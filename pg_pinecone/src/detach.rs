//! Implementation of `pinecone.detach_storage` / `pinecone.attach_storage`.
//! The `storage = 'detached'` state's own correctness guards (the
//! `_sync_trigger` detached branches, the row-level id guard, the registry
//! `CHECK`s) live elsewhere — this
//! module is the pair of verbs that actually walk a table into and back out
//! of that state, plus the preconditions gating the walk.
//!
//! ## `detach_storage(tbl, force => false)`
//! Gates run in a fixed order; the first four are CORRECTNESS preconditions
//! and are **never** bypassable by `force` (only the backup gate, a CUSTODY
//! check, is):
//!
//! 1. Ownership (`require_table_owner`).
//! 2. Registry row exists, `mode` is EXACTLY `'remote'` (a `'promoting'` row
//!    is mid-rebuild — detaching it would reopen the sync-wipe hole once the
//!    backfill eventually lands).
//! 3. `synced_by = 'trigger'` (AM-indexed tables are a hard, permanent
//!    non-goal — see `registry.rs`'s `tables_storage_synced_by_check`).
//! 4. A single-column `UNIQUE` index or `PRIMARY KEY` on `id_column` MUST
//!    exist. A one-shot `count(*) = count(DISTINCT id)`
//!    snapshot was RETIRED — it proves nothing about the future: without a
//!    continuously-enforced constraint, an `INSERT` of an existing id with a
//!    `NULL` vector (the detached-INSERT guard checks the vector, not the id)
//!    makes two local rows share one id, and `attach_storage`'s write-back
//!    then writes the one fetched remote vector onto BOTH — fabricated data.
//!    A real unique constraint makes that `INSERT` impossible at the storage
//!    layer, so only it is accepted here. See [`require_id_uniqueness`].
//! 5. The outbox is drained: no `'recover'`/`'backfill'` (rebuild-class) job
//!    may be queued for `tbl` at all — regardless of `force` — since either
//!    races the batched NULLing pass below. A residual backlog of ordinary
//!    `'upsert'`/`'delete'`/`'update'`/`'reupsert'` rows (self-contained
//!    payloads) blocks too, by default: `force => true` bypasses only THIS
//!    part (a custody choice — those rows still drain normally whenever a
//!    worker next runs; they just aren't guaranteed to have landed before
//!    the NULLing pass below).
//! 6. The backup gate (`backup_readiness::detach_gate`, reusing
//!    `pinecone.backup_readiness`'s own three checks): every check must read
//!    `'ok'`, or `force => true` is required to proceed anyway. The gate
//!    always RUNS and its failures are always reported — `force` changes
//!    only whether they block, recorded durably as `backup_gate = 'forced'`
//!    (vs `'covered'`) on `pinecone.tables`.
//!
//! Execution (one transaction — see below for why): persist the detach audit
//! columns + flip `storage = 'detached'` ([`registry::mark_detached`]),
//! install the row-level id guard
//! ([`sync_trigger::install_detached_id_guard`]), THEN the batched NULLing
//! pass — in that order, so the guards are active for every row the NULLing
//! statement itself touches (otherwise `_sync_trigger`
//! must see `storage = 'detached'` before it ever fires against the
//! NULLing statement's own transition tables).
//!
//! **Single-transaction NULLing, chosen over a worker-chunked pass**: the
//! whole gate-check + registry flip + guard install + `UPDATE tbl SET vcol
//! = NULL WHERE vcol IS NOT NULL` runs as ONE ordinary SQL-invoked function
//! call, so it is one Postgres transaction start to finish
//! (the ~6-10GB WAL/1M rows is an acceptable
//! one-time cost at the scales this targets). This makes crash-idempotency
//! trivial rather than requiring "resume mid-NULLing" bookkeeping: a crash
//! anywhere in the call rolls back EVERYTHING (the flip included), so a
//! re-run is just calling `detach_storage` again from a clean, unchanged
//! table — the `WHERE vcol IS NOT NULL` clause is what makes that re-run
//! (or a deliberate second call after a genuine partial failure some other
//! way) a correct no-op/resume either way, never a re-scan of already-NULL
//! rows. The trade-off, documented rather than engineered around: a very
//! large table pays one long-held lock and a large WAL burst in a single
//! commit, rather than the smaller, incremental bursts a worker-chunked
//! design would produce — acceptable at the scales it targets,
//! a documented limit at 100M+ rows (mirrors `recover_from_remote`'s own
//! honest scale ceiling).
//!
//! **Known residual limitation (not closed by this milestone)**: a
//! `REPEATABLE READ`/`SERIALIZABLE` transaction that began before
//! `detach_storage` commits, and makes its FIRST write to `tbl` only after
//! that commit, still runs under its own frozen snapshot — which never
//! observes the `storage` flip, no matter what locking `detach_storage`
//! takes (locks gate WAITING, not MVCC visibility). Dropping local vector indexes
//! first would be a partial fence; that
//! only protects a transaction already holding a lock on `tbl` at flip
//! time, not this dormant-then-first-touch case, so it was not implemented
//! here as a false sense of completeness. Long-lived transactions that
//! touch a table for the first time long after `BEGIN` are already an
//! unusual pattern; flagged for hardening if evidence ever surfaces it
//! matters in practice.
//!
//! ## `attach_storage(tbl)`
//! Gates: ownership, `storage = 'detached'` (a no-op NOTICE otherwise —
//! matches `detach_storage`'s own idempotent-re-run posture). Enqueues (or
//! updates in place) a `'recover'` job via
//! [`recover::enqueue_attach_recover`] — the exact same machinery
//! `pinecone.recover_from_remote` uses, with `ghosts` defaulting to `'delete'`
//! (nothing else reconciles a detached table's ghosts)
//! and an internal `attach => true` flag. `outbox::dispatch_recover`'s
//! completion arm is the actual flip: once that job's last chunk exhausts,
//! it commits `storage = 'full'` + removes the id guard in the SAME
//! transaction as the job row's own deletion. Never-unavailable throughout:
//! reads were remote-only the whole time `tbl` was detached (topk pins
//! `'remote'`), so this widens what LOCAL writes are allowed, never what
//! search serves. Crash-idempotent: the job's persisted cursor resumes
//! exactly like any other `'recover'` job; re-running `attach_storage`
//! itself just updates the same queued job in place.

use pgrx::{
    PgRelation, pg_sys,
    prelude::*,
    spi::{Spi, quote_qualified_identifier},
};

use crate::{backend, backup_readiness, catalog, outbox, recover, registry, sync_trigger};

/// Seconds Pinecone's own staleness allowance adds on top of the schedule
/// interval — same constant `backup_readiness`'s freshness check uses, but
/// this module states the SIMPLE, verbatim RPO qualifier (`interval + 15
/// min`), not that check's own more generous gate-tolerance window
/// (`interval + max(25%,1h) + 15min`) — the two are deliberately different
/// numbers for deliberately different purposes (see `registry.rs`'s
/// `detach_rpo_secs` column comment).
const RPO_STALENESS_ALLOWANCE_SECS: i64 = 15 * 60;

/// Row-count estimate above which `detach_storage` WARNs about its
/// single-transaction shed pass (lock duration + WAL burst). Not a ceiling —
/// nothing is enforced; the documented prohibitive scale is ~100M
/// (see the single-transaction note in the module rustdoc).
const LARGE_SHED_WARN_ROWS: i64 = 10_000_000;

/// Requires the calling role to own the TABLE `heap_relid` — or be a member
/// of the owning role, or superuser. Mirrors `promote::require_table_owner`/
/// `lifecycle::require_table_owner`/`maintain::require_table_owner` (each
/// module keeps its own copy with wording tailored to its own verbs, the
/// established convention in this crate).
fn require_table_owner(heap_relid: pg_sys::Oid, display: &str, func: &str) {
    let caller = unsafe { pg_sys::GetUserId() };
    let owns = unsafe { pg_sys::object_ownercheck(pg_sys::RelationRelationId, heap_relid, caller) };
    if !owns {
        error!("{func}: must own {display} (or be superuser)");
    }
}

/// Implements `pinecone.detach_storage`. See this module's own rustdoc for the
/// full gate order and execution design.
pub(crate) fn detach_storage(tbl: PgRelation, force: bool) {
    const FUNC: &str = "pinecone.detach_storage";
    let relid = tbl.oid();
    let display = catalog::display_name(&tbl);

    // GATE 1 (never bypassable): ownership.
    require_table_owner(relid, &display, FUNC);

    // GATE (never bypassable): plain tables only. A partitioned table
    // (relkind 'p') passes every registry-level gate below, but the
    // statement-level `_sync_trigger`/`_detached_id_guard` do NOT fire for DML
    // routed directly to a partition — so a detached partitioned table's
    // vector/id columns would be silently unprotected. Refuse structurally,
    // before touching the registry. See [`require_plain_table`].
    require_plain_table(&tbl, FUNC);

    let row = registry::lookup(relid).unwrap_or_else(|| {
        error!(
            "{FUNC}: {display} is not tracked — run pinecone.track(...) or pinecone.promote(...) first"
        )
    });
    if let Some(partition) = &row.partition {
        backend::error_with_code(
            PgSqlErrorCode::ERRCODE_FEATURE_NOT_SUPPORTED,
            format!(
                "{FUNC}: {display} has the partition column \"{}\": detached storage does not \
                 work with partitions yet",
                partition.column
            ),
            None,
            Some("the vectors of a partitioned table stay in the table"),
        );
    }

    // Idempotent re-run posture (see this module's rustdoc): already
    // detached RESUMES rather than erroring or purely no-oping — the
    // single-transaction design below means the verb itself can never leave
    // a mixed NULL/non-NULL state behind (a crash rolls back the flip too),
    // but a mixed state is still REACHABLE (a direct registry UPDATE, as
    // in tests; a restore of a half-shed replica) and the convergent
    // answer is: re-ensure the id guard (idempotent
    // `CREATE OR REPLACE TRIGGER`) and re-run the `WHERE vcol IS NOT NULL`
    // shed pass, which is a genuine no-op on an already-clean table. No
    // HTTP: the backup gate ran (or was forced) when this row first
    // detached — its recorded outcome stands; re-running it here would let
    // a resume silently re-litigate (or worse, upgrade) a 'forced' audit
    // mark.
    if row.storage == "detached" {
        // A resume must re-check Gate 5, exactly as the
        // fresh-detach path does — otherwise a `detach_storage` called
        // during an in-flight `attach_storage` would run its shed pass
        // concurrently with that attach's recover write-back, racing it.
        // Advisory lock first (uniform order), so this check can't miss a
        // job a worker round is mid-dispatch on either.
        outbox::lock_relid(relid);

        // Guards an unrecoverable remote wipe: the
        // `row.storage` read above happened BEFORE the lock. An attach job's
        // final drain round atomically commits `storage = 'full'` + removes
        // the guard + deletes its own job row (`outbox::dispatch_recover`'s
        // completion arm). Had we branched only on that stale read + the
        // queue, a resume that lost the race to the lock would find the queue
        // empty (job gone) and shed a now-`'full'` table — whose `_sync_trigger`
        // reads `storage` FRESH as `'full'`, takes the NON-detached branch,
        // and enqueues a `'delete'` for every id the attach just recovered:
        // total, unrecoverable loss of exactly what attach restored. So
        // RE-READ `storage` under the lock and bail if it's no longer
        // detached — now both the storage check and the queue check are
        // post-lock and mutually consistent.
        let locked = registry::lookup(relid).unwrap_or_else(|| {
            error!("{FUNC}: {display} vanished from pinecone.tables under the lock")
        });
        if locked.storage != "detached" {
            notice!(
                "{FUNC}: {display} is now storage = 'full' — a concurrent pinecone.attach_storage \
                 completed while this call waited for the lock; nothing to resume (its recovered \
                 vectors are intact and must NOT be shed)"
            );
            return;
        }
        if outbox::has_queued_op(relid, "recover") || outbox::has_queued_op(relid, "backfill") {
            error!(
                "{FUNC}: {display} is detached with a recover/backfill job in flight — most \
                 likely a pinecone.attach_storage('{display}') in progress. Refusing to resume the \
                 shed pass while it runs (the shed would race the recover write-back); wait for \
                 it to finish (see pinecone.status()), or drain/cancel it first."
            );
        }
        // Re-prove id-uniqueness on the resume path too.
        // The constraint could have been DROPped since the original detach; a
        // resume that re-arms the guard and re-sheds must refuse if a future
        // attach could no longer safely write back (else it re-opens the
        // dup-id fabrication window). Never bypassable — no `force` branch.
        require_id_uniqueness(&tbl, &locked, FUNC);
        sync_trigger::install_detached_id_guard(tbl.clone());
        let shed_rows = shed_vector_column(&tbl, &locked.vector_column);
        notice!(
            "{FUNC}: {display} is already detached — resumed the shed pass instead: \
             {shed_rows} remaining non-NULL row(s) shed (0 = nothing left to do); see \
             pinecone.status() for detached_at/backup_gate; run \
             pinecone.attach_storage('{display}') to reverse"
        );
        return;
    }

    // GATE 2 (never bypassable): mode EXACTLY 'remote' — a 'promoting' row
    // is mid-rebuild (the sync-wipe hole would reopen once that
    // backfill eventually lands on top of a detached table).
    if row.mode != "remote" {
        error!(
            "{FUNC}: {display} is not promoted (mode = '{}') — run pinecone.promote(...) first \
             (a 'promoting' row must finish its backfill before it can be detached)",
            row.mode
        );
    }

    // GATE 3 (never bypassable): synced_by = 'trigger' only — AM-indexed
    // tables are a hard, permanent non-goal (see registry.rs's
    // tables_storage_synced_by_check and this module's own rustdoc).
    if row.synced_by != "trigger" {
        error!(
            "{FUNC}: {display} is managed by the pg_pinecone index (synced_by = '{}') — AM-indexed \
             tables can never be detached: the NULLing rewrite would re-mint every TID and the \
             next VACUUM would remote-delete the only remaining copy (a permanent non-goal)",
            row.synced_by
        );
    }

    // GATE 4 (never bypassable): id-column uniqueness proof.
    require_id_uniqueness(&tbl, &row, FUNC);

    // GATE 5: outbox drained. Advisory lock first (the uniform
    // advisory-then-table lock order every worker-round-adjacent caller
    // uses — outbox::lock_relid's own rustdoc), so this check can't race a
    // concurrent worker round's own dispatch of tbl's queue.
    outbox::lock_relid(relid);
    if outbox::has_queued_op(relid, "recover") || outbox::has_queued_op(relid, "backfill") {
        error!(
            "{FUNC}: {display} has a recover/backfill job in flight in pinecone.outbox — finish it \
             (see pinecone.status()) before detaching; never bypassable, even with force => true \
             (racing the batched NULLing pass below against a rebuild job would corrupt both)"
        );
    }
    let backlog = outbox::backlog(relid);
    if backlog > 0 {
        if force {
            warning!(
                "{FUNC}: {display} has {backlog} queued pinecone.outbox row(s) — force => true \
                 bypasses the drain requirement (a custody bypass, not a correctness one): those \
                 rows are self-contained and will still drain normally whenever a worker next \
                 runs, but they are not guaranteed to have reached Pinecone before this call's \
                 NULLing pass — review pinecone.status() for {display} afterward"
            );
        } else {
            error!(
                "{FUNC}: {display} has {backlog} queued pinecone.outbox row(s) — run \
                 pinecone.flush('{display}', <timeout_ms>) first, or pass force => true to bypass \
                 (see pinecone.status())"
            );
        }
    }

    // GATE 6: the backup gate. Always RUNS and reports; force decides only
    // whether a failure blocks.
    let gate = backup_readiness::detach_gate(&tbl);
    let failed: Vec<String> = gate
        .rows
        .iter()
        .filter(|(_, status, _)| status != "ok")
        .map(|(check, status, detail)| format!("{check}={status} ({detail})"))
        .collect();
    let backup_gate_value = if failed.is_empty() {
        "covered"
    } else if force {
        warning!(
            "{FUNC}: {display}'s backup gate did not pass — proceeding anyway (force => true, a \
             custody bypass; recorded as backup_gate = 'forced', a permanent audit mark): {}",
            failed.join("; ")
        );
        "forced"
    } else {
        error!(
            "{FUNC}: {display}'s backup gate did not pass: {} — run pinecone.backup_readiness(\
             '{display}') for the live detail, or pass force => true to bypass (a forced detach \
             is outside support and may be permanently unrecoverable if Pinecone loses this \
             index)",
            failed.join("; ")
        );
    };
    let detach_backup_id = gate.backup_id.clone();
    let rpo_secs = gate
        .schedule_interval_secs
        .map(|interval| (interval + RPO_STALENESS_ALLOWANCE_SECS) as i32);

    // Size WARNING (no ceiling is ENFORCED today): the shed
    // pass below NULLs every non-NULL row in ONE transaction, holding a lock
    // and emitting a large WAL burst; above ~10M rows an operator should know
    // that before it runs, not discover it mid-commit. A cheap planner
    // estimate (`pg_class.reltuples`), never a full count. Warn only —
    // the one-time cost is accepted at the target
    // scales; ~100M is the documented prohibitive limit.
    let est_rows = estimated_row_count(relid);
    if est_rows > LARGE_SHED_WARN_ROWS {
        warning!(
            "{FUNC}: {display} has ~{est_rows} rows — detach_storage NULLs them in ONE \
             transaction, holding a lock for its duration and emitting a large WAL burst \
             (order ~6-10GB per 1M x 6KB values). This is a documented one-time cost; at ~100M \
             rows it is prohibitive (see docs). Run it in a maintenance window."
        );
    }

    // ---- EXECUTE (one transaction — see this module's rustdoc) ----

    // Persist the audit trail + flip storage, BEFORE installing the guard or
    // NULLing — `_sync_trigger`/`_detached_id_guard` both read this row
    // fresh every fire, and Postgres commands within the SAME transaction
    // always see earlier commands' writes regardless of isolation level, so
    // ordering these three writes first is what makes the guards active
    // by the time the NULLing UPDATE's own AFTER STATEMENT trigger fires.
    registry::mark_detached(
        relid,
        detach_backup_id.as_deref(),
        backup_gate_value,
        rpo_secs,
    );
    sync_trigger::install_detached_id_guard(tbl.clone());

    let shed_rows = shed_vector_column(&tbl, &row.vector_column);

    let bytes_estimate = shed_rows * row.dimension as i64 * 4;
    let backup_id_display = detach_backup_id
        .as_deref()
        .unwrap_or("<none — force-bypassed the backup gate>");

    notice!(
        "{FUNC}: {display} is now storage = 'detached' — {shed_rows} row(s) shed (~{} local \
         bytes reclaimable via VACUUM; VACUUM FULL/pg_repack to return the space to the OS). If \
         a local pgvector index still exists on {vcol}, DROP it — it indexes nothing now and \
         churns on every write. backup_gate = '{backup_gate_value}' (backup id: \
         {backup_id_display}). RPO = backup-schedule interval + 15 min, never zero — Pinecone \
         backups only include vectors written at least 15 minutes before the backup was taken. \
         Pinecone is now the only full copy of this table's vector column; run \
         pinecone.attach_storage('{display}') to reverse.",
        human_bytes(bytes_estimate),
        vcol = row.vector_column,
    );
}

/// Implements `pinecone.attach_storage`. See this module's own rustdoc.
pub(crate) fn attach_storage(tbl: PgRelation) {
    const FUNC: &str = "pinecone.attach_storage";
    let relid = tbl.oid();
    let display = catalog::display_name(&tbl);

    require_table_owner(relid, &display, FUNC);

    let row = registry::lookup(relid).unwrap_or_else(|| {
        error!(
            "{FUNC}: {display} is not tracked — run pinecone.track(...) or pinecone.promote(...) first"
        )
    });

    if row.storage != "detached" {
        notice!("{FUNC}: {display} is already storage = 'full' — nothing to do");
        return;
    }

    // The write-back this enqueues stamps one fetched
    // remote vector per id via `UPDATE ... WHERE id_col::text = $id`. If the
    // UNIQUE index/PK was DROPped during the detached period and a duplicate
    // id inserted (a NULL-vector INSERT the detached-INSERT guard permits), that write-back
    // would fan the single remote vector onto BOTH rows — fabricating data.
    // Re-prove the constraint here, before enqueueing, so attach refuses
    // rather than fabricating. Never bypassable (attach has no `force`). The
    // write-back itself carries a defense-in-depth fan-out guard for the
    // narrower drop-DURING-job race (see `recover::recover_chunk_trigger`).
    require_id_uniqueness(&tbl, &row, FUNC);

    recover::enqueue_attach_recover(tbl);
}

/// GATE: `tbl` must be a plain, ordinary table (`relkind = 'r'`) — never
/// a partitioned parent (`'p'`), foreign table, or any other relkind. The
/// detached-mode write guards are statement-level triggers on `tbl` itself;
/// Postgres does NOT fire a partitioned parent's statement triggers for DML
/// routed straight at a partition, so a detached partitioned table's
/// vector/id columns would be writable with no guard at all. Read straight
/// off the open relcache entry (same access pattern as
/// `lifecycle`/`recover`'s own `rd_rel` reads) — no catalog round-trip.
fn require_plain_table(tbl: &PgRelation, func: &str) {
    let relkind = unsafe { (*(*tbl.as_ptr()).rd_rel).relkind } as u8;
    if relkind != b'r' {
        let display = catalog::display_name(tbl);
        let kind = match relkind {
            b'p' => "a partitioned table (relkind 'p')",
            b'f' => "a foreign table (relkind 'f')",
            b'v' => "a view (relkind 'v')",
            b'm' => "a materialized view (relkind 'm')",
            _ => "not an ordinary table",
        };
        error!(
            "{func}: {display} is {kind} — detach_storage supports ONLY plain tables. The \
             detached-mode write guards are statement-level triggers that do not fire for DML \
             routed directly to a partition, so a detached partitioned table's vector and id \
             columns would be silently unprotected. Detach the individual leaf partitions \
             instead, or restructure as a plain table."
        );
    }
}

/// A cheap planner-stat row-count estimate for `relid` (`pg_class.reltuples`,
/// no scan) — used only to decide whether to WARN about the shed pass's
/// size. `reltuples` is `-1` for a never-analyzed table and can be stale;
/// both are fine here (a stale/low estimate just skips the advisory warning,
/// never blocks or miscounts anything — the shed pass itself is exact).
fn estimated_row_count(relid: pg_sys::Oid) -> i64 {
    Spi::get_one_with_args::<f32>(
        "SELECT reltuples FROM pg_catalog.pg_class WHERE oid = $1::oid",
        &[relid.into()],
    )
    .expect("pg_pinecone: detach_storage reltuples query failed")
    .map(|r| r.max(0.0) as i64)
    .unwrap_or(0)
}

/// The shed pass itself: `UPDATE tbl SET vcol = NULL WHERE vcol IS NOT
/// NULL`, returning how many rows it touched. The `WHERE` clause is what
/// makes a re-run/resume a genuine no-op on already-shed rows (never a
/// full-table rewrite twice); the trigger machinery (already reading
/// `storage = 'detached'` by the time this runs — see the call sites'
/// ordering) treats every vector->NULL transition it produces as a
/// silent no-op, never a remote `'delete'`.
fn shed_vector_column(tbl: &PgRelation, vector_column: &str) -> i64 {
    let table = quote_qualified_identifier(tbl.namespace(), tbl.name());
    let vcol = pgrx::spi::quote_identifier(vector_column);
    Spi::get_one::<i64>(&format!(
        "WITH shed AS (UPDATE {table} SET {vcol} = NULL WHERE {vcol} IS NOT NULL RETURNING 1) \
         SELECT count(*) FROM shed"
    ))
    .expect("pg_pinecone: detach_storage NULLing query failed")
    .unwrap_or(0)
}

/// GATE 4: a single-column `UNIQUE` index or `PRIMARY KEY` on `id_column`
/// MUST exist — never bypassable, not even with `force`.
///
/// A one-shot `count(*) = count(DISTINCT id)` snapshot is NOT sufficient and
/// was deliberately RETIRED: it proves nothing about the future. With no
/// continuously-enforced constraint, an `INSERT` of an EXISTING id with a
/// `NULL` vector (which the detached-INSERT guard permits — it checks the
/// vector, not the id) creates two local rows sharing one id, purely
/// sequentially, no concurrency needed. A later `attach_storage`/
/// `recover_from_remote` fetches the one remote vector for that id and its
/// write-back `UPDATE ... WHERE id_col = $id` matches BOTH rows — writing the
/// same fetched vector onto each, fabricating data. It also closes the
/// TOCTOU window a snapshot check leaves open across Gate 6's own HTTP round.
/// A real `UNIQUE` index makes the duplicate `INSERT` impossible at the
/// storage layer — the sound, continuous invariant — so that is the only
/// thing accepted here.
fn require_id_uniqueness(tbl: &PgRelation, row: &registry::RegistryRow, func: &str) {
    if catalog::has_unique_single_column_index(tbl.oid(), &row.id_column) {
        return;
    }
    let display = catalog::display_name(tbl);
    let id = &row.id_column;
    error!(
        "{func}: {display}'s id column '{id}' has no single-column UNIQUE index or PRIMARY KEY \
         — pinecone.detach_storage REQUIRES one and will not proceed without it (never bypassable, \
         not even with force => true). Without a continuously-enforced unique constraint, an \
         INSERT of an existing id with a NULL vector would create two local rows sharing one id, \
         and pinecone.attach_storage would then write the one fetched remote vector onto BOTH — \
         fabricating data. Add one first, e.g. CREATE UNIQUE INDEX ON {display} ({id}); (or a \
         PRIMARY KEY), then retry."
    );
}

/// Human-scale byte formatting for `detach_storage`'s completion NOTICE —
/// deliberately coarse (whole-number units), since `bytes` is already an
/// estimate (`shed rows * dimension * 4`, ignoring TOAST/tuple-header
/// overhead), not a measured value.
fn human_bytes(bytes: i64) -> String {
    const KB: i64 = 1024;
    const MB: i64 = KB * 1024;
    const GB: i64 = MB * 1024;
    if bytes >= GB {
        format!("{:.1}GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1}MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1}KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes}B")
    }
}
