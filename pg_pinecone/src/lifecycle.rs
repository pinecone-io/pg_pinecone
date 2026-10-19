//! The lifecycle surface — `pinecone.promote_index`/`demote_index`
//! (the pg_pinecone access method's own local<->remote engine flip, operating on
//! an EXISTING `CREATE INDEX ... USING pinecone` index) and `pinecone.create_index`/
//! `set_engine` (the Level-1 "storage engines" unification — the
//! `pgvector | native | remote` model, one verb per lifecycle transition).
//!
//! ## Division from `promote.rs`
//! `promote::promote`/`demote` operate on a TABLE tracked the 0.2/0.3 way
//! (`pinecone.track`, a plain pgvector index + sync triggers, `synced_by =
//! 'trigger'`). This module operates on the pg_pinecone ACCESS METHOD's own index
//! (`synced_by = 'am'`), whose local/remote split lives in the index's own
//! metapage `mode` byte (`local::metapage::Mode`), not in `pinecone.tables.mode`
//! alone. The two are deliberately kept apart — `registry::require_not_am`
//! already rejects `promote()`/`demote()` on an AM-managed table, and this
//! module's own functions reject a `synced_by = 'trigger'` table the same way.
//!
//! ## promote_index: the mechanics
//! `local -> promoting` (metapage flip, [`local::metapage::flip_mode`], ONE
//! `GenericXLog` record — atomic) + a registry row created NOW (`synced_by =
//! 'am'`, `mode = 'promoting'`, `engine = 'remote'` — the engine column
//! records the TARGET/declared engine, not the transient serving mode; see
//! `registry.rs`'s own doc comment on that column) + a TID-keyed `'backfill'`
//! job enqueued (reusing 0.3's `promote::try_tid_backfill_chunk` machinery via
//! `outbox::dispatch_backfill`, exactly like a fresh remote `ambuild`).
//! Scans SERVE LOCAL throughout (`am::ambeginscan`'s `Promoting -> Local`
//! dispatch — never-partial-namespace: the backfill isn't done, so remote
//! would answer with a subset). Writes DUAL-write (`am::aminsert`'s
//! `Promoting` arm: append locally AND enqueue remotely) so local scans stay
//! fully correct for the WHOLE promoting window, not just up to the moment
//! `promote_index` was called — see `am::aminsert`'s own rustdoc for why this
//! is exactly the assumption `try_tid_backfill_chunk`'s "post-cursor inserts
//! converge" correctness argument already needed. The worker's own backfill
//! dispatch (`outbox::dispatch_backfill`'s `Ok(None)` arm) is the ONLY place
//! that flips the metapage `promoting -> remote` (another atomic
//! [`local::metapage::flip_mode`] call) once the TID scan is exhausted; it
//! also flips the registry row via the SAME `registry::flip_promoting_to_
//! remote` the old trigger-based async promote already used (that call was
//! always a no-op for a `synced_by = 'am'` row — see its own
//! rustdoc — because no AM row had ever been `'promoting'` before
//! `promote_index` existed).
//!
//! ## demote_index: the mechanics, and the guarantee ACTUALLY delivered
//! `remote -> demoting` (metapage flip) + a `pinecone.pending_demotes` row —
//! **`demote_index` itself returns immediately; it does not rebuild inline.**
//! The worker's own round ([`run_round`]) picks up pending demotes and runs
//! `ALTER INDEX ... SET (mode = 'local'); REINDEX INDEX ...;` — the ONLY
//! way this codebase can turn a remote-mode AM index into a local-mode one
//! (`local::vacuum`'s own "CIC/RIC decision": `REINDEX CONCURRENTLY` stays
//! rejected for a local-mode build target, so there is no non-blocking
//! rebuild-into-a-different-mode primitive to reach for here; a plain
//! `REINDEX` is a full, `AccessExclusiveLock`-held rebuild, same as any other
//! REINDEX in this codebase — see `maintain.rs`'s own module docs for the
//! identical, already-accepted tradeoff on the drift-triggered maintenance
//! REINDEX).
//!
//! **The honest v1 guarantee**: while a demote is QUEUED (between
//! `demote_index` returning and the worker's REINDEX actually starting),
//! scans/writes continue to serve REMOTE correctly — the metapage's
//! `Demoting` mode dispatches identically to `Remote` in every one of
//! `am.rs`'s mode-`match`es (`aminsert`, `ambulkdelete`, `ambeginscan`) —
//! this is genuinely "never unavailable" for that whole window, however long
//! the queue wait is. The moment the worker's `REINDEX INDEX` actually
//! starts, though, it takes the SAME `AccessExclusiveLock` any `REINDEX`
//! does: every concurrent scan of that index blocks for the rebuild's
//! duration (seconds to the churn-benchmark's ~100s-class figures at scale).
//! **That IS an availability gap, not zero-downtime**: demote = availability
//! gap equal to rebuild time.
//! **The exact guarantee this ships**: *never unavailable while queued,
//! blocking for the rebuild's duration once it starts* — not
//! zero-downtime-throughout. A true never-blocking demote would need a
//! `REINDEX CONCURRENTLY`-shaped rebuild-into-local, which this codebase does
//! not have (see the CIC/RIC decision cited above) — recorded as a
//! future candidate, not built here.
//!
//! On successful completion, [`run_round`] also cleans up the now-superseded
//! remote resource (deletes the whole Pinecone index if no other tracked
//! table shares it, else just that namespace's vectors — mirroring
//! `promote::demote`'s own `drop_remote_index` logic) and removes the
//! registry row entirely — a demoted-back-to-local AM index has no
//! `pinecone.tables` row, same as one that was always local (`local::build`'s
//! own "no registry row" invariant for local-mode AM indexes).
//!
//! ## The engine surface (Level 1)
//! [`create_index`]/[`set_engine`] are deliberately a MINIMAL v1: three
//! engines, one metric/vector column each, at most one filter column
//! (`filter_column`, engines `native` and `remote`; `partition_column` makes
//! it one namespace per value on `remote`), no BYOC
//! cloud/region override (reuses `pinecone.am_cloud`/`am_region`, same as the
//! plain AM path). `set_engine` performs `native <-> remote` (exactly
//! [`promote_index`]/[`demote_index]`) and `pgvector -> native` (a native
//! build, then the hnsw index is dropped), and gives a clear, named "not
//! supported yet" error for the other transitions touching `pgvector`.

use pgrx::{PgRelation, prelude::*, spi::Spi};

use crate::{
    am, am_partition, backend, catalog,
    local::metapage::{self, Mode},
    outbox,
    partition::{Claim, Lane, PartitionSpec},
    partition_registry, promote,
    registry::{self, RegistryRow},
    secret,
};

extension_sql!(
    r#"
    CREATE TABLE pinecone.pending_demotes (
        index_relid oid PRIMARY KEY,
        heap_relid oid NOT NULL,
        queue_parked boolean NOT NULL DEFAULT false,
        parked_reason text,
        attempts int NOT NULL DEFAULT 0,
        last_error text,
        requested_at timestamptz NOT NULL DEFAULT now()
    );
    REVOKE ALL ON pinecone.pending_demotes FROM PUBLIC;
    SELECT pg_catalog.pg_extension_config_dump('pinecone.pending_demotes', '');
    "#,
    name = "pg_pinecone_pending_demotes",
    requires = [pinecone],
);

// The retriable remote-cleanup queue. `demote_step` (phase 1)
// enqueues one row here — carrying the captured remote coordinates — in the
// SAME transaction that deletes the registry row and purges the
// `pending_demotes` retry record, so the "delete the superseded Pinecone
// index" obligation outlives phase 1's commit. `remote_cleanup` (phase 2)
// retries the HTTP delete on every worker round and only removes the row once
// the remote resource is confirmed gone (a 200/202 delete, or a 404 meaning
// it was already gone). This closes both the crash gap (a worker crash
// between phase 1 and phase 2 — the docs' D11) and the partition gap (a
// Pinecone outage outlasting the old best-effort, non-retried cleanup's
// bounded HTTP retry window): before this, phase 1 purged the only retry
// record, so a failed phase 2 orphaned the remote index forever. Transient
// failures (partition/timeout/5xx) never count toward `attempts` (they retry
// indefinitely until healed, the same policy `outbox.rs` uses); only
// permanent-class failures (e.g. a 403 deletion-protection guard) accrue
// `attempts` and park after `pinecone.outbox_max_attempts`. Config-dumped like
// `pending_demotes`, so a dump/restore mid-cleanup does not silently drop the
// obligation.
extension_sql!(
    r#"
    CREATE TABLE pinecone.pending_cleanups (
        id bigserial PRIMARY KEY,
        index_name text NOT NULL,
        namespace text,
        shared boolean NOT NULL,
        attempts int NOT NULL DEFAULT 0,
        last_error text,
        queue_parked boolean NOT NULL DEFAULT false,
        parked_reason text,
        requested_at timestamptz NOT NULL DEFAULT now()
    );
    REVOKE ALL ON pinecone.pending_cleanups FROM PUBLIC;
    SELECT pg_catalog.pg_extension_config_dump('pinecone.pending_cleanups', '');
    SELECT pg_catalog.pg_extension_config_dump('pinecone.pending_cleanups_id_seq', '');
    "#,
    name = "pg_pinecone_pending_cleanups",
    requires = [pinecone],
);

// Same "one plpgsql EXCEPTION WHEN OTHERS around the risky DDL" shape as
// `maintain.rs`'s own `pinecone._maintenance_attempt`, and the same
// owner-identity/naming hardening that function got (see its own comment block
// in `maintain.rs` for the
// full rationale, point by point):
// - the function-level `SET search_path = pg_catalog, pg_temp` pins name
//   resolution for the ENTIRE call, restored by Postgres on any exit path;
// - the target's schema-qualified name AND its table's CURRENT owner are
//   resolved fresh from `pg_class`/`pg_namespace`/`pg_index` (never a
//   caller-search_path-dependent `regclass::text` rendering, never a stale
//   recorded role), doubling as the existence check;
// - the ALTER INDEX + REINDEX lines run under a temporary `SET ROLE` to that
//   owner (both are owner-gated DDL — the worker's own, typically-superuser
//   identity must never rubber-stamp them), `RESET ROLE` on the success
//   path, and the plpgsql `EXCEPTION` subtransaction rolls the `SET ROLE`
//   back on any error (verified sound by the gate for
//   `_maintenance_attempt`'s identical shape).
extension_sql!(
    r#"
    CREATE FUNCTION pinecone._demote_attempt(
        p_index_relid oid,
        OUT done boolean,
        OUT gone boolean,
        OUT error text
    )
    LANGUAGE plpgsql
    SET search_path = pg_catalog, pg_temp
    AS $$
    DECLARE
        v_schema name;
        v_index name;
        v_owner name;
    BEGIN
        done := false;
        gone := false;
        error := NULL;

        SELECT n.nspname, c.relname, pg_catalog.pg_get_userbyid(t.relowner)
          INTO v_schema, v_index, v_owner
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_catalog.pg_index x ON x.indexrelid = c.oid
        JOIN pg_catalog.pg_class t ON t.oid = x.indrelid
        WHERE c.oid = p_index_relid;

        IF v_schema IS NULL THEN
            gone := true;
            RETURN;
        END IF;

        EXECUTE pg_catalog.format('SET ROLE %I', v_owner);
        EXECUTE pg_catalog.format('ALTER INDEX %I.%I SET (mode = ''local'')', v_schema, v_index);
        EXECUTE pg_catalog.format('REINDEX INDEX %I.%I', v_schema, v_index);
        RESET ROLE;
        done := true;
    EXCEPTION WHEN OTHERS THEN
        error := SQLERRM;
    END;
    $$;
    REVOKE EXECUTE ON FUNCTION pinecone._demote_attempt(oid) FROM PUBLIC;
    "#,
    name = "pg_pinecone_demote_attempt",
    requires = [pinecone, "pg_pinecone_pending_demotes"],
);

/// Requires the calling role to own the TABLE `heap_relid` — or be a member
/// of the owning role, or superuser — mirroring
/// `maintain::require_table_owner`'s `schedule_maintenance` gate
/// and Postgres's own `REINDEX` permission rule. Without this, any role
/// with `EXECUTE` on `pinecone.promote_index`/`demote_index`/`set_engine`
/// (PUBLIC by default, like any plain `#[pg_extern]`) could flip someone
/// ELSE's index between engines — promote pushes the whole table's vectors
/// to the stored remote account, demote queues a blocking REINDEX; both are
/// owner-class operations. Checked against the table (not the index's own
/// `relowner`), exactly like `REINDEX`.
fn require_table_owner(heap_relid: pg_sys::Oid, display: &str, func: &str) {
    let caller = unsafe { pg_sys::GetUserId() };
    let owns = unsafe { pg_sys::object_ownercheck(pg_sys::RelationRelationId, heap_relid, caller) };
    if !owns {
        error!("{func}: must own {display} (or be superuser) to change its engine");
    }
}

/// Errors clearly unless `idx` is a pg_pinecone access-method index — the shared
/// precondition [`promote_index`]/[`demote_index`] both start with.
fn require_pg_pinecone_am_index(idx: &PgRelation, func: &str) -> metapage::MetaPage {
    if !idx.is_index() || !crate::maintain::is_pg_pinecone_am_index(idx) {
        error!(
            "{func}: \"{}\" is not a pg_pinecone access-method index — CREATE INDEX ... USING pinecone \
             first (or pinecone.create_index(...))",
            idx.name()
        );
    }
    metapage::read(idx.as_ptr())
}

fn metric_registry_str(metric: crate::local::metric::Metric) -> &'static str {
    match metric {
        crate::local::metric::Metric::L2 => "l2",
        crate::local::metric::Metric::Cosine => "cosine",
        crate::local::metric::Metric::Ip => "ip",
    }
}

/// Warns (never refuses) if `idx` is `UNLOGGED` — see
/// `local::metapage::flip_mode`'s own rustdoc for the documented gap this
/// covers: an unlogged index's `INIT_FORKNUM` crash-reset image isn't updated
/// by a mode flip, so a crash during promote/demote can revert its lifecycle
/// state on restart.
fn warn_if_unlogged(idx: &PgRelation, func: &str) {
    // `relpersistence` is `c_char`, whose signedness flips per target (i8 on
    // x86_64/Apple, u8 on aarch64-linux); `RELPERSISTENCE_UNLOGGED` is always
    // `u8` (bindgen's char-literal constants aren't `c_char`-typed). Cast the
    // field through `u8` so the comparison compiles on every platform — the
    // constant needs no cast, and clippy correctly flags one as redundant.
    let persistence = unsafe { (*(*idx.as_ptr()).rd_rel).relpersistence } as u8;
    if persistence == pg_sys::RELPERSISTENCE_UNLOGGED {
        warning!(
            "{func}: \"{}\" is an UNLOGGED index — a crash before this transition completes may \
             revert its lifecycle state on restart (see local::metapage::flip_mode's docs); \
             proceeding anyway",
            idx.name()
        );
    }
}

/// Warns (never refuses — a hard rejection would also reject the pgrx test
/// harness's own enclosing transaction, and Postgres offers no reliable way
/// for a plain function to know its statement is top-level) when `func` runs
/// inside an explicit transaction block: the metapage mode flip these verbs
/// perform is a GenericXLog page write — immediate and NON-transactional —
/// so an enclosing ROLLBACK undoes the registry/queue writes but NOT the
/// flip, leaving the two out of sync. Both verbs order the flip LAST (after
/// every transactional write) and carry a recovery arm for exactly this
/// state (see each verb's mode `match`), so the remedy is simply re-running
/// the same verb.
fn warn_if_in_transaction_block(func: &str) {
    if unsafe { pg_sys::IsTransactionBlock() } {
        warning!(
            "{func}: running inside a transaction block — the index's mode flip is immediate \
             and will NOT roll back with this transaction; if the transaction aborts, re-run \
             {func} to restore consistency"
        );
    }
}

/// Implements `pinecone.promote_index(idx)` — see this module's own docs for
/// the full mechanics.
///
/// ## Abort/crash ordering (the crash matrix's promote column)
/// Every transactional write (registry row, backfill job) happens BEFORE the
/// non-transactional metapage flip, which is this function's last act — so
/// an error anywhere in this function aborts cleanly back to a fully-local
/// index with nothing dangling (the created Pinecone index is the one
/// exception: create-or-adopt is idempotent, a retry adopts it). The one
/// residual inconsistency window — flip durable, enclosing transaction
/// rolled back afterward ([`warn_if_in_transaction_block`]) or a crash
/// exactly between the flip's WAL record and the commit record — leaves
/// metapage `Promoting` with NO registry row: scans stay correct (promoting
/// serves local), and the `Promoting`-without-registry-row recovery arm
/// below lets a plain re-run of this function repair it (it re-does the
/// transactional writes and skips the already-done flip).
pub(crate) fn promote_index(
    idx: PgRelation,
    pinecone_index: Option<&str>,
    pinecone_namespace: Option<&str>,
    attach_unchecked: bool,
    partition_column: Option<&str>,
) {
    let meta = require_pg_pinecone_am_index(&idx, "pinecone.promote_index");
    let index_relid = idx.oid();
    let heap_relid = catalog::index_heap_relid(index_relid)
        .unwrap_or_else(|| error!("pinecone.promote_index: could not resolve the indexed table"));
    require_table_owner(
        heap_relid,
        &catalog::display_name_of(heap_relid),
        "pinecone.promote_index",
    );
    // The backfill would read the local pages, and an index below format 3 holds codes this build
    // cannot read (`set_engine(.., 'remote')` lands here too).
    if meta.needs_reindex() {
        unsafe { metapage::refuse_old_format(idx.as_ptr(), meta.stored()) };
    }

    // Advisory lock on the HEAP relid first — the exact lock
    // `outbox::dispatch_backfill`'s worker path already takes for this same
    // relid (via `drain_step`'s `Worker` credentials), so a concurrent
    // worker round and this call can never interleave on the same table
    // (the uniform lock order, reused here).
    outbox::lock_relid(heap_relid);

    // `already_flipped`: the recovery case — metapage says `Promoting` but
    // the registry row is missing (an enclosing-transaction rollback or a
    // flip-vs-commit crash window; see the rustdoc above). A re-run redoes
    // the transactional half and skips the flip.
    let already_flipped = match meta.mode {
        Mode::Local => false,
        Mode::Promoting if registry::lookup(heap_relid).is_none() => {
            warning!(
                "pinecone.promote_index: \"{}\" is mid-promotion with no registry row (an earlier \
                 call's transaction rolled back after its mode flip) — repairing by redoing \
                 the registration and backfill enqueue",
                idx.name()
            );
            true
        }
        Mode::Promoting => error!(
            "pinecone.promote_index: \"{}\" already has a promotion in progress",
            idx.name()
        ),
        other => error!(
            "pinecone.promote_index: \"{}\" is not local-mode (mode = {other:?}) — only a \
             local-mode pg_pinecone index can be promoted",
            idx.name()
        ),
    };
    if already_flipped && pinecone_index.is_some() {
        error!(
            "pinecone.promote_index: \"{}\" is mid-promotion from an earlier (non-attach) call — \
             re-run pinecone.promote_index/set_engine without pinecone_index to finish that first",
            idx.name()
        );
    }
    if !already_flipped && registry::lookup(heap_relid).is_some() {
        error!(
            "pinecone.promote_index: \"{}\"'s table already has a pinecone.tables row — should be \
             unreachable for a genuinely local-mode pg_pinecone-AM index",
            idx.name()
        );
    }
    if !secret::has_stored_api_key() {
        error!(
            "pinecone.promote_index: needs a stored API key — run pinecone.set_api_key(...) first"
        );
    }
    warn_if_unlogged(&idx, "pinecone.promote_index");
    warn_if_in_transaction_block("pinecone.promote_index");

    let (schema, table) = catalog::resolve_table_name(heap_relid)
        .unwrap_or_else(|| error!("pinecone.promote_index: could not resolve the indexed table"));
    let display = format!("{schema}.{table}");
    let id_column = am::single_primary_key_column(heap_relid, &display);
    let columns = catalog::all_columns(heap_relid);
    let column_named = |attnum: i16, what: &str| {
        columns
            .iter()
            .find(|c| c.attnum == attnum)
            .unwrap_or_else(|| error!("pinecone.promote_index: {what} not found on {display}"))
            .name
            .clone()
    };
    let vector_column = column_named(meta.heap_attnum, "indexed column");
    // A filtered local index promotes with its key: the backfill and every later write carry it
    // to Pinecone as metadata, scans push it down.
    let filter_column = meta
        .has_filter()
        .then(|| column_named(meta.filter_heap_attnum, "filter column"));

    let metric = metric_registry_str(meta.metric);
    let default_namespace = format!("{schema}.{table}");
    let attaching = pinecone_index.is_some();
    let partition = partition_column.map(|column| {
        if attaching {
            require_partition_engine("pinecone.promote_index", "remote", pinecone_index);
        }
        if filter_column.as_deref() != Some(column) {
            backend::error_with_code(
                PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
                format!(
                    "pinecone.promote_index: partition_column \"{column}\" is not the second key \
                     column of \"{}\" ({})",
                    idx.name(),
                    filter_column
                        .as_deref()
                        .map_or("it has none".to_string(), |key| format!("it is \"{key}\"")),
                ),
                None,
                Some(
                    "the partition column is the filter key of the index: CREATE INDEX .. USING \
                     pg_pinecone (vec, <partition column>) WITH (mode = 'local') first",
                ),
            );
        }
        let index_name = promote::default_index_name(&schema, &table);
        am_partition::declare(
            heap_relid,
            &schema,
            &table,
            column,
            &index_name,
            "pinecone.promote_index",
        )
    });

    let (index_name, namespace, read_capacity) = if let Some(pinecone_index) = pinecone_index {
        let namespace = pinecone_namespace
            .map(str::to_string)
            .unwrap_or(default_namespace);
        let claim = Claim::Exact(&namespace);
        if let Some(owner) = registry::claim_owner(pinecone_index, claim, heap_relid) {
            partition_registry::refuse_family_clash(
                "pinecone.promote_index",
                pinecone_index,
                claim,
                &owner,
            );
            let owner_display = catalog::display_name_of(owner.relid);
            error!(
                "pinecone.promote_index: index '{pinecone_index}' namespace '{namespace}' is \
                 already owned by {owner_display}"
            );
        }
        let mapped_metric = promote::map_metric(metric);
        let local_row_count = Spi::get_one::<i64>(&format!(
            "SELECT count(*) FROM {}",
            pgrx::spi::quote_qualified_identifier(&schema, &table)
        ))
        .expect("pinecone.promote_index: row-count query failed")
        .expect("pinecone.promote_index: count(*) unexpectedly NULL");
        let client = backend::client_for_effective_key();
        let read_capacity = promote::attach_index(
            &client,
            pinecone_index,
            meta.dim,
            mapped_metric,
            &namespace,
            local_row_count,
            attach_unchecked,
            &display,
            filter_column.as_deref(),
        );
        (pinecone_index.to_string(), namespace, read_capacity)
    } else {
        let index_name = promote::default_index_name(&schema, &table);
        let namespace = default_namespace;
        let claim = Claim::Exact(&namespace);
        if let Some(owner) = registry::claim_owner(&index_name, claim, heap_relid) {
            partition_registry::refuse_family_clash(
                "pinecone.promote_index",
                &index_name,
                claim,
                &owner,
            );
            let owner_display = catalog::display_name_of(owner.relid);
            error!(
                "pinecone.promote_index: index '{index_name}' namespace '{namespace}' is already \
                 owned by {owner_display}"
            );
        }
        let mapped_metric = promote::map_metric(metric);
        let (cloud, region) = am::am_cloud_region();
        let client = backend::client_for_effective_key();
        let read_capacity = promote::create_or_adopt_index(
            &client,
            &index_name,
            meta.dim,
            mapped_metric,
            serde_json::json!({"serverless": {"cloud": cloud, "region": region}}),
            &display,
        );
        promote::wait_ready(&client, &index_name);
        (index_name, namespace, read_capacity)
    };

    registry::insert(&RegistryRow {
        relid: heap_relid,
        vector_column,
        id_column,
        metric: metric.to_string(),
        dimension: meta.dim as i32,
        metadata_columns: Vec::new(),
        mode: if attaching { "remote" } else { "promoting" }.to_string(),
        sync_mode: "async".to_string(),
        synced_by: "am".to_string(),
        queue_parked: false,
        parked_reason: None,
        index_name: Some(index_name),
        namespace: Some(namespace),
        am_index_relid: Some(index_relid),
        last_vacuum_error: None,
        // The engine column records the TARGET engine (module docs).
        engine: Some("remote".to_string()),
        // An AM row (`synced_by = 'am'`) can never be detached — enforced
        // at the schema level too (`registry.rs`'s
        // `tables_storage_synced_by_check`).
        storage: "full".to_string(),
        detached_at: None,
        detach_backup_id: None,
        backup_gate: None,
        detach_rpo_secs: None,
        read_capacity,
        attached: attaching,
        partition,
        filter_column,
    });
    if !attaching {
        am::enqueue_backfill(heap_relid);
    }

    // LAST, after every transactional write above — see this function's
    // rustdoc for the abort/crash ordering argument. An attach skips the
    // `Promoting` window entirely — `attach_index` already confirmed the
    // namespace mirrors this table, so there's no backfill to protect scans
    // from serving a partial view of.
    if !already_flipped {
        unsafe {
            metapage::flip_mode(
                idx.as_ptr(),
                if attaching {
                    Mode::Remote
                } else {
                    Mode::Promoting
                },
            );
        }
    } else {
        // The original flip's
        // `CacheInvalidateRelcacheByRelid` (inside `flip_mode`) was
        // registered in a transaction that ABORTED — Postgres only
        // broadcasts queued invalidations at COMMIT, so no other backend
        // ever learned the metapage durably changed. This repair run's own
        // transaction WILL commit, so re-send the broadcast here; without
        // it, a warm backend keeps dispatching on its stale rd_amcache'd
        // mode indefinitely (the exact staleness the committed-flip path
        // already handles).
        unsafe {
            pg_sys::CacheInvalidateRelcacheByRelid(index_relid);
        }
    }

    if !outbox::launcher_lock_held() {
        let queued = if attaching {
            "future writes will queue"
        } else {
            "a backfill is queued and future writes will queue"
        };
        warning!(
            "pinecone.promote_index: {display} — no pg_pinecone worker launcher is running in this \
             cluster — {queued} in pinecone.outbox but nothing will drain it until one starts \
             (see pinecone.status())"
        );
    }
}

/// Implements `pinecone.demote_index(idx)` — schedules the transition; see this
/// module's own docs for the exact "never unavailable while queued, blocks
/// for the rebuild once it starts" guarantee.
///
/// ## Abort/crash ordering (the crash matrix's demote column)
/// Same discipline as [`promote_index`]: the transactional write (the
/// `pinecone.pending_demotes` row) happens BEFORE the non-transactional
/// metapage flip, which is last. The residual window — flip durable, row
/// rolled back — leaves metapage `Demoting` with no pending row: scans stay
/// correct (demoting serves remote, indistinguishable from plain remote),
/// and the `Demoting`-without-pending-row recovery arm below lets a re-run
/// re-queue it.
pub(crate) fn demote_index(idx: PgRelation) {
    let meta = require_pg_pinecone_am_index(&idx, "pinecone.demote_index");
    let index_relid = idx.oid();
    let heap_relid = catalog::index_heap_relid(index_relid)
        .unwrap_or_else(|| error!("pinecone.demote_index: could not resolve the indexed table"));
    require_table_owner(
        heap_relid,
        &catalog::display_name_of(heap_relid),
        "pinecone.demote_index",
    );

    outbox::lock_relid(heap_relid);

    let already_flipped = match meta.mode {
        Mode::Remote => false,
        Mode::Demoting if !pending_demote_exists(index_relid) => {
            warning!(
                "pinecone.demote_index: \"{}\" is mid-demote with no pending_demotes row (an \
                 earlier call's transaction rolled back after its mode flip) — repairing by \
                 re-queueing the rebuild",
                idx.name()
            );
            true
        }
        Mode::Demoting => {
            // The retry hatch, mirroring `pinecone.schedule_maintenance`'s
            // "re-run to clear the park" convention: a demote whose REINDEX
            // failed `pinecone.outbox_max_attempts` times parks its
            // pending_demotes row — re-running demote_index unparks it (and
            // resets attempts) so the worker retries; there is no separate
            // retry function. A live (unparked) in-flight demote still
            // errors.
            let unparked = Spi::get_one_with_args::<bool>(
                "WITH u AS (UPDATE pinecone.pending_demotes SET queue_parked = false, \
                 parked_reason = NULL, attempts = 0, last_error = NULL \
                 WHERE index_relid = $1 AND queue_parked RETURNING 1) \
                 SELECT EXISTS (SELECT 1 FROM u)",
                &[index_relid.into()],
            )
            .expect("pg_pinecone: pending_demotes unpark failed")
            .expect("pg_pinecone: EXISTS unexpectedly NULL");
            if unparked {
                warning!(
                    "pinecone.demote_index: \"{}\"'s parked demote was unparked — the worker will \
                     retry its rebuild",
                    idx.name()
                );
                return;
            }
            error!(
                "pinecone.demote_index: \"{}\" already has a demote in progress",
                idx.name()
            )
        }
        other => error!(
            "pinecone.demote_index: \"{}\" is not remote-mode (mode = {other:?}) — only a \
             remote-mode pg_pinecone index can be demoted",
            idx.name()
        ),
    };
    if registry::lookup(heap_relid).is_none() {
        error!(
            "pinecone.demote_index: \"{}\"'s table has no pinecone.tables row — should be unreachable \
             for a remote-mode pg_pinecone-AM index",
            idx.name()
        );
    }
    warn_if_unlogged(&idx, "pinecone.demote_index");
    warn_if_in_transaction_block("pinecone.demote_index");

    Spi::run_with_args(
        "INSERT INTO pinecone.pending_demotes (index_relid, heap_relid) VALUES ($1, $2)",
        &[index_relid.into(), heap_relid.into()],
    )
    .expect("pg_pinecone: pending_demotes insert failed");
    // LAST, after the transactional write above — see this function's
    // rustdoc for the abort/crash ordering argument.
    if !already_flipped {
        unsafe {
            metapage::flip_mode(idx.as_ptr(), Mode::Demoting);
        }
    } else {
        // Same reasoning as `promote_index`'s
        // recovery arm: the aborted original transaction discarded its
        // flip's relcache-invalidation broadcast, so this (committing)
        // repair run must re-send it for warm backends' rd_amcache.
        unsafe {
            pg_sys::CacheInvalidateRelcacheByRelid(index_relid);
        }
    }

    if !outbox::launcher_lock_held() {
        warning!(
            "pinecone.demote_index: \"{}\" is queued to demote but no pg_pinecone worker launcher is \
             running in this cluster — it will serve remote (never-unavailable) but the local \
             rebuild will not run until one starts",
            idx.name()
        );
    }
}

/// Whether a `pinecone.pending_demotes` row exists for `index_relid` —
/// [`demote_index`]'s recovery-arm probe.
fn pending_demote_exists(index_relid: pg_sys::Oid) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT EXISTS (SELECT 1 FROM pinecone.pending_demotes WHERE index_relid = $1)",
        &[index_relid.into()],
    )
    .expect("pg_pinecone: pending_demotes existence query failed")
    .expect("pg_pinecone: EXISTS unexpectedly NULL")
}

/// One `pinecone.pending_demotes` row — `pub(crate)` (with its fields) so
/// `worker.rs` can carry it ACROSS transactions: the worker runs
/// [`pending_demotes`], each [`demote_step`], and each [`remote_cleanup`] in
/// their own separate `BackgroundWorker::transaction` (see [`run_round`]'s
/// rustdoc for the ordering argument).
pub(crate) struct PendingDemote {
    pub(crate) index_relid: pg_sys::Oid,
    pub(crate) heap_relid: pg_sys::Oid,
}

pub(crate) fn pending_demotes() -> Vec<PendingDemote> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT index_relid::oid AS index_relid, heap_relid::oid AS heap_relid \
                 FROM pinecone.pending_demotes WHERE NOT queue_parked",
                None,
                &[],
            )
            .expect("pg_pinecone: pending_demotes scan failed")
            .map(|row| PendingDemote {
                index_relid: row
                    .get_by_name::<pg_sys::Oid, _>("index_relid")
                    .expect("pg_pinecone: pending_demotes index_relid query failed")
                    .expect("pg_pinecone: pending_demotes index_relid unexpectedly NULL"),
                heap_relid: row
                    .get_by_name::<pg_sys::Oid, _>("heap_relid")
                    .expect("pg_pinecone: pending_demotes heap_relid query failed")
                    .expect("pg_pinecone: pending_demotes heap_relid unexpectedly NULL"),
            })
            .collect()
    })
}

struct DemoteAttemptResult {
    done: bool,
    gone: bool,
    error: Option<String>,
}

fn attempt_demote(index_relid: pg_sys::Oid) -> DemoteAttemptResult {
    Spi::connect(|client| {
        let mut rows = client
            .select(
                "SELECT * FROM pinecone._demote_attempt($1)",
                None,
                &[index_relid.into()],
            )
            .expect("pg_pinecone: _demote_attempt call failed");
        let row = rows
            .next()
            .expect("pg_pinecone: _demote_attempt unexpectedly returned no rows");
        DemoteAttemptResult {
            done: row
                .get_by_name::<bool, _>("done")
                .expect("pg_pinecone: demote attempt done query failed")
                .expect("pg_pinecone: demote attempt done unexpectedly NULL"),
            gone: row
                .get_by_name::<bool, _>("gone")
                .expect("pg_pinecone: demote attempt gone query failed")
                .expect("pg_pinecone: demote attempt gone unexpectedly NULL"),
            error: row
                .get_by_name::<String, _>("error")
                .expect("pg_pinecone: demote attempt error query failed"),
        }
    })
}

fn purge_pending_demote(index_relid: pg_sys::Oid) {
    Spi::run_with_args(
        "DELETE FROM pinecone.pending_demotes WHERE index_relid::oid = $1::oid",
        &[index_relid.into()],
    )
    .expect("pg_pinecone: pending_demotes purge failed");
}

/// Mirrors `maintain.rs`'s `record_failure`: park after
/// `pinecone.outbox_max_attempts` consecutive failures (same GUC/vocabulary).
fn record_demote_failure(index_relid: pg_sys::Oid, message: &str) {
    let attempts = Spi::get_one_with_args::<i32>(
        "UPDATE pinecone.pending_demotes SET attempts = attempts + 1, last_error = $2 \
         WHERE index_relid::oid = $1::oid RETURNING attempts",
        &[index_relid.into(), message.into()],
    )
    .expect("pg_pinecone: pending_demotes failure record failed")
    .expect("pg_pinecone: pending_demotes RETURNING attempts unexpectedly NULL");

    if attempts >= crate::OUTBOX_MAX_ATTEMPTS.get() {
        Spi::run_with_args(
            "UPDATE pinecone.pending_demotes SET queue_parked = true, parked_reason = $2 \
             WHERE index_relid::oid = $1::oid",
            &[index_relid.into(), message.into()],
        )
        .expect("pg_pinecone: pending_demotes park failed");
    }
}

/// The remote coordinates captured for a follow-up cleanup HTTP call — from
/// the registry row BEFORE it's deleted (by cleanup time the row is gone).
/// [`demote_step`] builds one to persist a [`pinecone.pending_cleanups`] row;
/// [`pending_cleanups`] reads them back for [`remote_cleanup`] to retry.
pub(crate) struct RemoteCleanup {
    /// The `pinecone.pending_cleanups` row this came from (0 when freshly
    /// captured in `demote_step`, before its row exists) — the key
    /// [`remote_cleanup`] purges / records failure against.
    id: i64,
    index_name: String,
    namespace: Option<String>,
    /// Another registry row still uses `index_name` (at capture time) —
    /// empty only THIS namespace's vectors rather than deleting the whole
    /// index.
    shared: bool,
}

/// Phase 1 of one demote — everything TRANSACTIONAL, no HTTP: the
/// owner-identity `ALTER INDEX`/`REINDEX` (`_demote_attempt`), then — only
/// once the rebuild succeeded — the registry-row delete, the relid's
/// queued-outbox purge (a demoted table's stale queued rows must never
/// survive to replay against a registry row that no longer exists), the
/// pending-row purge, AND a `pinecone.pending_cleanups` row carrying the
/// captured remote coordinates, all in the CALLER's one transaction. Returns
/// whether the demote completed.
///
/// The split is what makes the crash/abort matrix sound: the rebuild + all
/// local bookkeeping commit FIRST (releasing the REINDEX's
/// `AccessExclusiveLock` before any network wait), and only THEN does any
/// remote deletion happen — an abort after the remote data is gone is
/// impossible, because nothing left in phase 2 can abort anything local.
///
/// The cleanup obligation is now a durable, retriable queue row rather
/// than a value handed straight to a single best-effort HTTP call. Enqueued
/// in THIS transaction (atomic with the pending-demote purge that removes the
/// old retry hook), it survives a worker crash or a Pinecone outage that
/// outlasts phase 2's HTTP retry window — [`remote_cleanup`] retries it every
/// round until the remote resource is confirmed gone. `cleanup` can
/// legitimately be `None` even on completion (no remote registration left to
/// clean, e.g. a concurrent DROP already purged the registry row).
pub(crate) fn demote_step(pd: &PendingDemote) -> bool {
    let result = attempt_demote(pd.index_relid);
    if result.gone {
        purge_pending_demote(pd.index_relid);
        return false;
    }
    if let Some(err) = result.error {
        record_demote_failure(pd.index_relid, &err);
        return false;
    }
    if !result.done {
        return false;
    }

    // Capture the remote coordinates BEFORE deleting the row they live on.
    let cleanup = registry::lookup(pd.heap_relid).and_then(|row| {
        row.index_name.clone().map(|index_name| RemoteCleanup {
            id: 0, // set on read-back; unused by the enqueue below
            shared: registry::other_row_uses_index(&index_name, pd.heap_relid),
            namespace: row.namespace.clone(),
            index_name,
        })
    });
    // A whole-index delete takes every namespace along; a shared index keeps the other tables',
    // so each namespace of a partitioned table is emptied on its own.
    let partitions = partition_registry::partitions_of(pd.heap_relid, Lane::Vector);
    if registry::lookup(pd.heap_relid).is_some() {
        registry::delete(pd.heap_relid);
    }
    partition_registry::forget_partitions(pd.heap_relid, Lane::Vector);
    // Queued outbox rows for the demoted relid die with
    // the registry row, in the same transaction — otherwise the worker's
    // next drain round dispatches them against a registry row that no
    // longer exists.
    outbox::clear_relid(pd.heap_relid);
    purge_pending_demote(pd.index_relid);
    // Persist the remote-cleanup obligation in THIS transaction — the
    // only retry hook (the pending-demote row) was just purged, so without
    // this a failed phase 2 would orphan the remote index forever.
    if let Some(cleanup) = cleanup {
        if cleanup.shared {
            for (_, namespace) in partitions {
                enqueue_cleanup(&RemoteCleanup {
                    id: 0,
                    index_name: cleanup.index_name.clone(),
                    namespace: Some(namespace),
                    shared: true,
                });
            }
        }
        enqueue_cleanup(&cleanup);
    }
    true
}

/// Persists one remote-cleanup obligation to `pinecone.pending_cleanups`.
fn enqueue_cleanup(cleanup: &RemoteCleanup) {
    Spi::run_with_args(
        "INSERT INTO pinecone.pending_cleanups (index_name, namespace, shared) VALUES ($1, $2, $3)",
        &[
            cleanup.index_name.clone().into(),
            cleanup.namespace.clone().into(),
            cleanup.shared.into(),
        ],
    )
    .expect("pg_pinecone: pending_cleanups insert failed");
}

/// The retriable remote cleanups still owed — one worker round's worth.
/// Skips parked rows (a permanent-class failure exhausted
/// `pinecone.outbox_max_attempts`); everything else is retried every round.
pub(crate) fn pending_cleanups() -> Vec<RemoteCleanup> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT id, index_name, namespace, shared FROM pinecone.pending_cleanups \
                 WHERE NOT queue_parked ORDER BY id",
                None,
                &[],
            )
            .expect("pg_pinecone: pending_cleanups scan failed")
            .map(|row| RemoteCleanup {
                id: row
                    .get_by_name::<i64, _>("id")
                    .expect("pg_pinecone: pending_cleanups id query failed")
                    .expect("pg_pinecone: pending_cleanups id unexpectedly NULL"),
                index_name: row
                    .get_by_name::<String, _>("index_name")
                    .expect("pg_pinecone: pending_cleanups index_name query failed")
                    .expect("pg_pinecone: pending_cleanups index_name unexpectedly NULL"),
                namespace: row
                    .get_by_name::<String, _>("namespace")
                    .expect("pg_pinecone: pending_cleanups namespace query failed"),
                shared: row
                    .get_by_name::<bool, _>("shared")
                    .expect("pg_pinecone: pending_cleanups shared query failed")
                    .expect("pg_pinecone: pending_cleanups shared unexpectedly NULL"),
            })
            .collect()
    })
}

/// Phase 2 of one demote — the remote cleanup HTTP call, nothing local: runs
/// in its OWN transaction (the worker's; needed anyway for the stored-key
/// SPI read), strictly AFTER phase 1's commit. Deletes the whole Pinecone
/// index, or (if it was shared at capture time) just this table's namespace.
///
/// RETRIABLE. The `pinecone.pending_cleanups` row is removed ONLY
/// once the remote resource is confirmed gone — a `200`/`202` delete, or a
/// `404` (`Error::IndexNotFound`) meaning it was already gone. A transient
/// failure (partition/timeout/5xx) is recorded but never counts toward
/// parking, so a Pinecone outage that outlasts any bounded HTTP retry window
/// is retried indefinitely by later worker rounds until it heals — this is
/// what closes the partition orphan (and the crash gap: a worker crash
/// between phase 1 and phase 2 just leaves the row for the next round). A
/// permanent-class failure (e.g. a 403 deletion-protection guard) accrues
/// `attempts` and parks after `pinecone.outbox_max_attempts`, since retrying it
/// forever cannot help. Returns whether the remote resource is now confirmed
/// gone (the row purged).
///
/// Stale-capture race: `cleanup.shared` was
/// captured inside `demote_step`'s transaction, but this call runs in a
/// LATER one — a table promoted onto the same `index_name` in between would
/// have its whole index deleted on the strength of that stale read. So the
/// registry is re-checked HERE, in this call's own transaction, and a new
/// claimant downgrades the deletion to namespace-only. One-directional by
/// design: a captured `shared = true` is never widened to a whole-index
/// delete even if the other sharer disappeared in the window — its own
/// demote's cleanup owns its namespace, and erring toward the smaller
/// deletion is the only safe default.
pub(crate) fn remote_cleanup(cleanup: &RemoteCleanup) -> bool {
    let client = backend::client_for_drain(backend::DrainCredentials::Worker);
    let shared_now = cleanup.shared || registry::any_row_uses_index(&cleanup.index_name);
    let result = if shared_now {
        client.delete_vectors(&pg_pinecone_client::DeleteVectorsRequest {
            index: cleanup.index_name.clone(),
            namespace: cleanup.namespace.clone(),
            ids: None,
            delete_all: true,
            filter: None,
        })
    } else {
        client.delete_index(&cleanup.index_name)
    };
    match result {
        // Gone: a real delete, or a 404 (whole-index path) meaning it already
        // was — either way the obligation is discharged.
        Ok(()) | Err(pg_pinecone_client::Error::IndexNotFound { .. }) => {
            purge_cleanup(cleanup.id);
            true
        }
        Err(e) if backend::is_transient(&e) => {
            // Partition/timeout/5xx: keep the row, do NOT count toward
            // parking — retried every round until Pinecone heals.
            record_cleanup_transient_failure(cleanup.id, &e.to_string());
            warning!(
                "pg_pinecone: demote_index remote cleanup for {} deferred (transient): {e} — the local \
                 demote is already committed; the worker will retry until it succeeds",
                cleanup.index_name
            );
            false
        }
        Err(e) => {
            // Permanent-class (auth/403 deletion-protection/bad request):
            // retrying cannot help, so accrue attempts and park after the max.
            record_cleanup_permanent_failure(cleanup.id, &e.to_string(), &cleanup.index_name);
            false
        }
    }
}

/// Removes a discharged cleanup obligation.
fn purge_cleanup(id: i64) {
    Spi::run_with_args(
        "DELETE FROM pinecone.pending_cleanups WHERE id = $1",
        &[id.into()],
    )
    .expect("pg_pinecone: pending_cleanups purge failed");
}

/// Records a transient cleanup failure — `last_error` only, never
/// `attempts` (mirrors `outbox::record_transient_failure`'s policy: blips
/// must never edge a row toward parking, so a long outage retries forever).
fn record_cleanup_transient_failure(id: i64, message: &str) {
    Spi::run_with_args(
        "UPDATE pinecone.pending_cleanups SET last_error = $2 WHERE id = $1",
        &[id.into(), message.into()],
    )
    .expect("pg_pinecone: pending_cleanups transient failure record failed");
}

/// Records a permanent-class cleanup failure — bumps `attempts` and parks
/// the row once it reaches `pinecone.outbox_max_attempts` (mirrors
/// `record_demote_failure`). A parked cleanup is skipped by
/// [`pending_cleanups`] and needs operator attention (the warned message).
fn record_cleanup_permanent_failure(id: i64, message: &str, index_name: &str) {
    let attempts = Spi::get_one_with_args::<i32>(
        "UPDATE pinecone.pending_cleanups SET attempts = attempts + 1, last_error = $2 \
         WHERE id = $1 RETURNING attempts",
        &[id.into(), message.into()],
    )
    .expect("pg_pinecone: pending_cleanups failure record failed")
    .expect("pg_pinecone: pending_cleanups RETURNING attempts unexpectedly NULL");

    if attempts >= crate::OUTBOX_MAX_ATTEMPTS.get() {
        Spi::run_with_args(
            "UPDATE pinecone.pending_cleanups SET queue_parked = true, parked_reason = $2 \
             WHERE id = $1",
            &[id.into(), message.into()],
        )
        .expect("pg_pinecone: pending_cleanups park failed");
        warning!(
            "pg_pinecone: demote_index remote cleanup for {index_name} parked after {attempts} \
             permanent-class failures: {message} — the orphaned remote resource needs manual \
             deletion"
        );
    }
}

/// Single-transaction driver over [`pending_demotes`] → [`demote_step`], then
/// [`pending_cleanups`] → [`remote_cleanup`] — the TEST-facing convenience (a
/// `#[pg_test]` has exactly one transaction to work with). **The worker does
/// NOT call this**: `worker.rs` drives these functions with one transaction
/// per step so the REINDEX's
/// `AccessExclusiveLock` never spans an HTTP call and an abort after remote
/// deletion is structurally impossible; in this combined form those two
/// hazards are accepted for the test transaction's convenience only. Returns
/// how many pending demotes completed.
#[cfg_attr(not(feature = "pg_test"), allow(dead_code))] // test-facing driver; worker uses the split fns
pub(crate) fn run_round() -> u32 {
    let mut completed = 0u32;
    for pd in pending_demotes() {
        if demote_step(&pd) {
            completed += 1;
        }
    }
    for cleanup in pending_cleanups() {
        remote_cleanup(&cleanup);
    }
    completed
}

// ---------------------------------------------------------------------------
// Engine surface (Level 1): pinecone.create_index / pinecone.set_engine
// ---------------------------------------------------------------------------

fn validate_engine(engine: &str, func: &str) {
    if !matches!(engine, "pgvector" | "native" | "remote") {
        error!(
            "{func}: engine '{engine}' is not supported — must be 'pgvector', 'native', or 'remote'"
        );
    }
}

fn opclass_for_metric(metric: &str, func: &str) -> &'static str {
    match metric {
        "cosine" => "vector_cosine_ops",
        "l2" => "vector_l2_ops",
        "ip" => "vector_ip_ops",
        other => {
            error!("{func}: metric '{other}' is not supported — must be 'cosine', 'l2', or 'ip'")
        }
    }
}

/// `create_index`'s `filter_column`: only the engines that keep a per-record key can carry one,
/// and an equality operator class has to exist for its type.
fn require_filter_column(relid: pg_sys::Oid, display: &str, engine: &str, column: &str) {
    if engine == "pgvector" {
        error!(
            "pinecone.create_index: filter_column only applies to engine 'native' or 'remote', got \
             '{engine}' — an hnsw index takes one vector column"
        );
    }
    let column_type = catalog::column_type(relid, column).unwrap_or_else(|| {
        error!("pinecone.create_index: filter column \"{column}\" not found on {display}")
    });
    if metapage::FilterKind::from_typname(&column_type).is_none() {
        error!(
            "pinecone.create_index: filter column \"{column}\" of {display} has type {column_type} \
             — a pg_pinecone filter column must be smallint, integer, bigint, text, or varchar"
        );
    }
}

/// `partition_column` makes one Pinecone namespace per value, which only a fresh remote index
/// can; on a native index the column is just its filter key.
fn require_partition_engine(func: &str, engine: &str, pinecone_index: Option<&str>) {
    if engine == "pgvector" {
        backend::error_with_code(
            PgSqlErrorCode::ERRCODE_FEATURE_NOT_SUPPORTED,
            format!(
                "{func}: partition_column does not apply to engine 'pgvector' — an hnsw index \
                 takes one vector column"
            ),
            None,
            Some("use engine => 'remote' for a Pinecone namespace per value"),
        );
    }
    if pinecone_index.is_some() {
        backend::error_with_code(
            PgSqlErrorCode::ERRCODE_FEATURE_NOT_SUPPORTED,
            format!(
                "{func}: partition_column cannot be combined with pinecone_index — an attached \
                 namespace holds one table's rows, not one namespace per value"
            ),
            None,
            Some(
                "create the partitioned index without pinecone_index: its backfill writes the \
                 <schema>.<table>:<value> namespaces",
            ),
        );
    }
}

/// A safe, deterministic Postgres index-name default: `<table>_<col>_pinecone_idx`,
/// truncated to fit Postgres's 63-byte identifier limit (NAMEDATALEN - 1).
fn default_pg_index_name(table: &str, col: &str) -> String {
    let raw = format!("{table}_{col}_pinecone_idx");
    if raw.len() <= 63 {
        raw
    } else {
        raw.chars().take(63).collect()
    }
}

/// Implements `pinecone.create_index(tbl, col, engine, metric, index_name,
/// pinecone_index, pinecone_namespace, attach_unchecked, filter_column,
/// partition_column)` — see
/// this module's own docs for the deliberately minimal v1 scope, and the "attach"
/// semantics on `pinecone_index`/`pinecone_namespace`/`attach_unchecked`
/// (only meaningful for `engine = 'remote'`). `filter_column` (engines `native`
/// and `remote`) builds `(col opclass, filter_column)`: the one equality key a
/// scan pushes down (remote: as a Pinecone metadata filter). `partition_column` (engine `remote`)
/// makes that key a namespace per value; on `native` it is the filter key.
#[allow(clippy::too_many_arguments)]
pub(crate) fn create_index(
    tbl: PgRelation,
    col: &str,
    engine: &str,
    metric: &str,
    index_name: Option<&str>,
    pinecone_index: Option<&str>,
    pinecone_namespace: Option<&str>,
    attach_unchecked: bool,
    filter_column: Option<&str>,
    partition_column: Option<&str>,
) {
    validate_engine(engine, "pinecone.create_index");
    let opclass = opclass_for_metric(metric, "pinecone.create_index");
    let relid = tbl.oid();
    let display = catalog::display_name(&tbl);

    if pinecone_index.is_some() && engine != "remote" {
        error!(
            "pinecone.create_index: pinecone_index only applies to engine 'remote', got '{engine}'"
        );
    }

    // One engine-surface-managed index per table — a
    // second create_index against a table that already has one (a live
    // pg_pinecone-AM index in any mode, or an engine-stamped registry row) must
    // reject clearly, naming the switch verb, rather than stacking a second
    // physical index whose bookkeeping would fight the first's.
    if let Some(current) = current_engine(relid) {
        error!(
            "pinecone.create_index: {display} is already managed with engine '{current}' — use \
             pinecone.set_engine('{display}', '{engine}') to switch, or DROP the existing index \
             first"
        );
    }

    let column_type = catalog::column_type(relid, col).unwrap_or_else(|| {
        error!("pinecone.create_index: column \"{col}\" not found on {display}")
    });
    if column_type != "vector" {
        error!(
            "pinecone.create_index: column \"{col}\" of {display} has type {column_type}, not vector"
        );
    }

    let schema = tbl.namespace().to_string();
    let table = tbl.name().to_string();
    if partition_column.is_some() {
        require_partition_engine("pinecone.create_index", engine, pinecone_index);
    }
    let partition = partition_column
        .filter(|_| engine == "remote")
        .map(|column| {
            let index = promote::default_index_name(&schema, &table);
            am_partition::declare(
                relid,
                &schema,
                &table,
                column,
                &index,
                "pinecone.create_index",
            )
        });
    let key_column = match (filter_column, partition_column) {
        (Some(filter), Some(partition)) if filter != partition => backend::error_with_code(
            PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
            "pinecone.create_index: filter_column and partition_column name different columns — \
             an index has one second key",
            None,
            Some("drop filter_column: the partition column is the key a query pins"),
        ),
        (filter, partition) => filter.or(partition),
    };
    if let Some(key) = key_column.filter(|_| partition.is_none()) {
        require_filter_column(relid, &display, engine, key);
    }
    if partition_column.is_some() && partition.is_none() {
        notice!(
            "pinecone.create_index: partition_column on engine '{engine}' is the filter key; \
             pinecone.set_engine(.., 'remote', partition_column => ..) makes it a namespace per value"
        );
    }
    let filter_key = key_column
        .map(|c| format!(", {}", pgrx::spi::quote_identifier(c)))
        .unwrap_or_default();

    let table_ident = pgrx::spi::quote_qualified_identifier(&schema, &table);
    let col_ident = pgrx::spi::quote_identifier(col);
    let name = index_name
        .map(str::to_string)
        .unwrap_or_else(|| default_pg_index_name(&table, col));
    let name_ident = pgrx::spi::quote_identifier(&name);

    // `tbl` must NOT still be open when the `CREATE INDEX` statement below
    // runs against the very same relation: Postgres's `CheckTableNotInUse`
    // refuses `CREATE INDEX`/most DDL on a relation this session itself still
    // has open (the `#[pg_extern]`-bound `PgRelation` argument holds it open
    // for as long as the value lives) — "cannot CREATE INDEX ... because it
    // is being used by active queries in this session", hit empirically
    // while testing this function. Every field this function still needs
    // from `tbl` is already extracted above; dropping it here releases that
    // hold before the DDL.
    drop(tbl);

    if let Some(pinecone_index) = pinecone_index {
        attach_remote(
            relid,
            col,
            metric,
            &schema,
            &table,
            &table_ident,
            pinecone_index,
            pinecone_namespace,
            attach_unchecked,
            &display,
            key_column,
        );
    }

    match engine {
        "pgvector" => {
            let ns = catalog::pgvector_namespace().unwrap_or_else(|e| {
                error!("pinecone.create_index: engine 'pgvector' requires pgvector — {e}")
            });
            let opclass_ident = pgrx::spi::quote_qualified_identifier(ns.as_str(), opclass);
            Spi::run(&format!(
                "CREATE INDEX {name_ident} ON {table_ident} USING hnsw ({col_ident} {opclass_ident})"
            ))
            .unwrap_or_else(|e| error!("pinecone.create_index: {e}"));
        }
        "native" => {
            Spi::run(&format!(
                "CREATE INDEX {name_ident} ON {table_ident} USING pinecone \
                 ({col_ident} {opclass}{filter_key}) WITH (mode = 'local')"
            ))
            .unwrap_or_else(|e| error!("pinecone.create_index: {e}"));
        }
        "remote" => {
            if !secret::has_stored_api_key() {
                error!(
                    "pinecone.create_index: engine 'remote' needs a stored API key — run \
                     pinecone.set_api_key(...) first"
                );
            }
            Spi::run(&format!(
                "CREATE INDEX {name_ident} ON {table_ident} USING pinecone \
                 ({col_ident} {opclass}{filter_key}) WITH (mode = 'remote')"
            ))
            .unwrap_or_else(|e| error!("pinecone.create_index: {e}"));
            if let Some(spec) = &partition {
                declare_partition(relid, spec);
            }
        }
        _ => unreachable!("validate_engine already rejected anything else"),
    }

    // NOT called for `engine == "native"`: `current_engine`'s own lookup
    // finds a native (local-mode) pg_pinecone-AM index directly via
    // `catalog::pg_pinecone_am_index_for_table` + its metapage, with no registry
    // row needed at all — and `lifecycle::promote_index`'s own precondition
    // (a genuinely local-mode AM index has NO `pinecone.tables` row yet — see
    // `local::build`'s invariant) would otherwise conflict with a bookkeeping
    // row this function planted for it. `pgvector` has no such catalog
    // signal (a plain index looks like any other to `pinecone.set_engine`), so
    // IT still needs one; `remote` already gets one from `am::ambuild`'s own
    // insert (attach or fresh) — this just stamps its `engine` column.
    if engine != "native" {
        ensure_engine_row(relid, col, metric, engine, &display);
    }
}

/// Stores the partition of the registry row `ambuild` just made for `relid`.
fn declare_partition(relid: pg_sys::Oid, spec: &PartitionSpec) {
    if registry::lookup(relid).is_none() {
        error!(
            "pinecone.create_index: CREATE INDEX left no pinecone.tables row for relid {relid:?} \
             — should be unreachable"
        );
    }
    registry::set_partition(relid, Some(spec));
}

/// Validates and pre-registers an attach of `pinecone_index` for `relid`
/// — `am::ambuild`'s own new dispatch arm (see its rustdoc) then just stamps
/// `am_index_relid` on the row this inserts, once the `CREATE INDEX` this
/// function's caller is about to run reaches it. Reuses
/// `registry::index_namespace_owner`'s collision check so two tables can't
/// attach to the same index/namespace pair.
#[allow(clippy::too_many_arguments)]
fn attach_remote(
    relid: pg_sys::Oid,
    col: &str,
    metric: &str,
    schema: &str,
    table: &str,
    table_ident: &str,
    pinecone_index: &str,
    pinecone_namespace: Option<&str>,
    attach_unchecked: bool,
    display: &str,
    filter_column: Option<&str>,
) {
    if !secret::has_stored_api_key() {
        error!(
            "pinecone.create_index: attaching an existing index needs a stored API key — run \
             pinecone.set_api_key(...) first"
        );
    }
    let namespace = pinecone_namespace
        .map(str::to_string)
        .unwrap_or_else(|| format!("{schema}.{table}"));

    let claim = Claim::Exact(&namespace);
    if let Some(owner) = registry::claim_owner(pinecone_index, claim, relid) {
        partition_registry::refuse_family_clash(
            "pinecone.create_index",
            pinecone_index,
            claim,
            &owner,
        );
        let owner_display = catalog::display_name_of(owner.relid);
        error!(
            "pinecone.create_index: index '{pinecone_index}' namespace '{namespace}' is already \
             owned by {owner_display} — pass a different pinecone_namespace"
        );
    }

    let id_column = am::single_primary_key_column(relid, display);
    let dimension = catalog::vector_dimension(relid, col);
    let mapped_metric = promote::map_metric(metric);
    let local_row_count = Spi::get_one::<i64>(&format!("SELECT count(*) FROM {table_ident}"))
        .expect("pinecone.create_index: row-count query failed")
        .expect("pinecone.create_index: count(*) unexpectedly NULL");

    let client = backend::client_for_effective_key();
    let read_capacity = promote::attach_index(
        &client,
        pinecone_index,
        dimension as u32,
        mapped_metric,
        &namespace,
        local_row_count,
        attach_unchecked,
        display,
        filter_column,
    );

    registry::insert(&RegistryRow {
        relid,
        vector_column: col.to_string(),
        id_column,
        metric: metric.to_string(),
        dimension,
        metadata_columns: Vec::new(),
        mode: "remote".to_string(),
        sync_mode: "async".to_string(),
        synced_by: "am".to_string(),
        queue_parked: false,
        parked_reason: None,
        index_name: Some(pinecone_index.to_string()),
        namespace: Some(namespace),
        am_index_relid: None,
        last_vacuum_error: None,
        engine: Some("remote".to_string()),
        storage: "full".to_string(),
        detached_at: None,
        detach_backup_id: None,
        backup_gate: None,
        detach_rpo_secs: None,
        read_capacity,
        attached: true,
        partition: None,
        filter_column: filter_column.map(str::to_string),
    });
}

/// Ensures `pinecone.tables` has a row for `relid` with `engine` recorded —
/// inserting a minimal one (for `pgvector`, which never had a registry
/// concept at all) or updating the existing AM-inserted one in place (for
/// `remote`, whose `ambuild` already created a full row — see
/// `am::ambuild`'s remote branch — this just stamps its `engine` column).
/// Never called for `engine == "native"` — see the call site's own comment.
fn ensure_engine_row(relid: pg_sys::Oid, col: &str, metric: &str, engine: &str, display: &str) {
    if let Some(existing) = registry::lookup(relid) {
        registry::set_engine(relid, Some(engine));
        let _ = existing; // only the engine column changes here
        return;
    }
    let id_column = am::single_primary_key_column(relid, display);
    let dimension = catalog::vector_dimension(relid, col);
    registry::insert(&RegistryRow {
        relid,
        vector_column: col.to_string(),
        id_column,
        metric: metric.to_string(),
        dimension,
        metadata_columns: Vec::new(),
        mode: "local".to_string(),
        sync_mode: "async".to_string(),
        synced_by: "engine".to_string(),
        queue_parked: false,
        parked_reason: None,
        index_name: None,
        namespace: None,
        am_index_relid: None,
        last_vacuum_error: None,
        engine: Some(engine.to_string()),
        // `synced_by = 'engine'` — likewise never detachable (schema-level
        // `tables_storage_synced_by_check`).
        storage: "full".to_string(),
        detached_at: None,
        detach_backup_id: None,
        backup_gate: None,
        detach_rpo_secs: None,
        read_capacity: None,
        attached: false,
        partition: None,
        filter_column: None,
    });
}

/// The current engine backing `heap_relid`'s vector column, per this
/// module's own bookkeeping — `None` if nothing the engine surface
/// recognizes governs it yet (a table `pinecone.create_index` has never touched,
/// and no pg_pinecone-AM index exists on it either).
fn current_engine(heap_relid: pg_sys::Oid) -> Option<&'static str> {
    if let Some(am_index) = catalog::pg_pinecone_am_index_for_table(heap_relid) {
        let idx =
            unsafe { PgRelation::with_lock(am_index, pg_sys::AccessShareLock as pg_sys::LOCKMODE) };
        let meta = metapage::read(idx.as_ptr());
        return Some(match meta.mode {
            Mode::Local | Mode::Promoting => "native",
            Mode::Remote | Mode::Demoting | Mode::Detached => "remote",
        });
    }
    match registry::lookup(heap_relid).and_then(|r| r.engine) {
        Some(e) if e == "pgvector" => Some("pgvector"),
        _ => None,
    }
}

/// A repeated `set_engine(.., 'remote', partition_column => c)` passes when the table is
/// already partitioned by `c`; any other pairing would silently ignore the declaration.
fn require_declared_partition(relid: pg_sys::Oid, display: &str, column: Option<&str>) {
    let declared = registry::lookup(relid).and_then(|row| row.partition);
    if declared.as_ref().map(|p| p.column.as_str()) != column {
        backend::error_with_code(
            PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
            format!(
                "pinecone.set_engine: {display} is already on that engine {}, so partition_column \
                 cannot change it",
                declared.map_or("unpartitioned".to_string(), |p| format!(
                    "partitioned by \"{}\"",
                    p.column
                )),
            ),
            None,
            Some(
                "a partition is declared when the index goes remote: set_engine(.., 'native') first",
            ),
        );
    }
}

/// Implements `pinecone.set_engine(idx_or_tbl, engine, pinecone_index,
/// pinecone_namespace, attach_unchecked, partition_column)` — the one-verb switch.
/// Accepts either a table or an index regclass; resolves to the underlying
/// table either way. See this module's own docs for exactly which
/// transitions this delivers vs. names as unsupported.
/// `pinecone_index`/`pinecone_namespace`/`attach_unchecked` only apply to a
/// `'native' -> 'remote'` transition (attach semantics); ignored (but
/// not rejected — a no-op transition is a no-op regardless) otherwise.
pub(crate) fn set_engine(
    target: PgRelation,
    engine: &str,
    pinecone_index: Option<&str>,
    pinecone_namespace: Option<&str>,
    attach_unchecked: bool,
    partition_column: Option<&str>,
) {
    validate_engine(engine, "pinecone.set_engine");
    let heap_relid = if target.is_index() {
        catalog::index_heap_relid(target.oid())
            .unwrap_or_else(|| error!("pinecone.set_engine: could not resolve the indexed table"))
    } else {
        target.oid()
    };
    let display = catalog::resolve_table_name(heap_relid)
        .map(|(s, t)| format!("{s}.{t}"))
        .unwrap_or_else(|| format!("<dropped:{heap_relid}>"));
    // Gate here too (not only inside the routed verbs): a no-op/unsupported
    // transition must be just as owner-gated as a real one.
    require_table_owner(heap_relid, &display, "pinecone.set_engine");

    let Some(current) = current_engine(heap_relid) else {
        error!(
            "pinecone.set_engine: {display} has no engine-surface-managed index yet — run \
             pinecone.create_index(...) first"
        );
    };
    if current == engine {
        if partition_column.is_some() {
            require_declared_partition(heap_relid, &display, partition_column);
        }
        return; // already there — a no-op, not an error.
    }
    if partition_column.is_some() && (current, engine) != ("native", "remote") {
        backend::error_with_code(
            PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
            format!(
                "pinecone.set_engine: partition_column only applies to a 'native' -> 'remote' \
                 transition, got '{current}' -> '{engine}'"
            ),
            None,
            None,
        );
    }
    if pinecone_index.is_some() && (current, engine) != ("native", "remote") {
        error!(
            "pinecone.set_engine: pinecone_index only applies to a 'native' -> 'remote' \
             transition, got '{current}' -> '{engine}'"
        );
    }

    match (current, engine) {
        ("native", "remote") => {
            let am_index = catalog::pg_pinecone_am_index_for_table(heap_relid)
                .expect("current_engine already confirmed a pg_pinecone-AM index exists");
            let idx = unsafe {
                PgRelation::with_lock(am_index, pg_sys::AccessShareLock as pg_sys::LOCKMODE)
            };
            promote_index(
                idx,
                pinecone_index,
                pinecone_namespace,
                attach_unchecked,
                partition_column,
            );
        }
        ("remote", "native") => {
            let am_index = catalog::pg_pinecone_am_index_for_table(heap_relid)
                .expect("current_engine already confirmed a pg_pinecone-AM index exists");
            let idx = unsafe {
                PgRelation::with_lock(am_index, pg_sys::AccessShareLock as pg_sys::LOCKMODE)
            };
            demote_index(idx);
        }
        ("pgvector", "native") => {
            // Closed before the DDL on it (`CheckTableNotInUse`), as in `create_index`.
            drop(target);
            pgvector_to_native(heap_relid, &display);
        }
        ("pgvector", "remote") | ("remote", "pgvector") => error!(
            "pinecone.set_engine: switching {display} from '{current}' to '{engine}' is not \
             supported yet — set_engine('{display}', 'native') first"
        ),
        _ => error!(
            "pinecone.set_engine: switching {display} from '{current}' to '{engine}' is not \
             supported yet — DROP INDEX the native index, then pinecone.create_index(.., engine \
             => '{engine}')"
        ),
    }
}

/// `pgvector -> native`: builds the native index while reads keep the hnsw one, then drops the
/// hnsw index, gives its name to the native one and deletes the registry row (a native index has
/// none). Only for the row `create_index` made: one `track` or `promote` also owns (or a demoted
/// one that still names its Pinecone index) is refused.
fn pgvector_to_native(heap_relid: pg_sys::Oid, display: &str) {
    // Writes wait and reads go on, as under CREATE INDEX; a second set_engine waits here instead
    // of deadlocking at the DROP INDEX.
    unsafe {
        pg_sys::LockRelationOid(
            heap_relid,
            pg_sys::ShareRowExclusiveLock as pg_sys::LOCKMODE,
        )
    };
    let Some(row) =
        registry::lookup(heap_relid).filter(|r| r.engine.as_deref() == Some("pgvector"))
    else {
        error!("pinecone.set_engine: {display} changed meanwhile — run it again");
    };
    if row.synced_by != "engine" || row.mode != "local" || row.index_name.is_some() {
        error!(
            "pinecone.set_engine: {display} is also registered by pinecone.track or promote \
             (synced_by '{}', mode '{}') — demote and untrack it, DROP INDEX the hnsw index, then \
             pinecone.create_index(..)",
            row.synced_by, row.mode
        );
    }
    if catalog::relkind(heap_relid) == Some('p') {
        error!(
            "pinecone.set_engine: {display} is partitioned — moving its hnsw index is not supported"
        );
    }
    if catalog::column_type(heap_relid, &row.vector_column).is_none() {
        error!(
            "pinecone.set_engine: column \"{}\" not found on {display}",
            row.vector_column
        );
    }
    let (schema, table) = catalog::resolve_table_name(heap_relid)
        .unwrap_or_else(|| error!("pinecone.set_engine: could not resolve {display}"));
    let hnsw = catalog::hnsw_index_names(heap_relid, &row.vector_column);
    if hnsw.len() > 1 {
        error!(
            "pinecone.set_engine: {display} has {} hnsw indexes on \"{}\" ({}) — drop all but one \
             first",
            hnsw.len(),
            row.vector_column,
            hnsw.join(", ")
        );
    }
    let opclass = opclass_for_metric(&row.metric, "pinecone.set_engine");
    let run = |sql: String| Spi::run(&sql).unwrap_or_else(|e| error!("pinecone.set_engine: {e}"));
    run(format!(
        "CREATE INDEX ON {} USING pinecone ({} {opclass}) WITH (mode = 'local')",
        pgrx::spi::quote_qualified_identifier(&schema, &table),
        pgrx::spi::quote_identifier(&row.vector_column)
    ));
    let built = catalog::pg_pinecone_am_index_for_table(heap_relid)
        .and_then(catalog::resolve_table_name)
        .map(|(_, name)| name)
        .expect("the CREATE INDEX above built it");
    let name = match hnsw.into_iter().next() {
        Some(old) => {
            run(format!(
                "DROP INDEX {}",
                pgrx::spi::quote_qualified_identifier(&schema, &old)
            ));
            old
        }
        None => default_pg_index_name(&table, &row.vector_column),
    };
    if built != name {
        run(format!(
            "ALTER INDEX {} RENAME TO {}",
            pgrx::spi::quote_qualified_identifier(&schema, &built),
            pgrx::spi::quote_identifier(&name)
        ));
    }
    // As the extension's owner, like `_on_sql_drop` (which usually took the row already): the
    // caller owns the table and needs no grant on the registry for it.
    crate::docs_registry::as_owner(|| registry::delete(heap_relid));
}
