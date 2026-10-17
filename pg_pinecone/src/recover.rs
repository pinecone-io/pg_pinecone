//! Implementation of `pinecone.recover_from_remote` — the standalone DR
//! primitive, independent of the native engine and detached storage:
//! restores a promoted table's vector column (and whatever metadata the
//! remote namespace carries for a row) from whatever Pinecone index it's
//! currently registered against. It is NOT a backup restore itself — see
//! `docs/disaster-recovery.md` for the full restore-to-new-index ->
//! `pinecone.repoint` -> `recover_from_remote` story, and the `pg_pinecone.
//! recover_from_remote` rustdoc in `lib.rs` for the RPO-honesty caveats
//! verbatim (never soften those: restores only what the remote currently
//! holds; RPO = backup-schedule interval + 15 min after a backup restore,
//! never zero; recover only fixes vector damage that hasn't yet propagated
//! to Pinecone).
//!
//! Mirrors `promote.rs`'s `'backfill'` job almost exactly (same "one chunk
//! per drain round, cursor persisted in the outbox payload, resumable across
//! worker restarts" shape — `outbox::dispatch_recover` is this module's
//! `outbox::dispatch_backfill` counterpart), but reads FROM Pinecone and
//! writes INTO the local table instead of the other way around, which is
//! where its three extra soundness requirements come from:
//!
//! 1. **NULL guard** (`synced_by = 'trigger'`): `UPDATE ... WHERE vcol IS
//!    NULL` unless `overwrite => true` — required even on an otherwise
//!    healthy, non-detached table: without it, a recover job replaying a
//!    stale Pinecone snapshot could clobber a fresher local write racing it.
//! 2. **Trigger suppression**: the local `UPDATE` below fires
//!    `pinecone._sync_trigger` like any other write — without suppression, that
//!    trigger would re-enqueue the very value this job just pulled FROM
//!    Pinecone right back into `pinecone.outbox`, a pure round-trip. The
//!    mechanism is [`RECOVERING_RELID`], a per-backend Rust static read by
//!    `pinecone._is_recovering(relid)` (a C function `_sync_trigger` calls
//!    first thing) — see the static's own rustdoc for why it must NOT be a
//!    GUC, and why it's scoped to the ONE relid being recovered.
//! 3. **`synced_by = 'am'` miss detection**: remote ids are heap TIDs there,
//!    which a concurrent non-HOT write can silently relocate — unlike the
//!    `'trigger'` path's stable id column, a mismatch between fetched and
//!    updated TIDs can't be resolved without quiescence. What happens to a
//!    missed id is [`MissPolicy`]'s call: `'park'` (default) fails the chunk
//!    with a clear pause-app-writes-and-retry error; `'quarantine'` copies
//!    the fetched bytes into `pinecone.recover_quarantine` and advances — the
//!    escape hatch for genuinely relocated rows, which would otherwise miss
//!    again on every retry forever (the parked-on-misses dead-end).
//!    `VACUUM` is hostile to a running or parked TID recover job
//!    (`ambulkdelete` would otherwise reap exactly the ids the job is still
//!    trying to restore — destroying the only copy of any relocated row
//!    whose local vector is NULL).
//! 4. **Partitioned tables** walk one namespace per value: the job reads the
//!    registered namespaces (`pinecone.partitions`) in order, keeps the one it is
//!    in beside the page token ([`Walking`]), and both write-backs match only rows
//!    of that namespace's partition, so an id or TID that belongs to another
//!    tenant's row by now is a ghost or a miss, never a vector written onto that
//!    row. The namespace decides the partition: remote metadata never writes the
//!    partition column.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicU32, Ordering},
};

use pg_pinecone_client::{
    Client, Error as ClientError, FetchVectorsRequest, ListVectorIdsPageRequest, Vector,
};
use pgrx::{
    JsonB, PgRelation, datum::DatumWithOid, pg_sys::submodules::panic::CaughtError, prelude::*,
    spi::quote_qualified_identifier,
};
use serde_json::{Value, json};

use crate::{
    catalog,
    filter::ColKind,
    outbox,
    partition::Lane,
    partition_lane, partition_registry,
    registry::{self, RegistryRow},
};

extension_sql!(
    r#"
    -- Where `pinecone.recover_from_remote(..., on_miss => 'quarantine')` parks
    -- the remote copies of TID-keyed vectors whose local row can no longer be
    -- found (a concurrent non-HOT write relocated the ctid mid-job — the
    -- parked-on-misses dead-end): the job advances past the chunk instead of
    -- parking forever, and the bytes survive the VACUUM that would otherwise
    -- remote-delete the only copy. `embedding` is real[] (float4[]) written
    -- straight from the fetched f32 datums — no text round-trip, so the
    -- quarantined bytes are exactly what Pinecone returned. `remote_id` is
    -- the STALE heap position ("block:offset"); it no longer identifies a
    -- live row, so re-attaching is a manual, application-knowledge join —
    -- see docs/disaster-recovery.md. `pg_extension_config_dump`: unlike
    -- pinecone.outbox (a queue), quarantined rows may be the ONLY copy of a
    -- vector, so they must survive dump/restore.
    CREATE TABLE pinecone.recover_quarantine (
        relid regclass NOT NULL,
        remote_id text NOT NULL,
        embedding real[] NOT NULL,
        metadata jsonb,
        quarantined_at timestamptz NOT NULL DEFAULT now()
    );
    REVOKE ALL ON pinecone.recover_quarantine FROM PUBLIC;
    SELECT pg_catalog.pg_extension_config_dump('pinecone.recover_quarantine', '');
    "#,
    name = "pg_pinecone_recover_quarantine",
    requires = [pinecone],
);

/// The relid THIS backend is currently applying recover write-backs to
/// (`0` = `InvalidOid` = none) — read by `pinecone._is_recovering(relid)`,
/// which `pinecone._sync_trigger`'s plpgsql body calls first thing so a recover
/// write never re-enqueues the value it just pulled FROM Pinecone.
///
/// A per-backend Rust static behind a C function, deliberately NOT a GUC: a
/// GUC checked by name from plpgsql is spoofable by a session that never
/// loaded the library (no `shared_preload_libraries`, so `_PG_init` and its
/// `MarkGUCPrefixReserved` haven't run), letting a writer skip the outbox.
/// Calling `pinecone._is_recovering()` forces the library to load, and this
/// static can only be set by [`RecoveringGuard`] — this module's own
/// write-back path — never from SQL.
///
/// Holds the ONE relid being recovered (not a boolean): a user trigger on
/// the recovered table cascading a write to a DIFFERENT promoted table
/// mid-recover must still sync that other table — only
/// the recovered relid's own trigger is suppressed. Atomic purely so safe
/// Rust can hold it in a static; a Postgres backend is single-threaded, so
/// `Relaxed` ordering is sufficient.
static RECOVERING_RELID: AtomicU32 = AtomicU32::new(0);

/// Implements `pinecone._is_recovering(relid)` — see [`RECOVERING_RELID`].
pub(crate) fn is_recovering(relid: pg_sys::Oid) -> bool {
    let current = RECOVERING_RELID.load(Ordering::Relaxed);
    current != 0 && current == relid.to_u32()
}

/// RAII scope for [`RECOVERING_RELID`]: set on construction, cleared on
/// [`Drop`] — which also runs during a Rust-panic unwind (how pgrx surfaces
/// a Postgres `ERROR` raised mid-write-back), so a failed chunk can never
/// leave suppression stuck on for the rest of the backend's life. Rollback
/// needs no special handling at all: the static is per-backend memory, not
/// transactional state, and the guard's lifetime is exactly the write-back
/// statement's.
struct RecoveringGuard;

impl RecoveringGuard {
    fn set(relid: pg_sys::Oid) -> Self {
        RECOVERING_RELID.store(relid.to_u32(), Ordering::Relaxed);
        Self
    }
}

impl Drop for RecoveringGuard {
    fn drop(&mut self) {
        RECOVERING_RELID.store(0, Ordering::Relaxed);
    }
}

/// Switches the executing role to the recovered table's OWNER, with
/// `SECURITY_RESTRICTED_OPERATION` set, for as long as this guard lives —
/// wrapped around each chunk's write-back `UPDATE` (CVE-2022-1552 class): that
/// statement runs arbitrary user code — the table's own triggers, `CHECK`
/// constraints, index expressions — and the
/// worker is effectively superuser, so without the switch a hostile table
/// owner's trigger escalates to superuser the moment an operator recovers
/// that owner's table. Switched to the owner, that code runs with exactly
/// the privileges its author already has; `SECURITY_RESTRICTED_OPERATION`
/// (the same flag `VACUUM`/`ANALYZE`/`REINDEX` set before running
/// user-owned code) additionally forbids it from re-switching identity
/// (`SET ROLE`/`SET SESSION AUTHORIZATION`) for the duration.
///
/// Restore happens in [`Drop`] — which runs during the Rust-panic unwind
/// pgrx surfaces a mid-statement `ERROR` as — and the enclosing
/// [`run_write_back_subtransaction`]'s abort path independently restores it
/// too (Postgres's own `AbortSubTransaction` resets the userid/sec-context
/// to the subtransaction's entry state), so no error path can leak the
/// switched identity.
///
/// Unlike `am.rs`'s `SecurityDefinerGuard` (which this mirrors), no
/// `RestrictSearchPath`: that guard ELEVATES to a definer and must shield
/// the elevated statements from the caller's search_path; this one
/// DE-escalates — the composed write-back SQL resolves under the worker's
/// own (trusted) session search_path, only its *execution identity* drops
/// to the owner — and clamping search_path here would break both the
/// unqualified `::vector` cast in the chunk SQL and any user trigger body
/// that references the owner's own tables unqualified.
struct TableOwnerGuard {
    saved_uid: pg_sys::Oid,
    saved_sec_context: i32,
}

impl TableOwnerGuard {
    /// # Safety
    /// Must be called on a live backend, inside a transaction — exactly the
    /// worker drain round's dispatch context.
    unsafe fn switch_to(owner: pg_sys::Oid) -> Self {
        unsafe {
            let mut saved_uid = pg_sys::Oid::INVALID;
            let mut saved_sec_context = 0i32;
            pg_sys::GetUserIdAndSecContext(&mut saved_uid, &mut saved_sec_context);
            pg_sys::SetUserIdAndSecContext(
                owner,
                saved_sec_context | pg_sys::SECURITY_RESTRICTED_OPERATION as i32,
            );
            Self {
                saved_uid,
                saved_sec_context,
            }
        }
    }
}

impl Drop for TableOwnerGuard {
    fn drop(&mut self) {
        unsafe {
            pg_sys::SetUserIdAndSecContext(self.saved_uid, self.saved_sec_context);
        }
    }
}

/// The owning role of `tbl`, straight off the open relcache entry — the
/// identity [`TableOwnerGuard`] drops each chunk's write-back to.
fn table_owner(tbl: &PgRelation) -> pg_sys::Oid {
    // # Safety: `tbl` is a live, locked relation for the duration of the
    // dispatch (`outbox::dispatch_recover` holds its `AccessShareLock`).
    unsafe { (*(*tbl.as_ptr()).rd_rel).relowner }
}

/// pgvector's `vector` type, schema-qualified for the chunk SQL's cast —
/// resolved by name against `search_path` otherwise, which is not guaranteed
/// to include pgvector's schema in every context this SQL can run under
/// (the worker's session default usually does, but nothing forces that).
/// Panics if pgvector is somehow absent — unreachable for a promoted table
/// with a live `vector` column, and inside a chunk the panic is caught by
/// [`run_write_back_subtransaction`] and parks the job with the message.
fn qualified_vector_type() -> String {
    let ns = catalog::pgvector_namespace()
        .unwrap_or_else(|e| panic!("pg_pinecone: recover chunk could not resolve pgvector: {e}"));
    format!("{}.vector", pgrx::spi::quote_identifier(&ns))
}

/// Whether a remote id with no matching local row (a "ghost", `synced_by =
/// 'trigger'` path only — see this module's rustdoc for why the `'am'` path
/// can't distinguish a ghost from a miss) is only counted, or also enqueued
/// for remote deletion.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GhostPolicy {
    Report,
    Delete,
}

impl GhostPolicy {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "report" => Some(Self::Report),
            "delete" => Some(Self::Delete),
            _ => None,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Report => "report",
            Self::Delete => "delete",
        }
    }
}

/// What a `synced_by = 'am'` chunk does when fetched TIDs have no matching
/// local row (a concurrent non-HOT write relocated them — this module's
/// rustdoc, point 3):
///
/// - [`Self::Park`] (default): fail the chunk and, past
///   `pinecone.outbox_max_attempts`, park the queue — safe, but a DEAD END if
///   rows really did relocate (the stale TIDs stay in the remote namespace
///   and miss again on every retry, forever), and a VACUUM while parked
///   remote-deletes the only copy of any relocated row whose local vector is
///   NULL.
/// - [`Self::Quarantine`]: copy each missed id's fetched vector (+ metadata)
///   into `pinecone.recover_quarantine`, byte-exact, then ADVANCE past the
///   chunk — the job completes (with a `WARNING` naming the count), the
///   bytes survive any later VACUUM, and re-attaching them to their rows is
///   a manual, application-knowledge step (docs/disaster-recovery.md).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum MissPolicy {
    Park,
    Quarantine,
}

impl MissPolicy {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "park" => Some(Self::Park),
            "quarantine" => Some(Self::Quarantine),
            _ => None,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Park => "park",
            Self::Quarantine => "quarantine",
        }
    }
}

/// One `'recover'` outbox job's persisted state — parsed from/serialized to
/// the row's `payload` by [`RecoverJob::parse`]/[`RecoverJob::to_json`].
/// Counts are cumulative across every chunk the job has processed so far
/// (including chunks from before a worker restart — the whole point of
/// persisting them here rather than in memory).
#[derive(Clone, Debug)]
pub(crate) struct RecoverJob {
    /// `None` requests the first `/vectors/list` page; `Some` resumes from
    /// wherever the last successfully-processed chunk left off.
    pub(crate) pagination_token: Option<String>,
    pub(crate) ghosts: GhostPolicy,
    /// `true` allows a chunk to clobber an already non-`NULL` local vector;
    /// `false` (default) only ever writes a `NULL` one.
    pub(crate) overwrite: bool,
    pub(crate) recovered: i64,
    pub(crate) skipped_non_null: i64,
    pub(crate) ghosts_count: i64,
    /// What to do when a `synced_by = 'am'` chunk detects misses — see
    /// [`MissPolicy`].
    pub(crate) on_miss: MissPolicy,
    /// Cumulative `synced_by = 'am'` miss count across every attempt at
    /// every chunk — including chunks that ultimately failed and were
    /// retried (a parked chunk never advances its own cursor, so for the
    /// [`MissPolicy::Park`] path this is the only place that history
    /// survives; see [`bump_misses`]).
    pub(crate) misses: i64,
    /// How many missed ids [`MissPolicy::Quarantine`] copied into
    /// `pinecone.recover_quarantine` — always <= `misses`.
    pub(crate) quarantined: i64,
    /// `true` for a job `pinecone.attach_storage` enqueued (or
    /// updated in place) — `outbox::dispatch_recover`'s completion arm reads
    /// this to decide whether to flip `storage` back to `'full'` and remove
    /// the detached id guard once this job exhausts. `false` for an ordinary
    /// `pinecone.recover_from_remote` call, which never touches `storage`.
    pub(crate) attach: bool,
    /// A partitioned table's job walks its registered namespaces in order, one after the other;
    /// this is the one it is in, with the partition value that names it. `None` before the first
    /// chunk, and always for a table without a partition column.
    pub(crate) partition: Option<Walking>,
    /// How many namespaces of a partitioned table the job has walked to the end.
    pub(crate) namespaces: i64,
}

/// The namespace a partitioned table's recover job is in, and the partition value of its rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Walking {
    pub(crate) value: String,
    pub(crate) namespace: String,
}

impl RecoverJob {
    pub(crate) fn new(
        ghosts: GhostPolicy,
        overwrite: bool,
        on_miss: MissPolicy,
        attach: bool,
    ) -> Self {
        Self {
            pagination_token: None,
            ghosts,
            overwrite,
            on_miss,
            recovered: 0,
            skipped_non_null: 0,
            ghosts_count: 0,
            misses: 0,
            quarantined: 0,
            attach,
            partition: None,
            namespaces: 0,
        }
    }

    pub(crate) fn to_json(&self) -> Value {
        let mut json = json!({
            "pagination_token": self.pagination_token,
            "ghosts": self.ghosts.as_str(),
            "overwrite": self.overwrite,
            "on_miss": self.on_miss.as_str(),
            "recovered": self.recovered,
            "skipped_non_null": self.skipped_non_null,
            "ghosts_count": self.ghosts_count,
            "misses": self.misses,
            "quarantined": self.quarantined,
            "attach": self.attach,
        });
        // Only a partitioned table's job carries a namespace cursor: every other payload keeps
        // its shape.
        if let Some(walking) = &self.partition {
            json["partition_value"] = json!(walking.value);
            json["partition_namespace"] = json!(walking.namespace);
            json["namespaces"] = json!(self.namespaces);
        }
        json
    }

    /// Parses a `'recover'` row's `payload` (built by [`Self::to_json`],
    /// possibly advanced in place by `outbox::set_recover_payload` since).
    pub(crate) fn parse(payload: Option<JsonB>) -> Self {
        let Some(JsonB(Value::Object(obj))) = payload else {
            panic!("pg_pinecone: 'recover' row has no jsonb object payload")
        };
        let pagination_token = match obj.get("pagination_token") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(other) => {
                panic!("pg_pinecone: 'recover' row pagination_token is not a string: {other}")
            }
        };
        let ghosts = match obj.get("ghosts") {
            Some(Value::String(s)) => GhostPolicy::parse(s).unwrap_or_else(|| {
                panic!("pg_pinecone: 'recover' row has an unknown ghosts policy: {s}")
            }),
            other => panic!("pg_pinecone: 'recover' row ghosts is not a string: {other:?}"),
        };
        let overwrite = obj
            .get("overwrite")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        // Absent on a payload written before the field existed — default to
        // the conservative policy, same as the SQL argument's own default.
        let on_miss = match obj.get("on_miss") {
            None | Some(Value::Null) => MissPolicy::Park,
            Some(Value::String(s)) => MissPolicy::parse(s).unwrap_or_else(|| {
                panic!("pg_pinecone: 'recover' row has an unknown on_miss policy: {s}")
            }),
            other => panic!("pg_pinecone: 'recover' row on_miss is not a string: {other:?}"),
        };
        let recovered = obj.get("recovered").and_then(Value::as_i64).unwrap_or(0);
        let skipped_non_null = obj
            .get("skipped_non_null")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        let ghosts_count = obj.get("ghosts_count").and_then(Value::as_i64).unwrap_or(0);
        let misses = obj.get("misses").and_then(Value::as_i64).unwrap_or(0);
        let quarantined = obj.get("quarantined").and_then(Value::as_i64).unwrap_or(0);
        // Absent on a payload written before this field existed (any older
        // job) — default to `false`, same conservative "not an attach
        // job" fallback `on_miss`'s own absent-field default already uses.
        let attach = obj.get("attach").and_then(Value::as_bool).unwrap_or(false);
        let partition = match (obj.get("partition_value"), obj.get("partition_namespace")) {
            (Some(Value::String(value)), Some(Value::String(namespace))) => Some(Walking {
                value: value.clone(),
                namespace: namespace.clone(),
            }),
            (None | Some(Value::Null), None | Some(Value::Null)) => None,
            other => panic!("pg_pinecone: 'recover' row has a partial partition cursor: {other:?}"),
        };
        let namespaces = obj.get("namespaces").and_then(Value::as_i64).unwrap_or(0);
        Self {
            pagination_token,
            ghosts,
            overwrite,
            on_miss,
            recovered,
            skipped_non_null,
            ghosts_count,
            misses,
            quarantined,
            attach,
            partition,
            namespaces,
        }
    }
}

/// [`try_recover_chunk`]'s outcome for one chunk.
pub(crate) enum RecoverOutcome {
    /// More pages remain — the job row stays queued, payload updated to
    /// `RecoverJob`'s new state (cursor + counts).
    InProgress(RecoverJob),
    /// The namespace listing is exhausted — final counts to report.
    Done(RecoverJob),
}

/// Looks up `tbl`'s registry row, requiring it to exist and be `'remote'` OR
/// `'promoting'` — any promoted table, `synced_by` unrestricted
/// (unlike `promote`/`demote`/`sync`/`track`/`untrack`/
/// `topk`, `pinecone.recover_from_remote` explicitly supports a
/// `synced_by = 'am'` row too — see this module's rustdoc, point 3 — so this
/// deliberately does NOT call `registry::require_not_am`, unlike
/// `promote::demotable_row`, whose mode check this otherwise mirrors).
fn recoverable_row(tbl: &PgRelation, func: &str) -> RegistryRow {
    let relid = tbl.oid();
    let display = catalog::display_name(tbl);
    let row = registry::lookup(relid).unwrap_or_else(|| {
        error!(
            "{func}: {display} is not tracked — run pinecone.track(...) or pinecone.promote(...) first"
        )
    });
    if row.mode != "remote" && row.mode != "promoting" {
        error!(
            "{func}: {display} is not promoted (mode = '{}') — run pinecone.promote(...) first",
            row.mode
        );
    }
    row
}

/// Implements `pinecone.recover_from_remote`: validates preconditions and
/// enqueues one `'recover'` outbox job — the worker (`outbox::Scheduler`)
/// does the actual chunk-by-chunk work via [`try_recover_chunk`]. Works on a
/// `'remote'` OR `'promoting'` row, `synced_by = 'trigger'` OR `'am'` — any
/// promoted table.
///
/// Called while a `'recover'` job is ALREADY queued for `tbl`, it does not
/// error (and never enqueues a second job): it updates the queued job's
/// OPTIONS in place — `overwrite`/`ghosts`/`on_miss` — preserving the
/// persisted cursor and counts, with a NOTICE. That's the escape hatch for
/// the parked-on-misses dead-end: a job parked under the
/// default `on_miss => 'park'` can be switched to `'quarantine'` and then
/// resumed with `pinecone.outbox_retry(...)`, completing where a plain retry
/// would just miss the same relocated TIDs forever.
pub(crate) fn recover_from_remote(
    tbl: PgRelation,
    overwrite: bool,
    ghosts: Option<&str>,
    on_miss: Option<&str>,
) {
    recover_from_remote_with_attach(tbl, overwrite, ghosts, on_miss, false)
}

/// `detach::attach_storage`'s entry point: enqueues (or
/// updates in place) the exact same `'recover'` job machinery, with the
/// completion-hook flag set. `ghosts`/`on_miss` are left at their defaults —
/// [`recover_from_remote_with_attach`]'s own detached-storage default already
/// resolves `ghosts` to `'delete'` (the table is still `storage = 'detached'`
/// at ENQUEUE time; only the job's own completion, later, flips that) —
/// matching `detach_storage`'s own ghost handling. `overwrite` stays `false`: a
/// detached row's local vector is NULL by construction (the detached write guard), so
/// there is nothing non-`NULL` an overwrite could ever need to clobber.
pub(crate) fn enqueue_attach_recover(tbl: PgRelation) {
    recover_from_remote_with_attach(tbl, false, None, None, true)
}

/// Shared body of [`recover_from_remote`]/[`enqueue_attach_recover`] — see
/// [`recover_from_remote`]'s own rustdoc for the argument contract `pg_pinecone.
/// recover_from_remote`'s SQL surface exposes; `attach` is an
/// internal-only addition, never SQL-facing (`pinecone.attach_storage` has no
/// `attach` argument of its own to expose — it's implied by calling that
/// function at all).
fn recover_from_remote_with_attach(
    tbl: PgRelation,
    overwrite: bool,
    ghosts: Option<&str>,
    on_miss: Option<&str>,
    attach: bool,
) {
    let relid = tbl.oid();
    let display = catalog::display_name(&tbl);
    let func = if attach {
        "pinecone.attach_storage"
    } else {
        "pinecone.recover_from_remote"
    };
    let row = recoverable_row(&tbl, func);

    // A detached table has no other way to
    // reconcile a remote id with no matching local row — `pinecone.sync` (which
    // would otherwise clean up ghosts via its own deleteAll+backfill) is
    // itself blocked while detached. Default to 'delete' there instead
    // of the ordinary 'report', so ghosts don't just accumulate forever with
    // nothing else ever reconciling them; an explicit `ghosts` argument
    // still always wins. `pinecone.attach_storage` always reaches this with
    // `row.storage == "detached"` still true (the flip to `'full'` only
    // happens at job COMPLETION), so it gets this same default for free.
    let ghosts_raw = ghosts.unwrap_or(if row.storage == "detached" {
        "delete"
    } else {
        "report"
    });
    let ghosts_policy = GhostPolicy::parse(ghosts_raw).unwrap_or_else(|| {
        error!("{func}: ghosts '{ghosts_raw}' is not supported — must be 'report' or 'delete'")
    });
    let on_miss_raw = on_miss.unwrap_or("park");
    let miss_policy = MissPolicy::parse(on_miss_raw).unwrap_or_else(|| {
        error!("{func}: on_miss '{on_miss_raw}' is not supported — must be 'park' or 'quarantine'")
    });

    // Same uniform advisory-then-table lock order every worker-round-adjacent
    // caller uses (`outbox::lock_relid`'s own rustdoc) — rules out this call
    // interleaving with a concurrent worker round's dispatch of the SAME
    // relid's queue in a way that could double-enqueue or update a payload
    // mid-chunk.
    outbox::lock_relid(relid);
    if outbox::has_queued_op(relid, "recover") {
        // NEVER DOWNGRADE an attach-flagged job. A plain
        // `pinecone.recover_from_remote` (attach == false) landing on a queued
        // `pinecone.attach_storage` job must PRESERVE that job's attach flag —
        // otherwise the job stops flipping `storage` back to `'full'` on
        // completion and the table is stranded `'detached'` forever. OR the
        // incoming flag with the stored one so it can only ever be raised,
        // never cleared (an attach call reaching an ordinary job correctly
        // upgrades it; a plain recover reaching an attach job leaves it set).
        let existing_attach = Spi::get_one_with_args::<bool>(
            "SELECT COALESCE((payload->>'attach')::boolean, false) FROM pinecone.outbox \
             WHERE relid::oid = $1::oid AND op = 'recover'",
            &[relid.into()],
        )
        .expect("pg_pinecone: recover_from_remote existing-attach probe failed")
        .unwrap_or(false);
        let effective_attach = attach || existing_attach;
        Spi::run_with_args(
            "UPDATE pinecone.outbox SET payload = payload || jsonb_build_object( \
                 'overwrite', $2::boolean, 'ghosts', $3::text, 'on_miss', $4::text, \
                 'attach', $5::boolean) \
             WHERE relid::oid = $1::oid AND op = 'recover'",
            &[
                relid.into(),
                overwrite.into(),
                ghosts_policy.as_str().into(),
                miss_policy.as_str().into(),
                effective_attach.into(),
            ],
        )
        .expect("pg_pinecone: recover_from_remote policy update failed");
        notice!(
            "{func}: {display} already has a recover job queued — updated its options in place \
             (overwrite => {overwrite}, ghosts => '{}', on_miss => '{}'{}), cursor and counts \
             preserved; if the job is parked, resume it with pinecone.outbox_retry('{display}')",
            ghosts_policy.as_str(),
            miss_policy.as_str(),
            if effective_attach {
                ", attach => true — storage flips back to 'full' once this job completes"
            } else {
                ""
            }
        );
        return;
    }

    let job = RecoverJob::new(ghosts_policy, overwrite, miss_policy, attach);
    Spi::run_with_args(
        "INSERT INTO pinecone.outbox (relid, op, payload) VALUES ($1::oid::regclass, 'recover', $2)",
        &[relid.into(), JsonB(job.to_json()).into()],
    )
    .expect("pg_pinecone: recover_from_remote enqueue failed");

    // Same "queued but nothing will ever drain it" warning `pinecone.promote`'s
    // async path gives — a `'recover'` job always goes through the outbox/
    // worker path (one chunk per WORKER drain round),
    // regardless of the table's own `sync_mode`.
    if !outbox::launcher_lock_held() {
        warning!(
            "{func}: {display}'s recover job is queued but no pg_pinecone worker launcher is running \
             in this cluster (preload absent or pinecone.worker_enabled off) — nothing will drain \
             it until one starts (see pinecone.status())"
        );
    }
}

/// Dispatches one chunk of `relid`'s `'recover'` job: reads one
/// `/vectors/list` page at `job.pagination_token`, fetches those ids' full
/// vectors, and writes them back via [`recover_chunk_trigger`] or
/// [`recover_chunk_am`] depending on `row.synced_by`. `job_id` is only used
/// to persist a miss-count bump ([`bump_misses`]) — the chunk's own
/// success/failure outcome and cursor advancement are otherwise returned to
/// the caller, mirroring `promote::try_backfill_chunk`'s contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_recover_chunk(
    tbl: &PgRelation,
    row: &RegistryRow,
    client: &Client,
    job_id: i64,
    index_name: &str,
    namespace: &str,
    job: &RecoverJob,
) -> Result<RecoverOutcome, ClientError> {
    // A partitioned table is read one registered namespace at a time, and a vector is written
    // back only to a row of that namespace's partition: the namespace says whose entry it is,
    // never the entry itself (a TID or an id may belong to another tenant's row by now).
    let (namespace, partition) = match &row.partition {
        None => (namespace.to_string(), None),
        Some(_) => match walking(tbl.oid(), job) {
            Some(walking) => (walking.namespace.clone(), Some(walking)),
            None => return Ok(RecoverOutcome::Done(job.clone())),
        },
    };
    let filter = row.partition.as_ref().zip(partition.as_ref());
    let filter = filter.map(|(spec, walking)| (spec.column.as_str(), walking.value.as_str()));
    let named = |err: ClientError| match filter {
        Some(_) => partition_lane::name_namespace(err, &namespace),
        None => err,
    };
    let page = client
        .list_vector_ids_page(&ListVectorIdsPageRequest {
            index: index_name.to_string(),
            namespace: Some(namespace.clone()),
            pagination_token: job.pagination_token.clone(),
        })
        .map_err(named)?;

    let fetched: HashMap<String, Vector> = if page.ids.is_empty() {
        HashMap::new()
    } else {
        client
            .fetch_vectors(&FetchVectorsRequest {
                index: index_name.to_string(),
                namespace: Some(namespace.clone()),
                ids: page.ids,
            })
            .map_err(named)?
    };

    // Validate the payload in Rust, before any SQL — a
    // wrong-dimension (or empty) remote vector previously surfaced as the
    // write-back SQL's own `::vector` assignment ERROR, an uncaught panic
    // that aborted the whole drain round's transaction, so `attempts` never
    // incremented and the poison chunk retried forever: a remote-controlled
    // DoS of every table's sync. As a returned permanent-class error it
    // parks after `pinecone.outbox_max_attempts` like any other poison row.
    let display = catalog::display_name(tbl);
    for (id, v) in &fetched {
        if v.values.is_empty() || v.values.len() != row.dimension as usize {
            return Err(ClientError::BadRequest {
                message: format!(
                    "pinecone.recover_from_remote: remote vector '{id}' has {} dimensions, but \
                     {display} is registered with dimension {} — refusing a mismatched (or \
                     hostile) remote payload; repoint at a matching index or fix the remote \
                     data, then pinecone.outbox_retry(...)",
                    v.values.len(),
                    row.dimension
                ),
            });
        }
    }

    let mut next = job.clone();
    next.pagination_token = page.next_token;

    if row.synced_by == "am" {
        let (updated, skipped, missed) = run_write_back_subtransaction(|| {
            recover_chunk_am(tbl, row, &fetched, job.overwrite, filter)
        })?;
        next.recovered += updated;
        next.skipped_non_null += skipped;
        if !missed.is_empty() {
            let miss_count = missed.len() as i64;
            next.misses += miss_count;
            match job.on_miss {
                // The conservative default: fail the chunk (cursor does NOT
                // advance — `next` is discarded with this Err), persisting
                // only the miss count via `bump_misses`. See this module's
                // rustdoc, point 3, for why a mismatch can't be resolved
                // without quiescence.
                MissPolicy::Park => {
                    bump_misses(job_id, miss_count);
                    return Err(miss_park_error(miss_count, fetched.len() as i64));
                }
                // The dead-end escape: copy each missed id's
                // fetched bytes into pinecone.recover_quarantine and ADVANCE —
                // runs OUTSIDE both the owner switch (quarantine is a pg_pinecone
                // table the owner rightly can't touch) and the chunk's
                // subtransaction (already released above; these INSERTs
                // commit with the round).
                MissPolicy::Quarantine => {
                    quarantine_misses(tbl.oid(), &fetched, &missed);
                    next.quarantined += miss_count;
                }
            }
        }
    } else {
        let (updated, skipped, ghost_ids) = run_write_back_subtransaction(|| {
            recover_chunk_trigger(tbl, row, &fetched, job.overwrite, job.attach, filter)
        })?;
        next.recovered += updated;
        next.skipped_non_null += skipped;
        next.ghosts_count += ghost_ids.len() as i64;
        if job.ghosts == GhostPolicy::Delete {
            // A ghost sits in this namespace, so its delete names the partition.
            let p = filter.map(|(_, value)| value);
            outbox::enqueue_delete_ids(tbl.oid(), &ghost_ids, p);
        }
    }

    let Some(walking) = partition else {
        return Ok(if next.pagination_token.is_none() {
            RecoverOutcome::Done(next)
        } else {
            RecoverOutcome::InProgress(next)
        });
    };
    if next.pagination_token.is_some() {
        next.partition = Some(walking);
        return Ok(RecoverOutcome::InProgress(next));
    }
    // This namespace is walked to the end: on to the next registered one, if there is one.
    next.namespaces += 1;
    let after =
        partition_registry::next_partition(tbl.oid(), Lane::Vector, Some(&walking.namespace));
    Ok(match after {
        Some((value, namespace)) => {
            next.partition = Some(Walking { value, namespace });
            RecoverOutcome::InProgress(next)
        }
        None => {
            next.partition = Some(walking);
            RecoverOutcome::Done(next)
        }
    })
}

/// The namespace a partitioned table's job reads next: the one it is in, else the first the
/// registry lists. `None` when nothing is registered, which ends the job.
fn walking(relid: pg_sys::Oid, job: &RecoverJob) -> Option<Walking> {
    job.partition.clone().or_else(|| {
        let first = partition_registry::next_partition(relid, Lane::Vector, None);
        first.map(|(value, namespace)| Walking { value, namespace })
    })
}

/// The permanent-class error a [`MissPolicy::Park`] chunk fails with — see
/// [`MissPolicy`] for the trade-off and `docs/disaster-recovery.md` for the
/// full remediation sequence this message summarizes. The
/// remediation deliberately says pause APPLICATION writes, never "take a
/// table lock" — a `SHARE` (or stronger) lock blocks this job's own
/// write-back `UPDATE`, freezing the retry for as long as it's held (proven
/// empirically by the drill's first leg-6 run) — and leads with the
/// do-not-VACUUM warning, since a VACUUM while parked destroys the remote
/// copies this job exists to restore.
fn miss_park_error(misses: i64, fetched: i64) -> ClientError {
    ClientError::BadRequest {
        message: format!(
            "pinecone.recover_from_remote: {misses} of {fetched} TID-keyed vector(s) in this \
             chunk had no matching row at all — a concurrent non-HOT write likely relocated a \
             row's ctid while this chunk ran (TID recover requires quiescence). Do NOT VACUUM \
             this table while the job is parked: ambulkdelete would remote-delete the very \
             copies this job is trying to restore. Do NOT take a table lock either — a SHARE \
             lock blocks the job's own write-back; pause the APPLICATION's writes instead. \
             This chunk's cursor did not advance. To resume: quiesce app writes, then \
             pinecone.outbox_retry(...); if rows genuinely relocated the same ids will miss \
             again — switch the job to quarantine first \
             (pinecone.recover_from_remote(tbl, on_miss => 'quarantine'), then outbox_retry) \
             and re-attach from pinecone.recover_quarantine per docs/disaster-recovery.md."
        ),
    }
}

/// [`MissPolicy::Quarantine`]'s copy step: one row per missed id into
/// `pinecone.recover_quarantine`, values passed as a `real[]` datum straight
/// from the fetched f32s — no text round-trip anywhere, so the quarantined
/// bytes are exactly what Pinecone returned (the byte-exactness the DR
/// drill's fidelity check verifies).
fn quarantine_misses(relid: pg_sys::Oid, fetched: &HashMap<String, Vector>, missed: &[String]) {
    for id in missed {
        let v = fetched.get(id).unwrap_or_else(|| {
            panic!("pg_pinecone: quarantine: missed id '{id}' not in the fetched map — a bug")
        });
        Spi::run_with_args(
            "INSERT INTO pinecone.recover_quarantine (relid, remote_id, embedding, metadata) \
             VALUES ($1::oid::regclass, $2, $3, $4)",
            &[
                relid.into(),
                id.as_str().into(),
                v.values.clone().into(),
                v.metadata.clone().map(JsonB).into(),
            ],
        )
        .expect("pg_pinecone: recover quarantine insert failed");
    }
}

/// Runs one chunk's write-back inside its own subtransaction, converting any
/// Postgres `ERROR` raised within (an uncastable metadata value, a
/// constraint/trigger failure on the user's own table — anything the
/// Rust-side payload validation can't see) into a returned permanent-class
/// [`ClientError`] instead of an uncaught panic (the SQL
/// half): without the subtransaction, the caught error would leave the drain
/// round's transaction aborted — `record_permanent_failure`'s own UPDATE
/// could never run, `attempts` never incremented, and the poison chunk
/// retried forever. The subtransaction bounds the rollback to exactly this
/// chunk's writes (so a failed chunk retries wholesale, same as a transport
/// failure), and the round's transaction stays healthy for the bookkeeping
/// that follows. Same `BeginInternalSubTransaction` /
/// `RollbackAndReleaseCurrentSubTransaction` + memory-context/resource-owner
/// save-restore sequence plpgsql's own `EXCEPTION` blocks use.
fn run_write_back_subtransaction<T>(
    body: impl FnOnce() -> T + std::panic::UnwindSafe,
) -> Result<T, ClientError> {
    unsafe {
        let saved_ctx = pg_sys::CurrentMemoryContext;
        let saved_owner = pg_sys::CurrentResourceOwner;
        pg_sys::BeginInternalSubTransaction(std::ptr::null());
        // `BeginInternalSubTransaction` leaves `CurTransactionContext`
        // current; the body's own allocations should live in the caller's
        // context, same as plpgsql's `MemoryContextSwitchTo(oldcontext)`
        // right after it begins the subxact.
        pg_sys::MemoryContextSwitchTo(saved_ctx);

        let result = PgTryBuilder::new(|| Ok(body()))
            .catch_others(|caught| {
                let message = match &caught {
                    CaughtError::PostgresError(e)
                    | CaughtError::ErrorReport(e)
                    | CaughtError::RustPanic { ereport: e, .. } => e.message().to_string(),
                };
                Err(ClientError::BadRequest {
                    message: format!("recover write-back failed: {message}"),
                })
            })
            .execute();

        match result {
            Ok(value) => {
                pg_sys::ReleaseCurrentSubTransaction();
                pg_sys::MemoryContextSwitchTo(saved_ctx);
                pg_sys::CurrentResourceOwner = saved_owner;
                Ok(value)
            }
            Err(e) => {
                pg_sys::RollbackAndReleaseCurrentSubTransaction();
                pg_sys::MemoryContextSwitchTo(saved_ctx);
                pg_sys::CurrentResourceOwner = saved_owner;
                Err(e)
            }
        }
    }
}

/// `synced_by = 'trigger'` chunk write-back: for every id in `fetched`,
/// `UPDATE ... SET vcol = <remote value> WHERE id_col = id AND (vcol IS NULL
/// OR overwrite)` — the NULL guard this module's rustdoc requires (point 1)
/// — plus, for each of `row.metadata_columns`, the same guarded assignment
/// from whatever key the remote metadata object carries (a key it doesn't
/// have leaves that column untouched, never NULLed). Returns `(updated_count,
/// skipped_non_null_count, ghost_ids)`; ghost ids are the fetched ids with NO
/// matching local row at all (distinct from a matching row that just failed
/// the NULL guard) — reachable here (unlike the `'am'` path) because the id
/// column is a stable identity, not a heap-position-derived one. Wrapped in
/// [`RecoveringGuard`] around the one `UPDATE` statement (point 2 of this
/// module's rustdoc).
///
/// With `partition` (`(column, value)`: the chunk was read from that partition's namespace) only
/// rows of that partition match, so an id that belongs to another tenant's row by now is a ghost
/// here, not a restore; and the partition column is never written back from remote metadata.
fn recover_chunk_trigger(
    tbl: &PgRelation,
    row: &RegistryRow,
    fetched: &HashMap<String, Vector>,
    overwrite: bool,
    attach: bool,
    partition: Option<(&str, &str)>,
) -> (i64, i64, Vec<String>) {
    if fetched.is_empty() {
        return (0, 0, Vec::new());
    }

    let relid = tbl.oid();
    let id_col = pgrx::spi::quote_identifier(&row.id_column);
    let vcol = pgrx::spi::quote_identifier(&row.vector_column);
    let table = quote_qualified_identifier(tbl.namespace(), tbl.name());

    // On the `pinecone.attach_storage` path
    // (`attach == true`), restore ONLY the vector column — NEVER the metadata
    // columns. Local metadata stayed AUTHORITATIVE for the whole detached
    // period (the detached `_sync_trigger` branch keeps pushing metadata edits
    // to Pinecone via `'update'`/`'reupsert'`), so copying the remote metadata
    // back would CLOBBER a local metadata edit that landed after attach began
    // but before this row's chunk drained — a silent lost update. An ordinary
    // `pinecone.recover_from_remote` (genuine DR / independent local loss) still
    // copies metadata: there the local columns are the DAMAGED side, and the
    // remote is the surviving source of truth.
    let metadata_sets: String = if attach {
        String::new()
    } else {
        let metadata_kinds: Vec<(String, ColKind)> = row
            .metadata_columns
            .iter()
            .filter(|name| partition.is_none_or(|(column, _)| column != name.as_str()))
            .map(|name| {
                let type_name = catalog::column_type(relid, name).unwrap_or_else(|| {
                    panic!(
                        "pg_pinecone: recover metadata column \"{name}\" of {table} disappeared since \
                         tracking — should be unreachable (backfill's own chunk read would \
                         already have failed first)"
                    )
                });
                (name.clone(), catalog::col_kind(&type_name))
            })
            .collect();
        metadata_kinds
            .iter()
            .map(|(name, kind)| format!(", {}", metadata_set_clause(kind, name)))
            .collect()
    };

    let mut incoming_map = serde_json::Map::with_capacity(fetched.len());
    for (id, v) in fetched {
        let mut obj = serde_json::Map::new();
        obj.insert("values".to_string(), json!(v.values));
        if let Some(meta) = &v.metadata {
            obj.insert("metadata".to_string(), meta.clone());
        }
        incoming_map.insert(id.clone(), Value::Object(obj));
    }

    let vector_type = qualified_vector_type();
    let in_partition = partition_predicate(partition);
    let sql = format!(
        "WITH incoming AS ( \
             SELECT key AS id, \
                    ARRAY(SELECT (jsonb_array_elements_text(value->'values'))::real) AS vec, \
                    (value->'metadata') AS meta \
             FROM jsonb_each($1::jsonb) \
         ), \
         ghosts AS ( \
             SELECT i.id FROM incoming i \
             WHERE NOT EXISTS (SELECT 1 FROM {table} t WHERE t.{id_col}::text = i.id{in_partition}) \
         ), \
         updated AS ( \
             UPDATE {table} t SET {vcol} = i.vec::{vector_type}{metadata_sets} \
             FROM incoming i \
             WHERE t.{id_col}::text = i.id{in_partition} AND (t.{vcol} IS NULL OR $2::boolean) \
             RETURNING i.id AS id \
         ), \
         fanned AS ( \
             SELECT id FROM updated GROUP BY id HAVING count(*) > 1 \
         ) \
         SELECT (SELECT count(*) FROM incoming)::bigint AS fetched_count, \
                (SELECT COALESCE(array_agg(id), '{{}}') FROM ghosts) AS ghost_ids, \
                (SELECT count(*) FROM updated)::bigint AS updated_count, \
                (SELECT COALESCE(array_agg(id), '{{}}') FROM fanned) AS fanned_ids"
    );

    let _owner = unsafe { TableOwnerGuard::switch_to(table_owner(tbl)) };
    let _recovering = RecoveringGuard::set(relid);
    let (fetched_count, ghost_ids, updated_count, fanned_ids) = Spi::connect(|client| {
        let mut rows = client
            .select(
                &sql,
                None,
                &chunk_arguments(incoming_map, overwrite, partition),
            )
            .expect("pg_pinecone: recover trigger-path chunk query failed");
        let row = rows
            .next()
            .expect("pg_pinecone: recover trigger-path chunk query returned no row");
        (
            row.get_by_name::<i64, _>("fetched_count")
                .expect("pg_pinecone: recover fetched_count")
                .unwrap_or(0),
            row.get_by_name::<Vec<String>, _>("ghost_ids")
                .expect("pg_pinecone: recover ghost_ids")
                .unwrap_or_default(),
            row.get_by_name::<i64, _>("updated_count")
                .expect("pg_pinecone: recover updated_count")
                .unwrap_or(0),
            row.get_by_name::<Vec<String>, _>("fanned_ids")
                .expect("pg_pinecone: recover fanned_ids")
                .unwrap_or_default(),
        )
    });
    drop(_recovering);
    drop(_owner);

    // Defense-in-depth for the drop-during-job race: if
    // the id column's UNIQUE index/PRIMARY KEY was dropped after this job was
    // enqueued and a duplicate id inserted (a NULL-vector INSERT the detached
    // INSERT guard permits), the single fetched remote vector for that id would
    // fan onto BOTH local rows — silently fabricating data. The gate
    // (`require_id_uniqueness`, run at detach_storage/attach_storage) closes
    // the common case; this closes the window where the constraint vanished
    // mid-job. Raise a permanent-class ERROR: the enclosing
    // `run_write_back_subtransaction` rolls this chunk's UPDATE back (nothing
    // fabricated) and parks the job for an operator to restore the constraint.
    if !fanned_ids.is_empty() {
        error!(
            "pinecone.recover_from_remote: id(s) {} each matched MORE THAN ONE local row in {table} \
             — the id column's UNIQUE index/PRIMARY KEY was dropped while this job was in flight \
             and a duplicate id inserted. Refusing to fan one remote vector onto multiple rows \
             (silent data fabrication). Restore the unique constraint (deduplicate the offending \
             rows first if needed), then pinecone.outbox_retry(...).",
            fanned_ids.join(", ")
        );
    }

    let skipped_non_null = (fetched_count - ghost_ids.len() as i64 - updated_count).max(0);
    (updated_count, skipped_non_null, ghost_ids)
}

/// The `SET` clause for one metadata column: only overwrites it when the
/// incoming remote metadata is an OBJECT that actually has that key
/// (`jsonb_typeof(i.meta) = 'object' AND i.meta ? key`) — a remote vector's
/// metadata is whatever was non-`NULL` at push time (`_sync_trigger`'s own
/// `jsonb_strip_nulls`), so a missing key means "this row never had a value
/// here", not "clear it". The `jsonb_typeof` half of the guard is load-
/// bearing: on a bare jsonb STRING scalar, `?` tests
/// equality — a hostile remote payload of just `"category"` makes
/// `i.meta ? 'category'` TRUE while `i.meta->>'category'` is NULL, which
/// without the typeof check silently NULLed the very column the guard
/// promises never to touch on absence. `->>'key'` always yields `text`;
/// every non-array kind relies on Postgres's built-in text ->
/// {integer/numeric/boolean} assignment casts rather than hand-rolled ones.
fn metadata_set_clause(kind: &ColKind, name: &str) -> String {
    let ident = pgrx::spi::quote_identifier(name);
    let key_lit = pgrx::spi::quote_literal(name);
    let value_expr = match kind {
        ColKind::TextArray => {
            format!("ARRAY(SELECT jsonb_array_elements_text(i.meta->{key_lit}))")
        }
        ColKind::Bool => format!("(i.meta->>{key_lit})::boolean"),
        ColKind::Int => format!("(i.meta->>{key_lit})::bigint"),
        ColKind::Float | ColKind::Numeric => format!("(i.meta->>{key_lit})::double precision"),
        ColKind::Text => format!("i.meta->>{key_lit}"),
        // Unreachable: track()/promote() validate metadata_columns via
        // catalog::col_kind before they ever reach the registry.
        ColKind::Unsupported(t) => {
            panic!("pg_pinecone: recover metadata column {name} has unsupported type {t}")
        }
    };
    format!(
        "{ident} = CASE WHEN jsonb_typeof(i.meta) = 'object' AND i.meta ? {key_lit} \
         THEN {value_expr} ELSE {ident} END"
    )
}

/// `synced_by = 'am'` chunk write-back: remote ids are heap TIDs, formatted
/// `"block:offset"` (`am::tid_to_id`) — reconstructed here via `split_part`
/// combined with a `tid` literal cast (`'(block,offset)'::tid`, Postgres's
/// native input format) rather than a Rust-side conversion, since this is
/// plain SQL, not the raw buffer manipulation
/// `promote::try_tid_backfill_chunk` needs.
///
/// **Miss detection, not ghost/skip disambiguation** (this module's rustdoc,
/// point 3): a fetched id with NO row at that `ctid` at all — `missing`,
/// below — is a "miss", distinct from one that merely failed the ordinary
/// NULL guard (`skipped_non_null`, exactly like [`recover_chunk_trigger`]'s
/// own split — a healthy, already-synced AM table recovered with
/// `overwrite => false` must NOT look like an emergency just because every
/// row's guard correctly declined to overwrite it). A TID's identity is only
/// stable absent concurrent non-HOT writes, so a genuine miss could mean a
/// dead TID (fine to skip) or a live row that just moved (NOT fine to skip —
/// the next list page will never re-offer this now-relocated row under its
/// NEW TID, since remote ids are frozen to this chunk's already-fetched
/// page) — indistinguishable without quiescence. What happens to the missed
/// ids is the CALLER's decision ([`MissPolicy`], applied in
/// [`try_recover_chunk`]): this function just performs the guarded
/// write-back and returns `(updated, skipped_non_null, missed_ids)`.
///
/// With `partition` (`(column, value)`: the chunk was read from that partition's namespace) a
/// position only matches a row of that partition, so a stale entry whose TID VACUUM recycled for
/// another tenant's row is a miss, never a vector written onto that row.
fn recover_chunk_am(
    tbl: &PgRelation,
    row: &RegistryRow,
    fetched: &HashMap<String, Vector>,
    overwrite: bool,
    partition: Option<(&str, &str)>,
) -> (i64, i64, Vec<String>) {
    if fetched.is_empty() {
        return (0, 0, Vec::new());
    }

    let vcol = pgrx::spi::quote_identifier(&row.vector_column);
    let table = quote_qualified_identifier(tbl.namespace(), tbl.name());

    let mut incoming_map = serde_json::Map::with_capacity(fetched.len());
    for (id, v) in fetched {
        incoming_map.insert(id.clone(), json!(v.values));
    }

    let vector_type = qualified_vector_type();
    let in_partition = partition_predicate(partition);
    let sql = format!(
        "WITH incoming AS ( \
             SELECT key, \
                    (('(' || split_part(key, ':', 1) || ',' || split_part(key, ':', 2) || \
                      ')')::tid) AS tid, \
                    ARRAY(SELECT (jsonb_array_elements_text(value))::real) AS vec \
             FROM jsonb_each($1::jsonb) \
         ), \
         missing AS ( \
             SELECT i.key FROM incoming i WHERE NOT EXISTS ( \
                 SELECT 1 FROM {table} t WHERE t.ctid = i.tid{in_partition} \
             ) \
         ), \
         updated AS ( \
             UPDATE {table} t SET {vcol} = i.vec::{vector_type} \
             FROM incoming i \
             WHERE t.ctid = i.tid{in_partition} AND (t.{vcol} IS NULL OR $2::boolean) \
             RETURNING i.tid \
         ) \
         SELECT (SELECT count(*) FROM incoming)::bigint AS fetched_count, \
                (SELECT count(*) FROM updated)::bigint AS updated_count, \
                (SELECT COALESCE(array_agg(key), '{{}}') FROM missing) AS missed_ids"
    );

    let _owner = unsafe { TableOwnerGuard::switch_to(table_owner(tbl)) };
    let _recovering = RecoveringGuard::set(tbl.oid());
    let (fetched_count, updated_count, missed_ids) =
        Spi::get_three_with_args::<i64, i64, Vec<String>>(
            &sql,
            &chunk_arguments(incoming_map, overwrite, partition),
        )
        .expect("pg_pinecone: recover am-path chunk query failed");
    drop(_recovering);
    // Restored here rather than at scope end: the CALLER's next steps
    // (bump_misses / quarantine inserts) write to pg_pinecone tables the table
    // owner (rightly) has no privileges on.
    drop(_owner);

    let fetched_count = fetched_count.unwrap_or(0);
    let updated_count = updated_count.unwrap_or(0);
    let missed_ids = missed_ids.unwrap_or_default();
    let skipped_non_null = fetched_count - missed_ids.len() as i64 - updated_count;
    (updated_count, skipped_non_null.max(0), missed_ids)
}

/// ` AND t.<column>::text = $3` for a chunk read from one partition's namespace: the text of the
/// value is what `partition::canon` writes (`col::text`), compared as bytes.
fn partition_predicate(partition: Option<(&str, &str)>) -> String {
    match partition {
        Some((column, _)) => format!(
            " AND t.{}::text COLLATE \"C\" = $3::text COLLATE \"C\"",
            pgrx::spi::quote_identifier(column)
        ),
        None => String::new(),
    }
}

/// The arguments of a chunk's SQL: the fetched vectors, `overwrite` and, for a partitioned table,
/// the partition value (`$3`, see [`partition_predicate`]).
fn chunk_arguments<'a>(
    incoming: serde_json::Map<String, Value>,
    overwrite: bool,
    partition: Option<(&str, &'a str)>,
) -> Vec<DatumWithOid<'a>> {
    let mut args: Vec<DatumWithOid<'a>> =
        vec![JsonB(Value::Object(incoming)).into(), overwrite.into()];
    if let Some((_, value)) = partition {
        args.push(value.into());
    }
    args
}

/// Bumps a `'recover'` job's persisted `misses` counter by `delta`,
/// independent of (and surviving) the chunk's own failed `Result` — see
/// [`RecoverJob::misses`]'s rustdoc for why this is the only place that
/// history is retained (a chunk that hits a miss never reaches
/// `outbox::set_recover_payload` through the normal success path, since it
/// returns `Err` and its cursor must NOT advance).
fn bump_misses(job_id: i64, delta: i64) {
    Spi::run_with_args(
        "UPDATE pinecone.outbox SET payload = jsonb_set(payload, '{misses}', \
         to_jsonb(COALESCE((payload->>'misses')::bigint, 0) + $2)) WHERE id = $1",
        &[job_id.into(), delta.into()],
    )
    .expect("pg_pinecone: recover misses bump failed");
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn job() -> RecoverJob {
        RecoverJob::new(GhostPolicy::Report, false, MissPolicy::Park, false)
    }

    #[test]
    fn a_job_without_a_partition_keeps_its_payload_shape() {
        let json = job().to_json();

        let object = json.as_object().expect("an object");
        for key in ["partition_value", "partition_namespace", "namespaces"] {
            assert!(!object.contains_key(key), "{key}");
        }
    }

    #[test]
    fn the_namespace_cursor_survives_a_round_trip_through_the_payload() {
        let mut job = job();
        job.pagination_token = Some("page-2".to_string());
        job.partition = Some(Walking {
            value: "Acme Corp".to_string(),
            namespace: "public.t:Acme%20Corp".to_string(),
        });
        job.namespaces = 3;
        job.recovered = 7;

        let back = RecoverJob::parse(Some(JsonB(job.to_json())));

        assert_eq!(back.partition, job.partition);
        assert_eq!(back.namespaces, 3);
        assert_eq!(back.recovered, 7);
        assert_eq!(back.pagination_token.as_deref(), Some("page-2"));
    }

    #[test]
    fn a_payload_written_before_the_cursor_existed_has_none() {
        let old = json!({"pagination_token": null, "ghosts": "report", "overwrite": false});

        let job = RecoverJob::parse(Some(JsonB(old)));

        assert_eq!(job.partition, None);
        assert_eq!(job.namespaces, 0);
    }

    #[test]
    #[should_panic(expected = "partial partition cursor")]
    fn half_a_cursor_is_refused() {
        let half = json!({"ghosts": "report", "partition_value": "1"});

        RecoverJob::parse(Some(JsonB(half)));
    }
}
