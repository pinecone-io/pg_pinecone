//! Worker-driven maintenance for the local (native) engine, with
//! evidence-gated re-cluster. CIC/RIC stays rejected (see
//! `local::vacuum`'s own "CIC/RIC decision" section) — the rebuild path here
//! is plain, blocking `REINDEX INDEX`, never `CONCURRENTLY`.
//!
//! ## Why this module owns a table of its own, not `pinecone.tables`/`pinecone.outbox`
//! A local-mode pg_pinecone-AM index gets **no `pinecone.tables` row at all** —
//! `am::ambuild`'s local branch returns before any registry-row/Pinecone
//! machinery runs (see `local::build`'s module docstring: "no `pinecone.tables`
//! row, no primary key requirement, no outbox"). That means there is no
//! heap-relid registry row to key an outbox job off of, and no PK-having
//! "table" for `pinecone.outbox`'s existing relid-keyed queue semantics to make
//! sense against. Rather than force local-index maintenance through a
//! table/HTTP-shaped queue built for a completely different job (remote sync
//! dispatch), this module keeps its own tiny registry —
//! [`pinecone.index_maintenance`], keyed by the pg_pinecone-AM **index's own oid** —
//! and its own round, evaluated by the SAME background worker
//! (`worker.rs`'s per-database loop calls [`run_round`] once per round,
//! right alongside `outbox::Scheduler::round`).
//!
//! ## The drift metric
//! [`index_drift`] is read-only and free-standing — it works on ANY
//! local-mode pg_pinecone index, scheduled or not. It reads the metapage's
//! `total_vectors` (frozen at build time) and `inserted_since_build` (fresh,
//! uncached — `metapage::read_inserted_count_fresh`, which bypasses the
//! per-backend `rd_amcache` since inserts never invalidate it). `total` is
//! the CURRENT effective row count (`total_vectors + inserted_since_build`),
//! and `drift_fraction = inserted_since_build / total` — e.g. a build of
//! 1000 rows with 300 further inserts reports `(300, 1300, 0.2308)`.
//!
//! ## Counter reconciliation (considered, resolved as-is)
//! `inserted_since_build` can OVERCOUNT: an aborted transaction's `aminsert`
//! bump survives the rollback (GenericXLog page writes are non-transactional
//! — `local::insert`'s "Aborted transactions overcount permanently" note),
//! and a crash between bump and append leaves it one high. Both err in the
//! conservative direction (drift looks worse than it is → maintenance fires
//! sooner, mode selection prefers exact — never a recall cost), and the
//! maintenance REINDEX this module triggers IS the reconciliation: `ambuild`
//! recounts everything and writes a fresh metapage with
//! `inserted_since_build = 0`, so accumulated aborted-txn overcount is
//! bounded by one maintenance cadence at the operator's own threshold. No
//! separate reconciliation pass (e.g. a VACUUM-time recount) is built — it
//! would add a metapage write to every VACUUM to shave a bias that already
//! self-corrects in the safe direction.
//!
//! ## Failure handling (poison/parking — the outbox op framework's own
//! vocabulary, reused here)
//! The entire "check drift, maybe REINDEX" sequence for one index runs
//! inside [`pinecone._maintenance_attempt`], a plain (`SECURITY INVOKER`)
//! plpgsql function whose `EXCEPTION WHEN OTHERS` clause is what makes a
//! caught Postgres ERROR (a lock conflict, a permissions error, whatever)
//! safe to continue past *within the same transaction*: a raw Rust-level
//! `PgTryBuilder` catch of a DDL error leaves the rest of that transaction
//! unusable for further SQL (confirmed elsewhere in this crate's own
//! tests — see the note on the reloption-validation tests in `lib.rs`),
//! which would strand this round's own bookkeeping update right after the
//! catch; plpgsql's own internal subtransaction is what makes recovery safe.
//! A consecutive-failure counter (mirroring `outbox::record_permanent_failure`
//! / `pinecone.outbox_max_attempts` — the SAME GUC, same "attempts" vocabulary)
//! parks the schedule after too many failures; re-running
//! `pinecone.schedule_maintenance` on a parked index clears the park (no
//! separate `..._retry` hatch — one enqueue path is enough surface).
//! An index whose schedule outlives the index itself (dropped between
//! `schedule_maintenance` and a later round) is silently purged with a
//! `WARNING`, mirroring `outbox::dispatch_backfill`'s dropped-relid purge.
//!
//! ## What this does NOT do (honest scope line for 0.6+/backlog)
//! No `REINDEX CONCURRENTLY` (rejected — see module docstring above), no
//! incremental per-list re-cluster/split-merge (a later optimization), no
//! adaptive/self-tuning threshold. A maintenance REINDEX is a full rebuild: it
//! takes the same locks as any
//! `REINDEX` (blocks writes and reads-via-index for its duration) — this
//! module's entire value is VISIBILITY (the drift metric) and
//! EVIDENCE-GATED TRIGGERING (an operator-set threshold, off by default via
//! `pinecone.auto_maintain`), not making the rebuild itself non-blocking.

use std::panic::AssertUnwindSafe;

use pgrx::{PgRelation, prelude::*, spi::Spi};

use crate::{
    backend, docs_guard,
    local::metapage::{self, Mode},
};

extension_sql!(
    r#"
    CREATE TABLE pinecone.index_maintenance (
        index_relid regclass PRIMARY KEY,
        drift_threshold double precision NOT NULL DEFAULT 0.2
            CHECK (drift_threshold > 0 AND drift_threshold <= 1),
        -- The role `pinecone.schedule_maintenance` verified as owning (or being
        -- superuser over) the index's table at scheduling time — an audit trail
        -- for who authorized this policy, independent of
        -- `_maintenance_attempt`'s own choice of
        -- which identity actually runs the REINDEX (see that function's
        -- rustdoc: it re-resolves the CURRENT table owner fresh, every
        -- round, rather than trusting this column not to have gone stale).
        scheduled_by regrole NOT NULL,
        queue_parked boolean NOT NULL DEFAULT false,
        parked_reason text,
        attempts int NOT NULL DEFAULT 0,
        last_error text,
        last_checked_at timestamptz,
        last_maintained_at timestamptz,
        last_drift_fraction double precision,
        created_at timestamptz NOT NULL DEFAULT now()
    );
    REVOKE ALL ON pinecone.index_maintenance FROM PUBLIC;
    -- State (an operator's standing maintenance policy per index), not a
    -- queue — dumped like `pinecone.tables`, unlike `pinecone.outbox`.
    SELECT pg_catalog.pg_extension_config_dump('pinecone.index_maintenance', '');
    "#,
    name = "pg_pinecone_index_maintenance",
    requires = [pinecone],
);

// The one risky step (`EXECUTE 'REINDEX INDEX ...'`) plus the drift read
// that gates it, both wrapped in ONE plpgsql `EXCEPTION WHEN OTHERS` —
// see this module's own docstring for why that (not a Rust-level
// `PgTryBuilder`) is what keeps the calling transaction usable afterward.
// `SECURITY INVOKER` (the default — no explicit clause): this function
// itself still runs as the background worker's OWN identity (whatever
// privilege it already has on `pinecone._maintenance_attempt`/`pg_pinecone.
// index_maintenance` — no separate grant needed per scheduled index, unlike
// the alternative of switching identity BEFORE this call, which would force
// every prospective index owner to hold their own `EXECUTE` grant on this
// internal function just for their own index to be maintainable).
// **The REINDEX line alone** runs under a temporary `SET ROLE` to the
// table's CURRENT owner — resolved fresh
// every attempt (not the possibly-stale role `pinecone.schedule_maintenance`
// recorded), so the REINDEX itself goes through Postgres's own ordinary
// ownership check as if the real owner ran it, never rubber-stamped by the
// worker's own (typically superuser) identity; `RESET ROLE` restores the
// worker's identity immediately after, and — since plpgsql wraps this WHOLE
// body in an implicit subtransaction — a REINDEX failure's `EXCEPTION WHEN
// OTHERS` catch itself rolls back the `SET ROLE` GUC change too, so the
// explicit `RESET ROLE` below only needs to cover the success path. `SET
// search_path` (the function's own option, below) pins name resolution to
// `pg_catalog`/`pg_temp` for the ENTIRE call, including while switched —
// closing the same search_path-shadowing hole `am::SecurityDefinerGuard`'s
// own `RestrictSearchPath` call exists for, and (unlike a plain in-body
// `SET search_path`) guaranteed by Postgres itself to restore on any exit
// path, success or error alike.
extension_sql!(
    r#"
    CREATE FUNCTION pinecone._maintenance_attempt(
        p_index_relid oid,
        p_drift_threshold double precision,
        OUT reindexed boolean,
        OUT drift_fraction double precision,
        OUT gone boolean,
        OUT error text
    )
    LANGUAGE plpgsql
    SET search_path = pg_catalog, pg_temp
    AS $$
    DECLARE
        d record;
        v_schema name;
        v_index name;
        v_owner name;
    BEGIN
        reindexed := false;
        gone := false;
        error := NULL;
        drift_fraction := NULL;

        -- Resolves the CURRENT schema-qualified name AND the underlying
        -- table's CURRENT owner straight from the catalogs rather than
        -- `p_index_relid::regclass::text` (whose rendering depends on the
        -- CALLER's `search_path`,
        -- unqualified whenever the index's schema happens to be on it) or
        -- the `scheduled_by` column (which can go stale if ownership
        -- changes after scheduling) — `%I.%I`/`%I` below are always
        -- unambiguous, and this doubles as the existence check
        -- `_maintenance_attempt` needs anyway.
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

        SELECT * INTO d FROM pinecone.index_drift(p_index_relid::regclass);
        drift_fraction := d.drift_fraction;

        IF drift_fraction >= p_drift_threshold THEN
            EXECUTE pg_catalog.format('SET ROLE %I', v_owner);
            EXECUTE pg_catalog.format('REINDEX INDEX %I.%I', v_schema, v_index);
            RESET ROLE;
            reindexed := true;
        END IF;
    EXCEPTION WHEN OTHERS THEN
        error := SQLERRM;
    END;
    $$;
    REVOKE EXECUTE ON FUNCTION pinecone._maintenance_attempt(oid, double precision) FROM PUBLIC;
    "#,
    name = "pg_pinecone_maintenance_attempt",
    requires = [pinecone, "pg_pinecone_index_maintenance", index_drift],
);

/// Errors clearly (naming `func`) unless `idx` is a local-mode pg_pinecone
/// access-method index — the one precondition both [`index_drift`] and
/// [`schedule_maintenance`] share. A plain pgvector/other-AM index, or a
/// REMOTE-mode pg_pinecone index (Pinecone-backed — no local per-record storage,
/// hence no drift/maintenance story of its own), both error here rather than
/// silently no-op.
fn require_local_pg_pinecone_index(idx: &PgRelation, func: &'static str) -> metapage::MetaPage {
    if !is_pg_pinecone_am_index(idx) {
        error!(
            "{func}: \"{}\" is not a pg_pinecone access-method index — CREATE INDEX ... USING pinecone \
             first",
            idx.name()
        );
    }
    let meta = metapage::read(idx.as_ptr());
    if meta.mode != Mode::Local {
        error!(
            "{func}: \"{}\" is not a local-mode pg_pinecone index (mode = {:?}) — remote-mode pg_pinecone \
             indexes are served by Pinecone directly and have no local drift/maintenance story",
            idx.name(),
            meta.mode
        );
    }
    warn_if_offline(idx, &meta, func);
    meta
}

/// A native index below format 3 is offline until REINDEX: the drift readers and the scheduler
/// still answer (the counters sit at the same place in every format) but say so with a WARNING,
/// the maintenance worker's REINDEX being one way out. Once per backend, index and `site`: the
/// worker asks every round, and one line in its log says it.
fn warn_if_offline(idx: &PgRelation, meta: &metapage::MetaPage, site: &'static str) {
    use std::{cell::RefCell, collections::HashSet};
    thread_local! {
        static WARNED: RefCell<HashSet<(u32, &'static str)>> = RefCell::new(HashSet::new());
    }
    if meta.needs_reindex() && WARNED.with_borrow_mut(|w| w.insert((idx.oid().to_u32(), site))) {
        unsafe { metapage::warn_old_format(idx.as_ptr(), meta.stored()) };
    }
}

pub(crate) fn is_pg_pinecone_am_index(idx: &PgRelation) -> bool {
    if !idx.is_index() {
        return false;
    }
    Spi::get_one_with_args::<bool>(
        "SELECT COALESCE((SELECT a.amname = 'pinecone' FROM pg_catalog.pg_class c \
         JOIN pg_catalog.pg_am a ON a.oid = c.relam WHERE c.oid = $1::oid), false)",
        &[idx.oid().into()],
    )
    .expect("pg_pinecone: maintenance AM-check query failed")
    .expect("pg_pinecone: COALESCE(..., false) is never NULL")
}

// ---------------------------------------------------------------------------
// Ownership/ACL gating
// ---------------------------------------------------------------------------

/// Requires the calling role to own `idx`'s underlying TABLE — or be a
/// member of the owning role, or superuser — mirroring EXACTLY what
/// Postgres's own `REINDEX INDEX`/`REINDEX TABLE` permission check requires
/// (`object_ownercheck` against the TABLE, not the index itself: an index's
/// own `relowner` need not match its table's, e.g. when a more-privileged
/// role created it). Without this, any role with `EXECUTE` on
/// `pinecone.schedule_maintenance` (PUBLIC by default, like any plain
/// `#[pg_extern]`) could register someone ELSE's index for maintenance; once
/// `pinecone.auto_maintain` is on, the background worker — which runs with its
/// OWN identity, not the scheduling caller's — would eventually REINDEX it.
/// See `_maintenance_attempt`'s own `SET ROLE` for the complementary half:
/// even with this check, the REINDEX itself is kept from ever running with
/// the worker's own (typically superuser) privilege.
fn require_table_owner(idx: &PgRelation, func: &str) {
    let table_relid = unsafe { pg_sys::IndexGetRelation(idx.oid(), false) };
    let caller = unsafe { pg_sys::GetUserId() };
    let owns =
        unsafe { pg_sys::object_ownercheck(pg_sys::RelationRelationId, table_relid, caller) };
    if !owns {
        error!(
            "{func}: must own \"{}\"'s underlying table (or be superuser) to schedule \
             maintenance on it",
            idx.name()
        );
    }
}

/// Requires the calling role to have `SELECT` on `idx`'s underlying table —
/// the read-only counterpart of [`require_table_owner`], for surfaces that
/// only ever REPORT information (never trigger a REINDEX): [`index_drift`]
/// computes its `(inserted_since_build, total, drift_fraction)` straight
/// from the index's own metapage buffer, entirely outside SQL/the executor,
/// so — unlike a plain `SELECT count(*) FROM tbl`, which the executor's own
/// ACL check would already gate — nothing stopped an arbitrary caller with
/// `EXECUTE` on `pinecone.index_drift` (PUBLIC by default) from learning a
/// table's approximate row count regardless of whether they could actually
/// read the table. `SELECT`, not ownership, is
/// the right bar here — parity with `pinecone.status()`'s own `local_rows`
/// column, which already degrades to `NULL` rather than leak a count past
/// `catalog::has_select_privilege`.
fn require_readable(idx: &PgRelation, func: &str) {
    let table_relid = unsafe { pg_sys::IndexGetRelation(idx.oid(), false) };
    if !crate::catalog::has_select_privilege(table_relid) {
        error!(
            "{func}: insufficient privilege — SELECT on \"{}\"'s underlying table is required",
            idx.name()
        );
    }
}

/// Pure drift arithmetic — factored out so it's plain-`#[test]`-able (no
/// Postgres needed), the same "pure decision function, unit-tested directly"
/// pattern `local::insert::should_warn_flat_growth`/
/// `worker::parse_databases_list` already use. `total` is the CURRENT
/// effective row count (`total_vectors` — frozen at build time — plus
/// `inserted_since_build`), so e.g. a build of 1000 rows with 300 further
/// inserts reports `(300, 1300, 300/1300)`.
fn drift_from_counts(total_vectors: u32, inserted_since_build: u32) -> (i64, i64, f64) {
    let inserted = inserted_since_build as i64;
    let total = total_vectors as i64 + inserted;
    let drift_fraction = if total == 0 {
        0.0
    } else {
        inserted as f64 / total as f64
    };
    (inserted, total, drift_fraction)
}

/// Core drift computation, once `idx` is already known to be a local-mode
/// pg_pinecone index (both this module's own callers and [`index_drift`] itself
/// funnel through [`require_local_pg_pinecone_index`] first).
fn compute_drift(idx: &PgRelation, meta: &metapage::MetaPage) -> (i64, i64, f64) {
    let inserted = unsafe { metapage::read_inserted_count_fresh(idx.as_ptr()) };
    drift_from_counts(meta.total_vectors, inserted)
}

/// Implements `pinecone.index_drift`: `(inserted_since_build, total,
/// drift_fraction)` for `idx`, a local-mode pg_pinecone access-method index.
/// Requires `SELECT` on `idx`'s underlying table — read-only, but it would
/// otherwise leak an approximate row count to any caller with bare `EXECUTE`,
/// bypassing the table's own ACL entirely
/// (see [`require_readable`]'s own rustdoc).
pub(crate) fn index_drift(idx: PgRelation) -> (i64, i64, f64) {
    let meta = require_local_pg_pinecone_index(&idx, "pinecone.index_drift");
    require_readable(&idx, "pinecone.index_drift");
    compute_drift(&idx, &meta)
}

/// Implements `pinecone.index_format`: `(format_version, readable)` of `idx`, a pg_pinecone access-method
/// index of either mode. Reads block 0 fresh and never refuses (`metapage::peek`), so it can
/// diagnose the index every other call rejects; holds `AccessShareLock`, so a REINDEX finishes first.
pub(crate) fn index_format(idx: PgRelation) -> (Option<i32>, bool) {
    if !is_pg_pinecone_am_index(&idx) {
        error!(
            "pinecone.index_format: \"{}\" is not a pg_pinecone access-method index",
            idx.name()
        );
    }
    let idx =
        unsafe { PgRelation::with_lock(idx.oid(), pg_sys::AccessShareLock as pg_sys::LOCKMODE) };
    metapage::describe_format(unsafe { metapage::peek(idx.as_ptr()) })
}

/// What `pinecone.reindex_native_indexes` did with one index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reindexed {
    /// Below this build's format (a dry run says no more).
    Stale,
    Rebuilt,
    /// Gone since the listing (dropped, renamed, or its schema dropped).
    Skipped,
}

impl Reindexed {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Stale => "stale",
            Self::Rebuilt => "rebuilt",
            Self::Skipped => "skipped",
        }
    }
}

/// Implements `pinecone.index_calibration`: what the build calibrated for `idx`, a pg_pinecone
/// access-method index of either mode, and the probe count and pool its scans use by default
/// (`metapage::MetaPage::calibration_report`). Needs `SELECT` on the underlying table, like
/// [`index_drift`]: the numbers describe its data.
#[allow(clippy::type_complexity)]
pub(crate) fn index_calibration(
    idx: PgRelation,
) -> (
    Option<i32>,
    Option<i32>,
    Option<i32>,
    Option<i32>,
    Option<f32>,
    i32,
    Option<f32>,
    Option<bool>,
    Option<String>,
    i32,
) {
    if !is_pg_pinecone_am_index(&idx) {
        error!(
            "pinecone.index_calibration: \"{}\" is not a pg_pinecone access-method index",
            idx.name()
        );
    }
    require_readable(&idx, "pinecone.index_calibration");
    let idx =
        unsafe { PgRelation::with_lock(idx.oid(), pg_sys::AccessShareLock as pg_sys::LOCKMODE) };
    let report = metapage::read(idx.as_ptr()).calibration_report();
    (
        report.nprobe,
        report.pool,
        report.calib_nprobe,
        report.calib_pool,
        report.calib_recall,
        report.calib_samples,
        report.target_recall,
        report.calib_reached,
        report.queries_source.map(str::to_string),
        report.queries_count,
    )
}

/// Implements `pinecone.reindex_native_indexes`: the native-mode pg_pinecone indexes the calling role
/// can REINDEX (it owns their table, or is superuser or a member of the owner) that are below this
/// build's format, `REINDEX INDEX` each (unless `dry_run`), each with its outcome. Remote-mode
/// indexes carry no codes and are never touched; a second call finds nothing. Runs as the caller
/// (no privilege change): each statement is the REINDEX the caller could type.
pub(crate) fn reindex_native_indexes(dry_run: bool) -> Vec<(String, Reindexed)> {
    rebuild_native_indexes(stale_native_indexes(), dry_run)
}

/// The schema-qualified names of the native indexes the caller could REINDEX that are below this
/// build's format, in name order. The listing is the first half of [`reindex_native_indexes`];
/// the rebuilds follow it by minutes, so an index can be gone by then.
pub(crate) fn stale_native_indexes() -> Vec<String> {
    let candidates: Vec<(pg_sys::Oid, String)> = Spi::connect(|client| {
        client
            .select(
                "SELECT c.oid, pg_catalog.quote_ident(n.nspname) || '.' \
                     || pg_catalog.quote_ident(c.relname) AS name \
                 FROM pg_catalog.pg_class c \
                 JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace \
                 JOIN pg_catalog.pg_am a ON a.oid = c.relam \
                 WHERE a.amname = 'pinecone' AND c.relkind = 'i' \
                 ORDER BY n.nspname, c.relname",
                None,
                &[],
            )
            .expect("pg_pinecone: reindex_native_indexes catalog query failed")
            .map(|row| {
                (
                    row.get_by_name::<pg_sys::Oid, _>("oid")
                        .expect("pg_pinecone: reindex_native_indexes oid read failed")
                        .expect("pg_pinecone: pg_class.oid is never NULL"),
                    row.get_by_name::<String, _>("name")
                        .expect("pg_pinecone: reindex_native_indexes name read failed")
                        .expect("pg_pinecone: a quoted name is never NULL"),
                )
            })
            .collect()
    });

    let caller = unsafe { pg_sys::GetUserId() };
    let mut listed = Vec::new();
    for (oid, name) in candidates {
        // Same bar as REINDEX itself: ownership of the table, not of the index.
        let table = unsafe { pg_sys::IndexGetRelation(oid, true) };
        if table == pg_sys::Oid::INVALID
            || !unsafe { pg_sys::object_ownercheck(pg_sys::RelationRelationId, table, caller) }
        {
            continue;
        }
        // The peek holds `AccessShareLock` for its own length only; REINDEX takes its own locks.
        let stale = {
            let idx =
                unsafe { PgRelation::with_lock(oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE) };
            unsafe { metapage::stale_native_format(idx.as_ptr()) }.is_some()
        };
        if stale {
            listed.push(name);
        }
    }
    listed
}

/// The second half of [`reindex_native_indexes`]: rebuilds `names` in order. Each REINDEX runs in a
/// subtransaction of its own, so an index that is gone (a missing relation or schema) is reported
/// `skipped` and costs only itself; any other error aborts the call, with its own SQLSTATE. The
/// subtransactions keep the rebuilds: Postgres holds the locks of a committed one until the
/// statement's transaction ends, so every rebuilt index stays locked until the call returns
/// (`REINDEX INDEX` per statement releases each as it finishes).
pub(crate) fn rebuild_native_indexes(
    names: Vec<String>,
    dry_run: bool,
) -> Vec<(String, Reindexed)> {
    let outcome = |name: &str| {
        if dry_run {
            return Reindexed::Stale;
        }
        let rebuilt = docs_guard::isolated(AssertUnwindSafe(|| {
            Spi::run(&format!("REINDEX INDEX {name}"))
        }));
        match rebuilt {
            Ok(Ok(())) => Reindexed::Rebuilt,
            Ok(Err(e)) => {
                error!("pinecone.reindex_native_indexes: REINDEX INDEX {name} failed: {e}")
            }
            Err(gone) if gone.names_a_missing_relation() => Reindexed::Skipped,
            Err(caught) => backend::error_with_code(
                caught.code,
                format!(
                    "pinecone.reindex_native_indexes: REINDEX INDEX {name} failed: {}",
                    caught.message
                ),
                None,
                None,
            ),
        }
    };
    names
        .into_iter()
        .map(|name| {
            let done = outcome(&name);
            (name, done)
        })
        .collect()
}

/// Implements `pinecone.schedule_maintenance`: upserts `idx`'s
/// `pinecone.index_maintenance` row. Re-running this on an already-scheduled
/// index updates its threshold and clears any prior park — the poison-escape
/// hatch (see this module's own docstring); there is no separate
/// `..._retry` function. Requires the caller to own `idx`'s underlying table —
/// without this, an unprivileged caller could enqueue a REINDEX-triggering
/// schedule against an index they don't
/// own; `scheduled_by` records exactly who this check passed
/// for, for audit.
pub(crate) fn schedule_maintenance(idx: PgRelation, drift_threshold: f64) {
    require_local_pg_pinecone_index(&idx, "pinecone.schedule_maintenance");
    require_table_owner(&idx, "pinecone.schedule_maintenance");
    if !(drift_threshold > 0.0 && drift_threshold <= 1.0) {
        error!(
            "pinecone.schedule_maintenance: drift_threshold must be in the range (0, 1] — got \
             {drift_threshold}"
        );
    }
    Spi::run_with_args(
        "INSERT INTO pinecone.index_maintenance (index_relid, drift_threshold, scheduled_by) \
         VALUES ($1::oid::regclass, $2, current_user::regrole) \
         ON CONFLICT (index_relid) DO UPDATE SET \
             drift_threshold = EXCLUDED.drift_threshold, \
             scheduled_by = EXCLUDED.scheduled_by, \
             queue_parked = false, parked_reason = NULL, attempts = 0, last_error = NULL",
        &[idx.oid().into(), drift_threshold.into()],
    )
    .expect("pg_pinecone: schedule_maintenance upsert failed");
}

/// Current value of the `pinecone.auto_maintain` GUC — default OFF
/// (auto-REINDEX is opt-in). [`run_round`] is a complete no-op
/// while this is off, regardless of how stale any scheduled index's drift
/// is — scheduling alone only buys visibility (`pinecone.index_drift`/
/// `pinecone.maintenance_status`), never an unannounced blocking REINDEX.
fn auto_maintain_enabled() -> bool {
    crate::AUTO_MAINTAIN.get()
}

struct Candidate {
    index_relid: pg_sys::Oid,
    drift_threshold: f64,
}

fn candidates() -> Vec<Candidate> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT index_relid::oid AS index_relid, drift_threshold \
                 FROM pinecone.index_maintenance WHERE NOT queue_parked",
                None,
                &[],
            )
            .expect("pg_pinecone: maintenance candidate scan failed")
            .map(|row| Candidate {
                index_relid: row
                    .get_by_name::<pg_sys::Oid, _>("index_relid")
                    .expect("pg_pinecone: maintenance candidate index_relid query failed")
                    .expect("pg_pinecone: maintenance candidate index_relid unexpectedly NULL"),
                drift_threshold: row
                    .get_by_name::<f64, _>("drift_threshold")
                    .expect("pg_pinecone: maintenance candidate drift_threshold query failed")
                    .expect("pg_pinecone: maintenance candidate drift_threshold unexpectedly NULL"),
            })
            .collect()
    })
}

struct AttemptResult {
    reindexed: bool,
    drift_fraction: Option<f64>,
    gone: bool,
    error: Option<String>,
}

/// Whether `index_relid` is a native index below this build's format, offline until REINDEX. An
/// index that is gone is not: the attempt reports it.
fn is_offline(index_relid: pg_sys::Oid) -> bool {
    const LOCK: pg_sys::LOCKMODE = pg_sys::AccessShareLock as pg_sys::LOCKMODE;
    unsafe {
        let index = pg_sys::try_relation_open(index_relid, LOCK);
        if index.is_null() {
            return false;
        }
        let offline = metapage::stale_native_format(index).is_some();
        pg_sys::relation_close(index, LOCK);
        offline
    }
}

fn attempt(index_relid: pg_sys::Oid, drift_threshold: f64) -> AttemptResult {
    // An offline index counts as drift 1.0, which reaches any threshold: the rebuild is
    // the only way back, and the attempt would otherwise leave it offline until it had drifted.
    let drift_threshold = if is_offline(index_relid) {
        0.0
    } else {
        drift_threshold
    };
    Spi::connect(|client| {
        let mut rows = client
            .select(
                "SELECT * FROM pinecone._maintenance_attempt($1, $2)",
                None,
                &[index_relid.into(), drift_threshold.into()],
            )
            .expect("pg_pinecone: _maintenance_attempt call failed");
        let row = rows
            .next()
            .expect("pg_pinecone: _maintenance_attempt unexpectedly returned no rows");
        AttemptResult {
            reindexed: row
                .get_by_name::<bool, _>("reindexed")
                .expect("pg_pinecone: maintenance attempt reindexed query failed")
                .expect("pg_pinecone: maintenance attempt reindexed unexpectedly NULL"),
            drift_fraction: row
                .get_by_name::<f64, _>("drift_fraction")
                .expect("pg_pinecone: maintenance attempt drift_fraction query failed"),
            gone: row
                .get_by_name::<bool, _>("gone")
                .expect("pg_pinecone: maintenance attempt gone query failed")
                .expect("pg_pinecone: maintenance attempt gone unexpectedly NULL"),
            error: row
                .get_by_name::<String, _>("error")
                .expect("pg_pinecone: maintenance attempt error query failed"),
        }
    })
}

/// Deletes a schedule whose index no longer exists (dropped between
/// `schedule_maintenance` and this round) — mirrors
/// `outbox::dispatch_backfill`'s dropped-relid purge.
fn purge_gone(index_relid: pg_sys::Oid) {
    let deleted = Spi::get_one_with_args::<i64>(
        "WITH d AS (DELETE FROM pinecone.index_maintenance WHERE index_relid::oid = $1::oid \
         RETURNING 1) SELECT count(*) FROM d",
        &[index_relid.into()],
    )
    .expect("pg_pinecone: maintenance purge failed")
    .expect("pg_pinecone: maintenance purge count(*) unexpectedly NULL");
    if deleted > 0 {
        warning!(
            "pg_pinecone: maintenance schedule for index oid {index_relid} purged — the index no \
             longer exists"
        );
    }
}

/// Records a failed attempt: increments `attempts`, sets `last_error`, and —
/// past `pinecone.outbox_max_attempts` consecutive failures (the SAME GUC and
/// vocabulary the outbox op framework itself uses) — parks the schedule.
fn record_failure(index_relid: pg_sys::Oid, message: &str) {
    let attempts = Spi::get_one_with_args::<i32>(
        "UPDATE pinecone.index_maintenance SET attempts = attempts + 1, last_error = $2, \
         last_checked_at = now() WHERE index_relid::oid = $1::oid RETURNING attempts",
        &[index_relid.into(), message.into()],
    )
    .expect("pg_pinecone: maintenance failure record update failed")
    .expect("pg_pinecone: maintenance failure record RETURNING attempts unexpectedly NULL");

    if attempts >= crate::OUTBOX_MAX_ATTEMPTS.get() {
        Spi::run_with_args(
            "UPDATE pinecone.index_maintenance SET queue_parked = true, parked_reason = $2 \
             WHERE index_relid::oid = $1::oid",
            &[index_relid.into(), message.into()],
        )
        .expect("pg_pinecone: maintenance park update failed");
    }
}

/// Records a successful attempt (whether or not it actually REINDEXed) —
/// resets the failure counter, and records the drift fraction observed
/// (`0.0` post-REINDEX, since the rebuild resets the underlying counters).
fn record_success(index_relid: pg_sys::Oid, reindexed: bool, drift_fraction: f64) {
    let drift_fraction = if reindexed { 0.0 } else { drift_fraction };
    Spi::run_with_args(
        "UPDATE pinecone.index_maintenance SET attempts = 0, last_error = NULL, \
         last_checked_at = now(), last_drift_fraction = $2, \
         last_maintained_at = CASE WHEN $3 THEN now() ELSE last_maintained_at END \
         WHERE index_relid::oid = $1::oid",
        &[index_relid.into(), drift_fraction.into(), reindexed.into()],
    )
    .expect("pg_pinecone: maintenance success record update failed");
}

/// The worker's per-round maintenance entry point (`worker.rs` calls this
/// once per round, alongside `outbox::Scheduler::round`) — also
/// called directly, Rust-internal, by tests (same "no SQL surface" pattern
/// `Scheduler::round` itself uses). A complete no-op while
/// `pinecone.auto_maintain` is off. Returns how many indexes were actually
/// REINDEXed this round, for test assertions.
pub(crate) fn run_round() -> u32 {
    if !auto_maintain_enabled() {
        return 0;
    }
    let mut reindexed = 0u32;
    for c in candidates() {
        let result = attempt(c.index_relid, c.drift_threshold);
        if result.gone {
            purge_gone(c.index_relid);
            continue;
        }
        if let Some(err) = result.error {
            record_failure(c.index_relid, &err);
            continue;
        }
        record_success(
            c.index_relid,
            result.reindexed,
            result.drift_fraction.unwrap_or(0.0),
        );
        if result.reindexed {
            reindexed += 1;
        }
    }
    reindexed
}

/// One `pinecone.maintenance_status()` row.
pub(crate) struct MaintenanceStatusRow {
    pub(crate) index_name: String,
    pub(crate) drift_threshold: f64,
    pub(crate) inserted_since_build: Option<i64>,
    pub(crate) total: Option<i64>,
    pub(crate) drift_fraction: Option<f64>,
    pub(crate) queue_parked: bool,
    pub(crate) last_error: Option<String>,
    pub(crate) last_maintained_at: Option<TimestampWithTimeZone>,
}

struct ScheduleRow {
    index_relid: pg_sys::Oid,
    drift_threshold: f64,
    queue_parked: bool,
    last_error: Option<String>,
    last_maintained_at: Option<TimestampWithTimeZone>,
}

/// Implements `pinecone.maintenance_status`: one row per index registered via
/// `pinecone.schedule_maintenance`, with a FRESH (not stored) drift read —
/// cheap, since this table is expected to hold at most a handful of rows
/// (an operator-scheduled maintenance policy per index, not a per-write
/// queue). An index dropped since it was scheduled shows
/// `index_name = '<dropped:OID>'` and `NULL` drift columns, same convention
/// `status::status` uses for a dropped tracked table — never errors.
pub(crate) fn maintenance_status() -> Vec<MaintenanceStatusRow> {
    let rows: Vec<ScheduleRow> = Spi::connect(|client| {
        client
            .select(
                "SELECT index_relid::oid AS index_relid, drift_threshold, queue_parked, \
                 last_error, last_maintained_at \
                 FROM pinecone.index_maintenance ORDER BY index_relid::oid",
                None,
                &[],
            )
            .expect("pg_pinecone: maintenance_status scan failed")
            .map(|row| ScheduleRow {
                index_relid: row
                    .get_by_name::<pg_sys::Oid, _>("index_relid")
                    .expect("pg_pinecone: maintenance_status index_relid query failed")
                    .expect("pg_pinecone: maintenance_status index_relid unexpectedly NULL"),
                drift_threshold: row
                    .get_by_name::<f64, _>("drift_threshold")
                    .expect("pg_pinecone: maintenance_status drift_threshold query failed")
                    .expect("pg_pinecone: maintenance_status drift_threshold unexpectedly NULL"),
                queue_parked: row
                    .get_by_name::<bool, _>("queue_parked")
                    .expect("pg_pinecone: maintenance_status queue_parked query failed")
                    .expect("pg_pinecone: maintenance_status queue_parked unexpectedly NULL"),
                last_error: row
                    .get_by_name::<String, _>("last_error")
                    .expect("pg_pinecone: maintenance_status last_error query failed"),
                last_maintained_at: row
                    .get_by_name::<TimestampWithTimeZone, _>("last_maintained_at")
                    .expect("pg_pinecone: maintenance_status last_maintained_at query failed"),
            })
            .collect()
    });

    rows.into_iter()
        .map(|r| {
            let (index_name, drift) = live_drift(r.index_relid);
            MaintenanceStatusRow {
                index_name,
                drift_threshold: r.drift_threshold,
                inserted_since_build: drift.map(|d| d.0),
                total: drift.map(|d| d.1),
                drift_fraction: drift.map(|d| d.2),
                queue_parked: r.queue_parked,
                last_error: r.last_error,
                last_maintained_at: r.last_maintained_at,
            }
        })
        .collect()
}

/// `(index_name, drift)` for `index_relid` — `drift = None` (and a
/// `'<dropped:OID>'` name) if it no longer resolves.
fn live_drift(index_relid: pg_sys::Oid) -> (String, Option<(i64, i64, f64)>) {
    let exists = Spi::get_one_with_args::<bool>(
        "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_class WHERE oid = $1::oid)",
        &[index_relid.into()],
    )
    .expect("pg_pinecone: maintenance_status existence query failed")
    .expect("pg_pinecone: EXISTS unexpectedly NULL");
    if !exists {
        return (format!("<dropped:{index_relid}>"), None);
    }
    let idx =
        unsafe { PgRelation::with_lock(index_relid, pg_sys::AccessShareLock as pg_sys::LOCKMODE) };
    let name = idx.name().to_string();
    let meta = metapage::read(idx.as_ptr());
    warn_if_offline(&idx, &meta, "maintenance_status");
    (name, Some(compute_drift(&idx, &meta)))
}

/// Registers this module's GUC — called from the crate's `_PG_init`.
pub(crate) fn init_gucs() {
    pgrx::guc::GucRegistry::define_bool_guc(
        c"pinecone.auto_maintain",
        c"Whether the pg_pinecone background worker automatically REINDEXes a local-mode pg_pinecone \
          index once pinecone.schedule_maintenance's drift threshold is reached. A maintenance \
          REINDEX is a plain, blocking rebuild (same locks as any REINDEX) — off by default so \
          it never fires unannounced; schedule_maintenance/index_drift/maintenance_status work \
          regardless of this GUC.",
        c"",
        &crate::AUTO_MAINTAIN,
        pgrx::guc::GucContext::Userset,
        pgrx::guc::GucFlags::default(),
    );
}

// Pure-Rust (no pgrx/pg dependency), unlike this crate's `#[pg_test]`s — runs
// under plain `cargo test`/`cargo pgrx test` alike (same split `worker.rs`'s
// own `mod tests` uses).
#[cfg(test)]
mod tests {
    use super::drift_from_counts;

    #[test]
    fn drift_from_counts_build_1000_insert_300() {
        let (inserted, total, drift_fraction) = drift_from_counts(1000, 300);
        assert_eq!(inserted, 300);
        assert_eq!(total, 1300);
        assert!(
            (drift_fraction - 300.0 / 1300.0).abs() < 1e-9,
            "drift_fraction = {drift_fraction}"
        );
        // The task's own headline figure, rounded.
        assert!((drift_fraction - 0.230769).abs() < 1e-5);
    }

    #[test]
    fn drift_from_counts_no_inserts_is_zero_drift() {
        assert_eq!(drift_from_counts(1000, 0), (0, 1000, 0.0));
    }

    #[test]
    fn drift_from_counts_empty_index_is_zero_drift_not_nan() {
        // total_vectors == 0 and inserted_since_build == 0 (a freshly built,
        // still-empty index) must never divide by zero.
        assert_eq!(drift_from_counts(0, 0), (0, 0, 0.0));
    }

    #[test]
    fn drift_from_counts_all_inserted_is_full_drift() {
        // A flat/bootstrap index built with zero rows, then populated
        // entirely via aminsert: total_vectors stays 0 forever (build-frozen),
        // so drift_fraction must still read 1.0, not divide-by-zero.
        assert_eq!(drift_from_counts(0, 500), (500, 500, 1.0));
    }
}
