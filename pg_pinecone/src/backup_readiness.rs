//! Implementation of `pinecone.backup_readiness` (the backup
//! gate, decomposed into three standalone, read-only checks — the
//! `detach_storage()` precondition gate can reuse this logic once
//! 0.5's metapage v2 merges, but nothing here mutates anything: no shed
//! verbs, no `storage` registry column, no side effects at all. Every
//! Pinecone call this module makes is read-only
//! (`describe_index_stats`/`list_backups`/`get_backup_schedule`) — safe to
//! run with a ControlPlaneViewer-scoped key.
//!
//! Three checks, one row each (`check` in `('sync', 'coverage',
//! 'freshness')`):
//!
//! 1. **sync**: live per-namespace vector count (`describe_index_stats`) vs.
//!    local non-NULL-vector row count, tolerating an in-flight
//!    `pinecone.outbox` backlog (both numbers, plus the backlog, are always in
//!    `detail`).
//! 2. **coverage**: the newest `Ready` backup's `namespace_count`/
//!    `dimension` against the live index — catches a backup that predates
//!    this table's namespace, or one built for a different dimension; also
//!    reports whether this table's namespace is the index's only one
//!    (backups are whole-index, so a shared index's backup covers every
//!    tenant, not just this table — the "shared-index nuance").
//! 3. **freshness**: the newest `Ready` backup's age against
//!    `schedule_interval + max(25% of interval, 1h) + 15min` (Pinecone's own
//!    staleness allowance — backups only include vectors written >= 15 min
//!    before the backup was taken). No configured schedule is ITSELF a
//!    failing check: absent a schedule, worst-case RPO is unbounded.
//!
//! A table with a partition column counts its whole family (`prefix:*`) in the `sync` and
//! `coverage` checks, names the namespaces of the family the registry does not list, and
//! warns above 2,000 namespaces in the index (a per-backup figure not verified against
//! live Pinecone).
//!
//! A remote-call failure degrades that ONE row to a `status` of the shape
//! `'unavailable (...)'` (plan-gated 403, auth failure, network error, or an
//! unparsable schedule/timestamp) — it never aborts the other two checks,
//! and never raises a Postgres `ERROR`. The only things that DO raise: `tbl`
//! not being a promoted, tracked table (same precondition every other
//! `pinecone.*` remote operation uses) and credential resolution itself
//! (`backend::client_for_effective_key`, same as `pinecone.promote`/`sync`/
//! `repoint`).

use pg_pinecone_client::{BackupInfo, Error as ClientError, IndexStats};
use pgrx::{
    PgRelation, pg_sys,
    prelude::*,
    spi::{Spi, quote_qualified_identifier},
};
use serde_json::Value;

use crate::{
    backend, catalog, outbox,
    partition::{self, Lane, PartitionSpec},
    partition_ops,
    partition_registry::{self, Failed},
    registry::{self, RegistryRow},
};

/// One `pinecone.backup_readiness` output row.
type ReadinessRow = (String, String, String);

/// Looks up `tbl`'s registry row, requiring it to exist and be `'remote'`
/// exactly — same shape as `promote::remote_row`, but deliberately NOT
/// calling `registry::require_not_am`: unlike `promote`/`demote`/`sync`/
/// `track`/`untrack`/`topk`, this is a read-only diagnostic equally
/// meaningful for a pg_pinecone-AM-indexed table (its `vector_column`'s non-NULL
/// count is just as good a SYNC signal there).
fn readiness_row(tbl: &PgRelation, func: &str) -> RegistryRow {
    let relid = tbl.oid();
    let display = catalog::display_name(tbl);
    let row = registry::lookup(relid).unwrap_or_else(|| {
        error!(
            "{func}: {display} is not tracked — run pinecone.track(...) or pinecone.promote(...) first"
        )
    });
    if row.mode != "remote" {
        error!(
            "{func}: {display} is not promoted (mode = '{}') — run pinecone.promote(...) first",
            row.mode
        );
    }
    row
}

/// Implements `pinecone.backup_readiness`: three independent checks, each
/// degrading its own row to an `'unavailable (...)'` status on a remote-call
/// failure rather than aborting the other two.
pub(crate) fn backup_readiness(tbl: PgRelation) -> Vec<ReadinessRow> {
    compute_gate(&tbl, "pinecone.backup_readiness").rows
}

/// [`compute_gate`]'s full result — the three check rows `backup_readiness`
/// returns, plus two extra facts `detach::detach_storage`'s backup gate
/// needs from the SAME HTTP round rather than paying for
/// a second one: the newest `Ready` backup's id (persisted as
/// `pinecone.tables.detach_backup_id`) and the configured schedule's interval
/// in seconds (persisted as `pinecone.tables.detach_rpo_secs`, so
/// `pinecone.status()` can format a worst-case RPO with zero network access of
/// its own — see that column's rustdoc in `registry.rs`).
pub(crate) struct GateResult {
    pub(crate) rows: Vec<ReadinessRow>,
    pub(crate) backup_id: Option<String>,
    pub(crate) schedule_interval_secs: Option<i64>,
}

/// `detach::detach_storage`'s own entry point: identical HTTP calls to
/// `backup_readiness`, but returns the fuller [`GateResult`]. `func` in the
/// three check rows' own error paths still reads correctly for a
/// `detach_storage`-context error since [`readiness_row`] takes it as a
/// parameter.
pub(crate) fn detach_gate(tbl: &PgRelation) -> GateResult {
    compute_gate(tbl, "pinecone.detach_storage")
}

/// Shared by [`backup_readiness`] and [`detach_gate`] — one
/// `describe_index_stats`/`list_backups`/`get_backup_schedule` round,
/// producing both the three-row diagnostic and (for the latter caller) the
/// backup id/schedule interval extracted from the very same responses.
fn compute_gate(tbl: &PgRelation, func: &str) -> GateResult {
    let row = readiness_row(tbl, func);
    let relid = tbl.oid();
    let index_name = row
        .index_name
        .clone()
        .expect("pg_pinecone: backup gate: remote row always has index_name");
    let namespace = row
        .namespace
        .clone()
        .expect("pg_pinecone: backup gate: remote row always has namespace");

    let client = backend::client_for_effective_key();
    let stats = client.describe_index_stats(&index_name);
    let backups = client.list_backups(&index_name);
    let schedule = client.get_backup_schedule(&index_name);

    let backup_id = backups
        .as_ref()
        .ok()
        .and_then(|list| newest_ready_backup(list))
        .map(|b| b.backup_id.clone());
    let schedule_interval_secs = schedule
        .as_ref()
        .ok()
        .and_then(|opt| opt.as_ref())
        .and_then(parse_schedule_interval_secs);

    let family = row
        .partition
        .as_ref()
        .and_then(|spec| FamilyView::of(relid, &namespace, spec, &stats));
    let failed = match row.partition {
        Some(_) => partition_registry::failed_of(relid, Lane::Vector),
        None => Vec::new(),
    };
    let rows = vec![
        sync_check(
            tbl,
            &row,
            relid,
            &namespace,
            &stats,
            family.as_ref(),
            &failed,
        ),
        coverage_check(&row, &stats, &backups, family.as_ref()),
        freshness_check(&backups, &schedule),
    ];
    GateResult {
        rows,
        backup_id,
        schedule_interval_secs,
    }
}

/// Backups of an index with more namespaces than this are unverified: 2,000 is the per-backup
/// figure pg_pinecone' documentation assumes, not one confirmed against live Pinecone.
const BACKUP_NAMESPACES_UNVERIFIED_ABOVE: u64 = 2_000;

/// What the index's own namespace stats say of a partitioned table's family (`prefix:*`) against
/// the registry: the vectors it holds and how its namespaces sort out. The two checks that read
/// the index (`sync`, `coverage`) count the whole family instead of one namespace.
struct FamilyView {
    prefix: String,
    vectors: u64,
    family: partition_ops::Family,
}

impl FamilyView {
    /// `None` when the index stats are unavailable (the check then reports that, as always).
    fn of(
        relid: pg_sys::Oid,
        prefix: &str,
        spec: &PartitionSpec,
        stats: &Result<IndexStats, ClientError>,
    ) -> Option<Self> {
        let stats = stats.as_ref().ok()?;
        let held: Vec<String> = stats.namespaces.keys().cloned().collect();
        let registered: Vec<String> = partition_registry::partitions_of(relid, Lane::Vector)
            .into_iter()
            .map(|(_, namespace)| namespace)
            .collect();
        let family = partition_ops::classify(prefix, spec.kind, &held, &registered);
        let inside = partition::family_prefix(prefix);
        let vectors = (stats.namespaces.iter())
            .filter(|(name, _)| name.starts_with(&inside))
            .map(|(_, ns)| ns.vector_count)
            .sum();
        Some(Self {
            prefix: prefix.to_string(),
            vectors,
            family,
        })
    }

    /// `remote family 'public.t:*', 3 namespaces`: what the `live(..)` count is of.
    fn describe(&self) -> String {
        let n = self.family.held;
        let plural = if n == 1 { "namespace" } else { "namespaces" };
        format!("remote family '{}:*', {n} {plural}", self.prefix)
    }

    /// `; unregistered namespaces in the family: a, b (+N more)`, or nothing when the registry
    /// knows every namespace of the family: such vectors no row of the table accounts for.
    fn unregistered_note(&self) -> String {
        let names: Vec<&str> = (self.family.unregistered.iter().map(|(_, ns)| ns.as_str()))
            .chain(self.family.foreign.iter().map(String::as_str))
            .collect();
        if names.is_empty() {
            return String::new();
        }
        let shown = names.iter().take(5).copied().collect::<Vec<_>>().join(", ");
        let more = names.len().saturating_sub(5);
        let more = if more > 0 {
            format!(" (+{more} more)")
        } else {
            String::new()
        };
        format!("; unregistered namespaces in the family: {shown}{more}")
    }
}

/// `; failed partitions: public.t:2, public.t:7 (+1 more)`, or nothing when none is failed: the
/// vector lane skips those, so Pinecone lacks what they were last written with.
fn failed_note(failed: &[Failed]) -> String {
    if failed.is_empty() {
        return String::new();
    }
    let names: Vec<&str> = failed.iter().map(|f| f.namespace.as_str()).collect();
    let shown = names.iter().take(5).copied().collect::<Vec<_>>().join(", ");
    let more = names.len().saturating_sub(5);
    let more = if more > 0 {
        format!(" (+{more} more)")
    } else {
        String::new()
    };
    format!("; failed partitions: {shown}{more} (pinecone.failed_partitions says why)")
}

/// Non-NULL `vector_column` row count for `tbl`, or `None` if `current_user`
/// lacks `SELECT` on it — same degrade-rather-than-error convention
/// `status::status`'s own `local_row_count` uses.
fn local_non_null_count(tbl: &PgRelation, row: &RegistryRow) -> Option<i64> {
    if !catalog::has_select_privilege(tbl.oid()) {
        return None;
    }
    let table = quote_qualified_identifier(tbl.namespace(), tbl.name());
    let vcol = pgrx::spi::quote_identifier(&row.vector_column);
    Some(
        Spi::get_one::<i64>(&format!(
            "SELECT count(*) FROM {table} WHERE {vcol} IS NOT NULL"
        ))
        .expect("pg_pinecone: backup_readiness local count query failed")
        .expect("pg_pinecone: count(*) unexpectedly NULL"),
    )
}

/// SYNC check: live per-namespace vector count vs. local non-NULL count,
/// tolerating an in-flight outbox backlog. `diff == 0` is `'ok'`; a nonzero
/// diff is `'warn'` when a nonzero (or invisible-to-this-role) backlog could
/// explain it, `'fail'` when the backlog is visibly zero (no in-flight
/// explanation — a genuine drift).
///
/// A partitioned table's live count is the sum over every namespace of its family (`prefix:*`),
/// and the detail names the namespaces of the family that the registry does not list. A failed
/// partition is never `ok`, whatever the counts say, and the detail names it.
fn sync_check(
    tbl: &PgRelation,
    row: &RegistryRow,
    relid: pg_sys::Oid,
    namespace: &str,
    stats: &Result<IndexStats, ClientError>,
    family: Option<&FamilyView>,
    failed: &[Failed],
) -> ReadinessRow {
    let stats = match stats {
        Err(e) => return classify_generic_error("sync", e),
        Ok(s) => s,
    };
    let (live, what, unregistered) = match family {
        Some(family) => (
            family.vectors,
            family.describe(),
            family.unregistered_note(),
        ),
        None => (
            stats
                .namespaces
                .get(namespace)
                .map(|n| n.vector_count)
                .unwrap_or(0),
            format!("remote namespace '{namespace}'"),
            String::new(),
        ),
    };
    let backlog = outbox::backlog_for(relid);
    let notes = format!("{unregistered}{}", failed_note(failed));

    let Some(local) = local_non_null_count(tbl, row) else {
        let status = if failed.is_empty() { "warn" } else { "fail" };
        return (
            "sync".to_string(),
            status.to_string(),
            format!(
                "live({what})={live}; local count unavailable (no \
                 SELECT privilege on the tracked table for this role) — grant SELECT to verify{notes}"
            ),
        );
    };

    let diff = live as i64 - local;
    let backlog_text = backlog
        .map(|b| b.to_string())
        .unwrap_or_else(|| "<not visible to this role>".to_string());
    let detail = format!(
        "live({what})={live} local(non-NULL vector rows)={local} \
         diff={diff} outbox_backlog={backlog_text}{notes}"
    );
    let status = if !failed.is_empty() {
        "fail"
    } else if diff == 0 {
        "ok"
    } else {
        match backlog {
            Some(b) if b > 0 => "warn",
            Some(_) => "fail",
            None => "warn",
        }
    };
    ("sync".to_string(), status.to_string(), detail)
}

/// The newest backup with `status == "Ready"`, by `created_at` — an RFC3339
/// timestamp, lexically sortable as a plain string (see
/// `BackupInfo::created_at`'s own rustdoc). A backup with no `created_at` at
/// all is excluded — it can't be compared, and every observed shape reports
/// one once a backup exists.
fn newest_ready_backup(backups: &[BackupInfo]) -> Option<&BackupInfo> {
    backups
        .iter()
        .filter(|b| b.status == "Ready" && b.created_at.is_some())
        .max_by(|a, b| a.created_at.cmp(&b.created_at))
}

/// COVERAGE check: the newest Ready backup's `namespace_count`/`dimension`
/// against the live index, plus the shared-index nuance (whether this
/// table's namespace is the index's only one).
///
/// A partitioned table's note counts its family against the index, and an index above
/// [`BACKUP_NAMESPACES_UNVERIFIED_ABOVE`] namespaces turns an `ok` into a `warn`.
fn coverage_check(
    row: &RegistryRow,
    stats: &Result<IndexStats, ClientError>,
    backups: &Result<Vec<BackupInfo>, ClientError>,
    family: Option<&FamilyView>,
) -> ReadinessRow {
    let backups = match backups {
        Err(e) => return classify_backup_error("coverage", e),
        Ok(b) => b,
    };
    let live_namespace_count = stats.as_ref().ok().map(|s| s.namespaces.len() as u64);
    let sole_owner_note = match (live_namespace_count, family) {
        (Some(total), Some(f)) if total as usize == f.family.held => format!(
            "this table's {total} namespaces ('{}:*') are the ONLY namespaces in the index",
            f.prefix
        ),
        (Some(total), Some(f)) => format!(
            "index is SHARED — {total} namespaces total, {} of them in this table's family \
             '{}:*' (restoring this backup restores all of them, not just this table's)",
            f.family.held, f.prefix
        ),
        (Some(1), None) => "this table's namespace is the ONLY namespace in the index".to_string(),
        (Some(n), None) => format!(
            "index is SHARED — {n} namespaces total (restoring this backup restores all of \
             them, not just this table's)"
        ),
        (None, _) => "namespace count unknown (live index stats unavailable — see the sync check)"
            .to_string(),
    };
    let too_many = match (live_namespace_count, family) {
        (Some(total), Some(_)) if total > BACKUP_NAMESPACES_UNVERIFIED_ABOVE => Some(format!(
            "the index has {total} namespaces, above 2,000: a backup of that many is unverified \
             against live Pinecone (it may fail or leave namespaces out) — test a restore before \
             relying on it"
        )),
        _ => None,
    };
    let sole_owner_note = match &too_many {
        Some(warning) => format!("{sole_owner_note}; {warning}"),
        None => sole_owner_note,
    };

    let Some(backup) = newest_ready_backup(backups) else {
        return (
            "coverage".to_string(),
            "fail".to_string(),
            format!(
                "no Ready backup found for this index — create one before relying on \
                 backup-based recovery; {sole_owner_note}"
            ),
        );
    };

    let mut problems = Vec::new();
    match (backup.namespace_count, live_namespace_count) {
        (Some(bc), Some(live)) if bc < live => problems.push(format!(
            "backup namespace_count={bc} is less than live namespace_count={live} — the backup \
             predates a namespace that exists now"
        )),
        (None, _) => problems.push(
            "backup namespace_count not yet reported by Pinecone (still Pending/Initializing?)"
                .to_string(),
        ),
        _ => {}
    }
    match backup.dimension {
        Some(bd) if bd != row.dimension as u32 => problems.push(format!(
            "backup dimension={bd} does not match this table's registered dimension={}",
            row.dimension
        )),
        None => problems.push("backup dimension not reported by Pinecone".to_string()),
        _ => {}
    }

    let detail = format!(
        "newest Ready backup '{}' created_at={} namespace_count={:?} dimension={:?}; \
         {sole_owner_note}",
        backup.backup_id,
        backup.created_at.as_deref().unwrap_or("<unknown>"),
        backup.namespace_count,
        backup.dimension,
    );
    if problems.is_empty() {
        let status = if too_many.is_some() { "warn" } else { "ok" };
        ("coverage".to_string(), status.to_string(), detail)
    } else {
        (
            "coverage".to_string(),
            "fail".to_string(),
            format!("{}; {detail}", problems.join("; ")),
        )
    }
}

/// Seconds a schedule's frequency implies — Pinecone's documented schedule
/// cadences are exactly `daily`/`weekly`/`monthly`,
/// so once a schedule exists its interval is always known; `None` if the response doesn't
/// carry a recognized `frequency` field at all (the schedule response shape
/// is provisional/unverified — see `pg_pinecone-client::Client::get_backup_schedule`'s
/// rustdoc — this is the honest fallback for a real shape this hasn't seen
/// yet, not a bug).
fn parse_schedule_interval_secs(schedule: &Value) -> Option<i64> {
    let obj = schedule.get("schedule").unwrap_or(schedule);
    let frequency = obj.get("frequency")?.as_str()?;
    match frequency.to_ascii_lowercase().as_str() {
        "daily" => Some(24 * 3600),
        "weekly" => Some(7 * 24 * 3600),
        "monthly" => Some(30 * 24 * 3600),
        _ => None,
    }
}

/// 15-minute staleness allowance, in seconds — Pinecone backups only include
/// vectors written at least this long before the backup was taken
/// (`docs/disaster-recovery.md`'s qualifier 1, verbatim).
const STALENESS_ALLOWANCE_SECS: i64 = 15 * 60;

/// Minimum floor on the "grace" term of the freshness window — 1 hour, used
/// whenever 25% of the schedule interval is smaller than this.
const MIN_GRACE_SECS: i64 = 3600;

/// The freshness window: `interval + max(25% of interval, 1h) + 15min`
/// — how old the newest Ready backup is
/// allowed to be before FRESHNESS fails.
fn freshness_window_secs(interval_secs: i64) -> i64 {
    interval_secs + (interval_secs / 4).max(MIN_GRACE_SECS) + STALENESS_ALLOWANCE_SECS
}

/// `now() - created_at`, in seconds — computed via SQL (Postgres already
/// parses RFC3339 `timestamptz` literals) rather than a Rust date library,
/// wrapped in a subtransaction so an unparsable `created_at` (the backup
/// response shape is provisional — see `parse_schedule_interval_secs`'s
/// rustdoc for the same caveat) degrades this ONE check to `'warn'` instead
/// of aborting the whole `pinecone.backup_readiness` call.
fn backup_age_seconds(created_at: &str) -> Option<f64> {
    if created_at.is_empty() {
        return None;
    }
    // `extract(epoch from ...)` returns `numeric`, not `double precision` —
    // without the explicit cast, `Spi::get_one_with_args::<f64>` fails with a
    // type-mismatch `SpiError` (a plain `Err`, not a raised Postgres `ERROR`
    // `run_catching` would otherwise need to catch) on every call, which
    // silently degraded every FRESHNESS check to `'warn'` (caught during
    // pg_test development, not a live-only gap).
    run_catching(|| {
        Spi::get_one_with_args::<f64>(
            "SELECT extract(epoch from (now() - $1::timestamptz))::double precision",
            &[created_at.into()],
        )
    })
    .ok()
    .and_then(|r| r.ok())
    .flatten()
}

/// FRESHNESS check: no schedule is itself a failing check (unbounded
/// worst-case RPO); an unrecognized schedule shape is a `'warn'` (can't
/// verify, but a schedule clearly exists); otherwise the newest Ready
/// backup's age against [`freshness_window_secs`].
fn freshness_check(
    backups: &Result<Vec<BackupInfo>, ClientError>,
    schedule: &Result<Option<Value>, ClientError>,
) -> ReadinessRow {
    let schedule = match schedule {
        Err(e) => return classify_backup_error("freshness", e),
        Ok(s) => s,
    };
    let Some(schedule_value) = schedule else {
        return (
            "freshness".to_string(),
            "fail".to_string(),
            "no backup schedule configured for this index — worst-case RPO is unbounded (no \
             guaranteed backup cadence). Configure a schedule via the Pinecone console/API \
             (pg_pinecone has no schedule-creation SQL surface yet), or take a manual backup before \
             relying on backup-based DR."
                .to_string(),
        );
    };
    let Some(interval_secs) = parse_schedule_interval_secs(schedule_value) else {
        return (
            "freshness".to_string(),
            "warn".to_string(),
            "a backup schedule is configured, but pg_pinecone could not determine its interval from \
             the response shape (the backup-schedule API response shape is provisional/unverified \
             — see pg_pinecone-client's Client::get_backup_schedule) — treat RPO as unverified until \
             confirmed live"
                .to_string(),
        );
    };
    let window_secs = freshness_window_secs(interval_secs);

    let backups = match backups {
        Err(e) => return classify_backup_error("freshness", e),
        Ok(b) => b,
    };
    let Some(backup) = newest_ready_backup(backups) else {
        return (
            "freshness".to_string(),
            "fail".to_string(),
            format!(
                "no Ready backup exists to check freshness of (window would be {window_secs}s \
                 = schedule interval {interval_secs}s + max(25%,1h) + 15min staleness allowance)"
            ),
        );
    };
    let created_at = backup.created_at.as_deref().unwrap_or_default();
    match backup_age_seconds(created_at) {
        None => (
            "freshness".to_string(),
            "warn".to_string(),
            format!("could not parse backup created_at '{created_at}' as a timestamp"),
        ),
        Some(age) => {
            let status = if age <= window_secs as f64 {
                "ok"
            } else {
                "fail"
            };
            (
                "freshness".to_string(),
                status.to_string(),
                format!(
                    "backup age={age:.0}s window={window_secs}s (schedule interval=\
                     {interval_secs}s + max(25%,1h)={}s + 15min staleness allowance); backup \
                     created_at={created_at}",
                    (interval_secs / 4).max(MIN_GRACE_SECS),
                ),
            )
        }
    }
}

/// Maps a [`ClientError`] to a `'unavailable (...)'` status/detail pair, for
/// the classic (non-backup) control/data-plane surface
/// (`describe_index_stats`): a `BadRequest` here is an ordinary malformed
/// request, NOT the backup plan gate — see [`classify_backup_error`] for
/// the surface where it is.
fn classify_generic_error(check: &str, e: &ClientError) -> ReadinessRow {
    let (status, detail) = match e {
        ClientError::Auth => (
            "unavailable (authentication failed)".to_string(),
            "authentication failed — check pinecone.api_key / the stored credentials".to_string(),
        ),
        ClientError::Transport { url, reason } => (
            "unavailable (network error)".to_string(),
            format!("could not reach {url}: {reason}"),
        ),
        ClientError::Timeout { ms } => (
            "unavailable (network error)".to_string(),
            format!("request timed out after {ms}ms"),
        ),
        ClientError::BadRequest { message } => {
            ("unavailable (request refused)".to_string(), message.clone())
        }
        ClientError::IndexNotFound { index } => (
            "unavailable (error)".to_string(),
            format!("index '{index}' not found"),
        ),
        ClientError::BackupNotFound { backup_id } => (
            "unavailable (error)".to_string(),
            format!("backup '{backup_id}' not found"),
        ),
        ClientError::NamespaceNotFound { namespace } => (
            "unavailable (error)".to_string(),
            format!("namespace '{namespace}' not found"),
        ),
        ClientError::Conflict { name } => (
            "unavailable (error)".to_string(),
            format!("index '{name}' conflict"),
        ),
        ClientError::MissingApiKey => (
            "unavailable (error)".to_string(),
            "no API key configured".to_string(),
        ),
        ClientError::Api { status, message } => (
            "unavailable (error)".to_string(),
            format!("unexpected status {status}: {message}"),
        ),
    };
    (check.to_string(), status, detail)
}

/// Like [`classify_generic_error`], but for the backup control-plane surface
/// (`list_backups`/`describe_backup`/`get_backup_schedule`), where a
/// `BadRequest` is, in practice, always the 403 plan gate
/// (`pg_pinecone_client::client::backup_forbidden`: Standard/Enterprise-only
/// endpoints refusing a Starter/Builder-plan key) — the drill-discovered
/// reality. The server's own
/// message (or the client's fallback text) is quoted verbatim in `detail`.
fn classify_backup_error(check: &str, e: &ClientError) -> ReadinessRow {
    match e {
        ClientError::BadRequest { message } => (
            check.to_string(),
            "unavailable (plan does not include backups)".to_string(),
            message.clone(),
        ),
        other => classify_generic_error(check, other),
    }
}

/// Runs `body` inside its own subtransaction, converting any raised Postgres
/// `ERROR` (e.g. an unparsable `created_at` cast to `timestamptz`) into
/// `Err(())` instead of aborting the whole `backup_readiness` call — this is
/// a read-only diagnostic; one check's malformed input must degrade that
/// check's own row, not the other two. Self-contained rather than reusing
/// `recover::run_write_back_subtransaction` (same
/// `BeginInternalSubTransaction`/`RollbackAndReleaseCurrentSubTransaction`
/// shape, generalized).
fn run_catching<T>(body: impl FnOnce() -> T + std::panic::UnwindSafe) -> Result<T, ()> {
    unsafe {
        let saved_ctx = pg_sys::CurrentMemoryContext;
        let saved_owner = pg_sys::CurrentResourceOwner;
        pg_sys::BeginInternalSubTransaction(std::ptr::null());
        pg_sys::MemoryContextSwitchTo(saved_ctx);

        let result = PgTryBuilder::new(|| Ok(body()))
            .catch_others(|_caught| Err(()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freshness_window_daily_matches_the_verbatim_formula() {
        // 86400 + max(21600, 3600) + 900 = 108900
        assert_eq!(freshness_window_secs(24 * 3600), 108_900);
    }

    #[test]
    fn freshness_window_floors_the_grace_term_at_one_hour() {
        // A short (1h) interval: 25% (900s) is below the 1h floor.
        assert_eq!(freshness_window_secs(3600), 3600 + 3600 + 900);
    }

    #[test]
    fn parse_schedule_interval_recognizes_the_three_documented_cadences() {
        assert_eq!(
            parse_schedule_interval_secs(&serde_json::json!({"frequency": "daily"})),
            Some(86_400)
        );
        assert_eq!(
            parse_schedule_interval_secs(&serde_json::json!({"schedule": {"frequency": "Weekly"}})),
            Some(7 * 86_400)
        );
        assert_eq!(
            parse_schedule_interval_secs(&serde_json::json!({"frequency": "monthly"})),
            Some(30 * 86_400)
        );
    }

    #[test]
    fn parse_schedule_interval_is_none_for_an_unrecognized_shape() {
        assert_eq!(
            parse_schedule_interval_secs(&serde_json::json!({"cron": "0 0 * * *"})),
            None
        );
    }
}
