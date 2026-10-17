//! Implementation of `pinecone.status`. `#[pg_schema]` only attaches to an
//! inline `mod {}` (see the comment on `pg_pinecone` in `lib.rs`), so its
//! `#[pg_extern]` wrapper is kept thin and delegates here.

use pgrx::{
    pg_sys,
    prelude::TimestampWithTimeZone,
    spi::{Spi, quote_qualified_identifier},
};

use crate::{
    catalog, docs_outbox,
    docs_registry::{self, DocRegistryRow},
    outbox,
    partition::{Lane, PartitionSpec},
    partition_lane, partition_registry, registry,
};

// `storage`/`backup_gate` plus `detached_at`/`detach_backup_id`/`rpo` plus
// `read_capacity` plus `attached` are APPENDED at the end, not slotted in next
// to `mode` where they'd read most naturally — every existing
// `status_rows()[i].N` test in `lib.rs`
// indexes this tuple positionally, and prepending/inserting would silently
// renumber dozens of unrelated assertions instead of erroring; appending is
// the only change that can't quietly break an existing test's meaning.
// `backup_gate`'s CONTENT changed (a placeholder hint string ->
// the actual persisted `'covered'`/`'forced'` value) without moving its
// position. The seven `docs_*` columns (the document lane) follow the same rule.
type StatusRow = (
    String,
    String,
    String,
    String,
    String,
    i32,
    Option<String>,
    Option<String>,
    Vec<String>,
    Option<i64>,
    Option<i64>,
    Option<TimestampWithTimeZone>,
    Option<String>,
    Option<String>,
    String,
    Option<String>,
    Option<TimestampWithTimeZone>,
    Option<String>,
    Option<String>,
    Option<String>,
    bool,
    Option<i64>,
    Option<TimestampWithTimeZone>,
    Option<TimestampWithTimeZone>,
    Option<i64>,
    Option<i64>,
    Option<i64>,
    Option<String>,
    Option<String>,
    Option<i64>,
    i32,
);

/// Lists every registered table. Registry + local row counts only — no HTTP,
/// and never fails: a registry row whose `relid` no longer resolves (the table
/// was dropped without `pinecone.untrack`) shows `table_name = '<dropped:OID>'`
/// (the stale `relid`, so the row stays identifiable) and `local_rows = NULL`;
/// one whose table still exists but where `current_user` lacks `SELECT` also
/// shows `local_rows = NULL` rather than erroring out of `count(*)`.
///
/// A table declared with `track_docs` also reports its document lane in the
/// `docs_*` columns (see [`DocLane`]); one that has only that registration is a
/// row of its own with `mode = 'docs'`. The registry is read for the caller
/// (nobody holds a grant on it), but only the tables they can `SELECT` show a
/// document lane (`NULL` otherwise), so a role that monitors the vector lane
/// alone keeps working.
pub(crate) fn status() -> Vec<StatusRow> {
    // Checked/probed once, up front — neither can change mid-call, and this
    // avoids re-running the same query once per registered table.
    let outbox_stats_allowed = outbox::can_call_outbox_stats();
    let launcher_present = outbox::launcher_lock_held();
    let mut docs = docs_registry::list_visible();
    let doc_stats_allowed = !docs.is_empty() && docs_outbox::can_call_doc_outbox_stats();

    let mut rows: Vec<StatusRow> = registry::list_for_reader()
        .into_iter()
        .map(|row| {
            let lane = docs
                .iter()
                .position(|doc| doc.relid == row.relid)
                .map(|at| DocLane::read(&docs.remove(at), launcher_present, doc_stats_allowed));
            let (table_name, local_rows) = table_name_and_rows(row.relid);

            let (backlog, oldest_enqueued, outbox_last_error, in_backoff) =
                outbox_stats(row.relid, outbox_stats_allowed);
            // Priority: `parked_reason` (a plain `pinecone.tables` column,
            // visible to any `status()` caller regardless of `_outbox_stats`
            // `EXECUTE` — a parked table's reason is never hidden just
            // because the caller lacks that one grant) > `last_vacuum_error`
            // (`am::ambulkdelete`'s most recent failure, same visibility —
            // lets an operator tell a VACUUM/Pinecone
            // outage apart from an ordinary outbox backoff, which the third
            // fallback already surfaces) > the outbox's own `last_error`,
            // and only then the document lane's error: a table in both
            // registries shows the vector lane's.
            let last_error = row
                .parked_reason
                .clone()
                .or_else(|| row.last_vacuum_error.clone())
                .or(outbox_last_error)
                .or_else(|| lane.as_ref().and_then(|lane| lane.last_error.clone()));
            let queue_state = queue_state(row.queue_parked, backlog, in_backoff, launcher_present);
            let rpo = format_rpo(row.detach_rpo_secs);
            let (partition_column, partitions) = partition_columns(
                row.relid,
                row.partition.as_ref(),
                lane.as_ref().and_then(|lane| lane.partition.as_ref()),
            );
            let failed_partitions =
                failed_partitions(row.relid, row.syncs(), row.partition.as_ref());

            (
                table_name,
                row.mode,
                row.vector_column,
                row.id_column,
                row.metric,
                row.dimension,
                row.index_name,
                row.namespace,
                row.metadata_columns,
                local_rows,
                backlog,
                oldest_enqueued,
                last_error,
                queue_state,
                row.storage,
                row.backup_gate,
                row.detached_at,
                row.detach_backup_id,
                rpo,
                row.read_capacity,
                row.attached,
                lane.as_ref().and_then(|lane| lane.backlog),
                lane.as_ref().and_then(|lane| lane.oldest_enqueued),
                lane.as_ref().and_then(|lane| lane.last_delivered_at),
                lane.as_ref().map(|lane| lane.backfill_done),
                lane.as_ref().and_then(|lane| lane.backfill_total),
                lane.as_ref().and_then(|lane| lane.failed),
                lane.and_then(|lane| lane.queue_state),
                partition_column,
                partitions,
                failed_partitions,
            )
        })
        .collect();
    // Whatever is left is declared for documents only.
    rows.extend(
        docs.iter()
            .map(|doc| docs_only_row(doc, launcher_present, doc_stats_allowed)),
    );
    rows
}

/// A table that only `track_docs` registered. `mode = 'docs'`; the vector
/// fields every other row has are empty placeholders (`''`, `0`) and the
/// vector lane's queue and detach columns are `NULL` — there is no vector lane.
/// `last_error` is the document lane's error.
fn docs_only_row(doc: &DocRegistryRow, launcher_present: bool, stats_allowed: bool) -> StatusRow {
    let (table_name, local_rows) = table_name_and_rows(doc.relid);
    let lane = DocLane::read(doc, launcher_present, stats_allowed);
    let (partition_column, partitions) = partition_columns(doc.relid, None, doc.partition.as_ref());
    (
        table_name,
        "docs".to_string(),
        String::new(),
        doc.id_column.clone(),
        String::new(),
        0,
        Some(doc.index_name.clone()),
        Some(doc.namespace.clone()),
        doc.metadata_columns.clone(),
        local_rows,
        None,
        None,
        lane.last_error,
        None,
        "full".to_string(),
        None,
        None,
        None,
        None,
        None,
        doc.attached,
        lane.backlog,
        lane.oldest_enqueued,
        lane.last_delivered_at,
        Some(lane.backfill_done),
        lane.backfill_total,
        lane.failed,
        lane.queue_state,
        partition_column,
        partitions,
        0,
    )
}

/// The partition column of a table and how many namespaces it has written: the vector lane's if
/// it has one, else the document lane's; `NULL`s for an unpartitioned table.
fn partition_columns(
    relid: pg_sys::Oid,
    vector: Option<&PartitionSpec>,
    docs: Option<&PartitionSpec>,
) -> (Option<String>, Option<i64>) {
    let (spec, lane) = match (vector, docs) {
        (Some(spec), _) => (spec, Lane::Vector),
        (None, Some(spec)) => (spec, Lane::Docs),
        (None, None) => return (None, None),
    };
    (
        Some(spec.column.clone()),
        Some(partition_lane::count_registered(relid, lane)),
    )
}

/// How many partitions of the vector lane are failed: `0` without a partition column, for a table
/// that is not synced (a bare `demote` keeps the state) and for the document lane, which
/// dead-letters per document.
fn failed_partitions(relid: pg_sys::Oid, syncs: bool, vector: Option<&PartitionSpec>) -> i32 {
    if !syncs || vector.is_none() {
        return 0;
    }
    let failed = partition_registry::count_failed(relid, Lane::Vector);
    i32::try_from(failed).unwrap_or(i32::MAX)
}

/// `schema.table` and the table's `count(*)` (`NULL` without `SELECT`), or
/// `<dropped:OID>` and `NULL` when `relid` no longer resolves.
fn table_name_and_rows(relid: pg_sys::Oid) -> (String, Option<i64>) {
    match catalog::resolve_table_name(relid) {
        Some((schema, table)) => {
            let local_rows =
                catalog::has_select_privilege(relid).then(|| local_row_count(&schema, &table));
            (format!("{schema}.{table}"), local_rows)
        }
        None => (format!("<dropped:{relid}>"), None),
    }
}

/// The document lane of one tracked table, as the `docs_*` columns show it.
/// Registry columns (progress, last delivery, the park flag) are always
/// there; the queue numbers come from `pinecone._doc_outbox_stats` and are
/// `None` for a caller without `EXECUTE` on it, like `_outbox_stats` for the
/// vector lane.
struct DocLane {
    backlog: Option<i64>,
    oldest_enqueued: Option<TimestampWithTimeZone>,
    last_delivered_at: Option<TimestampWithTimeZone>,
    backfill_done: i64,
    backfill_total: Option<i64>,
    failed: Option<i64>,
    queue_state: Option<String>,
    /// The park reason, else the failure recorded on the oldest queued row.
    last_error: Option<String>,
    partition: Option<PartitionSpec>,
}

impl DocLane {
    fn read(doc: &DocRegistryRow, launcher_present: bool, stats_allowed: bool) -> Self {
        let stats = docs_outbox::stats_if(stats_allowed, doc.relid);
        let backlog = stats.as_ref().map(|stats| stats.backlog);
        let in_backoff = stats.as_ref().is_some_and(|stats| stats.in_backoff);
        let queue_error = stats.as_ref().and_then(|stats| stats.last_error.clone());
        Self {
            backlog,
            oldest_enqueued: stats.as_ref().and_then(|stats| stats.oldest_enqueued),
            last_delivered_at: doc.last_delivered_at,
            backfill_done: doc.backfill_done,
            backfill_total: doc.backfill_total,
            failed: stats.as_ref().map(|stats| stats.failed),
            queue_state: queue_state(doc.queue_parked, backlog, in_backoff, launcher_present),
            last_error: doc.parked_reason.clone().or(queue_error),
            partition: doc.partition.clone(),
        }
    }
}

/// Formats `pinecone.tables.detach_rpo_secs` (the schedule-
/// interval-derived worst case, computed once at detach time from
/// `backup_readiness::detach_gate`'s own HTTP round) into the verbatim RPO
/// qualifier — `NULL` for a `'full'` row, or a `'detached'` one whose
/// schedule interval couldn't be determined at detach time (an unrecognized
/// schedule shape, or none configured at all — `pinecone.backup_readiness(tbl)`
/// still gives the live detail). Deliberately makes NO network call of its
/// own — `status()`'s contract ("Registry + local row counts only — no
/// HTTP"); the number itself was already fetched once, durably, at detach
/// time.
fn format_rpo(rpo_secs: Option<i32>) -> Option<String> {
    rpo_secs.map(|secs| {
        let secs = secs.max(0) as i64;
        let (h, m) = (secs / 3600, (secs % 3600) / 60);
        format!(
            "worst-case RPO ~= {h}h{m:02}m (schedule interval + 15 min, recorded at detach \
             time), never zero"
        )
    })
}

/// `pinecone.tables.queue_state`'s state machine:
/// `'parked'` (the registry flag — a plain, always-visible `pinecone.tables`
/// column, so it's shown regardless of `_outbox_stats` access) beats
/// everything else; otherwise `NULL` if the caller can't see `backlog` at
/// all (no `EXECUTE` on `_outbox_stats`); else `'idle'` (no queued rows),
/// `'backoff'` (queued rows, and the head row's `next_retry_at` is still in
/// the future — `outbox::record_transient_failure`'s rustdoc explains why
/// this is an approximation, not the worker's own authoritative schedule),
/// `'NO WORKER'` (queued rows, no backoff in progress, and no live-and-enabled
/// launcher in this cluster — `launcher_present`, probed via
/// `outbox::launcher_lock_held`: plan item 10's "preload absent or launcher
/// lock free while backlog > 0". Deliberately NOT the per-database worker's
/// own lock: that worker idle-exits between backlogs by design and is
/// respawned on demand, so its lock being free is normal, not an outage),
/// or `'active'` (queued rows, no backoff in progress, launcher present —
/// even if the per-database worker itself is momentarily idle-exited, the
/// live launcher will respawn one for this backlog).
fn queue_state(
    parked: bool,
    backlog: Option<i64>,
    in_backoff: bool,
    launcher_present: bool,
) -> Option<String> {
    if parked {
        return Some("parked".to_string());
    }
    let backlog = backlog?;
    if backlog == 0 {
        return Some("idle".to_string());
    }
    if in_backoff {
        return Some("backoff".to_string());
    }
    Some(if launcher_present {
        "active".to_string()
    } else {
        "NO WORKER".to_string()
    })
}

fn local_row_count(schema: &str, table: &str) -> i64 {
    let qualified = quote_qualified_identifier(schema, table);
    Spi::get_one::<i64>(&format!("SELECT count(*) FROM {qualified}"))
        .expect("pg_pinecone: status row-count query failed")
        .expect("pg_pinecone: count(*) unexpectedly NULL")
}

/// `pinecone.outbox`'s backlog count / oldest `enqueued_at` / head-of-line
/// `last_error` / whether the head row is still inside its recorded
/// `next_retry_at` window, for `relid`, via the `SECURITY DEFINER`
/// `pinecone._outbox_stats` — or `(None, None, None, false)` if `allowed` is
/// false (the caller lacks EXECUTE on it, `REVOKE`d from PUBLIC; see
/// `outbox.rs`). `pinecone.status` itself is `SECURITY INVOKER`, so without
/// this most callers have no direct grant on `pinecone.outbox` itself.
/// `Spi::connect`/`get_by_name` rather than `Spi::get_three_with_args`:
/// pgrx 0.19's `Spi` only goes up to 3 typed columns, and `_outbox_stats`
/// now returns 4.
fn outbox_stats(
    relid: pg_sys::Oid,
    allowed: bool,
) -> (
    Option<i64>,
    Option<TimestampWithTimeZone>,
    Option<String>,
    bool,
) {
    if !allowed {
        return (None, None, None, false);
    }
    Spi::connect(|client| {
        let mut rows = client
            .select(
                "SELECT * FROM pinecone._outbox_stats($1::oid)",
                None,
                &[relid.into()],
            )
            .expect("pg_pinecone: status outbox stats query failed");
        let row = rows
            .next()
            .expect("pg_pinecone: _outbox_stats unexpectedly returned no rows");
        let backlog = row
            .get_by_name::<i64, _>("backlog")
            .expect("pg_pinecone: status outbox stats backlog query failed")
            .unwrap_or(0);
        let oldest_enqueued = row
            .get_by_name::<TimestampWithTimeZone, _>("oldest_enqueued")
            .expect("pg_pinecone: status outbox stats oldest_enqueued query failed");
        let last_error = row
            .get_by_name::<String, _>("last_error")
            .expect("pg_pinecone: status outbox stats last_error query failed");
        let in_backoff = row
            .get_by_name::<bool, _>("in_backoff")
            .expect("pg_pinecone: status outbox stats in_backoff query failed")
            .unwrap_or(false);
        (Some(backlog), oldest_enqueued, last_error, in_backoff)
    })
}
