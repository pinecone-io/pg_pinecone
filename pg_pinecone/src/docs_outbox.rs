//! `pinecone.doc_outbox`: the document lane's queue, kept apart from `pinecone.outbox` so the
//! vector lane stays untouched. Not dumped with the extension (a queue is not data; a restored
//! database re-sends from the heap), and `REVOKE ALL` leaves it to the SECURITY DEFINER code.

use std::{
    collections::HashMap,
    panic::AssertUnwindSafe,
    time::{Duration, Instant},
};

use pg_pinecone_client::Error as ClientError;
use pgrx::{JsonB, bgworkers::BackgroundWorker, pg_sys, prelude::*, spi::Spi};

use crate::{backend, catalog, docs_registry};

extension_sql!(
    r#"
    CREATE TABLE pinecone.doc_outbox (
        id bigserial PRIMARY KEY,
        -- regclass like pinecone.outbox; queries compare relid::oid.
        relid regclass NOT NULL,
        -- doc_upsert/doc_delete carry primary-key ids, doc_failed is a dead letter.
        op text NOT NULL CHECK (op IN (
            'doc_upsert', 'doc_delete', 'doc_delete_all', 'doc_backfill', 'doc_failed')),
        payload jsonb,
        attempts int NOT NULL DEFAULT 0,
        last_error text,
        enqueued_at timestamptz NOT NULL DEFAULT now()
    );
    CREATE INDEX pg_pinecone_doc_outbox_relid_id_idx ON pinecone.doc_outbox (relid, id);
    -- one dead letter per document: a new one replaces the old
    CREATE INDEX pg_pinecone_doc_outbox_failed_idx
        ON pinecone.doc_outbox (relid, (payload ->> 'id')) WHERE op = 'doc_failed';
    REVOKE ALL ON pinecone.doc_outbox FROM PUBLIC;

    -- backlog excludes dead letters (counted as failed); callers are gated by EXECUTE.
    CREATE FUNCTION pinecone._doc_outbox_stats(
        p_relid oid,
        OUT backlog bigint,
        OUT oldest_enqueued timestamptz,
        OUT failed bigint,
        OUT max_attempts int,
        OUT queued_ids bigint,
        OUT last_error text,
        OUT in_backoff boolean
    )
    LANGUAGE sql
    SECURITY DEFINER
    STABLE
    SET search_path = pg_catalog, pg_temp
    AS $$
        SELECT
            (SELECT count(*) FROM pinecone.doc_outbox
             WHERE relid::oid = p_relid AND op <> 'doc_failed'),
            (SELECT min(enqueued_at) FROM pinecone.doc_outbox
             WHERE relid::oid = p_relid AND op <> 'doc_failed'),
            (SELECT count(*) FROM pinecone.doc_outbox
             WHERE relid::oid = p_relid AND op = 'doc_failed'),
            (SELECT max(attempts) FROM pinecone.doc_outbox
             WHERE relid::oid = p_relid AND op <> 'doc_failed'),
            -- documents, not rows: a row carries up to 1,000 ids, a control row counts one
            (SELECT coalesce(sum(CASE WHEN jsonb_typeof(payload) = 'array'
                                      THEN jsonb_array_length(payload) ELSE 1 END), 0)
             FROM pinecone.doc_outbox
             WHERE relid::oid = p_relid AND op <> 'doc_failed'),
            -- a failed attempt that no delivery has followed (a row is deleted once delivered)
            (SELECT o.last_error FROM pinecone.doc_outbox o
             WHERE o.relid::oid = p_relid AND o.op <> 'doc_failed' AND o.last_error IS NOT NULL
             ORDER BY o.id LIMIT 1),
            EXISTS (SELECT 1 FROM pinecone.doc_outbox o
                    WHERE o.relid::oid = p_relid AND o.op <> 'doc_failed'
                      AND o.last_error IS NOT NULL)
    $$;
    REVOKE EXECUTE ON FUNCTION pinecone._doc_outbox_stats(oid) FROM PUBLIC;
    "#,
    name = "pg_pinecone_doc_outbox",
    requires = [pinecone],
);

/// Most rows one [`fetch_batch`] claims. Single-row writes enqueue one id per row, so the
/// batch also stops at [`DOC_DRAIN_BATCH_IDS`] ids: 500 one-id rows or 20 full ones.
pub(crate) const DOC_DRAIN_BATCH_ROWS: i64 = 500;
pub(crate) const DOC_DRAIN_BATCH_IDS: i64 = 20_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DocOp {
    Upsert,
    Delete,
    DeleteAll,
    Backfill,
    Failed,
}

impl DocOp {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            DocOp::Upsert => "doc_upsert",
            DocOp::Delete => "doc_delete",
            DocOp::DeleteAll => "doc_delete_all",
            DocOp::Backfill => "doc_backfill",
            DocOp::Failed => "doc_failed",
        }
    }

    pub(crate) fn parse(op: &str) -> Option<Self> {
        [
            DocOp::Upsert,
            DocOp::Delete,
            DocOp::DeleteAll,
            DocOp::Backfill,
            DocOp::Failed,
        ]
        .into_iter()
        .find(|candidate| candidate.as_str() == op)
    }
}

pub(crate) struct DocOutboxRow {
    pub(crate) id: i64,
    pub(crate) op: DocOp,
    pub(crate) payload: Option<JsonB>,
    #[allow(dead_code)] // read by tests and the status columns (a later change)
    pub(crate) attempts: i32,
}

/// How many of the oldest rows (their id counts given, control rows 0) make one batch: at most
/// [`DOC_DRAIN_BATCH_ROWS`], ending with the row that reaches [`DOC_DRAIN_BATCH_IDS`] ids.
pub(crate) fn claim_len(id_counts: &[i64]) -> usize {
    let cap = DOC_DRAIN_BATCH_ROWS as usize;
    let mut ids = 0;
    for (index, count) in id_counts.iter().take(cap).enumerate() {
        ids += count;
        if ids >= DOC_DRAIN_BATCH_IDS {
            return index + 1;
        }
    }
    id_counts.len().min(cap)
}

pub(crate) fn enqueue(relid: pg_sys::Oid, op: DocOp, payload: Option<JsonB>) -> i64 {
    Spi::get_one_with_args::<i64>(
        "INSERT INTO pinecone.doc_outbox (relid, op, payload) VALUES ($1::oid::regclass, $2, $3) RETURNING id",
        &[relid.into(), op.as_str().into(), payload.into()],
    )
    .expect("pg_pinecone: doc outbox enqueue failed")
    .expect("pg_pinecone: doc outbox enqueue RETURNING id unexpectedly NULL")
}

/// Claims the next batch of `relid`'s deliverable rows (no dead letters), oldest first, locked
/// `FOR UPDATE SKIP LOCKED` and bounded by [`claim_len`]. Read-write SPI, which a locking read
/// needs. The locks end with the step's transaction (one per table in the worker), and every verb
/// that updates these rows takes the lane lock first, so it waits for the batch once.
pub(crate) fn fetch_batch(relid: pg_sys::Oid) -> Vec<DocOutboxRow> {
    let mut claimed: Vec<(DocOutboxRow, i64)> = Spi::connect_mut(|client| {
        client
            .update(
                "SELECT id, op, payload, attempts, \
                        CASE WHEN jsonb_typeof(payload) = 'array' \
                             THEN jsonb_array_length(payload) ELSE 0 END AS id_count \
                 FROM pinecone.doc_outbox WHERE relid::oid = $1::oid AND op <> 'doc_failed' \
                 ORDER BY id LIMIT $2 FOR UPDATE SKIP LOCKED",
                None,
                &[relid.into(), DOC_DRAIN_BATCH_ROWS.into()],
            )
            .expect("pg_pinecone: doc outbox fetch failed")
            .map(|row| {
                let op: String = required(&row, "op");
                let entry = DocOutboxRow {
                    id: required(&row, "id"),
                    op: DocOp::parse(&op).unwrap_or_else(|| {
                        panic!("pg_pinecone: doc outbox row has unknown op '{op}'")
                    }),
                    payload: optional(&row, "payload"),
                    attempts: required(&row, "attempts"),
                };
                let id_count: i32 = optional(&row, "id_count").unwrap_or(0);
                (entry, i64::from(id_count))
            })
            .collect()
    });
    let counts: Vec<i64> = claimed.iter().map(|(_, count)| *count).collect();
    claimed.truncate(claim_len(&counts));
    claimed.into_iter().map(|(row, _)| row).collect()
}

pub(crate) fn delete_rows(ids: &[i64]) {
    Spi::run_with_args(
        "DELETE FROM pinecone.doc_outbox WHERE id = ANY($1)",
        &[ids.to_vec().into()],
    )
    .expect("pg_pinecone: doc outbox delete failed");
}

/// Counts one more failed attempt on `ids` and records `error`; returns the highest attempt
/// count among them (0 if `ids` is empty), for the caller's parking decision.
pub(crate) fn bump_attempts(ids: &[i64], error: &str) -> i32 {
    Spi::get_one_with_args::<i32>(
        "WITH bumped AS (UPDATE pinecone.doc_outbox SET attempts = attempts + 1, \
             last_error = $2 WHERE id = ANY($1) RETURNING attempts) \
         SELECT coalesce(max(attempts), 0) FROM bumped",
        &[ids.to_vec().into(), error.into()],
    )
    .expect("pg_pinecone: doc outbox bump_attempts failed")
    .expect("pg_pinecone: doc outbox bump_attempts max unexpectedly NULL")
}

/// Deletes every row of `relid`, dead letters included; returns how many.
pub(crate) fn clear_relid(relid: pg_sys::Oid) -> i64 {
    Spi::get_one_with_args::<i64>(
        "WITH d AS (DELETE FROM pinecone.doc_outbox WHERE relid::oid = $1::oid RETURNING 1) \
         SELECT pg_catalog.count(*) FROM d",
        &[relid.into()],
    )
    .expect("pg_pinecone: doc outbox clear failed")
    .expect("pg_pinecone: doc outbox clear count unexpectedly NULL")
}

/// Relids with deliverable rows whose lane is not parked. LEFT JOIN, so a queue whose registry
/// row vanished is still visited (and can be purged) rather than lingering as an orphan.
pub(crate) fn active_relids() -> Vec<pg_sys::Oid> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT DISTINCT o.relid::oid AS relid FROM pinecone.doc_outbox o \
                 LEFT JOIN pinecone.doc_tables t ON t.relid = o.relid \
                 WHERE o.op <> 'doc_failed' AND (t.relid IS NULL OR NOT t.queue_parked)",
                None,
                &[],
            )
            .expect("pg_pinecone: doc outbox active-relid scan failed")
            .map(|row| required(&row, "relid"))
            .collect()
    })
}

/// Deliverable rows queued for `relid` (rows, not documents: a row carries up to 1,000 ids).
pub(crate) fn backlog(relid: pg_sys::Oid) -> i64 {
    Spi::get_one_with_args::<i64>(
        "SELECT count(*) FROM pinecone.doc_outbox \
         WHERE relid::oid = $1::oid AND op <> 'doc_failed'",
        &[relid.into()],
    )
    .expect("pg_pinecone: doc outbox backlog query failed")
    .expect("pg_pinecone: doc outbox backlog count unexpectedly NULL")
}

/// Rows a round can still deliver: neither dead letters nor the backfill job.
pub(crate) fn live_backlog(relid: pg_sys::Oid) -> i64 {
    Spi::get_one_with_args::<i64>(
        "SELECT count(*) FROM pinecone.doc_outbox \
         WHERE relid::oid = $1::oid AND op NOT IN ('doc_failed', 'doc_backfill')",
        &[relid.into()],
    )
    .expect("pg_pinecone: doc outbox live backlog query failed")
    .expect("pg_pinecone: doc outbox live backlog count unexpectedly NULL")
}

/// Whether a `doc_delete_all` is queued for `relid`: it must reach Pinecone before any backfill
/// chunk, which would otherwise be wiped again.
pub(crate) fn wipe_queued(relid: pg_sys::Oid) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT EXISTS (SELECT 1 FROM pinecone.doc_outbox \
         WHERE relid::oid = $1::oid AND op = 'doc_delete_all')",
        &[relid.into()],
    )
    .expect("pg_pinecone: doc outbox wipe query failed")
    .expect("pg_pinecone: EXISTS unexpectedly NULL")
}

/// Records a transient failure on row `id` without counting an attempt (see [`bump_attempts`]).
pub(crate) fn record_error(id: i64, error: &str) {
    Spi::run_with_args(
        "UPDATE pinecone.doc_outbox SET last_error = $2 WHERE id = $1",
        &[id.into(), error.into()],
    )
    .expect("pg_pinecone: doc outbox record_error failed");
}

/// Forgets the failed attempt recorded on rows `ids`, without counting it as progress of their
/// attempts: a delivery went out since, so the lane no longer backs off.
pub(crate) fn clear_errors(ids: &[i64]) {
    Spi::run_with_args(
        "UPDATE pinecone.doc_outbox SET last_error = NULL \
         WHERE id = ANY($1) AND last_error IS NOT NULL",
        &[ids.to_vec().into()],
    )
    .expect("pg_pinecone: doc outbox clear_errors failed");
}

/// Writes the dead letter of document `id` (Pinecone refused it with `error`), replacing an
/// older one: a document that fails again after a partial retry is still one letter.
pub(crate) fn enqueue_failed(relid: pg_sys::Oid, id: &str, error: &str) {
    Spi::run_with_args(
        "WITH replaced AS ( \
             DELETE FROM pinecone.doc_outbox \
             WHERE relid = $1::oid::regclass AND op = 'doc_failed' AND payload ->> 'id' = $2 \
         ) INSERT INTO pinecone.doc_outbox (relid, op, payload) \
           VALUES ($1::oid::regclass, 'doc_failed', $3)",
        &[
            relid.into(),
            id.into(),
            JsonB(serde_json::json!({"id": id, "error": error})).into(),
        ],
    )
    .expect("pg_pinecone: doc outbox enqueue_failed failed");
}

/// Removes the dead letters of `ids`: a later delivery (upsert or delete) of a document that
/// had been refused means its letter is stale.
pub(crate) fn clear_failed(relid: pg_sys::Oid, ids: &[String]) {
    if ids.is_empty() {
        return;
    }
    Spi::run_with_args(
        "DELETE FROM pinecone.doc_outbox \
         WHERE relid = $1::oid::regclass AND op = 'doc_failed' AND payload ->> 'id' = ANY($2)",
        &[relid.into(), ids.to_vec().into()],
    )
    .expect("pg_pinecone: doc outbox clear_failed failed");
}

/// Removes every dead letter of `relid`: the namespace was emptied, so each one names a document
/// that is gone.
pub(crate) fn clear_all_failed(relid: pg_sys::Oid) {
    Spi::run_with_args(
        "DELETE FROM pinecone.doc_outbox WHERE relid::oid = $1::oid AND op = 'doc_failed'",
        &[relid.into()],
    )
    .expect("pg_pinecone: doc outbox clear_all_failed failed");
}

/// Whether the `doc_delete_all` row `row` has registered the namespaces of the table's values
/// (see [`mark_wipe_adopted`]).
pub(crate) fn wipe_adopted(row: i64) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT coalesce((payload ->> 'adopted')::boolean, false) \
         FROM pinecone.doc_outbox WHERE id = $1",
        &[row.into()],
    )
    .expect("pg_pinecone: doc outbox wipe_adopted failed")
    .unwrap_or(false)
}

/// Notes on a `doc_delete_all` row that the namespaces of the table's values were registered for
/// it: a wipe cut short by its budget resumes with what the registry still lists, and must not
/// register again the namespaces it emptied and forgot.
pub(crate) fn mark_wipe_adopted(row: i64) {
    Spi::run_with_args(
        "UPDATE pinecone.doc_outbox SET payload = '{\"adopted\": true}'::jsonb WHERE id = $1",
        &[row.into()],
    )
    .expect("pg_pinecone: doc outbox mark_wipe_adopted failed");
}

pub(crate) struct FailedDoc {
    pub(crate) id: String,
    pub(crate) error: String,
    pub(crate) failed_at: TimestampWithTimeZone,
}

/// `relid`'s dead letters, oldest first.
pub(crate) fn failed(relid: pg_sys::Oid) -> Vec<FailedDoc> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT payload ->> 'id' AS id, coalesce(payload ->> 'error', '') AS error, \
                        enqueued_at AS failed_at \
                 FROM pinecone.doc_outbox WHERE relid::oid = $1::oid AND op = 'doc_failed' \
                 ORDER BY id",
                None,
                &[relid.into()],
            )
            .expect("pg_pinecone: doc outbox failed-documents query failed")
            .filter_map(|row| {
                Some(FailedDoc {
                    id: optional(&row, "id")?,
                    error: required(&row, "error"),
                    failed_at: required(&row, "failed_at"),
                })
            })
            .collect()
    })
}

/// Ids per `doc_upsert` row, as the trigger chunks them.
const IDS_PER_ROW: i64 = 1_000;

/// Turns every dead letter of `relid` back into `doc_upsert` rows of up to 1,000 ids; returns
/// how many documents were requeued.
pub(crate) fn retry_failed(relid: pg_sys::Oid) -> i64 {
    Spi::get_one_with_args::<i64>(
        &format!(
            "WITH letters AS ( \
                 DELETE FROM pinecone.doc_outbox WHERE relid::oid = $1::oid AND op = 'doc_failed' \
                 RETURNING id AS row_id, payload ->> 'id' AS doc_id \
             ), numbered AS ( \
                 SELECT doc_id, pg_catalog.row_number() OVER (ORDER BY row_id) AS rn FROM letters \
                 WHERE doc_id IS NOT NULL \
             ), queued AS ( \
                 INSERT INTO pinecone.doc_outbox (relid, op, payload) \
                 SELECT $1::oid::regclass, 'doc_upsert', pg_catalog.jsonb_agg(doc_id ORDER BY rn) \
                 FROM numbered GROUP BY (rn - 1) / {IDS_PER_ROW} ORDER BY (rn - 1) / {IDS_PER_ROW} \
                 RETURNING 1 \
             ) SELECT pg_catalog.count(*) FROM numbered"
        ),
        &[relid.into()],
    )
    .expect("pg_pinecone: doc outbox retry_failed failed")
    .expect("pg_pinecone: doc outbox retry_failed count unexpectedly NULL")
}

/// One round over every active relid in the caller's transaction, for the tests (the worker runs
/// [`worker_doc_round`]). Each table's step is isolated, so a Postgres ERROR in one table's rows
/// is that lane's failure and the others carry on.
#[cfg(feature = "pg_test")]
pub(crate) fn drain_doc_round(scheduler: &mut DocScheduler) -> bool {
    scheduler.drain_round(crate::docs_drain::drain_isolated)
}

/// The worker's round: finding the active tables and every table's step each run in a
/// transaction of their own, after the vector drain's. A step's lane lock and the locks on the
/// rows it claimed end with its transaction, so a verb on one table waits for that table's batch,
/// not for the batches of the tables after it in the round.
pub(crate) fn worker_doc_round(scheduler: &mut DocScheduler) -> bool {
    let active = BackgroundWorker::transaction(AssertUnwindSafe(active_relids));
    scheduler.drain_round_over(Instant::now(), active, |relid| {
        BackgroundWorker::transaction(AssertUnwindSafe(|| {
            crate::docs_drain::drain_isolated(relid)
        }))
    })
}

/// The queue row a lane failure is recorded on: the oldest live row, or, with `job_first`, the
/// backfill job; the other kind if there is none of the preferred one.
pub(crate) fn head_row(relid: pg_sys::Oid, job_first: bool) -> Option<i64> {
    Spi::get_one_with_args::<i64>(
        "SELECT (SELECT id FROM pinecone.doc_outbox \
                 WHERE relid::oid = $1::oid AND op <> 'doc_failed' \
                 ORDER BY ((op = 'doc_backfill') = $2) DESC, id LIMIT 1)",
        &[relid.into(), job_first.into()],
    )
    .expect("pg_pinecone: doc outbox head row query failed")
}

/// Fails unless `relid` is tracked by `track_docs`.
pub(crate) fn require_tracked(relid: pg_sys::Oid, display: &str, function: &str) {
    if docs_registry::lookup(relid).is_none() {
        backend::error_with_code(
            PgSqlErrorCode::ERRCODE_UNDEFINED_OBJECT,
            format!("{function}: {display} is not tracked by pinecone.track_docs"),
            None,
            Some("run pinecone.track_docs first"),
        );
    }
}

/// The standard "permission denied" for a caller who may not read the table: what the queue
/// reveals (ids, error texts, whether anything is pending) is about the table's rows.
pub(crate) fn require_select(relid: pg_sys::Oid, relname: &str) {
    if !catalog::has_select_privilege(relid) {
        backend::error_with_code(
            PgSqlErrorCode::ERRCODE_INSUFFICIENT_PRIVILEGE,
            format!("permission denied for table {relname}"),
            None,
            None,
        );
    }
}

/// [`flush`]'s poll interval.
const FLUSH_POLL_INTERVAL: Duration = Duration::from_millis(200);

/// `pinecone.flush_docs`: waits until nothing is queued for `relid`, i.e. every committed change
/// reached Pinecone (searchable up to about a minute later). Returns `false` with a WARNING
/// at once if the lane is parked or no launcher is running (nothing would ever drain it), and
/// after `timeout_ms` otherwise. Never drains anything itself. Needs `SELECT` on the table,
/// nothing on the queue.
pub(crate) fn flush(relid: pg_sys::Oid, display: &str, relname: &str, timeout_ms: i32) -> bool {
    require_select(relid, relname);
    docs_registry::as_owner(|| require_tracked(relid, display, "pinecone.flush_docs"));
    let queued = || docs_registry::as_owner(|| queue_stats(relid));
    let mut now_queued = queued();
    if now_queued.backlog == 0 {
        return true;
    }
    let deadline = Instant::now() + Duration::from_millis(timeout_ms.max(0) as u64);
    loop {
        let parked = docs_registry::as_owner(|| docs_registry::lookup(relid));
        if let Some(row) = parked.filter(|row| row.queue_parked) {
            warning!(
                "pinecone.flush_docs: the document queue of {display} is parked ({}) — nothing \
                 will drain it until it is unparked (see pinecone.status())",
                row.parked_reason.unwrap_or_default()
            );
            return false;
        }
        if !crate::outbox::launcher_lock_held() {
            warning!(
                "pinecone.flush_docs: {display} has {} queued documents but no pg_pinecone launcher \
                 is running in this cluster (preload absent or pinecone.worker_enabled off) — \
                 nothing will drain them (see pinecone.status())",
                now_queued.queued_ids
            );
            return false;
        }
        let now = Instant::now();
        if now >= deadline {
            warning!(
                "pinecone.flush_docs: {display} still has {} queued documents after {}ms — timed \
                 out (see pinecone.status())",
                now_queued.queued_ids,
                timeout_ms.max(0)
            );
            return false;
        }
        backend::interruptible_sleep(FLUSH_POLL_INTERVAL.min(deadline - now));
        now_queued = queued();
        if now_queued.backlog == 0 {
            return true;
        }
    }
}

/// `pinecone.failed_docs`: the documents Pinecone refused, with its message. Needs `SELECT` on
/// the table: the queue holds its ids and error texts.
pub(crate) fn failed_docs(relid: pg_sys::Oid, display: &str, relname: &str) -> Vec<FailedDoc> {
    require_select(relid, relname);
    docs_registry::as_owner(|| {
        require_tracked(relid, display, "pinecone.failed_docs");
        failed(relid)
    })
}

/// `pinecone.retry_failed_docs`: requeues the dead letters; returns how many documents. Owner
/// only, like the other verbs that change what gets sent.
pub(crate) fn retry_failed_docs(relid: pg_sys::Oid, display: &str) -> i64 {
    const FUNCTION: &str = "pinecone.retry_failed_docs";
    crate::docs_track::require_owner(relid, display, FUNCTION);
    docs_registry::as_owner(|| {
        require_tracked(relid, display, FUNCTION);
        retry_failed(relid)
    })
}

/// `pinecone.unpark_docs`: reopens a parked document queue, so the worker delivers it again,
/// and clears the failed attempts, so the retry starts from a clean count. Requires ownership
/// of the table (it re-enables sending with the stored key); harmless on a queue that is not
/// parked.
pub(crate) fn unpark_docs(relid: pg_sys::Oid, display: &str) {
    const FUNCTION: &str = "pinecone.unpark_docs";
    crate::docs_track::require_owner(relid, display, FUNCTION);
    crate::docs_drain::lock_lane_for_verb(relid, display);
    docs_registry::as_owner(|| {
        require_tracked(relid, display, FUNCTION);
        docs_registry::unpark(relid);
        Spi::run_with_args(
            "UPDATE pinecone.doc_outbox SET attempts = 0, last_error = NULL \
             WHERE relid::oid = $1::oid AND op <> 'doc_failed' \
               AND (attempts <> 0 OR last_error IS NOT NULL)",
            &[relid.into()],
        )
        .expect("pg_pinecone: doc outbox attempts reset failed");
    });
}

fn optional<T: FromDatum + IntoDatum>(
    row: &pgrx::spi::SpiHeapTupleData<'_>,
    name: &str,
) -> Option<T> {
    row.get_by_name::<T, _>(name)
        .unwrap_or_else(|e| panic!("pg_pinecone: doc outbox row {name} query failed: {e}"))
}

fn required<T: FromDatum + IntoDatum>(row: &pgrx::spi::SpiHeapTupleData<'_>, name: &str) -> T {
    optional(row, name)
        .unwrap_or_else(|| panic!("pg_pinecone: doc outbox row {name} unexpectedly NULL"))
}

/// Test seam: how often the gate of the queue statistics was probed. Compiled out of the
/// extension itself.
#[cfg(feature = "pg_test")]
pub(crate) mod probes {
    use std::cell::Cell;

    thread_local! {
        static COUNT: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) fn note() {
        COUNT.with(|count| count.set(count.get() + 1));
    }

    pub(crate) fn take() -> usize {
        COUNT.with(|count| count.replace(0))
    }
}

/// Whether `current_user` may execute the SECURITY DEFINER `pinecone._doc_outbox_stats`; the
/// check has to be made by the caller, since inside a definer's function it would see the owner.
pub(crate) fn can_call_doc_outbox_stats() -> bool {
    #[cfg(feature = "pg_test")]
    probes::note();
    Spi::get_one::<bool>(
        "SELECT pg_catalog.has_function_privilege(current_user, \
         'pinecone._doc_outbox_stats(oid)', 'EXECUTE')",
    )
    .expect("pg_pinecone: has_function_privilege query failed")
    .expect("pg_pinecone: has_function_privilege unexpectedly NULL")
}

#[allow(dead_code)] // the status columns (a later change) read the remaining fields
pub(crate) struct DocOutboxStats {
    pub(crate) backlog: i64,
    pub(crate) queued_ids: i64,
    pub(crate) oldest_enqueued: Option<TimestampWithTimeZone>,
    pub(crate) failed: i64,
    pub(crate) max_attempts: Option<i32>,
    /// The error of the oldest queued row that has one.
    pub(crate) last_error: Option<String>,
    /// Some queued row carries a failed attempt the lane has not yet got past.
    pub(crate) in_backoff: bool,
}

/// `relid`'s queue statistics, or `None` for a caller without EXECUTE on the function (never an
/// error, so `status()` can degrade the same way as for the vector queue).
#[allow(dead_code)] // `status()` probes once for all tables and calls `stats_if`
pub(crate) fn stats(relid: pg_sys::Oid) -> Option<DocOutboxStats> {
    stats_if(can_call_doc_outbox_stats(), relid)
}

/// [`stats`] for a caller that probed [`can_call_doc_outbox_stats`] already.
pub(crate) fn stats_if(allowed: bool, relid: pg_sys::Oid) -> Option<DocOutboxStats> {
    allowed.then(|| queue_stats(relid))
}

/// [`stats`] without the EXECUTE gate, for the verbs that authorised their caller already (and
/// run this as the registry's owner, see [`docs_registry::as_owner`]).
pub(crate) fn queue_stats(relid: pg_sys::Oid) -> DocOutboxStats {
    Spi::connect(|client| {
        let row = client
            .select(
                "SELECT * FROM pinecone._doc_outbox_stats($1::oid)",
                None,
                &[relid.into()],
            )
            .expect("pg_pinecone: doc outbox stats query failed")
            .next()
            .expect("pg_pinecone: _doc_outbox_stats unexpectedly returned no rows");
        DocOutboxStats {
            backlog: optional(&row, "backlog").unwrap_or(0),
            queued_ids: optional(&row, "queued_ids").unwrap_or(0),
            oldest_enqueued: optional(&row, "oldest_enqueued"),
            failed: optional(&row, "failed").unwrap_or(0),
            max_attempts: optional(&row, "max_attempts"),
            last_error: optional(&row, "last_error"),
            in_backoff: optional(&row, "in_backoff").unwrap_or(false),
        }
    })
}

#[derive(Debug)]
#[allow(dead_code)] // the error is recorded on the queue row; callers may inspect it
pub(crate) enum DocDrainOutcome {
    Processed(u64),
    /// A transient failure: retry the relid after a growing delay.
    Backoff(ClientError),
    /// Attempts ran out and the lane is now parked in `doc_tables`.
    Parked(ClientError),
}

/// Same backoff as the vector `Scheduler`.
const BACKOFF_MIN: Duration = Duration::from_secs(1);
const BACKOFF_MAX: Duration = Duration::from_secs(60);

struct Backoff {
    until: Instant,
    next_delay: Duration,
}

/// The document lane's round-robin scheduler: one step per eligible relid per round, so one
/// table's failure or backlog never delays another's turn. Parking lives in `doc_tables`.
pub(crate) struct DocScheduler {
    backoff: HashMap<pg_sys::Oid, Backoff>,
    /// Which table the next backfill step serves first, so a slow one cannot starve the rest.
    turn: usize,
}

impl DocScheduler {
    pub(crate) fn new() -> Self {
        Self {
            backoff: HashMap::new(),
            turn: 0,
        }
    }

    /// Whether `relid` is outside its backoff window at `now`.
    pub(crate) fn is_eligible(&self, relid: pg_sys::Oid, now: Instant) -> bool {
        self.backoff.get(&relid).is_none_or(|b| now >= b.until)
    }

    /// Moves the start of `relids` on by one each call (the order they are served in).
    pub(crate) fn rotate(&mut self, relids: &mut [pg_sys::Oid]) {
        if !relids.is_empty() {
            relids.rotate_left(self.turn % relids.len());
            self.turn = self.turn.wrapping_add(1);
        }
    }

    /// Runs `step` once for every active relid outside its backoff window, in the caller's
    /// transaction. Returns whether any unparked relid had rows, backing-off ones included: a
    /// round where everything backs off is busy, not idle, and the worker must not exit under it.
    #[cfg(feature = "pg_test")]
    pub(crate) fn drain_round(&mut self, step: impl FnMut(pg_sys::Oid) -> DocDrainOutcome) -> bool {
        self.drain_round_at(Instant::now(), step)
    }

    /// [`Self::drain_round`] at a given instant, so backoff windows are testable without sleeping.
    #[cfg(feature = "pg_test")]
    pub(crate) fn drain_round_at(
        &mut self,
        now: Instant,
        step: impl FnMut(pg_sys::Oid) -> DocDrainOutcome,
    ) -> bool {
        self.drain_round_over(now, active_relids(), step)
    }

    /// Like [`Self::drain_round`] over relids the caller found (the worker finds them in a
    /// transaction of its own).
    pub(crate) fn drain_round_over(
        &mut self,
        now: Instant,
        active: Vec<pg_sys::Oid>,
        mut step: impl FnMut(pg_sys::Oid) -> DocDrainOutcome,
    ) -> bool {
        let any_active = !active.is_empty();
        for relid in self.eligible(active, now) {
            let outcome = step(relid);
            self.record_outcome(relid, &outcome, now);
        }
        any_active
    }

    fn eligible(&self, active: Vec<pg_sys::Oid>, now: Instant) -> Vec<pg_sys::Oid> {
        active
            .into_iter()
            .filter(|relid| self.is_eligible(*relid, now))
            .collect()
    }

    pub(crate) fn record_outcome(
        &mut self,
        relid: pg_sys::Oid,
        outcome: &DocDrainOutcome,
        now: Instant,
    ) {
        match outcome {
            DocDrainOutcome::Processed(_) | DocDrainOutcome::Parked(_) => {
                self.backoff.remove(&relid);
            }
            DocDrainOutcome::Backoff(_) => {
                let entry = self.backoff.entry(relid).or_insert(Backoff {
                    until: now,
                    next_delay: BACKOFF_MIN,
                });
                let delay = entry.next_delay;
                entry.until = now + delay;
                entry.next_delay = (delay * 2).min(BACKOFF_MAX);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use pg_pinecone_client::Error as ClientError;
    use pgrx::pg_sys;

    use super::*;

    fn relid(n: u32) -> pg_sys::Oid {
        pg_sys::Oid::from(n)
    }

    fn transient() -> DocDrainOutcome {
        DocDrainOutcome::Backoff(ClientError::Timeout { ms: 1 })
    }

    // ---- claim bound ----

    #[test]
    fn claim_takes_thirty_one_id_rows_in_one_batch() {
        assert_eq!(claim_len(&[1; 30]), 30);
    }

    #[test]
    fn claim_stops_after_the_row_that_reaches_twenty_thousand_ids() {
        assert_eq!(claim_len(&[1_000; 25]), 20);
        assert_eq!(claim_len(&[1_000; 20]), 20);
        assert_eq!(claim_len(&[1_000; 19]), 19);
    }

    #[test]
    fn claim_includes_the_row_that_crosses_the_bound() {
        assert_eq!(claim_len(&[15_000, 6_000, 10]), 2);
        assert_eq!(claim_len(&[19_999, 1, 1]), 2);
        assert_eq!(claim_len(&[20_000, 5]), 1);
        assert_eq!(claim_len(&[25_000]), 1);
    }

    #[test]
    fn claim_counts_control_rows_as_zero_ids_and_caps_rows_at_five_hundred() {
        assert_eq!(claim_len(&[]), 0);
        assert_eq!(claim_len(&[0, 0, 0]), 3);
        assert_eq!(claim_len(&[0; 600]), DOC_DRAIN_BATCH_ROWS as usize);
        assert_eq!(claim_len(&[1; 600]), 500);
        assert_eq!(DOC_DRAIN_BATCH_ROWS, 500);
        assert_eq!(DOC_DRAIN_BATCH_IDS, 20_000);
    }

    // ---- op names ----

    #[test]
    fn ops_round_trip_through_their_queue_names() {
        for (op, name) in [
            (DocOp::Upsert, "doc_upsert"),
            (DocOp::Delete, "doc_delete"),
            (DocOp::DeleteAll, "doc_delete_all"),
            (DocOp::Backfill, "doc_backfill"),
            (DocOp::Failed, "doc_failed"),
        ] {
            assert_eq!(op.as_str(), name);
            assert_eq!(DocOp::parse(name), Some(op));
        }
        assert_eq!(DocOp::parse("upsert"), None);
        assert_eq!(DocOp::parse(""), None);
    }

    // ---- scheduler backoff ----

    #[test]
    fn backoff_doubles_from_one_second_to_the_sixty_second_cap() {
        let mut scheduler = DocScheduler::new();
        let now = Instant::now();
        for seconds in [1, 2, 4, 8, 16, 32, 60, 60] {
            scheduler.record_outcome(relid(7), &transient(), now);
            let until = scheduler.backoff[&relid(7)].until;
            assert_eq!(until, now + Duration::from_secs(seconds));
        }
    }

    #[test]
    fn progress_or_parking_resets_the_backoff() {
        for outcome in [
            DocDrainOutcome::Processed(3),
            DocDrainOutcome::Parked(ClientError::Auth),
        ] {
            let mut scheduler = DocScheduler::new();
            let now = Instant::now();
            scheduler.record_outcome(relid(7), &transient(), now);
            scheduler.record_outcome(relid(7), &transient(), now);
            scheduler.record_outcome(relid(7), &outcome, now);
            assert!(!scheduler.backoff.contains_key(&relid(7)), "{outcome:?}");
            scheduler.record_outcome(relid(7), &transient(), now);
            assert_eq!(
                scheduler.backoff[&relid(7)].until,
                now + Duration::from_secs(1)
            );
        }
    }

    #[test]
    fn the_backfill_order_starts_one_table_later_each_step() {
        let mut scheduler = DocScheduler::new();
        let mut starts = Vec::new();
        for _ in 0..4 {
            let mut order = vec![relid(1), relid(2), relid(3)];
            scheduler.rotate(&mut order);
            starts.push(order[0]);
        }
        assert_eq!(starts, [relid(1), relid(2), relid(3), relid(1)]);
        scheduler.rotate(&mut []);
    }

    #[test]
    fn a_relid_inside_its_window_is_skipped_while_others_stay_eligible() {
        let mut scheduler = DocScheduler::new();
        let now = Instant::now();
        scheduler.record_outcome(relid(1), &transient(), now);
        let active = vec![relid(1), relid(2)];
        assert_eq!(scheduler.eligible(active.clone(), now), vec![relid(2)]);
        let later = now + Duration::from_secs(1);
        assert_eq!(scheduler.eligible(active, later), vec![relid(1), relid(2)]);
    }
}
