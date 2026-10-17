//! The document backfill: a resumable job that sends the rows a table already holds, queued by
//! `track_docs` and re-queued by `pinecone.backfill_docs` (the repair after a restore). It is
//! one `doc_backfill` row in `pinecone.doc_outbox` whose payload carries the cursor, the id of
//! the last row sent (`{"after_id": null}` = from the start). Every chunk, at most 1,000 rows
//! and about 1.5 MB, commits the cursor with its progress in `pinecone.doc_tables`, so a
//! restart loses at most the chunk in flight and Pinecone upserts it again by id. With a
//! partition column the rows are read in `(partition, id)` order, so a chunk is mostly one
//! namespace, and the cursor adds the partition's canonical text (`"after_p"`).
//!
//! Live rows are drained before a round's chunks but never hold a job back (a hot table would
//! starve its backfill): every step runs at least one chunk. Only a queued wipe
//! (`backfill_docs(.., clear => true)`, a TRUNCATE) blocks them, as it must reach Pinecone
//! first. The worker runs chunks back to back, one transaction each, until
//! `pinecone.backfill_round_ms` has elapsed, serving tables round-robin; `pinecone._docs_round`
//! runs the same loop inside the caller's transaction.
//! Documents come from the same row-to-document code as the live path (`docs_drain.rs`), and
//! what Pinecone refuses is isolated, dead-lettered and circuit-broken the same way.

use std::{
    collections::HashSet,
    panic::AssertUnwindSafe,
    sync::Arc,
    time::{Duration, Instant},
};

use pg_pinecone_client::{Client, Error as ClientError};
use pgrx::{
    JsonB, PgRelation, PgSqlErrorCode as Code,
    bgworkers::BackgroundWorker,
    guc::{GucContext, GucFlags, GucRegistry, GucSetting},
    pg_sys,
    prelude::*,
    spi::{Spi, quote_qualified_identifier},
};
use serde_json::{Value, json};

use crate::{
    backend::{self, DrainCredentials},
    catalog,
    docs_drain::{self, Cursor, Lane, Sender, Stop, Unreadable},
    docs_guard,
    docs_outbox::{self, DocDrainOutcome, DocOp, DocScheduler},
    docs_registry::{self, DocState},
    docs_track, outbox, secret,
};

/// `pinecone.backfill_round_ms`: how long one worker round spends running backfill chunks back
/// to back (document chunks, and the chunks of a vector table's backfill, trigger lane or access
/// method), how long a partitioned table's recovery keeps walking namespaces in one round, and
/// how long it spends rebuilding retried partitions. The next round starts at once while chunks
/// are left. `Userset`: a session can tune the budget `pinecone._docs_round` uses, while the
/// worker reads the cluster's configured value.
pub(crate) static BACKFILL_ROUND_MS: GucSetting<i32> = GucSetting::<i32>::new(1000);

pub(crate) fn init_gucs() {
    GucRegistry::define_int_guc(
        c"pinecone.backfill_round_ms",
        c"Milliseconds a worker round spends running backfill chunks (document and vector), \
          walking the namespaces of a partitioned recovery or rebuilding retried partitions, \
          back to back; the next round starts at once while chunks are left. Document and \
          rebuild rounds deliver live changes first.",
        c"",
        &BACKFILL_ROUND_MS,
        50,
        60_000,
        GucContext::Userset,
        GucFlags::default(),
    );
}

pub(crate) fn round_budget() -> Duration {
    Duration::from_millis(BACKFILL_ROUND_MS.get().max(1) as u64)
}

/// The queued job of a table: its queue row and where it stands.
struct Job {
    id: i64,
    after: Option<Cursor>,
}

/// The cursor of a job payload: `{"after_id": "<last id sent>"}` and, with a partition column,
/// `"after_p"`, the canonical text of that row's partition; `null` or absent = the start.
fn cursor_of(payload: Option<&Value>) -> Option<Cursor> {
    let payload = payload?;
    let partition = payload.get("after_p").and_then(Value::as_str);
    Some(Cursor {
        partition: partition.map(str::to_string),
        id: payload.get("after_id")?.as_str()?.to_string(),
    })
}

/// Claims the oldest job row of `relid`, skipping one another session holds.
fn claim_job(relid: pg_sys::Oid) -> Option<Job> {
    Spi::connect_mut(|client| {
        let mut rows = client
            .update(
                "SELECT id, payload FROM pinecone.doc_outbox \
                 WHERE relid::oid = $1::oid AND op = 'doc_backfill' \
                 ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED",
                None,
                &[relid.into()],
            )
            .expect("pg_pinecone: doc backfill job claim failed");
        let row = rows.next()?;
        let payload = row
            .get_by_name::<JsonB, _>("payload")
            .expect("pg_pinecone: doc backfill payload read failed");
        Some(Job {
            id: row
                .get_by_name::<i64, _>("id")
                .expect("pg_pinecone: doc backfill job id read failed")
                .expect("pg_pinecone: doc backfill job id unexpectedly NULL"),
            after: cursor_of(payload.as_ref().map(|json| &json.0)),
        })
    })
}

/// Whether a backfill job is queued for `relid`.
pub(crate) fn job_queued(relid: pg_sys::Oid) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT EXISTS (SELECT 1 FROM pinecone.doc_outbox \
         WHERE relid::oid = $1::oid AND op = 'doc_backfill')",
        &[relid.into()],
    )
    .expect("pg_pinecone: doc backfill job query failed")
    .expect("pg_pinecone: EXISTS unexpectedly NULL")
}

/// The rows of `relid`'s table the caller sees: what `backfill_total` reports, taken when the job
/// is queued (rows added or removed meanwhile make `done` end up a little off it). Counted as the
/// caller, like `status().local_rows`, so a `FORCE ROW LEVEL SECURITY` policy hides rows here too;
/// `None` without SELECT.
pub(crate) fn count_rows(relid: pg_sys::Oid) -> Option<i64> {
    if !catalog::has_select_privilege(relid) {
        return None;
    }
    let (schema, name) = catalog::resolve_table_name(relid)?;
    let sql = format!(
        "SELECT pg_catalog.count(*) FROM {}",
        quote_qualified_identifier(&schema, &name)
    );
    Spi::get_one::<i64>(&sql).expect("pg_pinecone: doc backfill row count failed")
}

/// Queues a fresh backfill job for a tracked table, replacing a queued one, and restarts its
/// progress: state `backfilling`, cursor and `backfill_done` at zero, `backfill_total` as given
/// (see [`count_rows`]). With `clear` the namespace is wiped first (a live row, so it reaches
/// Pinecone before any chunk). The caller holds the lane lock.
pub(crate) fn enqueue_job(relid: pg_sys::Oid, clear: bool, total: Option<i64>) {
    if clear {
        docs_outbox::enqueue(relid, DocOp::DeleteAll, None);
    }
    Spi::run_with_args(
        "DELETE FROM pinecone.doc_outbox WHERE relid::oid = $1::oid AND op = 'doc_backfill'",
        &[relid.into()],
    )
    .expect("pg_pinecone: doc backfill job replace failed");
    docs_outbox::enqueue(
        relid,
        DocOp::Backfill,
        Some(JsonB(json!({"after_id": null}))),
    );
    docs_registry::set_backfill_progress(relid, None, 0, total);
    docs_registry::set_state(relid, DocState::Backfilling);
}

/// Moves the cursor of `job` past a delivered page: the job's payload and the registry's
/// cursor and counters change in one statement, so they cannot disagree after a crash. The
/// job's failed-attempt count starts over. The registry shows the last id only.
fn persist_progress(job: i64, relid: pg_sys::Oid, last: &Cursor, rows: usize, sent: bool) {
    Spi::run_with_args(
        "WITH moved AS ( \
             UPDATE pinecone.doc_outbox \
             SET payload = jsonb_set(coalesce(payload, '{}'::jsonb), '{after_id}', to_jsonb($2::text)) \
                           || CASE WHEN $6::text IS NULL THEN '{}'::jsonb \
                                   ELSE jsonb_build_object('after_p', $6::text) END, \
                 attempts = 0, last_error = NULL \
             WHERE id = $1) \
         UPDATE pinecone.doc_tables \
         SET backfill_cursor = $2, backfill_done = backfill_done + $4, lane_failures = 0, \
             last_delivered_at = CASE WHEN $5 THEN pg_catalog.clock_timestamp() ELSE last_delivered_at END \
         WHERE relid::oid = $3::oid",
        &[
            job.into(),
            last.id.clone().into(),
            relid.into(),
            (rows as i64).into(),
            sent.into(),
            last.partition.clone().into(),
        ],
    )
    .expect("pg_pinecone: doc backfill progress write failed");
}

/// What one chunk did.
#[derive(Debug)]
pub(crate) enum ChunkOutcome {
    /// Rows were delivered or the job finished; `more` says whether the job is still queued.
    Progress { more: bool },
    /// Nothing to do for this table: no job, a parked lane, or a table that is gone.
    Idle,
    /// DDL holds the table, or another session the lane (`backfill_docs`, `untrack_docs`): it
    /// sits out the rest of the step.
    LockUnavailable,
    /// The chunk failed: a transient failure to retry after a backoff, or the lane parked.
    Failed(DocDrainOutcome),
}

/// Runs one chunk for `relid` in the current transaction: claims the job, reads the next page
/// after its cursor, sends it and commits cursor and progress with it. An empty page ends the
/// job (the table is `ready`, `done` and `total` agree). A failure leaves the cursor where it
/// was and is recorded on the job row like any lane failure, so a permanent one parks the lane
/// after `pinecone.outbox_max_attempts` and a transient one only backs off. `client` is built
/// on first use and kept by the caller, so the chunks of one step share its connection and its
/// resolved index host.
pub(crate) fn chunk(relid: pg_sys::Oid, client: &mut Option<Arc<Client>>) -> ChunkOutcome {
    if !docs_drain::try_lock_lane(relid) {
        return ChunkOutcome::LockUnavailable;
    }
    let Some(job) = claim_job(relid) else {
        return ChunkOutcome::Idle;
    };
    // A queued wipe (`backfill_docs(.., clear => true)`, a TRUNCATE) must reach Pinecone first.
    if docs_outbox::wipe_queued(relid) {
        return ChunkOutcome::Idle;
    }
    let Some(registry) = docs_registry::lookup(relid) else {
        docs_drain::purge(relid, "is no longer tracked in pinecone.doc_tables");
        return ChunkOutcome::Idle;
    };
    if registry.queue_parked {
        return ChunkOutcome::Idle;
    }
    if catalog::resolve_table_name(relid).is_none() {
        docs_registry::delete(relid);
        docs_drain::purge(relid, "no longer resolves to a table");
        return ChunkOutcome::Idle;
    }
    // Checked before building a client: stored_credentials() raises, which would abort the round.
    if !secret::has_stored_api_key() {
        docs_outbox::record_error(job.id, docs_drain::NO_STORED_KEY_ERROR);
        return ChunkOutcome::Failed(DocDrainOutcome::Backoff(ClientError::MissingApiKey));
    }
    let layout = match docs_drain::build_layout(relid, &registry) {
        Ok(layout) => layout,
        Err(error) => {
            return ChunkOutcome::Failed(docs_drain::record_failure(relid, job.id, error));
        }
    };
    // The table is locked for a read only, never across the request that follows.
    let mut unreadable = Unreadable::default();
    let page = match docs_drain::read_page(relid, &layout, job.after.as_ref(), &mut unreadable) {
        Ok(page) => page,
        Err(Stop::Busy | Stop::Budget) => return ChunkOutcome::LockUnavailable,
        Err(Stop::Failed(error)) => {
            return ChunkOutcome::Failed(docs_drain::record_failure(relid, job.id, error));
        }
        Err(Stop::Sql(caught)) => {
            return ChunkOutcome::Failed(docs_drain::record_sql_failure(relid, caught, true));
        }
    };
    let Some(last) = page.last else {
        docs_outbox::delete_rows(&[job.id]);
        let done = registry.backfill_done;
        docs_registry::set_backfill_progress(relid, None, done, Some(done));
        docs_registry::set_state(relid, DocState::Ready);
        docs_registry::reset_lane_failures(relid);
        return ChunkOutcome::Progress { more: false };
    };

    for (id, message) in &page.rejected {
        docs_outbox::enqueue_failed(relid, id, message);
    }
    let client = client
        .get_or_insert_with(|| backend::client_for_drain(DrainCredentials::Worker))
        .clone();
    let lane = Lane::new(client, &registry);
    let mut sender = Sender::new(&lane, relid);
    sender.adopt(unreadable, page.unreadable);
    let result = sender.send(page.docs).and_then(|()| sender.verdict());
    let sent = sender.sent();
    sender.settle();
    match result {
        Ok(()) => {
            persist_progress(job.id, relid, &last, page.rows, sent);
            ChunkOutcome::Progress { more: true }
        }
        Err(error) => {
            if sent {
                docs_registry::touch_delivered(relid);
            }
            ChunkOutcome::Failed(docs_drain::record_failure(relid, job.id, error))
        }
    }
}

/// [`chunk`] for the worker: an ERROR raised in it (the registry or queue SQL; the page's own
/// reads are handled where they run) is the job's failure, recorded like any other, not an abort
/// of the chunk's transaction.
pub(crate) fn chunk_isolated(relid: pg_sys::Oid, client: &mut Option<Arc<Client>>) -> ChunkOutcome {
    match docs_guard::isolated(AssertUnwindSafe(|| chunk(relid, client))) {
        Ok(outcome) => outcome,
        Err(caught) => ChunkOutcome::Failed(docs_drain::record_sql_failure(relid, caught, true)),
    }
}

/// What a budgeted run of chunks did.
#[derive(Debug, Default)]
pub(crate) struct StepReport {
    /// Chunks that delivered rows or finished a job.
    pub(crate) progressed: u64,
    /// The last failed chunk's outcome, if any.
    pub(crate) failure: Option<DocDrainOutcome>,
    /// The step stopped on its budget with a table that has chunks left or has not had its turn.
    pub(crate) chunks_left: bool,
}

/// What the worker's backfill step did.
pub(crate) struct WorkerStep {
    pub(crate) terminated: bool,
    pub(crate) chunks_left: bool,
}

/// Runs chunks round-robin over `relids` until `budget` has elapsed on `elapsed` (or `stop`
/// says so), which is checked after every chunk: each call runs at least one. A table drops out
/// of the rotation once its job is done or it has nothing to do, was locked, or failed. `chunk`
/// is told whether it is the table's first chunk of the step; later ones may decline (`Idle`)
/// when live rows arrived meanwhile.
pub(crate) fn run_chunks(
    budget: Duration,
    mut elapsed: impl FnMut() -> Duration,
    relids: Vec<pg_sys::Oid>,
    mut chunk: impl FnMut(pg_sys::Oid, bool) -> ChunkOutcome,
    mut stop: impl FnMut() -> bool,
) -> StepReport {
    let mut report = StepReport::default();
    let mut served = HashSet::new();
    let mut turn = relids;
    while !turn.is_empty() {
        let mut next = Vec::new();
        for (i, relid) in turn.iter().enumerate() {
            match chunk(*relid, served.insert(*relid)) {
                ChunkOutcome::Progress { more } => {
                    report.progressed += 1;
                    if more {
                        next.push(*relid);
                    }
                }
                ChunkOutcome::Idle | ChunkOutcome::LockUnavailable => {}
                ChunkOutcome::Failed(failure) => report.failure = Some(failure),
            }
            if stop() || elapsed() >= budget {
                report.chunks_left = !next.is_empty() || i + 1 < turn.len();
                return report;
            }
        }
        turn = next;
    }
    report
}

/// What `pinecone._docs_round` runs after the live rows: [`run_chunks`] for one table inside the
/// caller's transaction.
pub(crate) fn run_in_caller_transaction(relid: pg_sys::Oid) -> StepReport {
    let started = Instant::now();
    let mut client = None;
    run_chunks(
        round_budget(),
        || started.elapsed(),
        vec![relid],
        |relid, _| chunk(relid, &mut client),
        || false,
    )
}

/// Tables with a queued job, an unparked lane and no queued wipe, in `relid` order. Queued live
/// rows do not hold a job back (a table written to all the time would starve its backfill): the
/// worker delivers them first and runs at least one chunk after. A table whose registry row is
/// gone is included: its chunk purges the orphaned queue.
pub(crate) fn candidates() -> Vec<pg_sys::Oid> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT DISTINCT j.relid::oid AS relid FROM pinecone.doc_outbox j \
                 LEFT JOIN pinecone.doc_tables t ON t.relid = j.relid \
                 WHERE j.op = 'doc_backfill' AND (t.relid IS NULL OR NOT t.queue_parked) \
                   AND NOT EXISTS (SELECT 1 FROM pinecone.doc_outbox l WHERE l.relid = j.relid \
                                   AND l.op = 'doc_delete_all') \
                 ORDER BY 1",
                None,
                &[],
            )
            .expect("pg_pinecone: doc backfill candidate scan failed")
            .map(|row| {
                row.get_by_name::<pg_sys::Oid, _>("relid")
                    .expect("pg_pinecone: doc backfill candidate read failed")
                    .expect("pg_pinecone: doc backfill candidate unexpectedly NULL")
            })
            .collect()
    })
}

/// The worker's backfill step, after the live drain of its round: chunks back to back, one
/// transaction each (so the table lock is held only inside a chunk and every cursor commits at
/// once), until `pinecone.backfill_round_ms` has elapsed. Tables inside a backoff window sit
/// out; outcomes feed the same scheduler as the live drain. `terminated`: the worker was asked to
/// stop meanwhile.
pub(crate) fn worker_step(scheduler: &mut DocScheduler) -> WorkerStep {
    let started = Instant::now();
    let mut relids: Vec<pg_sys::Oid> = BackgroundWorker::transaction(AssertUnwindSafe(candidates))
        .into_iter()
        .filter(|relid| scheduler.is_eligible(*relid, started))
        .collect();
    scheduler.rotate(&mut relids);
    let (mut client, mut terminated) = (None, false);
    let report = run_chunks(
        round_budget(),
        || started.elapsed(),
        relids,
        |relid, first| {
            let outcome = BackgroundWorker::transaction(AssertUnwindSafe(|| {
                // Live rows that arrived since the last chunk go before the next one.
                if !first && docs_outbox::live_backlog(relid) > 0 {
                    return ChunkOutcome::Idle;
                }
                chunk_isolated(relid, &mut client)
            }));
            let worked = DocDrainOutcome::Processed(0);
            let seen = match &outcome {
                ChunkOutcome::Failed(failure) => failure,
                _ => &worked,
            };
            scheduler.record_outcome(relid, seen, Instant::now());
            outcome
        },
        || {
            terminated |= BackgroundWorker::sigterm_received();
            terminated
        },
    );
    WorkerStep {
        terminated,
        chunks_left: report.chunks_left,
    }
}

/// Implements `pinecone.backfill_docs`: queues a fresh backfill of every row of `tbl`,
/// replacing one in flight and restarting its progress; with `clear` the namespace is wiped
/// first, which also removes documents the table no longer has (the repair after a restore).
/// Owner only; refused while the lane is parked, since nothing would deliver it.
pub(crate) fn backfill_docs(tbl: PgRelation, clear: bool) {
    const FUNCTION: &str = "pinecone.backfill_docs";
    let relid = tbl.oid();
    let display = catalog::display_name(&tbl);
    docs_track::require_owner(relid, &display, FUNCTION);
    docs_drain::lock_lane_for_verb(relid, &display);
    docs_registry::as_owner(|| {
        docs_outbox::require_tracked(relid, &display, FUNCTION);
        let row =
            docs_registry::lookup(relid).expect("pg_pinecone: tracked table has a registry row");
        if row.queue_parked {
            let reason = row.parked_reason.as_deref().unwrap_or("no reason recorded");
            backend::error_with_code(
                Code::ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE,
                format!("{FUNCTION}: the queue for {display} is parked: {reason}"),
                None,
                Some(&format!(
                    "fix the cause, then pinecone.unpark_docs({})",
                    pgrx::spi::quote_literal(&display)
                )),
            );
        }
    });
    let total = count_rows(relid);
    docs_registry::as_owner(|| enqueue_job(relid, clear, total));
    if !outbox::launcher_lock_held() {
        warning!(
            "{FUNCTION}: no pg_pinecone worker launcher is running in this cluster (pg_pinecone is not \
             in shared_preload_libraries, or pinecone.worker_enabled is off) — the backfill waits \
             until the worker runs; see pinecone.status()"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    fn relid(n: u32) -> pg_sys::Oid {
        pg_sys::Oid::from(n)
    }

    fn at(partition: Option<&str>, id: &str) -> Cursor {
        Cursor {
            partition: partition.map(str::to_string),
            id: id.to_string(),
        }
    }

    #[test]
    fn the_cursor_is_the_after_id_string_and_anything_else_is_the_start() {
        assert_eq!(
            cursor_of(Some(&json!({"after_id": "42"}))),
            Some(at(None, "42"))
        );
        assert_eq!(cursor_of(Some(&json!({"after_id": null}))), None);
        assert_eq!(cursor_of(Some(&json!({}))), None);
        assert_eq!(cursor_of(Some(&json!({"after_id": 42}))), None);
        assert_eq!(cursor_of(Some(&json!([]))), None);
        assert_eq!(cursor_of(None), None);
    }

    #[test]
    fn a_partitioned_cursor_adds_the_partition_text_and_needs_the_id() {
        assert_eq!(
            cursor_of(Some(&json!({"after_id": "7", "after_p": "Acme Corp"}))),
            Some(at(Some("Acme Corp"), "7"))
        );
        assert_eq!(
            cursor_of(Some(&json!({"after_id": "7", "after_p": null}))),
            Some(at(None, "7"))
        );
        assert_eq!(cursor_of(Some(&json!({"after_p": "2"}))), None);
    }

    /// A clock the chunks advance: each chunk takes `cost` and the run reads the time after it.
    struct Fake {
        now: Cell<Duration>,
        cost: Duration,
        calls: Cell<usize>,
    }

    impl Fake {
        fn new(cost_ms: u64) -> Self {
            Self {
                now: Cell::new(Duration::ZERO),
                cost: Duration::from_millis(cost_ms),
                calls: Cell::new(0),
            }
        }

        fn run(
            &self,
            budget_ms: u64,
            relids: &[u32],
            mut chunk: impl FnMut(u32, bool) -> ChunkOutcome,
        ) -> StepReport {
            run_chunks(
                Duration::from_millis(budget_ms),
                || self.now.get(),
                relids.iter().map(|n| relid(*n)).collect(),
                |oid, first| {
                    self.calls.set(self.calls.get() + 1);
                    self.now.set(self.now.get() + self.cost);
                    chunk(oid.to_u32(), first)
                },
                || false,
            )
        }
    }

    fn more() -> ChunkOutcome {
        ChunkOutcome::Progress { more: true }
    }

    #[test]
    fn a_budget_smaller_than_one_chunk_still_runs_exactly_one() {
        let fake = Fake::new(250);
        let report = fake.run(50, &[7], |_, _| more());
        assert_eq!(fake.calls.get(), 1);
        assert_eq!(report.progressed, 1);
    }

    #[test]
    fn a_long_budget_runs_chunks_until_the_job_ends() {
        let fake = Fake::new(250);
        let mut left = 10;
        let report = fake.run(60_000, &[7], |_, _| {
            left -= 1;
            ChunkOutcome::Progress { more: left > 0 }
        });
        assert_eq!(fake.calls.get(), 10, "every chunk, one round");
        assert_eq!(report.progressed, 10);
        assert!(report.failure.is_none());
    }

    #[test]
    fn the_budget_is_checked_after_each_chunk() {
        let fake = Fake::new(250);
        fake.run(1_000, &[7], |_, _| more());
        assert_eq!(fake.calls.get(), 4, "250 ms x 4 reaches the second");
        let fake = Fake::new(250);
        fake.run(1_001, &[7], |_, _| more());
        assert_eq!(fake.calls.get(), 5);
    }

    #[test]
    fn tables_take_turns_chunk_by_chunk() {
        let fake = Fake::new(100);
        let mut order = Vec::new();
        fake.run(600, &[1, 2, 3], |oid, _| {
            order.push(oid);
            more()
        });
        assert_eq!(order, [1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn a_table_leaves_the_rotation_when_it_is_done_locked_idle_or_failed() {
        let fake = Fake::new(10);
        let mut order = Vec::new();
        let report = fake.run(60_000, &[1, 2, 3, 4], |oid, _| {
            order.push(oid);
            match oid {
                1 => ChunkOutcome::Progress { more: false },
                2 => ChunkOutcome::LockUnavailable,
                3 => ChunkOutcome::Failed(DocDrainOutcome::Backoff(ClientError::Timeout { ms: 1 })),
                _ if order.iter().filter(|o| **o == 4).count() < 3 => more(),
                _ => ChunkOutcome::Idle,
            }
        });
        assert_eq!(order, [1, 2, 3, 4, 4, 4]);
        assert_eq!(report.progressed, 3, "table 1 finished, table 4 twice");
        assert!(matches!(report.failure, Some(DocDrainOutcome::Backoff(_))));
    }

    #[test]
    fn a_table_is_told_once_per_step_that_a_chunk_is_its_first() {
        let fake = Fake::new(10);
        let mut firsts = Vec::new();
        fake.run(60, &[1, 2], |oid, first| {
            firsts.push((oid, first));
            more()
        });
        assert_eq!(
            firsts,
            [
                (1, true),
                (2, true),
                (1, false),
                (2, false),
                (1, false),
                (2, false)
            ]
        );
    }

    #[test]
    fn a_stop_request_ends_the_step_after_the_current_chunk() {
        let calls = Cell::new(0);
        let report = run_chunks(
            Duration::from_secs(60),
            || Duration::ZERO,
            vec![relid(1)],
            |_, _| {
                calls.set(calls.get() + 1);
                more()
            },
            || calls.get() >= 2,
        );
        assert_eq!(calls.get(), 2);
        assert_eq!(report.progressed, 2);
    }

    #[test]
    fn a_step_that_spends_its_budget_on_a_job_with_more_says_chunks_are_left() {
        let fake = Fake::new(250);
        let report = fake.run(1_000, &[7], |_, _| more());
        assert!(report.chunks_left);

        let fake = Fake::new(250);
        let mut left = 3;
        let report = fake.run(60_000, &[7], |_, _| {
            left -= 1;
            ChunkOutcome::Progress { more: left > 0 }
        });
        assert!(!report.chunks_left, "the job ended inside the budget");
    }

    #[test]
    fn a_table_that_has_not_had_its_turn_yet_counts_as_chunks_left() {
        let fake = Fake::new(100);
        let report = fake.run(100, &[1, 2], |oid, _| ChunkOutcome::Progress {
            more: oid != 1,
        });
        assert!(report.chunks_left, "table 2 is still waiting");

        let fake = Fake::new(100);
        let report = fake.run(100, &[1, 2], |_, _| ChunkOutcome::Progress { more: false });
        assert!(
            report.chunks_left,
            "table 2 is still waiting, table 1 is done"
        );

        let fake = Fake::new(100);
        let report = fake.run(100, &[1], |_, _| ChunkOutcome::Progress { more: false });
        assert!(!report.chunks_left, "the only table is done");
    }

    #[test]
    fn blocked_tables_are_no_chunks_left() {
        let fake = Fake::new(100);
        let report = fake.run(100, &[1], |_, _| ChunkOutcome::LockUnavailable);
        assert!(!report.chunks_left, "locked");
        let fake = Fake::new(100);
        let failed =
            || ChunkOutcome::Failed(DocDrainOutcome::Backoff(ClientError::Timeout { ms: 1 }));
        let report = fake.run(100, &[1], |_, _| failed());
        assert!(!report.chunks_left, "failed");
        let fake = Fake::new(100);
        let report = fake.run(100, &[1], |_, _| ChunkOutcome::Idle);
        assert!(!report.chunks_left, "idle");
    }

    #[test]
    fn no_tables_means_no_chunks() {
        let fake = Fake::new(10);
        let report = fake.run(1_000, &[], |_, _| more());
        assert_eq!(fake.calls.get(), 0);
        assert_eq!(report.progressed, 0);
    }
}
