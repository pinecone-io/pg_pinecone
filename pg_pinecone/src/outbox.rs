//! `pinecone.outbox`: the queue `pinecone._sync_trigger` enqueues every write on
//! a promoted table into, and `pinecone._drain_relid` drains — called inline,
//! from within the trigger itself, for `sync_mode = 'sync'` tables; also
//! called by the background worker for `'async'` ones.
//!
//! Declared here (rather than inside the `pg_pinecone` `#[pg_schema]` module in
//! `lib.rs`) for the same reason as `pinecone.tables` in `registry.rs`: an
//! `extension_sql!` with an explicit `requires` can't live inside an inline
//! `#[pg_schema] mod {}`.
//!
//! Deliberately **not** passed to `pg_extension_config_dump`: it's a queue,
//! not data — a database restored from a dump may lag
//! the remote index; `pinecone.sync(tbl)` repairs that. `REVOKE ALL ... FROM
//! PUBLIC` mirrors `pinecone.secret`/`pinecone.tables`'s ACL model: only code
//! running as the table owner (the `SECURITY DEFINER` sync trigger, and this
//! module's own internals, which it calls into) ever touches it directly.

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use pg_pinecone_client::{Client, Error as ClientError};
use pgrx::{JsonB, PgRelation, prelude::*, spi::Spi};
use serde_json::{Value, json};

use crate::{
    backend::{self, DrainCredentials},
    catalog, docs_backfill,
    partition::{self, Lane},
    partition_lane::{self, DeleteEntry, Gate},
    partition_registry, promote, recover, registry, secret, sync_trigger,
};

extension_sql!(
    r#"
    CREATE TABLE pinecone.outbox (
        id bigserial PRIMARY KEY,
        relid regclass NOT NULL,
        -- `'update'`/`'reupsert'`: the detached-table
        -- metadata-divergence fix's two dispatch legs — a merge-only
        -- `setMetadata` PATCH, and a fetch_vectors + full re-upsert for a
        -- metadata field being unset (merge can't unset a key) — see
        -- `sync_trigger.rs`'s `try_update_metadata`/`try_reupsert_metadata`.
        -- `'rebuild_partition'`: a retry's rebuild of one partition. It deletes the namespace, then
        -- runs as a `'backfill'` scoped to one value that never flips the table's mode when it ends.
        -- The worker runs it after the live rows, chunk by chunk, so it never holds them back.
        op text NOT NULL CHECK (op IN ('upsert', 'delete', 'delete_all', 'backfill', 'rebuild_partition', 'recover', 'update', 'reupsert')),
        payload jsonb,
        enqueued_at timestamptz NOT NULL DEFAULT now(),
        attempts int NOT NULL DEFAULT 0,
        last_error text,
        -- Observability only, set/cleared by `outbox::drain_step` on the
        -- row that failed (backs `status()`'s `'backoff'` state) —
        -- the *authoritative* retry schedule is `Scheduler`'s own in-memory,
        -- per-relid exponential backoff, which never reads this column
        -- back. See `record_transient_failure`'s rustdoc in `outbox.rs`.
        next_retry_at timestamptz
    );
    CREATE INDEX pg_pinecone_outbox_relid_id_idx ON pinecone.outbox (relid, id);
    CREATE INDEX pg_pinecone_outbox_jobs_idx ON pinecone.outbox (relid, id)
        WHERE op IN ('backfill', 'recover', 'rebuild_partition');
    REVOKE ALL ON pinecone.outbox FROM PUBLIC;

    -- Read-only aggregate backing `pinecone.status`'s backlog/oldest_enqueued/
    -- last_error/queue_state columns — `SECURITY DEFINER` so `pinecone.status`
    -- (itself `SECURITY INVOKER`) can read it without a direct grant on
    -- `pinecone.outbox`. `REVOKE EXECUTE FROM PUBLIC` below (a plain
    -- `has_table_privilege` check inside this function would run as its
    -- *definer*, not the caller, so it can't gate access itself); `status()`
    -- checks the *caller's* own EXECUTE privilege before calling this and
    -- degrades to NULL backlog columns otherwise (see `status.rs`).
    CREATE FUNCTION pinecone._outbox_stats(
        p_relid oid,
        OUT backlog bigint,
        OUT oldest_enqueued timestamptz,
        OUT last_error text,
        OUT in_backoff boolean
    )
    LANGUAGE sql
    SECURITY DEFINER
    STABLE
    SET search_path = pg_catalog, pg_temp
    AS $$
        SELECT
            count(*),
            min(enqueued_at),
            (array_agg(last_error ORDER BY id) FILTER (WHERE last_error IS NOT NULL))[1],
            max(next_retry_at) > now()
        FROM pinecone.outbox WHERE relid::oid = p_relid
    $$;
    "#,
    name = "pg_pinecone_outbox",
    requires = [pinecone],
);

// `_drain_relid` is an ordinary `#[pg_extern]` inside `pg_pinecone`'s
// `#[pg_schema]` module in `lib.rs` (delegating straight here), so this
// `REVOKE` must run after it exists — same reasoning as
// `sync_trigger_helpers_revoke` in `sync_trigger.rs`. It doesn't need
// `SECURITY DEFINER` itself: `pinecone._sync_trigger` (already `SECURITY
// DEFINER`) calls it inline for `sync_mode = 'sync'` tables, and a
// `SECURITY DEFINER` function's callees keep running as its definer for the
// rest of that call — REVOKE EXECUTE FROM PUBLIC only closes the *direct*
// call path a session with no other privileges could otherwise use to drain
// (and so push under the stored key for) an arbitrary tracked table.
extension_sql!(
    "REVOKE EXECUTE ON FUNCTION pinecone._drain_relid(oid) FROM PUBLIC;",
    name = "pg_pinecone_outbox_drain_revoke",
    requires = [pinecone, _drain_relid],
);

// `_outbox_stats` is `SECURITY DEFINER` (see its comment above), so leaving
// it `PUBLIC`-executable would let any role with schema `USAGE` learn another
// table's backlog size/oldest-enqueued timestamp/last error — none of which
// `pinecone.status`'s other, `SECURITY INVOKER` columns leak to a caller with
// no privileges on the table itself. `status()` degrades to NULL backlog
// columns for a caller this REVOKE denies, rather than erroring out.
extension_sql!(
    "REVOKE EXECUTE ON FUNCTION pinecone._outbox_stats(oid) FROM PUBLIC;",
    name = "pg_pinecone_outbox_stats_revoke",
    requires = [pinecone, "pg_pinecone_outbox"],
);

/// Outbox rows fetched per [`drain_step`] batch — matches the sync
/// trigger's/backfill's existing 1000-row batching budget divided down for a
/// single batch that also holds row locks for its duration. `pub(crate)` so
/// tests can force a multi-batch drain deterministically.
pub(crate) const DRAIN_BATCH_ROWS: i64 = 100;

/// One `pinecone.outbox` row as read by [`fetch_batch`].
struct OutboxRow {
    id: i64,
    op: String,
    payload: Option<JsonB>,
}

/// Implements `pinecone._drain_relid`: the inline `sync_mode = 'sync'` drain,
/// running in the writer's own session. Loops [`drain_step`] (always with
/// [`DrainCredentials::Inline`]) until the queue is empty, re-raising the
/// client error as a Postgres `ERROR` on any non-`Processed` outcome — which
/// aborts whatever statement/transaction called this. For this caller,
/// that's the writer's own statement: its just-made enqueue (and this call's
/// own `attempts` update) roll back with it, so a failure here leaves
/// `pinecone.outbox` exactly as empty as it was before the statement —
/// matching 0.2's fail-closed semantics exactly (see the `#[pg_test]`s in
/// `lib.rs`). Persisting `attempts`/`last_error` *across* such a rollback
/// needs an independent transaction (the background worker's own), so it
/// isn't exercised by this path.
pub(crate) fn drain_relid(relid: pg_sys::Oid) {
    loop {
        match drain_step(relid, DrainCredentials::Inline) {
            DrainOutcome::Processed(0) => return,
            DrainOutcome::Processed(_) => {}
            DrainOutcome::Backoff(err) | DrainOutcome::Parked(err) => backend::raise(err),
        }
    }
}

/// One batch's outcome, as returned by [`drain_step`].
#[derive(Debug)]
pub(crate) enum DrainOutcome {
    /// The batch (`n` rows, possibly 0 — an empty queue) fully succeeded and
    /// was deleted: live rows and the rebuild jobs that finished.
    Processed(u64),
    /// The batch's head-of-line row failed with a transient error (timeout,
    /// transport failure, or a 5xx/429 response — see
    /// [`backend::is_transient`]); its `attempts`/`last_error` were recorded,
    /// but it was left in place — the relid should be retried later, with
    /// backoff.
    Backoff(ClientError),
    /// The batch's head-of-line row failed with a permanent-class error
    /// `pinecone.outbox_max_attempts` times in a row;
    /// the relid's queue is now parked (`pinecone.tables.queue_parked`) and
    /// will be skipped until `pinecone.outbox_retry` unparks it.
    Parked(ClientError),
}

/// A [`drain_step`]'s outcome and whether a job it ran has chunks left for the next round.
pub(crate) struct Step {
    pub(crate) outcome: DrainOutcome,
    pub(crate) chunks_left: bool,
}

impl Step {
    fn of(outcome: DrainOutcome) -> Self {
        Self {
            outcome,
            chunks_left: false,
        }
    }
}

/// Drains one batch (up to [`DRAIN_BATCH_ROWS`] rows, oldest first) of
/// `relid`'s `pinecone.outbox` queue: for [`DrainCredentials::Worker`] only,
/// takes `relid`'s advisory transaction lock first (serializes this worker
/// round against a concurrent worker round on the same relid, and
/// `demote`/`sync`/`ambulkdelete`); skipped for
/// [`DrainCredentials::Inline`] (see the comment at that check for why it's
/// safe to), then claims the batch `FOR UPDATE` (plain — no `SKIP LOCKED`;
/// a single drainer per relid *for the worker path* is the invariant the
/// advisory lock establishes there), and dispatches each row in order
/// through `sync_trigger`'s `try_*` cores via `client` (resolved once per
/// batch, per `creds`). Rows before the first failure are deleted regardless
/// of outcome; the first failing row is never advanced past
/// — its `last_error` (and, for a permanent-class failure only, its
/// `attempts` — transient blips never count toward parking) are recorded, and
/// it is classified via [`backend::is_transient`] into
/// [`DrainOutcome::Backoff`] or (past `pinecone.outbox_max_attempts`)
/// [`DrainOutcome::Parked`].
///
/// The worker's step runs a `'backfill'` job at the head of the batch chunk after chunk until the
/// round's `pinecone.backfill_round_ms` has elapsed ([`run_chunks`]; the budget is shared by the
/// steps of a round, [`Scheduler::round`]), then spends what is left of it on the
/// `'rebuild_partition'` jobs of retried partitions ([`rebuild_partitions`]), which the batch
/// leaves out so that they never hold the live rows behind them back. The inline drain runs one
/// chunk of a backfill job and stops at a rebuild job.
pub(crate) fn drain_step(relid: pg_sys::Oid, creds: DrainCredentials) -> DrainOutcome {
    drain_step_reporting(relid, creds).outcome
}

/// [`drain_step`], saying also whether a backfill, rebuild or recover job stopped with chunks left.
pub(crate) fn drain_step_reporting(relid: pg_sys::Oid, creds: DrainCredentials) -> Step {
    drain_step_within(relid, creds, &mut ChunkBudget::new())
}

/// [`drain_step_reporting`] spending its chunks from `clock`, the budget of the round it is a step of.
fn drain_step_within(relid: pg_sys::Oid, creds: DrainCredentials, clock: &mut ChunkBudget) -> Step {
    // Only `Worker` needs the relid lock. It's held until the *caller's*
    // COMMIT — for `Inline` that's the writer's own statement/transaction,
    // so taking it there would serialize every concurrent writer to the
    // same table across the whole HTTP round trip it's about to make.
    // Skipping it for `Inline` is safe: a same-relid inline drain still
    // serializes on the outbox rows' own `FOR UPDATE` locks below (whichever
    // session gets there first blocks the other), and a concurrent `demote`
    // is excluded by its own `DROP TRIGGER`'s `AccessExclusiveLock` on the
    // table, which nothing running underneath an inline drain can be
    // holding. The worker path still needs the wider lock: without it, two
    // worker rounds (or a worker racing an inline drain left behind by an
    // earlier failed statement) could interleave batches on the same relid
    // out of FIFO order.
    let worker = matches!(creds, DrainCredentials::Worker);
    if worker {
        lock_relid(relid);
    }

    let rows = fetch_batch(relid, worker);
    let first_job = if rows.is_empty() {
        match worker.then(|| next_job(relid, 0)).flatten() {
            Some(job) => Some(job),
            None => return Step::of(DrainOutcome::Processed(0)),
        }
    } else {
        None
    };

    // Worker-only pre-flight, checked *before* building `client`: a database
    // with backlog but no stored key yet would otherwise reach
    // `client_for_drain` -> `stored_credentials`, whose `error!` aborts the
    // whole round's transaction — crash-looping
    // the worker every naptime instead of just waiting. `Inline` never needs
    // this: it resolves credentials via `effective_credentials`, which
    // raises its own two-remedy error straight back to the writer that
    // triggered it, not a background loop.
    if worker && !secret::has_stored_api_key() {
        let head = rows.first().or(first_job.as_ref());
        record_transient_failure(
            head.map(|row| row.id)
                .expect("a step with nothing queued returned above"),
            NO_STORED_KEY_LAST_ERROR,
        );
        return Step::of(DrainOutcome::Backoff(ClientError::MissingApiKey));
    }

    let client = backend::client_for_drain(creds);
    let mut gate = Gate::new(worker);
    let mut processed_ids: Vec<i64> = Vec::new();
    let mut failure: Option<(i64, ClientError)> = None;
    let mut halted = false;
    let mut chunks_left = false;
    // Coalesce a maximal run of consecutive `'upsert'` rows
    // into ONE batched `upsert_vectors` call instead of one HTTP round trip
    // per row (~100x on many-small-transaction workloads — the steady-state
    // trigger path used to pay one ~150ms RTT per outbox row). Per-id
    // last-op-wins within the run, and the run breaks at any non-upsert op
    // (`delete`/`delete_all`/`backfill`/`recover`/`update`/`reupsert`), so
    // per-id FIFO ordering across those boundaries is preserved exactly.
    let mut rows = rows.into_iter().peekable();
    while let Some(row) = rows.next() {
        if row.op == "upsert" {
            // The whole coalesced run is one retry unit: its head row is the
            // FIFO head-of-line that stays queued (and accrues attempts) on a
            // failure, matching the per-row invariant before coalescing.
            let head_id = row.id;
            let mut run_ids = vec![row.id];
            let mut combined: Vec<Value> = Vec::new();
            let mut index: HashMap<UpsertKey, usize> = HashMap::new();
            fold_upsert_payload(row.payload, &mut combined, &mut index);
            while rows.peek().is_some_and(|r| r.op == "upsert") {
                let next = rows.next().expect("peek just confirmed a row");
                run_ids.push(next.id);
                fold_upsert_payload(next.payload, &mut combined, &mut index);
            }
            match sync_trigger::try_push_rows(
                relid,
                JsonB(Value::Array(combined)),
                &client,
                &mut gate,
            ) {
                Ok(()) => processed_ids.extend(run_ids),
                Err(e) => {
                    failure = Some((head_id, e));
                    break;
                }
            }
            continue;
        }

        let id = row.id;
        // A `'recover'` head is worker-only work: the
        // inline drain runs in the WRITER's own statement, and a recover
        // chunk is a list+fetch HTTP round-trip plus a bulk local UPDATE —
        // a user's INSERT on a `sync_mode = 'sync'` table must neither pay
        // that latency nor abort on the chunk's own transient fetch errors.
        // Stopping here (rather than skipping past it) preserves FIFO: the
        // user's just-enqueued write stays queued behind the job until the
        // worker drains both, the same head-of-line convention 'backfill'
        // rows already follow. Not a failure: the statement succeeds, its
        // write safely persisted in the outbox. A `'rebuild_partition'`
        // rebuild waits for the worker the same way.
        if matches!(creds, DrainCredentials::Inline)
            && matches!(row.op.as_str(), "recover" | "rebuild_partition")
        {
            break;
        }
        // The worker runs a backfill job's chunks back to back; the inline drain, in the
        // writer's statement, runs one.
        let outcome = if worker && row.op == "backfill" {
            run_chunks(relid, id, row.payload, &client, &mut gate, false, clock)
        } else {
            dispatch(relid, row, &client, &mut gate)
        };
        match outcome {
            DispatchOutcome::Done => processed_ids.push(id),
            // A `'backfill'` row with more chunks to go (for the worker: the round's budget
            // is spent): it stays queued (never added to `processed_ids`) and — since it must
            // stay head-of-line until exhausted — no later row in this same batch is
            // dispatched this round either. Every chunk's cursor is
            // committed with the round. Not a failure: `Processed` is the right outcome so
            // `Scheduler` resets backoff state.
            DispatchOutcome::InProgress => {
                chunks_left = true;
                break;
            }
            // The dropped-relid purge already deleted every queued row for
            // `relid` — later rows of this same in-memory batch included, so
            // dispatching (or re-deleting) them would be wrong. A success:
            // the relid's queue is now empty by construction.
            DispatchOutcome::Purged => {
                halted = true;
                break;
            }
            // The heap relation's `AccessShareLock` probe came back
            // unavailable (see
            // `dispatch_backfill`'s own rustdoc for the worker↔DROP deadlock
            // this avoids). The row stays queued, completely untouched, to
            // be retried next round at the ordinary naptime cadence — no
            // penalty (no `attempts`/`last_error` write, no `Scheduler`
            // backoff): this is expected, usually-short-lived contention,
            // not a poison row.
            DispatchOutcome::LockUnavailable => {
                halted = true;
                break;
            }
            DispatchOutcome::Err(e) => {
                failure = Some((id, e));
                break;
            }
        }
    }

    let mut processed = processed_ids.len() as u64;
    if !processed_ids.is_empty() {
        delete_rows(&processed_ids);
    }

    if worker && failure.is_none() && !halted {
        let rebuilt = rebuild_partitions(relid, &client, &mut gate, first_job, clock);
        processed += rebuilt.done.len() as u64;
        if !rebuilt.done.is_empty() {
            delete_rows(&rebuilt.done);
        }
        chunks_left |= rebuilt.chunks_left;
        failure = rebuilt.failure;
    }

    let Some((failed_id, err)) = failure else {
        return Step {
            outcome: DrainOutcome::Processed(processed),
            chunks_left,
        };
    };

    // A failed step is blocked work, whatever ran before the failure.
    if backend::is_transient(&err) {
        record_transient_failure(failed_id, &backend::failure_text(&err));
        return Step::of(DrainOutcome::Backoff(err));
    }

    let attempts = record_permanent_failure(failed_id, &err);
    if attempts >= max_attempts() {
        registry::set_parked(relid, &backend::failure_text(&err));
        clear_next_retry_at(failed_id);
        Step::of(DrainOutcome::Parked(err))
    } else {
        set_next_retry_at(failed_id);
        Step::of(DrainOutcome::Backoff(err))
    }
}

/// [`drain_step`]'s worker-only "no stored key yet" `last_error` — distinct
/// from [`backend::stored_credentials`]'s own `error!` message (still the
/// defensive fallback for the narrow TOCTOU window between this check and
/// `client_for_drain` actually reading `pinecone.secret`), since this one
/// only ever reaches `pinecone.status`'s `last_error` column, not a raised
/// Postgres error.
const NO_STORED_KEY_LAST_ERROR: &str =
    "no stored API key — run pinecone.set_api_key(...) before the worker can drain this table";

/// Current value of the `pinecone.outbox_max_attempts` GUC.
fn max_attempts() -> i32 {
    crate::OUTBOX_MAX_ATTEMPTS.get()
}

/// Takes `relid`'s advisory transaction lock — held for the rest of the
/// current transaction, released automatically at commit/rollback.
/// The uniform lock the worker path takes before
/// touching a relid's outbox rows (see `drain_step`'s own comment for why
/// `Inline` skips it), and — `pub(crate)` for exactly this — `promote::demote`/
/// `promote::sync` now take FIRST too, before any table lock (`DROP TRIGGER`'s
/// `AccessExclusiveLock`) or registry/outbox write, so a worker round and a
/// concurrent `demote`/`sync` can never interleave, nor deadlock against each
/// other (a consistent advisory-then-table lock order rules out the ABBA
/// cycle a worker round's own `dispatch_backfill` — which takes this same
/// lock before its own, weaker `AccessShareLock` table read — would otherwise
/// risk against `demote`'s `DROP TRIGGER`).
pub(crate) fn lock_relid(relid: pg_sys::Oid) {
    Spi::run_with_args(
        "SELECT pg_catalog.pg_advisory_xact_lock($1)",
        &[advisory_lock_key(relid).into()],
    )
    .expect("pg_pinecone: outbox drain advisory lock failed");
}

/// [`lock_relid`]'s single-bigint advisory-lock key: `(MyDatabaseId << 32) |
/// relid`. A bare `relid` alone (the original key) isn't enough — Postgres
/// advisory locks are a *cluster-wide* keyspace, not scoped to
/// `current_database()` the way ordinary relation locks are, but a `regclass`
/// oid is only unique *within* its own database's `pg_class`. Two databases
/// can easily share the same numeric relid for unrelated tables (most
/// obviously a `CREATE DATABASE ... TEMPLATE`-cloned one, which starts out
/// with identical oids to its template) — without folding in the database
/// oid, a drain in one database would needlessly (and incorrectly) serialize
/// against a same-numbered relid's drain in a completely different database.
/// Still the single-bigint form (`pg_advisory_xact_lock(bigint)`,
/// `objsubid = 1` in `pg_locks`), so it can never collide with the worker's
/// own two-int4-key singleton lock (`WORKER_LOCK_MAGIC`, dbid) — see
/// `worker_lock_held`'s comment for why that holds regardless of the actual
/// numeric values, which can otherwise coincide (this key's high 32 bits —
/// `pg_locks.classid` — are the same database oid that appears as
/// `objid` in the worker lock's own `pg_locks` row).
fn advisory_lock_key(relid: pg_sys::Oid) -> i64 {
    let dbid = unsafe { pg_sys::MyDatabaseId }.to_u32() as i64;
    let relid = relid.to_u32() as i64;
    (dbid << 32) | relid
}

/// [`dispatch`]'s outcome for one outbox row — a superset of a plain
/// `Result<(), ClientError>`: `'backfill'` rows can also come back
/// [`Self::InProgress`] (more chunks remain; not a failure) without having
/// actually finished.
enum DispatchOutcome {
    /// The row's op fully completed — safe to delete.
    Done,
    /// A `'backfill'` row whose chunk read came back non-empty: more chunks
    /// remain, so this row must stay queued (and head-of-line) rather than
    /// being deleted — its persisted `payload.after_id` cursor already
    /// advanced, though.
    InProgress,
    /// [`dispatch_backfill`] found `relid`'s table dropped and purged every
    /// queued row for it —
    /// including any *later* rows of this same in-memory batch, which must
    /// therefore not be dispatched (nor re-deleted): [`drain_step`] stops the
    /// batch here, as a success.
    Purged,
    /// [`dispatch_backfill`]'s non-blocking probe for the heap relation's
    /// `AccessShareLock` came back unavailable — the row stays queued,
    /// untouched, to be retried next round (see
    /// `dispatch_backfill`'s own rustdoc).
    LockUnavailable,
    /// The op failed with `e` — [`drain_step`] classifies and records it.
    Err(ClientError),
}

/// What identifies one vector in a folded run of upserts: its id, and its partition value when
/// the table is partitioned (the same id may live in two namespaces for a moment).
type UpsertKey = (Option<String>, String);

/// Folds one `'upsert'` outbox row's payload (a JSON array of
/// `{id, values, metadata?, p?}` items) into `combined`, keyed by [`UpsertKey`] for a
/// last-op-wins dedup — a later write for the same key overwrites the earlier
/// one's value in place (position, hence FIFO ordering, preserved). A missing
/// or JSON-`null` payload contributes nothing. An item with a non-string or
/// missing `id` is passed through unfolded, so `try_push_rows`/`parse_vector`
/// raise on it exactly as they would have per-row (never silently dropped).
fn fold_upsert_payload(
    payload: Option<JsonB>,
    combined: &mut Vec<Value>,
    index: &mut HashMap<UpsertKey, usize>,
) {
    let items = match payload {
        Some(JsonB(Value::Array(items))) => items,
        None | Some(JsonB(Value::Null)) => return,
        Some(JsonB(other)) => {
            error!("pinecone._drain_relid: 'upsert' row payload must be a JSON array, got {other}")
        }
    };
    fold_upsert_items(items, combined, index);
}

/// [`fold_upsert_payload`] without the Postgres types: pure, so a plain unit test can run it.
fn fold_upsert_items(
    items: Vec<Value>,
    combined: &mut Vec<Value>,
    index: &mut HashMap<UpsertKey, usize>,
) {
    for item in items {
        // Own the key before moving `item`, so no borrow of `item` outlives
        // the move into `combined`.
        let partition = item.get("p").and_then(Value::as_str).map(str::to_string);
        match item.get("id").and_then(Value::as_str).map(str::to_string) {
            Some(id) => {
                let key = (partition, id);
                if let Some(&pos) = index.get(&key) {
                    combined[pos] = item; // last-op-wins
                } else {
                    index.insert(key, combined.len());
                    combined.push(item);
                }
            }
            None => combined.push(item),
        }
    }
}

/// Dispatches one non-upsert outbox row through the same Rust internals
/// `pinecone._delete_ids`/`_delete_all` delegate to (for `'delete'`/
/// `'delete_all'`), or [`dispatch_backfill`]/[`dispatch_recover`] (for
/// `'backfill'`/`'recover'`), plus the metadata legs. Both drain paths now
/// coalesce `'upsert'` rows in [`drain_step`] before dispatch, so the
/// `'upsert'` arm below is unreachable in practice; it's kept so `dispatch`
/// stays a complete, independently-correct single-row op dispatcher.
fn dispatch(
    relid: pg_sys::Oid,
    row: OutboxRow,
    client: &Client,
    gate: &mut Gate,
) -> DispatchOutcome {
    let result = match row.op.as_str() {
        "upsert" => sync_trigger::try_push_rows(
            relid,
            row.payload.unwrap_or(JsonB(Value::Array(Vec::new()))),
            client,
            gate,
        ),
        "delete" => {
            sync_trigger::try_delete_ids(relid, delete_ids_payload(row.payload), client, gate)
        }
        "delete_all" => sync_trigger::try_delete_all(relid, client, gate),
        "backfill" => return dispatch_backfill(relid, row.id, row.payload, client, gate, false),
        "rebuild_partition" => {
            return dispatch_backfill(relid, row.id, row.payload, client, gate, true);
        }
        "recover" => return dispatch_recover(relid, row.id, row.payload, client),
        "update" => sync_trigger::try_update_metadata(
            relid,
            row.payload.unwrap_or(JsonB(Value::Array(Vec::new()))),
            client,
        ),
        "reupsert" => sync_trigger::try_reupsert_metadata(
            relid,
            row.payload.unwrap_or(JsonB(Value::Array(Vec::new()))),
            client,
        ),
        other => panic!(
            "pinecone._drain_relid: outbox row {} has op '{other}', which draining does not \
             support",
            row.id
        ),
    };
    match result {
        Ok(()) => DispatchOutcome::Done,
        Err(e) => DispatchOutcome::Err(e),
    }
}

/// Dispatches a `'backfill'` job row: reads and pushes one chunk of `relid`'s
/// live table, starting after the payload's persisted `after_id` cursor —
/// in PK order (`promote::try_backfill_chunk`) for a `pinecone.promote`-driven
/// job, or in block-range heap order with HOT-chain-root ids
/// (`promote::try_tid_backfill_chunk`; the cursor is a block number there)
/// for an `am::ambuild`-driven one (`payload.id_kind = "tid"`).
///
/// - More rows remain past this chunk: persists the new cursor
///   ([`set_backfill_after_id`]) and returns [`DispatchOutcome::InProgress`]
///   — the job row stays queued, to be picked up again next round.
/// - The table is exhausted from `after_id` onward: flips the registry row
///   `'promoting' -> 'remote'` ([`registry::flip_promoting_to_remote`], a
///   no-op if it's already `'remote'` — e.g. a `pinecone.sync`-triggered
///   rebuild on an already-`'remote'` table) in the same (ambient)
///   transaction [`drain_step`]'s caller commits the job row's own deletion
///   in, and returns [`DispatchOutcome::Done`].
/// - The push itself fails: returns [`DispatchOutcome::Err`], same as any
///   other op — the persisted `after_id` cursor (from a *previous*,
///   already-committed chunk) is what makes resuming after this restart-safe.
/// - A job scoped to one partition (`'rebuild_partition'`) starts with
///   [`wipe_partition`], until its first chunk has committed a cursor.
/// - `relid` no longer resolves to a table at all (dropped, or renamed away
///   — `catalog::resolve_table_name`): there is nothing left to backfill
///   from, and opening it below would raise a hard Postgres `ERROR` for an
///   oid that no longer exists — which would abort the whole worker round
///   (and, retried every round thereafter, crash-loop the worker process).
///   Instead, purges every row still queued for `relid`
///   (this job included — they're all equally moot) with a `WARNING`, and
///   returns [`DispatchOutcome::Purged`] so the round moves on to its other
///   relids.
/// - The heap relation's `AccessShareLock` can't be acquired right now:
///   returns [`DispatchOutcome::LockUnavailable`] instead of waiting for it
///   — **avoids a worker↔DROP deadlock.**
///   This used to open the table with the blocking `PgRelation::with_lock`,
///   while `drain_step`'s `Worker` path already holds `relid`'s advisory
///   xact lock *and* (via `fetch_batch`'s `SELECT ... FOR UPDATE`) a row
///   lock on this very job row. A concurrent `DROP INDEX`/`DROP TABLE` on
///   this AM index needs `AccessExclusiveLock` on the table — it doesn't
///   contend with the advisory lock directly, but once it's queued waiting
///   for that `AccessExclusiveLock` (behind some unrelated, already-granted
///   share-mode lock), Postgres's own fairness rule makes THIS call's fresh
///   `AccessShareLock` request queue behind it too, even though the two
///   share-mode locks are mutually compatible — so the worker would now wait
///   on the DROP. The DROP, in turn, waits on the worker: its `sql_drop`
///   event trigger body (`am.rs`) does `DELETE FROM pinecone.outbox WHERE
///   relid = ...` as part of its own transaction, which needs the row lock
///   this job row's own `FOR UPDATE` claim is holding — released only at the
///   worker's commit. Two transactions, each waiting on a lock the other
///   holds: Postgres's deadlock detector eventually breaks this (aborting
///   one side after `deadlock_timeout`), but that still means either the
///   user's `DROP` or the worker's whole round fails with a raw "deadlock
///   detected" error. Probing non-blockingly and giving up the row instead
///   avoids the worker ever joining that wait queue in the first place —
///   it releases its row lock (via a normal batch commit) promptly, letting
///   the DROP proceed unimpeded.
fn dispatch_backfill(
    relid: pg_sys::Oid,
    job_id: i64,
    payload: Option<JsonB>,
    client: &Client,
    gate: &mut Gate,
    scoped: bool,
) -> DispatchOutcome {
    if catalog::resolve_table_name(relid).is_none() {
        let purged = clear_relid(relid);
        warning!(
            "pg_pinecone: backfill dispatch: relid {relid} no longer resolves to a table — purged \
             {purged} queued pinecone.outbox row(s) for it"
        );
        return DispatchOutcome::Purged;
    }

    // `am::ambuild` enqueues TID-keyed backfill jobs (`payload.id_kind =
    // "tid"` — ids are `"block:offset"` heap TIDs, never this table's
    // primary key — `registry::synced_by` names the same distinction); a
    // `pinecone.promote`-driven job stays PK-keyed with no `id_kind` at all.
    // `promote::try_backfill_chunk`'s native-type `after_id` cast (keyed to
    // the id COLUMN's type) would silently misdispatch against a TID
    // cursor, so this must branch before ever reaching it. Checked by
    // reference first (`JsonB` has no `Clone`): [`backfill_after_id`] below
    // consumes `payload`.
    let is_tid = backfill_id_kind(&payload).as_deref() == Some("tid");
    let after_p = backfill_after_p(&payload);
    let only = scoped.then(|| rebuild_partition_value(&payload));
    let after_id = backfill_after_id(payload);
    // Missing registration: warn + purge, never panic (same reasoning as
    // `sync_trigger::remote_target`): the table can outlive its registration
    // (e.g. a demote completing after
    // this backfill was enqueued), and a panic here abort-loops the worker
    // on one poison row forever; the job is moot without a target.
    let Some(row) = registry::lookup(relid) else {
        let purged = clear_relid(relid);
        warning!(
            "pg_pinecone: backfill dispatch: relid {relid} has no pinecone.tables registration anymore \
             (dropped or demoted since it was enqueued) — purged {purged} moot queued row(s)"
        );
        return DispatchOutcome::Purged;
    };
    let index_name = row
        .index_name
        .clone()
        .expect("pg_pinecone: promoting/remote row always has index_name");
    let namespace = row
        .namespace
        .clone()
        .expect("pg_pinecone: promoting/remote row always has namespace");
    // `AccessShareLock`, but probed via `ConditionalLockRelationOid` rather
    // than the blocking `PgRelation::with_lock` this used to call — see this
    // function's own rustdoc for the worker↔DROP deadlock that avoids.
    const LOCKMODE: pg_sys::LOCKMODE = pg_sys::AccessShareLock as pg_sys::LOCKMODE;
    if !unsafe { pg_sys::ConditionalLockRelationOid(relid, LOCKMODE) } {
        return DispatchOutcome::LockUnavailable;
    }
    // `RelationIdGetRelation` (rather than `relation_open`, which would
    // re-acquire the lock — already held above) mirrors `relation_open`'s
    // own "lock, then open" sequence split into its two steps; `from_pg_owned`
    // so `tbl`'s `Drop` balances this open with a `RelationClose` (as any
    // `RelationIdGetRelation` caller must), but — `lockmode: None`, unlike
    // `with_lock` — never attempts to unlock a lockmode it never recorded
    // taking itself. The lock is released at this transaction's end instead,
    // same as every other advisory/table lock the design takes.
    let heap_ptr = unsafe { pg_sys::RelationIdGetRelation(relid) };
    if heap_ptr.is_null() {
        // Should be unreachable: `catalog::resolve_table_name` above just
        // confirmed `relid` resolves, and holding its lock (just acquired)
        // rules out a concurrent drop finishing in between.
        panic!(
            "pg_pinecone: backfill dispatch: relid {relid} resolved moments ago but is now gone"
        );
    }
    let tbl = unsafe { PgRelation::from_pg_owned(heap_ptr) };

    if let Some(value) = &only
        && row.partition.is_some()
        && after_p.is_none()
        && after_id.is_none()
    {
        match wipe_partition(relid, client, gate, &index_name, &namespace, value) {
            Ok(true) => {}
            Ok(false) => return DispatchOutcome::Done,
            Err(e) => return DispatchOutcome::Err(e),
        }
    }
    gate.scope(only);
    let chunk_result: Result<Option<BackfillCursor>, ClientError> = if is_tid {
        promote::try_tid_backfill_chunk(
            &tbl,
            &row,
            client,
            &index_name,
            &namespace,
            after_id.as_deref(),
            gate,
        )
        .map(|last| last.map(BackfillCursor::id))
    } else if row.partition.is_some() {
        let after = after_p.as_deref().zip(after_id.as_deref());
        promote::try_backfill_chunk_partitioned(
            &tbl,
            &row,
            client,
            &index_name,
            &namespace,
            after,
            gate,
        )
        .map(|last| last.map(BackfillCursor::partitioned))
    } else if scoped {
        Ok(None)
    } else {
        promote::try_backfill_chunk(
            &tbl,
            &row,
            client,
            &index_name,
            &namespace,
            after_id.as_deref(),
        )
        .map(|last| last.map(BackfillCursor::id))
    };
    gate.scope(None);

    match chunk_result {
        Ok(Some(cursor)) => {
            set_backfill_cursor(job_id, &cursor);
            DispatchOutcome::InProgress
        }
        Ok(None) if scoped => DispatchOutcome::Done,
        Ok(None) => {
            // A no-op for an AM row created by a plain `CREATE INDEX ...
            // USING pinecone` (`am::ambuild` registers it `mode = 'remote'`
            // from the start — every write goes straight through
            // `aminsert`'s own outbox rows, never gated on a backfill
            // draining first) — `flip_promoting_to_remote`'s `AND mode =
            // 'promoting'` guard makes calling it on an already-`'remote'`
            // row a harmless no-op. Real for a `pinecone.promote` async row,
            // as it always was.
            //
            // Also: an AM row created by `lifecycle::promote_index`
            // STARTS `'promoting'` (registry AND metapage alike) — for that
            // case, this is the worker's one and only chance to also flip
            // the AM INDEX'S OWN metapage `promoting -> remote`
            // (`local::metapage::flip_mode`, one atomic GenericXLog record):
            // once this returns, scans switch from serving local to serving
            // remote. Conditioned on `synced_by == 'am'` and the metapage
            // still reading `Promoting` (an idempotency guard — a redundant
            // dispatch, e.g. a retried round after an earlier one already
            // flipped it, must not attempt to flip an already-`Remote`
            // metapage a second time).
            registry::flip_promoting_to_remote(relid);
            if row.synced_by == "am" && !flip_am_index_promoting_to_remote(row.am_index_relid) {
                // The index's own `AccessShareLock` wasn't available this
                // round (see that function's rustdoc for the worker<->DROP
                // deadlock this avoids) — the registry flip above already
                // committed (harmless either way), but the metapage flip is
                // the only thing that actually moves scans over to remote,
                // so the job must stay queued for a retry rather than being
                // deleted: `Ok(None)` will simply recur next round (nothing
                // left to backfill) until the lock is free.
                return DispatchOutcome::LockUnavailable;
            }
            DispatchOutcome::Done
        }
        Err(e) => DispatchOutcome::Err(e),
    }
}

/// The first step of a `'rebuild_partition'` job, before any chunk of its rows: deletes the
/// partition's namespace and forgets it, so that only the load registers it again (a partition
/// with no rows leaves no registry row behind). `Ok(false)` when there is nothing to load: the
/// partition failed again since the retry, or Pinecone refused this delete, which fails it again.
fn wipe_partition(
    relid: pg_sys::Oid,
    client: &Client,
    gate: &mut Gate,
    index_name: &str,
    prefix: &str,
    value: &str,
) -> Result<bool, ClientError> {
    if gate.skips(relid, value) {
        return Ok(false);
    }
    let namespace = partition::namespace_for(prefix, value);
    match client.delete_namespace(index_name, &namespace) {
        Ok(()) => {
            partition_registry::forget_partition(relid, Lane::Vector, &namespace);
            Ok(true)
        }
        Err(err) => {
            let err = partition_lane::name_namespace(err, &namespace);
            gate.refused(relid, value, &namespace, err).map(|()| false)
        }
    }
}

/// Flips a `lifecycle::promote_index`-initiated AM index's own metapage
/// `Promoting -> Remote` once its backfill is exhausted (the `Ok(None)` arm
/// above) — the worker's one and only chance to do so, since that's the
/// moment `local::metapage::read`'s dispatch (`am.rs`'s `aminsert`/
/// `ambulkdelete`/`ambeginscan`) should start serving remote instead of
/// local. Returns `false` only when the index relation's `AccessShareLock`
/// couldn't be acquired right now — the same non-blocking probe (and the
/// same worker<->DROP deadlock reasoning) `dispatch_backfill` already uses
/// for the heap relation, reused here for the INDEX relation instead, since
/// a concurrent `DROP INDEX` needs the same `AccessExclusiveLock` this would
/// otherwise queue behind. The caller must leave the backfill job queued in
/// that case, so a later round retries the flip — losing it here would
/// otherwise strand the index in `Promoting` (serving local) forever, since
/// the job row this call is triggered from is about to be deleted.
///
/// A pure no-op returning `true` if `am_index_relid` is `None` (a
/// `synced_by = 'trigger'` row never reaches here — see the call site's own
/// `synced_by == "am"` guard), if the index was concurrently dropped (its
/// `sql_drop` event trigger already purged the registry row this whole
/// dispatch is running against — nothing left to flip), or if the metapage
/// isn't `Promoting` (a redundant dispatch, e.g. a retried round after an
/// earlier one already flipped it — `am::ambuild`'s registered rows are
/// never `Promoting` in the first place, so this only ever matters for a
/// `promote_index`-created one).
fn flip_am_index_promoting_to_remote(am_index_relid: Option<pg_sys::Oid>) -> bool {
    let Some(index_relid) = am_index_relid else {
        return true;
    };
    const LOCKMODE: pg_sys::LOCKMODE = pg_sys::AccessShareLock as pg_sys::LOCKMODE;
    if !unsafe { pg_sys::ConditionalLockRelationOid(index_relid, LOCKMODE) } {
        return false;
    }
    let index_ptr = unsafe { pg_sys::RelationIdGetRelation(index_relid) };
    if index_ptr.is_null() {
        return true; // concurrently dropped — nothing left to flip.
    }
    let index = unsafe { PgRelation::from_pg_owned(index_ptr) };
    let meta = crate::local::metapage::read(index.as_ptr());
    if meta.mode == crate::local::metapage::Mode::Promoting {
        unsafe {
            crate::local::metapage::flip_mode(index.as_ptr(), crate::local::metapage::Mode::Remote);
        }
    }
    // Sticky-engine map too (update the map at
    // every lifecycle flip, rather than teaching ambuild to reconcile against
    // the registry row at rebuild time): a promoted index whose
    // `pinecone._am_engine` row still said 'local' would have a later plain
    // REINDEX (mode=auto path) rebuild it LOCAL — silently undoing the
    // promotion, the exact drift `resolve_mode`'s stickiness exists to
    // prevent. The demote side needs no counterpart here: its rebuild runs
    // with an explicit `ALTER INDEX ... SET (mode='local')` reloption, and
    // `resolve_mode` persists every explicit resolution.
    //
    // This upsert must NOT be nested inside the `if` above
    // (its original home) — the metapage flip just above is one atomic,
    // durable `GenericXLog` record that takes effect independently of this
    // call's own enclosing transaction (that's the whole crash-safety point
    // of a metapage flip), while the upsert below is an ordinary
    // transactional heap write. A worker crash between the two — the flip's
    // WAL record reaching disk (e.g. via a concurrent backend's unrelated
    // flush) before this call's transaction ever commits — replays the flip
    // on restart but rolls back the upsert, so `meta.mode` reads `Remote` on
    // every later round while the guard above (now false forever) never lets
    // the upsert run again: `pinecone._am_engine` sticks at its pre-promotion
    // value for good. Running the upsert unconditionally whenever the
    // metapage reads `Remote` — whether this call just flipped it or an
    // earlier, crashed round already did — makes each round a self-healing,
    // idempotent check instead of a one-shot that a crash can permanently
    // defeat.
    if meta.mode == crate::local::metapage::Mode::Remote
        || meta.mode == crate::local::metapage::Mode::Promoting
    {
        crate::am::persist_engine_mode(index_relid, crate::local::metapage::Mode::Remote);
    }
    true
}

/// Dispatches a `'recover'` job row: reads one page of
/// `relid`'s remote namespace (`Client::list_vector_ids_page`, persisted
/// `payload.pagination_token` cursor — mirrors [`dispatch_backfill`]'s
/// `after_id`), fetches those ids' full vectors, and writes them back to the
/// local table via [`recover::try_recover_chunk`] — see that function's
/// rustdoc for the `synced_by = 'trigger'`/`'am'` write-back branches and the
/// NULL-guard/miss-detection soundness argument. Same non-blocking
/// `AccessShareLock` probe and dropped-relid purge as [`dispatch_backfill`]
/// (its rustdoc explains the worker↔DROP deadlock this avoids) — a
/// `'recover'` job can run for a long time (a whole namespace, one page per
/// round), so the table can just as easily disappear out from under it.
fn dispatch_recover(
    relid: pg_sys::Oid,
    job_id: i64,
    payload: Option<JsonB>,
    client: &Client,
) -> DispatchOutcome {
    if catalog::resolve_table_name(relid).is_none() {
        let purged = clear_relid(relid);
        warning!(
            "pg_pinecone: recover dispatch: relid {relid} no longer resolves to a table — purged \
             {purged} queued pinecone.outbox row(s) for it"
        );
        return DispatchOutcome::Purged;
    }

    let mut job = recover::RecoverJob::parse(payload);
    let row = registry::lookup(relid).unwrap_or_else(|| {
        panic!(
            "pg_pinecone: recover dispatch: no pinecone.tables row for relid {relid} — should be unreachable"
        )
    });
    let index_name = row
        .index_name
        .clone()
        .expect("pg_pinecone: remote/promoting row always has index_name");
    let namespace = row
        .namespace
        .clone()
        .expect("pg_pinecone: remote/promoting row always has namespace");

    // Same non-blocking probe as `dispatch_backfill` — see its rustdoc.
    const LOCKMODE: pg_sys::LOCKMODE = pg_sys::AccessShareLock as pg_sys::LOCKMODE;
    if !unsafe { pg_sys::ConditionalLockRelationOid(relid, LOCKMODE) } {
        return DispatchOutcome::LockUnavailable;
    }
    let heap_ptr = unsafe { pg_sys::RelationIdGetRelation(relid) };
    if heap_ptr.is_null() {
        panic!("pg_pinecone: recover dispatch: relid {relid} resolved moments ago but is now gone");
    }
    let tbl = unsafe { PgRelation::from_pg_owned(heap_ptr) };

    // A partitioned table is walked one namespace after the other: a round keeps going until its
    // budget is spent, each page's cursor persisted, so recovering 1,000 tenants is not 1,000 naps.
    let started = Instant::now();
    let chunk = loop {
        let chunk =
            recover::try_recover_chunk(&tbl, &row, client, job_id, &index_name, &namespace, &job);
        match chunk {
            Ok(recover::RecoverOutcome::InProgress(next))
                if row.partition.is_some() && started.elapsed() < docs_backfill::round_budget() =>
            {
                set_recover_payload(job_id, &next);
                job = next;
            }
            other => break other,
        }
    };
    match chunk {
        Ok(recover::RecoverOutcome::InProgress(next)) => {
            set_recover_payload(job_id, &next);
            DispatchOutcome::InProgress
        }
        Ok(recover::RecoverOutcome::Done(final_job)) => {
            let display = catalog::resolve_table_name(relid)
                .map(|(schema, table)| format!("{schema}.{table}"))
                .unwrap_or_else(|| format!("<dropped:{relid}>"));
            let func = if final_job.attach {
                "pinecone.attach_storage"
            } else {
                "pinecone.recover_from_remote"
            };
            let summary = format!(
                "{func}: {display} finished — recovered {}, skipped (already non-NULL locally) \
                 {}, ghosts {} (policy '{}'), misses {}, quarantined {}",
                final_job.recovered,
                final_job.skipped_non_null,
                final_job.ghosts_count,
                final_job.ghosts.as_str(),
                final_job.misses,
                final_job.quarantined
            );
            let summary = match row.partition {
                Some(_) => format!("{summary}, namespaces {}", final_job.namespaces),
                None => summary,
            };
            if final_job.quarantined > 0 {
                // A WARNING, not a NOTICE: quarantined vectors are recovered
                // BYTES, not recovered ROWS — they sit in
                // pinecone.recover_quarantine until an operator re-attaches them
                // by hand (docs/disaster-recovery.md), and nothing else will
                // ever do it for them.
                warning!(
                    "{summary} — the quarantined vector(s) could not be matched to a live row \
                     (concurrent ctid relocation); re-attach them manually from \
                     pinecone.recover_quarantine per docs/disaster-recovery.md"
                );
            } else {
                notice!("{summary}");
            }
            // `pinecone.attach_storage`'s completion hook —
            // committed in the SAME ambient transaction as this job row's
            // own deletion (`drain_step`'s `delete_rows` call, right after
            // `dispatch` returns), mirroring `dispatch_backfill`'s
            // `flip_promoting_to_remote` pattern exactly. Never-unavailable
            // throughout: `pinecone.topk`/reads stayed remote-only the whole
            // time this job ran (detached storage never served local), so
            // this flip only ever widens what's ALLOWED, never what's served.
            if final_job.attach {
                registry::mark_attached(relid);
                sync_trigger::remove_detached_id_guard(tbl);
                notice!(
                    "pinecone.attach_storage: {display} is now storage = 'full' — its vector/id \
                     columns are writable again"
                );
            }
            DispatchOutcome::Done
        }
        Err(e) => DispatchOutcome::Err(e),
    }
}

/// Persists a `'recover'` job row's whole payload (cursor + running counts) —
/// a full replace rather than [`set_backfill_after_id`]'s single-field
/// `jsonb_set`, since a `'recover'` chunk always touches several counters at
/// once.
fn set_recover_payload(job_id: i64, job: &recover::RecoverJob) {
    Spi::run_with_args(
        "UPDATE pinecone.outbox SET payload = $2 WHERE id = $1",
        &[job_id.into(), JsonB(job.to_json()).into()],
    )
    .expect("pg_pinecone: outbox recover payload update failed");
}

/// Whether `relid` already has a queued row with op `op` — `pg_pinecone.
/// recover_from_remote` uses this to refuse enqueuing a second concurrent
/// `'recover'` job on the same table (two jobs racing each other's cursors
/// would double-count, not double-recover).
pub(crate) fn has_queued_op(relid: pg_sys::Oid, op: &str) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT EXISTS (SELECT 1 FROM pinecone.outbox WHERE relid::oid = $1::oid AND op = $2)",
        &[relid.into(), op.into()],
    )
    .expect("pg_pinecone: outbox has_queued_op query failed")
    .expect("pg_pinecone: EXISTS unexpectedly NULL")
}

/// Enqueues a `'delete'` row for `ids` — `recover::try_recover_chunk`'s
/// `ghosts => 'delete'` path, cleaning up remote ids a `'trigger'`-path
/// recover chunk found with no matching local row. A no-op if `ids` is
/// empty. Reuses the exact payload shape `pinecone._sync_trigger`'s own
/// `DELETE`/no-longer-matching-vector branches build (a jsonb array of id
/// strings, or of `{id, p}` objects when the ids sit in the namespace of
/// partition value `partition`), so [`delete_ids_payload`] parses it
/// identically either way.
pub(crate) fn enqueue_delete_ids(relid: pg_sys::Oid, ids: &[String], partition: Option<&str>) {
    if ids.is_empty() {
        return;
    }
    let entry = |id: &String| match partition {
        Some(p) => json!({"id": id, "p": p}),
        None => Value::String(id.clone()),
    };
    let payload = Value::Array(ids.iter().map(entry).collect());
    Spi::run_with_args(
        "INSERT INTO pinecone.outbox (relid, op, payload) VALUES ($1::oid::regclass, 'delete', $2)",
        &[relid.into(), JsonB(payload).into()],
    )
    .expect("pg_pinecone: outbox enqueue ghost-delete failed");
}

/// Parses a `'backfill'` row's `payload` (`{"after_id": <text or null>}`,
/// built by [`enqueue_rebuild`] and advanced in place by
/// [`set_backfill_after_id`]) into the cursor
/// `promote::try_backfill_chunk` expects.
fn backfill_after_id(payload: Option<JsonB>) -> Option<String> {
    let Some(JsonB(Value::Object(obj))) = payload else {
        panic!("pg_pinecone: 'backfill' row has no jsonb object payload")
    };
    match obj.get("after_id") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(other) => panic!("pg_pinecone: 'backfill' row after_id is not a string: {other}"),
    }
}

/// Parses a `'rebuild_partition'` row's `payload.value`: the partition the job rebuilds.
fn rebuild_partition_value(payload: &Option<JsonB>) -> String {
    let Some(JsonB(Value::Object(obj))) = payload else {
        panic!("pg_pinecone: 'rebuild_partition' row has no jsonb object payload")
    };
    match obj.get("value") {
        Some(Value::String(s)) => s.clone(),
        other => panic!("pg_pinecone: 'rebuild_partition' row value is not a string: {other:?}"),
    }
}

/// Enqueues the rebuild of one partition: a `'rebuild_partition'` job that deletes the partition's
/// namespace, then sends the table's rows with `value` to it, by TID for an access-method index.
pub(crate) fn enqueue_partition_rebuild(relid: pg_sys::Oid, value: &str, tid: bool) {
    let payload = if tid {
        json!({"value": value, "id_kind": "tid", "after_id": null})
    } else {
        json!({"value": value, "after_id": null})
    };
    Spi::run_with_args(
        "INSERT INTO pinecone.outbox (relid, op, payload) \
         VALUES ($1::oid::regclass, 'rebuild_partition', $2)",
        &[relid.into(), JsonB(payload).into()],
    )
    .expect("pg_pinecone: outbox enqueue rebuild_partition failed");
}

/// Parses a `'backfill'` row's `payload.after_p`: the partition half of the cursor of a
/// partitioned table (absent before its first chunk, and for every other table). By reference,
/// like [`backfill_id_kind`], because [`backfill_after_id`] consumes the payload.
fn backfill_after_p(payload: &Option<JsonB>) -> Option<String> {
    let Some(JsonB(Value::Object(obj))) = payload else {
        panic!("pg_pinecone: 'backfill' row has no jsonb object payload")
    };
    match obj.get("after_p") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(other) => panic!("pg_pinecone: 'backfill' row after_p is not a string: {other}"),
    }
}

/// Parses a `'backfill'` row's optional `payload.id_kind` — absent for every
/// `pinecone.promote`-driven backfill (PK-keyed, `after_id` alone), present as
/// `"tid"` for one `am::ambuild`/REINDEX enqueues (`am.rs`'s `enqueue_backfill`).
/// Takes `payload` by reference (unlike [`backfill_after_id`]) so
/// [`dispatch_backfill`] can check this first and still pass `payload` on
/// afterward.
fn backfill_id_kind(payload: &Option<JsonB>) -> Option<String> {
    let Some(JsonB(Value::Object(obj))) = payload else {
        panic!("pg_pinecone: 'backfill' row has no jsonb object payload")
    };
    match obj.get("id_kind") {
        None => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(other) => panic!("pg_pinecone: 'backfill' row id_kind is not a string: {other}"),
    }
}

/// Persists a `'backfill'` job row's new `payload.after_id` cursor —
/// [`dispatch_backfill`], once a chunk push commits, so a worker restart (or
/// the next drain round) resumes from here rather than re-reading rows
/// already pushed.
fn set_backfill_after_id(job_id: i64, after_id: &str) {
    Spi::run_with_args(
        "UPDATE pinecone.outbox SET payload = jsonb_set(payload, '{after_id}', to_jsonb($2::text)) \
         WHERE id = $1",
        &[job_id.into(), after_id.into()],
    )
    .expect("pg_pinecone: outbox backfill after_id update failed");
}

/// Where the next chunk of a `'backfill'` job starts: after the row with this id, or, for a
/// partitioned table, after this `(partition, id)`.
struct BackfillCursor {
    partition: Option<String>,
    id: String,
}

impl BackfillCursor {
    fn id(id: String) -> Self {
        Self {
            partition: None,
            id,
        }
    }

    fn partitioned((partition, id): (String, String)) -> Self {
        Self {
            partition: Some(partition),
            id,
        }
    }
}

/// Persists the cursor [`dispatch_backfill`] reached, in the job row's payload.
fn set_backfill_cursor(job_id: i64, cursor: &BackfillCursor) {
    let Some(partition) = &cursor.partition else {
        return set_backfill_after_id(job_id, &cursor.id);
    };
    Spi::run_with_args(
        "UPDATE pinecone.outbox SET payload = jsonb_set(jsonb_set(payload, '{after_id}', \
         to_jsonb($2::text)), '{after_p}', to_jsonb($3::text)) WHERE id = $1",
        &[
            job_id.into(),
            cursor.id.as_str().into(),
            partition.as_str().into(),
        ],
    )
    .expect("pg_pinecone: outbox backfill cursor update failed");
}

/// Enqueues a `delete_all` job followed by a `backfill` job (`payload =
/// {"after_id": null}` — PK semantics), first deleting any of `relid`'s own
/// already-queued rows (coalescing: they're all
/// superseded by the fresh `delete_all`+`backfill` about to run).
///
/// Safe against a worker mid-batch: `fetch_batch`'s `FOR UPDATE` claim on a
/// row blocks this `DELETE` until that batch's transaction commits — but
/// that does *not* mean every claimed row is gone by then (a head-of-line
/// failure leaves the failing row, and everything after it in that same
/// batch, still queued rather than deleted — see `drain_step`). Once
/// unblocked, this `DELETE` simply removes whatever is still queued for
/// `relid` at that point, full stop — exactly the coalescing semantics
/// wanted: a row the worker round already finished is moot either way, and
/// one it *didn't* get to (including a halted head-of-line failure) is
/// superseded by the fresh rebuild about to run, the same as a row no
/// worker round ever touched. Callers that also hold `relid`'s own advisory
/// lock first (`promote::sync`) additionally
/// rule out a *concurrent* worker round altogether — `drain_step`'s
/// `Worker` path takes that same lock before it ever reaches `fetch_batch`.
///
/// Used by both `pinecone.promote`'s async path (a fresh queue, so the
/// `DELETE` is a no-op there) and `pinecone.sync` on an async table (where
/// it's the coalescing itself).
///
/// Defense-in-depth `require_not_detached` check: both
/// current callers already gate on `storage != 'detached'` before ever
/// reaching here (`promote::sync`'s own check; `promote::promote`'s async
/// path can only ever reach a fresh/`'local'` row, never a detached one — see
/// `track::track`'s "not local" guard), so this should be unreachable in
/// practice — but a `delete_all` + `backfill`-that-skips-NULLs enqueued here
/// is *precisely* the total-remote-wipe footgun this whole guard exists to
/// prevent, so it's re-checked at the one place both callers converge rather
/// than trusted to stay that way forever.
pub(crate) fn enqueue_rebuild(relid: pg_sys::Oid) {
    if let Some(row) = registry::lookup(relid) {
        let display = catalog::resolve_table_name(relid)
            .map(|(schema, table)| format!("{schema}.{table}"))
            .unwrap_or_else(|| format!("<dropped:{relid}>"));
        registry::require_not_detached(&row, "pg_pinecone: rebuild", &display);
    }
    Spi::run_with_args(
        "DELETE FROM pinecone.outbox WHERE relid::oid = $1::oid",
        &[relid.into()],
    )
    .expect("pg_pinecone: outbox coalesce-delete failed");
    Spi::run_with_args(
        "INSERT INTO pinecone.outbox (relid, op, payload) VALUES ($1::oid::regclass, 'delete_all', NULL)",
        &[relid.into()],
    )
    .expect("pg_pinecone: outbox enqueue delete_all failed");
    Spi::run_with_args(
        "INSERT INTO pinecone.outbox (relid, op, payload) VALUES \
         ($1::oid::regclass, 'backfill', jsonb_build_object('after_id', NULL::text))",
        &[relid.into()],
    )
    .expect("pg_pinecone: outbox enqueue backfill failed");
}

/// Parses a `'delete'` row's payload (a jsonb array of ids, as built by
/// `_sync_trigger`'s plpgsql body: bare strings, or `{id, p}` objects for a partitioned table)
/// into the entries `sync_trigger::try_delete_ids` expects.
fn delete_ids_payload(payload: Option<JsonB>) -> Vec<DeleteEntry> {
    let Some(JsonB(Value::Array(items))) = payload else {
        panic!("pinecone._drain_relid: 'delete' row has no jsonb array payload")
    };
    delete_entries(items)
}

/// [`delete_ids_payload`] without the Postgres types: pure, so a plain unit test can run it.
fn delete_entries(items: Vec<Value>) -> Vec<DeleteEntry> {
    items
        .into_iter()
        .map(|v| match &v {
            Value::String(id) => DeleteEntry {
                id: id.clone(),
                p: None,
            },
            Value::Object(entry) => {
                let text = |key: &str| entry.get(key).and_then(Value::as_str).map(str::to_string);
                let (Some(id), Some(p)) = (text("id"), text("p")) else {
                    panic!("pinecone._drain_relid: delete entry needs a string id and p: {v}")
                };
                DeleteEntry { id, p: Some(p) }
            }
            _ => panic!("pinecone._drain_relid: non-string id in delete payload: {v}"),
        })
        .collect()
}

/// Claims one batch `FOR UPDATE`. Runs through `connect_mut`/`update` (SPI
/// `read_only = false`), NOT `select`: a `FOR UPDATE` row claim is a
/// locking (mutating-class) statement Postgres rejects under read-only SPI,
/// and pgrx only passes `read_only = false` to `select` once the transaction
/// already has an xid — true for every trigger-inline drain (the trigger's
/// own enqueue wrote first) and for worker rounds (`lock_relid`'s `update`
/// runs first), but NOT for a bare `SELECT pinecone._drain_relid(...)` as a
/// fresh transaction's first statement, which errored with "SELECT FOR
/// UPDATE is not allowed in a non-volatile function" (caught by crash-drill
/// leg D9's foreground drain driver).
///
/// `live_only` leaves out the `'rebuild_partition'` rows: only the worker runs them, after the
/// live rows ([`rebuild_partitions`]), so they never hold the live rows behind them back.
fn fetch_batch(relid: pg_sys::Oid, live_only: bool) -> Vec<OutboxRow> {
    let sql = if live_only {
        "SELECT id, op, payload FROM pinecone.outbox WHERE relid::oid = $1::oid \
         AND op <> 'rebuild_partition' ORDER BY id LIMIT $2 FOR UPDATE"
    } else {
        "SELECT id, op, payload FROM pinecone.outbox WHERE relid::oid = $1::oid \
         ORDER BY id LIMIT $2 FOR UPDATE"
    };
    Spi::connect_mut(|client| {
        client
            .update(sql, None, &[relid.into(), DRAIN_BATCH_ROWS.into()])
            .expect("pg_pinecone: outbox drain fetch failed")
            .map(|row| outbox_row(&row))
            .collect()
    })
}

fn outbox_row(row: &pgrx::spi::SpiHeapTupleData<'_>) -> OutboxRow {
    OutboxRow {
        id: row
            .get_by_name::<i64, _>("id")
            .expect("pg_pinecone: outbox row id query failed")
            .expect("pg_pinecone: outbox row id unexpectedly NULL"),
        op: row
            .get_by_name::<String, _>("op")
            .expect("pg_pinecone: outbox row op query failed")
            .expect("pg_pinecone: outbox row op unexpectedly NULL"),
        payload: row
            .get_by_name::<JsonB, _>("payload")
            .expect("pg_pinecone: outbox row payload query failed"),
    }
}

/// Claims the first queued job of `relid` after row `after`: a `'backfill'`, `'recover'` or
/// `'rebuild_partition'` row, whichever is oldest, so that a rebuild queued behind another job
/// waits for it, as it does in the queue.
fn next_job(relid: pg_sys::Oid, after: i64) -> Option<OutboxRow> {
    Spi::connect_mut(|client| {
        client
            .update(
                "SELECT id, op, payload FROM pinecone.outbox WHERE relid::oid = $1::oid \
                 AND id > $2 AND op IN ('backfill', 'recover', 'rebuild_partition') \
                 ORDER BY id LIMIT 1 FOR UPDATE",
                None,
                &[relid.into(), after.into()],
            )
            .expect("pg_pinecone: outbox job claim failed")
            .next()
            .map(|row| outbox_row(&row))
    })
}

fn job_payload(job_id: i64) -> Option<JsonB> {
    Spi::get_one_with_args::<JsonB>(
        "SELECT payload FROM pinecone.outbox WHERE id = $1",
        &[job_id.into()],
    )
    .expect("pg_pinecone: outbox job payload read failed")
}

/// Whether the worker was asked to stop. pgrx's SIGTERM handler raises `ShutdownRequestPending`
/// next to its own flag; reading it leaves that flag for the `wait_latch` that ends the worker
/// (`BackgroundWorker::sigterm_received` would consume it, and panics outside a worker).
pub(crate) fn shutdown_requested() -> bool {
    unsafe { (&raw const pg_sys::ShutdownRequestPending).read_volatile() != 0 }
}

/// What a worker round may spend on backfill chunks: `pinecone.backfill_round_ms`, counted from
/// the round's first chunk and shared by every table's step, so that each step still runs at
/// least one chunk and a job always advances. A shutdown request spends it at once.
struct ChunkBudget {
    started: Option<Instant>,
    budget: Duration,
}

impl ChunkBudget {
    fn new() -> Self {
        Self {
            started: None,
            budget: docs_backfill::round_budget(),
        }
    }

    /// Starts the clock with the round's first chunk; a no-op after that.
    fn begin(&mut self) {
        self.started.get_or_insert_with(Instant::now);
    }

    /// Whether no further chunk may start.
    fn spent(&self) -> bool {
        shutdown_requested() || expired(self.started, self.budget, Instant::now())
    }
}

/// Whether a clock started at `started` (never, if `None`) has run `budget` out at `now`.
fn expired(started: Option<Instant>, budget: Duration, now: Instant) -> bool {
    started.is_some_and(|started| now.saturating_duration_since(started) >= budget)
}

/// Runs the chunks of backfill job `job_id` one after the other until it is done, a chunk fails,
/// the budget is spent or the worker is asked to stop (`InProgress`, the job stays queued). Every
/// chunk persists its cursor in the step's transaction, so what the budget cuts off resumes next
/// round, and a chunk that fails leaves the cursor of the one before it. A stop request ends
/// the loop after the chunk in flight: pgrx's SIGTERM handler sets a flag, not `ProcDiePending`,
/// so no interrupt check inside SPI would.
fn run_chunks(
    relid: pg_sys::Oid,
    job_id: i64,
    payload: Option<JsonB>,
    client: &Client,
    gate: &mut Gate,
    scoped: bool,
    clock: &mut ChunkBudget,
) -> DispatchOutcome {
    clock.begin();
    if shutdown_requested() {
        return DispatchOutcome::InProgress;
    }
    let mut payload = payload;
    loop {
        match dispatch_backfill(relid, job_id, payload.take(), client, gate, scoped) {
            DispatchOutcome::InProgress if !clock.spent() => {
                pgrx::check_for_interrupts!();
                payload = job_payload(job_id);
            }
            outcome => return outcome,
        }
    }
}

struct Rebuilt {
    done: Vec<i64>,
    failure: Option<(i64, ClientError)>,
    chunks_left: bool,
}

/// The rest of a worker step, once the live rows are out: the queued `'rebuild_partition'` jobs,
/// oldest first, chunk after chunk until the step's budget ([`ChunkBudget`]) is spent (at least
/// one chunk), across as many jobs as fit. Each chunk persists its cursor, so what the budget cuts
/// off resumes next round. Safe against live writes because the heap is the source: a chunk
/// reads it after this step's live rows went out, and a write that lands later is sent by the
/// next step, after the chunk.
fn rebuild_partitions(
    relid: pg_sys::Oid,
    client: &Client,
    gate: &mut Gate,
    first: Option<OutboxRow>,
    clock: &mut ChunkBudget,
) -> Rebuilt {
    let mut rebuilt = Rebuilt {
        done: Vec::new(),
        failure: None,
        chunks_left: false,
    };
    let mut after = 0;
    let mut next = first;
    while let Some(job) = next.take().or_else(|| next_job(relid, after)) {
        if job.op != "rebuild_partition" {
            break;
        }
        after = job.id;
        match run_chunks(relid, job.id, job.payload, client, gate, true, clock) {
            DispatchOutcome::Done => rebuilt.done.push(job.id),
            DispatchOutcome::InProgress => {
                rebuilt.chunks_left = true;
                return rebuilt;
            }
            DispatchOutcome::Purged | DispatchOutcome::LockUnavailable => return rebuilt,
            DispatchOutcome::Err(e) => {
                rebuilt.failure = Some((job.id, e));
                return rebuilt;
            }
        }
        if clock.spent() {
            rebuilt.chunks_left = next_job(relid, after).is_some();
            break;
        }
    }
    rebuilt
}

fn delete_rows(ids: &[i64]) {
    Spi::run_with_args(
        "DELETE FROM pinecone.outbox WHERE id = ANY($1)",
        &[ids.to_vec().into()],
    )
    .expect("pg_pinecone: outbox delete processed rows failed");
}

/// Deletes every queued `pinecone.outbox` row for `relid`, returning how many
/// were removed. Called by `pinecone.demote` (see its rustdoc in
/// `promote.rs`) so a demoted table's stale backlog never outlives the
/// demote to replay over a later re-promote's fresh backfill; also by
/// [`dispatch_backfill`] to purge a relid whose table was dropped/renamed
/// out from under a still-queued job (the returned count is what that
/// call's `WARNING` names).
pub(crate) fn clear_relid(relid: pg_sys::Oid) -> i64 {
    Spi::get_one_with_args::<i64>(
        "WITH d AS (DELETE FROM pinecone.outbox WHERE relid::oid = $1::oid RETURNING 1) \
         SELECT count(*) FROM d",
        &[relid.into()],
    )
    .expect("pg_pinecone: outbox clear failed")
    .expect("pg_pinecone: outbox clear count(*) unexpectedly NULL")
}

/// Records a transient-class dispatch failure (or, for
/// [`DrainCredentials::Worker`], `drain_step`'s "no stored key yet"
/// pre-flight check) on outbox row `id`: sets `last_error` to `message` and
/// bumps `next_retry_at` to a near future timestamp, but never touches
/// `attempts` — `pinecone.outbox_max_attempts` only ever counts
/// permanent-class failures (its own GUC doc; see [`record_permanent_failure`]
/// for those), so a relid that's merely waiting out a blip (or a
/// not-yet-configured key) must never edge toward parking just because it
/// keeps retrying. `next_retry_at` is set [`BACKOFF_MIN`] ahead — a
/// deliberately conservative floor for `status()`'s `queue_state`
/// (`'backoff'` while it's still in the future): the *authoritative*
/// schedule is `Scheduler`'s own in-memory, per-relid exponential backoff,
/// which may run far longer than this floor and never reads this column
/// back — it's observability only. `message` is never the raw
/// `ClientError` payload's API key (see the
/// `outbox_drain_error_never_leaks_api_key`/`drain_step_transient_error_*`
/// tests in `lib.rs` — none of that type's `Display` impls carry it).
fn record_transient_failure(id: i64, message: &str) {
    Spi::run_with_args(
        &format!(
            "UPDATE pinecone.outbox SET last_error = $2, \
             next_retry_at = now() + interval '{} seconds' WHERE id = $1",
            BACKOFF_MIN.as_secs()
        ),
        &[id.into(), message.into()],
    )
    .expect("pg_pinecone: outbox transient failure record update failed");
}

/// Records a permanent-class dispatch failure on outbox row `id`:
/// increments `attempts` and sets `last_error`, returning the new
/// `attempts` count so [`drain_step`] can compare it against
/// `pinecone.outbox_max_attempts` without a second round trip. Leaves
/// `next_retry_at` untouched — the caller sets ([`set_next_retry_at`]) or
/// clears ([`clear_next_retry_at`]) it afterward, once it knows whether this
/// failure just parked the relid or not.
fn record_permanent_failure(id: i64, err: &ClientError) -> i32 {
    Spi::get_one_with_args::<i32>(
        "UPDATE pinecone.outbox SET attempts = attempts + 1, last_error = $2 \
         WHERE id = $1 RETURNING attempts",
        &[id.into(), backend::failure_text(err).into()],
    )
    .expect("pg_pinecone: outbox failure record update failed")
    .expect("pg_pinecone: outbox failure record RETURNING attempts unexpectedly NULL")
}

/// Sets outbox row `id`'s `next_retry_at` — see
/// [`record_transient_failure`] for why [`BACKOFF_MIN`] ahead and why it's
/// observability-only. Paired with a permanent-class failure that backs off
/// without parking; the transient path folds this into its own single
/// update instead ([`record_transient_failure`]).
fn set_next_retry_at(id: i64) {
    Spi::run_with_args(
        &format!(
            "UPDATE pinecone.outbox SET next_retry_at = now() + interval '{} seconds' \
             WHERE id = $1",
            BACKOFF_MIN.as_secs()
        ),
        &[id.into()],
    )
    .expect("pg_pinecone: outbox next_retry_at update failed");
}

/// Clears [`set_next_retry_at`]/[`record_transient_failure`]'s column:
/// `drain_step` calls this once a permanent-class failure actually parks the
/// relid (`'parked'` beats `'backoff'` in `status()`'s `queue_state` state
/// machine regardless, but a stale future timestamp should never linger past
/// that point), and `outbox_retry` clears it too (alongside resetting
/// `attempts`/`last_error`) so a freshly unparked queue never reads as
/// `'backoff'` before its first post-unpark attempt.
fn clear_next_retry_at(id: i64) {
    Spi::run_with_args(
        "UPDATE pinecone.outbox SET next_retry_at = NULL WHERE id = $1",
        &[id.into()],
    )
    .expect("pg_pinecone: outbox next_retry_at clear failed");
}

/// Whether `current_user` — the actual caller, not `_outbox_stats`'s own
/// `SECURITY DEFINER` owner — may `EXECUTE pinecone._outbox_stats`. Used by
/// `pinecone.status` (`SECURITY INVOKER`) to decide whether to call it at all:
/// a plain `has_table_privilege`/`has_function_privilege` check placed
/// *inside* a `SECURITY DEFINER` function would evaluate against the
/// definer, not the caller, so the gate has to live on the caller's side of
/// the call, here.
pub(crate) fn can_call_outbox_stats() -> bool {
    Spi::get_one::<bool>(
        "SELECT pg_catalog.has_function_privilege(current_user, \
         'pinecone._outbox_stats(oid)', 'EXECUTE')",
    )
    .expect("pg_pinecone: has_function_privilege query failed")
    .expect("pg_pinecone: has_function_privilege unexpectedly NULL")
}

/// `pinecone.outbox` backlog count for `relid` alone — gated the same way
/// [`can_call_outbox_stats`] gates `status::outbox_stats`'s own read of
/// `pinecone._outbox_stats`, but without paying for the other three columns
/// that call also reads. `None` for a caller with no `EXECUTE` grant on the
/// `SECURITY DEFINER` function (never an error) — used by
/// `backup_readiness`'s SYNC check to explain a live/local vector-count
/// mismatch as "in-flight outbox backlog" rather than a genuine drift.
pub(crate) fn backlog_for(relid: pg_sys::Oid) -> Option<i64> {
    if !can_call_outbox_stats() {
        return None;
    }
    Spi::get_one_with_args::<i64>(
        "SELECT backlog FROM pinecone._outbox_stats($1::oid)",
        &[relid.into()],
    )
    .expect("pg_pinecone: backup_readiness backlog_for query failed")
}

/// First key of the (future) background worker's per-database session
/// advisory lock (`pg_advisory_lock(WORKER_LOCK_MAGIC, <this database's
/// oid>)`) — the single-worker-per-database guard. Paired
/// with the database oid (rather than used alone) so workers in different
/// databases never contend for the same cluster-wide key.
pub(crate) const WORKER_LOCK_MAGIC: i32 = 0x5056574B; // b"PVWK"

/// Whether *some* backend currently holds the worker's per-database session
/// advisory lock — "is a per-database worker running *right now*", used by
/// `worker::try_take_singleton_lock`'s probe-before-acquire shape (see its
/// rustdoc). Deliberately *not* what `status()`'s `queue_state`, `flush()`,
/// or `promote()`'s async warning read: a
/// per-database worker legitimately idle-exits after `IDLE_EXIT_ROUNDS` of
/// nothing to drain and gets respawned on demand (`worker.rs`), so this flips
/// far too often to be a good "will this backlog ever be drained" signal —
/// see [`launcher_lock_held`] for the one those use instead.
pub(crate) fn worker_lock_held() -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT EXISTS (
            SELECT 1 FROM pg_catalog.pg_locks
            -- objsubid = 2 for the two-int4-key advisory lock form we use
            -- here (`pg_advisory_lock(int, int)`) — it encodes which lock-tag
            -- *shape* was taken (2 keys vs. a single bigint, which is
            -- objsubid = 1), not session- vs. transaction-scope, so it isn't
            -- (and doesn't need to be) restricted to session-level locks
            -- specifically. `lock_relid`'s own per-relid xact lock
            -- (`advisory_lock_key`) is the single-bigint form instead
            -- (objsubid = 1) — its high 32 bits (`pg_locks.classid`) are a
            -- database oid, the same value that appears as this lock's own
            -- `objid` here, but the two can never collide: objsubid alone
            -- already tells the shapes apart, regardless of whether the
            -- numeric key values happen to coincide.
            WHERE locktype = 'advisory' AND granted AND objsubid = 2
              AND classid = $1
              AND objid = (SELECT oid::int4 FROM pg_catalog.pg_database \
                           WHERE datname = pg_catalog.current_database())
         )",
        &[WORKER_LOCK_MAGIC.into()],
    )
    .expect("pg_pinecone: worker lock probe query failed")
    .expect("pg_pinecone: EXISTS unexpectedly NULL")
}

/// Like [`worker_lock_held`], but for an arbitrary database `dbid` rather
/// than `current_database()` — the launcher needs this because it connects
/// via `BackgroundWorker::connect_worker_to_spi(None, None)` (shared catalogs
/// only), so it has no
/// `current_database()` of its own to probe before deciding whether to spawn
/// a worker for each discovered database.
pub(crate) fn worker_lock_held_for(dbid: pg_sys::Oid) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT EXISTS (
            SELECT 1 FROM pg_catalog.pg_locks
            WHERE locktype = 'advisory' AND granted AND objsubid = 2
              AND classid = $1 AND objid = $2::int4
         )",
        &[WORKER_LOCK_MAGIC.into(), dbid.into()],
    )
    .expect("pg_pinecone: worker lock probe (by dbid) query failed")
    .expect("pg_pinecone: EXISTS unexpectedly NULL")
}

/// Cluster-wide, single-instance advisory-lock key for the launcher itself
/// — distinct from [`WORKER_LOCK_MAGIC`] (one per
/// database): there is at most one launcher process in the whole cluster, so
/// this one always pairs with a fixed second key (`0`) rather than a database
/// oid. `worker::pg_pinecone_launcher_main` takes (and releases) it in step with
/// `pinecone.worker_enabled`, not unconditionally at process start — see its
/// rustdoc — so [`launcher_lock_held`] reads `true` only while the launcher
/// is actually preloaded *and* enabled, matching the
/// "preload absent or launcher lock free" definition of `'NO
/// WORKER'` exactly (a `worker_enabled = off` cluster behaves the same as no
/// preload at all: nothing will ever drain an async backlog).
pub(crate) const LAUNCHER_LOCK_MAGIC: i32 = 0x5056_4C43; // b"PVLC"

/// Whether *some* backend currently holds the launcher's own singleton
/// advisory lock — see [`LAUNCHER_LOCK_MAGIC`]'s rustdoc. Unlike
/// [`worker_lock_held`] (which flips constantly as a per-database worker
/// legitimately idle-exits and gets respawned by design), this is the "will
/// pg_pinecone's background-worker machinery ever drain this backlog" signal
/// `status()`'s `queue_state` and `flush()` use.
pub(crate) fn launcher_lock_held() -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT EXISTS (
            SELECT 1 FROM pg_catalog.pg_locks
            WHERE locktype = 'advisory' AND granted AND objsubid = 2
              AND classid = $1 AND objid = 0
         )",
        &[LAUNCHER_LOCK_MAGIC.into()],
    )
    .expect("pg_pinecone: launcher lock probe query failed")
    .expect("pg_pinecone: EXISTS unexpectedly NULL")
}

/// Distinct relids with at least one queued row whose registry row isn't
/// parked — the round-robin candidate set for [`Scheduler::round`]. A
/// relid whose registry row no longer exists (should be unreachable in
/// practice — `pinecone.demote` clears its outbox rows) is silently excluded
/// rather than erroring the whole round out.
fn active_relids() -> Vec<pg_sys::Oid> {
    Spi::connect(|client| {
        client
            .select(
                // LEFT join: a queued relid
                // whose registration vanished underneath it (racing DROP /
                // demote completion) must still be VISITED — the dispatch
                // layer's own missing-registration handling (warn + purge,
                // `sync_trigger::remote_target` / `dispatch_backfill`) is
                // what converges it; an inner join made such a queue
                // invisible instead: never dispatched, never purged, an
                // eternal orphan backlog.
                "SELECT DISTINCT o.relid::oid AS relid FROM pinecone.outbox o \
                 LEFT JOIN pinecone.tables t ON t.relid = o.relid \
                 WHERE t.relid IS NULL OR NOT t.queue_parked ORDER BY 1",
                None,
                &[],
            )
            .expect("pg_pinecone: outbox active-relid scan failed")
            .map(|row| {
                row.get_by_name::<pg_sys::Oid, _>("relid")
                    .expect("pg_pinecone: active-relid query failed")
                    .expect("pg_pinecone: active relid unexpectedly NULL")
            })
            .collect()
    })
}

/// [`Scheduler`]'s per-relid backoff floor/ceiling:
/// 1s on a relid's first consecutive `Backoff`, doubling on every further one
/// while it keeps failing, capped at 60s.
const BACKOFF_MIN: Duration = Duration::from_secs(1);
const BACKOFF_MAX: Duration = Duration::from_secs(60);

/// A relid's in-memory backoff state inside a [`Scheduler`]: skip drains of
/// this relid until `until`; `next_delay` is the window the *next*
/// consecutive `Backoff` outcome will set.
struct RelidBackoff {
    until: Instant,
    next_delay: Duration,
}

/// Per-database round-robin drain scheduler — the background worker's
/// (`worker.rs`) core loop, factored out so it's testable without one.
/// One [`Self::round`] call drains at most one
/// batch from every relid that currently has queued, unparked rows and isn't
/// inside its own backoff window — so one relid's failure or oversized
/// backlog never delays another's turn (fairness).
/// Always drains via [`DrainCredentials::Worker`]: this is the worker's own
/// loop, never the writer-session inline path (that's [`drain_relid`]).
pub(crate) struct Scheduler {
    backoff: HashMap<pg_sys::Oid, RelidBackoff>,
    /// Which table's step leads the next round, so that one backlog does not take the round's
    /// chunk budget every time.
    turn: usize,
}

/// What one worker round found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Round {
    /// An unparked relid had queued rows, also one that its backoff window kept out of the round.
    pub(crate) active: bool,
    /// A backfill, rebuild or recover job stopped its step with chunks left to run: the next round
    /// has work, so the worker does not nap.
    pub(crate) chunks_left: bool,
}

impl Scheduler {
    pub(crate) fn new() -> Self {
        Self {
            backoff: HashMap::new(),
            turn: 0,
        }
    }

    /// Runs one round: for every currently-eligible relid, exactly one
    /// [`drain_step`] call (never more — "process each relid at most one
    /// batch per round", so a single oversized backlog can't starve the
    /// others). The steps share one [`ChunkBudget`], `pinecone.backfill_round_ms`: the
    /// table that leads runs backfill chunks until it is spent, every table behind it its one
    /// chunk, and the lead moves on by one table each round. A stop request ends the round after
    /// the step in progress. `active` is whether *any* unparked relid had queued rows this
    /// round at all — including ones skipped for still being inside their
    /// own backoff window. That's the signal `worker.rs`'s idle-round
    /// counter needs: a round where every active relid
    /// happened to be backing off processes nothing, but it's still a
    /// *busy* round with a real backlog, not an idle one — the worker must
    /// never exit out from under it. `chunks_left` is what keeps the worker
    /// from napping.
    pub(crate) fn round(&mut self) -> Round {
        let now = Instant::now();
        let active = active_relids();
        let any_active = !active.is_empty();
        let mut eligible: Vec<pg_sys::Oid> = active
            .into_iter()
            .filter(|relid| self.backoff.get(relid).is_none_or(|b| now >= b.until))
            .collect();
        if !eligible.is_empty() {
            let lead = self.turn % eligible.len();
            eligible.rotate_left(lead);
            self.turn = self.turn.wrapping_add(1);
        }
        let mut clock = ChunkBudget::new();
        let mut chunks_left = false;
        for relid in eligible {
            if shutdown_requested() {
                break;
            }
            let step = drain_step_within(relid, DrainCredentials::Worker, &mut clock);
            self.record_outcome(relid, &step.outcome, now);
            chunks_left |= step.chunks_left;
        }
        Round {
            active: any_active,
            chunks_left,
        }
    }

    /// [`Self::round`]'s `active`, for the tests that only drive rounds.
    #[cfg(feature = "pg_test")]
    pub(crate) fn drain_round(&mut self) -> bool {
        self.round().active
    }

    fn record_outcome(&mut self, relid: pg_sys::Oid, outcome: &DrainOutcome, now: Instant) {
        match outcome {
            // A drained (or now-parked, hence registry-skipped) relid starts
            // fresh at the backoff floor next time it fails.
            DrainOutcome::Processed(_) | DrainOutcome::Parked(_) => {
                self.backoff.remove(&relid);
            }
            DrainOutcome::Backoff(_) => {
                let entry = self.backoff.entry(relid).or_insert(RelidBackoff {
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

/// Implements `pinecone.outbox_retry`: unparks `relid` and resets the
/// `attempts`/`last_error`/`next_retry_at` of the row that failed — the poison-escape
/// operator hatch. Not necessarily the oldest row: a
/// rebuild job may sit ahead of the live row that failed. Clearing `next_retry_at` too so a
/// freshly unparked queue never reads back as `'backoff'` before its first
/// post-unpark drain attempt. Owner-gated by `pinecone.tables`/
/// `pinecone.outbox`'s own ACL (neither grants PUBLIC anything), so no extra
/// privilege check is needed here.
pub(crate) fn retry(relid: pg_sys::Oid, display: &str) {
    require_tracked(relid, display, "pinecone.outbox_retry");
    Spi::run_with_args(
        "UPDATE pinecone.outbox SET attempts = 0, last_error = NULL, next_retry_at = NULL \
         WHERE relid::oid = $1::oid \
           AND (attempts <> 0 OR last_error IS NOT NULL OR next_retry_at IS NOT NULL)",
        &[relid.into()],
    )
    .expect("pg_pinecone: outbox_retry reset failed");
    registry::unpark(relid);
}

/// Implements `pinecone.outbox_discard`: deletes every queued row for `relid`
/// with `id <= through_id` — the other poison-escape hatch. Deliberately
/// consistency-affecting (those writes never reach the remote index), so it
/// always warns the caller to run `pinecone.sync(tbl)` afterward; it does
/// *not* unpark `relid` itself (`pinecone.outbox_retry` is the separate,
/// explicit unpark step).
pub(crate) fn discard(relid: pg_sys::Oid, display: &str, through_id: i64) {
    require_tracked(relid, display, "pinecone.outbox_discard");
    let deleted = Spi::get_one_with_args::<i64>(
        "WITH d AS ( \
             DELETE FROM pinecone.outbox WHERE relid::oid = $1::oid AND id <= $2 RETURNING 1 \
         ) SELECT count(*) FROM d",
        &[relid.into(), through_id.into()],
    )
    .expect("pg_pinecone: outbox_discard delete failed")
    .expect("pg_pinecone: outbox_discard count(*) unexpectedly NULL");
    warning!(
        "pinecone.outbox_discard: dropped {deleted} queued row(s) for {display} up through id \
         {through_id} — those writes will never reach the remote index; run pinecone.sync(...) \
         to repair consistency"
    );
}

fn require_tracked(relid: pg_sys::Oid, display: &str, func: &str) {
    if registry::lookup(relid).is_none() {
        error!("{func}: {display} is not tracked");
    }
}

/// [`flush`]'s poll interval between backlog/liveness checks.
const FLUSH_POLL_INTERVAL: Duration = Duration::from_millis(200);

/// Implements `pinecone.flush`: blocks (an interruptible poll, so
/// `pg_cancel_backend`/`statement_timeout` still work) until `relid`'s
/// `pinecone.outbox` backlog is empty, returning `true`; on timeout, returns
/// `false` after a `WARNING`.
///
/// Two fast-fail cases, both checked upfront and re-checked every poll (so a
/// launcher dying, or the relid parking, mid-wait doesn't strand the caller
/// for the rest of the timeout), and both purely `'async'`-table concerns —
/// a `sync_mode = 'sync'` table's writes always drain inline, so its backlog
/// is already empty by the time this could even observe it:
///
/// - [`launcher_lock_held`] is false: no launcher is alive-and-enabled in
///   this cluster, so nothing will *ever* drain the backlog — waiting out
///   the timeout would be for nothing. Deliberately the LAUNCHER's liveness
///   lock, not the per-database worker's ([`worker_lock_held`]): a
///   per-database worker legitimately idle-exits between backlogs
///   (`worker.rs`'s `IDLE_EXIT_ROUNDS`) and gets respawned on demand — its
///   lock being momentarily free is the *normal* state this poll is
///   designed to wait through, not a liveness failure.
/// - The relid's queue is parked (`pinecone.tables.queue_parked`): a live
///   launcher/worker deliberately *skips* a parked relid until an operator
///   runs `pinecone.outbox_retry`, so this backlog won't drain either, no
///   matter how healthy the worker machinery is.
///
/// Owner-ish gated the same way `outbox_retry`/`outbox_discard` are: neither
/// `pinecone.tables` nor `pinecone.outbox` grants `PUBLIC` anything, so
/// [`require_tracked`]'s plain existence check is the only gate needed here.
pub(crate) fn flush(relid: pg_sys::Oid, display: &str, timeout_ms: i32) -> bool {
    require_tracked(relid, display, "pinecone.flush");
    if backlog(relid) == 0 {
        return true;
    }

    let deadline = Instant::now() + Duration::from_millis(timeout_ms.max(0) as u64);
    loop {
        if queue_parked(relid) {
            warning!(
                "pinecone.flush: {display}'s outbox queue is parked — nothing will drain it until \
                 pinecone.outbox_retry(...) unparks it (see pinecone.status())"
            );
            return false;
        }
        if !launcher_lock_held() {
            warning!(
                "pinecone.flush: {display} has a queued backlog but no pg_pinecone launcher is \
                 running in this cluster (preload absent or pinecone.worker_enabled off) — \
                 nothing will drain it (see pinecone.status())"
            );
            return false;
        }
        let now = Instant::now();
        if now >= deadline {
            warning!(
                "pinecone.flush: {display} still has a queued backlog after {}ms — timed out",
                timeout_ms.max(0)
            );
            return false;
        }
        backend::interruptible_sleep(FLUSH_POLL_INTERVAL.min(deadline - now));
        if backlog(relid) == 0 {
            return true;
        }
    }
}

/// Whether `relid`'s registry row is currently parked — [`flush`]'s own
/// re-checked-every-poll read (a fresh query, not a row fetched once up
/// front: a concurrent worker round can park the relid mid-wait).
fn queue_parked(relid: pg_sys::Oid) -> bool {
    registry::lookup(relid).is_some_and(|row| row.queue_parked)
}

/// Unconditional `pinecone.outbox` row count for `relid` — unlike
/// [`backlog_for`] (gated on the CALLER's `EXECUTE` privilege on the
/// `SECURITY DEFINER` `pinecone._outbox_stats`, `status()`'s own visibility
/// rule), this is an internal count any code already running inside the
/// extension can rely on regardless of the invoking role's grants —
/// `detach::detach_storage`'s outbox-drained gate needs a real number even
/// when called by a table owner with no explicit grant on that internal
/// function.
pub(crate) fn backlog(relid: pg_sys::Oid) -> i64 {
    Spi::get_one_with_args::<i64>(
        "SELECT count(*) FROM pinecone.outbox WHERE relid::oid = $1::oid",
        &[relid.into()],
    )
    .expect("pg_pinecone: flush backlog count query failed")
    .expect("pg_pinecone: count(*) unexpectedly NULL")
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn array(value: Value) -> Vec<Value> {
        value.as_array().cloned().expect("a JSON array")
    }

    fn folded(rows: Vec<Value>) -> Vec<Value> {
        let mut combined = Vec::new();
        let mut index = HashMap::new();
        for rows in rows {
            fold_upsert_items(array(rows), &mut combined, &mut index);
        }
        combined
    }

    #[test]
    fn a_chunk_budget_runs_out_only_once_started_and_spent() {
        let (started, budget) = (Instant::now(), Duration::from_secs(1));
        let at = |ms| started + Duration::from_millis(ms);

        assert!(!expired(None, budget, at(5_000)), "no chunk run yet");
        assert!(!expired(Some(started), budget, at(999)));
        assert!(expired(Some(started), budget, at(1_000)));
        assert!(expired(Some(started), budget, at(1_001)));
        assert!(
            !expired(Some(at(10)), budget, started),
            "a clock behind the start"
        );
    }

    #[test]
    fn a_later_upsert_of_an_id_replaces_the_earlier_one_in_place() {
        let combined = folded(vec![
            json!([{"id": "1", "values": [1.0]}, {"id": "2", "values": [1.0]}]),
            json!([{"id": "1", "values": [2.0]}]),
        ]);

        assert_eq!(
            combined,
            [
                json!({"id": "1", "values": [2.0]}),
                json!({"id": "2", "values": [1.0]})
            ]
        );
    }

    #[test]
    fn upserts_of_one_id_in_two_partitions_are_two_vectors() {
        let combined = folded(vec![
            json!([{"id": "1", "values": [1.0], "p": "a"}]),
            json!([{"id": "1", "values": [2.0], "p": "b"}, {"id": "1", "values": [3.0], "p": "a"}]),
        ]);

        assert_eq!(
            combined,
            [
                json!({"id": "1", "values": [3.0], "p": "a"}),
                json!({"id": "1", "values": [2.0], "p": "b"})
            ]
        );
    }

    #[test]
    fn an_item_without_a_string_id_is_passed_through_unfolded() {
        let combined = folded(vec![json!([{"id": 1}, {"id": 1}, {"values": []}])]);

        assert_eq!(combined.len(), 3);
    }

    fn entries(payload: Value) -> Vec<(String, Option<String>)> {
        let parsed = delete_entries(array(payload));
        parsed.into_iter().map(|e| (e.id, e.p)).collect()
    }

    #[test]
    fn a_delete_payload_holds_bare_ids_and_id_partition_objects() {
        let parsed = entries(json!(["7", {"id": "8", "p": "acme corp"}]));

        assert_eq!(
            parsed,
            [
                ("7".to_string(), None),
                ("8".to_string(), Some("acme corp".to_string()))
            ]
        );
    }

    #[test]
    #[should_panic(expected = "needs a string id and p")]
    fn a_delete_object_without_its_partition_is_refused() {
        entries(json!([{"id": "8"}]));
    }

    #[test]
    #[should_panic(expected = "non-string id")]
    fn a_delete_entry_that_is_a_number_is_refused() {
        entries(json!([8]));
    }
}
