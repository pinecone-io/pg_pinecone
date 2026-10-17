# Operator runbook

For `pinecone.promote()`-managed tables (any `sync_mode`) and AM-indexed tables (`CREATE INDEX
... USING pinecone`) — anything with a `pinecone.outbox` backlog. `sync_mode = 'sync'` tables never
have one (writes drain inline), so most of this doesn't apply to them. Tables declared for
full-text search (`track_docs`) have a queue of their own: see [Document sync](#document-sync-track_docs).

## Start here: `pinecone.status()`

```sql
SELECT table_name, mode, backlog, oldest_enqueued, queue_state, last_error
FROM pinecone.status();
```

`queue_state` is the one column to alert on: `idle` (nothing queued), `active` (queued, draining
normally), `backoff` (queued, retrying a transient failure — Pinecone outage, timeout), `parked`
(queued, stopped after `pinecone.outbox_max_attempts` permanent-class failures), `NO WORKER`
(queued, and nothing in the cluster will ever drain it), or `NULL` (caller lacks `EXECUTE` on
`pinecone._outbox_stats` — see [reference.md's Configuration](reference.md#configuration) grant
bundle). The document lane of a `track_docs` table reports the same states in `docs_queue_state`;
alert on both.

## Parked queue

**Symptom**: `queue_state = 'parked'`. **Diagnose**: `last_error` (from `pinecone.status()`) is the
failing row's error — usually a bad payload (dimension/type mismatch) or an auth/permission error,
not a transient outage (those back off forever, they never park).

**Recover**:

```sql
-- fix the root cause first (e.g. re-promote with the right dimension), then:
SELECT pinecone.outbox_retry('docs');    -- unparks, resets attempts/last_error, resumes draining

-- if the queued rows themselves are unrecoverable (e.g. schema changed underneath them):
SELECT pinecone.outbox_discard('docs', through_id => 12345);   -- drops rows id <= 12345
SELECT pinecone.sync('docs');            -- repair: deleteAll + backfill from current rows
```

`outbox_discard` is consistency-affecting by design (those writes never reach Pinecone) — it
always warns, and doesn't unpark by itself; run `outbox_retry` too if the queue was parked.

## `NO WORKER`

**Symptom**: `queue_state = 'NO WORKER'` with `backlog > 0`, or a `pinecone.promote()`/
`CREATE INDEX ... USING pinecone` call warned about it. Checklist, in order:

1. **Preloaded?** `SHOW shared_preload_libraries;` must include `pg_pinecone`. Not there → add it and
   restart Postgres (a config reload isn't enough — background workers only register at
   `_PG_init`, during preload).
2. **Enabled?** `SHOW pinecone.worker_enabled;` — `off` means the launcher never spawns anything.
   `ALTER SYSTEM SET pinecone.worker_enabled = on;` then `SELECT pg_reload_conf();` (it's `SIGHUP`,
   not a plain `SET`).
3. **Launcher actually running?**
   ```sql
   SELECT EXISTS (
       SELECT 1 FROM pg_catalog.pg_locks
       WHERE locktype = 'advisory' AND granted AND objsubid = 2
         AND classid = 1347832899 AND objid = 0
   ) AS launcher_alive;
   ```
   `false` despite preload + `worker_enabled = on` → check the server log for why the launcher BGW
   itself isn't up (crash-looping, `max_worker_processes` exhausted by other extensions, etc.).
4. **Library and extension out of step?** A preloaded `.so` never swaps itself out from under an
   already-started postmaster, and a restart does not update the extension: the worker idles until the two
   versions match, see [After `ALTER EXTENSION pg_pinecone UPDATE`](#after-alter-extension-pg_pinecone-update).
   `SELECT * FROM pinecone.worker_version();` says so with `idle = true`.

Don't check the *per-database worker's* own lock as a liveness signal — it idle-exits after 30
quiet rounds by design and is respawned by the launcher on demand (up to
`pinecone.worker_naptime_ms × 10` later if it just exited). The launcher lock above is the
authoritative "will anything ever drain this" check; `pinecone.flush()` and `promote()`'s own
warning already use it.

## After `ALTER EXTENSION pg_pinecone UPDATE`

An upgrade is install, restart, `ALTER EXTENSION pg_pinecone UPDATE` ([steps](#upgrading-from-0100-and-later)).
A preloaded library keeps its code until Postgres restarts, and a worker must not drain with code that does
not match the catalog. Every round the worker compares its compiled version with `pg_extension.extversion`;
while they differ it skips the whole round (no drain, backfill, recovery or maintenance), naps, and logs one
WARNING per database: `pg_pinecone worker 0.10.0 idle: extension is 0.9.0 — restart Postgres to load the new
worker`.

```sql
SELECT * FROM pinecone.worker_version();   -- compiled | installed | idle
```

`idle = true`, and which way the versions differ says what is missing:

- `compiled` newer than `installed`: the normal state between the restart and the update (the WARNING
  still says "restart", but the restart is done). Run the `ALTER EXTENSION`; the worker resumes within a
  naptime and drains what queued meanwhile.
- `compiled` older than `installed`: the update ran before the restart, or the old library file is still in
  place. Restart Postgres, with the new file installed.

Until then `pinecone.status()` still says `active` (the launcher is alive), so check `worker_version()` after
every upgrade. Only the worker has this guard; every other `pinecone.*` call runs the library its session
loaded against whatever catalog is installed.

## Backlog monitoring

Poll or alert on:

```sql
SELECT table_name, backlog, oldest_enqueued, queue_state
FROM pinecone.status()
WHERE backlog > 0;
```

`oldest_enqueued` age is the actual user-visible lag (Pinecone won't reflect that row until it
drains); `backlog` alone can't tell an outage from "no worker" apart — always read `queue_state`
alongside it. `pinecone.flush('docs', timeout_ms => ...)` blocks until a specific table's backlog
hits zero (or times out) — useful after a bulk load or a `sync()`, before running a query that
needs the write to be visible remotely. For the document lane the same two ideas apply to
`docs_backlog`/`docs_oldest_enqueued` and `flush_docs` (below).

## Backfill of a vector table

`create_index(.., engine => 'remote')`, `CREATE INDEX .. USING pinecone` and an async `promote` queue one
backfill job; `status()` counts it in `backlog` until it ends (a promoted table reads `promoting` until
then). The worker sends it one request per chunk, as many vectors as Pinecone's 2 MB request limit
allows (at most 1,000: about 600 at 768-d, 300 at 1,536-d), and runs the next round at once, with no
nap, while the job has chunks left. It naps `pinecone.worker_naptime_ms` (2 s) only when its queues are
idle or everything left is blocked: a table in backoff, a parked queue, a failed partition. Writes queued
behind the job wait for it. The same holds for a partition rebuild, `recover_from_remote` and a document
backfill.

**How long it takes** is rows per request / the time of a request: nothing to tune. On the 0.9.0 box
(768-d, chunks of 100 to 140 vectors, a 2 s nap after each round) it was 43 to 55 rows/s, 1M rows in about
5 h (extrapolated), and 220 rows/s with a 100 ms nap. Measured for 0.10 on the same box: 847 rows/s at 768-d from
`create_index` to worker idle (6,000 rows, defaults), 1M rows in about 20 minutes; the
[reference](reference.md#backfill-rate) has the arithmetic. A request of a 768-d table is about 5 MB of JSON: a proxy in front of Pinecone with a lower
body limit refuses it (the backfill then stops on its first chunk with that proxy's error), and over a link
under about 15 Mbit/s it can outlast `pinecone.timeout_ms` (5 s), which then needs raising.
`pinecone.backfill_round_ms` (1 s) no longer buys speed; it is how often live writes, other tables and
DDL get a turn.

A round holds `AccessShareLock` on its tables until it commits, for up to the budget plus one chunk per
table with a job, so an `ALTER TABLE` or `DROP INDEX` waits for it (and gets in before the next round). A
stop request to the worker ends the round after its current chunk.

**A failing chunk** (a 5xx, a timeout, a 429 on a write after its three attempts) leaves the cursor at the
last chunk that went out; the table backs off (1 s, doubling to 60 s) and resumes there. A killed worker
re-sends the chunks of its round, which is harmless: an upsert of the same id is idempotent.

## Reads that get a 429

An index scan, `pinecone.topk` or `search_docs` that Pinecone answers with a 429 (or a 5xx) fails its
statement: reads are not retried, writes are (three attempts). Pinecone's read ceiling is per
namespace and soft: the 0.9.0 box saw about 245 to 250 accepted queries per second on a 1M-row
namespace (queueing, no 429) and 429s after a burst above 290 per second on a 100k one. In the
application, retry with backoff and jitter. Or spread tenants over namespaces with the
[partition column](reference.md#partition-column-on-a-vector-table): 100 tenants of 5,000 rows served
1,084 queries per second at 16 clients with none. Keep the largest `LIMIT` within `pinecone.am_min_topk`
(100): a scan that must ask again sends two requests per query, which halves what a namespace accepts
(`LIMIT 100` on the 1M namespace at 16 clients: 124 queries per second with the old default of 50, 251
with `am_min_topk` 100).

## VACUUM vs. outage on AM tables

An AM-indexed table's `VACUUM` deletes dead tuples from the remote index synchronously, inline —
unlike ordinary writes, this is not queued through the outbox. If Pinecone is unreachable,
`VACUUM` on that table fails outright (after the client's own bounded HTTP retries); autovacuum
simply retries it on its own schedule, same as any other `VACUUM` failure.

**Diagnose**: `pinecone.status()`'s `last_error` surfaces the most recent `ambulkdelete` failure
for that table (ahead of an ordinary outbox `last_error`, so it's never masked by one). A string
that looks like an HTTP/timeout error, not a payload validation error, means "outage", not "bug" —
no action needed beyond waiting for Pinecone to recover; the next autovacuum run will succeed.

## Document sync (`track_docs`)

A table declared with `pinecone.track_docs` has a queue of its own (`pinecone.doc_outbox`), its own
park flag and its own `status()` columns, apart from the vector lane. A table can be in both, and
each lane backs off and parks on its own. The [`NO WORKER` checklist](#no-worker) applies to both.

### Read the document lane

```sql
SELECT table_name, docs_backlog, docs_oldest_enqueued, docs_last_delivered_at,
       docs_backfill_done, docs_backfill_total, docs_failed, docs_queue_state, last_error
FROM pinecone.status()
WHERE docs_backfill_done IS NOT NULL;   -- the tables declared with track_docs
```

`docs_queue_state` is the one to alert on. It has the states of `queue_state`: `idle`, `active`,
`backoff` (a delivery attempt failed and nothing has been delivered since), `parked`, `NO WORKER`.
Alert on `docs_failed > 0` (dead letters) and on the age of `docs_oldest_enqueued` (the delivery
lag) too. A table that only has `track_docs` is a row with `mode = 'docs'` and a `NULL`
`queue_state`: add `docs_queue_state` to the alert, don't swap it in. `last_error` shows the
document lane's error when the vector lane has none.

### Freshness

A committed change is delivered within about one worker interval (`pinecone.worker_naptime_ms`,
2 s; up to ten if the worker had idled out and is respawned) when nothing is queued ahead of it,
and is searchable once Pinecone has indexed it: up to about a minute after delivery. The worker
delivers live changes first and runs backfill chunks after them, so a backfill doesn't make new
writes stale.

- `docs_oldest_enqueued` is the lag: its age is how long the oldest undelivered change has waited.
- `docs_last_delivered_at` says when Pinecone last accepted a request. `status()` can't see
  Pinecone's own indexing, so allow up to a minute after that.
- A row deleted locally but not yet deleted remotely drops out of a result at the join.
- `engine => 'local'` has no delay (it reads the heap).

### Dead letters

**Symptom**: `docs_failed > 0`. Pinecone refused one document (a field it won't take), an
`on_oversize => 'error'` column refused it at the drain, or the table could not render the row
(`the row could not be read: ... out of range for type double precision`, a `numeric` that no
`float8` holds). That document is a dead letter; the rest of the table keeps syncing, and the lane
doesn't park. Dead letters aren't backlog. A TRUNCATE clears them: they name rows that are gone.

**Diagnose**: `SELECT * FROM pinecone.failed_docs('articles');` lists the id, Pinecone's message
and when. Fix the row, or the declaration (`on_oversize => 'truncate'` for long text).

**Recover**:

```sql
SELECT pinecone.retry_failed_docs('articles');   -- requeues them, returns how many
```

A document that is delivered later, by a retry or because the row changed, loses its dead letter.
When Pinecone refuses a whole batch (10 documents or more) or 100 documents in a row, or the table
cannot render that many rows, the cause isn't the documents: no dead letters are written, the rows
stay queued, and the lane takes one error with the message. It parks after
`pinecone.outbox_max_attempts` of them.

### Parked document lane

**Symptom**: `docs_queue_state = 'parked'`. **Diagnose**: `last_error` is the reason: a rejected
key (`authentication failed`), a declared column that was dropped behind the DDL guard, or a
refusal of everything (`Pinecone refused N documents in a row — the index schema or the declaration
has probably drifted`). A 429, a 5xx or a timeout never parks the lane: it backs off and retries.

**Recover**: fix the cause, then

```sql
SELECT pinecone.unpark_docs('articles');   -- table owner; resets attempts and last_error
```

Nothing was lost: the queued changes stay and the worker resumes with them. While the lane is
parked, `flush_docs` returns `false` at once and `backfill_docs` refuses (SQLSTATE 55000). If the
index itself is wrong (deleted, or its schema changed), `untrack_docs` and `track_docs` again
builds a fresh registration and backfills it.

`untrack_docs`, `backfill_docs` and `unpark_docs` wait for the batch the worker is running for that
table (a NOTICE says so; a batch ends after about ten seconds and a request). The worker holds a
table's lane lock for that table's batch only, so another table's batch doesn't make them wait.
`lock_timeout` applies to the wait: set it to fail instead.

### Flush

`pinecone.flush_docs('articles', timeout_ms => 30000)` waits until nothing is queued for the table,
a pending backfill included. Use it after a bulk load, and in migrations and tests before the next
step. `true` means everything committed before the call was delivered; it doesn't mean searchable
(allow up to a minute). `false` comes with a WARNING that points at `status()`: the lane is
parked, or no launcher runs (both return at once, since waiting can't help), or the timeout passed
with N documents still queued. `flush_docs` never drains anything itself.

### Backfill

`track_docs` queues one backfill job for the rows the table already holds. `status()` shows it:
`docs_backfill_done` / `docs_backfill_total` count rows (the count when the job was queued), and
`docs_backlog` includes the job until it ends. The worker runs it in chunks of up to 1,000 rows
after the live changes, for `pinecone.backfill_round_ms` per round, and starts the next round at once
while chunks are left; each chunk commits its cursor, so a restart re-sends at most one chunk. Searches during the backfill see part of the table, and
say so with a NOTICE.

**After a restore**, or whenever the index may no longer match the table (an index recreated,
documents deleted in the console): Postgres is the source of truth and Pinecone keeps no backup of a
document index. A plain `pg_dump` / `pg_restore` works: the table's key and sync triggers come back
after its data and the registry row, and the DDL guard lets them through (a missing key or trigger
isn't refused, dropping one still is). A dump carries the registry (`pinecone.doc_tables`) but not
the queue (a queue isn't data), so the restored table looks in sync until you re-send everything:

```sql
SELECT pinecone.backfill_docs('articles', clear => true);   -- empties the namespace, then re-sends every row
```

`clear => true` also removes documents the restored table no longer has; without it the job only
upserts. Search sees an empty or partial index until it finishes: watch `docs_backfill_done` reach
`docs_backfill_total` and `docs_queue_state` return to `idle`. `backfill_docs` is for the table
owner, replaces a backfill in flight and refuses while the lane is parked.

## Local-engine (`mode = 'local'`) maintenance

A local-mode pg_pinecone index (`CREATE INDEX ... USING pinecone (col vector_l2_ops) WITH (mode =
'local')`) never talks to Pinecone, so it isn't in `pinecone.status()`/the outbox at all — it has its
own, separate maintenance story.

**Self-builds.** The code scan's popcount kernels and the exact-distance kernels pick AVX-512, AVX2 or
NEON at runtime (a scalar path otherwise), so a source build needs no `target-cpu` flag for them; the
prebuilt x86_64 artifacts still target x86-64-v3 (README install section).

**Keep the vectors inline (`STORAGE PLAIN`).** `CREATE INDEX` prints a NOTICE when a wide vector column is stored out of line (pgvector's default, `EXTERNAL`): every rescored row then costs a TOAST fetch, about 0.65 ms per 200 rows (measured on the 0.9.0 profile box). The statement it prints, and the rewrite that moves the existing rows:

```sql
ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN;
VACUUM FULL items;   -- ACCESS EXCLUSIVE for the rewrite; rebuilds the pg_pinecone index
```

`UPDATE items SET embedding = embedding` leaves the rows out of line. A row must fit one 8 KB page, about 2,000 dimensions at most: see the [reference](reference.md#native-engine-probe-count-pool-and-storage-plain) for the limits and numbers.

**Calibrated defaults: check them, and when to override them.** `CREATE INDEX` and `REINDEX` calibrate the probe count and the rescore pool of a clustered index for `target_recall` (default 0.95) and say what they chose in a NOTICE. Read it back at any time:

```sql
SELECT * FROM pinecone.index_calibration('items_idx');
-- nprobe | pool | calib_nprobe | calib_pool | calib_recall | calib_samples | target_recall | calib_reached | queries_source | queries_count
```

- `nprobe` and `pool` are what an unfiltered scan uses (`SET pinecone.scan_fraction` and `pinecone.rescore_multiplier` override them per session). `calib_samples = 0` means the build measured nothing (a table under 500 rows, or a flat index, which scans exactly and shows `NULL`s): the 0.9.0 constants apply.
- `calib_reached = f` means no pair on the grid reached the target (the NOTICE said so) and the cheapest pair near the best recall was kept, which on data with no cluster structure can be a near-exhaustive scan. Lower `target_recall` to trade recall for speed, unset a `rescore_multiplier` that pins the pool too low, or raise `n_lists`.
- By default the sample is rows of the table (each left out of its own search), so the numbers describe queries that look like the rows; for a query set unlike the data, [calibrate on real queries](#calibrate-with-real-queries) instead. A query set unlike the data (question embeddings against passage embeddings, queries concentrated on a few tenants) can land below the target: on wiki_dpr rows read 0.01 to 0.04 above held-out queries, and the build's 0.01 offset covers a part of that. Measure recall@10 of real queries against the exact answers (the same query with `SET pinecone.am_enabled = off`), sweep `pinecone.scan_fraction` and `pinecone.rescore_multiplier` with `SET`, then pin the pool in the index with `ALTER INDEX items_idx SET (rescore_multiplier = 5)` and `REINDEX` (the build calibrates the probe count for that pool), or raise `target_recall` until the measured recall is where you want it.
- A query the first batch cannot fill (a `LIMIT` above the pool, a selective `WHERE` the index cannot push) goes on under `pinecone.iterative_scan = auto`, up to `pinecone.max_scan_tuples` rescored rows: watch `capped` in `pinecone.last_scan_stats()`; `SET pinecone.iterative_scan = off` returns the whole rescored pool and nothing more (the early stop, `pinecone.scan_prune`, is `auto`'s first batch only).
- **Capacity.** Behind a selective residual `WHERE`, the `auto` default fetches up to `pinecone.max_scan_tuples` rows (20,000, about 4 µs each when warm) for one query, and every client can do that at once. Lower it for roles whose queries carry such filters (`ALTER ROLE app SET pinecone.max_scan_tuples = 2000`) or set a `statement_timeout`.
- Inserts do not recalibrate: the pair stays what the build found. When `pinecone.index_drift` shows the data has moved, `REINDEX` recalibrates (scheduled maintenance does too).
- Calibration costs the build 200 sample queries of exact distances over the table (fewer above a million rows) and a scan per probe count: 1.7 s for a million 768-d rows with eight workers (0.5 s for 300,000, 0.04 s for 20,000), small next to the k-means it follows. [Reference](reference.md#native-engine-calibrated-defaults) has the method and its limits.

<a id="calibrate-with-real-queries"></a>
**Calibrate with real queries.** The build tunes the probe count and pool on rows of the table. When your queries come from another distribution than your rows (question embeddings against passage embeddings, a few busy tenants), give it a sample of the real ones:

```sql
CREATE TABLE real_queries (id serial PRIMARY KEY, embedding vector(768));
-- load 200 to 1,000 recent queries, embedded with the model of items.embedding
ALTER INDEX items_idx SET (calibration_queries = 'public.real_queries');  -- or WITH (...) at CREATE INDEX
REINDEX INDEX items_idx;                                                  -- the table is read at the next build
SELECT queries_source, queries_count, calib_nprobe, calib_recall FROM pinecone.index_calibration('items_idx');
-- queries_source | queries_count | calib_nprobe | calib_recall
-- table          |           200 |           15 |       0.9715
```

- **Which queries.** 200 to 1,000 real queries, embedded with the same model as the indexed column, and never the queries you evaluate recall with: the build tunes to them, so recall measured on the same set reads too well. Up to 200 distinct queries are drawn (by `seed`; repeats count once; a table below 200 is used whole); fewer than 50 distinct is an error. The build reads the table once (0.03 s for 1,000 768-d rows, 1.8 s for 100,000), so a few thousand rows are plenty (above 100,000 the build refuses the table: name a view over a sample); they must not be rows of the indexed table.
- **When.** Question/passage workloads, and tenants whose queries concentrate in a few regions of the space. Not needed when the queries look like the rows. On wiki_dpr 1M x 768-d six builds (seeds 42, 1, 2, 3, 4 and 5) read fresh recall@10 0.913 to 0.958 on 500 held-out queries calibrated on rows (two of six below 0.95), and 0.965 to 0.981 calibrated on 200 held-out vectors (six of six), at 15 to 22 probed lists instead of 8 to 14: about 0.01 ms of p50 per extra list (13 to 15 lists 0.54 to 0.55 ms, 8 to 15 lists 0.46 to 0.55 ms with recall 0.913 to 0.965; interleaved builds on one laptop, M4 Max, release build).
- **Privileges.** The table is read as the owner of the indexed table (`USAGE` on its schema, `SELECT` on the column), whoever runs the statement: a superuser's `REINDEX`, `reindex_native_indexes()` and the maintenance worker never read a table the owner cannot.
- **Keep the table.** At `CREATE INDEX` a table that cannot be used (missing, wrong type or dimension, fewer than 50 distinct queries, no privilege, locked by another transaction) is an error that names the problem. At a rebuild it is a WARNING and the build calibrates on table rows, so a dropped, renamed, emptied or unreadable table never fails a `REINDEX` or the worker; `queries_source` then reads `fallback`. `ALTER INDEX items_idx RESET (calibration_queries)` drops the setting. `ALTER TABLE .. ALTER COLUMN .. TYPE` on the indexed column re-creates the index like `CREATE INDEX`, so an unusable table fails that statement.
- **Dump and restore.** `pg_dump` writes the option into the `CREATE INDEX`, which a restore runs after loading the data but before it restores GRANTs and refreshes materialized views. The build then fails, and the restore goes on without the index (or stops, with `--exit-on-error` or `ON_ERROR_STOP`), when the owner of the indexed table reads the queries through a GRANT (a table or schema another role owns) or the queries are a materialized view. A parallel `pg_restore -j` can also build the index before the queries table is loaded (it does not know the dependency). Restore without `-j` when the queries are a table (or a view over tables) the indexed table's owner owns, in `public` or a schema that role owns; otherwise create the index after the restore.
- **Refresh.** Replace the rows when the workload moves and `REINDEX`: inserts into the table of queries recalibrate nothing.

**When centroids are rebuilt.** The IVF centroids (and the calibrated-Hamming stats) are computed
once at `CREATE INDEX` and recomputed **only** by a full build/REINDEX — there is no incremental or
scheduled centroid refresh, and `aminsert` never moves them (it assigns each new row to the nearest
existing centroid). So the centroid-rebuild cadence is exactly your REINDEX cadence: **none by
default** (`pinecone.auto_maintain` is `off`, so centroids never change after the initial build until
you `REINDEX` by hand), or — with drift-gated maintenance enabled — once a scheduled index reaches
`drift_threshold` inserts (default `0.2`, i.e. roughly every 20% of net row growth; see below).
Deletes don't count toward drift and never trigger a recompute on their own.

**The drift model.** `aminsert` appends new rows in place (nearest-centroid, 1-2 pages/record) —
cheap, but every append leaves the index's IVF clustering (its centroids and lists) a little
more stale relative to what a fresh build would produce. `pinecone.index_drift(idx)` reports exactly
how stale:

```sql
SELECT * FROM pinecone.index_drift('docs_idx');
-- inserted_since_build | total | drift_fraction
-- ------------------- +------+---------------
--                  300 | 1300 |          0.23
```

`inserted_since_build` is rows appended via `aminsert` since the last build/REINDEX;
`total` is the current effective row count; `drift_fraction = inserted_since_build / total`. It
works on any local-mode pg_pinecone index whether or not maintenance is scheduled on it — call it ad hoc
any time.

**Scheduling maintenance:**

```sql
SELECT pinecone.schedule_maintenance('docs_idx', drift_threshold => 0.2);  -- default 0.2
SELECT * FROM pinecone.maintenance_status();  -- per-scheduled-index drift, cheap
```

`schedule_maintenance` only registers a policy — it never runs a REINDEX by itself. The
background worker evaluates every scheduled index each round and REINDEXes it once
`drift_fraction >= drift_threshold`, but **only if `pinecone.auto_maintain` is `on`** (default
`off`):

```sql
ALTER SYSTEM SET pinecone.auto_maintain = on;   -- or: SET pinecone.auto_maintain = on; (session-scoped)
SELECT pg_reload_conf();
```

A scheduled index that is offline (built by an older index format, see [Upgrading from 0.10.0 and
later](#upgrading-from-0100-and-later)) counts as drift 1.0: with `auto_maintain` on, the worker rebuilds
it in its next round, whatever its threshold.

**The blocking-REINDEX caveat (read this before turning `auto_maintain` on).** There is no
`REINDEX CONCURRENTLY` here — the local engine's build has real reasons to reject it (see
`local/vacuum.rs`'s "CIC/RIC decision": a flat/bootstrap index has no TIDs to enumerate, engine
stickiness is keyed by a stable oid a concurrent rebuild wouldn't preserve, and the k-means
snapshot/catch-up-insert interplay is unvalidated). A maintenance REINDEX therefore takes the
**same locks as any plain `REINDEX INDEX`** — it blocks writes to the table and blocks reads that
would use this index for the whole rebuild. Turning `pinecone.auto_maintain` on means the worker can
trigger that blocking rebuild unattended, on its own naptime cadence; schedule it (or watch
`pinecone.maintenance_status()` and REINDEX by hand) inside a maintenance window, not against a table
serving live traffic you can't afford to stall. This module's value is the drift *visibility* and
the *evidence-gated trigger*, never a non-blocking rebuild.

**Recommended thresholds.** In the churn benchmark (1M x 768d), pgvector's own
maintenance cost ~2400-2450s per cycle as an in-place VACUUM, or ~97-145s as a full REINDEX of its
HNSW index. Those are pgvector's figures, not the native engine's: no native rebuild time has been
re-measured since the parallel build landed, so time a `REINDEX` on your own table before you
count on a number. A rebuild is a whole-index operation either way, so batching several points of
drift into one rebuild, rather than reindexing on every small increment of churn, is the right
default trade. A 20-30% drift threshold is the recommended default band: roughly an order of
magnitude fewer REINDEX cycles than rebuilding on every couple of percent of churn (rough figure
carried from those measurements, ~14x). Lower it if query latency/recall visibly degrades before
drift reaches that band; there's no evidence in that benchmark that it does.

**Who may schedule, and who the REINDEX runs as.** `pinecone.schedule_maintenance` requires the
caller to own the index's table (same rule as `REINDEX` itself; superuser bypasses), and records
who scheduled it (`pinecone.index_maintenance.scheduled_by`). The worker's REINDEX then runs under
`SET ROLE` to the table's *current* owner — never as the worker's own (typically superuser)
identity — so a non-superuser-owned table's owner needs everything a manual `REINDEX` needs:
`USAGE` on the index's schema, plus (pre-existing for any pg_pinecone index built by a plain role, since
Postgres runs index builds as the table owner) `USAGE` on schema `pg_pinecone` and
`SELECT, INSERT, UPDATE ON pinecone._am_engine` for the engine-stickiness bookkeeping.

**Warning — queue-table DML grants bypass the verb gates.** The ownership checks live in the
verbs (`schedule_maintenance`, `demote_index`, `promote`/`demote`/`sync`), not in the tables they
enqueue into. A role with DML on `pinecone.index_maintenance`, `pinecone.pending_demotes`, or
`pinecone.outbox` can insert rows directly and the worker will act on them — REINDEX or demote an
arbitrary index, empty a remote namespace — with no ownership check ever running. Treat DML on
those three tables as admin-only; see
[reference.md's grant bundles](reference.md#grant-bundles-per-verb) for the exact per-verb grants
that are actually needed (never blanket DML).

**Poison/parking.** A REINDEX attempt that fails (lock conflict, permission error, whatever) is
recorded (`pinecone.maintenance_status().last_error`); after `pinecone.outbox_max_attempts` consecutive
failures (the same GUC the outbox drain path uses) the schedule parks
(`queue_parked = true`) and the worker stops retrying it. Re-running
`pinecone.schedule_maintenance(idx, ...)` clears the park — there is no separate retry function.

**The owner-split parking case.** The worker's rebuild runs `SET ROLE` to the *table's* current
owner, but `ALTER INDEX`/`REINDEX` require owning the *index* — normally the same role, but the
two can drift apart (catalog surgery, `REASSIGN OWNED` edge cases, ownership changed between
scheduling and the attempt). A split shows up as a repeated `must be owner of index ...` in
`last_error` until the row parks — the worker retrying can never fix it. Remedy: realign with
`ALTER TABLE ... OWNER TO <role>` (which reassigns the table's indexes with it), then re-run the
verb (`schedule_maintenance`, or `demote_index` for a parked demote — same parking rules) to
unpark.

**What this does NOT do.** No `REINDEX CONCURRENTLY` (see above — a considered rejection, not a
gap). No incremental per-list re-cluster/split-merge: a scheduled REINDEX always rebuilds the
whole index from the live heap, never patches individual IVF lists in place. No adaptive/
self-tuning threshold — `drift_threshold` is whatever you set it to, forever, until you
reschedule. Background re-cluster below the full-rebuild threshold stays evidence-gated: the churn
benchmark found zero recall drift over several cycles, so nothing here tries to recluster
preemptively — REINDEX past the threshold is the only rebuild path.

**Build-path memory.** `ambuild`/a maintenance `REINDEX` buffers every collected row's vector in
RAM for the whole build (roughly 4KB/row at 768 dimensions — the raw `f32` vector plus per-row
overhead), then peaks at roughly double that while k-means clustering runs alongside the still-live
collected set. There is no spill-to-disk path: an allocation failure here crashes the backend
running the build, not just the statement. Size `maintenance_work_mem`-equivalent RAM (i.e. actual
available backend memory, since this isn't `maintenance_work_mem`-budgeted today) for the largest
table you'll `CREATE INDEX`/`REINDEX` before turning on `pinecone.auto_maintain` or scheduling
maintenance against it.

**The filtered probe-fraction ramp, and its limits.** A FILTERED local scan in mode H
automatically probes more IVF lists than an unfiltered one — `min(0.5, 4x the unfiltered
fraction)` — because in-loop filtering makes an unqualifying record cost one compare instead of a
full scoring pass. Both numbers come straight from the measured sweep (doubling probed lists cost
+2.3ms; fraction 0.5 delivered the 1%-correlated pass at 0.967 recall @ 9.7ms) — **they are
measured points, not a fitted curve**: the 4x multiplier is the largest swept multiple that stayed
inside the safe latency band, and the 0.5 cap is the sweep's own winning point, not an
extrapolation past it. On a corpus whose filter/cluster correlation differs a lot from the bench
fixtures, don't assume the ramp extrapolates — sweep `pinecone.scan_fraction` yourself (any explicit
value overrides the ramp, session- or query-scoped via `SET LOCAL`) and watch recall/latency
before standardizing on a value.

## Engine lifecycle (`pinecone.create_index` / `set_engine` / `promote_index` / `demote_index`)

pg_pinecone 0.5's storage-engine surface: one vector column, three interchangeable engines —
`'pgvector'` (a plain `USING hnsw` index; pgvector must be installed), `'native'` (the local
pg_pinecone engine, `USING pinecone WITH (mode = 'local')`), `'remote'` (Pinecone-backed, `USING pinecone
WITH (mode = 'remote')`). **Queries never change across engines** — the same `ORDER BY embedding
<-> $1 LIMIT k` is served by whichever engine currently backs the column.

```sql
-- stand up an index on a chosen engine (defaults: engine 'pgvector', metric 'cosine'):
SELECT pinecone.create_index('docs', 'embedding', engine => 'native', metric => 'l2');

-- the one-verb switch (native <-> remote; accepts the table or the index):
SELECT pinecone.set_engine('docs', 'remote');   -- routes through pinecone.promote_index
SELECT pinecone.set_engine('docs', 'native');   -- routes through pinecone.demote_index

-- or drive the pg_pinecone-AM index's lifecycle directly:
SELECT pinecone.promote_index('docs_embedding_pinecone_idx');
SELECT pinecone.demote_index('docs_embedding_pinecone_idx');
```

**Promote (`native -> remote`)**: requires a stored API key (`pinecone.set_api_key`). A filtered
local index promotes too: the backfill and every later write carry its key as Pinecone metadata.
Creates/adopts the Pinecone index, flips the index to a `promoting`
state, and queues a TID-keyed backfill for the background worker. Throughout the promoting
window, **scans keep serving the LOCAL engine** (the remote namespace is incomplete until the
backfill drains — pg_pinecone never serves a partial namespace) and **writes go to both** (local
append + remote queue), so local answers stay complete the whole way. The worker flips the index
to remote in the same round its backfill exhausts; from that moment scans dispatch remotely.
No availability gap at any point.

**Demote (`remote -> native`) — the exact guarantee**: `pinecone.demote_index` returns immediately;
it flips the index to a `demoting` state and queues the rebuild for the background worker.
*While the demote is queued — however long that is — scans and writes keep serving the remote
engine, correctly and without interruption.* The rebuild itself is a plain, blocking
`REINDEX` (there is no `REINDEX CONCURRENTLY` for the local engine — see the maintenance
section's CIC/RIC verdict), so **once the worker's REINDEX actually starts, reads that would
use this index and writes to the table block for the rebuild's duration** (seconds to a few
minutes, scaling with table size — the same cost as any REINDEX of this index). Plan demotes
into a maintenance window if that pause matters. In one line: **never unavailable while queued;
an availability gap equal to the rebuild time once the rebuild starts** — this is v1's honest
contract, not zero-downtime-throughout; a non-blocking demote needs a concurrent
rebuild-into-local that does not exist yet. If the rebuild fails, the demote stays queued and
retries (same parking rules as the outbox: `pinecone.outbox_max_attempts` consecutive failures
park it; re-run `pinecone.demote_index` after fixing the cause). On success the superseded remote
index (or just this table's namespace, if the index is shared) is deleted and the registry row
removed.

**Crash safety** (drilled in `scripts/crash-drill.sh` legs D9-D11): a crash mid-backfill
resumes from the persisted cursor with scans still local-correct; the promote flip is one
atomic WAL record; a crash mid-rebuild leaves the index remote-serving with the demote still
queued. If a *transaction you wrapped around* `promote_index`/`demote_index` rolls back, re-run
the same verb — it detects and repairs the half-done state (the verbs warn when called inside a
transaction block).

**Remote cleanup is retriable** (0.6.1): `demote_step` persists the "delete the superseded
remote index" obligation to `pinecone.pending_cleanups` in the same transaction it deletes the
registry row, and the worker retries the delete every round until Pinecone confirms it gone
(a 404). A worker crash between `demote_step` and the delete, or a Pinecone outage that outlasts
any single HTTP attempt, no longer orphans the remote index — the queued obligation self-heals
once reachable. A permanent-class refusal (e.g. `deletion_protection`) parks the row after
`pinecone.outbox_max_attempts` with a warning naming the index for manual deletion.

**Not supported yet**: `set_engine` to/from `'pgvector'` (stand up a fresh
`pinecone.create_index(..., engine => ...)` instead — the error says exactly this); BYOC
cloud/region overrides on this surface (uses `pinecone.am_cloud`/`pinecone.am_region`).

## Attach an existing index

Already have a Pinecone index — populated some other way — that matches a table you want to
query through `pg_pinecone`? `pinecone_index` (on `pinecone.create_index(engine => 'remote')` or
`pinecone.set_engine(tbl, 'remote')`) attaches it instead of creating a new one:

```sql
SELECT pinecone.create_index('docs', 'embedding', engine => 'remote',
                              pinecone_index => 'my-existing-index',
                              pinecone_namespace => 'my-namespace');  -- default: the usual pv-<schema>-<table> namespace
```

What it checks, in order, before touching anything: the index's dimension/metric/vector_type
match the column and opclass (else an ERROR naming both shapes); the namespace isn't already
owned by another tracked table; and (unless `attach_unchecked => true`) `describe_index_stats`'s
vector count for that namespace equals `docs`'s current live row count (else an ERROR naming both
counts — a real mismatch almost always means the wrong index/namespace, or a stale export).

What it does NOT do: create or backfill anything. The namespace is assumed to already hold this
table's rows, one vector per row, **with ids in the AM's TID encoding** (`"block:offset"`, the
same scheme `CREATE INDEX ... USING pinecone` always uses) — not your own id column's values.
This is normally only true of data exported from a `pg_pinecone`-AM table in the first place;
attaching a namespace populated any other way will look "attached" but every read will silently
return zero matches (the ids just won't resolve to any live tuple), and every subsequent write
still goes through the outbox exactly as usual. `pinecone.status()` shows `attached = true` for
the row.

**A filtered index needs the metadata too.** With `filter_column` (or an index created with a
filter key), every vector in the attached namespace must already carry `{"<col>": value}` metadata,
exactly as `aminsert` and the backfill write it (a NULL key carries none). Attach runs no backfill,
`attach_unchecked` skips only the count check, and nothing can check the metadata: a filtered read
silently misses every row that lacks it. Unfiltered reads are unaffected. The attach says so in a
NOTICE.

**A stale export invalidates the ids.** `VACUUM FULL`/`CLUSTER`, run against the source table
*after* the data was exported to the namespace you're about to attach, rewrite every row to a new
TID — the exported ids no longer match. Re-export (or skip `VACUUM FULL`/`CLUSTER` between export
and attach) rather than attach against a table that's been rewritten since.

`repoint`/`detach_storage`/`recover_from_remote` all keep working exactly as they do on any other
`pg_pinecone`-AM-indexed table (i.e. `repoint`/`detach_storage` still refuse it outright — same as
every AM-indexed table, attached or not; `recover_from_remote` already supports AM-indexed tables
regardless of how they were registered) — attaching changes nothing about how they see the row.

## Partitioned tables

A table with a [partition column](reference.md#partition-column-on-a-vector-table) keeps one Pinecone
namespace per value, `<namespace>:<value>`. `pinecone.partitions_of('docs')` lists the ones it wrote
(the registry table itself is closed to every role but its owner) and `pinecone.status()` counts them
(`partitions`). A drain registers a namespace when Pinecone takes its first write, so a write still
queued is not listed yet.

**Offboard a tenant.** Delete its rows, send the deletes, then drop the namespace:

```sql
DELETE FROM docs WHERE tenant_id = 42;          -- queues one id delete per row
SELECT pinecone.flush('docs');                  -- sends them (`flush_docs` for a document table)
SELECT pinecone.delete_partition('docs', 42);   -- deletes the namespace, forgets it
```

The order matters: a write still queued for the tenant recreates its namespace when it drains, and
`delete_partition` alone leaves the rows in Postgres, so the next write of one registers the
namespace again. `delete_partition` says so in a NOTICE when the table has anything queued. For a big
tenant the per-row deletes are wasted work: run the `DELETE` with `session_replication_role =
replica` (it needs the privilege) so none are queued.

**A failed tenant.** When Pinecone refuses one tenant's namespace for good (a 4xx other than 404 and
429: the plan's limit of namespaces on a new tenant, a request it calls bad), the worker marks that
partition failed, logs one WARNING and goes on: the other tenants keep draining and the table does not
park. The tenant's writes and deletes are dropped until you retry (the table is the source of truth, so
nothing is queued for later), and its namespace keeps what it held, so its search results are stale (a
tenant that failed during the initial load has nothing: `topk` returns no rows for it). A row that
moved out or was deleted stays in the namespace, and `topk` leaves it out by checking its ids against
the table: it can return fewer than `k` rows and raises a NOTICE naming the partition and `failed_at`.

```sql
SELECT * FROM pinecone.failed_partitions('docs');   -- value, namespace, failed_at, last_error
SELECT table_name, failed_partitions FROM pinecone.status();
```

Alert on `status().failed_partitions > 0`: a failed partition changes neither `queue_state` nor `last_error`.

1. `last_error` names the namespace (`... — namespace 'public.docs:42'`; the value follows the colon,
   odd bytes as `%XX`) and says why. A HINT about the plan's namespace limit means: delete namespaces
   you no longer need (`delete_partition`) or upgrade the plan. The same refusal of a second tenant
   is not marked: it parks the table (`queue_state = parked`, the reason names that tenant's
   namespace), because a cause every tenant meets (a dimension or metric that does not match the
   index, say) is the table's. Fix it, `pinecone.outbox_retry`, then retry the one failed partition,
   or `pinecone.sync` for the whole table.
2. Fix the cause: Pinecone refused the request, not the queue, so there is nothing to discard.
3. Rebuild from the table: `SELECT pinecone.retry_failed_partition('docs', 42);`, or
   `pinecone.retry_failed_partitions('docs')` for all of them. The call only queues the rebuild: the
   worker deletes the namespace and loads the tenant's rows again in the background, and until that
   ends a search of the tenant sees only what was written since. `backlog` falls back to 0 and
   `failed_partitions` stays 0. If Pinecone still refuses, the partition fails again with the new
   error.
4. While a rebuild runs, other tenants' writes keep flowing: every worker round sends the queued
   writes first, then rebuilds for up to `pinecone.backfill_round_ms` (1 s) and starts the next round at
   once while chunks are left, so a write made meanwhile is delivered within about a round, not after
   the rebuild. A round loads chunks of a
   thousand rows, of as many tenants as fit in the budget (at least one chunk): small tenants share
   rounds, a big one takes as many rounds as its rows need. `backlog` counts the rebuild jobs still
   queued beside the writes. A larger `pinecone.backfill_round_ms` rebuilds no faster, since
   rounds follow each other at once, but holds the table's lock for the whole round: `retry_failed_partition`, `delete_partition`,
   `demote`, `sync` and DDL on the table wait for it. An index rebuilds each tenant by scanning the
   heap for it, so many failed tenants of a big table cost a scan each.

Not isolated, because nothing says Pinecone throttles per namespace rather than per index: a 429, a
5xx and a timeout back the whole table off and clear when Pinecone does, and a bad or missing key or an
index that does not exist parks it (see [Parked queue](#parked-queue)). A `sync_mode = 'sync'` writer
gets the refusal itself and nothing is marked. `backup_readiness` fails while a partition is failed.
TRUNCATE, `sync`, REINDEX and `delete_partition` forget the failure with the namespace.

Document lane: a refused document is a dead letter ([Document sync](#document-sync-track_docs)), a
429 or 5xx backs the lane off for every tenant, and a refusal of a new namespace that names the plan's
limit stops the lane with the namespace and the plan HINT in `last_error`: free a namespace
(`delete_partition` of a tenant you no longer need) or upgrade, then `pinecone.unpark_docs('articles')`.

**Restore with many namespaces.** The steps are the [disaster-recovery](disaster-recovery.md) ones;
what changes is the size.

1. Restore the backup to a new index, then `pinecone.repoint('docs', 'docs-restored')`. It lists
   the family on the new index and registers the namespaces the registry lacks (`adopted N`). A
   WARNING names registered namespaces the new index lacks: recovery finds nothing in them.
2. `pinecone.recover_from_remote('docs')` walks the registered namespaces in order, a page at a
   time, and a worker round keeps walking until `pinecone.backfill_round_ms` (1 s) is spent: a
   thousand small tenants are a few rounds, not a thousand naps. A worker restart, or
   `outbox_retry` of a parked job, resumes at the namespace and page it was in.
3. A tenant offboarded after the backup is in the restored index and comes back with it: run
   `delete_partition` for it again.

`backup_readiness` warns above 2,000 namespaces in the index: pg_pinecone has no verified figure for what
one backup holds, so restore a test copy before you rely on it.

**Orphan namespaces.** A namespace Pinecone holds under `<namespace>:` and the registry does not list
comes from a `sync_mode = 'sync'` write that rolled back after Pinecone took it, a restore that lost
the registry, or a dropped table (`DROP TABLE`, `untrack` and `DROP INDEX` forget the registry and
leave Pinecone alone). Every wipe lists the family first: `pinecone.sync('docs')`, a `promote` that
declares the partition column, and the queued wipe behind TRUNCATE and a rebuild. Each orphan is named
in a WARNING and registered, so the wipe empties it; the load that follows brings it back only if the
table has rows with that value. So a table dropped, re-created and promoted again starts clean: no
search of a tenant returns an id of the old table. A name no write of the table makes
(`<namespace>:abc` on a bigint column) is reported and left alone: it may be another table's.
`status()` makes no request, so it does not show orphans: `sync` and `backup_readiness` do. A document
index cannot be listed: `backfill_docs(.., clear => true)` and `untrack_docs` also empty the
namespaces of the values the table holds, which catches a drain that rolled back.

**One database per index prefix.** The family is everything under `<namespace>:` in the index,
whoever wrote it. Two databases (or clusters) that share an index and keep the default names
(`pv-<schema>-<table>`, `<schema>.<table>`) see each other's tenants as orphans: a `sync` or a TRUNCATE
in one empties the other's namespaces (a WARNING names the first 20). Give each database its own
`index_name` (or `namespace`).

## Upgrading from 0.10.0 and later

In place: no drop, nothing re-sent to Pinecone, tables keep their indexes and queues. The only downtime is the restart. (From `pinegres`: [below](#upgrading-from-pinegres-090-or-earlier).) On every node, per Postgres major:

1. **Install the new files**: the tarball's library into `pg_config --pkglibdir`, its `.control` and every `.sql` (the install script and the `pg_pinecone--<old>--<new>.sql` upgrade scripts) into `$(pg_config --sharedir)/extension`, or `cargo pgrx install --release`. Delete the old library file first: the running server keeps it mapped, and overwriting it in place (`cp`, `cargo pgrx install`) can crash it (`install -m 755` replaces it safely).
2. **Restart Postgres** to load the new library. Its worker idles, with one WARNING per database, until step 3 (`worker_version()` shows `compiled` newer than `installed`). Until then the new library serves the old catalog (releases keep that safe, see [contributing](contributing.md#schema-changes-and-upgrade-scripts)), except that a function the release removed errors: do step 3 right after the restart.
3. **`ALTER EXTENSION pg_pinecone UPDATE;`** as a superuser, in every database that has `pg_pinecone`. Postgres chains the scripts, so 0.10.0 to 0.12.0 works. The worker resumes within a naptime and drains what queued meanwhile:

   ```sql
   SELECT * FROM pinecone.worker_version();   -- compiled = installed, idle = f
   ```

   `compiled` older than `installed` after the restart means the old library file is still in place: redo step 1.

**A native index of an older on-disk format is offline from the restart until `REINDEX`.** A release that changes the format says so in its changelog (0.10.0 writes format 3; under `pg_pinecone` nothing older exists, so this first applies at the next format change): a native index built before it cannot be read by the new library. Per entry point: the planner stops choosing it (one WARNING per backend, naming the index); a scan that is forced, and every `INSERT` into its table, fail with `pg_pinecone index '...' has format version 1 (this build reads 3): REINDEX it` (SQLSTATE 55000); `VACUUM` skips it with a WARNING and completes; `pinecone.index_format()` says `readable = false` and never raises. `ALTER EXTENSION pg_pinecone UPDATE` prints a WARNING listing these indexes. Rebuild them right after the update:

```sql
SELECT * FROM pinecone.reindex_native_indexes(dry_run => true);   -- (index_name, 'stale'): what it would rebuild
SELECT * FROM pinecone.reindex_native_indexes();                  -- REINDEX INDEX for each, as the caller: 'rebuilt'
SELECT * FROM pinecone.index_format('docs_idx');                  -- 3 | t
```

An index dropped or renamed while the call runs (a rebuild takes minutes) comes back `skipped` and the others are rebuilt; any other error aborts the call. The helper is one transaction: each rebuilt index keeps its `AccessExclusiveLock` (and the table a `ShareLock`) until the call returns, so writes to a table with a rebuilt index stay blocked for the rest of the run. To release them one index at a time, run the statements separately (in `psql`, `\gexec` runs each row as its own statement):

```sql
SELECT format('REINDEX INDEX %s', index_name) FROM pinecone.reindex_native_indexes(dry_run => true) \gexec
```

A REINDEX is blocking and rebuilds from the table: about 4 minutes per million 768-d rows (the 0.9.0 box build took 235 s). The new index is calibrated for `target_recall` instead of probing 127 lists and rescoring 1,000 rows at 1M rows (a NOTICE says what it chose and `pinecone.index_calibration()` shows it: [reference](reference.md#native-engine-calibrated-defaults)); the calibration adds about 2 s per million 768-d rows to the rebuild with eight workers. Writes to a table with a native index fail from the restart until that index is rebuilt, so run the update and the REINDEX together, one table at a time with `REINDEX INDEX` if the writers cannot all wait. Remote-mode indexes carry no codes and are unaffected: do not REINDEX them for this (it empties and refills the Pinecone namespace).

Do not update before the restart. With `pg_pinecone` in `shared_preload_libraries` the session still has the old library mapped, so a release that adds a function fails to update (`could not find function "..._wrapper" in file ".../pg_pinecone.so"`) and the update rolls back; restart and run it again. Check `status()`, `backup_readiness(tbl)` and one query per lane afterwards. CI runs install, restart, update against the last release on every PR (`upgrade-test`: all lanes against a fake Pinecone, in the window between restart and update too, catalog identical to a fresh install).

## Upgrading from pinegres (0.9.0 or earlier)

There is no upgrade path: 0.10.0 renamed the extension (`pinegres` became `pg_pinecone`; schema, settings and access method `pinecone`) and Postgres cannot rename an installed extension. Reinstall: `DROP EXTENSION pinegres CASCADE` (your tables, rows and vector columns stay; the registry, queues, sync triggers and `USING pinegres` indexes go, and nothing is sent to Pinecone), install `pg_pinecone`, replace `pinegres` by `pg_pinecone` in `shared_preload_libraries`, restart, `CREATE EXTENSION pg_pinecone`, then set up each table again as for a new install (`pinecone.promote`, `pinecone.create_index`, `pinecone.track_docs`). A promoted or full-text table reloads into Pinecone from the table. From 0.10.0 on, upgrades are [in place](#upgrading-from-0100-and-later).

## See also

- [README: the consistency contract](../README.md#the-consistency-contract) — what `async` buys
  and costs.
- [`reference.md`](reference.md) — every GUC, function support matrix, and error message.
