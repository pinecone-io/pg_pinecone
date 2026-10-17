# Reference

Full detail for everything the [README](../README.md) and [runbook](runbook.md) summarize:
every GUC, every function's mode/engine support, the error-message table, and known limits.

## Configuration

| GUC | Default | |
|---|---|---|
| `pinecone.api_key` | *(empty)* | Session-only API key. |
| `pinecone.control_plane_url` | `https://api.pinecone.io` | Base URL of the Pinecone control plane, paired with the session GUC key above. |
| `pinecone.timeout_ms` | `5000` | Timeout, in milliseconds, for each control- and data-plane request. |
| `pinecone.source_tag` | *(empty)* | Attributes a partner packaging's share of pg_pinecone's Pinecone usage — see "What pg_pinecone sends" below. Normalised to lowercase `[a-z0-9_ :]` (spaces collapse to `_`); rejected only if nothing is left. |
| `pinecone.sync_mode` | `async` | Default `pinecone.tables.sync_mode` a `pinecone.promote()` call with no explicit `sync_mode` argument uses, for a genuinely new promotion. |
| `pinecone.outbox_max_attempts` | `10` | Consecutive permanent-class dispatch failures a table's outbox queue (or a scheduled maintenance REINDEX) tolerates before parking. |
| `pinecone.worker_enabled` | `on` | Whether the launcher spawns per-database background workers at all. `SIGHUP` — `ALTER SYSTEM SET` + reload, not a plain `SET`. |
| `pinecone.worker_naptime_ms` | `2000` | Milliseconds the launcher and every per-database worker sleep between rounds. A worker does not sleep while a backfill, a partition rebuild or a `recover_from_remote` job has chunks left: it naps when its queues are idle or what is left is blocked (a table in backoff or parked, a failed partition). `SIGHUP`. |
| `pinecone.backfill_round_ms` | `1000` (`50`–`60000`) | Milliseconds a worker round spends running backfill chunks back to back, at least one: document chunks (live changes first) and the chunks of vector tables' backfills (an access-method index by heap block, `promote` by primary key; see [Backfill rate](#backfill-rate)); one chunk is one request of at most 1,000 rows or 1.9 MB of vectors. The budget is the round's, shared by the tables of a vector round: the table that leads runs chunks until it is spent, each other table with a job runs one, and the lead moves on every round. The next round starts at once while a job has chunks left, so the value is how often live changes, other tables and DDL get a turn, not how fast a backfill goes. A vector round holds its tables' locks until it commits, for up to that plus one chunk per table with a job (DDL on a table waits for it), and a stop request (SIGTERM) ends it after its current chunk. It also bounds how long one round of a partitioned table's `recover_from_remote` keeps walking namespaces, and how long a round rebuilds retried partitions. The worker uses the value of its own session: the cluster's (`postgresql.conf` or `ALTER SYSTEM`, then reload), unless an `ALTER DATABASE ... SET` of its database or an `ALTER ROLE ... SET` of the bootstrap superuser it runs as overrides it (those are read when the worker starts); a `SET` in your session only changes what `pinecone._docs_round` does. |
| `pinecone.databases` | *(empty = scan `pg_database`)* | CSV list of databases the launcher spawns workers for. `SIGHUP`. |
| `pinecone.am_enabled` | `on` | Planner kill switch for `pg_pinecone`-AM indexes — `off` reports `disable_cost`, so the planner always falls back to a seqscan. |
| `pinecone.am_min_topk` | `100` | Smallest `topK` a remote-mode AM index scan ever sends Pinecone, also its first query's `topK`: a `LIMIT` up to it is one request, a larger one asks again with `topK` x4 (see [Iterative scan](#iterative-scan)). No `LIMIT` reaches an index access method, so the first request cannot be sized to it. 0.9.0 sent 50: `SET pinecone.am_min_topk = 50` restores it (about 0.3 to 0.9 ms less per query at a `LIMIT` of 50 or less, one more request at `LIMIT` 51 to 100). |
| `pinecone.am_max_topk` | `10000` | Largest `topK` a remote-mode AM index scan may over-fetch to past a selective `WHERE`. The last regrow is clamped to it; a scan that needs more errors (`iterative_scan = off`, what a remote index runs by default) or ends quietly (`relaxed_order`, `strict_order`). |
| `pinecone.am_cloud` | `aws` | Cloud provider for the serverless index a fresh remote-mode `CREATE INDEX`/`pinecone.create_index(engine => 'remote')` build creates. |
| `pinecone.am_region` | `us-east-1` | Region for the serverless index a fresh remote-mode build creates. |
| `pinecone.auto_maintain` | `off` | Whether the background worker automatically `REINDEX`es a native-engine index once `pinecone.schedule_maintenance`'s drift threshold is reached (an index below format 3 counts as drift 1.0) — see [the runbook's local-engine maintenance section](runbook.md#local-engine-mode--local-maintenance). |
| `pinecone.build_parallelism` | `-1` (auto: `min(8, cores/2)`) | Worker-thread count for a native-engine `ambuild`'s k-means training/assignment. `0` forces fully serial (for determinism cross-checks); every value produces byte-identical index pages for the same `(seed, data)` on one machine (the distance kernels differ by instruction set in the last bits, so a near-tie can fall the other way on another CPU). |
| `pinecone.rescore_multiplier` | `0` (no override) | Overrides every native-engine index's rescore pool (this times 100 rows) for scans in this session. The pool is the most rows a scan rescores: `iterative_scan = off` rescores all of it, the first batch of `auto` fewer when the estimator's bound shows the rest cannot matter ([early stop](#native-engine-blocked-scan-and-error-bound-early-stop)). Without it a scan uses the index's `rescore_multiplier` reloption if one was set, else the pool its build calibrated, else 1,000 rows (a scan with a pushed key: 1,000 rows unless the reloption pins one). See [Calibrated defaults](#native-engine-calibrated-defaults). |
| `pinecone.scan_prune` | `on` | Whether the first batch of an `auto` scan stops rescoring early (see the [early stop](#native-engine-blocked-scan-and-error-bound-early-stop)); `off` rescores its whole pool, as 0.9 did. An explicit `iterative_scan = off`, `relaxed_order`, `strict_order` and what `auto` goes on with never stop early. |
| `pinecone.scan_fraction` | `0.0` (no override) | Overrides every native-engine index's IVF probe fraction (`nprobe / n_lists`) for scans in this session. Without it an unfiltered scan probes the index's own `nprobe`, calibrated at build ([defaults](#native-engine-calibrated-defaults)); filtered (mode-H) scans auto-ramp to `min(0.5, 16 / sqrt(n_lists))`, since in-loop filtering makes extra probed lists nearly free. That ramp is already `0.5` up to about 1,000 lists (1M rows), so for a scan with a pushed key `0.5` there changes nothing and only a higher value (up to `1.0`) probes more lists. `relaxed_order`, `strict_order` and what `auto` goes on with use `ceil(4 x sqrt(n_lists))` lists. |
| `pinecone.mode_l_threshold` | `0.03` | Estimated filter selectivity at or below which a filtered native-engine scan uses mode L (filter-first exact scan, recall 1.0) instead of mode H (IVF probe + in-loop filtering). |
| `pinecone.force_scan_mode` | `auto` | Forces every filtered native-engine scan's mode: `auto`, `l`, or `h`. |
| `pinecone.iterative_scan` | `auto` | `auto`, `off`, `relaxed_order` or `strict_order`: whether a `pg_pinecone`-AM index scan keeps fetching past its first batch while the executor still wants rows (a `WHERE` the index cannot push, or a `LIMIT` above the first batch). `auto`: a native index returns `off`'s first batch with the early stop and goes on as `relaxed_order` only if the executor asks for more, a remote index runs `off`. `off` stops after the first batch: on a native index the whole rescored pool. Prefer `relaxed_order` and re-sort its few rows in a CTE for an exact order; `strict_order` drops rows to keep the order, so it may return fewer rows and can rescore up to `pinecone.max_scan_tuples` rows for a handful. Not used by `engine => 'pgvector'` indexes. See [Iterative scan](#iterative-scan). |
| `pinecone.max_scan_tuples` | `20000` (min `1`) | Rows an iterative scan may fetch before it ends quietly: exact-rescored rows on the native engine (the first batch is never cut, a later slice is cut to what the cap leaves; the scan reads more lists only after rescoring the ones already probed, so from about 85,000 rows the default ends it first, see [Iterative scan](#iterative-scan)), rows requested from Pinecone on the remote one (also bounded by `pinecone.am_max_topk`). |
| `pinecone.scan_factor` | `0` (not sent) | Dedicated Read Node query-time tuning: fraction of IVF partitions scanned, `0.5`-`4.0`. Sent on every remote `/query` (AM and `pinecone.topk`); ignored by an on-demand index. |
| `pinecone.max_candidates` | `0` (not sent) | Dedicated Read Node query-time tuning: candidates reranked with exact distance, `>= 1`. Same reach/behavior as `pinecone.scan_factor`. |

`pinecone.worker_enabled`/`worker_naptime_ms`/`databases` only take effect with
`shared_preload_libraries = 'pg_pinecone'` — see the [README's install section](../README.md#installing).

### Native engine: RaBitQ codes

Since 0.10 a native index stores each vector as a RaBitQ code (Gao and Long, SIGMOD 2024; implemented clean-room from the paper): one bit per dimension of the column padded up to a multiple of 64 and to at least 256 (`D`: 768 bits, 96 bytes, at 768 dimensions; 256 bits, 32 bytes, at 24 or 128), taken from the vector's offset to the centroid of its list after a seeded random rotation. A record is the code, three floats (`rnorm`, `dot`, `rc`: 12 bytes), the code's popcount (`pb`, 2 bytes) and the heap TID: 116 bytes at 768 dimensions, 125 with a filter column; 52 and 61 bytes up to 256 dimensions (0.9.0 stored a sketch of 512 bits up to 256 dimensions, 1,024 above). The pages are on-disk format 3, page layout 1 (see [Index format version](#index-format-version)).

A scan rotates the query once, quantises it to 4 bits per coordinate once per probed list, and scores the records of a list a block at a time with table lookups over their codes (AVX-512, AVX2 or NEON, else scalar, chosen when the backend first scans; [blocked scan](#native-engine-blocked-scan-and-error-bound-early-stop)). The estimate is unbiased and comes with an error bound: the scan keeps the best `pool` records by estimated distance ([calibrated](#native-engine-calibrated-defaults) per index) and rescores them exactly (in the first batch of `auto`, best estimate first, stopping once the bound shows the rest cannot enter the top rows).

- **L2** estimates the squared distance from the list's centroid. For **inner product and cosine** each probed list uses whichever of `q - c` and `q` has the smaller norm, because the estimate's error scales with that norm. A cosine index encodes `x / |x|` and scores `q / |q|`.
- The `k_bits` reloption is still accepted and does nothing (the codes are `D` bits); `CREATE INDEX` and `REINDEX` say so in a NOTICE when it is set. `seed` seeds the rotation as well as k-means; the same `(seed, data)` build the same pages at any `pinecone.build_parallelism`.
- A vector with a NaN or infinite component (pgvector stores none) is refused at build and at insert with an error (the build names the heap tuple). A finite one whose square overflows single precision (components near 1e20) is stored: its estimate is NaN, which ranks it last in the pool.
- A native index built before 0.10 holds the old sketch and is offline until `REINDEX`: [`pinecone.reindex_native_indexes()`](#index-format-version).
- **Minimum code width.** A column of fewer than 256 dimensions is zero-padded to 256 and gets a 256-bit code: 150 records to a list page. The estimator's error shrinks with the square root of the code width whatever the data's dimension, and the rotation spreads the padding over all 256 bits, so a short column ranks like a 256-d one. On a 64-bit code a pool of 300 over every list read recall@10 0.919 on a synthetic 64-d mixture (20,000 rows, 50 weakly separated blobs) and 0.870 on 24-d uniform data behind a 1-in-333 residual `WHERE`; with 256 bits it reads 1.000 and 0.925. Format 3 is unreleased, so the wider code needed no format bump.
- **Known limitation: inner product over offset data.** The IVF lists of an inner-product index are assigned by maximum inner product, so data with a large common offset collapses into few lists (one list held 16,384 of 20,000 rows of a synthetic 256-d set with a norm-10 offset), and recall at a given probe count is lower. Prefer a cosine index, or centre the data.

### Native engine: blocked scan and error-bound early stop

**Blocks.** A list page stores its codes in blocks of 64, 32 or 16 records, byte-interleaved: one vector load reads the same byte of every record of a block. A 64-record block of 768-d codes is 6 KB, so a page holds one block, and a block of 1,024-d codes would not fit a page at all. The block is therefore the widest of 64, 32 and 16 records that fits the most records on a page. Block width, records a page, and records a page before blocks: 64 / 64 / 71 at 768 dimensions, 32 / 32 / 38 at 1,536, 16 / 48 / 55 at 1,024, 16 / 16 / 29 at 2,048, 32 / 150 / 163 at 256 and below. The last block of a page is zero-padded (`n_in_page` counts live records) and the code area starts on a cache line of the page buffer. VACUUM compacts a page and rebuilds its blocks, an insert rewrites the page. Indexes built before this layout are refused with `REINDEX` ([Index format version](#index-format-version)); a native index is about 10 % larger at 768 dimensions.

**Scan.** For each probed list the 4-bit query is expanded into one 16-entry table per four coordinates; the kernel then adds one table entry per nibble of a record's code, 16 to 64 records at a time. The sum is the integer the four AND+popcount passes gave, so estimates, keys and bounds are bit-identical to the one-record-at-a-time path (the build's calibration still uses that path). On one core it takes 2.0 (AVX-512), 2.7 (AVX2) and 2.4 (NEON) ns a record at 768 dimensions against 14.4, 20.5 and 7.4 ns (Ice Lake and M4 Max, `bench/vector/kernels`). The scan keeps the `pool` best candidates by estimate in a bounded buffer: once it is full, a block none of whose estimates beats its worst costs no candidate work. A pushed key (mode H) is tested per block from the stored filter values, and a block without a qualifying record is not scored.

**Early stop** (the first batch of `auto` only). The rescore walks the candidates best estimate first, 32 at a time, and keeps `tau`, the `k`-th best exact distance found so far. A candidate is fetched only if its bound is below `tau`; the walk ends when none of the rest is. The bound is the estimate moved by twice the estimator's own 1.9-sigma error (3.8 sigma): a candidate that belongs in the top rows is lost only if its estimate is off by more than that, about 7 in 100,000 under the estimator's error model. At the estimator's own bound, up to 16 % of the queries (1.9 % of all runs) on five synthetic sets (20,000 rows, pools of 100, 200 and 300) lost a neighbour of the unpruned pool's top 10; at 1.5 times it 3 of 9,000 runs did, at twice it none. `k` is a tenth of the pool, at least 10, so `rescore_multiplier = 10` still serves a `LIMIT 100`; the pool stays the cap. The walk goes on until at least `min(pool, 100)` rows are found (a candidate whose heap row is gone does not count), so a `LIMIT` up to 100 is filled as before. What is certified is the pool's top `k`: the ranks after it are the best of the rows the walk fetched, in exact order, not the pool's top (20,000 x 256-d inner product, 60 queries, `LIMIT 100`: other rows than the pool's top 100 on 59 queries at a pool of 200, 57 at 300 and 24 at 500).

The first batch of an `auto` scan returns the rows it rescored, sorted by exact distance. That is fewer than the pool when the bound allows, so a `LIMIT` above it, or a residual `WHERE` that discards some of the rows, leaves the executor asking for more, and `auto` goes on ([Iterative scan](#iterative-scan)). An explicit `iterative_scan = off` has nothing after its one batch, and the index cannot see a residual `WHERE`: it rescores the whole pool, as 0.9.0 did, and returns every row of it (`SET pinecone.scan_prune = off` does the same for `auto`'s first batch). How much the walk saves depends on the data and grows with the pool: on four synthetic sets (20,000 rows, `LIMIT 10`) a pool of 300 rescored on average 100, 100, 99.7 and 48 % of its rows, a pool of 1,000 rescored 77, 71, 61 and 32 % (768-d cosine mixture, 200-d unnormalised cosine, 128-d unnormalised L2, 256-d inner product with a large offset). The calibrated pool of those sets is the 100-row floor, which is never pruned: the scan phase is where such an index gains. The pool is the cap in every case.

An explicit `off`, `relaxed_order` or `strict_order` scan never stops early, and neither does what `auto` goes on with: a batch that left candidates out would hand them over in a later batch, after rows that are farther (and `strict_order` would drop them), so every batch is rescored whole, as before. Only the first batch of `auto` stops early: it is `off`'s batch with the early stop, and the continuation takes up the rest. `strict_order` and the `max_scan_tuples` cap work on the rows rescored, as before.

`pinecone.last_scan_stats()` reports the scan phase of the last scan in the session: `scanned` (records scored: live, and passing a pushed key), `admitted` (those that entered the candidate buffer), `blocks_skipped` (blocks that gave it none), next to `rescored` (heap fetches). The three are zero for a flat or mode-L scan.

### Native engine: calibrated defaults

A clustered native index picks its default probe count and rescore pool when it is built, instead of carrying constants. `CREATE INDEX` and `REINDEX` search a fixed grid for the cheapest `(nprobe, pool)` whose recall@10 reaches `target_recall` (plus a safety offset, below), keep the pair and what it measured in the index, and say what they chose:

```
NOTICE:  pg_pinecone: calibrated for recall@10 0.95: probing 9 of 142 lists and rescoring 100 rows reached 0.970 on 200 sample queries (0.04 s)
```

All of it runs on the build's data in memory, before the first page is written:

1. **Sample.** `min(200, 2 % of the rows)` rows of the table, seeded by the index's `seed`, at least 50 and fewer than 200 above a million rows (the exact pass below never costs more than 200 queries over a million rows), stand in for queries. A row is left out of its own search (leave one out): it is neither its own neighbour nor a candidate. With [`calibration_queries`](#native-engine-calibrated-defaults) the sample is instead up to 200 of your own queries, none left out.
2. **Truth.** The exact top 10 of every sample query over the table, by the distance the rescore uses (rows tied with the tenth count as neighbours), split across `pinecone.build_parallelism` workers.
3. **Search.** Probe counts 8, 12, 16, 25, 37, 50, 75, 100, 150 and 200, then 300, 400, 600, 800, 1,200 and so on up to `n_lists`, and pools of 100, 200, 300, 500, 1,000 and 2,000 rows (the smallest is the `LIMIT` a scan's pool is sized for: a smaller one would underfill `LIMIT 100`). A pair costs `nprobe x rows per list + 100 x pool`: the codes scanned plus 100 per rescored row (a heap fetch and an exact distance cost 55 to 170 codes' worth on the box, `STORAGE PLAIN` to TOAST). Pairs are tried in increasing cost and the first one whose sample recall, less two standard errors, reaches `target_recall + 0.01` wins (the two standard errors: a fresh query set from the same distribution then misses the target about one time in forty; the 0.01 offset: the sample reads above queries from outside the table, see below). The offset is one constant for every data set and stops halfway to 1.0, so `target_recall = 0.99` asks for 0.995. If no pair reaches it, the cheapest that reaches `target_recall` itself wins. Each probe count is scanned once, with the scan's own estimator over the build's codes, and serves every pool: the rescore is exact, so recall at pool `P` is the share of the true neighbours among the `P` best estimates. A probe count that cannot cost less than the pair already held is not scanned.
4. **Refine.** The winner's probe count is bisected back toward the grid value below it (at most four more scans of the sample, each held to the same margin), so the default can sit between grid values: 14 lists rather than 16.

- **`target_recall`**: reloption, default `0.95`, `0.5` to `0.999` (a value outside is refused with SQLSTATE 22023). It applies at the next `CREATE INDEX` or `REINDEX`; `ALTER INDEX .. SET` alone changes nothing.
- **`calibration_queries`**: reloption, text, default empty: `'schema.table'` or `'schema.table(column)'`, a table, view or materialized view of real queries to calibrate on instead of rows of the indexed table (below); a foreign table is refused (a remote that does not answer would stall the build), and a view over the indexed table passes the check but calibrates on its own rows (do not). The schema is required (a build runs with a restricted `search_path`, which the functions a view calls see too); the column defaults to the one `vector` column of the indexed dimension (name it when several match or the column is a bare `vector`); identifiers fold to lower case unless double-quoted. Like `target_recall` it applies at the next `CREATE INDEX` or `REINDEX` (`ALTER INDEX .. SET (calibration_queries = '..')` alone changes nothing; `RESET` drops it). It is ignored by a flat index and a table below 500 rows, which are not calibrated (a NOTICE says so), and by remote-mode indexes.
- **`pinecone.index_calibration(index)`**: one row: `nprobe` and `pool`, what an unfiltered scan uses by default; `calib_nprobe`, `calib_pool`, `calib_recall` (the sample's mean recall@10 at that pair; a table row is left out of its own search), `calib_samples`, `calib_reached` and `target_recall`, what the build measured; `queries_source`, what the sample was made of (`rows` of the table, `table` for a `calibration_queries` sample, `fallback` when that table could not be read at the last rebuild and rows were used; `NULL` when nothing was measured) and `queries_count`, the queries drawn from the table (`calib_samples` when `queries_source` is `table`, else 0). Needs `SELECT` on the table; works on any `pg_pinecone` index.

```sql
SELECT * FROM pinecone.index_calibration('items_idx');
--  nprobe | pool | calib_nprobe | calib_pool | calib_recall | calib_samples | target_recall | calib_reached | queries_source | queries_count
--       9 |  100 |            9 |        100 |         0.97 |           200 |          0.95 | t             | rows           |             0
```

- **Calibrating on real queries.** `calibration_queries` names a table (or view) of query vectors, embedded with the model of the indexed column and never the queries you evaluate recall with. A build draws up to 200 distinct queries (equal vectors count once; seeded by `seed`; a table below 200 is used whole; at least 50, the floor of the row sample), takes the exact top 10 of each over the whole indexed table with nothing left out (so the queries must not be rows of that table: naming it is refused), and runs the search, offset and refinement above unchanged. On wiki_dpr 1M x 768-d (six builds, seeds 42, 1, 2, 3, 4 and 5; fresh recall@10 on 500 held-out queries): calibrated on rows 0.913 to 0.958, two of six below 0.95 and the sample reading 0.021 above the fresh recall (median); calibrated on 200 held-out vectors 0.965 to 0.981, six of six, the sample 0.005 above (median), at 15 to 22 probed lists instead of 8 to 14.

  ```sql
  CREATE INDEX items_idx ON items USING pinecone (embedding vector_l2_ops)
      WITH (mode = 'local', calibration_queries = 'public.real_queries');
  -- NOTICE:  pg_pinecone: calibrated for recall@10 0.95: probing 15 of 1000 lists and rescoring 100 rows reached 0.972 on 200 queries of public.real_queries (1.46 s)
  ```

  The table is read as the owner of the indexed table, whoever runs the statement (row-level security applies to that role): it needs `USAGE` on the schema and `SELECT` on the column, and a superuser's `REINDEX`, `pinecone.reindex_native_indexes()` and the maintenance worker check the owner's privileges first, so they never read a table the owner cannot. It is read at `CREATE INDEX` and at every rebuild of a clustered index (`REINDEX`, `VACUUM FULL`, `CLUSTER`), under a share lock taken without waiting (a lock its indexes or TOAST need is waited for 2 s at most) and given back when the read ends. The read sorts every row of the table, so a table or view of more than 100,000 rows is refused (the planner's estimate, or a count when there is none: a table never vacuumed or analyzed, a partitioned table, a view): name a view over a sample. It runs in a subtransaction, so the build drops out of `pg_stat_progress_create_index` once the queries are read. A table that cannot be used at `CREATE INDEX` is an ERROR that names the problem (the table, the column, a column that is not a `vector` of the indexed dimension, two candidate columns, fewer than 50 distinct queries, a query of another dimension, a missing privilege, a table another transaction holds exclusively; [errors](#errors); `ALTER TABLE .. ALTER COLUMN .. TYPE` on the indexed column re-creates the index like `CREATE INDEX`, so an unusable table fails it too). At a rebuild the same problems are a WARNING and the build calibrates on table rows, so a dropped, renamed, emptied or unreadable table never fails a `REINDEX` or the worker; `queries_source` reads `fallback` until a rebuild finds the table again. `pg_dump` writes the option into the `CREATE INDEX`, which a restore runs before it restores GRANTs and refreshes materialized views, so a restore fails that build when the owner of the indexed table reads the queries through a GRANT or the queries are a materialized view; a parallel `pg_restore -j` can also build the index before the queries table is loaded (it does not know the dependency), which fails on too few queries. Restore without `-j` when the queries are a table (or a view over tables) the indexed table's owner owns, in `public` or a schema that role owns; otherwise create the index after the restore.
- **Nothing to calibrate.** A flat index (a table below `small_n_threshold`) scans every row and reports `NULL`s. A clustered index on a table of fewer than 500 rows is not calibrated (a sample of 50 would be a tenth of it): it keeps the 0.9.0 constants, `ceil(4 x sqrt(n_lists))` lists and 1,000 rows, `calib_samples` is `0` and `nprobe` and `pool` show the constants. An index built by 0.9.0 is offline until `REINDEX` and reports `NULL`s too.
- **A target out of reach.** If no pair reaches the target (offset or not), the build keeps the cheapest pair within 0.005 of the best recall it saw, `calib_reached` is `false` and the NOTICE says so. Data with no cluster structure at the index's list count, or a `rescore_multiplier` that pins the pool too low, can do this; the best pair can then be a near-exhaustive scan (300,000 uniform 768-d vectors: all 548 lists and a pool of 2,000, recall 0.928), so lower `target_recall` to trade recall for speed.
- **Overrides.** A `rescore_multiplier` reloption (default `0`: calibrate) pins the pool in rows (`x 100`); the build then calibrates the probe count for that pool. `pinecone.scan_fraction` and `pinecone.rescore_multiplier` win over the index on every scan path, per session (the planner's cost estimate leaves them out).
- **Not calibrated, kept as they were.** A scan with a pushed key (mode H) rescores the 0.9.0 pool of 1,000 rows unless a pool is pinned, and an iterative scan (`relaxed_order`, `strict_order`, and `auto` once it goes on past its first batch) probes the 0.9.0 list count: see [below](#native-engine-probe-count-pool-and-storage-plain).
- **What the sample cannot know.** By default calibration uses rows of the table as its queries, so it measures queries that look like the rows, and a query set unlike the data (question embeddings against passage embeddings, a few busy tenants) can land below the target. On the box, rows of wiki_dpr passages read 0.01 to 0.04 above held-out queries (0.06 at eight of 1,000 lists); the 0.01 offset covers a part of that, not all. Calibrate on a sample of the real queries with `calibration_queries` (next bullet), or measure recall@10 of real queries against the exact answers, sweep `pinecone.scan_fraction` and `pinecone.rescore_multiplier` with `SET`, and pin the pool with `ALTER INDEX .. SET (rescore_multiplier = N)` and a `REINDEX` (the probe count is then calibrated for it), or raise `target_recall`. The target is recall@10, and the calibrated pool is often the 100-row minimum: a `LIMIT 100` needs a bigger one (`rescore_multiplier = 10`, [below](#native-engine-probe-count-pool-and-storage-plain)). Rows inserted after the build do not recalibrate it; when `pinecone.index_drift` says the data has moved, a `REINDEX` does (so does scheduled maintenance).
- **Cost.** 200 sample queries over the table, one distance per row each (fewer queries above a million rows: `200 x rows` is capped at 2e8), plus a scan per probe count tried and up to four more to refine. On an Apple M4 Max (release build, eight workers, 768 dimensions): 0.04 s at 20,000 rows, 0.5 s at 300,000 and 1.7 s at 1,000,000, where the k-means takes 5.6 s and the assignment and encoding about 45 s of CPU. A target out of reach scans every probe count and costs a little more (0.6 s at 300,000 rows of uniform vectors).
- **Minimum code width.** See [RaBitQ codes](#native-engine-rabitq-codes).

### Native engine: probe count, pool and `STORAGE PLAIN`

An unfiltered native scan ranks the index's centroids, probes the `nprobe` nearest lists, keeps the best `pool` rows by estimated distance ([RaBitQ codes](#native-engine-rabitq-codes)) and rescores those exactly from the heap: all of them under `iterative_scan = off`, best estimate first and stopping as soon as the estimator's bound shows the rest cannot enter the top rows in the first batch of `auto` ([early stop](#native-engine-blocked-scan-and-error-bound-early-stop)); the pool is the most either rescores. Both numbers are the index's [calibrated defaults](#native-engine-calibrated-defaults); where nothing was calibrated they are the 0.9.0 constants (`ceil(4 x sqrt(n_lists))` lists, 127 at 1,000 lists, and 1,000 rows). A pool pinned by the `rescore_multiplier` reloption is `rescore_multiplier x 100` rows.

Which scans use them:

- **Unfiltered scans** (`pinecone.iterative_scan = off`, and the first batch of `auto`): both numbers as stored.
- **Filtered scans, mode H (a pushed key):** the pool is the 0.9.0 one, 1,000 rows, unless the reloption pins one, whatever the calibration found: the key runs inside the scan loop, so the pool is cut over candidates that all passed it, and a pool tuned for unfiltered recall@10 is too small there. On the box (1M x 768d wiki_dpr) a 300-row pool took a scan with a residual `WHERE` and `iterative_scan = off` from recall 0.751 to 0.281 behind a 1 % key (3.1 rows of 10 returned), and `relaxed_order` lost 0.02 to 0.05; at 1,000 rows recall equals the 0.9.0 report's to the third decimal. The lists come from the ramp in the `pinecone.scan_fraction` row above, which calibration does not touch.
- **Iterative scans** (`relaxed_order`, `strict_order`, and `auto` past its first batch): the 0.9.0 probe count, `ceil(4 x sqrt(n_lists))` lists, and the index's pool. The residual `WHERE` runs after the pool is cut and the candidates of the first lists decide which rows come back, so fewer lists cost recall: `LIMIT 100` behind a residual that passes 10 % of the rows gave recall@100 0.632 at 48 lists and 0.529 at 25 on a synthetic 20,000-row mixture. The pool does not matter there (0.636 at 1,000 rows), because the scan keeps rescoring until the executor has its rows. The first batch of `relaxed_order` or `strict_order` is `off`'s only when both probe the same lists, so set `pinecone.scan_fraction` for an identical one; the first batch of `auto` always is `off`'s.
- `pinecone.scan_fraction` and `pinecone.rescore_multiplier` win on every scan path (the planner's cost estimate leaves them out).

Calibration targets recall@10 and the pool does not follow the `LIMIT` (none reaches an index access method), so a larger `LIMIT` gets less over-fetch: a calibrated pool is often the 100-row minimum, which leaves a `LIMIT 100` no over-fetch at all (on the box a pool of 300 read recall@100 0.842 and a pool of 1,000 read 0.956). For `LIMIT 100` workloads set `rescore_multiplier = 10`.

**`STORAGE PLAIN` for the vector column.** pgvector declares `vector` `STORAGE EXTERNAL`: a 768-d vector (3,080 bytes) lives in the TOAST table, and each rescored row costs about 5 buffer accesses (heap 0.7, TOAST heap 1.05, TOAST index 3.15) and a detoast, 3.9 ms of the 11.45 ms default query on the box. With the column `STORAGE PLAIN` the vector sits in the heap tuple: fetching 200 rows took 0.62 ms against 1.1 to 1.3 ms, and the native query at pool 200 and 20 lists fell from 2.96 to 2.29 ms (box, 0.9.0 library). `CREATE INDEX` and `pinecone.create_index(.., engine => 'native')` print a NOTICE with the statement when the column is `EXTERNAL` or `EXTENDED`, the vector is wide enough to be moved out of line (over 500 dimensions) and a PLAIN row would fit a page. They never change the column.

```sql
ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN;   -- new rows
VACUUM FULL items;                                            -- existing rows; ACCESS EXCLUSIVE, rebuilds the index
CREATE TABLE items (id bigint PRIMARY KEY, embedding vector(768) STORAGE PLAIN);   -- or from the start
```

`UPDATE items SET embedding = embedding` leaves the rows out of line (it keeps the TOAST pointer). A PLAIN row must fit one 8 KB page: a `vector(d)` is `4d + 8` bytes, so about 2,030 dimensions beside a `bigint` id and fewer beside other wide columns (the insert then fails with `row is too big`); the notice stays quiet from 1,998 dimensions, and 3,072-d vectors cannot be PLAIN. Below about 500 dimensions the vector is inline anyway.

### Iterative scan

A `pg_pinecone`-AM index scan hands the executor one batch: on the native engine at most the rescore
pool of exact-rescored rows (100 at the calibrated minimum; fewer in the first batch of `auto` when the
[early stop](#native-engine-blocked-scan-and-error-bound-early-stop) ends the walk) from `nprobe` lists,
on the remote engine `topK` = `pinecone.am_min_topk`. A `WHERE` the index cannot push (a range, `IN`,
a second column) is applied after that, so a selective one returns fewer than `LIMIT` rows, and so
does a `LIMIT` above the batch. `pinecone.iterative_scan` lets the scan keep going while the executor
wants rows. The default, `auto`, does so on a native index and runs `off` on a remote one:

| Value | Native engine | Remote engine |
|---|---|---|
| `auto` (default) | The first batch is `off`'s (the index's calibrated lists and pool) with the early stop: its top `pool / 10` rows (at least 10) are `off`'s, and it rescores fewer of the pool when the bound allows. Only if the executor drains it and asks for more, the scan starts over as `relaxed_order` (below) and leaves out the rows it already returned. | As `off`. |
| `off` | One batch: the whole rescored pool (no early stop), so a `LIMIT` above it returns fewer rows, with no error. | Re-queries with `topK` x4 up to `pinecone.am_max_topk` (100, 400, 1,600, 6,400, 10,000 rows at the defaults; the last step is clamped to it), then errors. |
| `relaxed_order` | Rescores the next `pool` candidates ([the index's pool](#native-engine-probe-count-pool-and-storage-plain)) of the probed lists per batch, best estimated distance first (all of them: an iterative scan has no [early stop](#native-engine-blocked-scan-and-error-bound-early-stop)), and probes the next `nprobe` lists (each once) only when none are left; ends quietly at `pinecone.max_scan_tuples` rescored rows or when the lists run out. | The same loop, but ends quietly at `pinecone.max_scan_tuples` rows instead of erroring. |
| `strict_order` | As `relaxed_order`, dropping a row closer than the last one returned. | As `relaxed_order`, skipping a re-queried row that ranks before the last one returned. |

**`auto`.** A query the first batch satisfies, which is nearly every `LIMIT 10`, runs as under
`off`: the same lists and the same rows in the same order up to rank `pool / 10` (at least 10), at the
same cost or less (the early stop rescores fewer rows of the pool, which `rescored` in
`last_scan_stats()` shows). One it does not satisfy (a
`LIMIT` above the pool, a residual `WHERE` that discards most of the batch) pays for a second scan: the
first batch keeps only the best `pool` candidates, so the continuation probes the lists again from the
nearest, with every candidate kept, and `relaxed_order`'s rules apply from there (the 0.9.0 list count, a
pool of candidates a batch, rows ordered per batch only, a stop at `pinecone.max_scan_tuples` rescored
rows: the first batch counts against it and is never cut). The lists of the first batch are read twice
and count twice in `lists_probed`. A first batch that probed every list and fetched every row they hold
starts no second scan. The rows after the first batch are the next best by estimate over the
probed lists, so depth depends on the data: `LIMIT 200` read recall@200 0.93 on 20,000 uniform 6-d rows
(`off`: 100 rows, 0.50), 0.83 on 16-d ones and 0.80 on a 32-d mixture of 200 blobs (`off`: 0.49, 0.48), and
a larger `rescore_multiplier` buys more. A residual `WHERE` that passes few rows is paid for in rescored
rows up to the cap (20,000 at the defaults): lower `pinecone.max_scan_tuples` to bound it, or `SET
pinecone.iterative_scan = off` to stop after the first batch.

The first batch of `relaxed_order` and `strict_order` is exact and sorted by true distance and holds what
`off` returns when both probe the same lists. They probe the 0.9.0 count of lists and `off` the index's calibrated one
([calibrated defaults](#native-engine-calibrated-defaults)): 48 lists against 8 of 142 on a calibrated
20,000-row index, which costs a `LIMIT 10` several times the work, so choose them when the first batch is
known not to be enough. After the first batch, `relaxed_order` is ordered per batch only: a later
batch can return a closer row after a farther one, and in most multi-batch queries it does (1M x 768d
benchmark, 0.3 % residual: 96 % of the queries at `rescore_multiplier` 10, 2 % at 50). `strict_order` keeps the order by dropping those
rows ("ordered but lossy", like pgvector's), so it can return fewer rows than `relaxed_order`. Where
batches overlap in distance it returns little beyond the first batch and still rescores until the
cap or the lists run out: 3 to 4 rows for up to 100,000 rescored rows (352 ms at one client, 742 ms
at eight) in that benchmark. Prefer `relaxed_order` and sort its few rows again for an exact order:

```sql
WITH relaxed AS MATERIALIZED (
  SELECT id, embedding <-> '[1,2,3]' AS distance FROM items WHERE category_id = 123
  ORDER BY distance LIMIT 10
) SELECT * FROM relaxed ORDER BY distance + 0;
```

The batch-wise order reaches every plan node that consumes the index order, not only the final
`ORDER BY ... LIMIT`: an Incremental Sort for a secondary `ORDER BY` column, MergeAppend over
partitions, a merge join, `DISTINCT ON`, window frames and `OFFSET` all take it as exact, so a
later batch's out-of-order rows come out out of order (and `OFFSET` skips the first rows of the
batch-wise stream, not the nearest). Re-sort first (the CTE above) or use `strict_order`.

Native mode-L and flat scans already return every qualifying row and ignore the setting. So do
`engine => 'pgvector'` indexes, which keep pgvector's own `hnsw.iterative_scan` and
`hnsw.max_scan_tuples` (`ivfflat.iterative_scan` for ivfflat). `pinecone.last_scan_stats()`
returns `(batches, rescored, capped, deepened, lists_probed, scanned, admitted, blocks_skipped)` for the session's last scan:
`capped` means the cap, not the lists, ended it; `deepened` counts the native batches that
rescored more candidates of lists already probed; `lists_probed` the native lists read (twice for the
lists of an `auto` first batch the continuation probes again); `scanned`,
`admitted` and `blocks_skipped` count the scan phase (see the [early stop](#native-engine-blocked-scan-and-error-bound-early-stop)). The
counters belong to the session, not to a scan: each scan zeroes them when it computes its first
batch and every later batch of any scan adds to them, so with two pg_pinecone scans in one statement
they describe the scan that started last (in a nested loop its last inner rescan), plus any batch
another scan computed after that.

Batches follow the estimated-distance ranking of the probed lists, which the `WHERE` cannot see, so a
selective one is paid for in rescored rows (up to `pinecone.max_scan_tuples`; raise it if `capped`
shows), not lost to the rescore pool. Recall is still bounded by the lists probed: raise
`pinecone.scan_fraction` for a `WHERE` whose matches lie far from the query.

**The cap and widening.** The scan reads more lists only after it has rescored every candidate of
the lists already probed, that is after about `nprobe / lists x rows` rows: `4 x rows^0.75` at an
iterative scan's fraction (`4 / sqrt(n_lists)`), 126,000 at 1M rows. From about 85,000 rows that is more
than the default `pinecone.max_scan_tuples`, so a scan whose matches sit in other lists ends `capped` before it
reads one. At 1M rows with nothing pushed and 33 matching rows the default cap stopped every query
at 0.64 rows on the first 127 lists; a cap of 1,000,000 filled the `LIMIT` after rescoring 313,000
rows (1.2 s). A pushed key shrinks the first set to its own rows and never reached the default cap
in the four benchmark cells below. The levers are `pinecone.scan_fraction` (more lists in the first
batch) and a `pinecone.max_scan_tuples` of at least the probed set's size, both paid in heap
fetches (3.5 to 4 us per rescored row in the benchmark below, warm cache).

The candidates of the probed lists (12 bytes each: 1.5 MB at 1M rows, 48 MB at 100M with `4 / sqrt(n_lists)`)
are held in backend memory, outside `work_mem`, until the scan ends, and an open
cursor keeps them until it is closed or exhausted.

#### Measured: a pushed key plus a residual `WHERE` (1M x 768d)

Release Postgres 17.10, pgvector 0.8.6, `LIMIT 10`, 500 cross-tenant queries, one run per point. A
key the index pushes (1 % or 10 % of the rows) plus an uncorrelated residual it cannot (0.3 % or
1 %, 33 to 1,004 matches). `rm` is `rescore_multiplier`; times are p50 in ms at 1 / 8 clients.

| Key + residual (matches) | `off`: rows, recall | `relaxed_order`, rm 10: rows, recall, ms | Recall >= 0.95 reached at | Exact seqscan, ms |
|---|---|---|---|---|
| 1 % + 0.3 % (33) | 3.0, 0.30 | 10.0, 0.76, 15.5 / 15.4 | rm 50, `scan_fraction` 1.0: 0.95, 22 / 23 | 17 / 85 |
| 1 % + 1 % (113) | 9.6, 0.75 | 10.0, 0.79, 8.5 / 8.3 | not reached: 0.94 at rm 50 (higher not run), 17 / 21 | 20 / 97 |
| 10 % + 0.3 % (326) | 3.4, 0.32 | 10.0, 0.78, 15.3 / 18.1 | rm 100: 0.97, 34 / 43 | 19 / 90 |
| 10 % + 1 % (1,004) | 8.5, 0.67 | 10.0, 0.79, 11.7 / 12.1 | rm 50: 0.98, 21 / 22 | 22 / 103 |

- `relaxed_order` filled the `LIMIT` in every query of all four cells (`off`: 0 to 76 %) for 0 to 7
  ms more. Recall is bound by the rescore pool (`rescore_multiplier x 100`, cut before the residual)
  and the lists probed, so `rescore_multiplier` 50 to 100 is the lever.
- The exact parallel sequential scan is the bar. pgvector's planner took it in three of the four
  cells (16 to 19 ms at one client, 89 to 102 ms at eight); in the fourth it chose hnsw, which
  returned 0 rows at the default `hnsw.max_scan_tuples`. Forced, hnsw reached 10 rows in two cells
  only at `hnsw.max_scan_tuples` 100,000 (49 and 112 ms); the other two gave 3.3 and 8.3 rows (160
  to 180 ms).
- In the three cells that reach recall >= 0.95, pg_pinecone needs 0.95 to 1.8x the sequential scan's
  time at one client and runs 2 to 5x faster at eight (22 to 43 ms against 85 to 103 ms). The scan
  is cheap here: the vectors sit in TOAST (a 100 MB main heap) and the cache is warm.

### Backfill rate

A vector table's backfill is one queued job: `create_index(.., engine => 'remote')` and `CREATE INDEX ..
USING pinecone` queue it by heap block (ids are TIDs), `promote(.., sync_mode => 'async')` by primary
key. The worker runs it round after round. A round sends the table's queued live writes that precede
the job, then the job's chunks, one request each, until `pinecone.backfill_round_ms` is spent (at least
one chunk). The budget is the round's, shared by the tables that have a job: the table that leads the
round runs chunks until it is spent, each of the others runs one, and the lead moves to the next table
every round. The next round starts at once, with no nap, while a job has chunks left. The worker naps
`pinecone.worker_naptime_ms` only when its queues are idle or what is left is blocked (a table in
backoff or parked, a failed partition). The same holds for a partition rebuild, a `recover_from_remote`
and a document backfill. Live writes queued behind the job wait for it, except for a rebuild, which
yields to them every round. Each chunk's cursor commits with its round: a failed chunk leaves the
cursor after the one before it, and a crash re-sends what its round sent (upserts are idempotent).

A request holds as many vectors as Pinecone allows one: 1,000 records and 2 MB, where Pinecone counts
4 bytes per value, the id and the metadata as JSON, not the size of the JSON that carries them
(Pinecone's [table](https://docs.pinecone.io/guides/index-data/upsert-data#upsert-in-batches) puts 559
records of 768 values and 500 bytes of metadata in one request). pg_pinecone budgets 1.9 MB:

| Dimensions | No metadata | 500 bytes of metadata |
|---|---|---|
| 384 | 1,000 | 915 |
| 768 | 610 | 526 |
| 1,536 | 307 | 284 |

A `promote` chunk reads by primary key and fills its request. An access-method chunk is whole heap
blocks and ends at the last block that fits, so its request is usually 80 to 100% full. A partitioned
index sends one request per namespace and window. The body is compact JSON with each float in its
shortest form, about 11 bytes a value, so a 768-d request is about 5 MB on the wire: a proxy in front
of Pinecone with a lower body limit refuses it, and over a link under about 15 Mbit/s it can outlast
`pinecone.timeout_ms` (5 s).

Rows per second at 768 dimensions, one request at a time:

| | Chunk | Rows/s |
|---|---|---|
| 0.9.0 box, access method, 100k rows, 2 s nap | about 140 rows (a heap block) in 2 requests | 55 |
| 0.9.0 box, `promote`, 100k rows, 2 s nap | about 100 rows in 1 request | 43 |
| 0.9.0 box, partitioned index (100 namespaces), 500k rows, 2 s nap | 5,000-row window, 100 requests of 49 | 193 |
| 0.9.0 box, access method, 1M rows, 100 ms nap | about 140 rows | 220 |
| modelled, 0.10, the 0.9.0 chunks without the nap | about 140 rows | 260 |
| modelled, 0.10, requests of 610 | 610 rows | 260 to 420 |

The 0.9.0 rows are the box's measurements (2026-10-14); the chunks are what the
tests send (99 vectors of 768 values were 2.2 MB of JSON, each float at twice its digits on a line of
its own). The two access-method rates fix the work of a chunk at about 0.5 s (`rows / (nap + work)` is
55 at 2 s and 220 at 0.1 s): the nap was four fifths of every round. The modelled rows are arithmetic,
not a run: the same chunks without the nap, and requests of 610 at the time per vector the box saw (3.8
ms for the access method, 2.4 ms for `promote`), which is 1M rows in 40 to 65 minutes where the 0.9.0
defaults took about 5 hours.

On the mock (`worker_loop_*`, `backfill_size_*`; 20,000 2-d rows, 50 ms a request, rounds of 100 ms)
the trigger lane went from 11 rounds and 10 naps, 21 s, to 11 rounds, 1.3 s; the access method from
19 rounds, 38 s, 37 requests, to 12 rounds, 1.3 s, 22 requests. 6,000 rows of 768 floats are 10
requests on the trigger lane (were 61) and 11 on the access method (were 89).

`backfill_round_ms` no longer needs raising for speed. It is how often live changes, other tables and
DDL get a turn: a vector round holds `AccessShareLock` on its tables until it commits, for up to the
budget plus one chunk per table with a job (a concurrent `ALTER TABLE` or `DROP INDEX` waits for it, and
gets in before the next round), and a stop request (SIGTERM) ends the round after its current chunk.

### What pg_pinecone sends

Every request to Pinecone (control plane, data plane, document-index, embed, rerank) carries a
`User-Agent` identifying pg_pinecone to Pinecone, e.g.:

```
pg_pinecone-client-0.7.0; lang=rust; pg=17; client=0.6.0; source_tag=pg_pinecone
```

That's the extension version, the Postgres major version, the bundled client version, and the
tag — nothing about your data, queries, or identity. Setting `pinecone.source_tag = 'Acme Corp'`
normalises to lowercase and changes the tag to `source_tag=pg_pinecone:acme_corp`, so a partner
packaging (e.g. a managed-Postgres vendor bundling pg_pinecone) can attribute its own share of usage
to Pinecone. `SELECT pinecone.user_agent()` returns the exact current string, e.g. for support.

Key resolution differs by function:

- `pinecone.promote`/`demote`/`sync`/`recover_from_remote`/`repoint`/`backup_readiness`,
  remote-mode `pinecone.topk`, `pinecone.embed`/`embed_batch`/`rerank`, the doc-index functions, and
  `CREATE INDEX ... USING pinecone` (or `pinecone.create_index(engine => 'remote')`) use the session GUC
  if set, else the key stored via `pinecone.set_api_key` — paired with **its own** stored
  `control_plane_url`, never the session's, so a session that doesn't already have the key can't
  redirect it to another server by setting `pinecone.control_plane_url`.
- The background worker and `ambulkdelete` (a remote-mode AM-indexed table's `VACUUM`) always use
  the stored key — never a session GUC, since neither runs inside a session that set one.
  `aminsert` itself makes no HTTP call at all — it only enqueues; the worker pushes later, under
  its own key.

```sql
SELECT pinecone.set_api_key('your-api-key');   -- upserts pinecone.secret (REVOKE ALL FROM PUBLIC)
SELECT pinecone.clear_api_key();               -- deletes it
GRANT SELECT ON pinecone.secret TO app;        -- lets role "app" use the stored key
```

### Grant bundles per verb

Every mutating verb is **ownership-gated in code** (`object_ownercheck` on the table — the same
rule as `REINDEX`; superuser bypasses): `promote`/`demote`/`sync`, `promote_index`/`demote_index`/
`set_engine`, and `schedule_maintenance` all reject non-owners before doing anything. The grants
below are what a **non-superuser table owner** additionally needs, because the extension's own
tables carry `REVOKE ALL FROM PUBLIC`. All bundles assume `USAGE` on schema `pg_pinecone`.

| Verb | Needs on top of table ownership |
|---|---|
| `pinecone.promote` | `SELECT, INSERT, UPDATE ON pinecone.tables`; `EXECUTE ON pinecone._sync_trigger()` (trigger *creation* checks EXECUTE; installed triggers fire for every writer with no grants); async mode also `INSERT, DELETE ON pinecone.outbox` (the coalesced `delete_all`+`backfill` enqueue) |
| `pinecone.demote` | `SELECT, UPDATE, DELETE ON pinecone.tables`; `DELETE ON pinecone.outbox` (queue purge) |
| `pinecone.sync` | `SELECT ON pinecone.tables`; async tables also `INSERT, DELETE ON pinecone.outbox` and `UPDATE ON pinecone.tables` (the `remote -> promoting` flip) |
| `pinecone.retry_failed_partition` / `retry_failed_partitions` | nothing: table ownership is enough, and checked first. The call queues its job through the extension's owner and sends no request, so it needs neither a key nor a grant on `pinecone.outbox` (the worker needs the stored key, like for any async table) |
| `pinecone.promote_index` | `SELECT, INSERT ON pinecone.tables`; `INSERT ON pinecone.outbox` (the TID backfill job); `SELECT, INSERT, UPDATE ON pinecone._am_engine` |
| `pinecone.demote_index` | `SELECT, INSERT, UPDATE ON pinecone.pending_demotes` (INSERT queues it; SELECT/UPDATE are the recovery/unpark re-run paths); `SELECT ON pinecone.tables` |
| `pinecone.schedule_maintenance` | `SELECT, INSERT, UPDATE ON pinecone.index_maintenance` (one upsert; re-running it is also the unpark path) |
| `pinecone.status` (full detail) | `EXECUTE ON pinecone._outbox_stats(oid)` for real `backlog`/`oldest_enqueued`/`last_error`/`queue_state` instead of `NULL` — parked and `ambulkdelete`-failed tables still surface via `pinecone.tables`'s own `parked_reason`/`last_vacuum_error` without it; `EXECUTE ON pinecone._doc_outbox_stats(oid)` for the document lane's queue columns (`docs_backlog`, `docs_oldest_enqueued`, `docs_failed`, `docs_queue_state`) — its registry columns need only `SELECT` on the table, never a grant on `pinecone.doc_tables`. No grant on `pinecone.tables` either: without one the rows come from the registry's owner, only for tables the caller may `SELECT` (a role that holds `SELECT ON pinecone.tables` sees every row) |
| `pinecone.topk` | nothing on `pinecone.tables`: without that grant the registry row is read for the caller, and only for a table they may `SELECT` (42501 otherwise); `remote` mode also needs a key, a session one or `SELECT ON pinecone.secret` |
| `pinecone.track_docs` / `untrack_docs` / `backfill_docs` / `unpark_docs` / `retry_failed_docs` | nothing: table ownership is enough, and checked first (`must be owner of`, SQLSTATE 42501). `track_docs` and `untrack_docs` (with `drop_index`) also call Pinecone, so they need a key like every remote verb: a session key, or `SELECT ON pinecone.secret` |
| `pinecone.flush_docs` / `failed_docs` | nothing beyond `SELECT` on the table (`permission denied for table`, SQLSTATE 42501) |
| `pinecone.search_docs` / `headline` | nothing beyond `SELECT` on the table, which also makes `engine => 'local'` run with the caller's privileges and RLS; `engine => 'remote'` needs a key as above |

The **document lane's tables are closed to every role**: nobody needs, or should hold, a grant on
`pinecone.doc_tables` or `pinecone.doc_outbox`. The document verbs check the caller's right on the
table first, then reach the registry and the queue themselves, as the role that owns the
extension's tables, for those statements only (the Pinecone calls, the stored-key lookup and the
local search still run as the caller). A grant on either table is therefore not part of any
bundle and only widens access, so a table owner can never touch another table's registry row or
queue: no wipe of another tenant's namespace, no registry row that blocks DDL elsewhere.

> **Warning — table DML grants bypass the verb-level gates.** The ownership checks live in the
> *verbs*, not in the queue tables they write. A role granted DML on `pinecone.outbox`,
> `pinecone.pending_demotes`, or `pinecone.index_maintenance` can INSERT rows directly, and the
> background worker — which trusts those queues and runs its rebuilds `SET ROLE` to the *table
> owner* — will execute them: a hand-inserted `pending_demotes` row demotes (blocking REINDEX)
> an arbitrary index, an `index_maintenance` row schedules REINDEXes for it, an outbox
> `delete_all` row empties a remote namespace. Grant these tables only to roles you would trust
> with the verbs themselves against *every* table (i.e. admin roles) — per-table scoping is not
> possible through a table grant.

A demote's rebuild (and a scheduled maintenance REINDEX) runs `SET ROLE` to the table's *current*
owner — that owner also needs `USAGE` on the index's schema and (pre-existing for any
plain-role-built pg_pinecone index) `SELECT, INSERT, UPDATE ON pinecone._am_engine` for the
engine-stickiness bookkeeping.

**Non-owner `pinecone.promote` can spend remote quota before it fails.** `promote`'s ownership
enforcement is the `CREATE TRIGGER` DDL itself, and the remote index is created (and awaited
ready) *before* the triggers are installed — so a non-owner's call aborts locally but the
Pinecone index it created stays (HTTP doesn't roll back), counting against the account's index
quota/billing. Re-running as the owner adopts the existing index (create-or-adopt is idempotent);
if nobody re-promotes, delete it in the Pinecone console. The AM-path verbs
(`promote_index`/`set_engine`) gate ownership *before* any remote call and do not have this
window.

`pinecone.secret` (one row, `REVOKE ALL FROM PUBLIC`) beats `ALTER ROLE ... SET` for a key shared
across writer sessions: the sync-trigger function and the AM's insert path are both
`SECURITY DEFINER`-equivalent, so every writer session (and the background worker) uses the stored
key without ever reading it; everyone else needs the `GRANT SELECT` above. Unlike
`pg_db_role_setting` (what `ALTER ROLE ... SET` uses, readable by all users), that's an ordinary,
revocable Postgres grant.

`pinecone.secret` and `pinecone.outbox` are both excluded from `pg_extension_config_dump` — a queue
isn't data, and a restored database may lag the remote index until `pinecone.sync(tbl)` repairs it.
`pinecone.tables`, the tracking registry, is dumped normally.

## Function x mode support

| Function | Local table | Remote table | AM-indexed table | API key |
|---|---|---|---|---|
| `pinecone.track` | registers it | — (demote first) | errors, names `DROP INDEX` | no |
| `pinecone.promote` | consumes it | produces it | errors, names `DROP INDEX` | stored key required |
| `pinecone.demote` | produces it | consumes it | errors, names `DROP INDEX` | only with `drop_remote_index` |
| `pinecone.sync` | — | yes (a partitioned table: also registers and empties the orphan namespaces of its family) | errors, names `DROP INDEX` | stored key or session GUC |
| `pinecone.recover_from_remote` | errors, not promoted | yes (a partitioned table: per registered namespace) | yes | stored key or session GUC |
| `pinecone.repoint` | errors, not promoted | yes (`mode = 'remote'` only; a partitioned table moves its family) | errors, names `DROP INDEX` | stored key or session GUC |
| `pinecone.backup_readiness` | errors, not promoted | yes (a partitioned table: the whole family) | yes | stored key or session GUC |
| `pinecone.delete_partition` | n/a | yes — a table with a partition column; the namespace of one value | yes — a partitioned index | stored key or session GUC; table owner |
| `pinecone.partitions_of` | n/a | yes — lists the namespaces of a table with a partition column | yes — a partitioned index | no (registry only); table owner |
| `pinecone.failed_partitions` | empty, not promoted | yes — the partitions of a table with a partition column that Pinecone refused for good | yes — a partitioned index | no (registry only); table owner |
| `pinecone.retry_failed_partition` / `retry_failed_partitions` | errors, not promoted | yes — queue the rebuild of a failed partition from the table | yes — a partitioned index (loads by TID) | no (the worker deletes and loads: stored key); table owner |
| `pinecone.detach_storage` | errors, not promoted | yes (`synced_by = 'trigger'` only) | errors, permanent non-goal (never, even with `force`) | stored key or session GUC |
| `pinecone.attach_storage` | errors, not promoted | yes | errors, permanent non-goal | stored key or session GUC |
| `pinecone.untrack` | yes | — (demote first) | errors, names `DROP INDEX` | no |
| `pinecone.topk` | pgvector operators | Pinecone `/query` | errors, names `DROP INDEX` — query with `ORDER BY` instead | only remote (stored key or session GUC) |
| `pinecone.flush` / `outbox_retry` / `outbox_discard` | n/a | yes (`sync` tables are no-ops) | yes — same outbox, keyed by table | no |
| `pinecone.status` | yes | yes | yes | no — registry-only, never fails |
| `pinecone.worker_version` | n/a (no table involved) | n/a | n/a | no — any role; `(compiled, installed, idle)` of the worker version guard, see [Limits](#limits) |
| `pinecone.last_scan_mode` / `last_scan_stats` | n/a | n/a | yes — this session's last scan: native path (`flat`/`h`/`l`) and batch counters, or the remote `/query` count; `last_scan_stats` is new in 0.9 (see [Iterative scan](#iterative-scan)) | no |
| `pinecone.index_drift` / `schedule_maintenance` / `maintenance_status` | n/a | n/a | native-engine only (errors on a remote-mode or non-`pg_pinecone` index; a native index below format 3 answers with a WARNING) | no |
| `pinecone.index_format` | n/a | n/a | yes — any `pg_pinecone` index, native or remote (errors on a non-`pg_pinecone` index); never raises on an old one, see [Index format version](#index-format-version) | no |
| `pinecone.index_calibration` | n/a | n/a | yes — any `pg_pinecone` index (errors on a non-`pg_pinecone` one); the probe count and pool a native index's scans use, what its build measured and what its sample was made of, `NULL` for a flat, remote or old native one; needs `SELECT` on the table, see [Calibrated defaults](#native-engine-calibrated-defaults) | no |
| `pinecone.reindex_native_indexes` | n/a | n/a | native-mode indexes below format 3 whose table the caller owns; see [Index format version](#index-format-version) | no |
| `pinecone.create_index` | n/a | n/a | creates it (`engine` picks `pgvector`/`native`/`remote`; `filter_column` adds a filter key to `native`/`remote`; `partition_column` makes it a namespace per value on `remote`), or attaches an existing Pinecone index (`pinecone_index`, engine `remote` only, not with `partition_column` — see the runbook) | stored key required for `engine => 'remote'` |
| `pinecone.set_engine` / `promote_index` / `demote_index` | n/a | n/a | switches `native <-> remote` on an existing `pg_pinecone`-AM index (a native one below format 3 refuses with the REINDEX hint), or attaches (`pinecone_index`, `native -> remote` only); `set_engine(.., 'remote', partition_column => ..)` declares a [partition column](#partition-column-on-a-remote-index) as the index goes remote | stored key required to reach `remote` |
| `CREATE INDEX ... USING pinecone` | n/a | n/a | creates it (`mode = 'auto'\|'local'\|'remote'`) | stored key required for `mode = 'remote'`/resolved-remote `auto` |
| `pinecone.embed` / `embed_batch` / `rerank` | n/a (no table involved) | n/a | n/a | stored key or session GUC |
| `pinecone.track_docs` / `untrack_docs` / `flush_docs` / `failed_docs` / `retry_failed_docs` / `backfill_docs` / `unpark_docs` | any ordinary table — the document lane is independent of the vector modes, a table can be in both | any | errors on partitioned/inherited tables only | `track_docs`/`untrack_docs`: stored key or session GUC (index create/delete); the worker needs the stored key |
| `pinecone.search_docs` | `engine => 'local'`: any table (Postgres FTS; an untracked one needs `field`), no key; `'remote'`: a table declared with `track_docs` | any | `engine => 'remote'` or `'local'` | `'remote'` only: stored key or session GUC |
| `pinecone.headline` | any table's column (declared language, else `english`) | any | any | no |
| `pinecone.create_doc_index` / `drop_doc_index` / `upsert_docs` / `search_doc_index` | n/a (separate surface) | n/a | n/a | stored key or session GUC |

### Index format version

An access-method index records its on-disk format in its metapage: 0.9 indexes read as format 1, a
`CREATE INDEX` or `REINDEX` by 0.10 writes 3 (RaBitQ codes), and `pinecone.index_format(idx)` returns
`(format_version, readable)` (once Postgres has restarted onto 0.10). `ALTER EXTENSION pg_pinecone UPDATE`
rewrites no index. What an older one does depends on its engine:

- **A remote-mode index** carries no codes: it keeps working at any older format, and `REINDEX` stamps 3 (it
  empties and refills its namespace, as any REINDEX of it does).
- **A native index of an older page layout** (format 3, layout 0: built by a 0.10 development snapshot
  before the code blocks; none was released) is refused like one of an older format, with
  `pg_pinecone index '...' has format version 3 with page layout 0 (this build reads layout 1): REINDEX it`;
  `index_format` reports `(3, false)`. The layout is the second byte of the format word, so the layout
  changes of a format that has not shipped need no new format number.
- **A native index below 3** (every one 0.9.0 built) holds sketch codes this build cannot score, so it is
  **offline until `REINDEX`**, from the restart on. The planner walks away from it (one WARNING per backend
  and index); a scan, forced or chosen, and every `INSERT` into its table raise SQLSTATE 55000,
  `pg_pinecone index '...' has format version N (this build reads 3): REINDEX it`; `VACUUM` skips it with a
  WARNING and completes; `index_drift`, `schedule_maintenance` and `maintenance_status()` still answer, with
  a WARNING; `promote_index` and `set_engine(.., 'remote')` refuse it. `index_format` never raises: it
  answers `readable = false`. `ALTER EXTENSION pg_pinecone UPDATE` prints a WARNING listing these indexes.
- **`pinecone.reindex_native_indexes(dry_run => false)`** runs the `REINDEX INDEX` of each native index below
  3 whose table the caller owns (a superuser: all) and returns `(index_name, outcome)` for each: `rebuilt`,
  or `skipped` if the index was dropped or renamed since the listing (a rebuild takes minutes, so one can
  be; any other error aborts the call); `dry_run => true` only lists them, outcome `stale`. Remote-mode
  indexes are never touched, and a second call returns no rows. A plain, blocking REINDEX per index: about
  4 minutes per million 768-d rows. The call is one transaction, so every rebuilt index stays locked until
  it returns; to release each lock as its rebuild ends, run the statements one by one (in `psql`:
  `SELECT format('REINDEX INDEX %s', index_name) FROM pinecone.reindex_native_indexes(true) \gexec`).
- **A newer format** (a downgrade after a newer release built it) fails every scan, insert, `VACUUM` of dead
  rows, `index_drift`, `schedule_maintenance` and `maintenance_status()` with SQLSTATE 55000, `... (this
  build reads up to 3): REINDEX it`; `index_format` still answers (`readable = false`; `format_version` is
  `NULL` for a block 0 it does not know), and `REINDEX` repairs it.

## Partition column on a vector table

`pinecone.promote(tbl, .., partition_column => 'tenant_id')` maps each value of the column to its own
Pinecone namespace, `<namespace>:<value>` (the namespace defaults to `<schema>.<table>`), instead of
one namespace for the table. A search is bound to one value, so what it costs follows that tenant's
namespace, and a tenant can be removed without touching the others. The access-method lane has its
own [partition column on a remote index](#partition-column-on-a-remote-index), the document lane its
own [partition column](#partition-column).

- **Column.** NOT NULL (42P16), a `smallint`, `integer`, `bigint`, `text`, `varchar` or `uuid` (0A000),
  not the id column (22023); a missing column is 42703. A `text` or `varchar` column needs a
  deterministic collation (0A000): under a case-insensitive one `'acme'` and `'ACME'` are one tenant
  to `WHERE tenant = 'ACME'` but two namespaces, so a search of one would miss the other's rows. A
  value already in the table whose namespace passes 512 bytes is refused (54000, naming the id of a
  row that holds it), instead of failing the backfill hours later. The id must be unique by itself
  (0A000 with a HINT otherwise): a composite key `(tenant_id, id)` needs `id_column => 'id'` and a
  unique index on `id`, and ids stay unique across the table. While the table is registered the
  column cannot be dropped, renamed, retyped or made nullable (2BP01). A bare re-promote keeps the
  stored column; declaring one on a table promoted without it first empties the namespace it wrote.
  `demote(.., drop_remote_index => true)` ends the declaration.
- **Namespace.** `<namespace>:` plus the value as `col::text` (decimal, lowercase hyphenated uuid, raw
  text) with every byte outside `[A-Za-z0-9._-]` written `%XX`: `Acme Corp` becomes
  `public.docs:Acme%20Corp`. At most 512 bytes. The table owns every namespace of its index that
  starts with `<namespace>:`; another table's namespace inside that family, or a family around it, is
  refused (42710).
- **Writes.** The trigger queues each upsert with its partition (`"p"`, the value as text) and sends
  `{"tenant_id": v}` as metadata next to the declared `metadata_columns`. A delete, and a row whose
  value changes, queue `{"id", "p"}` of the old row (bare ids from an older queue are deleted from
  `<namespace>`): a move is the delete in the old namespace, then the upsert in the new one. A NULL
  value, possible only if the NOT NULL was dropped behind the guard, fails the statement (23502); so
  does a value whose namespace passes 512 bytes (54000). The drain sends one request per namespace.
  A refusal of a namespace's first write that reads like a limit gets a HINT, in the failed
  partition's `last_error`: Pinecone limits namespaces per index by plan (100 on Starter up to
  1,000,000 on Enterprise). A 403 whose body names the namespace limit is that refusal, not a bad key
  (`authentication failed`).
- **Failed partitions.** A namespace Pinecone refuses for good no longer parks the table for every
  tenant. A permanent error on a request for one namespace (a 4xx other than 404 and 429: a refused
  request, the plan's limit of namespaces on a new tenant) marks that partition failed in
  `pinecone.partitions` (`failed_at`, and `last_error`, which names the namespace and carries the plan
  HINT), logs a WARNING and lets the queue row complete for the other tenants. `queue_state` and the
  table's `last_error` do not change: alert on `status().failed_partitions > 0`. While a partition is
  failed the worker drops every upsert and delete of it before any request (the table is the source
  of truth: nothing is kept or retried) and the backfills skip its rows. Its namespace keeps what it
  held when it failed and still answers searches, so that tenant's results are stale (empty, for a
  partition that failed during the initial load), and a row that moves out of a failed partition, or
  is deleted, stays in its namespace until the retry. `topk` of a failed partition checks the ids it
  got against the table (`id = ANY(ids) AND <partition column> = value`, an indexed read of at most
  `k` rows) and leaves out the rows that left, so it may return fewer than `k` rows, and raises a
  NOTICE naming the partition and its `failed_at`; an index scan already rechecks the heap, and a
  healthy partition's search reads nothing from the table. `status().failed_partitions` counts them
  (`0` for any other table, and for one that is not promoted); `pinecone.failed_partitions(tbl)`
  returns `(value, namespace, failed_at, last_error)` (table owner, 42501 for anyone else);
  `partitions_of` shows `failed_at` and `last_error` next to every namespace; `backup_readiness`
  fails its `sync` check and names them.
- **A refusal every tenant meets.** A 400 whose cause is the table's (a dimension or metric that does
  not match the index) is not isolated: the first tenant that meets it fails as above, but a refusal
  with the same text as a failed partition's (the namespace left out) is not marked. The queue row
  fails as a table's does: `pinecone.outbox_max_attempts` attempts, then the queue parks with every
  write intact (`queue_state = parked`, the reason in `last_error`), and an initial load stays
  `promoting`, served from the heap, instead of going remote with every partition failed. At most
  one partition fails that way. After the fix, `outbox_retry` resumes the queue and
  `retry_failed_partition` loads that partition, or `pinecone.sync` redoes the table in one wipe and
  one load. Two tenants refused with the same text for reasons of their own park the table too; the
  plan's limit of namespaces is each new tenant's own and always fails that tenant alone.
- **Retry.** `pinecone.retry_failed_partition(tbl regclass, value anycompatible)` (table owner)
  rebuilds one failed partition from the table. The call clears the failure and queues a
  `rebuild_partition` job, nothing else: it sends no request, so it needs no key and no grant on the
  queue and holds the table's lock for queries only. The worker runs the job (for a
  `sync_mode = 'sync'` table too): it deletes the namespace, once, then loads the rows with that
  value, the lane's own backfill scoped to the partition (by TID for an index). While it runs, a
  search of that tenant sees only what was written since. The rebuild yields to live writes: every
  worker round sends the table's queued writes first, other tenants' too, then runs rebuild chunks
  (at most 1,000 rows each) until `pinecone.backfill_round_ms` is spent, across as many queued jobs
  as fit, so a write made meanwhile waits for the round in progress, not for the rebuild, and small
  partitions share a round instead of taking one each. A write that lands between two chunks is
  sent by the next round, and the table decides what the two leave. A job queued behind a `promote`
  backfill or a `recover_from_remote` waits for it. The round holds the table's lock for up to the
  budget plus one chunk (a `pg_pinecone` verb or DDL on the table waits for it), so a larger budget
  rebuilds faster and holds it longer; an index's job scans the heap for its value, so many failed
  partitions of a big heap cost a scan each. A retry that Pinecone refuses again fails the partition
  again, with the new error, and so does a delete it refuses; a delete it cannot
  answer backs the job off. A partition whose rows are gone has nothing to load: its namespace is
  deleted and it is forgotten. A write queued before the retry and refused again fails the partition
  before the job runs, and the job leaves its namespace alone. A partition that is not failed is
  22023, a table that is not promoted 55000 (a bare `demote` keeps the failed partitions; a promote
  loads every partition again). A parked table's worker skips the job, which a NOTICE says
  (`outbox_retry` resumes it). `retry_failed_partitions(tbl)` retries every failed partition and
  returns how many.
  `delete_partition` forgets the failure of its partition with the namespace; TRUNCATE,
  `pinecone.sync` and REINDEX wipe every namespace and forget every failure; `untrack` and
  `DROP TABLE` forget them too.
- **Not isolated.** A 429 (rate or quota), a 5xx, a timeout and a connection failure back the whole
  table off as before: nothing says that Pinecone throttles per namespace rather than per index. A
  bad or missing key and an index that does not exist stay table-wide too, and so do a 404 from the
  index host (deleted and created again behind it) and a 200 whose body cannot be read. So does a
  `sync_mode = 'sync'` table: its writer is the statement that wrote the row, no shared queue can
  park, and the mode exists so that a refused write is visible, so the refusal still reaches the
  writer and nothing is marked. The document lane dead-letters per document already.
- **Registry.** `pinecone.partitions` lists the namespaces the table wrote: one is added when a drain
  sees Pinecone accept a vector for it (dumped with the extension). The table is closed to every role
  but the registry's owner; the table's owner reads their own rows with `pinecone.partitions_of(tbl)`,
  which returns `(lane, value, namespace, failed_at, last_error)` (42501 for anyone else, nothing for
  a table without a partition column). TRUNCATE, `pinecone.sync` and a re-promote empty
  the registered namespaces and the orphans of the family (see Orphans), one
  `DELETE /namespaces/{ns}` each, and forget them. `demote(.., drop_remote_index => true)`, `untrack`
  and `DROP TABLE` forget the registered ones without a request. `DROP TABLE` of a partitioned table
  also forgets its registry row (an unpartitioned one stays as `<dropped:OID>` in `status()`), so
  the same table can be created and declared again; Pinecone keeps the namespaces, and the new
  declaration empties them (Orphans, below).
- **Backfill.** The initial load and `sync` read in `(partition, id)` order and send one request per
  namespace, at most 100 namespaces per drain round (the job's cursor adds `"after_p"`). An index on
  `(tenant_id, id)`, which a composite primary key is, keeps each chunk cheap.
- **Queries.** [`topk(.., partition => <value>)`](#function-x-mode-support) takes an integer, text or
  uuid read as the column's type (`'007'` is `7`; `anycompatible`, so cast any other type first), or
  an `$eq` on the column in `filter` (`{"tenant_id": 42}`, `{"tenant_id": {"$eq": 42}}`, also inside
  `$and`; never `$or`, `$in`, `$ne` or a range). With neither it is an error (22023 with a HINT);
  `partition` on a table without a partition column is one too. The `$eq` stays in the request. A
  tenant without rows returns nothing. While the table is `local` (after a bare demote) or
  `promoting`, the heap is searched with the partition as a predicate. The partition selects, it does
  not authorise: anyone who may call `topk` can name any value.
- **`status()`** reports `partition_column`, `partitions` (the namespaces written) and
  `failed_partitions`; `namespace` is the prefix.
- **Orphans.** A namespace under `<namespace>:` that Pinecone holds and the registry does not list is
  an orphan: a `sync_mode = 'sync'` write that rolled back after Pinecone took it leaves one (and
  more than that: a rolled-back move has already deleted the row from its old namespace and put it
  in the new one, a rolled-back delete has deleted the vector; `pinecone.sync` repairs it all), a
  restore that lost the registry does, and so does a dropped table (`DROP TABLE`, `untrack` and
  `DROP INDEX` forget the registry and leave Pinecone alone). Every wipe lists the family
  (`GET /namespaces?prefix=<namespace>:`) first: `pinecone.sync`, a `promote` that declares the
  partition column, and the queued wipe behind TRUNCATE and a rebuild. It warns once per orphan (the
  first 20 by name, the rest counted) and registers it, so the wipe empties it with the rest; the
  load that follows brings it back when the table has rows with that value. So a re-created table
  never serves its predecessor's vectors: `topk(.., partition => 42)` cannot return an id the new
  table did not write. A name no write of the table makes (`<namespace>:abc` for a bigint column) is
  reported and left alone. If the listing fails, `sync` and `promote` raise before they wipe anything
  and the queued wipe retries. `status()` stays registry-only (no request): orphans show in `sync` and
  in [`backup_readiness`](#backup-readiness-and-repoint-of-a-family).
  **Use one database per index prefix.** The family is everything under `<namespace>:`, so two
  databases that share an index and keep the default names (`pv-<schema>-<table>`,
  `<schema>.<table>`) see each other's tenants as orphans, and a wipe in one deletes the other's.
  Give each its own `index_name` (or `namespace`).
- **Recovery.** `recover_from_remote` walks the registered namespaces in namespace order, a page at a
  time, and a worker round keeps walking until `pinecone.backfill_round_ms` is spent: a thousand
  small tenants take a few rounds, not a thousand. The job holds the namespace it is in
  (`partition_namespace`, `partition_value`) next to the page token, so a restart resumes there. A
  vector is written back only to a row of that namespace's partition, so an id whose row belongs to
  another tenant by now is a ghost (reported; `ghosts => 'delete'` deletes it from the namespace
  that held it), and the partition column is never taken from remote metadata. A registered
  namespace the index does not hold is skipped. A row that moved to another partition after the backup is in neither
  namespace: `pinecone.sync` loads it. The completion notice adds `namespaces N`.
- **`delete_partition(tbl regclass, value anycompatible)`** returns nothing. It deletes the namespace
  of one value (`DELETE /namespaces/{ns}`) and forgets it in `pinecone.partitions`: tenant
  offboarding. It is the table owner's call (42501) and idempotent (a namespace nobody wrote is
  deleted all the same; the NOTICE says it was not registered). `value` is an integer, text or uuid
  that the column's type reads (errors: [Errors](#errors)). **The table's rows stay**: the next
  write of a row with that value registers the namespace again, holding only what that write sent.
  Delete the rows as well (a `DELETE` queues one id delete per row; with `session_replication_role =
  replica` it queues none), or load the namespace whole with `pinecone.sync`. Writes still queued for
  the value recreate its namespace when they drain, so flush first (the
  [runbook](runbook.md#partitioned-tables) has the order); a NOTICE says so when anything is queued
  for the table. It works on each lane the table has: for a
  document table it deletes every document in the namespace, and for a partitioned index the heap
  rows stay, so a heap scan still finds them while a scan of the index finds nothing.
- **Not yet.** `detach_storage` refuses a partitioned table (0A000: a detached table's vectors
  live in one namespace).
- **Limits.** `promote` raises a NOTICE with the count above 100 distinct values and a WARNING above
  1,000.

### Backup readiness and repoint of a family

A table with a partition column is backed up and moved as a family: every namespace `<namespace>:*`.

- **`backup_readiness(tbl)`** counts the family instead of one namespace. `sync` compares the vectors
  of all its namespaces with the table's non-NULL vectors, says how many namespaces that is, and names
  those the registry does not list (no row accounts for their vectors). `coverage` says how many of
  the index's namespaces are the family's and **warns above 2,000 namespaces in the index**: that is
  the per-backup figure pg_pinecone knows of, not verified against live Pinecone, so test a restore
  before relying on a backup of that many.
- **`repoint(tbl, new_index)`** keeps the prefix and changes the index. It lists `<namespace>:` on the
  new index and registers the namespaces of the family that the registry lacks (`adopted N`), so
  recovery and every wipe reach them. It warns about registered namespaces the index lacks (recovery
  finds nothing in them) and about names pg_pinecone never writes, which it leaves alone. A listing that
  fails is a warning and the repoint goes on. It refuses (42710) an index where another table owns a
  namespace inside the family, or a family around it. An access-method table is refused, as before.

## Partition column on a remote index

`pinecone.create_index(tbl, col, engine => 'remote', partition_column => 'tenant_id')` builds
`CREATE INDEX .. USING pinecone (col, tenant_id) WITH (mode = 'remote')` and gives every value of the
column its own Pinecone namespace, `<schema>.<table>:<value>`, the way
[`promote(.., partition_column => ..)`](#partition-column-on-a-vector-table) does for the trigger
lane (same naming, 512-byte limit, `pinecone.partitions` and `status()` columns).
`set_engine(tbl, 'remote', partition_column => 'tenant_id')` declares it as a native index
`(col, tenant_id)` goes remote.

- **Column.** The index's second key, NOT NULL (42P16), a `smallint`, `integer`, `bigint`, `text` or
  `varchar` (0A000: no `uuid` on the access method); a missing column is 42703, and a `set_engine`
  column that is not the index's second key is 22023. A text column needs a deterministic collation
  (0A000), and a value already in the table whose namespace passes 512 bytes is refused (54000,
  naming the row by its TID), both before any request. `create_index` refuses `engine => 'pgvector'`
  and `pinecone_index` (0A000; an attached namespace holds one table's rows) and a `filter_column`
  naming another column (22023). On `engine => 'native'` the column is only the filter key. The
  table needs a single-column primary key, as for every `pg_pinecone` index. The column cannot be
  renamed, retyped or made nullable (2BP01); dropping it drops the index with it, as for any indexed
  column. A partition is declared when the index is created or goes remote, not on an index that is
  already remote.
- **Writes.** `aminsert` queues each tuple with its value (`"p"`) and `{"tenant_id": v}` as metadata
  and touches no registry row: the drain registers the namespace in `pinecone.partitions` once
  Pinecone took a write of it, so loaders that meet the same new tenants never wait for each other
  (`partitions_of` and `status().partitions` follow within a drain round). A NULL value,
  possible only if the NOT NULL was dropped behind the guard, fails the statement (23502), and so
  does a namespace over 512 bytes (54000). Changing the value is a non-HOT update: the new tuple
  goes to the new namespace and the old entry waits for VACUUM. The drain claims 100 queued tuples a
  step and sends one request per namespace among them: tenants that alternate row by row make up to
  100 requests for 100 rows.
- **Queries.** `WHERE tenant_id = <value>` (any integer width) binds the scan to the namespace of the
  value; the `$eq` stays in the request. A query without it (no condition, another column, `IN`, a
  range) is refused at scan start (0A000, no request), also while the index is `promoting`. A NULL
  value returns nothing without a request. Partitioned scans are always rechecked against the heap, so
  an entry whose TID VACUUM recycled for another tenant's row is never returned to the wrong tenant.
- **VACUUM.** `ambulkdelete` lists every namespace in the registry (one list walk each: cost grows with
  the number of tenants) and deletes a dead TID from the namespace that holds it, 1,000 ids per
  request. A namespace missing from the registry is not visited. A failed partition's namespace is
  still listed and cleaned, and an error from it fails the VACUUM as before.
- **Failed partitions.** The drain is the trigger lane's, so a permanent refusal of one namespace fails
  that partition the same way (see [Failed partitions](#partition-column-on-a-vector-table)): the
  tuples `aminsert` queued for it are consumed without a request, the TID backfill skips its rows, and
  `retry_failed_partition` loads the partition again by TID, reading the heap blocks but sending only
  the rows of that value.
- **Backfill, REINDEX, TRUNCATE.** The TID backfill reads a window of 5,000 rows (or 64 MB of
  vectors, ended at a block boundary) and sends one request per namespace in it, 1,000 vectors at
  most: a heap that alternates tenants row by row costs about one request per tenant per window, not
  one per row. A window of tenants with one row each is still one request per row. REINDEX and
  TRUNCATE empty every registered namespace and the orphans of the family (see
  [Orphans](#partition-column-on-a-vector-table)), one `DELETE /namespaces/{ns}` each, and load the
  table again. A new `create_index` queues only the load: it empties nothing, and a stale entry that
  an earlier table left under the same TIDs is rechecked against the heap, never returned to
  another tenant.
  `demote_index`/`set_engine(.., 'native')` forgets them (and empties each when the Pinecone index is
  shared); `DROP INDEX` and `DROP TABLE` forget them and leave Pinecone alone, as they do for the one
  namespace of an unpartitioned index.
- **Recovery and offboarding.** `recover_from_remote`, `backup_readiness` and
  [`delete_partition`](#partition-column-on-a-vector-table) work as for the trigger lane: recovery
  walks the registered namespaces and matches a TID only to a row of that namespace's partition
  (an entry whose TID VACUUM recycled for another tenant's row is a miss, `on_miss` decides what
  follows). `repoint` refuses an access-method table, as before. `delete_partition` leaves the heap
  rows: a scan of the index finds nothing for the value until it is written again.
- **Not yet.** `uuid`, a composite primary key such as `(tenant_id, id)` (the access method needs a
  single-column key, with or without a partition; per-partition ids are a v2 question), attaching an
  index, a query over several values, and declaring a partition on an index that is already remote.

## Document sync (`track_docs`)

`pinecone.track_docs` keeps a Pinecone **document index** in sync with a table's text columns,
independently of the vector modes above (a table can be in both; the two lanes park separately).
[`pinecone.search_docs`](#search-search_docs) searches it by table; `pinecone.search_doc_index` queries an
ad-hoc document index.

```sql
SELECT pinecone.track_docs('articles',
  text_columns     => '{"title": {"language": "en"}, "body": {"on_oversize": "truncate"}, "sku": {"stemming": false}}',
  metadata_columns => ARRAY['author', 'published_at'],
  vector_column    => 'embedding');           -- optional: the one dense field
SELECT pinecone.flush_docs('articles');          -- true once every committed change was delivered
```

**Freshness.** Sync is asynchronous and search is eventually consistent. The worker delivers a
committed change within about one worker interval (`pinecone.worker_naptime_ms`, 2 s; up to ten
if the worker had idled out and is respawned) when nothing is queued ahead of it, and the change is
searchable once Pinecone has indexed it, up to about a minute later. There is no read-your-writes.
`flush_docs` waits for delivery, not for indexing, and `status()` reports delivery
(`docs_backlog`, `docs_last_delivered_at`), not searchability. `engine => 'local'` has no delay.

| Function | What it does |
|---|---|
| `track_docs(tbl, text_columns, metadata_columns => '{}', vector_column => NULL, id_column => NULL, index_name => NULL, adopt_existing => false, partition_column => NULL)` | Validates the declaration locally, creates the index (or attaches `index_name`), registers the table in `pinecone.doc_tables`, installs the `pg_pinecone_docs_*` triggers and queues the backfill. Owner only. A WARNING says so if no worker launcher runs. |
| `untrack_docs(tbl, drop_index => false, force => false)` | Removes the registry row, triggers and queued documents; the index stays unless `drop_index` (refused while another tracked table uses it, and for an attached index — one pg_pinecone did not create — unless `force`). A table with a [partition column](#partition-column) first has every namespace it wrote emptied (not with `drop_index`, which takes them with the index). Owner only. |
| `flush_docs(tbl, timeout_ms => 30000)` | Waits until nothing is queued: `true`, or `false` with a WARNING at once if the queue is parked or no launcher runs, and after the timeout otherwise. A pending backfill counts as queued. Delivered is not searchable: Pinecone indexes for up to about a minute. Needs `SELECT` on the table. |
| `failed_docs(tbl)` | `(id, error, failed_at)` of documents Pinecone refused one by one. Needs `SELECT` on the table. |
| `retry_failed_docs(tbl)` | Requeues them; returns how many. Owner only. |
| `backfill_docs(tbl, clear => false)` | Queues a fresh backfill of every row, replacing one in flight (the repair after a restore or a lost index); `clear => true` empties the namespace first (every registered namespace of a table with a [partition column](#partition-column)), which also removes documents the table no longer has. Owner only; refused (55000) while the queue is parked. |
| `unpark_docs(tbl)` | Reopens a parked queue and clears its failed attempts. Owner only. |

**Privileges.** `track_docs`, `untrack_docs`, `backfill_docs`, `retry_failed_docs` and `unpark_docs`
need ownership of the table; `flush_docs`, `failed_docs` and `search_docs` need `SELECT` on it.
Nobody needs a grant on `pinecone.doc_tables` or `pinecone.doc_outbox`: each verb checks the
caller's right on the table, then reads and writes the extension's tables itself (what is left to
grant is in the [grant bundles](#grant-bundles-per-verb)). The remote engine also needs a Pinecone
key: the stored key (`SELECT ON pinecone.secret`) or a session key.

`track_docs` takes:

| Parameter | Meaning |
|---|---|
| `tbl` | An ordinary table you own. Partitioned tables and tables with inheritance children are refused. |
| `text_columns` | The searchable columns: an object `{"<column>": {options}}`, or a list of names for the defaults (`'["title", "body"]'`). Required. |
| `metadata_columns` | Columns sent with every document so `search_docs` can `filter` on them: `smallint`, `integer`, `bigint`, `real`, `double precision`, `numeric`, `boolean`, `text`, `varchar`, `char`, `uuid`, `date`, `timestamp`, `timestamptz` (sent as epoch seconds) or `text[]`; not `jsonb` or composites. Default none. |
| `vector_column` | A `vector(n)` or `halfvec(n)` column, sent as the one dense field. The dimension is the column's; the metric comes from the operator class of the table's indexes on it (cosine without one). Needed for `search_docs(vector => ...)`. Default none. |
| `id_column` | The column that gives the document `_id`. Default: the single-column primary key. |
| `index_name` | Default `pd-<schema>-<table>`. 1 to 45 lowercase letters, digits and `-`. |
| `adopt_existing` | `true` attaches an existing `index_name` whose schema matches the declaration; documents in it that this table does not own never join. Default `false`. |
| `partition_column` | A NOT NULL `smallint`, `integer`, `bigint`, `text`, `varchar` or `uuid` column, not the id: every value gets its own namespace, `<schema>.<table>:<value>`. See [Partition column](#partition-column). Default none: one namespace for the table. |

- **`text_columns`** is `{"<column>": {options}}` or a list of names (defaults). Options: `language`
  (default `en`), `stemming` and `stop_words` (default on; stop words need stemming), `ngram`
  (`{"min_gram", "max_gram"}`, excludes both), `on_oversize` (`truncate`, the default, or `error`).
  Columns must be `text`, `varchar` or `char`; names may not start with `_` or `$` or `pg_pinecone_`.
- **Identity.** The document `_id` is the single-column primary key (or `id_column`, NOT NULL and
  UNIQUE) of type `smallint`, `integer`, `bigint`, `uuid`, `text`, `varchar` or `char`, at most 512
  bytes. Partitioned and inherited tables are refused.
- **Index.** Created in `aws`/`us-east-1` with the caller's key as `pd-<schema>-<table>` (or
  `index_name`), namespace `<schema>.<table>`. An existing index is attached only with
  `adopt_existing => true` and only if its schema matches the declaration exactly.
- **Sync is asynchronous.** Statement-level triggers queue primary keys in the writer's
  transaction (rollback leaves nothing); the worker reads each row at drain time, so Pinecone gets
  the newest committed text. An UPDATE that touches no declared column queues nothing; a changed
  key is a delete plus an upsert; TRUNCATE empties the namespace.
- **Size policy.** Text over 100 KB or 9,000 tokens is cut and the document carries
  `pg_pinecone_truncated: true` (`on_oversize => 'error'` fails the statement instead); metadata over
  40 KB fails the statement.
- **Dead letters.** A document Pinecone rejects is isolated by bisecting the request, and a row the
  table cannot render (a `numeric` beyond a `float8`, say) by bisecting its read; each is listed in
  `failed_docs` with the message (counted in `status().docs_failed`; not backlog), and the queue
  never parks on a per-document error. A TRUNCATE clears the table's dead letters, a delivered
  delete or upsert the one of its id. A 429 or 5xx backs off without counting an attempt; any
  other failure counts one and parks the queue after `pinecone.outbox_max_attempts`.
- **DDL guard.** Dropping, renaming or retyping a declared column, and dropping the table, fail
  with SQLSTATE `2BP01` until `untrack_docs`; other columns are free.
- **Backfill.** `track_docs` queues one job that sends the rows the table already holds, in
  primary-key order, 1,000 rows (about 1.5 MB) per chunk; each chunk commits its cursor, so a
  restart re-sends at most one chunk (upserts are idempotent). The worker delivers live changes
  first, then runs chunks back to back for `pinecone.backfill_round_ms` per round, one
  transaction each, and starts the next round at once while chunks are left; the table lock is held
  only inside a chunk. [`status()`](#pg_pineconestatus-columns)
  shows the progress as `docs_backfill_done` / `docs_backfill_total` — the exact row count when
  the job was queued, so rows added or removed meanwhile leave `done` a little off it (when the
  job finishes `total` is set to `done`) — and `docs_backlog` counts the pending job. A refused
  document is a dead letter, not a stop. While the job runs, a remote `search_docs` raises a
  NOTICE that its results are partial.
- **Systemic refusals.** When Pinecone refuses 100 documents in a row, or every document of a
  batch of at least 10 (the index schema or the declaration has drifted), or the table cannot
  render 100 rows in a row or every row of a batch of at least 10, bisecting stops: no
  dead letters are written, the rows stay queued, and one lane error with Pinecone's message
  counts as a failed attempt (the queue parks after `pinecone.outbox_max_attempts`). A document
  that is delivered later has its dead letter removed.

`pinecone.doc_tables` is dumped with the extension; `pinecone.doc_outbox` is not (a queue is not
data: after a restore the heap is the source of truth).

### Partition column

`partition_column => 'tenant_id'` maps each value of the column to its own Pinecone namespace,
`<schema>.<table>:<value>`, instead of one namespace for the table. A search ranks inside one
partition (BM25 statistics are per namespace, so scores of two partitions cannot be merged and a
search never fans out) and a partition's documents live in a namespace of their own.

- **Column.** NOT NULL (42P16), of a supported type (0A000), not the id column. A text column needs a
  deterministic collation (0A000). A composite key `PRIMARY KEY (tenant_id, id)` is accepted with
  `id_column => 'id'` and a unique index on `id` alone: ids stay unique across the whole table
  (0A000 with a HINT otherwise).
- **Namespace.** `<schema>.<table>:` plus the value as `col::text` (decimal, lowercase hyphenated
  uuid, raw text) with every byte outside `[A-Za-z0-9._-]` written `%XX`: `Acme Corp` becomes
  `public.articles:Acme%20Corp`, and `''` becomes `public.articles:`. Pinecone allows 512 bytes; a
  row over it fails its statement (54000). The table owns every namespace of its index that starts
  with `<schema>.<table>:`; another table's namespace inside that family, or a family containing it,
  is refused (42710). Attaching an existing index (`adopt_existing`) warns that the family may hold
  documents the table does not own.
- **Sync.** An upsert goes to the namespace of the live row's value, read at drain time. A delete,
  and a row whose value changes, need the old value, so the trigger queues `{"id", "p"}` (`p` the old
  partition's text) in `doc_delete`; bare ids from an older queue are deleted from the namespace
  `<schema>.<table>`. A move is such a delete plus an upsert. A NULL value, possible only if the
  NOT NULL was dropped behind the guard, fails the statement (23502). A refusal of a new
  namespace's first documents that names the plan's limit stops the lane (no dead letters) with the
  namespace and the plan HINT in `last_error`. The errors of a request (429, 5xx, a connection
  failure) name its namespace, and one tenant's holds the lane up for the others (the vector lanes
  isolate a permanent refusal per partition instead, see
  [Failed partitions](#partition-column-on-a-vector-table)).
- **Registry.** `pinecone.partitions` lists the namespaces the table wrote (a namespace is added when
  its first document is accepted; dumped with the extension). TRUNCATE, `backfill_docs(.., clear =>
  true)` and `untrack_docs` empty those, one request each. Nothing lists the namespaces of a
  document index, so a namespace missing from the registry (its drain rolled back after Pinecone took
  a document, or a restore lost it) would be missed: `backfill_docs(.., clear => true)` and `untrack_docs`
  (without `drop_index`) also empty the namespace of every value the table holds (a TRUNCATE has no
  rows left to name them, so it empties the registered ones only). TRUNCATE and `untrack_docs`
  forget them, and `delete_partition(tbl, value)` deletes one namespace
  (`documents/delete {delete_all}`), forgets it and leaves the rows, as for vectors. `status()`
  counts them. If the index itself is gone, `untrack_docs` has nothing to empty: it says so in a
  WARNING and finishes the teardown (`DROP TABLE` works again).
- **Backfill.** Rows are read in `(partition, id)` order, so a chunk is mostly one namespace; the job's
  cursor adds `"after_p"`. Give the table an index on `(tenant_id, id)` (the composite primary key is
  one): without it every chunk reads and sorts the table.
- **Limits.** Pinecone limits namespaces per index by plan (Starter 100, Standard 100,000, Enterprise
  1,000,000); `track_docs` raises a NOTICE with the count and a WARNING above 1,000 distinct values.
  A row whose value makes a namespace over 512 bytes (it got into the table before the trigger) is a
  dead letter.
- **Search.** [`search_docs(.., partition => <value>)`](#search-search_docs) names the namespace; with
  `engine => 'local'` it is a condition on the column instead (`AND org = <value>`), so both engines
  return one partition's ids. It selects, it does not authorise: anyone with `SELECT` on the table can
  name any value, and the join's row-level security decides what they read.

### Engines and full-text search

A table's full-text search is served by one of two engines today. Which one has nothing to do
with the table's vector engine (`pgvector`, `native` or `remote`).

| Where the table stands | Engine | Ranking | Query features | Visibility |
|---|---|---|---|---|
| Declared with `track_docs`; `engine => 'remote'` (the default) | Pinecone document index | BM25 | `mode => 'query_string'` (Lucene), `must` (text-match), `filter` (metadata), `vector` (hybrid in one request) | Eventual: delivery, then up to a minute of indexing |
| A `pgvector` or `native` table, or any other, that is not declared; or any table with `engine => 'local'` | Postgres full-text search: `websearch_to_tsquery` and `ts_rank_cd` | Not BM25 | `query` and `field`. `query_string`, `must`, `filter` and `vector` are refused in 0.8 | Immediate, on the table's own indexes |
| Planned | BM25 in Postgres, through `pg_textsearch` or RUM when installed | BM25 | The operator form, `ORDER BY body <@@> 'q'` | Immediate |

Hybrid search works with every vector engine: the vector leg is plain `ORDER BY embedding <=> $1`,
served by whatever index the table has ([recipe](#hybrid-search)).

### Search (`search_docs`)

```sql
SELECT a.id, a.title, s.score
FROM pinecone.search_docs('articles', 'postgres vector search', k => 10) s
JOIN articles a ON a.id = s.id::bigint
ORDER BY s.rank;
```

`search_docs(tbl, query => NULL, mode => 'text', field => NULL, must => NULL, filter => NULL,
vector => NULL, engine => 'remote', k => 10, partition => NULL)` returns `(id text, score real, rank integer)`: the ids
and scores Pinecone ranked, `rank` being the 1-based position. Nothing else leaves Pinecone, so the
columns, privileges and row-level security are your join's. The function declares `ROWS 100`; in a
`LATERAL` join it makes one request per outer row.

| Parameter | Meaning |
|---|---|
| `query` | BM25 over every declared text column (one `text` clause naming them all), or over `field` alone. Not blank. |
| `mode` | `'text'` (default), or `'query_string'`: Lucene syntax such as `title:(postgres AND "vector search"^2) -body:old`, sent unchanged. Qualify fields inside the query; `field =>` is refused in this mode. |
| `field` | One declared text column. |
| `must` | Text-match requirements on declared text columns: `{"body": {"$match_all": "TensorFlow"}}`; operators `$match_phrase` (exact phrase), `$match_all` (all tokens), `$match_any` (at least one). |
| `filter` | Conditions on declared `metadata_columns`, with the operators and per-type rules of `topk`'s filters. A `date`, `timestamp` or `timestamptz` value may be a string (`'2024-01-01T00:00:00Z'`); it is sent as epoch seconds like the documents' own values (`date` and `timestamp` count as UTC, a `timestamptz` string without an offset follows the session time zone). |
| `vector` | A `real[]` query vector: ranks by the dense field instead of BM25 (the table needs a `vector_column`; the length must match). `'[...]'::vector` and `vector` columns convert implicitly; pass a `halfvec` as `embedding::real[]`. With a `query`, `field` is required and the query becomes a `$match_all` requirement on that column (hybrid search in one request). |
| `engine` | `'remote'` (default): Pinecone, a table declared with `track_docs`. `'local'`: [Postgres FTS](#local-engine-postgres-fts-not-bm25), no Pinecone call. |
| `k` | 0 to 10,000; `k = 0` returns nothing without calling Pinecone. |
| `partition` | For a table declared with [`partition_column`](#partition-column): the value whose namespace to search, an integer, text or uuid read as the column's type (`partition => 42`, `partition => 'Acme Corp'`). Required then (22023), unless `filter` has an `$eq` on the column, which needs it in `metadata_columns`; refused (22023) on a table without one. A namespace nothing was written to returns no rows. `engine => 'local'` searches the heap and applies it as `AND <column> = <value>` (the column's own type, so `'Acme'` and `'acme'` stay two tenants): either engine returns one tenant's ids; a table without a partition column ignores it there. The parameter is `anycompatible`, so that an integer and a text literal both resolve; cast any other type first. |

- `must` and `filter` (and the requirement a hybrid `query` adds) are sent as one `{"$and": [...]}`;
  a single one is sent as it is. A request has exactly one scoring type: `mode => 'query_string'`
  cannot be combined with `vector`.
- **Visibility is eventual.** A search sees what was delivered (`flush_docs`) and indexed, about a
  minute later. A row deleted locally but not yet deleted remotely drops out at the join; a recently
  updated row may rank on its old text for up to a minute.
- **Row-level security applies to your join, not to Pinecone.** The remote engine returns every
  match, including rows the caller's policies hide; the join drops them, but their ids and scores are
  observable by anyone with `SELECT` on the table. The local engine runs as the caller and never
  returns such rows.
- A 429 is a quota error (SQLSTATE `53400`, the monthly unit limit; clears with the billing period or
  an upgrade) or a rate error (`58030`, transient), told apart by Pinecone's message. Other 5xx,
  timeouts and unreachable hosts are `58030` and point at `engine => 'local'`.

### Hybrid search

Two ways, both with any vector engine.

**One request.** `vector` with a `query` and a `field` (the table needs a `vector_column`): the text
is a `$match_all` requirement on that column and the dense field ranks what is left.

```sql
SELECT s.id, s.score
FROM pinecone.search_docs('articles', 'tensorflow', field => 'body',
       vector => (SELECT embedding FROM articles WHERE id = 42), k => 10) s;
```

**Two rankings, fused.** Rank by text in `search_docs` and by meaning in SQL, then add reciprocal
ranks (RRF, k = 60). Scores of different kinds are not comparable; ranks are.

```sql
WITH lex AS (
  SELECT s.id::bigint AS id, s.rank
  FROM pinecone.search_docs('articles', 'growling bear', k => 20) s),
sem AS (
  SELECT id, rank() OVER (ORDER BY embedding <=> $1) AS rank   -- $1: the query embedding
  FROM articles ORDER BY embedding <=> $1 LIMIT 20)
SELECT coalesce(lex.id, sem.id) AS id,
       coalesce(1.0 / (60 + lex.rank), 0) + coalesce(1.0 / (60 + sem.rank), 0) AS score
FROM lex FULL OUTER JOIN sem ON lex.id = sem.id
ORDER BY score DESC LIMIT 5;
```

### Local engine: Postgres FTS, not BM25

`engine => 'local'` answers the same statement from Postgres: no Pinecone call, no sync delay (a
committed row is found at once) and no index of its own. `query` is read as web search syntax
(`websearch_to_tsquery`: words are ANDed; `"a phrase"`, `or` and `-word` work) and rows are ranked by
`ts_rank_cd`, which weighs how often and how close the words occur but has no inverse document
frequency. Order and scores therefore differ from Pinecone's BM25, and a score only compares within
one query.

```sql
SELECT a.id, s.score
FROM pinecone.search_docs('articles', 'postgres "vector search"', engine => 'local', k => 10) s
JOIN articles a ON a.id = s.id::bigint
ORDER BY s.rank;
```

- **Runs as you.** The search is a query under the caller's role: `SELECT` privilege and row-level
  security apply to the function itself.
- **Takes `query`, `field` and `k`.** `mode => 'query_string'`, `must`, `filter` and `vector` raise
  `0A000`; for a vector leg, `ORDER BY embedding <=> $1` in SQL.
- **Tracked or not.** A tracked table is searched on its declared columns (or `field`), each with the
  configuration of its declared language. Any other table needs `field => '<column>'` (`text`,
  `varchar` or `char`), is searched as `english`, and returns its single-column primary key (of a
  type `track_docs` accepts) as the id.
- **Languages.** A column's declared `language` picks the Postgres configuration; a column with
  `stemming => false` (identifiers, `ngram`) is `simple`: words as written, no stop words.

  | `language` | Configuration | `language` | Configuration |
  |---|---|---|---|
  | `da` | `danish` | `nl` | `dutch` |
  | `de` | `german` | `no` | `norwegian` |
  | `en` | `english` | `pt` | `portuguese` |
  | `es` | `spanish` | `ru` | `russian` |
  | `fi` | `finnish` | `sv` | `swedish` |
  | `fr` | `french` | `tr` | `turkish` |
  | `hu` | `hungarian` | other (`ar`, `el`, `ro`, `ta`, ...) | `simple` |
  | `it` | `italian` | | |

  A stemmed column also drops its language's stop words, whatever `stop_words` says.
- **Indexes.** The engine creates none; it uses the table's own and otherwise scans, building the
  vector of every row and sorting every match, so index a large table. For the declared columns of
  one configuration it evaluates `to_tsvector(<config>, coalesce(<column>, ''))` of each, in column
  name order, joined by `||`; a GIN index on exactly that expression serves the search. Columns of
  another configuration are a second expression with its own index: a row matches if either does and
  the ranks add up. With `field` it is that column's expression alone.

  ```sql
  -- title and body in English, sku with stemming => false
  CREATE INDEX ON articles USING gin ((
    to_tsvector('english', coalesce(body, '')) || to_tsvector('english', coalesce(title, ''))));
  CREATE INDEX ON articles USING gin (to_tsvector('simple', coalesce(sku, '')));
  ```

### Snippets (`headline`)

`pinecone.headline(tbl, column, document, query, options => '')` returns `document` with the words of
`query` highlighted: core `ts_headline` with the configuration of the declared language of
`tbl`.`column` (`english` if the column is not declared). It runs in Postgres on the text you pass,
so it works after the join for either engine; `options` are `ts_headline`'s, for example
`'StartSel=<mark>, StopSel=</mark>, MaxWords=35'`.

```sql
SELECT a.id, pinecone.headline('articles', 'body', a.body, 'postgres vector search') AS snippet
FROM pinecone.search_docs('articles', 'postgres vector search', k => 10) s
JOIN articles a ON a.id = s.id::bigint
ORDER BY s.rank;
```

## `pinecone.status()` columns

| Column | Meaning |
|---|---|
| `table_name` | `schema.table`, or `<dropped:OID>` if dropped without `untrack`. |
| `mode` | `local`, `remote`, or `promoting`; `docs` for a table declared only with `track_docs`. |
| `vector_column` / `id_column` / `metric` / `dimension` | The tracked column and its registered metric/dimension. A `docs` row has the id column and `''`, `''`, `0` for the rest. |
| `index_name` / `namespace` | Pinecone index/namespace, or `NULL` if local-only. A `docs` row shows the document index. |
| `metadata_columns` | Columns registered for remote filter pushdown (a `docs` row: the declared metadata columns). |
| `local_rows` | `count(*)`, or `NULL` if unreadable or dropped. |
| `backlog` / `oldest_enqueued` / `last_error` / `queue_state` | Outbox status — `NULL` without `EXECUTE` on `_outbox_stats`. They describe the vector lane, so `backlog`, `oldest_enqueued` and `queue_state` are `NULL` in a `docs` row. `last_error` falls back to the document lane's error (its park reason, else the error recorded on its oldest queued row) when the vector lane has none. |
| `storage` | `full` or `detached`. |
| `backup_gate` | `'covered'`/`'forced'` at detach time, or `NULL`. |
| `detached_at` / `detach_backup_id` | Detach audit trail, or `NULL`. |
| `rpo` | Worst-case RPO at detach time, or `NULL`. |
| `read_capacity` | `'OnDemand'`, `'Dedicated <node_type> <shards>x<replicas> <state>'`, or `NULL` if unknown. |
| `attached` | `true` if this row was registered via `create_index`/`set_engine`'s attach path (an existing Pinecone index, no backfill); for a `docs` row, an index attached with `adopt_existing`. |
| `docs_backlog` | Queued document-lane rows, a pending backfill included, dead letters not. A row carries up to 1,000 ids. `0` means caught up. |
| `docs_oldest_enqueued` | When the oldest queued row was enqueued: the age is the delivery lag. `NULL` when nothing is queued. |
| `docs_last_delivered_at` | When Pinecone last accepted a request for this table (an upsert or a delete, live or from the backfill), or `NULL` before the first. |
| `docs_backfill_done` / `docs_backfill_total` | Backfill progress: rows passed, and the row count the owner sees when the job was queued (a row-level security policy that hides rows hides them here too, like in `local_rows`). When the job finishes, `total` is set to `done`. |
| `docs_failed` | Dead letters: documents Pinecone refused, or the table could not render, one by one (`failed_docs`). They are not backlog. |
| `docs_queue_state` | The document lane, in the vocabulary of `queue_state`: `idle`, `active`, `backoff`, `parked`, `NO WORKER`. |
| `partition_column` | The column that gives every value its own namespace (`namespace` is then the prefix), for a table promoted with [`partition_column`](#partition-column-on-a-vector-table) or tracked with the document lane's; else `NULL`. |
| `partitions` | How many namespaces the table has written (`pinecone.partitions`); `NULL` without a partition column. A drain registers a namespace when Pinecone takes its first write, so a write still queued is not counted (`partitions_of(tbl)` lists them). Registry only, like every column here: orphans Pinecone holds and the registry does not show up in `pinecone.sync` and `backup_readiness`. `delete_partition` lowers it, the next write of the value raises it again. |
| `failed_partitions` | How many partitions of the vector lane Pinecone refused for good and the worker skips (`pinecone.failed_partitions(tbl)` says which, and why). `0` for a table without a partition column, one that is not promoted and the document lane, which dead-letters per document. A failed partition changes neither `queue_state` nor `last_error`: alert on `failed_partitions > 0`. |

The `docs_*` columns describe a table's document lane and are `NULL` for a table that is not
declared with `track_docs`, and for a table the caller cannot `SELECT` (the registry itself is nobody's to read). Without
`EXECUTE` on `pinecone._doc_outbox_stats(oid)` (as for the vector columns on `_outbox_stats`),
`docs_backlog`, `docs_oldest_enqueued` and `docs_failed` are `NULL`, and so is `docs_queue_state`
unless the lane is parked. A table in both registries is one row with both lanes; they park
separately.

`docs_queue_state` is a snapshot of the queue, not of the worker's own schedule: `parked` (the
lane stopped after `pinecone.outbox_max_attempts` permanent failures, see
[`unpark_docs`](runbook.md#parked-document-lane)), `idle` (nothing queued), `backoff` (a delivery
attempt failed and nothing has been delivered since: Pinecone is down or rate limiting; the worker
retries after growing delays), `NO WORKER` (queued, and no launcher runs in the cluster), `active`
(queued, draining). Delivery is not searchability: Pinecone indexes for up to about a minute after
`docs_last_delivered_at` (see [Freshness](#document-sync-track_docs)).

## Errors

| Message starts with | Cause | Fix |
|---|---|---|
| `pg_pinecone: no API key available` | No key on the session GUC and none stored. | `SET pinecone.api_key = '...'`, or have an owner run `pinecone.set_api_key(...)` and `GRANT SELECT ON pinecone.secret`. |
| `pg_pinecone: index '...' not found` | Index name is wrong, or the key belongs to a different project. | Check the index name and project. |
| `pg_pinecone: index '...' already exists` | `create_doc_index` hit an index name already in use. (`promote`/`CREATE INDEX ... USING pinecone` adopt an existing index of the same name instead of erroring.) | Pick a different name, or drop the existing index. |
| `pg_pinecone: authentication failed` | The API key is invalid or lacks access. | Verify the key in the Pinecone console. |
| `pg_pinecone: Pinecone rejected the request: ...` | The request was malformed (e.g. dimension mismatch, bad doc-index schema). | Check the message; fix the offending argument. |
| `pg_pinecone: request timed out after ...ms` | The control or data plane didn't respond in time. | Raise `pinecone.timeout_ms`. |
| `pg_pinecone: could not reach ...` | Network problem, or a bad `pinecone.control_plane_url`. | Check connectivity and the GUC value. |
| `pg_pinecone: Pinecone returned unexpected status ...` | An unhandled API response. | Check the Pinecone status page; retry. |
| `pg_pinecone: Pinecone returned unexpected status 429: ...` (an index scan, `pinecone.topk`) | A read hit a Pinecone limit: the rate limit of the namespace, or the plan's monthly read units (the message says which). Reads are not retried (only writes are: 429 and 5xx, three attempts), so the statement fails at once. | Retry with backoff in the application, or spread the load over namespaces with a [partition column](#partition-column-on-a-vector-table); see [Limits](#limits). |
| `pg_pinecone index '...' has format version N (this build reads up to M): REINDEX it` | An index a newer release built, after a downgrade (SQLSTATE 55000). | `REINDEX INDEX ...`; see [Index format version](#index-format-version). |
| `pg_pinecone index '...' has format version N (this build reads M): REINDEX it` (an error on a scan or `INSERT`, a WARNING from the planner, `VACUUM` and the maintenance readers; for a format-3 index of an older page layout `... has format version 3 with page layout 0 (this build reads layout 1): REINDEX it`) | A native index built before 0.10 (its sketch codes), or by a development snapshot of 0.10 before the code blocks: offline until rebuilt (SQLSTATE 55000). | `SELECT pinecone.reindex_native_indexes()`, or `REINDEX INDEX ...`; see [Index format version](#index-format-version). |
| `pg_pinecone: the vector at heap tuple (b,o) has a non-finite value` / `cannot index a vector with a non-finite value` | A native build or insert met a vector with a NaN or infinite component (pgvector stores none; a scan errors the same way for such a query). | Fix or delete the row. |
| `pg_pinecone: id '...' is N bytes, exceeding Pinecone's 512-byte id limit` | A tracked row's id, cast to text, is too long. | Shorten the id, or use a different `id_column`. |
| `<function>: <arg> must not be NULL` / `<arg>[N] is NULL` | A required argument (or array element) was `NULL`. | Pass a non-`NULL` value, or omit the argument to use its default. |
| `pinecone.track: ... has no vector column` / `has multiple vector columns` | Autodetection needs exactly one `vector` column. | Pass `vector_column` explicitly. |
| `pinecone.track: ... has no primary key` / `has a composite primary key` | Autodetection needs a single-column PK. | Pass `id_column` explicitly. |
| `pinecone.track: column "..." ... which pinecone.track does not support for id_column` | `id_column`'s type isn't in the whitelist. | Use `text`/`varchar`/`uuid`/`int2`/`int4`/`int8`. |
| `pinecone.track/untrack/promote/demote/sync/topk: ... is managed by the pg_pinecone index — DROP INDEX to detach` | Called on an AM-indexed table. | `DROP INDEX` to detach, or query with `ORDER BY <col> <=> <value>` directly. |
| `pinecone.track/untrack: ... is tracked in 'remote' mode` | `track`/`untrack` was called on a promoted table. | `pinecone.demote(...)` first. |
| `pinecone.topk: table ... is not tracked` | `topk` on an untracked table. | `pinecone.track(...)` (or `promote`) first. |
| `pinecone.topk: query has N dimensions, but ... is tracked with dimension M` | Query vector's length doesn't match the tracked dimension. | Pass a query vector of the right length. |
| `pinecone.topk: unknown column '...'` / `unknown operator '...'` | A filter referenced a column not in scope, or an unsupported operator. | Fix the filter, or (remote) add the column to `metadata_columns` and re-promote. |
| `pinecone.promote: writers' sync triggers need pinecone.set_api_key(...)` | `promote` was called with only a session-GUC key. | `pinecone.set_api_key(...)` first. |
| `pinecone.promote: existing index '...' has dimension=..., metric=..., vector_type=... but ... needs ...` | Adopting an existing index whose shape doesn't match the table. | Use a different `index_name`, or fix the mismatch. |
| `pinecone.promote: sync_mode '...' is not supported` | `sync_mode` wasn't `'async'` or `'sync'`. | Pass `'async'` or `'sync'`. |
| `pinecone.promote: partition column "..." has type ...` / `must be NOT NULL` / `is its id column` / `does not exist` / `has a nondeterministic collation` | The partition column is unsupported (0A000), nullable (42P16), the id (22023), missing (42703) or a text column whose collation is nondeterministic (0A000; also `create_index`). | Use a NOT NULL `smallint`, `integer`, `bigint`, `text`, `varchar` or `uuid` column other than the id, with a deterministic collation (the default, or `COLLATE "C"`). |
| `pinecone.promote: id "..." of ... has a "..." value that makes a namespace of N bytes` | A text value already in the table makes a namespace over Pinecone's 512 bytes (54000; `create_index` names the row by TID). Nothing was requested. | Shorten the value, in that row and any other over the limit. |
| `pinecone.promote: id column "..." is not unique by itself` | A composite key such as `(tenant_id, id)` with a partition column: ids must be unique across the table (0A000). | Add a unique index on the id column, or a table-wide unique column. |
| `pinecone.promote: ... partition family ... overlaps / collides with / falls in ...` | Another table of the index holds a namespace the family `<namespace>:*` would cover, or the reverse (42710). | Pass a different `index_name` or `namespace`. |
| `... — while writing the first vector (or document) of partition namespace '...'` | Pinecone refused the first write of a new namespace with a message about a namespace limit or quota (400, or 403): the plan's limit of namespaces per index. The HINT is in `last_error` too: the failed partition's (`failed_partitions`) for vectors, `status()`'s for documents. | Delete namespaces you no longer need (`delete_partition`) or upgrade the plan, then `retry_failed_partition` (documents: `unpark_docs`). |
| `pinecone.topk: ... is partitioned by "...": a search needs one partition` / `has no partition column, so partition => does not apply` / `partition "..." is not a valid ... value` / `partition is of type ...` | `partition` is missing (22023, unless `filter` has an `$eq` on the column), given for a table without a partition column (22023), not a value of the column's type (22023) or of a type no partition column holds (42804). | Pass `partition => <value>` or `filter => '{"col": {"$eq": <value>}}'`; omit `partition` for an unpartitioned table. |
| `pinecone.detach_storage: ... has the partition column "..."` | A detached table keeps its vectors in one namespace, and a partitioned table has one per value (0A000). | Leave the table attached. |
| `pinecone.delete_partition: ... must be owner of ...` / `is not tracked by ...` / `has no partition column, so it has one namespace` / `value must not be NULL` / `"..." is not a valid ... value` / `the namespace of this partition is unusable` | The caller does not own the table (42501), or it has no `promote` or `track_docs` declaration (42704) or no partition column (22023); `value` is NULL (22004), is not a value of the column's type (22023, or 42804 for a type no partition column holds), or makes a namespace Pinecone refuses (54000 over 512 bytes). Nothing was requested. | Pass a value of the column's type. Without a partition column, `demote` or `untrack_docs` removes what the table wrote. |
| `pinecone.sync: orphan namespace '...'` / `pinecone.repoint: index '...' lacks N of the M registered namespaces` (WARNINGs) | `sync`, a `promote` that declares a partition column or the queued wipe behind TRUNCATE found a namespace under `<namespace>:` that Pinecone holds and the registry does not list (the wipe empties it); `repoint` found registered namespaces missing from the new index. Neither stops the call. | See [Orphans](#partition-column-on-a-vector-table) and the [runbook](runbook.md#partitioned-tables). |
| `pinecone.partitions_of: must be owner of ...` (also `pinecone.failed_partitions`) | The caller does not own the table (42501). | Ask the table's owner. |
| `pg_pinecone: partition '...' (...) of ... failed for good and is skipped until it is retried: ...` (WARNING, once per failure) | The worker, a backfill or a retry met a permanent refusal of one namespace and marked that partition failed (see [Failed partitions](#partition-column-on-a-vector-table)). The rest of the table goes on. | `pinecone.failed_partitions(tbl)` says why; fix the cause, then `pinecone.retry_failed_partition(tbl, value)`. |
| `pinecone.retry_failed_partition: partition '...' of ... is not failed` | The partition is healthy or was never written (22023); the HINT lists the failed ones. Also: `must be owner of ...` (42501), `has no partition column` (22023), `is not promoted (mode = 'local')` (55000: a bare `demote` keeps the failed partitions), `is not tracked by ...` (42704), `has no vector lane` (22023: a document lane dead-letters per document), `value must not be NULL` (22004). Nothing was requested. | Check `pinecone.failed_partitions(tbl)`. `retry_failed_partitions` does not mind and returns 0. |
| `pinecone.retry_failed_partition: ... is parked (...): the worker skips it, so nothing is rebuilt until pinecone.outbox_retry(...) unparks it` (NOTICE) | The rebuild is queued, but the worker skips a parked table. | Fix the cause the reason names, then `pinecone.outbox_retry(tbl)`. |
| `pinecone.topk: partition '...' of ... is failed since ...: results may be stale (rows that left it are left out); ...` (NOTICE) | The searched partition is failed: its namespace is frozen where it failed, so the ids came back checked against the table and may be fewer than `k`. | `pinecone.failed_partitions(tbl)` says why; fix the cause, then `retry_failed_partition`. |
| `cannot drop column / alter column "..." of ...: it is the partition column of pinecone.promote` (or `of its pg_pinecone index`) | The DDL guard (SQLSTATE 2BP01): the partition column of a promoted table or a partitioned `pg_pinecone` index cannot go, change name or type, or become nullable. | `pinecone.demote(tbl, drop_remote_index => true)` first; for an index, `DROP INDEX` it or `set_engine(tbl, 'native')`. |
| `pg_pinecone: a query on a partitioned remote index needs WHERE ... = <value>` | A scan of a partitioned `pg_pinecone` index without an equality on its partition column (0A000, nothing was requested). | Add `WHERE tenant_id = <value>`. |
| `pinecone.create_index / promote_index: partition_column ...` | `pgvector` engine or `pinecone_index` with a partition column (0A000), `filter_column` and `partition_column` naming different columns, or a column that is not the index's second key (22023). | Drop the extra argument; build the index with the partition column second. |
| `pg_pinecone: partition column "..." of ... is NULL` / `the partition value ... makes an unusable namespace` | A write to a partitioned index has no value (23502) or a value whose namespace passes 512 bytes (54000); nothing is queued. | Set the column, or shorten the value. |
| `pinecone.demote/sync: ... is not tracked` | `demote`/`sync` on an untracked table. | `pinecone.track(...)` or `pinecone.promote(...)` first. |
| `pinecone.demote/sync: ... is not promoted` | `demote`/`sync` on a table that's still local. | `pinecone.promote(...)` first. |
| `pinecone.demote: index '...' is also used by another tracked table` | `drop_remote_index => true` would strand another table's namespace. | Demote the other table first, or don't drop the index. |
| `pinecone.flush/outbox_retry/outbox_discard: ... is not tracked` | Called on a table with no `pinecone.tables` row. | `pinecone.track(...)`/`pinecone.promote(...)` first. |
| `pinecone.repoint: ... is not mode 'remote'` | `repoint` on a table that isn't (exactly) `'remote'` — e.g. still `'promoting'`. | Wait for promotion to finish, or `pinecone.promote(...)` first. |
| `pinecone.create_index: engine '...' is not supported` | `engine` wasn't `'pgvector'`, `'native'`, or `'remote'`. | Pass one of those three. |
| `pinecone.create_index: ... is already managed with engine '...'` | A second `create_index` call on an already-managed table. | Use `pinecone.set_engine(...)` to switch, or `DROP INDEX` the existing one first. |
| `pinecone.create_index: engine 'remote' needs a stored API key` | `engine => 'remote'` with no stored key. | `pinecone.set_api_key(...)` first. |
| `pinecone.set_engine: ...` (names `pinecone.create_index` instead) | Asked to switch to/from `'pgvector'`, or another unsupported transition. | Stand up a fresh index with `create_index(..., engine => ...)`. |
| `pinecone.create_index: pinecone_index only applies to engine 'remote'` / `pinecone.set_engine: pinecone_index only applies to a 'native' -> 'remote' transition` | `pinecone_index` passed with the wrong engine/transition. | Drop `pinecone_index`, or pick `engine => 'remote'` / a `native -> remote` switch. |
| `pg_pinecone: index '...' does not match ...'s column/opclass` (DETAIL names both shapes) | Attaching an index whose dimension/metric/vector_type doesn't match the column/opclass. | Attach a matching index, or fix the column/opclass. |
| `pg_pinecone: index '...' namespace '...' does not look like it holds ...'s current rows` (HINT names both counts) | Attaching a namespace whose vector count doesn't match the table's live row count. | Fix the mismatch, or pass `attach_unchecked => true` if you're sure they already match. |
| `pinecone.create_index/promote_index: index '...' namespace '...' is already owned by ...` | Attaching an index/namespace another tracked table already owns. | Pick a different `pinecone_namespace`, or attach the other table elsewhere first. |
| `pg_pinecone: the pg_pinecone access method supports one vector column plus at most one filter column` | `CREATE INDEX ... USING pinecone` named more than one filter column, or an expression. | Index one vector column, optionally with one plain filter column. |
| `pinecone.create_index: filter_column only applies to engine 'native' or 'remote'` / `column "..." not found` / `filter column "..." of ... has type ...` | `filter_column` on a `pgvector` index, or naming a missing or unsupported column. | Pick `native`/`remote` and a `smallint`, `integer`, `bigint`, `text` or `varchar` column. |
| `pg_pinecone: filter column "..." of ... was renamed` | A scan of a remote index whose key column was renamed since the build; Pinecone still holds the metadata under the old name. | `REINDEX` the index. |
| `pg_pinecone: column "..." of ... has type ..., not vector` | The indexed column isn't `vector`. | Index a `vector` column. |
| `pg_pinecone: ... has no primary key` / `has a composite primary key` — pg_pinecone access method | Same PK rule as `pinecone.track`, enforced by `CREATE INDEX ... USING pinecone`. | Add a single-column primary key. |
| `pg_pinecone: a pg_pinecone access-method index needs a stored API key` | A remote-mode `CREATE INDEX ... USING pinecone` with no stored key. | `pinecone.set_api_key(...)` first. |
| `pg_pinecone: ... already has a pg_pinecone access-method index` | A second AM index on the same table (the message says "doesn't match" when its vector or filter column differs). | `DROP INDEX` the existing one first. |
| `pg_pinecone: local mode supports at most N dimensions (a centroid must fit one ...-byte page)` | The vector column is wider than the native engine's page-format-derived cap. | Use `engine = 'pgvector'`/`'remote'` for this column, or reduce its dimension. |
| `pg_pinecone: k_bits (...) must be a multiple of 8` | An explicit `k_bits` reloption wasn't byte-aligned. (The value is ignored since 0.10, but a typo is still refused.) | Pass a multiple of 8, or omit it. |
| `value ... out of bounds for option "target_recall"` (SQLSTATE 22023) | `target_recall` outside 0.5 to 0.999, at `CREATE INDEX` or `ALTER INDEX .. SET`. | Pass a value in range; see [Calibrated defaults](#native-engine-calibrated-defaults). |
| `pg_pinecone: calibration_queries: ...` (SQLSTATE 22023 for a name that is not `schema.table` or `schema.table(column)` or a table with fewer than 50 distinct queries, 42P01 no such table, 42703 no such or no `vector` column, 42804 a column that is not a `vector` of the indexed dimension or a query of another dimension, 42702 two candidate columns, 42809 not a table or view, or a foreign table, 42501 the owner of the indexed table cannot read it, 55P03 the table is locked, 22000 a non-finite value) | `CREATE INDEX .. WITH (calibration_queries = ..)`, or `ALTER INDEX .. SET` for the name's shape. At a rebuild the same problems are a WARNING (`calibrating on table rows instead`) and `queries_source` reads `fallback`; an error raised while reading (a view's function) is reported by its SQLSTATE, its text in the server log. | Name `schema.table` or `schema.table(column)` of 50 or more queries of the indexed dimension that the owner of the indexed table can `SELECT`; `ALTER INDEX .. RESET (calibration_queries)` drops the setting; see [Calibrated defaults](#native-engine-calibrated-defaults). |
| `pg_pinecone: index scan needed more than pinecone.am_max_topk (...) remote matches` | A selective `WHERE` on a remote-mode AM-indexed table over-fetched past the cap. | Raise `pinecone.am_max_topk`, loosen the filter, or `SET pinecone.iterative_scan = 'relaxed_order'` to end the scan quietly at `pinecone.max_scan_tuples` instead. |
| `pinecone.outbox_discard: dropped N queued row(s) ...` (a `WARNING`, not an error) | `outbox_discard` ran — those writes never reached Pinecone. | Run `pinecone.sync(tbl)` to repair. |
| `pinecone.create_doc_index: schema must ... "fields"` | `schema` isn't a JSON object with a `"fields"` object. | Pass e.g. `'{"fields": {"body": {"type": "string", "full_text_search": true}}}'`. |
| `pinecone.upsert_docs: docs[N] is missing a string "_id"` | A document had no `_id`. | Give every document a unique `_id`. |
| `pinecone.search_doc_index: query must not be an empty array` | `query` (`score_by`) was `[]`. | Pass at least one clause object. |
| `pg_pinecone: pinecone.scan_factor must be 0 (omit) or between 0.5 and 4.0` | An out-of-gap `pinecone.scan_factor` reached a remote query. | `SET pinecone.scan_factor` to `0` or a value in `0.5..=4.0`. |
| `pinecone.track_docs: ... has no primary key` / `the primary key of ... has N columns` / `id column "..." has type ...` | The table has no single-column key of a supported type. | Add one, or pass `id_column => '<col>'` naming a NOT NULL UNIQUE column. |
| `pinecone.track_docs: column "..." does not exist in ...` / `text column "..." has type ...` / `metadata column "..." has type ...` | A declared column is missing, or has a type with no Pinecone counterpart (`jsonb`, composites). | Fix the declaration, or list a generated column of a supported type. |
| `pinecone.track_docs: column "..." is not a valid Pinecone field name` / `collides with a reserved field name` | Names may not start with `_`, `$` or `pg_pinecone_`, or exceed 64 bytes. | Expose the column under a valid name (a generated column). |
| `pinecone.track_docs: column "...": ngram cannot be combined with ...` / `stop_words requires stemming` / `unknown option "..."` | An invalid analyzer declaration (SQLSTATE 22023, checked before any HTTP). | Follow the HINT. |
| `pinecone.track_docs: index "..." exists with a different schema` / `already exists` | The attach target differs from the declaration (42P16), or `adopt_existing` is missing (55000). | Pass a new `index_name`, match the declaration, or pass `adopt_existing => true`. |
| `pinecone.track_docs: index "..." did not become ready within Ns` | Pinecone has not finished creating the index (SQLSTATE 55000). | Check the index in the Pinecone console, then run `track_docs` again with `adopt_existing => true`. |
| `pinecone.track_docs: index_name "..." is not a valid Pinecone index name` | 1 to 45 lowercase letters, digits or `-`, starting and ending with a letter or digit (22023). | Pass a valid `index_name`, or none. |
| `pinecone.track_docs: namespace "..." (the table's schema and name) is not valid for Pinecone` | The namespace `<schema>.<table>` must be printable ASCII (22023); it is not configurable per table in this release. | Rename the table to an ASCII name; `index_name` names the index only. |
| `pinecone.track_docs: vector column "..." has type ...` / `has no declared dimension` | The dense field needs a `vector(n)` or `halfvec(n)` column (42804, 42P16). | Declare the column with its dimension, or leave `vector_column` out. |
| `pinecone.track_docs` / `backfill_docs: no pg_pinecone worker launcher is running ...` (WARNING) | Nothing syncs or backfills until a worker runs: `pg_pinecone` is not in `shared_preload_libraries`, or `pinecone.worker_enabled` is off. | See the [runbook](runbook.md#no-worker). |
| `pg_pinecone worker ... idle: extension is ... — restart Postgres to load the new worker` (WARNING, once per database and change) | The preloaded library and the installed extension differ, between an upgrade's restart and its `ALTER EXTENSION pg_pinecone UPDATE` (or after an update that came before the restart): the worker skips every round, so nothing drains. | Run the update if `compiled` is newer, restart if it is older (`pinecone.worker_version()`); see the [runbook](runbook.md#after-alter-extension-pg_pinecone-update). |
| `pinecone.track_docs: Pinecone rejected the derived schema ...` | Pinecone refused the schema (DETAIL shows it); nothing was registered. | Fix the declaration and re-run. |
| `pinecone.track_docs: partition column "..." has type ...` / `must be NOT NULL` / `is the id column` / `column "..." does not exist` / `has a nondeterministic collation` | The partition column is unsupported (0A000), nullable (42P16), the id (22023), missing (42703) or a text column whose collation is nondeterministic (0A000). | Use a NOT NULL `smallint`, `integer`, `bigint`, `text`, `varchar` or `uuid` column other than the id, with a deterministic collation. |
| `pinecone.track_docs: id column "..." is unique only together with the other columns of its primary key` | A composite key such as `(tenant_id, id)` with a partition column: ids must be unique across the table (0A000). | Add a unique index on the id column (or an identity column) and pass it as `id_column`. |
| `pinecone.track_docs: ... partition family ... overlaps / collides with / falls in ...` | Another table of the index holds a namespace the family `<schema>.<table>:*` would cover, or the reverse (42710). | Pass a different `index_name`. |
| `pg_pinecone: ... partition column "..." is NULL` / `its partition value makes a namespace of N bytes` | A write to a table with a partition column has no value (23502) or a value whose namespace passes 512 bytes (54000); the statement fails and nothing is queued. | Set the column, or shorten the value. |
| `pinecone.track_docs: ... is already tracked` / `is a partitioned table` / `has inheritance children` / `must be owner of ...` | Already tracked (42710), unsupported table kind (0A000), or not the owner (42501). | `untrack_docs` first; track partitions or children separately; run as the owner. |
| `pg_pinecone: ... id is N bytes` / `column "..." is N bytes (~T tokens)` / `column "..." contains a N-byte token` / `metadata is N bytes` | A write exceeded what Pinecone accepts (SQLSTATE 54000); the statement fails and nothing is queued. | Shorten the value, `on_oversize => 'truncate'`, or drop the column from `metadata_columns`. |
| `cannot drop column / alter column "..." of ...: it is declared in pinecone.track_docs` / `cannot drop ...: it is tracked by pinecone.track_docs` | The DDL guard (SQLSTATE 2BP01). | `pinecone.untrack_docs(tbl)` first; changing declared columns needs a new index. |
| `pinecone.untrack_docs: index "..." is also used by another tracked table` | `drop_index => true` would strand another table's documents. | Untrack that table first, or keep the index. |
| `pinecone.untrack_docs: index "..." was attached ..., not created by pg_pinecone` | `drop_index => true` would delete an index pg_pinecone attached (SQLSTATE 55000). | Delete it in Pinecone if you mean it, or pass `force => true`. |
| `pinecone.backfill_docs: the queue for ... is parked: ...` | The lane parked after repeated permanent failures (SQLSTATE 55000); a backfill would never run. | Fix the cause named in the reason, then `pinecone.unpark_docs(tbl)`. |
| `no stored API key — run pinecone.set_api_key(...) before the worker can sync documents` (`status().last_error`) | The worker syncs with the stored key, never a session key. The queue waits; no attempt is counted. | `pinecone.set_api_key(...)`. |
| `public.t: the document lane's SQL failed — ...` / `the table could not render N documents in a row — the SQL that reads its rows is failing, not the documents: ...` (`status().last_error`) | The registry or queue SQL of the table's step raised, or the table could not render 100 rows in a row (or every row of a batch of at least 10). A single row it cannot render is a dead letter (`the row could not be read: ...`), not an error. | Read the message; for a table that fails everywhere check the declared columns and the table's health, then `pinecone.unpark_docs(tbl)`. |
| `Pinecone refused N documents in a row — the index schema or the declaration has probably drifted: ...` / `... — untrack and re-track the table` (`status().last_error`; the queue parks after `pinecone.outbox_max_attempts` of them) | Pinecone refuses everything the worker sends (the index changed outside pg_pinecone, or an attached index does not match), or a declared column is gone. No dead letters are written; the rows stay queued. | Fix the index or the declaration (`untrack_docs`, then `track_docs`), then `pinecone.unpark_docs(tbl)` if the table stays tracked. |
| `pinecone.untrack_docs` / `backfill_docs` / `unpark_docs` / `flush_docs` / `failed_docs` / `retry_failed_docs: ... is not tracked by pinecone.track_docs` / `must be owner of ...` | The table has no `doc_tables` row (42704), or the caller does not own a table the verb needs ownership of (42501). | `track_docs` first; run as the owner. |
| `pinecone.flush_docs: ... timed out` / `... is parked` (WARNINGs) | The queue did not drain in time, or is parked. | `pinecone.status()`; `failed_docs`; fix the cause. |
| `pinecone.search_docs: ... is not tracked` | The table has no `doc_tables` row (SQLSTATE 42704). | `pinecone.track_docs(...)` first; `pinecone.search_doc_index(...)` searches an ad-hoc index. |
| `pinecone.search_docs: ... is still backfilling: N of M rows delivered; results are partial` (NOTICE) | A remote search ran while the first backfill, or a `backfill_docs`, was still sending rows. Once per call; the local engine searches the heap and says nothing. | Wait for `status().docs_queue_state` to be `idle` and `docs_backfill_done = docs_backfill_total`. |
| `permission denied for table ...` | The caller has no `SELECT` on the table `search_docs`, `flush_docs` or `failed_docs` was asked about (42501, before any HTTP). | Grant `SELECT`. |
| `pinecone.search_docs: k must be between 0 and 10000` / `mode must be ...` / `engine must be ...` / `<arg> must not be NULL` | An argument is out of range (22023; NULL is 22004). | Fix the argument. |
| `pinecone.search_docs: nothing to rank by` / `query must not be empty` / `vector => with query => needs field =>` / `mode => 'query_string' cannot be combined with vector =>` | The arguments do not make one scoring clause (22023). | Pass `query` or `vector`; with both, name `field`; drop `query` or use `mode => 'text'`. |
| `pinecone.search_docs: ... is partitioned by "...": a search needs one partition` / `has no partition column, so partition => does not apply` / `partition "..." is not a valid ... value` / `partition is of type ...` | `partition` is missing (22023, unless `filter` has an `$eq` on the column) or given for a table without a partition column (22023), does not read as the column's type (22023) or has a type no partition column holds (42804). | Pass `partition => <value>`; omit it on an unpartitioned table. |
| `pinecone.search_docs: field "..." is not a declared text column` | `field` is not a declared text column (42703); the HINT lists them. | Pick a declared column. |
| `pinecone.search_docs: must: ...` / `filter: ...` | `must` needs declared text columns with `$match_phrase`/`$match_all`/`$match_any` and string values; `filter` needs declared metadata columns and the `topk` operators (22023). | Follow the message; text columns go in `must`, metadata in `filter`. |
| `pinecone.search_docs: vector has N dimensions — index "..." expects M` / `... has no vector column` | The query vector does not fit the dense field (22000), or the table declared none (22023). | Pass a vector of the right length; re-track with `vector_column`. |
| `pinecone.search_docs: Pinecone rejected the query: ...` | Pinecone refused the request (22023); in `query_string` mode usually Lucene syntax. | Quote or escape operators, or use `mode => 'text'`. |
| `pinecone.search_docs: Pinecone returned 429/5xx: ...` / `request timed out` / `could not reach ...` | A monthly quota (53400) or a rate limit, an outage, a timeout or an unreachable host (58030). | The HINT says whether it clears by itself; `engine => 'local'` avoids Pinecone. |
| `pinecone.search_docs: mode => 'query_string' is not supported on the local engine` / `must/filter are not supported ...` / `vector => is not supported ...` | `engine => 'local'` takes `query`, `field` and `k` only (0A000). | Use `mode => 'text'`, a `WHERE` on the join, an `ORDER BY` on the vector column, or `engine => 'remote'`. |
| `pinecone.search_docs: field => is required for the local engine on ..., which is not tracked` | An untracked table needs the column to search (22023). | Pass `field => '<column>'`, or `track_docs` the table. |
| `pinecone.search_docs: ... has no primary key` / `the primary key of ... has N columns` / `id column "..." of ... has type ...` | An untracked table's ids are its single-column primary key (42P16, 0A000). | Add one, or `track_docs` it with `id_column`. |
| `pinecone.search_docs: column "..." does not exist in ...` / `column "..." of ... has type jsonb` | `field` on an untracked table names no column (42703) or a column that is not `text`, `varchar` or `char` (42804). | Name a text column. |
| `pinecone.headline: column "..." does not exist in ...` | `column` names no column of `tbl` (42703). | Fix the name. |

## Limits

- **Reads are not retried.** An index scan, `pinecone.topk` and `search_docs` fail on a 429 or a 5xx
  at once (the client retries those on writes, three attempts with `Retry-After` honored; a read is
  retried only once, against a fresh host lookup, when the connection fails). Retry with backoff in
  the application, or spread tenants over namespaces with the partition column.
- **The per-namespace read ceiling is soft and per namespace.** The 0.9.0 box run saw about 245 to 250
  accepted queries per second on a 1M x 768d namespace (queueing, no 429) and 429s after a 14 s burst
  above 290 per second on a 100k one; 100 tenants of 5,000 rows served 1,084 per second at 16 clients
  with none. One run each on one day: the mechanism and the documented limit are not established.
- **Cancel is timeout-bound.** Every `pg_pinecone` remote call blocks; a cancelled statement only takes
  effect once the current HTTP call unblocks — up to a few multiples of `pinecone.timeout_ms` in the
  worst case, not immediately.
- **The session-GUC key is readable.** `SHOW pinecone.api_key` reveals it to the session that set
  it — prefer `pinecone.set_api_key` for anything long-lived.
- **`sync_mode = 'async'` tables (and every remote-mode AM-indexed table) are eventually
  consistent, with no read-your-writes** — see the [README's consistency contract](../README.md#the-consistency-contract).
- **Document search is eventually consistent, with no read-your-writes.** A committed change is
  delivered within about a worker interval and searchable up to about a minute later — see
  [Freshness](#document-sync-track_docs). There is no automatic fallback to the local engine in 0.8:
  an outage is an error whose HINT names `engine => 'local'`.
- **`pinecone.promote`'s `CREATE TRIGGER` lock window is milliseconds for `sync_mode = 'async'`**
  (default) — index create/wait runs first, before any table lock; `sync_mode = 'sync'` instead
  holds the lock for the whole inline backfill.
- **A remote-mode index pushes one equality filter column to Pinecone, nothing else.** With a
  filter column (`USING pinecone (embedding, col)`, or `create_index(..., filter_column => 'col')`),
  `WHERE col = v` is sent as the metadata filter `{"col": {"$eq": v}}`, so Pinecone filters inside
  the search; `aminsert` and the backfill sync the key as `{"col": v}` metadata (a NULL key sends
  none and never matches). Every other qual runs after the index returns a batch, so a selective
  one grows `topK` (capped by `pinecone.am_max_topk`) or ends the scan early — see
  [Iterative scan](#iterative-scan). A **native-engine** index pushes the same one equality column,
  compared per record inside the list scan (no bitmap); see the
  [README's filtered-search section](../README.md#filtered-search-natives-case).
- **Remote filter keys: limits.** Pinecone compares numbers as f64, so a key of 2^53 or more is
  rechecked against the heap. A text key counts against Pinecone's 40 KB of metadata per vector;
  a larger value fails its upsert and parks the queue. Renaming the key column needs a `REINDEX`;
  scans error until then. An attached index (`pinecone_index`) runs no backfill: its namespace
  must already carry the metadata, or filtered reads miss rows (see the runbook).
- **The planner does not price a pushed key's size.** `amcostestimate` prices a pushed key at the
  1,000-row pool (the pinned `rescore_multiplier`, if any) and the ramped lists its scan uses, and gives a 1 % and a 10 % key that same cost, so
  it can pick the exact filter-first scan (mode L, what a key at or below `pinecone.mode_l_threshold`
  gets) where the sequential scan is faster: 33 to 34 ms against 17 to 20 ms at one client and 109 to
  117 ms against 85 to 97 ms at eight (1M x 768d, 1 % key, measured before the estimate priced the pool
  and the lists). Teaching the estimate the key's selectivity is a known follow-up; until then
  `SET pinecone.am_enabled = off` steers a query away.
- **Vector ids are capped at 512 bytes** (Pinecone's own limit). `id_column` must be
  `text`/`varchar`/`uuid`/`int2`/`int4`/`int8` — types with a canonical, GUC-independent text form.
- **Document ids are capped at 512 bytes, text fields at 100 KB / 9,000 tokens (estimated as
  runs of letters and digits; Pinecone allows 10,000 real tokens), metadata at 40 KB per document**, and a
  request carries at most 1,000 documents and 1.5 MB (the drain packs by measured bytes).
- **A document table with a partition column makes one namespace per value** and Pinecone limits
  namespaces per index by plan (Starter 100, Standard 100,000, Enterprise 1,000,000, unverified here:
  the cap's error is Pinecone's own). A namespace name is at most 512 bytes; ids stay unique across
  the table; a search names exactly one partition.
- **Backups and many namespaces.** `backup_readiness` warns above 2,000 namespaces in the index: the
  figure is not verified against live Pinecone. A restore also brings back a tenant removed after the
  backup (`delete_partition` it again).
- **A preloaded `.so` changes only with a restart, for every `pinecone.*` call — not just the
  worker, so an upgrade restarts first and runs `ALTER EXTENSION` after.** `shared_preload_libraries =
  'pg_pinecone'` loads the library once per postmaster lifetime, shared by every backend until a real
  restart; `ALTER EXTENSION` updates the catalog, not that already-loaded code, and a script that
  creates a function the loaded library lacks fails. The worker has its own version guard: every
  round it compares its compiled version with `pg_extension.extversion` and, while they differ,
  skips the whole round (no drain, backfill, recovery or maintenance) after one WARNING per database;
  `pinecone.worker_version()` returns `(compiled, installed, idle)`. Every other call has no guard:
  it runs the loaded library against whatever catalog is installed. See "After `ALTER EXTENSION
  pg_pinecone UPDATE`" in [`docs/runbook.md`](runbook.md#after-alter-extension-pg_pinecone-update).
- **A native-engine index caps vector dimension at whatever fits one centroid directory entry per
  page** (page-format-derived, similar order of magnitude to pgvector's own `ivfflat` cap) —
  enforced with a clear error at `CREATE INDEX`/`REINDEX` time, regardless of table size.
- **A composite index takes at most one filter column** (`smallint`, `integer`, `bigint`, `text`
  or `varchar` with a deterministic collation): `pinecone.create_index(..., filter_column => 'col')`
  for `native` and `remote`, or `CREATE INDEX ... USING pinecone (col opclass, filter_col)
  WITH (mode = 'local' | 'remote')`.
