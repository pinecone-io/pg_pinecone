# Changelog

## Unreleased

- **`native` is the default engine of `pinecone.create_index`** (it was `pgvector`). The gate the README tied the
  default to is met: on the 1M x 768-d box (AWS `r6id.4xlarge`, 2026-10-19, one client, PostgreSQL 17), native takes
  1.12 / 1.74 ms p50 at recall 0.95 / 0.99 (its calibrated default: 0.958 at 1.29 ms) against hnsw's 1.59 / 3.86 ms;
  its index is 136 MB against 4,094 MB and builds in 39 s against 62 s, and the filtered cells were already native's.
  `engine => 'pgvector'` and `'remote'` and `CREATE INDEX` are unchanged: a script that relied on the old default
  says `engine => 'pgvector'`. Create the index after the bulk load: below `small_n_threshold` (10,000 rows) a native
  index is flat, an exact scan, until `REINDEX`.
- **`set_engine(tbl, 'native')` moves an hnsw index `create_index` built to native** (it refused: "not supported
  yet"). To move an index created before 0.10.1 to native: `SELECT pinecone.set_engine('docs', 'native')`. It builds
  the native index while reads keep the hnsw one, drops the hnsw index and gives its name to the native one; the table
  then leaves `status()` and `topk`, like any native table. A table `track` or `promote` also registered is refused;
  `pgvector -> remote` goes through `native`.
- **Fix: `DROP INDEX` of an hnsw index `create_index` built (or `DROP TABLE`) kept its `pinecone.tables` row**, so
  `create_index` refused the table (`already managed with engine 'pgvector'`). The `sql_drop` event trigger now deletes
  that row with the table's last hnsw index; a row `track` or `promote` also owns stays.
- **Fix: a native inner-product index put most rows of offset data in one list** (16,384 of 20,000 on a 256-d set
  with a norm-10 offset), so a query read most of the table. Rows are now clustered and assigned by L2, and lists are
  still ranked by inner product: at 1,000 rows scanned, recall@10 on that set goes from 0.17 to 0.92. No format
  change, but `REINDEX` an inner-product native index built before: it keeps its skewed lists, and a row inserted
  after the upgrade is placed by L2 against its old centroids, which queries may probe late.
- **First in-place upgrade, 0.10.0 to 0.10.1** (`pg_pinecone--0.10.0--0.10.1.sql`): the script replaces
  `pinecone.create_index` (the new default) and `pinecone._on_sql_drop` (the fix above), and deletes the rows 0.10.0
  kept for hnsw indexes dropped under it. Indexes already built stay as they are (an hnsw index made under 0.10.0
  too). Install, restart, then `ALTER EXTENSION pg_pinecone UPDATE`
  ([runbook](docs/runbook.md#upgrading-from-0100-and-later)): until the update runs, a call without `engine` still
  builds hnsw.

## 0.10.0 — 2026-10-17

At a glance:

- **Renamed: `pinegres` is `pg_pinecone`**, and everything SQL-visible is `pinecone` (`pinecone.topk(..)`,
  `SET pinecone.api_key`, `USING pinecone`). No upgrade from `pinegres`: reinstall (below). From 0.10.0 on,
  upgrades are in place: install, restart, `ALTER EXTENSION pg_pinecone UPDATE`.
- Native engine on public algorithms: RaBitQ codes, a blocked scan with an error-bound early stop, and
  every build calibrates its probe count and rescore pool to `target_recall` (on rows, or on a table of
  real queries via `calibration_queries`). On the 1M x 768-d box the default query went from 7.1 ms p50
  at recall 0.95 to 1.3 ms, before the blocked scan's further 20 to 40 %.
- Remote engine: backfill rounds without naps and requests sized to Pinecone's limits, 847 rows/s end
  to end at 768-d (was 55); `LIMIT 100` is one request.
- Apache License 2.0.

- **Renamed: `pinegres` is `pg_pinecone`.** The package is `pg_pinecone`: `CREATE EXTENSION pg_pinecone`,
  `shared_preload_libraries = 'pg_pinecone'`, the library, the crates, the tarballs
  (`pg_pinecone-<version>-pg17-linux-x86_64.tar.gz`). Everything SQL-visible is `pinecone`: the schema
  (`pinecone.promote(..)`, `pinecone.topk(..)`, `pinecone.status()`), the settings (`SET pinecone.api_key`,
  `pinecone.worker_naptime_ms`), the index access method (`USING pinecone`) and the default index name
  (`<table>_<column>_pinecone_idx`); the scripts' variables are `PG_PINECONE_*`. Postgres cannot rename an
  installed extension, so **there is no upgrade from `pinegres`: drop it and set `pg_pinecone` up as a new
  install** ([runbook](docs/runbook.md#upgrading-from-pinegres-090-or-earlier)). From 0.10.0 on, upgrades are
  in place.
- **License.** The repository is licensed under the Apache License 2.0 (`LICENSE`); the earlier
  binary-only terms are withdrawn.

### In-place upgrades from 0.10.0 on

- `ALTER EXTENSION pg_pinecone UPDATE` takes a release to the next without dropping the extension: every
  release from 0.10.0 on ships `pg_pinecone--<prev>--<new>.sql` in its tarballs. Install the new files,
  restart Postgres, then update: the worker idles until `pinecone.worker_version()` shows equal versions.
  See the [runbook](docs/runbook.md#upgrading-from-0100-and-later).
- Every PR proves the upgrade: CI upgrades the latest release's tarball (install, restart, update) under a
  fixture of all lanes (`upgrade-test`, `scripts/upgrade-drill.sh`) and compares the catalog and the fixture's
  own schema. Until v0.10.0 is tagged there is nothing to upgrade from and the job skips its drill (the
  script tooling is still tested); the drill ran against `pinegres` 0.9.0 before the rename.
- `scripts/upgrade/diff-schema.py` proposes the upgrade script from two generated schemas. Rules for
  schema changes: [`docs/contributing.md`](docs/contributing.md#schema-changes-and-upgrade-scripts).
- The access-method metapage carries a format version, so a later on-disk change is refused with `REINDEX`
  instead of misread: 0.10.0 writes format 3, a newer one raises 55000 `REINDEX it`, and
  `pinecone.index_format(idx)` reports it. See the [reference](docs/reference.md#index-format-version).

### Remote engine: backfill and fetch sizing

- **Backfill rounds without naps.** The worker runs the chunks of a vector table's backfill (an access-method
  index by heap block, `promote` by primary key) back to back until `pinecone.backfill_round_ms` (1 s) is
  spent, at least one per round, as it does for document backfills and partition rebuilds, and starts the next
  round at once while a backfill, a partition rebuild, a `recover_from_remote` or a document backfill has
  chunks left. It ran one chunk, then napped `pinecone.worker_naptime_ms` (2 s): 43 to 55 rows/s at the
  defaults on the 0.9.0 box, four fifths of every round asleep. It now naps when its queues are idle or what
  is left is blocked (a table in backoff or parked, a failed partition). Against a mock that answers in 50 ms,
  20,000 rows took 21 s on the trigger lane (11 rounds, 10 naps) and 38 s on the access method (19 rounds,
  18 naps); they take 1.3 s. The requests, the cursors and the flip to `remote` are unchanged, a chunk that
  fails mid-round leaves the cursor at the last good one, and live writes still go out first in every
  round. See the [reference](docs/reference.md#backfill-rate).
- **Upsert requests are sized to Pinecone's limits.** A request holds as many vectors as fit 1,000 records and
  2 MB, counted as Pinecone counts it (4 bytes per value, the id, the metadata; a 1.9 MB budget): 610 at 768
  dimensions, 307 at 1,536, 1,000 at 384. 0.9.0 budgeted the JSON text, each float at twice its digits, and
  sent 99 vectors of 768 values (2.2 MB of JSON). The body is now compact JSON with each float in its
  shortest form (22 KB a 768-d vector became 8.5 KB), document requests included, so a 768-d request is about
  5 MB on the wire: mind a proxy with a lower body limit. An access-method chunk ends before the heap block
  that would overflow its request, instead of sending the overflow as a second, small request. 6,000 rows of
  768 values are 10 requests on the trigger lane (were 61) and 11 on the access method (were 89); live
  writes are packed the same way (1,300 such rows: 3 requests, were 8). Measured on the box (6,000 rows x 768-d,
  defaults, a fresh serverless index): 847 rows/s from `create_index` to worker idle, 1,475 from the first
  request to the last response, against 43 to 55.
- **`pinecone.am_min_topk` defaults to 100 (was 50).** Postgres 17 and 18 tell an index access method
  nothing of the `LIMIT` above it, so the first request cannot be sized to it: `LIMIT 100` took two (50,
  then 200: 35.3 ms against 17.7 ms in the 0.9.0 box run, and half the accepted queries per second). At 100
  any `LIMIT` up to 100 is one request. It costs about 0.3 to 0.9 ms per query at a `LIMIT` of 50 or less
  (the 0.9.0 box cells at 1M and 100k rows; read units do not depend on `topK`). `SET pinecone.am_min_topk
  = 50` restores 0.9.0. The regrow ladder is 100, 400, 1,600, 6,400, 10,000.
- **Docs.** Reads are not retried on a 429 or a 5xx, and what to do about it
  ([runbook](docs/runbook.md#reads-that-get-a-429)); Pinecone's per-namespace read ceiling is soft and per
  namespace; the backfill rate of each lane at the default naptime.

### Native engine: quick wins

"Box" is the 0.10 profile on one AWS `r6id.4xlarge` (PostgreSQL 18.6, wiki_dpr 1M x 768d, one client, p50);
details in the [reference](docs/reference.md#native-engine-probe-count-pool-and-storage-plain).

- **SIMD exact distances.** L2, dot and cosine run on AVX-512F (Ice Lake, Zen 4 and later), AVX2+FMA or
  NEON, with a portable fallback, instead of one sequential f32 sum that waits on every add: a 768-d
  distance takes 37 ns where it took 348 (cosine 58 ns, was 1,014; Apple M4 Max). On the box the loop
  cost 1.1 ms in the rescore and 0.95 ms ranking the 1,000 centroids of each 11.45 ms default query.
  k-means and inserts use the kernels too. Each path is checked against the scalar one once per process;
  they agree to about 1e-6 relative, not bit for bit, so an index built from one seed can differ between
  CPUs (not between thread counts). Microbench: `cargo run --release --bin distance` in
  `bench/vector/kernels`.
- **`STORAGE PLAIN` notice.** `CREATE INDEX` and `create_index(.., engine => 'native')` print the
  `ALTER TABLE .. SET STORAGE PLAIN` statement when a vector column of 501 to 1,997 dimensions is
  `EXTERNAL` or `EXTENDED`: every rescored row then pays a TOAST fetch, 0.65 ms per 200 rows on the box
  (2.96 to 2.29 ms per query at pool 200 and 20 lists). The column is never changed; `VACUUM FULL` moves
  the existing rows. All three together, on a patched 0.9.0 (20, 25 and 50 lists, pools of 200, 300 and
  500): 1.38, 1.53 and 2.30 ms at recall 0.956, 0.973 and 0.990, against 11.45 ms at 0.998.

### Native engine: calibrated defaults

- **`CREATE INDEX` and `REINDEX` calibrate the probe count and the rescore pool.** The quick wins'
  constants (26 lists, 300 rows) held on wiki_dpr 1M x 768d and nowhere else: recall@10 0.912 on SIFT1M,
  0.648 on a harder synthetic corpus, 0.28 behind a 1 % pushed key with `iterative_scan = off`, `LIMIT 100`
  0.84 (box measurements of those defaults). A clustered index now draws `min(200, 2 % of the rows)`
  held-out rows as queries (at least 50; fewer above a million rows; 1.7 s at a million 768-d rows with
  eight workers, 0.04 s at 20,000), computes their exact top 10 and keeps the cheapest
  `(nprobe, pool)` of a fixed grid (probes 8, 12, 16, 25, 37, 50, 75, 100, 150, 200 and on to `n_lists`;
  pools 100, 200, 300, 500, 1,000, 2,000; cost = codes scanned + 100 per rescored row) whose sample
  recall@10 (a row is left out of its own search), less two standard errors, reaches the target plus a
  0.01 safety offset (the plain target when no pair does; the offset stops halfway to 1.0, so 0.99 asks
  for 0.995). The winner's probe count is then bisected back toward the grid value below it (at most four
  more scans of the sample). A NOTICE says what it chose; the pair and the sample recall live in the
  metapage (no format change: format 3 is new in 0.10).
  [Reference](docs/reference.md#native-engine-calibrated-defaults).
- **`target_recall`** reloption: default 0.95, 0.5 to 0.999 (SQLSTATE 22023 outside); `ALTER INDEX .. SET`
  applies at the next build.
- **`calibration_queries`** reloption (`'schema.table'` or `'schema.table(column)'`): calibrate on a sample of
  your own queries instead of table rows, for workloads whose queries are unlike their rows (questions
  against passages). On wiki_dpr 1M six builds read fresh recall@10 0.913 to 0.958 calibrated on rows and
  0.965 to 0.981 calibrated on 200 held-out vectors. Up to 200 queries are drawn by `seed` (at least 50),
  their exact top 10 is taken over the whole table with nothing left out, and the grid, offset and
  refinement are unchanged. The table is read as the table's owner (a superuser's `REINDEX`,
  `reindex_native_indexes()` and the maintenance worker never read what the owner cannot); at `CREATE INDEX`
  a table that cannot be used is an error, at a rebuild a WARNING and the build falls back to table rows
  (an error raised while reading, a view's function, is reported by its SQLSTATE and logged, not echoed to
  the rebuild's runner). A table over 100,000 rows is refused by a count, not the planner's estimate alone;
  a foreign table is refused. `ALTER INDEX .. SET` applies at the next `REINDEX`.
  [Runbook](docs/runbook.md#calibrate-with-real-queries).
- **`pinecone.index_calibration(index)`** (new; callable once Postgres has restarted onto 0.10) returns the
  probe count and pool a scan uses by default and what the build measured (`calib_nprobe`, `calib_pool`,
  `calib_recall`, `calib_samples`, `calib_reached`, `target_recall`) and what the sample was made of
  (`queries_source`: `rows`, `table` or `fallback`; `queries_count`). Needs `SELECT` on the table.
- **Fallbacks.** A flat index scans exactly and a table below 500 rows is not calibrated (its sample would
  be a tenth of it): they keep the 0.9.0 constants, `ceil(4 x sqrt(n_lists))` lists and 1,000 rows, which
  the function reports. A target no pair reaches keeps the cheapest pair near the best recall and says so.
  The `max(ceil(0.8 x sqrt(n_lists)), 25)` probe rule and the pool of 300 are gone.
- **Overrides.** `rescore_multiplier` now defaults to 0 (calibrate); set, it pins the pool and the build
  calibrates the probe count for it. `pinecone.scan_fraction` and `pinecone.rescore_multiplier` still win
  on every path. A scan with a pushed key keeps the 0.9.0 pool (1,000 rows) unless one is pinned, and an
  iterative scan its 0.9.0 list count: calibration measures unfiltered scans.
- **Limits.** By default calibration uses rows of the table as queries: a query set unlike the data
  (questions against passages) may land below the target (wiki_dpr rows read 0.01 to 0.04 above held-out
  queries; the 0.01 offset covers part of that): calibrate on real queries with `calibration_queries`, or use
  the knobs the docs give to measure and pin. It targets recall@10,
  and the pool it picks is often the 100-row minimum, so set `rescore_multiplier = 10` for `LIMIT 100`.
  Inserts do not recalibrate; `REINDEX` does.
- **The planner prices the scan it will get.** A scan with a pushed key is costed at the 1,000-row pool (the
  pinned `rescore_multiplier`, if any) and the ramped lists it uses, not the calibrated pair, and list pages
  are counted from the layout: 64 records a 768-d page and 48 at 1,024-d, where the estimate said 71 and 55.
  Session `SET`s of `pinecone.rescore_multiplier` and `scan_fraction` stay out of the estimate, so a sweep
  does not move plans.
- **A NaN distance ranks last in the calibration's exact truth, whatever its sign.** A NaN with its sign
  bit set sorted before every number (x86 makes one of the cosine distance's `inf / inf`, aarch64 one of
  the inner product's negated NaN), so rows of components near 1e20 became the true neighbours of every
  sample query (the scan ranks them last) and the build could report a target not reached or pick a wider
  pair.

### Native engine: RaBitQ codes (on-disk format 3)

- **A native index of an older on-disk format is offline until `REINDEX`.** The native engine stores
  RaBitQ codes (on-disk format 3), not the FJLT/SimHash sketch (formats 1 and 2, written by `pinegres` 0.9.0
  and by snapshots of main). Such an index is refused per entry point: the planner picks another plan (one
  WARNING per backend), a scan that is forced and every `INSERT` into its table raise 55000 `pg_pinecone index
  '...' has format version N (this build reads 3): REINDEX it`, `VACUUM` skips it with a WARNING,
  `pinecone.index_format()` reports `readable = false`, `promote_index` refuses it. `ALTER EXTENSION
  pg_pinecone UPDATE` lists them in a WARNING; `SELECT pinecone.reindex_native_indexes()` (new; returns
  `(index_name, outcome)`; `dry_run => true` only lists) rebuilds the ones the caller owns, about 4 minutes
  per million 768-d rows. Remote-mode indexes carry no codes and are unaffected. A `pinegres` index goes
  with `DROP EXTENSION pinegres` (the rename is a clean break), so a released install never meets this; the
  guard is for the next format change. Runbook:
  [Upgrading from 0.10.0 and later](docs/runbook.md#upgrading-from-0100-and-later).
- **RaBitQ** (Gao and Long, SIGMOD 2024), implemented clean-room from the paper. A record is a `D`-bit code
  (the dimension padded to 64, at least 256: 96 bytes at 768 dimensions) from a seeded random rotation of the offset to its
  list's centroid, three floats, the code's popcount and the TID: 116 bytes (125 with a filter column). A scan
  rotates the query once, quantises it to 4 bits per probed list and scores a list's records in blocks (AVX-512,
  AVX2, NEON or scalar, chosen at runtime); the estimate is unbiased and has an error bound, which ends the
  rescore of `auto`'s first batch early (see below). IP and cosine pick, per probed list, the centred or the plain form of the query, whichever has the
  smaller norm. The pool, the rescore, the iterative scan, the filter modes and `last_scan_stats()` are as
  before. [Reference](docs/reference.md#native-engine-rabitq-codes).
- **Minimum code width.** A column of fewer than 256 dimensions is zero-padded to 256 and gets a 256-bit
  code (a record is 52 bytes, 150 to a page), because the estimator's error shrinks with the square root of
  the code width whatever the data's dimension: a pool of 300 over every list read recall@10 0.919 on a
  synthetic 64-d mixture and 0.870 on 24-d uniform data behind a selective `WHERE` with 64-bit codes, 1.000 and
  0.925 with 256. Format 3 is new in 0.10, so no further bump. Inner product over data with a large common
  offset collapses into few IVF lists (a limitation of the list assignment, not of the codes).
- **`k_bits` is ignored** (the codes are `D` bits): still accepted, with a NOTICE at `CREATE INDEX` and
  `REINDEX`. A vector with a NaN or infinite component is refused at build and insert.
- **`VACUUM` of a native index honours `vacuum_cost_delay`**: the sweep calls `vacuum_delay_point()` between
  list pages (it ran unthrottled, autovacuum included).
- **The earlier random-projection code is gone** from the source (`projector.rs`, the Hamming kernel). Rule
  for new quantiser and scan code: [contributing](docs/contributing.md#provenance).

### Native engine: blocked scan and error-bound early stop

- **Breaking for snapshots of `main`: page layout 1.** A list page stores its codes in blocks (below), so a
  native index built between the RaBitQ codes and this change (format 3, layout 0) is refused like an older
  format: `has format version 3 with page layout 0 (this build reads layout 1): REINDEX it`. The layout is the
  second byte of the metapage's format word; `pinecone.index_format()` reports such an index as format 3,
  `readable = false`. No release is affected: 0.9.0 indexes are format 1 or 2 and need the `REINDEX` already.
  A page holds 64 records at 768 dimensions, 71 before (a block does not split), so a native index is about 10 %
  larger there; 150 up to 256 dimensions (163), 32 at 1,536 (38).
- **Blocked scan.** A page keeps its records in blocks of 64, 32 or 16, byte-interleaved (the widest width that
  fits most records on a page: a 64-record block of 1,024-d codes would not fit one). The query is expanded
  into one 16-entry table per four coordinates and a lookup (`vpshufb`, `tbl`) adds a nibble of 16 to 64
  records at once, as in FastScan (VLDB 2015); each kernel is checked against the scalar one when the backend
  first scans. Estimates and bounds are bit-identical to the one-record path. At 768 dimensions the kernel takes
  2.0 (AVX-512), 2.7 (AVX2) and 2.4 (NEON) ns a record against 14.4, 20.5 and 7.4 for AND+popcount; the scan
  phase of a 26-list probe set (kernel, scoring, candidate buffer) takes 9.7 ns a record on Ice Lake, from 32.5
  (3.4x; 4.2x at 256 dimensions, 1.8x at 1,536) and 5.3 on an M4 Max, from 11.6. A pushed key tests a block's
  stored filter values at once; a block with no match is not scored. VACUUM compacts a page into blocks.
- **Early stop (`pinecone.scan_prune`, on).** The rescore walks the pool best estimate first, 32 rows at a
  time, and ends when no further candidate's lower bound (its estimate moved by twice the estimator's error,
  3.8 sigma) can beat the k-th best exact distance found (k = a tenth of the pool, at least 10). The
  estimator's own 1.9-sigma bound is too tight to prune with: up to 16 % of the queries on five synthetic sets
  lost a top-10 neighbour; at twice it none of 9,000 runs did. `rescore_multiplier x 100` stays the hard cap,
  and the walk goes on until `min(pool, 100)` rows are found. Ids and order equal the whole-pool rescore on
  four synthetic sets (200 queries each, pools of 100 to 1,000). At the calibrated pool of those sets, the
  100-row floor, nothing is saved; at a pool of 1,000 the scan rescored on average 32 to 77 % of it. **Only
  the first batch of `auto` stops early; an explicit `iterative_scan = off` rescores the whole pool, as 0.9.0
  did** (it is one batch, and the index cannot see a residual `WHERE` that filters it). A `LIMIT` above the
  rows `auto`'s first batch rescored, or a residual `WHERE` that discards some, makes the executor ask for
  more and `auto` goes on (next bullet); `SET pinecone.scan_prune = off` gives `auto`'s first batch the whole
  pool too. An iterative scan, mode L and the flat path are unchanged: an iterative scan rescores every batch
  whole, as a batch that left candidates out would hand them over later, after farther rows.
- **`pinecone.iterative_scan` defaults to `auto`: a `LIMIT` above the rescored pool is filled.** With the
  calibrated pool (100 rows) and the early stop, `LIMIT 200` on a native index returned 100 rows and no error
  (0.9.0's pool of 1,000 hid it). A native scan now returns `off`'s first batch, from the same lists and
  pool, with the early stop above (the same top `pool / 10` rows, a lower `rescored` in
  `last_scan_stats()`, a `LIMIT 10` that costs what it did or less), and only if the executor asks for more
  goes on as `relaxed_order`: it probes again from the nearest lists, leaving out the rows already returned,
  up to `pinecone.max_scan_tuples` rescored rows. On 20,000 uniform 6-d rows `LIMIT 200` and `LIMIT 1000`
  return 200 and 1,000 rows (100 and 100 before) at recall@200 0.93 (0.50); on 16-d uniform rows and a 32-d
  mixture it reads 0.83 and 0.80 (0.49, 0.48): rows after the first batch are the next best by estimate, so
  depth depends on the data and a larger `rescore_multiplier` buys more. A residual `WHERE` that passes few
  rows now reads up to the cap (20,000 rescored rows) before the scan ends: lower `pinecone.max_scan_tuples`, or
  `SET pinecone.iterative_scan = off` for the old "at most the pool". A remote index runs `off` as before;
  `relaxed_order` and `strict_order` are unchanged (they start with the 0.9.0 list count: 48 lists, not 8, on
  a calibrated 20,000-row index). See the [reference](docs/reference.md#iterative-scan).
- **`pinecone.last_scan_stats()`** gains `scanned`, `admitted` and `blocks_skipped` (the scan phase).
  [Reference](docs/reference.md#native-engine-blocked-scan-and-error-bound-early-stop).
- **Page parsers are fuzzed.** The list-page and directory-page parsers take 10,000 random and mutated pages
  each in the unit tests: no panic, and inconsistent counts are refused (`corrupt list page at block N`).

### Hardening

- **Release workflow token.** The build job (it runs the PR's build scripts on the dry-run trigger) has a
  read-only token that is not left in `.git/config`; only `create-release` and a new `upload` job, which runs
  after all six legs passed and never on a PR, hold `contents: write`. The tarballs reach `upload` as workflow
  artifacts.
- **rustls, h2 and chacha20 updated** for RUSTSEC-2026-0346 to 0349 and -0258 (and a yanked chacha20).
  `cargo audit` is clean but for `serde_cbor` (unmaintained, pulled in by pgrx).
- **A SIGTERM ends a vector backfill round after its current chunk.** The worker used to finish the round's
  budget first (`backfill_round_ms`, up to 60 s, plus a chunk) holding the table's lock. The budget is now
  also the round's, shared by its tables, instead of each table's.
- **`scripts/upgrade-drill.sh` fails if the base tarball has no `.sha256`** (it skipped the check silently);
  `UPGRADE_ALLOW_UNVERIFIED=1` accepts that, with a warning. CI keeps verifying.
- **`pinecone.reindex_native_indexes()` skips an index dropped meanwhile** instead of aborting the call and
  rolling back the rebuilds before it: each REINDEX runs in a subtransaction and the result lists
  `(index_name, outcome)`, `rebuilt` or `skipped` (it returned names). The call is still one transaction, so
  a rebuilt index stays locked until it returns: [runbook](docs/runbook.md#upgrading-from-0100-and-later).
- **Offline native indexes are rebuilt by scheduled maintenance.** With `pinecone.auto_maintain` on, a
  scheduled index below format 3 counts as drift 1.0 and is rebuilt in the worker's next round, whatever its
  threshold (it stayed offline until it had drifted).
- **A corrupt filter-stats page is refused, not allocated.** Its entry count went straight into an
  allocation (4 billion entries: 64 GiB, an abort where the OS refuses it). The count is now checked
  against the 256 the build stores and the page, and a filtered scan raises `corrupt stats page at block N`.
- **CI runs `cargo audit`** in the lint job; the one advisory it ignores is `serde_cbor` (unmaintained,
  pulled in by pgrx).
- **Dispatching the release workflow no longer replaces released tarballs.** `workflow_dispatch` against a
  release that has tarballs fails with a message unless `overwrite: true` is set; only a tag push
  re-attaches with `--clobber`.

## 0.9.0 and earlier

Releases 0.2 to 0.9 shipped under earlier names (0.7 to 0.9 as `pinegres`). There is no upgrade path across the
0.10.0 rename; see [docs/runbook.md](docs/runbook.md#upgrading-from-pinegres-090-or-earlier).
