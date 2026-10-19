# pg_pinecone/pgvector ANN benchmark harness

What it measures: filtered and unfiltered ANN search of pgvector (hnsw, ivfflat, forced exact seqscan) against
pg_pinecone's native local engine (`pg_pinecone_native` = `CREATE INDEX ... USING pinecone ... WITH (mode = 'local')`), on synthetic
clustered data and on real embeddings (SIFT1M 128d, wiki_dpr 768d): tie-aware recall against exact ground truth, latency
percentiles, QPS under concurrency, build time and size, churn. The harness runs its **own throwaway Postgres**
(`harness/pgctl.py`); it never touches a shared instance and makes no Pinecone call, except the `remote*` configs, which
measure the Pinecone-backed lanes (`pg_pinecone_remote`, `pg_pinecone_remote_topk`) against a real serverless index (see
"Pinecone-backed leg (box)" below).

Evidence velocity over elegance, but reproducible: one entrypoint (`run.py --config ...`), configs as checked-in YAML,
results as timestamped JSON plus a generated markdown summary.

Write-ups:
- `REPORT-residual-filter.md`: iterative scan vs pgvector hnsw for a pushed key plus a residual `WHERE`, 1M x 768d, 2026-10-07
  (how to rerun: `configs/README-residual.md`).
- `REPORT-halfvec.md`: pgvector FP16 (halfvec) against FP32 hnsw and pg_pinecone native from `main` (0.10.0), 1M x 768d, one AWS box,
  2026-10-16 (`configs/halfvec_1m.yaml`; box facts `box/BOX-2026-10-16-halfvec.md`).

**Last verified:** 2026-10-14 on an AWS `r6id.4xlarge` (PostgreSQL 17.11 PGDG, pgvector 0.8.7, the pg_pinecone 0.9.0 release tarball, Python 3.12.3):
`local_1m`, `residual-c1`, `residual-c2`, `residual-ext` (its mode-L system re-run as `residual-ext-model`), `remote_1m`, `remote_filtered`,
`remote_partition` and both `remote_lanes_*` configs, all from clean commits (the residual cells' recall equals the 2026-10-07 run
within 0.023). On 2026-10-12 (main dc3e29c, M4 Max, pgrx cassert Postgres 17.10, Python 3.14.6) `configs/smoke-native.yaml` ran 50,000 rows in 411 s,
44/44 sweep points valid: pg_pinecone recall 0.812 / 0.879 / 0.930 at `rescore_multiplier` 4 / 10 / 20, hnsw 0.646 / 0.843 at `ef_search` 10 / 40;
`configs/ci_tiny_exact.yaml` 12 s, 8/8 points valid. On 2026-10-16, against pg_pinecone 0.10.0 built from main (d1b57ff) with PostgreSQL 17.11 /
pgvector 0.8.7 in Docker on a laptop, `configs/smoke-halfvec-native.yaml` ran in 80 s, 76/76 points valid, every native point on its index: `target_recall` 0.90 / 0.95 / 0.98 / 0.99
calibrated (`nprobe`, pool) to (50, 200) / (75, 500) / (75, 1,000) / (75, 2,000) of 224 lists. Datasets via `scripts/download-datasets.sh`: `sift` (25 s; `sift1m.npz` byte-identical to the
2026-09 cache) and `wikidpr` (9 min on the laptop, 7 min on the box; 5.3 GB of shards became a 3.38 GB `wikidpr_768d_1m.npz` whose base set, dataset key and
held-out query/bank cluster ids match the 2026-09 artifact's recorded fingerprints; its 160 warmup vectors have none). Not run in any of these: SIFT1M and every
config not named here.

## Environment

| variable | meaning | default |
|---|---|---|
| `PG_PINECONE_BENCH_PG` | Postgres 17 install prefix (`<prefix>/bin/pg_ctl`) with pgvector and pg_pinecone installed | `~/.pgrx/17.10/pgrx-install` (`PGRX_PG_BIN`, the `bin/` dir, still works) |
| `PG_PINECONE_BENCH_DATA` | datasets, ground-truth caches and scratch Postgres dirs, under `vector/` | `bench/.data` |
| `BENCH_TUNED_CONF` | settings appended to `postgresql.conf` after `initdb` | `pg/tuned.conf` (sized for a 36 GB laptop; `pg/tuned.linux-aws.conf` is the r6id.4xlarge box) |
| `BENCH_RESULTS_DIR` | where results JSON and markdown land | `bench/vector/results/` |
| `PINECONE_API_KEY` or `EE_ENV` | the key (or a file holding `PINECONE_API_KEY=...`) of the `remote*` configs; never printed or stored in results | unset |
| `STATE`, `RAW` | the Pinecone legs' spend tally, events, build state and backfill CSVs | `results/state`, `results/raw` |

Results land in `bench/vector/results/` (gitignored): `<name>_<timestamp>.json` plus the matching `.md`. Run every command
from `bench/vector`. Only the `remote*` configs call Pinecone; `env -u PINECONE_API_KEY` costs nothing and is used below for the others.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # numpy, psycopg[binary], pgvector, pyyaml
export PG_PINECONE_BENCH_DATA="$(cd .. && pwd)/.data"
```

### Install pg_pinecone

The harness never builds the engine: it needs a Postgres 17 with pgvector and pg_pinecone installed, in which it runs `CREATE
EXTENSION` inside its own throwaway instance. Use a PRIVATE pgrx home so other worktrees' `cargo pgrx install` / `test` cannot
overwrite the build under test. Re-install after every engine-side change before trusting a `pg_pinecone_native` result.

```
export PGRX_HOME=/path/to/private-pgrx-home      # any empty dir, e.g. $PG_PINECONE_BENCH_DATA/pgrx
mkdir -p "$PGRX_HOME" && cp -cR ~/.pgrx/17.10 "$PGRX_HOME/17.10"     # a pgrx PG 17.10 that has pgvector installed
printf '[configs]\npg17 = "%s/17.10/pgrx-install/bin/pg_config"\n' "$PGRX_HOME" > "$PGRX_HOME/config.toml"
(cd ../../pg_pinecone && env -u PINECONE_API_KEY cargo pgrx install --release \
    --pg-config "$PGRX_HOME/17.10/pgrx-install/bin/pg_config")
export PG_PINECONE_BENCH_PG="$PGRX_HOME/17.10/pgrx-install"
```

The `cp` line assumes `cargo pgrx init --pg17 download` was run once and pgvector installed into it with
`scripts/install-pgvector.sh <pg_config>` (the CI recipe; not run in this pass). That Postgres is a `--enable-cassert` build
(stamped per run as `pg_debug_assertions`): fine for plumbing and recall, but absolute latencies carry the assertion tax. For
timing runs use a release Postgres built from source (add pgvector with `../../scripts/install-pgvector.sh <pg_config>`).

### Datasets

```
scripts/download-datasets.sh sift       # SIFT1M: 168 MB download -> sift1m.npz 521 MB
scripts/download-datasets.sh wikidpr    # wiki_dpr: 5.3 GB download (4 shards) -> wikidpr_768d_1m.npz 3.4 GB
```

Both land in `$PG_PINECONE_BENCH_DATA/vector/datasets_cache`. The synthetic datasets (everything named smoke, sweep, spotcheck, full)
are generated on first use; nothing to download. SIFT1M is only read by configs whose `sift1m:` block is enabled; wiki_dpr
only by the `kind: real` configs (baselines_exact_real*, wikidpr_1m_pg_pinecone, followup_c1_wikidpr, residual-*, except residual-smoke).

## Run

```
env -u PINECONE_API_KEY .venv/bin/python run.py --config configs/smoke-native.yaml \
    --data-dir "$PG_PINECONE_BENCH_DATA/vector/pgdata-smoke" -v
```

`results/<name>_<timestamp>.json` holds the full data, the `.md` the summary (build stats, latency-at-recall tables, raw
sweeps, churn, SIFT1M sanity check). A run of record must come from a clean committed tree; `save_results` refuses a dirty one
(see "Provenance stamps" below) unless you pass `--allow-dirty`, which marks the file and report DIRTY.

**Running two instances at once (e.g. a long baseline series plus a plumbing check) needs a distinct `--data-dir` each.**
`PgInstance.initdb()` `rmtree`s whatever it finds at its data directory (default `bench/vector/.pgdata`) before creating a fresh
one, so two runs sharing the path delete each other's live data, not just clash on a port (ports are always freshly
OS-assigned; see `harness/util.free_port`). Pass an absolute path (the socket directory derived from it is written into `postgresql.conf`). `./cleanup.sh
[data-dir ...]` stops leftover scratch instances.

### Configs

| config | data | runs | wall time |
|---|---|---|---|
| `smoke-native` | 50k x 768d synthetic | pgvector hnsw + `pg_pinecone_native` (unfiltered, filtered, one residual cell); the pg_pinecone 0.8 port smoke | 411 s (verified 2026-10-12) |
| `ci_tiny_exact` | tiny synthetic | exact-seqscan regression gate before a long run | 12 s (verified 2026-10-12) |
| `smoke` | 50k x 768d synthetic + SIFT1M | pgvector hnsw/ivfflat only; the 2026-09 smoke | ~21 min (2026-09 record) |
| `smoke_pg_pinecone_native` | 50k x 768d synthetic | trimmed plumbing proof with forced mode L / mode H | not run in this pass |
| `residual-smoke` | 50k x 768d synthetic | iterative-scan grid, one residual cell; rehearsal of `residual-c1/c2` | ~7 min (its header) |
| `residual-c1`, `residual-c2`, `residual-ext`, `residual-ext2`, `residual-ext-model` | wiki_dpr 1M x 768d | residual-filter benchmark (`REPORT-residual-filter.md`), run from `configs/README-residual.md`; `residual-ext-model` is the mode-L system of `residual-ext` alone | c1, c2: 55 min each on the laptop, 83 and 88 min on the box (ext: 23 min, ext-model: 6 min) |
| `local_1m` | wiki_dpr 1M x 768d | unfiltered hnsw, ivfflat, `pg_pinecone_native` and forced exact at k = 10 and 100, 1 / 4 / 16 clients (`k_values`, `sweep_values_by_k`, `max_queries`) | box run 2026-10-14 |
| `local_1m_0.10`, `residual-c1-0.10`, `residual-c2-0.10`, `residual-ext-0.10`, `remote_filtered_keep` | wiki_dpr 1M x 768d | the 0.10.0 versions of the configs above: the native grid is `scan_fraction` x `rescore_multiplier` with the calibrated default as its first point, `iterative_scan` `auto` (the new default) beside `off` and `relaxed_order`, `remote_filtered_keep` is `remote_filtered` without deleting the 1M index | 0.10.0 release run on the box |
| `smoke-halfvec-native` | 50k x 768d synthetic | rehearsal of `halfvec_1m`: every arm kind in its order (FP32 hnsw, 5x rerank, the four `target_recall` native builds and the 0.95 grid, exact FP32, FP16 heap, exact FP16), k = 10 and 100, 1 / 4 clients; needs pg_pinecone from main | 80 s (verified 2026-10-16) |
| `halfvec_1m` | wiki_dpr 1M x 768d | FP32 hnsw; FP16 expression hnsw with and without 2x / 5x / 10x FP32 rerank; FP16 heap hnsw; m16 and m32 builds; `pg_pinecone_native` from main at `target_recall` 0.90 / 0.95 / 0.98 / 0.99 (calibrated defaults, plus a probe x pool grid on 0.95); exact FP32 and FP16 scans; k = 10 and 100, 1 / 4 / 16 clients | 95 min on the box (2026-10-16, `REPORT-halfvec.md`) |
| `remote-smoke`, `remote-smoke-lanes` | 5k x 768d synthetic | rehearsals of the Pinecone lanes against a real index: AM with a filter key and residuals, direct REST, a partitioned AM index, the trigger lane | 3 to 5 min, about 15 to 30k write units each |
| `remote_1m`, `remote_filtered`, `remote_partition`, `remote_lanes_am_100k`, `remote_lanes_topk_100k` | wiki_dpr 1M / 500k / 100k x 768d | the Pinecone-backed leg (staged: `--stage build`, then `--stage measure`); real money, see below | hours (backfill) plus about 1 h of queries |
| `baselines_exact`, `baselines_exact_real`, `baselines_exact_real_r50` | synthetic / wiki_dpr | forced-seqscan exact baselines (recall asserted == 1.0) | hours |
| `sweep_pg_pinecone_50k`, `spotcheck_pg_pinecone_50k` | 50k synthetic | `pg_pinecone_native` knob sweep and spot check | not run in this pass |
| `full`, `full_1m`, `full_1m_mid`, `full_1m_mid_pg_pinecone`, `sift_hard_pg_pinecone`, `wikidpr_1m_pg_pinecone`, `followup_c1_mid`, `followup_c1_wikidpr`, `followup_c5_sift_k512` | 1M synthetic / SIFT1M / wiki_dpr | the 2026-09 records and follow-ups | hours; budget before launching |

Only the configs named in the verification above were run. The rest load and generate valid DDL but are
unverified on this tree, and the 1M configs also need the datasets above.

## Pinecone-backed leg (box)

`pg_pinecone_remote` is `pinecone.create_index(tbl, col, engine => 'remote', metric, index_name[, filter_column][, partition_column])`
(what `CREATE INDEX ... USING pinecone ... WITH (mode = 'remote')` runs): the build creates a serverless index on Pinecone
(`pinecone.am_cloud` / `am_region`, default aws us-east-1, dimension and metric from the column and opclass), queues a backfill
that the background worker drains (`pinecone.worker_naptime_ms`, default 2000), and every query is the plain
`ORDER BY embedding <-> $1 LIMIT k` served by the access method. `pg_pinecone_remote_topk` is `pinecone.promote` plus
`pinecone.topk(tbl, q, k)`. What the harness adds on top of the sweep machinery (`harness/remote.py`):

- **Build and gate.** After `create_index` it polls `pinecone.status()` until the queue is idle in remote mode, counts the vectors Pinecone
  holds (`describe_index_stats`), calibrates the reported read units per query on the finished namespace, and runs the
  searchability gate: 100 sampled rows queried with their own vector through the lane under test, at least 99% must come back as
  their own nearest neighbour. Time-to-searchable is `create_index` to that gate. Backfill rows per second, the worker's CPU seconds
  and WAL are recorded; the number of upsert requests is **estimated** from the extension's chunking rule (at most 1,000
  rows or 1.9 MB of record size per request, counted as Pinecone counts it: 4 bytes a value, the id, the metadata; one request per
  tenant per 5,000-row window for a partitioned index), because the extension logs none.
- **Sweep points.** Same EXPLAIN gate as every system (a plan that does not use the pg_pinecone index invalidates the point), same
  tie-aware recall against exact ground truth (Pinecone's own recall, since it answers the query), plus `pinecone.last_scan_stats()`
  per query (`batches` = Pinecone `/query` calls per SQL query). Pinecone documents 100 queries per second per namespace and answers
  the excess with HTTP 429, which `pg_pinecone` raises as an error and never retries: a failed attempt is retried with backoff,
  its latency is not counted, and each point reports the rejected share. The box run saw about 245 per second on one namespace before
  queueing (1M) or 429s (100k). `remote.rest_seconds` pauses between points so the allowance refills.
- **Knobs.** `pinecone.am_min_topk` (the first `topK`, also for `LIMIT 10`: default 50 in the 0.9.0 run, 100 from 0.10; a `LIMIT` above it
  re-queries at `topK` x 4), `pinecone.iterative_scan` and `pinecone.max_scan_tuples` (rows requested from Pinecone, bounded by
  `pinecone.am_max_topk`, default 10,000) for a residual `WHERE`.
- **Direct lane.** The same `/query` over keep-alive HTTPS from Python (`remote.direct`), with Pinecone's own
  `x-pinecone-request-latency-ms`, the reported read units and the RTT (TCP, TLS, keep-alive GET).
- **Partition.** `partition:` in the config adds a `tenant_id` column (100 tenants, each a group of k-means clusters) and a
  per-tenant workload; per-tenant p50 is the median over tenants.
- **Spend.** Every request is tallied in `$STATE/cost-tally.jsonl` (write units from the logical record size, 1 WU per KB with 5 per
  request, which is what Pinecone reports in `x-pinecone-request-logical-size`; read units as the API reports them, rounded up per query,
  429s counted as reads) and the run stops at 6M WU, 1M RU or $40 at the high end of Standard list ($4.50 per M WU, $18 per M RU).
- **Cleanup.** `box/pc_indexes.py delete-all --yes` deletes every index whose name contains `pg-pinecone-bench-vec` (the access method
  names its index `pv-<schema>-<table>`, so the tables are called `pg_pinecone_bench_vec_*`) and lists what is left.

Stages: `run.py --config configs/remote_1m.yaml --data-dir D --stage build` loads the table, builds the index, waits for the backfill
and the gate, and stops; `--stage measure` (or `--reuse`) starts the cluster in `D` again, skips the load and the build, re-runs a short
gate and measures. `remote_filtered.yaml` runs on the same `D`. Rehearse any change first with `configs/remote-smoke.yaml` and
`remote-smoke-lanes.yaml` (about $0.10 each). The box recipe, phases and teardown are in `box/README.md`.

## Not reproducible from this tree

- The 2026-09 production-build runs on a GCP VM and in Docker, including their Pinecone-backed leg, which never completed (the free tier's
  write quota ran out): their scripts are not in this tree.
- Absolute latencies: they depend on the machine and on cassert vs release Postgres. The reports say which was used.
- The over-RAM workload's Docker variant (`overram/RUNBOOK.md`) was only smoke-checked locally; the capped runs of `REPORT-overram.md` use
  `box/overram-pg.sh` on a Linux host.
- The residual-filter write-up's table scripts and probe logs are not imported.

## Methodology guarantees

These are the properties the harness enforces mechanically -- each was a
concrete way an earlier version of this harness could lie, found in review and fixed:

**One index at a time (no cross-system contamination).** Before each
system's build+sweep, every other secondary index on the table is dropped
(`harness/workloads.py:drop_all_secondary_indexes`); the results JSON records
what was dropped. Churn likewise drops everything and rebuilds only the
system-under-test's index (verified via EXPLAIN before the first cycle;
hard-fails if the check doesn't hold).

**Plan-shape integrity.** Every sweep point captures `EXPLAIN (FORMAT JSON)`
at that point's exact session settings and tags the point with its plan
shape (`index_scan:<index_name>` / `seqscan_sort`) and the index actually
used. If an index scan uses an *unexpected* index, the point is recorded as
`valid: false` with an `abort_reason` and excluded from every derived
number. Seqscan points are legitimate data ("the planner gives up here") and
are reported as their own labeled series -- never blended into an index's
recall/latency curve. This matters: in the clean smoke run, the planner
flips hnsw filtered queries to a seqscan at `ef_search >= 80`, and those
points score recall 1.000 at ~16ms -- an earlier harness version reported
exactly those as the index "reaching recall 1.0". When *every* default-planner
point for a filter goes seqscan, the harness re-runs the same grid with
`enable_seqscan = off` as a clearly-labeled forced-plan companion series, so
"the planner gives up" is told apart from "the index is incapable" instead of
only the former being on record.

**Iterative-scan budget is swept, not assumed.** Filtered `relaxed_order`
scans are bounded by `hnsw.max_scan_tuples` (default 20000) /
`ivfflat.max_probes`. That default is comparable to the row count and can cap
filtered recall independent of `ef_search` -- so it is swept as its own
stamped axis (default vs. `auto` = `max(100000, 10% of rows)`), the effective
GUCs are `SHOW`-captured onto every point, and points where recall is flat
while the knob rises *and* a higher budget still moves recall are labeled
`budget_capped`. This is exactly the earlier "0.855 plateau": that number was
the 20000-tuple budget binding, not pgvector's ceiling (see the results
summary -- the fair cross-tenant ceiling is ~0.93).

**No plan drift within a point.** All connections use psycopg
`prepare_threshold=None` (no silent auto-prepare generic-plan switch), and
session GUCs are set once per connection per sweep point, not per query.

**Calibrated dataset difficulty.** The synthetic GMM (see
`harness/datasets.py` and the `dataset:` comment in `configs/smoke.yaml`) is
calibrated so the swept knob actually matters: mixed per-cluster scales, a
2% uniform background fraction, and `cluster_sigma` chosen empirically
against real pgvector recall curves (naive tight clusters are trivially easy
in 768d -- the curse of dimensionality makes "nearest-center impurity" a
misleading target; local density is the real lever). Every run records
geometry diagnostics (inter/intra distance ratio) and runs an automatic
hardness check: if the cheapest sweep setting achieves unfiltered
recall > 0.97, the run FAILS LOUD ("dataset too easy — recalibrate") instead
of producing flattering numbers. The clean smoke run's cheapest-point recall:
hnsw 0.670, ivfflat 0.545.

**Honest concurrency.** Workers are OS processes (`ProcessPoolExecutor`, one
psycopg connection per process, persistent across a sweep) -- not
GIL-sharing threads. All workers barrier after their untimed warmup. `qps_wall`
comes from a *separate* duration-based throughput pass: after a second
barrier, each worker streams its query set in a loop for at least
`min_throughput_seconds` (5s in full, 1.5s in smoke) and counts completed
queries -- a fixed 500-query set split 8 ways otherwise finishes in tens of
milliseconds, too short a window to measure throughput. Recall and the
latency percentiles still come from the fixed measured set (one pass), kept
separate from the throughput count.

**Fixed measured query set.** The measured pool (`n_queries`, 500 in both
configs) is identical at every concurrency; warmup queries come from a
disjoint held-out pool (`n_conns * warmup_queries` extra vectors generated
alongside the dataset) and are never counted.

**Bounded interpolation.** latency@recall interpolates only between two
points of the *same* plan shape whose recall gap is <= 0.15; otherwise the
report shows the bracketing pair and `n/a` rather than a made-up number.
Targets outside the achieved recall range are `n/a` (never extrapolated).

**Provenance stamps, machine-checked.** Every results JSON records the git
SHA, a `git_dirty` flag (`git status --porcelain` non-empty), a content hash
over every tracked file under `bench/vector/`, a stable config hash, `SELECT
extversion` for pgvector, per-point plan-shape tags + effective GUCs, and
per-filter mask hashes; filtered ground-truth caches are keyed on the hash of
the actual mask bytes, not a human label. `save_results` **refuses** to write
a run of record from a dirty tree (dev iteration needs an explicit
`--allow-dirty`, which tags the filename and banners the report DIRTY); the
run of record must come from a clean commit. NaN/Inf are serialized as `null`
(strict-JSON safe).

## What's implemented

**Datasets** (`harness/datasets.py`, `harness/sift.py`):
- Synthetic clustered Gaussian-mixture point cloud (fixed seed; `n_centers`,
  `cluster_sigma`, `cluster_scale_mix`, `background_frac` knobs), calibrated
  as described above. NOT uniform random. The GMM centers and per-cluster
  sigmas are persisted in the dataset cache -- churn and in-region query
  sampling draw from the same generative model. Held-out queries (measured +
  warmup pools) are drawn from the same distribution.
- SIFT1M (128d, classic ann-benchmarks corpus, ~168MB download from
  `ftp.irisa.fr`, `scripts/download-datasets.sh sift`) as a real-data sanity check. Downloads, parses,
  and caches to `datasets_cache/sift1m.npz`; **skips gracefully** (logs why,
  returns `None`, run continues) if the download or parse fails for any
  reason -- see `sift1m.available` in the results JSON.
- Real-embedding corpora (`kind: real`): a converted npz
  artifact in `datasets_cache/` (fp32 base vectors + held-out
  queries/warmup/query-bank + embedded acquisition provenance) plus a cached
  numpy k-means clustering that plays the GMM cluster-id role -- filter
  columns derive from k-means membership under the same
  correlated/uncorrelated rules, and in-region ("tenant") queries are REAL
  held-out bank vectors from the qualifying clusters (per-filter sampling
  info persisted, incl. the with-replacement fallback at thin selectivities
  where the finite bank can't cover the pool). First corpus:
  `wikidpr_768d_1m.npz` (facebook/wiki_dpr DPR embeddings, 768d x 1M; built by
  `scripts/download-datasets.sh wikidpr`) -- see
  `configs/baselines_exact_real.yaml`'s header for acquisition, holdout, and
  geometry notes. Churn hard-errors on real corpora (it needs the generative
  model).

**Ground truth** (`harness/groundtruth.py`): exact top-100 via brute-force
numpy (batched to bound peak memory), cached to disk keyed by a hash of the
dataset + metric + (for filtered ground truth) the qualifying-mask bytes.
Churn cycles always recompute against current live rows (the whole point is
the ground truth moves as the data does).

**Filtered search workloads** (`harness/datasets.py`, `harness/loader.py`):
one boolean column per (selectivity, mode) pair in `filters.run`.
`mode=correlated` picks whole clusters until their combined population hits
the target selectivity (qualifying rows are spatially contiguous -- the hard
case). `mode=uncorrelated` assigns independently at random. Selectivity is a
*target*; realized selectivity is always reported. Correlated filters run
**two query regimes** as separate sweeps: `out_region` (global queries, the
cross-tenant case -- query lands far from the qualifying region) and
`in_region` (queries sampled from the qualifying clusters, the tenant-style
case). The clean smoke run shows why both matter: at 1% correlated
selectivity with `relaxed_order` at the raised budget, hnsw reaches ~0.928
recall @ ~16ms for out-of-region queries but ~0.989 @ ~0.8ms for in-region
ones -- same index, same filter, ~20x latency and a real recall gap.

**Churn workload** (`harness/churn.py`): per cycle, re-embeds a % of rows
(fresh draw from the row's own cluster's generative distribution -- "same
topic, freshly encoded", not a random walk) and delete+reinserts another %
with fresh ids drawn from the generative model, with all filter columns
populated using the same assignment rules as the base load. Ground truth is
recomputed against current live rows every cycle; recall measured before and
after `VACUUM`. Churn runs with only the system-under-test's index present,
and the EXPLAIN plan shape is re-verified **after every cycle's ANALYZE**
(recorded per cycle; a non-index plan fails the run loud).

**Systems** (`harness/systems.py`): pgvector hnsw (`m`/`ef_construction`
pinned to pgvector docs' recommended defaults, `ef_search` swept) and
ivfflat (`lists` via the docs' heuristic, `probes` swept).
`hnsw.iterative_scan` / `ivfflat.iterative_scan` run as **separate
configurations** on filtered sweeps (`off` vs `relaxed_order`), each with the
scan-budget GUC swept as its own axis (default vs. table-scaled `auto`); not
swept on unfiltered queries where they don't matter. Plus `type: exact`:
the forced-seqscan baseline -- builds NO index, runs
every point with `enable_indexscan/enable_bitmapscan = off`, EXPLAIN-verifies
`plan_shape=seqscan_forced` per point, and **asserts** recall == 1.0 against
the cached brute-force ground truth instead of merely reporting it (an FP16-heap
exact scan against the FP16-rounded truth: "pgvector halfvec (FP16)" below)
(`configs/baselines_exact.yaml`; these measured exact-scan p50s are the
denominators for the filtered-search claims).

**`pg_pinecone_native` (`harness/systems.py`'s `System(kind="pg_pinecone_native")`):**
`CREATE INDEX ... USING pinecone (embedding <opclass>[, filter_column]) WITH
(mode = 'local', ...)` -- the native local engine.

- *Install first, separately.* This harness never builds/installs pg_pinecone --
  it assumes the extension is already installed into the Postgres install
  `PG_PINECONE_BENCH_PG` points at (the binaries `harness/pgctl.py`'s `PgInstance`
  launches its own throwaway `.pgdata` from); see "Install pg_pinecone" above.
  Re-install after every engine-side change before trusting a `pg_pinecone_native`
  result. `PgInstance` always `initdb`s a brand-new data directory per run, so a
  bench run itself never needs a stop/start restart.
- *Filter columns are equality-typed, not boolean.* pg_pinecone's AM only accepts
  an int2/int4/int8/text/varchar filter column (`am.rs`'s `ambuild`: bare
  `boolean` has no pg_pinecone opclass and is rejected at DDL time), so
  `harness/loader.py` writes an int2 companion column per filter spec
  (`loader.filter_eq_column_name`, `1`/`0` mirroring the SAME mask the
  boolean column pgvector systems query) -- a `pg_pinecone_native` system's
  `build_params.filter_column` names that companion column, and
  `runner.py`'s per-filter loop queries it (`WHERE {eq_col} = 1`) instead of
  the boolean one. `am.rs` supports AT MOST ONE filter column per index,
  fixed at `CREATE INDEX` time, so a `pg_pinecone_native` build only ever gets swept against ITS
  OWN filter column -- `runner.py` records every OTHER filter spec as
  `not_applicable` for that system rather than silently skip it or (worse)
  run it anyway as an unfiltered-scan-plus-executor-postfilter and
  misreport it as a filtered result. Sweeping a different filter cell means
  a different `pg_pinecone_native` system entry (a different build) in the config,
  same as sweeping `m`/`ef_construction` would for hnsw.
- *Protocol constraints, encoded as assertions:*
  - Query `k <= 100` (`workloads.run_sweep_point`): `local/scan.rs`'s
    `ASSUMED_TOP_K=100` fixes the exact-rescore pool size (no real `LIMIT`
    propagation into the AM), so a larger `k` doesn't error inside the
    engine -- it silently under-returns. The harness asserts this instead of
    reporting an artificially low "recall" that's actually a
    misconfiguration.
  - Per-measured-query `pinecone.last_scan_mode()` (`workloads._pool_worker_run`):
    every recall-pass query for a `pg_pinecone_native` system asserts the reported
    mode (`'flat'`/`'h'`/`'l'`) against `System.pg_pinecone_native_expected_mode`'s
    prediction (small-N bootstrap vs. filtered-vs-not vs. the
    estimate-vs-`mode_l_threshold` rule, using the harness's own already-
    measured realized selectivity) -- not just recorded, asserted, so a
    silently-wrong scan path shows up as a hard failure at the exact sweep
    point it happened, not a puzzling recall number three tables later.
  - Sweep knobs are session GUCs (`pinecone.rescore_multiplier <= 1000`,
    `pinecone.scan_fraction`/`pinecone.mode_l_threshold` in `[0, 1]`,
    `_assert_and_set_pg_pinecone_guc` bounds-checks every one before `SET`) --
    `System.sweep_param` must name one of the three; `k_bits`/`seed`/etc.
    are build-time reloptions and can never be `sweep_param` (that would
    silently rebuild mid-sweep). A `k_bits` sweep is multiple `pg_pinecone_native`
    system entries (multiple builds), each sweeping a GUC -- same pattern
    as sweeping hnsw's `m` via multiple systems. `target_recall` (0.10+, a real
    in `[0.5, 0.999]`) passes through `build_params` the same way; every native
    build records `pinecone.index_calibration()` (`build.calibration`, `None` on
    a build without it), and every point `SHOW`s `pinecone.scan_prune`.
- *Churn is REINDEX-shaped, not pgvector's in-place live-index churn* --
  an HONEST comparability difference, documented at the call site
  (`harness/churn.py:run_churn_pg_pinecone_native`), not a like-for-like
  substitute. (Stale since 0.8.0: the engine now has an incremental `aminsert`,
  `local/insert.rs`; the protocol below was written for an earlier engine
  and was not re-validated.)
  Local mode's `aminsert` is a hard-stub `ERROR` (no
  incremental insert yet), so ANY `INSERT`/vector-column `UPDATE` raises
  while this system's index exists on the table -- churn can only reach the
  index via a full rebuild. Per cycle: `VACUUM` (measured; reclaims already-
  dead TIDs) -> `DROP INDEX` (measured) -> mutate the table exactly like
  pgvector's churn (safe now, no index to reject the writes) -> `CREATE
  INDEX` fresh (measured -- **the number to compare**: REINDEX time is counted
  fully) -> recall against the fresh
  build. There's no "pre-vacuum recall" here (no live index existed to
  drift); recall reflects a normal fresh build every cycle, by construction.
- See `configs/smoke_pg_pinecone_native.yaml` for a runnable plumbing-proof config
  (unfiltered + forced-mode-L + forced-mode-H `pg_pinecone_native` systems
  alongside a trimmed `pgvector_hnsw` sweep, 50k rows) -- run it with
  `--data-dir` pointed at a scratch directory, never the default
  `bench/vector/.pgdata`, whenever another run might already own that path.

**Incremental checkpoints** (`harness/runner.py:_write_checkpoint`): after
every completed workload chunk the runner atomically writes
`results/<name>_<started>.partial.json` (marked `partial: true`, no
provenance stamps) so a multi-hour run can't lose everything to a crash
before the final write; `save_results` removes the partials on success.

**Metrics** (`harness/metrics.py`, `harness/workloads.py`): per sweep point,
recall@k, p50/p95/p99 + mean latency (ms) from the recall pass; `qps_wall`
from the separate duration-based throughput pass; plan-shape tag, effective
GUCs, `budget_capped` label, validity. latency@recall per (iterative_scan,
scan_budget, plan_shape, n_conns) group under the bounded-interpolation rule
above.

**Build stats** (`harness/workloads.py:build_index`): wall time,
`pg_relation_size`, peak RSS of the leader backend sampled every 200ms
(directional -- doesn't count parallel workers), plus any server NOTICEs
raised during the build (`build_notices`) and a `graph_spilled_to_disk` flag
derived from pgvector's "graph no longer fits into maintenance_work_mem"
notice.

## Postgres tuning

`pg/tuned.conf` is appended to `postgresql.conf` after `initdb`, pinned for
the Mac the harness was developed on (Apple Silicon, 14 cores, 36GB RAM): 8GB `shared_buffers`,
2GB `maintenance_work_mem`, `fsync`/`synchronous_commit` off (data is 100%
reproducible -- never worth durability), 7 parallel maintenance workers,
autovacuum off (the harness controls VACUUM explicitly).
`effective_io_concurrency` left at 0: nonzero fails startup on macOS. A config
can override any of these per run via a `postgres:` block (layered last, so it
wins) -- `full.yaml` raises `maintenance_work_mem` to 8GB so the 1M-row HNSW
build doesn't spill (the spill is captured and labeled if it happens anyway).

## pgvector halfvec (FP16)

pgvector's FP16 options against its FP32 hnsw and the native engine, on one table. In a system entry (`harness/systems.py`),
`index_type: halfvec` builds hnsw with `halfvec_*_ops`: over the FP32 heap (`storage_type: vector`, the default) on the expression
`embedding::halfvec(dim)`; with `storage_type: halfvec` the loader rounds the rows to IEEE FP16 on COPY, and the runner reloads the same
rows whenever the next system needs the other heap (so a halfvec run needs `stage: all`). `rerank_multiplier: 2/5/10` fetches that many
FP16 candidates per requested row and re-sorts them by the FP32 vector inside the measured query (FP32 heap only); timing, the EXPLAIN gate
and the cost probe run the same SQL. An `ef_search` below the candidate count is refused at startup, before the load or any build.

The top-level `fp32_column_storage` (`plain`, `main`, `external`, `extended`; unset = the type's default) is set on the FP32 heap's
`embedding` column before the COPY; the halfvec configs use `plain`. pgvector's default (`external`) TOASTs a vector over about 2 KB, so at
768d every FP32 vector read by the rerank, native rescore or exact FP32 arms would cost a TOAST lookup the FP16 heap (1,544 bytes, inline)
never pays. `plain` keeps 2 rows per 8 KB page, the same bytes on disk, and fits FP32 vectors up to about 1,990 dimensions; the build line
stamps the column storage and the TOAST size.

FP32 recall is the decision metric. A run with a halfvec system also records, on every point of every system, `recall_halfvec_truth` (the
report's `FP16 truth` column: recall against exact search over FP16-rounded corpus and queries, what rounding alone costs) and, per build,
heap, index and total relation size; the forced exact scan of an FP16 heap is asserted at 1.0 against that truth, not the FP32 one. A run
without a halfvec system records and reports exactly what it did before. `halfvec_1m.yaml` sweeps `ef_search` through at least each rerank
candidate count at k = 10 and 100, on m16 / ef_construction 64 and m32 / ef_construction 200 builds: report the recall-latency frontier and
the plan shape, not one hand-picked setting.

Its native arms are pg_pinecone from main (0.10), one build per `target_recall` (0.90, 0.95, 0.98, 0.99), each queried at the defaults its
build calibrated (sweep value 0: no session override) and at `rescore_multiplier` 10 for k = 100; the 0.95 build also sweeps
`pinecone.scan_fraction` x `pinecone.rescore_multiplier` to reach recall above its calibration (`docs/reference.md`, "Native engine:
calibrated defaults"). Every native build stamps `pinecone.index_calibration()` into its build result and report line.
Configs: `smoke-halfvec-native.yaml` (every arm kind of `halfvec_1m` at 50k x 768d, native included: run it on the same
pg_pinecone build before the record run), `halfvec_1m.yaml` (the record run, on the box with pg_pinecone built from main: `box/README.md`, "Halfvec session"). The halfvec
arms benchmark pgvector; they do not imply pg_pinecone supports a halfvec column.

## Over-RAM workload

`configs/overram_1m.yaml` runs FP32 hnsw, the FP16 expression index and native on an external Postgres 17 that `box/overram-pg.sh` starts as a
transient systemd unit, so `MemoryMax` (cgroup v2, page cache included) can drop after each uncapped build and lift after the sweep; the result
is `REPORT-overram.md`. `overram/RUNBOOK.md` is the earlier Docker-based variant: in 2026-09 a local smoke check (macOS Docker VM,
`pgvector/pgvector:pg17`, `--memory=512m`) confirmed that container config works.

## Smoke results (historical: 2026-09-09, clean run after the methodology fixes)

Wall time 1261s (~21m) for `configs/smoke.yaml` (50k rows, 768d). The run stamps
`git_dirty: False` against commit `8cc2a86`; 125/125 sweep points valid, the
hardness gate passed (cheapest-point recall hnsw 0.644 / ivfflat 0.606).

Headlines (all from plan-verified index-scan points unless labeled):

- **The raised `hnsw.max_scan_tuples` budget reveals the real filtered
  ceiling.** On the 1%-correlated *cross-tenant* filter with `relaxed_order`,
  the default 20000-tuple budget plateaus recall at **0.861** (ef=40, ~16.7ms
  p50, every point auto-labeled `budget_capped`); raising the budget to
  `auto` (=100000, 10% of rows) lifts it to **0.928** at equal latency. The
  earlier "0.855 plateau" was the budget binding, not pgvector's capability.
- **Tenant vs cross-tenant split** (same index, same 1% filter,
  `relaxed_order`, raised budget, ef=40, 1 conn): in-region **0.989 recall @
  0.8ms** vs out-of-region **0.928 @ 15.9ms** -- ~20x latency, and the tenant
  case nearly saturates recall while the cross-tenant case tops out at ~0.93
  even with the budget lifted. `iterative_scan=off` collapses cross-tenant to
  0.011-0.014 (planner keeps the index at ef<=40), then flips to seqscan at
  ef>=80 (recall 1.000, ~16ms -- its own labeled series, not index perf).
- **Unfiltered, latency@recall (p50, 1 conn):** hnsw 1.24ms @ 0.90, 2.47ms @
  0.95 (recall 0.99 not reached within `ef_search<=160`: max 0.962). ivfflat
  10.1ms @ 0.90, 15.2ms @ 0.99. hnsw build 26.9s / 204.8MB; ivfflat 5.1s /
  205.2MB.
- **Concurrency (8 process workers, duration-based throughput pass):** hnsw
  unfiltered `qps_wall` 1449 (1 conn) -> 10974 (8 conns) at `ef_search=10`;
  480 -> 3013 at `ef_search=160`. This is a >=1.5s measured window per point,
  not the earlier ~50ms noise window.
- **ivfflat under the 1% correlated filter:** the planner *does* pick the
  ivfflat index at probes=1,2 (recall 0.149-0.293) before flipping to seqscan
  at probes>=4 -- "low-probe index then planner fallback," not the earlier "all
  seqscan." (Because real index-scan points exist, the forced-plan companion
  series is correctly not triggered here.)
- **Churn (3 cycles, 5% reembed + 5% delete/reinsert, hnsw, ef_search=40):**
  plan shape re-verified after every cycle's ANALYZE (all `index_scan`).
  Recall 0.836/0.824/0.822 pre-VACUUM vs 0.829/0.825/0.808 post-VACUUM -- no
  clear post-VACUUM recovery signal at this scale. VACUUM dominates cost:
  156-162s per cycle for 5k dead tuples on a 50k-row HNSW table. **To
  re-verify at full scale** -- an observation from a 50k-row run, not a
  headline.
- **SIFT1M sanity (1M x 128d real corpus):** hnsw build 49.4s, 820MB;
  latency@recall 0.64ms @ 0.90, 1.09ms @ 0.95, 1.95ms @ 0.99. Confirms the
  harness produces sane recall/latency curves on real data.

### Deltas vs the earlier clean run

- The cross-tenant filtered ceiling moved from a **budget-capped 0.86** to a
  **fair 0.93** once `max_scan_tuples` is lifted -- the number the investment
  case needs. The earlier run reported the capped value as if it were the limit.
- ivfflat filtered is "low-probe index then seqscan," not "all seqscan" as
  the earlier run stated -- the per-point plan-shape gate shows the planner does pick
  it at probes=1,2.
- 8-conn QPS is now a duration-based measurement, not a ~50ms window.
- Hardness 0.644/0.606 (earlier run 0.670/0.545): same regime, both well under
  the 0.97 gate; churn recall trend and VACUUM cost are consistent.

## Layout

```
bench/vector/
  run.py                    single entrypoint
  requirements.txt
  configs/                  run configs (*.yaml); README-residual.md = runbook of the residual-filter benchmark
  pg/tuned.conf             appended to postgresql.conf after initdb (tuned.linux-aws.conf / tuned.linux-gcp.conf: the AWS and GCP boxes)
  harness/                  the package (pgctl, datasets, sift, groundtruth, loader, systems, workloads, remote, churn,
                            metrics, runner, report, config, util)
  scripts/                  download-datasets.sh, convert_wikidpr.py
  box/                      the AWS boxes: provision.sh, upload.sh, teardown.sh, conn.sh, step.sh, user-data.sh, install-pg_pinecone.sh,
                            install-pg18.sh, setup-box.sh, phase.sh, settings.py, pc_indexes.py, tables_halfvec.py, README.md,
                            BOX-2026-10-16-halfvec.md
  overram/                  docker-compose.yml + tuned conf + RUNBOOK.md (design only)
  kernels/                  standalone microbench crate, no dependencies: `cargo run --release` times the Hamming and 1-bit-code
                            kernels (x86_64); `cargo run --release --bin distance` the engine's exact f32 distance kernels (any architecture)
  cleanup.sh                stops leftover scratch Postgres instances
  REPORT-*.md               write-ups
  results/                  timestamped JSON + markdown (gitignored)
bench/.data/vector/         datasets_cache/ (datasets + ground truth), pgdata-*/ scratch (gitignored)
```
