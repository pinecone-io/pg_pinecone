# Residual-filter benchmark (iterative scan): `residual-c1.yaml`, `residual-c2.yaml`

RUN 2026-10-07 on a release Postgres 17.10 (analysis: `../REPORT-residual-filter.md`; raw results `results/residual_{c1,c2,ext,ext2}_DIRTY_*`, not committed). Added since the port: `explain_probe: true` (EXPLAIN ANALYZE cost probe per index point), an `exact_seqscan` reference system, `residual-ext.yaml` / `residual-ext2.yaml` (rescore pool, `scan_fraction` 1.0 and the shipping mode L). Question: when a query carries a predicate the index cannot push (a second
column), how many rows does `LIMIT 10` return, at what recall and latency, and what does
`pinecone.iterative_scan` buy over `off` and over pgvector's `hnsw.iterative_scan`?

| config | pushed key (index column) | residual (executor `Filter`) | combined rows (of 1M) |
|---|---|---|---|
| `residual-c1` | `filt_correlated_0010_eq` (1% correlated, 10,170 rows) | `filt_uncorrelated_0003_eq` (0.3%, 3,098) / `_0010_eq` (1%, 10,048) | 33 / 113 |
| `residual-c2` | `filt_correlated_0100_eq` (10% correlated, 100,844 rows) | same two | 326 / 1,004 |

(row counts computed from the cached k-means and the harness's own mask generator.)

wiki_dpr 1M x 768d, L2, k=10, cross-tenant queries only (`filters.query_modes: [out_region]`), same
artifact/seed/k-means as `wikidpr_1m_pg_pinecone.yaml` (dataset key `real_fd937e5862449934`).
`category = X` is the residual column: uncorrelated with the tenant key, so it passes the same share of
every cluster. Swap `mode: uncorrelated` for `correlated` in a residual to make it spatially clustered.

Sweep per filter cell, 1 and 8 clients:

- `pg_pinecone_native`: `pinecone.iterative_scan` {off, relaxed_order, strict_order} x
  `pinecone.max_scan_tuples` {2000, 20000, 100000} (not set when off) x
  `pinecone.rescore_multiplier` {10, 50} x `pinecone.scan_fraction` {0 = auto ramp, 0.5}: 56 points.
- `pgvector_hnsw` (comparison): `hnsw.iterative_scan` {off, relaxed_order, strict_order} x
  `hnsw.ef_search` {40, 200} x `hnsw.max_scan_tuples` {20000 default, 100000} (not set when off): 20 points.

Recorded per point (results JSON `systems.<s>.filtered.<cell>.query_modes.out_region.sweep[]`; the
`.md` report prints the same as a table): `recall`, `rows_returned_mean` / `rows_returned_min` /
`frac_filled` (rows at `LIMIT 10`), `p50_ms` / `p95_ms` / `p99_ms`, `qps_wall`, and for pg_pinecone
`scan_rescored_mean` / `scan_rescored_p95` / `scan_batches_mean` / `scan_capped_frac` (from
`pinecone.last_scan_stats()`, read after the timed query), plus `plan_shape` and the effective GUCs.

## Setup

1. A pg_pinecone build with the iterative-scan GUCs (`pinecone.iterative_scan`, `pinecone.max_scan_tuples`,
   `pinecone.last_scan_stats()`); the harness refuses to build the index on an older build. Install it as in
   `../README.md` ("Install pg_pinecone"). The 2026-10-07 numbers come from a release Postgres 17.10 (`-O2`, no cassert).
   pgrx's own PG 17 is a cassert build, 2.5x slower on the 50k smoke and up to 10x on rescore-heavy points: do not
   compare across builds.
2. The venv, as in `../README.md`.
3. Dataset: `scripts/download-datasets.sh wikidpr` (5.3 GB download, 3.4 GB `wikidpr_768d_1m.npz`). The k-means and
   ground-truth caches under `datasets_cache/` are keyed by dataset content and recomputed on a fresh checkout; the
   2026-10-07 run reused the 2026-09 caches (22 MB) and skipped the 1M brute force.
4. Room: ~12 GB free for `--data-dir` (table 3.1 GB, hnsw 4.1 GB, pg_pinecone 0.2 GB, WAL), ~16 GB free RAM
   (`pg/tuned.conf` asks for 8 GB shared_buffers, the config for 8 GB maintenance_work_mem). Close other
   benchmarks: 8-client points compete for cores.

## Run

```
cd bench/vector
export PG_PINECONE_BENCH_DATA="$(cd .. && pwd)/.data"
env -u PINECONE_API_KEY nohup nice -n 10 .venv/bin/python run.py \
    --config configs/residual-c1.yaml --data-dir "$PG_PINECONE_BENCH_DATA/vector/pgdata-residual-c1" -v \
    > "$PG_PINECONE_BENCH_DATA/vector/residual-c1.log" 2>&1 &
```
Add `--allow-dirty` only if the git tree is dirty (the 2026-10-07 runs were made from a git-excluded directory and used it).
`--data-dir` is mandatory next to any other run. Poll the log (`== system`, `build:`, `filtered ground truth`
lines); partial results land in `results/residual_c1_*.partial.json` after every filter cell (no resume: a crash keeps
the partial file, a rerun starts over). Run c1, then c2,
one at a time. `./cleanup.sh "$PG_PINECONE_BENCH_DATA/vector/pgdata-residual-c1"` stops a leftover scratch Postgres. Rehearse first
with `configs/residual-smoke.yaml`: the same
code path and the full iterative grid on 50k synthetic rows (one cell: 10% key + 0.5% residual = 24 rows, ~7 min).

## Time (estimate; measured 2026-10-07 on a release build: C1 55 min, C2 55 min, builds 90 s hnsw + 110 s pg_pinecone)

| step | time |
|---|---|
| load + ANALYZE + ground truth (unfiltered 1M x 500 is cached if copied; the residual masks only touch 33-100k rows) | 3-6 min |
| hnsw build (8 GB maintenance_work_mem) | 2.5-5 min |
| pg_pinecone build | 5-15 min (cassert pgrx Postgres: ~650 s; 50k took 8 s) |
| pg_pinecone sweep, 2 cells x 56 points | 60-120 min |
| hnsw sweep, 2 cells x 20 points | 15-70 min (the top end if the planner picks a seqscan for every point of a cell and the runner repeats the grid with `enable_seqscan = off`) |
| total per config | about 2.5 h (1.5-3.5 h); c1 + c2 about 5 h |

Point cost = 500 queries x per-query latency (split across clients) + 5 s throughput window + ~2 s overhead.
Per-query latency is about 28 us x exact-rescored rows (measured 138 ms for 4,900 rows at 50k on this Mac):
`off` 30 ms (rm 10) to 140 ms (rm 50); relaxed/strict 30-300 ms (first batch is never cut, the cap bounds the rest;
C1/0.3% can rescore the whole 10k-row key set, ~280 ms); hnsw relaxed at 100k budget 50-120 ms, a seqscan ~113 ms.
To cut the time: `concurrency: [1]` (about half), drop `max_scan_tuples` 100000 or `rescore_multiplier` 50.
`n_queries` must stay 500 (changing it changes the dataset key and recomputes k-means).

## Read before interpreting

- `mode_l_threshold: 0.005` forces the 1% key into mode H; at the shipping default (0.03) it routes to mode L
  (exact over the key rows), where the iterative GUCs are inert and the answer is complete.
- The rescore pool is cut BEFORE the residual qual: expect low
  recall at defaults even when rows fill; the deepen-before-widen follow-up targets exactly that.
- `strict_order` is "ordered but lossy" (drops rows closer than the last one served); `relaxed_order` returns
  batch-wise order. `max_scan_tuples` counts exact-rescored rows and never cuts the first batch.
- Few qualifying rows: C1's 0.3% cell has 33 rows in 1M, so `LIMIT 10` returns at most 33 and recall is
  noisy; the exact seqscan (recall 1.0; 25-34 ms on production builds, ~113 ms on the Mac cassert build)
  is the baseline an iterative scan has to beat.
- Planner: a point whose plan is a seqscan is kept as its own `seqscan_sort` series (never blended into
  index curves, mode assertion skipped); only if EVERY point of a cell is a seqscan does the runner repeat
  the grid with `enable_seqscan = off` (`forced_index_sweep`), which adds up to 20 / 56 points per cell.
  At 50k rows this happens for both systems (`residual-smoke`); at 1M it should not for pg_pinecone.
- pgrx Postgres is a cassert build: absolute latencies are inflated (stamped as `pg_debug_assertions`).
  - The 2026-10-07 runs were made from a git-excluded scratch directory, so their `git_dirty` and `bench_content_hash`
  stamps carry no information. From `bench/vector/` in the repo they do: commit before a run of record.
