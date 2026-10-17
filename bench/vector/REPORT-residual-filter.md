# Residual-filter benchmark: pg_pinecone iterative scan vs pgvector hnsw (2026-10-07)

> Imported from the session write-up; only file references changed. Raw results (`results/` is gitignored) and the `bench3/` analysis and probe scripts are not in the repo. Rerun: `configs/README-residual.md`.

1M x 768d wiki_dpr, `LIMIT 10`, 500 cross-tenant queries. Four cells: a key the index pushes (1% = C1, 10% = C2) plus a residual column it cannot push (0.3% or 1%, uncorrelated). Full grids: the two residual_c1/c2 results files of 2026-10-07 (not committed).

## Verdict
1. **pgvector mostly bypasses hnsw here.** In 3 of 4 cells the planner picks the exact parallel seqscan (recall 1.000, 16-19 ms, 89-102 ms at 8 clients). In C2/1% it picks hnsw and the default returns **0.0 rows**. Forced hnsw reaches 3.3 / 8.3 / 10 / 10 rows only at `max_scan_tuples` 100000, in 49-180 ms.
2. **pg_pinecone at defaults (`off`) underfills**: 3.0 / 9.6 / 3.4 / 8.5 rows (0-76% of queries filled), recall 0.30-0.75, 8.5-9.7 ms.
3. **`relaxed_order` fills `LIMIT 10` in all four cells** (100% of queries at the default cap) at 8.5-15.5 ms, +0 to +7 ms over `off`, recall 0.76-0.79.
4. **Recall is bound by the rescore pool, not by the scan.** The pool (rm x 100) is cut before the residual. rm 50 gives 0.81 / 0.94 / 0.88 / 0.98 at 17-21 ms; rm 100 gives 0.97 in C2/0.3% at 34 ms.
5. **The exact seqscan is the bar**: 17-22 ms at 1 client, recall 1.000 (main heap is 100 MB, vectors sit in TOAST). At recall 0.95 or better pg_pinecone needs 0.95-1.8x the seqscan's time at 1 client and runs 2-5x faster at 8 clients (22-43 ms against 85-103 ms).
6. **M1 (cap binds before widening) is real at 1M, but these cells do not hit it.** At defaults the cap never binds here (capped 0.00 in all four). With nothing pushed and 33 matching rows, the default cap stops 100% of queries (0.64 rows, never widens). Reaching widening costs 313k rescored rows (1.2 s).
7. **`strict_order` loses everywhere**: fewer rows, lower recall, up to 100k rescored rows (352 ms at 1 client, 742 ms at 8).

## Setup
- Mac, Apple Silicon, 14 cores, 36 GB. Not idle: load average 5-6 between runs from other apps (~1.5 cores); the runs push it to 15-50. `caffeinate -i`, `nice -n 10`. Load samples every 5 min (not committed).
- Postgres 17.10 from source, `-O2`, no cassert (`debug_assertions = off`, stamped per run). pgvector 0.8.6. pg_pinecone 0.8.0 at main 44bddad (iterative scan + deepen-before-widen).
- Postgres: shared_buffers 8 GB, work_mem 256 MB, maintenance_work_mem 8 GB, fsync off, autovacuum off, up to 14 parallel workers.
- Key sets: 10,170 rows (C1), 100,844 rows (C2). Matching rows: 33 / 113 (C1), 326 / 1,004 (C2). `mode_l_threshold 0.005` keeps the 1% key in mode H (the shipping 0.03 routes it to exact mode L).
- Grid per cell, 1 and 8 clients: pg_pinecone `iterative_scan` {off, relaxed, strict} x `max_scan_tuples` {2000, 20000, 100000} x `rescore_multiplier` {10, 50} x `scan_fraction` {default, 0.5}; hnsw `iterative_scan` {off, relaxed, strict} x `ef_search` {40, 200} x `hnsw.max_scan_tuples` {20000, 100000}.
- Builds: hnsw (m 16, ef_c 64) 90 s, 4.1 GB; pg_pinecone 107-111 s, 157 MB. Run time: C1 55 min, C2 55 min.
- Planner: the seqscan costs 14.5k. The hnsw index loses to it in 3 cells (its index rows are forced with `enable_seqscan = off`). The pg_pinecone index wins in all four (Limit cost 12.5k in C1, 6.5k in C2/0.3%).
- Harness additions (now in `harness/`): an `exact_seqscan` reference, a per-point EXPLAIN (ANALYZE, BUFFERS) probe on 8 queries at 1 client (outside the timers), `last_scan_stats()` with `deepened` and `lists_probed`.
- `scan_fraction` default and 0.5 are identical (the filtered ramp is already 0.5 at 1000 lists): same rows, recall and rescored in all 112 pairs. Tables show the default only.

## Headline (fills the LIMIT first, then recall, then latency)

| cell / row | rows (fill) | recall | p50/p95 @1 ms | p50/p95 @8 ms | per query |
|---|---|---|---|---|---|
| **C1 1% key + 0.3% residual (33 rows)** | | | | | |
| exact seqscan | 10.0 (100%) | 1.000 | 16.9/18.6 | 85.3/180.4 | - |
| pgvector default (seqscan) | 10.0 (100%) | 1.000 | 16.3/17.5 | 88.9/176.4 | - |
| pgvector hnsw relaxed, cap 100000 (forced) | 3.3 (2%) | 0.310 | 180.2/206.6 | 371.7/455.7 | 100123 fetched |
| pg_pinecone default (off, rm 10) | 3.0 (0%) | 0.296 | 8.6/9.8 | 7.8/9.0 | 1000 resc / 1000 fetched |
| pg_pinecone relaxed, cap 20000, rm 10 | 10.0 (100%) | 0.762 | 15.5/20.2 | 15.4/21.8 | 3418 resc / 2939 fetched |
| pg_pinecone relaxed, cap 20000, rm 50 | 10.0 (100%) | 0.805 | 17.3/21.5 | 18.7/21.9 | 4717 resc / 2864 fetched |
| **C1 1% key + 1% residual (113 rows)** | | | | | |
| exact seqscan | 10.0 (100%) | 1.000 | 19.7/21.3 | 97.1/231.5 | - |
| pgvector default (seqscan) | 10.0 (100%) | 1.000 | 19.3/20.8 | 102.0/252.7 | - |
| pgvector hnsw relaxed, cap 100000 (forced) | 8.3 (59%) | 0.649 | 160.3/188.2 | 281.6/339.4 | 67150 fetched |
| pg_pinecone default (off, rm 10) | 9.6 (76%) | 0.752 | 8.5/10.0 | 8.4/16.6 | 1000 resc / 700 fetched |
| pg_pinecone relaxed, cap 20000, rm 10 | 10.0 (100%) | 0.791 | 8.5/11.7 | 8.3/12.1 | 1236 resc / 700 fetched |
| pg_pinecone relaxed, cap 20000, rm 50 | 10.0 (100%) | 0.935 | 16.8/19.8 | 20.5/34.3 | 4617 resc / 749 fetched |
| **C2 10% key + 0.3% residual (326 rows)** | | | | | |
| exact seqscan | 10.0 (100%) | 1.000 | 18.9/20.4 | 90.2/215.3 | - |
| pgvector default (seqscan) | 10.0 (100%) | 1.000 | 17.9/21.9 | 89.9/197.1 | - |
| pgvector hnsw relaxed, cap 100000 (forced) | 10.0 (100%) | 0.937 | 111.8/133.5 | 297.9/397.4 | 32702 fetched |
| pg_pinecone default (off, rm 10) | 3.4 (0%) | 0.324 | 8.5/9.8 | 9.0/10.3 | 1000 resc / 1000 fetched |
| pg_pinecone relaxed, cap 20000, rm 10 | 10.0 (100%) | 0.778 | 15.3/20.6 | 18.1/26.0 | 3512 resc / 2942 fetched |
| pg_pinecone relaxed, cap 20000, rm 50 | 10.0 (100%) | 0.884 | 18.8/21.4 | 22.0/25.7 | 5070 resc / 2850 fetched |
| **C2 10% key + 1% residual (1,004 rows)** | | | | | |
| exact seqscan | 10.0 (100%) | 1.000 | 21.7/23.1 | 102.6/202.1 | - |
| pgvector default (hnsw index) | 0.0 (0%) | 0.003 | 0.8/1.3 | 1.2/2.0 | 40 fetched |
| pgvector hnsw relaxed, cap 100000 | 10.0 (100%) | 0.986 | 49.4/109.0 | 90.3/212.9 | 10244 fetched |
| pg_pinecone default (off, rm 10) | 8.5 (44%) | 0.665 | 9.7/11.6 | 10.3/14.8 | 1000 resc / 805 fetched |
| pg_pinecone relaxed, cap 20000, rm 10 | 10.0 (100%) | 0.786 | 11.7/14.9 | 12.1/15.9 | 1568 resc / 859 fetched |
| pg_pinecone relaxed, cap 20000, rm 50 | 10.0 (100%) | 0.979 | 20.7/25.2 | 22.3/25.7 | 5000 resc / 959 fetched |

### Best point per criterion
Lowest p50 at 1 client among index-scan points of the grid (`rel` = `relaxed_order`, cells read p50 at 1 / 8 clients); ties within 10% go to the default knobs. The extension column is the extra run below. The seqscan has recall 1.000.

| cell | criterion | pgvector hnsw index (best) | pg_pinecone, grid (best) | pg_pinecone, extension (best) | exact seqscan @1 / @8 ms |
|---|---|---|---|---|---|
| C1 1% + 0.3% (33 rows) | recall >= 0.95 | none (max recall 0.312) | none (max recall 0.805) | rel cap 20000 rm 50 sf 1: recall 0.953, 22 / 23 ms | 17 / 85 |
| C1 1% + 0.3% (33 rows) | fills 10 rows | none (max 3.3 rows) | rel cap 20000 rm 10: recall 0.762, 15 / 15 ms | - | 17 / 85 |
| C1 1% + 1% (113 rows) | recall >= 0.95 | none (max recall 0.650) | none (max recall 0.935) | not run | 20 / 97 |
| C1 1% + 1% (113 rows) | fills 10 rows | none (max 8.3 rows) | rel cap 20000 rm 10: recall 0.791, 9 / 8 ms | - | 20 / 97 |
| C2 10% + 0.3% (326 rows) | recall >= 0.95 | none (max recall 0.938) | none (max recall 0.884) | rel cap 20000 rm 100: recall 0.973, 34 / 43 ms | 19 / 90 |
| C2 10% + 0.3% (326 rows) | fills 10 rows | rel cap 100000 ef 40: recall 0.937, 112 / 298 ms | rel cap 20000 rm 10: recall 0.778, 15 / 18 ms | - | 19 / 90 |
| C2 10% + 1% (1,004 rows) | recall >= 0.95 | rel cap 100000 ef 40: recall 0.986, 49 / 90 ms | rel cap 20000 rm 50: recall 0.979, 21 / 22 ms | not run | 22 / 103 |
| C2 10% + 1% (1,004 rows) | fills 10 rows | rel cap 100000 ef 40: recall 0.986, 49 / 90 ms | rel cap 20000 rm 10: recall 0.786, 12 / 12 ms | - | 22 / 103 |

## pg_pinecone per configuration (`scan_fraction` default)
Cells are rows at `LIMIT 10` (share of queries that got 10), recall@10, p50/p95 at 1 client, p50 at 8 clients, rescored rows / batches / share of queries stopped by the cap. `rm` = `rescore_multiplier`. Rows with `cap` read the same at 20000 and 100000 unless shown.

| config (C1) | 0.3%: rows (fill) | recall | p50/p95 @1 | p50 @8 | resc / batches / capped | 1%: rows (fill) | recall | p50/p95 @1 | p50 @8 | resc / batches / capped |
|---|---|---|---|---|---|---|---|---|---|---|
| off cap - rm 10 | 3.0 (0%) | 0.296 | 8.6/9.8 | 7.8 | 1000 / 1.0 / 0.00 | 9.6 (76%) | 0.752 | 8.5/10.0 | 8.4 | 1000 / 1.0 / 0.00 |
| off cap - rm 50 | 10.0 (98%) | 0.800 | 17.5/19.9 | 19.1 | 4617 / 1.0 / 0.00 | 10.0 (100%) | 0.935 | 17.2/19.8 | 18.7 | 4617 / 1.0 / 0.00 |
| rel cap 2000 rm 10 | 6.4 (2%) | 0.562 | 11.1/13.7 | 11.2 | 2002 / 2.0 / 0.98 | 10.0 (100%) | 0.791 | 8.7/12.0 | 8.0 | 1236 / 1.2 / 0.00 |
| rel cap 20000 rm 10 | 10.0 (100%) | 0.762 | 15.5/20.2 | 15.4 | 3418 / 3.4 / 0.00 | 10.0 (100%) | 0.791 | 8.5/11.7 | 8.3 | 1236 / 1.2 / 0.00 |
| rel cap 20000 rm 50 | 10.0 (100%) | 0.805 | 17.3/21.5 | 18.7 | 4717 / 1.0 / 0.00 | 10.0 (100%) | 0.935 | 16.8/19.8 | 20.5 | 4617 / 1.0 / 0.00 |
| str cap 2000 rm 10 | 3.2 (0%) | 0.305 | 10.8/12.5 | 10.8 | 2002 / 2.0 / 1.00 | 9.7 (82%) | 0.752 | 7.9/11.2 | 7.8 | 1236 / 1.2 / 0.18 |
| str cap 20000 rm 10 | 3.6 (0%) | 0.305 | 36.3/39.2 | 42.6 | 10169 / 11.2 / 0.00 | 9.8 (92%) | 0.752 | 9.0/41.1 | 8.9 | 2393 / 2.5 / 0.00 |
| str cap 100000 rm 10 | 3.6 (0%) | 0.305 | 36.7/40.0 | 42.3 | 10169 / 11.2 / 0.00 | 9.8 (92%) | 0.752 | 8.3/36.9 | 8.0 | 2393 / 2.5 / 0.00 |

| config (C2) | 0.3%: rows (fill) | recall | p50/p95 @1 | p50 @8 | resc / batches / capped | 1%: rows (fill) | recall | p50/p95 @1 | p50 @8 | resc / batches / capped |
|---|---|---|---|---|---|---|---|---|---|---|
| off cap - rm 10 | 3.4 (0%) | 0.324 | 8.5/9.8 | 9.0 | 1000 / 1.0 / 0.00 | 8.5 (44%) | 0.665 | 9.7/11.6 | 10.3 | 1000 / 1.0 / 0.00 |
| off cap - rm 50 | 10.0 (99%) | 0.881 | 19.0/21.1 | 22.5 | 5000 / 1.0 / 0.00 | 10.0 (100%) | 0.979 | 21.2/26.4 | 24.8 | 5000 / 1.0 / 0.00 |
| rel cap 2000 rm 10 | 6.5 (7%) | 0.559 | 11.2/12.8 | 12.4 | 2000 / 2.0 / 0.93 | 10.0 (99%) | 0.785 | 11.7/14.3 | 12.2 | 1560 / 1.6 / 0.01 |
| rel cap 20000 rm 10 | 10.0 (100%) | 0.778 | 15.3/20.6 | 18.1 | 3512 / 3.5 / 0.00 | 10.0 (100%) | 0.786 | 11.7/14.9 | 12.1 | 1568 / 1.6 / 0.00 |
| rel cap 20000 rm 50 | 10.0 (100%) | 0.884 | 18.8/21.4 | 22.0 | 5070 / 1.0 / 0.00 | 10.0 (100%) | 0.979 | 20.7/25.2 | 22.3 | 5000 / 1.0 / 0.00 |
| str cap 2000 rm 10 | 3.5 (0%) | 0.328 | 11.1/12.6 | 12.2 | 2000 / 2.0 / 1.00 | 8.8 (52%) | 0.665 | 11.8/14.1 | 12.7 | 1560 / 1.6 / 0.48 |
| str cap 20000 rm 10 | 3.7 (0%) | 0.328 | 64.1/71.7 | 150.1 | 20000 / 20.0 / 1.00 | 8.9 (58%) | 0.665 | 12.3/76.6 | 17.9 | 9472 / 9.5 / 0.42 |
| str cap 100000 rm 10 | 3.9 (0%) | 0.328 | 351.8/389.3 | 742.5 | 100488 / 101.0 / 0.83 | 9.1 (63%) | 0.665 | 12.5/362.6 | 49.1 | 41612 / 41.8 / 0.33 |

## pgvector hnsw (ef 40; ef 200 is the same within noise)
Cells are rows (fill) / recall / p50 at 1 client / p50 at 8 clients, in ms. "default plan" is what the planner runs: seq = exact parallel seqscan, idx = hnsw. The index rows are forced where the planner prefers the seqscan.

| hnsw (ef 40) | C1 0.3% | C1 1% | C2 0.3% | C2 1% |
|---|---|---|---|---|
| default plan | 10.0 (100%) / 1.000 / 16 / 89 seq | 10.0 (100%) / 1.000 / 19 / 102 seq | 10.0 (100%) / 1.000 / 18 / 90 seq | 0.0 (0%) / 0.003 / 1 / 1 idx |
| index, off | 0.0 (0%) / 0.000 / 1 / 1 | 0.0 (0%) / 0.001 / 1 / 1 | 0.0 (0%) / 0.002 / 2 / 1 | 0.0 (0%) / 0.003 / 1 / 1 |
| index, relaxed cap 20000 | 0.7 (0%) / 0.069 / 29 / 38 | 2.1 (9%) / 0.191 / 27 / 37 | 6.2 (18%) / 0.512 / 26 / 37 | 9.8 (91%) / 0.791 / 22 / 31 |
| index, relaxed cap 100000 | 3.3 (2%) / 0.310 / 180 / 372 | 8.3 (59%) / 0.649 / 160 / 282 | 10.0 (100%) / 0.937 / 112 / 298 | 10.0 (100%) / 0.986 / 49 / 90 |
| index, strict cap 100000 | 3.3 (2%) / 0.309 / 170 / 315 | 8.3 (58%) / 0.640 / 161 / 277 | 10.0 (100%) / 0.915 / 112 / 195 | 10.0 (100%) / 0.944 / 50 / 77 |

## Where the cap binds (review M1)
Arithmetic: with nothing pushed the first batch probes 127 of 1000 lists (4/sqrt(n)), about 127k candidates, 6x the default cap of 20000. With a pushed key the ramp probes 500 lists and the candidates are the key's rows in them (C1 ~5k of 10,170; C2 ~50k of 100,844).

| case (rm 10 unless noted) | capped | what happens |
|---|---|---|
| pushed key, relaxed, cap 20000 (default), all four cells | 0.00 | stops after 1.2-3.5k rescored rows, 10 rows found |
| same, cap 2000, 0.3% cells | 0.93-0.98 | 6.4-6.5 rows, recall 0.56 |
| same, strict, cap 20000 / 100000, C2/0.3% | 1.00 / 0.83 | 20,000 / 100,488 rescored; lists probed 500 / 1000 |
| nothing pushed, 1% predicate, 10,170 matches (100 queries) | 0.10 | 9.56 rows, recall 0.51; 127 lists at every cap up to 10M (matches were always in the first set) |
| nothing pushed, 33 matches (50 queries), cap 20000 | 1.00 | 0.64 rows, recall 0.06, 127 lists (never widens), 87 ms |
| same, cap 100000 | 0.98 | 2.6 rows, recall 0.24, 374 ms |
| same, cap 1,000,000 | 0.00 | 10 rows, recall 0.64, widens (376 lists), 313k rescored (p95 507k), 1.2 s p50 / 1.9 s p95 |

M1 holds. At 1M the default cap ends the scan before it can widen whenever matches are sparse. The pushed-key cells hide it: the key shrinks the first set and `relaxed` stops at 10 rows. Raising the cap is a poor fix (313k heap fetches). The review's widen trigger (next sketch estimate against the next list's centroid distance) is the real fix. Probes: one-off scripts and logs (not committed).

## Strict vs relaxed order
- **Rows and recall.** Strict at rm 10 returns 3.2-3.9 rows in the 0.3% cells (0% filled) against 10.0 for relaxed; recall 0.31-0.33 against 0.76-0.78. At rm 50 one batch suffices, so strict = relaxed = off.
- **Cost.** Strict spends the cap. C1/0.3%: 10,169 rescored (the whole key set), 36 ms. C2/0.3%: 20,000 rescored, 64 ms (150 ms at 8 clients) at cap 20000; 100,488 rescored, 352 ms (742 ms) at cap 100000.
- **Order** (C2/0.3% probe, 100 queries): relaxed rm 10 returns its 10 rows out of distance order in 96% of queries (7.4 of 45 pairs inverted); relaxed rm 50 in 2%; strict in 0%, with 3.6 rows. A CTE re-sort of the 10 rows (`ORDER BY distance + 0`) restores order for free, so order does not justify strict.

## Cost per query (heap fetches)
`rescored` = rows pg_pinecone fetched and scored inside the AM (about 3.5 us each: 352 ms / 100k). `fetched` = tuples the executor pulled through the index scan (EXPLAIN ANALYZE on 8 queries per point, 1 client). `buffers` = shared buffer accesses of the whole query.

| path | rescored | fetched | buffers |
|---|---|---|---|
| exact seqscan (warm) | vectors of the 326-1,004 matches only | 1M main-heap tuples, 12,050 pages | 12.1k-17.5k |
| hnsw forced, relaxed cap 20000 / 100000 | - | 11-20k / 10-100k | 33-44k / 80-222k |
| pg_pinecone `off`, rm 10 | 1,000 | 700-1,000 | 15k |
| pg_pinecone relaxed, rm 10, cap 20000 | 1.2-3.5k | 0.7-2.9k | 15-31k |
| pg_pinecone relaxed, rm 50 | 4.6-5.1k | 0.7-2.9k | 32-38k |
| pg_pinecone strict, rm 10, cap 100000 (C2/0.3%) | 100,488 | 1,160 | 520k |

## What reaches recall 0.95 (extension, not in the brief)
The grid stops at rm 50, so 0.95 is out of reach in most cells. Same harness, `relaxed_order`, cap 20000 (`configs/residual-ext.yaml`, `residual-ext2.yaml`). Two limits decide recall: list coverage (the ramp probes 50% of the lists) and the pool (rm x 100). C1/0.3% is coverage-bound: 0.805 at any rm until sf 1.0 (rm 50: 0.953 at 22 ms; rm 100: 1.000 at 34 ms). C2/0.3% is pool-bound: rm 100 gives 0.973 at 34 ms and sf does nothing. The shipping default for the 1% key is mode L (exact over the key rows): recall 1.000, 33 ms at 1 client but 109-117 ms at 8, slower than the seqscan (17-20 ms, 85-97 ms). The planner picks it anyway: the cost model ignores the key's size.

| cell | config | rows (fill) | recall | p50/p95 @1 ms | p50/p95 @8 ms | rescored / fetched |
|---|---|---|---|---|---|---|
| C1 1% key + 0.3% (33 rows) | exact seqscan | 10.0 (100%) | 1.000 | 16.9/18.6 | 85.3/180.4 | - |
| C1 1% key + 0.3% (33 rows) | relaxed, cap 20000, rm 50, sf 0.5 | 10.0 (100%) | 0.805 | 16.8/21.9 | 19.0/23.8 | 4717 resc / 2864 fetched |
| C1 1% key + 0.3% (33 rows) | relaxed, cap 20000, rm 100, sf 0.5 | 10.0 (100%) | 0.805 | 17.4/21.2 | 18.8/24.3 | 5005 resc / 2870 fetched |
| C1 1% key + 0.3% (33 rows) | relaxed, cap 20000, rm 200, sf 0.5 | 10.0 (100%) | 0.805 | 16.9/20.6 | 18.5/22.3 | 5005 resc / 2870 fetched |
| C1 1% key + 0.3% (33 rows) | relaxed, cap 20000, rm 50, sf 1.0 | 10.0 (100%) | 0.953 | 22.3/24.2 | 23.3/26.7 | 5000 resc / 3128 fetched |
| C1 1% key + 0.3% (33 rows) | relaxed, cap 20000, rm 100, sf 1.0 | 10.0 (100%) | 1.000 | 34.0/37.0 | 38.8/120.5 | 10000 resc / 3106 fetched |
| C1 1% key + 0.3% (33 rows) | relaxed, cap 20000, rm 200, sf 1.0 | 10.0 (100%) | 1.000 | 32.2/35.0 | 38.7/110.0 | 10170 resc / 3106 fetched |
| C2 10% key + 0.3% (326 rows) | exact seqscan | 10.0 (100%) | 1.000 | 18.9/20.4 | 90.2/215.3 | - |
| C2 10% key + 0.3% (326 rows) | relaxed, cap 20000, rm 50, sf 0.5 | 10.0 (100%) | 0.884 | 20.8/23.9 | 25.2/31.3 | 5070 resc / 2850 fetched |
| C2 10% key + 0.3% (326 rows) | relaxed, cap 20000, rm 100, sf 0.5 | 10.0 (100%) | 0.973 | 34.3/37.8 | 42.5/88.1 | 10000 resc / 2713 fetched |
| C2 10% key + 0.3% (326 rows) | relaxed, cap 20000, rm 200, sf 0.5 | 10.0 (100%) | 0.991 | 61.5/71.3 | 118.6/163.5 | 20000 resc / 2757 fetched |
| C2 10% key + 0.3% (326 rows) | relaxed, cap 20000, rm 50, sf 1.0 | 10.0 (100%) | 0.882 | 27.2/33.2 | 29.3/44.8 | 5060 resc / 2834 fetched |
| C2 10% key + 0.3% (326 rows) | relaxed, cap 20000, rm 100, sf 1.0 | 10.0 (100%) | 0.974 | 40.4/45.3 | 49.0/96.9 | 10000 resc / 2779 fetched |
| C2 10% key + 0.3% (326 rows) | relaxed, cap 20000, rm 200, sf 1.0 | 10.0 (100%) | 0.997 | 67.0/75.3 | 129.7/169.8 | 20000 resc / 2784 fetched |
| C1 1% key + 0.3% residual (33 rows) | shipping default, mode L (threshold 0.03) | 10.0 (100%) | 1.000 | 32.5/34.6 | 108.7/192.8 | 10170 resc / 3106 fetched |
| C1 1% key + 1% residual (113 rows) | shipping default, mode L (threshold 0.03) | 10.0 (100%) | 1.000 | 33.8/39.1 | 116.8/196.8 | 10170 resc / 715 fetched |

## Defaults: what I would change
1. **`pinecone.iterative_scan = relaxed_order` (your call).** It fills the LIMIT in every cell (100% against 0-76%) for +0-7 ms, and its first batch is byte-identical to `off`, so queries that stop inside batch 1 do not change. The price is order: batch-wise, 96% unordered at rm 10 in C2 (re-sort in a CTE). Keep `off` if order beats fill.
2. **Keep `max_scan_tuples` 20000.** It never binds for a keyed scan under `relaxed`. It is too low for sparse un-pushed residuals at 1M (M1), and raising it buys 313k heap fetches. Fix it with the widen trigger or a cap that scales with the table.
3. **Keep `rescore_multiplier` 10, document the lever.** Under a selective residual the pool decides recall: rm 50 gives 0.81-0.98 at 17-21 ms, rm 100 gives 0.97 in C2/0.3% at 34 ms. A residual-aware pool (cut after the filter) is the design fix.
4. **`scan_fraction`: no change.** The filtered ramp already probes 50% of the lists. sf 1.0 helps only where coverage binds (C1/0.3%) and does nothing in C2.
5. **Docs: warn off `strict_order`** (up to 100k rescored rows for 3.9 rows) and say that a 1% key runs exact mode L by default (33 ms), which the seqscan beats at 17 ms.
6. **Look at `amcostestimate`.** It prices a 1% and a 10% key the same (index cost 28.6k both), so it picks mode L at 33 ms where the seqscan takes 17 ms.

## Caveats
- Laptop, not idle (other apps took ~1.5 cores), `nice -n 10`. One run per point, 500 queries. The 33-row cell has few matches, so recall moves in steps.
- Noise: 20 identical seqscan points differ by 10-12% at 1 client and 11-44% at 8; identical pg_pinecone configs differ by about +-5% (p10-p90). Read 8-client p95 as indicative.
- Release Postgres. The cassert pgrx build was 2.5x slower on the 50k smoke (434 s against 176 s) and up to 10x on rescore-heavy points (rm 50: 130 ms against 11.5 ms). Do not compare across builds.
- The seqscan baseline is cheap because vectors sit in TOAST (12k-page main heap) and the cache is warm. A cold first run took 51 ms.
- hnsw index rows are forced where the planner prefers the seqscan. The planner rows are what a user gets.
- Cost probe: 8 queries at 1 client per index point, not 500.
- The 1% key is forced into mode H (`mode_l_threshold 0.005`). The residuals are uncorrelated with the key and the query. A correlated residual is the worst case for widening and ran only as the sparse probe.

## Not run
Real Pinecone and the remote engine; `LIMIT` above 10; 0.1% filters; more than 8 clients; Linux/AVX-512; inserts or VACUUM during a scan; pgvector ivfflat iterative scan; `hnsw.scan_mem_multiplier`; the M1 widen trigger and M3 early end (design items).

## Files
Not committed: `results/residual_{c1,c2,ext,ext2}_DIRTY_*.{json,md}` (raw and harness reports), `bench3/` (scripts, probe logs, `summarize.py`, `final_tables.py`), `progress-bench.md` (step log). The configs that produced them are `configs/residual-{c1,c2,ext,ext2}.yaml`.
