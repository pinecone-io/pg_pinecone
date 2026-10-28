# Vector benchmark: pg_pinecone 0.11.0 against pgvector HNSW on wiki_dpr 10M (2026-10-27)

*One AWS box (`r6id.4xlarge`), PostgreSQL 17.11, pgvector 0.8.7, **pg_pinecone 0.11.0 release tarball** (pg17, sha256 `2d083429...bf86`, installed `pg_pinecone.so` `a838fad2...e45a`), wiki_dpr 10M x 768d, L2, k = 10. `configs/local_10m.yaml`: unfiltered, a 1% correlated filter, and a 10% correlated filter, out-of-region queries only. `pinecone.mode_l_threshold` left at the engine default 0.03. 500 held-out queries for the approximate systems; the exact scan runs 50 of those 500, because one unfiltered exact query takes about 15 s. Concurrency 1 and 4 means that many Postgres sessions running the same queries at once. 110 of 110 points valid, 3.9 h. Facts of the box: `box/BOX-2026-10-27-10m.md`, `box/settings-2026-10-27-10m.json`.*

## Verdict

1. **Unfiltered, native is ahead once recall is 0.95 or higher.** Its cheapest point is already recall 0.967 at 3.8 ms (25 of 3,163 lists). HNSW reaches 0.95 at 4.6 ms and 0.985 at 10.2 ms (`ef_search` 320), and does not reach 0.99 in this sweep. Native reaches 0.993 at 8.5 ms (78 lists).
2. **At four concurrent sessions the unfiltered gap shrinks.** Near recall 0.97 the two are 4.2 ms / 920 QPS against 4.7 ms / 897 QPS. Native still goes higher: 0.993 at 8.8 ms, against HNSW's best of 0.985 at 7.7 ms.
3. **A 1% filter is under the default threshold, so native uses mode L** and scans all 103,509 matching rows. Recall is 1.000 at 669 ms. A filtered sequential scan is also exact and faster, at 391 ms: the vector column is stored out of line, so mode L pays a TOAST fetch per matching row. HNSW stays at recall 0.84 to 0.85 across the whole `ef_search` sweep, median about 470 ms, p99 about 3.1 s.
4. **A 10% filter is over the threshold, so native uses mode H.** Recall is 1.000 at 77 ms, p95 83 ms (900 lists, `rescore_multiplier` 2). HNSW's best is recall 0.904 at 27 ms, p95 167 ms. The median is faster. The recall and the tail are not. This is the same shape as the 1M filtered run, at higher latency.
5. **The native index is 1.3 GB and builds in about 11 minutes. HNSW is 41 GB and took 62 minutes.** The HNSW graph spilled out of `maintenance_work_mem` (8 GB) after 2,287,034 tuples, so that build time includes disk spill. A native index holds one filter column, so this run built three of them (unfiltered, 1%, 10%). Together they are about 34 minutes and 3.9 GB, still less time and much less disk than the one HNSW index, which serves all three workloads.

## What 10M adds to the 1M runs

The 1M filtered numbers are `REPORT-halfvec-filtered.md` (pg_pinecone 0.10.0, threshold pinned at 0.005 so both keys were mode H). The 1M unfiltered numbers are `REPORT-halfvec.md`.

| Workload | 1M | 10M |
|---|---|---|
| Unfiltered, recall 0.99, one session | native 2.3 ms, HNSW 6.7 ms | native 8.5 ms at 0.993; HNSW does not reach 0.99 (0.985 at 10.2 ms) |
| 1% filter, one session | mode H: native 0.981 @ 8.3 ms, HNSW 0.926 @ 58 ms, exact 42 ms | mode L: native 1.000 @ 669 ms, HNSW 0.853 @ 488 ms (p99 3.1 s), exact 391 ms |
| 10% filter, one session | mode H: native 1.000 @ 12.3 ms, HNSW 0.920 @ 5.9 ms, exact 157 ms | mode H: native 1.000 @ 77 ms, HNSW 0.904 @ 27 ms, exact 1.7 s |
| Index size | native 136 MB, HNSW 4.1 GB | native 1.3 GB, HNSW 41 GB |

The 10% row is the scale check: native still returns recall 1.0, HNSW still tops out near 0.90, and HNSW is still faster at the median. The 1% row is a different engine path. At 1M that key was forced through the approximate index and native was the fast one. At 10M, with the threshold left at 0.03, native scans every match and loses to a sequential scan.

## Unfiltered, one session

Native sweeps `scan_fraction` at `rescore_multiplier` 0. HNSW sweeps `ef_search`. Recall@10 and p50. QPS is the harness wall throughput.

| Target | pg_pinecone native | pgvector HNSW |
|---|---|---|
| 0.90 | no point this low; cheapest is 0.967 @ 3.8 ms (25 lists), 260 QPS | 3.0 ms |
| 0.95 | 3.8 ms (already 0.967) | 4.6 ms |
| 0.99 | 8.5 ms (78 lists, recall 0.993), 116 QPS | not reached; best is 0.985 @ 10.2 ms (`ef_search` 320), 131 QPS |

The build calibrates to 68 of 3,163 lists for a target recall of 0.95, and that point scores held-out recall 0.990. Its one-session p50 of 17.2 ms is the first point of the sweep, read cold. The same point with four sessions is 8.5 ms, in line with the later one-session points (50 lists at 6.2 ms and recall 0.987, 78 lists at 8.5 ms and recall 0.993).

Four sessions, near recall 0.97: native 4.2 ms / 920 QPS (25 lists), HNSW 4.7 ms / 897 QPS (`ef_search` 160). An exact scan of all 10 million rows is 15.2 s (50 queries).

## Filtered, out-of-region, one session

Realized selectivity is 1.04% and 10.06%. The recorded session GUC `pinecone.mode_l_threshold` is 0.03 on every native point. HNSW uses one index for both filters, `iterative_scan = relaxed_order`, budget 1,000,000 tuples.

| Filter | pg_pinecone native | pgvector HNSW | Exact seqscan (50 queries) |
|---|---|---|---|
| 1% | 1.000 @ 669 ms, p99 707 ms (mode L: all 3,163 lists, 103,509 rows rescored) | 0.853 @ 488 ms, p99 3.1 s (`ef_search` 320; the sweep is 0.844 to 0.853) | 1.000 @ 391 ms |
| 10% | 1.000 @ 77 ms, p95 83 ms (mode H: 900 lists, `rescore_multiplier` 2, 185 rows rescored) | 0.904 @ 27 ms, p95 167 ms (`ef_search` 320; recall 0.95 is not reached, 0.90 is 26.5 ms) | 1.000 @ 1.7 s |

At four sessions the 10% native point stays at 79 ms and 51 QPS. HNSW is 26 ms and 70 QPS, still at recall 0.904.

Each native build warned that `embedding` is `STORAGE EXTERNAL`, so every rescored row is a TOAST fetch. That is the whole mode L cost at 1%: 103,509 fetches per query.

## Build and size

| Index | Build | Size | Peak RSS during the build |
|---|---|---|---|
| native, unfiltered | 682 s | 1,306 MB | 83 GB |
| native, 1% filter | 698 s | 1,306 MB | 83 GB |
| native, 10% filter | 689 s | 1,306 MB | 83 GB |
| pgvector HNSW (m 16, ef_construction 64) | 3,705 s | 40,906 MB | 26 GB |

Peak RSS is the most RAM the backend held at once. Each native build reported the same calibration: 68 of 3,163 lists, recall 0.984 on 50 sample rows, target 0.95. HNSW notice: `hnsw graph no longer fits into maintenance_work_mem after 2287034 tuples`.

## Not verified

- An HNSW build with `maintenance_work_mem` large enough to hold the graph. The spill slows the build. This run does not say by how much, and the spill does not change which edges are written.
- Rebuilding the table with the vector column `STORAGE PLAIN`, which is what the native build notice recommends. Mode L's gap against the sequential scan is tied to the TOAST fetches.
- `ef_search` above 320, k other than 10, in-region queries, a repeated first sweep point with a warm cache.
- The 1M and 10M binaries differ (0.10.0 against 0.11.0). The harness path is `CREATE INDEX ... USING pinecone` with `mode = local` in both.
