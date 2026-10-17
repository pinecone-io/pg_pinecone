# Vector benchmark: pgvector halfvec (FP16) against FP32 hnsw and pg_pinecone native from main (2026-10-16)

*One AWS box (`r6id.4xlarge`), PostgreSQL 17.11, pgvector 0.8.7, **pg_pinecone 0.10.0 built from `main` at `7763b35`** (unreleased), wiki_dpr 1M x 768d, L2, 500 held-out queries, unfiltered. `configs/halfvec_1m.yaml`, 16 systems, 531 sweep points, 95 min. Facts of the box: `box/BOX-2026-10-16-halfvec.md`. Tables: `box/tables_halfvec.py`. Rerun: `box/README.md`, "Halfvec session".*

## Verdict

1. **FP16 costs no measurable recall on wiki_dpr.** The forced exact scan of the FP16 heap scores recall@10 1.000 against the FP32 truth, and on every one of the 531 points recall against FP32 truth and against exact search over FP16-rounded corpus and queries differ by at most 0.0002.
2. **pgvector's FP16 index is faster and half the size of its FP32 one.** At the same `ef_search` the FP16 expression index is 30 to 35% faster at equal recall (ef 80: 1.75 ms against 2.50 ms, recall 0.955 against 0.947; ef 160: 2.72 against 4.17 ms). Latency at recall@10 0.95 / 0.99, one client: FP32 2.67 / 6.72 ms, FP16 1.71 / 4.65 ms (m16); FP32 2.49 / 5.95 ms, FP16 heap 1.57 / 4.07 ms (m32). Index 4.1 GB to 2.0 GB; heap + index 8.2 GB (FP32) to 6.2 GB (FP16 index over the FP32 heap) to 3.7 GB (FP16 heap). Builds 79 to 55 to 62 s (m16), 382 to 270 to 281 s (m32).
3. **FP32 rerank buys nothing here and costs latency.** With rounding free, 2x rerank returns the same recall as the plain FP16 index at the same `ef_search` (within 0.004 from ef 80 up, inside the spread between two builds) and adds 0.2 to 0.4 ms. At k = 100 the 5x and 10x arms must ask hnsw for 500 and 1,000 candidates: 12.4 to 27.6 ms at one client, against 3.2 ms for the FP16 heap at recall 0.95.
4. **Native from main is the k = 10 latency winner at one client at every target, and at 16 clients from 0.95 up, with a 136 MB index.** (At 16 clients and 0.90 the FP16 expression index is 1.65 ms against the 0.90 build's 1.94 ms; at 0.95, 2.13 against 2.22 ms for FP16 heap m32.) Its 0.95 build serves recall@10 0.953 at 1.27 ms by default (12 of 1,000 lists, a 100-row pool), 0.99 at 2.28 ms (one client) and 3.26 ms (16 clients), against 4.07 / 4.98 ms for the best pgvector arm (FP16 heap, m32); 8,319 QPS at 16 clients at its default. Build 45 s. Its total footprint is 4.26 GB, because it keeps the FP32 heap: more than FP16 heap + hnsw (3.7 GB), half of FP32 hnsw (8.2 GB).
5. **FP16 closes 47 to 88% of pgvector's latency gap to native, and all of it at the easy end.** At k = 10 and one client, FP32 hnsw is 1.74 / 2.13 / 2.84 times native's p50 at recall 0.90 / 0.95 / 0.99, FP16 hnsw 1.16 / 1.35 / 1.97 times. FP16 is as fast as native or faster at recall 0.90 (except at k = 10 with one client, 1.16 times) and at k = 100, 0.95 with 16 clients; at 0.99 native stays 1.3 to 2.0 times faster than FP16 (2.0 to 2.8 times faster than FP32). On size FP16 halves the gap: the index is 15 times native's (FP32: 30 times), and an FP16 heap makes pgvector's total footprint 3.7 GB, below native's 4.3 GB. Section "Gap to native".
6. **Calibration holds on held-out queries.** The four builds (`target_recall` 0.90 / 0.95 / 0.98 / 0.99) chose 8 / 12 / 25 / 37 lists and a 100-row pool, and serve held-out recall@10 0.921 / 0.953 / 0.983 / 0.990; the build's own sample of table rows reads 0.006 to 0.029 higher, as `docs/reference.md` warns. At k = 10 the probe count is the only lever: at 12 lists a pool of 100 to 2,000 rows leaves recall at 0.953 and raises p50 from 1.27 to 3.06 ms.
7. **k = 100 needs a pinned pool on native.** At the calibrated defaults recall@100 is 0.754 to 0.799 on all four builds; a 1,000-row pool (`rescore_multiplier` 10) gives 0.880 / 0.926 / 0.970 / 0.982. With the probes swept at that pool, native reaches recall@100 0.95 at 2.72 ms and 0.99 at 4.03 ms (one client), against 3.18 and 5.77 ms for the best pgvector arms.
8. **`STORAGE PLAIN` on the FP32 heap: the forced exact scan is 324 ms, against 1,120 ms on the 2026-10-14 box with pgvector's default TOAST** (same data, same instance type). Across the two sessions pgvector FP32 hnsw is 20 to 28% slower at the same `ef_search` (from ef 20 up; equal at ef 10); not isolated, so this report compares within its own session only.
9. **Cost:** AWS 1.92 h, $2.32 for the instance; no Pinecone.

## Gap to native

The fastest measured point of each family that reaches the recall target (no interpolation): pgvector FP32 is `pgvector_fp32_hnsw` and its m32 build; pgvector FP16 the FP16 expression and FP16 heap builds, m16 and m32 (the rerank arms are left out: never faster than plain FP16); native any of the four builds at any pool and probe count. `x native` is the family's p50 over native's; `FP16 closes` is (FP32 - FP16) / (FP32 - native), `all` where FP16 is at or below native.

| k | Clients | Recall >= | pgvector FP32 p50 ms (x native) | pgvector FP16 p50 ms (x native) | Native p50 ms | FP16 closes of the FP32 gap | Best QPS at 16 clients, FP32 / FP16 / native |
|---|---|---|---|---|---|---|---|
| 10 | 1 | 0.90 | 1.85 (1.74x) | 1.23 (1.16x) | 1.06 | 78% |  |
| 10 | 1 | 0.95 | 2.70 (2.13x) | 1.71 (1.35x) | 1.27 | 69% |  |
| 10 | 1 | 0.99 | 6.46 (2.84x) | 4.48 (1.97x) | 2.28 | 47% |  |
| 10 | 16 | 0.90 | 2.81 (1.45x) | 1.75 (0.90x) | 1.94 | all | 7,091 / 10,615 / 9,304 |
| 10 | 16 | 0.95 | 3.54 (1.66x) | 2.30 (1.08x) | 2.13 | 88% | 4,980 / 7,229 / 8,319 |
| 10 | 16 | 0.99 | 8.34 (2.48x) | 5.46 (1.62x) | 3.36 | 58% | 1,917 / 2,726 / 5,287 |
| 100 | 1 | 0.90 | 3.38 (1.35x) | 2.30 (0.92x) | 2.50 | all |  |
| 100 | 1 | 0.95 | 5.15 (2.06x) | 3.24 (1.29x) | 2.50 | 72% |  |
| 100 | 1 | 0.99 | 11.34 (2.16x) | 7.70 (1.47x) | 5.24 | 60% |  |
| 100 | 16 | 0.90 | 5.11 (1.34x) | 3.38 (0.89x) | 3.81 | all | 4,100 / 5,761 / 4,660 |
| 100 | 16 | 0.95 | 7.53 (1.83x) | 4.03 (0.98x) | 4.12 | all | 2,591 / 4,234 / 3,848 |
| 100 | 16 | 0.99 | 14.72 (1.95x) | 9.82 (1.30x) | 7.56 | 68% | 1,082 / 1,568 / 2,108 |

| Family | Index MB | Heap + index GB | Build s | Build peak RSS GB |
|---|---|---|---|---|
| pgvector FP32 | 4,094 | 8.2 | 79 to 382 | 8.6 to 9.6 |
| pgvector FP16 | 2,047 to 2,123 | 3.7 to 6.2 | 55 to 274 | 4.6 to 6.0 |
| native | 136 | 4.3 | 45 | 7.9 |

- **Latency: FP16 removes 69 to 88% of the FP32 gap at recall 0.95 (all of it at k = 100 with 16 clients) and 47 to 68% at 0.99.** The gap left at 0.99 is native's: it reaches 0.99 by probing more lists with a 100-row exact rescore, where hnsw has to widen `ef_search` (160 to 320) whatever the precision.
- **FP16 hnsw matches or beats native at recall 0.90, except at k = 10 with one client (1.16 times), and at k = 100, 0.95 with 16 clients** (0.89 to 0.98 times its p50; 10,615 against 9,304 QPS at k = 10, 0.90). Native's lead is at high recall and at k = 10 with one client.
- **Throughput at 16 clients follows latency:** at k = 10, 0.95, native serves 8,319 QPS, FP16 7,229, FP32 4,980; at 0.99, 5,287 / 2,726 / 1,917.
- **Size: FP16 halves the index gap but not the footprint gap unless the heap is FP16 too.** Index: FP32 4,094 MB, FP16 2,047 to 2,123 MB, native 136 MB. Heap + index: FP32 8.2 GB, FP16 index over the FP32 heap 6.2 GB, FP16 heap 3.7 GB, native 4.3 GB (it keeps the FP32 heap). Builds: native 45 s against 55 to 62 s (FP16 m16) and 79 s (FP32 m16); the m32 builds take 270 to 382 s.
- **What FP16 gives up to close the gap: nothing measurable in recall on this corpus** (section "FP16 rounding"); native's FP32 heap keeps exact distances for its rescore.

## Setup

- **Machine:** one `r6id.4xlarge` (16 vCPU Xeon Platinum 8375C 2.9 GHz, 128 GB, 950 GB NVMe), us-east-1a, Ubuntu 24.04.5, kernel 7.0.0-1014-aws, launched 13:43 UTC; the recipe of the 2026-10-14 0.9.0 run.
- **Software:** PostgreSQL 17.11 (PGDG), pgvector 0.8.7, pg_pinecone from `main` at `7763b35` (0.10.0: RaBitQ codes, build-time calibration, the blocked scan with its early stop; built on the box with cargo-pgrx 0.19.2 and `-C target-cpu=x86-64-v3`, `pg_pinecone.so` sha256 `50136328...eff3dc`). Postgres: `pg/tuned.linux-aws.conf` (`shared_buffers` 32 GB, `work_mem` 256 MB, `maintenance_work_mem` 8 GB, `jit` and `fsync` off, up to 16 parallel workers).
- **Data:** wiki_dpr, 1,000,000 x 768 float32 (`base_content_sha` `dd93ee85811b7494`, dataset key `real_fd937e5862449934`, the key of `local_1m.yaml`), 500 held-out queries, tie-aware recall against exact FP32 ground truth (top 100, cached). Latency is client time around `execute` and `fetchall`, one connection per client, warm-up untimed; concurrency points run a second pass of at least 5 s. Every point is plan-gated (`EXPLAIN` shows the system's own index); none fell back to a sequential scan. Latency at recall is interpolated linearly between two bracketing points of one curve (`harness/metrics.py`); `<=x` means the cheapest point already exceeds the target and x is its p50.
- **Storage:** the FP32 heap's `embedding` is `STORAGE PLAIN` (`fp32_column_storage: plain`): 2 rows per 8 KB page, nothing TOASTed (`heap` 4,096 MB, TOAST 0). pgvector's default `EXTERNAL` would put every 3,080-byte FP32 vector in the TOAST table and charge the rerank, native and exact FP32 arms a TOAST fetch per vector read that the FP16 heap (1,544 bytes, inline) never pays. The FP16 heap is 1,638 MB.
- **Arms:** pgvector hnsw over the FP32 heap (`vector_l2_ops`); hnsw over `embedding::halfvec(768)` on the FP32 heap (`halfvec_l2_ops`, "expression"); the same with 2x / 5x / 10x FP32 rerank (`LIMIT k x n` FP16 candidates in a materialised CTE, re-sorted by the FP32 vector inside the measured query); hnsw on an FP16 heap (rows rounded on `COPY`, the runner reloads the table); m16 / ef_construction 64 and m32 / ef_construction 200 builds; native at four `target_recall` values, each at its calibrated defaults and with a 1,000-row pool at k = 100, the 0.95 build also over `pinecone.scan_fraction` 0.025 to 0.4 x `pinecone.rescore_multiplier` 2 to 50; forced exact scans of both heaps. The FP16 heap arms ran last, after the reload; native ran on the FP32 heap.
- **Who shared the box:** nothing beside `halfvec_1m` (the smoke ran during `hv_prep`, before it). No CPU samples were kept (`box/BOX-2026-10-16-halfvec.md`, Gaps).

## Build and size

One build per system, `maintenance_work_mem` 8 GB, no graph spilled.

| System | Heap | Index | Column storage | Build s | Index MB | Heap MB | Heap + index MB | Peak RSS GB (leader) |
|---|---|---|---|---|---|---|---|---|
| `pgvector_fp32_hnsw` | vector | vector | plain | 79 | 4,094 | 4,096 | 8,213 | 8.60 |
| `pgvector_fp32_hnsw_m32` | vector | vector | plain | 382 | 4,094 | 4,096 | 8,213 | 9.55 |
| `pgvector_fp16_expression` | vector | halfvec | plain | 62 | 2,047 | 4,096 | 6,166 | 5.44 |
| `pgvector_fp16_expression_m32` | vector | halfvec | plain | 270 | 2,123 | 4,096 | 6,242 | 6.04 |
| `pgvector_fp16_rerank_2x` | vector | halfvec | plain | 57 | 2,047 | 4,096 | 6,166 | 5.38 |
| `pgvector_fp16_rerank_5x` | vector | halfvec | plain | 57 | 2,047 | 4,096 | 6,166 | 5.42 |
| `pgvector_fp16_rerank_5x_m32` | vector | halfvec | plain | 281 | 2,123 | 4,096 | 6,241 | 5.83 |
| `pgvector_fp16_rerank_10x` | vector | halfvec | plain | 67 | 2,047 | 4,096 | 6,166 | 5.43 |
| `pg_pinecone_native_t090` | vector | native | plain | 45 | 136 | 4,096 | 4,255 | 7.85 |
| `pg_pinecone_native_t095` | vector | native | plain | 45 | 136 | 4,096 | 4,255 | 7.86 |
| `pg_pinecone_native_t098` | vector | native | plain | 45 | 136 | 4,096 | 4,255 | 7.90 |
| `pg_pinecone_native_t099` | vector | native | plain | 45 | 136 | 4,096 | 4,255 | 7.89 |
| `fp32_exact_seqscan` | vector | vector | plain | 0 | 0 | 4,096 | 4,119 | - |
| `pgvector_fp16_heap` | halfvec | halfvec | external | 55 | 2,047 | 1,638 | 3,708 | 4.63 |
| `pgvector_fp16_heap_m32` | halfvec | halfvec | external | 274 | 2,123 | 1,638 | 3,784 | 5.54 |
| `fp16_exact_seqscan` | halfvec | halfvec | external | 0 | 0 | 1,638 | 1,661 | - |

- **FP16 halves the hnsw index** (4,094 to 2,047 MB at m16; 2,123 MB at m32) and its build memory (leader RSS 8.6 to 5.4 GB); builds are 20 to 30% shorter.
- **The FP16 heap is 1,638 MB against 4,096 MB** (`PLAIN`, 2 rows a page): heap + index 3.7 GB against 8.2 GB for FP32 hnsw.
- **Native: 136 MB in 45 s at every target** (calibration is part of the build; the four builds differ only in the chosen probe count). The 0.9.0 index on the same data was 147 MB in 235 s.

## Unfiltered, k = 10

p50 in ms at recall@10 0.90 / 0.95 / 0.99, at 1, 4 and 16 clients. pgvector arms along `ef_search`; native's 0.95 build along its probe count (12 calibrated, then 25 to 400 lists) at each pool. The other three native builds are single points (next section).

| System | Curve | c=1 p50 ms at recall 0.90 / 0.95 / 0.99 | c=4 | c=16 | Best recall |
|---|---|---|---|---|---|
| `pgvector_fp32_hnsw` | ef_search | 1.86 / 2.67 / 6.72 | 1.97 / 2.91 / 6.79 | 2.59 / 3.60 / 8.55 | 0.996 |
| `pgvector_fp32_hnsw_m32` | ef_search | 1.79 / 2.49 / 5.95 | 2.04 / 2.64 / 6.24 | 2.74 / 3.36 / 7.68 | 1.000 |
| `pgvector_fp16_expression` | ef_search | 1.33 / 1.71 / 4.65 | 1.26 / 1.90 / 4.58 | 1.65 / 2.42 / 5.94 | 0.998 |
| `pgvector_fp16_expression_m32` | ef_search | 1.23 / 1.75 / 4.14 | 1.45 / 1.83 / 4.19 | 1.86 / 2.34 / 5.31 | 1.000 |
| `pgvector_fp16_rerank_2x` | ef_search | 1.41 / 2.06 / 4.51 | 1.57 / 1.99 / 4.71 | 2.12 / 2.81 / 5.81 | 0.999 |
| `pgvector_fp16_rerank_5x` | ef_search | <=1.93 / 2.72 / 5.59 | <=2.14 / 2.77 / 5.43 | <=3.10 / 3.59 / 6.83 | 0.998 |
| `pgvector_fp16_rerank_5x_m32` | ef_search | <=3.48 / <=3.48 / 5.75 | <=3.64 / <=3.64 / 5.83 | <=4.86 / <=4.86 / 7.35 | 1.000 |
| `pgvector_fp16_rerank_10x` | ef_search | <=3.90 / <=3.90 / 7.12 | <=4.11 / <=4.11 / 7.29 | <=5.40 / <=5.40 / 9.01 | 0.998 |
| `pg_pinecone_native_t095` | probes swept, calibrated pool | <=1.27 / <=1.27 / 2.28 | <=1.41 / <=1.41 / 2.28 | <=2.13 / <=2.13 / 3.26 | 0.999 |
| `pg_pinecone_native_t095` | probes swept, pool 200 | <=1.39 / <=1.39 / 2.42 | <=1.53 / <=1.53 / 2.40 | <=2.19 / <=2.19 / 3.36 | 1.000 |
| `pg_pinecone_native_t095` | probes swept, pool 500 | <=1.82 / <=1.82 / 2.88 | <=1.90 / <=1.90 / 2.84 | <=2.65 / <=2.65 / 3.93 | 1.000 |
| `pg_pinecone_native_t095` | probes swept, pool 1000 | <=2.31 / <=2.31 / 3.30 | <=2.45 / <=2.45 / 3.44 | <=3.27 / <=3.27 / 4.63 | 1.000 |
| `pg_pinecone_native_t095` | probes swept, pool 2000 | <=3.06 / <=3.06 / 4.13 | <=3.35 / <=3.35 / 4.43 | <=4.41 / <=4.41 / 5.94 | 1.000 |
| `pgvector_fp16_heap` | ef_search | 1.34 / 1.73 / 5.25 | 1.35 / 1.76 / 5.15 | 1.84 / 2.27 / 6.69 | 0.998 |
| `pgvector_fp16_heap_m32` | ef_search | 1.17 / 1.57 / 4.07 | 1.26 / 1.64 / 4.08 | 1.74 / 2.22 / 4.98 | 1.000 |

- **FP16 is the faster pgvector representation at every target.** At 0.95: 1.71 ms (expression) and 1.73 ms (heap) against 2.67 ms (FP32), m16. At 0.99 the m32 builds lead: 4.07 ms (FP16 heap), 4.14 ms (expression), 5.95 ms (FP32). At 16 clients the order holds (0.99: 4.98 / 5.31 / 7.68 ms).
- **Rerank moves latency, not recall.** At `ef_search` 80 / 160 / 320 the 2x arm scores 0.954 / 0.981 / 0.994 against the plain expression index's 0.955 / 0.977 / 0.991 (two different builds) and takes 2.11 / 3.08 / 5.07 ms against 1.75 / 2.72 / 4.86. 5x and 10x need `ef_search` of at least 50 and 100 to return their candidates, so their cheapest points (1.93 and 3.90 ms) are already slower than plain FP16 at 0.95.
- **Native's default is the fastest point at 0.90 and 0.95** (1.27 ms, recall 0.953) and its probe sweep reaches 0.99 at 2.28 ms, 0.993 at 50 lists (2.54 ms), 0.997 at 100 (4.09 ms). At 16 clients: 2.13 ms at its default (8,319 QPS), 3.26 ms at 0.99.
- **The forced exact scans:** 324 ms (FP32, `PLAIN`) and 219 ms (FP16) at one client, 11 and 17 QPS at 16 clients.

## Unfiltered, k = 100

The same curves at `LIMIT 100`. pgvector arms from `ef_search` 100 (the rerank arms from `k x n`); native's 0.95 build along its probe count at pools of 1,000, 2,000 and 5,000 rows (its calibrated pool never reaches 0.90: best 0.804).

| System | Curve | c=1 p50 ms at recall 0.90 / 0.95 / 0.99 | c=4 | c=16 | Best recall |
|---|---|---|---|---|---|
| `pgvector_fp32_hnsw` | ef_search | <=3.38 / 4.42 / 11.04 | <=3.72 / 4.28 / 11.09 | <=5.11 / 5.78 / 14.28 | 0.996 |
| `pgvector_fp32_hnsw_m32` | ef_search | <=5.15 / <=5.15 / 8.40 | <=5.37 / <=5.37 / 8.73 | <=7.53 / <=7.53 / 11.03 | 1.000 |
| `pgvector_fp16_expression` | ef_search | <=3.07 / 3.79 / 8.40 | <=3.15 / 3.72 / 7.91 | <=4.38 / 4.85 / 10.10 | 0.996 |
| `pgvector_fp16_expression_m32` | ef_search | <=4.32 / <=4.32 / 6.60 | <=4.30 / <=4.30 / 6.63 | <=6.00 / <=6.00 / 8.35 | 1.000 |
| `pgvector_fp16_rerank_2x` | ef_search | <=5.68 / <=5.68 / 9.43 | <=5.86 / <=5.86 / 9.62 | <=8.10 / <=8.10 / 12.79 | 0.996 |
| `pgvector_fp16_rerank_5x` | ef_search | <=12.39 / <=12.39 / <=12.39 | <=12.99 / <=12.99 / <=12.99 | <=17.64 / <=17.64 / <=17.64 | 0.996 |
| `pgvector_fp16_rerank_5x_m32` | ef_search | <=19.72 / <=19.72 / <=19.72 | <=20.07 / <=20.07 / <=20.07 | <=26.19 / <=26.19 / <=26.19 | 1.000 |
| `pgvector_fp16_rerank_10x` | ef_search | <=27.57 / <=27.57 / <=27.57 | <=27.64 / <=27.64 / <=27.64 | <=37.39 / <=37.39 / <=37.39 | 0.996 |
| `pg_pinecone_native_t095` | probes swept, calibrated pool | - / - / - | - / - / - | - / - / - | 0.804 |
| `pg_pinecone_native_t095` | probes swept, pool 1000 | <=2.54 / 2.72 / 4.03 | <=2.79 / 2.86 / 4.25 | <=3.81 / 3.98 / 5.71 | 1.000 |
| `pg_pinecone_native_t095` | probes swept, pool 2000 | <=3.15 / 3.50 / 5.05 | <=3.61 / 3.86 / 5.13 | <=4.79 / 5.08 / 7.03 | 1.000 |
| `pg_pinecone_native_t095` | probes swept, pool 5000 | <=5.14 / 5.60 / 7.49 | <=5.52 / 5.98 / 7.72 | <=7.52 / 8.07 / 10.46 | 1.000 |
| `pgvector_fp16_heap` | ef_search | <=2.30 / 3.18 / 7.63 | <=2.37 / 3.05 / 7.88 | <=3.38 / 3.99 / 10.20 | 0.996 |
| `pgvector_fp16_heap_m32` | ef_search | <=3.29 / <=3.29 / 5.77 | <=3.44 / <=3.44 / 5.58 | <=4.96 / <=4.96 / 7.25 | 1.000 |

- **Native with a 1,000-row pool is fastest at 0.95 and 0.99:** 2.72 and 4.03 ms against 3.18 ms (FP16 heap, m16) and 5.77 ms (FP16 heap, m32); at 16 clients 3.98 and 5.71 ms against 3.99 and 7.25.
- **The rerank arms are the slowest hnsw arms at k = 100:** 5.7 ms (2x, ef 200), 12.4 ms (5x, ef 500), 27.6 ms (10x, ef 1,000) at one client, for the same recall as plain FP16 at ef 100 to 160.

## Native calibration

Each build's `pinecone.index_calibration()` (stamped in the results), held-out recall and p50 at its defaults, recall@100 at the calibrated pool and at a 1,000-row pool (`rescore_multiplier` 10).

| target_recall | nprobe (of 1,000) | Pool | Build sample recall@10 | Held-out recall@10 | c=1 p50 ms | recall@100, calibrated pool | recall@100, pool 1,000 |
|---|---|---|---|---|---|---|---|
| 0.90 | 8 | 100 | 0.950 | 0.921 | 1.06 | 0.754 | 0.880 |
| 0.95 | 12 | 100 | 0.972 | 0.953 | 1.27 | 0.777 | 0.926 |
| 0.98 | 25 | 100 | 0.992 | 0.983 | 1.83 | 0.795 | 0.970 |
| 0.99 | 37 | 100 | 0.996 | 0.990 | 2.28 | 0.799 | 0.982 |

- **Every build met its target on the held-out queries**, the 0.99 build by 0.0002 (0.9902), the others by 0.003 to 0.021; every build's sample read 0.006 to 0.029 above what the held-out queries got (the sample queries are table rows).
- **Every target chose the 100-row pool and raised the probe count instead.** The calibration's cost model charges a rescored row 100 codes, so on this corpus the cheapest pair that reaches the target is the smallest pool and more lists.
- **Calibration targets recall@10; a `LIMIT 100` needs a pinned pool** (`ALTER INDEX .. SET (rescore_multiplier = 10)` and a `REINDEX`, or the session override). Even then the 0.98 and 0.99 builds stop at 0.970 and 0.982: at 25 and 37 lists the true top 100 is not in the probed lists; more probes (the 0.95 build's grid) reach 0.99.
- **The same 50k synthetic corpus calibrated differently on the box and on the laptop** (`smoke-halfvec-native`: 75 against 50 lists at 0.90, a 1,000 against a 2,000-row pool at 0.99); the cause was not looked into.

## FP16 rounding

`recall_halfvec_truth` is recall against an exact search over the FP16-rounded corpus and queries; recall is against the FP32 truth.

- On every point of every system the two differ by at most 0.0002 (one neighbour in 5,000 at k = 10), in both directions.
- The forced exact scan of the FP16 heap scores 1.000 against both truths, at k = 10 (the exact arms run k = 10 only, `max_queries` 200).
- So on these 768-d embeddings FP16 rounding never moves a true neighbour out of the top 10. That is a property of this corpus and query set; other embeddings (smaller norms, more ties, other dimensions) may differ.

## Against 2026-10-14

The same data, queries, instance type, AMI and tuned settings as the 2026-10-14 0.9.0 run (leg A), on a different instance.

| System | 2026-10-14 | 2026-10-16 |
|---|---|---|
| pgvector FP32 hnsw m16, p50 at ef 10 / 20 / 40 / 80 / 160 / 320 (recall) | 0.92 / 1.01 / 1.43 / 2.00 / 3.29 / 5.42 ms (0.700 / 0.807 / 0.900 / 0.955 / 0.975 / 0.990) | 0.94 / 1.23 / 1.70 / 2.50 / 4.17 / 6.93 ms (0.694 / 0.809 / 0.888 / 0.947 / 0.976 / 0.991) |
| pgvector FP32 hnsw m16 index, build | 4,094 MB, 69.6 s | 4,094 MB, 79.1 s |
| Forced exact FP32 scan, c = 1 | 1,120 ms (default storage, TOASTed) | 324 ms (`STORAGE PLAIN`) |
| pg_pinecone native, p50 at recall 0.90 / 0.95 / 0.99 | 0.9.0: <=6.09 / 6.32 / 7.57 ms, 147 MB, 235 s | `main`, 0.95 build: <=1.27 / <=1.27 / 2.28 ms, 136 MB, 45 s |

- **hnsw is 20 to 28% slower from ef 20 up and equal at ef 10**, with the same index size and recall within 0.012. Both runs had the box to themselves. The heap layout differs (`PLAIN` here, so a returned row's heap tuple is on its own 8 KB page), and so does the physical host; neither was isolated.
- **The exact scan is 3.5 times faster with `PLAIN`**: it reads every vector, and with the default storage every read is a TOAST fetch.
- **Native is 5 times faster at 0.95**, but both the engine (calibrated 12 lists against the 0.9.0 constant of 127, RaBitQ codes, the blocked scan) and the storage changed; this run does not separate them.

## Notes

- **Build-to-build variance.** One build per arm. The three m16 FP16 builds (expression, 2x rerank, heap) differ by up to 0.017 recall@10 at the same `ef_search` (at ef 40; 0.004 or less from ef 80 up, 0.002 or less at k = 100); read differences of that size as noise.
- **The smoke.** `smoke-halfvec-native` (50k x 768d, every arm kind of this config, native from the same build): 76/76 valid, 131 s, no native point on a sequential scan.

## Deviations

- **D1. The box's state and logs were not kept.** The copy before teardown brought back the results files and empty `/data/state` and `/data/logs`: there is no `settings-2026-10-16-halfvec.json`, and the `pip freeze`, the shard checksums, the Rust toolchain version and the `sar` samples are missing (`box/BOX-2026-10-16-halfvec.md`, Gaps).
- **D2. pg_pinecone is not a release.** `main` at `7763b35`, built on the box; its numbers describe 0.10.0 as of that commit.
- **D3. `STORAGE PLAIN` is a choice of this config, not pgvector's default.** pgvector's FP32 arms under default storage would pay TOAST reads only where they read the vector (rerank, exact); the hnsw search itself reads the index.

## Not verified

- Filtered queries (the config is unfiltered), other corpora, dimensions and metrics (L2 only), PostgreSQL 18.
- Run-to-run and build-to-build spread: one run, one build per arm.
- FP16 on the native engine: pg_pinecone has no halfvec column support; its arms use the FP32 heap.
- The cause of the cross-session hnsw difference (above).

## Timeline (2026-10-16, UTC)

| Time | Step |
|---|---|
| 13:43 | instance launched (`provision.sh`), ready 13:44 |
| 13:45 to 13:57 | harness at `204592c` and `main`'s source at `7763b35` uploaded; 0.9.0 tarball installed, `setup-box.sh` |
| 13:57 | `hv_install` (pg_pinecone from `main`, 13:57:20 to 13:59:37) and `data` (wiki_dpr, 13:57:21 to 13:58:06) |
| 14:00 | `hv_prep` (k-means, FP32 and FP16-rounded ground truth) and `hv_smoke` (the rehearsal smoke and a 2.5k-row pgvector-only plumbing check, 14:00:18 to 14:02:43) |
| 14:03 | `hv_1m` started alone |
| 15:38 | `hv_1m` done at 15:38:27 (95 min), results copied, instance terminated 15:38:45 |

## Cost

| | |
|---|---|
| AWS | `r6id.4xlarge` $1.2096 per hour, 1.92 h from launch to termination: $2.32 for the instance (EBS a few cents more) |
| Pinecone | none |

## Cleanup

Instance terminated 15:38:45 UTC; key pair and security group deleted, private key removed; no tagged instance or volume left (`teardown.sh`).

## Every point, k = 10

`Rows rescored` is native's mean per query (`pinecone.last_scan_stats()`).

| System | Setting | recall@10 | c=1 p50 / p95 / p99 ms | Rows rescored | c=16 p50 ms (QPS) |
|---|---|---|---|---|---|
| `pgvector_fp32_hnsw` | ef 10 | 0.694 | 0.94 / 1.68 / 1.93 | - | 1.78 (15,815/s) |
| `pgvector_fp32_hnsw` | ef 20 | 0.809 | 1.23 / 2.04 / 2.47 | - | 1.80 (11,517/s) |
| `pgvector_fp32_hnsw` | ef 40 | 0.888 | 1.70 / 2.58 / 2.86 | - | 2.39 (7,822/s) |
| `pgvector_fp32_hnsw` | ef 80 | 0.947 | 2.50 / 3.70 / 4.09 | - | 3.40 (5,056/s) |
| `pgvector_fp32_hnsw` | ef 160 | 0.976 | 4.17 / 6.32 / 6.79 | - | 5.27 (3,103/s) |
| `pgvector_fp32_hnsw` | ef 320 | 0.991 | 6.93 / 10.54 / 11.71 | - | 8.82 (1,806/s) |
| `pgvector_fp32_hnsw` | ef 640 | 0.996 | 11.89 / 18.22 / 20.75 | - | 15.99 (1,015/s) |
| `pgvector_fp32_hnsw_m32` | ef 10 | 0.834 | 1.57 / 2.62 / 3.12 | - | 2.47 (9,864/s) |
| `pgvector_fp32_hnsw_m32` | ef 20 | 0.916 | 1.85 / 2.70 / 3.28 | - | 2.81 (7,091/s) |
| `pgvector_fp32_hnsw_m32` | ef 40 | 0.961 | 2.70 / 4.09 / 4.66 | - | 3.54 (4,980/s) |
| `pgvector_fp32_hnsw_m32` | ef 80 | 0.981 | 4.00 / 6.14 / 6.89 | - | 5.09 (3,160/s) |
| `pgvector_fp32_hnsw_m32` | ef 160 | 0.992 | 6.46 / 10.51 / 11.73 | - | 8.34 (1,917/s) |
| `pgvector_fp32_hnsw_m32` | ef 320 | 0.998 | 11.01 / 18.59 / 20.37 | - | 14.31 (1,123/s) |
| `pgvector_fp32_hnsw_m32` | ef 640 | 1.000 | 18.95 / 31.73 / 37.42 | - | 24.96 (628/s) |
| `pgvector_fp32_hnsw_m32` | ef 1000 | 1.000 | 27.80 / 45.80 / 53.25 | - | 36.18 (441/s) |
| `pgvector_fp16_expression` | ef 10 | 0.710 | 0.73 / 1.25 / 1.60 | - | 1.25 (19,116/s) |
| `pgvector_fp16_expression` | ef 20 | 0.812 | 0.88 / 1.38 / 1.81 | - | 1.38 (13,822/s) |
| `pgvector_fp16_expression` | ef 40 | 0.898 | 1.32 / 1.94 / 2.46 | - | 1.62 (10,874/s) |
| `pgvector_fp16_expression` | ef 80 | 0.955 | 1.75 / 2.62 / 2.93 | - | 2.49 (6,946/s) |
| `pgvector_fp16_expression` | ef 160 | 0.977 | 2.72 / 4.11 / 4.52 | - | 3.67 (4,174/s) |
| `pgvector_fp16_expression` | ef 320 | 0.991 | 4.86 / 7.39 / 8.39 | - | 6.18 (2,485/s) |
| `pgvector_fp16_expression` | ef 640 | 0.997 | 8.57 / 13.21 / 15.12 | - | 10.95 (1,443/s) |
| `pgvector_fp16_expression` | ef 1000 | 0.998 | 13.00 / 20.32 / 23.13 | - | 16.15 (968/s) |
| `pgvector_fp16_expression_m32` | ef 10 | 0.837 | 1.06 / 1.75 / 2.22 | - | 1.74 (13,485/s) |
| `pgvector_fp16_expression_m32` | ef 20 | 0.913 | 1.26 / 1.91 / 2.36 | - | 1.88 (10,079/s) |
| `pgvector_fp16_expression_m32` | ef 40 | 0.959 | 1.86 / 2.85 / 3.18 | - | 2.46 (6,791/s) |
| `pgvector_fp16_expression_m32` | ef 80 | 0.983 | 2.86 / 4.48 / 5.07 | - | 3.75 (4,328/s) |
| `pgvector_fp16_expression_m32` | ef 160 | 0.993 | 4.59 / 7.68 / 8.37 | - | 5.85 (2,669/s) |
| `pgvector_fp16_expression_m32` | ef 320 | 0.999 | 7.75 / 12.94 / 14.97 | - | 10.04 (1,542/s) |
| `pgvector_fp16_expression_m32` | ef 640 | 1.000 | 13.26 / 22.02 / 26.16 | - | 18.37 (889/s) |
| `pgvector_fp16_expression_m32` | ef 1000 | 1.000 | 19.11 / 32.07 / 38.15 | - | 24.78 (625/s) |
| `pgvector_fp16_rerank_2x` | ef 20 | 0.811 | 1.11 / 1.76 / 2.15 | - | 1.96 (11,735/s) |
| `pgvector_fp16_rerank_2x` | ef 40 | 0.900 | 1.41 / 1.99 / 2.38 | - | 2.12 (8,613/s) |
| `pgvector_fp16_rerank_2x` | ef 80 | 0.954 | 2.11 / 3.05 / 3.35 | - | 2.87 (6,128/s) |
| `pgvector_fp16_rerank_2x` | ef 160 | 0.981 | 3.08 / 4.60 / 4.91 | - | 4.06 (3,864/s) |
| `pgvector_fp16_rerank_2x` | ef 320 | 0.994 | 5.07 / 7.61 / 8.42 | - | 6.50 (2,360/s) |
| `pgvector_fp16_rerank_2x` | ef 640 | 0.997 | 8.73 / 13.40 / 14.94 | - | 11.56 (1,355/s) |
| `pgvector_fp16_rerank_2x` | ef 1000 | 0.999 | 13.08 / 19.86 / 23.24 | - | 17.09 (931/s) |
| `pgvector_fp16_rerank_5x` | ef 50 | 0.917 | 1.93 / 2.75 / 3.50 | - | 3.10 (6,537/s) |
| `pgvector_fp16_rerank_5x` | ef 80 | 0.944 | 2.48 / 3.44 / 3.81 | - | 3.32 (5,188/s) |
| `pgvector_fp16_rerank_5x` | ef 160 | 0.972 | 3.55 / 5.13 / 5.63 | - | 4.55 (3,542/s) |
| `pgvector_fp16_rerank_5x` | ef 320 | 0.990 | 5.64 / 8.23 / 9.29 | - | 6.88 (2,235/s) |
| `pgvector_fp16_rerank_5x` | ef 640 | 0.996 | 9.18 / 14.04 / 15.85 | - | 11.67 (1,333/s) |
| `pgvector_fp16_rerank_5x` | ef 1000 | 0.998 | 13.47 / 20.37 / 23.54 | - | 17.26 (908/s) |
| `pgvector_fp16_rerank_5x_m32` | ef 50 | 0.971 | 3.48 / 4.98 / 5.92 | - | 4.86 (3,931/s) |
| `pgvector_fp16_rerank_5x_m32` | ef 80 | 0.985 | 4.30 / 6.31 / 6.97 | - | 5.61 (3,009/s) |
| `pgvector_fp16_rerank_5x_m32` | ef 160 | 0.992 | 6.40 / 9.90 / 11.28 | - | 8.12 (1,958/s) |
| `pgvector_fp16_rerank_5x_m32` | ef 320 | 0.998 | 10.28 / 17.06 / 18.77 | - | 13.12 (1,189/s) |
| `pgvector_fp16_rerank_5x_m32` | ef 640 | 1.000 | 17.53 / 28.49 / 33.83 | - | 23.06 (690/s) |
| `pgvector_fp16_rerank_5x_m32` | ef 1000 | 1.000 | 25.20 / 41.39 / 49.94 | - | 32.53 (493/s) |
| `pgvector_fp16_rerank_10x` | ef 100 | 0.960 | 3.90 / 5.38 / 5.98 | - | 5.40 (3,400/s) |
| `pgvector_fp16_rerank_10x` | ef 160 | 0.978 | 4.82 / 6.79 / 7.44 | - | 6.19 (2,609/s) |
| `pgvector_fp16_rerank_10x` | ef 320 | 0.992 | 7.42 / 10.69 / 11.80 | - | 9.38 (1,689/s) |
| `pgvector_fp16_rerank_10x` | ef 640 | 0.998 | 12.35 / 18.61 / 20.76 | - | 15.39 (1,016/s) |
| `pgvector_fp16_rerank_10x` | ef 1000 | 0.998 | 17.72 / 26.55 / 30.14 | - | 22.59 (713/s) |
| `pgvector_fp16_heap` | ef 10 | 0.711 | 0.69 / 1.24 / 1.42 | - | 1.24 (21,084/s) |
| `pgvector_fp16_heap` | ef 20 | 0.804 | 0.87 / 1.37 / 1.59 | - | 1.26 (16,267/s) |
| `pgvector_fp16_heap` | ef 40 | 0.883 | 1.20 / 1.83 / 2.25 | - | 1.68 (11,297/s) |
| `pgvector_fp16_heap` | ef 80 | 0.953 | 1.75 / 2.62 / 2.85 | - | 2.30 (7,042/s) |
| `pgvector_fp16_heap` | ef 160 | 0.979 | 2.93 / 4.46 / 4.92 | - | 3.61 (4,470/s) |
| `pgvector_fp16_heap` | ef 320 | 0.989 | 4.83 / 7.49 / 8.20 | - | 6.10 (2,556/s) |
| `pgvector_fp16_heap` | ef 640 | 0.996 | 8.25 / 12.84 / 14.53 | - | 10.81 (1,443/s) |
| `pgvector_fp16_heap` | ef 1000 | 0.998 | 12.57 / 19.16 / 22.30 | - | 16.62 (928/s) |
| `pgvector_fp16_heap_m32` | ef 10 | 0.848 | 0.99 / 1.77 / 2.27 | - | 1.70 (14,641/s) |
| `pgvector_fp16_heap_m32` | ef 20 | 0.918 | 1.23 / 1.93 / 2.38 | - | 1.75 (10,615/s) |
| `pgvector_fp16_heap_m32` | ef 40 | 0.964 | 1.71 / 2.64 / 2.91 | - | 2.42 (7,229/s) |
| `pgvector_fp16_heap_m32` | ef 80 | 0.983 | 2.89 / 4.58 / 5.12 | - | 3.57 (4,477/s) |
| `pgvector_fp16_heap_m32` | ef 160 | 0.992 | 4.48 / 7.57 / 8.40 | - | 5.46 (2,726/s) |
| `pgvector_fp16_heap_m32` | ef 320 | 0.999 | 8.20 / 13.54 / 15.17 | - | 9.55 (1,590/s) |
| `pgvector_fp16_heap_m32` | ef 640 | 1.000 | 13.70 / 22.90 / 26.89 | - | 17.51 (889/s) |
| `pgvector_fp16_heap_m32` | ef 1000 | 1.000 | 19.05 / 32.72 / 39.58 | - | 24.62 (617/s) |
| `pg_pinecone_native_t090` | pool calibrated, probes 8 | 0.921 | 1.06 / 1.24 / 1.37 | 100 | 1.94 (9,304/s) |
| `pg_pinecone_native_t095` | pool calibrated, probes 12 | 0.953 | 1.27 / 1.50 / 1.64 | 100 | 2.13 (8,319/s) |
| `pg_pinecone_native_t095` | pool 200, probes 12 | 0.953 | 1.39 / 1.75 / 1.87 | 152 | 2.19 (7,736/s) |
| `pg_pinecone_native_t095` | pool 500, probes 12 | 0.953 | 1.82 / 2.44 / 2.80 | 319 | 2.65 (6,229/s) |
| `pg_pinecone_native_t095` | pool 1000, probes 12 | 0.953 | 2.31 / 3.29 / 3.70 | 553 | 3.27 (5,032/s) |
| `pg_pinecone_native_t095` | pool 2000, probes 12 | 0.953 | 3.06 / 4.31 / 4.84 | 933 | 4.41 (3,730/s) |
| `pg_pinecone_native_t095` | pool calibrated, probes 25 | 0.983 | 1.73 / 1.95 / 2.04 | 100 | 2.56 (6,294/s) |
| `pg_pinecone_native_t095` | pool 200, probes 25 | 0.983 | 1.88 / 2.20 / 2.31 | 155 | 2.67 (5,931/s) |
| `pg_pinecone_native_t095` | pool 500, probes 25 | 0.983 | 2.25 / 2.72 / 2.83 | 333 | 3.19 (5,020/s) |
| `pg_pinecone_native_t095` | pool 1000, probes 25 | 0.983 | 2.82 / 3.69 / 3.95 | 593 | 3.99 (4,037/s) |
| `pg_pinecone_native_t095` | pool 2000, probes 25 | 0.983 | 3.70 / 5.27 / 5.67 | 1,030 | 5.22 (3,047/s) |
| `pg_pinecone_native_t095` | pool calibrated, probes 50 | 0.993 | 2.54 / 2.79 / 2.89 | 100 | 3.59 (4,450/s) |
| `pg_pinecone_native_t095` | pool 200, probes 50 | 0.994 | 2.71 / 3.05 / 3.22 | 156 | 3.74 (4,264/s) |
| `pg_pinecone_native_t095` | pool 500, probes 50 | 0.994 | 3.22 / 3.76 / 3.90 | 339 | 4.33 (3,716/s) |
| `pg_pinecone_native_t095` | pool 1000, probes 50 | 0.994 | 3.57 / 4.53 / 4.69 | 614 | 4.98 (3,145/s) |
| `pg_pinecone_native_t095` | pool 2000, probes 50 | 0.994 | 4.37 / 6.28 / 6.63 | 1,087 | 6.33 (2,453/s) |
| `pg_pinecone_native_t095` | pool calibrated, probes 100 | 0.997 | 4.09 / 4.40 / 4.56 | 100 | 5.70 (2,851/s) |
| `pg_pinecone_native_t095` | pool 200, probes 100 | 0.998 | 4.20 / 4.59 / 4.70 | 157 | 5.79 (2,732/s) |
| `pg_pinecone_native_t095` | pool 500, probes 100 | 0.998 | 4.55 / 5.12 / 5.27 | 342 | 6.23 (2,521/s) |
| `pg_pinecone_native_t095` | pool 1000, probes 100 | 0.998 | 5.07 / 6.02 / 6.22 | 621 | 7.36 (2,188/s) |
| `pg_pinecone_native_t095` | pool 2000, probes 100 | 0.998 | 6.26 / 8.08 / 8.50 | 1,112 | 8.74 (1,810/s) |
| `pg_pinecone_native_t095` | pool calibrated, probes 200 | 0.999 | 7.24 / 7.76 / 8.05 | 100 | 10.00 (1,642/s) |
| `pg_pinecone_native_t095` | pool 200, probes 200 | 1.000 | 7.23 / 7.75 / 7.98 | 157 | 10.11 (1,596/s) |
| `pg_pinecone_native_t095` | pool 500, probes 200 | 1.000 | 7.85 / 8.52 / 8.76 | 343 | 10.68 (1,505/s) |
| `pg_pinecone_native_t095` | pool 1000, probes 200 | 1.000 | 8.24 / 9.31 / 9.53 | 625 | 11.53 (1,388/s) |
| `pg_pinecone_native_t095` | pool 2000, probes 200 | 1.000 | 9.55 / 11.65 / 12.01 | 1,125 | 12.90 (1,227/s) |
| `pg_pinecone_native_t095` | pool calibrated, probes 400 | 0.999 | 13.36 / 14.01 / 14.23 | 100 | 18.44 (883/s) |
| `pg_pinecone_native_t095` | pool 200, probes 400 | 1.000 | 13.57 / 14.24 / 14.64 | 157 | 18.55 (868/s) |
| `pg_pinecone_native_t095` | pool 500, probes 400 | 1.000 | 14.30 / 15.07 / 15.36 | 343 | 19.19 (848/s) |
| `pg_pinecone_native_t095` | pool 1000, probes 400 | 1.000 | 14.59 / 15.66 / 16.09 | 627 | 19.87 (809/s) |
| `pg_pinecone_native_t095` | pool 2000, probes 400 | 1.000 | 15.28 / 17.31 / 17.89 | 1,130 | 21.44 (741/s) |
| `pg_pinecone_native_t098` | pool calibrated, probes 25 | 0.983 | 1.83 / 2.10 / 2.22 | 100 | 2.85 (6,321/s) |
| `pg_pinecone_native_t099` | pool calibrated, probes 37 | 0.990 | 2.28 / 2.59 / 2.72 | 100 | 3.36 (5,287/s) |
| `fp32_exact_seqscan` | forced seqscan | 1.000 | 323.93 / 363.36 / 371.18 | - | 1,102.98 (11/s) |
| `fp16_exact_seqscan` | forced seqscan | 1.000 | 219.16 / 225.44 / 226.61 | - | 723.96 (17/s) |

## Every point, k = 100

| System | Setting | recall@100 | c=1 p50 / p95 / p99 ms | Rows rescored | c=16 p50 ms (QPS) |
|---|---|---|---|---|---|
| `pgvector_fp32_hnsw` | ef 100 | 0.913 | 3.38 / 5.30 / 6.43 | - | 5.11 (4,100/s) |
| `pgvector_fp32_hnsw` | ef 160 | 0.950 | 4.42 / 6.58 / 7.18 | - | 5.77 (2,959/s) |
| `pgvector_fp32_hnsw` | ef 320 | 0.982 | 7.11 / 10.70 / 11.70 | - | 9.25 (1,758/s) |
| `pgvector_fp32_hnsw` | ef 640 | 0.993 | 12.39 / 19.10 / 21.40 | - | 16.01 (993/s) |
| `pgvector_fp32_hnsw` | ef 1000 | 0.996 | 18.23 / 27.58 / 31.83 | - | 24.19 (676/s) |
| `pgvector_fp32_hnsw_m32` | ef 100 | 0.968 | 5.15 / 8.17 / 10.03 | - | 7.53 (2,591/s) |
| `pgvector_fp32_hnsw_m32` | ef 160 | 0.986 | 6.76 / 11.04 / 12.45 | - | 8.96 (1,847/s) |
| `pgvector_fp32_hnsw_m32` | ef 320 | 0.997 | 11.34 / 18.63 / 21.00 | - | 14.72 (1,082/s) |
| `pgvector_fp32_hnsw_m32` | ef 640 | 0.999 | 19.83 / 32.76 / 39.31 | - | 25.69 (628/s) |
| `pgvector_fp32_hnsw_m32` | ef 1000 | 1.000 | 27.78 / 46.00 / 54.58 | - | 35.25 (449/s) |
| `pgvector_fp16_expression` | ef 100 | 0.912 | 3.07 / 4.35 / 5.10 | - | 4.38 (4,390/s) |
| `pgvector_fp16_expression` | ef 160 | 0.950 | 3.80 / 5.40 / 5.85 | - | 4.86 (3,403/s) |
| `pgvector_fp16_expression` | ef 320 | 0.981 | 5.78 / 8.57 / 9.20 | - | 7.04 (2,215/s) |
| `pgvector_fp16_expression` | ef 640 | 0.994 | 9.60 / 14.49 / 16.38 | - | 11.51 (1,295/s) |
| `pgvector_fp16_expression` | ef 1000 | 0.996 | 13.99 / 20.95 / 24.96 | - | 17.88 (907/s) |
| `pgvector_fp16_expression_m32` | ef 100 | 0.969 | 4.32 / 6.57 / 7.58 | - | 6.00 (3,060/s) |
| `pgvector_fp16_expression_m32` | ef 160 | 0.986 | 5.57 / 8.80 / 9.56 | - | 7.23 (2,296/s) |
| `pgvector_fp16_expression_m32` | ef 320 | 0.997 | 8.55 / 13.99 / 15.69 | - | 10.48 (1,426/s) |
| `pgvector_fp16_expression_m32` | ef 640 | 0.999 | 14.87 / 24.62 / 28.85 | - | 18.52 (852/s) |
| `pgvector_fp16_expression_m32` | ef 1000 | 1.000 | 20.49 / 34.01 / 39.66 | - | 26.98 (595/s) |
| `pgvector_fp16_rerank_2x` | ef 200 | 0.967 | 5.68 / 7.77 / 8.54 | - | 8.10 (2,208/s) |
| `pgvector_fp16_rerank_2x` | ef 320 | 0.982 | 6.88 / 9.63 / 10.45 | - | 9.32 (1,749/s) |
| `pgvector_fp16_rerank_2x` | ef 640 | 0.993 | 10.37 / 14.90 / 16.74 | - | 14.08 (1,132/s) |
| `pgvector_fp16_rerank_2x` | ef 1000 | 0.996 | 14.99 / 21.64 / 24.53 | - | 19.80 (807/s) |
| `pgvector_fp16_rerank_5x` | ef 500 | 0.991 | 12.39 / 16.55 / 17.80 | - | 17.64 (989/s) |
| `pgvector_fp16_rerank_5x` | ef 640 | 0.993 | 13.67 / 18.49 / 20.21 | - | 18.55 (889/s) |
| `pgvector_fp16_rerank_5x` | ef 1000 | 0.996 | 17.72 / 24.74 / 27.54 | - | 22.95 (711/s) |
| `pgvector_fp16_rerank_5x_m32` | ef 500 | 0.999 | 19.72 / 29.26 / 33.87 | - | 26.19 (628/s) |
| `pgvector_fp16_rerank_5x_m32` | ef 640 | 0.999 | 22.52 / 34.07 / 39.55 | - | 29.88 (542/s) |
| `pgvector_fp16_rerank_5x_m32` | ef 1000 | 1.000 | 31.05 / 47.73 / 55.96 | - | 40.38 (399/s) |
| `pgvector_fp16_rerank_10x` | ef 1000 | 0.996 | 27.57 / 36.59 / 40.67 | - | 37.39 (451/s) |
| `pgvector_fp16_heap` | ef 100 | 0.912 | 2.30 / 3.50 / 4.06 | - | 3.38 (5,761/s) |
| `pgvector_fp16_heap` | ef 160 | 0.953 | 3.24 / 4.81 / 5.26 | - | 4.03 (4,234/s) |
| `pgvector_fp16_heap` | ef 320 | 0.981 | 4.85 / 7.55 / 8.23 | - | 6.19 (2,502/s) |
| `pgvector_fp16_heap` | ef 640 | 0.993 | 8.39 / 13.09 / 14.48 | - | 11.29 (1,404/s) |
| `pgvector_fp16_heap` | ef 1000 | 0.996 | 12.88 / 19.86 / 23.29 | - | 16.57 (937/s) |
| `pgvector_fp16_heap_m32` | ef 100 | 0.969 | 3.29 / 5.35 / 6.24 | - | 4.96 (3,753/s) |
| `pgvector_fp16_heap_m32` | ef 160 | 0.987 | 4.84 / 7.88 / 8.68 | - | 6.01 (2,634/s) |
| `pgvector_fp16_heap_m32` | ef 320 | 0.997 | 7.70 / 12.73 / 14.71 | - | 9.82 (1,568/s) |
| `pgvector_fp16_heap_m32` | ef 640 | 0.999 | 13.55 / 22.30 / 26.06 | - | 17.56 (877/s) |
| `pgvector_fp16_heap_m32` | ef 1000 | 1.000 | 19.80 / 33.03 / 39.18 | - | 25.68 (616/s) |
| `pg_pinecone_native_t090` | pool calibrated, probes 8 | 0.754 | 1.21 / 1.42 / 1.63 | 100 | 2.09 (8,663/s) |
| `pg_pinecone_native_t090` | pool 1000, probes 8 | 0.880 | 2.30 / 3.26 / 3.64 | 519 | 3.49 (5,098/s) |
| `pg_pinecone_native_t095` | pool calibrated, probes 12 | 0.777 | 1.34 / 1.56 / 1.68 | 100 | 2.28 (7,788/s) |
| `pg_pinecone_native_t095` | pool 1000, probes 12 | 0.926 | 2.54 / 3.56 / 4.13 | 553 | 3.81 (4,660/s) |
| `pg_pinecone_native_t095` | pool 2000, probes 12 | 0.926 | 3.15 / 4.40 / 4.85 | 933 | 4.79 (3,563/s) |
| `pg_pinecone_native_t095` | pool 5000, probes 12 | 0.926 | 5.14 / 6.82 / 7.38 | 1,832 | 7.52 (2,263/s) |
| `pg_pinecone_native_t095` | pool calibrated, probes 25 | 0.795 | 1.81 / 2.04 / 2.16 | 100 | 2.59 (5,940/s) |
| `pg_pinecone_native_t095` | pool 1000, probes 25 | 0.970 | 2.88 / 3.78 / 4.06 | 593 | 4.12 (3,789/s) |
| `pg_pinecone_native_t095` | pool 2000, probes 25 | 0.970 | 3.78 / 5.26 / 5.80 | 1,030 | 5.33 (2,981/s) |
| `pg_pinecone_native_t095` | pool 5000, probes 25 | 0.970 | 5.99 / 8.44 / 9.24 | 2,083 | 8.54 (1,897/s) |
| `pg_pinecone_native_t095` | pool calibrated, probes 50 | 0.801 | 2.61 / 2.88 / 3.22 | 100 | 3.79 (4,245/s) |
| `pg_pinecone_native_t095` | pool 1000, probes 50 | 0.988 | 3.71 / 4.66 / 4.96 | 614 | 5.23 (2,977/s) |
| `pg_pinecone_native_t095` | pool 2000, probes 50 | 0.988 | 4.74 / 6.53 / 6.98 | 1,087 | 6.57 (2,393/s) |
| `pg_pinecone_native_t095` | pool 5000, probes 50 | 0.988 | 7.11 / 10.26 / 11.57 | 2,264 | 9.94 (1,576/s) |
| `pg_pinecone_native_t095` | pool calibrated, probes 100 | 0.804 | 4.11 / 4.46 / 4.74 | 100 | 5.85 (2,722/s) |
| `pg_pinecone_native_t095` | pool 1000, probes 100 | 0.996 | 5.24 / 6.28 / 6.58 | 621 | 7.56 (2,108/s) |
| `pg_pinecone_native_t095` | pool 2000, probes 100 | 0.996 | 6.24 / 8.10 / 8.50 | 1,112 | 8.79 (1,785/s) |
| `pg_pinecone_native_t095` | pool 5000, probes 100 | 0.996 | 8.92 / 12.83 / 14.39 | 2,368 | 12.44 (1,254/s) |
| `pg_pinecone_native_t095` | pool calibrated, probes 200 | 0.804 | 7.20 / 7.78 / 8.08 | 100 | 10.13 (1,575/s) |
| `pg_pinecone_native_t095` | pool 1000, probes 200 | 0.999 | 8.55 / 9.61 / 9.84 | 625 | 11.67 (1,371/s) |
| `pg_pinecone_native_t095` | pool 2000, probes 200 | 0.999 | 9.27 / 11.25 / 11.67 | 1,125 | 13.14 (1,206/s) |
| `pg_pinecone_native_t095` | pool 5000, probes 200 | 0.999 | 12.25 / 16.69 / 18.01 | 2,425 | 16.73 (928/s) |
| `pg_pinecone_native_t095` | pool calibrated, probes 400 | 0.804 | 13.38 / 14.06 / 14.39 | 100 | 18.86 (845/s) |
| `pg_pinecone_native_t095` | pool 1000, probes 400 | 1.000 | 14.82 / 15.98 / 16.42 | 627 | 20.35 (790/s) |
| `pg_pinecone_native_t095` | pool 2000, probes 400 | 1.000 | 15.84 / 17.80 / 18.32 | 1,130 | 21.65 (738/s) |
| `pg_pinecone_native_t095` | pool 5000, probes 400 | 1.000 | 18.24 / 22.91 / 24.63 | 2,450 | 25.63 (617/s) |
| `pg_pinecone_native_t098` | pool calibrated, probes 25 | 0.795 | 1.93 / 2.25 / 2.41 | 100 | 2.98 (5,880/s) |
| `pg_pinecone_native_t098` | pool 1000, probes 25 | 0.970 | 3.10 / 4.19 / 4.68 | 593 | 4.56 (3,848/s) |
| `pg_pinecone_native_t099` | pool calibrated, probes 37 | 0.799 | 2.30 / 2.64 / 2.92 | 100 | 3.44 (5,661/s) |
| `pg_pinecone_native_t099` | pool 1000, probes 37 | 0.982 | 2.50 / 3.50 / 3.90 | 607 | 4.39 (3,787/s) |
