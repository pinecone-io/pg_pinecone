# Vector benchmark: search under a 4 GB memory cap, pgvector hnsw (FP32, FP16) and pg_pinecone 0.10.0 (2026-10-23)

*One AWS box (`r6id.4xlarge`), PostgreSQL 17.11 as the transient unit of `box/overram-pg.sh` (cgroup v2, no swap, `shared_buffers` 1 GB), pgvector 0.8.7, **pg_pinecone 0.10.0 release tarball** (pg17), wiki_dpr 1M x 768d, L2, 500 held-out queries, unfiltered, k = 10, one client. `configs/overram_1m.yaml` run twice: uncapped and with `MemoryMax=4G` set after each index build (page cache counts against it), lifted after the sweep. Each sweep repeats its first point: the first pass refills the cache the cap emptied, the repeat is the one read. 30 of 30 points valid. Facts of the box: `box/BOX-2026-10-22-overram.md` (both sessions), `box/settings-2026-10-23-overram.json`. Rerun: `box/README.md`, "Over-RAM session".*

## Verdict

1. **FP32 hnsw slows down 2.8 to 3.8 times at recall 0.95 to 0.99:** its 3.9 GB index does not fit beside 1.2 GB of shared memory under 4 GB. During its capped sweep the unit hit the cap 7,557 times and re-read 1.24 M evicted pages; p50 at `ef_search` 80 / 160 / 320 went from 2.62 / 4.29 / 7.23 to 7.32 / 13.07 / 27.24 ms, p95 from 4.3 / 7.0 / 12.7 to 19.1 / 35.8 / 73.7 ms. At `ef_search` 40 (recall 0.89) the part of the graph it walks fits and it does not slow down (1.64 against 1.70 ms).
2. **FP16 hnsw does not slow down:** its 2.0 GB index fits. No cap hit during its sweep, 2,479 re-read pages, p50 within 4% of uncapped at every point.
3. **pg_pinecone does not slow down:** a 129 MB index; its rescore reads the vectors from the 4.1 GB TOAST table, 22,145 re-read pages over the sweep, no cap hit. p50 within 3% of uncapped at every point after the warm-up (recall 0.953 to 0.990, 1.19 to 1.82 ms).
4. **The cold pass is the 10x reading.** The first pass after the cap, before the cache refills, reads FP32 hnsw at 7.7 times its uncapped p50 (ef 40: 14.8 against 1.9 ms) and native at 1.9 times (2.9 against 1.5 ms); FP16 hnsw at 1.07. A workload whose working set is evicted repeatedly (other tables competing for memory) sits nearer this pass than the warm one; this run does not measure that.
5. **The first session (2026-10-22) agrees.** Without the repeated point it read FP32 hnsw at 3.2 / 3.4 / 4.4 times at `ef_search` 80 / 160 / 320 and 9.1 at the cold first point, FP16 hnsw and native unslowed past the first point. Its first attempt was OOM-killed by a cap lowered while the native build's backend still held 3.6 GB of anonymous memory; `overram-pg.sh cap` now waits for it to drain (0 to 1 s here).
6. **Cost:** AWS 0.30 h, about $0.40 (the 2026-10-22 session 0.47 h); no Pinecone.

## p50 ms (recall@10), one client, after the warm-up pass

| System | Knob | Uncapped | 4 GB cap | capped / uncapped |
|---|---|---|---|---|
| pgvector hnsw, FP32 | ef 40 | 1.70 (0.893) | 1.64 (0.887) | 0.96 |
| | ef 80 | 2.62 (0.953) | 7.32 (0.947) | 2.79 |
| | ef 160 | 4.29 (0.974) | 13.07 (0.977) | 3.05 |
| | ef 320 | 7.23 (0.990) | 27.24 (0.991) | 3.77 |
| pgvector hnsw, FP16 expression | ef 40 | 1.42 (0.901) | 1.37 (0.898) | 0.96 |
| | ef 80 | 2.09 (0.953) | 2.14 (0.953) | 1.02 |
| | ef 160 | 3.46 (0.975) | 3.53 (0.978) | 1.02 |
| | ef 320 | 5.95 (0.988) | 5.96 (0.990) | 1.00 |
| pg_pinecone native | defaults (13 lists) | 1.21 (0.958) | 1.19 (0.958) | 0.98 |
| | 12 lists | 1.18 (0.953) | 1.20 (0.953) | 1.02 |
| | 25 lists | 1.55 (0.983) | 1.50 (0.983) | 0.97 |
| | 37 lists | 1.83 (0.990) | 1.82 (0.990) | 1.00 |

The cold first pass (not in the table): FP32 hnsw ef 40 1.92 against 14.79 ms, FP16 1.67 against 1.79, native defaults 1.52 against 2.94 (p95 10.5). Each run builds its own hnsw graph (parallel builds are not deterministic), so hnsw recall differs by up to 0.006 between the columns; native's is identical.

## Memory during the capped sweeps

From the unit's `memory.stat` and `memory.events`, read by the hooks before and after each sweep (`build.hooks` in the results).

| System | Index MB | Cap hits during the sweep | Re-read pages (`workingset_refault_file`) |
|---|---|---|---|
| pgvector hnsw, FP32 | 3,904 | 7,557 | 1,238,463 |
| pgvector hnsw, FP16 expression | 1,952 | 0 | 2,479 |
| pg_pinecone native | 129 | 0 | 22,145 |

## Not verified

- Caps other than 4 GB, more clients, k = 100, filtered queries, `STORAGE PLAIN`, repeated eviction by a competing workload.
