# Vector benchmark: pgvector halfvec (FP16) in the filtered cells, against FP32 hnsw and pg_pinecone 0.10.0 (2026-10-22)

*One AWS box (`r6id.4xlarge`), PostgreSQL 18.6, pgvector 0.8.7, **pg_pinecone 0.10.0 release tarball** (pg18, sha256 `6c7589f7...cb09`), wiki_dpr 1M x 768d, L2, 500 held-out queries. `configs/halfvec_filtered_1m_0.10.yaml`: the 1% and 10% filtered cells (FP32 hnsw and native) with pgvector's FP16 expression index added, so the three systems share one session. 128 of 128 points valid, 30 min. Facts of the box: `box/BOX-2026-10-22-hvfilt.md`, `box/settings-2026-10-22-hvfilt.json`. Rerun: `box/README.md`, "Filtered halfvec session".*

## Verdict

1. **FP16 changes nothing in the filtered cells.** At the same `ef_search` with `relaxed_order` the FP16 index reads the recall of the FP32 one within 0.002 in every cell, and its p50 is 2 to 25% lower out of region.
2. **Neither hnsw reaches recall 0.95 out of region** at `ef_search` 40 to 160: FP32 tops out at 0.926 (1% key) and 0.920 (10% key), FP16 at 0.924 and 0.922. Native with the key pushed reads 0.981 and 1.000.
3. **Latency splits by key size, as in the 0.10.0 release run (another host):** at the 1% key native is about 7 times faster than either hnsw (8.3 against 56 to 58 ms); at the 10% key both hnsw are faster than native (5.1 / 5.9 against 12.3 ms) at recall 0.92 against 1.000.
4. **The FP32 and native rows reproduce the 0.10.0 release run** (same recipe, another host): hnsw 0.926 / 0.920 against 0.919 / 0.924, native 0.981 / 1.000 against 0.981 / 1.000.
5. **Cost:** AWS 0.96 h, about $1.20; no Pinecone.

## Filtered cells: out-region queries, one client

hnsw at `ef_search` 160, `relaxed_order`, `max_scan_tuples` 100000; native at `rescore_multiplier` 2, `iterative_scan` auto, cap 20000 (the rows of the 0.10.0 release run's table). Every query returns 10 rows; every point runs on the system's own index.

| Key | pgvector hnsw, FP16 expression | pgvector hnsw, FP32 | pg_pinecone native, key pushed | exact scan | 0.10.0 release run (hnsw / native) |
|---|---|---|---|---|---|
| 1% | 0.924 @ 55.7 ms | 0.926 @ 57.8 ms | 0.981 @ 8.3 ms | 1.000 @ 41.9 ms | 0.919 @ 55.8 / 0.981 @ 8.1 |
| 10% | 0.922 @ 5.1 ms | 0.920 @ 5.9 ms | 1.000 @ 12.3 ms | 1.000 @ 157.0 ms | 0.924 @ 4.5 / 1.000 @ 12.4 |

`ef_search` 40 / 80 / 160, recall @ p50 ms:

| Key | FP16 | FP32 |
|---|---|---|
| 1% | 0.918 @ 53.5, 0.921 @ 58.6, 0.924 @ 55.7 | 0.920 @ 64.2, 0.922 @ 67.3, 0.926 @ 57.8 |
| 10% | 0.896 @ 3.7, 0.908 @ 5.8, 0.922 @ 5.1 | 0.896 @ 5.0, 0.908 @ 5.9, 0.920 @ 5.9 |

- **FP16 truth equals FP32 truth** in these cells (the recall against exact search over FP16-rounded rows and queries differs by at most 0.001).
- **In-region (tenant-style) queries** fill and read high for every system: at `ef_search` 160 FP16 0.989 / 0.983 at 3.2 ms, FP32 0.987 / 0.981 at 3.7 to 4.5 ms; native 1.000 at 8.5 / 12.8 ms (1% / 10% key).

## Build and size

| System | Build s | Index MB |
|---|---|---|
| pgvector hnsw, FP32 | 98.2 | 4,094.2 |
| pgvector hnsw, FP16 expression | 58.7 | 2,047.1 |
| pg_pinecone native (1% key / 10% key) | 44.7 / 44.9 | 136.2 |

## Not verified

- Other `ef_search` above 160, other k than 10, the FP16 heap and FP32 rerank arms of `REPORT-halfvec.md`, residual (unpushed) predicates.
