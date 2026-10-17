# Benchmarks

Harnesses, configs, dataset scripts, the fake Pinecone and the write-ups behind pg_pinecone's performance numbers. Nothing here is
built or run by CI: `.github/workflows/ci.yml` formats and lints the Rust crates and runs the Rust tests only.

| dir | measures |
|---|---|
| [`vector/`](vector/README.md) | Filtered and unfiltered ANN search: pgvector hnsw / ivfflat / halfvec against pg_pinecone's native local engine and the remote engine; residual `WHERE` filters and the iterative scan, on synthetic data, SIFT1M and wiki_dpr 768d |

## Common setup

- State, downloads and scratch clusters live in `bench/.data/` (gitignored). Move them with `PG_PINECONE_BENCH_DATA=/path`.
- The Postgres under test is `PG_PINECONE_BENCH_PG` (an install prefix with `bin/pg_ctl`). `vector/` defaults to a pgrx-managed
  PG 17 (`~/.pgrx/17.10/pgrx-install`) that has pgvector. pgrx's own PG is a `--enable-cassert` build: fine for plumbing and
  recall, not for timing.
- Build pg_pinecone into a private pgrx home (`PGRX_HOME`), never into one other worktrees install into.
- The smoke below needs no Pinecone key and runs with `env -u PINECONE_API_KEY`. The remote configs read a key from a file and spend
  real money (see `vector/box/README.md`).
- The Python venv is `vector/.venv` (verified with Python 3.14). Paths must not contain spaces.

## Smoke command (verified 2026-10-12; sizes and timings in `vector/README.md`)

From `bench/vector` (`PG_PINECONE_BENCH_PG` needs pgvector and pg_pinecone installed: see "Install pg_pinecone" in `vector/README.md`):

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export PG_PINECONE_BENCH_DATA="$(cd .. && pwd)/.data" PG_PINECONE_BENCH_PG=/path/to/postgres-install-prefix
scripts/download-datasets.sh sift                # 168 MB; `wikidpr` is 5.3 GB
env -u PINECONE_API_KEY .venv/bin/python run.py --config configs/smoke-native.yaml \
    --data-dir "$PG_PINECONE_BENCH_DATA/vector/pgdata-smoke" -v      # 50k x 768d, 411 s
```

Results land in `vector/results/` (one JSON and one markdown file per run), gitignored. The committed write-ups are the records of past runs:

- `vector/REPORT-residual-filter.md`: iterative scan vs pgvector, 1M x 768d, 2026-10-07.
- `vector/REPORT-halfvec.md`: pgvector halfvec vs FP32 hnsw and native, 1M x 768d, 2026-10-16.

## Not reproducible from this tree

- **Anything against real Pinecone** needs a paid Pinecone project (and, for the box recipe, AWS): the `remote_*` configs and their
  phases in `vector/box/phase.sh`.
- **Absolute numbers.** Reports quote a laptop (M4 Max, 36 GB, other apps running) or an AWS box; compare rows inside a table.
