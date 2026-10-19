# Box for the vector benchmark (pg_pinecone 0.10.0)

One `r6id.4xlarge` (16 vCPU, 128 GB, 950 GB NVMe instance store) on Ubuntu 24.04 x86_64 in us-east-1 with PGDG Postgres 17 (18 on demand) and pgvector, the
pg_pinecone release tarball and this harness on the same machine. The scripts in this directory provision it (`provision.sh`, `user-data.sh`,
`postgresql.bench.conf`), upload the committed harness (`upload.sh`), run steps detached (`step.sh`, `step-run.sh`) and tear it down (`teardown.sh`).
The resolved facts of each session (AMI, kernel, versions, settings, checksums) are in `BOX-2026-10-16-halfvec.md`, `BOX-2026-10-22-hvfilt.md` and
`BOX-2026-10-22-overram.md` with their `settings-*.json`; the sessions below say which one to follow.

## Run it again

Needs the AWS CLI v2 logged in (`aws sso login`) with `AWS_PROFILE` set, `BOX_OWNER` (the Owner tag), and `jq`. The Pinecone-backed phases also need a file `ee.env` holding
`PINECONE_API_KEY=...` (never committed, never printed) and spend real money: the harness tallies every request and stops at 6M write units, 1M read units or $40 (`harness/remote.py`).

```
curl -fsSLO https://github.com/pinecone-io/pg_pinecone/releases/download/v0.10.0/pg_pinecone-0.10.0-pg17-linux-x86_64.tar.gz
export AWS_PROFILE=<profile> BOX_OWNER=<you> BOX_DATE=<date> BOX_JSON=$HOME/.pg_pinecone-bench/box-<date>.json
bench/vector/box/provision.sh                                # key pair, security group (SSH from this IP only), instance
EE_ENV=/path/to/ee.env TARBALL=$PWD/pg_pinecone-0.10.0-pg17-linux-x86_64.tar.gz bench/vector/box/upload.sh
source bench/vector/box/conn.sh && bssh                      # a shell on the box
```

On the box (`B=~/pg_pinecone/bench`; every long step detached with `vector/box/step.sh <name> <command>`, log `/data/logs/<name>.log`, state `/data/state/step-<name>.json`):

```
$B/vector/box/install-pg_pinecone.sh                         # sha256-checked, into the PGDG directories
$B/vector/box/setup-box.sh                                   # env file, venv, key file mode, sar
$B/vector/box/step.sh data   $B/vector/box/phase.sh data     # wiki_dpr: 5.3 GB of shards, 3.4 GB npz, sha256 list
$B/vector/box/step.sh prep   $B/vector/box/phase.sh prep     # k-means and ground-truth caches of every config, no Postgres
$B/vector/box/step.sh smoke  $B/vector/box/phase.sh smoke    # 5k-row rehearsal of the Pinecone lane
```

`phase.sh` phases (finished ones leave `/data/state/done-<phase>` and are skipped when the chain is started again):

| Phase | What | Pinecone |
|---|---|---|
| `local_1m`, `c1`, `c2`, `ext`, `ext_model` | leg A: `local_1m_0.10.yaml`, `residual-c1-0.10.yaml`, `residual-c2-0.10.yaml`, `residual-ext-0.10.yaml`, `residual-ext-model.yaml` (its mode-L system alone) | no |
| `remote_1m_build`, `remote_1m`, `remote_filtered` | `remote_1m.yaml` (build, then measure) and `remote_filtered_keep.yaml` on the same index | one 1M-row index |
| `lane_am_build`, `lane_am`, `lane_topk_build`, `lane_topk` | the two 100k lane configs | two 100k-row indexes |
| `partition_build`, `partition` | `remote_partition.yaml`, 100 tenants on 500k rows | one index, 100 namespaces |
| `cleanup` | `pc_indexes.py delete-all --yes`: every index whose name contains `pg-pinecone-bench-vec` | deletes |

A `*_build` phase loads the table, creates the index, waits for the backfill and the gate and stops, leaving the cluster under `/data/cl/pgdata-<name>` and the index in place; the matching measure phase reuses both (`run.py --stage measure`). Backfills take hours at the default worker settings, so the build phases run beside leg A (a backfill is network-bound and uses well under one core); the measure phases run after it.

## Kernel microbench

The kernel microbench (Hamming over the native sketches, 1-bit code x 4-bit query planes in scalar, AVX2 and AVX-512, record-major and 64-code blocks, read roofline, core-frequency probe) needs only a Rust toolchain (`phase.sh tools` installs one; the VM has no PMU, so `perf` runs with `-e cpu-clock`):

```
cd bench/vector/kernels && cargo run --release -- --cpu 3
```

## Halfvec session

`configs/halfvec_1m.yaml` (pgvector FP16 arms against FP32 hnsw and native) on the box recipe above, no Pinecone, no key file. Its native arms measure **pg_pinecone built from main** on the box, not the release tarball: the tarball is still uploaded and installed only because `upload.sh` and `setup-box.sh` expect one, and `hv_install` replaces it. The pg_pinecone commit is the `origin/main` uploaded below, stamped at run time in `/data/state/pg_pinecone-build.txt` (commit and library sha256); the run itself stamps `pg_pinecone_extversion` 0.10.0. Copy both into the session's facts file, and run `settings.py <label>` and copy `/data/state`, `/data/logs` and `/data/raw` off the box (check them) before `teardown.sh`: the 2026-10-16 session lost its state and logs (`BOX-2026-10-16-halfvec.md`, Gaps). Results: `../REPORT-halfvec.md`.

```
export BOX_DATE=<date>-halfvec BOX_JSON=$HOME/.pg_pinecone-bench/box-<date>-halfvec.json BOX_OWNER=<you> AWS_PROFILE=<profile>
bench/vector/box/provision.sh
TARBALL=$PWD/pg_pinecone-0.10.0-pg17-linux-x86_64.tar.gz bench/vector/box/upload.sh
git fetch origin && SRC_SHA=$(git rev-parse origin/main) && source bench/vector/box/conn.sh
git archive --format=tar --prefix=pg_pinecone-src/ "$SRC_SHA" Cargo.toml Cargo.lock pg_pinecone pg_pinecone-client \
    | bssh "rm -rf ~/pg_pinecone-src && tar -x -C ~ && echo $SRC_SHA > ~/pg_pinecone-src/COMMIT"
```

On the box (`B=~/pg_pinecone/bench`, `S=$B/vector/box/step.sh`, `P=$B/vector/box/phase.sh`): `$B/vector/box/install-pg_pinecone.sh`, `$B/vector/box/setup-box.sh`, then

```
$S hvinstall $P hv_install         # rustup, cargo-pgrx 0.19.2 (the workspace pin) under its own --root, pg_pinecone from ~/pg_pinecone-src into the PGDG Postgres 17
$S data $P data                    # wiki_dpr, as above
$S hvprep $P hv_prep               # k-means, FP32 and FP16-rounded ground truth (same dataset key as local_1m)
$S hvsmoke $P hv_smoke             # smoke-halfvec-native.yaml: every arm kind of hv_1m at 50k rows on the main build
$S hv1m $P hv_1m                   # halfvec_1m.yaml: 2.5 to 3 h, an estimate
```

`hv_install` builds with `RUSTFLAGS='-C target-cpu=x86-64-v3'`, as the release tarballs are. Nothing else runs beside `hv_1m`. Teardown as below.



## Teardown

```
$VENV/bin/python $B/vector/box/pc_indexes.py delete-all --yes     # then it lists: must print []
bench/vector/box/teardown.sh                                     # terminate, key pair, security group, leftovers listing
```

## Notes

- `/data` is the instance store: its contents are gone after a stop. Logs, state and raw results sit on the EBS root (`/data/logs`, `/data/state`, `/data/raw` are links); the clusters are on the NVMe.
- The AM names its Pinecone index `pv-<schema>-<table>` and offers no way to change it, so the tables are named `pg_pinecone_bench_vec_*` and every cleanup and listing check matches the substring `pg-pinecone-bench-vec`. The trigger lane takes an `index_name`: `pg-pinecone-bench-vec-lane-topk`.
- Postgres settings: `pg/tuned.linux-aws.conf` (memory, planner and `jit` as `postgresql.bench.conf`; durability traded for speed and autovacuum off as the harness always does); remote configs lower `shared_buffers` to 4 GB because several clusters run at once.
