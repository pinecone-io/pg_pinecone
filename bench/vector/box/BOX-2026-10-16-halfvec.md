# Box 2026-10-16: resolved facts of the halfvec session

What this records: versions, settings and checksums of the machine of `REPORT-halfvec.md` (`configs/halfvec_1m.yaml` and its rehearsal `configs/smoke-halfvec-native.yaml`), so it can be rebuilt from `bench/vector/box/` (`README.md`, "Halfvec session"). It is the recipe of the 2026-10-14 box on a fresh instance, with pg_pinecone built from `main` on the box instead of the release tarball and no Pinecone: no key file was copied, no Pinecone call was made. Region and zone only: no account id, instance id, address or key.

The box's `/data/state` and `/data/logs` (step states, `pg_pinecone-build.txt`, `data-sha256.txt`, `pip-freeze.txt`, the step logs and `sar`) came back empty from the copy made before teardown, and the instance was terminated before that was noticed. What they held is recorded below only where the session's console output or the results JSON carry it, and there is no `settings-2026-10-16-halfvec.json` (`settings.py` did not run before teardown; see Gaps).

## Machine

| | |
|---|---|
| Region / zone | us-east-1 / us-east-1a |
| Instance type | `r6id.4xlarge`: 16 vCPU Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz, 123.8 GiB visible of 128 GB, 1 x 950 GB NVMe instance store at `/data` |
| Root volume | 100 GB gp3 |
| AMI | `ami-0fa5967347d08d2df` = `ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-20261004` (pinned in `box/provision.sh`, as on the 2026-10-14 box) |
| OS | Ubuntu 24.04.5 LTS, kernel `7.0.0-1014-aws` |
| Tags, access | `Project=pg_pinecone-bench`, `Owner` (`BOX_OWNER`), `TTL=3d`; IMDSv2 required; SSH from one address only; powers itself off after 72 h |
| Launched | 2026-10-16T13:43:47+00:00 |

## Software

| | |
|---|---|
| PostgreSQL 17 | `PostgreSQL 17.11 (Ubuntu 17.11-1.pgdg24.04+2)` from PGDG (`postgresql-17=17.11-1.pgdg24.04+2`, `postgresql-server-dev-17=17.11-1.pgdg24.04+2`), `pg_debug_assertions` off |
| pgvector | 0.8.7 (`postgresql-17-pgvector=0.8.7-1.pgdg24.04+1`) |
| pg_pinecone | **0.10.0 from `main` at `7763b35d647aad894fb3311cf175061a5183b9d8`** (unreleased), built on the box by `phase.sh hv_install`: cargo-pgrx 0.19.2 (the workspace pin) under its own `--root`, `cargo pgrx install --release` with `RUSTFLAGS='-C target-cpu=x86-64-v3'` into the PGDG directories, 2 min 17 s; installed `pg_pinecone.so` sha256 `50136328c87959062d66bac03140ea8198220c227d1c4ea886a99e1989eff3dc`; `pg_pinecone.control` `default_version = '0.10.0'`. The 0.9.0 tarball (`pg_pinecone-0.9.0-pg17-linux-x86_64.tar.gz`, sha256 `8c77895b7471973e4c0d1ad16b65caa661a00c112dfd1827665660d8b367136a`) was installed first by `install-pg_pinecone.sh`, as the recipe expects, and replaced |
| Rust toolchain | rustup stable of the day; the `rustc` version was in the lost `hvinstall` log |
| Python | the AMI's `python3`, venv `~/bench-data/vector/venv` from `requirements.txt`; the `pip freeze` was in the lost `/data/state` |
| Harness | `bench/vector` at `204592c4fee88bd0c2e4c1f389371abf75af8b06` (stamped in every result, none dirty, none changed mid-run) |

## Runs

| Run | Started (UTC) | Wall | Harness commit | Points valid |
|---|---|---|---|---|
| `smoke_halfvec_native` | 2026-10-16 14:00:32 | 131 s | `204592c` | 76 / 76 |
| `halfvec_1m` | 2026-10-16 14:03:06 | 95 min | `204592c` | 531 / 531 |

The smoke ran while `hv_prep` computed the 1M k-means and ground truth (`hv_prep` and `hv_smoke` were started together); `halfvec_1m` ran alone.

## Postgres settings (as stamped into each result's `pg_settings`)

The 2026-10-14 leg-A settings (`pg/tuned.linux-aws.conf`), unchanged: `shared_buffers` 32 GB, `effective_cache_size` 96 GB, `work_mem` 256 MB, `maintenance_work_mem` 8 GB, `max_parallel_workers` 16, `max_parallel_maintenance_workers` 8, `fsync` and `jit` off. `halfvec_1m.yaml` adds `min_wal_size` 1 GB and `max_wal_size` 4 GB. `shared_preload_libraries` is not set. The embedding column of the FP32 heap is `STORAGE PLAIN` (`fp32_column_storage: plain`); the FP16 heap keeps halfvec's default (`external`, never TOASTed at 1,544 bytes).

## Datasets

wiki_dpr: `phase.sh data` downloaded the four shards and rebuilt `wikidpr_768d_1m.npz` in 45 s and printed `base_content_sha dd93ee85811b7494 (matches the 2026-09 runs: dd93ee85811b7494)`; the run stamps dataset key `real_fd937e5862449934`, the key of `local_1m.yaml`. The per-file sha256 list (`data-sha256.txt`) was in the lost `/data/state`; the files of the same download on the 2026-10-14 box, whose `base_content_sha` matches:

| File | sha256 |
|---|---|
| `wiki_passages_0` | `c277bc2ba1ef027ee86cb24b1df7015aaa32972da88d5183f6e5a9563d8705b5` |
| `wiki_passages_1` | `9bcfb323ccb0c0f8ed5de949287129c630bfee52738cd2ae232a52eaa6cf6faf` |
| `wiki_passages_2` | `c66c771b0363ceb78e19f7fdc7348159c9b4d548837b3856da5d867811f42191` |
| `wiki_passages_3` | `3659fccf01a2640fcc5da35161b7afb89ea669f295b7570d78678afd32c8b9b4` |
| `wikidpr_768d_1m.npz` | `fcb07ccfd32a88cb619b4f6ad1371a5e44d6d11e3e20da9c4e46d781bc88c146` |

## Pinecone

None. No key file was copied to the box (`upload.sh` without `EE_ENV`), no Pinecone call was made.

## Gaps

- `pip freeze`, the shard checksums, the Rust toolchain version and the `sar` CPU samples of this session are not recorded.
- For a next session: run `settings.py <label>` and check the copy (`ls`) before `teardown.sh`.

## Lifecycle

| | |
|---|---|
| Launched / ready | 2026-10-16 13:43:47 UTC / 13:44:53 UTC |
| Terminated | 2026-10-16 15:38:45 UTC (`describe-instances`: `terminated`), 1.92 h after launch |
| Leftovers after teardown | none: no tagged instance or volume, key pair and security group deleted, private key file removed |
