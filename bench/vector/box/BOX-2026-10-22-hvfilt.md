# Box 2026-10-22: resolved facts of the filtered halfvec session

What this records: versions and checksums of the machine of `REPORT-halfvec-filtered.md` (`configs/halfvec_filtered_1m_0.10.yaml` and its rehearsal `configs/smoke-halfvec-filtered.yaml`), so it can be rebuilt from `bench/vector/box/` (`README.md`, "Filtered halfvec session"). No Pinecone: no key file was copied. Region and zone only: no account id, instance id, address or key. Machine-read facts: `settings-2026-10-22-hvfilt.json`.

## Machine

| | |
|---|---|
| Region / zone | us-east-1 / us-east-1a |
| Instance type | `r6id.4xlarge`: 16 vCPU Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz, 123.8 GiB visible, 950 GB NVMe instance store at `/data` |
| AMI | `ami-0fa5967347d08d2df` (pinned in `box/provision.sh`) |
| OS | Ubuntu 24.04.5 LTS, kernel `7.0.0-1014-aws` |
| Launched / terminated | 2026-10-22 22:00:26 / about 22:58 UTC |

## Software

| | |
|---|---|
| PostgreSQL 18 | 18.6 from PGDG (`postgresql-18=18.6-1.pgdg24.04+2`), `phase.sh pg18_install`, `pg_debug_assertions` off |
| pgvector | 0.8.7 (`postgresql-18-pgvector=0.8.7-1.pgdg24.04+1`) |
| pg_pinecone | 0.10.0 release tarballs: pg18 sha256 `6c7589f7f34f0c0819e767187ed875e796cdb8e77d10e7bdc343a2966ecdcb09` (installed `pg_pinecone.so` `3d1995c3f11b643dd062d059779dded49ef0c291f3607d38777b6452f5269685`); pg17 `52167cc7a2883fc14d480ce2f4a2564b87780f63efebf7b3d2c72f72bc8a5ec2`, installed because `setup-box.sh` expects it, unused |
| Python | 3.12.3; numpy 2.5.4, psycopg 3.3.6, pgvector 0.5.1, PyYAML 6.0.3 |
| Harness | `bench/vector` at `98455fb690e17e9414d95443bb933cd46a4e7010` (stamped in both results, not dirty, not changed mid-run) |
| Postgres settings | `pg/tuned.linux-aws.conf`; the config adds `maintenance_work_mem` 8 GB, `min_wal_size` 1 GB, `max_wal_size` 4 GB |

## Data

wiki_dpr: `base_content_sha` `dd93ee85811b7494` (matches every earlier run); dataset key `real_fd937e5862449934`.

| File | sha256 |
|---|---|
| `wiki_passages_0` | `c277bc2ba1ef027ee86cb24b1df7015aaa32972da88d5183f6e5a9563d8705b5` |
| `wiki_passages_1` | `9bcfb323ccb0c0f8ed5de949287129c630bfee52738cd2ae232a52eaa6cf6faf` |
| `wiki_passages_2` | `c66c771b0363ceb78e19f7fdc7348159c9b4d548837b3856da5d867811f42191` |
| `wiki_passages_3` | `3659fccf01a2640fcc5da35161b7afb89ea669f295b7570d78678afd32c8b9b4` |
| `wikidpr_768d_1m.npz` | `fcb07ccfd32a88cb619b4f6ad1371a5e44d6d11e3e20da9c4e46d781bc88c146` |

## Runs

One detached step, `phase.sh data pg18_install hvf_prep hvf_smoke hvf_1m`, 22:14:45 to 22:51:45 UTC, rc 0; nothing else ran beside it.

| Run | Started (UTC) | Wall | Points valid |
|---|---|---|---|
| `smoke_halfvec_filtered` | 22:19:21 | 157 s | 60 / 60 |
| `halfvec_filtered_1m_0.10` | 22:21:59 | 1,786 s | 128 / 128 |

`/data/raw`, `/data/logs` and `/data/state` were copied off and checked (sha256, 18 files) before `teardown.sh`.
