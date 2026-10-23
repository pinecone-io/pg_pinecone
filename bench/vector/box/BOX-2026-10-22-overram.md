# Box 2026-10-22: resolved facts of the over-RAM session

What this records: versions and checksums of the two machines of `REPORT-overram.md` (the reported run is the second session, below) (`configs/overram_1m.yaml` and its rehearsal `configs/smoke-overram.yaml`), so it can be rebuilt from `bench/vector/box/` (`README.md`, "Over-RAM session"). No Pinecone: no key file was copied. Region and zone only: no account id, instance id, address or key. Machine-read facts: `settings-2026-10-22-overram.json`.

## Machine

| | |
|---|---|
| Region / zone | us-east-1 / us-east-1a |
| Instance type | `r6id.4xlarge`: 16 vCPU Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz, 123.8 GiB visible, 950 GB NVMe instance store at `/data`, cgroup v2 |
| AMI | `ami-0fa5967347d08d2df` (pinned in `box/provision.sh`) |
| OS | Ubuntu 24.04.5 LTS, kernel `7.0.0-1014-aws` |
| Launched / terminated | 2026-10-22 23:30:39 / about 23:59 UTC |

## Software

| | |
|---|---|
| PostgreSQL 17 | 17.11 from PGDG (`postgresql-17=17.11-1.pgdg24.04+2`), started by `box/overram-pg.sh` as `pg-overram.service` (`MemorySwapMax=0`), `pg/tuned.linux-aws.conf` with `shared_buffers` 1 GB, `effective_cache_size` 3 GB, port 55433 |
| pgvector | 0.8.7 (`postgresql-17-pgvector=0.8.7-1.pgdg24.04+1`) |
| pg_pinecone | 0.10.0 release tarball, pg17 sha256 `52167cc7a2883fc14d480ce2f4a2564b87780f63efebf7b3d2c72f72bc8a5ec2` (installed `pg_pinecone.so` `08352bfc486d241669415c559ba2831a251f433d970cc8595dc2e3701fc86f4f`) |
| Harness | `bench/vector` at `ec17c02b8959fc869363847d39647802bd1a7566` for both reported runs (not dirty, not changed mid-run) |

## Data

wiki_dpr as `BOX-2026-10-22-hvfilt.md` (same five sha256 values, `base_content_sha` `dd93ee85811b7494`).

## Runs

| Step | Started (UTC) | Ended | Result |
|---|---|---|---|
| `ovr`: `phase.sh data overram_prep overram_smoke overram_1m` at `4c034b9` | 23:33:28 | 23:45:31 | smoke 5 / 5 points valid; uncapped run complete; capped run OOM-killed at the native cap (`raw/failed/` of the session copy, not reported) |
| `ovr2`: `phase.sh overram_1m` at `ec17c02` (cap waits for anonymous memory) | 23:46:34 | 23:56:14 | `overram_1m_cap_infinity` 12 / 12, `overram_1m_cap_4G` 12 / 12 |

`/data/raw`, `/data/logs` and `/data/state` were copied off and checked (sha256, 25 files) before `teardown.sh`.

## Second session (2026-10-23): the reported run

The same recipe on a fresh instance (launched 00:10:29, terminated about 00:28 UTC), harness at `4c8f16ba9f998835b4896234a19c9dee2cb11175`: the sweeps repeat their first point as a cache warm-up. Same packages, tarball and installed library sha256 as above; the five dataset sha256 values match. Machine-read facts: `settings-2026-10-23-overram.json`.

| Step | Started (UTC) | Ended | Result |
|---|---|---|---|
| `ovr`: `phase.sh data overram_prep overram_smoke overram_1m` | 00:14:10 | 00:26:56 | smoke 7 / 7; `overram_1m_cap_infinity` 15 / 15 (281 s), `overram_1m_cap_4G` 15 / 15 (315 s) |

Copied off and checked (sha256, 19 files) before `teardown.sh`.
