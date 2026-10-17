# Over-RAM workload: Docker variant + runbook

Goal: measure what happens to recall/latency when the index + hot working set
no longer fits in `shared_buffers` + OS page cache -- the regime real deployments
hit long before they hit CPU limits. This is the Docker-based variant for a
Linux host; the capped runs of `REPORT-overram.md` use `box/overram-pg.sh`
instead (see `../README.md`, "Over-RAM workload"). It is not credible on a
laptop Docker VM; the local smoke check only verified that the mechanics
(container, tuned conf, pgvector) work.

## What was verified locally (2026-09-09, local macOS)

- Docker ran in a local Linux VM (2 CPU, 2 GB RAM, 10 GB disk).
- Pulled `pgvector/pgvector:pg17` (official image, pgvector preinstalled).
- Ran it memory-capped (`--memory=512m --memory-swap=512m`, no swap headroom)
  with `bench/vector/overram/postgresql.overram.conf` mounted in as the *entire*
  config (`-c config_file=...`), confirming:
  - `CREATE EXTENSION vector`, HNSW build, and a `<->` query all work inside
    the capped container.
  - **Gotcha we hit and fixed**: a custom `config_file=` bypasses the
    docker-entrypoint's usual `listen_addresses='*'` patch, so Postgres bound
    to loopback-only inside the container and the host-side port mapping
    connected-then-immediately-closed (looks like "server closed the
    connection unexpectedly", not a permissions/auth error). Our checked-in
    `postgresql.overram.conf` now sets `listen_addresses = '*'` explicitly --
    if you fork this conf, keep that line.
  - The compose plugin was not installed there; the checked-in
    `docker-compose.yml` is the canonical config for a host with it, but the
    smoke test used the equivalent `docker run` (see below).
- No 1M-scale dataset was loaded into the capped container -- that's the
  actual over-RAM experiment, deliberately not attempted here (see "Why not
  here" below).

Repro command for the smoke check above:
```
docker run -d --name bench-overram-smoke \
  --memory=512m --memory-swap=512m --shm-size=128mb \
  -e POSTGRES_PASSWORD=bench -e POSTGRES_USER=bench -e POSTGRES_DB=bench \
  -v "$(pwd)/postgresql.overram.conf:/etc/postgresql/postgresql.conf:ro" \
  -p 55432:5432 \
  pgvector/pgvector:pg17 postgres -c config_file=/etc/postgresql/postgresql.conf
```

## Why not the real experiment here

1. **macOS Docker memory limits run inside a Linux VM, which
   itself competes with the host for real RAM.** `--memory` caps the
   container's cgroup inside that VM, but the VM's own memory (and macOS's
   unified memory management, compressed memory, etc.) sit underneath it.
   Page cache eviction behavior under memory pressure -- exactly the effect
   we want to measure -- is not the same two layers removed from bare metal.
2. **virtiofs/9p-backed volumes** (a VM's default mount type) have their own
   caching layer with different characteristics than a real block device;
   disk-bound index scans would be measuring the VM's file-sharing protocol
   as much as Postgres.
3. A meaningful over-RAM run needs the *dataset* (index + table) to exceed
   the memory cap by a real margin (e.g. a 1M x 768 fp32 dataset here is
   ~3GB of raw vectors before index overhead) -- multi-GB inside a nested VM
   on a shared laptop isn't a fair or reproducible number, and would compete
   with the smoke/full runs for the same machine's RAM.

## Runbook: real over-RAM run (Linux host, e.g. cloud VM or bare metal)

1. Provision a Linux host with **known, fixed RAM** (e.g. a cloud VM sized
   deliberately: pick a memory size that's a fraction of the working set --
   see step 3) and a local SSD (not network block storage with its own
   caching tier).
2. Install Docker (native cgroups, no nested VM) or run the tuned Postgres
   directly on the host with a cgroup memory limit
   (`systemd-run --scope -p MemoryMax=<N>G`) if you want to skip Docker
   entirely.
3. Pick the memory cap as a fraction of the expected on-disk working set:
   - Run `bench/vector/run.py --config bench/vector/configs/full.yaml` once, uncapped, and
     note `index_size_bytes` (from the JSON results, per system) plus raw
     vector size (`n * dim * 4` bytes). That's the working set.
   - Choose 2-3 memory caps that are e.g. 150%, 75%, and 30% of that working
     set, to sweep from "fits comfortably" to "mostly disk-bound".
4. `docker compose -f bench/vector/overram/docker-compose.yml up -d` with
   `OVERRAM_MEM_LIMIT=<cap>` set per run (compose plugin needed; use the
   equivalent `docker run` from this file's "verified" section if compose
   isn't installed). For the runs of record use `box/overram-pg.sh` instead.
5. Point `harness/pgctl.py`-equivalent connection logic at the container
   instead of a local `initdb`'d instance (swap in a raw `psycopg.connect`
   using the container's mapped port; the rest of the harness -- loader,
   workloads, metrics -- is system-agnostic and doesn't care how Postgres was
   started).
6. Run the same unfiltered + filtered sweeps as `configs/full.yaml`, once per
   memory cap. Report latency-at-recall curves *per cap* on the same chart:
   the interesting result is where the curve breaks away from the uncapped
   baseline, not a single number.
7. Watch `docker stats` (or cgroup `memory.current`) during the run to
   confirm the container is actually memory-pressured (steady-state near the
   cap, not idling well under it) -- a cap that's never hit isn't a
   measurement, it's a no-op.
