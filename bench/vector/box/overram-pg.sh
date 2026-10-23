#!/usr/bin/env bash
# usage (on the box): overram-pg.sh start | cap <size> | uncap | snap | stop   (the external Postgres 17 of configs/overram_1m.yaml)
# A transient unit so MemoryMax can change while it runs: builds uncapped, sweeps capped. cgroup v2 charges page cache to the unit.
set -euo pipefail
[ ! -f "$HOME/.bench-env" ] || source "$HOME/.bench-env"
UNIT=pg-overram.service
CG=/sys/fs/cgroup/system.slice/$UNIT
BIN=/usr/lib/postgresql/17/bin
PGDATA=${CL:-/data/cl}/pgdata-overram
PORT=55433
SHARED_BUFFERS=${OVERRAM_SHARED_BUFFERS:-1GB}

snap() {
    local s; s=$(cat "$CG/memory.stat")
    st() { awk -v k="$1" '$1 == k { print $2 }' <<< "$s"; }
    printf '{"max": "%s", "current": %s, "peak": %s, "file": %s, "anon": %s, "shmem": %s, "pgmajfault": %s, "workingset_refault_file": %s, "events": "%s"}\n' \
        "$(cat "$CG/memory.max")" "$(cat "$CG/memory.current")" "$(cat "$CG/memory.peak" 2> /dev/null || echo null)" \
        "$(st file)" "$(st anon)" "$(st shmem)" "$(st pgmajfault)" "$(st workingset_refault_file)" "$(tr '\n' ' ' < "$CG/memory.events")"
}

set_max() {
    if [ "$1" != infinity ]; then
        local limit pinned waited=0
        limit=$(numfmt --from=iec "$1")
        while :; do
            pinned=$(awk '$1 == "anon" || $1 == "shmem" { s += $2 } END { print s }' "$CG/memory.stat")
            [ "$pinned" -lt $((limit * 3 / 4)) ] && break
            [ "$waited" -lt 300 ] || { echo "anon + shmem $pinned still above 3/4 of $1 after ${waited}s" >&2; exit 1; }
            sleep 1; waited=$((waited + 1))
        done
        echo "waited ${waited}s for anon + shmem ($pinned) below 3/4 of $1"
    fi
    sudo systemctl set-property --runtime "$UNIT" MemoryMax="$1"
    snap
}

case ${1:-} in
    start)
        sudo systemctl stop "$UNIT" 2> /dev/null || true
        sudo systemctl reset-failed "$UNIT" 2> /dev/null || true
        rm -rf "$PGDATA"
        "$BIN/initdb" -D "$PGDATA" -A trust -E UTF8 --locale=C.UTF-8 > /dev/null
        { cat "$BENCH_TUNED_CONF"
          printf "\nshared_buffers = %s\neffective_cache_size = 3GB\nport = %s\nlisten_addresses = 'localhost'\nunix_socket_directories = '/tmp'\n" "$SHARED_BUFFERS" "$PORT"
        } >> "$PGDATA/postgresql.conf"
        sudo systemd-run --unit="$UNIT" --uid="$(id -u)" --gid="$(id -g)" -p MemoryMax=infinity -p MemorySwapMax=0 \
            "$BIN/postgres" -D "$PGDATA" > /dev/null
        for _ in $(seq 1 60); do "$BIN/pg_isready" -q -h localhost -p "$PORT" && break; sleep 1; done
        "$BIN/pg_isready" -h localhost -p "$PORT"
        snap
        ;;
    cap) set_max "${2:?size, e.g. 4G}" ;;
    uncap) set_max infinity ;;
    snap) snap ;;
    stop) sudo systemctl stop "$UNIT" || true ;;
    *) echo "usage: $0 start | cap <size> | uncap | snap | stop" >&2; exit 2 ;;
esac
