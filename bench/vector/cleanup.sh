#!/bin/sh
# Stops leftover scratch Postgres instances and bench processes of the vector harness and removes scratch data dirs.
# Scratch convention: run.py --data-dir "$DATA/pgdata-<name>", where DATA = $PG_PINECONE_BENCH_DATA/vector (default bench/.data/vector).
# Keeps: datasets_cache/, results/, logs. Usage: ./cleanup.sh [extra-data-dir ...] (extra dirs are stopped, not deleted).
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${PG_PINECONE_BENCH_DATA:-$HERE/../.data}/vector
if [ -n "$PG_PINECONE_BENCH_PG" ]; then PGBIN=$PG_PINECONE_BENCH_PG/bin; else PGBIN=${PGRX_PG_BIN:-$HOME/.pgrx/17.10/pgrx-install/bin}; fi
PIDS=${PG_PINECONE_BENCH_PIDS:-$DATA/pids}

kill_tree() {  # kill a pid and all its descendants (spawn-pool workers, postmasters)
  for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c"; done
  kill "$1" 2>/dev/null
}

# 1. scratch postgres instances (data dirs only; *.sock dirs have no postmaster.pid).
#    Extra data dirs can be passed as arguments (stopped, not deleted).
for d in "$DATA"/pgdata* "$HERE"/.pgdata "$@"; do
  if [ -f "$d/postmaster.pid" ]; then
    echo "stopping postgres in $d"
    "$PGBIN/pg_ctl" -D "$d" -m fast -w -t 20 stop >/dev/null 2>&1 \
      || "$PGBIN/pg_ctl" -D "$d" -m immediate stop >/dev/null 2>&1
  fi
done

# 2. registered processes: "pid|command-pattern|label"; the pattern guards against pid reuse
if [ -f "$PIDS" ]; then
  while IFS='|' read -r pid pattern label; do
    [ -n "$pid" ] || continue
    cmd=$(ps -p "$pid" -o command= 2>/dev/null)
    if [ -n "$cmd" ] && echo "$cmd" | grep -q -- "$pattern"; then
      echo "killing $pid ($label)"
      kill_tree "$pid"
    fi
  done < "$PIDS"
  sleep 2
  while IFS='|' read -r pid pattern label; do
    [ -n "$pid" ] || continue
    cmd=$(ps -p "$pid" -o command= 2>/dev/null)
    if [ -n "$cmd" ] && echo "$cmd" | grep -q -- "$pattern"; then
      echo "force-killing $pid ($label)"; kill -9 "$pid" 2>/dev/null
    fi
  done < "$PIDS"
  : > "$PIDS"
fi

# 3. scratch directories
rm -rf "$DATA"/pgdata* "$HERE"/.pgdata "$HERE"/.pgdata.sock

# 4. verify nothing is left
left=$(ps -axo pid=,command= | grep -E "$DATA/pgdata|$HERE/.pgdata|run\.py --config" | grep -v -E 'grep|cleanup.sh' )
if [ -n "$left" ]; then echo "STILL RUNNING:"; echo "$left"; else echo "cleanup ok: no bench processes left"; fi
du -sh "$DATA"/pgdata* 2>/dev/null
