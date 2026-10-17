#!/usr/bin/env bash
# pg_pinecone fresh-install drill: a fresh `CREATE EXTENSION pg_pinecone` at the current
# crate version. In-place upgrades (from 0.9.0 on) are gated by upgrade-drill.sh.
#
# Against the shared pgrx pg17 instance this proves, end to end:
#   1. `cargo pgrx install` writes the base-install script as
#      `pg_pinecone--<crate version>.sql` into the sharedir (the release artifact's
#      filename — see .github/workflows/release.yml).
#   2. `CREATE EXTENSION pg_pinecone` installs at exactly that version.
#   3. the core schema objects exist (pinecone.tables / pinecone.outbox /
#      pinecone.status()).
#   4. a local track -> topk round-trip works mock-free (no network, no API key,
#      pgvector engine only).
#   5. `DROP EXTENSION pg_pinecone` tears the schema down cleanly.
#
# Shared-lock + instance-lifecycle scaffolding mirrors vacuum-drill.sh: the whole
# install/drill/restore sequence runs as one atomic section under the shared pgrx
# lock, and this repo's own HEAD install is restored on exit.
set -euo pipefail

LOCKED=0
for arg in "$@"; do
    case "$arg" in
        --locked) LOCKED=1 ;;
    esac
done

# Re-execs itself once under the shared pgrx lock so the whole install/drill/
# restore sequence is one atomic section against every other cooperating
# `cargo pgrx ...` invocation on this machine.
LOCK=/tmp/pg_pinecone-pgrx.lock
if [ "$LOCKED" -eq 0 ]; then
    exec lockf -t 1800 "$LOCK" "$0" --locked "$@"
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PG_PINECONE_DIR="$REPO_ROOT/pg_pinecone"
PG_CONFIG="$(cd "$PG_PINECONE_DIR" && cargo pgrx info pg-config pg17)"
# pgrx's own fixed per-major-version layout (pgrx-install/ and data-17/ are
# siblings under ~/.pgrx, not nested under each other).
PGRX_HOME="${PGRX_HOME:-$HOME/.pgrx}"
PGDATA="$PGRX_HOME/data-17"

PID=$$

was_running=0
PORT=""
DBNAME=""
restore_install=0

pg_port() {
    sed -n '4p' "$PGDATA/postmaster.pid"
}

wait_for_ready() {
    local tries=0
    until psql -h 127.0.0.1 -p "$(pg_port)" -U "$(whoami)" -d postgres -c 'SELECT 1' >/dev/null 2>&1; do
        tries=$((tries + 1))
        if [ "$tries" -ge 60 ]; then
            echo "postgres never accepted connections" >&2
            return 1
        fi
        sleep 1
    done
}

# A plain `cargo pgrx install` only overwrites the `.so` *file* — an already-
# running postmaster keeps serving whatever each backend first dlopen'ed. A
# restart guarantees the newly-written `.so` is what actually gets loaded.
restart_pg() {
    cargo pgrx stop pg17
    cargo pgrx start pg17
    wait_for_ready
    PORT="$(pg_port)"
}

cleanup() {
    local status=$?
    set +e
    if [ -n "$PORT" ] && [ -n "$DBNAME" ]; then
        psql -h 127.0.0.1 -p "$PORT" -U "$(whoami)" -d postgres \
            -c "DROP DATABASE IF EXISTS \"$DBNAME\" WITH (FORCE)" >/dev/null 2>&1
    fi
    if [ "$restore_install" -eq 1 ]; then
        # Leave the shared instance's installed extension files matching this
        # repo's checked-out HEAD again (the drill only ever installs HEAD, so
        # this is a no-op reinstall — kept for symmetry with the other drills).
        (cd "$PG_PINECONE_DIR" && cargo pgrx install -c "$PG_CONFIG") >/dev/null 2>&1
    fi
    if [ "$was_running" -eq 0 ]; then
        cargo pgrx stop pg17 >/dev/null 2>&1
    fi
    exit "$status"
}
trap cleanup EXIT

if cargo pgrx status pg17 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -qi "is running"; then
    was_running=1
else
    cargo pgrx start pg17
fi
wait_for_ready
PORT="$(pg_port)"
psql_pg() { psql -h 127.0.0.1 -p "$PORT" -U "$(whoami)" -v ON_ERROR_STOP=1 "$@"; }

cur_version="$(grep -m1 '^version = ' "$PG_PINECONE_DIR/Cargo.toml" | sed -E 's/version = "(.*)"/\1/')"
if [ -z "$cur_version" ]; then
    echo "FAIL: could not read version from $PG_PINECONE_DIR/Cargo.toml" >&2
    exit 1
fi
echo "== fresh-install drill for pg_pinecone $cur_version ==" >&2

echo "== building + installing pg_pinecone $cur_version (current tree) ==" >&2
(cd "$PG_PINECONE_DIR" && cargo pgrx install -c "$PG_CONFIG")
restore_install=1
restart_pg # guarantee the just-installed .so is what's actually loaded

# 1. the base-install script the release artifact ships, named for the version.
base_install="$("$PG_CONFIG" --sharedir)/extension/pg_pinecone--$cur_version.sql"
if [ ! -f "$base_install" ]; then
    echo "FAIL: cargo pgrx install did not write pg_pinecone--$cur_version.sql (looked for $base_install)" >&2
    exit 1
fi
echo "PASS: base-install script present at $base_install" >&2

DBNAME="pg_pinecone_install_drill_$PID"
psql_pg -d postgres -c "CREATE DATABASE \"$DBNAME\""
db() { psql_pg -d "$DBNAME" "$@"; }

# 2. fresh CREATE EXTENSION installs at exactly the crate version.
db -c "CREATE EXTENSION IF NOT EXISTS vector"   # pg_pinecone auto-installs its opclasses against pgvector
db -c "CREATE EXTENSION pg_pinecone"
installed="$(db -tAc "SELECT extversion FROM pg_extension WHERE extname = 'pg_pinecone'")"
if [ "$installed" != "$cur_version" ]; then
    echo "FAIL: CREATE EXTENSION installed '$installed', expected '$cur_version'" >&2
    exit 1
fi
echo "PASS: CREATE EXTENSION pg_pinecone installed at $installed" >&2

# 3. core schema objects exist.
for obj in tables outbox secret; do
    have="$(db -tAc "SELECT to_regclass('pinecone.$obj') IS NOT NULL")"
    if [ "$have" != "t" ]; then
        echo "FAIL: pinecone.$obj missing after fresh install" >&2
        exit 1
    fi
done
db -c "SELECT * FROM pinecone.status()" >/dev/null
echo "PASS: core schema objects present (tables/outbox/secret/status())" >&2

# 4. a local track -> topk round-trip, mock-free (pgvector engine, no network).
db -c "CREATE TABLE drill_docs (id bigint PRIMARY KEY, embedding vector(3))"
db -c "INSERT INTO drill_docs (id, embedding) VALUES (1, '[1,0,0]'::vector), (2, '[0,1,0]'::vector)"
db -c "SELECT pinecone.track('drill_docs')"
top_id="$(db -tAc "SELECT id FROM pinecone.topk('drill_docs', ARRAY[1,0,0]::real[], 1) AS r ORDER BY r.distance")"
if [ "$top_id" != "1" ]; then
    echo "FAIL: local topk() did not return the nearest row (got '$top_id', expected '1')" >&2
    exit 1
fi
echo "PASS: local track -> topk round-trip returned the nearest row" >&2

# 5. DROP EXTENSION tears the schema down cleanly.
db -c "DROP EXTENSION pg_pinecone"
still_there="$(db -tAc "SELECT count(*) FROM pg_extension WHERE extname = 'pg_pinecone'")"
if [ "$still_there" != "0" ]; then
    echo "FAIL: DROP EXTENSION pg_pinecone left the extension registered" >&2
    exit 1
fi
schema_gone="$(db -tAc "SELECT to_regnamespace('pinecone') IS NULL")"
if [ "$schema_gone" != "t" ]; then
    echo "FAIL: DROP EXTENSION pg_pinecone left the pg_pinecone schema behind" >&2
    exit 1
fi
echo "PASS: DROP EXTENSION pg_pinecone removed the extension and its schema" >&2

echo "PASS: fresh-install drill succeeded for pg_pinecone $cur_version." >&2
