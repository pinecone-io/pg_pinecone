#!/usr/bin/env bash
# Installs a packaged pg_pinecone tarball into the Postgres it was built for and runs the
# mock-free checks of install-drill.sh on a throwaway cluster (no Pinecone, no API key).
#   ./scripts/release-smoke.sh <pg_config> <pg_pinecone-...tar.gz>
# Writes into <pg_config>'s pkglibdir/sharedir (via sudo if needed): never aim it at a live server.
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <pg_config> <tarball>" >&2
    exit 2
fi
PG_CONFIG="$1"
TARBALL="$2"
[ -x "$PG_CONFIG" ] || { echo "FAIL: pg_config is not executable: $PG_CONFIG" >&2; exit 2; }
[ -f "$TARBALL" ] || { echo "FAIL: tarball not found: $TARBALL" >&2; exit 2; }

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
TARBALL="$(cd "$(dirname "$TARBALL")" && pwd)/$(basename "$TARBALL")"

BINDIR="$("$PG_CONFIG" --bindir)"
PKGLIBDIR="$("$PG_CONFIG" --pkglibdir)"
EXTDIR="$("$PG_CONFIG" --sharedir)/extension"
PG_VERSION="$("$PG_CONFIG" --version)"
PG_MAJOR="$(printf '%s' "$PG_VERSION" | sed -E 's/^PostgreSQL ([0-9]+).*/\1/')"
echo "== release smoke: $(basename "$TARBALL") on $PG_VERSION ==" >&2

WORK="$(mktemp -d /tmp/pg_pinecone-release-smoke.XXXXXX)"
STAGE="$WORK/stage"
PGDATA="$WORK/data"
SOCKDIR="$WORK/sock"
LOG="$WORK/postgres.log"
started=0

cleanup() {
    local status=$?
    set +e
    if [ "$started" -eq 1 ]; then
        "$BINDIR/pg_ctl" -D "$PGDATA" -m immediate stop >/dev/null 2>&1
    fi
    if [ "$status" -ne 0 ] && [ -f "$LOG" ]; then
        echo "---- postgres log (tail) ----" >&2
        tail -n 60 "$LOG" >&2
    fi
    rm -rf "$WORK"
    exit "$status"
}
trap cleanup EXIT

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

tar_major="$(basename "$TARBALL" | sed -nE 's/.*-pg([0-9]+)-.*/\1/p')"
if [ -n "$tar_major" ] && [ "$tar_major" != "$PG_MAJOR" ]; then
    fail "$(basename "$TARBALL") is built for pg$tar_major, but $PG_CONFIG is Postgres $PG_MAJOR"
fi

mkdir -p "$STAGE" "$SOCKDIR"
tar -xzf "$TARBALL" -C "$STAGE"

# The layout mirrors the packaging machine's install prefix: find files, don't assume paths.
find_one() {
    local what="$1" found count
    shift
    found="$(find "$STAGE" -type f "$@")"
    count="$(printf '%s' "$found" | grep -c . || true)"
    if [ "$count" -ne 1 ]; then
        echo "FAIL: expected exactly one $what in the tarball, found $count" >&2
        [ -z "$found" ] || printf '%s\n' "$found" >&2
        exit 1
    fi
    printf '%s\n' "$found"
}

lib="$(find_one 'pg_pinecone shared library' \( -name pg_pinecone.so -o -name pg_pinecone.dylib \))"
control="$(find_one 'pg_pinecone.control' -name pg_pinecone.control)"
ext_version="$(sed -n "s/^default_version *= *'\(.*\)'.*/\1/p" "$control")"
[ -n "$ext_version" ] || fail "no default_version in the tarball's pg_pinecone.control"
find_one "pg_pinecone--$ext_version.sql" -name "pg_pinecone--$ext_version.sql" >/dev/null
upgrade_scripts=0
for script in "$REPO_ROOT"/pg_pinecone/sql/pg_pinecone--*--*.sql; do
    [ -f "$script" ] || continue
    find_one "${script##*/}" -name "${script##*/}" >/dev/null
    upgrade_scripts=$((upgrade_scripts + 1))
done
[ "$ext_version" = 0.10.0 ] || [ -n "$(find "$STAGE" -type f -name "pg_pinecone--*--$ext_version.sql")" ] \
    || fail "tarball carries no upgrade script into $ext_version (pg_pinecone/sql/pg_pinecone--<prev>--$ext_version.sql)"
echo "PASS: tarball carries $(basename "$lib"), pg_pinecone.control, pg_pinecone--$ext_version.sql and $upgrade_scripts upgrade scripts" >&2

if [ -f "$REPO_ROOT/pg_pinecone/Cargo.toml" ]; then
    crate_version="$(grep -m1 '^version = ' "$REPO_ROOT/pg_pinecone/Cargo.toml" | sed -E 's/version = "(.*)"/\1/')"
    [ "$ext_version" = "$crate_version" ] \
        || fail "tarball is pg_pinecone $ext_version, but this tree's crate version is $crate_version"
fi

SUDO=""
if [ ! -w "$PKGLIBDIR" ] || [ ! -w "$EXTDIR" ]; then
    SUDO="sudo"
fi
# rm first: overwriting a signed Mach-O in place can get the process SIGKILLed on macOS.
put() {
    local mode="$1" src="$2" destdir="$3"
    $SUDO rm -f "$destdir/$(basename "$src")"
    $SUDO install -m "$mode" "$src" "$destdir/"
}
put 755 "$lib" "$PKGLIBDIR"
put 644 "$control" "$EXTDIR"
for sql in $(find "$STAGE" -type f -name 'pg_pinecone--*.sql'); do
    put 644 "$sql" "$EXTDIR"
done
echo "PASS: tarball installed into $PKGLIBDIR and $EXTDIR" >&2

$SUDO "$HERE/install-pgvector.sh" "$PG_CONFIG"

PORT="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])' 2>/dev/null || echo $((20000 + RANDOM % 20000)))"
"$BINDIR/initdb" -D "$PGDATA" -U postgres -A trust -E UTF8 --locale=C --no-sync >/dev/null
cat >> "$PGDATA/postgresql.conf" <<CONF
port = $PORT
listen_addresses = ''
unix_socket_directories = '$SOCKDIR'
shared_preload_libraries = 'pg_pinecone'
fsync = off
CONF
"$BINDIR/pg_ctl" -D "$PGDATA" -l "$LOG" -w -t 60 start >/dev/null
started=1

psql_pg() { "$BINDIR/psql" -X -q -h "$SOCKDIR" -p "$PORT" -U postgres -d postgres -v ON_ERROR_STOP=1 "$@"; }
scalar() { psql_pg -tAc "$1"; }

server_major="$(scalar "SELECT current_setting('server_version_num')::int / 10000")"
[ "$server_major" = "$PG_MAJOR" ] || fail "server is Postgres $server_major, expected $PG_MAJOR"
[ "$(scalar 'SHOW shared_preload_libraries')" = "pg_pinecone" ] || fail "pg_pinecone is not preloaded"
launcher=0
for _ in $(seq 1 30); do
    launcher="$(scalar "SELECT count(*) FROM pg_stat_activity WHERE backend_type = 'pg_pinecone launcher'")"
    [ "$launcher" -ge 1 ] && break
    sleep 1
done
[ "$launcher" -ge 1 ] || fail "the pg_pinecone launcher background worker did not start"
echo "PASS: Postgres $PG_MAJOR started with pg_pinecone preloaded (launcher running)" >&2

psql_pg -c "CREATE EXTENSION vector"
psql_pg -c "CREATE EXTENSION pg_pinecone"
installed="$(scalar "SELECT extversion FROM pg_extension WHERE extname = 'pg_pinecone'")"
[ "$installed" = "$ext_version" ] || fail "CREATE EXTENSION installed '$installed', expected '$ext_version'"
echo "PASS: CREATE EXTENSION pg_pinecone installed at $installed" >&2

for obj in tables outbox secret; do
    [ "$(scalar "SELECT to_regclass('pinecone.$obj') IS NOT NULL")" = "t" ] \
        || fail "pinecone.$obj missing after fresh install"
done
psql_pg -c "SELECT * FROM pinecone.status()" >/dev/null
echo "PASS: core schema objects present (tables/outbox/secret/status())" >&2

psql_pg -c "CREATE TABLE smoke_docs (id bigint PRIMARY KEY, embedding vector(3))"
psql_pg -c "INSERT INTO smoke_docs (id, embedding) VALUES (1, '[1,0,0]'), (2, '[0,1,0]')"
psql_pg -c "SELECT pinecone.track('smoke_docs')" >/dev/null
top_id="$(scalar "SELECT id FROM pinecone.topk('smoke_docs', ARRAY[1,0,0]::real[], 1) AS r ORDER BY r.distance")"
[ "$top_id" = "1" ] || fail "local topk() returned '$top_id', expected the nearest row '1'"
echo "PASS: local track -> topk round-trip returned the nearest row" >&2

psql_pg -c "CREATE TABLE smoke_am (id bigint PRIMARY KEY, embedding vector(3))"
psql_pg -c "INSERT INTO smoke_am SELECT g, ('[' || g || ',1,0]')::vector FROM generate_series(1, 50) g"
psql_pg -c "SELECT pinecone.create_index('smoke_am', 'embedding', engine => 'native')" >/dev/null
am_query="SELECT id FROM smoke_am ORDER BY embedding <=> '[50,1,0]' LIMIT 1"
am_plan="$(PGOPTIONS='-c enable_seqscan=off' scalar "EXPLAIN (COSTS OFF) $am_query")"
case "$am_plan" in
    *smoke_am_embedding_pinecone_idx*) ;;
    *) fail "the planner did not use the pg_pinecone index: $am_plan" ;;
esac
am_id="$(PGOPTIONS='-c enable_seqscan=off' scalar "$am_query")"
[ "$am_id" = "50" ] || fail "native index scan returned '$am_id', expected the nearest row '50'"
echo "PASS: native index (access method) scan returned the nearest row" >&2

psql_pg -c "DROP TABLE smoke_am"
psql_pg -c "DROP TABLE smoke_docs"
psql_pg -c "DROP EXTENSION pg_pinecone"
[ "$(scalar "SELECT count(*) FROM pg_extension WHERE extname = 'pg_pinecone'")" = "0" ] \
    || fail "DROP EXTENSION pg_pinecone left the extension registered"
[ "$(scalar "SELECT to_regnamespace('pinecone') IS NULL")" = "t" ] \
    || fail "DROP EXTENSION pg_pinecone left the pg_pinecone schema behind"
echo "PASS: DROP EXTENSION pg_pinecone removed the extension and its schema" >&2

"$BINDIR/pg_ctl" -D "$PGDATA" -m fast -w stop >/dev/null
started=0
echo "PASS: release smoke succeeded for $(basename "$TARBALL") on $PG_VERSION" >&2
