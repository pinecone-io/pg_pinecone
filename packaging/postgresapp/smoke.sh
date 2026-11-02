#!/usr/bin/env bash
# Load a Postgres.app install root the way Postgres.app would, without writing
# into ~/Library/Application Support. Postgres.app scans that folder on every
# server start, so a smoke test must not put files there.
#
#   packaging/postgresapp/smoke.sh <install-root> [pg_config]
#
# <install-root> is the prefix build.zsh produced (lib/ and share/ at the top).
# pg_config defaults to Postgres.app's PostgreSQL 18. The server is started
# with dynamic_library_path and extension_control_path pointed at a temp copy,
# $libdir and $system first, plus shared_preload_libraries=pg_pinecone.
# Uses the pgvector bundled with that Postgres.app.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 <install-root> [pg_config]" >&2
    exit 2
fi

INSTALL_ROOT=$1
PG_CONFIG=${2:-/Applications/Postgres.app/Contents/Versions/18/bin/pg_config}
[[ -d $INSTALL_ROOT/lib/postgresql ]] || { echo "FAIL: no lib/postgresql in $INSTALL_ROOT" >&2; exit 2; }
[[ -d $INSTALL_ROOT/share/postgresql/extension ]] || { echo "FAIL: no extension directory in $INSTALL_ROOT" >&2; exit 2; }
[[ -x $PG_CONFIG ]] || { echo "FAIL: pg_config is not executable: $PG_CONFIG" >&2; exit 2; }

HERE=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$HERE/../.." && pwd)
BINDIR=$("$PG_CONFIG" --bindir)
PG_VERSION=$("$PG_CONFIG" --version)
PG_MAJOR=$(printf '%s' "$PG_VERSION" | sed -E 's/^PostgreSQL ([0-9]+).*/\1/')
VECTOR_CONTROL=$("$PG_CONFIG" --sharedir)/extension/vector.control
[[ -f $VECTOR_CONTROL ]] || { echo "FAIL: no bundled vector.control at $VECTOR_CONTROL" >&2; exit 1; }
VECTOR_CKSUM=$(cksum "$VECTOR_CONTROL")
echo "== postgres.app smoke on $PG_VERSION (bundled $(grep -m1 default_version "$VECTOR_CONTROL")) ==" >&2

WORK=$(mktemp -d)
DEST=$WORK/pg_pinecone
PGDATA=$WORK/data
SOCKDIR=$WORK/sock
LOG=$WORK/postgres.log
started=0

cleanup() {
    local status=$?
    set +e
    if [[ $started -eq 1 ]]; then
        "$BINDIR/pg_ctl" -D "$PGDATA" -m immediate stop >/dev/null 2>&1
    fi
    if [[ $status -ne 0 && -f $LOG ]]; then
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

mkdir -p "$DEST" "$SOCKDIR"
cp -R "$INSTALL_ROOT/lib" "$INSTALL_ROOT/share" "$DEST/"
DYLIB_DIR=$DEST/lib/postgresql
CONTROL_PARENT=$DEST/share/postgresql
[[ -f $DYLIB_DIR/pg_pinecone.dylib ]] || fail "copied tree has no pg_pinecone.dylib"
control=$CONTROL_PARENT/extension/pg_pinecone.control
[[ -f $control ]] || fail "copied tree has no pg_pinecone.control"
ext_version=$(sed -n "s/^default_version *= *'\(.*\)'.*/\1/p" "$control")
[[ -n $ext_version ]] || fail "no default_version in pg_pinecone.control"
if [[ -f $REPO_ROOT/pg_pinecone/Cargo.toml ]]; then
    crate_version=$(grep -m1 '^version = ' "$REPO_ROOT/pg_pinecone/Cargo.toml" | sed -E 's/version = "(.*)"/\1/')
    [[ $ext_version == "$crate_version" ]] \
        || fail "install root is pg_pinecone $ext_version, but this tree's crate version is $crate_version"
fi

PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])' 2>/dev/null || echo $((20000 + RANDOM % 20000)))
"$BINDIR/initdb" -D "$PGDATA" -U postgres -A trust -E UTF8 --locale=C --no-sync >/dev/null
# $libdir and $system stay literal so Postgres expands them. The temp dirs are
# what Server.swift would have appended after scanning an extension folder.
cat >> "$PGDATA/postgresql.conf" <<CONF
port = $PORT
listen_addresses = ''
unix_socket_directories = '$SOCKDIR'
shared_preload_libraries = 'pg_pinecone'
dynamic_library_path = '\$libdir:$DYLIB_DIR'
extension_control_path = '\$system:$CONTROL_PARENT'
fsync = off
CONF
"$BINDIR/pg_ctl" -D "$PGDATA" -l "$LOG" -w -t 60 start >/dev/null
started=1

psql_pg() { "$BINDIR/psql" -X -q -h "$SOCKDIR" -p "$PORT" -U postgres -d postgres -v ON_ERROR_STOP=1 "$@"; }
scalar() { psql_pg -tAc "$1"; }

server_major=$(scalar "SELECT current_setting('server_version_num')::int / 10000")
[[ $server_major == "$PG_MAJOR" ]] || fail "server is Postgres $server_major, expected $PG_MAJOR"
[[ $(scalar 'SHOW shared_preload_libraries') == "pg_pinecone" ]] || fail "pg_pinecone is not preloaded"
launcher=0
for _ in $(seq 1 30); do
    launcher=$(scalar "SELECT count(*) FROM pg_stat_activity WHERE backend_type = 'pg_pinecone launcher'")
    [[ $launcher -ge 1 ]] && break
    sleep 1
done
[[ $launcher -ge 1 ]] || fail "the pg_pinecone launcher background worker did not start"
echo "PASS: Postgres $PG_MAJOR started with pg_pinecone preloaded (launcher running)" >&2

psql_pg -c "CREATE EXTENSION vector"
psql_pg -c "CREATE EXTENSION pg_pinecone"
installed=$(scalar "SELECT extversion FROM pg_extension WHERE extname = 'pg_pinecone'")
[[ $installed == "$ext_version" ]] || fail "CREATE EXTENSION installed '$installed', expected '$ext_version'"
echo "PASS: CREATE EXTENSION pg_pinecone installed at $installed" >&2

for obj in tables outbox secret; do
    [[ $(scalar "SELECT to_regclass('pinecone.$obj') IS NOT NULL") == "t" ]] \
        || fail "pinecone.$obj missing after fresh install"
done
psql_pg -c "SELECT * FROM pinecone.status()" >/dev/null
echo "PASS: core schema objects present (tables/outbox/secret/status())" >&2

psql_pg -c "CREATE TABLE smoke_docs (id bigint PRIMARY KEY, embedding vector(3))"
psql_pg -c "INSERT INTO smoke_docs (id, embedding) VALUES (1, '[1,0,0]'), (2, '[0,1,0]')"
psql_pg -c "SELECT pinecone.track('smoke_docs')" >/dev/null
top_id=$(scalar "SELECT id FROM pinecone.topk('smoke_docs', ARRAY[1,0,0]::real[], 1) AS r ORDER BY r.distance")
[[ $top_id == "1" ]] || fail "local topk() returned '$top_id', expected the nearest row '1'"
echo "PASS: local track -> topk round-trip returned the nearest row" >&2

psql_pg -c "CREATE TABLE smoke_am (id bigint PRIMARY KEY, embedding vector(3))"
psql_pg -c "INSERT INTO smoke_am SELECT g, ('[' || g || ',1,0]')::vector FROM generate_series(1, 50) g"
psql_pg -c "SELECT pinecone.create_index('smoke_am', 'embedding', engine => 'native')" >/dev/null
am_query="SELECT id FROM smoke_am ORDER BY embedding <=> '[50,1,0]' LIMIT 1"
am_plan=$(PGOPTIONS='-c enable_seqscan=off' scalar "EXPLAIN (COSTS OFF) $am_query")
case $am_plan in
    *smoke_am_embedding_pinecone_idx*) ;;
    *) fail "the planner did not use the pg_pinecone index: $am_plan" ;;
esac
am_id=$(PGOPTIONS='-c enable_seqscan=off' scalar "$am_query")
[[ $am_id == "50" ]] || fail "native index scan returned '$am_id', expected the nearest row '50'"
echo "PASS: native index (access method) scan returned the nearest row" >&2

psql_pg -c "DROP TABLE smoke_am"
psql_pg -c "DROP TABLE smoke_docs"
psql_pg -c "DROP EXTENSION pg_pinecone"
[[ $(scalar "SELECT count(*) FROM pg_extension WHERE extname = 'pg_pinecone'") == "0" ]] \
    || fail "DROP EXTENSION pg_pinecone left the extension registered"
[[ $(scalar "SELECT to_regnamespace('pinecone') IS NULL") == "t" ]] \
    || fail "DROP EXTENSION pg_pinecone left the pg_pinecone schema behind"
echo "PASS: DROP EXTENSION pg_pinecone removed the extension and its schema" >&2

[[ $(cksum "$VECTOR_CONTROL") == "$VECTOR_CKSUM" ]] || fail "bundled vector.control was modified"
echo "PASS: bundled pgvector was left in place" >&2

"$BINDIR/pg_ctl" -D "$PGDATA" -m fast -w stop >/dev/null
started=0
echo "PASS: postgres.app smoke succeeded on $PG_VERSION" >&2
