#!/usr/bin/env bash
# Out-of-process VACUUM write-path drill.
#
# `am_ambulkdelete_purges_dead_tuple_via_real_vacuum` (pg_pinecone/src/lib.rs)
# documents a hard `pgrx-tests` limitation: every `#[pg_test]` runs inside one
# client-side transaction, whose pinned snapshot blocks VACUUM's `OldestXmin`
# horizon for the whole test — a row deleted *during* the test can never look
# reclaimable to a VACUUM triggered from inside it. That leaves ambulkdelete's
# write half (the outbox purge SPI DELETE + the real `/vectors/delete` HTTP
# call) unverified by the pg_test suite. This script closes that gap with
# genuine, separate `psql` connections — no enclosing transaction at all.
#
# Needs PINECONE_API_KEY (creates a real, temporary serverless index) and
# drives the pgrx-managed pg17 instance directly, so it is NOT part of the
# non-live suite — run by hand / in a live-e2e CI stage, not `cargo pgrx test`.
set -euo pipefail

if [ -z "${PINECONE_API_KEY:-}" ]; then
    echo "PINECONE_API_KEY is not set — skipping vacuum-drill (out-of-process VACUUM write-path check)." >&2
    exit 0
fi

# Re-execs itself once under the shared pgrx lock (see below) so the whole
# install/restart/drill/restore sequence is one atomic section against every
# other cooperating `cargo pgrx ...` invocation on this machine — not just
# each command in isolation, which would let another invocation see (or
# clobber) a half-restarted instance.
LOCK=/tmp/pg_pinecone-pgrx.lock
if [ "${1:-}" != "--locked" ]; then
    exec lockf -t 900 "$LOCK" "$0" --locked
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PG_PINECONE_DIR="$REPO_ROOT/pg_pinecone"
PG_CONFIG="$(cd "$PG_PINECONE_DIR" && cargo pgrx info pg-config pg17)"
# pgrx's own fixed per-major-version layout (same one scripts/install-pgvector.sh
# assumes for its own $PG_CONFIG default) — not derived from $PG_CONFIG's path,
# since pgrx-install/ and data-17/ are siblings under ~/.pgrx, not nested.
PGRX_HOME="${PGRX_HOME:-$HOME/.pgrx}"
PGDATA="$PGRX_HOME/data-17"

PID=$$
TABLE="pg_pinecone_vacuum_drill_$PID"
DBNAME="pg_pinecone_vacuum_drill_$PID"
API_VERSION="2026-04"
CONTROL_PLANE_URL="https://api.pinecone.io"

echo "== vacuum-drill: table=$TABLE db=$DBNAME pg_config=$PG_CONFIG ==" >&2

# ---------------------------------------------------------------------------
# Instance lifecycle: build+install the CURRENT tree, then a stop/configure/
# start round trip that guarantees both a fresh .so (a plain `cargo pgrx
# install` never swaps an already-loaded preloaded library) and
# shared_preload_libraries/worker_enabled are on, regardless of this shared
# instance's prior state.
# ---------------------------------------------------------------------------

was_running=0
if cargo pgrx status pg17 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -qi "is running"; then
    was_running=1
fi
orig_worker_enabled=""
PORT=""
INDEX_NAME=""

pg_port() {
    # postmaster.pid line 4 is the port — robust against pgrx's port
    # convention changing, unlike hardcoding 28800 + major version.
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

cleanup() {
    local status=$?
    set +e
    if [ -n "$PORT" ]; then
        if [ -n "$INDEX_NAME" ]; then
            curl -fsS -X DELETE "$CONTROL_PLANE_URL/indexes/$INDEX_NAME" \
                -H "Api-Key: $PINECONE_API_KEY" \
                -H "X-Pinecone-Api-Version: $API_VERSION" >/dev/null 2>&1
        fi
        psql -h 127.0.0.1 -p "$PORT" -U "$(whoami)" -d postgres \
            -c "DROP DATABASE IF EXISTS \"$DBNAME\" WITH (FORCE)" >/dev/null 2>&1
        if [ -n "$orig_worker_enabled" ]; then
            psql -h 127.0.0.1 -p "$PORT" -U "$(whoami)" -d postgres -v ON_ERROR_STOP=1 \
                -c "ALTER SYSTEM SET pinecone.worker_enabled = '$orig_worker_enabled'" \
                -c "SELECT pg_reload_conf()" >/dev/null 2>&1
        fi
    fi
    # Instance running/stopped state restored: only stop it if THIS script
    # is the one that started it. shared_preload_libraries is left as
    # 'pg_pinecone' (a dev/test instance wanting that is the common case, and
    # reverting it would mean yet another disruptive restart of a
    # potentially shared instance).
    if [ "$was_running" -eq 0 ]; then
        cargo pgrx stop pg17 >/dev/null 2>&1
    fi
    exit "$status"
}
trap cleanup EXIT

(cd "$PG_PINECONE_DIR" && cargo pgrx install -c "$PG_CONFIG")

cargo pgrx start pg17
wait_for_ready
PORT="$(pg_port)"
psql_pg() { psql -h 127.0.0.1 -p "$PORT" -U "$(whoami)" -v ON_ERROR_STOP=1 "$@"; }

orig_worker_enabled="$(psql_pg -d postgres -tAc "SHOW pinecone.worker_enabled" 2>/dev/null || echo "")"
psql_pg -d postgres \
    -c "ALTER SYSTEM SET shared_preload_libraries = 'pg_pinecone'" \
    -c "ALTER SYSTEM SET pinecone.worker_enabled = on" \
    -c "SELECT pg_reload_conf()"

cargo pgrx stop pg17
cargo pgrx start pg17
wait_for_ready
PORT="$(pg_port)"

psql_pg -d postgres -c "CREATE DATABASE \"$DBNAME\""
db() { psql_pg -d "$DBNAME" "$@"; }

# ---------------------------------------------------------------------------
# Fixture: a real AM-indexed table against a temp Pinecone index.
# ---------------------------------------------------------------------------

db -c "CREATE EXTENSION IF NOT EXISTS vector"
db -c "CREATE EXTENSION pg_pinecone"
db -c "CREATE TABLE $TABLE (id bigint PRIMARY KEY, embedding vector(3))"
# Key via stdin, not argv — argv is world-readable in `ps` on shared machines.
db <<SQL
SELECT pinecone.set_api_key('$PINECONE_API_KEY');
SQL
db -c "CREATE INDEX ${TABLE}_idx ON $TABLE USING pinecone (embedding vector_cosine_ops)"

# `ambuild` derives the Pinecone index name/namespace itself — read them
# back rather than recomputing the sanitize/truncate rule in bash.
INDEX_NAME="$(db -tAc "SELECT index_name FROM pinecone.tables WHERE relid = '$TABLE'::regclass")"
NAMESPACE="$(db -tAc "SELECT namespace FROM pinecone.tables WHERE relid = '$TABLE'::regclass")"
echo "== temp index: $INDEX_NAME (namespace $NAMESPACE) ==" >&2

db -c "INSERT INTO $TABLE (id, embedding) VALUES (1, '[1,0,0]'::vector), (2, '[0,1,0]'::vector)"

# Two dead-tuple shapes in one VACUUM, matching ambulkdelete_core's own
# candidate set (remote ids UNION still-queued upsert ids — am.rs's own
# rustdoc): id 1 is flushed to the remote index before it dies (proves the
# real /vectors/delete call); id 3 dies before its own upsert ever drains
# (proves the outbox purge SPI DELETE on a row that's genuinely still there).
if ! db -tAc "SELECT pinecone.flush('$TABLE', 60000)" | grep -q '^t$'; then
    echo "pinecone.flush timed out waiting for the initial insert to drain — is the worker running?" >&2
    exit 1
fi
DEAD_FLUSHED_ID="$(db -tAc "SELECT ctid::text FROM $TABLE WHERE id = 1" | tr -d '()' | tr ',' ':')"
db -c "DELETE FROM $TABLE WHERE id = 1"

# Worker paused (deterministically, not just "hope it's slower than us")
# so id 3's own upsert genuinely never drains before it dies — restored to
# its pre-drill value by the cleanup trap.
db -c "ALTER SYSTEM SET pinecone.worker_enabled = off"
db -c "SELECT pg_reload_conf()"
sleep 3 # >= one worker naptime (default 2000ms), so it's actually exited.

db -c "INSERT INTO $TABLE (id, embedding) VALUES (3, '[0,0,1]'::vector)"
DEAD_QUEUED_ID="$(db -tAc "SELECT ctid::text FROM $TABLE WHERE id = 3" | tr -d '()' | tr ',' ':')"
db -c "DELETE FROM $TABLE WHERE id = 3"

# No enclosing transaction (unlike the `#[pg_test]` this
# replaces) — this VACUUM's OldestXmin horizon genuinely advances past both
# deletes above, so both dead tuples are reclaimable.
db -c "VACUUM $TABLE"

# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

fail=0

remaining_queued="$(db -tAc \
    "SELECT count(*) FROM pinecone.outbox WHERE relid = '$TABLE'::regclass \
     AND payload->0->>'id' = '$DEAD_QUEUED_ID'")"
if [ "$remaining_queued" != "0" ]; then
    echo "FAIL: dead TID $DEAD_QUEUED_ID's still-queued outbox row was not purged by VACUUM" >&2
    fail=1
fi

remaining_flushed="$(db -tAc \
    "SELECT count(*) FROM pinecone.outbox WHERE relid = '$TABLE'::regclass \
     AND payload->0->>'id' = '$DEAD_FLUSHED_ID'")"
if [ "$remaining_flushed" != "0" ]; then
    echo "FAIL: dead TID $DEAD_FLUSHED_ID left a stray outbox row after VACUUM" >&2
    fail=1
fi

HOST="$(curl -fsS "$CONTROL_PLANE_URL/indexes/$INDEX_NAME" \
    -H "Api-Key: $PINECONE_API_KEY" \
    -H "X-Pinecone-Api-Version: $API_VERSION" | jq -r '.host')"
case "$HOST" in
    http*) ;;
    *) HOST="https://$HOST" ;;
esac
ENCODED_NS="$(jq -rn --arg v "$NAMESPACE" '$v|@uri')"
REMOTE_IDS="$(curl -fsS "$HOST/vectors/list?limit=100&namespace=$ENCODED_NS" \
    -H "Api-Key: $PINECONE_API_KEY" \
    -H "X-Pinecone-Api-Version: $API_VERSION" | jq -r '.vectors[]?.id')"
LIVE_ID="$(db -tAc "SELECT ctid::text FROM $TABLE WHERE id = 2" | tr -d '()' | tr ',' ':')"

if echo "$REMOTE_IDS" | grep -qx "$DEAD_FLUSHED_ID"; then
    echo "FAIL: dead TID $DEAD_FLUSHED_ID is still present in the remote namespace after VACUUM" >&2
    fail=1
fi
if ! echo "$REMOTE_IDS" | grep -qx "$LIVE_ID"; then
    echo "FAIL: the still-live row's own remote id is unexpectedly gone — VACUUM over-deleted" >&2
    fail=1
fi

db -c "DROP TABLE $TABLE"

if [ "$fail" -eq 0 ]; then
    echo "PASS: ambulkdelete's write half (outbox purge + real /vectors/delete) verified via a genuine out-of-process VACUUM." >&2
else
    exit 1
fi
