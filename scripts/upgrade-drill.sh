#!/usr/bin/env bash
# In-place upgrade drill (docs/contributing.md): installs release <base-tag> from its tarball, loads a
# fixture (promoted, partitioned, access-method remote, full-text, native and local tables) against a fake
# Pinecone (scripts/upgrade/fake_pinecone: control, vector, namespace and document planes), builds this tree
# with `cargo pgrx package`, restarts onto it, runs ALTER EXTENSION pg_pinecone UPDATE and proves the result
# equals a fresh install of this tree (pg_dump of the extension's objects, normalized) and still works.
# The access-method indexes the base built read as format 1: the remote one keeps working, the native one
# (clustered: a flat one has no codes to refuse) is offline until pinecone.reindex_native_indexes() rebuilds
# it as format 3 with a calibration (planner walks away with a WARNING, a forced scan and an insert raise 55000,
# VACUUM warns, pinecone.index_calibration() reports nothing until the rebuild, then the calibrated pair).
#   ./scripts/upgrade-drill.sh <base-tag> [pg_config]
# Exits non-zero on the first difference. Needs gh (tarball download), cargo-pgrx, python3, curl; no
# Pinecone key is ever set. Replaces pg_pinecone files in <pg_config>'s pkglibdir/sharedir (via sudo if
# needed): never aim it at a live server. CI's base is `scripts/upgrade/pick-base.py derive` (the latest
# release below this tree's version); the drill refuses a base without an upgrade script path to this tree.
# Env: UPGRADE_ORDER = restart-first (default, the only order CI runs: restart onto the new library, whose
# worker idles until the UPDATE) or update-first (UPDATE while the old library still runs, then restart;
# local only: it fails once a script creates a C function the preloaded library lacks);
# UPGRADE_BASE_DIR (tarball cache, default /tmp/pg_pinecone-upgrade-base); UPGRADE_BASE_TARBALL (use this
# file); UPGRADE_ALLOW_UNVERIFIED=1 (accept a base tarball without its .sha256, local only: CI verifies);
# UPGRADE_KEEP=1 (keep the scratch dir); UPGRADE_PACKAGE_FLAGS (`cargo pgrx package` flags, --debug).
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 <base-tag> [pg_config]" >&2
    exit 2
fi
BASE_TAG="$1"
PG_CONFIG="${2:-$(command -v pg_config || true)}"
[ -x "$PG_CONFIG" ] || { echo "FAIL: pg_config is not executable: ${PG_CONFIG:-<none on PATH>}" >&2; exit 2; }
unset PINECONE_API_KEY

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
UPGRADE_DIR="$HERE/upgrade"
CACHE="${UPGRADE_BASE_DIR:-/tmp/pg_pinecone-upgrade-base}"

BINDIR="$("$PG_CONFIG" --bindir)"
PKGLIBDIR="$("$PG_CONFIG" --pkglibdir)"
EXTDIR="$("$PG_CONFIG" --sharedir)/extension"
PG_VERSION="$("$PG_CONFIG" --version)"
PG_MAJOR="$(printf '%s' "$PG_VERSION" | sed -E 's/^PostgreSQL ([0-9]+).*/\1/')"
HEAD_VERSION="$(grep -m1 '^version = ' "$REPO_ROOT/pg_pinecone/Cargo.toml" | sed -E 's/version = "(.*)"/\1/')"
BASE_VERSION="${BASE_TAG#v}"
check_error="$(python3 "$UPGRADE_DIR/pick-base.py" check "$BASE_TAG" 2>&1)" || { echo "FAIL: $check_error" >&2; exit 1; }

case "$(uname -s)" in
    Linux) OS=linux ;;
    Darwin) OS=darwin ;;
    *) echo "FAIL: unsupported OS $(uname -s)" >&2; exit 2 ;;
esac
case "$(uname -m)" in
    x86_64 | amd64) ARCH=x86_64 ;;
    arm64 | aarch64) ARCH=arm64 ;;
    *) echo "FAIL: unsupported CPU $(uname -m)" >&2; exit 2 ;;
esac
ASSET="pg_pinecone-$BASE_VERSION-pg$PG_MAJOR-$OS-$ARCH.tar.gz"
TARBALL="${UPGRADE_BASE_TARBALL:-$CACHE/$BASE_TAG/$ASSET}"

WORK="$(mktemp -d /tmp/pg_pinecone-upgrade-drill.XXXXXX)"
PGDATA="$WORK/data"
SOCK="$WORK/sock"
PG_LOG="$WORK/postgres.log"
FAKE_LOG="$WORK/fake.log"
HEAD_PKG="$WORK/head-package"
DRILL_DB=drill
FRESH_DB=fresh
mkdir -p "$SOCK" "$WORK/stage"

free_ports() {
    python3 - <<'PY'
import socket


def bound(port=0):
    s = socket.socket()
    s.bind(("127.0.0.1", port))
    return s


while True:
    pg, fake = bound(), bound()
    try:
        ctl = bound(fake.getsockname()[1] + 1)
    except OSError:
        continue
    print(fake.getsockname()[1], pg.getsockname()[1])
    break
PY
}
read -r FAKE_PORT PG_PORT <<< "$(free_ports)"
FAKE_URL="http://127.0.0.1:$FAKE_PORT"
FAKE_CTL="http://127.0.0.1:$((FAKE_PORT + 1))"

pg_started=0
fake_pid=""

cleanup() {
    local status=$?
    set +e
    [ "$pg_started" -eq 1 ] && "$BINDIR/pg_ctl" -D "$PGDATA" -m immediate stop >/dev/null 2>&1
    [ -n "$fake_pid" ] && kill "$fake_pid" >/dev/null 2>&1
    if [ "$status" -ne 0 ]; then
        echo "---- postgres log (tail) ----" >&2
        tail -n 40 "$PG_LOG" >&2 2>/dev/null
        echo "FAIL: scratch dir kept for inspection: $WORK" >&2
    elif [ "${UPGRADE_KEEP:-0}" != 1 ]; then
        rm -rf "$WORK"
    fi
    exit "$status"
}
trap cleanup EXIT

step() { echo "== $*" >&2; }
pass() { echo "PASS: $*" >&2; }
fail() { echo "FAIL: $*" >&2; exit 1; }

psql_db() {
    local db="$1"
    shift
    "$BINDIR/psql" -X -q -At -h "$SOCK" -p "$PG_PORT" -U postgres -d "$db" -v ON_ERROR_STOP=1 "$@"
}
sql() { psql_db "$DRILL_DB" "$@"; }
scalar() { sql -c "$1"; }

wait_until() {
    local seconds="$1" what="$2" query="$3" deadline
    deadline=$((SECONDS + seconds))
    until [ "$(scalar "$query")" = "t" ]; do
        [ "$SECONDS" -lt "$deadline" ] || fail "timed out after ${seconds}s waiting for: $what"
        sleep 0.5
    done
}

SUDO=""
if [ ! -w "$PKGLIBDIR" ] || [ ! -w "$EXTDIR" ]; then
    SUDO="sudo"
fi

fetch_base() {
    if [ ! -f "$TARBALL" ]; then
        mkdir -p "$(dirname "$TARBALL")"
        (cd "$REPO_ROOT" && gh release download "$BASE_TAG" -p "$ASSET" -p "$ASSET.sha256" -D "$(dirname "$TARBALL")" --clobber) \
            || fail "could not download $ASSET of $BASE_TAG (gh release download)"
    fi
    # Same release, same channel as the tarball: this catches a corrupt download, not a tampered release.
    if [ -f "$TARBALL.sha256" ]; then
        if command -v sha256sum >/dev/null 2>&1; then check=(sha256sum -c); else check=(shasum -a 256 -c); fi
        (cd "$(dirname "$TARBALL")" && "${check[@]}" "$(basename "$TARBALL").sha256" >/dev/null) \
            || fail "checksum mismatch for $TARBALL"
    elif [ "${UPGRADE_ALLOW_UNVERIFIED:-0}" = 1 ]; then
        echo "WARN: no $(basename "$TARBALL").sha256: $TARBALL is NOT checksum-verified (UPGRADE_ALLOW_UNVERIFIED=1)" >&2
    else
        fail "no $(basename "$TARBALL").sha256 next to $TARBALL, so the base is unverified (UPGRADE_ALLOW_UNVERIFIED=1 accepts that)"
    fi
}

find_one() {
    local root="$1" what="$2" found count
    shift 2
    found="$(find "$root" -type f "$@")"
    count="$(printf '%s' "$found" | grep -c . || true)"
    [ "$count" -eq 1 ] || fail "expected exactly one $what in ${root##*/}, found $count"
    printf '%s\n' "$found"
}

# rm first: the running server keeps the old inode mapped, an in-place overwrite would crash it
put() {
    local mode="$1" src="$2" destdir="$3"
    $SUDO rm -f "$destdir/$(basename "$src")"
    $SUDO install -m "$mode" "$src" "$destdir/"
}

install_base() {
    tar -xzf "$TARBALL" -C "$WORK/stage"
    local lib control installed
    lib="$(find_one "$WORK/stage" 'pg_pinecone shared library' \( -name pg_pinecone.so -o -name pg_pinecone.dylib \))"
    control="$(find_one "$WORK/stage" 'pg_pinecone.control' -name pg_pinecone.control)"
    installed="$(sed -n "s/^default_version *= *'\(.*\)'.*/\1/p" "$control")"
    [ "$installed" = "$BASE_VERSION" ] || fail "$ASSET is pg_pinecone $installed, not $BASE_VERSION"
    $SUDO rm -f "$PKGLIBDIR"/pg_pinecone.so "$PKGLIBDIR"/pg_pinecone.dylib "$EXTDIR"/pg_pinecone.control "$EXTDIR"/pg_pinecone--*.sql
    put 755 "$lib" "$PKGLIBDIR"
    put 644 "$control" "$EXTDIR"
    while read -r script; do
        put 644 "$script" "$EXTDIR"
    done < <(find "$WORK/stage" -type f -name 'pg_pinecone--*.sql')
    $SUDO "$HERE/install-pgvector.sh" "$PG_CONFIG" >/dev/null 2>&1 || fail "pgvector is not installed for $PG_CONFIG"
}

build_head() {
    # shellcheck disable=SC2086
    (cd "$REPO_ROOT/pg_pinecone" && cargo pgrx package --pg-config "$PG_CONFIG" --out-dir "$HEAD_PKG" ${UPGRADE_PACKAGE_FLAGS:---debug}) \
        > "$WORK/package-head.log" 2>&1 \
        || { tail -n 30 "$WORK/package-head.log" >&2; fail "cargo pgrx package of this tree failed"; }
}

start_pg() {
    "$BINDIR/pg_ctl" -D "$PGDATA" -l "$PG_LOG" -w -t 60 start >/dev/null
    pg_started=1
}

stop_pg() {
    "$BINDIR/pg_ctl" -D "$PGDATA" -m fast -w -t 60 stop >/dev/null
    pg_started=0
}

init_cluster() {
    "$BINDIR/initdb" -D "$PGDATA" -U postgres -A trust -E UTF8 --locale=C --no-sync >/dev/null
    cat >> "$PGDATA/postgresql.conf" <<CONF
port = $PG_PORT
listen_addresses = ''
unix_socket_directories = '$SOCK'
shared_preload_libraries = 'pg_pinecone'
pinecone.worker_naptime_ms = 200
fsync = off
CONF
}

start_fake() {
    PYTHONDONTWRITEBYTECODE=1 python3 "$UPGRADE_DIR/fake_pinecone/fake_pinecone.py" "$FAKE_PORT" > "$FAKE_LOG" 2>&1 &
    fake_pid=$!
    disown "$fake_pid"
    for _ in $(seq 1 50); do
        curl -s --max-time 2 "$FAKE_CTL/__state" >/dev/null && return 0
        sleep 0.2
    done
    fail "the fake Pinecone did not start (see $FAKE_LOG)"
}

LANE_TABLES="('fx.t_trig', 'fx.t_am', 'fx.t_fts', 'fx.t_part')"
STATUS_TABLES="('fx.t_trig', 'fx.t_am', 'fx.t_fts', 'fx.t_part', 'fx.t_loc')"
LANES_READY="SELECT count(*) = 4 AND bool_and(CASE WHEN mode = 'docs'
    THEN docs_backlog = 0 AND docs_backfill_done = docs_backfill_total AND docs_queue_state = 'idle'
    ELSE mode = 'remote' AND backlog = 0 AND queue_state = 'idle' END)
    FROM pinecone.status() WHERE table_name IN $LANE_TABLES"
QUERY_VECTOR="'[1,2,3,4,5,6,7,8]'"
TID_AS_FAKE_ID="regexp_replace(ctid::text, '^\\((\\d+),(\\d+)\\)\$', '\\1:\\2')"
STATUS_KEYS="'table_name','mode','storage','vector_column','id_column','metric','dimension','index_name','namespace',
'metadata_columns','local_rows','backlog','queue_state','attached','docs_backlog','docs_backfill_done',
'docs_backfill_total','docs_failed','docs_queue_state','partition_column','partitions','failed_partitions','last_error'"

load_fixture() {
    sql -c "CREATE EXTENSION vector" -c "CREATE EXTENSION pg_pinecone" \
        -c "SELECT pinecone.set_api_key('drill-key', '$FAKE_URL')" >/dev/null
    sql >/dev/null <<'SQL'
CREATE SCHEMA fx;
CREATE TABLE fx.t_trig (id bigint PRIMARY KEY, tenant_id int NOT NULL, note text, embedding vector(8));
INSERT INTO fx.t_trig
  SELECT g, g % 5, 'note ' || g,
         (SELECT array_agg(((g*7 + k*13) % 17 - 8)::float4 + 0.01*k) FROM generate_series(1,8) k)::vector
  FROM generate_series(1,300) g;
CREATE TABLE fx.t_am (id bigint PRIMARY KEY, embedding vector(8));
INSERT INTO fx.t_am
  SELECT g, (SELECT array_agg(((g*5 + k*11) % 19 - 9)::float4 + 0.01*k) FROM generate_series(1,8) k)::vector
  FROM generate_series(1,400) g;
CREATE TABLE fx.t_fts (id bigint PRIMARY KEY, title text, body text, tenant_id int);
INSERT INTO fx.t_fts
  SELECT g, 'title ' || g, 'the quick brown fox number ' || g || ' jumps over the lazy dog ' || (g % 7), g % 3
  FROM generate_series(1,250) g;
CREATE TABLE fx.t_nat (id bigint PRIMARY KEY, tenant_id int NOT NULL, embedding vector(8));
INSERT INTO fx.t_nat
  SELECT g, g % 4, (SELECT array_agg(((g*37 + k*k*11 + g*k) % 211 - 105)::float4 + 0.01*k) FROM generate_series(1,8) k)::vector
  FROM generate_series(1,600) g;
CREATE TABLE fx.t_part (id bigint PRIMARY KEY, tenant_id int NOT NULL, embedding vector(8));
INSERT INTO fx.t_part
  SELECT g, g % 4, (SELECT array_agg(((g*11 + k*3) % 29 - 14)::float4 + 0.01*k) FROM generate_series(1,8) k)::vector
  FROM generate_series(1,200) g;
CREATE TABLE fx.t_loc (id bigint PRIMARY KEY, embedding vector(8));
INSERT INTO fx.t_loc
  SELECT g, (SELECT array_agg(((g*13 + k*5) % 31 - 15)::float4 + 0.01*k) FROM generate_series(1,8) k)::vector
  FROM generate_series(1,60) g;
SELECT pinecone.promote('fx.t_trig', metadata_columns => ARRAY['tenant_id']);
SELECT pinecone.promote('fx.t_part', partition_column => 'tenant_id');
SELECT pinecone.track('fx.t_loc');
SELECT pinecone.create_index('fx.t_am', 'embedding', engine => 'remote');
SELECT pinecone.track_docs('fx.t_fts', '["title","body"]'::jsonb, metadata_columns => ARRAY['tenant_id']);
CREATE INDEX t_nat_idx ON fx.t_nat USING pinecone (embedding vector_cosine_ops, tenant_id int4_ops) WITH (mode = 'local', small_n_threshold = 50);
SELECT pinecone.schedule_maintenance('fx.t_nat_idx', 0.2);
SQL
    wait_until 90 "the four Pinecone lanes to drain" "$LANES_READY"
    sql -c "SELECT pinecone.flush('fx.t_trig')" -c "SELECT pinecone.flush('fx.t_am')" \
        -c "SELECT pinecone.flush_docs('fx.t_fts')" >/dev/null
}

lane_ids() {
    local table="$1" kind="$2" partitioned="${3:-0}" index namespace
    IFS='|' read -r index namespace <<< "$(sql -F '|' -c "SELECT index_name, namespace FROM pinecone.status() WHERE table_name = '$table'")"
    curl -s --max-time 30 "$FAKE_CTL/__dump?index=$index&kind=$kind" > "$WORK/dump.json"
    python3 - "$namespace" "$WORK/dump.json" "$partitioned" <<'PY'
import json
import sys

namespace, dump, partitioned = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3] == "1"
print("\n".join(i for ns, ids in dump.items() if ns == namespace or (partitioned and ns.startswith(namespace + ":")) for i in ids))
PY
}

assert_fake_matches_heap() {
    local label="$1" lane heap fake
    for lane in "trig|fx.t_trig|vec|id" "am|fx.t_am|vec|$TID_AS_FAKE_ID" "fts|fx.t_fts|doc|id" "part|fx.t_part|vec|id|1"; do
        IFS='|' read -r _ table kind expr partitioned <<< "$lane"
        heap="$(sql -c "SELECT $expr::text FROM $table ORDER BY 1")"
        fake="$(lane_ids "$table" "$kind" "${partitioned:-0}" | sort)"
        [ "$(printf '%s\n' "$heap" | sort)" = "$fake" ] || fail "$label: the fake Pinecone's ids for $table differ from the table's rows"
    done
    pass "$label: the fake Pinecone holds exactly the table's ids for the promoted, partitioned, access-method and full-text lanes"
}

fixture_schema() {
    "$BINDIR/pg_dump" -h "$SOCK" -p "$PG_PORT" -U postgres -d "$DRILL_DB" --schema-only -n fx > "$1.raw"
    python3 "$UPGRADE_DIR/normalize-dump.py" "$1.raw" > "$1"
}

# The access method's metapage format word (local/metapage.rs FORMAT_OFFSET): the last word of the payload,
# after the 24-byte page header. A base before 0.10 never wrote it, so its pages hold zero there.
FORMAT_INDEXES="fx.t_am_embedding_pinecone_idx fx.t_nat_idx"
FORMAT_WORD_AT=108
# What a build by this tree writes there: format 3 in the low byte and the list page layout (1) in the next
# (local/metapage.rs Stored::CURRENT), 3 + 256.
HEAD_FORMAT_WORD=259

format_word() {
    sql -c "SELECT get_byte(p, $FORMAT_WORD_AT) + (get_byte(p, $FORMAT_WORD_AT + 1) << 8)
        + (get_byte(p, $FORMAT_WORD_AT + 2) << 16) + (get_byte(p, $FORMAT_WORD_AT + 3) << 24)
        FROM get_raw_page('$1', 'main', 0) p"
}

format_words() {
    local index
    for index in "$@"; do echo "$index $(format_word "$index")"; done
}

index_format_of() {
    scalar "SELECT format('%s|%s', format_version, readable) FROM pinecone.index_format('$1')"
}

# no default reported | nothing measured | samples | no target: `t|t|0|t` for an index with no calibration
calibration_of() {
    scalar "SELECT format('%s|%s|%s|%s', nprobe IS NULL, calib_nprobe IS NULL, calib_samples, target_recall IS NULL) FROM pinecone.index_calibration('$1')"
}

version_lt() {
    [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)" = "$1" ]
}

# snapshot <name> [native]: the native index's search and plan only while it is readable (`native`).
snapshot() {
    local dir="$WORK/$1" native="${2:-}"
    mkdir -p "$dir"
    sql -c "SELECT s.table_name || ' ' || (SELECT jsonb_object_agg(k, v) FROM jsonb_each(to_jsonb(s)) e(k, v)
            WHERE k IN ($STATUS_KEYS)) FROM pinecone.status() s WHERE table_name IN $STATUS_TABLES ORDER BY 1" > "$dir/status.txt"
    sql -c "SELECT 'sync ' || t || ' ' || status FROM (VALUES ('fx.t_trig'), ('fx.t_am'), ('fx.t_part')) v(t),
            LATERAL pinecone.backup_readiness(v.t::regclass) WHERE \"check\" = 'sync' ORDER BY 1" > "$dir/readiness.txt"
    sql -c "SELECT index_name || ' ' || drift_threshold FROM pinecone.maintenance_status() ORDER BY 1" > "$dir/maintenance.txt"
    {
        echo "== topk"
        sql -c "SELECT id::text FROM pinecone.topk('fx.t_trig', ARRAY[1,2,3,4,5,6,7,8]::real[], 5) ORDER BY distance, id"
        echo "== access method, remote"
        PGOPTIONS='-c enable_seqscan=off' sql -c "SELECT id FROM fx.t_am ORDER BY embedding <=> $QUERY_VECTOR LIMIT 5"
        echo "== search_docs"
        sql -c "SELECT id FROM pinecone.search_docs('fx.t_fts', 'quick fox', k => 5) ORDER BY rank, id"
        echo "== topk, partitioned"
        sql -c "SELECT id::text FROM pinecone.topk('fx.t_part', ARRAY[1,2,3,4,5,6,7,8]::real[], 5, partition => 2) ORDER BY distance, id"
        echo "== topk, local"
        sql -c "SELECT id::text FROM pinecone.topk('fx.t_loc', ARRAY[1,2,3,4,5,6,7,8]::real[], 5) ORDER BY distance, id"
        echo "== partitions"
        sql -c "SELECT lane || ' ' || value || ' ' || namespace || ' ' || (failed_at IS NULL) FROM pinecone.partitions_of('fx.t_part') ORDER BY 1"
    } > "$dir/searches.txt"
    if [ "$native" = native ]; then
        PGOPTIONS='-c enable_seqscan=off' sql -c "SELECT id FROM fx.t_nat WHERE tenant_id = 2 ORDER BY embedding <=> $QUERY_VECTOR LIMIT 5" > "$dir/native.txt"
        [ -s "$dir/native.txt" ] || fail "$1: the native search came back empty"
    fi
    fixture_schema "$dir/fixture-schema.txt"
    [ "$(grep -c . "$dir/status.txt")" -eq 5 ] || fail "$1: status() should list the four Pinecone lanes and the local table"
    for object in "TRIGGER fx.t_part pg_pinecone_sync_ins" "TRIGGER fx.t_fts pg_pinecone_docs_ins" "INDEX fx.t_am_embedding_pinecone_idx" "INDEX fx.t_nat_idx"; do
        grep -q "^== $object\$" "$dir/fixture-schema.txt" || fail "$1: the dump of the fixture schema lacks $object"
    done
    for lane in fx.t_trig fx.t_am fx.t_part; do
        grep -q "^sync $lane ok$" "$dir/readiness.txt" || fail "$1: the backup_readiness() sync check of $lane is not ok"
    done
    local lanes=("id FROM fx.t_am ORDER BY embedding <=> $QUERY_VECTOR LIMIT 5|t_am_embedding_pinecone_idx")
    if [ "$native" = native ]; then
        lanes+=("id FROM fx.t_nat WHERE tenant_id = 2 ORDER BY embedding <=> $QUERY_VECTOR LIMIT 5|t_nat_idx")
    fi
    for lane in "${lanes[@]}"; do
        plan="$(PGOPTIONS='-c enable_seqscan=off' sql -c "EXPLAIN (COSTS OFF) SELECT ${lane%%|*}")"
        printf '%s' "$plan" | grep -q "${lane##*|}" || fail "$1: the planner does not use ${lane##*|}"
    done
    awk '/^==/ { if (seen && !rows) bad = 1; seen = 1; rows = 0; next } { rows++ } END { exit (bad || !rows) }' "$dir/searches.txt" \
        || fail "$1: a search came back empty"
}

assert_same_snapshot() {
    local what="$1" f
    for f in status.txt readiness.txt maintenance.txt searches.txt fixture-schema.txt; do
        diff -u "$WORK/before/$f" "$WORK/$what/$f" > "$WORK/$what/$f.diff" \
            || { cat "$WORK/$what/$f.diff" >&2; fail "$what: $f differs from the pre-upgrade snapshot"; }
    done
    pass "$what: status(), backup_readiness(), maintenance_status(), every search and the fixture's schema equal the pre-upgrade snapshot"
}

dump_catalog() {
    local db="$1" out="$2"
    # --binary-upgrade is the only mode in which pg_dump prints an extension's member objects
    "$BINDIR/pg_dump" -h "$SOCK" -p "$PG_PORT" -U postgres -d "$db" --schema-only --binary-upgrade -e pg_pinecone -N fx > "$out.raw"
    python3 "$UPGRADE_DIR/normalize-dump.py" "$out.raw" > "$out"
    psql_db "$db" -c "SELECT 'extconfig: ' || coalesce(string_agg(c::regclass::text || ' [' || coalesce(cond, '') || ']', ', '
        ORDER BY c::regclass::text), '') FROM pg_extension e, unnest(e.extconfig, e.extcondition) AS u(c, cond)
        WHERE e.extname = 'pg_pinecone'" >> "$out"
}

assert_same_catalog() {
    psql_db postgres -c "CREATE DATABASE $FRESH_DB" >/dev/null
    psql_db "$FRESH_DB" -c "CREATE EXTENSION vector" -c "CREATE EXTENSION pg_pinecone" >/dev/null
    dump_catalog "$FRESH_DB" "$WORK/catalog-fresh.txt"
    dump_catalog "$DRILL_DB" "$WORK/catalog-upgraded.txt"
    [ "$(grep -c '^== ' "$WORK/catalog-fresh.txt")" -gt 100 ] || fail "the extension dump of the fresh database is implausibly small"
    diff -u "$WORK/catalog-fresh.txt" "$WORK/catalog-upgraded.txt" > "$WORK/catalog.diff" \
        || { head -n 80 "$WORK/catalog.diff" >&2; fail "the upgraded catalog differs from a fresh install of $HEAD_VERSION (full diff: $WORK/catalog.diff)"; }
    pass "the upgraded catalog equals a fresh install of $HEAD_VERSION ($(grep -c '^== ' "$WORK/catalog-fresh.txt") extension objects, pg_dump --schema-only normalized)"
}

assert_worker_version() {
    local compiled installed idle
    IFS='|' read -r compiled installed idle <<< "$(scalar "SELECT compiled, installed, idle FROM pinecone.worker_version()")"
    [ "$compiled|$installed|$idle" = "$1|$2|$3" ] \
        || fail "worker_version() is $compiled|$installed|idle=$idle, expected $1|$2|idle=$3"
    pass "worker_version(): compiled $compiled, installed $installed, idle $idle"
}

IDLE_ID=9001

queue_write_while_idle() {
    sql -c "INSERT INTO fx.t_trig (id, tenant_id, note, embedding) VALUES ($IDLE_ID, 1, 'queued while idle', $QUERY_VECTOR::vector)" >/dev/null
    sleep 2
    [ "$(scalar "SELECT backlog > 0 FROM pinecone.status() WHERE table_name = 'fx.t_trig'")" = "t" ] \
        || fail "a worker whose version differs from the extension's drained a write"
    pass "a write queued while the worker idles stays queued"
}

drain_queued_write() {
    wait_until 60 "the write queued while the worker idled to drain" \
        "SELECT backlog = 0 FROM pinecone.status() WHERE table_name = 'fx.t_trig'"
    assert_fake_matches_heap "after the worker resumed"
    sql -c "DELETE FROM fx.t_trig WHERE id = $IDLE_ID" >/dev/null
    wait_until 60 "the delete to drain" "$LANES_READY"
}

# Indexes the base built read as format 1 (a zero word: absent) and keep the word they were built with: the
# remote one stays readable, the native one is not (its sketch codes are offline until REINDEX).
assert_formats_unchanged() {
    local label="$1" index
    # shellcheck disable=SC2086
    format_words $FORMAT_INDEXES > "$WORK/format-words-$label.txt"
    diff -u "$WORK/format-words-before.txt" "$WORK/format-words-$label.txt" >&2 \
        || fail "$label: the format word of an access-method index the base built changed"
    for index in $FORMAT_INDEXES; do
        echo "$index $(index_format_of "$index")"
    done > "$WORK/format-reads-$label.txt"
    awk '{ print $1, (($2 > 1 ? $2 : 1) ($1 ~ /t_nat_idx$/ ? "|f" : "|t")) }' "$WORK/format-words-before.txt" \
        | diff -u - "$WORK/format-reads-$label.txt" >&2 \
        || fail "$label: pinecone.index_format() does not report the base's remote index as readable and its native one as not, a zero word as format 1"
    pass "$label: the access-method indexes $BASE_VERSION built read through pinecone.index_format() with the format word they were built with (the remote one readable, the native one offline)"
}

# REINDEX writes this release's format 3, and so does a new index (the native index the helper rebuilt
# already did).
assert_formats_rebuilt() {
    local index
    sql -c "CREATE INDEX t_nat_idx_fresh ON fx.t_nat USING pinecone (embedding vector_l2_ops) WITH (mode = 'local', small_n_threshold = 50)" \
        -c "REINDEX INDEX fx.t_am_embedding_pinecone_idx" >/dev/null
    wait_until 90 "the REINDEX of the access-method remote index to drain" "$LANES_READY"
    assert_fake_matches_heap "after the REINDEX of an access-method index"
    for index in fx.t_nat_idx_fresh fx.t_nat_idx fx.t_am_embedding_pinecone_idx; do
        [ "$(index_format_of "$index")" = "3|t" ] || fail "$index is not format 3 after a build by $HEAD_VERSION: $(index_format_of "$index")"
        [ "$(format_word "$index")" = "$HEAD_FORMAT_WORD" ] || fail "$index stores format word $(format_word "$index") after a build by $HEAD_VERSION, not $HEAD_FORMAT_WORD"
    done
    pass "a new index, the helper's REINDEX of the base's native index and the REINDEX of its remote access-method index write format 3"
}

# The native index the base built (clustered) is offline from the restart until REINDEX, per entry point.
# `format` also asks pinecone.index_format(), which exists only after ALTER EXTENSION UPDATE.
assert_native_refused() {
    local label="$1" format="${2:-}" out plan
    out="$(sql -c "EXPLAIN (COSTS OFF) SELECT id FROM fx.t_nat ORDER BY embedding <=> $QUERY_VECTOR LIMIT 5" 2>&1)" \
        || fail "$label: EXPLAIN over a table with the offline native index failed: $out"
    plan="$(printf '%s\n' "$out" | grep -v -E '^(WARNING|HINT|NOTICE|DETAIL):')"
    if ! printf '%s' "$plan" | grep -q 'Seq Scan' || printf '%s' "$plan" | grep -q 't_nat_idx'; then
        fail "$label: the planner does not walk away from the offline native index: $out"
    fi
    printf '%s' "$out" | grep -q "WARNING:  pg_pinecone index 't_nat_idx' has format version 1 (this build reads 3): REINDEX it" \
        || fail "$label: the planner did not warn about the offline native index: $out"
    if out="$(PGOPTIONS='-c enable_seqscan=off' sql -v VERBOSITY=verbose -c "SELECT id FROM fx.t_nat ORDER BY embedding <=> $QUERY_VECTOR LIMIT 5" 2>&1)"; then
        fail "$label: a scan of the offline native index succeeded: $out"
    fi
    if ! printf '%s' "$out" | grep -q "55000" \
        || ! printf '%s' "$out" | grep -q "pg_pinecone index 't_nat_idx' has format version 1 (this build reads 3): REINDEX it"; then
        fail "$label: a forced scan of the offline native index did not raise 55000 with the REINDEX text: $out"
    fi
    if out="$(sql -v VERBOSITY=verbose -c "INSERT INTO fx.t_nat VALUES (9100, 1, $QUERY_VECTOR::vector)" 2>&1)"; then
        fail "$label: an insert into a table with the offline native index succeeded: $out"
    fi
    if ! printf '%s' "$out" | grep -q "55000" \
        || ! printf '%s' "$out" | grep -q "has format version 1 (this build reads 3): REINDEX it"; then
        fail "$label: an insert did not raise 55000 with the REINDEX text: $out"
    fi
    out="$(sql -c "VACUUM fx.t_nat" 2>&1)" || fail "$label: VACUUM of a table with the offline native index failed: $out"
    printf '%s' "$out" | grep -q "WARNING:  pg_pinecone index 't_nat_idx' has format version 1 .*REINDEX it" \
        || fail "$label: VACUUM did not warn about the offline native index: $out"
    sql -c "SELECT * FROM pinecone.index_drift('fx.t_nat_idx')" -c "SELECT * FROM pinecone.maintenance_status()" >/dev/null 2>&1 \
        || fail "$label: index_drift or maintenance_status() raised on the offline native index"
    if [ -n "$format" ]; then
        [ "$(index_format_of fx.t_nat_idx)" = "1|f" ] \
            || fail "$label: pinecone.index_format() reports $(index_format_of fx.t_nat_idx) for the offline native index, not 1|f"
        [ "$(calibration_of fx.t_nat_idx)" = "t|t|0|t" ] \
            || fail "$label: pinecone.index_calibration() reports $(calibration_of fx.t_nat_idx) for the offline native index, not t|t|0|t"
    fi
    pass "$label: the native index $BASE_VERSION built is offline: the planner walks away with a WARNING, a forced scan and an insert raise 55000, VACUUM warns and completes"
}

# pinecone.reindex_native_indexes() rebuilds exactly the native index (a dry run only lists it as stale), the
# remote one is never touched, a second call is a no-op, and the rebuilt index scans and takes inserts.
assert_native_repaired() {
    local listed
    listed="$(scalar "SELECT coalesce(string_agg(n.index_name || ':' || n.outcome, ','), '') FROM pinecone.reindex_native_indexes(true) n")"
    [ "$listed" = "fx.t_nat_idx:stale" ] || fail "reindex_native_indexes(dry_run => true) lists '$listed', not just fx.t_nat_idx:stale"
    [ "$(index_format_of fx.t_nat_idx)" = "1|f" ] || fail "reindex_native_indexes(dry_run => true) changed the native index"
    listed="$(scalar "SELECT coalesce(string_agg(n.index_name || ':' || n.outcome, ','), '') FROM pinecone.reindex_native_indexes() n")"
    [ "$listed" = "fx.t_nat_idx:rebuilt" ] || fail "reindex_native_indexes() returned '$listed', not just fx.t_nat_idx:rebuilt"
    [ "$(index_format_of fx.t_nat_idx)" = "3|t" ] || fail "the native index is $(index_format_of fx.t_nat_idx) after reindex_native_indexes(), not 3|t"
    [ "$(scalar "SELECT count(*) FROM pinecone.reindex_native_indexes()")" = 0 ] || fail "a second reindex_native_indexes() was not a no-op"
    [ "$(index_format_of fx.t_am_embedding_pinecone_idx)" = "1|t" ] \
        || fail "reindex_native_indexes() touched the remote access-method index: $(index_format_of fx.t_am_embedding_pinecone_idx)"
    [ "$(scalar "SELECT calib_samples >= 50 AND calib_nprobe >= 1 AND calib_pool >= 100 AND calib_recall > 0 AND abs(target_recall - 0.95) < 1e-6 AND nprobe = calib_nprobe AND pool = calib_pool FROM pinecone.index_calibration('fx.t_nat_idx')")" = t ] \
        || fail "the rebuilt native index reports no calibration: $(scalar "SELECT row(i.*) FROM pinecone.index_calibration('fx.t_nat_idx') i")"
    [ "$(calibration_of fx.t_am_embedding_pinecone_idx)" = "t|t|0|t" ] \
        || fail "pinecone.index_calibration() reports a calibration for the remote access-method index: $(calibration_of fx.t_am_embedding_pinecone_idx)"
    snapshot reindexed native
    [ "$(PGOPTIONS='-c enable_seqscan=off' scalar "SELECT count(*) FROM (SELECT id FROM fx.t_nat WHERE tenant_id = 2 ORDER BY embedding <=> $QUERY_VECTOR LIMIT 5) q")" = 5 ] \
        || fail "the rebuilt native index does not answer"
    pass "reindex_native_indexes() rebuilt fx.t_nat_idx (format 3, scans work) and left the remote access-method index alone"
}

new_writes_drain() {
    sql >/dev/null <<'SQL'
INSERT INTO fx.t_trig SELECT 2000 + g, g % 5, 'new ' || g, '[8,7,6,5,4,3,2,1]'::vector FROM generate_series(1, 20) g;
DELETE FROM fx.t_trig WHERE id <= 5;
UPDATE fx.t_trig SET embedding = '[1,1,1,1,1,1,1,1]' WHERE id BETWEEN 6 AND 10;
INSERT INTO fx.t_am SELECT 2000 + g, '[8,7,6,5,4,3,2,1]'::vector FROM generate_series(1, 20) g;
INSERT INTO fx.t_fts SELECT 2000 + g, 'new ' || g, 'quick fox appended ' || g, g % 3 FROM generate_series(1, 20) g;
DELETE FROM fx.t_fts WHERE id <= 5;
INSERT INTO fx.t_part SELECT 2000 + g, g % 4, '[8,7,6,5,4,3,2,1]'::vector FROM generate_series(1, 20) g;
INSERT INTO fx.t_part VALUES (3000, 9, '[1,1,1,1,1,1,1,1]');
DELETE FROM fx.t_part WHERE id <= 5;
SQL
    wait_until 60 "the new writes of all four lanes to drain" "$LANES_READY"
    assert_fake_matches_heap "after new writes"
}

# The rebuilt native index takes inserts again, and answers with them.
native_writes() {
    sql -c "INSERT INTO fx.t_nat SELECT 2000 + g, g % 4, '[8,7,6,5,4,3,2,1]'::vector FROM generate_series(1, 20) g" >/dev/null
    [ "$(PGOPTIONS='-c enable_seqscan=off' scalar "SELECT count(*) FROM (SELECT id FROM fx.t_nat WHERE tenant_id = 2
        ORDER BY embedding <=> '[8,7,6,5,4,3,2,1]' LIMIT 5) q WHERE id > 2000")" = 5 ] \
        || fail "the rebuilt native index does not return the rows inserted after the REINDEX"
    [ "$(scalar "SELECT inserted_since_build > 0 FROM pinecone.index_drift('fx.t_nat_idx')")" = t ] \
        || fail "the native inserts did not bump the rebuilt index's counter"
    pass "after the REINDEX: the native index takes inserts and returns them"
}

# update-first cannot create a C function the old library lacks: the session that runs the UPDATE still
# has the old library mapped, and Postgres resolves the new symbol there
ORDER="${UPGRADE_ORDER:-restart-first}"
case "$ORDER" in
    update-first | restart-first) ;;
    *) fail "UPGRADE_ORDER must be update-first or restart-first, not $ORDER" ;;
esac
[ "$BASE_VERSION" != "$HEAD_VERSION" ] || fail "$BASE_TAG is this tree's own version ($HEAD_VERSION): nothing to upgrade"

restart_pg() {
    stop_pg
    start_pg
}

install_head_while_running() {
    local lib control script
    lib="$(find_one "$HEAD_PKG" 'pg_pinecone shared library' \( -name pg_pinecone.so -o -name pg_pinecone.dylib \))"
    control="$(find_one "$HEAD_PKG" 'pg_pinecone.control' -name pg_pinecone.control)"
    put 755 "$lib" "$PKGLIBDIR"
    put 644 "$control" "$EXTDIR"
    while read -r script; do
        put 644 "$script" "$EXTDIR"
    done < <(find "$HEAD_PKG" -type f -name 'pg_pinecone--*.sql')
    for script in "$REPO_ROOT"/pg_pinecone/sql/pg_pinecone--*--*.sql; do
        [ -f "$EXTDIR/${script##*/}" ] || fail "cargo pgrx package did not carry ${script##*/}"
    done
    [ -f "$EXTDIR/pg_pinecone--$HEAD_VERSION.sql" ] || fail "cargo pgrx package did not write pg_pinecone--$HEAD_VERSION.sql"
}

step "pg_pinecone $BASE_VERSION (release $ASSET) on $PG_VERSION"
fetch_base
build_head
install_base
init_cluster
start_pg
start_fake
psql_db postgres -c "CREATE DATABASE $DRILL_DB" >/dev/null
load_fixture
[ "$(scalar "SELECT extversion FROM pg_extension WHERE extname = 'pg_pinecone'")" = "$BASE_VERSION" ] \
    || fail "the fixture database is not at $BASE_VERSION"
assert_worker_version "$BASE_VERSION" "$BASE_VERSION" f
assert_fake_matches_heap "before the upgrade"
snapshot before native
sql -c "CREATE EXTENSION IF NOT EXISTS pageinspect" >/dev/null
# shellcheck disable=SC2086
format_words $FORMAT_INDEXES > "$WORK/format-words-before.txt"
if version_lt "$BASE_VERSION" 0.10.0; then
    [ "$(awk '$2 != 0' "$WORK/format-words-before.txt" | wc -l)" -eq 0 ] \
        || { cat "$WORK/format-words-before.txt" >&2; fail "an access-method index built by $BASE_VERSION has a nonzero format word"; }
    pass "the access-method indexes $BASE_VERSION built have no format word (zero in the page)"
fi

step "install $HEAD_VERSION files into the running server's install ($ORDER)"
install_head_while_running
if [ "$ORDER" = restart-first ]; then
    restart_pg
    assert_worker_version "$HEAD_VERSION" "$BASE_VERSION" t
    snapshot window
    assert_same_snapshot window
    assert_native_refused "in the window between restart and update"
    queue_write_while_idle
fi
sql -c "ALTER EXTENSION pg_pinecone UPDATE" >/dev/null 2> "$WORK/update.stderr" \
    || { cat "$WORK/update.stderr" >&2; fail "ALTER EXTENSION pg_pinecone UPDATE failed in $ORDER order (update-first cannot create C functions the old library lacks)"; }
[ "$(scalar "SELECT extversion FROM pg_extension WHERE extname = 'pg_pinecone'")" = "$HEAD_VERSION" ] \
    || fail "ALTER EXTENSION pg_pinecone UPDATE did not reach $HEAD_VERSION"
pass "ALTER EXTENSION pg_pinecone UPDATE: $BASE_VERSION -> $HEAD_VERSION"
grep -q "WARNING:  pg_pinecone 0.10: native indexes built before format 3 are offline until REINDEX: fx.t_nat_idx$" "$WORK/update.stderr" \
    || { cat "$WORK/update.stderr" >&2; fail "the update did not list the native index that needs REINDEX in a WARNING"; }
pass "the update's WARNING lists the native index that needs REINDEX (and only it)"
if [ "$ORDER" = update-first ]; then
    assert_worker_version "$BASE_VERSION" "$HEAD_VERSION" t
    snapshot window
    assert_same_snapshot window
    queue_write_while_idle
    restart_pg
fi
assert_worker_version "$HEAD_VERSION" "$HEAD_VERSION" f
drain_queued_write
snapshot after
assert_same_snapshot after
assert_same_catalog
assert_formats_unchanged "after the upgrade"
assert_native_refused "after the update" format
new_writes_drain
assert_formats_unchanged "after new writes"
assert_native_repaired
native_writes
assert_formats_rebuilt

pass "in-place upgrade $BASE_TAG -> $HEAD_VERSION ($ORDER) on $PG_VERSION, ${SECONDS}s"
