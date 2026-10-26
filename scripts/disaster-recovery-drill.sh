#!/usr/bin/env bash
# LIVE disaster-recovery drill for pg_pinecone
# (docs/disaster-recovery.md): restore -> repoint -> recover_from_remote, the
# joint Postgres+Pinecone restore, plus the AM-path concurrent-writer/quiescence
# probe. Version-agnostic: asserts the installed extension matches the crate's
# own Cargo version, so it tracks releases instead of pinning one. Precedent:
# scripts/vacuum-drill.sh / scripts/install-drill.sh (shared-pgrx-instance
# restart pattern, API key via stdin heredoc only — never argv).
#
# Needs PINECONE_API_KEY in the environment. Creates ONLY indexes/tables
# tagged pg_pinecone-drill-* and deletes them all on exit (see cleanup()).
#
# PG_PINECONE_DRILL_LEG6_ONLY=1 runs just SETUP-lite (install/restart/db/key) +
# the CONCURRENT leg + TEARDOWN — the re-verification mode for the leg-6
# remediation fixes (F1/F2), skipping the trigger-path legs that already
# passed.
set -uo pipefail

if [ -z "${PINECONE_API_KEY:-}" ]; then
    echo "PINECONE_API_KEY is not set — the disaster-recovery drill cannot run live." >&2
    exit 1
fi

LEG6_ONLY="${PG_PINECONE_DRILL_LEG6_ONLY:-0}"

LOCK=/tmp/pg_pinecone-pgrx.lock
if [ "${1:-}" != "--locked" ]; then
    exec lockf -t 900 "$LOCK" "$0" --locked
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PG_PINECONE_DIR="$REPO_ROOT/pg_pinecone"
CLIENT_DIR="$REPO_ROOT/pg_pinecone-client"
EXPECT_VER="$(grep -m1 '^version' "$PG_PINECONE_DIR/Cargo.toml" | sed -E 's/.*"([^"]+)".*/\1/')"
PG_CONFIG="$(cd "$PG_PINECONE_DIR" && cargo pgrx info pg-config pg17)"
PGRX_HOME="${PGRX_HOME:-$HOME/.pgrx}"
PGDATA="$PGRX_HOME/data-17"
SERVER_LOG="$PGRX_HOME/17.log"

PID=$$
DBNAME="pg_pinecone_dr_drill_$PID"
TABLE_A="drill_docs_trigger"
TABLE_B="pg_pinecone_drill_b_$PID"
INDEX_A="pg_pinecone-drill-a-$PID"
INDEX_RESTORED="pg_pinecone-drill-restored-$PID"
INDEX_MISMATCH="pg_pinecone-drill-mismatch-$PID"
TABLE_C="pg_pinecone_drill_c_$PID"
DBNAME_J="pg_pinecone_dr_drill_joint_$PID"
INDEX_C_RESTORED="pg_pinecone-drill-c-restored-$PID"
PGBIN="$(dirname "$PG_CONFIG")"
BACKUP_NAME="pg_pinecone-drill-backup-$PID"
API_VERSION="2026-04"
BACKUP_API_VERSION="2026-07"
CONTROL_PLANE_URL="https://api.pinecone.io"
DIM=16
N_ROWS=800

echo "== disaster-recovery-drill: db=$DBNAME table_a=$TABLE_A table_b=$TABLE_B pg_config=$PG_CONFIG ==" >&2

# ---------------------------------------------------------------------------
# Bookkeeping + reporting
# ---------------------------------------------------------------------------
# macOS ships bash 3.2 (no associative arrays) — per-leg RESULT_<LEG>/
# NOTE_<LEG> scalars, set/read via printf -v / indirect (${!name})
# expansion instead of declare -A.
LEG_ORDER=(SETUP BACKUP RESTORE REPOINT RECOVER CONCURRENT JOINT TEARDOWN)
CREATED_INDEXES=()
BACKUP_ID=""

record() {
    printf -v "RESULT_$1" '%s' "$2"
    printf -v "NOTE_$1" '%s' "$3"
    echo "== leg $1: $2 -- $3 ==" >&2
}

# ---------------------------------------------------------------------------
# Instance lifecycle helpers (vacuum-drill.sh / install-drill.sh pattern)
# ---------------------------------------------------------------------------
was_running=0
orig_worker_enabled=""
orig_max_attempts=""
PORT=""

pg_port() { sed -n '4p' "$PGDATA/postmaster.pid"; }

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
    record TEARDOWN RUNNING "deleting drill indexes, dropping test DB(s), restoring instance state"

    if [ -n "$PORT" ]; then
        if [ -n "$orig_worker_enabled" ]; then
            psql -h 127.0.0.1 -p "$PORT" -U "$(whoami)" -d postgres -v ON_ERROR_STOP=1 \
                -c "ALTER SYSTEM SET pinecone.worker_enabled = '$orig_worker_enabled'" \
                -c "SELECT pg_reload_conf()" >/dev/null 2>&1
        fi
        if [ -n "$orig_max_attempts" ]; then
            psql -h 127.0.0.1 -p "$PORT" -U "$(whoami)" -d postgres -v ON_ERROR_STOP=1 \
                -c "ALTER SYSTEM SET pinecone.outbox_max_attempts = $orig_max_attempts" \
                -c "SELECT pg_reload_conf()" >/dev/null 2>&1
        fi
        psql -h 127.0.0.1 -p "$PORT" -U "$(whoami)" -d postgres \
            -c "DROP DATABASE IF EXISTS \"$DBNAME\" WITH (FORCE)" >/dev/null 2>&1
        psql -h 127.0.0.1 -p "$PORT" -U "$(whoami)" -d postgres \
            -c "DROP DATABASE IF EXISTS \"$DBNAME_J\" WITH (FORCE)" >/dev/null 2>&1
    fi

    # bash 3.2: "${arr[@]}" on a possibly-empty array throws "unbound
    # variable" under set -u; "${arr[@]+"${arr[@]}"}" is the portable guard
    # (expands to nothing at all when empty, unlike "${arr[@]:-}" which
    # yields one bogus empty-string element).
    for idx in "${CREATED_INDEXES[@]+"${CREATED_INDEXES[@]}"}"; do
        echo "deleting Pinecone index: $idx" >&2
        curl -fsS -X DELETE "$CONTROL_PLANE_URL/indexes/$idx" \
            -H "Api-Key: $PINECONE_API_KEY" -H "X-Pinecone-Api-Version: $API_VERSION" >/dev/null 2>&1
    done
    if [ -n "$BACKUP_ID" ]; then
        echo "deleting Pinecone backup: $BACKUP_ID" >&2
        curl -fsS -X DELETE "$CONTROL_PLANE_URL/backups/$BACKUP_ID" \
            -H "Api-Key: $PINECONE_API_KEY" -H "X-Pinecone-Api-Version: $BACKUP_API_VERSION" >/dev/null 2>&1
    fi

    echo "== verifying account is clean of drill artifacts ==" >&2
    # DELETE /indexes/{name} is ASYNC (202) — the index lingers in the list
    # for a while after a successful delete. The first drill run flagged
    # perfectly-deleted indexes as leftovers because it checked immediately
    #; poll until they disappear, and only call it a leftover if
    # they are still listed after the grace window.
    leftover=""
    for _ in $(seq 1 45); do
        leftover="$(curl -fsS "$CONTROL_PLANE_URL/indexes" -H "Api-Key: $PINECONE_API_KEY" \
            -H "X-Pinecone-Api-Version: $API_VERSION" 2>/dev/null \
            | jq -r '.indexes[]?.name' 2>/dev/null | grep -Ei 'pg_pinecone-drill|drill.b' || true)"
        [ -z "$leftover" ] && break
        sleep 2
    done
    if [ -n "$leftover" ]; then
        echo "FINDING: drill-named indexes still present 90s after cleanup: $leftover" >&2
        RESULT_TEARDOWN="FAIL"
        NOTE_TEARDOWN="leftover indexes: $leftover"
        status=1
    else
        echo "confirmed: no drill-named indexes remain in the account" >&2
        if [ "${RESULT_TEARDOWN:-}" != "FAIL" ]; then
            record TEARDOWN PASS "all drill indexes/backup deleted and verified gone; test DB dropped"
        fi
    fi

    if [ "$was_running" -eq 0 ]; then
        cargo pgrx stop pg17 >/dev/null 2>&1
    fi

    echo "" >&2
    echo "================= DISASTER-RECOVERY DRILL REPORT =================" >&2
    for leg in "${LEG_ORDER[@]}"; do
        rv="RESULT_$leg"
        nv="NOTE_$leg"
        printf '%-11s %-24s %s\n' "$leg" "${!rv:-NOT-RUN}" "${!nv:-}" >&2
    done
    echo "====================================================================" >&2

    exit "$status"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# curl helpers against the real Pinecone control/data plane
# ---------------------------------------------------------------------------
index_host() {
    curl -fsS "$CONTROL_PLANE_URL/indexes/$1" -H "Api-Key: $PINECONE_API_KEY" \
        -H "X-Pinecone-Api-Version: $API_VERSION" | jq -r '.host'
}
normalize_host() { case "$1" in http*) echo "$1" ;; *) echo "https://$1" ;; esac; }
describe_stats() {
    local host
    host="$(normalize_host "$(index_host "$1")")"
    curl -fsS -X POST "$host/describe_index_stats" -H "Api-Key: $PINECONE_API_KEY" \
        -H "X-Pinecone-Api-Version: $API_VERSION" -H 'Content-Type: application/json' -d '{}'
}
wait_index_ready() {
    local name="$1" tries=0
    while true; do
        local ready
        ready="$(curl -fsS "$CONTROL_PLANE_URL/indexes/$name" -H "Api-Key: $PINECONE_API_KEY" \
            -H "X-Pinecone-Api-Version: $API_VERSION" | jq -r '.status.ready')"
        [ "$ready" = "true" ] && return 0
        tries=$((tries + 1))
        [ "$tries" -ge 90 ] && return 1
        sleep 2
    done
}

# ---------------------------------------------------------------------------
# LEG 1: SETUP
# ---------------------------------------------------------------------------
record SETUP RUNNING "installing $EXPECT_VER + restart, tracking table A (trigger path)"

if cargo pgrx status pg17 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -qi "is running"; then
    was_running=1
fi

if ! (cd "$PG_PINECONE_DIR" && cargo pgrx install -c "$PG_CONFIG"); then
    record SETUP FAIL "cargo pgrx install failed"
    exit 1
fi

cargo pgrx stop pg17 >/dev/null 2>&1
if ! cargo pgrx start pg17; then
    record SETUP FAIL "cargo pgrx start failed"
    exit 1
fi
wait_for_ready || { record SETUP FAIL "postgres never came up"; exit 1; }
PORT="$(pg_port)"
psql_pg() { psql -h 127.0.0.1 -p "$PORT" -U "$(whoami)" -v ON_ERROR_STOP=1 "$@"; }

orig_worker_enabled="$(psql_pg -d postgres -tAc "SHOW pinecone.worker_enabled" 2>/dev/null || echo "off")"
orig_max_attempts="$(psql_pg -d postgres -tAc "SHOW pinecone.outbox_max_attempts" 2>/dev/null || echo "10")"

psql_pg -d postgres \
    -c "ALTER SYSTEM SET shared_preload_libraries = 'pg_pinecone'" \
    -c "ALTER SYSTEM SET pinecone.worker_enabled = on" \
    -c "SELECT pg_reload_conf()"

cargo pgrx stop pg17
cargo pgrx start pg17
wait_for_ready || { record SETUP FAIL "postgres never came back up after restart"; exit 1; }
PORT="$(pg_port)"

psql_pg -d postgres -c "CREATE DATABASE \"$DBNAME\""
db() { psql_pg -d "$DBNAME" "$@"; }
db_noerr() { psql -h 127.0.0.1 -p "$PORT" -U "$(whoami)" -d "$DBNAME" "$@"; }

db -c "CREATE EXTENSION IF NOT EXISTS vector"
db -c "CREATE EXTENSION pg_pinecone"
installed_version="$(db -tAc "SELECT extversion FROM pg_extension WHERE extname='pg_pinecone'")"
if [ -z "$installed_version" ]; then
    record SETUP FAIL "pg_pinecone extension not installed"
    exit 1
fi
if [ -n "$EXPECT_VER" ] && [ "$installed_version" != "$EXPECT_VER" ]; then
    record SETUP FAIL "expected extversion $EXPECT_VER (crate Cargo.toml), got '$installed_version'"
    exit 1
fi

db <<SQL
SELECT pinecone.set_api_key('$PINECONE_API_KEY');
SQL

if [ "$LEG6_ONLY" = "1" ]; then
    record SETUP PASS "leg6-only rerun: $EXPECT_VER installed+restarted, db/extension/key ready (table-A legs skipped)"
    record BACKUP SKIPPED "leg6-only rerun"
    record RESTORE SKIPPED "leg6-only rerun"
    record REPOINT SKIPPED "leg6-only rerun"
    record RECOVER SKIPPED "leg6-only rerun"
    record JOINT SKIPPED "leg6-only rerun"
else

db -c "CREATE TABLE $TABLE_A (id bigint PRIMARY KEY, embedding vector($DIM), tag int, label text)"

db -c "SELECT pinecone.promote('$TABLE_A', index_name => '$INDEX_A', \
    metadata_columns => ARRAY['tag','label'], sync_mode => 'async')" || {
    record SETUP FAIL "pinecone.promote(table A) failed"
    exit 1
}
CREATED_INDEXES+=("$INDEX_A")

synced_by_a="$(db -tAc "SELECT synced_by FROM pinecone.tables WHERE relid = '$TABLE_A'::regclass")"
namespace_a="$(db -tAc "SELECT namespace FROM pinecone.tables WHERE relid = '$TABLE_A'::regclass")"
if [ "$synced_by_a" != "trigger" ]; then
    record SETUP FAIL "expected synced_by='trigger' for table A, got '$synced_by_a'"
    exit 1
fi

# Deterministic vectors: (((id+d) % 17) - 8) / 8.0 is an exact binary
# fraction (multiple of 0.125) for every component, so the f32 -> Pinecone
# JSON -> f32 round trip is bit-exact — needed for leg RECOVER's fidelity
# check.
db -c "INSERT INTO $TABLE_A (id, embedding, tag, label)
       SELECT gs,
              (ARRAY(SELECT ((((gs + d) % 17) - 8)::float8 / 8.0) FROM generate_series(0, $((DIM - 1))) d))::vector,
              gs % 97,
              'label-' || gs
       FROM generate_series(1, $N_ROWS) gs"

flushed="$(db -tAc "SELECT pinecone.flush('$TABLE_A', 180000)")"
if [ "$flushed" != "t" ]; then
    record SETUP FAIL "pinecone.flush(table A) did not drain within 180s (worker not running?)"
    exit 1
fi

remote_stats="$(describe_stats "$INDEX_A")"
remote_count="$(echo "$remote_stats" | jq -r '.totalVectorCount // 0')"
if [ "$remote_count" != "$N_ROWS" ]; then
    record SETUP FAIL "describe_index_stats reports $remote_count vectors, expected $N_ROWS"
    exit 1
fi

# Snapshot for the RECOVER leg's bit-fidelity check.
db -c "CREATE TABLE drill_originals AS SELECT id, embedding, tag, label FROM $TABLE_A"

record SETUP PASS "$EXPECT_VER installed+restarted; table A promoted (trigger, index=$INDEX_A); $N_ROWS rows flushed; remote count verified via describe_index_stats"

# ---------------------------------------------------------------------------
# LEG 2: BACKUP (may be plan-gated)
# ---------------------------------------------------------------------------
record BACKUP RUNNING "attempting pg_pinecone-client create_backup against index A via a one-off --ignored test"

backup_log="$(mktemp)"
if (cd "$CLIENT_DIR" && \
    PG_PINECONE_LIVE_TESTS=1 \
    PG_PINECONE_DRILL_INDEX_A="$INDEX_A" \
    PG_PINECONE_DRILL_BACKUP_NAME="$BACKUP_NAME" \
    PG_PINECONE_DRILL_INDEX_RESTORED="$INDEX_RESTORED" \
    cargo test --test backup_drill -- --ignored --nocapture --test-threads=1 \
    live_backup_create_and_restore_drill) >"$backup_log" 2>&1
then
    backup_ok=1
else
    backup_ok=0
fi
cat "$backup_log" >&2

drill_backup_id="$(grep -m1 '^DRILL_BACKUP_ID=' "$backup_log" | cut -d= -f2- || true)"
[ -n "$drill_backup_id" ] && BACKUP_ID="$drill_backup_id"

if [ "$backup_ok" -eq 1 ] && grep -q '^DRILL_RESTORED_READY=true' "$backup_log"; then
    CREATED_INDEXES+=("$INDEX_RESTORED")
    record BACKUP PASS "create_backup + describe_backup(Ready) + create_index_from_backup + wait_index_ready all succeeded (backup_id=$drill_backup_id)"
    RESTORE_MODE="real"
else
    error_line="$(grep -m1 -E 'DRILL_(BACKUP_CREATE|BACKUP_DESCRIBE|RESTORE_CREATE|RESTORE_WAIT)_ERROR=' "$backup_log" || true)"
    if echo "$error_line" | grep -qiE 'plan|not (available|supported|enabled)|upgrade|forbidden|403'; then
        record BACKUP SKIPPED-plan-gated "$error_line"
    else
        record BACKUP FAIL "$error_line"
    fi
    RESTORE_MODE="simulated"
fi
rm -f "$backup_log"

# ---------------------------------------------------------------------------
# LEG 3: DISASTER + RESTORE
# ---------------------------------------------------------------------------
if [ "$RESTORE_MODE" = "real" ]; then
    record RESTORE PASS "real path: backup restored to a new index ($INDEX_RESTORED) via create_index_from_backup, Ready"
else
    record RESTORE RUNNING "SIMULATED path: creating $INDEX_RESTORED and copying A's vectors via list+fetch+upsert"

    if ! curl -fsS -X POST "$CONTROL_PLANE_URL/indexes" \
        -H "Api-Key: $PINECONE_API_KEY" -H "X-Pinecone-Api-Version: $API_VERSION" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg name "$INDEX_RESTORED" --argjson dim "$DIM" \
              '{name:$name, vector_type:"dense", dimension:$dim, metric:"cosine", spec:{serverless:{cloud:"aws",region:"us-east-1"}}, deletion_protection:"disabled"}')" \
        >/dev/null
    then
        record RESTORE FAIL "SIMULATED: create_index for $INDEX_RESTORED failed"
    else
        CREATED_INDEXES+=("$INDEX_RESTORED")
        if ! wait_index_ready "$INDEX_RESTORED"; then
            record RESTORE FAIL "SIMULATED: $INDEX_RESTORED never became ready"
        else
            src_host="$(normalize_host "$(index_host "$INDEX_A")")"
            dst_host="$(normalize_host "$(index_host "$INDEX_RESTORED")")"
            enc_ns="$(jq -rn --arg v "$namespace_a" '$v|@uri')"
            token=""
            copied=0
            copy_ok=1
            while true; do
                url="$src_host/vectors/list?limit=100&namespace=$enc_ns"
                [ -n "$token" ] && url="$url&paginationToken=$(jq -rn --arg v "$token" '$v|@uri')"
                page="$(curl -fsS "$url" -H "Api-Key: $PINECONE_API_KEY" -H "X-Pinecone-Api-Version: $API_VERSION")"
                ids="$(echo "$page" | jq -r '.vectors[]?.id')"
                if [ -n "$ids" ]; then
                    idparams=""
                    while IFS= read -r id; do
                        [ -z "$id" ] && continue
                        idparams="$idparams&ids=$(jq -rn --arg v "$id" '$v|@uri')"
                    done <<<"$ids"
                    # ONE url with everything pre-encoded — the first run
                    # passed BOTH a -G/--data-urlencode form and this url,
                    # making curl fetch two URLs per page (the -G one 400'd
                    # every time).
                    fetched="$(curl -fsS \
                        -H "Api-Key: $PINECONE_API_KEY" -H "X-Pinecone-Api-Version: $API_VERSION" \
                        "$src_host/vectors/fetch?namespace=$enc_ns${idparams}")"
                    upsert_body="$(echo "$fetched" | jq -c --arg ns "$namespace_a" \
                        '{vectors: [.vectors[] | {id: .id, values: .values, metadata: (.metadata // {})}], namespace: $ns}')"
                    n="$(echo "$upsert_body" | jq -r '.vectors | length')"
                    if [ "$n" -gt 0 ]; then
                        if ! curl -fsS -X POST "$dst_host/vectors/upsert" \
                            -H "Api-Key: $PINECONE_API_KEY" -H "X-Pinecone-Api-Version: $API_VERSION" \
                            -H "Content-Type: application/json" -d "$upsert_body" >/dev/null
                        then
                            copy_ok=0
                            break
                        fi
                        copied=$((copied + n))
                    fi
                fi
                token="$(echo "$page" | jq -r '.pagination.next // empty')"
                [ -z "$token" ] && break
            done
            if [ "$copy_ok" -eq 1 ]; then
                record RESTORE PASS "SIMULATED: copied $copied/$N_ROWS vectors from $INDEX_A into new index $INDEX_RESTORED (list+fetch+upsert, NOT a real Pinecone backup restore)"
            else
                record RESTORE FAIL "SIMULATED: vector copy loop failed partway (copied $copied so far)"
            fi
        fi
    fi
fi

# ---------------------------------------------------------------------------
# LEG 4: REPOINT
# ---------------------------------------------------------------------------
if [ "${RESULT_RESTORE:-}" != "PASS" ]; then
    record REPOINT SKIPPED "RESTORE leg did not produce a usable target index"
else
    record REPOINT RUNNING "dim/metric mismatch validation, then real repoint to $INDEX_RESTORED"

    if ! curl -fsS -X POST "$CONTROL_PLANE_URL/indexes" \
        -H "Api-Key: $PINECONE_API_KEY" -H "X-Pinecone-Api-Version: $API_VERSION" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg name "$INDEX_MISMATCH" --argjson dim "$((DIM + 1))" \
              '{name:$name, vector_type:"dense", dimension:$dim, metric:"cosine", spec:{serverless:{cloud:"aws",region:"us-east-1"}}, deletion_protection:"disabled"}')" \
        >/dev/null
    then
        record REPOINT FAIL "could not create mismatch-dimension probe index"
    else
        CREATED_INDEXES+=("$INDEX_MISMATCH")
        wait_index_ready "$INDEX_MISMATCH" || true

        mismatch_out="$(db_noerr -c "SELECT pinecone.repoint('$TABLE_A', '$INDEX_MISMATCH')" 2>&1)"
        mismatch_rc=$?
        index_before="$(db -tAc "SELECT index_name FROM pinecone.tables WHERE relid = '$TABLE_A'::regclass")"

        curl -fsS -X DELETE "$CONTROL_PLANE_URL/indexes/$INDEX_MISMATCH" \
            -H "Api-Key: $PINECONE_API_KEY" -H "X-Pinecone-Api-Version: $API_VERSION" >/dev/null 2>&1
        # Already deleted above; left in CREATED_INDEXES too (teardown's
        # delete-by-name is idempotent/error-suppressed) as a safety net.

        if [ "$mismatch_rc" -eq 0 ] || [ "$index_before" != "$INDEX_A" ]; then
            record REPOINT FAIL "repoint should have refused a dimension mismatch, but rc=$mismatch_rc and index_name is now '$index_before'"
        else
            db -c "SELECT pinecone.repoint('$TABLE_A', '$INDEX_RESTORED')"
            repoint_rc=$?
            index_after="$(db -tAc "SELECT index_name FROM pinecone.tables WHERE relid = '$TABLE_A'::regclass")"
            if [ "$repoint_rc" -eq 0 ] && [ "$index_after" = "$INDEX_RESTORED" ]; then
                record REPOINT PASS "mismatch refused (rc=$mismatch_rc, index unchanged); real repoint updated registry to $INDEX_RESTORED"
            else
                record REPOINT FAIL "real repoint failed or registry not updated (rc=$repoint_rc, index_name=$index_after)"
            fi
        fi
    fi
fi

# ---------------------------------------------------------------------------
# LEG 5: RECOVER (+ bit-fidelity)
# ---------------------------------------------------------------------------
if [ "${RESULT_REPOINT:-}" != "PASS" ]; then
    record RECOVER SKIPPED "REPOINT leg did not succeed"
else
    record RECOVER RUNNING "NULLing a subset locally (trigger disabled), recover_from_remote, bit-fidelity check"

    db -c "BEGIN;" -c "ALTER TABLE $TABLE_A DISABLE TRIGGER USER;" \
        -c "UPDATE $TABLE_A SET embedding = NULL, tag = NULL, label = NULL WHERE id <= 50;" \
        -c "ALTER TABLE $TABLE_A ENABLE TRIGGER USER;" -c "COMMIT;"

    db -c "SELECT pinecone.recover_from_remote('$TABLE_A')"

    recover_state=""
    for _ in $(seq 1 120); do
        row="$(db -tAc "SELECT queue_state, backlog FROM pinecone.status() WHERE table_name = 'public.$TABLE_A'")"
        qs="${row%%|*}"
        bl="${row##*|}"
        echo "RECOVER poll: queue_state=$qs backlog=$bl" >&2
        if [ "$qs" = "idle" ] && [ "${bl:-0}" = "0" ]; then
            recover_state="done"
            break
        fi
        if [ "$qs" = "parked" ]; then
            recover_state="parked"
            break
        fi
        sleep 1
    done

    if [ "$recover_state" != "done" ]; then
        last_err="$(db -tAc "SELECT last_error FROM pinecone.status() WHERE table_name = 'public.$TABLE_A'")"
        record RECOVER FAIL "recover job did not complete cleanly (state=$recover_state, last_error=$last_err)"
    else
        mismatched="$(db -tAc "SELECT count(*) FROM $TABLE_A t JOIN drill_originals o USING (id) \
            WHERE o.id <= 50 AND t.embedding IS DISTINCT FROM o.embedding")"
        still_null="$(db -tAc "SELECT count(*) FROM $TABLE_A WHERE id <= 50 AND embedding IS NULL")"
        meta_mismatched="$(db -tAc "SELECT count(*) FROM $TABLE_A t JOIN drill_originals o USING (id) \
            WHERE o.id <= 50 AND (t.tag IS DISTINCT FROM o.tag OR t.label IS DISTINCT FROM o.label)")"
        untouched_changed="$(db -tAc "SELECT count(*) FROM $TABLE_A t JOIN drill_originals o USING (id) \
            WHERE o.id > 50 AND (t.embedding IS DISTINCT FROM o.embedding OR t.tag IS DISTINCT FROM o.tag OR t.label IS DISTINCT FROM o.label)")"

        if [ "$mismatched" = "0" ] && [ "$still_null" = "0" ] && [ "$meta_mismatched" = "0" ] && [ "$untouched_changed" = "0" ]; then
            record RECOVER PASS "all 50 NULLed rows restored, byte-exact vs. pre-loss originals (0 ULP deviation); metadata restored; 750 untouched rows unchanged"
        else
            record RECOVER FAIL "bit-fidelity/scope check failed: vector_mismatch=$mismatched still_null=$still_null metadata_mismatch=$meta_mismatched untouched_rows_changed=$untouched_changed"
        fi
    fi
fi

fi # LEG6_ONLY gate around table-A setup + legs 2-5

# ---------------------------------------------------------------------------
# LEG 6: CONCURRENT-WRITER PROBE (AM path) + F1/F2 remediation
# ---------------------------------------------------------------------------
record CONCURRENT RUNNING "AM-indexed table, non-HOT update racing recover_from_remote"

db -c "ALTER SYSTEM SET pinecone.outbox_max_attempts = 1" -c "SELECT pg_reload_conf()"

db -c "CREATE TABLE $TABLE_B (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, embedding vector($DIM), tag int, label text)"
db -c "CREATE INDEX ${TABLE_B}_tagidx ON $TABLE_B (tag)"
db -c "CREATE INDEX ${TABLE_B}_idx ON $TABLE_B USING pinecone (embedding vector_cosine_ops)"

am_index_name="$(db -tAc "SELECT index_name FROM pinecone.tables WHERE relid = '$TABLE_B'::regclass")"
am_synced_by="$(db -tAc "SELECT synced_by FROM pinecone.tables WHERE relid = '$TABLE_B'::regclass")"
if [ -n "$am_index_name" ]; then
    CREATED_INDEXES+=("$am_index_name")
fi

if [ "$am_synced_by" != "am" ]; then
    record CONCURRENT FAIL "CREATE INDEX ... USING pinecone did not register synced_by='am' (got '$am_synced_by')"
else
    db -c "INSERT INTO $TABLE_B (embedding, tag, label)
           SELECT (ARRAY(SELECT ((((gs + d) % 17) - 8)::float8 / 8.0) FROM generate_series(0, $((DIM - 1))) d))::vector,
                  gs, 'b-' || gs
           FROM generate_series(1, 60) gs"

    flushed_b="$(db -tAc "SELECT pinecone.flush('$TABLE_B', 120000)")"
    if [ "$flushed_b" != "t" ]; then
        record CONCURRENT FAIL "initial AM backfill/upsert never drained"
    else
        id_list="$(db -tAc "SELECT string_agg(id::text, ',') FROM (SELECT id FROM $TABLE_B ORDER BY id LIMIT 20) s")"

        # Pre-loss snapshot: the quarantine byte-fidelity check and the
        # manual re-attach demo below both join through this — deliberately
        # so, since that IS the documented re-attach story (the quarantined
        # remote_id is a stale heap position; only application knowledge can
        # match bytes back to rows).
        db -c "CREATE TABLE drill_b_originals AS SELECT id, embedding FROM $TABLE_B"

        # Deliberately NOT nulling any vectors first: embedding is itself
        # covered by the pg_pinecone AM index, so an UPDATE that nulls it is
        # ALREADY a non-HOT write that relocates the ctid on its own —
        # nulling would make every row a guaranteed miss before recover even
        # starts, collapsing the "concurrent writer during an in-flight job"
        # scenario into a foregone conclusion. recover_chunk_am's miss
        # detection (existing_count < fetched_count) doesn't depend on the
        # NULL guard anyway, so a genuine mid-flight race is enough on its
        # own: enqueue the job, then race a non-HOT UPDATE on an unrelated
        # (but btree-indexed, so still non-HOT) column against it.
        db -c "SELECT pinecone.recover_from_remote('$TABLE_B')"
        # Race window: worker naptime is 2000ms; fire the non-HOT UPDATE now.
        # tag is btree-indexed, so any change to it forces a non-HOT update
        # (new ctid for every index on the table, including pg_pinecone's own).
        db -c "UPDATE $TABLE_B SET tag = tag + 1000 WHERE id IN ($id_list)" &
        RACE_PID=$!
        wait "$RACE_PID"

        concurrent_state=""
        for _ in $(seq 1 30); do
            row="$(db -tAc "SELECT queue_state, backlog FROM pinecone.status() WHERE table_name = 'public.$TABLE_B'")"
            qs="${row%%|*}"
            echo "CONCURRENT poll: queue_state=$qs" >&2
            if [ "$qs" = "parked" ]; then
                concurrent_state="parked"
                break
            fi
            if [ "$qs" = "idle" ]; then
                concurrent_state="idle-no-park"
                break
            fi
            sleep 1
        done

        if [ "$concurrent_state" != "parked" ]; then
            record CONCURRENT FAIL "expected the recover job to PARK on the quiescence/miss error; got state=$concurrent_state (race not reproduced or silent data loss)"
        else
            last_error_b="$(db -tAc "SELECT last_error FROM pinecone.status() WHERE table_name = 'public.$TABLE_B'")"
            parked_flag="$(db -tAc "SELECT queue_parked FROM pinecone.tables WHERE relid = '$TABLE_B'::regclass")"
            echo "PARKED job status() output: queue_state=parked queue_parked=$parked_flag last_error=$last_error_b" >&2

            if ! echo "$last_error_b" | grep -qi "quiescence" || ! echo "$last_error_b" | grep -qi "no matching row"; then
                record CONCURRENT FAIL "job parked but error text doesn't match the documented quiescence/miss message: $last_error_b"
            elif echo "$last_error_b" | grep -qi "SHARE lock on the table"; then
                record CONCURRENT FAIL "park message still recommends taking a SHARE lock: $last_error_b"
            elif ! echo "$last_error_b" | grep -qi "quarantine" || ! echo "$last_error_b" | grep -qi "VACUUM"; then
                record CONCURRENT FAIL "park message is missing the quarantine option or the do-not-VACUUM warning: $last_error_b"
            else
                # The corrected remediation: quiesce = STOP
                # writing — no table locks (a SHARE lock blocks the job's own
                # write-back; the first drill run proved it empirically).
                # The writers here are this script's own UPDATEs, already
                # finished, so the app is quiesced by construction. Rows
                # genuinely relocated, so a plain retry would just park
                # again: switch the parked job to quarantine (in-place policy
                # update, cursor preserved), THEN retry.
                log_mark="$(wc -l < "$SERVER_LOG" 2>/dev/null | tr -d ' ' || echo 0)"
                db -c "SELECT pinecone.recover_from_remote('$TABLE_B', on_miss => 'quarantine')"
                db -c "SELECT pinecone.outbox_retry('$TABLE_B')"

                retry_state=""
                for _ in $(seq 1 60); do
                    row="$(db -tAc "SELECT queue_state, backlog FROM pinecone.status() WHERE table_name = 'public.$TABLE_B'")"
                    qs="${row%%|*}"
                    bl="${row##*|}"
                    echo "RETRY poll: queue_state=$qs backlog=$bl" >&2
                    if [ "$qs" = "idle" ] && [ "${bl:-0}" = "0" ]; then
                        retry_state="done"
                        break
                    fi
                    if [ "$qs" = "parked" ]; then
                        retry_state="parked-again"
                        break
                    fi
                    sleep 1
                done

                if [ "$retry_state" != "done" ]; then
                    record CONCURRENT FAIL "quarantine retry did not complete (state=$retry_state)"
                else
                    # The job must have completed with a WARNING naming the
                    # quarantine (worker output -> server log).
                    completion_warning="$(tail -n "+$((log_mark + 1))" "$SERVER_LOG" 2>/dev/null \
                        | grep -m1 -E 'recover_from_remote.*finished.*quarantined [1-9]' || true)"

                    q_count="$(db -tAc "SELECT count(*) FROM pinecone.recover_quarantine \
                        WHERE relid = '$TABLE_B'::regclass")"
                    q_bad_bytes="$(db -tAc "SELECT count(*) FROM pinecone.recover_quarantine q \
                        WHERE q.relid = '$TABLE_B'::regclass AND NOT EXISTS ( \
                            SELECT 1 FROM drill_b_originals o WHERE o.embedding::real[] = q.embedding)")"
                    still_null_b="$(db -tAc "SELECT count(*) FROM $TABLE_B WHERE embedding IS NULL")"
                    tag_preserved="$(db -tAc "SELECT count(*) FROM $TABLE_B WHERE id IN ($id_list) AND tag < 1000")"

                    # Manual re-attach demo (the documented step 5): NULL one
                    # relocated row's vector (an app-level write — the job is
                    # complete, so this is safe), then re-attach its
                    # quarantined bytes via the application-knowledge join.
                    reattach_id="$(db -tAc "SELECT min(id) FROM $TABLE_B WHERE id IN ($id_list)")"
                    db -c "UPDATE $TABLE_B SET embedding = NULL WHERE id = $reattach_id"
                    db -c "UPDATE $TABLE_B t SET embedding = q.embedding::vector \
                           FROM pinecone.recover_quarantine q, drill_b_originals o \
                           WHERE q.relid = '$TABLE_B'::regclass \
                             AND t.id = o.id AND t.id = $reattach_id \
                             AND t.embedding IS NULL \
                             AND q.embedding = o.embedding::real[]"
                    reattached="$(db -tAc "SELECT count(*) FROM $TABLE_B t \
                        JOIN drill_b_originals o USING (id) \
                        WHERE t.id = $reattach_id AND t.embedding IS NOT DISTINCT FROM o.embedding")"

                    # Only NOW is VACUUM safe (job complete, bytes accounted
                    # for) — and it must be harmless: reaping the stale TIDs
                    # remote-deletes positions nothing references anymore.
                    vacuum_ok=0
                    db -c "VACUUM $TABLE_B" && vacuum_ok=1
                    null_after_vacuum="$(db -tAc "SELECT count(*) FROM $TABLE_B WHERE embedding IS NULL")"

                    if [ -z "$completion_warning" ]; then
                        record CONCURRENT FAIL "job completed but no quarantine WARNING found in the server log after line $log_mark"
                    elif [ "$q_count" != "20" ] || [ "$q_bad_bytes" != "0" ]; then
                        record CONCURRENT FAIL "quarantine wrong: count=$q_count (want 20) byte_mismatches=$q_bad_bytes"
                    elif [ "$still_null_b" != "0" ] || [ "$tag_preserved" != "0" ]; then
                        record CONCURRENT FAIL "post-retry data wrong: still_null=$still_null_b tag_not_bumped=$tag_preserved"
                    elif [ "$reattached" != "1" ]; then
                        record CONCURRENT FAIL "the sample re-attach UPDATE did not restore the quarantined bytes (reattached=$reattached)"
                    elif [ "$vacuum_ok" != "1" ] || [ "$null_after_vacuum" != "0" ]; then
                        record CONCURRENT FAIL "post-completion VACUUM not harmless: rc_ok=$vacuum_ok null_after=$null_after_vacuum"
                    else
                        record CONCURRENT PASS "parked on first miss (corrected message, no lock advice); quiesce -> on_miss=>'quarantine' policy switch -> outbox_retry COMPLETED with WARNING; quarantined 20/20 byte-exact vs pre-loss; sample re-attach UPDATE restored the bytes; VACUUM afterward harmless"
                    fi
                fi
            fi
        fi
    fi
fi

db -c "ALTER SYSTEM SET pinecone.outbox_max_attempts = $orig_max_attempts" -c "SELECT pg_reload_conf()"

# ---------------------------------------------------------------------------
# LEG 7: JOINT (restore Postgres AND Pinecone together)
# docs/disaster-recovery.md#restoring-both-postgres-and-pinecone-together
# ---------------------------------------------------------------------------
dbj() { psql -h 127.0.0.1 -p "$PORT" -U "$(whoami)" -v ON_ERROR_STOP=1 -d "$DBNAME_J" "$@"; }

c_remote_count() { describe_stats "$1" | jq -r '.totalVectorCount // 0'; }

wait_drain_j() {
    local tbl="$1" tries=0 row qs bl
    while [ "$tries" -lt 120 ]; do
        row="$(dbj -tAc "SELECT queue_state, backlog FROM pinecone.status() WHERE table_name = 'public.$tbl'")"
        qs="${row%%|*}"; bl="${row##*|}"
        [ "$qs" = "idle" ] && [ "${bl:-0}" = "0" ] && return 0
        [ "$qs" = "parked" ] && return 1
        tries=$((tries + 1)); sleep 1
    done
    return 1
}

readiness_sync_ok() {
    local detail
    detail="$(dbj -tAc "SELECT detail FROM pinecone.backup_readiness('$1') WHERE \"check\" = 'sync'")"
    echo "$detail" | grep -q 'diff=0' && echo "$detail" | grep -q 'outbox_backlog=0'
}

run_joint() {
    local M=200 dumpfile src_count r_before r_after secret_rows restored_count
    local null_after_detach reattach_mismatch local_after_attach c_index_name

    record JOINT RUNNING "AM table dumped+restored into a keyless DB; order-safety, sever, full sync, detached attach"

    db -c "CREATE TABLE $TABLE_C (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, embedding vector($DIM), tag int)"
    db -c "CREATE INDEX ${TABLE_C}_idx ON $TABLE_C USING pinecone (embedding vector_cosine_ops)"
    c_index_name="$(db -tAc "SELECT index_name FROM pinecone.tables WHERE relid = '$TABLE_C'::regclass")"
    [ -n "$c_index_name" ] && CREATED_INDEXES+=("$c_index_name")

    db -c "INSERT INTO $TABLE_C (embedding, tag)
           SELECT (ARRAY(SELECT ((((gs + d) % 17) - 8)::float8 / 8.0) FROM generate_series(0, $((DIM - 1))) d))::vector, gs
           FROM generate_series(1, $M) gs"
    [ "$(db -tAc "SELECT pinecone.flush('$TABLE_C', 120000)")" = "t" ] || { record JOINT FAIL "source TABLE_C never drained"; return 1; }
    src_count="$(c_remote_count "$c_index_name")"
    [ "$src_count" = "$M" ] || { record JOINT FAIL "source remote count $src_count != $M"; return 1; }

    dumpfile="$(mktemp -t pg_pinecone_joint.XXXXXX)"
    "$PGBIN/pg_dump" -h 127.0.0.1 -p "$PORT" -U "$(whoami)" -Fc -d "$DBNAME" -f "$dumpfile" \
        || { record JOINT FAIL "pg_dump failed"; rm -f "$dumpfile"; return 1; }

    psql_pg -d postgres -c "CREATE DATABASE \"$DBNAME_J\"" || { record JOINT FAIL "createdb joint failed"; rm -f "$dumpfile"; return 1; }
    # pg_restore of a pg_pinecone DB emits non-fatal notices; ignore its exit, verify the data instead.
    "$PGBIN/pg_restore" -h 127.0.0.1 -p "$PORT" -U "$(whoami)" -d "$DBNAME_J" "$dumpfile" >/dev/null 2>&1
    rm -f "$dumpfile"
    [ "$(dbj -tAc "SELECT to_regclass('public.$TABLE_C') IS NOT NULL")" = "t" ] \
        || { record JOINT FAIL "restore did not recreate $TABLE_C"; return 1; }

    # The dump excludes pinecone.secret by design: the restored DB is keyless,
    # so every queued rebuild waits instead of firing delete_all at the index.
    secret_rows="$(dbj -tAc "SELECT count(*) FROM pinecone.secret")"
    [ "$secret_rows" = "0" ] || { record JOINT FAIL "restored DB unexpectedly has a key ($secret_rows rows in pinecone.secret)"; return 1; }

    # Drop the other restored tables so keying this DB later can't fire THEIR rebuilds.
    dbj -c "DROP TABLE IF EXISTS $TABLE_A, $TABLE_B CASCADE" >/dev/null 2>&1

    # ORDER SAFETY: a keyless restore must not have rewritten the source's index.
    sleep 5
    r_before="$(c_remote_count "$c_index_name")"
    [ "$r_before" = "$M" ] || { record JOINT FAIL "keyless restore clobbered the source index (count $r_before != $M)"; return 1; }

    # SEVER before keying: AM lane -> DROP INDEX discards the queued rebuild, Pinecone untouched.
    dbj -c "DROP INDEX ${TABLE_C}_idx" || { record JOINT FAIL "sever DROP INDEX failed"; return 1; }
    r_after="$(c_remote_count "$c_index_name")"
    [ "$r_after" = "$M" ] || { record JOINT FAIL "sever disturbed the source index (count $r_after != $M)"; return 1; }

    dbj -c "CREATE TABLE drill_c_j_originals AS SELECT id, embedding FROM $TABLE_C"

    dbj -c "SELECT pinecone.set_api_key('$PINECONE_API_KEY')" || { record JOINT FAIL "set_api_key in joint DB failed"; return 1; }
    dbj -c "SELECT pinecone.create_index('$TABLE_C', 'embedding', engine => 'remote', index_name => '$INDEX_C_RESTORED')" \
        || { record JOINT FAIL "create_index to restored index failed"; return 1; }
    CREATED_INDEXES+=("$INDEX_C_RESTORED")

    # FULL storage -> local is truth -> sync (local -> remote).
    dbj -c "SELECT pinecone.sync('$TABLE_C')" || { record JOINT FAIL "sync failed"; return 1; }
    wait_drain_j "$TABLE_C" || { record JOINT FAIL "full-path sync never drained"; return 1; }
    restored_count="$(c_remote_count "$INDEX_C_RESTORED")"
    [ "$restored_count" = "$M" ] || { record JOINT FAIL "restored index count $restored_count != $M after sync"; return 1; }
    readiness_sync_ok "$TABLE_C" || { record JOINT FAIL "backup_readiness sync row not clean after full sync"; return 1; }

    # DETACHED storage -> remote is truth -> attach (remote -> local).
    dbj -c "SELECT pinecone.detach_storage('$TABLE_C', force => true)" || { record JOINT FAIL "detach_storage failed"; return 1; }
    null_after_detach="$(dbj -tAc "SELECT count(*) FROM $TABLE_C WHERE embedding IS NOT NULL")"
    [ "$null_after_detach" = "0" ] || { record JOINT FAIL "detach left $null_after_detach non-NULL vectors locally"; return 1; }
    dbj -c "SELECT pinecone.attach_storage('$TABLE_C')" || { record JOINT FAIL "attach_storage failed"; return 1; }
    wait_drain_j "$TABLE_C" || { record JOINT FAIL "attach recover never drained"; return 1; }
    local_after_attach="$(dbj -tAc "SELECT count(*) FROM $TABLE_C WHERE embedding IS NOT NULL")"
    reattach_mismatch="$(dbj -tAc "SELECT count(*) FROM $TABLE_C t JOIN drill_c_j_originals o USING (id) WHERE t.embedding IS DISTINCT FROM o.embedding")"
    [ "$local_after_attach" = "$M" ] || { record JOINT FAIL "attach restored $local_after_attach/$M vectors locally"; return 1; }
    [ "$reattach_mismatch" = "0" ] || { record JOINT FAIL "attach restored $reattach_mismatch rows with wrong bytes"; return 1; }
    readiness_sync_ok "$TABLE_C" || { record JOINT FAIL "backup_readiness sync row not clean after attach"; return 1; }

    record JOINT PASS "keyless restore left the source index intact ($M); sever discarded the rebuild; create_index+sync repopulated the restored index to $M (readiness diff=0); detach(force)->attach round-tripped all $M vectors byte-exact (readiness diff=0)"
}

run_joint

echo "== all legs complete; handing off to teardown (EXIT trap) ==" >&2
