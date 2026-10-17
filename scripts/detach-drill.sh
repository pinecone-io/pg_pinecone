#!/usr/bin/env bash
# LIVE detach/attach disaster-recovery drill: the full detach_storage/attach_storage
# loop against the shared pgrx instance + real Pinecone, plus the attach-race live
# proof and a plan-gated backup leg. Precedent: scripts/disaster-recovery-drill.sh
# (shared-instance restart pattern, API key via environment only, never
# argv), whose SETUP/instance-lifecycle/curl helpers this script mirrors.
#
# Needs PINECONE_API_KEY in the environment. Creates ONLY indexes tagged
# pg_pinecone-d5-* and deletes them all on exit (see cleanup()).
set -uo pipefail

if [ -z "${PINECONE_API_KEY:-}" ]; then
    echo "PINECONE_API_KEY is not set — the detach drill cannot run live." >&2
    exit 1
fi

LOCK=/tmp/pg_pinecone-pgrx.lock
if [ "${1:-}" != "--locked" ]; then
    exec lockf -t 900 "$LOCK" "$0" --locked
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PG_PINECONE_DIR="$REPO_ROOT/pg_pinecone"
CLIENT_DIR="$REPO_ROOT/pg_pinecone-client"
PG_CONFIG="$(cd "$PG_PINECONE_DIR" && cargo pgrx info pg-config pg17)"
PGRX_HOME="${PGRX_HOME:-$HOME/.pgrx}"
PGDATA="$PGRX_HOME/data-17"
SERVER_LOG="$PGRX_HOME/17.log"

PID=$$
DBNAME="pg_pinecone_d5_drill_$PID"
TABLE_A="detach_drill_a"
TABLE_C="detach_drill_c"
INDEX_A="pg_pinecone-d5-a-$PID"
INDEX_C="pg_pinecone-d5-c-$PID"
INDEX_RESTORED="pg_pinecone-d5-restored-$PID"
BACKUP_NAME="pg_pinecone-d5-backup-$PID"
API_VERSION="2026-04"
BACKUP_API_VERSION="2026-07"
CONTROL_PLANE_URL="https://api.pinecone.io"
DIM=16
N_ROWS=300

echo "== detach-drill: db=$DBNAME table_a=$TABLE_A pg_config=$PG_CONFIG ==" >&2

# ---------------------------------------------------------------------------
# Bookkeeping + reporting (bash 3.2 on macOS: no associative arrays)
# ---------------------------------------------------------------------------
LEG_ORDER=(SETUP DETACH GUARDS ATTACH ATTACHRACE DUPID BACKUP RESTORE TEARDOWN)
CREATED_INDEXES=()
BACKUP_ID=""

record() {
    printf -v "RESULT_$1" '%s' "$2"
    printf -v "NOTE_$1" '%s' "$3"
    echo "== leg $1: $2 -- $3 ==" >&2
}

# ---------------------------------------------------------------------------
# Instance lifecycle helpers (disaster-recovery-drill.sh pattern)
# ---------------------------------------------------------------------------
was_running=0
orig_worker_enabled=""
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
    record TEARDOWN RUNNING "deleting drill indexes, dropping test DB, restoring instance state"

    if [ -n "$PORT" ]; then
        if [ -n "$orig_worker_enabled" ]; then
            psql -h 127.0.0.1 -p "$PORT" -U "$(whoami)" -d postgres -v ON_ERROR_STOP=1 \
                -c "ALTER SYSTEM SET pinecone.worker_enabled = '$orig_worker_enabled'" \
                -c "SELECT pg_reload_conf()" >/dev/null 2>&1
        fi
        psql -h 127.0.0.1 -p "$PORT" -U "$(whoami)" -d postgres \
            -c "DROP DATABASE IF EXISTS \"$DBNAME\" WITH (FORCE)" >/dev/null 2>&1
    fi

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
    leftover=""
    for _ in $(seq 1 45); do
        leftover="$(curl -fsS "$CONTROL_PLANE_URL/indexes" -H "Api-Key: $PINECONE_API_KEY" \
            -H "X-Pinecone-Api-Version: $API_VERSION" 2>/dev/null \
            | jq -r '.indexes[]?.name' 2>/dev/null | grep -Ei 'pg_pinecone-d5-' || true)"
        [ -z "$leftover" ] && break
        sleep 2
    done
    if [ -n "$leftover" ]; then
        echo "FINDING: drill-named indexes still present after cleanup: $leftover" >&2
        RESULT_TEARDOWN="FAIL"
        NOTE_TEARDOWN="leftover indexes: $leftover"
        status=1
    else
        echo "confirmed: no pg_pinecone-d5-* indexes remain in the account" >&2
        if [ "${RESULT_TEARDOWN:-}" != "FAIL" ]; then
            record TEARDOWN PASS "all drill indexes/backup deleted and verified gone; test DB dropped"
        fi
    fi

    if [ "$was_running" -eq 0 ]; then
        cargo pgrx stop pg17 >/dev/null 2>&1
    fi

    echo "" >&2
    echo "===================== DETACH DRILL REPORT =====================" >&2
    for leg in "${LEG_ORDER[@]}"; do
        rv="RESULT_$leg"
        nv="NOTE_$leg"
        printf '%-11s %-24s %s\n' "$leg" "${!rv:-NOT-RUN}" "${!nv:-}" >&2
    done
    echo "=================================================================" >&2

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
remote_count() { describe_stats "$1" | jq -r '.totalVectorCount // 0'; }
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
fetch_metadata() {
    # index namespace id -> metadata jsonb (or "null")
    local host ns_enc
    host="$(normalize_host "$(index_host "$1")")"
    ns_enc="$(jq -rn --arg v "$2" '$v|@uri')"
    curl -fsS "$host/vectors/fetch?namespace=$ns_enc&ids=$3" -H "Api-Key: $PINECONE_API_KEY" \
        -H "X-Pinecone-Api-Version: $API_VERSION" | jq -c ".vectors[\"$3\"].metadata // null"
}

# ---------------------------------------------------------------------------
# LEG: SETUP -- install, restart, track+promote table A (trigger path), load
# N_ROWS, flush, verify remote count. Snapshot for later byte-fidelity checks.
# ---------------------------------------------------------------------------
record SETUP RUNNING "installing dev build + restart, promoting table A (trigger path)"

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
# No ON_ERROR_STOP -- used for calls we EXPECT to fail, to capture rc + text.
db_out() { psql -h 127.0.0.1 -p "$PORT" -U "$(whoami)" -d "$DBNAME" "$@" 2>&1; }

db -c "CREATE EXTENSION IF NOT EXISTS vector"
db -c "CREATE EXTENSION pg_pinecone"
installed_version="$(db -tAc "SELECT extversion FROM pg_extension WHERE extname='pg_pinecone'")"
echo "installed extversion: $installed_version (pre-release-train dev build -- not asserted)" >&2

db <<SQL
SELECT pinecone.set_api_key('$PINECONE_API_KEY');
SQL

db -c "CREATE TABLE $TABLE_A (id bigint CONSTRAINT ${TABLE_A}_pkey PRIMARY KEY, embedding vector($DIM), tag int, label text)"

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
# fraction for every component, so f32 -> Pinecone JSON -> f32 is bit-exact --
# needed for the ATTACH/ATTACHRACE byte-fidelity checks.
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

remote_n="$(remote_count "$INDEX_A")"
if [ "$remote_n" != "$N_ROWS" ]; then
    record SETUP FAIL "describe_index_stats reports $remote_n vectors, expected $N_ROWS"
    exit 1
fi

db -c "CREATE TABLE drill_originals AS SELECT id, embedding, tag, label FROM $TABLE_A"

record SETUP PASS "table A promoted (trigger, index=$INDEX_A); $N_ROWS rows flushed; remote count verified"

# ---------------------------------------------------------------------------
# LEG: DETACH -- live gate-order proof: refuse without UNIQUE/PK, refuse
# without a fresh backup, succeed with force=>true (backup_gate='forced').
# ---------------------------------------------------------------------------
record DETACH RUNNING "gate-order probes: no-PK refusal, no-backup refusal, then force=>true"

db -c "ALTER TABLE $TABLE_A DROP CONSTRAINT ${TABLE_A}_pkey"
no_pk_out="$(db_out -c "SELECT pinecone.detach_storage('$TABLE_A')")"
no_pk_rc=$?
db -c "ALTER TABLE $TABLE_A ADD CONSTRAINT ${TABLE_A}_pkey PRIMARY KEY (id)"

no_backup_out="$(db_out -c "SELECT pinecone.detach_storage('$TABLE_A')")"
no_backup_rc=$?

if [ "$no_pk_rc" -eq 0 ]; then
    record DETACH FAIL "detach_storage should have refused with no UNIQUE/PK on id, but succeeded"
elif ! echo "$no_pk_out" | grep -qi "UNIQUE index or PRIMARY KEY"; then
    record DETACH FAIL "no-PK refusal message doesn't match the documented gate-4 text: $no_pk_out"
elif [ "$no_backup_rc" -eq 0 ]; then
    record DETACH FAIL "detach_storage should have refused with no backup schedule/force, but succeeded"
elif ! echo "$no_backup_out" | grep -qi "backup gate"; then
    record DETACH FAIL "no-backup refusal message doesn't mention the backup gate: $no_backup_out"
else
    db -c "SELECT pinecone.detach_storage('$TABLE_A', force => true)"
    force_rc=$?

    still_non_null="$(db -tAc "SELECT count(*) FROM $TABLE_A WHERE embedding IS NOT NULL")"
    row_count="$(db -tAc "SELECT count(*) FROM $TABLE_A")"
    remote_after="$(remote_count "$INDEX_A")"
    row="$(db -tAc "SELECT storage, backup_gate, detached_at IS NOT NULL FROM pinecone.tables WHERE relid = '$TABLE_A'::regclass")"

    if [ "$force_rc" -ne 0 ]; then
        record DETACH FAIL "detach_storage(force=>true) failed unexpectedly"
    elif [ "$still_non_null" != "0" ] || [ "$row_count" != "$N_ROWS" ] || [ "$remote_after" != "$N_ROWS" ]; then
        record DETACH FAIL "post-detach state wrong: non_null_local=$still_non_null row_count=$row_count remote=$remote_after"
    elif [ "$row" != "detached|forced|t" ]; then
        record DETACH FAIL "pinecone.tables row wrong after forced detach: got '$row', want 'detached|forced|t'"
    else
        record DETACH PASS "refused without UNIQUE/PK; refused without backup+force; force=>true succeeded (storage=detached, backup_gate=forced, detached_at set); $N_ROWS rows NULLed locally, remote intact"
    fi
fi

# ---------------------------------------------------------------------------
# LEG: GUARDS -- live write-guard proof while detached.
# ---------------------------------------------------------------------------
record GUARDS RUNNING "id UPDATE / vector UPDATE+INSERT / id-permutation errors; metadata UPDATE + DELETE work"

VEC_LIT="'[$(python3 -c 'print(",".join(["0.25"]*'"$DIM"'))')]'::vector"

id_update_out="$(db_out -c "UPDATE $TABLE_A SET id = 999001 WHERE id = 5")"
id_update_rc=$?

id_swap_out="$(db_out -c "UPDATE $TABLE_A SET id = CASE id WHEN 3 THEN 4 WHEN 4 THEN 3 END WHERE id IN (3,4)")"
id_swap_rc=$?

vec_update_out="$(db_out -c "UPDATE $TABLE_A SET embedding = $VEC_LIT WHERE id = 6")"
vec_update_rc=$?

vec_insert_out="$(db_out -c "INSERT INTO $TABLE_A (id, embedding, tag, label) VALUES (999002, $VEC_LIT, 1, 'x')")"
vec_insert_rc=$?

db -c "UPDATE $TABLE_A SET tag = 555 WHERE id = 1"
meta_update_rc=$?
db -c "DELETE FROM $TABLE_A WHERE id = 2"
delete_rc=$?

flushed2="$(db -tAc "SELECT pinecone.flush('$TABLE_A', 120000)")"

remote_meta="$(fetch_metadata "$INDEX_A" "$namespace_a" 1)"
remote_after_delete="$(remote_count "$INDEX_A")"

fail_note=""
[ "$id_update_rc" -eq 0 ] && fail_note="id UPDATE should have errored but succeeded; "
echo "$id_update_out" | grep -qi "immutable while detached" || fail_note="${fail_note}id UPDATE error text wrong: $id_update_out; "
[ "$id_swap_rc" -eq 0 ] && fail_note="${fail_note}id-permutation UPDATE should have errored but succeeded; "
echo "$id_swap_out" | grep -qi "immutable while detached" || fail_note="${fail_note}id-permutation error text wrong: $id_swap_out; "
[ "$vec_update_rc" -eq 0 ] && fail_note="${fail_note}vector-value UPDATE should have errored but succeeded; "
echo "$vec_update_out" | grep -qi "remote-only while detached" || fail_note="${fail_note}vector UPDATE error text wrong: $vec_update_out; "
[ "$vec_insert_rc" -eq 0 ] && fail_note="${fail_note}vector-value INSERT should have errored but succeeded; "
echo "$vec_insert_out" | grep -qi "remote-only while detached" || fail_note="${fail_note}vector INSERT error text wrong: $vec_insert_out; "
[ "$meta_update_rc" -ne 0 ] && fail_note="${fail_note}metadata-only UPDATE should have succeeded but errored; "
[ "$delete_rc" -ne 0 ] && fail_note="${fail_note}DELETE should have succeeded but errored; "
[ "$flushed2" != "t" ] && fail_note="${fail_note}post-guard flush did not drain; "
echo "$remote_meta" | grep -q "555" || fail_note="${fail_note}remote metadata.tag not updated to 555 (got: $remote_meta); "
[ "$remote_after_delete" != "$((N_ROWS - 1))" ] && fail_note="${fail_note}remote count after DELETE is $remote_after_delete, want $((N_ROWS - 1)); "

if [ -n "$fail_note" ]; then
    record GUARDS FAIL "$fail_note"
else
    record GUARDS PASS "id UPDATE/permutation ERROR; vector UPDATE+INSERT ERROR (remote-only-while-detached); metadata-only UPDATE reached Pinecone (tag=555); DELETE reached Pinecone (remote count $remote_after_delete)"
fi

# ---------------------------------------------------------------------------
# LEG: ATTACH -- worker recovers, storage flips to full, byte-exact vs
# pre-detach snapshot, id guard removed (id UPDATE works again).
# ---------------------------------------------------------------------------
record ATTACH RUNNING "attach_storage; poll for storage=full; byte-fidelity + guard-removed checks"

db -c "SELECT pinecone.attach_storage('$TABLE_A')"

attach_state=""
for _ in $(seq 1 120); do
    row="$(db -tAc "SELECT storage, queue_state FROM pinecone.status() WHERE table_name = 'public.$TABLE_A'")"
    st="${row%%|*}"
    qs="${row##*|}"
    echo "ATTACH poll: storage=$st queue_state=$qs" >&2
    if [ "$st" = "full" ] && [ "$qs" = "idle" ]; then
        attach_state="done"
        break
    fi
    if [ "$qs" = "parked" ]; then
        attach_state="parked"
        break
    fi
    sleep 1
done

if [ "$attach_state" != "done" ]; then
    record ATTACH FAIL "attach did not complete cleanly (state=$attach_state)"
else
    mismatched="$(db -tAc "SELECT count(*) FROM $TABLE_A t JOIN drill_originals o USING (id) \
        WHERE o.id NOT IN (2) AND t.embedding IS DISTINCT FROM o.embedding")"
    still_null="$(db -tAc "SELECT count(*) FROM $TABLE_A WHERE embedding IS NULL")"
    guard_present="$(db -tAc "SELECT count(*) FROM pg_trigger WHERE tgname = 'pg_pinecone_detached_id_guard' AND tgrelid = '$TABLE_A'::regclass")"

    db -c "UPDATE $TABLE_A SET id = 100007 WHERE id = 7"
    id_update_after_attach_rc=$?
    flushed3="$(db -tAc "SELECT pinecone.flush('$TABLE_A', 60000)")"

    if [ "$mismatched" != "0" ]; then
        record ATTACH FAIL "vector mismatch vs pre-detach snapshot: $mismatched row(s) differ"
    elif [ "$still_null" != "0" ]; then
        record ATTACH FAIL "$still_null row(s) still NULL after attach"
    elif [ "$guard_present" != "0" ]; then
        record ATTACH FAIL "detached id guard trigger still present after attach"
    elif [ "$id_update_after_attach_rc" -ne 0 ] || [ "$flushed3" != "t" ]; then
        record ATTACH FAIL "id UPDATE after attach should succeed (guard removed) but did not (rc=$id_update_after_attach_rc, flushed=$flushed3)"
    else
        record ATTACH PASS "storage=full; 0 vector mismatches vs pre-detach snapshot; id guard trigger gone; id UPDATE succeeds again"
    fi
fi

# ---------------------------------------------------------------------------
# LEG: ATTACHRACE -- the attach-race live proof: re-detach, kick attach_storage, race
# a concurrent detach_storage while its recover job drains. Must NOT re-shed,
# must NOT enqueue a delete; recovered vectors must survive.
# ---------------------------------------------------------------------------
record ATTACHRACE RUNNING "re-detach (force); kick attach_storage; race a concurrent detach_storage"

db -c "SELECT pinecone.detach_storage('$TABLE_A', force => true)"
redetach_rc=$?

if [ "$redetach_rc" -ne 0 ]; then
    record ATTACHRACE FAIL "re-detach before the race setup failed"
else
    db -c "SELECT pinecone.attach_storage('$TABLE_A')"
    race_out="$(db_out -c "SELECT pinecone.detach_storage('$TABLE_A', force => true)")"
    race_rc=$?
    echo "ATTACHRACE: race call rc=$race_rc output=<<$race_out>>" >&2

    race_verdict=""
    if [ "$race_rc" -ne 0 ] && echo "$race_out" | grep -qi "in flight" && echo "$race_out" | grep -qi "Refusing to resume"; then
        race_verdict="errored-in-flight"
    elif [ "$race_rc" -eq 0 ] && echo "$race_out" | grep -qi "nothing to resume"; then
        race_verdict="noticed-no-op"
    elif [ "$race_rc" -eq 0 ] && echo "$race_out" | grep -qi "is now storage = 'detached'"; then
        race_verdict="timing-miss"
    else
        race_verdict="unexpected"
    fi

    if [ "$race_verdict" = "timing-miss" ]; then
        # Self-heal so TEARDOWN/BACKUP legs still see a consistent table --
        # this is a drill-timing miss (the recover job outran the race call),
        # not evidence the fix is broken; report FAIL so a human reruns.
        db -c "SELECT pinecone.attach_storage('$TABLE_A')" >/dev/null 2>&1
        record ATTACHRACE FAIL "race window missed: the recover job (or a prior resume) completed before the racing detach_storage call landed, so a genuine re-detach happened instead of racing it -- rerun the drill (increase N_ROWS or shrink pinecone.worker_naptime_ms if this recurs)"
    elif [ "$race_verdict" = "unexpected" ]; then
        record ATTACHRACE FAIL "race call result matched neither documented outcome: rc=$race_rc out=$race_out"
    else
        recover_state=""
        for _ in $(seq 1 120); do
            row="$(db -tAc "SELECT storage, queue_state FROM pinecone.status() WHERE table_name = 'public.$TABLE_A'")"
            st="${row%%|*}"
            qs="${row##*|}"
            echo "ATTACHRACE poll: storage=$st queue_state=$qs" >&2
            if [ "$st" = "full" ] && [ "$qs" = "idle" ]; then
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
            record ATTACHRACE FAIL "original attach_storage job did not complete after the race (state=$recover_state)"
        else
            mismatched2="$(db -tAc "SELECT count(*) FROM $TABLE_A t JOIN drill_originals o USING (id) \
                WHERE o.id NOT IN (2, 7) AND t.embedding IS DISTINCT FROM o.embedding")"
            renamed_ok="$(db -tAc "SELECT count(*) FROM $TABLE_A t JOIN drill_originals o ON o.id = 7 \
                WHERE t.id = 100007 AND t.embedding IS NOT DISTINCT FROM o.embedding")"
            remote_final="$(remote_count "$INDEX_A")"

            if [ "$mismatched2" != "0" ] || [ "$renamed_ok" != "1" ]; then
                record ATTACHRACE FAIL "recovered vectors did not survive the race intact: mismatched=$mismatched2 renamed_ok=$renamed_ok"
            elif [ "$remote_final" != "$((N_ROWS - 1))" ]; then
                record ATTACHRACE FAIL "remote count changed as a side effect of the race: got $remote_final, want $((N_ROWS - 1)) (a spurious delete would show up here)"
            else
                record ATTACHRACE PASS "race outcome=$race_verdict (never re-shed, never enqueued a delete); original attach completed; recovered vectors byte-exact; remote count unchanged"
            fi
        fi
    fi
fi

# ---------------------------------------------------------------------------
# LEG: DUPID -- the UNIQUE/PK requirement blocks the fabrication path.
# ---------------------------------------------------------------------------
record DUPID RUNNING "on a dedicated detached table, a duplicate-id NULL-vector INSERT must fail (23505)"

db -c "CREATE TABLE $TABLE_C (id bigint PRIMARY KEY, embedding vector($DIM))"
db -c "SELECT pinecone.promote('$TABLE_C', index_name => '$INDEX_C', sync_mode => 'async')" || {
    record DUPID FAIL "pinecone.promote(table C) failed"
}
if [ "${RESULT_DUPID:-}" != "FAIL" ]; then
    CREATED_INDEXES+=("$INDEX_C")
    db -c "INSERT INTO $TABLE_C (id, embedding)
           SELECT gs, (ARRAY(SELECT ((((gs + d) % 17) - 8)::float8 / 8.0) FROM generate_series(0, $((DIM - 1))) d))::vector
           FROM generate_series(1, 5) gs"
    db -c "SELECT pinecone.flush('$TABLE_C', 60000)"
    db -c "SELECT pinecone.detach_storage('$TABLE_C', force => true)"

    dup_out="$(db_out -c "INSERT INTO $TABLE_C (id, embedding) VALUES (1, NULL)")"
    dup_rc=$?

    if [ "$dup_rc" -eq 0 ]; then
        record DUPID FAIL "duplicate-id NULL-vector INSERT should have violated the UNIQUE/PK constraint but succeeded — fabrication path is OPEN"
    elif ! echo "$dup_out" | grep -qi "duplicate key value violates unique constraint"; then
        record DUPID FAIL "INSERT failed but not with the expected unique-violation text: $dup_out"
    else
        record DUPID PASS "duplicate-id NULL-vector INSERT rejected with a unique-constraint violation (23505) -- the trigger's own detached-INSERT guard permits NULL-vector inserts, so only the continuously-enforced PK closes the fabrication path"
    fi

    db -c "SELECT pinecone.attach_storage('$TABLE_C')" >/dev/null 2>&1
fi

# ---------------------------------------------------------------------------
# LEG: BACKUP + RESTORE -- plan-gated real leg (record the exact 403), then a
# clearly-labeled SIMULATED restore (new index + copy) exercising the
# detach -> disaster -> restore -> repoint -> recover chain end to end.
# ---------------------------------------------------------------------------
record BACKUP RUNNING "attempting a real create_backup against index A via the D2 backup_drill test harness"

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
    record BACKUP PASS "create_backup + describe_backup(Ready) + create_index_from_backup + wait_index_ready all succeeded (backup_id=$drill_backup_id) -- this account is NOT plan-gated after all"
    RESTORE_MODE="real"
else
    error_line="$(grep -m1 -E 'DRILL_(BACKUP_CREATE|BACKUP_DESCRIBE|RESTORE_CREATE|RESTORE_WAIT)_ERROR=' "$backup_log" || true)"
    if echo "$error_line" | grep -qiE 'plan|not (available|supported|enabled)|upgrade|forbidden|403'; then
        record BACKUP SKIPPED-plan-gated "$error_line -- a Standard-plan account would additionally prove: real create_index_from_backup dimension/metric fidelity and namespace-scoped restore correctness, without this leg's manual list+fetch+upsert fallback"
    else
        record BACKUP FAIL "$error_line"
    fi
    RESTORE_MODE="simulated"
fi
rm -f "$backup_log"

if [ "$RESTORE_MODE" = "real" ]; then
    record RESTORE PASS "real path: backup restored to a new index ($INDEX_RESTORED) via create_index_from_backup, Ready"
else
    record RESTORE RUNNING "SIMULATED path (NOT a real Pinecone backup restore): creating $INDEX_RESTORED and copying A's vectors via list+fetch+upsert"

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
            if [ "$copy_ok" -eq 0 ]; then
                record RESTORE FAIL "SIMULATED: vector copy loop failed partway (copied $copied so far)"
            else
                # Exercise repoint() + recover_from_remote() against the
                # simulated restore target, closing the detach -> disaster ->
                # restore -> repoint -> recover chain end to end.
                db -c "SELECT pinecone.repoint('$TABLE_A', '$INDEX_RESTORED')"
                repoint_rc=$?
                # Simulate local-only corruption (independent of Pinecone):
                # trigger disabled around the NULLing so it does NOT enqueue
                # a remote delete against the just-repointed index -- a plain
                # UPDATE here would race its own delete against the recover
                # job below and could delete id=10 out of INDEX_RESTORED
                # before recover ever fetches it back.
                db -c "BEGIN;" -c "ALTER TABLE $TABLE_A DISABLE TRIGGER USER;" \
                    -c "UPDATE $TABLE_A SET embedding = NULL WHERE id = 10;" \
                    -c "ALTER TABLE $TABLE_A ENABLE TRIGGER USER;" -c "COMMIT;"
                db -c "SELECT pinecone.recover_from_remote('$TABLE_A')"
                recovered=""
                for _ in $(seq 1 60); do
                    row="$(db -tAc "SELECT queue_state, backlog FROM pinecone.status() WHERE table_name = 'public.$TABLE_A'")"
                    qs="${row%%|*}"; bl="${row##*|}"
                    if [ "$qs" = "idle" ] && [ "${bl:-0}" = "0" ]; then recovered="done"; break; fi
                    sleep 1
                done
                recovered_ok="$(db -tAc "SELECT embedding IS NOT NULL FROM $TABLE_A WHERE id = 10")"

                if [ "$repoint_rc" -ne 0 ] || [ "$recovered" != "done" ] || [ "$recovered_ok" != "t" ]; then
                    record RESTORE FAIL "SIMULATED: copied $copied/$((N_ROWS - 1)) but repoint/recover chain did not complete (repoint_rc=$repoint_rc recovered=$recovered recovered_ok=$recovered_ok)"
                else
                    record RESTORE PASS "SIMULATED (NOT a real backup restore): copied $copied/$((N_ROWS - 1)) vectors from $INDEX_A into new index $INDEX_RESTORED via list+fetch+upsert; repoint() + recover_from_remote() against it both succeeded, closing detach -> disaster -> restore -> repoint -> recover end to end"
                fi
            fi
        fi
    fi
fi

echo "== all legs complete; handing off to teardown (EXIT trap) ==" >&2
