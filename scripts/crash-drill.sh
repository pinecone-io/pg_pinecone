#!/usr/bin/env bash
# Crash/WAL kill drills for the pg_pinecone 0.5 LOCAL (native) engine
# (pg_ctl kill; a correctness check, not a bench bar). Precedent: scripts/vacuum-drill.sh /
# scripts/install-drill.sh (shared-lock + instance-lifecycle pattern) and
# scripts/disaster-recovery-drill.sh (multi-leg record()/report pattern) --
# but unlike every one of those, the LOCAL legs (D1-D8) need NO Pinecone
# credentials at all: local mode never leaves the index relation's own pages.
# The LIFECYCLE legs (D9-D11) exercise pinecone.promote_index/demote_index,
# whose flows genuinely talk to a control/data plane -- fixtured by a tiny
# local HTTP mock (started in SETUP, one process, survives every postmaster
# kill by construction since it lives outside the cluster), still zero real
# credentials.
#
# Each leg: setup -> kill -9 at the target moment -> restart -> verify
# (EXPLAIN-gated index scan + recall-vs-brute-force spot-check + a server-log
# PANIC/ERROR scan across the restart). Why each moment matters:
#   D1 mid-ambuild       -- transactional DDL: an aborted CREATE INDEX must
#                            leave no index relation at all.
#   D2 post-build commit -- GenericXLog WAL replay alone (no REINDEX) must
#                            reconstruct a just-committed index.
#   D3 mid-scan           -- a crash during a pure read must never harm the
#                            index (local-mode amgettuple writes nothing).
#   D4 UNLOGGED reset     -- the init-fork reset path (am.rs's
#                            `ambuildempty`) must leave a VALID EMPTY index,
#                            never an ERROR; repopulating via plain INSERTs
#                            (aminsert, flat mode) + REINDEX then makes
#                            it a functional clustered index again.
#   D5 mid-VACUUM sweep   -- the O(churn) list-page sweep (local::vacuum) is
#                            one GenericXLog record per *changed* page; a
#                            torn sweep must still leave every page valid.
#   D6 streaming replica  -- nice-to-have, not gate (SKIPs with a reason if
#                            it doesn't come up cleanly): GenericXLog is
#                            ordinary WAL, so a hot standby should serve the
#                            same local-mode scans as the primary.
#   D7 mid-insert batch   -- local aminsert: kill -9 partway through a
#                            long autocommit INSERT batch against a clustered
#                            index; committed inserts must replay into the
#                            index (each self-findable), uncommitted ones
#                            must be absent, and a follow-up VACUUM + scans
#                            must work over the insert-written pages. Also
#                            asserts CREATE INDEX CONCURRENTLY stays
#                            rejected (the documented CIC verdict --
#                            local::vacuum's docstring) -- checked here, not
#                            in a #[pg_test], because CIC can't run inside
#                            the test harness's enclosing transaction.
#   D8 concurrent inserts -- two parallel backends inserting through
#                            aminsert into 768-d lists (64 records a page) to
#                            force constant chain extensions: exercises the
#                            relation-extension serialization (ExtendBufferedRel WITHOUT
#                            EB_SKIP_EXTENSION_LOCK). Asserts full chain
#                            integrity via pinecone._debug_local_list_pages:
#                            no block in two chains, chain TIDs exactly ==
#                            heap ctids (both directions, duplicates
#                            included), total record count, plus sampled
#                            end-to-end self-findability.
#   D9 mid-promote-backfill -- kill -9 while pinecone.promote_index's TID
#                            backfill is mid-drain (cursor advanced, chunks
#                            remaining). Post-restart: metapage/registry
#                            still 'promoting', scans still LOCAL-correct
#                            (never-partial-namespace), and once the worker
#                            resumes, the backfill drains from its persisted
#                            block cursor and the flip to remote completes.
#   D10 flip window  -- kill -9 immediately after the worker's
#                            promoting->remote metapage flip commits (no
#                            checkpoint in between): the flip is ONE
#                            GenericXLog record, so WAL replay must
#                            reproduce it atomically -- post-restart scans
#                            dispatch REMOTE, registry says remote, no
#                            leftover backfill job. (Killing *inside* the
#                            flip's own microsecond window isn't steerable
#                            from bash; single-record atomicity is what
#                            makes any intra-window kill equivalent to
#                            before-or-after, which this leg verifies from
#                            the after side. The crash matrix's
#                            metapage-vs-registry divergence case -- flip
#                            durable, registry transaction rolled back -- is
#                            argued in outbox.rs's flip call-site comment:
#                            it converges next round via the idempotency
#                            guard, serving a COMPLETE namespace either way.)
#   D11 mid-demote-rebuild -- kill -9 while the demote's blocking
#                            REINDEX-to-local is mid-build (AEL held,
#                            a 120k-row serial rebuild). Transactional DDL:
#                            the aborted REINDEX leaves the index REMOTE-
#                            serving (metapage still 'demoting' -- never
#                            unavailable, never wrong), the pending_demotes
#                            row survives, and the re-enabled worker retries
#                            to completion (local rebuild + remote cleanup +
#                            registry-row removal).
#
# Health-gate hardening: every leg starts with a
# leg_precheck (instance reachable, else one restart attempt -- a prior
# leg's failure no longer cascades), and every post-kill restart goes
# through restart_instance (one retry with an immediate-stop scrub between
# attempts).
#
# Uses a DEDICATED, throwaway instance (own initdb, own port, own unix
# socket) rather than the shared pgrx dev instance in ~/.pgrx/data-17 --
# every leg here kill -9's the postmaster, which the shared dev instance
# (and whatever else is using it) must never be subjected to. The shared
# pgrx Postgres *install* (binaries + this tree's freshly built .so) is
# still reused -- only the DATA directory is private -- so the whole run
# stays under the same `lockf` convention as every other cargo-pgrx-touching
# script here (serializes against concurrent installs from other worktrees
# sharing ~/.pgrx).
set -uo pipefail

LOCK=/tmp/pg_pinecone-pgrx.lock
if [ "${1:-}" != "--locked" ]; then
    exec lockf -t 1800 "$LOCK" "$0" --locked
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PG_PINECONE_DIR="$REPO_ROOT/pg_pinecone"

PGDATA=/tmp/pg_pinecone_crash_pgdata
SOCKDIR=/tmp/pg_pinecone_crash_sock
SERVER_LOG=/tmp/pg_pinecone_crash_server.log
DBNAME=crashdrill
PGUSER="$(whoami)"
PORT=""

echo "== crash-drill: dedicated instance PGDATA=$PGDATA (own port/socket, release build) ==" >&2

# ---------------------------------------------------------------------------
# Bookkeeping + reporting (disaster-recovery-drill.sh's pattern: macOS ships
# bash 3.2, no associative arrays -- per-leg RESULT_<LEG>/NOTE_<LEG> scalars
# via printf -v / indirect expansion).
# ---------------------------------------------------------------------------
LEG_ORDER=(SETUP D1_AMBUILD D2_POST_BUILD D3_MIDSCAN D4_UNLOGGED D5_VACUUM D6_REPLICA D7_INSERT D8_CONCURRENT D9_PROMOTE D10_FLIP D11_DEMOTE TEARDOWN)

record() {
    printf -v "RESULT_$1" '%s' "$2"
    printf -v "NOTE_$1" '%s' "$3"
    echo "== leg $1: $2 -- $3 ==" >&2
}

# ---------------------------------------------------------------------------
# Instance lifecycle helpers
# ---------------------------------------------------------------------------
free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'
}

postmaster_pid() { sed -n '1p' "$PGDATA/postmaster.pid" 2>/dev/null; }

pgdata_pids() {
    local pmpid; pmpid="$(postmaster_pid)"
    [ -z "$pmpid" ] && return 0
    printf '%s\n' "$pmpid"
    pgrep -P "$pmpid" 2>/dev/null || true
}

wait_all_dead() {
    local max_tries="$1" tries=0
    while true; do
        local any_alive=0 p
        for p in $(pgdata_pids); do
            kill -0 "$p" 2>/dev/null && any_alive=1
        done
        [ "$any_alive" -eq 0 ] && return 0
        tries=$((tries + 1))
        [ "$tries" -ge "$max_tries" ] && return 1
        sleep 0.1
    done
}

# The actual "crash". IMPORTANT: this snapshots and SIGKILLs the postmaster
# *and every current child* (including whatever backend is mid-flight on
# our target operation) essentially simultaneously -- it does NOT rely on
# killing the postmaster alone and waiting for cascading postmaster-death
# detection to tear the rest down. That was this script's own first-draft
# bug (caught empirically): a backend only notices postmaster death at its
# own CHECK_FOR_INTERRUPTS()/wait-event points, so a CPU/IO-bound in-flight
# backend (mid-ambuild, mid-VACUUM-sweep) can simply keep running --
# entirely independent of the postmaster process -- and COMMIT before it
# ever notices, silently turning an intended "kill mid-build" into "kill
# after it finished anyway". Killing every pid visible at this instant is
# the only way the kill lands where the caller actually intends it to.
crash_kill() {
    local pmpid; pmpid="$(postmaster_pid)"
    if [ -z "$pmpid" ]; then
        echo "crash_kill: no postmaster.pid under $PGDATA" >&2
        return 1
    fi
    local pids; pids="$(pgdata_pids)"
    echo "crash_kill: SIGKILL postmaster pid=$pmpid + $(echo "$pids" | wc -l | tr -d ' ') pgdata pids total" >&2
    local p; for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
    if ! wait_all_dead 200; then
        echo "crash_kill: stragglers after 20s, force-killing remaining pgdata pids" >&2
        for p in $(pgdata_pids); do kill -9 "$p" 2>/dev/null || true; done
        wait_all_dead 100 || echo "crash_kill: WARNING stragglers may remain" >&2
    fi
}

start_instance() {
    # -t 120: crash recovery can legitimately take a while when a leg's
    # pre-kill WAL is large (observed: D5's default-timeout start gave up
    # mid-recovery once the suite grew to eight legs, cascading failures
    # into every later leg).
    "$PG_CTL" -D "$PGDATA" -l "$SERVER_LOG" -w -t 120 start
}

wait_for_ready() {
    local tries=0
    until "$PSQL" -h "$SOCKDIR" -p "$PORT" -U "$PGUSER" -d postgres -c 'SELECT 1' >/dev/null 2>&1; do
        tries=$((tries + 1))
        [ "$tries" -ge 480 ] && return 1
        sleep 0.25
    done
}

# Post-kill restart with ONE retry (a single transient start
# hiccup -- socket-file leftovers, a slow recovery pass racing the -w timeout
# -- used to fail the leg AND cascade into every later leg).
restart_instance() {
    if start_instance && wait_for_ready; then return 0; fi
    echo "restart_instance: first start attempt failed, scrubbing and retrying once" >&2
    "$PG_CTL" -D "$PGDATA" -m immediate stop >/dev/null 2>&1 || true
    crash_kill >/dev/null 2>&1 || true
    sleep 2
    start_instance && wait_for_ready
}

db() { "$PSQL" -h "$SOCKDIR" -p "$PORT" -U "$PGUSER" -v ON_ERROR_STOP=1 -d "$DBNAME" "$@"; }

# Per-leg setup check: each leg verifies the instance is
# actually reachable before doing anything, attempting one restart if not --
# a prior leg's failure mode (instance left down) no longer cascades into a
# wall of misleading later-leg FAILs.
leg_precheck() {
    local leg="$1"
    if db -tAc 'SELECT 1' >/dev/null 2>&1; then return 0; fi
    echo "leg_precheck($leg): instance unreachable at leg start, attempting restart" >&2
    if restart_instance && db -tAc 'SELECT 1' >/dev/null 2>&1; then return 0; fi
    record "$leg" FAIL "instance unreachable at leg start (prior leg left it down; restart retry failed too)"
    return 1
}

# Toggle the background worker cluster-wide (SIGHUP GUC). The lifecycle legs
# use this to make their kill windows deterministic: with the worker off, a
# queued promote-backfill/pending-demote sits untouched until the leg either
# drives one step itself (via pinecone._drain_relid / pinecone._demote_attempt from
# a killable foreground backend) or re-enables the worker to finish the job
# end-to-end. ALTER SYSTEM persists across the leg's kill -9 (it lands in
# postgresql.auto.conf), which is exactly what makes the post-restart
# verification window race-free.
set_worker() {
    db -c "ALTER SYSTEM SET pinecone.worker_enabled = $1" >/dev/null
    db -c "SELECT pg_reload_conf()" >/dev/null
    sleep 0.5 # one launcher tick at the drill's 100ms naptime, with margin
}

# Poll helper: retries a scalar query until it returns $2 (or tries run out).
wait_for_value() {
    local query="$1" expected="$2" max_tries="$3" tries=0 got
    while true; do
        got="$("$PSQL" -h "$SOCKDIR" -p "$PORT" -U "$PGUSER" -d "$DBNAME" -tAc "$query" 2>/dev/null | tail -n1)"
        [ "$got" = "$expected" ] && return 0
        tries=$((tries + 1))
        [ "$tries" -ge "$max_tries" ] && return 1
        sleep 0.25
    done
}

log_mark() { wc -l <"$SERVER_LOG" 2>/dev/null | tr -d ' '; }

# Scoped PANIC/ERROR scan across exactly the restart-through-ready window
# (the WAL-replay-correctness check) -- not the whole log, so an unrelated
# benign message elsewhere never taints a leg's verdict.
log_since_clean() {
    local mark="${1:-0}" bad
    bad="$(tail -n "+$((mark + 1))" "$SERVER_LOG" 2>/dev/null | grep -E 'PANIC:|ERROR:' || true)"
    if [ -n "$bad" ]; then
        echo "log_since_clean: found PANIC/ERROR since restart:" >&2
        echo "$bad" >&2
        return 1
    fi
    return 0
}

# Polls pg_stat_activity (a separate watcher connection each time) until
# some OTHER backend's query text contains $1 and it's genuinely 'active' --
# used instead of a blind sleep so the kill lands mid-operation regardless
# of machine speed, no timing calibration needed. `pid <> pg_backend_pid()`
# is load-bearing, not defensive: the watcher's OWN query text embeds the
# search pattern as a string literal (it's right there in the SQL below),
# so without this exclusion the watcher's own row always self-matches and
# this returns "found" on the very first poll, regardless of whether the
# real target operation has even started yet (caught empirically).
wait_for_active_query() {
    local pattern="$1" max_tries="$2" tries=0
    while true; do
        local n
        n="$("$PSQL" -h "$SOCKDIR" -p "$PORT" -U "$PGUSER" -d "$DBNAME" -tAc \
            "SELECT count(*) FROM pg_stat_activity WHERE state = 'active' AND pid <> pg_backend_pid() AND query ILIKE '%${pattern}%'" 2>/dev/null)"
        case "$n" in '' | *[!0-9]*) n=0 ;; esac
        [ "$n" -ge 1 ] && return 0
        tries=$((tries + 1))
        [ "$tries" -ge "$max_tries" ] && return 1
        sleep 0.05
    done
}

# Writes one single-row INSERT statement per line to $4 -- psql -f runs each
# as its own autocommit transaction (no enclosing BEGIN). Ids $2..$3 into
# table $1, vectors drawn from the table's centers fixture (round-robin).
gen_insert_batch() {
    local table="$1" from="$2" to="$3" outfile="$4"
    seq "$from" "$to" | awk -v nclust="$NCLUST_FOR_BATCH" -v tbl="$table" '{
        print "INSERT INTO " tbl " (id, embedding) SELECT " $1 ", (ARRAY(SELECT (e + (random()-0.5)*0.5)::real FROM unnest(c.v::real[]) WITH ORDINALITY AS u(e, ord) ORDER BY ord))::vector FROM " tbl "_centers c WHERE c.c = 1 + (" $1 " % " nclust ");"
    }' >"$outfile"
}

# `n` rows of `dim` dims clustered around `nclust` well-separated random
# centers (jitter range << center range, same ratio lib.rs's own
# `deterministic_clustered_vectors` test fixture uses) -- generated entirely
# in SQL so no client-side vector formatting is needed for 200k-row tables.
# NOTE the `GROUP BY c` aggregate form for the centers, not a bare `ARRAY(
# SELECT random() ...)` subquery: the latter doesn't reference the outer row
# at all, so Postgres's planner is free to hoist it into a single InitPlan
# evaluated ONCE for every row (a well-known volatile-subquery gotcha) --
# every "cluster" would silently collapse onto the same center.
gen_clustered_table() {
    local table="$1" n="$2" dim="$3" nclust="$4"
    db -c "CREATE TABLE $table (id bigint PRIMARY KEY, embedding vector($dim))"
    db -c "CREATE TABLE ${table}_centers AS
           SELECT c, (array_agg(random() * 20 - 10 ORDER BY d))::vector AS v
           FROM generate_series(1, $nclust) c, generate_series(1, $dim) d
           GROUP BY c"
    insert_clustered_rows "$table" "$n" "$nclust"
}

insert_clustered_rows() {
    local table="$1" n="$2" nclust="$3"
    db -c "INSERT INTO $table (id, embedding)
           SELECT gs,
             (ARRAY(SELECT (e + (random() - 0.5) * 0.5)::real
                    FROM unnest(c.v::real[]) WITH ORDINALITY AS u(e, ord)
                    ORDER BY ord))::vector
           FROM generate_series(1, $n) gs
           JOIN ${table}_centers c ON c.c = 1 + (gs % $nclust)"
}

# EXPLAIN-gated: fails unless the pg_pinecone-AM index is actually the plan node
# chosen (enable_seqscan off + pinecone.am_enabled on), not just "some plan ran
# without erroring".
explain_uses_index() {
    local table="$1" idxname="$2" order_expr="$3" plan
    plan="$(db -c "SET enable_seqscan = off; SET pinecone.am_enabled = on; \
        EXPLAIN SELECT id FROM $table ORDER BY $order_expr LIMIT 10" 2>&1)"
    echo "$plan" | grep -q "$idxname"
}

# Mean recall@k over evenly-spaced probe rows: exact = brute-force seqscan
# (enable_seqscan on + pinecone.am_enabled off, same "brute force via pgvector's
# own operator" ground truth the pg_test suite's `nearest_ids` helper uses),
# approx = the pg_pinecone-AM index scan under test. `tail -n1` strips the literal
# "SET"/"SET" acknowledgement lines psql -tAc prints for the two leading SET
# statements in the same multi-statement string.
# The approximate side runs with the 0.9.0 pool (rescore_multiplier 10: 1,000 rows). These legs prove the
# pages and the WAL replay are intact, not the estimator: at the default pool of 300 a 64-d column (64-bit
# codes) on tight synthetic clusters reaches recall@10 0.85 to 1.0 across random draws of the same recipe, at
# 1,000 always 1.0.
recall_check() {
    local table="$1" k="$2" num_queries="$3" n_rows="$4"
    local step=$((n_rows / num_queries))
    local total=0 i qid q exact approx hit
    for i in $(seq 1 "$num_queries"); do
        qid=$(((i - 1) * step + 1))
        q="(SELECT embedding FROM $table WHERE id = $qid)"
        exact="$(db -tAc "SET enable_seqscan = on; SET pinecone.am_enabled = off; \
            SELECT string_agg(id::text, ',') FROM (SELECT id FROM $table ORDER BY embedding <-> $q LIMIT $k) s" | tail -n1)"
        approx="$(db -tAc "SET enable_seqscan = off; SET pinecone.am_enabled = on; SET pinecone.rescore_multiplier = 10; \
            SELECT string_agg(id::text, ',') FROM (SELECT id FROM $table ORDER BY embedding <-> $q LIMIT $k) s" | tail -n1)"
        hit="$(comm -12 <(echo "$exact" | tr ',' '\n' | sort) <(echo "$approx" | tr ',' '\n' | sort) | wc -l | tr -d ' ')"
        total=$((total + hit))
    done
    awk -v t="$total" -v n="$num_queries" -v k="$k" 'BEGIN { printf "%.3f", t / (n * k) }'
}

recall_ok() { awk -v r="$1" 'BEGIN { exit !(r >= 0.8) }'; }

# The approximate top-k of `num_queries` rows' own vectors (the queries of recall_check), one sorted id list per
# line: what a crash must not change in an index that committed.
approx_answers() {
    local table="$1" k="$2" num_queries="$3" n_rows="$4"
    local step=$((n_rows / num_queries)) i qid
    for i in $(seq 1 "$num_queries"); do
        qid=$(((i - 1) * step + 1))
        db -tAc "SET enable_seqscan = off; SET pinecone.am_enabled = on; SET pinecone.rescore_multiplier = 10; \
            SELECT string_agg(id::text, ',' ORDER BY id) FROM (SELECT id FROM $table ORDER BY embedding <-> (SELECT embedding FROM $table WHERE id = $qid) LIMIT $k) s" | tail -n1
    done
}

# ---------------------------------------------------------------------------
# cleanup: stop the dedicated instance (however it needs stopping), remove
# every /tmp artifact, print the final PASS/FAIL table.
# ---------------------------------------------------------------------------
cleanup() {
    local status=$?
    set +e
    record TEARDOWN RUNNING "stopping the dedicated instance and removing /tmp artifacts"

    if [ -n "${PGDATA:-}" ] && [ -f "$PGDATA/postmaster.pid" ]; then
        "$PG_CTL" -D "$PGDATA" -m immediate -w stop >/dev/null 2>&1
        crash_kill >/dev/null 2>&1
    fi
    [ -n "${MOCK_PID:-}" ] && kill "$MOCK_PID" 2>/dev/null
    rm -rf "$PGDATA" "$SOCKDIR" "$SERVER_LOG" /tmp/pg_pinecone_crash_replica_pgdata /tmp/pg_pinecone_crash_replica_sock \
        /tmp/pg_pinecone_crash_initdb.log /tmp/pg_pinecone_crash_basebackup.log /tmp/pg_pinecone_crash_replica_server.log \
        /tmp/pg_pinecone_crash_mock.py /tmp/pg_pinecone_crash_mock.log /tmp/pg_pinecone_crash_drains.* /tmp/pg_pinecone_crash_inserts.*

    record TEARDOWN PASS "instance stopped, /tmp dirs and server log removed (failing legs already captured what they needed into their own NOTE before this point)"

    echo "" >&2
    echo "================= CRASH-DRILL REPORT =================" >&2
    local leg rv nv any_fail=0
    for leg in "${LEG_ORDER[@]}"; do
        rv="RESULT_$leg"; nv="NOTE_$leg"
        printf '%-14s %-9s %s\n' "$leg" "${!rv:-NOT-RUN}" "${!nv:-}" >&2
        [ "${!rv:-}" = "FAIL" ] && any_fail=1
    done
    echo "=====================================================================================" >&2

    [ "$any_fail" -eq 1 ] && status=1
    exit "$status"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# SETUP: release-build install (shared pgrx Postgres install, under the
# lock), initdb a private cluster, own port/socket, start it.
# ---------------------------------------------------------------------------
record SETUP RUNNING "release build + initdb + start dedicated instance"

PG_CONFIG="$(cd "$PG_PINECONE_DIR" && cargo pgrx info pg-config pg17)"
if ! (cd "$PG_PINECONE_DIR" && cargo pgrx install --release -c "$PG_CONFIG"); then
    record SETUP FAIL "cargo pgrx install --release failed"
    exit 1
fi
PG_BINDIR="$(dirname "$PG_CONFIG")"
PSQL="$PG_BINDIR/psql"
INITDB="$PG_BINDIR/initdb"
PG_CTL="$PG_BINDIR/pg_ctl"
PG_BASEBACKUP="$PG_BINDIR/pg_basebackup"

rm -rf "$PGDATA" "$SOCKDIR"
mkdir -p "$SOCKDIR"
if ! "$INITDB" -D "$PGDATA" -U "$PGUSER" --auth=trust -E UTF8 --data-checksums >/tmp/pg_pinecone_crash_initdb.log 2>&1; then
    record SETUP FAIL "initdb failed: $(tail -5 /tmp/pg_pinecone_crash_initdb.log | tr '\n' ' ')"
    exit 1
fi

PORT="$(free_port)"
cat >>"$PGDATA/postgresql.conf" <<CONF
port = $PORT
unix_socket_directories = '$SOCKDIR'
listen_addresses = 'localhost'
shared_preload_libraries = 'pg_pinecone'
checkpoint_timeout = 1h
max_wal_size = 4GB
autovacuum = off
wal_level = replica
max_wal_senders = 4
# Lifecycle legs (D9-D11): the real background worker drives the
# promote-backfill/flip/demote flows -- a snappy naptime keeps their
# polling loops fast. Harmless for the local legs (their outbox is empty).
pinecone.worker_naptime_ms = 100
CONF

if ! restart_instance; then
    record SETUP FAIL "dedicated instance never came up: $(tail -10 "$SERVER_LOG" 2>/dev/null | tr '\n' ' ')"
    exit 1
fi

"$PSQL" -h "$SOCKDIR" -p "$PORT" -U "$PGUSER" -v ON_ERROR_STOP=1 -d postgres -c "CREATE DATABASE $DBNAME"
db -c "CREATE EXTENSION vector"
db -c "CREATE EXTENSION pg_pinecone"
installed_version="$(db -tAc "SELECT extversion FROM pg_extension WHERE extname = 'pg_pinecone'" | tail -n1)"

# Pinecone control+data-plane mock for the lifecycle legs (D9-D11): one
# tiny local HTTP process serving exactly the endpoints promote_index/
# demote_index and the worker's backfill/cleanup touch. It lives OUTSIDE the
# cluster, so every kill -9 leaves it (and its in-memory index registry)
# intact -- the "remote side" deliberately survives each crash, like the
# real Pinecone would.
MOCK_PY=/tmp/pg_pinecone_crash_mock.py
MOCK_LOG=/tmp/pg_pinecone_crash_mock.log
MOCK_PORT="$(free_port)"
MOCK_URL="http://127.0.0.1:$MOCK_PORT"
cat >"$MOCK_PY" <<'PYEOF'
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1])
BASE = "http://127.0.0.1:%d" % PORT
INDEXES = {}

class H(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw or b"{}")
        except Exception:
            return {}

    def _index_desc(self, name):
        info = INDEXES.get(name, {"dimension": 0, "metric": "cosine"})
        return {
            "host": BASE,
            "dimension": info["dimension"],
            "metric": info["metric"],
            "vector_type": "dense",
            "status": {"ready": True, "state": "Ready"},
        }

    def do_POST(self):
        if self.path == "/indexes":
            body = self._body()
            name = body.get("name", "")
            INDEXES[name] = {
                "dimension": body.get("dimension", 0),
                "metric": body.get("metric", "cosine"),
            }
            return self._json(201, self._index_desc(name))
        if self.path.startswith("/vectors/upsert"):
            body = self._body()
            return self._json(200, {"upsertedCount": len(body.get("vectors", []))})
        if self.path.startswith("/vectors/delete"):
            return self._json(200, {})
        if self.path.startswith("/query"):
            return self._json(200, {"matches": []})
        return self._json(404, {"message": "no route: POST %s" % self.path})

    def do_GET(self):
        if self.path.startswith("/indexes/"):
            name = self.path.split("/indexes/", 1)[1].split("?")[0]
            if name in INDEXES:
                return self._json(200, self._index_desc(name))
            return self._json(404, {"message": "index not found"})
        if self.path.startswith("/vectors/list"):
            return self._json(200, {"vectors": []})
        return self._json(404, {"message": "no route: GET %s" % self.path})

    def do_DELETE(self):
        if self.path.startswith("/indexes/"):
            name = self.path.split("/indexes/", 1)[1]
            INDEXES.pop(name, None)
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        return self._json(404, {"message": "no route: DELETE %s" % self.path})

    def log_message(self, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
PYEOF
python3 "$MOCK_PY" "$MOCK_PORT" >"$MOCK_LOG" 2>&1 &
MOCK_PID=$!
mock_tries=0
until (exec 3<>"/dev/tcp/127.0.0.1/$MOCK_PORT") 2>/dev/null; do
    mock_tries=$((mock_tries + 1))
    if [ "$mock_tries" -ge 40 ]; then
        record SETUP FAIL "Pinecone mock never came up on port $MOCK_PORT: $(tail -3 "$MOCK_LOG" 2>/dev/null | tr '\n' ' ')"
        exit 1
    fi
    sleep 0.25
done

record SETUP PASS "dedicated instance up on port $PORT (data checksums on, pg_pinecone $installed_version, release build); Pinecone mock on $MOCK_URL"

# ---------------------------------------------------------------------------
# D1: kill mid-ambuild -- transactional DDL must roll back cleanly.
# ---------------------------------------------------------------------------
drill_ambuild() {
    record D1_AMBUILD RUNNING "200k-row local build, kill -9 mid-ambuild"
    leg_precheck D1_AMBUILD || return
    local table=crash1 n=200000 dim=64 nclust=400

    db -c "DROP TABLE IF EXISTS $table, ${table}_centers CASCADE" >/dev/null
    gen_clustered_table "$table" "$n" "$dim" "$nclust"

    local outfile; outfile="$(mktemp)"
    # Serial build (parallelism 0, session-scoped): the parbuild speedup
    # (~6x) shrank this leg's mid-build window below its tuned sleep — the
    # kill's target is transactional-DDL rollback, which is
    # parallelism-independent (the determinism matrix proves builds are
    # byte-identical across thread counts), so pinning serial here restores
    # the window without weakening what the leg verifies.
    # The SIMD distance kernels then took the serial build from ~7.5 s to ~2.5 s and a fixed sleep missed it:
    # time one serial build of this table first and kill at 40 % of it.
    local pilot_start pilot_end build_ms
    pilot_start="$(python3 -c 'import time; print(int(time.time() * 1000))')"
    if ! db -c "SET pinecone.build_parallelism = 0; CREATE INDEX ${table}_idx ON $table USING pinecone (embedding vector_l2_ops) WITH (mode = 'local', seed = 42)" >/dev/null 2>&1; then
        record D1_AMBUILD FAIL "the pilot build failed"
        return
    fi
    pilot_end="$(python3 -c 'import time; print(int(time.time() * 1000))')"
    build_ms=$((pilot_end - pilot_start))
    db -c "DROP INDEX ${table}_idx" >/dev/null
    db -c "SET pinecone.build_parallelism = 0; CREATE INDEX ${table}_idx ON $table USING pinecone (embedding vector_l2_ops) WITH (mode = 'local', seed = 42)" \
        >"$outfile" 2>&1 &
    local bgpid=$!

    if ! wait_for_active_query "CREATE INDEX" 200; then
        kill "$bgpid" 2>/dev/null; wait "$bgpid" 2>/dev/null
        record D1_AMBUILD FAIL "CREATE INDEX never showed up as an active backend"
        return
    fi
    sleep "$(awk -v ms="$build_ms" 'BEGIN { printf "%.2f", ms * 0.4 / 1000 }')" # 40 % into the build -- genuinely mid-flight, not "at the first tuple".
    local mark; mark="$(log_mark)"
    crash_kill
    wait "$bgpid" 2>/dev/null
    local raced_ahead=0
    grep -q "^CREATE INDEX$" "$outfile" && raced_ahead=1
    rm -f "$outfile"

    if ! restart_instance; then
        record D1_AMBUILD FAIL "instance did not come back up after the mid-ambuild crash"
        return
    fi
    if ! log_since_clean "$mark"; then
        record D1_AMBUILD FAIL "PANIC/ERROR during recovery from the mid-ambuild crash"
        return
    fi

    local relcount rowcount
    relcount="$(db -tAc "SELECT count(*) FROM pg_class WHERE relname = '${table}_idx'" | tail -n1)"
    rowcount="$(db -tAc "SELECT count(*) FROM $table" | tail -n1)"
    if [ "$relcount" != "0" ]; then
        # A kill that provably missed its window (the
        # build's own psql client printed a completed 'CREATE INDEX' before
        # the kill landed) is a DRILL-TIMING miss, not a transactional-DDL
        # violation -- SKIP with the rerun instruction instead of a false
        # FAIL. A surviving relation WITHOUT that race evidence stays a hard
        # FAIL: that would be the real bug this leg exists to catch.
        if [ "$raced_ahead" -eq 1 ]; then
            record D1_AMBUILD SKIPPED "kill missed the window (the build committed first; index legitimately exists) -- rerun the drill for a mid-build kill"
            return
        fi
        record D1_AMBUILD FAIL "aborted CREATE INDEX left a ${table}_idx relation behind (relcount=$relcount) -- transactional DDL violated"
        return
    fi
    if [ "$rowcount" != "$n" ]; then
        record D1_AMBUILD FAIL "heap table lost rows across the crash: expected $n, got $rowcount"
        return
    fi
    if ! db -c "CREATE INDEX ${table}_idx ON $table USING pinecone (embedding vector_l2_ops) WITH (mode = 'local', seed = 42)"; then
        record D1_AMBUILD FAIL "CREATE INDEX did not succeed on retry after the crash"
        return
    fi
    if ! explain_uses_index "$table" "${table}_idx" "embedding <-> (SELECT embedding FROM $table WHERE id = 1)"; then
        record D1_AMBUILD FAIL "rebuilt index is not chosen by the planner"
        return
    fi
    local recall; recall="$(recall_check "$table" 10 10 "$n")"
    local note=""; [ "$raced_ahead" -eq 1 ] && note=" (race note: build finished before the kill this run)"
    if recall_ok "$recall"; then
        record D1_AMBUILD PASS "aborted build left index+catalog absent (rows intact: $rowcount); retry succeeded; recall@10=$recall$note"
    else
        record D1_AMBUILD FAIL "post-recovery recall too low: recall@10=$recall$note"
    fi
}

# ---------------------------------------------------------------------------
# D2: kill immediately after CREATE INDEX commits -- deterministic (no race:
# CREATE INDEX blocks until commit, so the kill always lands post-commit,
# pre-checkpoint).
# ---------------------------------------------------------------------------
drill_post_build_crash() {
    record D2_POST_BUILD RUNNING "kill -9 right after CREATE INDEX commits; verify GenericXLog replay alone, no REINDEX"
    leg_precheck D2_POST_BUILD || return
    local table=crash2 n=150000 dim=64 nclust=300

    db -c "DROP TABLE IF EXISTS $table, ${table}_centers CASCADE" >/dev/null
    gen_clustered_table "$table" "$n" "$dim" "$nclust"
    if ! db -c "CREATE INDEX ${table}_idx ON $table USING pinecone (embedding vector_l2_ops) WITH (mode = 'local', seed = 7)"; then
        record D2_POST_BUILD FAIL "CREATE INDEX itself failed before we ever got to crash it"
        return
    fi

    # What the committed index answers before the crash: after the replay it must answer the same.
    local answers_before; answers_before="$(approx_answers "$table" 10 10 "$n")"
    local mark; mark="$(log_mark)"
    crash_kill
    if ! restart_instance; then
        record D2_POST_BUILD FAIL "instance did not come back up after the post-build crash"
        return
    fi
    if ! log_since_clean "$mark"; then
        record D2_POST_BUILD FAIL "PANIC/ERROR replaying the just-built index's WAL"
        return
    fi
    local recovered=0
    grep -q "database system was not properly shut down" "$SERVER_LOG" && recovered=1

    local relcount
    relcount="$(db -tAc "SELECT count(*) FROM pg_class WHERE relname = '${table}_idx'" | tail -n1)"
    if [ "$relcount" != "1" ]; then
        record D2_POST_BUILD FAIL "committed index did not survive the crash (relcount=$relcount)"
        return
    fi
    if ! explain_uses_index "$table" "${table}_idx" "embedding <-> (SELECT embedding FROM $table WHERE id = 1)"; then
        record D2_POST_BUILD FAIL "surviving index is not chosen by the planner"
        return
    fi
    local answers_after; answers_after="$(approx_answers "$table" 10 10 "$n")"
    if [ -z "$answers_before" ] || [ "$answers_before" != "$answers_after" ]; then
        record D2_POST_BUILD FAIL "the replayed index answers differently from the index that committed"
        return
    fi
    local recall; recall="$(recall_check "$table" 10 10 "$n")"
    if recall_ok "$recall"; then
        record D2_POST_BUILD PASS "index survived with NO REINDEX (recovery-mode replay observed: $([ $recovered -eq 1 ] && echo yes || echo 'not logged')); its answers equal the pre-crash ones; recall@10=$recall"
    else
        record D2_POST_BUILD FAIL "surviving index gives wrong answers: recall@10=$recall"
    fi
}

# ---------------------------------------------------------------------------
# D3: kill mid-scan -- a read-only op; must never harm the index.
# ---------------------------------------------------------------------------
drill_mid_scan() {
    record D3_MIDSCAN RUNNING "long-running forced-full-fraction scan, kill -9 mid-flight"
    leg_precheck D3_MIDSCAN || return
    local table=crash3 n=50000 dim=768 nclust=100

    db -c "DROP TABLE IF EXISTS $table, ${table}_centers CASCADE" >/dev/null
    gen_clustered_table "$table" "$n" "$dim" "$nclust"
    if ! db -c "CREATE INDEX ${table}_idx ON $table USING pinecone (embedding vector_l2_ops) WITH (mode = 'local', seed = 5, small_n_threshold = 1000)"; then
        record D3_MIDSCAN FAIL "CREATE INDEX failed before the scan drill could start"
        return
    fi

    # scan_fraction=1.0 (probe every list) + rescore_multiplier=1000 (max;
    # 100k-candidate exact-rescore pool) -- 768d is large enough that
    # pgvector's own TOASTed vector storage makes each rescore fetch pay a
    # real toast lookup (measured in the sweeps), which is what turns
    # this into a multi-second scan worth killing mid-flight.
    local outfile; outfile="$(mktemp)"
    db -c "SET enable_seqscan = off; SET pinecone.am_enabled = on; SET pinecone.scan_fraction = 1.0; SET pinecone.rescore_multiplier = 1000; \
        SELECT count(*) FROM (SELECT id FROM $table ORDER BY embedding <-> (SELECT embedding FROM $table WHERE id = 1) LIMIT 5000) s" \
        >"$outfile" 2>&1 &
    local bgpid=$!

    if ! wait_for_active_query "SELECT id FROM $table" 200; then
        kill "$bgpid" 2>/dev/null; wait "$bgpid" 2>/dev/null
        record D3_MIDSCAN FAIL "the long-running scan never showed up as an active backend"
        return
    fi
    sleep 0.7 # ~40% into the ~1.6s forced-full-fraction scan.
    local mark; mark="$(log_mark)"
    crash_kill
    wait "$bgpid" 2>/dev/null
    local raced_ahead=0
    grep -qE '^[0-9]+$' "$outfile" && raced_ahead=1
    rm -f "$outfile"

    if ! restart_instance; then
        record D3_MIDSCAN FAIL "instance did not come back up after the mid-scan crash"
        return
    fi
    if ! log_since_clean "$mark"; then
        record D3_MIDSCAN FAIL "PANIC/ERROR after a crash mid-read (a read-only scan should never need WAL replay to matter)"
        return
    fi
    if ! explain_uses_index "$table" "${table}_idx" "embedding <-> (SELECT embedding FROM $table WHERE id = 1)"; then
        record D3_MIDSCAN FAIL "index unusable after the mid-scan crash"
        return
    fi
    local recall; recall="$(recall_check "$table" 10 10 "$n")"
    local note=""; [ "$raced_ahead" -eq 1 ] && note=" (race note: scan finished before the kill this run)"
    if recall_ok "$recall"; then
        record D3_MIDSCAN PASS "index unharmed by a crash mid-read; recall@10=$recall$note"
    else
        record D3_MIDSCAN FAIL "post-crash recall too low: recall@10=$recall$note"
    fi
}

# ---------------------------------------------------------------------------
# D4: UNLOGGED table -- the init-fork reset path must leave a
# VALID EMPTY index; repopulating via plain INSERTs (local aminsert --
# the reset image is a flat empty index, so inserts land as counter bumps
# and flat scans read the heap) + a REINDEX then makes it a functional
# clustered index again.
# ---------------------------------------------------------------------------
drill_unlogged() {
    record D4_UNLOGGED RUNNING "UNLOGGED table + local index; crash must reset to a valid EMPTY index"
    leg_precheck D4_UNLOGGED || return
    local table=crash4 n=20000 dim=32 nclust=40

    db -c "DROP TABLE IF EXISTS $table, ${table}_centers CASCADE" >/dev/null
    db -c "CREATE UNLOGGED TABLE $table (id bigint PRIMARY KEY, embedding vector($dim))"
    db -c "CREATE TABLE ${table}_centers AS
           SELECT c, (array_agg(random() * 20 - 10 ORDER BY d))::vector AS v
           FROM generate_series(1, $nclust) c, generate_series(1, $dim) d
           GROUP BY c"
    insert_clustered_rows "$table" "$n" "$nclust"

    if ! db -c "CREATE INDEX ${table}_idx ON $table USING pinecone (embedding vector_l2_ops) WITH (mode = 'local', seed = 3)"; then
        record D4_UNLOGGED FAIL "CREATE INDEX on the unlogged table failed before we ever crashed it"
        return
    fi

    local mark; mark="$(log_mark)"
    crash_kill
    if ! restart_instance; then
        record D4_UNLOGGED FAIL "instance did not come back up after crashing the unlogged-table index"
        return
    fi
    if ! log_since_clean "$mark"; then
        record D4_UNLOGGED FAIL "PANIC/ERROR during the unlogged-index init-fork reset"
        return
    fi

    # UNLOGGED semantics reset the HEAP to zero rows too (Postgres's own
    # guarantee, not pg_pinecone-specific) -- a cheap sanity check that this is
    # genuinely the reset path, not a no-op restart.
    local heap_rows
    heap_rows="$(db -tAc "SELECT count(*) FROM $table" | tail -n1)"
    if [ "$heap_rows" != "0" ]; then
        record D4_UNLOGGED FAIL "expected the unlogged heap to reset to 0 rows post-crash, got $heap_rows"
        return
    fi

    local scan_out scan_rc
    scan_out="$(db -c "SET enable_seqscan = off; SET pinecone.am_enabled = on; \
        SELECT count(*) FROM (SELECT id FROM $table ORDER BY embedding <-> (SELECT v FROM ${table}_centers LIMIT 1) LIMIT 5) s" 2>&1)"
    scan_rc=$?
    if [ "$scan_rc" -ne 0 ]; then
        record D4_UNLOGGED FAIL "querying the reset (empty) index errored instead of returning empty: $(echo "$scan_out" | tr '\n' ' ')"
        return
    fi

    # Repopulating: the reset image is a flat EMPTY index, so plain
    # INSERTs land through local aminsert (flat mode: counter bump only; the
    # flat scan path reads the heap, exact by construction). The n=20000
    # bulk insert crosses 2x the default small_n_threshold (10000), so the
    # documented flat-growth WARNING fires mid-statement -- expected, not an
    # error. A REINDEX afterwards builds the real clustered engine. (This leg
    # used to DROP + re-CREATE the index because aminsert was a hard
    # stub error.)
    local warn_out; warn_out="$(mktemp)"
    insert_clustered_rows "$table" "$n" "$nclust" 2>"$warn_out"
    # The documented flat->IVF transition WARNING must fire end-to-end: the
    # 20k-row repopulation crosses exactly 2x the default small_n_threshold
    # (10000), so psql's stderr must carry the advisory.
    if ! grep -q "bootstrapped as a flat" "$warn_out"; then
        rm -f "$warn_out"
        record D4_UNLOGGED FAIL "flat-growth WARNING did not fire during the 20k INSERT repopulation"
        return
    fi
    rm -f "$warn_out"
    local flat_recall; flat_recall="$(recall_check "$table" 10 5 "$n")"
    if ! recall_ok "$flat_recall"; then
        record D4_UNLOGGED FAIL "flat (reset) index gives wrong answers after INSERT repopulation: recall@10=$flat_recall"
        return
    fi
    if ! db -c "REINDEX INDEX ${table}_idx"; then
        record D4_UNLOGGED FAIL "REINDEX after INSERT repopulation did not succeed"
        return
    fi
    if ! explain_uses_index "$table" "${table}_idx" "embedding <-> (SELECT embedding FROM $table WHERE id = 1)"; then
        record D4_UNLOGGED FAIL "post-REINDEX index is not chosen by the planner"
        return
    fi
    local recall; recall="$(recall_check "$table" 10 10 "$n")"
    if recall_ok "$recall"; then
        record D4_UNLOGGED PASS "crash reset the unlogged index to empty/valid; INSERT repopulation served exact flat scans (recall@10=$flat_recall) and fired the flat-growth WARNING; REINDEX rebuilt the clustered engine; recall@10=$recall"
    else
        record D4_UNLOGGED FAIL "post-REINDEX index gives wrong answers: recall@10=$recall"
    fi
}

# ---------------------------------------------------------------------------
# D5: kill mid-VACUUM's list-page sweep -- a torn sweep must still leave
# every page valid (one GenericXLog record per page that actually changed).
# ---------------------------------------------------------------------------
drill_vacuum_sweep() {
    record D5_VACUUM RUNNING "kill -9 mid-VACUUM sweep on a table with deletions"
    leg_precheck D5_VACUUM || return
    local table=crash5 n=200000 dim=64 nclust=400

    db -c "DROP TABLE IF EXISTS $table, ${table}_centers CASCADE" >/dev/null
    gen_clustered_table "$table" "$n" "$dim" "$nclust"
    if ! db -c "CREATE INDEX ${table}_idx ON $table USING pinecone (embedding vector_l2_ops) WITH (mode = 'local', seed = 11)"; then
        record D5_VACUUM FAIL "CREATE INDEX failed before the VACUUM drill could start"
        return
    fi
    db -c "DELETE FROM $table WHERE id % 5 = 0"
    # Bound the post-kill recovery to the SWEEP's own WAL -- the thing this
    # leg actually tests. Without this, recovery replays the whole 200k-row
    # build first, which once outran the restart timeout and failed the leg spuriously.
    db -c "CHECKPOINT"

    # The sweep is 850 pages of 26-byte records, over in a few milliseconds: no sleep lands in it. A cost delay
    # makes it last: with a dirty-page cost of 200 the sweep (`vacuum_delay_point` between list pages) sleeps
    # 1 ms after every page it rewrites, about a second whatever the machine. A timing lever only, never a
    # correctness knob.
    local outfile; outfile="$(mktemp)"
    PGOPTIONS='-c vacuum_cost_delay=1 -c vacuum_cost_page_dirty=200 -c vacuum_cost_limit=200' \
        "$PSQL" -h "$SOCKDIR" -p "$PORT" -U "$PGUSER" -v ON_ERROR_STOP=1 -d "$DBNAME" -c "VACUUM $table" >"$outfile" 2>&1 &
    local bgpid=$!
    if ! wait_for_active_query "VACUUM $table" 200; then
        kill "$bgpid" 2>/dev/null; wait "$bgpid" 2>/dev/null
        record D5_VACUUM FAIL "VACUUM never showed up as an active backend"
        return
    fi
    sleep 0.5 # ~half way into the ~1 s sweep (the cost delay above sets its length).
    local mark; mark="$(log_mark)"
    crash_kill
    wait "$bgpid" 2>/dev/null
    local raced_ahead=0
    grep -q "^VACUUM$" "$outfile" && raced_ahead=1
    rm -f "$outfile"
    if [ "$raced_ahead" -eq 1 ]; then
        record D5_VACUUM SKIPPED "kill missed the window (the VACUUM finished first) -- rerun the drill for a mid-sweep kill"
        return
    fi

    if ! restart_instance; then
        record D5_VACUUM FAIL "instance did not come back up after the mid-VACUUM crash"
        return
    fi
    if ! log_since_clean "$mark"; then
        record D5_VACUUM FAIL "PANIC/ERROR recovering from the mid-VACUUM crash"
        return
    fi

    # Scans must work BEFORE any re-VACUUM -- proves the torn sweep itself
    # left valid pages, not just "a second VACUUM papers over it".
    if ! explain_uses_index "$table" "${table}_idx" "embedding <-> (SELECT embedding FROM $table WHERE id = 2)"; then
        record D5_VACUUM FAIL "index unusable immediately after the mid-VACUUM crash (before re-VACUUM)"
        return
    fi
    local scan_out scan_rc
    scan_out="$(db -c "SET enable_seqscan = off; SET pinecone.am_enabled = on; \
        SELECT count(*) FROM (SELECT id FROM $table ORDER BY embedding <-> (SELECT embedding FROM $table WHERE id = 2) LIMIT 10) s" 2>&1)"
    scan_rc=$?
    if [ "$scan_rc" -ne 0 ]; then
        record D5_VACUUM FAIL "post-crash scan errored -- likely a torn sweep page or a stale dead TID: $(echo "$scan_out" | tr '\n' ' ')"
        return
    fi

    if ! db -c "VACUUM $table"; then
        record D5_VACUUM FAIL "re-VACUUM after the crash did not complete"
        return
    fi
    local recall; recall="$(recall_check "$table" 10 10 "$n")"
    if recall_ok "$recall"; then
        record D5_VACUUM PASS "mid-sweep crash left valid pages (pre-re-VACUUM scans already worked); re-VACUUM completed; recall@10=$recall"
    else
        record D5_VACUUM FAIL "post-recovery recall too low: recall@10=$recall"
    fi
}

# ---------------------------------------------------------------------------
# D6: streaming-replica smoke -- nice-to-have (SKIP with reason if setup
# exceeds ~30min). SKIPs cleanly on any setup hiccup rather than
# spending the budget debugging replication plumbing unrelated to the engine.
# ---------------------------------------------------------------------------
drill_replica_smoke() {
    record D6_REPLICA RUNNING "pg_basebackup + hot standby smoke"
    leg_precheck D6_REPLICA || return
    local table=crash2 # reuse D2's already-built, already-verified fixture

    if [ "$(db -tAc "SELECT count(*) FROM pg_class WHERE relname = '${table}_idx'" | tail -n1)" != "1" ]; then
        record D6_REPLICA SKIPPED "prerequisite table/index from D2 isn't in a good state"
        return
    fi

    local rep_pgdata=/tmp/pg_pinecone_crash_replica_pgdata
    local rep_sock=/tmp/pg_pinecone_crash_replica_sock
    local rep_log=/tmp/pg_pinecone_crash_replica_server.log
    rm -rf "$rep_pgdata" "$rep_sock"
    mkdir -p "$rep_sock"

    if ! "$PG_BASEBACKUP" -h "$SOCKDIR" -p "$PORT" -U "$PGUSER" -D "$rep_pgdata" -Fp -Xstream -R -c fast \
        >/tmp/pg_pinecone_crash_basebackup.log 2>&1
    then
        record D6_REPLICA SKIPPED "pg_basebackup failed: $(tail -3 /tmp/pg_pinecone_crash_basebackup.log | tr '\n' ' ')"
        rm -rf "$rep_pgdata" "$rep_sock"
        return
    fi

    local rep_port; rep_port="$(free_port)"
    touch "$rep_pgdata/standby.signal"
    cat >>"$rep_pgdata/postgresql.conf" <<CONF
port = $rep_port
unix_socket_directories = '$rep_sock'
listen_addresses = 'localhost'
primary_conninfo = 'host=$SOCKDIR port=$PORT user=$PGUSER'
CONF

    if ! "$PG_CTL" -D "$rep_pgdata" -l "$rep_log" -w start >/dev/null 2>&1; then
        record D6_REPLICA SKIPPED "standby did not start: $(tail -5 "$rep_log" 2>/dev/null | tr '\n' ' ')"
        rm -rf "$rep_pgdata" "$rep_sock"
        return
    fi

    local tries=0
    until "$PSQL" -h "$rep_sock" -p "$rep_port" -U "$PGUSER" -d "$DBNAME" -c 'SELECT 1' >/dev/null 2>&1; do
        tries=$((tries + 1))
        if [ "$tries" -ge 60 ]; then
            record D6_REPLICA SKIPPED "standby never accepted connections within 30s"
            "$PG_CTL" -D "$rep_pgdata" -m immediate stop >/dev/null 2>&1
            rm -rf "$rep_pgdata" "$rep_sock"
            return
        fi
        sleep 0.5
    done

    local q="SET enable_seqscan = off; SET pinecone.am_enabled = on; \
        SELECT string_agg(id::text, ',') FROM (SELECT id FROM $table ORDER BY embedding <-> (SELECT embedding FROM $table WHERE id = 1) LIMIT 10) s"
    local primary_ids replica_ids
    primary_ids="$(db -tAc "$q" | tail -n1)"
    replica_ids="$("$PSQL" -h "$rep_sock" -p "$rep_port" -U "$PGUSER" -d "$DBNAME" -tAc "$q" | tail -n1)"

    "$PG_CTL" -D "$rep_pgdata" -m fast -w stop >/dev/null 2>&1
    rm -rf "$rep_pgdata" "$rep_sock" "$rep_log"

    if [ -n "$primary_ids" ] && [ "$primary_ids" = "$replica_ids" ]; then
        record D6_REPLICA PASS "standby from pg_basebackup served an identical local-mode index scan: [$primary_ids]"
    else
        record D6_REPLICA FAIL "standby's scan diverged: primary=[$primary_ids] replica=[$replica_ids]"
    fi
}

# ---------------------------------------------------------------------------
# D7: kill mid-insert-batch (local aminsert). A long AUTOCOMMIT batch of
# single-row INSERTs against a clustered local index -- each insert is its
# own transaction, so the kill leaves a committed prefix and an uncommitted
# (or never-run) tail. After restart: committed inserts must have replayed
# into the index (each self-findable through an index scan), the one insert
# possibly in flight at kill time must be all-or-nothing, and a follow-up
# VACUUM + scans must work over insert-written/extended pages. Also asserts
# the CIC verdict (CREATE INDEX CONCURRENTLY stays rejected for local mode)
# here, since CIC can't run inside a #[pg_test]'s enclosing transaction.
# ---------------------------------------------------------------------------
drill_insert_batch() {
    record D7_INSERT RUNNING "autocommit INSERT batch against a clustered index, kill -9 mid-batch"
    leg_precheck D7_INSERT || return
    local table=crash7 n=100000 dim=64 nclust=200 n_insert=20000

    db -c "DROP TABLE IF EXISTS $table, ${table}_centers CASCADE" >/dev/null
    gen_clustered_table "$table" "$n" "$dim" "$nclust"
    if ! db -c "CREATE INDEX ${table}_idx ON $table USING pinecone (embedding vector_l2_ops) WITH (mode = 'local', seed = 13)"; then
        record D7_INSERT FAIL "CREATE INDEX failed before the insert drill could start"
        return
    fi

    # CIC verdict check (see leg comment): must fail with the documented
    # local-mode rejection, and the INVALID index a failed CIC leaves behind
    # must drop cleanly.
    local cic_out
    cic_out="$(db -c "CREATE INDEX CONCURRENTLY ${table}_cic ON $table USING pinecone (embedding vector_l2_ops) WITH (mode = 'local')" 2>&1)"
    if [ $? -eq 0 ]; then
        record D7_INSERT FAIL "CREATE INDEX CONCURRENTLY unexpectedly SUCCEEDED against local mode (the documented rejection is gone)"
        return
    fi
    if ! echo "$cic_out" | grep -q "not supported for a local-mode pg_pinecone index"; then
        record D7_INSERT FAIL "CIC failed with the wrong error: $(echo "$cic_out" | tr '\n' ' ')"
        return
    fi
    db -c "DROP INDEX IF EXISTS ${table}_cic" >/dev/null 2>&1

    # Bound the post-kill recovery to the insert batch's own WAL (the thing
    # this leg tests), not the 100k-row table+index setup's -- same rationale
    # as D5's pre-kill CHECKPOINT.
    db -c "CHECKPOINT"

    # One INSERT statement per line -> psql -f runs each as its own
    # autocommit transaction (no enclosing BEGIN), which is what makes
    # "committed prefix" well-defined at kill time.
    local sqlfile; sqlfile="$(mktemp /tmp/pg_pinecone_crash_inserts.XXXXXX)"
    NCLUST_FOR_BATCH="$nclust" gen_insert_batch "$table" $((n + 1)) $((n + n_insert)) "$sqlfile"

    local outfile; outfile="$(mktemp)"
    "$PSQL" -h "$SOCKDIR" -p "$PORT" -U "$PGUSER" -d "$DBNAME" -q -v ON_ERROR_STOP=1 -f "$sqlfile" >"$outfile" 2>&1 &
    local bgpid=$!

    if ! wait_for_active_query "INSERT INTO crash7" 200; then
        kill "$bgpid" 2>/dev/null; wait "$bgpid" 2>/dev/null
        rm -f "$sqlfile" "$outfile"
        record D7_INSERT FAIL "the INSERT batch never showed up as an active backend"
        return
    fi
    sleep 2 # partway into the multi-second batch -- a committed prefix + a live tail.
    local mark; mark="$(log_mark)"
    crash_kill
    wait "$bgpid" 2>/dev/null
    rm -f "$sqlfile" "$outfile"

    if ! restart_instance; then
        record D7_INSERT FAIL "instance did not come back up after the mid-insert crash"
        return
    fi
    if ! log_since_clean "$mark"; then
        record D7_INSERT FAIL "PANIC/ERROR replaying the insert batch's WAL"
        return
    fi

    local committed
    committed="$(db -tAc "SELECT count(*) FROM $table WHERE id > $n" | tail -n1)"
    local race=""
    if [ "$committed" = "0" ]; then
        record D7_INSERT FAIL "no committed inserts survived -- the kill landed before the first commit (widen the window) or committed inserts were lost"
        return
    fi
    [ "$committed" = "$n_insert" ] && race=" (race note: the whole batch committed before the kill this run)"

    if ! explain_uses_index "$table" "${table}_idx" "embedding <-> (SELECT embedding FROM $table WHERE id = 1)"; then
        record D7_INSERT FAIL "index unusable after the mid-insert crash"
        return
    fi

    # ALL committed rows (base build + every committed insert) must be
    # structurally present in the index chains -- asserted via the
    # pinecone._debug_local_list_pages walker, not by sampling: every visible
    # heap ctid must appear in some chain. (The REVERSE containment is
    # deliberately not asserted: the one insert possibly in flight at kill
    # time may have gotten its index append into WAL without its heap commit
    # -- that record is structurally present but MVCC-invisible to scans,
    # since the exact rescore's heap fetch is snapshot-checked, and the next
    # VACUUM sweep reclaims it.)
    local missing
    missing="$(db -tAc "WITH chain AS (SELECT unnest(tids) AS t FROM pinecone._debug_local_list_pages('${table}_idx'))
                        SELECT count(*) FROM (SELECT ctid::text FROM $table EXCEPT ALL SELECT t FROM chain) x" | tail -n1)"
    if [ "$missing" != "0" ]; then
        record D7_INSERT FAIL "$missing committed row(s) missing from the index chains after replay"
        return
    fi

    # Sampled end-to-end on top of the structural check (full fraction +
    # oversized pool = deterministic: a chain-resident record wins rank 1 at
    # distance ~0).
    local step=$((committed / 5)); [ "$step" -lt 1 ] && step=1
    local i sid got
    for i in 1 2 3 4 5; do
        sid=$((n + (i - 1) * step + 1))
        got="$(db -tAc "SET enable_seqscan = off; SET pinecone.am_enabled = on; SET pinecone.scan_fraction = 1.0; SET pinecone.rescore_multiplier = 1000;             SELECT id FROM $table ORDER BY embedding <-> (SELECT embedding FROM $table WHERE id = $sid) LIMIT 1" | tail -n1)"
        if [ "$got" != "$sid" ]; then
            record D7_INSERT FAIL "committed inserted row $sid did not replay into the index (top-1 was '$got')"
            return
        fi
    done

    # Uncommitted tail: absent from heap AND index SCAN alike. NOTE the MVCC
    # masking: an aborted/in-flight insert's index record can be structurally
    # present in a chain, but the rescore's snapshot heap-fetch hides it --
    # so this check proves scans never SERVE an uncommitted row, not that no
    # such record exists on disk (the walker containment above is one-way
    # for exactly this reason; VACUUM reclaims the strays).
    local idx_committed
    idx_committed="$(db -tAc "SET enable_seqscan = off; SET pinecone.am_enabled = on; SET pinecone.scan_fraction = 1.0; SET pinecone.rescore_multiplier = 1000;         SELECT count(*) FROM (SELECT id FROM $table ORDER BY embedding <-> (SELECT embedding FROM $table WHERE id = $((n + 1))) LIMIT 100000) s WHERE s.id > $n" | tail -n1)"
    if [ "$idx_committed" -gt "$committed" ]; then
        record D7_INSERT FAIL "index served MORE inserted rows ($idx_committed) than the heap committed ($committed) -- an uncommitted insert leaked"
        return
    fi

    # VACUUM interop over insert-written pages, then the overall recall gate.
    db -c "DELETE FROM $table WHERE id > $n AND id % 3 = 0"
    if ! db -c "VACUUM $table"; then
        record D7_INSERT FAIL "post-crash VACUUM over insert-written pages did not complete"
        return
    fi
    local recall; recall="$(recall_check "$table" 10 10 "$n")"
    if recall_ok "$recall"; then
        record D7_INSERT PASS "committed prefix $committed/$n_insert replayed into the index (ALL committed ctids chain-resident + 5/5 sampled self-findable); no uncommitted leak served; post-crash VACUUM + scans OK; CIC rejection verified; recall@10=$recall$race"
    else
        record D7_INSERT FAIL "post-recovery recall too low: recall@10=$recall$race"
    fi
}

# ---------------------------------------------------------------------------
# D8: two-backend concurrent inserts (no crash -- a pure concurrency leg).
# 768-d lists hold 64 records a page (one block of 64 records: a 96-byte code
# and 20 bytes of scalars and TID each), so 2 x 20k parallel inserts across ~174
# lists force on the order of six hundred chain extensions -- plenty of simultaneous-extension
# windows for the relation-extension race the gate flagged (two backends
# handed the same new block => one backend's committed record erased +
# cross-linked chains). (Before 0.10 a `k_bits = 4096` list held 15 records
# a page; the codes are D bits now, so the width comes from the column.)
# Integrity is asserted STRUCTURALLY via pinecone._debug_local_list_pages, not
# by sampling: every chain page listed once (no block in two chains), and
# the multiset of chain TIDs exactly equals the heap's ctids (EXCEPT ALL in
# both directions, so duplicates count too).
# ---------------------------------------------------------------------------
drill_concurrent_inserts() {
    record D8_CONCURRENT RUNNING "2 parallel backends x 20k inserts into 768-d lists, chain-extension heavy"
    leg_precheck D8_CONCURRENT || return
    local table=crash8 n=30000 dim=768 nclust=200 per_backend=20000

    db -c "DROP TABLE IF EXISTS $table, ${table}_centers CASCADE" >/dev/null
    gen_clustered_table "$table" "$n" "$dim" "$nclust"
    if ! db -c "CREATE INDEX ${table}_idx ON $table USING pinecone (embedding vector_l2_ops) WITH (mode = 'local', seed = 21)"; then
        record D8_CONCURRENT FAIL "CREATE INDEX failed before the concurrency drill could start"
        return
    fi

    local f1 f2; f1="$(mktemp /tmp/pg_pinecone_crash_c1.XXXXXX)"; f2="$(mktemp /tmp/pg_pinecone_crash_c2.XXXXXX)"
    NCLUST_FOR_BATCH="$nclust" gen_insert_batch "$table" $((n + 1)) $((n + per_backend)) "$f1"
    NCLUST_FOR_BATCH="$nclust" gen_insert_batch "$table" $((n + per_backend + 1)) $((n + 2 * per_backend)) "$f2"

    local t0; t0="$(date +%s)"
    "$PSQL" -h "$SOCKDIR" -p "$PORT" -U "$PGUSER" -d "$DBNAME" -q -v ON_ERROR_STOP=1 -f "$f1" &
    local pid1=$!
    "$PSQL" -h "$SOCKDIR" -p "$PORT" -U "$PGUSER" -d "$DBNAME" -q -v ON_ERROR_STOP=1 -f "$f2" &
    local pid2=$!
    local rc1=0 rc2=0
    wait "$pid1" || rc1=$?
    wait "$pid2" || rc2=$?
    local elapsed=$(( $(date +%s) - t0 ))
    rm -f "$f1" "$f2"
    if [ "$rc1" -ne 0 ] || [ "$rc2" -ne 0 ]; then
        record D8_CONCURRENT FAIL "a concurrent insert backend errored (rc1=$rc1 rc2=$rc2)"
        return
    fi

    local total=$((n + 2 * per_backend))
    local heap_rows
    heap_rows="$(db -tAc "SELECT count(*) FROM $table" | tail -n1)"
    if [ "$heap_rows" != "$total" ]; then
        record D8_CONCURRENT FAIL "heap row count wrong after concurrent inserts: expected $total, got $heap_rows"
        return
    fi

    # (a) No block appears in two chains (or twice in one).
    local dup_blocks
    dup_blocks="$(db -tAc "SELECT count(*) - count(DISTINCT blkno) FROM pinecone._debug_local_list_pages('${table}_idx')" | tail -n1)"
    if [ "$dup_blocks" != "0" ]; then
        record D8_CONCURRENT FAIL "$dup_blocks chain page(s) appear in more than one chain position -- cross-linked chains"
        return
    fi
    # (b) Chain record count == heap count.
    local chain_records
    chain_records="$(db -tAc "SELECT coalesce(sum(n_records), 0) FROM pinecone._debug_local_list_pages('${table}_idx')" | tail -n1)"
    if [ "$chain_records" != "$total" ]; then
        record D8_CONCURRENT FAIL "chain record count $chain_records != heap count $total (lost or duplicated inserts)"
        return
    fi
    # (c) Chain TID multiset exactly equals the heap's ctids, both directions
    # (EXCEPT ALL keeps duplicates, so a TID stored twice is caught even
    # though the counts in (b) would balance).
    local tid_diff
    tid_diff="$(db -tAc "WITH chain AS (SELECT unnest(tids) AS t FROM pinecone._debug_local_list_pages('${table}_idx')),
                              heap AS (SELECT ctid::text AS t FROM $table)
                         SELECT (SELECT count(*) FROM (SELECT t FROM chain EXCEPT ALL SELECT t FROM heap) x)
                              + (SELECT count(*) FROM (SELECT t FROM heap EXCEPT ALL SELECT t FROM chain) y)" | tail -n1)"
    if [ "$tid_diff" != "0" ]; then
        record D8_CONCURRENT FAIL "chain TIDs diverge from heap ctids by $tid_diff entries (lost/duplicated/phantom records)"
        return
    fi

    # Sampled end-to-end: inserted rows from BOTH backends self-findable.
    local i sid got
    for i in 0 1 2 3 4 5 6 7 8 9; do
        sid=$((n + 1 + i * (2 * per_backend - 1) / 9))
        got="$(db -tAc "SET enable_seqscan = off; SET pinecone.am_enabled = on; SET pinecone.scan_fraction = 1.0; SET pinecone.rescore_multiplier = 1000;             SELECT id FROM $table ORDER BY embedding <-> (SELECT embedding FROM $table WHERE id = $sid) LIMIT 1" | tail -n1)"
        if [ "$got" != "$sid" ]; then
            record D8_CONCURRENT FAIL "concurrently-inserted row $sid not self-findable (top-1 was '$got')"
            return
        fi
    done

    record D8_CONCURRENT PASS "2x${per_backend} concurrent inserts in ${elapsed}s: chains disjoint, chain TIDs == heap ctids (multiset), $chain_records records, 10/10 sampled self-findable"
}

# ---------------------------------------------------------------------------
# D9: kill -9 mid-promote-backfill (lifecycle). Deterministic kill window:
# the worker stays DISABLED for the whole first act (ALTER SYSTEM, which also
# survives the kill via postgresql.auto.conf), and the backfill's first
# chunks are driven by a killable foreground backend (pinecone._drain_relid --
# the same drain_step code path the worker runs). The kill lands once the
# job's persisted block cursor proves a chunk committed (mid-backfill by
# construction, no timing calibration). Post-restart, with the worker still
# off: metapage/registry must still say 'promoting' and scans must still be
# LOCAL-correct (never-partial-namespace). Then the worker is re-enabled and
# must finish the job end-to-end: resume from the persisted cursor, exhaust,
# and flip to remote -- verified by the registry row AND by scan dispatch
# (the mock's /query returns no matches, so a remote-served ORDER BY LIMIT
# comes back EMPTY where the local-served one returned rows: the metapage
# flip is directly observable from plan-visible behavior).
# ---------------------------------------------------------------------------
drill_promote_backfill() {
    record D9_PROMOTE RUNNING "promote_index, kill -9 mid-TID-backfill, resume + flip"
    leg_precheck D9_PROMOTE || return
    local table=crash9 n=30000 dim=32 nclust=100

    set_worker off
    db -c "DROP TABLE IF EXISTS $table, ${table}_centers CASCADE" >/dev/null
    gen_clustered_table "$table" "$n" "$dim" "$nclust"
    if ! db -c "CREATE INDEX ${table}_idx ON $table USING pinecone (embedding vector_l2_ops) WITH (mode = 'local', seed = 31, small_n_threshold = 1000)"; then
        record D9_PROMOTE FAIL "local CREATE INDEX failed before the promote drill could start"
        return
    fi
    db -c "SELECT pinecone.set_api_key('crash-drill-key', '$MOCK_URL')" >/dev/null

    local promote_out
    promote_out="$(db -c "SELECT pinecone.promote_index('${table}_idx')" 2>&1)"
    if [ $? -ne 0 ]; then
        record D9_PROMOTE FAIL "pinecone.promote_index failed: $(echo "$promote_out" | tr '\n' ' ')"
        return
    fi

    local mode
    mode="$(db -tAc "SELECT mode FROM pinecone.tables WHERE relid = '$table'::regclass" | tail -n1)"
    if [ "$mode" != "promoting" ]; then
        record D9_PROMOTE FAIL "expected registry mode 'promoting' right after promote_index, got '$mode'"
        return
    fi
    local recall0; recall0="$(recall_check "$table" 10 5 "$n")"
    if ! recall_ok "$recall0"; then
        record D9_PROMOTE FAIL "scans not local-correct while promoting (pre-kill): recall@10=$recall0"
        return
    fi

    # Drive the backfill from a killable foreground backend (worker is off).
    # One pinecone._drain_relid call processes exactly ONE backfill chunk (an
    # in-progress backfill ends the drain loop's round), so the driver is a
    # psql -f file of paced drain calls -- the pg_sleep between chunks is
    # what makes the kill window wide and deterministic instead of racing
    # 30 fast chunks.
    local sqlfile; sqlfile="$(mktemp /tmp/pg_pinecone_crash_drains.XXXXXX)"
    local i
    for i in $(seq 1 200); do
        printf "SELECT pinecone._drain_relid('%s'::regclass::oid);\nSELECT pg_sleep(0.2);\n" "$table"
    done >"$sqlfile"
    local outfile; outfile="$(mktemp)"
    "$PSQL" -h "$SOCKDIR" -p "$PORT" -U "$PGUSER" -d "$DBNAME" -q -v ON_ERROR_STOP=1 -f "$sqlfile" >"$outfile" 2>&1 &
    local bgpid=$!
    local tries=0 after_id=""
    while true; do
        after_id="$(db -tAc "SELECT coalesce(payload->>'after_id', '') FROM pinecone.outbox WHERE relid = '$table'::regclass AND op = 'backfill'" 2>/dev/null | tail -n1)"
        [ -n "$after_id" ] && break
        if ! kill -0 "$bgpid" 2>/dev/null; then
            record D9_PROMOTE FAIL "the foreground drain finished before any cursor was observed (backfill raced to completion): $(tail -2 "$outfile" | tr '\n' ' ')"
            rm -f "$outfile" "$sqlfile"
            return
        fi
        tries=$((tries + 1))
        if [ "$tries" -ge 400 ]; then
            record D9_PROMOTE FAIL "backfill cursor never advanced (no chunk committed within 20s)"
            kill "$bgpid" 2>/dev/null; wait "$bgpid" 2>/dev/null; rm -f "$outfile" "$sqlfile"
            return
        fi
        sleep 0.05
    done
    local mark; mark="$(log_mark)"
    crash_kill
    wait "$bgpid" 2>/dev/null
    rm -f "$outfile" "$sqlfile"

    if ! restart_instance; then
        record D9_PROMOTE FAIL "instance did not come back up after the mid-backfill crash"
        return
    fi
    if ! log_since_clean "$mark"; then
        record D9_PROMOTE FAIL "PANIC/ERROR recovering from the mid-backfill crash"
        return
    fi

    # Worker still off (auto.conf survived the kill): verify at leisure.
    mode="$(db -tAc "SELECT mode FROM pinecone.tables WHERE relid = '$table'::regclass" | tail -n1)"
    if [ "$mode" != "promoting" ]; then
        record D9_PROMOTE FAIL "post-crash registry mode should still be 'promoting', got '$mode'"
        return
    fi
    local cursor
    cursor="$(db -tAc "SELECT coalesce(payload->>'after_id', '') FROM pinecone.outbox WHERE relid = '$table'::regclass AND op = 'backfill'" | tail -n1)"
    if [ -z "$cursor" ]; then
        record D9_PROMOTE FAIL "the backfill job (or its persisted cursor) did not survive the crash"
        return
    fi
    local recall1; recall1="$(recall_check "$table" 10 5 "$n")"
    if ! recall_ok "$recall1"; then
        record D9_PROMOTE FAIL "scans not local-correct after the mid-backfill crash: recall@10=$recall1"
        return
    fi

    # Re-enable the worker; it must resume from the persisted cursor and flip.
    set_worker on
    if ! wait_for_value "SELECT mode FROM pinecone.tables WHERE relid = '$table'::regclass" "remote" 480; then
        record D9_PROMOTE FAIL "worker never completed the backfill + flip within 120s of re-enable"
        return
    fi
    local backlog
    backlog="$(db -tAc "SELECT count(*) FROM pinecone.outbox WHERE relid = '$table'::regclass" | tail -n1)"
    if [ "$backlog" != "0" ]; then
        record D9_PROMOTE FAIL "outbox not empty after the flip (backlog=$backlog)"
        return
    fi
    # The metapage flip is observable from scan dispatch: remote-served scans
    # hit the mock's /query (no matches) and come back EMPTY.
    local idx_rows
    idx_rows="$(db -tAc "SET enable_seqscan = off; SET pinecone.am_enabled = on; \
        SELECT count(*) FROM (SELECT id FROM $table ORDER BY embedding <-> (SELECT embedding FROM $table WHERE id = 1) LIMIT 5) s" | tail -n1)"
    if [ "$idx_rows" != "0" ]; then
        record D9_PROMOTE FAIL "post-flip index scan still served locally (got $idx_rows rows; expected 0 = remote-dispatched against the empty mock)"
        return
    fi
    record D9_PROMOTE PASS "mid-backfill crash kept 'promoting' + local-correct scans (recall@10=$recall1, cursor=$cursor survived); worker resumed and flipped to remote (scan dispatch verified)"
}

# ---------------------------------------------------------------------------
# D10: the flip window (lifecycle). The promoting->remote metapage flip
# is ONE GenericXLog record; killing inside its microsecond window isn't
# steerable from bash, but single-record atomicity means any such kill is
# equivalent to before-or-after -- so this leg kills IMMEDIATELY after the
# flip is first observed (no checkpoint in between: checkpoint_timeout=1h)
# and asserts pure WAL replay reproduces the flipped state: registry remote,
# scans dispatch remote, no leftover backfill job.
# ---------------------------------------------------------------------------
drill_flip_window() {
    record D10_FLIP RUNNING "kill -9 right after the promoting->remote flip; verify replay"
    leg_precheck D10_FLIP || return
    local table=crash10 n=3000 dim=32 nclust=30

    set_worker on
    db -c "DROP TABLE IF EXISTS $table, ${table}_centers CASCADE" >/dev/null
    gen_clustered_table "$table" "$n" "$dim" "$nclust"
    if ! db -c "CREATE INDEX ${table}_idx ON $table USING pinecone (embedding vector_l2_ops) WITH (mode = 'local', seed = 37, small_n_threshold = 1000)"; then
        record D10_FLIP FAIL "local CREATE INDEX failed before the flip drill could start"
        return
    fi
    db -c "SELECT pinecone.set_api_key('crash-drill-key', '$MOCK_URL')" >/dev/null
    if ! db -c "SELECT pinecone.promote_index('${table}_idx')"; then
        record D10_FLIP FAIL "pinecone.promote_index failed"
        return
    fi

    if ! wait_for_value "SELECT mode FROM pinecone.tables WHERE relid = '$table'::regclass" "remote" 480; then
        record D10_FLIP FAIL "worker never flipped the promotion to remote within 120s"
        return
    fi
    local mark; mark="$(log_mark)"
    crash_kill

    if ! restart_instance; then
        record D10_FLIP FAIL "instance did not come back up after the post-flip crash"
        return
    fi
    if ! log_since_clean "$mark"; then
        record D10_FLIP FAIL "PANIC/ERROR replaying the flip's WAL"
        return
    fi

    local mode
    mode="$(db -tAc "SELECT mode FROM pinecone.tables WHERE relid = '$table'::regclass" | tail -n1)"
    if [ "$mode" != "remote" ]; then
        record D10_FLIP FAIL "registry mode did not survive replay as 'remote' (got '$mode')"
        return
    fi
    local leftover
    leftover="$(db -tAc "SELECT count(*) FROM pinecone.outbox WHERE relid = '$table'::regclass AND op = 'backfill'" | tail -n1)"
    if [ "$leftover" != "0" ]; then
        record D10_FLIP FAIL "a backfill job survived past the flip ($leftover rows) -- flip and job deletion diverged"
        return
    fi
    local idx_rows heap_rows
    idx_rows="$(db -tAc "SET enable_seqscan = off; SET pinecone.am_enabled = on; \
        SELECT count(*) FROM (SELECT id FROM $table ORDER BY embedding <-> (SELECT embedding FROM $table WHERE id = 1) LIMIT 5) s" | tail -n1)"
    heap_rows="$(db -tAc "SELECT count(*) FROM $table" | tail -n1)"
    if [ "$idx_rows" != "0" ] || [ "$heap_rows" != "$n" ]; then
        record D10_FLIP FAIL "post-replay dispatch wrong: index scan returned $idx_rows rows (expected 0 = remote-dispatched) / heap has $heap_rows (expected $n)"
        return
    fi
    record D10_FLIP PASS "flip survived kill -9 via WAL replay of its single GenericXLog record: registry remote, scans dispatch remote, no leftover backfill job, heap intact"
}

# ---------------------------------------------------------------------------
# D11: kill -9 mid-demote-rebuild (lifecycle). The demote's REINDEX-to-
# local is plain transactional DDL, so an aborted rebuild must leave the
# index REMOTE-serving (metapage still 'demoting' -- the never-unavailable
# guarantee's crash case: never wrong, never absent-without-fallback), with
# the pending_demotes row intact for the worker to retry to completion.
# Deterministic window: worker disabled, the rebuild driven by a killable
# foreground backend running the same pinecone._demote_attempt the worker's
# round calls; kill lands while the REINDEX's AccessExclusiveLock is
# observably held (a serial rebuild of 150k 128-d rows: a few seconds).
# ---------------------------------------------------------------------------
drill_demote_rebuild() {
    record D11_DEMOTE RUNNING "demote_index, kill -9 mid-REINDEX-to-local, worker retries to completion"
    leg_precheck D11_DEMOTE || return
    local table=crash11 n=150000 dim=128 nclust=200

    set_worker on
    db -c "DROP TABLE IF EXISTS $table, ${table}_centers CASCADE" >/dev/null
    gen_clustered_table "$table" "$n" "$dim" "$nclust"
    db -c "SELECT pinecone.set_api_key('crash-drill-key', '$MOCK_URL')" >/dev/null
    if ! db -c "CREATE INDEX ${table}_idx ON $table USING pinecone (embedding vector_l2_ops) WITH (mode = 'remote', seed = 41, small_n_threshold = 1000)"; then
        record D11_DEMOTE FAIL "remote CREATE INDEX failed before the demote drill could start"
        return
    fi
    # Let the worker drain the initial TID backfill so the fixture starts
    # fully remote-synced.
    if ! wait_for_value "SELECT count(*) FROM pinecone.outbox WHERE relid = '$table'::regclass" "0" 480; then
        record D11_DEMOTE FAIL "initial remote backfill never drained within 120s"
        return
    fi

    set_worker off
    if ! db -c "SELECT pinecone.demote_index('${table}_idx')"; then
        record D11_DEMOTE FAIL "pinecone.demote_index failed"
        return
    fi
    local pending
    pending="$(db -tAc "SELECT count(*) FROM pinecone.pending_demotes WHERE index_relid = '${table}_idx'::regclass" | tail -n1)"
    if [ "$pending" != "1" ]; then
        record D11_DEMOTE FAIL "expected one pinecone.pending_demotes row after demote_index, got '$pending'"
        return
    fi
    # Queued state serves REMOTE (never unavailable): index-dispatched scans
    # hit the mock (no matches -> 0 rows); the heap is fully intact.
    local idx_rows heap_rows
    idx_rows="$(db -tAc "SET enable_seqscan = off; SET pinecone.am_enabled = on; \
        SELECT count(*) FROM (SELECT id FROM $table ORDER BY embedding <-> (SELECT embedding FROM $table WHERE id = 1) LIMIT 5) s" | tail -n1)"
    heap_rows="$(db -tAc "SELECT count(*) FROM $table" | tail -n1)"
    if [ "$idx_rows" != "0" ] || [ "$heap_rows" != "$n" ]; then
        record D11_DEMOTE FAIL "queued demote not remote-serving: index scan returned $idx_rows rows (expected 0) / heap $heap_rows (expected $n)"
        return
    fi

    # Drive the rebuild from a killable foreground backend (the same
    # _demote_attempt the worker's round calls), then kill mid-REINDEX.
    local outfile; outfile="$(mktemp)"
    # Serial rebuild (same rationale as D1's serial build): keeps the
    # REINDEX a multi-second window so the kill lands mid-flight
    # instead of racing a ~6x-faster parallel build.
    db -c "SET pinecone.build_parallelism = 0; SELECT * FROM pinecone._demote_attempt('${table}_idx'::regclass::oid)" >"$outfile" 2>&1 &
    local bgpid=$!
    local tries=0 ael=0
    while true; do
        ael="$(db -tAc "SELECT count(*) FROM pg_locks WHERE relation = '${table}_idx'::regclass AND mode = 'AccessExclusiveLock' AND granted" 2>/dev/null | tail -n1)"
        [ "$ael" = "1" ] && break
        if ! kill -0 "$bgpid" 2>/dev/null; then
            record D11_DEMOTE FAIL "_demote_attempt finished before its REINDEX lock was ever observed: $(tail -2 "$outfile" | tr '\n' ' ')"
            rm -f "$outfile"
            return
        fi
        tries=$((tries + 1))
        if [ "$tries" -ge 400 ]; then
            record D11_DEMOTE FAIL "REINDEX's AccessExclusiveLock never observed within 20s"
            kill "$bgpid" 2>/dev/null; wait "$bgpid" 2>/dev/null; rm -f "$outfile"
            return
        fi
        sleep 0.05
    done
    sleep 1 # into the multi-second serial rebuild -- genuinely mid-REINDEX
    local mark; mark="$(log_mark)"
    crash_kill
    wait "$bgpid" 2>/dev/null
    rm -f "$outfile"

    if ! restart_instance; then
        record D11_DEMOTE FAIL "instance did not come back up after the mid-REINDEX crash"
        return
    fi
    if ! log_since_clean "$mark"; then
        record D11_DEMOTE FAIL "PANIC/ERROR recovering from the mid-REINDEX crash"
        return
    fi

    # Worker still off: the aborted REINDEX must have left remote-serving
    # state (transactional DDL), pending row intact.
    local scan_out scan_rc
    scan_out="$(db -tAc "SET enable_seqscan = off; SET pinecone.am_enabled = on; \
        SELECT count(*) FROM (SELECT id FROM $table ORDER BY embedding <-> (SELECT embedding FROM $table WHERE id = 1) LIMIT 5) s" 2>&1)"
    scan_rc=$?
    if [ "$scan_rc" -ne 0 ]; then
        record D11_DEMOTE FAIL "post-crash index scan errored instead of serving remote (or local, if raced): $(echo "$scan_out" | tr '\n' ' ')"
        return
    fi
    idx_rows="$(echo "$scan_out" | tail -n1)"
    heap_rows="$(db -tAc "SELECT count(*) FROM $table" | tail -n1)"
    local race=""
    if [ "$idx_rows" != "0" ]; then
        # The kill can lose the race if the rebuild committed first: then the
        # index is already LOCAL-serving. Verify that state honestly instead.
        race=" (race note: the REINDEX committed before the kill this run -- verified the completed-local state instead)"
        if [ "$heap_rows" != "$n" ]; then
            record D11_DEMOTE FAIL "heap lost rows across the crash: expected $n, got $heap_rows$race"
            return
        fi
    else
        if [ "$heap_rows" != "$n" ]; then
            record D11_DEMOTE FAIL "heap lost rows across the crash: expected $n, got $heap_rows"
            return
        fi
        pending="$(db -tAc "SELECT count(*) FROM pinecone.pending_demotes WHERE index_relid = '${table}_idx'::regclass" | tail -n1)"
        if [ "$pending" != "1" ]; then
            record D11_DEMOTE FAIL "pending_demotes row did not survive the mid-REINDEX crash (got '$pending')"
            return
        fi
    fi

    # Re-enable the worker: it must retry (or finish) the demote end-to-end
    # -- local rebuild, remote cleanup, registry-row removal.
    set_worker on
    if ! wait_for_value "SELECT count(*) FROM pinecone.pending_demotes WHERE index_relid = '${table}_idx'::regclass" "0" 960; then
        record D11_DEMOTE FAIL "worker never completed the demote within 240s of re-enable"
        return
    fi
    if ! wait_for_value "SELECT count(*) FROM pinecone.tables WHERE relid = '$table'::regclass" "0" 120; then
        record D11_DEMOTE FAIL "registry row was not removed after the demote completed"
        return
    fi
    # Now genuinely local: index-dispatched scans return real rows, correctly.
    local recall; recall="$(recall_check "$table" 10 10 "$n")"
    if recall_ok "$recall"; then
        record D11_DEMOTE PASS "mid-REINDEX crash left remote-serving state + pending row; worker retried to completion (local rebuild + registry cleanup); post-demote local recall@10=$recall$race"
    else
        record D11_DEMOTE FAIL "post-demote local index gives wrong answers: recall@10=$recall$race"
    fi
}

# ---------------------------------------------------------------------------
# Run all eleven legs. Each is independent (its own table/index), so one
# leg's failure never blocks the rest -- the report is the deliverable.
# ---------------------------------------------------------------------------
drill_ambuild
drill_post_build_crash
drill_mid_scan
drill_unlogged
drill_vacuum_sweep
drill_replica_smoke
drill_insert_batch
drill_concurrent_inserts
drill_promote_backfill
drill_flip_window
drill_demote_rebuild

echo "== all legs complete; handing off to teardown (EXIT trap) ==" >&2
