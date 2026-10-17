"""Index builds and query-sweep execution (unfiltered, filtered, concurrent).

Concurrency is process-based: each simulated
connection is a real OS process with its own psycopg connection, not a
Python thread sharing one interpreter/GIL -- avoids the GIL understating
what independent client connections would actually achieve. Workers barrier
after warmup so the measured wall clock only covers measured queries.

Every sweep point is gated on an EXPLAIN (FORMAT JSON) plan check:
if the planner isn't using the index under test
(stale index from another system, or the planner gave up and fell back to a
seqscan), that's recorded on the point instead of silently blended into
"index performance" numbers.
"""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from . import metrics as metrics_mod
from . import remote as remote_mod
from .groundtruth import recall_at_k
from .systems import System, distance_operator
from .util import RssSampler, timed


def _assert_phase_b_pg_pinecone_installed(conn, system: System | None = None) -> None:
    """Fails loud, with the actual cause named, instead of a confusing
    downstream error if the Postgres install this harness launches
    (`PG_PINECONE_BENCH_PG`; see `systems.py`'s module docstring -- the harness never
    installs pg_pinecone itself) holds a different pg_pinecone build than expected.

    `SELECT extversion` cannot tell builds apart (several branches report the
    same version string), so the check is whether the SQL surface the run
    needs is present: `pinecone.last_scan_mode()` always; plus, when `system`
    asks for iterative-scan axes or `record_scan_stats`, the
    `pinecone.iterative_scan` / `pinecone.max_scan_tuples` GUCs and
    `pinecone.last_scan_stats()`.
    """
    try:
        conn.execute("SELECT pinecone.last_scan_mode()")
    except Exception as e:
        raise RuntimeError(
            "pg_pinecone_native: the pg_pinecone extension in the install under PG_PINECONE_BENCH_PG does not "
            "have pinecone.last_scan_mode() (native-engine AM) -- wrong or stale build. "
            "Re-install: cd <repo>/pg_pinecone && PGRX_HOME=<private pgrx home> "
            "cargo pgrx install --release --pg-config <that home>/17.10/pgrx-install/bin/pg_config. "
            f"Underlying error: {e}"
        ) from e
    if system is None:
        return
    wants_iter = (any(v is not None for v in system.iterative_scan_values)
                  or any(v is not None for v in system.filtered_iterative_scan_values)
                  or system.scan_budget_param is not None)
    missing = []
    if wants_iter:
        for guc in ("pinecone.iterative_scan", "pinecone.max_scan_tuples"):
            try:
                conn.execute(f"SHOW {guc}")
            except Exception:
                missing.append(guc)
    if system.record_scan_stats:
        try:
            conn.execute("SELECT batches, rescored, capped FROM pinecone.last_scan_stats()")
        except Exception:
            missing.append("pinecone.last_scan_stats()")
    if missing:
        raise RuntimeError(
            f"pg_pinecone_native system {system.name!r} needs {missing} but the installed pg_pinecone "
            "build lacks them -- the iterative-scan work is not in "
            "this build. Rebuild from a tree that has it, or drop iterative_scan_values/"
            "scan_budget_param/record_scan_stats from the config.")


def drop_all_secondary_indexes(conn, table: str) -> list[str]:
    """Drops every non-primary-key index on `table`. Guards against
    cross-system index contamination: a leftover
    index from a system that already built+swept can otherwise sit alongside
    the system under test and get chosen by the planner, or just physically
    exist and confuse later `pg_indexes` bookkeeping. Returns the names
    dropped, for the results record.
    """
    rows = conn.execute(
        "SELECT indexname FROM pg_indexes WHERE tablename = %s AND indexname !~ %s",
        (table, r"_pkey$"),
    ).fetchall()
    dropped = [r[0] for r in rows]
    for idxname in dropped:
        conn.execute(f"DROP INDEX IF EXISTS {idxname}")
    return dropped


def build_index(pg, table: str, column: str, metric: str, system: System, n_rows: int,
                remote: dict | None = None) -> dict:
    """Builds `system`'s index, tracking wall time, leader-backend peak RSS, and
    on-disk size. Drops every OTHER index on the table first (not just this
    system's own index name) so the table never carries more than one ANN
    index at a time -- see `drop_all_secondary_indexes`.

    Captures every server NOTICE raised during the build: pgvector's hnsw build emits "hnsw graph no longer fits into
    maintenance_work_mem after N tuples" as a NOTICE, not a log-level warning
    gated by log_min_messages -- it's sent straight to the client regardless
    of server logging config, so a psycopg notice handler is the reliable way
    to capture it. `graph_spilled_to_disk` is a quick boolean flag derived
    from that specific message so any spill regime is visible without
    grepping `build_notices`.

    `system.kind == "exact"` (the forced-seqscan baseline) builds
    NO index at all -- it exists to measure the honest incumbent (a plain
    table scan), so "build" is instantaneous and every OTHER index is still
    dropped first (the table must carry zero secondary indexes while this
    series runs, or the planner could pick one of them instead of the seqscan
    we're trying to measure).
    """
    if system.kind in remote_mod.REMOTE_KINDS:
        return remote_mod.build(pg, table, column, metric, system, n_rows, **(remote or {}))
    with pg.connect() as conn:
        dropped = drop_all_secondary_indexes(conn, table)
        if system.kind == "exact":
            return dict(build_seconds=0.0, peak_rss_bytes=0, index_size_bytes=0,
                        dropped_indexes=dropped, build_notices=[], graph_spilled_to_disk=False)
        for stmt in system.pre_build_sql(table, column, metric, n_rows):
            conn.execute(stmt)
        if system.kind == "pg_pinecone_native":
            _assert_phase_b_pg_pinecone_installed(conn, system)
        pid = conn.execute("SELECT pg_backend_pid()").fetchone()[0]
        sql = system.build_sql(table, column, metric, n_rows)
        notices: list[str] = []
        conn.add_notice_handler(lambda diag: notices.append(diag.message_primary))
        with RssSampler(pid) as sampler, timed() as t:
            conn.execute(sql)
        size_bytes = conn.execute(
            "SELECT pg_relation_size(%s)", (system.index_name(table),)
        ).fetchone()[0]
        calibration = (_index_calibration(conn, system.index_name(table))
                       if system.kind == "pg_pinecone_native" else None)
    graph_spilled = any(n and "no longer fits into maintenance_work_mem" in n for n in notices)
    out = dict(build_seconds=t["seconds"], peak_rss_bytes=sampler.peak_bytes,
               index_size_bytes=size_bytes, dropped_indexes=dropped,
               build_notices=notices, graph_spilled_to_disk=graph_spilled)
    if system.kind == "pg_pinecone_native":
        out["calibration"] = calibration
    return out


def _index_calibration(conn, index: str) -> dict | None:
    """Every column of `pinecone.index_calibration(index)` (0.10+): the probe count and pool a scan
    uses by default and what the build measured for `target_recall`. None on a build without it."""
    if conn.execute("SELECT to_regprocedure('pinecone.index_calibration(regclass)')").fetchone()[0] is None:
        return None
    cur = conn.execute("SELECT * FROM pinecone.index_calibration(%s::regclass)", (index,))
    row = cur.fetchone()
    return dict(zip([d.name for d in cur.description], row))


_PLAN_INDEX_NODE_TYPES = {"Index Scan", "Index Only Scan"}


def _classify_plan_node(node: dict):
    """Recursively finds the first Index Scan / Index Only Scan or Seq Scan
    node in an EXPLAIN plan tree. Returns (plan_shape, index_name_or_None)."""
    node_type = node.get("Node Type", "")
    if node_type in _PLAN_INDEX_NODE_TYPES:
        idx = node.get("Index Name")
        return f"index_scan:{idx}", idx
    if node_type == "Seq Scan":
        return "seqscan_sort", None
    for child in node.get("Plans", []):
        shape, idx = _classify_plan_node(child)
        if shape is not None:
            return shape, idx
    return None, None


# GUCs to SHOW and stamp on every sweep point, per system kind: the harness's *intent* (what it SET) and the session's
# *effective* value can differ (e.g. a config that only sets the budget for
# relaxed_order leaves ivfflat.max_probes at whatever the server default is)
# -- recording the effective value is what makes a budget-capped point
# provable from the results JSON alone.
_EFFECTIVE_GUC_NAMES = {
    "hnsw": ("hnsw.max_scan_tuples", "hnsw.scan_mem_multiplier"),
    "ivfflat": ("ivfflat.max_probes",),
    "pg_pinecone_native": ("pinecone.rescore_multiplier", "pinecone.scan_fraction",
                        "pinecone.mode_l_threshold", "pinecone.force_scan_mode",
                        "pinecone.iterative_scan", "pinecone.max_scan_tuples",
                        "pinecone.scan_prune"),
    "pg_pinecone_remote": ("pinecone.am_min_topk", "pinecone.am_max_topk", "pinecone.iterative_scan",
                        "pinecone.max_scan_tuples", "pinecone.worker_naptime_ms"),
}


# pg_pinecone GUCs a build without the iterative-scan work or the
# early stop does not define: recorded as "n/a" instead of failing the SHOW.
_OPTIONAL_GUCS = {"pinecone.iterative_scan", "pinecone.max_scan_tuples", "pinecone.scan_prune"}


def _capture_effective_gucs(conn, system: System) -> dict:
    out = {}
    for name in _EFFECTIVE_GUC_NAMES.get(system.kind, ()):
        try:
            out[name] = conn.execute(f"SHOW {name}").fetchone()[0]
        except psycopg.Error:
            if name not in _OPTIONAL_GUCS:
                raise
            out[name] = "n/a"
    return out


def capture_plan(dsn: str, table: str, column: str, op: str, system: System, sweep_value,
                  iterative_scan, where_clause: str, q_vec, k: int, scan_budget=None,
                  force_index: bool = False, force_seqscan: bool = False,
                  scan_fraction=None):
    """Runs EXPLAIN (FORMAT JSON) for one representative query at this sweep
    point's exact session settings and classifies the plan shape
   . EXPLAIN without ANALYZE doesn't execute the
    query, so this is cheap and doesn't perturb the measured connections --
    it uses its own short-lived connection.

    `force_index=True` sets `enable_seqscan = off` first: used for the forced-plan companion series that answers "did
    the planner give up, or is the index really incapable" when every
    default-planner point for a filter fell back to a seqscan.

    `force_seqscan=True` is the opposite lever (the forced-seqscan
    baseline series): sets `enable_seqscan = on`, `enable_indexscan = off`,
    `enable_bitmapscan = off` so the plan is a plain table scan regardless of
    whatever index (if any) happens to exist -- the honest exact-scan
    incumbent, not "whatever the planner picked." A resulting `seqscan_sort`
    shape is relabeled `seqscan_forced` so it's never confused with an
    ANN system's organic planner-gives-up fallback in the same results file.

    Returns `(plan_shape, index_name, effective_gucs)` -- `effective_gucs` is
    this system kind's iterative-scan-budget GUCs as SHOW reports them under
    this exact session setup, not just what the harness thinks it set.
    """
    conn = psycopg.connect(dsn, autocommit=True, prepare_threshold=None)
    register_vector(conn)
    try:
        # pgvector's GUCs (hnsw.*, ivfflat.*) are only registered once its
        # shared library is loaded into THIS backend -- which normally
        # happens lazily, the first time a query actually needs one of its
        # functions (e.g. the planner costing an index path). SET on an
        # unloaded GUC namespace is harmless (Postgres creates a placeholder
        # it reconciles once the library loads), but SHOW on a GUC that was
        # never SET and whose extension isn't loaded yet raises
        # "unrecognized configuration parameter". Loading explicitly here
        # makes every SET/SHOW below deterministic regardless of whether
        # this particular plan happens to cost the index.
        conn.execute("LOAD 'vector'")
        if system.kind in ("pg_pinecone_native", "pg_pinecone_remote"):
            conn.execute("LOAD 'pg_pinecone'")
        if force_index:
            conn.execute("SET enable_seqscan = off")
        if force_seqscan:
            conn.execute("SET enable_seqscan = on")
            conn.execute("SET enable_indexscan = off")
            conn.execute("SET enable_bitmapscan = off")
        system.set_session_params(conn, sweep_value, iterative_scan, scan_budget, scan_fraction)
        effective_gucs = _capture_effective_gucs(conn, system)
        sql, params = system.query_sql(table, column, op, where_clause, q_vec, k)
        plan = conn.execute("EXPLAIN (FORMAT JSON) " + sql, params).fetchone()[0]
        root = plan[0]["Plan"]
        shape, index_name = _classify_plan_node(root)
        if shape is None:
            shape = f"other:{root.get('Node Type', 'unknown')}"
        if force_seqscan and shape == "seqscan_sort":
            shape = "seqscan_forced"
        return shape, index_name, effective_gucs
    finally:
        conn.close()


def _find_index_node(node: dict):
    """First Index Scan / Index Only Scan node of an EXPLAIN plan tree, or None."""
    if node.get("Node Type") in _PLAN_INDEX_NODE_TYPES:
        return node
    for child in node.get("Plans", []):
        hit = _find_index_node(child)
        if hit is not None:
            return hit
    return None


def explain_cost_probe(dsn: str, table: str, column: str, op: str, system: System, sweep_value,
                       iterative_scan, where_clause: str, probe_queries, k: int, scan_budget=None,
                       force_index: bool = False, scan_fraction=None) -> dict:
    """Per-query cost at one sweep point from EXPLAIN (ANALYZE, BUFFERS, TIMING OFF) on a few
    queries, on its own connection after every timer has stopped (never perturbs latency).
    `ea_fetched_mean` = tuples the executor pulled through the index scan node (rows kept +
    rows removed by the residual Filter / recheck) = its heap fetches; buffers are the whole
    query (index pages, heap pages, pg_pinecone's own rescore fetches). Never raises."""
    conn = psycopg.connect(dsn, autocommit=True, prepare_threshold=None)
    try:
        register_vector(conn)
        conn.execute("LOAD 'vector'")
        if system.kind == "pg_pinecone_native":
            conn.execute("LOAD 'pg_pinecone'")
        if force_index:
            conn.execute("SET enable_seqscan = off")
        system.set_session_params(conn, sweep_value, iterative_scan, scan_budget, scan_fraction)
        fetched, removed, hit, read = [], [], [], []
        for q in probe_queries:
            sql, params = system.query_sql(table, column, op, where_clause, q, k)
            top = conn.execute("EXPLAIN (ANALYZE, BUFFERS, TIMING OFF, FORMAT JSON) " + sql,
                               params).fetchone()[0][0]["Plan"]
            node = _find_index_node(top)
            if node is None:
                continue
            rm = node.get("Rows Removed by Filter", 0) + node.get("Rows Removed by Index Recheck", 0)
            removed.append(rm)
            fetched.append(node.get("Actual Rows", 0) + rm)
            hit.append(top.get("Shared Hit Blocks", 0))
            read.append(top.get("Shared Read Blocks", 0))
        if not fetched:
            return {}
        return dict(ea_n=len(fetched), ea_fetched_mean=float(np.mean(fetched)),
                    ea_removed_mean=float(np.mean(removed)), ea_buf_hit_mean=float(np.mean(hit)),
                    ea_buf_read_mean=float(np.mean(read)))
    except Exception as e:  # a probe failure must never cost a multi-hour run
        return dict(ea_error=f"{type(e).__name__}: {e}"[:200])
    finally:
        conn.close()


def _run_one_query(conn, table, column, op, system, q_vec, k, where_clause):
    sql, params = system.query_sql(table, column, op, where_clause, q_vec, k)
    start = time.perf_counter()
    rows = conn.execute(sql, params).fetchall()
    elapsed = time.perf_counter() - start
    return [r[0] for r in rows], elapsed


# Per-worker-process global connection, set once by _pool_worker_init. Spawn
# start method (default on macOS) means each pool worker is a fresh
# interpreter -- module-level state here is per-process, not shared.
_worker_conn = None
_SCAN_STATS_SQL = None


def _scan_stats_sql(conn) -> str:
    """`last_scan_stats()` has (batches, rescored, capped) in iterative-scan builds and gains
    (deepened, lists_probed) with deepen-before-widen; use the wider row when the build has it."""
    global _SCAN_STATS_SQL
    if _SCAN_STATS_SQL is None:
        try:
            conn.execute("SELECT deepened, lists_probed FROM pinecone.last_scan_stats()")
            _SCAN_STATS_SQL = ("SELECT batches, rescored, capped, deepened, lists_probed "
                               "FROM pinecone.last_scan_stats()")
        except psycopg.Error:
            _SCAN_STATS_SQL = "SELECT batches, rescored, capped FROM pinecone.last_scan_stats()"
    return _SCAN_STATS_SQL


def _pool_worker_init(dsn: str):
    global _worker_conn
    _worker_conn = psycopg.connect(dsn, autocommit=True, prepare_threshold=None)
    register_vector(_worker_conn)


def _pool_worker_run(table, column, op, system: System, sweep_value, iterative_scan, scan_budget,
                      force_index, force_seqscan, warmup_queries, measured_queries, k, where_clause,
                      recall_barrier, throughput_barrier, min_throughput_seconds,
                      pg_pinecone_expected_mode=None, scan_fraction=None):
    """Runs in a worker *process* (real OS-level
    concurrency, not GIL-shared threads). Session GUCs are set exactly once
    for this point, then warmup runs untimed, then every worker
    barriers before starting its own wall clock.

    Two measured passes, deliberately separated:

    1. Recall pass: the fixed `measured_queries` set, once through -- this is
       what recall@k and the p50/p95/p99 latency percentiles come from.
    2. Throughput pass: the SAME query stream, looped until at least
       `min_throughput_seconds` of wall clock have elapsed for this worker,
       counting completed queries. A fixed 500-query set split 8 ways can
       finish in tens of milliseconds at a cheap sweep point -- far too short
       a window for a wall-clock QPS to mean anything (scheduler jitter
       dominates). This pass's queries are never counted toward recall.

    `pg_pinecone_expected_mode`, if given (pg_pinecone_native systems only -- see
    `System.pg_pinecone_native_expected_mode`), is asserted against
    `pinecone.last_scan_mode()` after EVERY recall-pass query (a protocol
    constraint: a benchmark cell must prove which scan path
    actually served it, not just report a number and assume). Checked only
    on the recall pass, not the throughput pass -- the mode can't differ
    between them (same query set, same session settings) and the extra
    round trip per query would cost real throughput-pass wall clock, which
    IS a measured number there.
    """
    conn = _worker_conn
    if force_index:
        conn.execute("SET enable_seqscan = off")
    if force_seqscan:
        conn.execute("SET enable_seqscan = on")
        conn.execute("SET enable_indexscan = off")
        conn.execute("SET enable_bitmapscan = off")
    system.set_session_params(conn, sweep_value, iterative_scan, scan_budget, scan_fraction)
    backend_pid = conn.execute("SELECT pg_backend_pid()").fetchone()[0]
    budget = system.point_budget_seconds
    warm_end = time.perf_counter() + budget / 5 if budget else None
    for q in warmup_queries:
        _run_one_query(conn, table, column, op, system, q, k, where_clause)
        if warm_end is not None and time.perf_counter() > warm_end:
            break

    recall_barrier.wait()
    results = []
    # Query-time peak RSS of THIS worker's own backend, sampled across the
    # recall pass (report it, no bar yet). Same
    # directional-signal caveats as the build-time sampler (`util.RssSampler`).
    recall_end = time.perf_counter() + budget if budget else None
    with RssSampler(backend_pid) as rss:
        for q in measured_queries:
            if recall_end is not None and results and time.perf_counter() > recall_end:
                break
            result = _run_one_query(conn, table, column, op, system, q, k, where_clause)
            if pg_pinecone_expected_mode is not None:
                mode = conn.execute("SELECT pinecone.last_scan_mode()").fetchone()[0]
                assert mode == pg_pinecone_expected_mode, (
                    f"pg_pinecone_native: expected scan mode {pg_pinecone_expected_mode!r}, "
                    f"pinecone.last_scan_mode() reported {mode!r} "
                    f"(where_clause={where_clause!r}, sweep_value={sweep_value!r})"
                )
            scan_stats = None
            if system.record_scan_stats:
                # (batches, rescored, capped[, deepened, lists_probed]) of the query just run --
                # read on the recall pass only, after the timer, so it never costs latency.
                scan_stats = conn.execute(_scan_stats_sql(conn)).fetchone()
            results.append((*result, scan_stats))
    query_peak_rss = rss.peak_bytes

    throughput_barrier.wait()
    n_completed = 0
    t_start = time.perf_counter()
    t_now = t_start
    while t_now - t_start < min_throughput_seconds:
        for q in measured_queries:
            _run_one_query(conn, table, column, op, system, q, k, where_clause)
            n_completed += 1
            t_now = time.perf_counter()
            if t_now - t_start >= min_throughput_seconds:
                break
    t_end = time.perf_counter()

    return results, n_completed, t_start, t_end, query_peak_rss


def run_sweep_point(pool, manager, dsn: str, table: str, column: str, metric: str,
                     system: System, sweep_value, iterative_scan, measured_queries: np.ndarray,
                     warmup_pool: np.ndarray, ground_truth: np.ndarray, k: int,
                     where_clause: str = "", n_conns: int = 1, warmup_n: int = 10,
                     expected_index: str | None = None, scan_budget=None,
                     force_index: bool = False, force_seqscan: bool = False,
                     min_throughput_seconds: float = 5.0, pg_pinecone_mode_ctx: dict | None = None,
                     scan_fraction=None, plan_where: str | None = None,
                     halfvec_ground_truth=None) -> dict:
    """Runs the full measured query set at one (sweep_value, iterative_scan,
    scan_budget, scan_fraction, n_conns) point, gated on an EXPLAIN plan check.

    `measured_queries` is the SAME fixed pool at every concurrency
    (exactly one measured set, split n_conns ways,
    never shrunk by subtracting warmup from it). `warmup_pool` is a disjoint
    set of query vectors never counted in recall/QPS; each of the n_conns
    workers gets its own `warmup_n`-sized slice of it.

    `ground_truth` is either `(ids, dists)` from `brute_force_topk` (recall
    is tie-aware; see groundtruth.py's module docstring) or a bare ids array
    (SIFT1M's corpus-provided ids-only ground truth; strict id-set recall).

    `force_seqscan=True` (the P1 forced-seqscan baseline) replaces the usual
    "did the planner use the expected index" gate with the opposite check:
    the plan MUST be a seqscan, or the point is invalid. `expected_index` is
    meaningless here (there's no index under test) and is ignored.

    `pg_pinecone_mode_ctx`, if given (pg_pinecone_native systems only), carries the
    grid-constant inputs of `System.pg_pinecone_native_expected_mode` (`filtered`/
    `realized_selectivity`/`is_flat`); the expected mode itself is computed
    HERE, per point, with this point's own `sweep_value` -- a
    `mode_l_threshold` sweep changes the expected mode across its own grid
    (that sweep's whole purpose is crossing the L/H boundary), so a
    once-per-grid mode string would assert wrongly on exactly the sweep it
    matters most for.
    """
    if system.kind in remote_mod.REMOTE_KINDS:
        return remote_mod.run_sweep_point_remote(
            pool, manager, dsn, table, column, metric, system, sweep_value, iterative_scan,
            measured_queries, warmup_pool, ground_truth, k, where_clause, n_conns, warmup_n,
            expected_index=expected_index, scan_budget=scan_budget, force_index=force_index,
            min_throughput_seconds=min_throughput_seconds, plan_where=plan_where)
    pg_pinecone_expected_mode = None
    if system.kind == "pg_pinecone_native" and pg_pinecone_mode_ctx is not None:
        pg_pinecone_expected_mode = system.pg_pinecone_native_expected_mode(
            **pg_pinecone_mode_ctx, sweep_value=sweep_value)
    if system.kind == "pg_pinecone_native":
        # local/scan.rs's ASSUMED_TOP_K=100 caps the exact-rescore pool at a
        # FIXED size (no real LIMIT propagation into the AM -- see that
        # module's own docstring): a query k above it doesn't error, it
        # silently UNDER-returns rather than misorders, which would show up
        # here as an artificially low "recall" that's actually a harness
        # misconfiguration, not an engine quality number. Fail the point
        # loudly instead (protocol constraint: "query k <= 100").
        assert k <= 100, (
            f"pg_pinecone_native: query k={k} exceeds local/scan.rs's fixed ASSUMED_TOP_K=100 "
            "rescore-pool cap -- lower k, or this benchmark cell measures the cap, not recall"
        )
    gt_ids, gt_dists = (ground_truth if isinstance(ground_truth, tuple)
                        else (ground_truth, None))
    half_ids, half_dists = (halfvec_ground_truth if halfvec_ground_truth is not None
                            else (None, None))
    op = distance_operator(metric)
    plan_shape, index_used, effective_gucs = capture_plan(
        dsn, table, column, op, system, sweep_value, iterative_scan, where_clause,
        measured_queries[0], k, scan_budget=scan_budget, force_index=force_index,
        force_seqscan=force_seqscan, scan_fraction=scan_fraction)
    index_plan = plan_shape.startswith("index_scan:")
    if pg_pinecone_expected_mode is not None and not index_plan:
        # 0.8's AM cost model lets the planner pick a seqscan for a selective cell on
        # a small table: no pg_pinecone scan runs, so last_scan_mode() is stale ('none').
        # The point stays on record as its own plan shape; the forced-index companion
        # series (runner) is where the mode assertion applies.
        pg_pinecone_expected_mode = None
    base = dict(sweep_param=system.sweep_param, sweep_value=sweep_value,
                iterative_scan=iterative_scan, scan_budget=scan_budget,
                scan_fraction=scan_fraction, force_index=force_index,
                force_seqscan=force_seqscan, n_conns=n_conns, k=k, plan_shape=plan_shape,
                index_used=index_used, effective_gucs=effective_gucs)

    if force_seqscan:
        if not plan_shape.startswith("seqscan"):
            # The whole point of this series is an honest exact scan -- if
            # Postgres somehow still used an index (shouldn't happen: no
            # secondary index exists on the table during this series, see
            # build_index's kind=="exact" branch), don't silently report
            # index performance as if it were the seqscan baseline.
            stats = metrics_mod.latency_stats([])
            stats.update(base, valid=False, recall=float("nan"),
                          abort_reason=f"forced-seqscan baseline got plan shape {plan_shape!r}, expected a seqscan")
            return stats
    elif plan_shape.startswith("index_scan:") and expected_index is not None \
            and index_used != expected_index:
        # Cross-system contamination or a stale index: abort this point rather
        # than silently report another index's performance as this system's
        #.
        stats = metrics_mod.latency_stats([])
        stats.update(base, valid=False, recall=float("nan"),
                      abort_reason=f"plan used index {index_used!r}, expected {expected_index!r}")
        return stats

    m_slices = [measured_queries[i::n_conns] for i in range(n_conns)]
    gt_slices = [gt_ids[i::n_conns] for i in range(n_conns)]
    gtd_slices = [gt_dists[i::n_conns] if gt_dists is not None else None
                  for i in range(n_conns)]
    half_slices = [(half_ids[i::n_conns], half_dists[i::n_conns])
                   if half_ids is not None else None for i in range(n_conns)]
    w_slices = [warmup_pool[i * warmup_n:(i + 1) * warmup_n] for i in range(n_conns)]

    recall_barrier = manager.Barrier(n_conns)
    throughput_barrier = manager.Barrier(n_conns)
    futures = [pool.submit(_pool_worker_run, table, column, op, system, sweep_value,
                            iterative_scan, scan_budget, force_index, force_seqscan, w_slices[i],
                            m_slices[i], k, where_clause, recall_barrier, throughput_barrier,
                            min_throughput_seconds, pg_pinecone_expected_mode, scan_fraction)
               for i in range(n_conns)]
    per_proc = [f.result() for f in futures]

    latencies = []
    recalls = []
    half_recalls = []
    rows_returned = []
    scan_stats_all = []
    for proc_idx, (results, _n_completed, _ts, _te, _rss) in enumerate(per_proc):
        dist_rows = gtd_slices[proc_idx]
        for qi, ((ids, elapsed, scan_stats), truth_row) in enumerate(
                zip(results, gt_slices[proc_idx])):
            latencies.append(elapsed)
            rows_returned.append(len(ids))
            if scan_stats is not None:
                scan_stats_all.append(scan_stats)
            recalls.append(recall_at_k(ids, truth_row, k,
                                        truth_dists_row=None if dist_rows is None
                                        else dist_rows[qi]))
            if half_slices[proc_idx] is not None:
                h_ids, h_dists = half_slices[proc_idx]
                half_recalls.append(recall_at_k(ids, h_ids[qi], k,
                                                truth_dists_row=h_dists[qi]))

    throughput_completed = sum(p[1] for p in per_proc)
    throughput_starts = [p[2] for p in per_proc]
    throughput_ends = [p[3] for p in per_proc]
    throughput_wall = max(throughput_ends) - min(throughput_starts) if per_proc else float("nan")

    stats = metrics_mod.latency_stats(latencies)
    stats["qps_wall"] = (throughput_completed / throughput_wall
                          if throughput_wall and throughput_wall > 0 else float("nan"))
    stats["throughput_completed_queries"] = throughput_completed
    stats["throughput_wall_seconds"] = throughput_wall
    stats["recall"] = float(np.nanmean(recalls)) if recalls else float("nan")
    if half_ids is not None:
        stats["recall_halfvec_truth"] = (float(np.nanmean(half_recalls))
                                         if half_recalls else float("nan"))
    # Rows returned at LIMIT k (a residual filter can leave a scan under-filled)
    # and, when `record_scan_stats`, the engine's own per-scan counters.
    stats["rows_returned_mean"] = float(np.mean(rows_returned)) if rows_returned else float("nan")
    stats["rows_returned_min"] = int(min(rows_returned)) if rows_returned else 0
    stats["queries_done"] = len(latencies)
    stats["queries_planned"] = len(measured_queries)
    stats["frac_filled"] = (float(np.mean([r >= k for r in rows_returned]))
                            if rows_returned else float("nan"))
    if scan_stats_all and index_plan:
        cols = [np.array(c, dtype=float) for c in zip(*scan_stats_all)]
        batches, rescored, capped = cols[:3]
        stats["scan_batches_mean"] = float(batches.mean())
        stats["scan_rescored_mean"] = float(rescored.mean())
        stats["scan_rescored_p95"] = float(np.percentile(rescored, 95))
        stats["scan_capped_frac"] = float(capped.mean())
        if len(cols) >= 5:  # deepen-before-widen build: slices beyond the first of a probed set, lists read
            stats["scan_deepened_mean"] = float(cols[3].mean())
            stats["scan_lists_probed_mean"] = float(cols[4].mean())
    # Query-time peak backend RSS, recall pass only,
    # max across this point's workers (one backend each). Directional --
    # same `ps`-based sampler as build RSS, and a 200ms tick can miss a
    # short-lived spike inside a single fast query.
    stats["query_peak_backend_rss_bytes"] = max((p[4] for p in per_proc), default=0)
    if system.explain_probe and index_plan and n_conns == 1:
        probe_q = measured_queries[:: max(1, len(measured_queries) // 8)][:8]
        stats.update(explain_cost_probe(dsn, table, column, op, system, sweep_value, iterative_scan,
                                        where_clause, probe_q, k, scan_budget=scan_budget,
                                        force_index=force_index, scan_fraction=scan_fraction))
    stats.update(base, valid=True, abort_reason=None)
    return stats


def run_sweep(dsn, table, column, metric, system: System, measured_queries, warmup_pool,
              ground_truth, k, where_clause="", n_conns_list=(1,), warmup_n=10,
              iterative_scan_values=None, expected_index=None, n_rows: int | None = None,
              force_index: bool = False, force_seqscan: bool = False,
              min_throughput_seconds: float = 5.0, pg_pinecone_mode_ctx: dict | None = None,
              plan_where: str | None = None, halfvec_ground_truth=None) -> list[dict]:
    """Runs `system`'s full (iterative_scan x scan_budget x scan_fraction x
    sweep_value x n_conns) grid.

    Owns one process pool (sized to the largest n_conns this sweep needs) for
    its whole duration -- workers keep a persistent connection
    (`_pool_worker_init`) across every point instead of reconnecting per
    point. `n_rows` resolves any "auto" scan-budget value to the table-size
    scaled formula (see `System.scan_budgets_for`). `force_index=True` runs
    every point with `enable_seqscan = off` -- the forced-plan companion
    series. `force_seqscan=True` runs every
    point with `enable_seqscan = on, enable_indexscan = off, enable_bitmapscan
    = off` -- the forced-seqscan exact-scan baseline series.

    `pg_pinecone_mode_ctx` (pg_pinecone_native systems only) carries the grid-constant
    inputs of the per-measured-query scan-mode assertion (`filtered`/
    `realized_selectivity`/`is_flat` -- fixed for any one `run_sweep` call,
    since the filter/build is fixed for its duration); the expected mode
    itself is computed per POINT in `run_sweep_point`, because a
    `mode_l_threshold` sweep legitimately changes the expected mode across
    its own grid.
    """
    max_conns = max(n_conns_list)
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    out = []
    try:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_conns, mp_context=ctx,
                                  initializer=_pool_worker_init, initargs=(dsn,)) as pool:
            for iterative_scan in (iterative_scan_values or system.iterative_scan_values):
                for scan_budget in system.scan_budgets_for(iterative_scan, n_rows):
                    for scan_fraction in system.scan_fractions_for():
                        for sweep_value in system.sweep_values:
                            for n_conns in n_conns_list:
                                out.append(run_sweep_point(
                                    pool, manager, dsn, table, column, metric, system,
                                    sweep_value, iterative_scan, measured_queries, warmup_pool,
                                    ground_truth, k, where_clause, n_conns, warmup_n,
                                    expected_index=expected_index, scan_budget=scan_budget,
                                    force_index=force_index, force_seqscan=force_seqscan,
                                    min_throughput_seconds=min_throughput_seconds,
                                    pg_pinecone_mode_ctx=pg_pinecone_mode_ctx,
                                    scan_fraction=scan_fraction, plan_where=plan_where,
                                    halfvec_ground_truth=halfvec_ground_truth))
    finally:
        manager.shutdown()
    metrics_mod.label_budget_capped(out)
    return out
