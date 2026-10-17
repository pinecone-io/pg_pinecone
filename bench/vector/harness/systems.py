"""System/index definitions under test: pgvector hnsw, pgvector ivfflat,
the exact-scan baseline, pg_pinecone_native (the native local engine's
`USING pinecone ... WITH (mode = 'local')` index) and the remote engine.

Each system exposes: DDL to build its index, the GUC(s) to set per sweep
point on the *querying* connection, and the ORDER BY operator to use for the
configured distance metric.

## pg_pinecone_native -- install first (read before running)

`pg_pinecone_native`'s DDL (`build_sql` below) assumes the `pg_pinecone` extension
(control file/`.sql`/shared library) is ALREADY installed into the Postgres
install `harness/pgctl.py`'s `PgInstance` launches its throwaway `.pgdata`
from (`PG_PINECONE_BENCH_PG` = the install prefix, or its bin/ dir as `PGRX_PG_BIN`;
default `~/.pgrx/17.10/pgrx-install`) -- this harness never builds or installs
the engine itself. Use a PRIVATE pgrx home so other worktrees'
`cargo pgrx install`/`test` cannot overwrite the build under test (exact
commands: README.md, "Install pg_pinecone"):

    cp -cR ~/.pgrx "$PGRX_HOME"       # then repoint $PGRX_HOME/config.toml's pg17 path
    cd <repo>/pg_pinecone && cargo pgrx install --release \
        --pg-config "$PGRX_HOME/17.10/pgrx-install/bin/pg_config"
    export PG_PINECONE_BENCH_PG="$PGRX_HOME/17.10/pgrx-install"

`PgInstance` always `initdb`s a brand-new data directory per run, so a bench
run never needs a restart dance. Re-run the install after every engine-side
code change before trusting a `pg_pinecone_native` result.

## pg_pinecone_native -- GUC mapping (extension 0.8.0+)

Sweep axes: `sweep_param` (rescore_multiplier | scan_fraction |
mode_l_threshold), `iterative_scan_values` -> `pinecone.iterative_scan`
(off | relaxed_order | strict_order, and `auto` from 0.10, the shipped default;
needs an iterative-scan build; `null` leaves the GUC alone, so a worker's persistent connection keeps
the previous point's value: name `auto` to measure the default),
`filtered_scan_budget_values`
-> `pinecone.max_scan_tuples`, `scan_fraction_values` ->
`pinecone.scan_fraction` as a further grid axis (0 = no override, the auto
ramp). `record_scan_stats: true` additionally reads
`pinecone.last_scan_stats()` after every recall-pass query.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

_OPCLASS = {
    "l2": "vector_l2_ops",
    "cosine": "vector_cosine_ops",
    "ip": "vector_ip_ops",
}
_HALFVEC_OPCLASS = {metric: opclass.replace("vector_", "halfvec_")
                    for metric, opclass in _OPCLASS.items()}
_OPERATOR = {
    "l2": "<->",
    "cosine": "<=>",
    "ip": "<#>",
}


def distance_operator(metric: str) -> str:
    return _OPERATOR[metric]


def ivfflat_lists_heuristic(n: int) -> int:
    """Per pgvector docs: rows/1000 for up to 1M rows, sqrt(rows) beyond that."""
    if n <= 1_000_000:
        return max(1, round(n / 1000))
    return max(1, round(math.sqrt(n)))


def _norm_iscan(v):
    return "off" if v is False else v


@dataclass
class System:
    name: str
    kind: str  # "hnsw" | "ivfflat" | "exact" | "pg_pinecone_native" | "pg_pinecone_remote" | "pg_pinecone_remote_topk"
    build_params: dict
    sweep_param: str
    sweep_values: list
    # Used for unfiltered sweeps. None = don't touch the GUC (session/server default).
    iterative_scan_values: list = field(default_factory=lambda: [None])
    # Used for filtered sweeps -- this is where iterative_scan actually matters
    # (a thin filter can starve a graph/IVF scan of candidates). Defaults to
    # iterative_scan_values if a system config doesn't override it.
    filtered_iterative_scan_values: list | None = None
    # GUC name for this system's iterative-scan visitation budget (e.g.
    # "hnsw.max_scan_tuples"), or None if this system kind has no such knob
    # worth sweeping. See `scan_budgets_for`.
    scan_budget_param: str | None = None
    # Values to sweep for `scan_budget_param` on filtered relaxed/strict-order
    # scans: ints, or the literal string "auto" to resolve at run time to
    # max(100_000, round(0.1 * n_rows)) -- scaled to table size rather than
    # pinned to one dataset's scale.
    filtered_scan_budget_values: list | None = None
    # Skip this system's unfiltered sweep entirely (config key
    # `skip_unfiltered`). For runs whose exact-scan system exists only to
    # measure per-filter denominators: the unfiltered exact scan is
    # corpus-geometry-independent (~8.9s p50 at 1M x 768d, measured
    # identically on two corpora) and re-measuring it costs ~2.5h of a
    # record run for a number that's already on record.
    skip_unfiltered: bool = False
    # pg_pinecone_native only: extra grid axis over `pinecone.scan_fraction`
    # (0 = no override = the engine's own auto ramp), alongside `sweep_param`.
    # None = not an axis (no stamping, no extra points).
    scan_fraction_values: list | None = None
    # pg_pinecone_native only: read `pinecone.last_scan_stats()` after every
    # recall-pass query and aggregate batches/rescored/capped per point.
    # Needs an extension build that has the function.
    record_scan_stats: bool = False
    # After each index-plan point at 1 client, run EXPLAIN (ANALYZE, BUFFERS) on a few
    # queries (outside every timer) and record tuples pulled through the index scan
    # node + buffer counts (`workloads.explain_cost_probe`): the per-query heap-fetch cost.
    explain_probe: bool = False
    concurrency: list | None = None
    k_values: list | None = None
    sweep_values_by_k: dict | None = None
    max_queries: int | None = None
    # Seconds one worker may spend on the recall pass of one sweep point; its warm-up gets a fifth of it. Queries not
    # reached are dropped, the point stamps queries_done / queries_planned. For a system that streams through a very
    # selective filter and takes seconds per query. None = run every query.
    point_budget_seconds: float | None = None
    # pgvector comparison arms: FP32 heap with a halfvec expression index,
    # or a halfvec heap with a halfvec index. Reranking requires the FP32 heap.
    storage_type: str = "vector"
    index_type: str = "vector"
    rerank_multiplier: int | None = None
    dimension: int | None = None  # filled from the dataset before index build

    def __post_init__(self):
        if self.storage_type not in ("vector", "halfvec"):
            raise ValueError(f"{self.name}: unknown storage_type {self.storage_type!r}")
        if self.index_type not in ("vector", "halfvec"):
            raise ValueError(f"{self.name}: unknown index_type {self.index_type!r}")
        if (self.storage_type == "halfvec" or self.index_type == "halfvec") and self.kind not in ("hnsw", "exact"):
            raise ValueError(f"{self.name}: halfvec is supported for pgvector hnsw or exact only")
        if self.storage_type == "halfvec" and self.index_type != "halfvec":
            raise ValueError(f"{self.name}: a halfvec heap needs a halfvec index")
        if self.rerank_multiplier is not None:
            if (self.kind != "hnsw" or self.storage_type != "vector"
                    or self.index_type != "halfvec"
                    or type(self.rerank_multiplier) is not int or self.rerank_multiplier < 1):
                raise ValueError(f"{self.name}: FP32 reranking requires vector heap, halfvec hnsw, multiplier >= 1")
        # YAML 1.1 parses an unquoted `off` as False; normalise so `off` stays
        # the string every GUC (and `scan_budgets_for`) expects.
        self.iterative_scan_values = [_norm_iscan(v) for v in self.iterative_scan_values]
        if self.filtered_iterative_scan_values is None:
            self.filtered_iterative_scan_values = self.iterative_scan_values
        self.filtered_iterative_scan_values = [_norm_iscan(v)
                                               for v in self.filtered_iterative_scan_values]
        if self.scan_fraction_values is not None:
            assert self.kind == "pg_pinecone_native", "scan_fraction_values is pg_pinecone_native only"
            assert self.sweep_param != "scan_fraction", (
                "scan_fraction is already the sweep_param; drop scan_fraction_values")
        if self.kind == "pg_pinecone_native" and self.build_params.get("target_recall") is not None:
            lo, hi = _PG_PINECONE_NATIVE_TARGET_RECALL
            tr = float(self.build_params["target_recall"])
            assert lo <= tr <= hi, f"{self.name}: target_recall out of range [{lo}, {hi}]: {tr}"

    def scan_fractions_for(self) -> list:
        return list(self.scan_fraction_values) if self.scan_fraction_values else [None]

    def scan_budgets_for(self, iterative_scan, n_rows: int | None) -> list:
        """Values to sweep for the iterative-scan visitation-budget GUC at
        this `iterative_scan` setting.

        Returns `[None]` (GUC left at its session default, not swept) unless
        this system declares `scan_budget_param` AND iterative scanning is
        actually engaged -- the budget is meaningless when iterative_scan is
        off/unset. This is the fix for the "0.855 plateau" methodology bug:
        hnsw.max_scan_tuples defaults to 20000, which can be comparable to or
        smaller than the table's row count and silently cap what a filtered
        relaxed_order scan can ever find, independent of how high ef_search
        is swept. Treating the budget as its own swept-and-stamped axis
        (default vs. a raised, table-size-scaled budget) tells "the knob
        plateaued" apart from "the budget capped the knob."
        """
        if self.scan_budget_param is None or iterative_scan in (None, "off"):
            return [None]
        values = self.filtered_scan_budget_values or [None]
        out = []
        for v in values:
            if v == "auto":
                out.append(max(100_000, round(0.1 * n_rows)) if n_rows else None)
            else:
                out.append(v)
        return out

    def index_name(self, table: str) -> str | None:
        # kind="exact" (the forced-seqscan baseline) builds no
        # index -- there's nothing to name, and callers use `None` here to
        # skip the "did the planner use MY index" contamination check that
        # doesn't apply to a deliberately index-free series (see
        # workloads.run_sweep_point's force_seqscan branch instead).
        if self.kind in ("exact", "pg_pinecone_remote_topk"):
            return None
        return f"{table}_{self.name}_idx"

    def pre_build_sql(self, table: str, column: str, metric: str, n_rows: int) -> list[str]:
        """Statements to run (on the build connection) before `build_sql`.
        Only `pg_pinecone_native` needs one so far: the extension itself, since
        this harness never installs pg_pinecone -- see the module docstring's
        shared-install note."""
        if self.kind == "pg_pinecone_native":
            return ["CREATE EXTENSION IF NOT EXISTS pg_pinecone"]
        return []

    def build_sql(self, table: str, column: str, metric: str, n_rows: int) -> str:
        opclass = _OPCLASS[metric]
        idx = self.index_name(table)
        if self.kind in ("pg_pinecone_remote", "pg_pinecone_remote_topk"):
            from . import remote
            return remote.build_sql(self, table, column, metric)
        if self.kind == "hnsw":
            m = self.build_params.get("m", 16)
            ef_c = self.build_params.get("ef_construction", 64)
            if self.index_type == "halfvec":
                opclass = _HALFVEC_OPCLASS[metric]
                if self.storage_type == "vector" and self.dimension is None:
                    raise ValueError(f"{self.name}: dimension required for halfvec expression index")
                key = (f"({column}::halfvec({self.dimension}))"
                       if self.storage_type == "vector" else column)
            else:
                key = column
            return (f"CREATE INDEX {idx} ON {table} USING hnsw ({key} {opclass}) "
                    f"WITH (m = {m}, ef_construction = {ef_c})")
        if self.kind == "ivfflat":
            lists = self.build_params.get("lists") or ivfflat_lists_heuristic(n_rows)
            return (f"CREATE INDEX {idx} ON {table} USING ivfflat ({column} {opclass}) "
                    f"WITH (lists = {lists})")
        if self.kind == "pg_pinecone_native":
            # am.rs's ambuild: one vector column, then AT MOST one filter
            # column -- `build_params["filter_column"]` (if set) must be the
            # int2 EQUALITY-mirror column (`loader.filter_eq_column_name`),
            # never the boolean one pgvector systems query (bool has no
            # pg_pinecone opclass and is rejected at DDL time). `None`/absent means
            # an unfiltered build (the unfiltered/SIFT/synthetic-floor
            # criteria).
            bp = self.build_params
            filter_col = bp.get("filter_column")
            key_cols = f"{column} {opclass}" + (f", {filter_col}" if filter_col else "")
            opts = ["mode = 'local'"]
            for name, cast in _PG_PINECONE_NATIVE_RELOPTIONS.items():
                if bp.get(name) is not None:
                    opts.append(f"{name} = {cast(bp[name])}")
            return (f"CREATE INDEX {idx} ON {table} USING pinecone ({key_cols}) "
                    f"WITH ({', '.join(opts)})")
        raise ValueError(f"unknown system kind {self.kind!r}")

    def query_sql(self, table: str, column: str, op: str, where_clause: str,
                  q_vec, k: int) -> tuple[str, tuple]:
        """One query definition shared by timing, EXPLAIN and cost probes."""
        dim = len(q_vec)
        if self.index_type == "halfvec" or self.storage_type == "halfvec":
            key = (f"({column}::halfvec({dim}))" if self.storage_type == "vector"
                   else column)
            q_expr = f"%s::halfvec({dim})"
        else:
            key, q_expr = column, "%s"
        if self.rerank_multiplier is not None:
            candidates = max(k, k * self.rerank_multiplier)
            sql = (f"WITH candidates AS MATERIALIZED (SELECT id, {column} FROM {table} "
                   f"{where_clause} ORDER BY {key} {op} {q_expr} LIMIT %s) "
                   f"SELECT id FROM candidates ORDER BY {column} {op} %s LIMIT %s")
            return sql, (q_vec, candidates, q_vec, k)
        return (f"SELECT id FROM {table} {where_clause} ORDER BY {key} {op} {q_expr} LIMIT %s",
                (q_vec, k))

    def drop_sql(self, table: str) -> str:
        return f"DROP INDEX IF EXISTS {self.index_name(table)}"

    def set_session_params(self, conn, sweep_value, iterative_scan=None, scan_budget=None,
                           scan_fraction=None):
        if self.kind == "hnsw":
            conn.execute(f"SET hnsw.ef_search = {int(sweep_value)}")
            if iterative_scan is not None:
                conn.execute(f"SET hnsw.iterative_scan = {iterative_scan}")
            if scan_budget is not None:
                conn.execute(f"SET hnsw.max_scan_tuples = {int(scan_budget)}")
        elif self.kind == "ivfflat":
            conn.execute(f"SET ivfflat.probes = {int(sweep_value)}")
            if iterative_scan is not None:
                conn.execute(f"SET ivfflat.iterative_scan = {iterative_scan}")
            if scan_budget is not None:
                conn.execute(f"SET ivfflat.max_probes = {int(scan_budget)}")
        elif self.kind == "exact":
            # No knob -- the forced-seqscan GUCs (enable_seqscan/enable_indexscan/
            # enable_bitmapscan) are set by workloads.py's force_seqscan lever,
            # not here; sweep_value is just a placeholder to reuse run_sweep's
            # grid machinery for a system with nothing to sweep.
            pass
        elif self.kind == "pg_pinecone_native":
            self._set_pg_pinecone_native_session_params(conn, sweep_value, iterative_scan,
                                                     scan_budget, scan_fraction)
        elif self.kind == "pg_pinecone_remote":
            assert self.sweep_param in ("am_min_topk", "none"), (
                f"pg_pinecone_remote sweep_param must be am_min_topk or none, got {self.sweep_param!r}")
            if self.sweep_param == "am_min_topk":
                conn.execute(f"SET pinecone.am_min_topk = {int(sweep_value)}")
            if self.build_params.get("am_max_topk") is not None:
                conn.execute(f"SET pinecone.am_max_topk = {int(self.build_params['am_max_topk'])}")
            if iterative_scan is not None:
                assert iterative_scan in ("off", "relaxed_order", "strict_order"), iterative_scan
                conn.execute(f"SET pinecone.iterative_scan = {iterative_scan}")
            if scan_budget is not None:
                conn.execute(f"SET pinecone.max_scan_tuples = {int(scan_budget)}")
        elif self.kind == "pg_pinecone_remote_topk":
            pass
        else:
            raise ValueError(self.kind)

    def _set_pg_pinecone_native_session_params(self, conn, sweep_value, iterative_scan=None,
                                            scan_budget=None, scan_fraction=None):
        """B3's whole reason for `pinecone.rescore_multiplier`/`pinecone.scan_fraction`/
        `pinecone.mode_l_threshold` existing as session GUCs (`local/scan.rs`'s
        B2 scan-time overrides): a sweep across `sweep_values` never
        rebuilds this system's index -- `sweep_param` must name one of
        those three, and every OTHER one of the three, if fixed for this
        System instance via `build_params`, is re-applied every session too
        (a `mode_l_threshold` sweep still needs a FIXED `rescore_multiplier`
        applied, and vice versa). Bounds are asserted here, not left to
        Postgres's own GUC-range error, so a bad config fails at the harness
        level with the sweep point it broke, not a bare SQL error deep in a
        worker process.
        """
        assert self.sweep_param in _PG_PINECONE_NATIVE_SESSION_GUCS, (
            f"pg_pinecone_native sweep_param must be one of {sorted(_PG_PINECONE_NATIVE_SESSION_GUCS)} "
            f"(a scan-time GUC -- no rebuild within a sweep), got {self.sweep_param!r}"
        )
        _assert_and_set_pg_pinecone_guc(conn, self.sweep_param, sweep_value)
        bp = self.build_params
        for guc in _PG_PINECONE_NATIVE_SESSION_GUCS:
            if guc != self.sweep_param and guc in bp:
                _assert_and_set_pg_pinecone_guc(conn, guc, bp[guc])
        forced = bp.get("force_scan_mode")
        if forced is not None:
            assert forced in ("auto", "l", "h"), f"pinecone.force_scan_mode must be auto/l/h, got {forced!r}"
            conn.execute(f"SET pinecone.force_scan_mode = '{forced}'")
        # Iterative-scan axes (`pinecone.iterative_scan` enum,
        # `pinecone.max_scan_tuples` int). Left untouched when None so a build
        # without them keeps working for every non-residual config.
        if scan_fraction is not None:
            _assert_and_set_pg_pinecone_guc(conn, "scan_fraction", scan_fraction)
        if iterative_scan is not None:
            assert iterative_scan in ("auto", "off", "relaxed_order", "strict_order"), (
                f"pinecone.iterative_scan must be auto/off/relaxed_order/strict_order, got "
                f"{iterative_scan!r}")
            conn.execute(f"SET pinecone.iterative_scan = {iterative_scan}")
        if scan_budget is not None:
            conn.execute(f"SET pinecone.max_scan_tuples = {int(scan_budget)}")

    def pg_pinecone_native_expected_mode(self, *, filtered: bool, realized_selectivity: float | None = None,
                                   is_flat: bool = False, sweep_value=None) -> str:
        """The scan mode `pinecone.last_scan_mode()` must report for a query
        against this system's build, given whether the query carries THIS
        system's own filter equality qual (`filtered`) -- mirrors
        `local::scan`'s mode dispatch so a broken estimator, a
        misconfigured `mode_l_threshold`, or a harness/engine drift shows up
        as an assertion failure (`workloads._pool_worker_run`'s
        per-measured-query check) instead of a silently-blended result.

        `is_flat` (small-N exact bootstrap; `am.rs`/`local/scan.rs`'s "MODE_FLAT")
        beats every other rule. An unfiltered query against a NON-flat
        index -- filtered-built or not, since a query without this index's
        filter equality qual supplies no scankey at all -- is always mode H.
        A filtered query's mode follows `pinecone.force_scan_mode` if forced,
        else the same estimate-vs-threshold rule `choose_filtered_mode`
        applies, using the harness's own already-measured
        `realized_selectivity` (not the target one) as the estimate's stand-in.

        `sweep_value` is the CURRENT sweep point's value of `sweep_param`.
        It matters exactly when `sweep_param == "mode_l_threshold"`: that
        sweep's whole purpose is crossing the L/H boundary, so the effective
        threshold -- and therefore the expected mode -- varies per point and
        must come from the swept value, not once-per-grid from
        `build_params` (`workloads.run_sweep_point` computes this per
        point). The other two sweepable GUCs (`rescore_multiplier`/
        `scan_fraction`) never affect mode selection.
        """
        assert self.kind == "pg_pinecone_native"
        if is_flat:
            return "flat"
        if not filtered:
            return "h"
        forced = self.build_params.get("force_scan_mode")
        if forced in ("l", "h"):
            return forced
        if self.sweep_param == "mode_l_threshold" and sweep_value is not None:
            threshold = float(sweep_value)
        else:
            threshold = self.build_params.get("mode_l_threshold", 0.03)
        assert realized_selectivity is not None, (
            "pg_pinecone_native_expected_mode(filtered=True) needs realized_selectivity "
            "to predict L vs H when force_scan_mode isn't pinned to 'l'/'h'"
        )
        return "l" if realized_selectivity <= threshold else "h"


# Build-time reloptions (`local/options.rs`'s `PgPineconeOptionsRaw`) `build_sql`
# passes through from `build_params` when present, name -> type; `mode`/
# `filter_column` are handled separately above. `target_recall` (0.10+) is a
# real reloption, bounded like `local/options.rs`'s MIN/MAX_TARGET_RECALL.
_PG_PINECONE_NATIVE_RELOPTIONS = {"seed": int, "k_bits": int, "rescore_multiplier": int,
                               "small_n_threshold": int, "n_lists": int, "target_recall": float}
_PG_PINECONE_NATIVE_TARGET_RECALL = (0.5, 0.999)

# Scan-time GUCs (`local/scan.rs`'s B2 overrides) -- name -> (lo, hi) for the
# assertion in `_assert_and_set_pg_pinecone_guc`, matching each GUC's own
# `GucRegistry::define_*` bounds exactly (`pinecone.rescore_multiplier`:
# 0..=1000 int; the other two: 0.0..=1.0 float).
_PG_PINECONE_NATIVE_SESSION_GUCS = {
    "rescore_multiplier": (0, 1000, int),
    "scan_fraction": (0.0, 1.0, float),
    "mode_l_threshold": (0.0, 1.0, float),
}


def _assert_and_set_pg_pinecone_guc(conn, name: str, value) -> None:
    lo, hi, cast = _PG_PINECONE_NATIVE_SESSION_GUCS[name]
    value = cast(value)
    assert lo <= value <= hi, f"pinecone.{name} out of range [{lo}, {hi}]: {value}"
    conn.execute(f"SET pinecone.{name} = {value}")


def default_systems(cfg_systems: list[dict]) -> list[System]:
    return [System(name=s["name"], kind=s["type"], build_params=s.get("build_params", {}),
                    sweep_param=s["sweep_param"], sweep_values=s["sweep_values"],
                    iterative_scan_values=s.get("iterative_scan_values", [None]),
                    filtered_iterative_scan_values=s.get("filtered_iterative_scan_values"),
                    scan_budget_param=s.get("scan_budget_param"),
                    filtered_scan_budget_values=s.get("filtered_scan_budget_values"),
                    skip_unfiltered=s.get("skip_unfiltered", False),
                    scan_fraction_values=s.get("scan_fraction_values"),
                    record_scan_stats=s.get("record_scan_stats", False),
                    explain_probe=s.get("explain_probe", False),
                    concurrency=s.get("concurrency"), k_values=s.get("k_values"),
                    max_queries=s.get("max_queries"),
                    point_budget_seconds=s.get("point_budget_seconds"),
                    storage_type=s.get("storage_type", "vector"),
                    index_type=s.get("index_type", "vector"),
                    rerank_multiplier=s.get("rerank_multiplier"),
                    sweep_values_by_k={int(k): v for k, v in (s.get("sweep_values_by_k") or {}).items()} or None)
            for s in cfg_systems]
