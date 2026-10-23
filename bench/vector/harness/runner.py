"""Orchestrates one full benchmark run from a config dict to a results JSON."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

from . import churn as churn_mod
from . import datasets as ds_mod
from . import groundtruth as gt_mod
from . import loader as loader_mod
from . import metrics as metrics_mod
from . import remote as remote_mod
from . import sift as sift_mod
from .pgctl import PgInstance
from .systems import default_systems
from .util import bench_content_hash, bytes_hash, git_dirty, git_sha, sanitize_for_json, stable_hash, timed

log = logging.getLogger(__name__)
RESULTS_DIR = Path(os.environ.get("BENCH_RESULTS_DIR") or Path(__file__).resolve().parent.parent / "results")
BENCH_DIR = Path(__file__).resolve().parent.parent
TABLE = "items"
COLUMN = "embedding"


def _write_checkpoint(run_name: str, started_tag: str, payload: dict) -> None:
    """Crash-safe partial-results checkpoint, written after every completed
    workload chunk (per-system unfiltered sweep, each filter column, churn,
    SIFT). An 18-hour run must never hold every measured point hostage to the
    single results write at the very end -- a crash/OOM/power event at hour 17
    should cost the in-flight chunk, not the whole run's data.

    The file is `results/<name>_<started>.partial.json` (gitignored like all
    raw results), atomically replaced via tmp+rename so a crash mid-checkpoint
    leaves the previous checkpoint intact, and marked `"partial": true` so it
    can never be mistaken for a run of record -- it deliberately carries no
    provenance stamps (those are only meaningful at successful completion).
    `save_results` removes the run's partials once the final write succeeds.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{run_name}_{started_tag}.partial.json"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(sanitize_for_json(dict(payload, partial=True)), f, indent=2,
                  default=str, allow_nan=False)
    tmp.replace(path)


def _machine_info() -> dict:
    info = dict(platform=platform.platform(), processor=platform.processor())
    if platform.system() == "Darwin":
        try:
            info["cpu_count"] = int(subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                                                    capture_output=True, text=True).stdout.strip())
            mem_bytes = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                            capture_output=True, text=True).stdout.strip())
            info["memory_gb"] = round(mem_bytes / (1024**3), 1)
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        info["memory_gb"] = round(int(line.split()[1]) / (1024**2), 1)
                        break
            import os as _os
            info["cpu_count"] = _os.cpu_count()
        except Exception:
            pass
        # GCE identity, if running on one (results
        # must stamp the box, not just "Linux"). 0.5s timeout: a non-GCE
        # Linux host loses half a second once per run, nothing else.
        try:
            import urllib.request
            req = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/instance/machine-type",
                headers={"Metadata-Flavor": "Google"})
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                info["gce_machine_type"] = resp.read().decode().rsplit("/", 1)[-1]
        except Exception:
            pass
    return info


def _load_dataset(cfg: dict):
    d = cfg["dataset"]
    n_warmup_pool = max(cfg["concurrency"]) * cfg["warmup_queries"]
    if d["kind"] == "synthetic":
        return ds_mod.synthetic_dataset(
            n=d["n"], dim=d["dim"], n_centers=d["n_centers"], cluster_sigma=d["cluster_sigma"],
            n_queries=d.get("n_queries", 500), n_warmup_pool=n_warmup_pool, seed=cfg["seed"],
            cluster_scale_mix=d.get("cluster_scale_mix"), background_frac=d.get("background_frac", 0.0))
    if d["kind"] == "real":
        # Converted real-embedding artifact + cached k-means clustering (P1
        # task #3) -- see datasets.real_dataset.
        return ds_mod.real_dataset(
            artifact=d["artifact"], n=d["n"], dim=d["dim"], n_centers=d["n_centers"],
            n_queries=d.get("n_queries", 500), n_warmup_pool=n_warmup_pool, seed=cfg["seed"],
            kmeans_sample=d.get("kmeans_sample", 100_000), kmeans_iters=d.get("kmeans_iters", 15))
    raise ValueError(f"unknown dataset kind {d['kind']!r}")


def _latency_at_recall_grouped(sweep: list[dict], n_conns_list, iterative_scan_values) -> dict:
    """Interpolates latency-at-recall separately per (iterative_scan,
    scan_budget, plan_shape, n_conns) group.

    Mixing iterative_scan=off and iterative_scan=relaxed_order points into one
    recall-latency curve would blur which *configuration* achieves a given
    recall (they're separate configs, not two settings of the same knob) --
    see systems.py's iterative_scan_values docstring. Mixing plan shapes
    (index scan vs a seqscan fallback) would be worse: those aren't even
    measuring the same execution strategy. The
    scan_budget axis is the same kind of hazard: the default 20000-tuple hnsw.max_scan_tuples budget and a raised,
    table-size-scaled budget are two different configurations of the same
    iterative_scan setting, not one knob -- blending them would hide exactly
    the "budget capped, not the knob" signal this fix exists to surface.
    """
    out = {}
    plan_shapes = sorted({p["plan_shape"] for p in sweep})
    for iscan in iterative_scan_values:
        label = "default" if iscan is None else str(iscan)
        out[label] = {}
        # (scan_budget, scan_fraction) are both configuration axes, never one
        # curve: scan_fraction is only stamped for systems that sweep it.
        groups = sorted({(p.get("scan_budget"), p.get("scan_fraction"))
                         for p in sweep if p["iterative_scan"] == iscan},
                        key=lambda g: (g[0] is None, g[0] or 0, g[1] is None, g[1] or 0))
        for budget, sfrac in groups:
            blabel = ("default" if budget is None else str(budget)) \
                + ("" if sfrac is None else f" sf={sfrac}")
            out[label][blabel] = {}
            for shape in plan_shapes:
                out[label][blabel][shape] = {
                    str(nc): metrics_mod.interpolate_latency_at_recall(
                        [p for p in sweep if p["n_conns"] == nc and p["iterative_scan"] == iscan
                         and p.get("scan_budget") == budget and p.get("scan_fraction") == sfrac
                         and p["plan_shape"] == shape])
                    for nc in n_conns_list
                }
    return out


def _all_seqscan(sweep: list[dict]) -> bool:
    """True if every plan-valid point in `sweep` fell back to a seqscan -- the
    planner never chose the index at all for this filter. Distinguishes "the index can't do this" from "the planner
    gives up", which need a forced-plan (`enable_seqscan = off`) companion
    series to tell apart.
    """
    valid = [p for p in sweep if p.get("valid", True)]
    return bool(valid) and all(p["plan_shape"] == "seqscan_sort" for p in valid)


def _assert_exact_recall(sweep: list[dict], label: str, tol: float = 1e-9,
                         recall_key: str = "recall") -> None:
    """The forced-seqscan baseline visits every qualifying row, so
    recall@k is 1.0 BY CONSTRUCTION -- assert it rather than just report it.

    Recall is the tie-aware definition (groundtruth.recall_at_k): real corpora
    contain exact duplicate vectors (434 in the wiki_dpr 1M set) and
    sub-fp32-noise near-ties at the k boundary, where numpy ground truth and
    Postgres legitimately order two (near-)equidistant neighbors differently
    (observed: fp64 relative boundary gaps of 3.5e-6/7.6e-6 flipped one
    neighbor and scored 0.9998). Under the tie-aware definition an exact scan
    must still be PERFECT -- this stays a hard gate; a violation now means a
    real methodology bug (wrong metric, wrong mask, stale ground truth), not
    tie noise.
    """
    for p in sweep:
        if not p.get("valid", True):
            continue
        recall = p.get(recall_key)
        if recall is None or (isinstance(recall, float) and np.isnan(recall)):
            continue
        if abs(recall - 1.0) > tol:
            raise RuntimeError(
                f"forced-seqscan baseline ({label}) got {recall_key}={recall!r} != 1.0 at "
                f"sweep point {p!r} -- exact scan should match cached ground truth exactly."
            )


def _hardness_check(system, unfiltered_sweep: list[dict], n_conns_list, iterative_scan_values,
                     threshold: float = 0.97):
    """If the cheapest swept setting already achieves
    near-perfect recall, the dataset isn't exercising the index at all -- fail
    loud instead of reporting flattering numbers from a trivial benchmark.
    """
    cheapest_value = min(system.sweep_values)
    default_iscan = iterative_scan_values[0]
    min_conns = min(n_conns_list)
    pts = [p for p in unfiltered_sweep if p["sweep_value"] == cheapest_value
           and p["iterative_scan"] == default_iscan and p["n_conns"] == min_conns]
    if not pts:
        log.warning("hardness check: no matching cheapest-point sweep result for system %s", system.name)
        return None
    point = pts[0]
    recall = point["recall"]
    if point.get("valid", True) and recall is not None and not np.isnan(recall) and recall > threshold:
        raise RuntimeError(
            f"dataset too easy — recalibrate: system={system.name} cheapest sweep point "
            f"({system.sweep_param}={cheapest_value}) achieved unfiltered recall={recall:.4f} > "
            f"{threshold} at n_conns={min_conns}. Raise dataset.cluster_sigma, widen "
            f"cluster_scale_mix, and/or raise background_frac in the config."
        )
    return recall


def _run_sift_sanity(cfg: dict) -> dict:
    """Best-effort real-data sanity workload: unfiltered hnsw sweep only, since
    SIFT1M has no metadata to filter on and this is a sanity check, not the
    main methodology surface. Not subject to the synthetic-dataset hardness
    check -- SIFT1M's geometry is real data, not ours to recalibrate."""
    sift = sift_mod.try_load_sift1m()
    if sift is None:
        return dict(available=False, reason="download or parse failed; see logs")

    from . import workloads as wl_mod
    n_queries = cfg.get("sift_n_queries", 500)
    warmup_n = cfg.get("warmup_queries", 10)
    n_conns_list = (1,)
    n_warmup_pool = max(n_conns_list) * warmup_n
    rng = np.random.default_rng(cfg["seed"])
    q_idx = rng.choice(len(sift.queries), size=n_queries + n_warmup_pool, replace=False)
    measured_idx, warmup_idx = q_idx[:n_queries], q_idx[n_queries:]
    queries = sift.queries[measured_idx]
    warmup_pool = sift.queries[warmup_idx]
    truth = sift.groundtruth[measured_idx]

    # `data_dir` MUST follow the main run's override (threaded in via the
    # private `_data_dir` cfg key -- see run()): this instance starts after
    # the main one stopped, so REUSING the same directory is safe
    # sequentially, but falling back to the default bench/vector/.pgdata here would
    # rmtree a directory a CONCURRENT run (the whole reason --data-dir
    # exists) may currently own.
    with PgInstance(data_dir=cfg.get("_data_dir"), external=cfg.get("_external_pg")) as pg:
        with pg.connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            loader_mod.create_and_load(conn, TABLE, sift.vectors,
                                        np.zeros(len(sift.vectors), dtype=np.int32),
                                        filter_specs=[], seed=cfg["seed"], n_centers=1)
        systems = default_systems(cfg["sift_systems"])
        out = {}
        for system in systems:
            build = wl_mod.build_index(pg, TABLE, COLUMN, "l2", system, len(sift.vectors))
            # pg_pinecone_native in the SIFT section gets the same per-measured-query
            # scan-mode assertion as the main loop: unfiltered scans on a
            # clustered build are mode H (computed via the same
            # small_n_threshold rule as the main loop, not assumed).
            sift_pg_pinecone_ctx = (dict(filtered=False,
                                     is_flat=len(sift.vectors)
                                     <= system.build_params.get("small_n_threshold", 10_000))
                                if system.kind == "pg_pinecone_native" else None)
            sweep = wl_mod.run_sweep(pg.dsn(), TABLE, COLUMN, "l2", system, queries, warmup_pool,
                                      truth, k=cfg.get("k", 10), n_conns_list=n_conns_list,
                                      warmup_n=warmup_n, expected_index=system.index_name(TABLE),
                                      n_rows=len(sift.vectors),
                                      min_throughput_seconds=cfg.get("min_throughput_seconds", 5.0),
                                      pg_pinecone_mode_ctx=sift_pg_pinecone_ctx)
            out[system.name] = dict(build=build, sweep=sweep,
                                     latency_at_recall=_latency_at_recall_grouped(
                                         sweep, n_conns_list, system.iterative_scan_values))
    return dict(available=True, n=len(sift.vectors), dim=sift.vectors.shape[1], systems=out)


def _table_rows(conn, table: str) -> int | None:
    if not conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0]:
        return None
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


_ATTSTORAGE = {"p": "plain", "e": "external", "m": "main", "x": "extended"}


def _table_storage_type(conn, table: str) -> tuple[str, str, str] | None:
    """(type name, column storage, the type's default storage) of `table.embedding`,
    e.g. ("vector", "plain", "external")."""
    row = conn.execute(
        "SELECT t.typname, a.attstorage, t.typstorage FROM pg_attribute a JOIN pg_type t ON t.oid = a.atttypid "
        "WHERE a.attrelid = to_regclass(%s) AND a.attname = 'embedding' AND NOT a.attisdropped",
        (table,),
    ).fetchone()
    return (row[0], _ATTSTORAGE[row[1]], _ATTSTORAGE[row[2]]) if row else None


def _check_halfvec_ef_search(system, k_list, k: int, filtered: bool) -> None:
    """A halfvec hnsw arm asks the index for LIMIT k (k x rerank_multiplier candidates when reranking); pgvector
    returns at most ef_search rows, so a smaller ef_search scores an artificially low recall. Checked for every k the
    run will query (its `sweep_values_by_k`, and `k` with the plain `sweep_values` for filtered cells) before the
    table load or any index build."""
    if system.kind != "hnsw" or system.index_type != "halfvec":
        return
    mult = system.rerank_multiplier or 1
    cells = [] if system.skip_unfiltered else [
        (kk, (system.sweep_values_by_k or {}).get(kk, system.sweep_values)) for kk in k_list]
    if filtered:
        cells.append((k, system.sweep_values))
    for kk, values in cells:
        too_small = [v for v in values if int(v) < kk * mult]
        if too_small:
            raise ValueError(
                f"{system.name}: ef_search {too_small} is below LIMIT {kk * mult} "
                f"(k={kk}, rerank_multiplier={mult}); "
                "pgvector can return fewer candidates, making recall artificially low")


def _run_hook(cmd: str | None, system_name: str) -> dict | None:
    if not cmd:
        return None
    with timed() as t:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           env=dict(os.environ, SYSTEM=system_name))
    if p.returncode:
        raise RuntimeError(f"hook {cmd!r} failed ({p.returncode}): {p.stderr.strip()}")
    log.info("  hook %r: %.1fs", cmd, t["seconds"])
    return dict(cmd=cmd, seconds=t["seconds"], output=p.stdout.strip())


def _pg_settings(conn) -> dict:
    rows = conn.execute("SELECT name, setting, coalesce(unit, ''), source FROM pg_settings "
                        "WHERE source NOT IN ('default', 'override') ORDER BY name").fetchall()
    out = dict(non_default=[list(r) for r in rows],
               version=conn.execute("SELECT version()").fetchone()[0])
    for ext in ("vector", "pg_pinecone"):
        r = conn.execute("SELECT extversion FROM pg_extension WHERE extname = %s", (ext,)).fetchone()
        out[f"{ext}_extversion"] = r[0] if r else None
    return out


def _run_partition(wl_mod, pg, table, metric, system, part_workload, part_cfg, n_conns_list, warmup_n,
                   min_throughput_seconds, expected_index) -> dict:
    queries, warmup, gt = part_workload
    col = system.build_params["partition_column"]
    out = {}
    for kk in part_cfg.get("k_values", [10]):
        out[str(kk)] = wl_mod.run_sweep(
            pg.dsn(), table, COLUMN, metric, system, queries, warmup, gt, kk,
            where_clause=f"WHERE {col} = %s", plan_where=f"WHERE {col} = {queries[0][0]}",
            n_conns_list=n_conns_list, warmup_n=warmup_n, expected_index=expected_index,
            min_throughput_seconds=min_throughput_seconds)
    return out


def run(cfg: dict, data_dir: str | None = None, prep_only: bool = False) -> dict | None:
    """`data_dir`, if given, overrides `PgInstance`'s default throwaway
    `bench/vector/.pgdata` location -- REQUIRED whenever another run (e.g. a
    long-running baseline series) might already own that default directory:
    `PgInstance.initdb()` does `shutil.rmtree(data_dir)` on whatever it
    finds there, so two runs sharing the default path is a real "delete the
    other run's live data out from under it" hazard, not just a port clash
    (ports are always freshly OS-assigned regardless -- see
    `harness/util.free_port`). `run.py --data-dir` is the CLI surface for
    this.
    """
    started = dt.datetime.now(dt.timezone.utc)
    started_tag = started.strftime("%Y%m%dT%H%M%SZ")
    # Fingerprint the code at START as well as at the end (the usual stamps):
    # commits landing in this tree MID-RUN are a real failure mode -- the
    # parent process keeps its already-imported modules, but multiprocessing
    # spawn workers re-import from DISK, so a long run silently mixes two code
    # versions (observed 2026-09-12: a worker-return-arity change landed 4h
    # into the real-corpus baseline run; freshly spawned pool workers returned
    # 5-tuples to a parent unpacking 4 -- ValueError after 4.4h of work). The
    # end-of-run stamp alone would have attributed the results to the NEW sha.
    # save_results refuses a run whose start/end hashes differ, same as dirty.
    content_hash_start = bench_content_hash(BENCH_DIR)
    dataset = _load_dataset(cfg)
    metric = cfg["metric"]
    k = cfg["k"]
    warmup_n = cfg["warmup_queries"]
    n_conns_list = tuple(cfg["concurrency"])
    filter_specs = cfg["filters"].get("run", [])
    col_specs = loader_mod.column_specs(filter_specs + cfg["filters"].get("load", []))
    table = cfg.get("table", TABLE)
    stage = cfg.get("stage") or "all"
    reuse = bool(cfg.get("reuse")) or stage == "measure"
    remote_mod.configure(cfg.get("remote"))
    part_cfg = cfg.get("partition")
    tenant_ids = None
    if part_cfg:
        n_tenants = int(part_cfg["n_tenants"])
        per_t = max(1, -(-int(cfg["dataset"]["n_centers"]) // n_tenants))
        tenant_ids = np.minimum(dataset.cluster_ids // per_t, n_tenants - 1).astype(np.int32)
    # Which query regimes to run for filtered cells (default: out_region, plus
    # in_region for correlated filters). A residual-filter benchmark only needs
    # the cross-tenant (out_region) regime.
    query_modes_wanted = cfg["filters"].get("query_modes")
    want_out = query_modes_wanted is None or "out_region" in query_modes_wanted
    want_in = query_modes_wanted is None or "in_region" in query_modes_wanted
    systems = default_systems(cfg["systems"])
    for system in systems:
        system.dimension = dataset.vectors.shape[1]
        if (system.storage_type == "halfvec" or system.index_type == "halfvec") and system.dimension > 4000:
            raise ValueError(f"{system.name}: pgvector halfvec HNSW supports at most 4000 dimensions")
    uses_halfvec = any(s.index_type == "halfvec" or s.storage_type == "halfvec"
                       for s in systems)
    if uses_halfvec and cfg.get("stage") in ("build", "measure"):
        raise ValueError("halfvec comparison requires stage=all so heap reloads cannot reuse a mismatched index")
    if uses_halfvec and any(s.kind in remote_mod.REMOTE_KINDS for s in systems):
        raise ValueError("halfvec comparison cannot share a run with remote systems")
    if uses_halfvec and cfg.get("churn", {}).get("enabled"):
        raise ValueError("halfvec comparison does not support the churn workload")
    # `fp32_column_storage: plain` keeps a 768d FP32 vector (3,080 bytes) in the heap tuple instead of
    # pgvector's default out-of-line TOAST, which an FP16 vector (1,544 bytes) never needs: without it the
    # FP32-heap arms pay a TOAST fetch per vector read that the FP16 heap does not.
    fp32_storage = cfg.get("fp32_column_storage")
    if fp32_storage is not None:
        if fp32_storage not in _ATTSTORAGE.values():
            raise ValueError(f"fp32_column_storage must be one of {sorted(_ATTSTORAGE.values())}, got {fp32_storage!r}")
        if fp32_storage == "plain" and 4 * dataset.vectors.shape[1] + 8 > 8000:
            raise ValueError(f"fp32_column_storage: plain cannot hold a {dataset.vectors.shape[1]}d FP32 vector "
                             "in one 8 KB heap page")
    if cfg.get("sift1m", {}).get("enabled") and any(
        s.get("storage_type") == "halfvec" or s.get("index_type") == "halfvec"
        for s in cfg["sift1m"].get("sift_systems", [])):
        raise ValueError("SIFT1M's bundled FP32 truth cannot validate halfvec precision; disable halfvec in sift1m")
    for system in systems:
        _check_halfvec_ef_search(system, system.k_values or cfg.get("k_values") or [k], k,
                                 filtered=bool(filter_specs))
    n_centers = cfg["dataset"]["n_centers"]
    n_rows = dataset.vectors.shape[0]
    min_throughput_seconds = cfg["min_throughput_seconds"]

    log.info("dataset %s: n=%d dim=%d", dataset.key, dataset.vectors.shape[0], dataset.vectors.shape[1])
    geometry = ds_mod.geometry_diagnostics(dataset.centers, dataset.cluster_sigmas, dataset.vectors.shape[1])
    impurity = ds_mod.cluster_impurity(dataset.vectors, dataset.cluster_ids, dataset.centers)
    log.info("geometry: inter-center=%.1f intra-radius=%.1f ratio=%.3f cluster_impurity=%.4f",
              geometry["mean_inter_center_dist"], geometry["mean_intra_cluster_radius"],
              geometry["ratio"], impurity)

    unfiltered_gt_key = stable_hash(dict(dataset=dataset.key, metric=metric, k=100, kind="unfiltered"))
    with timed() as gt_t:
        unfiltered_truth = gt_mod.brute_force_topk(dataset.queries, dataset.vectors, k=100,
                                                     metric=metric, cache_key=unfiltered_gt_key)
    log.info("unfiltered ground truth: %.1fs", gt_t["seconds"])

    filtered_truth = {}
    filtered_realized = {}
    filtered_mask_hash = {}
    in_region_queries = {}
    in_region_warmup = {}
    in_region_truth = {}
    in_region_sampling = {}  # real corpora only: bank-sampling info per filter
    filtered_combined = {}
    filtered_row_ids = {}
    for spec in filter_specs:
        col = loader_mod.filter_key(spec)
        spec_modes = spec.get("query_modes", query_modes_wanted)
        want_out = spec_modes is None or "out_region" in spec_modes
        want_in = spec_modes is None or "in_region" in spec_modes
        base_mask = ds_mod.assign_filter_values(dataset.cluster_ids, n_centers,
                                                 spec["selectivity"], spec["mode"], cfg["seed"])
        mask = base_mask
        resid = spec.get("residual")
        if resid:
            # The residual column is NOT part of the index key: ground truth is the
            # conjunction, the pushed key's own selectivity stays `base_mask`'s.
            mask = base_mask & ds_mod.assign_filter_values(
                dataset.cluster_ids, n_centers, resid["selectivity"], resid["mode"], cfg["seed"])
        mask_hash = bytes_hash(mask.tobytes())
        filtered_mask_hash[col] = mask_hash
        row_ids = np.nonzero(mask)[0]
        filtered_row_ids[col] = row_ids
        filtered_realized[col] = float(base_mask.mean())
        filtered_combined[col] = float(mask.mean())
        if len(row_ids) < k:
            log.warning("filter %s: only %d qualifying rows (< k=%d) -- recall is over fewer",
                        col, len(row_ids), k)
        # Cache key is the mask's own hash, not just a label derived from
        # selectivity/mode -- if the assignment rule ever changes without the
        # label changing, this still invalidates correctly.
        gt_key = stable_hash(dict(dataset=dataset.key, metric=metric, k=100, mask_hash=mask_hash))
        if want_out:
            with timed() as ft:
                filtered_truth[col] = gt_mod.brute_force_topk(
                    dataset.queries, dataset.vectors[row_ids], k=100, metric=metric,
                    row_ids=row_ids, cache_key=gt_key,
                )
            log.info("filtered ground truth (%s, realized selectivity %.4f, combined %.5f, "
                     "mask_hash=%s): %.1fs", col, filtered_realized[col], filtered_combined[col],
                     mask_hash, ft["seconds"])

        if spec["mode"] == "correlated" and want_in:
            # In-region ("tenant-style") query workload alongside the default
            # global/out-of-region one: queries
            # sampled from the SAME clusters the filter qualifies.
            chosen = ds_mod.correlated_chosen_clusters(dataset.cluster_ids, n_centers,
                                                        spec["selectivity"], cfg["seed"])
            qualifying_clusters = np.nonzero(chosen)[0]
            region_seed = int(stable_hash(dict(seed=cfg["seed"], sel=spec["selectivity"],
                                                 region="in")), 16) % (2**32)
            n_warmup_pool = max(n_conns_list) * warmup_n
            n_q = cfg["dataset"].get("n_queries", 500)
            if dataset.query_bank is not None:
                # Real corpus: tenant queries are REAL held-out vectors from
                # the qualifying k-means clusters; the sampling info (unique
                # availability / replacement) is persisted per filter.
                iq, iw, region_info = ds_mod.sample_region_queries_real(
                    dataset, qualifying_clusters, n_q, n_warmup_pool, region_seed)
                if region_info["sampled_with_replacement"]:
                    log.warning("in-region query pool for %s sampled WITH REPLACEMENT: "
                                 "only %d unique bank vectors in region for %d requested",
                                 col, region_info["n_unique_available"], region_info["n_requested"])
                in_region_sampling[col] = region_info
            else:
                iq, iw = ds_mod.sample_region_queries(dataset, qualifying_clusters, n_q,
                                                        n_warmup_pool, region_seed)
            in_region_queries[col] = iq
            in_region_warmup[col] = iw
            region_gt_key = stable_hash(dict(dataset=dataset.key, metric=metric, k=100,
                                               mask_hash=mask_hash, region="in", region_seed=region_seed))
            with timed() as rt:
                in_region_truth[col] = gt_mod.brute_force_topk(
                    iq, dataset.vectors[row_ids], k=100, metric=metric, row_ids=row_ids,
                    cache_key=region_gt_key)
            log.info("in-region ground truth (%s): %.1fs", col, rt["seconds"])

    # The primary denominator stays exact original FP32 for every arm. The
    # secondary denominator answers how much ranking was lost by rounding
    # vectors AND queries to FP16, independent of ANN search quality.
    unfiltered_half_truth = None
    filtered_half_truth = {}
    in_region_half_truth = {}
    if uses_halfvec:
        half_vectors = dataset.vectors.astype(np.float16).astype(np.float32)
        half_queries = dataset.queries.astype(np.float16).astype(np.float32)
        unfiltered_half_truth = gt_mod.brute_force_topk(
            half_queries, half_vectors, k=100, metric=metric,
            cache_key=stable_hash(dict(dataset=dataset.key, metric=metric, k=100,
                                       representation="rounded_fp16", kind="unfiltered")))
        for col, row_ids in filtered_row_ids.items():
            mask_hash = filtered_mask_hash[col]
            if col in filtered_truth:
                filtered_half_truth[col] = gt_mod.brute_force_topk(
                    half_queries, half_vectors[row_ids], k=100, metric=metric, row_ids=row_ids,
                    cache_key=stable_hash(dict(dataset=dataset.key, metric=metric, k=100,
                                               representation="rounded_fp16", mask_hash=mask_hash)))
            if col in in_region_truth:
                iq = in_region_queries[col].astype(np.float16).astype(np.float32)
                in_region_half_truth[col] = gt_mod.brute_force_topk(
                    iq, half_vectors[row_ids], k=100, metric=metric, row_ids=row_ids,
                    cache_key=stable_hash(dict(dataset=dataset.key, metric=metric, k=100,
                                               representation="rounded_fp16", mask_hash=mask_hash,
                                               region="in", queries_hash=bytes_hash(iq.tobytes()))))
        del half_vectors, half_queries

    part_workload = None
    if part_cfg:
        part_workload = remote_mod.partition_workload(
            dataset, tenant_ids, n_tenants, int(part_cfg.get("queries_per_tenant", 20)), 100, metric, cfg["seed"],
            max(n_conns_list) * warmup_n)
        log.info("partition workload: %d queries over %d tenants", len(part_workload[0]), n_tenants)

    if prep_only:
        log.info("prep only: ground truth and clustering cached")
        return None

    with PgInstance(data_dir=data_dir, conf_overrides=cfg["postgres"],
                    external=cfg.get("external_pg"), reuse=reuse) as pg:
        with pg.connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            pgvector_version = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0]
            # In-band production-build proof (production-build runs): the
            # debug_assertions GUC is on iff the server was compiled with
            # --enable-cassert -- stamping it from the live server beats
            # trusting an out-of-band pg_config parse of whatever binary
            # happened to be on PATH.
            pg_server_version = conn.execute("SHOW server_version").fetchone()[0]
            pg_debug_assertions = conn.execute("SHOW debug_assertions").fetchone()[0]
            pg_settings = _pg_settings(conn)
            first_storage = systems[0].storage_type
            stored = _table_storage_type(conn, table)
            loaded = (reuse and _table_rows(conn, table) == n_rows and stored is not None
                      and stored[0] == first_storage
                      and (first_storage != "vector" or stored[1] == (fp32_storage or stored[2])))
        if loaded:
            log.info("reusing the loaded table %s (%d rows)", table, n_rows)
        else:
            with timed() as load_t, pg.connect() as conn:
                loader_mod.create_and_load(conn, table, dataset.vectors, dataset.cluster_ids,
                                            filter_specs=col_specs, seed=cfg["seed"],
                                            n_centers=n_centers, tenant_ids=tenant_ids,
                                            storage_type=first_storage,
                                            column_storage=fp32_storage if first_storage == "vector" else None)
            log.info("loaded %d rows: %.1fs", dataset.vectors.shape[0], load_t["seconds"])

        # Delayed import to avoid a cycle at module load time.
        from . import workloads as wl_mod

        hardness_check = {}
        system_results = {}
        # Incremental checkpoint state: `hardness_check`/`system_results` (and
        # the per-system `filtered` dict below) are mutated in place, so every
        # `checkpoint(...)` call dumps everything measured so far.
        ckpt_state = dict(run_name=cfg["name"], started=started.isoformat(),
                          machine=_machine_info(), config=cfg,
                          hardness_check=hardness_check, systems=system_results,
                          churn=None, sift1m=None)

        def checkpoint(stage: str):
            _write_checkpoint(cfg["name"], started_tag, dict(ckpt_state, stage=stage))

        active_storage = first_storage
        for system in systems:
            log.info("== system %s (%s) ==", system.name, system.kind)
            if system.storage_type != active_storage:
                with timed() as reload_t, pg.connect() as conn:
                    loader_mod.create_and_load(conn, table, dataset.vectors, dataset.cluster_ids,
                                               filter_specs=col_specs, seed=cfg["seed"],
                                               n_centers=n_centers, tenant_ids=tenant_ids,
                                               storage_type=system.storage_type,
                                               column_storage=(fp32_storage if system.storage_type == "vector"
                                                               else None))
                active_storage = system.storage_type
                log.info("  reloaded %s heap in %.1fs", active_storage, reload_t["seconds"])
            # kind="exact" is the P1 forced-seqscan baseline: no index exists,
            # every sweep point is run with enable_seqscan=on/enable_indexscan=
            # off/enable_bitmapscan=off (workloads.run_sweep's force_seqscan),
            # and recall is asserted ==1.0 rather than hardness-gated (an exact
            # scan visiting every qualifying row is supposed to be "too easy" --
            # that's not a calibration bug here, it's the point).
            hooks = cfg.get("hooks") or {}
            hook_runs = dict(before_build=_run_hook(hooks.get("before_build"), system.name))
            force_seqscan = system.kind == "exact"
            is_remote = system.kind in remote_mod.REMOTE_KINDS
            # build_index drops every OTHER index on the table first:
            # at no point do two systems' indexes coexist.
            build = wl_mod.build_index(
                pg, table, COLUMN, metric, system, dataset.vectors.shape[0],
                remote=dict(dataset=dataset, reuse=reuse, tenant_ids=tenant_ids) if is_remote else None)
            if uses_halfvec or fp32_storage is not None:
                with pg.connect() as conn:
                    build["heap_size_bytes"], build["total_storage_bytes"], build["toast_size_bytes"] = conn.execute(
                        "SELECT pg_table_size(c.oid), pg_total_relation_size(c.oid), "
                        "coalesce(pg_relation_size(nullif(c.reltoastrelid, 0)), 0) "
                        "FROM pg_class c WHERE c.oid = %s::regclass", (table,)).fetchone()
                    build["column_storage"] = _table_storage_type(conn, table)[1]
            log.info("  build: %.1fs, %.1f MB, peak RSS %.1f MB", build["build_seconds"],
                      build["index_size_bytes"] / 1e6, build["peak_rss_bytes"] / 1e6)
            if "calibration" in build:
                log.info("  calibration: %s", build["calibration"])
            if is_remote and stage == "build":
                system_results[system.name] = dict(kind=system.kind, build_params=system.build_params,
                                                     build=build, unfiltered=dict(sweep=[], latency_at_recall={}),
                                                     filtered={})
                checkpoint(f"{system.name}/build")
                continue
            hook_runs["before_measure"] = _run_hook(hooks.get("before_measure"), system.name)
            if hooks:
                build["hooks"] = hook_runs
            expected_index = system.index_name(table)
            n_conns_sys = tuple(system.concurrency or n_conns_list)
            k_list = system.k_values or cfg.get("k_values") or [k]
            # pg_pinecone_native only: whether THIS build resolved to the small-N
            # exact bootstrap (`am.rs`'s `is_flat`) -- `small_n_threshold`
            # defaults to `local::options::DEFAULT_SMALL_N_THRESHOLD` (10000)
            # when the config doesn't override it, matching `ambuild_local`'s
            # own default resolution exactly (see `System.pg_pinecone_native_expected_mode`).
            pg_pinecone_is_flat = (system.kind == "pg_pinecone_native"
                              and n_rows <= system.build_params.get("small_n_threshold", 10_000))

            unfiltered_pg_pinecone_ctx = (dict(filtered=False, is_flat=pg_pinecone_is_flat)
                                      if system.kind == "pg_pinecone_native" else None)
            by_k = {}
            unfiltered_sweep = []
            unfiltered_first = []
            if system.skip_unfiltered:
                # Config opted out (`skip_unfiltered`, System's field doc):
                # this system exists for its filtered cells only; an empty
                # sweep renders as an empty table, never as blended zeros.
                # Hardness gating is meaningless without an unfiltered sweep
                # (and the exact baseline never gates anyway).
                unfiltered_sweep = []
                hardness_check[system.name] = None
            else:
                for kk in k_list:
                    sys_k = system
                    if system.sweep_values_by_k and kk in system.sweep_values_by_k:
                        sys_k = dataclasses.replace(system, sweep_values=system.sweep_values_by_k[kk])
                    mq = system.max_queries
                    sweep_k = wl_mod.run_sweep(pg.dsn(), table, COLUMN, metric, sys_k,
                                               dataset.queries[:mq] if mq else dataset.queries, dataset.warmup_pool,
                                               tuple(t[:mq] for t in unfiltered_truth) if mq else unfiltered_truth, kk,
                                               n_conns_list=n_conns_sys, warmup_n=warmup_n,
                                               expected_index=expected_index, n_rows=n_rows,
                                               force_seqscan=force_seqscan,
                                               min_throughput_seconds=min_throughput_seconds,
                                               pg_pinecone_mode_ctx=unfiltered_pg_pinecone_ctx,
                                               halfvec_ground_truth=(tuple(t[:mq] for t in unfiltered_half_truth)
                                                                     if mq and unfiltered_half_truth is not None
                                                                     else unfiltered_half_truth))
                    by_k[str(kk)] = dict(sweep=sweep_k, latency_at_recall=_latency_at_recall_grouped(
                        sweep_k, n_conns_sys, system.iterative_scan_values))
                    unfiltered_sweep += sweep_k
                    checkpoint(f"{system.name}/unfiltered/k{kk}")
                unfiltered_first = by_k[str(k_list[0])]["sweep"]
                if force_seqscan:
                    _assert_exact_recall(unfiltered_sweep, f"{system.name}/unfiltered",
                                         recall_key=("recall_halfvec_truth" if system.index_type == "halfvec"
                                                     else "recall"))
                    hardness_check[system.name] = None
                elif is_remote:
                    hardness_check[system.name] = None
                else:
                    hardness_check[system.name] = _hardness_check(system, unfiltered_first, n_conns_sys,
                                                                    system.iterative_scan_values,
                                                                    threshold=cfg["hardness_recall_threshold"])
            unfiltered_result = dict(
                sweep=unfiltered_sweep,
                latency_at_recall=_latency_at_recall_grouped(unfiltered_first, n_conns_sys,
                                                              system.iterative_scan_values),
            )
            if len(by_k) > 1:
                unfiltered_result["by_k"] = by_k

            filtered_result = {}
            # Register the system entry BEFORE the filter loop and mutate
            # `filtered_result` in place, so each per-filter checkpoint below
            # captures the system's completed filters too.
            system_results[system.name] = dict(kind=system.kind, build_params=system.build_params,
                                                 build=build, unfiltered=unfiltered_result,
                                                 filtered=filtered_result)
            if uses_halfvec:
                system_results[system.name].update(storage_type=system.storage_type,
                                                   index_type=system.index_type,
                                                   rerank_multiplier=system.rerank_multiplier)
            checkpoint(f"{system.name}/unfiltered")

            for spec in filter_specs:
                col = loader_mod.filter_key(spec)
                spec_modes = spec.get("query_modes", query_modes_wanted)
                want_out = spec_modes is None or "out_region" in spec_modes
                base_col = loader_mod.filter_column_name(spec["selectivity"], spec["mode"])
                eq_col = loader_mod.filter_eq_column_name(spec["selectivity"], spec["mode"])
                resid = spec.get("residual")
                if system.kind in ("pg_pinecone_native", "pg_pinecone_remote") and system.build_params.get("filter_column") != eq_col:
                    # am.rs's ambuild: at most ONE filter column per pg_pinecone
                    # index, fixed at CREATE INDEX time -- this build wasn't
                    # made with this filter's column, so its equality qual
                    # can't be pushed into this index's scan key at all (it
                    # isn't one of the index's key columns). Recorded as
                    # explicitly not-applicable rather than silently skipped
                    # or, worse, run anyway and misreported as this filter's
                    # result (it would just be the unfiltered scan plus an
                    # executor post-filter, not a test of filtered behavior).
                    filtered_result[col] = dict(
                        not_applicable=True,
                        reason=f"pg_pinecone_native index built with filter_column="
                                f"{system.build_params.get('filter_column')!r}, not {eq_col!r} "
                                "-- rebuild against this filter to measure it",
                    )
                    checkpoint(f"{system.name}/filtered/{col}")
                    continue
                # A residual is a SECOND column that is never an index key: the planner
                # pushes the pg_pinecone key column into the AM and leaves the residual as
                # an executor Filter (pgvector: both are Filters over the index scan).
                if system.kind in ("pg_pinecone_native", "pg_pinecone_remote"):
                    where = f"WHERE {eq_col} = 1"
                    if resid:
                        r_eq = loader_mod.filter_eq_column_name(resid["selectivity"], resid["mode"])
                        where += f" AND {r_eq} = 1"
                else:
                    where = f"WHERE {base_col}"
                    if resid:
                        where += " AND " + loader_mod.filter_column_name(resid["selectivity"],
                                                                         resid["mode"])
                filtered_pg_pinecone_ctx = (
                    dict(filtered=True, realized_selectivity=filtered_realized[col],
                         is_flat=pg_pinecone_is_flat)
                    if system.kind == "pg_pinecone_native" else None)
                query_modes = {}
                for mode_name, mq, mw, mtruth in (
                    *([("out_region", dataset.queries, dataset.warmup_pool, filtered_truth[col])]
                      if want_out else []),
                    *([("in_region", in_region_queries[col], in_region_warmup[col],
                        in_region_truth[col])] if col in in_region_queries else []),
                ):
                    mode_sweep = wl_mod.run_sweep(
                        pg.dsn(), table, COLUMN, metric, system, mq, mw, mtruth, k,
                        where_clause=where, n_conns_list=n_conns_sys, warmup_n=warmup_n,
                        iterative_scan_values=system.filtered_iterative_scan_values,
                        expected_index=expected_index, n_rows=n_rows,
                        force_seqscan=force_seqscan,
                        min_throughput_seconds=min_throughput_seconds,
                        pg_pinecone_mode_ctx=filtered_pg_pinecone_ctx,
                        halfvec_ground_truth=(filtered_half_truth.get(col) if mode_name == "out_region"
                                              else in_region_half_truth.get(col)))
                    if force_seqscan:
                        _assert_exact_recall(mode_sweep, f"{system.name}/filtered/{col}/{mode_name}",
                                             recall_key=("recall_halfvec_truth" if system.index_type == "halfvec"
                                                         else "recall"))
                    forced = None
                    if not force_seqscan and _all_seqscan(mode_sweep):
                        # The default planner never
                        # chose the index for ANY point in this sweep -- run the
                        # SAME grid again with enable_seqscan=off so the report
                        # can distinguish "the planner gives up" from "the index
                        # genuinely can't do this," instead of leaving only the
                        # former on record. Not applicable to the force_seqscan
                        # baseline itself -- there's no index to force.
                        log.info("  filtered %s/%s: planner never chose the index (all seqscan) "
                                  "-- running forced-index companion series", system.name, mode_name)
                        forced_sweep = wl_mod.run_sweep(
                            pg.dsn(), table, COLUMN, metric, system, mq, mw, mtruth, k,
                            where_clause=where, n_conns_list=n_conns_sys, warmup_n=warmup_n,
                            iterative_scan_values=system.filtered_iterative_scan_values,
                            expected_index=expected_index, n_rows=n_rows,
                            min_throughput_seconds=min_throughput_seconds, force_index=True,
                            pg_pinecone_mode_ctx=filtered_pg_pinecone_ctx,
                            halfvec_ground_truth=(filtered_half_truth.get(col) if mode_name == "out_region"
                                                  else in_region_half_truth.get(col)))
                        forced = dict(sweep=forced_sweep,
                                      latency_at_recall=_latency_at_recall_grouped(
                                          forced_sweep, n_conns_sys,
                                          system.filtered_iterative_scan_values))
                    query_modes[mode_name] = dict(
                        sweep=mode_sweep,
                        latency_at_recall=_latency_at_recall_grouped(
                            mode_sweep, n_conns_sys, system.filtered_iterative_scan_values),
                        forced_index_sweep=forced,
                    )

                filtered_result[col] = dict(
                    realized_selectivity=filtered_realized[col], mode=spec["mode"],
                    target_selectivity=spec["selectivity"], mask_hash=filtered_mask_hash[col],
                    where_clause=where, residual=resid,
                    combined_realized_selectivity=filtered_combined[col],
                    in_region_sampling=in_region_sampling.get(col),
                    query_modes=query_modes,
                )
                checkpoint(f"{system.name}/filtered/{col}")

            if is_remote:
                rcfg = cfg.get("remote") or {}
                extras = {}
                if part_workload is not None and system.build_params.get("partition_column"):
                    extras["partition"] = _run_partition(
                        wl_mod, pg, table, metric, system, part_workload, part_cfg, n_conns_sys, warmup_n,
                        min_throughput_seconds, expected_index)
                    system_results[system.name]["remote_extras"] = extras
                    checkpoint(f"{system.name}/partition")
                elif rcfg.get("direct"):
                    dcfg = rcfg["direct"]
                    host = remote_mod.index_host(build["remote"]["index"])
                    extras["direct"] = []
                    for top_k in dcfg.get("top_k", [50]):
                        extras["direct"] += remote_mod.direct_lane(
                            host, build["remote"]["namespace"], dataset.queries, int(top_k),
                            concurrency=tuple(dcfg.get("concurrency", n_conns_sys)), n=int(dcfg.get("n", 1000)),
                            label=f"direct:{system.name}")
                    system_results[system.name]["remote_extras"] = extras
                    checkpoint(f"{system.name}/direct")
                if rcfg.get("flood") and not system.build_params.get("partition_column"):
                    fcfg = rcfg["flood"]
                    extras["flood"] = remote_mod.flood_probe(
                        remote_mod.index_host(build["remote"]["index"]), build["remote"]["namespace"], dataset.queries,
                        top_k=int(fcfg.get("top_k", 50)), clients=int(fcfg.get("clients", 4)), secs=int(fcfg.get("secs", 30)),
                        rest_s=int(fcfg.get("rest_s", 120)), label=f"flood:{system.name}")
                    system_results[system.name]["remote_extras"] = extras
                    checkpoint(f"{system.name}/flood")
                delete_after = bool(rcfg.get("delete_index_after", True))
                extras["teardown"] = remote_mod.teardown(pg, table, system, delete_after)
                if delete_after:
                    with pg.connect() as conn:
                        remote_mod.clear_key(conn)
                system_results[system.name]["remote_extras"] = extras
                checkpoint(f"{system.name}/teardown")

            hook_runs["after_measure"] = _run_hook(hooks.get("after_measure"), system.name)

        with pg.connect() as conn:
            pg_settings = _pg_settings(conn)

        churn_results = None
        churn_cfg = cfg.get("churn", {})
        if churn_cfg.get("enabled") and cfg["dataset"]["kind"] != "synthetic":
            # Churn re-embeds/reinserts by drawing from the GMM generative
            # model (same-topic fresh draws); a real corpus has no generative
            # model to draw from. Fail loud at config level rather than
            # produce synthetic-looking churn on real data.
            raise ValueError("churn is only supported on synthetic datasets "
                             "(it draws replacement rows from the generative model)")
        if churn_cfg.get("enabled"):
            churn_system = default_systems([churn_cfg["system"]])[0] \
                if "system" in churn_cfg else systems[0]
            sweep_value = churn_cfg.get("sweep_value", churn_system.sweep_values[len(churn_system.sweep_values) // 2])
            log.info("== churn: %d cycles on %s (sweep_value=%s) ==", churn_cfg["cycles"],
                      churn_system.name, sweep_value)
            # Churn runs with ONLY the system-under-test's index present too
            #: drop everything left over from the systems loop above
            # and build fresh, regardless of whether churn_system == one of
            # the systems already swept.
            with pg.connect() as conn:
                dropped = wl_mod.drop_all_secondary_indexes(conn, table)
            log.info("  churn index isolation: dropped %s, building %s fresh", dropped, churn_system.name)
            churn_build = wl_mod.build_index(pg, table, COLUMN, metric, churn_system,
                                              dataset.vectors.shape[0])
            if churn_system.kind == "pg_pinecone_native":
                # REINDEX-shaped protocol, not `run_churn`'s in-place live-index
                # churn -- local mode's aminsert is a hard-stub ERROR, so churn's
                # mutations can only reach the index via a full rebuild (see
                # `churn_mod.run_churn_pg_pinecone_native`'s own docstring for the
                # honest-comparability note this is; the rebuild time is
                # counted fully).
                cycles = churn_mod.run_churn_pg_pinecone_native(
                    pg, table, COLUMN, metric, churn_system, dataset, n_centers,
                    cycles=churn_cfg["cycles"], reembed_pct=churn_cfg["reembed_pct"],
                    delete_pct=churn_cfg["delete_reinsert_pct"], seed=cfg["seed"],
                    sweep_value=sweep_value, filter_specs=col_specs, k=k,
                    expected_index=churn_system.index_name(table),
                )
            else:
                cycles = churn_mod.run_churn(
                    pg, table, COLUMN, metric, churn_system, dataset, n_centers,
                    cycles=churn_cfg["cycles"], reembed_pct=churn_cfg["reembed_pct"],
                    delete_pct=churn_cfg["delete_reinsert_pct"], sweep_value=sweep_value,
                    seed=cfg["seed"], filter_specs=col_specs,
                    vacuum_after_each_cycle=churn_cfg.get("vacuum_after_each_cycle", True),
                    k=k, expected_index=churn_system.index_name(table),
                )
            churn_results = dict(
                system=churn_system.name, sweep_value=sweep_value, build=churn_build,
                # This was hardcoded to 10 regardless
                # of the run config's own `k` -- silently inconsistent bookkeeping
                # if a config ever set k != 10. Also records the pre-churn drop
                # list (previously computed but only logged, never persisted).
                pre_churn_dropped_indexes=dropped,
                reindex_shaped=churn_system.kind == "pg_pinecone_native",
                cycles=cycles,
            )
            ckpt_state["churn"] = churn_results
            checkpoint("churn")

    sift_results = None
    if cfg.get("sift1m", {}).get("enabled"):
        log.info("== SIFT1M sanity check ==")
        sift_results = _run_sift_sanity({**cfg["sift1m"], "seed": cfg["seed"], "k": cfg["k"],
                                          "warmup_queries": warmup_n,
                                          "min_throughput_seconds": cfg["min_throughput_seconds"],
                                          "_data_dir": data_dir,
                                          "_external_pg": cfg.get("external_pg")})
        ckpt_state["sift1m"] = sift_results
        checkpoint("sift1m")

    finished = dt.datetime.now(dt.timezone.utc)
    # Git_sha alone doesn't prove the working
    # tree matches that commit. git_dirty says yes/no; bench_content_hash is
    # a reproducible fingerprint of the exact bytes under bench/vector/ that this
    # run actually executed against, independent of commit boundaries.
    dirty = git_dirty(cwd=str(BENCH_DIR))
    content_hash_end = bench_content_hash(BENCH_DIR)
    remote_info = None
    if any(sr["kind"] in remote_mod.REMOTE_KINDS for sr in system_results.values()):
        remote_info = dict(tally=remote_mod.tally_totals(), units=remote_mod.state_glob("units"),
                           indexes=remote_mod.state_glob("index"), stage=stage, reuse=reuse)
    return dict(
        run_name=cfg["name"] + ("-build" if stage == "build" else ""), started=started.isoformat(), finished=finished.isoformat(),
        wall_seconds=(finished - started).total_seconds(), machine=_machine_info(),
        stamps=dict(git_sha=git_sha(cwd=str(BENCH_DIR)), git_dirty=dirty,
                    bench_content_hash=content_hash_end,
                    bench_content_hash_at_start=content_hash_start,
                    # See the start-of-run comment: spawn workers re-import
                    # from disk, so a mid-run code change means the run mixed
                    # two code versions -- the end-of-run sha/hash alone would
                    # misattribute every point measured before the change.
                    code_changed_mid_run=(content_hash_start != content_hash_end),
                    config_hash=stable_hash(cfg), pgvector_extversion=pgvector_version,
                    pg_server_version=pg_server_version,
                    pg_debug_assertions=pg_debug_assertions),
        config=cfg,
        dataset=dict(key=dataset.key, n=int(dataset.vectors.shape[0]), dim=int(dataset.vectors.shape[1]),
                      meta=dataset.meta, geometry=geometry, cluster_impurity=impurity),
        filters_realized_selectivity=filtered_realized, hardness_check=hardness_check,
        systems=system_results, churn=churn_results, sift1m=sift_results,
        pg_settings=pg_settings, remote=remote_info,
    )


def save_results(results: dict, allow_dirty: bool = False) -> Path:
    """Writes results JSON, NaN/Inf sanitized to null (strict-JSON
    compatibility).

    Refuses to save a dirty-tree run under its plain name unless
    `allow_dirty=True`: a run meant to stand as
    "the run of record" must come from a clean, committed tree -- this is the
    mechanical backstop for that requirement, not just process discipline.
    `allow_dirty=True` is for local dev iteration only; such a run's filename
    and rendered report are both marked DIRTY so it can never be mistaken for
    a committed run of record.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamps = results.get("stamps") or {}
    dirty = stamps.get("git_dirty")
    changed_mid_run = stamps.get("code_changed_mid_run")
    if changed_mid_run and not allow_dirty:
        raise RuntimeError(
            "refusing to save: the bench code changed WHILE this run was executing "
            f"(content hash {stamps.get('bench_content_hash_at_start')} at start vs "
            f"{stamps.get('bench_content_hash')} at end). Spawn workers re-import from "
            "disk, so points in this run may have been measured under two different code "
            "versions -- rerun from a quiescent committed tree (or pass allow_dirty=True "
            "for a throwaway dev run, marked DIRTY).")
    if dirty and not allow_dirty:
        raise RuntimeError(
            "refusing to save results from a dirty working tree as the run of record -- "
            "commit the code changes first and re-run, or pass allow_dirty=True for a "
            "throwaway/dev iteration run (its filename and report will be marked DIRTY)")
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "_DIRTY" if (dirty or changed_mid_run) else ""
    path = RESULTS_DIR / f"{results['run_name']}{tag}_{ts}.json"
    with open(path, "w") as f:
        json.dump(sanitize_for_json(results), f, indent=2, default=str, allow_nan=False)
    # The final write succeeded -- this run's incremental checkpoints
    # (`_write_checkpoint`) are now strictly redundant with the complete file.
    for partial in RESULTS_DIR.glob(f"{results['run_name']}_*.partial.json*"):
        partial.unlink(missing_ok=True)
    return path
