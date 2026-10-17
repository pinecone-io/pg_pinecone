"""Renders a results JSON dict into a compact markdown summary."""
from __future__ import annotations


def _fmt_ms(v):
    return "n/a" if v is None else f"{v:.2f}"


def _fmt_lar_cell(lar: dict | None) -> str:
    """Renders one {'value':..,'bracket':..} interpolation cell. When value is
    None but a bracket exists, shows the bracketing pair instead of a bare
    'n/a' (honest about *why* it's n/a)."""
    if lar is None:
        return "n/a"
    if lar.get("value") is not None:
        return f"{lar['value']:.2f}"
    b = lar.get("bracket")
    if b:
        (r0, l0), (r1, l1) = b
        return f"n/a (bracket {r0:.3f}@{l0:.2f}ms .. {r1:.3f}@{l1:.2f}ms)"
    return "n/a"


def _latency_at_recall_table(lar_by_iscan_budget_shape_conns: dict) -> list[str]:
    """Renders a (iterative_scan, scan_budget, plan_shape, n_conns) -> latency
    table. `scan_budget` is a nesting level: the default hnsw.max_scan_tuples budget and a
    raised, table-size-scaled budget are separate configurations, never blended into
    one recall-latency curve (see runner.py's `_latency_at_recall_grouped`).
    """
    lines = ["| iter_scan | scan_budget | plan_shape | n_conns | recall@0.90 | recall@0.95 | recall@0.99 |",
             "|---|---|---|---|---|---|---|"]
    for iscan, by_budget in lar_by_iscan_budget_shape_conns.items():
        for budget, by_shape in by_budget.items():
            for shape, by_conns in by_shape.items():
                for nc, lar in by_conns.items():
                    lines.append(f"| {iscan} | {budget} | {shape} | {nc} | "
                                  f"{_fmt_lar_cell(lar.get(0.90))} | {_fmt_lar_cell(lar.get(0.95))} | "
                                  f"{_fmt_lar_cell(lar.get(0.99))} |")
    return lines


def _fmt_calibration(cal: dict | None) -> str:
    if cal is None:
        return "n/a (this pg_pinecone build has no index_calibration)"
    return ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in cal.items())


def _sweep_table(sweep: list[dict]) -> list[str]:
    """`rows` = mean rows returned at LIMIT k; `rescored` = mean exact-rescored
    tuples per query (`pinecone.last_scan_stats()`, only when the system sets
    `record_scan_stats`); `scan_frac` only for systems that sweep it."""
    has_half_truth = any("recall_halfvec_truth" in p for p in sweep)
    recall_header = "FP32 recall | FP16 truth" if has_half_truth else "recall"
    lines = ["| sweep_value | iter_scan | scan_budget | scan_frac | n_conns | plan_shape | valid | "
             f"{recall_header} | rows | rescored | budget_capped | p50 ms | p95 ms | p99 ms | QPS |",
             "|---|---|---|---|---|---|---|---|" + ("---|" if has_half_truth else "")
             + "---|---|---|---|---|---|---|"]
    for p in sweep:
        valid = p.get("valid", True)
        recall = "n/a" if p["recall"] is None or (isinstance(p["recall"], float) and p["recall"] != p["recall"]) \
            else f"{p['recall']:.3f}"
        half = p.get("recall_halfvec_truth")
        half_s = ("n/a" if half is None or half != half else f"{half:.3f}")
        half_cell = f"{half_s} | " if has_half_truth else ""
        budget = p.get("scan_budget")
        budget_s = "-" if budget is None else str(budget)
        sfrac = p.get("scan_fraction")
        sfrac_s = "-" if sfrac is None else str(sfrac)
        rows = p.get("rows_returned_mean")
        rows_s = "-" if rows is None or rows != rows else f"{rows:.1f}"
        resc = p.get("scan_rescored_mean")
        resc_s = "-" if resc is None else f"{resc:.0f}"
        bc = p.get("budget_capped")
        bc_s = "-" if bc is None else ("YES" if bc else "no")
        if not valid:
            lines.append(f"| {p['sweep_value']} | {p['iterative_scan']} | {budget_s} | {sfrac_s} | "
                          f"{p['n_conns']} | {p.get('plan_shape')} | "
                          f"**INVALID: {p.get('abort_reason')}** | {recall} | {half_cell}- | - | {bc_s} | "
                          "n/a | n/a | n/a | n/a |")
            continue
        lines.append(f"| {p['sweep_value']} | {p['iterative_scan']} | {budget_s} | {sfrac_s} | "
                      f"{p['n_conns']} | {p.get('plan_shape')} | ok | {recall} | {half_cell}{rows_s} | "
                      f"{resc_s} | {bc_s} | {p['p50_ms']:.2f} | {p['p95_ms']:.2f} | "
                      f"{p['p99_ms']:.2f} | {p['qps_wall']:.1f} |")
    return lines


def _render_query_mode(name: str, mode_res: dict) -> list[str]:
    lines = [f"#### query mode: {name}", "", "Latency at target recall:"]
    lines += _latency_at_recall_table(mode_res["latency_at_recall"])
    lines.append("")
    lines.append("Raw sweep:")
    lines += _sweep_table(mode_res["sweep"])
    lines.append("")
    forced = mode_res.get("forced_index_sweep")
    if forced:
        # Every point above fell back to a
        # seqscan under the default planner -- this companion series forces
        # the index (`SET enable_seqscan = off`) so the report can show
        # whether the index itself is capable, separate from whether the
        # planner would ever choose it unprompted. Never blended with the
        # planner-choice series above.
        lines.append("**Forced-index companion series** (`enable_seqscan = off`): the default "
                      "planner never chose the index for any point above (all seqscan). This "
                      "answers \"planner gives up\" vs. \"index incapable\" -- kept as its own "
                      "series, never blended with the planner-choice numbers above.")
        lines.append("")
        lines.append("Latency at target recall (forced index):")
        lines += _latency_at_recall_table(forced["latency_at_recall"])
        lines.append("")
        lines.append("Raw sweep (forced index):")
        lines += _sweep_table(forced["sweep"])
        lines.append("")
    return lines


def render(results: dict) -> str:
    lines = [f"# Bench run: {results['run_name']}", ""]
    stamps = results.get("stamps") or {}
    dirty = stamps.get("git_dirty")
    if stamps.get("code_changed_mid_run"):
        # Same severity as a dirty tree: points in this run may have been
        # measured under two different code versions (spawn workers re-import
        # from disk) -- see runner.run's start/end content-hash stamps.
        dirty = True
    if dirty or dirty is None:
        # An unknown dirty state is never
        # allowed to read as clean either -- only an explicit `dirty is
        # False` earns the "run of record" framing below.
        lines.append(f"**{'DIRTY TREE' if dirty else 'PROVENANCE UNKNOWN'} -- NOT the run of "
                      "record.** " +
                      ("Uncommitted changes were present when this ran; re-run from a clean "
                       "commit before trusting/publishing any number below."
                       if dirty else
                       "git status could not be determined for this run; provenance is unverified."))
        lines.append("")
    lines.append(f"- started: {results['started']}")
    lines.append(f"- wall time: {results['wall_seconds']:.1f}s")
    lines.append(f"- machine: {results['machine']}")
    lines.append(f"- git SHA: {stamps.get('git_sha')}, dirty: {dirty}, "
                  f"bench content hash: {stamps.get('bench_content_hash')}, "
                  f"config hash: {stamps.get('config_hash')}, pgvector: {stamps.get('pgvector_extversion')}")
    ds = results["dataset"]
    lines.append(f"- dataset: {ds['key']} (n={ds['n']}, dim={ds['dim']}), meta={ds['meta']}")
    geom = ds.get("geometry")
    if geom:
        lines.append(f"- geometry: mean inter-center dist={geom['mean_inter_center_dist']:.1f}, "
                      f"mean intra-cluster radius={geom['mean_intra_cluster_radius']:.1f}, "
                      f"ratio={geom['ratio']:.3f}, cluster_impurity={ds.get('cluster_impurity'):.4f}")
    hardness = results.get("hardness_check")
    if hardness:
        lines.append(f"- hardness check (cheapest-point unfiltered recall, must be <= "
                      f"{results['config'].get('hardness_recall_threshold', 0.97)}): {hardness}")
    lines.append("")

    for name, sysres in results["systems"].items():
        lines.append(f"## {name} ({sysres['kind']}, build_params={sysres['build_params']})")
        if "storage_type" in sysres:
            lines.append(f"- heap: {sysres['storage_type']}; index: {sysres['index_type']}; "
                         f"FP32 rerank multiplier: {sysres['rerank_multiplier']}")
        b = sysres["build"]
        lines.append(f"- build time: {b['build_seconds']:.2f}s, index size: {b['index_size_bytes']/1e6:.1f} MB, "
                      f"peak backend RSS: {b['peak_rss_bytes']/1e6:.1f} MB, "
                      f"dropped other indexes first: {b.get('dropped_indexes')}")
        if "calibration" in b:
            lines.append(f"- calibration (`pinecone.index_calibration`): {_fmt_calibration(b['calibration'])}")
        if "heap_size_bytes" in b:
            storage = (f" (column storage {b['column_storage']}, TOAST {b['toast_size_bytes']/1e6:.1f} MB)"
                       if "column_storage" in b else "")
            lines.append(f"- heap size: {b['heap_size_bytes']/1e6:.1f} MB{storage}; "
                         f"heap + index: {b['total_storage_bytes']/1e6:.1f} MB")
        if b.get("graph_spilled_to_disk"):
            # Pgvector's build-time NOTICE that the
            # graph no longer fit in maintenance_work_mem, surfaced here instead
            # of only living in a psycopg notice buffer nobody reads.
            lines.append(f"- **GRAPH SPILLED TO DISK** (maintenance_work_mem too small for this "
                          f"build): {b.get('build_notices')}")
        lines.append("")
        lines.append("### Unfiltered: latency at target recall")
        lines += _latency_at_recall_table(sysres["unfiltered"]["latency_at_recall"])
        lines.append("")
        lines.append("### Unfiltered: raw sweep")
        lines += _sweep_table(sysres["unfiltered"]["sweep"])
        lines.append("")

        for col, fres in sysres["filtered"].items():
            if fres.get("not_applicable"):
                # pg_pinecone_native only: this system's index was built with a
                # DIFFERENT (or no) filter column -- am.rs's ambuild fixes at
                # most one filter column per index at CREATE INDEX time, so
                # this filter's equality qual can't be pushed into this
                # index's scan key at all. See runner.py's per-filter loop.
                lines.append(f"### Filtered `{col}`: not applicable to {name}")
                lines.append(f"- {fres.get('reason')}")
                lines.append("")
                continue
            lines.append(f"### Filtered `{col}` (mode={fres['mode']}, target selectivity="
                          f"{fres['target_selectivity']}, realized={fres['realized_selectivity']:.4f}, "
                          f"mask_hash={fres.get('mask_hash')})")
            if fres.get("residual"):
                lines.append(f"- residual (not pushed into the index): `{fres['residual']}`, combined "
                              f"pass rate {fres.get('combined_realized_selectivity', float('nan')):.6f}; "
                              f"`{fres.get('where_clause')}`")
            lines.append("")
            for mode_name, mode_res in fres["query_modes"].items():
                lines += _render_query_mode(mode_name, mode_res)

    if results.get("churn"):
        c = results["churn"]
        b = c.get("build")
        lines.append(f"## Churn ({c['system']}, sweep_value={c['sweep_value']})")
        if c.get("pre_churn_dropped_indexes") is not None:
            lines.append(f"- pre-churn index isolation: dropped {c['pre_churn_dropped_indexes']} "
                          f"before rebuilding fresh")
        if b:
            lines.append(f"- index rebuilt in isolation: {b['build_seconds']:.2f}s, "
                          f"{b['index_size_bytes']/1e6:.1f} MB, dropped first: {b.get('dropped_indexes')}")
        if c.get("reindex_shaped"):
            # pg_pinecone_native's REINDEX-shaped protocol (churn.py:
            # run_churn_pg_pinecone_native's own docstring) -- no live index exists
            # mid-cycle to have a "pre-vacuum" recall, so this is a DIFFERENT
            # table shape from pgvector's in-place churn, not the same
            # columns with blanks: the number is
            # maintenance_seconds_total (vacuum + drop + full rebuild,
            # counted fully), not a vacuum time alone.
            lines.append("**REINDEX-shaped protocol** (local mode's `aminsert` is a hard-stub ERROR -- "
                         "see `churn.py:run_churn_pg_pinecone_native`'s docstring): each cycle's mutations "
                         "only reach the index via a full rebuild, an HONEST comparability "
                         "difference from pgvector's in-place churn above, not a like-for-like one.")
            lines.append("")
            lines.append("| cycle | live_before | reembedded | deleted/reinserted | plan_shape | "
                          "recall post-rebuild | vacuum s | drop s | rebuild s | maintenance s (total) |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for cyc in c["cycles"]:
                lines.append(f"| {cyc['cycle']} | {cyc['n_live_before']} | {cyc['n_reembedded']} | "
                              f"{cyc['n_deleted']} | {cyc.get('plan_shape')} | "
                              f"{cyc['recall_at_k_post_rebuild']:.3f} | {cyc['vacuum_seconds']:.2f} | "
                              f"{cyc['drop_index_seconds']:.2f} | {cyc['rebuild_seconds']:.2f} | "
                              f"{cyc['maintenance_seconds_total']:.2f} |")
            lines.append("")
        else:
            # plan_shape column: every cycle re-checks
            # EXPLAIN after its own ANALYZE, not just once before cycle 0.
            lines.append("| cycle | live_before | reembedded | deleted/reinserted | plan_shape | "
                          "recall pre-vacuum | recall post-vacuum | cycle s | vacuum s |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for cyc in c["cycles"]:
                pre = "n/a" if cyc["recall_at_k_pre_vacuum"] is None else f"{cyc['recall_at_k_pre_vacuum']:.3f}"
                post = "n/a" if cyc["recall_at_k_post_vacuum"] is None else f"{cyc['recall_at_k_post_vacuum']:.3f}"
                vac = "n/a" if cyc["vacuum_seconds"] is None else f"{cyc['vacuum_seconds']:.2f}"
                lines.append(f"| {cyc['cycle']} | {cyc['n_live_before']} | {cyc['n_reembedded']} | "
                              f"{cyc['n_deleted']} | {cyc.get('plan_shape')} | "
                              f"{pre} | {post} | "
                              f"{cyc['cycle_seconds']:.2f} | {vac} |")
            lines.append("")

    if results.get("sift1m") is not None:
        s = results["sift1m"]
        lines.append("## SIFT1M real-data sanity check")
        if not s["available"]:
            lines.append(f"- SKIPPED: {s['reason']}")
        else:
            lines.append(f"- n={s['n']}, dim={s['dim']}")
            for name, sysres in s["systems"].items():
                b = sysres["build"]
                lines.append(f"### {name}")
                lines.append(f"- build time: {b['build_seconds']:.2f}s, index size: "
                              f"{b['index_size_bytes']/1e6:.1f} MB")
                lines += _latency_at_recall_table(sysres["latency_at_recall"])
        lines.append("")

    return "\n".join(lines)
