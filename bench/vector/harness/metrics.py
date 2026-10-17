"""Latency percentiles, QPS, and recall-vs-latency interpolation."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .util import percentile


def latency_stats(latencies_s: list[float]) -> dict:
    """p50/p95/p99 (ms) + QPS + mean recall placeholder, over a batch of per-query latencies."""
    if not latencies_s:
        return dict(p50_ms=float("nan"), p95_ms=float("nan"), p99_ms=float("nan"),
                    mean_ms=float("nan"), qps=float("nan"), n=0)
    s = sorted(latencies_s)
    total = sum(latencies_s)
    return dict(
        p50_ms=percentile(s, 50) * 1000,
        p95_ms=percentile(s, 95) * 1000,
        p99_ms=percentile(s, 99) * 1000,
        mean_ms=(total / len(s)) * 1000,
        qps=len(s) / total if total > 0 else float("nan"),
        n=len(s),
    )


def interpolate_latency_at_recall(sweep_points: list[dict], targets=(0.90, 0.95, 0.99),
                                   latency_key: str = "p50_ms", max_recall_gap: float = 0.15) -> dict:
    """Given sweep points [{'recall': r, 'p50_ms': l, 'valid': bool, ...}, ...],
    linearly interpolate the latency achieving each target recall.

    Callers are expected to have already grouped `sweep_points` to a single
    plan shape (never blend index-scan and
    seqscan-fallback points into one recall-latency curve -- they're not
    comparable "index performance"). Invalid points (plan didn't match the
    expected index -- see workloads.run_sweep_point) are always excluded here
    too, as a second line of defense.

    We sort by recall (ties broken by latency) and interpolate along that
    recall-latency curve, but ONLY between a bracketing pair whose recall gap
    is <= `max_recall_gap` -- a wide, sparse gap in
    the swept knob doesn't license pretending the curve is linear across it.
    Every target reports both a `value` (None if not interpolable) and the
    `bracket` of real points used, so "n/a" is always paired with the honest
    bracketing data instead of just silence. If a target recall is above the
    max achieved (index can't get there at any swept setting) or below the
    min, `value` is None and `bracket` is None too -- there's nothing to
    bracket with.
    """
    pts = sorted(((p["recall"], p[latency_key]) for p in sweep_points
                  if p.get("valid", True) and not np.isnan(p["recall"])),
                 key=lambda t: (t[0], t[1]))
    out = {}
    for target in targets:
        if not pts:
            out[target] = dict(value=None, bracket=None)
            continue
        recalls = [r for r, _ in pts]
        if target < recalls[0] or target > recalls[-1]:
            out[target] = dict(value=None, bracket=None)
            continue
        # Find bracketing points and linearly interpolate latency in recall-space.
        lo = max(i for i, (r, _) in enumerate(pts) if r <= target)
        hi = min(i for i, (r, _) in enumerate(pts) if r >= target)
        if lo == hi:
            out[target] = dict(value=pts[lo][1], bracket=[list(pts[lo]), list(pts[lo])])
            continue
        r0, l0 = pts[lo]
        r1, l1 = pts[hi]
        bracket = [[r0, l0], [r1, l1]]
        if (r1 - r0) > max_recall_gap:
            out[target] = dict(value=None, bracket=bracket)
            continue
        frac = (target - r0) / (r1 - r0) if r1 != r0 else 0.0
        out[target] = dict(value=l0 + frac * (l1 - l0), bracket=bracket)
    return out


def label_budget_capped(sweep: list[dict], epsilon: float = 0.02) -> list[dict]:
    """Flags points where the iterative-scan visitation budget (e.g.
    hnsw.max_scan_tuples), not the swept knob, is why recall stalls (the
    "0.855 plateau" bug).

    Within each (iterative_scan, n_conns, scan_fraction) group, a point is `budget_capped:
    True` when BOTH hold:
      (a) a sibling point at the SAME sweep_value but the next-HIGHER swept
          scan_budget scores meaningfully higher recall ("the budget is
          binding" -- raising it alone, with the knob held fixed, still moves
          recall), AND
      (b) recall barely moved from the previous (lower) sweep_value within
          THIS point's own budget series ("the knob itself has plateaued").

    `budget_capped` is `None` when scan_budget wasn't swept at this point (a
    single budget value -- nothing to compare against) and `False` when it
    was swept and compared but not found capped. Mutates `sweep` in place and
    returns it.
    """
    groups = defaultdict(list)
    for p in sweep:
        groups[(p.get("iterative_scan"), p.get("n_conns"), p.get("scan_fraction"))].append(p)

    def _finite(r):
        return r is not None and not (isinstance(r, float) and np.isnan(r))

    for grp in groups.values():
        valid = [p for p in grp if p.get("valid", True)]
        budgets = sorted({p["scan_budget"] for p in valid if p.get("scan_budget") is not None})
        if len(budgets) < 2:
            for p in grp:
                p["budget_capped"] = None
            continue
        by_budget_sv = {(p["scan_budget"], p["sweep_value"]): p for p in valid
                         if p.get("scan_budget") is not None}
        for p in grp:
            if p.get("scan_budget") is None or not p.get("valid", True):
                p["budget_capped"] = None
                continue
            higher = [b for b in budgets if b > p["scan_budget"]]
            if not higher:
                p["budget_capped"] = False
                continue
            sibling = by_budget_sv.get((higher[0], p["sweep_value"]))
            budget_binding = bool(
                sibling is not None and _finite(sibling["recall"]) and _finite(p["recall"])
                and (sibling["recall"] - p["recall"]) > epsilon)
            same_budget_series = sorted(
                (q for q in valid if q.get("scan_budget") == p["scan_budget"]),
                key=lambda q: q["sweep_value"])
            svs = [q["sweep_value"] for q in same_budget_series]
            idx = svs.index(p["sweep_value"])
            recall_flat = True
            if idx > 0:
                prev = same_budget_series[idx - 1]
                recall_flat = (_finite(p["recall"]) and _finite(prev["recall"])
                               and (p["recall"] - prev["recall"]) < epsilon)
            p["budget_capped"] = bool(budget_binding and recall_flat)
    return sweep
