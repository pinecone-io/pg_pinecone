"""Markdown tables of REPORT-halfvec.md, straight from the halfvec_1m results JSON (the newest per run name in the directory).
Latency at recall uses the harness's own interpolation (harness/metrics.py); `<=x` means the cheapest point already exceeds the target.
usage: tables_halfvec.py <results dir> [<out file>]   (stdout without an out file)"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from harness.metrics import interpolate_latency_at_recall  # noqa: E402
from tables import load, n, table  # noqa: E402

TARGETS = (0.90, 0.95, 0.99)
CONNS = (1, 4, 16)


def sweep(r, name, k):
    u = r["systems"][name]["unfiltered"]
    pts = (u["by_k"].get(str(k)) or {}).get("sweep", []) if u.get("by_k") else [p for p in u["sweep"] if p.get("k") == k]
    return [p for p in pts if p.get("valid", True)]


def lar(pts, c):
    pts = [p for p in pts if p["n_conns"] == c]
    if not pts:
        return ["-"] * len(TARGETS)
    lo = min(pts, key=lambda p: p["recall"])
    got = interpolate_latency_at_recall(pts, TARGETS)
    return [n(got[t]["value"]) if got[t]["value"] is not None else ("<=" + n(lo["p50_ms"]) if t < lo["recall"] else "-")
            for t in TARGETS]


def curves(r, name, k):
    s = r["systems"][name]
    pts = sweep(r, name, k)
    if s["kind"] != "pg_pinecone_native":
        return [("ef_search", pts)]
    out = []
    for rm in sorted({p["sweep_value"] for p in pts}):
        pool = "calibrated pool" if rm == 0 else f"pool {rm * 100}"
        out.append((pool if len({p.get('scan_fraction') for p in pts}) == 1 else f"probes swept, {pool}",
                    [p for p in pts if p["sweep_value"] == rm]))
    return out


def build_size(r):
    rows = []
    for name, s in r["systems"].items():
        b = s["build"]
        rows.append((f"`{name}`", s.get("storage_type", "vector"), s.get("index_type", "-") if s["kind"] != "pg_pinecone_native" else "native",
                     b.get("column_storage", "-"), n(b["build_seconds"], 0), n(b["index_size_bytes"] / 1e6, 0),
                     n(b["heap_size_bytes"] / 1e6, 0), n(b["total_storage_bytes"] / 1e6, 0),
                     n(b["peak_rss_bytes"] / 1e9, 2) if b["peak_rss_bytes"] else "-"))
    return table(["System", "Heap", "Index", "Column storage", "Build s", "Index MB", "Heap MB", "Heap + index MB", "Peak RSS GB (leader)"], rows)


def frontier(r, k):
    rows = []
    for name, s in r["systems"].items():
        if s["kind"] == "exact":
            continue
        for label, pts in curves(r, name, k):
            if len({p["recall"] for p in pts if p["n_conns"] == 1}) == 1 and s["kind"] == "pg_pinecone_native" and "swept" not in label:
                continue
            best = max((p["recall"] for p in pts), default=None)
            rows.append((f"`{name}`", label, *(" / ".join(lar(pts, c)) for c in CONNS), n(best, 3)))
    return table(["System", "Curve", "c=1 p50 ms at recall 0.90 / 0.95 / 0.99", "c=4", "c=16", "Best recall"], rows)


def points(r, k, names):
    rows = []
    for name in names:
        by = {}
        for p in sweep(r, name, k):
            by.setdefault((p["sweep_value"], p.get("scan_fraction")), {})[p["n_conns"]] = p
        for (sv, sf), cs in by.items():
            c1, c16 = cs.get(1), cs.get(16)
            knob = (f"pool {'calibrated' if sv == 0 else sv * 100}" + (f", probes {c1.get('scan_lists_probed_mean'):.0f}" if c1 and c1.get("scan_lists_probed_mean") else "")
                    if r["systems"][name]["kind"] == "pg_pinecone_native" else "forced seqscan" if r["systems"][name]["kind"] == "exact" else f"ef {sv}")
            rows.append((f"`{name}`", knob, n(c1["recall"], 3), f"{n(c1['p50_ms'])} / {n(c1['p95_ms'])} / {n(c1['p99_ms'])}",
                         n(c1.get("scan_rescored_mean"), 0) if c1.get("scan_rescored_mean") is not None else "-",
                         f"{n(c16['p50_ms'])} ({n(c16['qps_wall'], 0)}/s)" if c16 else "-"))
    return table(["System", "Setting", f"recall@{k}", "c=1 p50 / p95 / p99 ms", "Rows rescored", "c=16 p50 ms (QPS)"], rows)


def calibration(r):
    rows = []
    for name, s in r["systems"].items():
        c = s["build"].get("calibration")
        if not c:
            continue
        d10 = [p for p in sweep(r, name, 10) if p["sweep_value"] == 0 and not p.get("scan_fraction") and p["n_conns"] == 1]
        d100 = {p["sweep_value"]: p for p in sweep(r, name, 100) if not p.get("scan_fraction") and p["n_conns"] == 1}
        rows.append((n(c["target_recall"], 2), c["calib_nprobe"], c["calib_pool"], n(c["calib_recall"], 3),
                     n(d10[0]["recall"], 3) if d10 else "-", n(d10[0]["p50_ms"]) if d10 else "-",
                     n(d100[0]["recall"], 3) if 0 in d100 else "-", n(d100[10]["recall"], 3) if 10 in d100 else "-"))
    return table(["target_recall", "nprobe (of 1,000)", "Pool", "Build sample recall@10", "Held-out recall@10", "c=1 p50 ms",
                  "recall@100, calibrated pool", "recall@100, pool 1,000"], rows)


def precision(r):
    rows = []
    for name, s in r["systems"].items():
        pts = [p for p in s["unfiltered"]["sweep"] if p.get("valid", True) and p.get("recall_halfvec_truth") is not None]
        if not pts:
            continue
        gap = max(abs(p["recall_halfvec_truth"] - p["recall"]) for p in pts)
        rows.append((f"`{name}`", len(pts), n(gap, 4)))
    return table(["System", "Points", "Largest |FP16-truth recall - FP32 recall|"], rows)


FAMILIES = (("pgvector FP32", lambda s: s["kind"] == "hnsw" and s.get("index_type") == "vector"),
            ("pgvector FP16", lambda s: s["kind"] == "hnsw" and s.get("index_type") == "halfvec" and not s.get("rerank_multiplier")),
            ("native", lambda s: s["kind"] == "pg_pinecone_native"))


def fastest(r, test, k, c, t):
    cand = [p for name, s in r["systems"].items() if test(s) for p in sweep(r, name, k) if p["n_conns"] == c and p["recall"] >= t]
    return (min(p["p50_ms"] for p in cand), max(p["qps_wall"] for p in cand)) if cand else None


def gap(r):
    rows = []
    for k in (10, 100):
        for c in (1, 16):
            for t in TARGETS:
                fp32, fp16, nat = (fastest(r, test, k, c, t) for _, test in FAMILIES)
                closes = ("all" if fp16[0] <= nat[0] else f"{100 * (fp32[0] - fp16[0]) / (fp32[0] - nat[0]):.0f}%") if fp32 and fp16 and nat else "-"
                cell = lambda v: "-" if v is None or nat is None else f"{n(v[0])} ({v[0] / nat[0]:.2f}x)"
                rows.append((k, c, n(t, 2), cell(fp32), cell(fp16), n(nat[0]) if nat else "-", closes,
                             " / ".join(n(v[1], 0) if v else "-" for v in (fp32, fp16, nat)) if c == 16 else ""))
    return table(["k", "Clients", "Recall >=", "pgvector FP32 p50 ms (x native)", "pgvector FP16 p50 ms (x native)", "Native p50 ms",
                  "FP16 closes of the FP32 gap", "Best QPS at 16 clients, FP32 / FP16 / native"], rows)


def footprint(r):
    rows = []
    for label, test in FAMILIES:
        b = [s["build"] for s in r["systems"].values() if test(s)]
        rng = lambda xs, f, nd=0: n(min(xs) / f, nd) if n(min(xs) / f, nd) == n(max(xs) / f, nd) else f"{n(min(xs) / f, nd)} to {n(max(xs) / f, nd)}"
        rows.append((label, rng([x["index_size_bytes"] for x in b], 1e6), rng([x["total_storage_bytes"] for x in b], 1e9, 1),
                     rng([x["build_seconds"] for x in b], 1), rng([x["peak_rss_bytes"] for x in b], 1e9, 1)))
    return table(["Family", "Index MB", "Heap + index GB", "Build s", "Build peak RSS GB"], rows)


def main():
    r = load(sys.argv[1])["halfvec_1m"]
    pg_arms = [nm for nm, s in r["systems"].items() if s["kind"] == "hnsw"]
    native = [nm for nm, s in r["systems"].items() if s["kind"] == "pg_pinecone_native"]
    exact = [nm for nm, s in r["systems"].items() if s["kind"] == "exact"]
    parts = ["## Gap to native", gap(r), "## Footprint by family", footprint(r),
             "## Build and size", build_size(r),
             "## Latency at recall, k = 10", frontier(r, 10),
             "## Latency at recall, k = 100", frontier(r, 100),
             "## Native calibration", calibration(r),
             "## Points, k = 10", points(r, 10, pg_arms + native + exact),
             "## Points, k = 100", points(r, 100, pg_arms + native),
             "## FP16 rounding", precision(r)]
    text = "\n\n".join(parts) + "\n"
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
