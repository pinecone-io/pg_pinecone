"""Exact brute-force top-k ground truth, computed once and cached to disk.

Cache key is a hash of (dataset key, metric, k, and -- for filtered ground
truth -- the boolean qualifying mask), so re-running the same config is free
and changing any input invalidates the cache.

Ground truth carries DISTANCES alongside ids (cached as one .npz) because
recall is tie-aware: real corpora contain exact duplicate vectors (434 in the
1M wiki_dpr set) and near-ties whose ordering is below fp32 evaluation-order
noise -- when the k-th and (k+1)-th neighbors are (near-)equidistant, numpy's
brute force and Postgres can rank them differently while both are correct.
A retrieved neighbor therefore counts as correct if its ground-truth distance
is within a small relative epsilon of the true k-th neighbor's distance (the
ann-benchmarks convention), instead of demanding id-set equality. Measured
failing case that forced this: wiki_dpr query with fp64 squared-distance
boundary gap 3.5e-6 relative -- under fp32 arithmetic that ordering is noise,
and the forced-seqscan baseline (which must score EXACTLY 1.0) flipped it.
"""
from __future__ import annotations

import numpy as np

from . import util

CACHE_DIR = util.datasets_cache_dir() / "groundtruth"

# Relative epsilon for tie-aware recall, applied to the (squared, for l2)
# ground-truth distance of the true k-th neighbor. Chosen to sit comfortably
# ABOVE fp32 evaluation-order divergence (~1e-5 relative for 768-term dot
# products; the observed baseline flips were at 3.5e-6 and 7.6e-6) and far
# BELOW any retrieval-quality-relevant distance difference (0.01% of the
# distance). A neighbor this close to the k-th neighbor is equivalent for
# every purpose this benchmark measures.
TIE_EPS_REL = 1e-4


def _distances(queries: np.ndarray, vectors: np.ndarray, metric: str) -> np.ndarray:
    """Return (n_queries, n_vectors) matrix where LOWER is closer, regardless of metric."""
    if metric == "l2":
        # ||q - v||^2 = ||q||^2 + ||v||^2 - 2 q.v ; monotonic in true L2 distance.
        q2 = np.sum(queries ** 2, axis=1, keepdims=True)
        v2 = np.sum(vectors ** 2, axis=1, keepdims=True).T
        return q2 + v2 - 2.0 * (queries @ vectors.T)
    elif metric == "cosine":
        qn = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        vn = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return -(qn @ vn.T)  # negative cosine similarity: lower = closer
    elif metric == "ip":
        return -(queries @ vectors.T)
    else:
        raise ValueError(f"unknown metric {metric!r}")


def brute_force_topk(queries: np.ndarray, vectors: np.ndarray, k: int, metric: str,
                      row_ids: np.ndarray | None = None,
                      cache_key: str | None = None,
                      query_batch: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k neighbors per query, batched to bound peak memory.

    Returns `(ids, dists)`, each (n_queries, k): the neighbor ids and their
    ground-truth distances (same lower-is-closer scale `_distances` produces;
    SQUARED distance for l2). Rows with fewer than k candidates are padded
    with id -1 / dist +inf. `dists` exists so recall can be tie-aware (see
    module docstring) -- the threshold for "counts as a top-k hit" is a
    distance, not an id set.

    row_ids: if given, maps local vector row index -> external id to report
    (used when `vectors` is already a filtered subset). Otherwise ids are
    just 0..n-1. Cache is one `<key>.npz` holding both arrays.
    """
    if cache_key:
        path = CACHE_DIR / f"{cache_key}.npz"
        if path.exists():
            d = np.load(path)
            return d["ids"], d["dists"]

    n_q = queries.shape[0]
    k_eff = min(k, vectors.shape[0])
    out_ids = np.full((n_q, k), -1, dtype=np.int64)
    out_dists = np.full((n_q, k), np.inf, dtype=np.float64)
    for start in range(0, n_q, query_batch):
        end = min(start + query_batch, n_q)
        d = _distances(queries[start:end], vectors, metric)
        part = np.argpartition(d, k_eff - 1, axis=1)[:, :k_eff]
        rows = np.arange(end - start)[:, None]
        order = np.argsort(d[rows, part], axis=1)
        idx = part[rows, order]
        out_ids[start:end, :k_eff] = idx if row_ids is None else row_ids[idx]
        out_dists[start:end, :k_eff] = d[rows, idx]

    if cache_key:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE_DIR / f"{cache_key}.npz", ids=out_ids, dists=out_dists)
    return out_ids, out_dists


def recall_at_k(retrieved: list[int], truth_ids_row: np.ndarray, k: int,
                truth_dists_row: np.ndarray | None = None,
                eps_rel: float = TIE_EPS_REL) -> float:
    """Tie-aware recall@k: fraction of the top-k retrieved ids that are true
    top-k neighbors, where "true top-k" includes every ground-truth neighbor
    whose distance is within `eps_rel` (relative) of the true k-th neighbor's
    distance -- (near-)ties at the k boundary count for either ordering, since
    the metric cannot distinguish them (see module docstring).

    `truth_ids_row`/`truth_dists_row` are one query's rows from
    `brute_force_topk` (computed at k' >= k, typically 100, so boundary ties
    up to k'-deep are visible). If `truth_dists_row` is None (e.g. SIFT1M's
    corpus-provided ids-only ground truth), falls back to strict id-set
    recall -- the pre-tie-aware behavior.
    """
    if truth_dists_row is None:
        truth_set = set(int(x) for x in truth_ids_row[:k] if x >= 0)
        if not truth_set:
            return float("nan")
        got = set(retrieved[:k])
        return len(got & truth_set) / len(truth_set)

    valid = truth_ids_row >= 0
    ids = truth_ids_row[valid]
    dists = truth_dists_row[valid]
    if len(ids) == 0:
        return float("nan")
    kk = min(k, len(ids))
    t = float(dists[kk - 1])
    # Works for negative distance scales too (cosine/ip store negated
    # similarity): the allowance is always toward "slightly farther".
    threshold = t + eps_rel * abs(t)
    accept = set(int(i) for i, dv in zip(ids, dists) if dv <= threshold)
    got = set(retrieved[:k])
    return len(got & accept) / kk
