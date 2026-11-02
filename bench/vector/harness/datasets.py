"""Synthetic clustered dataset generation + metadata filter assignment.

Deliberately NOT uniform random: real embedding corpora cluster (topics,
products, users...), and clustering is exactly what makes ANN indexes'
recall/latency tradeoffs interesting. Uniform random in high dimension is
close to worst-case-uniform-distance and unrealistically easy for graph
indexes to route through.

Calibration note: naive tight clusters (small sigma
relative to center spacing) are ALSO unrealistically easy -- in 768d,
independent Gaussian noise around a cluster center is nearly orthogonal to
the direction toward any other center, so it doesn't "reach" neighboring
clusters the way low-dimensional intuition suggests (curse of
dimensionality). Recall only becomes genuinely knob-sensitive once cluster
sigma is a substantial fraction of typical inter-center spacing. We calibrate
empirically (see `geometry_diagnostics` + the recall curve this produces
against real pgvector, checked in the smoke run) rather than trust a formula.
Mixed per-cluster scales and a small uniform background fraction add
realistic heterogeneity on top (some topics/products are tighter than
others; a little unclustered noise is always present in real corpora).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import util

CACHE_DIR = util.datasets_cache_dir()


@dataclass
class Dataset:
    key: str
    vectors: np.ndarray          # (n, dim) float32, base set
    queries: np.ndarray          # (n_queries, dim) float32, held out, MEASURED pool
    warmup_pool: np.ndarray      # (n_warmup_pool, dim) float32, held out, never measured/recalled
    cluster_ids: np.ndarray      # (n,) int32, base set cluster assignment
    query_cluster_ids: np.ndarray  # (n_queries,) int32
    centers: np.ndarray          # (n_centers, dim) float32 -- generative centers (synthetic)
                                 #   or k-means centroids (real)
    cluster_sigmas: np.ndarray   # (n_centers,) float32 -- per-cluster sigma used (synthetic)
                                 #   or empirical RMS-dist-to-centroid / sqrt(dim) (real)
    meta: dict
    # Real corpora only (see real_dataset): a large held-out pool of REAL
    # vectors for in-region ("tenant") query sampling, with each bank vector's
    # nearest-centroid assignment. None for synthetic datasets, which draw
    # in-region queries from the generative model instead.
    query_bank: np.ndarray | None = None
    bank_cluster_ids: np.ndarray | None = None


def _gen_gmm(n: int, dim: int, n_centers: int, cluster_sigma: float, seed: int,
             rng: np.random.Generator | None = None,
             cluster_scale_mix: list[float] | None = None,
             background_frac: float = 0.0):
    """Gaussian-mixture point cloud: n_centers centers on a sphere-ish scale,
    points drawn N(center, (cluster_sigma * mult_c)^2 I) where mult_c is a
    per-cluster scale multiplier (heterogeneous cluster tightness) drawn from
    `cluster_scale_mix` (uniform mix over the given multipliers) if given,
    else 1.0 for every cluster (homogeneous, the old behavior).

    `background_frac`: this fraction of the n points is overwritten with
    points drawn uniformly from the bounding box of the generated cloud --
    unclustered "noise" rows that don't belong to any topic, same as real
    corpora always have some. Their cluster_id bookkeeping is left as-is (for
    filter-assignment purposes) even though their position is not actually
    generated from that cluster.

    Returns (points, cluster_ids, centers, cluster_sigmas).
    """
    rng = rng or np.random.default_rng(seed)
    # Centers spread with unit-scale coordinates so sigma is meaningful relative
    # to typical inter-center distance (~sqrt(2*dim) apart in expectation).
    centers = rng.normal(loc=0.0, scale=1.0, size=(n_centers, dim)).astype(np.float32)
    if cluster_scale_mix:
        mults = rng.choice(np.asarray(cluster_scale_mix, dtype=np.float64), size=n_centers)
    else:
        mults = np.ones(n_centers, dtype=np.float64)
    cluster_sigmas = (cluster_sigma * mults).astype(np.float32)

    cluster_ids = rng.integers(0, n_centers, size=n, dtype=np.int64)
    per_row_sigma = cluster_sigmas[cluster_ids]
    noise = rng.normal(loc=0.0, scale=1.0, size=(n, dim)).astype(np.float32) * per_row_sigma[:, None]
    points = centers[cluster_ids] + noise

    if background_frac > 0:
        n_bg = int(round(background_frac * n))
        if n_bg > 0:
            bg_idx = rng.choice(n, size=n_bg, replace=False)
            lo, hi = points.min(axis=0), points.max(axis=0)
            points[bg_idx] = rng.uniform(lo, hi, size=(n_bg, dim)).astype(np.float32)

    return points.astype(np.float32), cluster_ids.astype(np.int32), centers, cluster_sigmas


def _sample_gmm_queries(centers: np.ndarray, cluster_sigmas: np.ndarray, n: int,
                         rng: np.random.Generator, n_centers: int | None = None,
                         allowed_cluster_ids: np.ndarray | None = None):
    """Draw `n` fresh query vectors from the GMM (fresh noise draw, same
    centers/sigmas). If `allowed_cluster_ids` is given, cluster ids are drawn
    only from that subset (in-region/tenant-style queries); otherwise
    uniformly over 0..n_centers-1 (global/out-of-region queries)."""
    if allowed_cluster_ids is not None:
        q_cluster_ids = rng.choice(np.asarray(allowed_cluster_ids), size=n).astype(np.int64)
    else:
        q_cluster_ids = rng.integers(0, n_centers, size=n, dtype=np.int64)
    q_sigma = cluster_sigmas[q_cluster_ids]
    q_noise = rng.normal(loc=0.0, scale=1.0, size=(n, centers.shape[1])).astype(np.float32) * q_sigma[:, None]
    queries = (centers[q_cluster_ids] + q_noise).astype(np.float32)
    return queries, q_cluster_ids.astype(np.int32)


def synthetic_dataset(n: int, dim: int, n_centers: int, cluster_sigma: float,
                       n_queries: int, n_warmup_pool: int, seed: int,
                       cluster_scale_mix: list[float] | None = None,
                       background_frac: float = 0.0) -> Dataset:
    """Generate (or load from cache) a synthetic clustered dataset + held-out
    measured queries + a held-out warmup pool.

    `n_warmup_pool` extra query vectors are generated alongside the
    `n_queries` measured ones so that at every concurrency level the *same*
    `n_queries`-sized measured set is used --
    warmup queries are drawn from a disjoint pool, never counted in recall or
    QPS, and never subtracted from the measured pool.
    """
    cfg = dict(kind="synthetic", n=n, dim=dim, n_centers=n_centers,
               cluster_sigma=cluster_sigma, n_queries=n_queries,
               n_warmup_pool=n_warmup_pool, seed=seed,
               cluster_scale_mix=cluster_scale_mix, background_frac=background_frac)
    key = f"synthetic_{util.stable_hash(cfg)}"
    path = CACHE_DIR / f"{key}.npz"
    if path.exists():
        d = np.load(path)
        return Dataset(key=key, vectors=d["vectors"], queries=d["queries"],
                        warmup_pool=d["warmup_pool"], cluster_ids=d["cluster_ids"],
                        query_cluster_ids=d["query_cluster_ids"], centers=d["centers"],
                        cluster_sigmas=d["cluster_sigmas"], meta=cfg)

    rng = np.random.default_rng(seed)
    vectors, cluster_ids, centers, cluster_sigmas = _gen_gmm(
        n, dim, n_centers, cluster_sigma, seed, rng,
        cluster_scale_mix=cluster_scale_mix, background_frac=background_frac)

    total_q = n_queries + n_warmup_pool
    all_queries, all_q_cluster_ids = _sample_gmm_queries(
        centers, cluster_sigmas, total_q, rng, n_centers=n_centers)
    queries, warmup_pool = all_queries[:n_queries], all_queries[n_queries:]
    query_cluster_ids = all_q_cluster_ids[:n_queries]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, vectors=vectors, queries=queries, warmup_pool=warmup_pool,
              cluster_ids=cluster_ids, query_cluster_ids=query_cluster_ids,
              centers=centers, cluster_sigmas=cluster_sigmas)
    return Dataset(key=key, vectors=vectors, queries=queries, warmup_pool=warmup_pool,
                    cluster_ids=cluster_ids, query_cluster_ids=query_cluster_ids,
                    centers=centers, cluster_sigmas=cluster_sigmas, meta=cfg)


def sample_region_queries(dataset: Dataset, allowed_cluster_ids: np.ndarray, n: int,
                           n_warmup_pool: int, seed: int):
    """In-region ('tenant') query workload for a correlated filter: queries
    are drawn from the same GMM but restricted to the filter's qualifying
    clusters, so a query's own distribution matches the region it filters
    into -- the opposite regime from the default global/out-of-region
    queries. Returns (measured_queries, warmup_pool).
    """
    rng = np.random.default_rng(seed)
    total_q = n + n_warmup_pool
    all_queries, _ = _sample_gmm_queries(dataset.centers, dataset.cluster_sigmas, total_q, rng,
                                          allowed_cluster_ids=allowed_cluster_ids)
    return all_queries[:n], all_queries[n:]


# --- Real-embedding corpus support ------------------------------
#
# A "real" dataset is a converted artifact in datasets_cache/ (npz with fp32
# arrays: vectors, queries, warmup_pool, query_bank, plus meta_json
# provenance) and a k-means clustering computed here (cached alongside) that
# plays the role the GMM generative cluster ids play for synthetic data.
# A 10M wiki_dpr npz omits `vectors` and stores them in a sibling
# `<stem>.vectors.npy`, which is memory-mapped so the 31 GB matrix is not
# pinned next to Postgres (that pairing OOM-killed the 10M Cohere run).
# Filter columns derive from cluster membership exactly as before (correlated
# = whole clusters until target selectivity via correlated_chosen_clusters;
# uncorrelated = random), and in-region ("tenant") queries are sampled from
# the held-out query_bank restricted to qualifying clusters -- real vectors,
# not synthetic draws, so the query geometry stays real.


def _assign_to_centroids(x: np.ndarray, centroids: np.ndarray, chunk: int = 100_000) -> np.ndarray:
    """Nearest-centroid assignment, chunked to bound peak memory."""
    c2 = (centroids.astype(np.float64) ** 2).sum(axis=1)
    out = np.empty(len(x), dtype=np.int32)
    for s in range(0, len(x), chunk):
        block = x[s:s + chunk]
        out[s:s + chunk] = np.argmin(c2[None, :] - 2.0 * (block @ centroids.T), axis=1)
    return out


def _kmeans(vectors: np.ndarray, k: int, sample: int, iters: int, seed: int):
    """Plain Lloyd k-means on a random sample -- deliberately numpy-only (no
    sklearn dependency) and deterministic for a given seed. Empty clusters
    keep their stale centroid (harmless: zero population, so the correlated
    filter selection just never picks them). Cluster quality only needs to
    yield spatially coherent regions -- the correlated filter is DEFINED by
    these clusters, not judged against hidden truth. Returns (centroids,
    assignments-for-`vectors`).
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(vectors.shape[0], size=min(sample, vectors.shape[0]), replace=False)
    pts = vectors[idx].astype(np.float32)
    centroids = pts[rng.choice(len(pts), size=k, replace=False)].copy()
    for _ in range(iters):
        a = _assign_to_centroids(pts, centroids)
        for c in range(k):
            members = pts[a == c]
            if len(members):
                centroids[c] = members.mean(axis=0)
    return centroids.astype(np.float32), _assign_to_centroids(vectors, centroids)


def real_dataset(artifact: str, n: int, dim: int, n_centers: int, n_queries: int,
                 n_warmup_pool: int, seed: int, kmeans_sample: int = 100_000,
                 kmeans_iters: int = 15) -> Dataset:
    """Load a converted real-embedding artifact + its (cached) k-means
    clustering. `centers`/`cluster_sigmas` are the k-means centroids and each
    cluster's empirical sigma (RMS distance to centroid / sqrt(dim)) so
    geometry_diagnostics reports numbers comparable to the synthetic path.
    The dataset key fingerprints the artifact's content hash (from the
    converter's meta) plus every clustering parameter -- ground-truth caches
    keyed on it invalidate if either the data or the clustering changes.
    """
    path = CACHE_DIR / artifact
    d = np.load(path, allow_pickle=False)
    import json as _json
    meta_src = _json.loads(str(d["meta_json"]))

    if "vectors" in d.files:
        vectors = d["vectors"][:n]
    else:
        # 10M wiki_dpr: the base matrix is a sidecar .npy (see convert_wikidpr.py --scale 10m).
        sidecar = path.parent / f"{path.stem}.vectors.npy"
        if not sidecar.is_file():
            raise ValueError(f"artifact {artifact} has no vectors array and {sidecar.name} is not beside it")
        vectors = np.load(sidecar, mmap_mode="r")[:n]
    if vectors.shape != (n, dim):
        raise ValueError(f"artifact {artifact}: vectors {vectors.shape}, config wants {(n, dim)}")
    if len(d["queries"]) < n_queries:
        raise ValueError(f"artifact holds {len(d['queries'])} queries, config wants {n_queries}")
    if len(d["warmup_pool"]) < n_warmup_pool:
        raise ValueError(f"artifact holds {len(d['warmup_pool'])} warmup vectors, config "
                         f"wants {n_warmup_pool} (= max(concurrency) * warmup_queries)")
    queries = d["queries"][:n_queries]
    warmup_pool = d["warmup_pool"][:n_warmup_pool]
    bank = d["query_bank"]

    key_cfg = dict(kind="real", artifact=artifact, content=meta_src.get("base_content_sha"),
                   n=n, dim=dim, n_centers=n_centers, n_queries=n_queries,
                   n_warmup_pool=n_warmup_pool, seed=seed,
                   kmeans_sample=kmeans_sample, kmeans_iters=kmeans_iters)
    key = f"real_{util.stable_hash(key_cfg)}"

    km_path = CACHE_DIR / f"{key}_kmeans.npz"
    if km_path.exists():
        km = np.load(km_path)
        centroids, cluster_ids = km["centroids"], km["cluster_ids"]
        query_cluster_ids, bank_cluster_ids = km["query_cluster_ids"], km["bank_cluster_ids"]
        cluster_sigmas = km["cluster_sigmas"]
    else:
        centroids, cluster_ids = _kmeans(vectors, n_centers, kmeans_sample, kmeans_iters, seed)
        query_cluster_ids = _assign_to_centroids(queries, centroids)
        bank_cluster_ids = _assign_to_centroids(bank, centroids)
        # Per-cluster empirical sigma from a bounded sample, so
        # geometry_diagnostics has real numbers to report.
        rng = np.random.default_rng(seed)
        s_idx = rng.choice(n, size=min(200_000, n), replace=False)
        s_pts, s_ids = vectors[s_idx], cluster_ids[s_idx]
        dists = np.linalg.norm(s_pts.astype(np.float32) - centroids[s_ids], axis=1)
        cluster_sigmas = np.zeros(n_centers, dtype=np.float32)
        for c in range(n_centers):
            mask = s_ids == c
            if mask.any():
                cluster_sigmas[c] = float(np.sqrt(np.mean(dists[mask] ** 2)) / np.sqrt(dim))
        np.savez(km_path, centroids=centroids, cluster_ids=cluster_ids,
                 query_cluster_ids=query_cluster_ids, bank_cluster_ids=bank_cluster_ids,
                 cluster_sigmas=cluster_sigmas)

    return Dataset(key=key, vectors=vectors, queries=queries, warmup_pool=warmup_pool,
                   cluster_ids=cluster_ids.astype(np.int32),
                   query_cluster_ids=query_cluster_ids.astype(np.int32),
                   centers=centroids, cluster_sigmas=cluster_sigmas,
                   meta=dict(key_cfg, source_meta=meta_src),
                   query_bank=bank, bank_cluster_ids=bank_cluster_ids.astype(np.int32))


def sample_region_queries_real(dataset: Dataset, allowed_cluster_ids: np.ndarray, n: int,
                                n_warmup_pool: int, seed: int):
    """In-region ('tenant') queries for a REAL corpus: drawn from the held-out
    query_bank restricted to the filter's qualifying k-means clusters -- real
    vectors whose own nearest centroid lies inside the qualifying region, the
    real-data analog of the synthetic path's same-cluster GMM draws.

    A finite bank may not hold n + n_warmup_pool DISTINCT in-region vectors at
    thin selectivities; in that case we sample WITH REPLACEMENT and say so:
    returns (measured, warmup, info) where info records n_unique_available vs
    n_requested and a sampled_with_replacement flag. Latency and recall stay
    meaningful per execution; what shrinks is query diversity -- callers
    persist `info` in the results JSON so a reader can never mistake the pool
    for n distinct queries.
    """
    rng = np.random.default_rng(seed)
    allowed = np.zeros(len(dataset.centers), dtype=bool)
    allowed[np.asarray(allowed_cluster_ids)] = True
    in_region = np.nonzero(allowed[dataset.bank_cluster_ids])[0]
    total = n + n_warmup_pool
    info = dict(n_unique_available=int(len(in_region)), n_requested=int(total),
                sampled_with_replacement=bool(len(in_region) < total))
    if len(in_region) == 0:
        raise RuntimeError("query_bank holds no vectors in the qualifying region -- "
                           "bank too small for this selectivity")
    chosen = (rng.permutation(in_region)[:total] if len(in_region) >= total
              else rng.choice(in_region, size=total, replace=True))
    q = dataset.query_bank[chosen]
    return q[:n], q[n:], info


def cluster_impurity(vectors: np.ndarray, cluster_ids: np.ndarray, centers: np.ndarray,
                      sample: int = 2000, seed: int = 0) -> float:
    """Diagnostic: fraction of a sample of points whose *nearest* center is not
    their own generative center. In high dimension this stays near zero even
    with substantial sigma (independent noise is nearly orthogonal to the
    direction toward any other center) -- it's reported for transparency, not
    used as the primary calibration signal (see `geometry_diagnostics` and
    the module docstring)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vectors), size=min(sample, len(vectors)), replace=False)
    pts = vectors[idx]
    true_ids = cluster_ids[idx]
    q2 = np.sum(pts ** 2, axis=1, keepdims=True)
    v2 = np.sum(centers ** 2, axis=1, keepdims=True).T
    d2 = q2 + v2 - 2.0 * (pts @ centers.T)
    nearest = np.argmin(d2, axis=1)
    return float(np.mean(nearest != true_ids))


def geometry_diagnostics(centers: np.ndarray, cluster_sigmas: np.ndarray, dim: int,
                          sample: int = 300, seed: int = 0) -> dict:
    """Reports the inter/intra-cluster distance ratio the task asks us to
    verify: mean pairwise distance among a sample of centers, vs. mean
    expected intra-cluster radius (sigma * sqrt(dim)). A ratio well under 1
    with a genuinely knob-sensitive recall curve (checked empirically, see
    `run.hardness_check`) is the real evidence; this number is corroborating,
    not sufficient on its own (see module docstring on the curse of
    dimensionality)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(centers), size=min(sample, len(centers)), replace=False)
    sub = centers[idx]
    d = np.sqrt(np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.nan)
    mean_inter = float(np.nanmean(d))
    mean_intra_radius = float(np.mean(cluster_sigmas) * np.sqrt(dim))
    return dict(mean_inter_center_dist=mean_inter, mean_intra_cluster_radius=mean_intra_radius,
                ratio=(mean_intra_radius / mean_inter) if mean_inter else float("nan"))


def correlated_chosen_clusters(cluster_ids: np.ndarray, n_centers: int, selectivity: float,
                                seed: int) -> np.ndarray:
    """Deterministic boolean-per-cluster mask for mode="correlated": chooses a
    subset of clusters whose combined population share ~= selectivity. Split
    out of `assign_filter_values` so churn can apply the *same* chosen-cluster
    set to freshly reinserted rows instead of
    re-deriving an inconsistent one.
    """
    derived_seed = int(util.stable_hash(dict(seed=seed, sel=selectivity, mode="correlated")), 16) % (2**32)
    rng = np.random.default_rng(derived_seed)
    counts = np.bincount(cluster_ids, minlength=n_centers)
    order = rng.permutation(n_centers)
    target = selectivity * len(cluster_ids)
    chosen = np.zeros(n_centers, dtype=bool)
    running = 0
    for c in order:
        if running >= target:
            break
        chosen[c] = True
        running += counts[c]
    return chosen


def assign_filter_values(cluster_ids: np.ndarray, n_centers: int, selectivity: float,
                          mode: str, seed: int) -> np.ndarray:
    """Assign an integer metadata `tag` column value per row for a WHERE-filter workload.

    mode="correlated": tag is a (deterministic) function of cluster_id, so the
      qualifying rows cluster together in vector space -- the hard case for
      graph/IVF indexes, which route by proximity and can strand a whole
      subtree outside the filter.
    mode="uncorrelated": tag assigned uniformly at random, independent of
      cluster_id -- qualifying rows are spread evenly through the index.

    Selectivity is approximate (multiple tag values can map to "qualifies");
    we report the *realized* selectivity alongside every result rather than
    trusting the target number.
    """
    n = len(cluster_ids)
    if mode == "uncorrelated":
        derived_seed = int(util.stable_hash(dict(seed=seed, sel=selectivity, mode=mode)), 16) % (2**32)
        rng = np.random.default_rng(derived_seed)
        qualifies = rng.random(n) < selectivity
    elif mode == "correlated":
        chosen = correlated_chosen_clusters(cluster_ids, n_centers, selectivity, seed)
        qualifies = chosen[cluster_ids]
    else:
        raise ValueError(f"unknown filter mode {mode!r}")
    return qualifies.astype(bool)


def assign_new_row_filter_values(new_cluster_ids: np.ndarray, n_centers: int, selectivity: float,
                                  mode: str, seed: int, rng: np.random.Generator,
                                  base_cluster_ids: np.ndarray | None = None) -> np.ndarray:
    """Same rule as `assign_filter_values`, applied to freshly-generated rows
    (churn reinserts) rather than the original dataset -- used so reinserted
    rows get their filter columns populated consistently with the rest of the
    table instead of left NULL.

    mode="correlated" reuses the exact chosen-cluster set derived from the
    *original* dataset's cluster population (`base_cluster_ids`) so a
    reinserted row qualifies iff its cluster was already one of the chosen
    ones -- not re-derived from the (smaller, post-churn) live population.
    mode="uncorrelated" draws fresh independent coin flips (correct by
    construction: uncorrelated rows are independent of cluster_id and of each
    other, so a fresh draw is exactly as valid as any other).
    """
    if mode == "uncorrelated":
        return rng.random(len(new_cluster_ids)) < selectivity
    elif mode == "correlated":
        if base_cluster_ids is None:
            raise ValueError("mode='correlated' requires base_cluster_ids")
        chosen = correlated_chosen_clusters(base_cluster_ids, n_centers, selectivity, seed)
        return chosen[new_cluster_ids]
    else:
        raise ValueError(f"unknown filter mode {mode!r}")
