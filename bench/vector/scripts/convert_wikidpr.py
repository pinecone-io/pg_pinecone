#!/usr/bin/env python3
"""wiki_dpr shards -> wikidpr_768d_1m.npz, the real-embedding artifact the harness reads (harness/datasets.py:real_dataset).

Recipe, as recorded in the header of configs/baselines_exact_real.yaml: the four DPR shards wiki_passages_0..3
(1,681,224 x 768 fp32, corpus order); the first 1,000,000 are the base set; from the remaining 681,224 a seed-42
shuffle takes 500 measured queries, 160 warmup vectors and a 100,000-vector query bank (all held out).
Reconstructed from that record: the original converter was never committed. BASE_SHA is the base_content_sha
the 2026-09 runs stamped; a different value means the base set is not the one those runs used. Checked 2026-10-12: the
base_content_sha, the harness's dataset key (real_fd937e5862449934) and the cluster ids of the 500 queries and the 100,000
bank vectors against the cached k-means of the 2026-09 runs all match; the 160 warmup vectors have no fingerprint.

usage: convert_wikidpr.py --shards DIR --out FILE
"""
import argparse, gc, hashlib, json, pickle, sys
import numpy as np

URLS = [f"https://dl.fbaipublicfiles.com/dpr/data/wiki_encoded/single/nq/wiki_passages_{i}" for i in range(4)]
N_BASE, N_QUERIES, N_WARMUP, N_BANK, SEED, DIM = 1_000_000, 500, 160, 100_000, 42, 768
BASE_SHA = "dd93ee85811b7494"


def load_shard(path):
    with open(path, "rb") as f:
        rows = pickle.load(f)                       # [(passage id as str, float32[768]), ...]
    vecs = np.stack([v for _, v in rows]).astype(np.float32, copy=False)
    assert vecs.shape[1] == DIM, vecs.shape
    return vecs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    parts = []
    for i in range(4):
        parts.append(load_shard(f"{a.shards}/wiki_passages_{i}"))
        print(f"shard {i}: {parts[-1].shape}", flush=True)
        gc.collect()
    allv = np.concatenate(parts); del parts; gc.collect()
    print("all:", allv.shape, flush=True)
    base, rest = allv[:N_BASE], allv[N_BASE:]
    perm = np.random.default_rng(SEED).permutation(len(rest))
    queries = rest[perm[:N_QUERIES]]
    warmup = rest[perm[N_QUERIES:N_QUERIES + N_WARMUP]]
    bank = rest[perm[N_QUERIES + N_WARMUP:N_QUERIES + N_WARMUP + N_BANK]]
    sha = hashlib.sha256(np.ascontiguousarray(base).tobytes()).hexdigest()[:16]
    print(f"base_content_sha {sha} ({'matches' if sha == BASE_SHA else 'DIFFERS FROM'} the 2026-09 runs: {BASE_SHA})", flush=True)
    meta = dict(source="facebook/wiki_dpr (DPR-encoded English Wikipedia passages, single-nq encoder)", shard_urls=URLS,
                n_source_vectors=int(len(allv)), dim=DIM, dtype="float32", n_base=N_BASE, n_queries=N_QUERIES,
                n_warmup=N_WARMUP, n_bank=N_BANK, holdout_seed=SEED, base_content_sha=sha)
    np.savez(a.out, vectors=base, queries=queries, warmup_pool=warmup, query_bank=bank, meta_json=np.array(json.dumps(meta)))
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
