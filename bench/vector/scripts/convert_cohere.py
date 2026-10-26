#!/usr/bin/env python3
"""Cohere wikipedia-22-12 HDF5 -> harness npz (cohere_768d_{1,10}m.npz).

Source: s3://vector-datasets/{1M,10M}/cohere-wikipedia-22-12-*-angular.hdf5, published by Tiger Data
(ann-benchmarks layout: train, test, neighbors, distances). No labels. HDF5 neighbors are unfiltered
and unused: filtered ground truth is recomputed by the harness.

Wiki_dpr had 681k extra passages for a 100k query bank. These files have 10k test
vectors only. Split of test, seed 42: 500 measured queries, 160 warmup, remainder
(~9340) query_bank. The Cohere configs pin query_modes to out_region, so the thin
bank is unused until in_region is turned on.

usage: convert_cohere.py --hdf5 FILE --out FILE [--scale 1m|10m]
"""
import argparse, hashlib, json, sys
import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("convert_cohere.py needs h5py (pip install h5py)")

N_QUERIES, N_WARMUP, SEED, DIM = 500, 160, 42, 768
# sha256 of the public HDF5 files.
SCALES = {
    "1m": dict(
        n_base=1_000_000,
        sha="d08003508b881ccd0d82e67429d2ff794ebc950eaca7fdb0d270b23c9a79ef95",
        uri="s3://vector-datasets/1M/cohere-wikipedia-22-12-1M-angular.hdf5",
        source="ann-benchmarks cohere-wikipedia-22-12-1M-angular (s3://vector-datasets)",
    ),
    "10m": dict(
        n_base=10_000_000,
        sha="b2891c56e31e0a6c8f3da63bc63e8735a3fb428989c615b98d5f9473f425df4d",
        uri="s3://vector-datasets/10M/cohere-wikipedia-22-12-10M-angular.hdf5",
        source="ann-benchmarks cohere-wikipedia-22-12-10M-angular (s3://vector-datasets)",
    ),
}


def file_sha256(path, buf=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", choices=sorted(SCALES), default="1m")
    a = ap.parse_args()
    spec = SCALES[a.scale]
    n_base, expect_sha, uri = spec["n_base"], spec["sha"], spec["uri"]

    sha = file_sha256(a.hdf5)
    if sha != expect_sha:
        sys.exit(f"hdf5 sha256 {sha} != expected {expect_sha}")
    print(f"hdf5 sha256 {sha} scale={a.scale}", flush=True)

    with h5py.File(a.hdf5, "r") as f:
        keys = list(f.keys())
        print(f"hdf5 datasets: {keys}", flush=True)
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)

    if train.shape != (n_base, DIM):
        sys.exit(f"train {train.shape}, expected {(n_base, DIM)}")
    if test.ndim != 2 or test.shape[1] != DIM:
        sys.exit(f"test {test.shape}, expected (N, {DIM})")
    if len(test) < N_QUERIES + N_WARMUP:
        sys.exit(f"test has {len(test)} rows, need {N_QUERIES + N_WARMUP} for queries+warmup")

    perm = np.random.default_rng(SEED).permutation(len(test))
    queries = test[perm[:N_QUERIES]]
    warmup = test[perm[N_QUERIES:N_QUERIES + N_WARMUP]]
    bank = test[perm[N_QUERIES + N_WARMUP:]]
    train = np.ascontiguousarray(train)
    base_sha = hashlib.sha256(train.data).hexdigest()[:16]
    print(f"base_content_sha {base_sha}; queries {queries.shape} warmup {warmup.shape} "
          f"bank {bank.shape}", flush=True)

    meta = dict(
        source=spec["source"],
        source_uri=uri,
        hdf5_sha256=sha,
        n_source_vectors=int(len(train)),
        n_source_test=int(len(test)),
        dim=DIM,
        dtype="float32",
        metric="angular",
        n_base=n_base,
        n_queries=N_QUERIES,
        n_warmup=N_WARMUP,
        n_bank=int(len(bank)),
        holdout_seed=SEED,
        base_content_sha=base_sha,
    )
    np.savez(a.out, vectors=train, queries=queries, warmup_pool=warmup, query_bank=bank,
             meta_json=np.array(json.dumps(meta)))
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
