#!/usr/bin/env python3
"""wiki_dpr shards -> the real-embedding artifacts the harness reads (harness/datasets.py:real_dataset).

1M (default): the four DPR shards wiki_passages_0..3 (1,681,224 x 768 fp32, corpus order); the first
1,000,000 are the base set; from the remaining 681,224 a seed-42 shuffle takes 500 measured queries,
160 warmup vectors and a 100,000-vector query bank (all held out). Recipe as recorded in the header of
configs/baselines_exact_real.yaml. Reconstructed from that record: the original converter was never
committed. BASE_SHA is the base_content_sha the 2026-09 runs stamped; a different value means the base
set is not the one those runs used.

10M (--scale 10m): the same corpus, shards in order, stopping at the first shard that covers a
10,000,000 base plus that same holdout (25 shards, wiki_passages_0..24, about 10.51M vectors).
One shard is resident at a time. The base is written as a sibling .vectors.npy the harness memory-maps;
the npz holds only the holdout and meta. The first 1,000,000 rows must hash to BASE_SHA.

usage: convert_wikidpr.py --shards DIR --out FILE [--scale 1m|10m]
"""
import argparse, gc, hashlib, json, os, pickle, sys
from pathlib import Path
import numpy as np
from numpy.lib.format import open_memmap

URL = "https://dl.fbaipublicfiles.com/dpr/data/wiki_encoded/single/nq"
URLS_1M = [f"{URL}/wiki_passages_{i}" for i in range(4)]
N_BASE, N_QUERIES, N_WARMUP, N_BANK, SEED, DIM = 1_000_000, 500, 160, 100_000, 42, 768
N_BASE_10M = 10_000_000
BASE_SHA = "dd93ee85811b7494"
SOURCE = "facebook/wiki_dpr (DPR-encoded English Wikipedia passages, single-nq encoder)"


def load_shard(path):
    with open(path, "rb") as f:
        rows = pickle.load(f)                       # [(passage id as str, float32[768]), ...]
    vecs = np.stack([v for _, v in rows]).astype(np.float32, copy=False)
    assert vecs.shape[1] == DIM, vecs.shape
    return vecs


def list_shards(shards_dir):
    shards, i = [], 0
    while True:
        p = Path(shards_dir) / f"wiki_passages_{i}"
        if not p.exists():
            break
        shards.append(p)
        i += 1
    return shards


def take_holdout(rest, n_queries, n_warmup, n_bank, seed):
    need = n_queries + n_warmup + n_bank
    if len(rest) < need:
        sys.exit(f"holdout has {len(rest)} vectors, need {need}")
    perm = np.random.default_rng(seed).permutation(len(rest))
    queries = rest[perm[:n_queries]]
    warmup = rest[perm[n_queries:n_queries + n_warmup]]
    bank = rest[perm[n_queries + n_warmup:need]]
    return queries, warmup, bank


def content_sha(rows):
    """sha256 prefix of the raw fp32 bytes, matching hashlib of array.tobytes()."""
    h = hashlib.sha256()
    step = 65_536
    for s in range(0, len(rows), step):
        h.update(np.ascontiguousarray(rows[s:s + step]).tobytes())
    return h.hexdigest()[:16]


def convert_1m(shards_dir, out):
    parts = []
    for i in range(4):
        parts.append(load_shard(f"{shards_dir}/wiki_passages_{i}"))
        print(f"shard {i}: {parts[-1].shape}", flush=True)
        gc.collect()
    allv = np.concatenate(parts); del parts; gc.collect()
    print("all:", allv.shape, flush=True)
    base, rest = allv[:N_BASE], allv[N_BASE:]
    queries, warmup, bank = take_holdout(rest, N_QUERIES, N_WARMUP, N_BANK, SEED)
    sha = hashlib.sha256(np.ascontiguousarray(base).tobytes()).hexdigest()[:16]
    print(f"base_content_sha {sha} ({'matches' if sha == BASE_SHA else 'DIFFERS FROM'} the 2026-09 runs: {BASE_SHA})", flush=True)
    meta = dict(source=SOURCE, shard_urls=URLS_1M,
                n_source_vectors=int(len(allv)), dim=DIM, dtype="float32", n_base=N_BASE, n_queries=N_QUERIES,
                n_warmup=N_WARMUP, n_bank=N_BANK, holdout_seed=SEED, base_content_sha=sha)
    np.savez(out, vectors=base, queries=queries, warmup_pool=warmup, query_bank=bank, meta_json=np.array(json.dumps(meta)))
    print("wrote", out)


def convert_10m(shards_dir, out, n_base=N_BASE_10M, n_queries=N_QUERIES, n_warmup=N_WARMUP,
                n_bank=N_BANK, seed=SEED, check_prefix_sha=BASE_SHA, prefix_rows=N_BASE):
    """Stream shards into a base .vectors.npy plus an npz of the holdout.

    `check_prefix_sha` compares the first `prefix_rows` of the base to the 1M artifact.
    Pass None to skip. A failed check deletes the vectors file.
    """
    out = Path(out)
    vectors_path = out.with_name(out.stem + ".vectors.npy")
    tmp_npz = out.with_name(out.stem + ".writing.npz")

    def die(msg):
        vectors_path.unlink(missing_ok=True)
        tmp_npz.unlink(missing_ok=True)
        sys.exit(msg)

    shards = list_shards(shards_dir)
    if not shards:
        die(f"no wiki_passages_* in {shards_dir}")
    need = n_base + n_queries + n_warmup + n_bank
    chosen, total = [], 0
    for p in shards:
        vecs = load_shard(p)
        n = len(vecs)
        del vecs
        gc.collect()
        chosen.append(p)
        total += n
        print(f"{p.name}: {n} rows (running {total})", flush=True)
        if total >= need:
            break
    else:
        die(f"shards total {total} vectors, need {need} "
            f"({n_base} base + {n_queries + n_warmup + n_bank} holdout). "
            f"Download further wiki_passages_* (0..49 exist).")

    mm = open_memmap(vectors_path, mode="w+", dtype=np.float32, shape=(n_base, DIM))
    try:
        offset = 0
        rest_parts = []
        for p in chosen:
            vecs = load_shard(p)
            if offset < n_base:
                take = min(len(vecs), n_base - offset)
                mm[offset:offset + take] = vecs[:take]
                offset += take
                if take < len(vecs):
                    rest_parts.append(np.ascontiguousarray(vecs[take:]))
            else:
                rest_parts.append(np.ascontiguousarray(vecs))
            del vecs
            gc.collect()
        if offset != n_base:
            die(f"wrote {offset} base rows, expected {n_base}")
        mm.flush()
        rest = np.concatenate(rest_parts)
        del rest_parts
        gc.collect()
        queries, warmup, bank = take_holdout(rest, n_queries, n_warmup, n_bank, seed)
        del rest
        gc.collect()
        prefix_sha = None
        if n_base >= prefix_rows:
            prefix_sha = content_sha(mm[:prefix_rows])
            if check_prefix_sha is not None and prefix_sha != check_prefix_sha:
                die(f"first {prefix_rows} vectors sha {prefix_sha} != {check_prefix_sha} "
                    f"(the 1M wiki_dpr base). Shards are not the DPR corpus in order.")
        sha = content_sha(mm)
        print(f"base_content_sha {sha} prefix_1m_sha {prefix_sha}", flush=True)
        meta = dict(source=SOURCE,
                    shard_urls=[f"{URL}/{p.name}" for p in chosen],
                    n_source_vectors=int(total), n_shards=len(chosen),
                    dim=DIM, dtype="float32", n_base=n_base,
                    n_queries=n_queries, n_warmup=n_warmup, n_bank=n_bank,
                    holdout_seed=seed, base_content_sha=sha,
                    prefix_1m_sha=prefix_sha,
                    vectors_file=vectors_path.name)
        np.savez(tmp_npz, queries=queries, warmup_pool=warmup, query_bank=bank,
                 meta_json=np.array(json.dumps(meta)))
        os.replace(tmp_npz, out)
    except SystemExit:
        vectors_path.unlink(missing_ok=True)
        tmp_npz.unlink(missing_ok=True)
        raise
    except Exception:
        vectors_path.unlink(missing_ok=True)
        tmp_npz.unlink(missing_ok=True)
        raise
    finally:
        del mm
    print("wrote", out, "and", vectors_path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", choices=("1m", "10m"), default="1m")
    ap.add_argument("--check-prefix", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args(argv)
    if a.scale == "1m":
        convert_1m(a.shards, a.out)
        return 0
    convert_10m(a.shards, a.out, check_prefix_sha=BASE_SHA if a.check_prefix else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
