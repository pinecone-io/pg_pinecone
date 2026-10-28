#!/usr/bin/env python3
"""Converter, sidecar load, and 10M ground-truth batching. No network, no 10M matrix.

Run from bench/vector:  .venv/bin/python scripts/test_wikidpr_10m.py
"""
import hashlib, importlib.util, json, pickle, sys, tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

spec = importlib.util.spec_from_file_location("convert_wikidpr", HERE / "convert_wikidpr.py")
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)

from harness import datasets as ds_mod
from harness import groundtruth as gt_mod


def shard(path, start, n, poison=False):
    x = np.zeros((n, conv.DIM), np.float32)
    x[:, 0] = -1 if poison else np.arange(start, start + n)
    with open(path, "wb") as f:
        pickle.dump([(str(i), row) for i, row in enumerate(x)], f)


def test_content_sha_matches_tobytes():
    a = np.random.default_rng(0).standard_normal((1000, conv.DIM)).astype(np.float32)
    got = conv.content_sha(a)
    want = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
    assert got == want, (got, want)


def test_squared_row_norms_match_one_shot():
    x = np.random.default_rng(1).standard_normal((3000, conv.DIM)).astype(np.float32)
    got = gt_mod._squared_row_norms(x)
    want = np.sum(x ** 2, axis=1)
    assert got.shape == want.shape and got.dtype == want.dtype
    assert np.array_equal(got, want)


def test_query_batch_cap_leaves_1m_alone():
    assert gt_mod._query_batch_for(1_000_000, 200) == 200
    capped = gt_mod._query_batch_for(10_000_000, 200)
    assert capped == 26
    assert capped * 10_000_000 * 8 <= 2 * 1024**3


def test_convert_streams_and_holds_out():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shards = root / "shards"
        shards.mkdir()
        shard(shards / "wiki_passages_0", 0, 100)
        shard(shards / "wiki_passages_1", 1000, 100)
        shard(shards / "wiki_passages_2", 0, 50, poison=True)
        out = root / "wikidpr_768d_10m.npz"
        conv.convert_10m(shards, out, n_base=150, n_queries=10, n_warmup=5, n_bank=20,
                         check_prefix_sha=None)
        npy = root / "wikidpr_768d_10m.vectors.npy"
        assert npy.is_file() and out.is_file()
        base = np.load(npy, mmap_mode="r")
        assert base.shape == (150, conv.DIM)
        assert base[0, 0] == 0 and base[100, 0] == 1000
        assert not np.any(base[:, 0] == -1)
        z = np.load(out, allow_pickle=False)
        meta = json.loads(str(z["meta_json"]))
        assert meta["n_shards"] == 2 and meta["n_source_vectors"] == 200
        assert meta["n_base"] == 150 and meta["base_content_sha"] == conv.content_sha(base)
        assert z["queries"].shape == (10, conv.DIM)
        assert z["warmup_pool"].shape == (5, conv.DIM)
        assert z["query_bank"].shape == (20, conv.DIM)
        assert not np.any(z["queries"][:, 0] == -1)
        held = np.concatenate([z["queries"][:, 0], z["warmup_pool"][:, 0], z["query_bank"][:, 0]])
        assert set(np.unique(held)).isdisjoint(set(np.unique(base[:, 0])))

        ds_mod.CACHE_DIR = root
        loaded = ds_mod.real_dataset("wikidpr_768d_10m.npz", n=150, dim=conv.DIM, n_centers=4,
                                     n_queries=10, n_warmup_pool=4, seed=42,
                                     kmeans_sample=50, kmeans_iters=2)
        assert isinstance(loaded.vectors, np.memmap)
        assert loaded.vectors.shape == (150, conv.DIM)
        assert loaded.queries.shape == (10, conv.DIM)
        assert loaded.meta["source_meta"]["base_content_sha"] == meta["base_content_sha"]

        inline = root / "inline.npz"
        np.savez(inline, vectors=np.array(base), queries=z["queries"], warmup_pool=z["warmup_pool"],
                 query_bank=z["query_bank"], meta_json=np.array(json.dumps(meta)))
        loaded_inline = ds_mod.real_dataset("inline.npz", n=150, dim=conv.DIM, n_centers=4,
                                            n_queries=10, n_warmup_pool=4, seed=42,
                                            kmeans_sample=50, kmeans_iters=2)
        assert not isinstance(loaded_inline.vectors, np.memmap)
        assert loaded_inline.vectors.shape == (150, conv.DIM)


def test_bad_prefix_deletes_the_matrix():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shards = root / "shards"
        shards.mkdir()
        shard(shards / "wiki_passages_0", 0, 200)
        out = root / "out.npz"
        try:
            conv.convert_10m(shards, out, n_base=150, n_queries=10, n_warmup=5, n_bank=20,
                             check_prefix_sha="0" * 16, prefix_rows=40)
        except SystemExit as e:
            assert "0" * 16 in str(e)
        else:
            raise AssertionError("prefix mismatch must fail")
        assert not (root / "out.vectors.npy").exists()
        assert not out.exists()


if __name__ == "__main__":
    test_content_sha_matches_tobytes()
    test_squared_row_norms_match_one_shot()
    test_query_batch_cap_leaves_1m_alone()
    test_convert_streams_and_holds_out()
    test_bad_prefix_deletes_the_matrix()
    print("ok")
