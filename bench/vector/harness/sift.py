"""SIFT1M (ann-benchmarks classic corpus) loader: real-data sanity check.

Format: http://corpus-texmex.irisa.fr/ .fvecs/.ivecs -- each vector is
prefixed by a little-endian int32 dimension, then that many float32s (fvecs)
or int32s (ivecs). Downloads and caches ~168MB tar.gz; skips gracefully
(returns None) if the download fails for any reason, per the harness's
"skip gracefully" requirement rather than hard-failing a whole run.
"""
from __future__ import annotations

import logging
import tarfile
import urllib.request
from dataclasses import dataclass

import numpy as np

from . import util

CACHE_DIR = util.datasets_cache_dir()
SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
SIFT_TARBALL = CACHE_DIR / "sift.tar.gz"
SIFT_NPZ = CACHE_DIR / "sift1m.npz"

log = logging.getLogger(__name__)


@dataclass
class SiftDataset:
    key: str
    vectors: np.ndarray   # (1_000_000, 128) float32
    queries: np.ndarray   # (10_000, 128) float32 -- we subsample 500 in the runner
    groundtruth: np.ndarray  # (10_000, 100) int32, provided by the corpus
    meta: dict


def _read_fvecs(fh, dtype) -> np.ndarray:
    dim = np.frombuffer(fh.read(4), dtype=np.int32)[0]
    fh.seek(0)
    raw = np.frombuffer(fh.read(), dtype=np.int32)
    row_len = dim + 1
    raw = raw.reshape(-1, row_len)
    assert (raw[:, 0] == dim).all(), "corrupt fvecs/ivecs: inconsistent dim header"
    return raw[:, 1:].copy().view(dtype)


def try_load_sift1m() -> SiftDataset | None:
    """Best-effort load; returns None (with a logged reason) instead of raising."""
    if SIFT_NPZ.exists():
        d = np.load(SIFT_NPZ)
        return SiftDataset(key="sift1m", vectors=d["vectors"], queries=d["queries"],
                            groundtruth=d["groundtruth"], meta=dict(kind="sift1m"))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not SIFT_TARBALL.exists():
        try:
            log.info("downloading SIFT1M from %s ...", SIFT_URL)
            urllib.request.urlretrieve(SIFT_URL, SIFT_TARBALL)
        except Exception as e:
            log.warning("SIFT1M download failed (%s); skipping the real-data sanity set.", e)
            return None

    try:
        with tarfile.open(SIFT_TARBALL) as tf:
            def member(name):
                f = tf.extractfile(f"sift/{name}")
                if f is None:
                    raise RuntimeError(f"missing member sift/{name} in tarball")
                return f

            vectors = _read_fvecs(member("sift_base.fvecs"), np.float32)
            queries = _read_fvecs(member("sift_query.fvecs"), np.float32)
            groundtruth = _read_fvecs(member("sift_groundtruth.ivecs"), np.int32)
    except Exception as e:
        log.warning("SIFT1M extract/parse failed (%s); skipping the real-data sanity set.", e)
        return None

    np.savez(SIFT_NPZ, vectors=vectors, queries=queries, groundtruth=groundtruth)
    SIFT_TARBALL.unlink(missing_ok=True)
    return SiftDataset(key="sift1m", vectors=vectors, queries=queries,
                        groundtruth=groundtruth, meta=dict(kind="sift1m"))
