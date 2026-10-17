"""Small shared helpers: free ports, hashing, RSS sampling, timing."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import socket
import subprocess
import threading
import time
from pathlib import Path


def datasets_cache_dir() -> Path:
    """Dataset and ground-truth cache: $PG_PINECONE_BENCH_DATA/vector/datasets_cache
    (default bench/.data/vector/datasets_cache; gitignored)."""
    root = os.environ.get("PG_PINECONE_BENCH_DATA") or str(Path(__file__).resolve().parents[2] / ".data")
    return Path(root) / "vector" / "datasets_cache"


def _archive_sha(cwd: str | None) -> str | None:
    """A box copy is a `git archive` of a committed tree with its commit in HARNESS_SHA (box/upload.sh)."""
    f = Path(cwd or ".") / "HARNESS_SHA"
    return f.read_text().strip() if f.exists() else None


def git_sha(cwd: str | None = None) -> str | None:
    """Short git SHA of the current checkout, for results provenance. None if
    not in a git repo or git isn't on PATH."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=cwd, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return _archive_sha(cwd)


def git_dirty(cwd: str | None = None) -> bool | None:
    """Whether `git status --porcelain` reports anything under `cwd`.

    None means "unknown" (not a repo, git missing) -- callers must never treat
    unknown as clean (a results file's "clean
    run of record" label has to be backed by an actual clean-tree check, not
    an absence of evidence).
    """
    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                              text=True, cwd=cwd, timeout=10)
        if out.returncode != 0:
            return False if _archive_sha(cwd) else None
        return bool(out.stdout.strip())
    except Exception:
        return False if _archive_sha(cwd) else None


def bench_content_hash(bench_dir) -> str | None:
    """Sha256 fingerprint over every git-tracked file under `bench_dir`,
    hashed from each file's CURRENT on-disk bytes (not the git blob) -- it
    changes the instant a tracked file is edited, even before staging.

    Complements `git_sha` (which commit) and `git_dirty` (whether the tree
    still matches that commit) with a reproducible fingerprint of the exact
    bytes a run actually executed against, independent of commit boundaries.
    """
    bench_dir = str(bench_dir)
    try:
        out = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                              cwd=bench_dir, timeout=10)
        if out.returncode != 0:
            if not _archive_sha(bench_dir):
                return None
            files = [str(p.relative_to(bench_dir)) for p in Path(bench_dir).rglob("*")
                     if p.is_file() and not (set(p.relative_to(bench_dir).parts) & {"results", "__pycache__", ".venv"})]
        else:
            files = out.stdout.splitlines()
        h = hashlib.sha256()
        for rel in sorted(files):
            path = Path(bench_dir) / rel
            try:
                data = path.read_bytes()
            except OSError:
                continue
            h.update(rel.encode())
            h.update(hashlib.sha256(data).digest())
        return h.hexdigest()[:16]
    except Exception:
        return None


def sanitize_for_json(obj):
    """Recursively replaces non-finite floats (NaN/+-Inf) with None.

    Python's `json` module happily emits the literal
    tokens `NaN`/`Infinity` by default, which is not valid per the JSON spec
    and chokes strict parsers in other languages. Every recall/latency number
    that can legitimately be "not applicable" already flows through here
    rather than through a bespoke per-field check.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def bytes_hash(b: bytes) -> str:
    """Deterministic short hash of raw bytes (e.g. a boolean mask), for cache keys
    that must invalidate whenever the actual data changes -- not just whenever a
    human-readable label describing it changes."""
    return hashlib.sha256(b).hexdigest()[:16]


def free_port() -> int:
    """Ask the OS for a free TCP port, then release it immediately.

    There's a tiny TOCTOU race (something else could grab it before we bind
    Postgres), but it's good enough for a throwaway local benchmark instance.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def stable_hash(obj) -> str:
    """Deterministic short hash of a JSON-serializable object, for cache keys."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@contextlib.contextmanager
def timed():
    """Context manager yielding a dict populated with {'seconds': float} on exit."""
    result: dict = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start


class RssSampler:
    """Polls RSS (bytes) of a given OS pid in a background thread.

    Used to record a peak-memory watermark for a Postgres backend during
    index builds. Approximate: on macOS we shell out to `ps` each tick, and we
    only track the leader backend, not parallel workers it may spawn for
    CREATE INDEX. Good enough as a directional signal, not a precise figure.
    """

    def __init__(self, pid: int, interval_s: float = 0.2):
        self.pid = pid
        self.interval_s = interval_s
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample_once(self) -> int:
        try:
            out = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(self.pid)],
                capture_output=True, text=True, timeout=2,
            )
            rss_kb = int(out.stdout.strip() or 0)
            return rss_kb * 1024
        except Exception:
            return 0

    def _run(self):
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._sample_once())
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self.peak_bytes = self._sample_once()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)


def percentile(sorted_values, p: float) -> float:
    """Nearest-rank percentile over an already-sorted sequence. p in [0, 100]."""
    if not sorted_values:
        return float("nan")
    n = len(sorted_values)
    idx = min(n - 1, max(0, int(round(p / 100.0 * (n - 1)))))
    return sorted_values[idx]
