"""YAML config loading. Deliberately just validated dicts, not a schema
framework -- one entrypoint, few configs, not worth the abstraction weight."""
from __future__ import annotations

from pathlib import Path

import yaml

REQUIRED_TOP_KEYS = {"name", "seed", "dataset", "metric", "systems"}


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = REQUIRED_TOP_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"config {path} missing required keys: {missing}")
    cfg.setdefault("filters", {"run": []})
    cfg.setdefault("concurrency", [1])
    cfg.setdefault("churn", {"enabled": False})
    cfg.setdefault("warmup_queries", 10)
    cfg.setdefault("k", 10)
    # Single source of truth for the hardness gate's threshold:
    # runner.py enforces this exact value, report.py displays
    # it -- "reported == enforced" only holds if both read the same cfg key.
    cfg.setdefault("hardness_recall_threshold", 0.97)
    # Duration-based throughput-pass floor per sweep point, seconds: a fixed query set split across many workers can
    # finish in tens of milliseconds, too short a window for qps_wall to mean
    # anything. See harness/workloads.py's two-pass sweep-point design.
    cfg.setdefault("min_throughput_seconds", 5.0)
    # Per-run postgresql.conf overrides layered on top of pg/tuned.conf (e.g.
    # a larger maintenance_work_mem for a big index build) -- see
    # harness/pgctl.py:PgInstance.
    cfg.setdefault("postgres", {})
    # Point the run at an already-running Postgres (e.g. a containerized
    # production build, as the over-RAM cells do) instead of
    # the harness's own throwaway initdb'd instance. Dict of
    # {host, port, user?, password?, admin_dbname?} -- see
    # harness/pgctl.py:PgInstance's `external` param. None (default) is the
    # harness's normal local-instance behavior, unchanged.
    cfg.setdefault("external_pg", None)
    return cfg
