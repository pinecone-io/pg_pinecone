#!/usr/bin/env python3
"""Single entrypoint for the pg_pinecone/pgvector ANN benchmark harness.

Usage:
    bench/vector/.venv/bin/python bench/vector/run.py --config bench/vector/configs/smoke.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness.config import load_config
from harness.runner import run, save_results
from harness.report import render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to a YAML run config")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true",
                     help="allow saving results from an uncommitted tree (dev iteration only -- "
                          "the filename and report are marked DIRTY and can never be mistaken "
                          "for the committed run of record; see harness/runner.py:save_results)")
    ap.add_argument("--data-dir", default=None,
                     help="override PgInstance's throwaway data directory (default: bench/vector/.pgdata). "
                          "REQUIRED alongside any run that might overlap another one already using "
                          "the default path -- initdb() rmtree's whatever it finds there, so sharing "
                          "the default path with a live run deletes its data out from under it, not "
                          "just a port clash (ports are always freshly OS-assigned regardless).")
    ap.add_argument("--prep-only", action="store_true",
                     help="compute and cache datasets, k-means and ground truth, then exit (no Postgres)")
    ap.add_argument("--stage", choices=["all", "build", "measure"], default=None,
                     help="remote configs: `build` stops after the backfill and gate, `measure` reuses "
                          "the built index of an earlier `build` run in the same --data-dir")
    ap.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                     help="override one config value; PATH is a /-separated key path, VALUE is YAML "
                          "(e.g. --set postgres/pinecone.worker_naptime_ms=100)")
    ap.add_argument("--reuse", action="store_true",
                     help="keep an existing --data-dir, loaded table and built remote index")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                         format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    for item in args.set:
        path, _, value = item.partition("=")
        node = cfg
        keys = path.split("/")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(value)
    if args.stage:
        cfg["stage"] = args.stage
    if args.reuse:
        cfg["reuse"] = True
    results = run(cfg, data_dir=args.data_dir, prep_only=args.prep_only)
    if results is None:
        print("prep done")
        return
    json_path = save_results(results, allow_dirty=args.allow_dirty)
    md_path = json_path.with_suffix(".md")
    md_path.write_text(render(results))

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
