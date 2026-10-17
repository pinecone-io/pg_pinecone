"""Lifecycle management for the harness's own throwaway Postgres instance.

Never touches the shared pgrx dev instance. initdb's a fresh data dir under
bench/vector/.pgdata, starts it on a free port with the tuned conf, and provides a
psycopg connection helper. Always stop() what you start().

`external=` (see `PgInstance.__init__`) is the one exception to "the harness
always starts its own instance": it points `PgInstance` at an ALREADY-RUNNING
Postgres over TCP instead of initdb'ing one -- e.g. a containerized
production-build Postgres (as the over-RAM cells use). initdb()/start()/
stop() become no-ops (the container's lifecycle is managed outside the
harness, by whatever started it); `create_bench_db()` (drop+recreate the
`bench` database) still runs and is what gives each `PgInstance(...)` construction
its "fresh state" guarantee in external mode, the same role `initdb()`'s
rmtree plays for a local throwaway instance.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import psycopg

from . import util

BENCH_DIR = Path(__file__).resolve().parent.parent
# PG_PINECONE_BENCH_PG = Postgres install prefix (<prefix>/bin/pg_ctl); PGRX_PG_BIN (the bin/ dir) is the older spelling.
DEFAULT_PG_BIN = Path(
    os.path.join(os.environ["PG_PINECONE_BENCH_PG"], "bin") if os.environ.get("PG_PINECONE_BENCH_PG")
    else os.environ.get("PGRX_PG_BIN", str(Path.home() / ".pgrx/17.10/pgrx-install/bin"))
)
# BENCH_TUNED_CONF selects a different tuned conf (e.g. pg/tuned.linux-gcp.conf
# on a GCP box -- the default file is sized to the Mac the harness was
# developed on). Same append-verbatim mechanics either way.
TUNED_CONF = Path(os.environ.get("BENCH_TUNED_CONF", str(BENCH_DIR / "pg" / "tuned.conf")))


class PgInstance:
    def __init__(self, data_dir: Path | None = None, pg_bin: Path = DEFAULT_PG_BIN,
                 extra_libs: list[str] | None = None, conf_overrides: dict | None = None,
                 external: dict | None = None, reuse: bool = False):
        """`external`, if given, is a dict of `{host, port, user, password?,
        admin_dbname?}` for an already-running Postgres (e.g. a Docker
        container) -- see the module docstring. All other constructor args
        besides `conf_overrides` are ignored in this mode (no local pg_bin/
        data_dir involved)."""
        self.external = dict(external) if external else None
        self.reuse = reuse
        if self.external is not None:
            self.pg_bin = None
            self.data_dir = None
            self.port = self.external["port"]
            self.host = self.external["host"]
            self.user = self.external.get("user", "postgres")
            self.password = self.external.get("password")
            self.admin_dbname = self.external.get("admin_dbname", "postgres")
            self.conf_overrides = conf_overrides or {}
            self._started = False
            return
        self.pg_bin = Path(pg_bin)
        if not (self.pg_bin / "pg_ctl").exists():
            raise RuntimeError(
                f"pg_ctl not found under {self.pg_bin}. Set PG_PINECONE_BENCH_PG to the "
                "Postgres 17 install prefix (the directory that holds bin/pg_ctl)."
            )
        self.data_dir = Path(data_dir or (BENCH_DIR / ".pgdata"))
        self.port = util.free_port()
        self.socket_dir = self.data_dir.parent / f"{self.data_dir.name}.sock"
        if len(str(self.socket_dir)) + len("/.s.PGSQL.65535") > 100:
            # AF_UNIX paths top out at ~103 bytes: a deep checkout or --data-dir would make Postgres refuse to start.
            self.socket_dir = Path(tempfile.gettempdir()) / f"pgb-{util.stable_hash(str(self.data_dir))[:10]}.sock"
        self.extra_libs = extra_libs or []
        # Per-run postgresql.conf overrides (e.g. a bigger maintenance_work_mem
        # for a large index build), layered
        # AFTER pg/tuned.conf in initdb() so they win (Postgres conf files use
        # last-setting-wins for repeated keys).
        self.conf_overrides = conf_overrides or {}
        self._started = False

    def _run(self, *args, check=True):
        return subprocess.run(
            [str(self.pg_bin / args[0]), *args[1:]],
            capture_output=True, text=True, check=check,
        )

    def initdb(self):
        if self.external is not None:
            return
        if self.reuse and (self.data_dir / "PG_VERSION").exists():
            self.socket_dir.mkdir(parents=True, exist_ok=True)
            return
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
        self.socket_dir.mkdir(parents=True, exist_ok=True)
        self._run("initdb", "-D", str(self.data_dir), "-U", "postgres",
                   "--auth=trust", "-E", "UTF8")
        conf_path = self.data_dir / "postgresql.conf"
        with open(conf_path, "a") as f:
            f.write("\n# --- bench harness overrides (pg/tuned.conf) ---\n")
            f.write(TUNED_CONF.read_text())
            f.write(f"\nport = {self.port}\n")
            f.write(f"unix_socket_directories = '{self.socket_dir}'\n")
            f.write("listen_addresses = ''\n")  # unix socket only; no TCP, no port clashes
            if self.extra_libs:
                f.write(f"shared_preload_libraries = '{','.join(self.extra_libs)}'\n")
            if self.conf_overrides:
                f.write("\n# --- per-run overrides (config YAML's `postgres:` section) ---\n")
                for key, value in self.conf_overrides.items():
                    f.write(f"{key} = {value}\n")

    def start(self):
        if self.external is not None:
            # Already running (started outside the harness -- e.g. `docker
            # run`); just confirm reachability and apply this run's
            # postgresql.conf overrides via ALTER SYSTEM (initdb-time conf
            # file appending isn't available for an instance we didn't
            # initdb). All three overrides every external-Postgres config uses
            # (maintenance_work_mem, min_wal_size, max_wal_size) are
            # reloadable -- no restart needed.
            for _ in range(60):
                try:
                    with self.connect(dbname=self.admin_dbname) as conn:
                        conn.execute("SELECT 1")
                    break
                except Exception:
                    time.sleep(0.5)
            else:
                raise RuntimeError(
                    f"external Postgres at {self.host}:{self.port} did not become reachable")
            if self.conf_overrides:
                with self.connect(dbname=self.admin_dbname) as conn:
                    for key, value in self.conf_overrides.items():
                        conn.execute(f"ALTER SYSTEM SET {key} = '{value}'")
                    conn.execute("SELECT pg_reload_conf()")
            self._started = True
            return
        log_path = self.data_dir.parent / f"{self.data_dir.name}.log"
        if self.reuse:
            running = self._run("pg_ctl", "-D", str(self.data_dir), "status", check=False).returncode == 0
            if running:
                pid_lines = (self.data_dir / "postmaster.pid").read_text().splitlines()
                self.port, self.socket_dir = int(pid_lines[3]), Path(pid_lines[4])
            else:
                self._run("pg_ctl", "-D", str(self.data_dir), "-l", str(log_path), "-w", "-t", "60",
                          "-o", f"-p {self.port} -k {self.socket_dir}", "start")
        else:
            self._run("pg_ctl", "-D", str(self.data_dir), "-l", str(log_path),
                      "-w", "-t", "30", "start")
        self._started = True
        # Wait until it actually accepts connections (pg_ctl -w can race on slow disks).
        for _ in range(60):
            try:
                with self.connect(dbname="postgres") as conn:
                    conn.execute("SELECT 1")
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError(f"Postgres did not become ready; see {log_path}")

    def stop(self):
        if self.external is not None:
            # The container's lifecycle belongs to whatever started it, not
            # to this harness -- never stop it here.
            self._started = False
            return
        if self._started:
            self._run("pg_ctl", "-D", str(self.data_dir), "-m", "fast", "stop", check=False)
            self._started = False

    def dsn(self, dbname="bench") -> str:
        if self.external is not None:
            auth = f"user={self.user}"
            if self.password:
                auth += f" password={self.password}"
            return f"host={self.host} port={self.port} {auth} dbname={dbname}"
        return f"host={self.socket_dir} port={self.port} user=postgres dbname={dbname}"

    def connect(self, dbname="bench", autocommit=True) -> psycopg.Connection:
        # prepare_threshold=None: never let psycopg auto-prepare a statement
        # after N executions -- a generic plan chosen after auto-prepare can
        # silently differ in shape from the custom plan used on every prior
        # execution. Sweep workers set this too
        # (harness/workloads.py); this is for the harness's own bookkeeping
        # connections (loader, ground truth, churn).
        conn = psycopg.connect(self.dsn(dbname), autocommit=autocommit, prepare_threshold=None)
        return conn

    def create_bench_db(self):
        with self.connect(dbname="postgres") as conn:
            if self.reuse and conn.execute("SELECT 1 FROM pg_database WHERE datname = 'bench'").fetchone():
                return
            conn.execute("DROP DATABASE IF EXISTS bench")
            conn.execute("CREATE DATABASE bench")

    def __enter__(self):
        self.initdb()
        self.start()
        self.create_bench_db()
        return self

    def __exit__(self, *exc):
        self.stop()
