"""Pinecone-backed lanes: `pg_pinecone_remote` (CREATE INDEX ... USING pinecone, remote mode) and `pg_pinecone_remote_topk`
(pinecone.promote + pinecone.topk). The key comes from $PINECONE_API_KEY or the file $EE_ENV and is never printed or stored."""
from __future__ import annotations

import csv
import http.client
import json
import math
import os
import socket
import ssl
import statistics
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from . import metrics as metrics_mod
from .groundtruth import recall_at_k
from .systems import distance_operator
from .util import percentile

REMOTE_KINDS = ("pg_pinecone_remote", "pg_pinecone_remote_topk")
NAME_MARK = "pg-pinecone-bench-vec"
API_VERSION = "2026-04"
CP_HOST = "api.pinecone.io"
# Ceilings of one run; PG_PINECONE_BENCH_{WU,RU,USD}_MAX lower (or raise) them, e.g. $35 for the 0.10.0 box run.
WU_MAX = int(os.environ.get("PG_PINECONE_BENCH_WU_MAX", 6_000_000))
RU_MAX = int(os.environ.get("PG_PINECONE_BENCH_RU_MAX", 1_000_000))
USD_MAX = float(os.environ.get("PG_PINECONE_BENCH_USD_MAX", 40.0))
USD_PER_WU, USD_PER_RU = 4.50e-6, 18.0e-6
BENCH_DIR = Path(__file__).resolve().parent.parent
TRANSPORT = (socket.gaierror, ConnectionError, TimeoutError, ssl.SSLError, http.client.HTTPException, OSError)

_cfg: dict = {}
_current: dict = {"index": None, "ru": 1.0}
_last_point_end = 0.0
_key: str | None = None


class Stop(Exception):
    pass


def configure(remote_cfg: dict | None) -> None:
    global _cfg
    _cfg = dict(remote_cfg or {})


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _dir(env: str, default: str) -> Path:
    d = Path(os.environ.get(env) or BENCH_DIR / "results" / default)
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_dir() -> Path:
    return _dir("STATE", "state")


def raw_dir() -> Path:
    return _dir("RAW", "raw")


def state_get(name: str, default=None):
    try:
        return json.loads((state_dir() / f"{name}.json").read_text())
    except FileNotFoundError:
        return default


def state_set(name: str, obj) -> None:
    p = state_dir() / f"{name}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    tmp.replace(p)


def event(kind: str, **kw) -> None:
    with open(state_dir() / "events.jsonl", "a") as f:
        f.write(json.dumps({"t": round(time.time(), 3), "ts": time.strftime("%F %T"), "kind": kind, **kw}, default=str) + "\n")


def api_key() -> str:
    global _key
    if _key:
        return _key
    k = os.environ.get("PINECONE_API_KEY")
    if not k and os.environ.get("EE_ENV"):
        for ln in open(os.environ["EE_ENV"]):
            if ln.startswith("PINECONE_API_KEY="):
                k = ln.split("=", 1)[1].strip().strip("\"'")
    if not k:
        raise RuntimeError("no Pinecone API key: set PINECONE_API_KEY or EE_ENV")
    _key = k
    return k


def tally_add(kind: str, requests: int, units: float, note: str = "") -> None:
    with open(state_dir() / "cost-tally.jsonl", "a") as f:
        f.write(json.dumps({"ts": time.strftime("%F %T"), "kind": kind, "requests": requests, "units": round(units, 3), "note": note}) + "\n")


def tally_totals() -> dict:
    t = {"wu": 0.0, "ru": 0.0, "wu_json": 0.0, "write_requests": 0, "read_requests": 0}
    try:
        for ln in open(state_dir() / "cost-tally.jsonl"):
            e = json.loads(ln)
            if e["kind"] == "wu":
                t["wu"] += e["units"]
                t["write_requests"] += e["requests"]
            elif e["kind"] == "ru":
                t["ru"] += e["units"]
                t["read_requests"] += e["requests"]
            elif e["kind"] == "wu_json":
                t["wu_json"] += e["units"]
    except FileNotFoundError:
        pass
    t["usd"] = t["wu"] * USD_PER_WU + t["ru"] * USD_PER_RU
    return t


def tally_check() -> dict:
    t = tally_totals()
    if t["wu"] > WU_MAX or t["ru"] > RU_MAX or t["usd"] > USD_MAX:
        raise Stop(f"spend guard: {t['wu']:,.0f} WU, {t['ru']:,.0f} RU, ${t['usd']:.2f} (ceilings {WU_MAX:,} WU, {RU_MAX:,} RU, ${USD_MAX:.0f})")
    return t


def tally_reads(calls: float, note: str = "") -> None:
    if calls:
        tally_add("ru", int(calls), calls * _current["ru"], note)
        tally_check()


def _request(host: str, method: str, path: str, body=None, ver: str = API_VERSION, timeout: float = 30):
    c = http.client.HTTPSConnection(host, timeout=timeout, context=ssl.create_default_context())
    try:
        c.request(method, path, body=None if body is None else json.dumps(body, separators=(",", ":")),
                  headers={"Api-Key": api_key(), "X-Pinecone-Api-Version": ver, "Content-Type": "application/json", "Accept": "application/json"})
        r = c.getresponse()
        data = r.read()
        js = json.loads(data) if data and data[:1] in b"{[" else (data.decode("utf-8", "replace")[:300] if data else None)
        return r.status, js, {k.lower(): v for k, v in r.getheaders()}
    finally:
        c.close()


def _retried(fn, what: str, max_wait: float = 900):
    t0, delay = time.time(), 1.0
    while True:
        try:
            return fn()
        except TRANSPORT as e:
            if time.time() - t0 > max_wait:
                raise
            event("transport_pause", what=what, err=type(e).__name__)
            time.sleep(delay)
            delay = min(30.0, delay * 2)


def cp(method: str, path: str, body=None, ver: str = "2026-07"):
    st, js, _ = _retried(lambda: _request(CP_HOST, method, path, body, ver), f"cp {method} {path}")
    return st, js


def list_indexes() -> list[dict]:
    st, js = cp("GET", "/indexes")
    if st != 200:
        raise RuntimeError(f"list indexes: HTTP {st}")
    return js.get("indexes", [])


def index_names() -> list[str]:
    return sorted(i["name"] for i in list_indexes())


def describe_index(name: str):
    return cp("GET", f"/indexes/{urllib.parse.quote(name, safe='')}")


def index_host(name: str) -> str:
    st, js = describe_index(name)
    if st != 200:
        raise RuntimeError(f"describe {name}: HTTP {st}")
    return js["host"].replace("https://", "").rstrip("/")


def index_stats(host: str) -> dict:
    st, js, _ = _retried(lambda: _request(host, "POST", "/describe_index_stats", {}), "describe_index_stats")
    if st != 200:
        raise RuntimeError(f"describe_index_stats: HTTP {st}")
    return js


def namespace_count(host: str, namespace: str, prefix: bool = False) -> int:
    ns = index_stats(host).get("namespaces", {})
    if prefix:
        return sum(v.get("vectorCount", 0) for k, v in ns.items() if k.startswith(namespace + ":"))
    return ns.get(namespace, {}).get("vectorCount", 0)


def delete_index(name: str) -> int:
    assert NAME_MARK in name, name
    st, _ = cp("DELETE", f"/indexes/{urllib.parse.quote(name, safe='')}")
    event("index_deleted", name=name, http=st)
    return st


def register_index(name: str, **kw) -> None:
    state_set(f"index-{name}", {**kw, "name": name})


def state_glob(prefix: str) -> dict:
    out = {}
    for f in sorted(state_dir().glob(f"{prefix}-*.json")):
        out[f.stem[len(prefix) + 1:]] = json.loads(f.read_text())
    return out


def query_body(vec, top_k: int, namespace: str, flt=None) -> bytes:
    b = {"vector": [float(str(np.float32(x))) for x in vec], "topK": int(top_k), "includeMetadata": False, "namespace": namespace}
    if flt is not None:
        b["filter"] = flt
    return json.dumps(b, separators=(",", ":")).encode()


class Direct:
    def __init__(self, host: str, timeout: float = 30):
        self.host, self.timeout = host, timeout
        self.hdr = {"Api-Key": api_key(), "X-Pinecone-Api-Version": API_VERSION, "Content-Type": "application/json", "Accept": "application/json"}
        self.conn = None

    def _connect(self):
        c = http.client.HTTPSConnection(self.host, timeout=self.timeout, context=ssl.create_default_context())
        c.connect()
        c.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.conn = c

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def query(self, body: bytes):
        for attempt in (0, 1):
            if self.conn is None:
                self._connect()
            t0 = time.perf_counter_ns()
            try:
                self.conn.request("POST", "/query", body=body, headers=self.hdr)
                r = self.conn.getresponse()
                data = r.read()
                ms = (time.perf_counter_ns() - t0) / 1e6
                return r.status, (json.loads(data) if data[:1] == b"{" else None), {k.lower(): v for k, v in r.getheaders()}, ms
            except (http.client.RemoteDisconnected, http.client.CannotSendRequest, BrokenPipeError, ConnectionResetError):
                self.close()
                if attempt:
                    raise
            except TRANSPORT:
                self.close()
                raise


def rtt_probe(host: str, n: int = 20) -> dict:
    ctx = ssl.create_default_context()
    tcp, tls, ka = [], [], []
    for _ in range(n):
        t0 = time.perf_counter()
        s = socket.create_connection((host, 443), timeout=10)
        t1 = time.perf_counter()
        ss = ctx.wrap_socket(s, server_hostname=host)
        t2 = time.perf_counter()
        ss.close()
        tcp.append((t1 - t0) * 1e3)
        tls.append((t2 - t1) * 1e3)
    c = http.client.HTTPSConnection(host, timeout=10, context=ctx)
    c.connect()
    c.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for _ in range(n):
        t0 = time.perf_counter()
        c.request("GET", "/", headers={"Accept": "application/json"})
        c.getresponse().read()
        ka.append((time.perf_counter() - t0) * 1e3)
    c.close()
    f = lambda v: {"min": round(min(v), 2), "p50": round(statistics.median(v), 2), "max": round(max(v), 2)}
    return {"tcp_connect_ms": f(tcp), "tls_handshake_ms": f(tls), "keepalive_get_ms": f(ka), "n": n}


def calibrate(host: str, namespace: str, vec, index: str, n: int = 12, top_k: int = 50) -> dict:
    """Read units the API reports per query on the finished namespace: the unit price of the read tally."""
    d = Direct(host)
    body = query_body(vec, top_k, namespace)
    ru, svc = [], []
    for _ in range(n):
        st, js, h, _ms = d.query(body)
        if st == 200:
            ru.append((js.get("usage") or {}).get("readUnits", 1))
            if "x-pinecone-request-latency-ms" in h:
                svc.append(float(h["x-pinecone-request-latency-ms"]))
    d.close()
    ru_q = float(max(ru)) if ru else 1.0
    _current.update(index=index, ru=ru_q)
    tally_add("ru", len(ru), sum(ru), f"calibrate {index}")
    out = {"ru_per_query": ru_q, "ru_seen": sorted(set(ru)), "service_ms_p50": round(statistics.median(svc), 1) if svc else None, "n": len(ru)}
    state_set(f"units-{index}", out)
    return out


def set_unit_price(index: str) -> None:
    u = state_get(f"units-{index}")
    _current.update(index=index, ru=float(u["ru_per_query"]) if u else 1.0)


def store_key(conn) -> None:
    conn.execute("SELECT pinecone.set_api_key(%s)", (api_key(),))


def clear_key(conn) -> None:
    conn.execute("SELECT pinecone.clear_api_key()")


def pg_status(conn, table: str) -> dict:
    r = conn.execute(
        "SELECT mode, coalesce(backlog, 0), coalesce(queue_state, '-'), coalesce(last_error, ''), index_name, namespace "
        "FROM pinecone.status() WHERE table_name = %s", (f"public.{table}",)).fetchone()
    if r is None:
        return {"mode": None, "backlog": 0, "queue": "-", "last_error": "", "index": None, "namespace": None}
    return dict(zip(("mode", "backlog", "queue", "last_error", "index", "namespace"), r))


def _worker_cpu(conn):
    r = conn.execute("SELECT pid FROM pg_stat_activity WHERE backend_type LIKE 'pg_pinecone worker%' LIMIT 1").fetchone()
    if not r:
        return None, 0.0
    try:
        f = open(f"/proc/{r[0]}/stat").read().rsplit(")", 1)[1].split()
        return r[0], (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError):
        return r[0], 0.0


def _cpu_busy():
    try:
        v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    except OSError:
        return None
    return sum(v) - v[3] - v[4], sum(v)


def _lit(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def build_sql(system, table: str, column: str, metric: str) -> str:
    bp = system.build_params
    if system.kind == "pg_pinecone_remote":
        args = [f"{_lit(table)}::regclass", _lit(column), "engine => 'remote'", f"metric => {_lit(metric)}",
                f"index_name => {_lit(system.index_name(table))}"]
        if bp.get("filter_column"):
            args.append(f"filter_column => {_lit(bp['filter_column'])}")
        if bp.get("partition_column"):
            args.append(f"partition_column => {_lit(bp['partition_column'])}")
        return f"SELECT pinecone.create_index({', '.join(args)})"
    args = [f"{_lit(table)}::regclass", f"index_name => {_lit(bp.get('index_name') or f'{NAME_MARK}-topk')}", f"metric => {_lit(metric)}",
            f"sync_mode => {_lit(bp.get('sync_mode', 'async'))}"]
    return f"SELECT pinecone.promote({', '.join(args)})"


def vector_sizes(vectors: np.ndarray, id_len: int, meta_json: str | None, sample: int = 500) -> tuple[float, float]:
    """(JSON bytes per row as the extension serialises it, logical bytes per record = what Pinecone's write units count)."""
    idx = np.random.default_rng(1).choice(len(vectors), size=min(sample, len(vectors)), replace=False)
    nums = sum(len(str(np.float32(x))) + 1 for v in vectors[idx] for x in v) / len(idx)
    meta_len = len(meta_json) if meta_json else 0
    json_row = nums + len(f'{{"id":"{"0" * id_len}","values":[]}}') + (len(',"metadata":') + meta_len if meta_json else 0)
    return json_row, 4.0 * vectors.shape[1] + id_len + meta_len + 9


def simulate_backfill(n_rows: int, rows_per_block: float, json_per_row: float, logical_per_row: float,
                      tenant_ids: np.ndarray | None = None) -> dict:
    """Requests and write units of the extension's backfill from its chunking rules (promote.rs): a request carries at
    most 1,000 rows or 1.9 MB of record size (`logical_per_row` plus 23 bytes of framing); a partitioned index reads
    windows of 5,000 rows and sends one request per tenant in a window. Write units: 1 per KB of logical record size,
    5 minimum per request; `wu_json` counts the JSON text instead."""
    max_rows, max_bytes = 1000, 1_900_000
    per_req = max(1, min(max_rows, int(max_bytes // (logical_per_row + 23))))
    kb = lambda rows, per: max(5, math.ceil(rows * per / 1024))
    reqs = wu = wu_json = 0

    def sizes(rows: int):
        full, rem = divmod(rows, per_req)
        return [per_req] * full + ([rem] if rem else [])

    if tenant_ids is None:
        parts = sizes(n_rows)
        reqs, wu, wu_json = len(parts), sum(kb(s, logical_per_row) for s in parts), sum(kb(s, json_per_row) for s in parts)
    else:
        win, bs = 5000, max(1, int(round(rows_per_block)))
        pos = 0
        while pos < n_rows:
            end = min(n_rows, math.ceil(min(n_rows, pos + win) / bs) * bs)
            counts = np.bincount(tenant_ids[pos:end])
            for c in counts[counts > 0]:
                for s in sizes(int(c)):
                    reqs += 1
                    wu += kb(s, logical_per_row)
                    wu_json += kb(s, json_per_row)
            pos = end
    return {"requests": reqs, "wu": int(wu), "wu_json": int(wu_json)}


def _index_size(pg, idx_name: str) -> int:
    with pg.connect() as conn:
        return int(conn.execute("SELECT pg_relation_size(%s)", (idx_name,)).fetchone()[0])


def build(pg, table: str, column: str, metric: str, system, n_rows: int, dataset=None, reuse: bool = False,
          tenant_ids: np.ndarray | None = None) -> dict:
    from . import workloads as wl
    rcfg = _cfg
    with pg.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_pinecone")
        store_key(conn)
        for _ in range(60):
            if conn.execute("SELECT count(*) FROM pg_stat_activity WHERE backend_type = 'pg_pinecone launcher'").fetchone()[0]:
                break
            time.sleep(1)
        else:
            raise Stop("pg_pinecone launcher is not running: check shared_preload_libraries and pinecone.databases")
    am = system.kind == "pg_pinecone_remote"
    idx_name = system.index_name(table)
    prefix = bool(system.build_params.get("partition_column"))
    saved = state_get(f"build-{table}")
    if reuse and saved:
        with pg.connect() as conn:
            st = pg_status(conn, table)
            have = bool(conn.execute("SELECT 1 FROM pg_class WHERE relname = %s", (idx_name,)).fetchone()) if am else True
        if have and st["mode"] == "remote" and st["backlog"] == 0:
            say(f"reusing the built remote index {st['index']} ({st['namespace']})")
            set_unit_price(st["index"])
            again = _post_build(pg, table, column, metric, system, dataset, st, tenant_ids, rcfg, quick=True)
            return {**saved["build"], "reused": True, "measure_stage": again}
    with pg.connect() as conn:
        dropped = wl.drop_all_secondary_indexes(conn, table)
        sql = build_sql(system, table, column, metric)
        cpu0 = _cpu_busy()
        lsn0 = conn.execute("SELECT pg_current_wal_lsn()").fetchone()[0]
        t_create0 = time.time()
        conn.execute(sql)
        create_call_s = time.time() - t_create0
    with pg.connect() as conn:
        st = pg_status(conn, table)
    index, namespace = st["index"], st["namespace"]
    register_index(index, table=table, namespace=namespace, created=time.strftime("%F %T"))
    say(f"{system.kind}: create returned in {create_call_s:.1f} s; index {index}, namespace {namespace}, mode {st['mode']}")
    host = index_host(index)
    sync = _wait_synced(pg, table, host, namespace, prefix, n_rows, t_create0, rcfg)
    delivered_s = sync["delivered_at"] - t_create0
    cpu1 = _cpu_busy()
    with pg.connect() as conn:
        wal = float(conn.execute("SELECT pg_wal_lsn_diff(pg_current_wal_lsn(), %s::pg_lsn)", (lsn0,)).fetchone()[0] or 0)
        rpb = conn.execute("SELECT CASE WHEN relpages > 0 THEN reltuples / relpages END FROM pg_class WHERE relname = %s", (table,)).fetchone()[0]
    out = {"create_call_s": round(create_call_s, 2), "delivered_s": round(delivered_s, 1), "rows": n_rows,
           "rows_per_s": round(n_rows / max(delivered_s, 0.1)), "worker_cpu_s": round(sync["worker_cpu_s"], 1),
           "box_cpu_pct_mean": round(100.0 * (cpu1[0] - cpu0[0]) / max(1, cpu1[1] - cpu0[1]), 2) if cpu0 and cpu1 else None,
           "wal_mb": round(wal / 1e6, 1),
           "index": index, "namespace": namespace, "mode_final": sync["final"]["mode"], "queue_final": sync["final"]["queue"],
           "last_error": sync["final"]["last_error"][:200], "naptime_ms": sync["naptime_ms"], "backfill_round_ms": sync["round_ms"],
           "progress_csv": sync["csv"], "t_create0_epoch": t_create0}
    if dataset is not None:
        bp = system.build_params
        col = bp.get("filter_column") or bp.get("partition_column")
        meta = json.dumps({col: 1}, separators=(",", ":")) if am and col else None
        json_row, logical_row = vector_sizes(dataset.vectors[:n_rows], 9 if am else 6, meta)
        sim = simulate_backfill(n_rows, rpb or 100.0, json_row, logical_row, tenant_ids if prefix else None)
        out.update(backfill_requests_est=sim["requests"], backfill_wu_est=sim["wu"], backfill_wu_json_basis_est=sim["wu_json"],
                   json_bytes_per_row_est=round(json_row), logical_bytes_per_row_est=round(logical_row), rows_per_block=round(rpb or 0, 1))
        tally_add("wu", sim["requests"], sim["wu"], f"backfill {index}: {sim['requests']} requests, estimated from the chunking rules")
        tally_add("wu_json", sim["requests"], sim["wu_json"], f"backfill {index}: JSON-size basis")
        tally_check()
    t = tally_totals()
    say(f"backfill delivered: {n_rows} rows in {delivered_s:.0f} s ({out['rows_per_s']} rows/s); tally {t['wu']:,.0f} WU, {t['ru']:,.0f} RU, ${t['usd']:.2f}")
    with pg.connect() as conn:
        st = pg_status(conn, table)
    res = {"build_seconds": out["create_call_s"], "peak_rss_bytes": 0, "index_size_bytes": _index_size(pg, idx_name) if am else 0,
           "dropped_indexes": dropped, "build_notices": [], "graph_spilled_to_disk": False, "remote": out}
    state_set(f"build-{table}", {"build": res, "at": time.strftime("%F %T"), "gate": "pending"})
    out.update(_post_build(pg, table, column, metric, system, dataset, st, tenant_ids, rcfg, quick=False, t_create0=t_create0))
    state_set(f"build-{table}", {"build": res, "at": time.strftime("%F %T")})
    return res


def _wait_synced(pg, table, host, namespace, prefix, n_rows, t0, rcfg) -> dict:
    timeout_s = float(rcfg.get("max_backfill_s", 12 * 3600))
    path = raw_dir() / f"backfill-{table}.csv"
    new = not path.exists()
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["epoch", "elapsed_s", "pinecone_vectors", "backlog", "queue", "mode", "worker_cpu_s", "last_error"])
    cpu_acc, last_pid, last_val = 0.0, None, 0.0
    last_stats = last_cpu = last_print = last_retry = 0.0
    vectors = 0
    with pg.connect() as conn:
        naptime = conn.execute("SHOW pinecone.worker_naptime_ms").fetchone()[0]
        round_ms = conn.execute("SHOW pinecone.backfill_round_ms").fetchone()[0]
        while True:
            now = time.time()
            st = pg_status(conn, table)
            if now - last_stats >= float(rcfg.get("stats_every_s", 20)):
                vectors = namespace_count(host, namespace, prefix)
                last_stats = now
            if now - last_cpu >= 5:
                pid, cs = _worker_cpu(conn)
                if pid is not None:
                    cpu_acc += max(0.0, cs - (last_val if pid == last_pid else 0.0))
                    last_pid, last_val = pid, cs
                last_cpu = now
            w.writerow([round(now, 2), round(now - t0, 1), vectors, st["backlog"], st["queue"], st["mode"], round(cpu_acc, 1), st["last_error"][:100]])
            f.flush()
            if now - last_print >= 60:
                say(f"backfill: {vectors}/{n_rows} vectors in Pinecone, backlog {st['backlog']}, queue {st['queue']}, mode {st['mode']} {st['last_error'][:80]}")
                last_print = now
            if st["queue"] == "parked" and now - last_retry > 60:
                event("outbox_retry", table=table, last_error=st["last_error"][:160])
                conn.execute("SELECT pinecone.outbox_retry(%s::regclass)", (table,))
                last_retry = now
            if st["mode"] == "remote" and st["backlog"] == 0 and st["queue"] == "idle":
                break
            if now - t0 > timeout_s:
                raise Stop(f"backfill of {table} not finished after {timeout_s:.0f} s: {st}")
            time.sleep(2.0)
    f.close()
    return {"delivered_at": time.time(), "worker_cpu_s": cpu_acc, "final": st, "naptime_ms": int(naptime), "round_ms": int(round_ms), "csv": str(path)}


def _post_build(pg, table, column, metric, system, dataset, st, tenant_ids, rcfg, quick: bool, t_create0: float | None = None) -> dict:
    index, namespace = st["index"], st["namespace"]
    host = index_host(index)
    prefix = bool(system.build_params.get("partition_column"))
    names = index_stats(host).get("namespaces", {})
    out = {"pinecone_vectors": sum(v.get("vectorCount", 0) for k, v in names.items() if k == namespace or k.startswith(namespace + ":")),
           "pinecone_namespaces": len([k for k in names if k == namespace or k.startswith(namespace + ":")]),
           "index_spec": {k: v for k, v in describe_index(index)[1].items() if k in ("dimension", "metric", "vector_type", "deletion_protection", "spec", "status")}}
    if dataset is None:
        return out
    if not quick:
        probe_ns = f"{namespace}:{int(tenant_ids[0])}" if prefix else namespace
        out["calibration"] = calibrate(host, probe_ns, dataset.vectors[0], index, n=int(rcfg.get("calibrate_n", 12)))
        say(f"calibration: {out['calibration']}")
    out["gate"] = gate(pg, table, column, metric, system, dataset, tenant_ids, rcfg, n=int(rcfg.get("gate_n", 200 if not quick else 100)))
    if t_create0 is not None:
        out["time_to_searchable_s"] = round(out["gate"]["passed_at"] - t_create0, 1)
    out["rtt"] = rtt_probe(host, int(rcfg.get("rtt_n", 20)))
    return out


def gate(pg, table, column, metric, system, dataset, tenant_ids, rcfg, n: int = 200, min_pct: float = 99.0) -> dict:
    """Self-retrieval of sampled rows through the lane under test; a hit is distance 0 at rank 1 (duplicate vectors count)."""
    op = distance_operator(metric)
    rows = np.random.default_rng(20261014).choice(len(dataset.vectors), size=min(n, len(dataset.vectors)), replace=False)
    am = system.kind == "pg_pinecone_remote"
    part = system.build_params.get("partition_column")

    def one(conn, i):
        q = dataset.vectors[i]
        if am:
            where = f"WHERE {part} = {int(tenant_ids[i])}" if part else ""
            r = conn.execute(f"SELECT ({column} {op} %s) FROM {table} {where} ORDER BY {column} {op} %s LIMIT 1", (q, q)).fetchone()
        else:
            r = conn.execute("SELECT distance FROM pinecone.topk(%s::regclass, (%s::vector)::real[], 1)", (table, q)).fetchone()
        return bool(r) and r[0] is not None and float(r[0]) <= 1e-3

    def round_():
        hits, errs, lock = [0], [0], threading.Lock()

        def work(chunk):
            with pg.connect() as conn:
                register_vector(conn)
                for i in chunk:
                    try:
                        ok = one(conn, int(i))
                    except psycopg.Error:
                        ok = False
                        with lock:
                            errs[0] += 1
                    with lock:
                        hits[0] += ok

        with ThreadPoolExecutor(4) as ex:
            list(ex.map(work, np.array_split(rows, 4)))
        tally_reads(len(rows), f"gate {table}")
        return 100.0 * hits[0] / len(rows), errs[0]

    t0 = time.time()
    rounds = []
    max_wait = float(rcfg.get("gate_max_wait_s", 900))
    settle_s = float(rcfg.get("gate_settle_s", 90))
    floor_pct = float(rcfg.get("gate_floor_pct", 90))
    while True:
        pct, err = round_()
        el = round(time.time() - t0, 1)
        rounds.append({"t_s": el, "pct": round(pct, 2), "errors": err})
        say(f"gate: self-retrieval {pct:.1f}% ({err} errors) at +{el} s")
        if pct >= min_pct:
            return {"passed": True, "passed_at": time.time(), "rounds": rounds, "n": len(rows)}
        last = [r["pct"] for r in rounds[-4:]]
        if el >= settle_s and len(last) == 4 and max(last) - min(last) <= 0.5 and pct >= floor_pct:
            return {"passed": True, "stable_below_threshold": True, "passed_at": time.time(), "rounds": rounds, "n": len(rows)}
        if el > max_wait:
            if pct < floor_pct:
                raise Stop(f"gate for {table} stays below {floor_pct}% after {max_wait:.0f} s: {rounds[-3:]}")
            return {"passed": False, "passed_at": time.time(), "rounds": rounds, "n": len(rows)}
        time.sleep(10)


def teardown(pg, table: str, system, delete_index_after: bool) -> dict:
    if not delete_index_after:
        return {"kept": True}
    out = {}
    with pg.connect() as conn:
        st = pg_status(conn, table)
        if system.kind == "pg_pinecone_remote_topk":
            conn.execute("SELECT pinecone.demote(%s::regclass, drop_remote_index => %s)", (table, bool(delete_index_after)))
        else:
            conn.execute(f"DROP INDEX IF EXISTS {system.index_name(table)}")
    if delete_index_after and st["index"] and st["index"] in index_names():
        out["delete_http"] = delete_index(st["index"])
        deadline = time.time() + 300
        while st["index"] in index_names() and time.time() < deadline:
            time.sleep(3)
        out["gone"] = st["index"] not in index_names()
    return out


def pace() -> None:
    wait = _last_point_end + float(_cfg.get("rest_seconds", 0)) - time.time()
    if wait > 0:
        time.sleep(wait)


def _classify(exc: Exception) -> str:
    msg = str(exc).lower()
    if any(s in msg for s in ("429", "rate limit", "rate-limit", "qps limit", "too many requests")):
        return "429"
    if "am_max_topk" in msg:
        return "topk_cap"
    return "err"


def _remote_worker_run(table, column, op, system, sweep_value, iterative_scan, scan_budget, warmup, measured, k, where_clause,
                       recall_barrier, throughput_barrier, min_throughput_seconds, retry_max, record_stats):
    from . import workloads as wl
    conn = wl._worker_conn
    try:
        system.set_session_params(conn, sweep_value, iterative_scan, scan_budget)
        am = system.kind == "pg_pinecone_remote"
        sql = (f"SELECT id FROM {table} {where_clause} ORDER BY {column} {op} %s LIMIT %s" if am
               else "SELECT id FROM pinecone.topk(%s::regclass, (%s::vector)::real[], %s)")

        def run_one(item):
            if isinstance(item, tuple):
                params, group = (item[0], item[1], k), item[0]
            else:
                params, group = ((item, k) if am else (table, item, k)), None
            t0 = time.perf_counter()
            rows = conn.execute(sql, params).fetchall()
            return [int(r[0]) for r in rows], time.perf_counter() - t0, group

        seen_errors = {}

        def attempt(item):
            fails, err = 0, None
            group = item[0] if isinstance(item, tuple) else None
            for a in range(retry_max + 1):
                try:
                    ids, el, group = run_one(item)
                    return ids, el, fails, err, group
                except psycopg.Error as e:
                    err = _classify(e)
                    key = f"{err}: {str(e).strip().splitlines()[0][:200]}"
                    seen_errors[key] = seen_errors.get(key, 0) + 1
                    fails += 1
                    if err == "topk_cap" or a == retry_max:
                        break
                    time.sleep(min(1.0, 0.025 * (2 ** a)))
            return None, None, fails, err, group

        for q in warmup:
            attempt(q)
        recall_barrier.wait()
        results, consecutive = [], 0
        for q in measured:
            ids, el, fails, err, group = attempt(q)
            stats = None
            if ids is not None and record_stats and am:
                stats = conn.execute("SELECT batches, rescored, capped FROM pinecone.last_scan_stats()").fetchone()
            results.append((ids, el, fails, err, stats, group))
            consecutive = consecutive + 1 if ids is None and err != "topk_cap" else 0
            if consecutive >= 200:
                raise RuntimeError(f"200 consecutive failed queries (last error class {err})")
        throughput_barrier.wait()
        ok = fail = 0
        t_start = time.perf_counter()
        t_now = t_start
        while t_now - t_start < min_throughput_seconds:
            for q in measured:
                ids, _el, fails, _err, _g = attempt(q)
                ok += ids is not None
                fail += fails
                t_now = time.perf_counter()
                if t_now - t_start >= min_throughput_seconds:
                    break
        return results, ok, fail, t_start, time.perf_counter(), seen_errors
    except BaseException:
        recall_barrier.abort()
        throughput_barrier.abort()
        raise


def _as_vec(item):
    return item[1] if isinstance(item, tuple) else item


def run_sweep_point_remote(pool, manager, dsn, table, column, metric, system, sweep_value, iterative_scan, measured_queries,
                           warmup_pool, ground_truth, k, where_clause="", n_conns=1, warmup_n=10, expected_index=None,
                           scan_budget=None, force_index=False, min_throughput_seconds=5.0, plan_where=None, **_ignored) -> dict:
    global _last_point_end
    from . import workloads as wl
    pace()
    gt_ids, gt_dists = ground_truth
    op = distance_operator(metric)
    am = system.kind == "pg_pinecone_remote"
    base = dict(sweep_param=system.sweep_param, sweep_value=sweep_value, iterative_scan=iterative_scan, scan_budget=scan_budget,
                scan_fraction=None, force_index=force_index, force_seqscan=False, n_conns=n_conns, k=k)
    if am:
        shape, index_used, eff = wl.capture_plan(dsn, table, column, op, system, sweep_value, iterative_scan,
                                                 plan_where if plan_where is not None else where_clause, _as_vec(measured_queries[0]), k,
                                                 scan_budget=scan_budget, force_index=force_index)
        base.update(plan_shape=shape, index_used=index_used, effective_gucs=eff)
        bad = None
        if not shape.startswith("index_scan:"):
            bad = f"plan is {shape}: the pg_pinecone index is not used"
        elif expected_index is not None and index_used != expected_index:
            bad = f"plan used index {index_used!r}, expected {expected_index!r}"
        if bad:
            stats = metrics_mod.latency_stats([])
            stats.update(base, valid=False, recall=float("nan"), abort_reason=bad)
            return stats
    else:
        base.update(plan_shape="function:pinecone.topk", index_used=None, effective_gucs={})
    m_slices = [measured_queries[i::n_conns] for i in range(n_conns)]
    gt_slices = [gt_ids[i::n_conns] for i in range(n_conns)]
    gtd_slices = [gt_dists[i::n_conns] for i in range(n_conns)]
    w_slices = [warmup_pool[i * warmup_n:(i + 1) * warmup_n] for i in range(n_conns)]
    recall_barrier = manager.Barrier(n_conns)
    throughput_barrier = manager.Barrier(n_conns)
    futures = [pool.submit(_remote_worker_run, table, column, op, system, sweep_value, iterative_scan, scan_budget, w_slices[i], m_slices[i],
                           k, where_clause, recall_barrier, throughput_barrier, min_throughput_seconds, int(_cfg.get("retry_max", 12)),
                           bool(system.record_scan_stats)) for i in range(n_conns)]
    per = [f.result() for f in futures]
    lat, recalls, rows_ret, batches, rescored, capped, by_group = [], [], [], [], [], [], {}
    n_q = n_fail_q = n_fail_attempts = n_cap = n_err = 0
    errors_seen = {}
    for pi, (results, _ok, _fail, _ts, _te, seen) in enumerate(per):
        for key, cnt in seen.items():
            errors_seen[key] = errors_seen.get(key, 0) + cnt
        for qi, (ids, el, fails, err, stats, group) in enumerate(results):
            n_q += 1
            n_fail_attempts += fails
            if ids is None:
                n_fail_q += 1
                n_cap += err == "topk_cap"
                n_err += err not in ("topk_cap", "429")
                continue
            lat.append(el)
            rows_ret.append(len(ids))
            recalls.append(recall_at_k(ids, gt_slices[pi][qi], k, truth_dists_row=gtd_slices[pi][qi]))
            if stats is not None:
                batches.append(stats[0])
                rescored.append(stats[1])
                capped.append(bool(stats[2]))
            if group is not None:
                by_group.setdefault(group, []).append(el)
    tp_ok = sum(p[1] for p in per)
    tp_fail = sum(p[2] for p in per)
    wall = max(p[4] for p in per) - min(p[3] for p in per)
    out = metrics_mod.latency_stats(lat)
    out["qps_wall"] = tp_ok / wall if wall > 0 else float("nan")
    out["throughput_completed_queries"] = tp_ok
    out["throughput_wall_seconds"] = wall
    out["recall"] = float(np.nanmean(recalls)) if recalls else float("nan")
    out["rows_returned_mean"] = float(np.mean(rows_ret)) if rows_ret else float("nan")
    out["rows_returned_min"] = int(min(rows_ret)) if rows_ret else 0
    out["frac_filled"] = float(np.mean([r >= k for r in rows_ret])) if rows_ret else float("nan")
    out["n_queries"] = n_q
    if errors_seen:
        out["error_messages"] = dict(sorted(errors_seen.items(), key=lambda kv: -kv[1])[:5])
    out["failed_queries"] = n_fail_q
    out["failed_topk_cap"] = n_cap
    out["failed_other"] = n_err
    out["rate_limited_attempts_recall_pass"] = n_fail_attempts
    out["rate_limited_frac_recall_pass"] = n_fail_attempts / max(1, n_fail_attempts + len(lat) + n_fail_q)
    out["rate_limited_frac_throughput_pass"] = tp_fail / max(1, tp_fail + tp_ok)
    if batches:
        out["scan_batches_mean"] = float(np.mean(batches))
        out["scan_rescored_mean"] = float(np.mean(rescored))
        out["scan_capped_frac"] = float(np.mean(capped))
    calls_q = out.get("scan_batches_mean", 1.0)
    calls = (len(lat) + n_fail_q + tp_ok + n_conns * (warmup_n + 1)) * calls_q + n_fail_attempts + tp_fail
    out["pinecone_calls_est"] = calls
    tally_reads(calls, f"{system.name} k={k} c={n_conns} sweep={sweep_value}")
    if by_group:
        full = [v for v in by_group.values() if len(v) >= 3]
        if full:
            p50s = sorted(percentile(sorted(v), 50) * 1000 for v in full)
            out["groups"] = len(by_group)
            out["group_p50_ms_min"], out["group_p50_ms_median"], out["group_p50_ms_max"] = p50s[0], statistics.median(p50s), p50s[-1]
            out["group_p95_ms_median"] = statistics.median(percentile(sorted(v), 95) * 1000 for v in full)
    out.update(base, valid=True, abort_reason=None)
    _last_point_end = time.time()
    return out


def direct_lane(host: str, namespace: str, queries, top_k: int, concurrency=(1, 4, 16), n: int = 1000, flt=None,
                warmup: int = 30, label: str = "direct") -> list[dict]:
    """The same /query over keep-alive HTTPS from Python, no Postgres: client time, Pinecone's own service time, reported units, 429s."""
    out = []
    idx = np.random.default_rng(7).choice(len(queries), size=min(n, len(queries)), replace=len(queries) < n)
    bodies = [query_body(_as_vec(queries[i]), top_k, namespace, flt) for i in idx]
    for c in concurrency:
        pace()
        res, ru_sum, fails = [[] for _ in range(c)], [0] * c, [0] * c
        bar = threading.Barrier(c)

        def work(w):
            d = Direct(host)
            for b in bodies[:warmup]:
                try:
                    d.query(b)
                except TRANSPORT:
                    pass
            bar.wait()
            for b in bodies[w::c]:
                try:
                    st, js, h, ms = d.query(b)
                except TRANSPORT:
                    fails[w] += 1
                    continue
                if st == 200:
                    res[w].append((ms, float(h.get("x-pinecone-request-latency-ms", "nan"))))
                    ru_sum[w] += (js.get("usage") or {}).get("readUnits", 1)
                else:
                    fails[w] += 1
            d.close()

        t0 = time.time()
        ths = [threading.Thread(target=work, args=(w,)) for w in range(c)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        wall = time.time() - t0
        client = sorted(x[0] for r in res for x in r)
        svc = sorted(x[1] for r in res for x in r if x[1] == x[1])
        extra = sum(fails) + c * warmup
        tally_add("ru", len(client) + extra, sum(ru_sum) + extra * _current["ru"], f"{label} c={c} top_k={top_k}")
        tally_check()
        pct = lambda v, p: round(v[min(len(v) - 1, int(p / 100 * len(v)))], 2) if v else None
        out.append({"label": label, "n_conns": c, "top_k": top_k, "ok": len(client), "failed_or_429": sum(fails),
                    "client_p50_ms": pct(client, 50), "client_p95_ms": pct(client, 95), "client_p99_ms": pct(client, 99),
                    "service_p50_ms": pct(svc, 50), "service_p95_ms": pct(svc, 95), "ru_reported": sum(ru_sum), "ok_per_s": round(len(client) / wall, 1)})
        say(f"direct c={c} top_k={top_k}: {out[-1]}")
    return out


def flood_probe(host: str, namespace: str, queries, top_k: int = 50, clients: int = 4, secs: int = 30, rest_s: int = 120,
                label: str = "flood") -> dict:
    """Direct /query at full speed from a rested start: when the first 429 comes and what is accepted per second."""
    time.sleep(rest_s)
    idx = np.random.default_rng(11).choice(len(queries), size=min(len(queries), 200), replace=False)
    bodies = [query_body(_as_vec(queries[i]), top_k, namespace) for i in idx]
    events = []
    lock = threading.Lock()
    t_start = time.time() + 1.0
    deadline = t_start + secs

    def work(w):
        d = Direct(host)
        i = w
        while time.time() < t_start:
            time.sleep(0.01)
        while time.time() < deadline:
            try:
                st, _js, _h, ms = d.query(bodies[i % len(bodies)])
            except TRANSPORT:
                st, ms = 0, 0.0
            i += clients
            with lock:
                events.append((time.time() - t_start, st, ms))
        d.close()

    ths = [threading.Thread(target=work, args=(w,)) for w in range(clients)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    per = {}
    for t, st, ms in events:
        sec = int(t)
        ok, rej = per.get(sec, (0, 0))
        per[sec] = (ok + (st == 200), rej + (st == 429))
    ok_ms = sorted(ms for _t, st, ms in events if st == 200)
    rejected = [t for t, st, _ms in events if st == 429]
    out = {"label": label, "clients": clients, "secs": secs, "rest_s": rest_s, "top_k": top_k,
           "accepted": sum(v[0] for v in per.values()), "rejected": sum(v[1] for v in per.values()),
           "first_429_at_s": round(min(rejected), 1) if rejected else None,
           "accepted_per_s_first10": round(sum(per.get(s_, (0, 0))[0] for s_ in range(10)) / 10, 1),
           "accepted_per_s_last10": round(sum(per.get(s_, (0, 0))[0] for s_ in range(secs - 10, secs)) / 10, 1),
           "accepted_p50_ms": round(ok_ms[len(ok_ms) // 2], 2) if ok_ms else None,
           "per_second": [[s_, per[s_][0], per[s_][1]] for s_ in sorted(per)]}
    total = out["accepted"] + out["rejected"]
    tally_add("ru", total, total * _current["ru"], f"{label} flood")
    tally_check()
    say(f"flood: { {k: v for k, v in out.items() if k != 'per_second'} }")
    return out


def partition_workload(dataset, tenant_ids: np.ndarray, n_tenants: int, per_tenant: int, k_gt: int, metric: str, seed: int, n_warmup: int):
    """Per tenant: held-out bank vectors from the tenant's own region, exact top-k among the tenant's rows only.
    Returns (queries as (tenant, vector), warmup, (gt_ids, gt_dists)), shuffled across tenants so concurrent clients meet different namespaces."""
    from . import datasets as ds_mod
    from . import groundtruth as gt_mod
    from .util import stable_hash
    n_centers = int(dataset.centers.shape[0])
    per_t = max(1, math.ceil(n_centers / n_tenants))
    items, gts_i, gts_d, warm = [], [], [], []
    for t in range(n_tenants):
        rows = np.nonzero(tenant_ids == t)[0]
        if len(rows) == 0:
            continue
        clusters = np.arange(t * per_t, min(n_centers, (t + 1) * per_t))
        seed_t = int(stable_hash(dict(seed=seed, t=t)), 16) % (2 ** 32)
        if dataset.query_bank is not None:
            qv, wv, _info = ds_mod.sample_region_queries_real(dataset, clusters, per_tenant, 2, seed_t)
        else:
            qv, wv = ds_mod.sample_region_queries(dataset, clusters, per_tenant, 2, seed_t)
        key = stable_hash(dict(dataset=dataset.key, metric=metric, k=k_gt, tenant=t, n_t=len(rows), nq=per_tenant, seed=seed, kind="partition"))
        ids, dists = gt_mod.brute_force_topk(qv, dataset.vectors[rows], k=k_gt, metric=metric, row_ids=rows, cache_key=key)
        for j in range(len(qv)):
            items.append((t, qv[j]))
            gts_i.append(ids[j])
            gts_d.append(dists[j])
        warm.append((t, wv[0]))
    order = np.random.default_rng(seed).permutation(len(items))
    queries = [items[i] for i in order]
    gt = (np.array([gts_i[i] for i in order]), np.array([gts_d[i] for i in order]))
    pick = np.random.default_rng(seed + 1).choice(len(warm), size=n_warmup, replace=len(warm) < n_warmup)
    return queries, [warm[i] for i in pick], gt
