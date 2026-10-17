"""Core of the partition chaos fake Pinecone: shared state, fault gate, HTTP base (127.0.0.1 only).

State is keyed by index name: vecs[index][ns][id] = (values, metadata), docs[index][ns][id] = doc.
Every data-plane request is logged in arrival order (op, index, ns, ids, n, status, t, done).
Fault knobs: global STATE["chaos"] (mode, latency_ms, scope, ns_cap, ns_delete_latency_ms, ...) and
per-namespace STATE["rules"] (ns | ns_prefix | ns_re, ops, mode, hits, latency_ms, status, message).
"""
import re
import threading
import time

LOCK = threading.RLock()
GEN = {"n": 0}  # bumped by __down/__reset/abort_sleepers so sleeping handlers give up
RATE_MSG = ("Request failed. You've reached the max upsert requests per second per namespace. "
            "Pace your upsert requests or contact Pinecone Support to request a higher limit.")
QUOTA_MSG = ("Request failed. You've reached your write unit limit for the current month. "
             "To continue writing data, upgrade your plan.")
CAP_MSG = ("Request failed. You've reached the maximum number of namespaces allowed for this index "
           "on your plan ({cap}). Delete unused namespaces or upgrade your plan.")  # wording assumed, unverified live
CHAOS0 = {"mode": "ok", "latency_ms": 0, "scope": "data", "retry_after": None, "ns_cap": 0,
          "cap_status": 400, "cap_message": "", "ns_delete_latency_ms": 0, "log_ids": True}
SCOPES = {"docs": {"docs"}, "vec": {"vec"}, "data": {"docs", "vec"}, "all": {"docs", "vec", "ctl"}}
STATE = {"indexes": {}, "vecs": {}, "docs": {}, "accepts": {}, "log": [], "seq": 0,
         "chaos": dict(CHAOS0), "rules": [], "counts": {}}


def now():
    return round(time.time(), 3)


def bump(key):
    STATE["counts"][key] = STATE["counts"].get(key, 0) + 1


def err(code, msg, status, hdr=None):
    return (status, {"error": {"code": code, "message": msg}, "status": status}, hdr or {})


def valid_ns(ns):
    return ns.isascii() and 1 <= len(ns) <= 512 and "\x00" not in ns


def log_entry(op, index, ns, ids=(), n=None):
    with LOCK:
        STATE["seq"] += 1
        keep = STATE["chaos"]["log_ids"]
        e = {"seq": STATE["seq"], "t": now(), "op": op, "index": index, "ns": ns,
             "ids": list(ids) if keep else [], "n": len(ids) if n is None else n, "status": None}
        if not keep and ids:
            e["first"], e["last"] = ids[0], ids[-1]
        STATE["log"].append(e)
        return e


def rule_matches(r, op, ns):
    if r.get("ops") and op not in r["ops"]:
        return False
    ns = ns or ""
    if "ns" in r and r["ns"] != ns:
        return False
    if "ns_prefix" in r and not ns.startswith(r["ns_prefix"]):
        return False
    return not ("ns_re" in r and not re.search(r["ns_re"], ns))


def gate(plane, op, ns):
    """Applies latency + the first matching rule or the global mode. Returns (status, body, hdr) or None."""
    with LOCK:
        c, gen, rule = dict(STATE["chaos"]), GEN["n"], None
        for r in STATE["rules"]:
            if (r.get("hits") is None or r["hits"] > 0) and rule_matches(r, op, ns):
                rule = r
                r["fired"] = r.get("fired", 0) + 1
                if r.get("hits") is not None:
                    r["hits"] -= 1
                break
    inscope = plane in SCOPES.get(c["scope"], ())
    left = ((c["latency_ms"] if inscope else 0) + (rule.get("latency_ms", 0) if rule else 0)
            + (c["ns_delete_latency_ms"] if op == "ns_delete" else 0)) / 1000.0
    while left > 0:
        time.sleep(min(0.05, left))
        left -= 0.05
        if GEN["n"] != gen:
            return (0, None, None)
    hdr = {"Retry-After": c["retry_after"]} if c["retry_after"] is not None else {}
    mode = (rule.get("mode") if rule and rule.get("mode") else (c["mode"] if inscope else "ok"))
    if rule and mode == "custom":
        return err("CUSTOM", rule.get("message", "custom"), rule.get("status", 400), hdr)
    table = {"429rate": ("RESOURCE_EXHAUSTED", RATE_MSG, 429), "429quota": ("RESOURCE_EXHAUSTED", QUOTA_MSG, 429),
             "503": ("UNAVAILABLE", "service unavailable", 503), "500": ("UNKNOWN", "internal error", 500),
             "400": ("INVALID_ARGUMENT", "Bad request injected by the chaos rule", 400),
             "403": ("FORBIDDEN", "Forbidden injected by the chaos rule", 403)}
    if mode == "401":
        return (401, "Unauthorized", {"__text": True})
    return err(*table[mode], hdr) if mode in table else None


def cap_refusal(index, ns):
    """The namespace-cap error when `ns` would be a new namespace beyond ns_cap in `index`, else None."""
    with LOCK:
        c = STATE["chaos"]
        if not c["ns_cap"]:
            return None
        have = {k for k, v in STATE["vecs"].get(index, {}).items() if v} | {k for k, v in STATE["docs"].get(index, {}).items() if v}
        if ns in have or len(have) < c["ns_cap"]:
            return None
        bump("cap_refused")
        return err("INVALID_ARGUMENT" if c["cap_status"] == 400 else "FORBIDDEN",
                   c["cap_message"] or CAP_MSG.format(cap=c["ns_cap"]), c["cap_status"])
