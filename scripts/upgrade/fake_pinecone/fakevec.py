"""Vector data plane of the partition chaos fake, writes: upsert and delete (+ shared prelude)."""
from array import array

from fakecore import LOCK, STATE, bump, cap_refusal, gate, log_entry, now, valid_ns
from fakefilter import match


def _ns(x):
    return x or "__default__"


def _prelude(h, op, ns, ids=(), plane="vec", check_ns=True):
    """Index check, log entry, namespace validity, fault gate. Returns (entry, response-or-None)."""
    e = log_entry(op, h.index, ns, ids)
    with LOCK:
        if h.index is not None and h.index not in STATE["indexes"]:
            e["status"] = 404
            return e, (404, {"error": {"code": "NOT_FOUND", "message": f"Index {h.index} not found"}, "status": 404}, None)
    if check_ns and not valid_ns(ns):
        e["status"] = 400
        return e, (400, {"error": {"code": "INVALID_ARGUMENT", "message": "Namespace must be ASCII, must not contain the NUL character, and must be 1-512 characters long"}, "status": 400}, None)
    res = gate(plane, op, ns)
    if res:
        e["status"] = res[0] or -1
        bump(f"{op}_{e['status']}")
    return e, res


def _missing(e):
    e["status"] = 404
    return (404, {"code": 5, "message": "Namespace not found"}, None)


def finish(h, e, res):
    h.answer(res)
    e["done"] = now()


def upsert(h, body):
    ns, vecs = _ns(body.get("namespace")), body.get("vectors") or []
    e, res = _prelude(h, "upsert", ns, [v["id"] for v in vecs])
    if not res:
        res = cap_refusal(h.index, ns)
        if res:
            e["status"] = res[0]
    if not res:
        dim = (STATE["indexes"].get(h.index) or {}).get("dimension")
        bad = [v["id"] for v in vecs if dim and len(v.get("values") or []) != dim]
        if bad:
            e["status"] = 400
            res = (400, {"error": {"message": f"Vector dimension {len(vecs[0].get('values') or [])} does not match the dimension of the index {dim}"}}, None)
    if not res:
        with LOCK:
            st = STATE["vecs"].setdefault(h.index, {}).setdefault(ns, {})
            acc = STATE["accepts"].setdefault((h.index, ns), {})
            for v in vecs:
                st[v["id"]] = (array("f", v.get("values") or []), v.get("metadata"))
                acc[v["id"]] = acc.get(v["id"], 0) + 1
            e["status"] = 200
            bump("upsert_ok")
        res = (200, {"upsertedCount": len(vecs)}, None)
    finish(h, e, res)


def delete(h, body):
    ns, ids = _ns(body.get("namespace")), list(body.get("ids") or [])
    e, res = _prelude(h, "delete", ns, ids)
    e["all"], e["filter"] = bool(body.get("deleteAll")), body.get("filter")
    if not res:
        with LOCK:
            idx = STATE["vecs"].setdefault(h.index, {})
            st = idx.get(ns)
            if st is None:
                res = _missing(e)
            else:
                if body.get("deleteAll"):
                    idx.pop(ns, None)
                for i in ids:
                    st.pop(i, None)
                if body.get("filter"):
                    for i in [i for i, (_, m) in st.items() if match(m, body["filter"])]:
                        del st[i]
                if not st:
                    idx.pop(ns, None)
                e["status"] = 200
                bump("delete_ok")
                res = (200, {}, None)
    finish(h, e, res)
