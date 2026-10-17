"""Document data plane + control plane (/indexes) of the partition chaos fake."""
import re

from fakecore import LOCK, STATE, bump, cap_refusal, err, gate, log_entry, now, valid_ns
from fakefilter import match

PUBLIC = {"url": ""}


def documents(h, ns, op, body):
    body = body or {}
    docs = (body.get("documents") or []) if op == "upsert" else []
    ids = [d.get("_id") for d in docs] if op == "upsert" else list(body.get("ids") or [])
    e = log_entry("doc_" + op, h.index, ns, ids, n=len(ids) if ids else int(bool(body.get("delete_all"))))
    e["all"] = bool(body.get("delete_all"))
    with LOCK:
        gone = h.index is not None and h.index not in STATE["indexes"]
    if gone:
        e["status"] = 404
        h.send_json(404, {"error": {"code": "NOT_FOUND", "message": f"Index {h.index} not found"}, "status": 404})
    elif not valid_ns(ns):
        e["status"] = 400
        h.send_json(400, {"error": {"code": "INVALID_ARGUMENT", "message": "Namespace must be ASCII, must not contain the NUL character, and must be 1-512 characters long"}, "status": 400})
    else:
        res = gate("docs", "doc_" + op, ns) or (cap_refusal(h.index, ns) if op == "upsert" else None)
        if res:
            e["status"] = res[0] or -1
            h.answer(res)
        else:
            h.answer(_apply(h.index, ns, op, body, docs, ids, e))
    e["done"] = now()


def _apply(index, ns, op, body, docs, ids, e):
    with LOCK:
        planes = STATE["docs"].setdefault(index, {})
        if op == "upsert":
            st, acc = planes.setdefault(ns, {}), STATE["accepts"].setdefault((index, ns), {})
            for d in docs:
                st[d["_id"]] = d
                acc[d["_id"]] = acc.get(d["_id"], 0) + 1
            e["status"] = 202
            bump("doc_upsert_ok")
            return (202, {"upserted_count": len(docs)}, None)
        if op == "delete":
            st = planes.get(ns, {})
            if body.get("delete_all"):
                planes.pop(ns, None)
            for i in ids:
                st.pop(i, None)
            if ns in planes and not planes[ns]:
                planes.pop(ns)
            e["status"] = 202
            bump("doc_delete_ok")
            return (202, {}, None)
        if op == "search":
            st = planes.get(ns)
            if st is None:
                e["status"] = 404
                return (404, {"code": 5, "message": "Namespace not found"}, None)
            words = {w.lower() for c in (body.get("score_by") or []) for w in re.findall(r"\w+", str(c.get("query", "")))}
            hits = []
            for i, d in st.items():
                text = " ".join(str(v) for k, v in d.items() if k != "_id").lower()
                s = sum(text.count(w) for w in words)
                if s and match({k: v for k, v in d.items()}, body.get("filter")):
                    hits.append((s, i, d))
            hits.sort(key=lambda t: (-t[0], t[1]))
            out = [{"_id": i, "_score": float(s), **{k: v for k, v in d.items() if k != "_id"}} for s, i, d in hits[: int(body.get("top_k") or 10)]]
            e["status"], e["n"] = 200, len(out)
            return (200, {"matches": out}, None)
        e["status"] = 404
        return (404, {"error": {"message": "no route"}}, None)


def create_index(h, body):
    idx = {"name": body["name"], "host": f"{PUBLIC['url']}/idx/{body['name']}", "schema": body.get("schema"),
           "status": {"ready": True, "state": "Ready"}, "deletion_protection": "disabled", "deployment": body.get("deployment")}
    if "dimension" in body:
        idx.update({"dimension": body["dimension"], "metric": body.get("metric", "cosine"), "vector_type": "dense", "spec": body.get("spec")})
    with LOCK:
        bump("create_index")
        if body["name"] in STATE["indexes"]:
            return h.send_json(409, {"error": {"message": "exists"}, "status": 409})
        STATE["indexes"][body["name"]] = idx
    h.send_json(201, idx)
