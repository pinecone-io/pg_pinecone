"""Vector data plane of the partition chaos fake, reads: query, list, fetch, describe_index_stats."""
import math

from fakecore import LOCK, STATE
from fakefilter import match
from fakevec import _ns, _missing, _prelude, finish


def _score(metric, a, b):
    dot = sum(x * y for x, y in zip(a, b))
    if metric == "dotproduct":
        return dot
    if metric == "euclidean":
        return sum((x - y) ** 2 for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def query(h, body):
    ns = _ns(body.get("namespace"))
    e, res = _prelude(h, "query", ns)
    if not res:
        metric = (STATE["indexes"].get(h.index) or {}).get("metric", "cosine")
        with LOCK:
            st = STATE["vecs"].get(h.index, {}).get(ns)
            if st is None:
                res = _missing(e)
            else:
                q = body.get("vector") or []
                hits = [(_score(metric, q, list(v)), i, m) for i, (v, m) in st.items() if match(m, body.get("filter"))]
                hits.sort(key=lambda t: (t[0] if metric == "euclidean" else -t[0], t[1]))
                out = []
                for s, i, m in hits[: int(body.get("topK") or 10)]:
                    d = {"id": i, "score": s}
                    if body.get("includeMetadata") and m is not None:
                        d["metadata"] = m
                    out.append(d)
                e["status"], e["n"] = 200, len(out)
                res = (200, {"matches": out, "namespace": ns}, None)
    finish(h, e, res)


def list_ids(h, q):
    ns = _ns(q.get("namespace", [""])[0])
    e, res = _prelude(h, "list", ns)
    if not res:
        prefix, limit, tok = q.get("prefix", [""])[0], int(q.get("limit", ["100"])[0]), q.get("paginationToken", [""])[0]
        with LOCK:
            st = STATE["vecs"].get(h.index, {}).get(ns)
            if st is None:
                res = _missing(e)
            else:
                ids = sorted(i for i in st if i.startswith(prefix) and (not tok or i > tok))
                page = ids[:limit]
                body = {"vectors": [{"id": i} for i in page], "namespace": ns}
                if len(ids) > limit:
                    body["pagination"] = {"next": page[-1]}
                e["status"], e["n"] = 200, len(page)
                res = (200, body, None)
    finish(h, e, res)


def fetch(h, q):
    ns, ids = _ns(q.get("namespace", [""])[0]), q.get("ids", [])
    e, res = _prelude(h, "fetch", ns, ids)
    if not res:
        with LOCK:
            st = STATE["vecs"].get(h.index, {}).get(ns)
            if st is None:
                res = _missing(e)
            else:
                out = {i: {"id": i, "values": list(st[i][0]), "metadata": st[i][1] or {}} for i in ids if i in st}
                e["status"], e["n"] = 200, len(out)
                res = (200, {"vectors": out, "namespace": ns}, None)
    finish(h, e, res)


def stats(h, body):
    e, res = _prelude(h, "stats", "", check_ns=False)
    if not res:
        with LOCK:
            ns = {k: {"vectorCount": len(v)} for k, v in STATE["vecs"].get(h.index, {}).items() if v}
            dim = (STATE["indexes"].get(h.index) or {}).get("dimension", 0)
            e["status"], e["n"] = 200, len(ns)
            res = (200, {"namespaces": ns, "dimension": dim, "totalVectorCount": sum(x["vectorCount"] for x in ns.values())}, None)
    finish(h, e, res)
