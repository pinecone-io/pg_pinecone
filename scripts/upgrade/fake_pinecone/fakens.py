"""Namespace operations of the partition chaos fake: GET /namespaces (prefix, paging), GET/DELETE /namespaces/{ns}."""
from fakecore import LOCK, STATE, bump
from fakevec import _missing, _prelude, finish


def counts(index):
    """{namespace: records} over vectors and documents of `index` (empty namespaces do not exist)."""
    with LOCK:
        out = {}
        for plane in ("vecs", "docs"):
            for ns, st in STATE[plane].get(index, {}).items():
                if st:
                    out[ns] = out.get(ns, 0) + len(st)
        return out


def list_ns(h, q):
    prefix = q.get("prefix", [""])[0]
    limit, tok = min(int(q.get("limit", ["100"])[0]), 100), q.get("paginationToken", [""])[0]
    e, res = _prelude(h, "ns_list", prefix, check_ns=False)
    if not res:
        c = counts(h.index)
        names = sorted(n for n in c if n.startswith(prefix) and (not tok or n > tok))
        page = names[:limit]
        body = {"namespaces": [{"name": n, "record_count": str(c[n])} for n in page], "total_count": len(names)}
        if len(names) > limit:
            body["pagination"] = {"next": page[-1]}
        e["status"], e["n"] = 200, len(page)
        res = (200, body, None)
    finish(h, e, res)


def describe_ns(h, ns):
    e, res = _prelude(h, "ns_describe", ns)
    if not res:
        n = counts(h.index).get(ns)
        res = _missing(e) if n is None else (200, {"name": ns, "record_count": str(n)}, None)
        e["status"] = res[0]
    finish(h, e, res)


def delete_ns(h, ns):
    e, res = _prelude(h, "ns_delete", ns)
    if not res:
        with LOCK:
            hit = [STATE[p].get(h.index, {}).pop(ns, None) for p in ("vecs", "docs")]
        if any(x is not None for x in hit):
            e["status"] = 200
            bump("ns_delete_ok")
            res = (200, {}, None)
        else:
            res = _missing(e)
    finish(h, e, res)
