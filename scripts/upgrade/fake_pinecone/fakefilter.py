"""Pinecone metadata filter evaluation for the chaos fake ($eq $ne $in $nin $gt $gte $lt $lte $exists $and $or).

Numbers compare as f64, like Pinecone's metadata store (an integer above 2^53 equals its neighbours).
"""


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _eq(a, b):
    return float(a) == float(b) if _num(a) and _num(b) else a == b


def _cmp(a, b, op):
    if not (_num(a) and _num(b)):
        return False
    a, b = float(a), float(b)
    return {"$gt": a > b, "$gte": a >= b, "$lt": a < b, "$lte": a <= b}[op]


def match(meta, flt):
    if not flt:
        return True
    meta = meta or {}
    for k, cond in flt.items():
        if k == "$and":
            if not all(match(meta, f) for f in cond):
                return False
        elif k == "$or":
            if not any(match(meta, f) for f in cond):
                return False
        else:
            v = meta.get(k)
            cond = cond if isinstance(cond, dict) else {"$eq": cond}
            for op, arg in cond.items():
                ok = {"$eq": lambda: v is not None and _eq(v, arg), "$ne": lambda: v is None or not _eq(v, arg),
                      "$in": lambda: v is not None and any(_eq(v, x) for x in arg),
                      "$nin": lambda: v is None or not any(_eq(v, x) for x in arg),
                      "$exists": lambda: (v is not None) == bool(arg)}.get(op) or (lambda: _cmp(v, arg, op))
                if not ok():
                    return False
    return True
