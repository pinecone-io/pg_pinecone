"""Control listener of the partition chaos fake (127.0.0.1 only): knobs, state, log, dumps for the oracle."""
from array import array

from fakecore import CHAOS0, GEN, LOCK, STATE
from fakedocs import PUBLIC
from fakehttp import STATS, Base
from fakens import counts


class Ctl(Base):
    hooks = {}  # api_start / api_stop, set by the main module

    def do_GET(self):
        parts, q = self.parts()
        one = lambda k, d="": q.get(k, [d])[0]
        with LOCK:
            if parts[0] == "__state":
                return self.send_json(200, {
                    "indexes": list(STATE["indexes"]), "counts": STATE["counts"], "seq": STATE["seq"],
                    "chaos": STATE["chaos"], "rules": STATE["rules"], "api_up": self.hooks["up"](),
                    "namespaces": {i: len(counts(i)) for i in STATE["indexes"]}, "connections": STATS["connections"]})
            if parts[0] == "__rules":
                return self.send_json(200, STATE["rules"])
            if parts[0] == "__dump":
                plane = "docs" if one("kind", "vec") == "doc" else "vecs"
                out = {}
                for ns, st in STATE[plane].get(one("index"), {}).items():
                    if not ns.startswith(one("prefix")) or not st:
                        continue
                    if one("meta"):
                        out[ns] = {i: (v[1] or {}).get(one("meta")) if plane == "vecs" else v.get(one("meta")) for i, v in st.items()}
                    else:
                        out[ns] = sorted(st)
                return self.send_json(200, out)
            if parts[0] == "__docs":
                docs = STATE["docs"].get(one("index"), {}).get(one("ns"), {})
                return self.send_json(200, {i: {k: v for k, v in d.items() if k != "_id"} for i, d in docs.items()})
            if parts[0] == "__nscount":
                c = counts(one("index"))
                return self.send_json(200, {"namespaces": sum(1 for n in c if n.startswith(one("prefix"))),
                                            "records": sum(v for n, v in c.items() if n.startswith(one("prefix")))})
            if parts[0] == "__accepts":  # per namespace, or summed over the index when ns is omitted
                accs = [v for (i, n), v in STATE["accepts"].items() if i == one("index") and (not one("ns") or n == one("ns"))]
                total, distinct = sum(sum(a.values()) for a in accs), sum(len(a) for a in accs)
                return self.send_json(200, {"distinct": distinct, "total": total, "dups": total - distinct})
            if parts[0] == "__log":
                since, limit = int(one("since", "0")), int(one("limit", "10000000"))
                out = []
                for e in STATE["log"]:
                    if e["seq"] <= since or (one("index") and e["index"] != one("index")) or (one("op") and e["op"] != one("op")):
                        continue
                    if (one("ns") and e["ns"] != one("ns")) or not e["ns"].startswith(one("ns_prefix")):
                        continue
                    out.append(e if one("ids", "1") == "1" else {**e, "ids": []})
                    if len(out) >= limit:
                        break
                return self.send_json(200, out)
        self.send_json(404, {"error": "no route"})

    def do_POST(self):
        parts, q = self.parts()
        body = self.read_body() or {}
        if parts[0] == "__chaos":
            with LOCK:
                if body.get("reset"):
                    STATE["chaos"] = dict(CHAOS0)
                STATE["chaos"].update({k: v for k, v in body.items() if k in CHAOS0})
                GEN["n"] += 1 if body.get("abort_sleepers") else 0
                return self.send_json(200, STATE["chaos"])
        if parts[0] == "__rules":
            with LOCK:
                STATE["rules"] = list(body.get("rules", [])) if "rules" in body else STATE["rules"] + [body]
                GEN["n"] += 1 if body.get("abort_sleepers") else 0
                return self.send_json(200, STATE["rules"])
        if parts[0] == "__down":
            self.hooks["stop"]()
            return self.send_json(200, {"api_up": False})
        if parts[0] == "__up":
            if not self.hooks["up"]():
                self.hooks["start"]()
            return self.send_json(200, {"api_up": True})
        if parts[0] == "__seed":  # create vectors straight into a namespace (orphans): {index, ns, ids, dim, meta}
            with LOCK:
                if body.get("kind") == "doc":  # a document namespace: {"_id": id, "title": "seeded"}
                    st = STATE["docs"].setdefault(body["index"], {}).setdefault(body["ns"], {})
                    for i in body["ids"]:
                        st[str(i)] = {"_id": str(i), "title": "seeded", "body": "seeded"}
                    return self.send_json(200, {"seeded": len(body["ids"])})
                st = STATE["vecs"].setdefault(body["index"], {}).setdefault(body["ns"], {})
                for i in body["ids"]:
                    st[str(i)] = (array("f", [1.0] * int(body.get("dim", 8))), body.get("meta"))
            return self.send_json(200, {"seeded": len(body["ids"])})
        if parts[0] == "__copy_index":  # a Pinecone backup restore: {from, to} -> new index with the same namespaces and data
            with LOCK:
                new = dict(STATE["indexes"][body["from"]])
                new.update({"name": body["to"], "host": f"{PUBLIC['url']}/idx/{body['to']}"})
                STATE["indexes"][body["to"]] = new
                for p in ("vecs", "docs"):
                    STATE[p][body["to"]] = {ns: dict(st) for ns, st in STATE[p].get(body["from"], {}).items()}
            return self.send_json(200, {"copied": body["to"]})
        if parts[0] == "__drop_ns":  # silently lose a namespace: {index, ns}
            with LOCK:
                for p in ("vecs", "docs"):
                    STATE[p].get(body["index"], {}).pop(body["ns"], None)
            return self.send_json(200, {})
        if parts[0] == "__reset":
            with LOCK:
                for k in ("vecs", "docs", "accepts", "counts"):
                    STATE[k] = {}
                STATE["log"], STATE["seq"], STATE["rules"] = [], 0, []
                STATE["chaos"] = dict(CHAOS0)
                GEN["n"] += 1
                if body.get("indexes"):
                    STATE["indexes"] = {}
            return self.send_json(200, {"reset": True})
        self.send_json(404, {"error": "no route"})
