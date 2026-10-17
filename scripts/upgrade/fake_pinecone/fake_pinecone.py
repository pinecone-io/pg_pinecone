"""Fault-injecting stand-in for Pinecone (control plane, vector + namespace + document data planes) for the
partition chaos suite. Two listeners, both 127.0.0.1 only: API on P, CONTROL on P+1 (see fakectl.py).
`__down` really closes the API listener (connection refused); state survives down/up, not a restart.
Every API request is logged in arrival order with the status it was answered with (`__log`).
"""
import socket
import sys
import threading

import fakedocs
import fakens
import fakeread
import fakevec
from fakecore import GEN, LOCK, STATE, bump, gate
from fakectl import Ctl
from fakehttp import CONN_LOCK, CONNS, Base, Srv

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18190
API = {"server": None}
fakedocs.PUBLIC["url"] = f"http://127.0.0.1:{PORT}"
NO_ROUTE = {"error": {"message": "no route"}}


class Api(Base):
    index = None

    def parts(self):
        parts, q = super().parts()
        self.index = None
        if parts[0] == "idx" and len(parts) >= 3:
            self.index, parts = parts[1], parts[2:]
        return parts, q

    def do_GET(self):
        parts, q = self.parts()
        if parts[0] == "vectors" and len(parts) == 2 and parts[1] in ("list", "fetch"):
            return (fakeread.list_ids if parts[1] == "list" else fakeread.fetch)(self, q)
        if parts[0] == "namespaces" and len(parts) <= 2:
            return fakens.list_ns(self, q) if len(parts) == 1 else fakens.describe_ns(self, parts[1])
        if parts[0] == "indexes" and len(parts) == 2:
            res = gate("ctl", "ctl_describe", "")
            if res:
                return self.answer(res)
            with LOCK:
                bump("describe")
                idx = STATE["indexes"].get(parts[1])
            if idx is None:
                return self.send_json(404, {"error": {"code": "NOT_FOUND", "message": "not found"}, "status": 404})
            return self.send_json(200, idx)
        self.send_json(404, NO_ROUTE)

    def do_DELETE(self):
        parts, _ = self.parts()
        if parts[0] == "namespaces" and len(parts) == 2:
            return fakens.delete_ns(self, parts[1])
        if parts[0] == "indexes" and len(parts) == 2:
            res = gate("ctl", "ctl_delete", "")
            if res:
                return self.answer(res)
            with LOCK:
                existed = STATE["indexes"].pop(parts[1], None) is not None
                STATE["vecs"].pop(parts[1], None)
                STATE["docs"].pop(parts[1], None)
                bump("delete_index")
            return self.send_json(202 if existed else 404, {})
        self.send_json(404, NO_ROUTE)

    def do_POST(self):
        parts, _ = self.parts()
        body = self.read_body()
        if parts[0] == "indexes" and len(parts) == 1:
            res = gate("ctl", "ctl_create", "")
            return self.answer(res) if res else fakedocs.create_index(self, body)
        if parts[0] == "namespaces" and len(parts) == 4 and parts[2] == "documents":
            return fakedocs.documents(self, parts[1], parts[3], body)
        if parts[0] == "vectors" and len(parts) == 2 and parts[1] in ("upsert", "delete"):
            return (fakevec.upsert if parts[1] == "upsert" else fakevec.delete)(self, body or {})
        if parts[0] == "vectors" and len(parts) == 2 and parts[1] == "update":
            return self.send_json(200, {})
        if parts[0] == "query":
            return fakeread.query(self, body or {})
        if parts[0] == "describe_index_stats":
            return fakeread.stats(self, body or {})
        self.send_json(404, NO_ROUTE)


def api_start():
    srv = Srv(("127.0.0.1", PORT), Api)
    API["server"] = srv
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()


def api_stop():
    srv = API["server"]
    if srv is None:
        return
    GEN["n"] += 1
    API["server"] = None
    srv.shutdown()
    srv.server_close()
    with CONN_LOCK:
        conns = list(CONNS)
    for c in conns:
        try:
            c.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


if __name__ == "__main__":
    Ctl.hooks.update({"start": api_start, "stop": api_stop, "up": lambda: API["server"] is not None})
    api_start()
    print(f"fake pinecone: api http://127.0.0.1:{PORT}  control http://127.0.0.1:{PORT + 1}", flush=True)
    Srv(("127.0.0.1", PORT + 1), Ctl).serve_forever()
