"""HTTP plumbing of the partition chaos fake: threading server that tracks connections, JSON base handler."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

CONNS, CONN_LOCK = set(), threading.Lock()
STATS = {"connections": 0}


class Srv(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 256

    def get_request(self):
        conn, addr = super().get_request()
        with CONN_LOCK:
            CONNS.add(conn)
            STATS["connections"] += 1
        return conn, addr

    def shutdown_request(self, request):
        with CONN_LOCK:
            CONNS.discard(request)
        super().shutdown_request(request)


class Base(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send_json(self, status, body=None, headers=None):
        raw = json.dumps(body if body is not None else {}).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for k, v in (headers or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw) if raw else None
        except ValueError:
            return {}

    def parts(self):
        u = urlparse(self.path)
        return [unquote(p) for p in u.path.strip("/").split("/")], parse_qs(u.query)

    def answer(self, res):
        status, body, hdr = res
        if status == 0:
            self.close_connection = True
            return
        if hdr and hdr.get("__text"):
            raw = str(body).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except OSError:
                pass
            return
        self.send_json(status, body, hdr)
