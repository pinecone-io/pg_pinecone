"""Writes settings-<label>.json on the box: instance, OS, Postgres and pgvector packages, the pg_pinecone tarball and installed library
checksums, Python and pip freeze, dataset checksums, harness commit, and per run the Postgres settings the harness stamped.
No account id, address or key. usage: settings.py <label>"""
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request


def imds(path):
    try:
        tok = urllib.request.urlopen(urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}), timeout=2).read().decode()
        return urllib.request.urlopen(urllib.request.Request("http://169.254.169.254/latest/meta-data/" + path,
                                      headers={"X-aws-ec2-metadata-token": tok}), timeout=2).read().decode()
    except Exception:
        return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


label = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
bench = os.path.dirname(here)
pg = os.environ.get("PG_PINECONE_BENCH_PG", "/usr/lib/postgresql/17")
pkglib = sh(f"{pg}/bin/pg_config --pkglibdir")
tarballs = glob.glob(os.path.expanduser("~/pg_pinecone-*.tar.gz"))
runs = {}
for f in sorted(glob.glob(os.path.join(os.environ.get("BENCH_RESULTS_DIR", os.path.join(os.environ.get("RAW", "/data/raw"), "results")), "*.json"))):
    if ".partial" in f:
        continue
    try:
        r = json.load(open(f))
    except ValueError:
        continue
    runs[os.path.basename(f)] = {"run_name": r.get("run_name"), "started": r.get("started"), "wall_seconds": r.get("wall_seconds"), "stamps": r.get("stamps"),
                                  "postgres_overrides": (r.get("config") or {}).get("postgres"), "pg_settings": r.get("pg_settings"), "machine": r.get("machine")}
data_sums = {}
sums = os.path.join(os.environ.get("STATE", "/data/state"), "data-sha256.txt")
if os.path.exists(sums):
    data_sums = {ln.split()[1]: ln.split()[0] for ln in open(sums) if ln.strip()}
out = {
    "label": label, "captured_at": time.strftime("%F %T %Z"),
    "host": {"kernel": platform.release(), "os": sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '\"'"), "cpu": sh("lscpu | sed -n 's/^Model name: *//p'"),
             "vcpus": os.cpu_count(), "mem_kib": sh("sed -n 's/^MemTotal: *\\([0-9]*\\).*/\\1/p' /proc/meminfo"),
             "instance_type": imds("instance-type"), "az": imds("placement/availability-zone"), "ami_id": imds("ami-id"), "ami_name": None},
    "postgres": {"server": sh(f"{pg}/bin/postgres --version"),
                 "packages": sh("dpkg-query -W -f '${Package}=${Version}\\n' postgresql-17 postgresql-client-17 postgresql-17-pgvector postgresql-common")},
    "pg_pinecone": {"tarball_sha256": {os.path.basename(p): sha256(p) for p in tarballs}, "so_sha256_installed": sha256(os.path.join(pkglib, "pg_pinecone.so"))},
    "python": {"version": sys.version.split()[0], "pip_freeze": sh(f"{sys.executable} -m pip freeze").splitlines()},
    "postgres18": {"server": sh("/usr/lib/postgresql/18/bin/postgres --version"),
                   "packages": sh("dpkg-query -W -f '${Package}=${Version}\\n' postgresql-18 postgresql-client-18 postgresql-18-pgvector"),
                   "pg_pinecone_so_sha256_installed": sha256("/usr/lib/postgresql/18/lib/pg_pinecone.so")}
                  if os.path.exists("/usr/lib/postgresql/18/lib/pg_pinecone.so") else None,
    "dataset_sha256": data_sums,
    "harness_sha": open(os.path.join(bench, "HARNESS_SHA")).read().strip() if os.path.exists(os.path.join(bench, "HARNESS_SHA")) else None,
    "tuned_conf": open(os.environ["BENCH_TUNED_CONF"]).read() if os.environ.get("BENCH_TUNED_CONF") else None,
    "pinecone_api_version": {"vector data plane (pg_pinecone-client)": "2026-04", "control plane (REST helper)": "2026-07"},
    "runs": runs,
}
path = os.path.join(os.environ.get("RAW", "/data/raw"), f"settings-{label}.json")
json.dump(out, open(path, "w"), indent=1)
print(path)
