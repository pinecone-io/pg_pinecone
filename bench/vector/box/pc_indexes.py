"""List or delete the Pinecone indexes of this benchmark: every index whose name contains `pg-pinecone-bench-vec`
(the access method names its index pv-<schema>-<table>, so the mark is not a prefix). Names only, never the key.
usage: pc_indexes.py list | delete-all --yes"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from harness import remote

cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
names = remote.index_names()
mine = [n for n in names if remote.NAME_MARK in n]
print("indexes:", names, "| matching:", mine)
if cmd == "delete-all":
    if "--yes" not in sys.argv:
        raise SystemExit("add --yes")
    for n in mine:
        print("delete", n, "-> HTTP", remote.delete_index(n))
    deadline = time.time() + 600
    while time.time() < deadline and any(remote.NAME_MARK in n for n in remote.index_names()):
        time.sleep(5)
    print("final index listing:", remote.index_names())
