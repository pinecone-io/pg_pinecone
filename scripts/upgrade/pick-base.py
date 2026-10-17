#!/usr/bin/env python3
"""Pick and check the release the upgrade drill starts from.

  pick-base.py derive        the highest release tag from v0.10.0 (the first pg_pinecone release) below this
                             tree's crate version (git tag --list 'v*'), or nothing when there is none yet;
                             $UPGRADE_BASES, if set, is printed instead
  pick-base.py check <tag>   fail unless pg_pinecone/sql chains upgrade scripts from <tag> to the crate version
"""

import collections
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
FIRST_IN_PLACE = (0, 10, 0)
SCRIPT = re.compile(r"pg_pinecone--(\d+\.\d+\.\d+)--(\d+\.\d+\.\d+)\.sql")


def version(text):
    found = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", text)
    return tuple(int(part) for part in found.groups()) if found else None


def crate_version():
    with open(os.path.join(ROOT, "pg_pinecone", "Cargo.toml"), encoding="utf-8") as fh:
        for line in fh:
            found = re.match(r'version = "(.*)"', line)
            if found:
                return found.group(1)
    sys.exit("no version in pg_pinecone/Cargo.toml")


def release(crate):
    return version(crate) or sys.exit(f"the crate version {crate} is not X.Y.Z")


def derive(tags, crate):
    below = sorted(v for v in map(version, tags) if v and FIRST_IN_PLACE <= v < release(crate))
    if not below:
        if version(crate) == FIRST_IN_PLACE:
            return ""
        sys.exit(f"no release tag from v0.10.0 (the first pg_pinecone release) up to below {crate}: fetch the tags "
                 "(git fetch --tags; CI checks out with fetch-depth: 0)")
    return "v" + ".".join(map(str, below[-1]))


def check(base, crate, sql_dir):
    if version(base) is None or not FIRST_IN_PLACE <= version(base) < release(crate):
        sys.exit(f"{base} is not a release from v0.10.0 (the first pg_pinecone release) up to below this tree's version {crate}")
    edges = collections.defaultdict(set)
    for name in os.listdir(sql_dir):
        found = SCRIPT.fullmatch(name)
        if found:
            edges[found.group(1)].add(found.group(2))
    seen, queue = set(), [base.lstrip("v")]
    while queue:
        at = queue.pop()
        if at == crate:
            return
        if at not in seen:
            seen.add(at)
            queue.extend(edges[at])
    sys.exit(f"no upgrade script path {base} -> {crate}: add pg_pinecone/sql/pg_pinecone--{base.lstrip('v')}--{crate}.sql "
             "(an empty one is fine) in the PR that bumps the version; every release pair needs one (docs/contributing.md)")


def main(argv):
    if argv[:1] == ["derive"] and len(argv) == 1:
        override = os.environ.get("UPGRADE_BASES", "").strip()
        tags = subprocess.run(["git", "-C", ROOT, "tag", "--list", "v*"], capture_output=True, text=True, check=True).stdout.split()
        print(override or derive(tags, crate_version()))
    elif argv[:1] == ["check"] and len(argv) == 2:
        check(argv[1], crate_version(), os.path.join(ROOT, "pg_pinecone", "sql"))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
