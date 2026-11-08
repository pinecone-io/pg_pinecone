#!/usr/bin/env python3
"""Pick and check the release the upgrade drill starts from.

  pick-base.py derive        the highest tag from v0.10.0 (the first pg_pinecone release) below this tree's
                             crate version (git tag --list 'v*') that still has a published GitHub release to
                             download — a tag whose release was deleted/burned is skipped — or nothing when
                             there is none yet; $UPGRADE_BASES, if set, is printed instead
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


def derive(tags, crate, released=None):
    below = sorted(v for v in map(version, tags) if v and FIRST_IN_PLACE <= v < release(crate))
    if released is not None:  # keep only tags the drill can still download (a burned release's tag is dropped)
        below = [v for v in below if "v" + ".".join(map(str, v)) in released]
    if not below:
        if version(crate) == FIRST_IN_PLACE:
            return ""
        sys.exit(f"no published release from v0.10.0 (the first pg_pinecone release) up to below {crate}: fetch "
                 "the tags (git fetch --tags; CI checks out with fetch-depth: 0) and check `gh release list`")
    return "v" + ".".join(map(str, below[-1]))


def published_releases():
    """Tags that have a downloadable GitHub release; the drill can only start from one of these."""
    done = subprocess.run(["gh", "release", "list", "--json", "tagName", "--jq", ".[].tagName"],
                          cwd=ROOT, capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"gh release list failed: {done.stderr.strip()}")
    return set(done.stdout.split())


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
        print(override or derive(tags, crate_version(), published_releases()))
    elif argv[:1] == ["check"] and len(argv) == 2:
        check(argv[1], crate_version(), os.path.join(ROOT, "pg_pinecone", "sql"))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
