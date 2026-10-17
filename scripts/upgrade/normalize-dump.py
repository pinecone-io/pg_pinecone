#!/usr/bin/env python3
"""Normalize `pg_dump --schema-only` output (also with --binary-upgrade) so equal catalogs print equal text.

Drops what is not schema: comments, OIDs, relfrozenxid, sequence positions, `\\restrict` keys, the missing
values of ADD COLUMN ... DEFAULT and the stubs of dropped columns. Entries are sorted by (type, schema, name).
"""

import re
import sys

HEADER = re.compile(r"^-- Name: (?P<name>.*); Type: (?P<type>.*); Schema: (?P<schema>.*); Owner: .*$")
OID_CALL = re.compile(r"^SELECT pg_catalog\.(binary_upgrade_set_next_\w+|setval)\(")
# only an upgraded database prints these, some over several lines
UPGRADE_ONLY = re.compile(r"^(UPDATE pg_catalog\.pg_(class|attribute)\b|SELECT (pg_catalog\.)?binary_upgrade[._]set_missing_value\()")
DROPPED = r'"\.{8}pg\.dropped\.\d+\.{8}"'
DROPPED_STUB = re.compile(rf"^\s*{DROPPED} \w+ /\* dummy \*/,?$")
DROP_STUB = re.compile(rf"^ALTER TABLE ONLY \S+ DROP COLUMN {DROPPED};$")
EMPTY_EXTENSION = re.compile(r"^(SELECT pg_catalog\.binary_upgrade_create_empty_extension\('[^']*', '[^']*', \w+, '[^']*', )"
                             r"'\{([0-9,]*)\}', '\{((?:[^']|'')*)\}'(.*)$")
ARRAY_ELEMENT = re.compile(r'"((?:[^"\\]|\\.)*)"|([^,"]+)')
MEMBER_LINE = re.compile(r"^    (OPERATOR|FUNCTION|STORAGE) \d")


def clean(lines):
    out, skipping = [], False
    for line in lines:
        if skipping:
            skipping = not line.endswith(";")
            continue
        text = line.strip()
        if not text or text.startswith(("--", "\\restrict", "\\unrestrict")):
            continue
        if OID_CALL.match(line) or DROP_STUB.match(line):
            continue
        if UPGRADE_ONLY.match(line):
            skipping = not line.endswith(";")
            continue
        if DROPPED_STUB.match(line):
            if not line.endswith(",") and out and out[-1].endswith(","):
                out[-1] = out[-1][:-1]
            continue
        ext = EMPTY_EXTENSION.match(line)
        if ext:
            # config table OIDs and their order are not schema; the drill compares name/condition pairs
            count = len([o for o in ext.group(2).split(",") if o])
            conditions = sorted(a or b for a, b in ARRAY_ELEMENT.findall(ext.group(3)))
            line = f"{ext.group(1)}'{{{count} config tables}}', '{{{','.join(conditions)}}}'{ext.group(4)}"
        out.append(line.rstrip())
    return sort_members(out)


def sort_members(lines):
    out, run = [], []

    def flush():
        if run:
            ends_statement = run[-1].endswith(";")
            members = sorted(re.sub(r"[ ,;]+$", "", m) for m in run)
            out.extend(m + (" ," if k < len(members) - 1 else (";" if ends_statement else " ,")) for k, m in enumerate(members))
            run.clear()

    for line in lines:
        if MEMBER_LINE.match(line):
            run.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return out


def entries(text):
    found, current = [], None
    for line in text.splitlines():
        head = HEADER.match(line)
        if head:
            current = (head.group("type"), head.group("schema"), head.group("name"), [])
            found.append(current)
        elif current is not None:
            current[3].append(line)
    return [(t, s, n, clean(body)) for t, s, n, body in found]


def normalize(text):
    out = []
    for kind, schema, name, body in sorted(entries(text), key=lambda e: (e[0], e[1], e[2], e[3])):
        if body:
            out.append(f"== {kind} {schema}.{name}")
            out.extend(body)
    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: normalize-dump.py <pg_dump output file>")
    with open(sys.argv[1], encoding="utf-8") as fh:
        sys.stdout.write(normalize(fh.read()))


if __name__ == "__main__":
    main()
