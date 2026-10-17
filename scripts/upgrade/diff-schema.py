#!/usr/bin/env python3
"""Propose the statements of pg_pinecone--<base>--<head>.sql from two generated schemas."""

import argparse
import collections
import os
import re
import sys
import tarfile

Tok = collections.namedtuple("Tok", "kind text start end")

SPACE = re.compile(r"\s+")
DOLLAR = re.compile(r"\$([A-Za-z_\x80-\U0010ffff][A-Za-z0-9_\x80-\U0010ffff]*)?\$")
WORD = re.compile(r"[A-Za-z0-9_\x80-\U0010ffff][A-Za-z0-9_$\x80-\U0010ffff]*")
SIMPLE_IDENT = re.compile(r"[a-z_][a-z0-9_$]*")


def end_of_block_comment(text, i):
    depth, j = 1, i + 2
    while j < len(text) and depth:
        if text.startswith("/*", j):
            depth, j = depth + 1, j + 2
        elif text.startswith("*/", j):
            depth, j = depth - 1, j + 2
        else:
            j += 1
    return j


def end_of_quoted(text, i, quote, backslash=False):
    j = i + 1
    while j < len(text):
        if backslash and text[j] == "\\":
            j += 2
        elif text.startswith(quote * 2, j):
            j += 2
        elif text[j] == quote:
            return j + 1
        else:
            j += 1
    return len(text)


def scan(text):
    toks, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        dollar = DOLLAR.match(text, i) if c == "$" else None
        if c.isspace():
            kind, j = "ws", SPACE.match(text, i).end()
        elif text.startswith("--", i):
            j = text.find("\n", i)
            kind, j = "comment", n if j < 0 else j
        elif text.startswith("/*", i):
            kind, j = "comment", end_of_block_comment(text, i)
        elif c == "'":
            escaped = bool(toks) and toks[-1].end == i and toks[-1].text in ("e", "E")
            kind, j = "str", end_of_quoted(text, i, "'", escaped)
        elif c == '"':
            kind, j = "qident", end_of_quoted(text, i, '"')
        elif dollar:
            close = text.find(dollar.group(0), dollar.end())
            kind, j = "dstr", n if close < 0 else close + len(dollar.group(0))
        elif WORD.match(text, i):
            kind, j = "word", WORD.match(text, i).end()
        else:
            kind, j = "punct", i + 1
        toks.append(Tok(kind, text[i:j], i, j))
        i = j
    return toks


def significant(toks):
    return [t for t in toks if t.kind not in ("ws", "comment")]


def unquote(tok):
    if tok.kind == "qident":
        name = tok.text[1:-1].replace('""', '"')
        return name if SIMPLE_IDENT.fullmatch(name) else tok.text
    return tok.text.lower()


TYPE_ALIASES = {
    "int": "integer", "int4": "integer", "int8": "bigint", "int2": "smallint", "bool": "boolean",
    "float8": "double precision", "float4": "real", "varchar": "character varying", "decimal": "numeric",
    "timestamptz": "timestamp with time zone", "timetz": "time with time zone",
}
TYPE_START = {"double", "character", "bit", "national", "timestamp", "time", "interval"}
TYPE_SECOND = {"precision", "varying", "with", "without", "character", "year", "month", "day", "hour",
               "minute", "second", "to"}


def glued(prev, tok):
    if tok.kind == "punct" and tok.text in "()[],.":
        return True
    if prev.kind == "punct" and prev.text in "([.":
        return True
    return prev.kind == "punct" and tok.kind == "punct" and prev.text not in ",)]"


def render(toks, lang=None):
    out, prev = "", None
    for t in significant(toks):
        if t.kind in ("word", "qident"):
            s = unquote(t)
        elif t.kind == "dstr" and lang in ("plpgsql", "sql"):
            tag = DOLLAR.match(t.text).group(0)
            s = f"{tag} {render(scan(t.text[len(tag):-len(tag)]))} {tag}"
        else:
            s = t.text
        if out and not glued(prev, t):
            out += " "
        out += s
        prev = t
    return out


def render_type(toks):
    return re.sub(r"\b[a-z0-9_]+\b", lambda m: TYPE_ALIASES.get(m.group(0), m.group(0)), render(toks))


def language(toks):
    for a, b in zip(toks, toks[1:]):
        if a.kind == "word" and a.text.lower() == "language":
            return unquote(b)
    return None


def split_top(toks, sep=","):
    out, depth, cur = [], 0, []
    for t in toks:
        if t.kind == "punct" and t.text in "([":
            depth += 1
        elif t.kind == "punct" and t.text in ")]":
            depth -= 1
        elif t.kind == "punct" and t.text == sep and depth == 0:
            out.append(cur)
            cur = []
            continue
        cur.append(t)
    if cur:
        out.append(cur)
    return out


def match_paren(toks, i):
    depth = 0
    for k in range(i, len(toks)):
        if toks[k].kind == "punct" and toks[k].text == "(":
            depth += 1
        elif toks[k].kind == "punct" and toks[k].text == ")":
            depth -= 1
            if depth == 0:
                return k
    return len(toks) - 1


def take_name(toks, i):
    parts = []
    while i < len(toks) and toks[i].kind in ("word", "qident"):
        parts.append(unquote(toks[i]))
        i += 1
        if i + 1 < len(toks) and toks[i].kind == "punct" and toks[i].text == ".":
            i += 1
            continue
        break
    return ".".join(parts), i


def leading_words(toks, n=8):
    out = []
    for t in toks[:n]:
        if t.kind != "word":
            break
        out.append(t.text.upper())
    return out


Param = collections.namedtuple("Param", "mode name type default")
CLAUSES = {"language", "immutable", "stable", "volatile", "strict", "called", "returns", "parallel", "security",
           "cost", "rows", "set", "as", "window", "leakproof", "not", "support", "transform", "external"}


def parse_param(toks):
    toks, default, depth = significant(toks), None, 0
    for k, t in enumerate(toks):
        if t.kind == "punct" and t.text in "([":
            depth += 1
        elif t.kind == "punct" and t.text in ")]":
            depth -= 1
        elif depth == 0 and (t.text == "=" or (t.kind == "word" and t.text.upper() == "DEFAULT")):
            default, toks = render(toks[k + 1:]), toks[:k]
            break
    mode = "in"
    if toks and toks[0].kind == "word" and toks[0].text.lower() in ("in", "out", "inout", "variadic"):
        mode, toks = toks[0].text.lower(), toks[1:]
    name = None
    if len(toks) >= 2 and toks[1].kind in ("word", "qident"):
        type_only = toks[0].text.lower() in TYPE_START and toks[1].text.lower() in TYPE_SECOND
        if not type_only:
            name, toks = unquote(toks[0]), toks[1:]
    return Param(mode, name, render_type(toks), default)


def parse_signature(toks, i):
    name, i = take_name(toks, i)
    if i >= len(toks) or toks[i].text != "(":
        return name, None, i
    close = match_paren(toks, i)
    return name, [parse_param(p) for p in split_top(toks[i + 1:close])], close + 1


def func_key(name, params):
    return f"{name}({', '.join(p.type for p in params if p.mode != 'out')})"


def returns_clause(toks):
    for k, t in enumerate(toks):
        if t.kind == "word" and t.text.lower() == "returns":
            depth, end = 0, k + 1
            while end < len(toks):
                u = toks[end]
                if u.kind == "punct" and u.text in "([":
                    depth += 1
                elif u.kind == "punct" and u.text in ")]":
                    depth -= 1
                elif depth == 0 and u.kind == "word" and u.text.lower() in CLAUSES:
                    break
                end += 1
            return render_type(toks[k + 1:end])
    return ""


class Stmt:
    def __init__(self, text, toks):
        self.toks = toks
        self.origin = toks[0].start
        self.text = text[toks[0].start:toks[-1].end]
        self.words = leading_words(toks)
        self.norm = render(toks, "plpgsql" if self.words[:1] == ["DO"] else language(toks))

    def slice(self, toks):
        return self.text[toks[0].start - self.origin:toks[-1].end - self.origin]


def statements(text):
    out, cur = [], []
    for t in significant(scan(text)):
        if t.kind == "punct" and t.text == ";":
            if cur:
                out.append(Stmt(text, cur))
            cur = []
        else:
            cur.append(t)
    if cur:
        out.append(Stmt(text, cur))
    return out


class Func:
    def __init__(self, stmt):
        toks = stmt.toks
        i = next(k for k, t in enumerate(toks) if t.text.upper() in ("FUNCTION", "PROCEDURE"))
        self.kind = toks[i].text.upper()
        self.name, self.params, end = parse_signature(toks, i + 1)
        self.key = func_key(self.name, self.params)
        outs = ", ".join(f"{p.name} {p.type}" for p in self.params if p.mode == "out")
        self.result = (returns_clause(toks[end:]) + (f" out({outs})" if outs else "")).strip()
        self.norm = render(toks[i:], language(toks))
        self.stmt, self.attrs = stmt, []

    @property
    def in_params(self):
        return [p for p in self.params if p.mode != "out"]

    def replace_blocker(self, old):
        if self.result != old.result:
            return f"the result changed ({old.result} -> {self.result})"
        for a, b in zip(old.in_params, self.in_params):
            if a.name and a.name != b.name:
                return f'input parameter "{a.name}" is now "{b.name}"'
        had = sum(p.default is not None for p in old.in_params)
        has = sum(p.default is not None for p in self.in_params)
        return "parameter defaults were removed" if has < had else None


COL_START = {"CONSTRAINT", "NOT", "NULL", "DEFAULT", "CHECK", "PRIMARY", "UNIQUE", "REFERENCES", "GENERATED", "COLLATE"}
TABLE_ELEMENT = {"CONSTRAINT", "PRIMARY", "UNIQUE", "CHECK", "FOREIGN", "EXCLUDE", "LIKE"}


def column_chunks(toks):
    chunks, cur, depth = [], [], 0
    for t in toks:
        if t.kind == "punct" and t.text in "([":
            depth += 1
        elif t.kind == "punct" and t.text in ")]":
            depth -= 1
        starts = depth == 0 and t.kind == "word" and t.text.upper() in COL_START
        if starts and cur and not (t.text.upper() == "NULL" and cur[-1].text.upper() == "NOT"):
            chunks.append(cur)
            cur = []
        cur.append(t)
    if cur:
        chunks.append(cur)
    return chunks


class Col:
    def __init__(self, toks, slicer):
        self.name = unquote(toks[0])
        k = 1
        while k < len(toks) and not (toks[k].kind == "word" and toks[k].text.upper() in COL_START):
            k += 1
        self.type = render_type(toks[1:k])
        self.notnull, self.default, self.checks, self.other = False, None, [], []
        self.check_text = {}
        pending = None
        for chunk in column_chunks(toks[k:]):
            head = chunk[0].text.upper()
            if head == "CONSTRAINT":
                pending = unquote(chunk[1])
                continue
            text = render(chunk)
            if head == "NOT":
                self.notnull = True
            elif head == "NULL":
                self.notnull = False
            elif head == "DEFAULT":
                self.default = render(chunk[1:])
            elif head == "CHECK":
                self.checks.append((pending, text))
                self.check_text[text] = slicer(chunk)
            else:
                self.other.append(text if pending is None else f"constraint {pending} {text}")
            pending = None
        self.other.sort()

    def sig(self):
        return (self.type, self.notnull, self.default, tuple(self.checks), tuple(self.other))


class Table:
    def __init__(self, stmt):
        toks = stmt.toks
        i = stmt.words.index("TABLE") + 1
        while toks[i].text.upper() in ("IF", "NOT", "EXISTS"):
            i += 1
        self.name, i = take_name(toks, i)
        self.stmt = stmt
        self.cols, self.col_text, self.constraints = {}, {}, {}
        for el in split_top(toks[i + 1:match_paren(toks, i)]):
            if el[0].kind == "word" and el[0].text.upper() in TABLE_ELEMENT:
                self.constraints[render(el)] = stmt.slice(el)
            else:
                col = Col(el, stmt.slice)
                self.cols[col.name] = col
                self.col_text[col.name] = stmt.slice(el)


class Index:
    def __init__(self, stmt):
        toks = stmt.toks
        i = stmt.words.index("INDEX") + 1
        while toks[i].text.upper() in ("CONCURRENTLY", "IF", "NOT", "EXISTS"):
            i += 1
        name, i = take_name(toks, i)
        self.table = ""
        if toks[i].text.upper() == "ON":
            i += 1
            if toks[i].text.upper() == "ONLY":
                i += 1
            self.table = take_name(toks, i)[0]
        schema = self.table.rpartition(".")[0]
        self.name = f"{schema}.{name}" if schema else name
        self.stmt, self.norm = stmt, stmt.norm


MODIFIERS = {"OR", "REPLACE", "TEMP", "TEMPORARY", "UNLOGGED", "UNIQUE", "RECURSIVE", "GLOBAL", "LOCAL"}
TWO_WORD = {("EVENT", "TRIGGER"), ("ACCESS", "METHOD"), ("OPERATOR", "CLASS"), ("OPERATOR", "FAMILY"),
            ("MATERIALIZED", "VIEW"), ("FOREIGN", "TABLE"), ("FOREIGN", "DATA"), ("TEXT", "SEARCH"),
            ("PROCEDURAL", "LANGUAGE")}


def hand_label(st, i, kind):
    kind_words = list(kind) if isinstance(kind, tuple) else [kind]
    if kind_words in (["TEXT", "SEARCH"], ["FOREIGN", "DATA"]) and len(st.words) > i + 2:
        kind_words.append(st.words[i + 2])
    toks, j = st.toks, i + len(kind_words)
    while j < len(toks) and toks[j].text.upper() in ("IF", "NOT", "EXISTS"):
        j += 1
    if j < len(toks) and toks[j].kind in ("word", "qident"):
        label = take_name(toks, j)[0]
    elif j < len(toks) and toks[j].text == "(":
        label = render(toks[j:match_paren(toks, j) + 1])
    else:
        end = next((k for k in range(j, len(toks)) if toks[k].text == "("), len(toks))
        label = render(toks[j:end])
    if kind_words[-1] in ("CLASS", "FAMILY"):
        using = next((k for k, t in enumerate(toks) if t.text.upper() == "USING"), None)
        if using is not None and using + 1 < len(toks):
            label += f" using {unquote(toks[using + 1])}"
    return f"{' '.join(w.lower() for w in kind_words)} {label}".strip()


def function_ref(st):
    if st.words[:1] not in (["ALTER"], ["REVOKE"], ["GRANT"], ["COMMENT"]):
        return None
    toks = st.toks
    for k in range(1, len(toks)):
        if toks[k].text.upper() in ("FUNCTION", "PROCEDURE") and toks[k - 1].text.upper() in ("ALTER", "ON"):
            return k + 1
    return None


class Schema:
    def __init__(self, text):
        self.funcs, self.tables, self.indexes = {}, {}, {}
        self.event_triggers, self.views, self.hand, self.loose = {}, {}, {}, []
        refs = []
        for st in statements(text):
            if st.words[:1] == ["CREATE"]:
                self.add_create(st)
            elif function_ref(st) is not None:
                refs.append(st)
            else:
                self.loose.append(st)
        by_name = collections.defaultdict(list)
        for f in self.funcs.values():
            by_name[f.name].append(f)
        for st in refs:
            owner = self.owner_of(st, by_name)
            (owner.attrs if owner else self.loose).append(st)

    def add_create(self, st):
        w = st.words
        i = 1
        while i < len(w) and w[i] in MODIFIERS:
            i += 1
        two = tuple(w[i:i + 2])
        kind = two if two in TWO_WORD else (w[i] if i < len(w) else "")
        if kind in ("FUNCTION", "PROCEDURE"):
            f = Func(st)
            self.funcs[f.key] = f
        elif kind == "TABLE":
            t = Table(st)
            self.tables[t.name] = t
        elif kind == "INDEX":
            ix = Index(st)
            self.indexes[ix.name] = ix
        elif kind == ("EVENT", "TRIGGER"):
            self.event_triggers[take_name(st.toks, i + 2)[0]] = st
        elif kind == "VIEW":
            self.views[take_name(st.toks, i + 1)[0]] = st
        else:
            self.hand[hand_label(st, i, kind)] = st

    @staticmethod
    def owner_of(st, by_name):
        name, params, _ = parse_signature(st.toks, function_ref(st))
        if params is None:
            candidates = by_name.get(name, [])
            return candidates[0] if len(candidates) == 1 else None
        return next((f for f in by_name.get(name, []) if f.key == func_key(name, params)), None)


def read_schema(path):
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tar:
            found = [m for m in tar.getmembers()
                     if m.isfile() and re.fullmatch(r"pg_pinecone--(?!.*--).+\.sql", os.path.basename(m.name))]
            if len(found) != 1:
                sys.exit(f"{path}: expected one pg_pinecone--<version>.sql in the tarball, found {len(found)}")
            return tar.extractfile(found[0]).read().decode("utf-8")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class Plan:
    def __init__(self):
        self.steps = collections.defaultdict(list)
        self.hand = []
        self.counts = collections.Counter()
        self.bundled = set()

    def add(self, step, *lines):
        self.steps[step].extend(lines)


def unmatched(items, others):
    have = collections.Counter(s.norm for s in others)
    out = []
    for s in sorted(items, key=lambda s: s.norm):
        if have[s.norm] > 0:
            have[s.norm] -= 1
        else:
            out.append(s)
    return out


def bundle_for(table, schema):
    bare = table.rpartition(".")[2]
    found = [ix.stmt for _, ix in sorted(schema.indexes.items()) if ix.table == table]
    for st in sorted(schema.loose, key=lambda s: s.norm):
        dump = re.search(r"pg_extension_config_dump\('([^']+)'", st.norm)
        if dump and (dump.group(1) == table or dump.group(1).rpartition(".")[2].startswith(bare + "_")):
            found.append(st)
        elif st.words[:1] in (["REVOKE"], ["GRANT"]) and re.search(rf"\bon (table )?{re.escape(table)}\b", st.norm):
            found.append(st)
        elif st.words[:2] == ["ALTER", "TABLE"] and re.match(rf"alter table (only )?{re.escape(table)}\b", st.norm):
            found.append(st)
    return found


def changed_column(table, name, b, h):
    alter = f"ALTER TABLE {table} ALTER COLUMN {name}"
    lines = []
    if b.type != h.type:
        lines.append(f"-- CHECK BY HAND: {table}.{name}: type {b.type} -> {h.type}; write the conversion (ALTER COLUMN .. TYPE .. USING ..)")
    if b.default != h.default:
        lines += [f"-- CHECK BY HAND: {table}.{name}: default changed",
                  f"{alter} SET DEFAULT {h.default};" if h.default is not None else f"{alter} DROP DEFAULT;"]
    if b.notnull != h.notnull:
        lines += [f"-- CHECK BY HAND: {table}.{name}: NOT NULL " + ("added, fails on existing NULLs" if h.notnull else "dropped"),
                  f"{alter} SET NOT NULL;" if h.notnull else f"{alter} DROP NOT NULL;"]
    if b.checks != h.checks:
        if len(h.checks) == 1 and len(b.checks) <= 1:
            cname = h.checks[0][0] or f"{table.rpartition('.')[2]}_{name}_check"
            expr = h.check_text[h.checks[0][1]]
            lines += [f"-- CHECK BY HAND: {table}.{name}: CHECK changed, existing rows are validated",
                      f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {cname};",
                      f"ALTER TABLE {table} ADD CONSTRAINT {cname} {expr};"]
        else:
            lines.append(f"-- CHECK BY HAND: {table}.{name}: CHECK constraints changed {[c[1] for c in b.checks]} -> {[c[1] for c in h.checks]}")
    if b.other != h.other:
        lines.append(f"-- CHECK BY HAND: {table}.{name}: column constraints changed {b.other} -> {h.other}")
    return lines


def changed_table(name, b, h):
    lines = []
    order = list(h.cols)
    for pos, cn in enumerate(order):
        col = h.cols[cn]
        if cn in b.cols:
            if b.cols[cn].sig() != col.sig():
                lines += changed_column(name, cn, b.cols[cn], col)
            continue
        risky = " (NOT NULL without a DEFAULT fails on existing rows)" if col.notnull and col.default is None else ""
        lines.append(f'-- CHECK BY HAND: {name}: new column "{cn}"{risky}')
        before = next((c for c in order[pos + 1:] if c in b.cols), None)
        if before:
            lines.append(f'-- CHECK BY HAND: "{cn}" precedes existing column "{before}" in HEAD; ADD COLUMN appends, so move it last '
                         "in the Rust CREATE TABLE (the catalog compare checks column order)")
        lines.append(f"ALTER TABLE {name} ADD COLUMN IF NOT EXISTS {h.col_text[cn]};")
    for cn in b.cols:
        if cn not in h.cols:
            lines += [f'-- CHECK BY HAND: {name}: column "{cn}" is gone in HEAD; dropping it loses data',
                      f"-- ALTER TABLE {name} DROP COLUMN IF EXISTS {cn};"]
    for text in sorted(h.constraints.keys() - b.constraints.keys()):
        lines += [f"-- CHECK BY HAND: {name}: new table constraint, existing rows are validated",
                  f"ALTER TABLE {name} ADD {h.constraints[text]};"]
    for text in sorted(b.constraints.keys() - h.constraints.keys()):
        lines.append(f"-- CHECK BY HAND: {name}: table constraint gone in HEAD (find its name in pg_constraint): {text}")
    return lines


def diff_tables(base, head, plan):
    for name in sorted(head.tables.keys() - base.tables.keys()):
        bundle = bundle_for(name, head)
        plan.bundled.update(id(s) for s in bundle)
        plan.counts["tables new"] += 1
        plan.add("tables", f"-- CHECK BY HAND: new table {name}; its indexes, grants and extension config dump come with it",
                 head.tables[name].stmt.text + ";", *[s.text + ";" for s in bundle], "")
    for name in sorted(base.tables.keys() - head.tables.keys()):
        plan.counts["tables dropped"] += 1
        plan.add("tables", f"-- CHECK BY HAND: table {name} is gone in HEAD; dropping it loses its rows",
                 f"-- DROP TABLE IF EXISTS {name};", "")
    for name in sorted(base.tables.keys() & head.tables.keys()):
        lines = changed_table(name, base.tables[name], head.tables[name])
        if lines:
            plan.counts["tables changed"] += 1
            plan.add("tables", *lines, "")


def create_text(func, replace=False):
    keyword = "CREATE OR REPLACE " if replace else "CREATE "
    return re.sub(r"^CREATE\s+(OR\s+REPLACE\s+)?", keyword, func.stmt.text, flags=re.I) + ";"


def diff_functions(base, head, plan):
    new_names = {head.funcs[k].name for k in head.funcs.keys() - base.funcs.keys()}
    for key in sorted(base.funcs.keys() - head.funcs.keys()):
        f = base.funcs[key]
        plan.counts["functions dropped"] += 1
        if f.name in new_names:
            plan.add("drop functions", f"-- CHECK BY HAND: {f.name} changed signature; callers of the old one break, "
                     "keep it as an overload unless that is intended")
        plan.add("drop functions", f"DROP {f.kind} IF EXISTS {key};")
    for key in sorted(base.funcs.keys() & head.funcs.keys()):
        b, h = base.funcs[key], head.funcs[key]
        added = unmatched(h.attrs, b.attrs)
        if b.norm == h.norm:
            plan.add("function attributes", *[s.text + ";" for s in added])
        elif h.replace_blocker(b):
            plan.counts["functions recreated"] += 1
            plan.add("drop functions", f"-- CHECK BY HAND: {key}: CREATE OR REPLACE cannot apply, {h.replace_blocker(b)}; "
                     "dependants (event triggers, opclasses) block the DROP",
                     f"DROP {h.kind} IF EXISTS {key};")
            plan.add("create functions", create_text(h), *[s.text + ";" for s in sorted(h.attrs, key=lambda s: s.norm)], "")
        else:
            plan.counts["functions changed"] += 1
            reapply = [s for s in sorted(h.attrs, key=lambda s: s.norm) if s.words[0] == "ALTER" or s in added]
            plan.add("replace functions", create_text(h, replace=True), *[s.text + ";" for s in reapply], "")
        for s in unmatched(b.attrs, h.attrs):
            if b.norm == h.norm or (s.words[0] != "ALTER" and not h.replace_blocker(b)):
                plan.add("function attributes", f"-- CHECK BY HAND: {key}: BASE had it, HEAD does not: {s.norm}")
    for key in sorted(head.funcs.keys() - base.funcs.keys()):
        h = head.funcs[key]
        plan.counts["functions new"] += 1
        plan.add("create functions", create_text(h), *[s.text + ";" for s in sorted(h.attrs, key=lambda s: s.norm)], "")


def create_index_text(ix):
    def add_guard(m):
        return f"CREATE {m.group(1) or ''}INDEX IF NOT EXISTS "
    return re.sub(r"^CREATE\s+(UNIQUE\s+)?INDEX\s+(IF\s+NOT\s+EXISTS\s+)?", add_guard, ix.stmt.text, flags=re.I) + ";"


def diff_indexes(base, head, plan):
    new_tables = head.tables.keys() - base.tables.keys()
    for name in sorted(base.indexes.keys() - head.indexes.keys()):
        plan.counts["indexes dropped"] += 1
        plan.add("indexes", f"DROP INDEX IF EXISTS {name};")
    for name in sorted(base.indexes.keys() & head.indexes.keys()):
        if base.indexes[name].norm != head.indexes[name].norm:
            plan.counts["indexes changed"] += 1
            plan.add("indexes", f"-- CHECK BY HAND: {name} is rebuilt, which locks its table",
                     f"DROP INDEX IF EXISTS {name};", create_index_text(head.indexes[name]))
    for name in sorted(head.indexes.keys() - base.indexes.keys()):
        if head.indexes[name].table not in new_tables:
            plan.counts["indexes new"] += 1
            plan.add("indexes", create_index_text(head.indexes[name]))


def diff_event_triggers(base, head, plan):
    for name in sorted(base.event_triggers.keys() - head.event_triggers.keys()):
        plan.counts["event triggers dropped"] += 1
        plan.add("drop event triggers", f"DROP EVENT TRIGGER IF EXISTS {name};")
    for name in sorted(base.event_triggers.keys() & head.event_triggers.keys()):
        if base.event_triggers[name].norm != head.event_triggers[name].norm:
            plan.counts["event triggers changed"] += 1
            plan.add("drop event triggers", f"DROP EVENT TRIGGER IF EXISTS {name};")
            plan.add("create event triggers", head.event_triggers[name].text + ";")
    for name in sorted(head.event_triggers.keys() - base.event_triggers.keys()):
        plan.counts["event triggers new"] += 1
        plan.add("create event triggers", head.event_triggers[name].text + ";")


def diff_views(base, head, plan):
    for name in sorted(base.views.keys() - head.views.keys()):
        plan.counts["views dropped"] += 1
        plan.add("views", f"DROP VIEW IF EXISTS {name};")
    for name in sorted(base.views.keys() & head.views.keys()):
        if base.views[name].norm != head.views[name].norm:
            plan.counts["views changed"] += 1
            text = re.sub(r"^CREATE\s+(OR\s+REPLACE\s+)?", "CREATE OR REPLACE ", head.views[name].text, flags=re.I)
            plan.add("views", "-- CHECK BY HAND: CREATE OR REPLACE VIEW only appends columns; otherwise DROP VIEW first", text + ";")
    for name in sorted(head.views.keys() - base.views.keys()):
        plan.counts["views new"] += 1
        plan.add("views", head.views[name].text + ";")


def diff_hand(base, head, plan):
    for label in sorted(head.hand.keys() - base.hand.keys()):
        plan.hand.append((f"new: {label}", head.hand[label].text))
    for label in sorted(base.hand.keys() & head.hand.keys()):
        if base.hand[label].norm != head.hand[label].norm:
            plan.hand.append((f"changed: {label}", f"was: {base.hand[label].text}\nnow: {head.hand[label].text}"))
    for label in sorted(base.hand.keys() - head.hand.keys()):
        plan.hand.append((f"removed: {label}", base.hand[label].text))
    if plan.hand:
        plan.counts["hand-written"] = len(plan.hand)


def diff_loose(base, head, plan):
    for st in unmatched([s for s in head.loose if id(s) not in plan.bundled], base.loose):
        plan.counts["statements new"] += 1
        plan.add("other statements", "-- CHECK BY HAND: new in HEAD", st.text + ";")
    for st in unmatched(base.loose, head.loose):
        plan.counts["statements gone"] += 1
        plan.add("other statements", f"-- CHECK BY HAND: in BASE, not in HEAD: {st.norm}")


def install_calls(head):
    defined = {f.name for f in head.funcs.values()}
    found = set()
    for st in head.loose:
        call = re.fullmatch(r"select ([a-z0-9_.]+)\(\)", st.norm)
        if call and call.group(1) in defined:
            found.add(call.group(1))
    return sorted(found)


STEPS = [
    ("tables", "TABLES"),
    ("indexes", "INDEXES"),
    ("drop event triggers", "EVENT TRIGGERS: drop"),
    ("drop functions", "FUNCTIONS: drop"),
    ("replace functions", "FUNCTIONS: replace"),
    ("create functions", "FUNCTIONS: create"),
    ("function attributes", "FUNCTIONS: grants and attributes"),
    ("create event triggers", "EVENT TRIGGERS: create"),
    ("views", "VIEWS"),
    ("other statements", "OTHER STATEMENTS"),
]


def propose(base_text, head_text, base_name="base", head_name="head"):
    base, head, plan = Schema(base_text), Schema(head_text), Plan()
    diff_tables(base, head, plan)
    diff_indexes(base, head, plan)
    diff_event_triggers(base, head, plan)
    diff_functions(base, head, plan)
    diff_views(base, head, plan)
    diff_hand(base, head, plan)
    diff_loose(base, head, plan)
    out = [f"-- diff-schema.py: base={base_name} head={head_name}",
           "-- Proposal for pg_pinecone/sql/pg_pinecone--<prev>--<new>.sql: review every statement, nothing is applied for you."]
    if not any(plan.steps.values()) and not plan.hand:
        out.append("-- No schema differences: the upgrade script needs no statements.")
    else:
        out.append("-- Summary: " + ", ".join(f"{k} {v}" for k, v in sorted(plan.counts.items())))
    for key, title in STEPS:
        lines = list(plan.steps.get(key, []))
        while lines and lines[-1] == "":
            lines.pop()
        if lines:
            out += ["", f"-- === {title} ==="] + lines
    if plan.hand:
        out += ["", "-- === HAND-WRITTEN NEEDED: operators, opclasses, access method, types and other objects ==="]
        for title, text in plan.hand:
            out += [f"-- {title}"] + [f"--   {ln}" for ln in text.splitlines()]
    calls = install_calls(head)
    if calls:
        out += ["", "-- === NOTE: install-time calls in HEAD's schema ===",
                "-- A Rust change behind one of these is invisible to a schema diff; repeat the idempotent ones when it changed:"]
        out += [f"--   SELECT {c}();" for c in calls]
    return "\n".join(out) + "\n"


EPILOG = """\
HEAD:  (cd pg_pinecone && cargo pgrx schema pg17 --out /tmp/head.sql)
BASE:  a release tarball carries its schema, no build needed:
         gh release download v0.9.0 -p 'pg_pinecone-0.9.0-pg17-linux-x86_64.tar.gz' -D /tmp/pg_pinecone-upgrade-base/v0.9.0
         diff-schema.py /tmp/pg_pinecone-upgrade-base/v0.9.0/pg_pinecone-0.9.0-pg17-linux-x86_64.tar.gz /tmp/head.sql
       (scripts/upgrade-drill.sh downloads and caches the same file). From a tag build instead:
         git worktree add /tmp/pg_pinecone-base v0.9.0
         (cd /tmp/pg_pinecone-base/pg_pinecone && cargo pgrx schema pg17 --out /tmp/pg_pinecone-upgrade-base/v0.9.0/schema.sql)
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="schema of the previous release (cargo pgrx schema output or a release tarball)")
    ap.add_argument("head", help="schema of this tree (cargo pgrx schema output)")
    args = ap.parse_args(argv)
    sys.stdout.write(propose(read_schema(args.base), read_schema(args.head), os.path.basename(args.base), os.path.basename(args.head)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
