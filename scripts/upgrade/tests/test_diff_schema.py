import importlib.util
import os
import random
import subprocess
import sys
import tarfile
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, os.pardir, "diff-schema.py")

spec = importlib.util.spec_from_file_location("diff_schema", TOOL)
ds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ds)


def read(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return fh.read()


def propose(base="base.sql", head="head.sql"):
    return ds.propose(read(base), read(head), base, head)


class ProposalTest(unittest.TestCase):
    def test_matches_golden_file(self):
        self.assertEqual(propose(), read("expected.sql"))

    def test_cli_prints_the_same_proposal(self):
        run = subprocess.run([sys.executable, TOOL, os.path.join(HERE, "base.sql"), os.path.join(HERE, "head.sql")],
                             capture_output=True, text=True, check=True)
        self.assertEqual(run.stdout, read("expected.sql"))

    def test_block_order_is_irrelevant(self):
        marker = "/* <begin connected objects> */"
        blocks = read("head.sql").split(marker)[1:]
        random.Random(7).shuffle(blocks)
        shuffled = "".join(marker + b for b in blocks)
        self.assertEqual(ds.propose(read("base.sql"), shuffled, "base.sql", "head.sql"), read("expected.sql"))

    def test_identical_schemas_need_nothing(self):
        out = propose("base.sql", "base.sql")
        self.assertIn("No schema differences", out)
        self.assertNotIn("CHECK BY HAND", out)

    def test_release_tarball_as_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            member_dir = os.path.join(tmp, "usr", "share", "postgresql", "17", "extension")
            os.makedirs(member_dir)
            with open(os.path.join(member_dir, "pg_pinecone--0.1.0.sql"), "w", encoding="utf-8") as fh:
                fh.write(read("base.sql"))
            with open(os.path.join(member_dir, "pg_pinecone--0.0.9--0.1.0.sql"), "w", encoding="utf-8") as fh:
                fh.write("SELECT 'upgrade scripts are not schemas';")
            tarball = os.path.join(tmp, "pg_pinecone-0.1.0-pg17-linux-x86_64.tar.gz")
            with tarfile.open(tarball, "w:gz") as tar:
                tar.add(os.path.join(tmp, "usr"), arcname="usr")
            out = ds.propose(ds.read_schema(tarball), read("head.sql"), "base.sql", "head.sql")
        self.assertEqual(out, read("expected.sql"))

    def test_table_changes_are_marked_for_a_human(self):
        marked = False
        for line in propose().splitlines():
            if line.startswith("-- CHECK BY HAND"):
                marked = True
            elif not line:
                marked = False
            elif line.startswith("ALTER TABLE"):
                self.assertTrue(marked, line)


class ScannerTest(unittest.TestCase):
    def split(self, sql):
        return [st.text for st in ds.statements(sql)]

    def test_semicolons_inside_quotes_and_comments_do_not_split(self):
        sql = "SELECT 'a;b'; SELECT $$x;y$$; SELECT $t$ $$ ; $t$; -- c;\nSELECT \"q;\" /* ; /* nested ; */ */;"
        self.assertEqual(len(self.split(sql)), 4)

    def test_escape_strings_and_parameters(self):
        self.assertEqual(self.split("SELECT E'it\\'s; fine'; SELECT $1 + $2;"), ["SELECT E'it\\'s; fine'", "SELECT $1 + $2"])

    def test_function_bodies_compare_without_comments_or_case(self):
        a = ds.Schema("CREATE FUNCTION s.f() RETURNS int LANGUAGE plpgsql AS $$ BEGIN -- x\n RETURN 1; END $$;")
        b = ds.Schema("create function s.f() returns int language plpgsql as $$ begin /* y */ return 1; end $$;")
        c = ds.Schema("CREATE FUNCTION s.f() RETURNS int LANGUAGE plpgsql AS $$ BEGIN RETURN '--'; END $$;")
        self.assertEqual(a.funcs["s.f()"].norm, b.funcs["s.f()"].norm)
        self.assertNotEqual(a.funcs["s.f()"].norm, c.funcs["s.f()"].norm)


class FunctionTest(unittest.TestCase):
    def func(self, sql):
        return ds.Schema(sql).funcs

    def test_identity_uses_normalized_input_types_only(self):
        funcs = self.func('CREATE FUNCTION s."f"("a" INT, b double precision, c text[] DEFAULT NULL, OUT d bigint) '
                          "RETURNS record LANGUAGE sql AS 'select 1';")
        self.assertEqual(list(funcs), ["s.f(integer, double precision, text[])"])
        self.assertEqual(funcs["s.f(integer, double precision, text[])"].result, "record out(d bigint)")

    def test_attributes_attach_by_signature(self):
        schema = ds.Schema("CREATE FUNCTION s.f(a int) RETURNS int LANGUAGE sql AS 'select 1';"
                           "CREATE FUNCTION s.f(a int, b int) RETURNS int LANGUAGE sql AS 'select 2';"
                           "ALTER FUNCTION s.f(integer, int) ROWS 5;"
                           "REVOKE EXECUTE ON FUNCTION s.f(integer) FROM PUBLIC;")
        self.assertEqual([len(f.attrs) for f in schema.funcs.values()], [1, 1])
        self.assertEqual(schema.loose, [])

    def test_create_or_replace_blockers(self):
        base = self.func("CREATE FUNCTION s.f(a int DEFAULT 1) RETURNS int LANGUAGE sql AS 'select 1';")["s.f(integer)"]
        body = self.func("CREATE FUNCTION s.f(a int DEFAULT 2) RETURNS int LANGUAGE sql AS 'select 2';")["s.f(integer)"]
        renamed = self.func("CREATE FUNCTION s.f(b int DEFAULT 1) RETURNS int LANGUAGE sql AS 'select 1';")["s.f(integer)"]
        retyped = self.func("CREATE FUNCTION s.f(a int DEFAULT 1) RETURNS bigint LANGUAGE sql AS 'select 1';")["s.f(integer)"]
        undefaulted = self.func("CREATE FUNCTION s.f(a int) RETURNS int LANGUAGE sql AS 'select 1';")["s.f(integer)"]
        self.assertIsNone(body.replace_blocker(base))
        self.assertIn("is now", renamed.replace_blocker(base))
        self.assertIn("result", retyped.replace_blocker(base))
        self.assertIn("defaults", undefaulted.replace_blocker(base))


class TableTest(unittest.TestCase):
    def test_column_parts(self):
        table = ds.Schema("CREATE TABLE s.t (a int PRIMARY KEY, m text NOT NULL DEFAULT 'x' CHECK (m IN ('x', 'y')), "
                          "n numeric(10, 2), UNIQUE (a, m));").tables["s.t"]
        col = table.cols["m"]
        self.assertEqual((col.type, col.notnull, col.default), ("text", True, "'x'"))
        self.assertEqual(col.checks, [(None, "check(m in('x', 'y'))")])
        self.assertEqual(table.cols["n"].type, "numeric(10, 2)")
        self.assertEqual(table.cols["a"].other, ["primary key"])
        self.assertEqual(list(table.constraints), ["unique(a, m)"])


if __name__ == "__main__":
    unittest.main()
