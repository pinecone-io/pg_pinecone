import importlib.util
import itertools
import os
import shutil
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("normalize_dump", os.path.join(HERE, os.pardir, "normalize-dump.py"))
nd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nd)


def bindir():
    config = os.environ.get("PG_CONFIG") or shutil.which("pg_config")
    if not config:
        return None
    return subprocess.run([config, "--bindir"], capture_output=True, text=True, check=True).stdout.strip()


BIN = bindir()
DATABASES = itertools.count()


@unittest.skipUnless(BIN, "needs PG_CONFIG or pg_config on PATH")
class LiveDumpTest(unittest.TestCase):
    """Real pg_dump --binary-upgrade of this Postgres: a table an upgrade script altered must normalize
    like the same table created fresh."""

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="pg_pinecone-nd.", dir="/tmp")
        cls.addClassCleanup(shutil.rmtree, cls.work, True)
        data, cls.sock, cls.port = os.path.join(cls.work, "data"), os.path.join(cls.work, "s"), str(20000 + os.getpid() % 20000)
        os.mkdir(cls.sock)
        subprocess.run([f"{BIN}/initdb", "-D", data, "-U", "postgres", "-A", "trust", "-E", "UTF8", "--locale=C", "--no-sync"],
                       check=True, capture_output=True)
        options = f"-c listen_addresses='' -c unix_socket_directories={cls.sock} -c port={cls.port} -c fsync=off"
        subprocess.run([f"{BIN}/pg_ctl", "-D", data, "-l", os.path.join(cls.work, "log"), "-w", "-t", "60", "-o", options, "start"],
                       check=True, capture_output=True)
        cls.addClassCleanup(subprocess.run, [f"{BIN}/pg_ctl", "-D", data, "-m", "immediate", "stop"], capture_output=True)

    def client(self, tool, db, *args):
        cmd = [f"{BIN}/{tool}", "-h", self.sock, "-p", self.port, "-U", "postgres", "-d", db, *args]
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout

    def dumps(self, fresh, upgraded):
        raw = []
        for kind, statements in (("fresh", fresh), ("upgraded", upgraded)):
            db = f"{kind}_{next(DATABASES)}"
            self.client("psql", "postgres", "-X", "-q", "-c", f"CREATE DATABASE {db}")
            self.client("psql", db, "-X", "-q", "-v", "ON_ERROR_STOP=1", "-c", ";\n".join(statements))
            raw.append(self.client("pg_dump", db, "--schema-only", "--binary-upgrade"))
        return raw

    def test_added_columns_with_defaults_equal_a_fresh_table(self):
        fresh, upgraded = self.dumps(
            ["CREATE TABLE t (a int PRIMARY KEY, e int, c int NOT NULL DEFAULT 0, d text DEFAULT 'x', f timestamptz DEFAULT now())"],
            ["CREATE TABLE t (a int PRIMARY KEY, e int)", "ALTER TABLE t ADD COLUMN c int NOT NULL DEFAULT 0",
             "ALTER TABLE t ADD COLUMN d text DEFAULT 'x'", "ALTER TABLE t ADD COLUMN f timestamptz DEFAULT now()"])
        self.assertIn("binary_upgrade_set_missing_value", upgraded)
        self.assertNotEqual(fresh, upgraded)
        self.assertEqual(nd.normalize(fresh), nd.normalize(upgraded))

    def test_dropped_columns_equal_a_fresh_table(self):
        fresh, upgraded = self.dumps(
            ["CREATE TABLE u (a int, c int, tail int)", "CREATE TABLE v (a int, b int)"],
            ["CREATE TABLE u (a int, b int, c int, d text, tail int)", "ALTER TABLE u DROP COLUMN b", "ALTER TABLE u DROP COLUMN d",
             "CREATE TABLE v (a int, b int, z int)", "ALTER TABLE v DROP COLUMN z"])
        self.assertIn("pg.dropped.", upgraded)
        self.assertNotEqual(fresh, upgraded)
        self.assertEqual(nd.normalize(fresh), nd.normalize(upgraded))

    def test_a_real_difference_is_still_a_difference(self):
        fresh, upgraded = self.dumps(["CREATE TABLE t (a int, c int)"], ["CREATE TABLE t (a int, c int)", "ALTER TABLE t ADD COLUMN x int"])
        self.assertNotEqual(nd.normalize(fresh), nd.normalize(upgraded))
        fresh, upgraded = self.dumps(["CREATE TABLE t (a int, c int)"], ["CREATE TABLE t (a int, c bigint)"])
        self.assertNotEqual(nd.normalize(fresh), nd.normalize(upgraded))


if __name__ == "__main__":
    unittest.main()
