import hashlib
import importlib.util
import os
import re
import shutil
import stat
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
DRILL = os.path.join(HERE, os.pardir, os.pardir, "upgrade-drill.sh")
spec = importlib.util.spec_from_file_location("pick_base", os.path.join(HERE, os.pardir, "pick-base.py"))
pb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pb)

FAKE_PG_CONFIG = """#!/bin/sh
case "$1" in
    --bindir) echo /nonexistent/bin ;;
    --pkglibdir) echo /nonexistent/lib ;;
    --sharedir) echo /nonexistent/share ;;
    --version) echo "PostgreSQL 17.0" ;;
esac
"""


def a_base_with_an_upgrade_path():
    sql_dir = os.path.join(pb.ROOT, "pg_pinecone", "sql")
    crate = pb.crate_version()
    for name in sorted(os.listdir(sql_dir)) if os.path.isdir(sql_dir) else []:
        found = pb.SCRIPT.fullmatch(name)
        if found:
            try:
                pb.check("v" + found.group(1), crate, sql_dir)
                return "v" + found.group(1)
            except SystemExit:
                pass
    return None


BASE = a_base_with_an_upgrade_path()


def executable(path, text):
    with open(path, "w") as fh:
        fh.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


@unittest.skipUnless(BASE and shutil.which("bash"), "needs an upgrade script chain and bash")
class FetchBaseChecksumTest(unittest.TestCase):
    """The drill's base tarball is verified, or the run says it was not. `cargo` is a stub that fails,
    so a run that gets past the checksum stops at the package step instead of building."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        executable(os.path.join(self.dir, "pg_config"), FAKE_PG_CONFIG)
        os.mkdir(os.path.join(self.dir, "bin"))
        executable(os.path.join(self.dir, "bin", "cargo"), "#!/bin/sh\nexit 1\n")
        self.tarball = os.path.join(self.dir, "base.tar.gz")
        with open(self.tarball, "wb") as fh:
            fh.write(b"not a real tarball")

    def write_checksum(self, content=None):
        if content is None:
            with open(self.tarball, "rb") as fh:
                content = fh.read()
        digest = hashlib.sha256(content).hexdigest()
        with open(self.tarball + ".sha256", "w") as fh:
            fh.write(f"{digest}  base.tar.gz\n")

    def drill(self, **env):
        run_env = {k: v for k, v in os.environ.items() if k != "UPGRADE_ALLOW_UNVERIFIED"}
        run_env.update(env)
        run_env["PATH"] = os.path.join(self.dir, "bin") + os.pathsep + run_env["PATH"]
        run_env["UPGRADE_BASE_TARBALL"] = self.tarball
        done = subprocess.run(
            ["bash", DRILL, BASE, os.path.join(self.dir, "pg_config")],
            capture_output=True, text=True, env=run_env, timeout=120,
        )
        for kept in re.findall(r"scratch dir kept for inspection: (\S+)", done.stderr):
            shutil.rmtree(kept, ignore_errors=True)
        return done

    def test_a_base_without_its_sha256_is_refused(self):
        done = self.drill()
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("unverified", done.stderr)
        self.assertNotIn("cargo pgrx package", done.stderr)

    def test_a_wrong_checksum_is_refused(self):
        self.write_checksum(b"another file")
        done = self.drill()
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("checksum mismatch", done.stderr)

    def test_a_matching_checksum_goes_on_to_the_package_step(self):
        self.write_checksum()
        done = self.drill()
        self.assertIn("cargo pgrx package of this tree failed", done.stderr)
        self.assertNotIn("unverified", done.stderr)
        self.assertNotIn("checksum mismatch", done.stderr)

    def test_allow_unverified_goes_on_and_says_so(self):
        done = self.drill(UPGRADE_ALLOW_UNVERIFIED="1")
        self.assertIn("NOT checksum-verified", done.stderr)
        self.assertIn("cargo pgrx package of this tree failed", done.stderr)


if __name__ == "__main__":
    unittest.main()
