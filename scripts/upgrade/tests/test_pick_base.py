import importlib.util
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("pick_base", os.path.join(HERE, os.pardir, "pick-base.py"))
pb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pb)

TAGS = ["v0.2.0", "v0.8.0", "v0.9.0", "v0.9.1", "v0.10.0", "v0.10.1", "v0.11.0", "v0.11.0-rc1", "latest"]


class DeriveTest(unittest.TestCase):
    def test_the_highest_release_below_the_crate_version(self):
        self.assertEqual(pb.derive(TAGS, "0.12.0"), "v0.11.0")
        self.assertEqual(pb.derive(TAGS, "0.11.0"), "v0.10.1")
        self.assertEqual(pb.derive(TAGS, "0.10.1"), "v0.10.0")

    def test_versions_sort_numerically_and_a_released_or_later_tag_is_never_the_base(self):
        self.assertEqual(pb.derive(["v0.10.0", "v0.11.0", "v0.12.0"], "0.11.0"), "v0.10.0")

    def test_the_first_release_has_no_base_and_pinegres_tags_are_never_one(self):
        for tags in ([], ["v0.8.0"], ["v0.8.0", "v0.9.0", "v0.9.1"]):
            self.assertEqual(pb.derive(tags, "0.10.0"), "")

    def test_a_later_version_without_a_base_is_an_error(self):
        for tags, crate in ((["v0.9.0"], "0.11.0"), ([], "0.10.1")):
            with self.assertRaises(SystemExit):
                pb.derive(tags, crate)


class CheckTest(unittest.TestCase):
    def sql_dir(self, *scripts):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for name in scripts + ("pg_pinecone--0.11.0.sql", "pg_pinecone.control"):
            open(os.path.join(tmp.name, name), "w").close()
        return tmp.name

    def test_the_script_of_the_pair_is_required(self):
        sql = self.sql_dir("pg_pinecone--0.10.0--0.11.0.sql")
        pb.check("v0.10.0", "0.11.0", sql)
        with self.assertRaises(SystemExit) as raised:
            pb.check("v0.11.0", "0.12.0", sql)
        self.assertIn("pg_pinecone--0.11.0--0.12.0.sql", str(raised.exception))

    def test_a_chain_of_scripts_is_enough(self):
        sql = self.sql_dir("pg_pinecone--0.10.0--0.11.0.sql", "pg_pinecone--0.11.0--0.12.0.sql")
        pb.check("v0.10.0", "0.12.0", sql)
        pb.check("v0.11.0", "0.12.0", sql)
        with self.assertRaises(SystemExit):
            pb.check("v0.10.0", "0.13.0", sql)

    def test_a_base_that_is_not_older_or_a_pinegres_release_is_refused(self):
        sql = self.sql_dir("pg_pinecone--0.10.0--0.11.0.sql")
        for base in ("v0.11.0", "v0.12.0", "v0.9.0", "v0.8.0", "main"):
            with self.assertRaises(SystemExit):
                pb.check(base, "0.11.0", sql)


if __name__ == "__main__":
    unittest.main()
