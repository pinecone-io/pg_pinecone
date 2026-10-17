import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("normalize_dump", os.path.join(HERE, os.pardir, "normalize-dump.py"))
nd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nd)

PREAMBLE = """--
-- PostgreSQL database dump
--

\\restrict {key}

SET statement_timeout = 0;
"""

ENTRIES = {
    "extension": """--
-- Name: pg_pinecone; Type: EXTENSION; Schema: -; Owner: -
--

DROP EXTENSION IF EXISTS pg_pinecone;
SELECT pg_catalog.binary_upgrade_create_empty_extension('pg_pinecone', 'public', false, '0.10.0', '{{{oids}}}', '{{{conditions}}}', ARRAY[]::pg_catalog.text[]);
""",
    "function": """--
-- Name: f(); Type: FUNCTION; Schema: pg_pinecone; Owner: postgres
--

CREATE FUNCTION pinecone.f() RETURNS integer
    LANGUAGE plpgsql
    AS $$
{comment}BEGIN
    RETURN {result};
END
$$;
""",
    "table": """--
-- Name: outbox; Type: TABLE; Schema: pg_pinecone; Owner: postgres
--

-- For binary upgrade, must preserve pg_class oids and relfilenodes
SELECT pg_catalog.binary_upgrade_set_next_heap_pg_class_oid('{oid}'::pg_catalog.oid);

CREATE TABLE pinecone.outbox (
    id bigint NOT NULL{extra}
);

-- For binary upgrade, set heap's relfrozenxid and relminmxid
UPDATE pg_catalog.pg_class
SET relfrozenxid = '{xid}', relminmxid = '1'
WHERE oid = 'pinecone.outbox'::pg_catalog.regclass;

ALTER EXTENSION pg_pinecone ADD TABLE pinecone.outbox;
""",
    "sequence": """--
-- Name: outbox_id_seq; Type: SEQUENCE SET; Schema: pg_pinecone; Owner: postgres
--

SELECT pg_catalog.setval('pinecone.outbox_id_seq', {seq}, true);
""",
    "family": """--
-- Name: integer_ops; Type: OPERATOR FAMILY; Schema: public; Owner: postgres
--

CREATE OPERATOR FAMILY public.integer_ops USING pinecone;
ALTER OPERATOR FAMILY public.integer_ops USING pinecone ADD
{members};
""",
}
MEMBERS = ["    OPERATOR 1 =(bigint,smallint) ,", "    OPERATOR 1 =(bigint,integer) ,", "    OPERATOR 1 =(integer,smallint)"]


def dump(order=("extension", "table", "sequence", "family"), key="abc", oid="16745", xid="744", seq="5",
         oids="16716,16754", extra="", members=MEMBERS, conditions='"",""', comment="", result="1"):
    body = "\n".join(
        ENTRIES[name].format(oids=oids, oid=oid, xid=xid, seq=seq, extra=extra, members="\n".join(members),
                             conditions=conditions, comment=comment, result=result)
        for name in order)
    return PREAMBLE.format(key=key) + "\n" + body + "\n\\unrestrict " + key + "\n"


class NormalizeTest(unittest.TestCase):
    def test_noise_is_removed(self):
        out = nd.normalize(dump())
        for noise in ("binary_upgrade_set_next", "relfrozenxid", "setval", "restrict", "16745", "744", "16716", "-- "):
            self.assertNotIn(noise, out)
        self.assertIn("{2 config tables}", out)
        self.assertNotIn("SEQUENCE SET", out)

    def test_oids_xids_sequence_positions_and_entry_order_do_not_matter(self):
        a = nd.normalize(dump())
        b = nd.normalize(dump(order=("family", "sequence", "table", "extension"), key="zzz", oid="99999", xid="31337",
                              seq="123", oids="1,2"))
        self.assertEqual(a, b)

    def test_operator_family_member_order_does_not_matter(self):
        reordered = ["    OPERATOR 1 =(integer,smallint) ,", "    OPERATOR 1 =(bigint,smallint) ,", "    OPERATOR 1 =(bigint,integer)"]
        self.assertEqual(nd.normalize(dump()), nd.normalize(dump(members=reordered)))

    def test_a_schema_difference_is_kept(self):
        a, b = nd.normalize(dump()), nd.normalize(dump(extra=",\n    probe text"))
        self.assertNotEqual(a, b)
        self.assertIn("probe text", b)

    def test_config_table_count_is_kept(self):
        self.assertNotEqual(nd.normalize(dump()), nd.normalize(dump(oids="1,2,3")))

    def test_config_table_order_does_not_matter_but_the_conditions_do(self):
        a = nd.normalize(dump(conditions='"","WHERE x < 1"'))
        self.assertEqual(a, nd.normalize(dump(conditions='"WHERE x < 1",""', oids="16754,16716")))
        self.assertNotEqual(a, nd.normalize(dump(conditions='"","WHERE x < 2"')))

    def test_comments_in_function_bodies_do_not_matter_indented_or_not(self):
        plain = nd.normalize(dump(order=("function",)))
        for comment in ("-- at column 0\n", "    -- indented\n"):
            self.assertEqual(plain, nd.normalize(dump(order=("function",), comment=comment)))
        self.assertNotEqual(plain, nd.normalize(dump(order=("function",), result="2")))


def read(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return fh.read()


class RealDumpTest(unittest.TestCase):
    """Raw `pg_dump --schema-only --binary-upgrade` of PostgreSQL 17.10: a table built by ALTER TABLE
    (two ADD COLUMN ... DEFAULT; two dropped columns in the middle; the last column dropped) and the
    table with the final shape created fresh."""

    def test_the_fixture_holds_the_noise(self):
        upgraded = read("dump_upgraded.sql")
        for noise in ("binary_upgrade_set_missing_value", "pg.dropped.2", "UPDATE pg_catalog.pg_attribute", "DROP COLUMN"):
            self.assertIn(noise, upgraded)
            self.assertNotIn(noise, read("dump_fresh.sql"))

    def test_added_columns_with_defaults_and_dropped_columns_equal_a_fresh_table(self):
        self.assertEqual(nd.normalize(read("dump_fresh.sql")), nd.normalize(read("dump_upgraded.sql")))

    def test_a_dropped_column_does_not_hide_a_difference(self):
        upgraded = read("dump_upgraded.sql")
        for old, new in (("    e integer,\n", "    e bigint,\n"), ("    c integer,\n", ""), ("    a integer,\n    b integer,", "    a integer,")):
            self.assertIn(old, upgraded)
            self.assertNotEqual(nd.normalize(read("dump_fresh.sql")), nd.normalize(upgraded.replace(old, new)))

    def test_the_missing_value_is_stripped_but_the_default_is_kept(self):
        out = nd.normalize(read("dump_upgraded.sql"))
        self.assertNotIn("missing_value", out)
        self.assertIn("c integer DEFAULT 0 NOT NULL", out)
        self.assertIn("d text DEFAULT 'x'::text", out)


if __name__ == "__main__":
    unittest.main()
