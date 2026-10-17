--
-- PostgreSQL database dump
--

\restrict D62bVOBnH4pewVRcdXJNAd1hKHR3AHw5fT57goudMmVRP4txK3qWYE6wjdwfvG1

-- Dumped from database version 17.10
-- Dumped by pg_dump version 17.10

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: t; Type: TABLE; Schema: public; Owner: postgres
--


-- For binary upgrade, must preserve pg_type oid
SELECT pg_catalog.binary_upgrade_set_next_pg_type_oid('16397'::pg_catalog.oid);


-- For binary upgrade, must preserve pg_type array oid
SELECT pg_catalog.binary_upgrade_set_next_array_pg_type_oid('16396'::pg_catalog.oid);


-- For binary upgrade, must preserve pg_class oids and relfilenodes
SELECT pg_catalog.binary_upgrade_set_next_heap_pg_class_oid('16395'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_heap_relfilenode('16395'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_toast_pg_class_oid('16402'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_toast_relfilenode('16402'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_index_pg_class_oid('16403'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_index_relfilenode('16403'::pg_catalog.oid);

CREATE TABLE public.t (
    a integer NOT NULL,
    e integer,
    c integer DEFAULT 0 NOT NULL,
    d text DEFAULT 'x'::text
);

-- set missing value.
SELECT pg_catalog.binary_upgrade_set_missing_value('public.t'::pg_catalog.regclass,'c','{0}');


-- set missing value.
SELECT pg_catalog.binary_upgrade_set_missing_value('public.t'::pg_catalog.regclass,'d','{x}');


-- For binary upgrade, set heap's relfrozenxid and relminmxid
UPDATE pg_catalog.pg_class
SET relfrozenxid = '741', relminmxid = '1'
WHERE oid = 'public.t'::pg_catalog.regclass;

-- For binary upgrade, set toast's relfrozenxid and relminmxid
UPDATE pg_catalog.pg_class
SET relfrozenxid = '743', relminmxid = '1'
WHERE oid = '16402';


ALTER TABLE public.t OWNER TO postgres;

--
-- Name: u; Type: TABLE; Schema: public; Owner: postgres
--


-- For binary upgrade, must preserve pg_type oid
SELECT pg_catalog.binary_upgrade_set_next_pg_type_oid('16409'::pg_catalog.oid);


-- For binary upgrade, must preserve pg_type array oid
SELECT pg_catalog.binary_upgrade_set_next_array_pg_type_oid('16408'::pg_catalog.oid);


-- For binary upgrade, must preserve pg_class oids and relfilenodes
SELECT pg_catalog.binary_upgrade_set_next_heap_pg_class_oid('16407'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_heap_relfilenode('16407'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_toast_pg_class_oid('16410'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_toast_relfilenode('16410'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_index_pg_class_oid('16411'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_index_relfilenode('16411'::pg_catalog.oid);

CREATE TABLE public.u (
    a integer,
    "........pg.dropped.2........" INTEGER /* dummy */,
    c integer,
    "........pg.dropped.4........" INTEGER /* dummy */,
    tail integer
);

-- For binary upgrade, recreate dropped columns.
UPDATE pg_catalog.pg_attribute
SET attlen = v.dlen, attalign = v.dalign, attbyval = false
FROM (VALUES ('........pg.dropped.2........', 4, 'i'),
             ('........pg.dropped.4........', -1, 'i')) v(dname, dlen, dalign)
WHERE attrelid = 'public.u'::pg_catalog.regclass
  AND attname = v.dname;
ALTER TABLE ONLY public.u DROP COLUMN "........pg.dropped.2........";
ALTER TABLE ONLY public.u DROP COLUMN "........pg.dropped.4........";

-- For binary upgrade, set heap's relfrozenxid and relminmxid
UPDATE pg_catalog.pg_class
SET relfrozenxid = '745', relminmxid = '1'
WHERE oid = 'public.u'::pg_catalog.regclass;

-- For binary upgrade, set toast's relfrozenxid and relminmxid
UPDATE pg_catalog.pg_class
SET relfrozenxid = '745', relminmxid = '1'
WHERE oid = '16410';


ALTER TABLE public.u OWNER TO postgres;

--
-- Name: v; Type: TABLE; Schema: public; Owner: postgres
--


-- For binary upgrade, must preserve pg_type oid
SELECT pg_catalog.binary_upgrade_set_next_pg_type_oid('16417'::pg_catalog.oid);


-- For binary upgrade, must preserve pg_type array oid
SELECT pg_catalog.binary_upgrade_set_next_array_pg_type_oid('16416'::pg_catalog.oid);


-- For binary upgrade, must preserve pg_class oids and relfilenodes
SELECT pg_catalog.binary_upgrade_set_next_heap_pg_class_oid('16415'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_heap_relfilenode('16415'::pg_catalog.oid);

CREATE TABLE public.v (
    a integer,
    b integer,
    "........pg.dropped.3........" INTEGER /* dummy */
);

-- For binary upgrade, recreate dropped columns.
UPDATE pg_catalog.pg_attribute
SET attlen = v.dlen, attalign = v.dalign, attbyval = false
FROM (VALUES ('........pg.dropped.3........', 4, 'i')) v(dname, dlen, dalign)
WHERE attrelid = 'public.v'::pg_catalog.regclass
  AND attname = v.dname;
ALTER TABLE ONLY public.v DROP COLUMN "........pg.dropped.3........";

-- For binary upgrade, set heap's relfrozenxid and relminmxid
UPDATE pg_catalog.pg_class
SET relfrozenxid = '749', relminmxid = '1'
WHERE oid = 'public.v'::pg_catalog.regclass;


ALTER TABLE public.v OWNER TO postgres;

--
-- Name: t t_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--


-- For binary upgrade, must preserve pg_class oids and relfilenodes
SELECT pg_catalog.binary_upgrade_set_next_index_pg_class_oid('16398'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_index_relfilenode('16398'::pg_catalog.oid);

ALTER TABLE ONLY public.t
    ADD CONSTRAINT t_pkey PRIMARY KEY (a);


--
-- PostgreSQL database dump complete
--

\unrestrict D62bVOBnH4pewVRcdXJNAd1hKHR3AHw5fT57goudMmVRP4txK3qWYE6wjdwfvG1

