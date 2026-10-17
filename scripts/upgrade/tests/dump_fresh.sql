--
-- PostgreSQL database dump
--

\restrict VdInNNnfO7RrdrJZuNlfvR5IwfCuJXTOHTrvswZlg5NhFcGAj12ZsNKBPnoO2wF

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
SELECT pg_catalog.binary_upgrade_set_next_pg_type_oid('16388'::pg_catalog.oid);


-- For binary upgrade, must preserve pg_type array oid
SELECT pg_catalog.binary_upgrade_set_next_array_pg_type_oid('16387'::pg_catalog.oid);


-- For binary upgrade, must preserve pg_class oids and relfilenodes
SELECT pg_catalog.binary_upgrade_set_next_heap_pg_class_oid('16386'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_heap_relfilenode('16386'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_toast_pg_class_oid('16391'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_toast_relfilenode('16391'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_index_pg_class_oid('16392'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_index_relfilenode('16392'::pg_catalog.oid);

CREATE TABLE public.t (
    a integer NOT NULL,
    e integer,
    c integer DEFAULT 0 NOT NULL,
    d text DEFAULT 'x'::text
);

-- For binary upgrade, set heap's relfrozenxid and relminmxid
UPDATE pg_catalog.pg_class
SET relfrozenxid = '740', relminmxid = '1'
WHERE oid = 'public.t'::pg_catalog.regclass;

-- For binary upgrade, set toast's relfrozenxid and relminmxid
UPDATE pg_catalog.pg_class
SET relfrozenxid = '740', relminmxid = '1'
WHERE oid = '16391';


ALTER TABLE public.t OWNER TO postgres;

--
-- Name: u; Type: TABLE; Schema: public; Owner: postgres
--


-- For binary upgrade, must preserve pg_type oid
SELECT pg_catalog.binary_upgrade_set_next_pg_type_oid('16406'::pg_catalog.oid);


-- For binary upgrade, must preserve pg_type array oid
SELECT pg_catalog.binary_upgrade_set_next_array_pg_type_oid('16405'::pg_catalog.oid);


-- For binary upgrade, must preserve pg_class oids and relfilenodes
SELECT pg_catalog.binary_upgrade_set_next_heap_pg_class_oid('16404'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_heap_relfilenode('16404'::pg_catalog.oid);

CREATE TABLE public.u (
    a integer,
    c integer,
    tail integer
);

-- For binary upgrade, set heap's relfrozenxid and relminmxid
UPDATE pg_catalog.pg_class
SET relfrozenxid = '744', relminmxid = '1'
WHERE oid = 'public.u'::pg_catalog.regclass;


ALTER TABLE public.u OWNER TO postgres;

--
-- Name: v; Type: TABLE; Schema: public; Owner: postgres
--


-- For binary upgrade, must preserve pg_type oid
SELECT pg_catalog.binary_upgrade_set_next_pg_type_oid('16414'::pg_catalog.oid);


-- For binary upgrade, must preserve pg_type array oid
SELECT pg_catalog.binary_upgrade_set_next_array_pg_type_oid('16413'::pg_catalog.oid);


-- For binary upgrade, must preserve pg_class oids and relfilenodes
SELECT pg_catalog.binary_upgrade_set_next_heap_pg_class_oid('16412'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_heap_relfilenode('16412'::pg_catalog.oid);

CREATE TABLE public.v (
    a integer,
    b integer
);

-- For binary upgrade, set heap's relfrozenxid and relminmxid
UPDATE pg_catalog.pg_class
SET relfrozenxid = '748', relminmxid = '1'
WHERE oid = 'public.v'::pg_catalog.regclass;


ALTER TABLE public.v OWNER TO postgres;

--
-- Name: t t_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--


-- For binary upgrade, must preserve pg_class oids and relfilenodes
SELECT pg_catalog.binary_upgrade_set_next_index_pg_class_oid('16393'::pg_catalog.oid);
SELECT pg_catalog.binary_upgrade_set_next_index_relfilenode('16393'::pg_catalog.oid);

ALTER TABLE ONLY public.t
    ADD CONSTRAINT t_pkey PRIMARY KEY (a);


--
-- PostgreSQL database dump complete
--

\unrestrict VdInNNnfO7RrdrJZuNlfvR5IwfCuJXTOHTrvswZlg5NhFcGAj12ZsNKBPnoO2wF

