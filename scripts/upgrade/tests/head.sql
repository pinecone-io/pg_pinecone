/* <begin connected objects> */
-- demo/src/views.rs:1
-- requires:
--   demo

    CREATE VIEW demo.v_status AS SELECT relid, mode, tier FROM demo.tables;
    CREATE VIEW demo.v_new AS SELECT 2 AS two;
/* </end connected objects> */

/* <begin connected objects> */
-- demo/src/lib.rs:1
CREATE SCHEMA IF NOT EXISTS demo; /* demo::demo */
/* </end connected objects> */

/* <begin connected objects> */
-- demo/src/lib.rs:25
-- demo::demo::plain_c
CREATE FUNCTION demo."plain_c"(
	"tbl" regclass /* PgRelation, renamed in Rust only */
) RETURNS void
STRICT
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'plain_c_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- demo/src/lib.rs:26
-- requires:
--   plain_c

REVOKE EXECUTE ON FUNCTION demo.plain_c(regclass) FROM PUBLIC;
/* </end connected objects> */

/* <begin connected objects> */
-- demo/src/registry.rs:10
-- requires:
--   demo

    CREATE TABLE demo.tables (
        relid regclass PRIMARY KEY,
        mode text NOT NULL DEFAULT 'local' CHECK (mode IN ('local', 'remote', 'promoting')),
        -- a note
        note text DEFAULT 'n/a',
        retries int DEFAULT 3,
        tier text NOT NULL DEFAULT 'free',
        created_at timestamptz NOT NULL DEFAULT now(),
        region text,
        strict_col text NOT NULL,
        CONSTRAINT tables_region_check CHECK (region IS NULL OR tier <> 'x')
    );
    REVOKE ALL ON demo.tables FROM PUBLIC;
    SELECT pg_catalog.pg_extension_config_dump('demo.tables', '');
/* </end connected objects> */

/* <begin connected objects> */
-- demo/src/queue.rs:30
-- requires:
--   demo

    CREATE TABLE demo.outbox (
        id bigserial PRIMARY KEY,
        op text NOT NULL CHECK (op IN ('upsert', 'delete', 'update')),
        payload jsonb
    );
    CREATE INDEX demo_outbox_op_idx ON demo.outbox (op);
    CREATE INDEX demo_outbox_created_idx ON demo.outbox (id) WHERE op = 'update';
/* </end connected objects> */

/* <begin connected objects> */
-- demo/src/audit.rs:1
-- requires:
--   demo

    CREATE TABLE demo.audit (
        id bigserial PRIMARY KEY,
        what text NOT NULL
    );
    CREATE INDEX demo_audit_what_idx ON demo.audit (what);
    REVOKE ALL ON demo.audit FROM PUBLIC;
    SELECT pg_catalog.pg_extension_config_dump('demo.audit', '');
    SELECT pg_catalog.pg_extension_config_dump('demo.audit_id_seq', '');
    INSERT INTO demo.audit (what) VALUES ('seed');
/* </end connected objects> */

/* <begin connected objects> */
-- demo/src/lib.rs:40
-- requires:
--   demo

    CREATE FUNCTION demo.body_change(x integer) RETURNS integer
    LANGUAGE plpgsql
    AS $$
    BEGIN
        RETURN x + 2;
    END;
    $$;
    REVOKE EXECUTE ON FUNCTION demo.body_change(integer) FROM PUBLIC;
/* </end connected objects> */

/* <begin connected objects> */
-- demo/src/lib.rs:60
-- requires:
--   demo

    create function demo.comment_only(x int) returns int
    language plpgsql
    as $$
    begin
        /* a new comment */
        return x;
    end;
    $$;
/* </end connected objects> */

/* <begin connected objects> */
-- demo/src/lib.rs:90
-- demo::demo::table_fn
CREATE  FUNCTION demo."table_fn"() RETURNS TABLE (
	"a" INT,  /* i32 */
	"b" TEXT  /* String */
)
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'table_fn_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- demo/src/lib.rs:95
-- requires:
--   table_fn

REVOKE EXECUTE ON FUNCTION demo.table_fn() FROM PUBLIC;
/* </end connected objects> */

/* <begin connected objects> */
-- demo/src/lib.rs:100
-- demo::demo::renamed_param
CREATE  FUNCTION demo."renamed_param"(
	"new_name" TEXT /* &str */
) RETURNS void
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'renamed_param_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- demo/src/lib.rs:110
-- demo::demo::sig_change
CREATE  FUNCTION demo."sig_change"(
	"a" INT, /* i32 */
	"b" TEXT DEFAULT 'x' /* &str */
) RETURNS INT /* i32 */
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'sig_change_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- demo/src/lib.rs:120
-- demo::demo::search
CREATE  FUNCTION demo."search"(
	"q" TEXT, /* &str */
	"k" INT DEFAULT 20 /* Option < i32 > */
) RETURNS TABLE (
	"id" TEXT,  /* String */
	"score" real  /* f32 */
)
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'search_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- demo/src/lib.rs:130
-- requires:
--   search

ALTER FUNCTION demo.search(text, integer) ROWS 200;
/* </end connected objects> */

/* <begin connected objects> */
-- demo/src/lib.rs:140
-- demo::demo::fresh
CREATE  FUNCTION demo."fresh"(
	"x" TEXT /* &str */
) RETURNS TEXT /* String */
STRICT
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'fresh_wrapper';
REVOKE EXECUTE ON FUNCTION demo.fresh(text) FROM PUBLIC;
/* </end connected objects> */

/* <begin connected objects> */
-- demo/src/hooks.rs:1
-- requires:
--   demo

    CREATE FUNCTION demo._on_drop() RETURNS event_trigger
    LANGUAGE plpgsql
    AS $$ BEGIN PERFORM 2; END; $$;
    CREATE EVENT TRIGGER demo_on_drop ON sql_drop
        EXECUTE FUNCTION demo._on_drop();
    CREATE EVENT TRIGGER demo_on_ddl ON ddl_command_end WHEN TAG IN ('ALTER TABLE', 'CREATE TABLE')
        EXECUTE FUNCTION demo._on_drop();
    CREATE EVENT TRIGGER demo_on_start ON ddl_command_start
        EXECUTE FUNCTION demo._on_drop();
/* </end connected objects> */

/* <begin connected objects> */
-- demo/src/am.rs:1
-- requires:
--   demo

SELECT demo._install_opclasses();
COMMENT ON TABLE demo.tables IS 'registry v2';
/* </end connected objects> */

/* <begin connected objects> */
-- demo/src/am.rs:100
CREATE  FUNCTION demo."_install_opclasses"() RETURNS void
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'install_opclasses_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- demo/src/am.rs:200
-- requires:
--   demo

CREATE ACCESS METHOD demo_am TYPE INDEX HANDLER demo._am_handler;
CREATE OPERATOR <#> (LEFTARG = real[], RIGHTARG = real[], FUNCTION = demo.dist2);
CREATE OPERATOR CLASS demo.vec_ops DEFAULT FOR TYPE real[] USING demo_am AS OPERATOR 1 <#>;
CREATE TYPE demo.kind AS ENUM ('a', 'b');
/* </end connected objects> */
