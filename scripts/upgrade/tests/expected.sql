-- diff-schema.py: base=base.sql head=head.sql
-- Proposal for pg_pinecone/sql/pg_pinecone--<prev>--<new>.sql: review every statement, nothing is applied for you.
-- Summary: event triggers changed 1, event triggers dropped 1, event triggers new 1, functions changed 3, functions dropped 2, functions new 2, functions recreated 2, hand-written 3, indexes new 1, statements gone 1, statements new 2, tables changed 2, tables dropped 1, tables new 1, views changed 1, views dropped 1, views new 1

-- === TABLES ===
-- CHECK BY HAND: new table demo.audit; its indexes, grants and extension config dump come with it
CREATE TABLE demo.audit (
        id bigserial PRIMARY KEY,
        what text NOT NULL
    );
CREATE INDEX demo_audit_what_idx ON demo.audit (what);
REVOKE ALL ON demo.audit FROM PUBLIC;
SELECT pg_catalog.pg_extension_config_dump('demo.audit', '');
SELECT pg_catalog.pg_extension_config_dump('demo.audit_id_seq', '');

-- CHECK BY HAND: table demo.legacy_queue is gone in HEAD; dropping it loses its rows
-- DROP TABLE IF EXISTS demo.legacy_queue;

-- CHECK BY HAND: demo.outbox.op: CHECK changed, existing rows are validated
ALTER TABLE demo.outbox DROP CONSTRAINT IF EXISTS outbox_op_check;
ALTER TABLE demo.outbox ADD CONSTRAINT outbox_op_check CHECK (op IN ('upsert', 'delete', 'update'));

-- CHECK BY HAND: demo.tables.mode: CHECK changed, existing rows are validated
ALTER TABLE demo.tables DROP CONSTRAINT IF EXISTS tables_mode_check;
ALTER TABLE demo.tables ADD CONSTRAINT tables_mode_check CHECK (mode IN ('local', 'remote', 'promoting'));
-- CHECK BY HAND: demo.tables.note: default changed
ALTER TABLE demo.tables ALTER COLUMN note SET DEFAULT 'n/a';
-- CHECK BY HAND: demo.tables.retries: NOT NULL dropped
ALTER TABLE demo.tables ALTER COLUMN retries DROP NOT NULL;
-- CHECK BY HAND: demo.tables: new column "tier"
-- CHECK BY HAND: "tier" precedes existing column "created_at" in HEAD; ADD COLUMN appends, so move it last in the Rust CREATE TABLE (the catalog compare checks column order)
ALTER TABLE demo.tables ADD COLUMN IF NOT EXISTS tier text NOT NULL DEFAULT 'free';
-- CHECK BY HAND: demo.tables: new column "region"
ALTER TABLE demo.tables ADD COLUMN IF NOT EXISTS region text;
-- CHECK BY HAND: demo.tables: new column "strict_col" (NOT NULL without a DEFAULT fails on existing rows)
ALTER TABLE demo.tables ADD COLUMN IF NOT EXISTS strict_col text NOT NULL;
-- CHECK BY HAND: demo.tables: column "dropped_col" is gone in HEAD; dropping it loses data
-- ALTER TABLE demo.tables DROP COLUMN IF EXISTS dropped_col;
-- CHECK BY HAND: demo.tables: new table constraint, existing rows are validated
ALTER TABLE demo.tables ADD CONSTRAINT tables_region_check CHECK (region IS NULL OR tier <> 'x');

-- === INDEXES ===
CREATE INDEX IF NOT EXISTS demo_outbox_created_idx ON demo.outbox (id) WHERE op = 'update';

-- === EVENT TRIGGERS: drop ===
DROP EVENT TRIGGER IF EXISTS demo_removed;
DROP EVENT TRIGGER IF EXISTS demo_on_ddl;

-- === FUNCTIONS: drop ===
DROP FUNCTION IF EXISTS demo.removed_fn(text);
-- CHECK BY HAND: demo.sig_change changed signature; callers of the old one break, keep it as an overload unless that is intended
DROP FUNCTION IF EXISTS demo.sig_change(integer);
-- CHECK BY HAND: demo.renamed_param(text): CREATE OR REPLACE cannot apply, input parameter "old_name" is now "new_name"; dependants (event triggers, opclasses) block the DROP
DROP FUNCTION IF EXISTS demo.renamed_param(text);
-- CHECK BY HAND: demo.table_fn(): CREATE OR REPLACE cannot apply, the result changed (table(a integer) -> table(a integer, b text)); dependants (event triggers, opclasses) block the DROP
DROP FUNCTION IF EXISTS demo.table_fn();

-- === FUNCTIONS: replace ===
CREATE OR REPLACE FUNCTION demo._on_drop() RETURNS event_trigger
    LANGUAGE plpgsql
    AS $$ BEGIN PERFORM 2; END; $$;

CREATE OR REPLACE FUNCTION demo.body_change(x integer) RETURNS integer
    LANGUAGE plpgsql
    AS $$
    BEGIN
        RETURN x + 2;
    END;
    $$;

CREATE OR REPLACE FUNCTION demo."search"(
	"q" TEXT, /* &str */
	"k" INT DEFAULT 20 /* Option < i32 > */
) RETURNS TABLE (
	"id" TEXT,  /* String */
	"score" real  /* f32 */
)
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'search_wrapper';
ALTER FUNCTION demo.search(text, integer) ROWS 200;

-- === FUNCTIONS: create ===
CREATE FUNCTION demo."renamed_param"(
	"new_name" TEXT /* &str */
) RETURNS void
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'renamed_param_wrapper';

CREATE FUNCTION demo."table_fn"() RETURNS TABLE (
	"a" INT,  /* i32 */
	"b" TEXT  /* String */
)
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'table_fn_wrapper';
REVOKE EXECUTE ON FUNCTION demo.table_fn() FROM PUBLIC;

CREATE FUNCTION demo."fresh"(
	"x" TEXT /* &str */
) RETURNS TEXT /* String */
STRICT
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'fresh_wrapper';
REVOKE EXECUTE ON FUNCTION demo.fresh(text) FROM PUBLIC;

CREATE FUNCTION demo."sig_change"(
	"a" INT, /* i32 */
	"b" TEXT DEFAULT 'x' /* &str */
) RETURNS INT /* i32 */
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'sig_change_wrapper';

-- === FUNCTIONS: grants and attributes ===
REVOKE EXECUTE ON FUNCTION demo.plain_c(regclass) FROM PUBLIC;

-- === EVENT TRIGGERS: create ===
CREATE EVENT TRIGGER demo_on_ddl ON ddl_command_end WHEN TAG IN ('ALTER TABLE', 'CREATE TABLE')
        EXECUTE FUNCTION demo._on_drop();
CREATE EVENT TRIGGER demo_on_start ON ddl_command_start
        EXECUTE FUNCTION demo._on_drop();

-- === VIEWS ===
DROP VIEW IF EXISTS demo.v_gone;
-- CHECK BY HAND: CREATE OR REPLACE VIEW only appends columns; otherwise DROP VIEW first
CREATE OR REPLACE VIEW demo.v_status AS SELECT relid, mode, tier FROM demo.tables;
CREATE VIEW demo.v_new AS SELECT 2 AS two;

-- === OTHER STATEMENTS ===
-- CHECK BY HAND: new in HEAD
COMMENT ON TABLE demo.tables IS 'registry v2';
-- CHECK BY HAND: new in HEAD
INSERT INTO demo.audit (what) VALUES ('seed');
-- CHECK BY HAND: in BASE, not in HEAD: comment on table demo.tables is 'registry'

-- === HAND-WRITTEN NEEDED: operators, opclasses, access method, types and other objects ===
-- new: operator class demo.vec_ops using demo_am
--   CREATE OPERATOR CLASS demo.vec_ops DEFAULT FOR TYPE real[] USING demo_am AS OPERATOR 1 <#>
-- new: type demo.kind
--   CREATE TYPE demo.kind AS ENUM ('a', 'b')
-- changed: operator <#>
--   was: CREATE OPERATOR <#> (LEFTARG = real[], RIGHTARG = real[], FUNCTION = demo.dist)
--   now: CREATE OPERATOR <#> (LEFTARG = real[], RIGHTARG = real[], FUNCTION = demo.dist2)

-- === NOTE: install-time calls in HEAD's schema ===
-- A Rust change behind one of these is invisible to a schema diff; repeat the idempotent ones when it changed:
--   SELECT demo._install_opclasses();
