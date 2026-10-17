//! The DDL guard of the document lane: what the sync depends on cannot be changed under it. The
//! Pinecone index schema is immutable and the queue and drain are keyed by the declared columns
//! (id, text, metadata, vector); the statement triggers see writes to the table itself only and
//! must stay in place; the id must stay a NOT NULL, unique and immediate key. Two event
//! triggers, both `SECURITY DEFINER` so any role's unrelated DDL passes without grants on the
//! registry:
//!
//! - `sql_drop` refuses `DROP TABLE` of a tracked table, `DROP COLUMN` of a declared column,
//!   dropping a `pg_pinecone_docs_*` trigger, and dropping the unique constraint or index that
//!   identifies the documents.
//! - `ddl_command_end` for `ALTER TABLE` and `CREATE TABLE` re-checks the tracked tables the
//!   statement altered or is related to (a partition or inheritance parent or child) and only
//!   those: a registry row that has drifted must not block DDL on other tables. Declared columns
//!   are compared against `pg_attribute` (an ALTER's subcommands cannot be read from plpgsql,
//!   so state is compared): a renamed or retyped column no longer matches the `typname:atttypmod`
//!   recorded when the table was tracked (`format_type` would render a type differently under
//!   this function's search_path, hence the plain `typname:typmod` signature). Also refused:
//!   becoming a partition or inheritance parent or child, an id column that is nullable or whose
//!   unique index is deferrable, a partition column that is nullable (a row without a value has
//!   no namespace), a disabled or replica-only sync trigger. A missing key or trigger is not
//!   refused: a restored table gets them after its data and this registry row.
//!
//! `untrack_docs` deletes the registry row first, which ends the protection. Both functions are
//! members of the extension, so `DROP EXTENSION` removes the triggers before they could fire.

use pgrx::prelude::*;

extension_sql!(
    r#"
    CREATE FUNCTION pinecone._docs_on_sql_drop() RETURNS event_trigger
    LANGUAGE plpgsql
    SECURITY DEFINER
    SET search_path = pg_catalog, pg_temp
    AS $$
    DECLARE
        obj record;
        t record;
        v_col text;
        v_rel text;
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pinecone.doc_tables) THEN
            RETURN;
        END IF;
        FOR obj IN SELECT * FROM pg_event_trigger_dropped_objects()
                   WHERE object_type IN ('table', 'table column') LOOP
            SELECT d.column_types INTO t FROM pinecone.doc_tables d WHERE d.relid::oid = obj.objid;
            IF NOT FOUND THEN
                CONTINUE;
            END IF;
            IF obj.object_type = 'table' THEN
                RAISE EXCEPTION 'cannot drop %: it is tracked by pinecone.track_docs', obj.object_identity
                    USING ERRCODE = '2BP01',
                          HINT = format('pinecone.untrack_docs(%L, drop_index => true) first', obj.object_identity);
            END IF;
            v_col := obj.address_names[3];
            IF t.column_types ? v_col THEN
                v_rel := format('%I.%I', obj.address_names[1], obj.address_names[2]);
                RAISE EXCEPTION 'cannot drop column "%" of %: it is declared in pinecone.track_docs', v_col, v_rel
                    USING ERRCODE = '2BP01',
                          HINT = format('pinecone.untrack_docs(%L) first — changing declared columns needs a new document index', v_rel);
            END IF;
        END LOOP;
        FOR obj IN SELECT * FROM pg_event_trigger_dropped_objects()
                   WHERE object_type = 'trigger'
                     AND address_names[3] IN ('pg_pinecone_docs_ins', 'pg_pinecone_docs_upd', 'pg_pinecone_docs_del', 'pg_pinecone_docs_trunc') LOOP
            v_rel := format('%I.%I', obj.address_names[1], obj.address_names[2]);
            IF EXISTS (SELECT 1 FROM pinecone.doc_tables d WHERE d.relid = to_regclass(v_rel)) THEN
                RAISE EXCEPTION 'cannot drop trigger "%" of %: it keeps the document index in sync (pinecone.track_docs)', obj.address_names[3], v_rel
                    USING ERRCODE = '2BP01',
                          HINT = format('pinecone.untrack_docs(%L) first', v_rel);
            END IF;
        END LOOP;
        IF EXISTS (SELECT 1 FROM pg_event_trigger_dropped_objects() WHERE object_type IN ('index', 'table constraint')) THEN
            FOR t IN SELECT d.id_column, n.nspname, k.relname
                       FROM pinecone.doc_tables d
                       JOIN pg_class k ON k.oid = d.relid::oid
                       JOIN pg_namespace n ON n.oid = k.relnamespace
                      WHERE n.nspname IN (SELECT o.schema_name FROM pg_event_trigger_dropped_objects() o
                                           WHERE o.object_type IN ('index', 'table constraint'))
                        AND NOT EXISTS (
                          SELECT 1 FROM pg_index i JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0]
                           WHERE i.indrelid = k.oid AND i.indisunique AND i.indisvalid AND i.indimmediate AND i.indpred IS NULL
                             AND cardinality(i.indkey::int2[]) = 1 AND a.attname = d.id_column) LOOP
                v_rel := format('%I.%I', t.nspname, t.relname);
                RAISE EXCEPTION 'cannot drop the unique constraint of id column "%" of %: it identifies the documents (pinecone.track_docs)', t.id_column, v_rel
                    USING ERRCODE = '2BP01',
                          HINT = format('pinecone.untrack_docs(%L) first', v_rel);
            END LOOP;
        END IF;
    END;
    $$;

    CREATE FUNCTION pinecone._docs_on_ddl_end() RETURNS event_trigger
    LANGUAGE plpgsql
    SECURITY DEFINER
    SET search_path = pg_catalog, pg_temp
    AS $$
    DECLARE
        t record;
        c record;
        v_now text;
        v_rel text;
        v_hint text;
        v_name text;
        v_altered oid[];
    BEGIN
        SELECT coalesce(array_agg(objid), '{}') INTO v_altered
          FROM pg_event_trigger_ddl_commands() WHERE classid = 'pg_class'::regclass;
        IF cardinality(v_altered) = 0 THEN
            RETURN;
        END IF;
        FOR t IN SELECT d.relid::oid AS relid, d.id_column, d.partition_column, d.column_types, n.nspname, k.relname
                   FROM pinecone.doc_tables d
                   JOIN pg_class k ON k.oid = d.relid::oid
                   JOIN pg_namespace n ON n.oid = k.relnamespace
                  WHERE d.relid::oid = ANY (v_altered)
                     OR d.relid::oid IN (SELECT i.inhrelid FROM pg_inherits i WHERE i.inhparent = ANY (v_altered))
                     OR d.relid::oid IN (SELECT i.inhparent FROM pg_inherits i WHERE i.inhrelid = ANY (v_altered)) LOOP
            v_rel := format('%I.%I', t.nspname, t.relname);
            v_hint := format('pinecone.untrack_docs(%L) first — changing declared columns needs a new document index', v_rel);
            FOR c IN SELECT key AS col, value AS expected FROM jsonb_each_text(t.column_types) LOOP
                SELECT y.typname::text || ':' || a.atttypmod INTO v_now
                  FROM pg_attribute a JOIN pg_type y ON y.oid = a.atttypid
                 WHERE a.attrelid = t.relid AND a.attname = c.col AND a.attnum > 0 AND NOT a.attisdropped;
                IF v_now IS DISTINCT FROM c.expected THEN
                    RAISE EXCEPTION 'cannot alter column "%" of %: it is declared in pinecone.track_docs', c.col, v_rel
                        USING ERRCODE = '2BP01', HINT = v_hint;
                END IF;
            END LOOP;
            IF EXISTS (SELECT 1 FROM pg_inherits i WHERE i.inhrelid = t.relid OR i.inhparent = t.relid) THEN
                RAISE EXCEPTION 'cannot make % part of a partition or inheritance hierarchy: it is tracked by pinecone.track_docs, whose statement-level triggers do not see writes made through a parent or a child', v_rel
                    USING ERRCODE = '2BP01', HINT = format('pinecone.untrack_docs(%L) first', v_rel);
            END IF;
            -- a table restored from a dump gets its data (and this registry row) before its key
            -- and its triggers, so neither a missing key nor a missing trigger is refused here;
            -- dropping them is refused by the sql_drop trigger
            IF EXISTS (
                SELECT 1 FROM pg_attribute a
                 WHERE a.attrelid = t.relid AND a.attname = t.id_column AND NOT a.attnotnull AND NOT a.attisdropped)
               OR (EXISTS (
                       SELECT 1 FROM pg_index i JOIN pg_attribute x ON x.attrelid = i.indrelid AND x.attnum = i.indkey[0]
                        WHERE i.indrelid = t.relid AND i.indisunique AND i.indpred IS NULL
                          AND cardinality(i.indkey::int2[]) = 1 AND x.attname = t.id_column)
                   AND NOT EXISTS (
                       SELECT 1 FROM pg_index i JOIN pg_attribute x ON x.attrelid = i.indrelid AND x.attnum = i.indkey[0]
                        WHERE i.indrelid = t.relid AND i.indisunique AND i.indisvalid AND i.indimmediate AND i.indpred IS NULL
                          AND cardinality(i.indkey::int2[]) = 1 AND x.attname = t.id_column)) THEN
                RAISE EXCEPTION 'cannot alter id column "%" of %: it must stay NOT NULL and UNIQUE — it identifies the documents (pinecone.track_docs)', t.id_column, v_rel
                    USING ERRCODE = '2BP01', HINT = format('pinecone.untrack_docs(%L) first', v_rel);
            END IF;
            IF t.partition_column IS NOT NULL AND EXISTS (
                SELECT 1 FROM pg_attribute a
                 WHERE a.attrelid = t.relid AND a.attname = t.partition_column AND NOT a.attnotnull AND NOT a.attisdropped) THEN
                RAISE EXCEPTION 'cannot alter column "%" of %: it is the partition column of pinecone.track_docs and must stay NOT NULL', t.partition_column, v_rel
                    USING ERRCODE = '2BP01', HINT = v_hint;
            END IF;
            FOREACH v_name IN ARRAY ARRAY['pg_pinecone_docs_ins', 'pg_pinecone_docs_upd', 'pg_pinecone_docs_del', 'pg_pinecone_docs_trunc'] LOOP
                IF EXISTS (SELECT 1 FROM pg_trigger g WHERE g.tgrelid = t.relid AND g.tgname = v_name AND g.tgenabled NOT IN ('O', 'A')) THEN
                    RAISE EXCEPTION 'cannot disable trigger "%" of %: it keeps the document index in sync (pinecone.track_docs)', v_name, v_rel
                        USING ERRCODE = '2BP01', HINT = format('pinecone.untrack_docs(%L) first', v_rel);
                END IF;
            END LOOP;
        END LOOP;
    END;
    $$;

    REVOKE EXECUTE ON FUNCTION pinecone._docs_on_sql_drop() FROM PUBLIC;
    REVOKE EXECUTE ON FUNCTION pinecone._docs_on_ddl_end() FROM PUBLIC;
    CREATE EVENT TRIGGER pg_pinecone_docs_on_sql_drop ON sql_drop
        EXECUTE FUNCTION pinecone._docs_on_sql_drop();
    CREATE EVENT TRIGGER pg_pinecone_docs_on_ddl_end ON ddl_command_end WHEN TAG IN ('ALTER TABLE', 'CREATE TABLE')
        EXECUTE FUNCTION pinecone._docs_on_ddl_end();
    "#,
    name = "pg_pinecone_docs_ddl_guard",
    requires = [pinecone, "pg_pinecone_doc_tables"],
);
