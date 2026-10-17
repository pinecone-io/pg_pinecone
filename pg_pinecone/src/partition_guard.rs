//! The DDL guard of the partition column of `pinecone.tables`: every namespace comes from it, so
//! while a table is registered it cannot be dropped, must keep its name and type and stays NOT
//! NULL. Two event triggers, both `SECURITY DEFINER` so any role's unrelated DDL passes without
//! grants on the registry; both return at once while no table is partitioned.

use pgrx::prelude::*;

extension_sql!(
    r#"
    CREATE FUNCTION pinecone._partition_on_sql_drop() RETURNS event_trigger
    LANGUAGE plpgsql
    SECURITY DEFINER
    SET search_path = pg_catalog, pg_temp
    AS $$
    DECLARE
        obj record;
        v_rel text;
        v_by text;
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pinecone.tables WHERE partition_column IS NOT NULL) THEN
            RETURN;
        END IF;
        FOR obj IN SELECT * FROM pg_event_trigger_dropped_objects() WHERE object_type = 'table column' LOOP
            SELECT t.synced_by INTO v_by FROM pinecone.tables t
             WHERE t.relid::oid = obj.objid AND t.partition_column = obj.address_names[3];
            IF FOUND
               AND NOT EXISTS (SELECT 1 FROM pg_event_trigger_dropped_objects() d
                                WHERE d.object_type = 'table' AND d.objid = obj.objid) THEN
                v_rel := format('%I.%I', obj.address_names[1], obj.address_names[2]);
                RAISE EXCEPTION 'cannot drop column "%" of %: it is the partition column of %', obj.address_names[3], v_rel,
                                CASE WHEN v_by = 'am' THEN 'its pg_pinecone index' ELSE 'pinecone.promote' END
                    USING ERRCODE = '2BP01',
                          HINT = CASE WHEN v_by = 'am'
                                      THEN format('DROP INDEX the pg_pinecone index of %s first, or pinecone.set_engine(%L, ''native'')', v_rel, v_rel)
                                      ELSE format('pinecone.demote(%L, drop_remote_index => true) first', v_rel) END;
            END IF;
        END LOOP;
    END;
    $$;

    CREATE FUNCTION pinecone._partition_on_ddl_end() RETURNS event_trigger
    LANGUAGE plpgsql
    SECURITY DEFINER
    SET search_path = pg_catalog, pg_temp
    AS $$
    DECLARE
        t record;
        v_rel text;
        v_kind text;
        v_not_null boolean;
        v_altered oid[];
        v_what text;
        v_hint text;
    BEGIN
        SELECT coalesce(array_agg(objid), '{}') INTO v_altered
          FROM pg_event_trigger_ddl_commands() WHERE classid = 'pg_class'::regclass;
        IF cardinality(v_altered) = 0 THEN
            RETURN;
        END IF;
        FOR t IN SELECT x.relid::oid AS relid, x.partition_column, x.partition_kind, x.synced_by, n.nspname, k.relname
                   FROM pinecone.tables x
                   JOIN pg_class k ON k.oid = x.relid::oid
                   JOIN pg_namespace n ON n.oid = k.relnamespace
                  WHERE x.partition_column IS NOT NULL AND x.relid::oid = ANY (v_altered) LOOP
            v_rel := format('%I.%I', t.nspname, t.relname);
            v_what := CASE WHEN t.synced_by = 'am' THEN 'its pg_pinecone index' ELSE 'pinecone.promote' END;
            v_hint := CASE WHEN t.synced_by = 'am'
                           THEN format('DROP INDEX the pg_pinecone index of %s first, or pinecone.set_engine(%L, ''native'')', v_rel, v_rel)
                           ELSE format('pinecone.demote(%L, drop_remote_index => true) first', v_rel) END;
            SELECT CASE y.typname::text WHEN 'varchar' THEN 'text' ELSE y.typname::text END, a.attnotnull
              INTO v_kind, v_not_null
              FROM pg_attribute a JOIN pg_type y ON y.oid = a.atttypid
             WHERE a.attrelid = t.relid AND a.attname = t.partition_column AND a.attnum > 0 AND NOT a.attisdropped;
            IF v_kind IS DISTINCT FROM t.partition_kind THEN
                RAISE EXCEPTION 'cannot alter column "%" of %: it is the partition column of % and must keep its name and type', t.partition_column, v_rel, v_what
                    USING ERRCODE = '2BP01', HINT = v_hint;
            END IF;
            IF NOT v_not_null THEN
                RAISE EXCEPTION 'cannot alter column "%" of %: the partition column of % must stay NOT NULL', t.partition_column, v_rel, v_what
                    USING ERRCODE = '2BP01', HINT = v_hint;
            END IF;
        END LOOP;
    END;
    $$;

    REVOKE EXECUTE ON FUNCTION pinecone._partition_on_sql_drop() FROM PUBLIC;
    REVOKE EXECUTE ON FUNCTION pinecone._partition_on_ddl_end() FROM PUBLIC;
    CREATE EVENT TRIGGER pg_pinecone_partition_on_sql_drop ON sql_drop
        EXECUTE FUNCTION pinecone._partition_on_sql_drop();
    CREATE EVENT TRIGGER pg_pinecone_partition_on_ddl_end ON ddl_command_end WHEN TAG IN ('ALTER TABLE')
        EXECUTE FUNCTION pinecone._partition_on_ddl_end();
    "#,
    name = "pg_pinecone_partition_ddl_guard",
    requires = [pinecone, "pg_pinecone_tables"],
);
