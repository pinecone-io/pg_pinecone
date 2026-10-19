-- create_index builds 'native' when engine is left out (it built 'pgvector'); OR REPLACE keeps owner and grants.
CREATE OR REPLACE FUNCTION pinecone."create_index"(
	"tbl" regclass, /* PgRelation */
	"col" TEXT, /* & str */
	"engine" TEXT DEFAULT 'native', /* & str */
	"metric" TEXT DEFAULT 'cosine', /* & str */
	"index_name" TEXT DEFAULT NULL, /* Option < & str > */
	"pinecone_index" TEXT DEFAULT NULL, /* Option < & str > */
	"pinecone_namespace" TEXT DEFAULT NULL, /* Option < & str > */
	"attach_unchecked" bool DEFAULT false, /* bool */
	"filter_column" TEXT DEFAULT NULL, /* Option < & str > */
	"partition_column" TEXT DEFAULT NULL /* Option < & str > */
) RETURNS void
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'create_index_wrapper';

-- DROP INDEX of an hnsw index create_index built forgets its pinecone.tables row (it kept it). The body is am.rs's
-- byte for byte, indentation included: the catalog compare checks it.
    CREATE OR REPLACE FUNCTION pinecone._on_sql_drop() RETURNS event_trigger
    LANGUAGE plpgsql
    SECURITY DEFINER
    SET search_path = pg_catalog, pg_temp
    AS $$
    DECLARE
        obj record;
        v_relid oid;
    BEGIN
        -- A dropped table's partition namespaces are forgotten (Pinecone is left alone), whichever
        -- lane synced it; the AM index below takes its registry row with it. A trigger-lane row stays
        -- as `<dropped:OID>` for `pinecone.status()`, unless the table was partitioned: its row
        -- claims a namespace family that nothing could release, and the namespaces are already gone.
        FOR obj IN SELECT * FROM pg_event_trigger_dropped_objects() WHERE object_type = 'table' LOOP
            DELETE FROM pinecone.partitions WHERE relid::oid = obj.objid AND lane = 'vector';
            IF EXISTS (SELECT 1 FROM pinecone.tables WHERE relid::oid = obj.objid
                       AND synced_by = 'trigger' AND partition_column IS NOT NULL) THEN
                DELETE FROM pinecone.outbox WHERE relid::oid = obj.objid;
                DELETE FROM pinecone.tables WHERE relid::oid = obj.objid;
            END IF;
        END LOOP;
        FOR obj IN SELECT * FROM pg_event_trigger_dropped_objects() WHERE object_type = 'index' LOOP
            SELECT relid::oid INTO v_relid FROM pinecone.tables WHERE am_index_relid = obj.objid;
            IF FOUND THEN
                DELETE FROM pinecone.outbox WHERE relid::oid = v_relid;
                DELETE FROM pinecone.partitions WHERE relid::oid = v_relid AND lane = 'vector';
                DELETE FROM pinecone.tables WHERE am_index_relid = obj.objid;
            END IF;
            -- `pinecone._am_engine` (engine stickiness across
            -- REINDEX): keyed by the index's own oid directly, unconditional
            -- (no matching `pinecone.tables` row needed — local-mode indexes
            -- never have one), a no-op DELETE for anything not tracked here.
            DELETE FROM pinecone._am_engine WHERE index_relid = obj.objid;
        END LOOP;
        -- create_index(engine => 'pgvector')'s own row has no `am_index_relid`: it goes with the table's
        -- last hnsw index. A row promote() or track() made stays, and so does one naming a Pinecone index
        -- (a demote keeps it); a track() on top of create_index does not take the row over.
        IF EXISTS (SELECT 1 FROM pg_event_trigger_dropped_objects() WHERE object_type = 'index') THEN
            DELETE FROM pinecone.tables t
             WHERE t.engine = 'pgvector' AND t.synced_by = 'engine' AND t.mode = 'local' AND t.index_name IS NULL
               AND NOT EXISTS (SELECT 1 FROM pg_index i
                                 JOIN pg_class c ON c.oid = i.indexrelid
                                 JOIN pg_am a ON a.oid = c.relam
                                WHERE i.indrelid = t.relid::oid AND a.amname = 'hnsw');
        END IF;
    END;
    $$;

-- The rows 0.10.0 kept for hnsw indexes dropped under it.
DELETE FROM pinecone.tables t
 WHERE t.engine = 'pgvector' AND t.synced_by = 'engine' AND t.mode = 'local' AND t.index_name IS NULL
   AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_index i
                     JOIN pg_catalog.pg_class c ON c.oid = i.indexrelid
                     JOIN pg_catalog.pg_am a ON a.oid = c.relam
                    WHERE i.indrelid = t.relid::oid AND a.amname = 'hnsw');
