//! The write path of the document lane: four statement-level triggers per tracked table that
//! queue ids, never documents. `pinecone._docs_sync_trigger()` runs in the writer's transaction
//! (a rollback takes the queue rows with it), checks what Pinecone could not accept anyway (id
//! length, oversize text under `on_oversize => 'error'`, metadata over 40 KB) so the statement
//! fails where the writer can see it (one scan of the statement's rows does all of it), and enqueues `doc_upsert`/`doc_delete` rows of up to 1,000
//! canonical ids each, or one `doc_delete_all` for a TRUNCATE. The worker reads the rows at
//! drain time (`docs_drain.rs`); an UPDATE that changes no declared column enqueues nothing.
//!
//! With a partition column the `doc_delete` entries are `{"id", "p"}` objects, `p` being the
//! canonical text of the partition the row leaves (the drain reads the live row to route an
//! upsert, but a deleted or moved row is gone); a changed partition value is such a delete next
//! to the upsert. A row without a value (23502) or whose namespace would pass 512 bytes (54000)
//! fails the statement.

use pgrx::{
    PgTryBuilder, pg_sys::submodules::panic::CaughtError, prelude::*,
    spi::quote_qualified_identifier,
};

use crate::backend;

/// The trigger names, in the order they are created.
pub(crate) const TRIGGERS: [&str; 4] = [
    "pg_pinecone_docs_ins",
    "pg_pinecone_docs_upd",
    "pg_pinecone_docs_del",
    "pg_pinecone_docs_trunc",
];

extension_sql!(
    r#"
    CREATE FUNCTION pinecone._docs_sync_trigger() RETURNS trigger
    LANGUAGE plpgsql
    SECURITY DEFINER
    SET search_path = pg_catalog, pg_temp
    AS $$
    DECLARE
        v_id_col text;
        v_id_type text;
        v_text_cols jsonb;
        v_meta_cols text[];
        v_vec_col text;
        v_rel text;
        v_id_fmt text;
        v_enqueue text;
        v_rows text;
        v_changed text := '';
        v_col text;
        v_cols text[];
        v_err_cols text[];
        v_big text[];
        v_pairs text;
        v_expr text := '';
        v_idx text;
        v_parts text[];
        v_checks text := '';
        v_ids text[];
        v_found jsonb;
        v_bad text;
        v_n bigint;
        v_msg text;
        v_i int;
        v_part_col text;
        v_prefix text;
        v_enqueue_p text;
    BEGIN
        SELECT id_column, id_type, text_columns, metadata_columns, vector_column, partition_column, namespace
          INTO v_id_col, v_id_type, v_text_cols, v_meta_cols, v_vec_col, v_part_col, v_prefix
          FROM pinecone.doc_tables WHERE relid::oid = TG_RELID;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'pinecone._docs_sync_trigger: table % (relid %) is not registered in pinecone.doc_tables',
                TG_TABLE_NAME, TG_RELID;
        END IF;
        v_rel := format('%I.%I', TG_TABLE_SCHEMA, TG_TABLE_NAME);
        -- canonical, GUC-independent id text; char(n) is compared without its padding
        v_id_fmt := CASE WHEN v_id_type = 'bpchar' THEN 'rtrim(%1$s.%2$I::text)' ELSE '%1$s.%2$I::text' END;
        -- one outbox row per 1,000 ids of the query given as %3$s (a column "id")
        v_enqueue := 'INSERT INTO pinecone.doc_outbox (relid, op, payload) '
            || 'SELECT %1$L::oid::regclass, %2$L, jsonb_agg(c.id ORDER BY c.rn) '
            || 'FROM (SELECT s.id, row_number() OVER () AS rn FROM (%3$s) s) c '
            || 'GROUP BY (c.rn - 1) / 1000 ORDER BY (c.rn - 1) / 1000';
        -- the same for the deletes of a partitioned table: each entry names the partition it leaves
        v_enqueue_p := 'INSERT INTO pinecone.doc_outbox (relid, op, payload) '
            || 'SELECT %1$L::oid::regclass, %2$L, jsonb_agg(jsonb_build_object(''id'', c.id, ''p'', c.p) ORDER BY c.rn) '
            || 'FROM (SELECT s.id, s.p, row_number() OVER () AS rn FROM (%3$s) s) c '
            || 'GROUP BY (c.rn - 1) / 1000 ORDER BY (c.rn - 1) / 1000';

        IF TG_OP = 'TRUNCATE' THEN
            INSERT INTO pinecone.doc_outbox (relid, op, payload)
                VALUES (TG_RELID::regclass, 'doc_delete_all', NULL);
            RETURN NULL;
        END IF;

        IF TG_OP = 'DELETE' THEN
            IF v_part_col IS NULL THEN
                EXECUTE format(v_enqueue, TG_RELID, 'doc_delete',
                    format('SELECT %s AS id FROM old_table o WHERE o.%I IS NOT NULL',
                           format(v_id_fmt, 'o', v_id_col), v_id_col));
            ELSE
                -- a row without a partition value never reached Pinecone
                EXECUTE format(v_enqueue_p, TG_RELID, 'doc_delete',
                    format('SELECT %1$s AS id, o.%2$I::text AS p FROM old_table o WHERE o.%3$I IS NOT NULL AND o.%2$I IS NOT NULL',
                           format(v_id_fmt, 'o', v_id_col), v_part_col, v_id_col));
            END IF;
            RETURN NULL;
        END IF;

        IF TG_OP = 'INSERT' THEN
            v_rows := 'SELECT n.* FROM new_table n';
        ELSE
            -- UPDATE: ids that left the table are deletes; rows with a new id or a changed
            -- declared column are upserts (rows are matched by id, no row identity exists)
            IF v_part_col IS NULL THEN
                EXECUTE format(v_enqueue, TG_RELID, 'doc_delete',
                    format('SELECT %s AS id FROM old_table o WHERE o.%I IS NOT NULL AND NOT EXISTS (SELECT 1 FROM new_table n WHERE n.%I = o.%I)',
                           format(v_id_fmt, 'o', v_id_col), v_id_col, v_id_col, v_id_col));
            ELSE
                -- with a partition column also the rows that moved: deleted from the partition they
                -- leave (the text is compared byte for byte, whatever the column's collation)
                EXECUTE format(v_enqueue_p, TG_RELID, 'doc_delete',
                    format('SELECT %1$s AS id, o.%2$I::text AS p FROM old_table o WHERE o.%3$I IS NOT NULL AND o.%2$I IS NOT NULL '
                           'AND NOT EXISTS (SELECT 1 FROM new_table n WHERE n.%3$I = o.%3$I '
                           'AND n.%2$I::text COLLATE "C" IS NOT DISTINCT FROM o.%2$I::text COLLATE "C")',
                           format(v_id_fmt, 'o', v_id_col), v_part_col, v_id_col));
            END IF;
            SELECT array_agg(k) INTO v_cols FROM (
                SELECT jsonb_object_keys(v_text_cols) AS k
                UNION SELECT unnest(v_meta_cols)
                UNION SELECT v_vec_col WHERE v_vec_col IS NOT NULL) d;
            FOREACH v_col IN ARRAY coalesce(v_cols, '{}') LOOP
                IF v_col = v_vec_col THEN
                    -- real[] comparison: the vector type's own operators are not on this search_path
                    v_changed := v_changed || format(' OR n.%1$I::real[] IS DISTINCT FROM o.%1$I::real[]', v_col);
                ELSE
                    v_changed := v_changed || format(' OR n.%1$I IS DISTINCT FROM o.%1$I', v_col);
                END IF;
            END LOOP;
            -- a row that moves to another partition is a change even if nothing else is
            IF v_part_col IS NOT NULL THEN
                v_changed := v_changed || format(' OR n.%1$I::text COLLATE "C" IS DISTINCT FROM o.%1$I::text COLLATE "C"', v_part_col);
            END IF;
            v_rows := format('SELECT n.* FROM new_table n LEFT JOIN old_table o ON o.%1$I = n.%1$I WHERE o.%1$I IS NULL%2$s',
                             v_id_col, v_changed);
        END IF;

        -- text under on_oversize => 'error': what truncation would have to cut is refused
        SELECT coalesce(array_agg(key ORDER BY key), '{}') INTO v_err_cols
          FROM jsonb_each(v_text_cols) WHERE value ->> 'on_oversize' = 'error';

        -- metadata: 40 KB per document; only columns that can be large are measured
        SELECT coalesce(array_agg(a.attname::text ORDER BY a.attnum), '{}') INTO v_big
          FROM pg_attribute a
         WHERE a.attrelid = TG_RELID AND a.attname = ANY (v_meta_cols) AND a.attnum > 0 AND NOT a.attisdropped
           AND a.atttypid IN ('text'::regtype, 'varchar'::regtype, 'bpchar'::regtype, 'text[]'::regtype);
        IF cardinality(v_big) > 0 THEN
            -- jsonb_build_object takes at most 100 arguments
            FOR v_i IN 0 .. (cardinality(v_big) - 1) / 40 LOOP
                SELECT string_agg(format('%L, r.%I', c, c), ', ') INTO v_pairs
                  FROM unnest(v_big[v_i * 40 + 1 : v_i * 40 + 40]) c;
                v_expr := v_expr || CASE WHEN v_i > 0 THEN ' || ' ELSE '' END
                    || format('jsonb_build_object(%s)', v_pairs);
            END LOOP;
        END IF;

        -- one scan of the rows: the ids to enqueue and the first offender of every check
        v_idx := format(v_id_fmt, 'r', v_id_col);
        -- Pinecone takes ids of 1 to 512 bytes
        v_parts := ARRAY[format('%L, min(%s) FILTER (WHERE r.%I IS NOT NULL AND octet_length(%s) NOT BETWEEN 1 AND 512)',
                                'id', v_idx, v_id_col, v_idx)];
        IF v_part_col IS NOT NULL THEN
            -- every row needs a partition value, and the namespace it makes must fit Pinecone's 512 bytes
            v_parts := v_parts || format('%L, min(%s) FILTER (WHERE r.%I IS NOT NULL AND r.%I IS NULL)',
                                         'pn', v_idx, v_id_col, v_part_col);
            v_parts := v_parts || format('%L, min(%s) FILTER (WHERE r.%I IS NOT NULL AND pinecone._partition_namespace_bytes(%L, r.%I::text) > 512)',
                                         'pl', v_idx, v_id_col, v_prefix, v_part_col);
        END IF;
        FOR v_i IN 1 .. cardinality(v_err_cols) LOOP
            v_parts := v_parts || format('%L, min(%s) FILTER (WHERE pinecone._doc_text_over_limit(r.%I))',
                                         'e' || v_i, v_idx, v_err_cols[v_i]);
        END LOOP;
        IF v_expr <> '' THEN
            v_parts := v_parts || format('%L, min(%s) FILTER (WHERE octet_length((%s)::text) > 40960)',
                                         'meta', v_idx, v_expr);
        END IF;
        FOR v_i IN 0 .. (cardinality(v_parts) - 1) / 40 LOOP
            v_checks := v_checks || CASE WHEN v_i > 0 THEN ' || ' ELSE '' END
                || format('jsonb_build_object(%s)', array_to_string(v_parts[v_i * 40 + 1 : v_i * 40 + 40], ', '));
        END LOOP;
        EXECUTE format('SELECT array_agg(%1$s) FILTER (WHERE r.%2$I IS NOT NULL), %3$s FROM (%4$s) r',
                       v_idx, v_id_col, v_checks, v_rows) INTO v_ids, v_found;

        v_bad := v_found ->> 'id';
        IF v_bad IS NOT NULL THEN
            EXECUTE format('SELECT octet_length(%1$s) FROM (%2$s) r WHERE %1$s = $1 LIMIT 1', v_idx, v_rows)
                INTO v_n USING v_bad;
            IF v_n = 0 THEN
                RAISE EXCEPTION 'pg_pinecone: % has an empty id — Pinecone ids are 1 to 512 bytes', v_rel
                    USING ERRCODE = '54000';
            END IF;
            RAISE EXCEPTION 'pg_pinecone: % id is % bytes — Pinecone allows 512', v_rel, v_n
                USING ERRCODE = '54000';
        END IF;
        IF v_part_col IS NOT NULL THEN
            v_bad := v_found ->> 'pn';
            IF v_bad IS NOT NULL THEN
                RAISE EXCEPTION 'pg_pinecone: % id "%": partition column "%" is NULL — every row of a partitioned document table needs a value', v_rel, v_bad, v_part_col
                    USING ERRCODE = '23502', HINT = format('give "%s" a value: the table is partitioned by it', v_part_col);
            END IF;
            v_bad := v_found ->> 'pl';
            IF v_bad IS NOT NULL THEN
                EXECUTE format('SELECT pinecone._partition_namespace_bytes(%1$L, r.%2$I::text) FROM (%3$s) r WHERE %4$s = $1 LIMIT 1',
                               v_prefix, v_part_col, v_rows, v_idx) INTO v_n USING v_bad;
                RAISE EXCEPTION 'pg_pinecone: % id "%": its partition value makes a namespace of % bytes — Pinecone allows 512', v_rel, v_bad, v_n
                    USING ERRCODE = '54000', HINT = format('shorten the value of "%s"', v_part_col);
            END IF;
        END IF;
        FOR v_i IN 1 .. cardinality(v_err_cols) LOOP
            v_bad := v_found ->> ('e' || v_i);
            IF v_bad IS NOT NULL THEN
                EXECUTE format('SELECT pinecone._doc_text_oversize_message(%3$L, r.%4$I) FROM (%2$s) r WHERE %1$s = $1 LIMIT 1',
                               v_idx, v_rows, v_err_cols[v_i], v_err_cols[v_i]) INTO v_msg USING v_bad;
                RAISE EXCEPTION 'pg_pinecone: % id "%": %', v_rel, v_bad, v_msg
                    USING ERRCODE = '54000',
                          HINT = 'declare on_oversize => ''truncate'' for this column, or shorten the value';
            END IF;
        END LOOP;
        v_bad := v_found ->> 'meta';
        IF v_bad IS NOT NULL THEN
            EXECUTE format('SELECT octet_length((%1$s)::text) FROM (%2$s) r WHERE %3$s = $1 LIMIT 1', v_expr, v_rows, v_idx)
                INTO v_n USING v_bad;
            RAISE EXCEPTION 'pg_pinecone: % id "%": metadata is % bytes — Pinecone allows 40 KB', v_rel, v_bad, v_n
                USING ERRCODE = '54000', HINT = 'remove large columns from metadata_columns';
        END IF;

        IF v_ids IS NOT NULL THEN
            INSERT INTO pinecone.doc_outbox (relid, op, payload)
            SELECT TG_RELID::regclass, 'doc_upsert', jsonb_agg(u.id ORDER BY u.rn)
              FROM unnest(v_ids) WITH ORDINALITY AS u(id, rn)
             GROUP BY (u.rn - 1) / 1000 ORDER BY (u.rn - 1) / 1000;
        END IF;
        RETURN NULL;
    END;
    $$;
    REVOKE EXECUTE ON FUNCTION pinecone._docs_sync_trigger() FROM PUBLIC;
    "#,
    name = "pg_pinecone_docs_sync_trigger_fn",
    requires = [
        pinecone,
        "pg_pinecone_doc_tables",
        "pg_pinecone_doc_outbox",
        _doc_text_over_limit,
        _doc_text_tokens,
        _doc_text_oversize_message,
        _partition_namespace_bytes
    ],
);

/// Runs one DDL statement of the registry's owner on `table`. A privilege error says what that
/// role is missing (`hint`); any other ERROR is raised again unchanged, with its SQLSTATE.
fn run_ddl(statement: &str, failure: &str, hint: &str) {
    PgTryBuilder::new(|| {
        if Spi::run(statement).is_err() {
            backend::error_with_code(PgSqlErrorCode::ERRCODE_INTERNAL_ERROR, failure, None, None);
        }
    })
    .catch_others(|caught| {
        let (CaughtError::PostgresError(report)
        | CaughtError::ErrorReport(report)
        | CaughtError::RustPanic {
            ereport: report, ..
        }) = &caught;
        if report.sql_error_code() == PgSqlErrorCode::ERRCODE_INSUFFICIENT_PRIVILEGE {
            backend::error_with_code(
                PgSqlErrorCode::ERRCODE_INSUFFICIENT_PRIVILEGE,
                report.message(),
                report.detail(),
                Some(hint),
            );
        }
        caught.rethrow()
    })
    .execute();
}

/// Creates the four triggers on `table` (a quoted, qualified name), as the calling role: it needs
/// the TRIGGER privilege on the table and EXECUTE on `pinecone._docs_sync_trigger()`.
pub(crate) fn install_triggers(schema: &str, table: &str) {
    let table = quote_qualified_identifier(schema, table);
    let function = "FOR EACH STATEMENT EXECUTE FUNCTION pinecone._docs_sync_trigger()";
    let statements = [
        format!(
            "CREATE TRIGGER {} AFTER INSERT ON {table} REFERENCING NEW TABLE AS new_table {function}",
            TRIGGERS[0]
        ),
        format!(
            "CREATE TRIGGER {} AFTER UPDATE ON {table} \
             REFERENCING NEW TABLE AS new_table OLD TABLE AS old_table {function}",
            TRIGGERS[1]
        ),
        format!(
            "CREATE TRIGGER {} AFTER DELETE ON {table} REFERENCING OLD TABLE AS old_table {function}",
            TRIGGERS[2]
        ),
        format!(
            "CREATE TRIGGER {} AFTER TRUNCATE ON {table} {function}",
            TRIGGERS[3]
        ),
    ];
    let failure = format!("pinecone.track_docs: could not install the sync triggers on {table}");
    let hint = format!(
        "the owner of pinecone.doc_tables needs TRIGGER on {table} and EXECUTE on \
         pinecone._docs_sync_trigger() (a superuser has both)"
    );
    for statement in statements {
        run_ddl(&statement, &failure, &hint);
    }
}

/// Drops the four triggers; missing ones are fine, so an interrupted untrack can be re-run.
pub(crate) fn drop_triggers(schema: &str, table: &str) {
    let table = quote_qualified_identifier(schema, table);
    let failure = format!("pinecone.untrack_docs: could not drop the sync triggers of {table}");
    let hint = format!(
        "the owner of pinecone.doc_tables must own {table} to drop its sync triggers (a superuser \
         can)"
    );
    for name in TRIGGERS {
        run_ddl(
            &format!("DROP TRIGGER IF EXISTS {name} ON {table}"),
            &failure,
            &hint,
        );
    }
}
