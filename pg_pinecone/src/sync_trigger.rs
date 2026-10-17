//! `pinecone._sync_trigger()`: the ONE generic statement-level trigger function
//! `pinecone.promote` installs on a promoted table (four triggers —
//! INSERT/UPDATE/DELETE/TRUNCATE — all pointing at this single function; see
//! `promote::install_triggers`). `SECURITY DEFINER` so arbitrary writer
//! sessions can push to Pinecone without needing any privilege on
//! `pinecone.secret` themselves; per-table identifiers never appear in a
//! trigger body — the function reads the registry (and transition tables) by
//! `TG_RELID`/`TG_OP` at runtime, so no generated-DDL injection surface exists
//! here.
//!
//! Declared here (rather than inside the `pg_pinecone` `#[pg_schema]` module in
//! `lib.rs`) for the same reason as `pinecone.tables` in `registry.rs`: an
//! `extension_sql!` with an explicit `requires` can't live inside an inline
//! `#[pg_schema] mod {}`.
//!
//! ## Dispatch (plpgsql body)
//! - `INSERT` on a `storage = 'full'` table: pushes every row of `new_table`
//!   with a non-`NULL` vector.
//! - `INSERT` on a `storage = 'detached'` table: any non-`NULL` vector
//!   in the statement ERRORs (remote-only column, same rule as the detached
//!   UPDATE write guard; the whole statement aborts — never a partial
//!   NULL-rows-kept split); an all-`NULL`-vector statement succeeds and
//!   enqueues nothing (no remote counterpart to create).
//! - `UPDATE` on a `storage = 'full'` table (every table before 0.6, and
//!   every 0.6 one that hasn't been detached): (a) deletes ids from
//!   `old_table` that no longer appear in `new_table` (a primary-key
//!   change); (b) skips rows whose id, vector, and every metadata column are
//!   all `IS NOT DISTINCT FROM` their old counterpart (the no-op-update
//!   guard — zero HTTP calls); (c) deletes ids whose vector went from
//!   non-`NULL` to `NULL`; (d) pushes everything else (a genuine change, or a
//!   row with no old counterpart because its id changed away from
//!   something else).
//! - `UPDATE` on a `storage = 'detached'` table (Pinecone
//!   holds the only full copy of the vector column): an id change ERRORs
//!   (the old id's remote vector would be deleted with nothing to
//!   replace it); a write that gives the vector column an actual new
//!   non-`NULL` value ERRORs too (it's remote-only while detached — a local
//!   write can't reach Pinecone); a transition TO `NULL` is a silent no-op
//!   for the vector itself (no `delete` is ever enqueued, since that
//!   would wipe Pinecone's only copy); an unchanged non-`NULL` vector whose
//!   metadata changed is an ordinary full re-upsert (the real values are
//!   already local); a metadata edit on any row whose NEW vector is `NULL` —
//!   whether it was already `NULL` or went `NULL` in this same statement
//!   — becomes an `'update'` (merge-only `setMetadata` patch)
//!   or a `'reupsert'` (fetch_vectors + full re-upsert, needed when a
//!   changed field's new value is `NULL` — `setMetadata` can only ever merge,
//!   never unset a key) outbox op, so remote metadata never silently goes
//!   stale just because the local vector was shed.
//! - `DELETE`: deletes every id in `old_table` — correct regardless of
//!   `storage` (the row, and so its only remaining copy, is gone either way).
//! - `TRUNCATE`: `deleteAll` the namespace (no transition tables exist for
//!   `TRUNCATE`, and none are needed — the whole namespace empties) — also
//!   correct regardless of `storage`: emptying the namespace is exactly what
//!   `TRUNCATE` means, detached or not.
//!
//! Batches are built as a single `jsonb` value per statement (via
//! `jsonb_agg`/`jsonb_build_object`, correlating `old_table`/`new_table` by
//! `id_column` — matched-by-identity is not attempted, so a same-statement id
//! swap between two rows converges correctly but doesn't take the no-op-guard
//! fast path; see the module tests) and, since 0.3, enqueued whole into
//! `pinecone.outbox` (`op` = `'upsert'`/`'delete'`/`'delete_all'`, `payload`
//! = the batch itself) rather than pushed directly. If the table's
//! `sync_mode = 'sync'`, this same function then calls
//! `pinecone._drain_relid(TG_RELID)` (see `outbox.rs`) before returning, which
//! drains — and so, for that mode only, still synchronously HTTP-pushes —
//! everything just enqueued (chunking at ≤1000 rows/1.9MB, matching
//! `pinecone.promote`'s backfill budget) through the internal helpers below.
//! `'async'` tables (the default) enqueue only; nothing pushes until
//! something drains (a background worker).
//!
//! ## Internal helpers (`_push_rows`/`_delete_ids`/`_delete_all`)
//! Declared as ordinary `#[pg_extern]` functions inside `pg_pinecone`'s
//! `#[pg_schema]` module in `lib.rs` (so they're schema-qualified
//! `pinecone._push_rows` etc.), with their SQL-facing logic delegated back
//! here (as a `Result`-returning `try_*` core each, plus a panicking
//! wrapper — `outbox::drain_relid` uses the former to record a failing
//! row's `attempts`/`last_error` before re-raising). They run `SECURITY
//! INVOKER`, but since the only intended callers are this file's `SECURITY
//! DEFINER` trigger function (inline, for `sync_mode = 'sync'`) and
//! `pinecone._drain_relid` (which the trigger calls, so it too runs as the
//! trigger's definer), they execute with the trigger owner's privileges by
//! call-chain inheritance — no privilege on `pinecone.secret` is required of
//! the writer session itself. `REVOKE EXECUTE FROM PUBLIC` (below) closes
//! the direct-call path: without it, any session with `pinecone.api_key` set
//! could call `pinecone._push_rows(<any relid>, <arbitrary batch>)` directly
//! and push fabricated vectors into another tracked table's index/namespace.
use std::collections::HashMap;

use pg_pinecone_client::{
    Client, DeleteVectorsRequest, Error as ClientError, FetchVectorsRequest, UPSERT_MAX_BYTES,
    UPSERT_MAX_RECORDS, UpdateVectorMetadataRequest, UpsertVectorsRequest, Vector,
};
use pgrx::{JsonB, prelude::*};
use serde_json::Value;

use crate::{
    backend,
    partition::{Lane, PartitionSpec},
    partition_lane::{self, DeleteEntry, Gate},
    partition_ops, partition_registry, registry,
};

/// Vectors per `_push_rows` request; also the row budget of [`try_reupsert_metadata`]. `pub(crate)`
/// so tests can build a fixture one row past it.
pub(crate) const PUSH_MAX_ROWS: usize = UPSERT_MAX_RECORDS;
/// Pinecone's limit of ids per delete request.
pub(crate) const DELETE_MAX_IDS: usize = 1000;

extension_sql!(
    r#"
    CREATE FUNCTION pinecone._partition_check_batch(p_table text, p_column text, p_prefix text, p_batch jsonb)
    RETURNS void
    LANGUAGE plpgsql
    SET search_path = pg_catalog, pg_temp
    AS $$
    DECLARE
        v_id text;
        v_n int;
    BEGIN
        SELECT e ->> 'id' INTO v_id FROM jsonb_array_elements(p_batch) e WHERE e -> 'p' = 'null'::jsonb LIMIT 1;
        IF FOUND THEN
            RAISE EXCEPTION 'pg_pinecone: % id "%": partition column "%" is NULL: every row of a partitioned table needs a value', p_table, v_id, p_column
                USING ERRCODE = '23502', HINT = format('give "%s" a value: the table is partitioned by it', p_column);
        END IF;
        SELECT e ->> 'id', pinecone._partition_namespace_bytes(p_prefix, e ->> 'p') INTO v_id, v_n
          FROM jsonb_array_elements(p_batch) e
         WHERE pinecone._partition_namespace_bytes(p_prefix, e ->> 'p') > 512 LIMIT 1;
        IF FOUND THEN
            RAISE EXCEPTION 'pg_pinecone: % id "%": its partition value makes a namespace of % bytes, Pinecone allows 512', p_table, v_id, v_n
                USING ERRCODE = '54000', HINT = format('shorten the value of "%s"', p_column);
        END IF;
    END;
    $$;
    REVOKE EXECUTE ON FUNCTION pinecone._partition_check_batch(text, text, text, jsonb) FROM PUBLIC;

    CREATE FUNCTION pinecone._sync_trigger() RETURNS trigger
    LANGUAGE plpgsql
    SECURITY DEFINER
    SET search_path = pg_catalog, pg_temp
    AS $$
    DECLARE
        v_vector_col text;
        v_id_col text;
        v_metadata_cols text[];
        v_metadata_pairs text[] := '{}';
        v_metadata_field text := '';
        v_metadata_distinct text := '';
        v_metadata_null_transition text := '';
        v_col text;
        v_batch jsonb;
        v_delete_ids jsonb;
        v_sync_mode text;
        v_storage text;
        v_id_changed boolean;
        v_vector_written boolean;
        v_meta_update_batch jsonb;
        v_meta_reupsert_batch jsonb;
        v_namespace text;
        v_part_col text;
        v_part_moved text := '';
        v_item_tail text;
        v_table text;
    BEGIN
        -- `pinecone.recover_from_remote`'s dispatch flags the
        -- ONE relid it is writing recovered values back into (a per-backend
        -- Rust static behind this C function — see `recover.rs`'s
        -- `RECOVERING_RELID` for why it must NOT be a GUC: a GUC is
        -- spoofable by a session that never loaded the library, whereas
        -- calling a C function forces the load, and the static is only
        -- ever set by the recover code itself). Without this early return,
        -- the write-back would re-enqueue the very value it just pulled FROM
        -- Pinecone, a pure round-trip. Keyed to TG_RELID so a user trigger
        -- cascading a write to a DIFFERENT promoted table mid-recover still
        -- syncs normally.
        IF pinecone._is_recovering(TG_RELID) THEN
            RETURN NULL;
        END IF;

        SELECT vector_column, id_column, metadata_columns, sync_mode, storage, namespace, partition_column
          INTO v_vector_col, v_id_col, v_metadata_cols, v_sync_mode, v_storage, v_namespace, v_part_col
          FROM pinecone.tables WHERE relid::oid = TG_RELID;

        IF NOT FOUND THEN
            RAISE EXCEPTION 'pinecone._sync_trigger: table % (relid %) is not registered in pinecone.tables',
                TG_TABLE_NAME, TG_RELID;
        END IF;

        FOREACH v_col IN ARRAY v_metadata_cols LOOP
            v_metadata_pairs := v_metadata_pairs || format('%L, new_table.%I', v_col, v_col);
            v_metadata_distinct := v_metadata_distinct
                || format(' OR new_table.%I IS DISTINCT FROM old_table.%I', v_col, v_col);
            -- Scoped metadata-divergence fix: a CHANGED
            -- metadata column whose NEW value is NULL is an "unset" —
            -- Pinecone's setMetadata endpoint only ever MERGES, so it can
            -- never clear a key; a row with at least one such transition
            -- needs fetch_vectors + full re-upsert ('reupsert') instead of
            -- the cheap merge-only PATCH ('update').
            v_metadata_null_transition := v_metadata_null_transition
                || format(' OR (new_table.%I IS NULL AND old_table.%I IS NOT NULL)', v_col, v_col);
        END LOOP;
        IF v_part_col IS NOT NULL THEN
            v_table := format('%I.%I', TG_TABLE_SCHEMA, TG_TABLE_NAME);
            v_part_moved := format('new_table.%1$I::text COLLATE "C" IS DISTINCT FROM old_table.%1$I::text COLLATE "C"', v_part_col);
            v_metadata_distinct := v_metadata_distinct || ' OR ' || v_part_moved;
            IF NOT (v_part_col = ANY (v_metadata_cols)) THEN
                v_metadata_pairs := v_metadata_pairs || format('%L, new_table.%I', v_part_col, v_part_col);
            END IF;
        END IF;
        IF array_length(v_metadata_pairs, 1) > 0 THEN
            v_metadata_field := format(', %L, jsonb_strip_nulls(jsonb_build_object(%s))',
                                        'metadata', array_to_string(v_metadata_pairs, ', '));
        END IF;
        v_item_tail := v_metadata_field;
        IF v_part_col IS NOT NULL THEN
            v_item_tail := v_item_tail || format(', %L, new_table.%I::text', 'p', v_part_col);
        END IF;

        IF TG_OP = 'INSERT' AND v_storage = 'detached' THEN
            -- Same write guard as the detached UPDATE
            -- branch below — the vector column is remote-only while detached,
            -- so an INSERT carrying a non-NULL vector ERRORs (a locally-
            -- inserted value can never coherently reach Pinecone's
            -- authoritative copy). Statement-level, so a multi-row statement
            -- mixing NULL and non-NULL vectors aborts WHOLE — never a
            -- partial "NULL rows kept" split. NULL-vector inserts succeed
            -- and enqueue NOTHING: there is no remote counterpart to create.
            EXECUTE format(
                'SELECT EXISTS (SELECT 1 FROM new_table WHERE new_table.%I IS NOT NULL)',
                v_vector_col
            ) INTO v_vector_written;
            IF v_vector_written THEN
                RAISE EXCEPTION 'pg_pinecone: % is detached (pinecone.tables.storage = ''detached'') — % is remote-only while detached; INSERTs may not carry a vector value — run pinecone.attach_storage(...) first',
                    TG_TABLE_NAME, v_vector_col;
            END IF;

        ELSIF TG_OP = 'INSERT' THEN
            EXECUTE format(
                'SELECT COALESCE(jsonb_agg(jsonb_build_object(%L, new_table.%I::text, %L, to_jsonb(new_table.%I::real[])%s)), ''[]''::jsonb) FROM new_table WHERE new_table.%I IS NOT NULL',
                'id', v_id_col, 'values', v_vector_col, v_item_tail, v_vector_col
            ) INTO v_batch;

            IF v_part_col IS NOT NULL AND jsonb_array_length(v_batch) > 0 THEN
                PERFORM pinecone._partition_check_batch(v_table, v_part_col, v_namespace, v_batch);
            END IF;

            IF jsonb_array_length(v_batch) > 0 THEN
                INSERT INTO pinecone.outbox (relid, op, payload) VALUES (TG_RELID::regclass, 'upsert', v_batch);
            END IF;

        ELSIF TG_OP = 'UPDATE' AND v_storage = 'detached' THEN
            -- Detached semantics replace the ordinary
            -- (a)-(d) dispatch in the sibling branch below entirely:
            -- Pinecone holds the only remaining full copy of this table's
            -- vector column, so id/vector writes need different rules (see
            -- this file's module rustdoc for the detached write rules and the metadata split).

            -- id_column is immutable while detached — the ordinary
            -- branch (a) would enqueue a 'delete' for the OLD id with
            -- nothing to replace it (a detached row's vector is NULL, so
            -- nothing re-upserts under the new id), silently losing
            -- Pinecone's only copy of that row.
            EXECUTE format(
                'SELECT EXISTS (SELECT 1 FROM old_table WHERE NOT EXISTS (SELECT 1 FROM new_table WHERE new_table.%I = old_table.%I))',
                v_id_col, v_id_col
            ) INTO v_id_changed;
            IF v_id_changed THEN
                RAISE EXCEPTION 'pg_pinecone: % is detached (pinecone.tables.storage = ''detached'') — % is immutable while detached (Pinecone holds the only remaining copy of this table''s vectors; changing it would silently lose one) — run pinecone.attach_storage(...) first',
                    TG_TABLE_NAME, v_id_col;
            END IF;

            -- The companion write guard: the vector column is
            -- remote-only while detached. A transition TO NULL (the future
            -- detach_storage NULLing pass itself, or any other NULL-ing
            -- write) is the expected state and must stay a silent no-op — it
            -- must NOT enqueue a 'delete' (that would wipe Pinecone's only
            -- copy). Any OTHER write to it (a new non-NULL value, whether
            -- replacing NULL or a different non-NULL value) is incoherent
            -- while detached: a local-only write can never reach Pinecone's
            -- authoritative copy, so it errors instead of silently diverging
            -- or silently being dropped.
            EXECUTE format(
                'SELECT EXISTS (SELECT 1 FROM new_table JOIN old_table ON old_table.%I = new_table.%I WHERE new_table.%I IS NOT NULL AND new_table.%I::real[] IS DISTINCT FROM old_table.%I::real[])',
                v_id_col, v_id_col, v_vector_col, v_vector_col, v_vector_col
            ) INTO v_vector_written;
            IF v_vector_written THEN
                RAISE EXCEPTION 'pg_pinecone: % is detached (pinecone.tables.storage = ''detached'') — % is remote-only while detached; local writes to it are incoherent — run pinecone.attach_storage(...) first',
                    TG_TABLE_NAME, v_vector_col;
            END IF;

            -- Every surviving vector transition is either unchanged (same
            -- non-NULL value, or still NULL) or non-NULL -> NULL — neither
            -- ever enqueues a 'delete' for a detached table's UPDATE.

            -- An unchanged, still non-NULL vector whose metadata changed:
            -- same shape as the non-detached branch (d) below — the write
            -- guard above already proved the vector itself never took on a
            -- genuinely new value, so this is an ordinary full re-upsert
            -- (the real values are already sitting right here locally).
            EXECUTE format(
                'SELECT COALESCE(jsonb_agg(jsonb_build_object(%L, new_table.%I::text, %L, to_jsonb(new_table.%I::real[])%s)), ''[]''::jsonb) FROM new_table JOIN old_table ON old_table.%I = new_table.%I WHERE new_table.%I IS NOT NULL AND (FALSE%s)',
                'id', v_id_col, 'values', v_vector_col, v_metadata_field,
                v_id_col, v_id_col,
                v_vector_col,
                v_metadata_distinct
            ) INTO v_batch;

            IF jsonb_array_length(v_batch) > 0 THEN
                INSERT INTO pinecone.outbox (relid, op, payload) VALUES (TG_RELID::regclass, 'upsert', v_batch);
            END IF;

            -- Metadata edits on rows whose NEW vector is NULL (the scoped
            -- metadata-divergence fix): these never reach the upsert above
            -- (its "vector IS NOT NULL" guard excludes them), yet Pinecone
            -- still holds that id's vector+metadata from before detach, so a
            -- metadata edit here DOES have a remote target and must not go
            -- stale silently. Keyed on the NEW vector alone —
            -- a row combining vector->NULL with a metadata change in ONE
            -- statement belongs here too: its local NULLing is the expected
            -- silent no-op, but dropping the metadata half of the same write
            -- would silently diverge the remote copy. Split per row by
            -- whether any CHANGED metadata column's new value is NULL (an
            -- unset, `v_metadata_null_transition`): a row with none gets the
            -- cheap merge-only 'update' op; a row with at least one gets
            -- 'reupsert' (fetch_vectors + full re-upsert with freshly
            -- rebuilt, already-null-stripped metadata) instead, since
            -- Pinecone's setMetadata can only ever merge, never unset a key.
            EXECUTE format(
                'SELECT COALESCE(jsonb_agg(jsonb_build_object(%L, new_table.%I::text, %L, jsonb_strip_nulls(jsonb_build_object(%s)))) FILTER (WHERE NOT (FALSE%s)), ''[]''::jsonb), COALESCE(jsonb_agg(jsonb_build_object(%L, new_table.%I::text, %L, jsonb_strip_nulls(jsonb_build_object(%s)))) FILTER (WHERE (FALSE%s)), ''[]''::jsonb) FROM new_table JOIN old_table ON old_table.%I = new_table.%I WHERE new_table.%I IS NULL AND (FALSE%s)',
                'id', v_id_col, 'metadata', array_to_string(v_metadata_pairs, ', '), v_metadata_null_transition,
                'id', v_id_col, 'metadata', array_to_string(v_metadata_pairs, ', '), v_metadata_null_transition,
                v_id_col, v_id_col,
                v_vector_col,
                v_metadata_distinct
            ) INTO v_meta_update_batch, v_meta_reupsert_batch;

            IF jsonb_array_length(v_meta_update_batch) > 0 THEN
                INSERT INTO pinecone.outbox (relid, op, payload) VALUES (TG_RELID::regclass, 'update', v_meta_update_batch);
            END IF;
            IF jsonb_array_length(v_meta_reupsert_batch) > 0 THEN
                INSERT INTO pinecone.outbox (relid, op, payload) VALUES (TG_RELID::regclass, 'reupsert', v_meta_reupsert_batch);
            END IF;

        ELSIF TG_OP = 'UPDATE' THEN
            -- (a) ids whose row no longer exists under that id (PK changed away), plus
            -- (c) ids whose vector went NULL for a still-matching id.
            IF v_part_col IS NULL THEN
                EXECUTE format(
                    'SELECT COALESCE(jsonb_agg(DISTINCT d.id_val), ''[]''::jsonb) FROM (SELECT old_table.%I::text AS id_val FROM old_table WHERE NOT EXISTS (SELECT 1 FROM new_table WHERE new_table.%I = old_table.%I) UNION ALL SELECT new_table.%I::text AS id_val FROM new_table JOIN old_table ON old_table.%I = new_table.%I WHERE new_table.%I IS NULL AND old_table.%I IS NOT NULL) d',
                    v_id_col, v_id_col, v_id_col,
                    v_id_col, v_id_col, v_id_col,
                    v_vector_col, v_vector_col
                ) INTO v_delete_ids;
            ELSE
                -- (a) and (c) as `{id, p}` of the OLD row, plus (d) a vector that moves partition
                EXECUTE format(
                    'SELECT COALESCE(jsonb_agg(DISTINCT jsonb_build_object(%1$L, d.id_val, %2$L, d.p_val)), ''[]''::jsonb) FROM ('
                    || 'SELECT old_table.%3$I::text AS id_val, old_table.%4$I::text AS p_val FROM old_table WHERE NOT EXISTS (SELECT 1 FROM new_table WHERE new_table.%3$I = old_table.%3$I) '
                    || 'UNION ALL SELECT new_table.%3$I::text, old_table.%4$I::text FROM new_table JOIN old_table ON old_table.%3$I = new_table.%3$I WHERE old_table.%5$I IS NOT NULL AND (new_table.%5$I IS NULL OR %6$s)'
                    || ') d WHERE d.p_val IS NOT NULL',
                    'id', 'p', v_id_col, v_part_col, v_vector_col, v_part_moved
                ) INTO v_delete_ids;
            END IF;

            IF jsonb_array_length(v_delete_ids) > 0 THEN
                INSERT INTO pinecone.outbox (relid, op, payload) VALUES (TG_RELID::regclass, 'delete', v_delete_ids);
            END IF;

            -- (b)/(d): everything with a non-NULL new vector that either has no old
            -- counterpart (PK changed) or differs from it (vector or any metadata
            -- column) — i.e. everything NOT covered by the no-op-update guard.
            EXECUTE format(
                'SELECT COALESCE(jsonb_agg(jsonb_build_object(%L, new_table.%I::text, %L, to_jsonb(new_table.%I::real[])%s)), ''[]''::jsonb) FROM new_table LEFT JOIN old_table ON old_table.%I = new_table.%I WHERE new_table.%I IS NOT NULL AND (old_table.%I IS NULL OR new_table.%I::real[] IS DISTINCT FROM old_table.%I::real[]%s)',
                'id', v_id_col, 'values', v_vector_col, v_item_tail,
                v_id_col, v_id_col,
                v_vector_col,
                v_id_col, v_vector_col, v_vector_col,
                v_metadata_distinct
            ) INTO v_batch;

            IF v_part_col IS NOT NULL AND jsonb_array_length(v_batch) > 0 THEN
                PERFORM pinecone._partition_check_batch(v_table, v_part_col, v_namespace, v_batch);
            END IF;

            IF jsonb_array_length(v_batch) > 0 THEN
                INSERT INTO pinecone.outbox (relid, op, payload) VALUES (TG_RELID::regclass, 'upsert', v_batch);
            END IF;

        ELSIF TG_OP = 'DELETE' THEN
            IF v_part_col IS NULL THEN
                EXECUTE format(
                    'SELECT COALESCE(jsonb_agg(old_table.%I::text), ''[]''::jsonb) FROM old_table',
                    v_id_col
                ) INTO v_delete_ids;
            ELSE
                EXECUTE format(
                    'SELECT COALESCE(jsonb_agg(jsonb_build_object(%L, old_table.%I::text, %L, old_table.%I::text)), ''[]''::jsonb) FROM old_table WHERE old_table.%I IS NOT NULL',
                    'id', v_id_col, 'p', v_part_col, v_part_col
                ) INTO v_delete_ids;
            END IF;

            IF jsonb_array_length(v_delete_ids) > 0 THEN
                INSERT INTO pinecone.outbox (relid, op, payload) VALUES (TG_RELID::regclass, 'delete', v_delete_ids);
            END IF;

        ELSIF TG_OP = 'TRUNCATE' THEN
            INSERT INTO pinecone.outbox (relid, op, payload) VALUES (TG_RELID::regclass, 'delete_all', NULL);
        END IF;

        IF v_sync_mode = 'sync' THEN
            PERFORM pinecone._drain_relid(TG_RELID);
        END IF;

        RETURN NULL;
    END;
    $$;
    "#,
    name = "pg_pinecone_sync_trigger_fn",
    requires = [pinecone, "pg_pinecone_tables", _partition_namespace_bytes],
);

// `_push_rows`/`_delete_ids`/`_delete_all` are declared as `#[pg_extern]`s
// inside `pg_pinecone`'s `#[pg_schema]` module in `lib.rs` (delegating straight
// here) so pgrx schema-qualifies them `pinecone._push_rows` etc.; this
// `REVOKE` must run after all three exist, hence `requires` names them
// explicitly rather than just the `pg_pinecone` schema.
extension_sql!(
    r#"
    REVOKE EXECUTE ON FUNCTION pinecone._push_rows(oid, jsonb) FROM PUBLIC;
    REVOKE EXECUTE ON FUNCTION pinecone._delete_ids(oid, text[]) FROM PUBLIC;
    REVOKE EXECUTE ON FUNCTION pinecone._delete_all(oid) FROM PUBLIC;
    -- _sync_trigger itself must not be PUBLIC either: CREATE TRIGGER checks
    -- EXECUTE on the function at creation time, so with the default PUBLIC
    -- grant any role could hand-attach this SECURITY DEFINER function to its
    -- own table over a hand-forged mode='remote' registry row and push (or
    -- TRUNCATE→deleteAll) under the STORED key — bypassing promote()'s
    -- stored-key gate. Installed triggers keep firing for every writer
    -- regardless of this ACL (EXECUTE is not re-checked at fire time), so
    -- writers still need no grants; only trigger *creation* is restricted.
    -- Self-service promote by a non-owner role therefore needs an explicit
    -- GRANT EXECUTE ON FUNCTION pinecone._sync_trigger() — documented in the
    -- README alongside the rest of that grant bundle.
    REVOKE EXECUTE ON FUNCTION pinecone._sync_trigger() FROM PUBLIC;
    "#,
    name = "pg_pinecone_sync_trigger_helpers_revoke",
    requires = [
        pinecone,
        "pg_pinecone_sync_trigger_fn",
        _push_rows,
        _delete_ids,
        _delete_all
    ],
);

extension_sql!(
    r#"
    -- An exact guard, not a documented
    -- limitation: the row-level BEFORE UPDATE half of the detached
    -- id-immutability rule. The statement-level check in _sync_trigger's
    -- detached UPDATE branch compares the transition tables' id SETS
    -- ("does any old id no longer exist in new_table?") — which a
    -- same-statement id PERMUTATION (e.g. UPDATE t SET id = CASE WHEN
    -- id = 1 THEN 2 ELSE 1 END) structurally evades: the id set is
    -- preserved, so no id "disappears", yet every permuted row's identity
    -- has silently swapped under the remote mapping — Pinecone's vector
    -- for old-id-1 now describes the row that calls itself id 2, a
    -- permanent, silent local<->remote mapping corruption that no later
    -- sync can even detect (sync() is blocked while detached). Only a
    -- row-level trigger, seeing each row's own OLD/NEW pair, can catch
    -- it — matching by identity in the statement-level transition-table
    -- join is impossible (Postgres exposes no row correspondence between
    -- old_table and new_table).
    --
    -- Installed/removed per table by pinecone._install_detached_id_guard /
    -- _remove_detached_id_guard (which detach_storage/attach_storage
    -- call); INERT (RETURN NEW) whenever the table's registry row is
    -- missing or storage != 'detached', so a leftover or prematurely
    -- installed trigger can never break ordinary writes. SECURITY DEFINER
    -- for the same reason as _sync_trigger: writer sessions have no
    -- SELECT on pinecone.tables. The id comparison goes through
    -- to_jsonb(...)->>id_column text: every supported id column type has
    -- a canonical, GUC-independent text representation (pinecone.track's
    -- ID_COLUMN_TYPES contract), so text distinctness == value
    -- distinctness.
    CREATE FUNCTION pinecone._detached_id_guard() RETURNS trigger
    LANGUAGE plpgsql
    SECURITY DEFINER
    SET search_path = pg_catalog, pg_temp
    AS $$
    DECLARE
        v_id_col text;
        v_storage text;
    BEGIN
        SELECT id_column, storage INTO v_id_col, v_storage
          FROM pinecone.tables WHERE relid::oid = TG_RELID;
        IF NOT FOUND OR v_storage IS DISTINCT FROM 'detached' THEN
            RETURN NEW;
        END IF;
        IF (to_jsonb(OLD) ->> v_id_col) IS DISTINCT FROM (to_jsonb(NEW) ->> v_id_col) THEN
            RAISE EXCEPTION 'pg_pinecone: % is detached (pinecone.tables.storage = ''detached'') — % is immutable while detached (Pinecone holds the only remaining copy of this table''s vectors, keyed by id; this row-level guard also catches same-statement id permutations the statement-level check cannot) — run pinecone.attach_storage(...) first',
                TG_TABLE_NAME, v_id_col;
        END IF;
        RETURN NEW;
    END;
    $$;
    -- Same trigger-creation ACL as _sync_trigger (see the revoke block
    -- above): only creation is restricted; installed triggers fire for
    -- every writer regardless. Self-service detach by a non-owner role
    -- needs an explicit GRANT EXECUTE, same README grant bundle.
    REVOKE EXECUTE ON FUNCTION pinecone._detached_id_guard() FROM PUBLIC;
    "#,
    name = "pg_pinecone_detached_id_guard_fn",
    requires = [pinecone, "pg_pinecone_tables"],
);

/// Implements `pinecone._install_detached_id_guard`: installs the row-level
/// BEFORE UPDATE id-immutability trigger on `tbl` — see the
/// `pinecone._detached_id_guard` SQL above for what it guards and why the
/// statement-level check can't. Idempotently re-runnable (`CREATE OR REPLACE
/// TRIGGER`): `detach_storage` is specified re-runnable after a
/// mid-flight crash, and this must not be the step that breaks that.
/// `SECURITY INVOKER` — the `CREATE TRIGGER` inside requires the caller to
/// own `tbl` (or hold `TRIGGER` privilege), the natural gate.
pub(crate) fn install_detached_id_guard(tbl: pgrx::PgRelation) {
    let table = pgrx::spi::quote_qualified_identifier(tbl.namespace(), tbl.name());
    Spi::run(&format!(
        "CREATE OR REPLACE TRIGGER pg_pinecone_detached_id_guard BEFORE UPDATE ON {table} \
         FOR EACH ROW EXECUTE FUNCTION pinecone._detached_id_guard()"
    ))
    .unwrap_or_else(|e| error!("pinecone._install_detached_id_guard: {e}"));
}

/// Implements `pinecone._remove_detached_id_guard`: drops the trigger
/// [`install_detached_id_guard`] installs — the `attach_storage` leg.
/// `IF EXISTS`, so it too is idempotently re-runnable.
pub(crate) fn remove_detached_id_guard(tbl: pgrx::PgRelation) {
    let table = pgrx::spi::quote_qualified_identifier(tbl.namespace(), tbl.name());
    Spi::run(&format!(
        "DROP TRIGGER IF EXISTS pg_pinecone_detached_id_guard ON {table}"
    ))
    .unwrap_or_else(|e| error!("pinecone._remove_detached_id_guard: {e}"));
}

/// Implements `pinecone._push_rows`: upserts a pre-filtered batch of `{id,
/// values, metadata?}` objects (built by `_sync_trigger`'s plpgsql body, one
/// per statement) to the sync target for `relid`'s tracked table, chunking at
/// [`PUSH_MAX_ROWS`]/[`UPSERT_MAX_BYTES`]. Skips nothing itself — the plpgsql
/// caller has already filtered out no-op and NULL-vector rows.
pub(crate) fn push_rows(relid: pg_sys::Oid, batch: JsonB) {
    let client = backend::client_for_effective_key();
    try_push_rows(relid, batch, &client, &mut Gate::new(false))
        .unwrap_or_else(|e| backend::raise(e))
}

/// Core of [`push_rows`]: same behavior, but returns the client error instead
/// of raising it (used by `outbox::drain_step`, so a failing outbox row can
/// have its `attempts`/`last_error` recorded before the batch's outcome is
/// decided) and takes `client` as a parameter rather than resolving it
/// itself — `outbox::drain_step` resolves it once per batch, via
/// `backend::client_for_drain`, so the same dispatch code serves both the
/// inline `sync_mode = 'sync'` path and the background worker.
pub(crate) fn try_push_rows(
    relid: pg_sys::Oid,
    batch: JsonB,
    client: &Client,
    gate: &mut Gate,
) -> Result<(), ClientError> {
    let Value::Array(items) = batch.0 else {
        error!("pinecone._push_rows: batch must be a JSON array");
    };
    if items.is_empty() {
        return Ok(());
    }

    let Some(target) = lane_target(relid, "_push_rows") else {
        return Ok(()); // registration gone; the queue was just purged (see remote_target)
    };
    if target.partition.is_some() {
        return push_partitioned(relid, client, &target, items, gate);
    }
    push_items(client, &target.index, &target.namespace, items, &mut false)
}

/// Upserts `items` into one namespace in budgeted requests; `wrote` says whether any request
/// succeeded, also when a later one failed.
fn push_items(
    client: &Client,
    index_name: &str,
    namespace: &str,
    items: Vec<Value>,
    wrote: &mut bool,
) -> Result<(), ClientError> {
    let mut chunk: Vec<Vector> = Vec::new();
    let mut chunk_bytes: usize = 0;
    for item in items {
        let vector = parse_vector(&item);
        let size = vector.record_bytes();
        if !chunk.is_empty()
            && (chunk.len() + 1 > PUSH_MAX_ROWS || chunk_bytes + size > UPSERT_MAX_BYTES)
        {
            flush_push(client, index_name, namespace, &mut chunk, &mut chunk_bytes)?;
            *wrote = true;
        }
        chunk_bytes += size;
        chunk.push(vector);
    }
    flush_push(client, index_name, namespace, &mut chunk, &mut chunk_bytes)?;
    *wrote = true;
    Ok(())
}

/// One namespace at a time, each registered once it holds a vector. The namespaces go in name
/// order, so two sessions that first write the same new values register them in one order and
/// wait for each other's registry rows instead of deadlocking. A failed partition is skipped,
/// and one that Pinecone refuses now is marked failed while the others go on.
fn push_partitioned(
    relid: pg_sys::Oid,
    client: &Client,
    target: &LaneTarget,
    items: Vec<Value>,
    gate: &mut Gate,
) -> Result<(), ClientError> {
    let mut groups = partition_lane::group_upserts(&target.namespace, items)?;
    groups.sort_by(|a, b| a.namespace.cmp(&b.namespace));
    for group in groups {
        if gate.skips(relid, &group.value) {
            continue;
        }
        let mut wrote = false;
        let result = push_items(
            client,
            &target.index,
            &group.namespace,
            group.items,
            &mut wrote,
        );
        partition_lane::settle_write(relid, &group.value, &group.namespace, wrote, result, gate)?;
    }
    Ok(())
}

/// Implements `pinecone._delete_ids`: deletes `ids` from the sync target for
/// `relid`'s tracked table. A no-op (no HTTP call) if `ids` is empty.
pub(crate) fn delete_ids(relid: pg_sys::Oid, ids: Vec<Option<String>>) {
    let client = backend::client_for_effective_key();
    let entries = ids
        .into_iter()
        .enumerate()
        .map(|(i, v)| DeleteEntry {
            id: v.unwrap_or_else(|| {
                panic!("pinecone._delete_ids: ids[{}] is unexpectedly NULL", i + 1)
            }),
            p: None,
        })
        .collect();
    try_delete_ids(relid, entries, &client, &mut Gate::new(false))
        .unwrap_or_else(|e| backend::raise(e))
}

/// Core of [`delete_ids`]; see [`try_push_rows`] for why it returns a
/// `Result` and takes `client` as a parameter instead of resolving it itself.
pub(crate) fn try_delete_ids(
    relid: pg_sys::Oid,
    entries: Vec<DeleteEntry>,
    client: &Client,
    gate: &mut Gate,
) -> Result<(), ClientError> {
    for entry in &entries {
        backend::require_id_len(&entry.id);
    }
    if entries.is_empty() {
        return Ok(());
    }

    let Some(target) = lane_target(relid, "_delete_ids") else {
        return Ok(()); // registration gone; the queue was just purged (see remote_target)
    };
    if target.partition.is_some() {
        for group in partition_lane::group_deletes(&target.namespace, entries) {
            let skipped = group.value.as_deref().is_some_and(|v| gate.skips(relid, v));
            if skipped {
                continue;
            }
            for ids in group.ids.chunks(DELETE_MAX_IDS) {
                let result = client
                    .delete_vectors(&DeleteVectorsRequest {
                        index: target.index.clone(),
                        namespace: Some(group.namespace.clone()),
                        ids: Some(ids.to_vec()),
                        delete_all: false,
                        filter: None,
                    })
                    .map_err(|e| partition_lane::name_namespace(e, &group.namespace));
                let Err(err) = result else { continue };
                match &group.value {
                    Some(value) => gate.refused(relid, value, &group.namespace, err)?,
                    None => return Err(err),
                }
                break;
            }
        }
        return Ok(());
    }
    client.delete_vectors(&DeleteVectorsRequest {
        index: target.index,
        namespace: Some(target.namespace),
        ids: Some(entries.into_iter().map(|e| e.id).collect()),
        delete_all: false,
        filter: None,
    })
}

/// Implements `pinecone._delete_all`: deletes every vector in the sync
/// target's namespace for `relid`'s tracked table (the `TRUNCATE` case).
pub(crate) fn delete_all(relid: pg_sys::Oid) {
    let client = backend::client_for_effective_key();
    try_delete_all(relid, &client, &mut Gate::new(false)).unwrap_or_else(|e| backend::raise(e))
}

/// Core of [`delete_all`]; see [`try_push_rows`] for why it returns a
/// `Result` and takes `client` as a parameter instead of resolving it itself.
pub(crate) fn try_delete_all(
    relid: pg_sys::Oid,
    client: &Client,
    gate: &mut Gate,
) -> Result<(), ClientError> {
    let Some(target) = lane_target(relid, "_delete_all") else {
        return Ok(()); // registration gone; the queue was just purged (see remote_target)
    };
    if let Some(spec) = &target.partition {
        // The registry only lists what a drain registered. The family may hold more (the leftovers
        // of a dropped table, a write that rolled back after Pinecone took it): adopt those first,
        // so the wipe empties them too. A listing that fails is a retry, before anything is wiped.
        partition_ops::try_adopt_orphans(
            "pinecone._delete_all",
            relid,
            spec,
            &target.index,
            &target.namespace,
            client,
        )?;
        partition_registry::wipe_partitions(relid, Lane::Vector, client, &target.index)?;
        partition_registry::forget_partitions(relid, Lane::Vector);
        gate.wiped();
        return Ok(());
    }
    client.delete_vectors(&DeleteVectorsRequest {
        index: target.index,
        namespace: Some(target.namespace),
        ids: None,
        delete_all: true,
        filter: None,
    })
}

/// Dispatches an `'update'` outbox row (the scoped
/// metadata-divergence fix): a batch of `{id, metadata}` objects built by the
/// detached branch of `_sync_trigger`'s plpgsql body, one merge-only
/// `setMetadata` PATCH per id (Pinecone's `/vectors/update` has no bulk
/// form). Only ever built for rows whose vector stayed `NULL` and whose
/// changed metadata fields all kept a non-`NULL` new value — see the
/// trigger's own rustdoc for why a `NULL`-valued field can never reach this
/// op (merge can't unset a key; that's `try_reupsert_metadata`'s job
/// instead).
///
/// **Ghost pre-check (a `detach_storage` gate precondition, live-verified
/// 2026-09-13 — `pg_pinecone-client/tests/update_missing_id_probe.rs`):**
/// `/vectors/update` on an id Pinecone has no vector for returns a plain
/// `200` no-op — it does NOT create a phantom vector, and NEVER a `404`.
/// That 200 carries no signal that the id didn't exist, and unlike
/// [`try_reupsert_metadata`] (which already calls `fetch_vectors` to get real
/// values to re-upsert, so a ghost falls straight out of that same call)
/// this op needs no values at all — nothing would otherwise ever notice a
/// ghost here, silently burying exactly the loss signal this whole feature
/// exists to surface. So this batch-fetches every id's existence FIRST (one
/// `fetch_vectors` call, values discarded) and only issues the `setMetadata`
/// PATCH for ids Pinecone actually still has; a ghost is skipped and WARNed,
/// same wording/shape as `try_reupsert_metadata`'s own ghost handling —
/// never parks the queue.
pub(crate) fn try_update_metadata(
    relid: pg_sys::Oid,
    batch: JsonB,
    client: &Client,
) -> Result<(), ClientError> {
    let Value::Array(items) = batch.0 else {
        error!("pg_pinecone: 'update' outbox row batch must be a JSON array");
    };
    if items.is_empty() {
        return Ok(());
    }
    let Some((index_name, namespace)) = remote_target(relid, "_update_metadata") else {
        return Ok(()); // registration gone; the queue was just purged (see remote_target)
    };

    let mut metadata_by_id: HashMap<String, Value> = HashMap::with_capacity(items.len());
    let mut ids: Vec<String> = Vec::with_capacity(items.len());
    for item in items {
        let id = metadata_item_id(&item, "'update'");
        let metadata = item.get("metadata").cloned().unwrap_or_default();
        metadata_by_id.insert(id.clone(), metadata);
        ids.push(id);
    }

    let fetched = client.fetch_vectors(&FetchVectorsRequest {
        index: index_name.clone(),
        namespace: Some(namespace.clone()),
        ids: ids.clone(),
    })?;

    let mut ghosts: Vec<String> = Vec::new();
    for id in ids {
        if !fetched.contains_key(&id) {
            ghosts.push(id);
            continue;
        }
        let metadata = metadata_by_id.remove(&id).unwrap_or_default();
        client.update_vector_metadata(&UpdateVectorMetadataRequest {
            index: index_name.clone(),
            namespace: Some(namespace.clone()),
            id,
            metadata,
        })?;
    }
    if !ghosts.is_empty() {
        let listed: Vec<String> = ghosts.iter().map(|id| format!("'{id}'")).collect();
        warning!(
            "pg_pinecone: update: {} id(s) in this batch have no remote vector to set metadata on — \
             on a detached table the remote copy is the ONLY copy, so a ghost here means that \
             vector is already gone (see pinecone.recover_from_remote's ghost accounting): {}",
            ghosts.len(),
            listed.join(", ")
        );
    }
    Ok(())
}

/// Dispatches a `'reupsert'` outbox row (the scoped
/// metadata-divergence fix, the `->NULL` leg): a batch of `{id, metadata}`
/// objects — the row's freshly rebuilt, already-`jsonb_strip_nulls`'d
/// complete metadata object — for rows whose vector stayed `NULL` but at
/// least one changed metadata field is being UNSET (a `NULL` new value,
/// which `try_update_metadata`'s merge-only PATCH can never express). Fetches
/// each id's real values from Pinecone (`fetch_vectors` — the local column
/// holds nothing, by definition, while detached) and re-upserts the full
/// vector with the given metadata REPLACING (not merging with) whatever
/// Pinecone had, chunked at [`PUSH_MAX_ROWS`]/[`UPSERT_MAX_BYTES`] like
/// `try_push_rows`. An id Pinecone has no vector for at all (a ghost —
/// the remote copy was independently deleted, e.g. via `TRUNCATE`/`DELETE`
/// racing this same drain round) is skipped, but WARNed with the ids: on a
/// detached table a ghost here is exactly the loss signal this whole feature
/// exists to surface — the remote copy is the ONLY copy,
/// and it's gone — so silence would bury it; `recover_from_remote` counts
/// its ghosts for the same reason.
pub(crate) fn try_reupsert_metadata(
    relid: pg_sys::Oid,
    batch: JsonB,
    client: &Client,
) -> Result<(), ClientError> {
    let Value::Array(items) = batch.0 else {
        error!("pg_pinecone: 'reupsert' outbox row batch must be a JSON array");
    };
    if items.is_empty() {
        return Ok(());
    }
    let Some((index_name, namespace)) = remote_target(relid, "_reupsert_metadata") else {
        return Ok(()); // registration gone; the queue was just purged (see remote_target)
    };

    let mut metadata_by_id: HashMap<String, Value> = HashMap::with_capacity(items.len());
    let mut ids: Vec<String> = Vec::with_capacity(items.len());
    for item in items {
        let id = metadata_item_id(&item, "'reupsert'");
        let metadata = item.get("metadata").cloned().unwrap_or_default();
        metadata_by_id.insert(id.clone(), metadata);
        ids.push(id);
    }

    let fetched = client.fetch_vectors(&FetchVectorsRequest {
        index: index_name.clone(),
        namespace: Some(namespace.clone()),
        ids: ids.clone(),
    })?;

    // Chunked at the same [`PUSH_MAX_ROWS`]/[`UPSERT_MAX_BYTES`] budget as
    // `try_push_rows`: one statement clearing metadata
    // across a few hundred NULL-vector detached rows builds ONE 'reupsert'
    // row whose re-upsert would otherwise be a single oversized request — a
    // permanent 400 that parks the queue unrecoverably (retrying can never
    // shrink the request).
    let mut chunk: Vec<Vector> = Vec::new();
    let mut chunk_bytes: usize = 0;
    let mut ghosts: Vec<String> = Vec::new();
    for id in ids {
        let Some(v) = fetched.get(&id) else {
            ghosts.push(id); // nothing to re-upsert a value onto — see rustdoc.
            continue;
        };
        // `.get().cloned()`, NOT `.remove()`.
        // If a dropped UNIQUE index let two local rows share one id, this
        // batch carries that id twice; `.remove()` handed the SECOND
        // occurrence `None`, silently clearing that id's remote metadata to
        // nothing. Cloning gives every occurrence the same metadata (a remote
        // last-write-wins), never a spurious `metadata: None` wipe.
        let metadata = metadata_by_id.get(&id).cloned();
        let vector = Vector {
            id,
            values: v.values.clone(),
            metadata,
        };
        let size = vector.record_bytes();
        if !chunk.is_empty()
            && (chunk.len() + 1 > PUSH_MAX_ROWS || chunk_bytes + size > UPSERT_MAX_BYTES)
        {
            flush_push(
                client,
                &index_name,
                &namespace,
                &mut chunk,
                &mut chunk_bytes,
            )?;
        }
        chunk_bytes += size;
        chunk.push(vector);
    }
    if !ghosts.is_empty() {
        let listed: Vec<String> = ghosts.iter().map(|id| format!("'{id}'")).collect();
        warning!(
            "pg_pinecone: reupsert: {} id(s) in this batch have no remote vector to re-upsert their \
             metadata onto — on a detached table the remote copy is the ONLY copy, so a ghost \
             here means that vector is already gone (see pinecone.recover_from_remote's ghost \
             accounting): {}",
            ghosts.len(),
            listed.join(", ")
        );
    }
    flush_push(
        client,
        &index_name,
        &namespace,
        &mut chunk,
        &mut chunk_bytes,
    )
}

/// Extracts and length-validates the `"id"` string of one `{id, metadata}`
/// batch item, as built by `_sync_trigger`'s detached-branch plpgsql (shared
/// by [`try_update_metadata`]/[`try_reupsert_metadata`]); `op_label` names
/// the outbox op in the panic message, matching [`parse_vector`]'s own
/// "batch item missing ..." wording for the `'upsert'` path.
fn metadata_item_id(item: &Value, op_label: &str) -> String {
    let id = item
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_else(|| {
            panic!("pg_pinecone: {op_label} batch item missing string \"id\": {item}")
        })
        .to_string();
    backend::require_id_len(&id);
    id
}

/// [`lane_target`] without the partition column, for the ops that never carry one.
fn remote_target(relid: pg_sys::Oid, func: &str) -> Option<(String, String)> {
    lane_target(relid, func).map(|target| (target.index, target.namespace))
}

/// Where the sync of a table writes: its index, its namespace (the prefix of the family when the
/// table is partitioned) and the partition column, if any.
pub(crate) struct LaneTarget {
    pub(crate) index: String,
    pub(crate) namespace: String,
    pub(crate) partition: Option<PartitionSpec>,
}

/// Looks up `relid`'s [`LaneTarget`].
///
/// A MISSING registry row returns `None` after warning and purging `relid`'s
/// queued outbox rows — it is reachable, not
/// a bug: a row can be enqueued while registered and only dispatched after
/// something removed the registration out from under the queue (the
/// `sql_drop` trigger on a concurrent `DROP INDEX`, a demote completion
/// racing an in-flight write's commit). Panicking here turned such a row
/// into poison that aborted the worker's round — and, retried every round,
/// abort-looped the worker forever; the queued write is moot without a
/// registration (there is no target to push to), so warn-and-purge is the
/// only outcome that ever converges.
///
/// A registered row in a non-`'remote'`/`'promoting'` mode still panics
/// (a genuine invariant: `pinecone.demote` clears the queue in the same
/// transaction as its mode flip, so a `'local'`-mode row with queued writes
/// means a bug in that transactional pairing, not an operational race).
/// `'promoting'` is a valid target: an async promote's
/// triggers can enqueue real writes before its backfill exhausts.
fn lane_target(relid: pg_sys::Oid, func: &str) -> Option<LaneTarget> {
    let Some(row) = registry::lookup(relid) else {
        let purged = crate::outbox::clear_relid(relid);
        warning!(
            "pinecone.{func}: relid {relid} has no pinecone.tables registration anymore (dropped or \
             demoted since these writes were queued) — purged {purged} moot queued row(s) \
             instead of retrying them forever"
        );
        return None;
    };
    if row.mode != "remote" && row.mode != "promoting" {
        panic!(
            "pinecone.{func}: relid {relid} is not remote or promoting (mode = '{}') — should be \
             unreachable",
            row.mode
        );
    }
    Some(LaneTarget {
        index: row
            .index_name
            .expect("pg_pinecone: remote row always has index_name"),
        namespace: row
            .namespace
            .expect("pg_pinecone: remote row always has namespace"),
        partition: row.partition,
    })
}

/// Parses one `{id, values, metadata?}` batch item (as built by
/// `_sync_trigger`'s plpgsql body) into a [`Vector`].
fn parse_vector(item: &Value) -> Vector {
    let id = item
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("pinecone._push_rows: batch item missing string \"id\": {item}"))
        .to_string();
    backend::require_id_len(&id);
    let values = item
        .get("values")
        .and_then(Value::as_array)
        .unwrap_or_else(|| {
            panic!("pinecone._push_rows: batch item missing array \"values\": {item}")
        })
        .iter()
        .map(|v| {
            v.as_f64().unwrap_or_else(|| {
                panic!("pinecone._push_rows: non-numeric element in \"values\": {item}")
            }) as f32
        })
        .collect();
    let metadata = item.get("metadata").cloned();
    Vector {
        id,
        values,
        metadata,
    }
}

fn flush_push(
    client: &pg_pinecone_client::Client,
    index_name: &str,
    namespace: &str,
    chunk: &mut Vec<Vector>,
    chunk_bytes: &mut usize,
) -> Result<(), ClientError> {
    if chunk.is_empty() {
        return Ok(());
    }
    let vectors = std::mem::take(chunk);
    *chunk_bytes = 0;
    client.upsert_vectors(&UpsertVectorsRequest {
        index: index_name.to_string(),
        namespace: Some(namespace.to_string()),
        vectors,
    })?;
    Ok(())
}
