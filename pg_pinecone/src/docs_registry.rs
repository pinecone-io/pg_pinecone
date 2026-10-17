//! `pinecone.doc_tables`: the registry of tables declared with `track_docs`, the document
//! lane's counterpart of `pinecone.tables`. Dumped with the extension (configuration, unlike
//! the queue in `docs_outbox.rs`).

use pgrx::{JsonB, pg_sys, prelude::*, spi::Spi};
use serde_json::Value;

use crate::{
    partition::{Claim, PartitionSpec},
    partition_registry::{self, Owner},
};

extension_sql!(
    r#"
    CREATE TABLE pinecone.doc_tables (
        relid regclass PRIMARY KEY,
        id_column text NOT NULL,
        id_type text NOT NULL,
        -- Resolved per-column options (`docs_schema::text_columns_json`).
        text_columns jsonb NOT NULL,
        metadata_columns text[] NOT NULL DEFAULT '{}',
        vector_column text,
        vector_dimension int,
        vector_metric text,
        index_name text NOT NULL,
        namespace text NOT NULL,
        -- The Pinecone schema derived from the declaration, kept to detect drift.
        schema jsonb NOT NULL,
        -- Every declared column (id, text, metadata, vector) -> `typname:atttypmod` at track
        -- time; the DDL guard refuses ALTER TABLE changes to any of them.
        column_types jsonb NOT NULL DEFAULT '{}',
        attached boolean NOT NULL DEFAULT false,
        state text NOT NULL CHECK (state IN ('backfilling', 'ready')),
        backfill_cursor text,
        backfill_total bigint,
        backfill_done bigint NOT NULL DEFAULT 0,
        queue_parked boolean NOT NULL DEFAULT false,
        parked_reason text,
        -- Consecutive permanent failures of the lane (any row, any step): it parks at
        -- pinecone.outbox_max_attempts, and a step that gets through starts the count over.
        lane_failures int NOT NULL DEFAULT 0,
        last_delivered_at timestamptz,
        -- Each value of the column gets its own namespace, `<namespace>:<value>`; `namespace`
        -- then holds the prefix. NULL (both columns) = one namespace.
        partition_column text,
        partition_kind text CHECK (partition_kind IN ('int2', 'int4', 'int8', 'text', 'uuid')),
        created_at timestamptz NOT NULL DEFAULT now(),
        UNIQUE (index_name, namespace),
        CONSTRAINT doc_tables_partition_pair_check
            CHECK ((partition_column IS NULL) = (partition_kind IS NULL))
    );
    REVOKE ALL ON pinecone.doc_tables FROM PUBLIC;
    SELECT pg_catalog.pg_extension_config_dump('pinecone.doc_tables', '');
    "#,
    name = "pg_pinecone_doc_tables",
    requires = [pinecone],
);

/// The role that owns `pinecone.doc_tables` and `pinecone.doc_outbox` (whoever created the
/// extension). Read from the catalog caches: a query would resolve its operators through the
/// caller's `search_path`, before the switch.
fn tables_owner() -> pg_sys::Oid {
    unsafe {
        let schema = pg_sys::get_namespace_oid(c"pinecone".as_ptr(), false);
        let relid = pg_sys::get_relname_relid(c"doc_tables".as_ptr(), schema);
        let tuple = pg_sys::SearchSysCache1(
            pg_sys::SysCacheIdentifier::RELOID as i32,
            relid.into_datum().expect("pg_sys::Oid is never SQL NULL"),
        );
        if tuple.is_null() {
            error!("pg_pinecone: the document registry pinecone.doc_tables is missing");
        }
        let owner = (*pg_sys::heap_tuple_get_struct::<pg_sys::FormData_pg_class>(tuple)).relowner;
        pg_sys::ReleaseSysCache(tuple);
        owner
    }
}

/// Puts identity and settings back when the scope of [`as_owner`] ends, an ERROR unwinding it
/// included. Settings first, as `fmgr_security_definer` does.
struct RestoreIdentity {
    user: pg_sys::Oid,
    sec_context: ::core::ffi::c_int,
    guc_nest_level: ::core::ffi::c_int,
}

impl Drop for RestoreIdentity {
    fn drop(&mut self) {
        unsafe {
            pg_sys::AtEOXact_GUC(false, self.guc_nest_level);
            pg_sys::SetUserIdAndSecContext(self.user, self.sec_context);
        }
    }
}

/// Runs `f` as the owner of the registry and the queue, the only way the verbs reach them: no
/// role holds a grant on either table. The verbs stay `SECURITY INVOKER` and check the caller's
/// right first; whatever must run as the caller (Pinecone calls, the key lookup, the local search,
/// the count of the caller's own table) stays outside `f`.
///
/// `search_path` is pinned to `pg_catalog, pg_temp` inside, for the owner too: names resolved
/// through the caller's path would run the caller's operators, casts and aggregates with the
/// owner's rights. The scope is a security-restricted operation as well.
pub(crate) fn as_owner<T>(f: impl FnOnce() -> T) -> T {
    let owner = tables_owner();
    let (mut user, mut sec_context) = (pg_sys::InvalidOid, 0);
    unsafe { pg_sys::GetUserIdAndSecContext(&mut user, &mut sec_context) };
    let switched = sec_context
        | pg_sys::SECURITY_LOCAL_USERID_CHANGE as ::core::ffi::c_int
        | pg_sys::SECURITY_RESTRICTED_OPERATION as ::core::ffi::c_int;
    let _restore = unsafe {
        pg_sys::SetUserIdAndSecContext(owner, switched);
        RestoreIdentity {
            user,
            sec_context,
            guc_nest_level: pg_sys::NewGUCNestLevel(),
        }
    };
    unsafe { pg_sys::RestrictSearchPath() };
    f()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DocState {
    Backfilling,
    Ready,
}

impl DocState {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            DocState::Backfilling => "backfilling",
            DocState::Ready => "ready",
        }
    }

    fn parse(state: &str) -> Self {
        match state {
            "backfilling" => DocState::Backfilling,
            "ready" => DocState::Ready,
            other => panic!("pg_pinecone: doc_tables row has unknown state '{other}'"),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DocRegistryRow {
    pub(crate) relid: pg_sys::Oid,
    pub(crate) id_column: String,
    pub(crate) id_type: String,
    pub(crate) text_columns: Value,
    pub(crate) metadata_columns: Vec<String>,
    pub(crate) vector_column: Option<String>,
    pub(crate) vector_dimension: Option<i32>,
    pub(crate) vector_metric: Option<String>,
    pub(crate) index_name: String,
    pub(crate) namespace: String,
    pub(crate) schema: Value,
    pub(crate) attached: bool,
    pub(crate) state: DocState,
    pub(crate) backfill_cursor: Option<String>,
    pub(crate) backfill_total: Option<i64>,
    pub(crate) backfill_done: i64,
    pub(crate) queue_parked: bool,
    pub(crate) parked_reason: Option<String>,
    #[allow(dead_code)] // read by tests and the status columns (a later change)
    pub(crate) last_delivered_at: Option<TimestampWithTimeZone>,
    pub(crate) partition: Option<PartitionSpec>,
}

const SELECT: &str = "SELECT relid::oid AS relid, id_column, id_type, text_columns, \
     metadata_columns, vector_column, vector_dimension, vector_metric, index_name, namespace, \
     schema, attached, state, backfill_cursor, backfill_total, backfill_done, queue_parked, \
     parked_reason, last_delivered_at, partition_column, partition_kind FROM pinecone.doc_tables";

pub(crate) fn lookup(relid: pg_sys::Oid) -> Option<DocRegistryRow> {
    Spi::connect(|client| {
        client
            .select(
                &format!("{SELECT} WHERE relid::oid = $1::oid"),
                None,
                &[relid.into()],
            )
            .expect("pg_pinecone: doc registry lookup query failed")
            .next()
            .map(|row| row_from_tuple(&row))
    })
}

pub(crate) fn list() -> Vec<DocRegistryRow> {
    Spi::connect(|client| {
        client
            .select(&format!("{SELECT} ORDER BY relid::oid"), None, &[])
            .expect("pg_pinecone: doc registry list query failed")
            .map(|row| row_from_tuple(&row))
            .collect()
    })
}

/// The registry rows `status()` may show the caller: the tables they can `SELECT`, read as the
/// registry's owner (nobody holds a grant on the registry). A row whose table is gone is shown to
/// a superuser only, since no one else can have a privilege on it.
pub(crate) fn list_visible() -> Vec<DocRegistryRow> {
    as_owner(list)
        .into_iter()
        .filter(|row| can_see(row.relid))
        .collect()
}

pub(crate) fn can_see(relid: pg_sys::Oid) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT CASE WHEN EXISTS (SELECT 1 FROM pg_catalog.pg_class WHERE oid = $1::oid) \
                     THEN pg_catalog.has_table_privilege(current_user, $1::oid, 'SELECT') \
                     ELSE pg_catalog.current_setting('is_superuser') = 'on' END",
        &[relid.into()],
    )
    .expect("pg_pinecone: doc registry visibility query failed")
    .expect("pg_pinecone: doc registry visibility unexpectedly NULL")
}

/// Inserts a row; the parked and delivery columns start at their defaults.
pub(crate) fn insert(row: &DocRegistryRow) {
    Spi::run_with_args(
        "INSERT INTO pinecone.doc_tables \
         (relid, id_column, id_type, text_columns, metadata_columns, vector_column, \
          vector_dimension, vector_metric, index_name, namespace, schema, attached, state, \
          backfill_cursor, backfill_total, backfill_done, partition_column, partition_kind) \
         VALUES ($1::oid::regclass, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, \
                 $15, $16, $17, $18)",
        &[
            row.relid.into(),
            row.id_column.clone().into(),
            row.id_type.clone().into(),
            JsonB(row.text_columns.clone()).into(),
            row.metadata_columns.clone().into(),
            row.vector_column.clone().into(),
            row.vector_dimension.into(),
            row.vector_metric.clone().into(),
            row.index_name.clone().into(),
            row.namespace.clone().into(),
            JsonB(row.schema.clone()).into(),
            row.attached.into(),
            row.state.as_str().into(),
            row.backfill_cursor.clone().into(),
            row.backfill_total.into(),
            row.backfill_done.into(),
            row.partition.as_ref().map(|p| p.column.clone()).into(),
            row.partition.as_ref().map(|p| p.kind.as_str()).into(),
        ],
    )
    .expect("pg_pinecone: doc registry insert failed");
}

/// Records the declared columns' type signatures (see the `column_types` column).
pub(crate) fn set_column_types(relid: pg_sys::Oid, column_types: &Value) {
    Spi::run_with_args(
        "UPDATE pinecone.doc_tables SET column_types = $2 WHERE relid::oid = $1::oid",
        &[relid.into(), JsonB(column_types.clone()).into()],
    )
    .expect("pg_pinecone: doc registry set_column_types failed");
}

pub(crate) fn delete(relid: pg_sys::Oid) {
    Spi::run_with_args(
        "DELETE FROM pinecone.doc_tables WHERE relid::oid = $1::oid",
        &[relid.into()],
    )
    .expect("pg_pinecone: doc registry delete failed");
}

pub(crate) fn set_state(relid: pg_sys::Oid, state: DocState) {
    Spi::run_with_args(
        "UPDATE pinecone.doc_tables SET state = $2 WHERE relid::oid = $1::oid",
        &[relid.into(), state.as_str().into()],
    )
    .expect("pg_pinecone: doc registry set_state failed");
}

/// Overwrites the backfill cursor (the last primary key sent, `None` = before the first row),
/// the number of rows done and the total, if known.
pub(crate) fn set_backfill_progress(
    relid: pg_sys::Oid,
    cursor: Option<&str>,
    done: i64,
    total: Option<i64>,
) {
    Spi::run_with_args(
        "UPDATE pinecone.doc_tables SET backfill_cursor = $2, backfill_done = $3, \
         backfill_total = $4 WHERE relid::oid = $1::oid",
        &[relid.into(), cursor.into(), done.into(), total.into()],
    )
    .expect("pg_pinecone: doc registry set_backfill_progress failed");
}

/// Parks the table's document queue: the scheduler skips it until [`unpark`].
pub(crate) fn set_parked(relid: pg_sys::Oid, reason: &str) {
    Spi::run_with_args(
        "UPDATE pinecone.doc_tables SET queue_parked = true, parked_reason = $2 \
         WHERE relid::oid = $1::oid",
        &[relid.into(), reason.into()],
    )
    .expect("pg_pinecone: doc registry set_parked failed");
}

/// Counts one more permanent failure of the lane and returns the count so far (0 if the row is
/// gone), for the parking decision.
pub(crate) fn count_lane_failure(relid: pg_sys::Oid) -> i32 {
    Spi::get_one_with_args::<i32>(
        "WITH bumped AS (UPDATE pinecone.doc_tables SET lane_failures = lane_failures + 1 \
             WHERE relid::oid = $1::oid RETURNING lane_failures) \
         SELECT coalesce((SELECT lane_failures FROM bumped), 0)",
        &[relid.into()],
    )
    .expect("pg_pinecone: doc registry count_lane_failure failed")
    .expect("pg_pinecone: doc registry lane failure count unexpectedly NULL")
}

/// A step got through: the lane's run of failures ends.
pub(crate) fn reset_lane_failures(relid: pg_sys::Oid) {
    Spi::run_with_args(
        "UPDATE pinecone.doc_tables SET lane_failures = 0 \
         WHERE relid::oid = $1::oid AND lane_failures <> 0",
        &[relid.into()],
    )
    .expect("pg_pinecone: doc registry reset_lane_failures failed");
}

pub(crate) fn unpark(relid: pg_sys::Oid) {
    Spi::run_with_args(
        "UPDATE pinecone.doc_tables SET queue_parked = false, parked_reason = NULL, \
         lane_failures = 0 WHERE relid::oid = $1::oid",
        &[relid.into()],
    )
    .expect("pg_pinecone: doc registry unpark failed");
}

pub(crate) fn touch_delivered(relid: pg_sys::Oid) {
    Spi::run_with_args(
        "UPDATE pinecone.doc_tables SET last_delivered_at = pg_catalog.clock_timestamp() \
         WHERE relid::oid = $1::oid",
        &[relid.into()],
    )
    .expect("pg_pinecone: doc registry touch_delivered failed");
}

/// Whether the caller may read `pinecone.doc_tables`. The vector verbs' guards consult the
/// document registry only if so: a vector-only role holds no grant on it, and Pinecone
/// rejects a clashing index anyway.
pub(crate) fn readable() -> bool {
    Spi::get_one::<bool>(
        "SELECT pg_catalog.has_table_privilege(current_user, 'pinecone.doc_tables', 'SELECT')",
    )
    .expect("pg_pinecone: has_table_privilege query failed")
    .expect("pg_pinecone: has_table_privilege unexpectedly NULL")
}

/// The table holding the exact namespace `namespace` in `index_name`: see [`claim_owner`].
#[allow(
    dead_code,
    reason = "shorthand for an exact claim; the verbs call claim_owner"
)]
pub(crate) fn index_in_use(index_name: &str, namespace: &str) -> Option<pg_sys::Oid> {
    claim_owner(index_name, Claim::Exact(namespace)).map(|owner| owner.relid)
}

/// The table whose claim in `index_name` clashes with `claim`: the same namespace, or a
/// partition family covering it or covered by it.
pub(crate) fn claim_owner(index_name: &str, claim: Claim<'_>) -> Option<Owner> {
    Spi::connect(|client| {
        client
            .select(
                &format!(
                    "SELECT relid::oid AS relid, namespace, partition_column IS NOT NULL AS partitioned \
                     FROM pinecone.doc_tables WHERE index_name = $1 AND {} \
                     ORDER BY relid::oid LIMIT 1",
                    partition_registry::overlap(claim)
                ),
                None,
                &[index_name.into(), claim.namespace().into()],
            )
            .expect("pg_pinecone: doc registry claim owner query failed")
            .next()
            .map(|row| Owner {
                relid: get(&row, "relid"),
                namespace: get(&row, "namespace"),
                partitioned: get(&row, "partitioned"),
            })
    })
}

/// Whether another tracked table (any namespace) uses document index `index_name`.
pub(crate) fn other_row_uses_index(index_name: &str, relid: pg_sys::Oid) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT EXISTS (SELECT 1 FROM pinecone.doc_tables \
         WHERE index_name = $1 AND relid::oid <> $2::oid)",
        &[index_name.into(), relid.into()],
    )
    .expect("pg_pinecone: doc registry shared-index query failed")
    .expect("pg_pinecone: EXISTS unexpectedly NULL")
}

/// The owner of a document index named `index_name`, whatever its namespace.
pub(crate) fn index_owner(index_name: &str) -> Option<pg_sys::Oid> {
    Spi::get_one_with_args::<pg_sys::Oid>(
        "SELECT (SELECT relid::oid FROM pinecone.doc_tables WHERE index_name = $1 \
                 ORDER BY relid::oid LIMIT 1)",
        &[index_name.into()],
    )
    .expect("pg_pinecone: doc registry index owner query failed")
}

fn get<T: FromDatum + IntoDatum>(row: &pgrx::spi::SpiHeapTupleData<'_>, name: &str) -> T {
    row.get_by_name::<T, _>(name)
        .unwrap_or_else(|e| panic!("pg_pinecone: doc registry row {name} query failed: {e}"))
        .unwrap_or_else(|| panic!("pg_pinecone: doc registry row {name} unexpectedly NULL"))
}

fn get_opt<T: FromDatum + IntoDatum>(
    row: &pgrx::spi::SpiHeapTupleData<'_>,
    name: &str,
) -> Option<T> {
    row.get_by_name::<T, _>(name)
        .unwrap_or_else(|e| panic!("pg_pinecone: doc registry row {name} query failed: {e}"))
}

fn row_from_tuple(row: &pgrx::spi::SpiHeapTupleData<'_>) -> DocRegistryRow {
    DocRegistryRow {
        relid: get(row, "relid"),
        id_column: get(row, "id_column"),
        id_type: get(row, "id_type"),
        text_columns: get::<JsonB>(row, "text_columns").0,
        metadata_columns: get(row, "metadata_columns"),
        vector_column: get_opt(row, "vector_column"),
        vector_dimension: get_opt(row, "vector_dimension"),
        vector_metric: get_opt(row, "vector_metric"),
        index_name: get(row, "index_name"),
        namespace: get(row, "namespace"),
        schema: get::<JsonB>(row, "schema").0,
        attached: get(row, "attached"),
        state: DocState::parse(&get::<String>(row, "state")),
        backfill_cursor: get_opt(row, "backfill_cursor"),
        backfill_total: get_opt(row, "backfill_total"),
        backfill_done: get(row, "backfill_done"),
        queue_parked: get(row, "queue_parked"),
        parked_reason: get_opt(row, "parked_reason"),
        last_delivered_at: get_opt(row, "last_delivered_at"),
        partition: PartitionSpec::from_columns(
            get_opt(row, "partition_column"),
            get_opt(row, "partition_kind"),
        )
        .unwrap_or_else(|e| panic!("pg_pinecone: doc registry row {e}")),
    }
}
