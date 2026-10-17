//! `pinecone.track_docs` and `pinecone.untrack_docs`: declare a table's text, metadata and vector
//! columns for full-text search, create (or attach) the Pinecone document index, register the
//! table and arm the sync triggers. Everything that can be checked locally is checked before
//! the first HTTP call, so a refused declaration costs nothing and leaves nothing behind.

use pg_pinecone_client::{CreateDocIndexRequest, Error as ClientError};
use pgrx::{JsonB, PgRelation, PgSqlErrorCode as Code, pg_sys, prelude::*, spi::Spi};
use serde_json::{Map, Value};

use crate::{
    backend, catalog, docs, docs_backfill, docs_drain, docs_outbox,
    docs_registry::{self, DocRegistryRow, DocState},
    docs_schema::{self, DocsError, TextColumn, VectorField},
    docs_sync, outbox,
    partition::{Claim, Lane, PartitionKind, PartitionSpec},
    partition_lane, partition_ops, partition_registry, promote,
};

const TRACK: &str = "pinecone.track_docs";
/// Where `track_docs` creates indexes; `create_doc_index` has the same defaults.
const CLOUD: &str = "aws";
const REGION: &str = "us-east-1";
const MAX_INDEX_NAME_LEN: usize = 45;

fn fail(code: Code, message: impl Into<String>) -> ! {
    DocsError::new(code, message).raise()
}

fn fail_hint(code: Code, message: impl Into<String>, hint: impl Into<String>) -> ! {
    DocsError::new(code, message).hint(hint).raise()
}

/// Pinecone's index name rule: 1 to 45 lowercase letters, digits and hyphens, starting and
/// ending with a letter or digit.
pub(crate) fn valid_index_name(name: &str) -> bool {
    let edge = |c: Option<char>| c.is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit());
    name.len() <= MAX_INDEX_NAME_LEN
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && edge(name.chars().next())
        && edge(name.chars().next_back())
}

/// Pinecone's namespace rule: printable ASCII only, so a table named `tablé` cannot have one.
pub(crate) fn valid_namespace(namespace: &str) -> bool {
    !namespace.is_empty() && namespace.bytes().all(|b| (0x20..=0x7e).contains(&b))
}

/// `pd-<schema>-<table>`, sanitised like the vector default (`pv-...`) but a different name, so
/// a table in both registries gets two indexes.
pub(crate) fn default_index_name(schema: &str, table: &str) -> String {
    promote::sanitize_index_name(&format!("pd-{schema}-{table}"))
}

/// The dense field's metric from the opclasses of the indexes on the vector column (pgvector's
/// `vector_*_ops` and `halfvec_*_ops`); cosine without any. With several distinct metrics the
/// first of cosine, l2, ip wins, and the flag says it was a choice.
pub(crate) fn metric_from_opclasses(opclasses: &[String]) -> (&'static str, bool) {
    let found: Vec<&'static str> = ["cosine", "l2", "ip"]
        .into_iter()
        .filter(|metric| {
            opclasses.iter().any(|opclass| {
                *opclass == format!("vector_{metric}_ops")
                    || *opclass == format!("halfvec_{metric}_ops")
            })
        })
        .collect();
    (found.first().copied().unwrap_or("cosine"), found.len() > 1)
}

struct Col {
    name: String,
    attnum: i16,
    typoid: pg_sys::Oid,
    typname: String,
    typmod: i32,
    type_text: String,
    not_null: bool,
}

impl Col {
    /// What the DDL guard compares: unlike `format_type`, independent of the search_path.
    fn signature(&self) -> String {
        format!("{}:{}", self.typname, self.typmod)
    }
}

fn column<T: FromDatum + IntoDatum>(row: &pgrx::spi::SpiHeapTupleData<'_>, name: &str) -> T {
    row.get_by_name::<T, _>(name)
        .unwrap_or_else(|e| panic!("pg_pinecone: column catalog {name} query failed: {e}"))
        .unwrap_or_else(|| panic!("pg_pinecone: column catalog {name} unexpectedly NULL"))
}

fn read_columns(relid: pg_sys::Oid) -> Vec<Col> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT a.attname::text AS name, a.attnum, a.atttypid AS typoid, \
                        t.typname::text AS typname, a.atttypmod AS typmod, \
                        pg_catalog.format_type(a.atttypid, a.atttypmod) AS type_text, \
                        a.attnotnull AS not_null \
                 FROM pg_catalog.pg_attribute a \
                 JOIN pg_catalog.pg_type t ON t.oid = a.atttypid \
                 WHERE a.attrelid = $1::oid AND a.attnum > 0 AND NOT a.attisdropped \
                 ORDER BY a.attnum",
                None,
                &[relid.into()],
            )
            .expect("pg_pinecone: column catalog query failed")
            .map(|row| Col {
                name: column(&row, "name"),
                attnum: column(&row, "attnum"),
                typoid: column(&row, "typoid"),
                typname: column(&row, "typname"),
                typmod: column(&row, "typmod"),
                type_text: column(&row, "type_text"),
                not_null: column(&row, "not_null"),
            })
            .collect()
    })
}

fn find<'a>(cols: &'a [Col], name: &str, display: &str) -> &'a Col {
    cols.iter().find(|c| c.name == name).unwrap_or_else(|| {
        fail(
            Code::ERRCODE_UNDEFINED_COLUMN,
            format!("column \"{name}\" does not exist in {display}"),
        )
    })
}

/// Requires the caller to own `relid` (or be a member of the owning role, or a superuser).
pub(crate) fn require_owner(relid: pg_sys::Oid, display: &str, function: &str) {
    let caller = unsafe { pg_sys::GetUserId() };
    if !unsafe { pg_sys::object_ownercheck(pg_sys::RelationRelationId, relid, caller) } {
        backend::error_with_code(
            Code::ERRCODE_INSUFFICIENT_PRIVILEGE,
            format!("{function}: must be owner of {display}"),
            None,
            None,
        );
    }
}

/// Statement-level triggers see writes to the table itself only: not to partitions, not to
/// inheritance children, and not to anything that is not an ordinary table.
fn require_plain_table(relid: pg_sys::Oid, display: &str) {
    match catalog::relkind(relid) {
        Some('r') => {}
        Some('p') => fail_hint(
            Code::ERRCODE_FEATURE_NOT_SUPPORTED,
            format!(
                "{display} is a partitioned table — statement-level triggers do not see writes to \
                 partitions"
            ),
            "track each partition separately",
        ),
        _ => fail(
            Code::ERRCODE_FEATURE_NOT_SUPPORTED,
            format!("{display} is not an ordinary table — only ordinary tables can be tracked"),
        ),
    }
    if catalog::has_inheritance_children(relid) {
        fail_hint(
            Code::ERRCODE_FEATURE_NOT_SUPPORTED,
            format!(
                "{display} has inheritance children — statement-level triggers do not see writes \
                 to them"
            ),
            "track the child tables separately",
        );
    }
}

/// The sync triggers take fixed names (`docs_sync::TRIGGERS`): a trigger of the table's own that
/// already has one would fail the registration after Pinecone was called.
fn require_free_trigger_names(relid: pg_sys::Oid, display: &str) {
    if let Some(name) = catalog::trigger_names_among(relid, &docs_sync::TRIGGERS).first() {
        fail_hint(
            Code::ERRCODE_DUPLICATE_OBJECT,
            format!("{display} already has a trigger named \"{name}\""),
            "rename or drop it: pg_pinecone reserves the trigger names pg_pinecone_docs_*",
        );
    }
}

const ID_HINT: &str = "pass id_column => '<col>' naming a NOT NULL UNIQUE column";

/// The id column: `explicit` (NOT NULL and UNIQUE) or the single-column primary key, of a type
/// with a canonical text rendering. With a partition column ids stay unique across the whole
/// table, so a key that is only unique together with the partition (`PRIMARY KEY (org, id)`)
/// needs a unique index on `id` of its own.
fn resolve_id<'a>(
    relid: pg_sys::Oid,
    cols: &'a [Col],
    explicit: Option<&str>,
    partitioned: bool,
    display: &str,
) -> &'a Col {
    let col = match explicit {
        Some(name) => {
            let col = find(cols, name, display);
            if !col.not_null || !catalog::has_immediate_unique_single_column_index(relid, name) {
                let keys = catalog::primary_key_columns(relid);
                if partitioned
                    && col.not_null
                    && keys.len() > 1
                    && keys.iter().any(|k| k.name == name)
                {
                    fail_hint(
                        Code::ERRCODE_FEATURE_NOT_SUPPORTED,
                        format!(
                            "id column \"{name}\" of {display} is unique only together with the \
                             other columns of its primary key — a document id identifies one \
                             row of the whole table, not of one partition"
                        ),
                        "add a table-wide unique column (identity) or a unique index on the id \
                         column, or wait for per-partition ids",
                    );
                }
                fail_hint(
                    Code::ERRCODE_INVALID_TABLE_DEFINITION,
                    format!(
                        "id column \"{name}\" of {display} must be NOT NULL and UNIQUE — a \
                         document id identifies exactly one row"
                    ),
                    "ALTER TABLE ... ALTER COLUMN ... SET NOT NULL, and add a unique constraint \
                     or unique index that is not DEFERRABLE",
                );
            }
            col
        }
        None => match catalog::primary_key_columns(relid).as_slice() {
            [single] => find(cols, &single.name, display),
            [] => fail_hint(
                Code::ERRCODE_INVALID_TABLE_DEFINITION,
                format!("{display} has no primary key — document ids are derived from it"),
                format!("add a primary key, or {ID_HINT}"),
            ),
            keys => fail_hint(
                Code::ERRCODE_FEATURE_NOT_SUPPORTED,
                format!(
                    "the primary key of {display} has {} columns — this release supports \
                     single-column keys",
                    keys.len()
                ),
                ID_HINT,
            ),
        },
    };
    if !catalog::DOC_ID_COLUMN_TYPES.contains(&col.typname.as_str()) {
        fail(
            Code::ERRCODE_FEATURE_NOT_SUPPORTED,
            format!(
                "id column \"{}\" has type {} — supported: smallint, integer, bigint, uuid, \
                 text, varchar, char",
                col.name, col.type_text
            ),
        );
    }
    col
}

fn check_text_columns(cols: &[Col], declared: &[TextColumn], display: &str) {
    for column in declared {
        let col = find(cols, &column.name, display);
        if !catalog::is_text_column_type(col.typoid) {
            fail(
                Code::ERRCODE_DATATYPE_MISMATCH,
                format!(
                    "text column \"{}\" has type {} — declared text columns must be text, \
                     varchar or char",
                    col.name, col.type_text
                ),
            );
        }
    }
}

/// The metadata columns, each existing and typable; none may repeat or also be a text column.
fn resolve_metadata(
    cols: &[Col],
    declared: Option<Vec<Option<String>>>,
    text: &[TextColumn],
    display: &str,
) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for (at, name) in declared.unwrap_or_default().into_iter().enumerate() {
        let Some(name) = name else {
            fail(
                Code::ERRCODE_INVALID_PARAMETER_VALUE,
                format!("metadata_columns[{}] is NULL", at + 1),
            );
        };
        if names.contains(&name) || text.iter().any(|t| t.name == name) {
            fail(
                Code::ERRCODE_INVALID_PARAMETER_VALUE,
                format!("column \"{name}\" is declared twice in text_columns and metadata_columns"),
            );
        }
        let col = find(cols, &name, display);
        docs_schema::check_metadata_column(&name, &col.typname).unwrap_or_else(|e| e.raise());
        names.push(name);
    }
    names
}

/// The dense field: a `vector` or `halfvec` column with a declared dimension.
fn resolve_vector(relid: pg_sys::Oid, cols: &[Col], column: &str, display: &str) -> VectorField {
    let col = find(cols, column, display);
    if !["vector", "halfvec"].contains(&col.typname.as_str()) {
        fail(
            Code::ERRCODE_DATATYPE_MISMATCH,
            format!(
                "vector column \"{column}\" has type {} — it must be vector or halfvec",
                col.type_text
            ),
        );
    }
    let Some(dimension) = u32::try_from(col.typmod).ok().filter(|d| *d > 0) else {
        fail_hint(
            Code::ERRCODE_INVALID_TABLE_DEFINITION,
            format!("vector column \"{column}\" of {display} has no declared dimension"),
            "declare it as vector(<n>) or halfvec(<n>)",
        );
    };
    let opclasses = catalog::index_opclass_names(relid, col.attnum);
    let (metric, chosen) = metric_from_opclasses(&opclasses);
    if chosen {
        notice!(
            "{TRACK}: {display} has indexes on \"{column}\" with several distance metrics — the \
             dense field uses {metric}"
        );
    }
    VectorField {
        column: column.to_string(),
        dimension,
        metric: metric.to_string(),
    }
}

/// Distinct values of the partition column above which a declaration says how many namespaces it
/// is about to create.
const MANY_PARTITIONS: i64 = 1_000;

/// The partition column: it exists, is NOT NULL (a row without a value has no namespace), has a
/// type with a canonical text rendering, and is not the id.
fn resolve_partition(
    relid: pg_sys::Oid,
    cols: &[Col],
    name: &str,
    id: &Col,
    display: &str,
) -> PartitionSpec {
    let col = find(cols, name, display);
    let Some(kind) = PartitionKind::from_typname(&col.typname) else {
        fail(
            Code::ERRCODE_FEATURE_NOT_SUPPORTED,
            format!(
                "partition column \"{name}\" has type {} — supported: smallint, integer, bigint, \
                 text, varchar, uuid",
                col.type_text
            ),
        );
    };
    if !col.not_null {
        fail_hint(
            Code::ERRCODE_INVALID_TABLE_DEFINITION,
            format!(
                "partition column \"{name}\" of {display} must be NOT NULL — a row without a \
                 value has no namespace"
            ),
            "ALTER TABLE ... ALTER COLUMN ... SET NOT NULL",
        );
    }
    if col.name == id.name {
        fail(
            Code::ERRCODE_INVALID_PARAMETER_VALUE,
            format!(
                "partition column \"{name}\" is the id column — a document id and its namespace \
                 come from different columns"
            ),
        );
    }
    if kind == PartitionKind::Text && !catalog::column_is_deterministic(relid, name) {
        fail_hint(
            Code::ERRCODE_FEATURE_NOT_SUPPORTED,
            format!(
                "partition column \"{name}\" of {display} {}",
                partition_lane::NONDETERMINISTIC
            ),
            partition_lane::NONDETERMINISTIC_HINT,
        );
    }
    PartitionSpec {
        column: name.to_string(),
        kind,
    }
}

/// Says how many namespaces the table's rows would fill when that is a lot (Pinecone limits them
/// per index by plan); the count is read as the caller.
fn note_partition_count(schema: &str, table: &str, display: &str, column: &str) {
    use pgrx::spi::{quote_identifier, quote_qualified_identifier};
    let sql = format!(
        "SELECT pg_catalog.count(DISTINCT {}) FROM {}",
        quote_identifier(column),
        quote_qualified_identifier(schema, table)
    );
    let distinct = Spi::get_one::<i64>(&sql)
        .expect("pg_pinecone: partition value count failed")
        .unwrap_or(0);
    if distinct > MANY_PARTITIONS {
        notice!(
            "{TRACK}: {display} holds {distinct} distinct values of \"{column}\", one Pinecone \
             namespace each"
        );
        warning!(
            "{TRACK}: {distinct} namespaces in one index — Pinecone limits namespaces per index by \
             plan (Starter 100, Standard 100,000, Enterprise 1,000,000); check the plan before the \
             backfill runs"
        );
    }
}

fn wait_ready(client: &pg_pinecone_client::Client, index: &str) {
    client
        .wait_doc_index_ready(index, docs::WAIT_READY_DEADLINE, docs::WAIT_READY_POLL)
        .unwrap_or_else(|e| match e {
            ClientError::Timeout { .. } => fail(
                Code::ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE,
                format!(
                    "index \"{index}\" did not become ready within {}s — check its status in the \
                     Pinecone console and run track_docs again with adopt_existing => true",
                    docs::WAIT_READY_DEADLINE.as_secs()
                ),
            ),
            other => backend::raise(other),
        });
}

/// Checks an existing index against the derived schema and, if it matches, adopts it. With a
/// partition column the table takes a whole family of namespaces, `namespace:*`.
fn attach(
    client: &pg_pinecone_client::Client,
    info: pg_pinecone_client::DocIndexInfo,
    index: &str,
    namespace: &str,
    schema: &Value,
    adopt_existing: bool,
    partitioned: bool,
) {
    if let Some(mismatch) = docs_schema::schema_mismatch(schema, &info.schema) {
        fail_hint(
            Code::ERRCODE_INVALID_TABLE_DEFINITION,
            format!(
                "index \"{index}\" exists with a different schema — {}",
                mismatch.describe()
            ),
            "pass a new index_name, or declare matching columns",
        );
    }
    if !adopt_existing {
        let held = match partitioned {
            true => format!("its namespaces \"{namespace}:*\""),
            false => format!("its namespace \"{namespace}\""),
        };
        fail_hint(
            Code::ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE,
            format!(
                "index \"{index}\" already exists — {held} may hold documents this table does not own"
            ),
            "pass adopt_existing => true to attach it (ids without a row never join), or pass \
             a new index_name",
        );
    }
    if partitioned {
        warning!(
            "{TRACK}: index \"{index}\" is attached with a partition column — every namespace \
             starting with \"{namespace}:\" may hold documents this table does not own (ids \
             without a row never join)"
        );
    }
    if !info.ready {
        wait_ready(client, index);
    }
}

/// Creates the index for `schema`: `true` once it is ready, `false` if the name was taken in the
/// meantime (409). Pinecone refusing the schema (4xx) is reported with the schema.
fn create(client: &pg_pinecone_client::Client, index: &str, schema: &Value) -> bool {
    let request = CreateDocIndexRequest {
        name: index.to_string(),
        schema: schema.clone(),
        cloud: CLOUD.to_string(),
        region: REGION.to_string(),
    };
    match client.create_doc_index(&request) {
        Ok(()) => {
            wait_ready(client, index);
            true
        }
        Err(ClientError::Conflict { .. }) => false,
        Err(ClientError::BadRequest { message }) => backend::error_with_code(
            Code::ERRCODE_INVALID_PARAMETER_VALUE,
            format!(
                "{TRACK}: Pinecone rejected the derived schema for index \"{index}\": {message}"
            ),
            Some(&format!("schema: {schema}")),
            Some("fix the declaration and re-run; nothing was registered"),
        ),
        Err(other) => backend::raise(other),
    }
}

/// Creates the index, or attaches the existing one (also when it appears between the lookup and
/// the create); returns whether it was attached. Runs in the caller's own Pinecone project.
fn create_or_attach(
    index: &str,
    namespace: &str,
    schema: &Value,
    adopt_existing: bool,
    partitioned: bool,
) -> bool {
    let client = backend::client_for_effective_key();
    match client.describe_doc_index(index) {
        Ok(info) => {
            attach(
                &client,
                info,
                index,
                namespace,
                schema,
                adopt_existing,
                partitioned,
            );
            return true;
        }
        Err(ClientError::IndexNotFound { .. }) => {}
        Err(other) => backend::raise(other),
    }
    if create(&client, index, schema) {
        return false;
    }
    match client.describe_doc_index(index) {
        Ok(info) => attach(
            &client,
            info,
            index,
            namespace,
            schema,
            adopt_existing,
            partitioned,
        ),
        Err(other) => backend::raise(other),
    }
    true
}

/// The declared columns' type signatures, which the DDL guard compares.
fn column_types(cols: &[Col], names: impl IntoIterator<Item = String>) -> Value {
    let mut types = Map::new();
    for name in names {
        if let Some(col) = cols.iter().find(|c| c.name == name) {
            types.insert(name, Value::String(col.signature()));
        }
    }
    Value::Object(types)
}

/// Implements `pinecone.track_docs`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn track_docs(
    tbl: PgRelation,
    text_columns: Option<JsonB>,
    metadata_columns: Option<Vec<Option<String>>>,
    vector_column: Option<&str>,
    id_column: Option<&str>,
    index_name: Option<&str>,
    adopt_existing: bool,
    partition_column: Option<&str>,
) {
    let relid = tbl.oid();
    let display = catalog::display_name(&tbl);
    let (schema_name, table_name) = (tbl.namespace().to_string(), tbl.name().to_string());
    require_owner(relid, &display, TRACK);
    require_plain_table(relid, &display);
    if let Some(existing) = docs_registry::as_owner(|| docs_registry::lookup(relid)) {
        fail(
            Code::ERRCODE_DUPLICATE_OBJECT,
            format!(
                "{display} is already tracked (index \"{}\") — run pinecone.untrack_docs first",
                existing.index_name
            ),
        );
    }
    require_free_trigger_names(relid, &display);
    let Some(JsonB(declared)) = text_columns else {
        fail(
            Code::ERRCODE_INVALID_PARAMETER_VALUE,
            "text_columns must not be NULL",
        );
    };
    let text = docs_schema::parse_text_columns(&declared).unwrap_or_else(|e| e.raise());
    let cols = read_columns(relid);
    let id = resolve_id(
        relid,
        &cols,
        id_column,
        partition_column.is_some(),
        &display,
    );
    let partition =
        partition_column.map(|column| resolve_partition(relid, &cols, column, id, &display));
    check_text_columns(&cols, &text, &display);
    let metadata = resolve_metadata(&cols, metadata_columns, &text, &display);
    let vector = vector_column.map(|column| resolve_vector(relid, &cols, column, &display));
    if let Some(vector) = vector.as_ref().filter(|v| metadata.contains(&v.column)) {
        fail(
            Code::ERRCODE_INVALID_PARAMETER_VALUE,
            format!(
                "column \"{}\" is declared twice in metadata_columns and vector_column",
                vector.column
            ),
        );
    }
    let schema = docs_schema::derive_schema(&text, vector.as_ref()).unwrap_or_else(|e| e.raise());

    let index = match index_name {
        Some(name) if valid_index_name(name) => name.to_string(),
        Some(name) => fail_hint(
            Code::ERRCODE_INVALID_PARAMETER_VALUE,
            format!("index_name \"{name}\" is not a valid Pinecone index name"),
            "use 1 to 45 lowercase letters, digits or \"-\", starting and ending with a letter \
             or digit",
        ),
        None => default_index_name(&schema_name, &table_name),
    };
    let namespace = format!("{schema_name}.{table_name}");
    if !valid_namespace(&namespace) {
        fail_hint(
            Code::ERRCODE_INVALID_PARAMETER_VALUE,
            format!(
                "namespace \"{namespace}\" (the table's schema and name) is not valid for \
                 Pinecone: it must be printable ASCII"
            ),
            "index_name names the index only; per-table namespaces are not configurable in this \
             release, so rename the table to an ASCII name",
        );
    }
    let claim = Claim::of(&namespace, partition.is_some());
    let in_use = docs_registry::as_owner(|| docs_registry::claim_owner(&index, claim));
    if let Some(owner) = in_use {
        partition_registry::refuse_family_clash(TRACK, &index, claim, &owner);
        fail_hint(
            Code::ERRCODE_DUPLICATE_OBJECT,
            format!(
                "index \"{index}\" namespace \"{namespace}\" is already used by {}",
                catalog::display_name_of(owner.relid)
            ),
            "pass a different index_name",
        );
    }

    if let Some(spec) = &partition {
        note_partition_count(&schema_name, &table_name, &display, &spec.column);
    }
    let attached = create_or_attach(
        &index,
        &namespace,
        &schema,
        adopt_existing,
        partition.is_some(),
    );
    let declared_names = [id.name.clone()]
        .into_iter()
        .chain(text.iter().map(|t| t.name.clone()))
        .chain(metadata.iter().cloned())
        .chain(vector.iter().map(|v| v.column.clone()))
        .chain(partition.iter().map(|p| p.column.clone()));
    let types = column_types(&cols, declared_names);
    let row = DocRegistryRow {
        relid,
        id_column: id.name.clone(),
        id_type: id.typname.clone(),
        text_columns: docs_schema::text_columns_json(&text),
        metadata_columns: metadata,
        vector_column: vector.as_ref().map(|v| v.column.clone()),
        vector_dimension: vector.as_ref().map(|v| v.dimension as i32),
        vector_metric: vector.as_ref().map(|v| v.metric.clone()),
        index_name: index,
        namespace,
        schema,
        attached,
        state: DocState::Backfilling,
        backfill_cursor: None,
        backfill_total: None,
        backfill_done: 0,
        queue_parked: false,
        parked_reason: None,
        last_delivered_at: None,
        partition,
    };
    // As the registry's owner: neither the registry nor the queue is writable by the caller,
    // and the triggers name a function only that role may execute.
    docs_registry::as_owner(|| {
        docs_registry::insert(&row);
        docs_registry::set_column_types(relid, &types);
        docs_sync::install_triggers(&schema_name, &table_name);
    });
    let total = docs_backfill::count_rows(relid);
    // The backfill job sends the rows that exist now, once the worker runs.
    docs_registry::as_owner(|| docs_backfill::enqueue_job(relid, false, total));
    if !outbox::launcher_lock_held() {
        warning!(
            "{TRACK}: no pg_pinecone worker launcher is running in this cluster (pg_pinecone is not in \
             shared_preload_libraries, or pinecone.worker_enabled is off) — nothing will be \
             synced until the worker runs; see pinecone.status()"
        );
    }
}

/// Empties every namespace a partitioned table wrote: the registry's, and those of the values the
/// table holds (a drain that rolled back after Pinecone took a document of a new value lost that
/// value's registration, and nothing lists the namespaces of a document index). A refusal by
/// Pinecone leaves the table tracked; an index that is gone leaves nothing to empty.
fn empty_partitions(tbl: &PgRelation, row: &DocRegistryRow, spec: &PartitionSpec) {
    let relid = tbl.oid();
    let table = pgrx::spi::quote_qualified_identifier(tbl.namespace(), tbl.name());
    let column = pgrx::spi::quote_identifier(&spec.column);
    partition_ops::adopt_table_values(relid, Lane::Docs, &row.namespace, &table, &column);
    if partition_registry::partitions_of(relid, Lane::Docs).is_empty() {
        return;
    }
    let client = backend::client_for_effective_key();
    match partition_registry::wipe_partitions(relid, Lane::Docs, &client, &row.index_name) {
        Ok(_) => {}
        Err(ClientError::IndexNotFound { .. }) => warning!(
            "pinecone.untrack_docs: index '{}' not found, so its namespaces cannot be emptied; \
             the teardown goes on",
            row.index_name
        ),
        Err(other) => backend::raise(other),
    }
}

/// Implements `pinecone.untrack_docs`: the registry row goes first (which ends the DDL guard's
/// protection), then the triggers and the queue; the Pinecone index only with `drop_index`,
/// which an attached index (one pg_pinecone did not create) needs `force` for. Without it a table
/// with a partition column has its namespaces emptied first: the registry is the only list of
/// them, and `drop_index` takes them with the index.
pub(crate) fn untrack_docs(tbl: PgRelation, drop_index: bool, force: bool) {
    const UNTRACK: &str = "pinecone.untrack_docs";
    let relid = tbl.oid();
    let display = catalog::display_name(&tbl);
    require_owner(relid, &display, UNTRACK);
    // The same lock order as a worker round (lane lock, then table locks).
    docs_drain::lock_lane_for_verb(relid, &display);
    let row = docs_registry::as_owner(|| {
        let Some(row) = docs_registry::lookup(relid) else {
            backend::error_with_code(
                Code::ERRCODE_UNDEFINED_OBJECT,
                format!("{UNTRACK}: {display} is not tracked by pinecone.track_docs"),
                None,
                None,
            );
        };
        // Refused before anything is torn down, so a refusal leaves the table tracked.
        if drop_index && row.attached && !force {
            backend::error_with_code(
                Code::ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE,
                format!(
                    "{UNTRACK}: index \"{}\" was attached to {display}, not created by pg_pinecone \
                     — drop_index => true would delete it",
                    row.index_name
                ),
                None,
                Some(
                    "the index was attached, not created by pg_pinecone; delete it in Pinecone if \
                     you mean it, or pass force => true",
                ),
            );
        }
        if drop_index && docs_registry::other_row_uses_index(&row.index_name, relid) {
            backend::error_with_code(
                Code::ERRCODE_OBJECT_IN_USE,
                format!(
                    "{UNTRACK}: index \"{}\" is also used by another tracked table — untrack \
                     that table first, or delete the index in Pinecone yourself",
                    row.index_name
                ),
                None,
                Some("run untrack_docs without drop_index to keep the index"),
            );
        }
        row
    });
    if let Some(spec) = &row.partition
        && !drop_index
    {
        empty_partitions(&tbl, &row, spec);
    }
    docs_registry::as_owner(|| {
        docs_registry::delete(relid);
        docs_sync::drop_triggers(tbl.namespace(), tbl.name());
        docs_outbox::clear_relid(relid);
    });
    partition_registry::forget_partitions(relid, Lane::Docs);
    if drop_index {
        match backend::client_for_effective_key().delete_doc_index(&row.index_name) {
            Ok(()) | Err(ClientError::IndexNotFound { .. }) => {}
            Err(other) => backend::raise(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|name| name.to_string()).collect()
    }

    #[test]
    fn index_names_follow_pinecones_rule() {
        for ok in [
            "a",
            "fts-articles",
            "pd-public-articles",
            "a1-b2",
            &"x".repeat(45),
        ] {
            assert!(valid_index_name(ok), "{ok}");
        }
        for bad in ["", "-a", "a-", "A", "a_b", "a b", "é", &"x".repeat(46)] {
            assert!(!valid_index_name(bad), "{bad:?}");
        }
    }

    #[test]
    fn namespaces_are_printable_ascii() {
        for ok in ["public.articles", "my schema.t a-b_c", "public.~tilde!"] {
            assert!(valid_namespace(ok), "{ok}");
        }
        for bad in ["", "public.tablé", "public.a\nb", "public.a\u{7f}b", "ü.t"] {
            assert!(!valid_namespace(bad), "{bad:?}");
        }
    }

    #[test]
    fn the_default_index_name_is_valid_and_differs_from_the_vector_default() {
        let name = default_index_name("public", "Articles_2026");
        assert_eq!(name, "pd-public-articles-2026");
        assert_ne!(name, promote::default_index_name("public", "Articles_2026"));
        let long = default_index_name("public", &"t".repeat(80));
        assert!(valid_index_name(&long), "{long}");
        assert!(valid_index_name(&default_index_name(
            "my schema",
            "ü-table"
        )));
    }

    #[test]
    fn the_metric_follows_the_opclass_and_defaults_to_cosine() {
        assert_eq!(metric_from_opclasses(&[]), ("cosine", false));
        assert_eq!(
            metric_from_opclasses(&names(&["vector_l2_ops"])),
            ("l2", false)
        );
        assert_eq!(
            metric_from_opclasses(&names(&["vector_ip_ops"])),
            ("ip", false)
        );
        assert_eq!(
            metric_from_opclasses(&names(&["halfvec_l2_ops"])),
            ("l2", false)
        );
        assert_eq!(
            metric_from_opclasses(&names(&["vector_cosine_ops"])),
            ("cosine", false)
        );
        assert_eq!(
            metric_from_opclasses(&names(&["vector_l1_ops", "btree_ops"])),
            ("cosine", false),
            "opclasses Pinecone has no metric for are ignored"
        );
    }

    #[test]
    fn several_metrics_pick_the_first_of_cosine_l2_ip_and_say_so() {
        let both = names(&["vector_ip_ops", "vector_l2_ops"]);
        assert_eq!(metric_from_opclasses(&both), ("l2", true));
        let all = names(&["halfvec_ip_ops", "vector_cosine_ops", "vector_l2_ops"]);
        assert_eq!(metric_from_opclasses(&all), ("cosine", true));
    }
}
