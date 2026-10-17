//! `pinecone.search_docs`: validates every argument, builds the Pinecone request in pure code
//! ([`plan`]) and returns ids, scores and ranks for the caller to join back to the table.

use std::collections::BTreeMap;

use pg_pinecone_client::{Error as ClientError, SearchDocsRequest};
use pgrx::{
    AnyElement, JsonB, PgRelation, PgSqlErrorCode as Code,
    callconv::{Arg, ArgAbi},
    pg_sys,
    spi::Spi,
};
use serde_json::{Map, Value, json};

use crate::{
    backend, catalog, docs_local, docs_outbox,
    docs_registry::{self, DocRegistryRow, DocState},
    docs_schema::{self, MetaKind},
    filter::{self, ColKind},
    partition::{self, NamespaceError, PartitionKind, PartitionSpec},
};

const SEARCH: &str = "pinecone.search_docs";
const MAX_K: i32 = 10_000;
const TEXT_MATCH_OPERATORS: [&str; 3] = ["$match_phrase", "$match_all", "$match_any"];
const QUERY_STRING_HINT: &str = "mode => 'query_string' passes Lucene syntax through unchanged — \
     quote or escape operators, or use mode => 'text'";
const TEXT_QUERY_HINT: &str =
    "the request is built from query, must and filter: check them against the declared columns";
const LOCAL_ENGINE_HINT: &str = "pass engine => 'local' to search this table with Postgres FTS";
const OUTAGE_HINT: &str = "transient — retry in a moment, or pass engine => 'local' to search \
     this table with Postgres FTS";
const UNTRACKED_HINT: &str = "pass engine => 'local' with field => '<column>' to search this \
     table with Postgres FTS, or use pinecone.search_doc_index(...) for an ad-hoc Pinecone index";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SearchError {
    pub(crate) code: Code,
    pub(crate) message: String,
    pub(crate) hint: Option<String>,
}

impl SearchError {
    pub(crate) fn new(code: Code, message: impl Into<String>) -> Self {
        Self {
            code,
            message: format!("{SEARCH}: {}", message.into()),
            hint: None,
        }
    }

    pub(crate) fn hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    pub(crate) fn raise(self) -> ! {
        backend::error_with_code(self.code, self.message, None, self.hint.as_deref())
    }
}

pub(crate) fn invalid(message: impl Into<String>) -> SearchError {
    SearchError::new(Code::ERRCODE_INVALID_PARAMETER_VALUE, message)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Text,
    QueryString,
}

fn parse_mode(mode: &str) -> Result<Mode, SearchError> {
    match mode {
        "text" => Ok(Mode::Text),
        "query_string" => Ok(Mode::QueryString),
        other => Err(invalid(format!(
            "mode must be 'text' or 'query_string', got '{other}'"
        ))),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Engine {
    Remote,
    Local,
}

fn parse_engine(engine: &str) -> Result<Engine, SearchError> {
    match engine {
        "remote" => Ok(Engine::Remote),
        "local" => Ok(Engine::Local),
        other => Err(invalid(format!(
            "engine must be 'remote' or 'local', got '{other}'"
        ))),
    }
}

fn check_k(k: i32) -> Result<u32, SearchError> {
    match u32::try_from(k) {
        Ok(k) if k <= MAX_K as u32 => Ok(k),
        _ => Err(invalid(format!(
            "k must be between 0 and {MAX_K} (got {k})"
        ))),
    }
}

fn null_error(what: impl std::fmt::Display) -> SearchError {
    SearchError::new(
        Code::ERRCODE_NULL_VALUE_NOT_ALLOWED,
        format!("{what} must not be NULL"),
    )
}

fn vector_values(vector: Option<Vec<Option<f32>>>) -> Result<Option<Vec<f32>>, SearchError> {
    let all_set = |elements: Vec<Option<f32>>| {
        let numbered = elements.into_iter().enumerate();
        numbered
            .map(|(i, v)| v.ok_or_else(|| null_error(format!("vector[{}]", i + 1))))
            .collect()
    };
    vector.map(all_set).transpose()
}

struct Declared {
    table: String,
    index: String,
    text_columns: Vec<String>,
    metadata: BTreeMap<String, MetaKind>,
    vector: Option<(String, i32)>,
}

fn declared_hint(declared: &[String]) -> String {
    format!("declared text columns: {}", declared.join(", "))
}

impl Declared {
    fn text_columns_hint(&self) -> String {
        declared_hint(&self.text_columns)
    }
}

struct Args<'a> {
    query: Option<&'a str>,
    mode: Mode,
    field: Option<&'a str>,
    must: Option<&'a Value>,
    filter: Option<&'a Value>,
    vector: Option<&'a [f32]>,
}

struct Body {
    score_by: Value,
    filter: Option<Value>,
}

type EpochSeconds<'a> = &'a mut dyn FnMut(MetaKind, &str) -> Option<f64>;

fn plan(declared: &Declared, args: &Args, epoch: EpochSeconds) -> Result<Body, SearchError> {
    let query = match args.query {
        Some(query) if query.trim().is_empty() => return Err(invalid("query must not be empty")),
        other => other,
    };
    if let Some(field) = args.field {
        check_field(declared, field)?;
    }
    if query.is_some() {
        check_query_arguments(args)?;
    }
    let must = args.must.map(|m| check_must(declared, m)).transpose()?;
    let filter = args
        .filter
        .map(|f| type_filter(declared, f, epoch))
        .transpose()?;

    let (score_by, requirement) = match (args.vector, query) {
        (Some(values), query) => {
            let column = check_vector(declared, values)?;
            let requirement = query.zip(args.field).map(|(q, f)| match_all(f, q));
            let clause = json!({"type": "dense_vector", "fields": [column], "values": values});
            (clause, requirement)
        }
        (None, Some(query)) => (scoring_clause(declared, args, query), None),
        (None, None) => {
            return Err(invalid(
                "nothing to rank by — pass query => (text) or vector => (a query vector)",
            ));
        }
    };

    let mut parts: Vec<Value> = must.into_iter().chain(requirement).chain(filter).collect();
    let filter = match parts.len() {
        0 => None,
        1 => parts.pop(),
        _ => Some(json!({ "$and": parts })),
    };
    Ok(Body { score_by, filter })
}

fn check_query_arguments(args: &Args) -> Result<(), SearchError> {
    match (args.mode, args.field, args.vector) {
        (Mode::QueryString, Some(_), _) => Err(invalid(
            "field => cannot narrow mode => 'query_string': the API takes field qualifiers inside \
             the query",
        )
        .hint("write the qualifier into the query, e.g. body:(vector search)")),
        (Mode::QueryString, None, Some(_)) => Err(invalid(
            "mode => 'query_string' cannot be combined with vector => — a request has one \
             scoring type",
        )
        .hint("use mode => 'text' with field =>, or drop query =>")),
        (Mode::Text, None, Some(_)) => Err(invalid(
            "vector => with query => needs field => (one text column) — the query becomes a \
             $match_all requirement on it",
        )),
        _ => Ok(()),
    }
}

fn scoring_clause(declared: &Declared, args: &Args, query: &str) -> Value {
    match args.mode {
        Mode::QueryString => json!({"type": "query_string", "query": query}),
        Mode::Text => {
            let fields = match args.field {
                Some(field) => vec![field.to_string()],
                None => declared.text_columns.clone(),
            };
            json!({"type": "text", "fields": fields, "query": query})
        }
    }
}

fn match_all(field: &str, query: &str) -> Value {
    let mut requirement = Map::new();
    requirement.insert(field.to_string(), json!({"$match_all": query}));
    Value::Object(requirement)
}

fn check_field(declared: &Declared, field: &str) -> Result<(), SearchError> {
    if declared.text_columns.iter().any(|column| column == field) {
        return Ok(());
    }
    Err(undeclared_field(
        &declared.table,
        field,
        &declared.text_columns,
    ))
}

/// `field` names no declared text column of `table` (42703); the HINT lists the declared ones.
pub(crate) fn undeclared_field(table: &str, field: &str, declared: &[String]) -> SearchError {
    SearchError::new(
        Code::ERRCODE_UNDEFINED_COLUMN,
        format!("field \"{field}\" is not a declared text column of {table}"),
    )
    .hint(declared_hint(declared))
}

fn check_must(declared: &Declared, must: &Value) -> Result<Value, SearchError> {
    const EXAMPLE: &str = r#"{"body": {"$match_all": "text"}}"#;
    let Some(columns) = must.as_object().filter(|columns| !columns.is_empty()) else {
        return Err(invalid(format!(
            "must must be a non-empty JSON object like {EXAMPLE}"
        )));
    };
    for (column, operators) in columns {
        if !declared.text_columns.contains(column) {
            return Err(invalid(format!(
                "must: \"{column}\" is not a declared text column of {}",
                declared.table
            ))
            .hint(declared.text_columns_hint()));
        }
        let Some(operators) = operators.as_object().filter(|ops| !ops.is_empty()) else {
            return Err(invalid(format!(
                "must: \"{column}\" needs an operator object like {EXAMPLE}"
            )));
        };
        for (operator, text) in operators {
            if !TEXT_MATCH_OPERATORS.contains(&operator.as_str()) {
                return Err(invalid(format!(
                    "must: operator \"{operator}\" is not supported on \"{column}\""
                ))
                .hint(format!("use {}", TEXT_MATCH_OPERATORS.join(", "))));
            }
            if !text.as_str().is_some_and(|text| !text.trim().is_empty()) {
                return Err(invalid(format!(
                    "must: \"{column}\" {operator} needs a non-empty string, got {text}"
                )));
            }
        }
    }
    Ok(must.clone())
}

fn check_vector<'a>(declared: &'a Declared, values: &[f32]) -> Result<&'a str, SearchError> {
    let Some((column, dimension)) = &declared.vector else {
        return Err(invalid(format!(
            "{} has no vector column — vector => needs one",
            declared.table
        ))
        .hint(
            "untrack_docs the table, then track_docs it again with vector_column => '<column>'",
        ));
    };
    if i64::try_from(values.len()) != Ok(i64::from(*dimension)) {
        return Err(SearchError::new(
            Code::ERRCODE_DATA_EXCEPTION,
            format!(
                "vector has {} dimensions — index \"{}\" expects {dimension}",
                values.len(),
                declared.index
            ),
        ));
    }
    if let Some(position) = values.iter().position(|value| !value.is_finite()) {
        return Err(invalid(format!(
            "vector[{}] is not a finite number",
            position + 1
        )));
    }
    Ok(column)
}

fn column_kind(kind: MetaKind) -> ColKind {
    match kind {
        MetaKind::Int => ColKind::Int,
        MetaKind::Float | MetaKind::Date | MetaKind::Timestamp | MetaKind::Timestamptz => {
            ColKind::Float
        }
        MetaKind::Numeric => ColKind::Numeric,
        MetaKind::Bool => ColKind::Bool,
        MetaKind::Text | MetaKind::Uuid => ColKind::Text,
        MetaKind::TextArray => ColKind::TextArray,
    }
}

fn is_instant(kind: MetaKind) -> bool {
    matches!(
        kind,
        MetaKind::Date | MetaKind::Timestamp | MetaKind::Timestamptz
    )
}

fn instant_name(kind: MetaKind) -> &'static str {
    match kind {
        MetaKind::Date => "date",
        MetaKind::Timestamp => "timestamp",
        _ => "timestamptz",
    }
}

fn type_filter(
    declared: &Declared,
    filter: &Value,
    epoch: EpochSeconds,
) -> Result<Value, SearchError> {
    let typed = type_instants(filter, &declared.metadata, epoch)
        .map_err(|message| invalid(format!("filter: {message}")))?;
    let columns: BTreeMap<String, ColKind> = declared
        .metadata
        .iter()
        .map(|(name, kind)| (name.clone(), column_kind(*kind)))
        .collect();
    filter::translate(&typed, &columns).map_err(|message| filter_error(declared, &message))?;
    Ok(typed)
}

fn filter_error(declared: &Declared, message: &str) -> SearchError {
    let unknown = message
        .strip_prefix("unknown column '")
        .and_then(|rest| rest.strip_suffix('\''));
    match unknown {
        Some(column) => invalid(format!(
            "filter: '{column}' is not a declared metadata column of {}",
            declared.table
        ))
        .hint(format!(
            "filter takes metadata columns ({}); use must => for text-match on a text column",
            declared
                .metadata
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        )),
        None if message.starts_with("filter ") => invalid(message),
        None => invalid(format!("filter: {message}")),
    }
}

/// Rewrites the date and timestamp strings of `node` to epoch seconds, the way the documents
/// carry them; everything else, including what `filter::translate` will reject, is left as is.
fn type_instants(
    node: &Value,
    metadata: &BTreeMap<String, MetaKind>,
    epoch: EpochSeconds,
) -> Result<Value, String> {
    let Value::Object(entries) = node else {
        return Ok(node.clone());
    };
    let mut typed = Map::new();
    for (key, value) in entries {
        let converted = match (key.as_str(), metadata.get(key)) {
            ("$and" | "$or", _) => match value {
                Value::Array(items) => {
                    let mut branches = Vec::with_capacity(items.len());
                    for item in items {
                        branches.push(type_instants(item, metadata, &mut *epoch)?);
                    }
                    Value::Array(branches)
                }
                other => other.clone(),
            },
            (_, Some(kind)) if is_instant(*kind) => type_column(key, *kind, value, &mut *epoch)?,
            _ => value.clone(),
        };
        typed.insert(key.clone(), converted);
    }
    Ok(Value::Object(typed))
}

fn type_column(
    column: &str,
    kind: MetaKind,
    value: &Value,
    epoch: EpochSeconds,
) -> Result<Value, String> {
    let mut seconds = |value: &Value| match value {
        Value::String(text) => epoch(kind, text)
            .and_then(serde_json::Number::from_f64)
            .map(Value::Number)
            .ok_or_else(|| {
                let name = instant_name(kind);
                format!("\"{text}\" is not a valid {name} value for column \"{column}\"")
            }),
        other => Ok(other.clone()),
    };
    let Value::Object(operators) = value else {
        return seconds(value);
    };
    let mut typed = Map::new();
    for (operator, operand) in operators {
        let converted = match operand {
            Value::Array(items) => Value::Array(
                items
                    .iter()
                    .map(&mut seconds)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            other => seconds(other)?,
        };
        typed.insert(operator.clone(), converted);
    }
    Ok(Value::Object(typed))
}

fn or_raise<T>(result: Result<T, SearchError>) -> T {
    result.unwrap_or_else(|error| error.raise())
}

fn untracked(display: &str) -> ! {
    SearchError::new(
        Code::ERRCODE_UNDEFINED_OBJECT,
        format!("{display} is not tracked — run pinecone.track_docs first"),
    )
    .hint(UNTRACKED_HINT)
    .raise()
}

fn unsupported(message: impl Into<String>) -> SearchError {
    SearchError::new(Code::ERRCODE_FEATURE_NOT_SUPPORTED, message)
}

/// Refuses what the local engine cannot do rather than search differently; returns the query.
fn check_local<'a>(args: &Args<'a>) -> Result<&'a str, SearchError> {
    if args.mode == Mode::QueryString {
        return Err(
            unsupported("mode => 'query_string' is not supported on the local engine")
                .hint("use mode => 'text', or engine => 'remote'"),
        );
    }
    if args.must.is_some() || args.filter.is_some() {
        return Err(unsupported(
            "must/filter are not supported on the local engine in this release",
        )
        .hint("put the condition in a WHERE clause on the join, or use engine => 'remote'"));
    }
    if args.vector.is_some() {
        return Err(
            unsupported("vector => is not supported on the local engine in this release").hint(
                "rank with ORDER BY <vector column> <=> $1 in SQL, or use engine => 'remote'",
            ),
        );
    }
    match args.query {
        Some(query) if !query.trim().is_empty() => Ok(query),
        Some(_) => Err(invalid("query must not be empty")),
        None => Err(invalid("nothing to search — pass query => (text)")),
    }
}

fn search_local(
    tbl: &PgRelation,
    row: Option<&DocRegistryRow>,
    args: &Args,
    k: u32,
    partition: Option<&PartitionArg>,
) -> Vec<(String, f32, i32)> {
    let query = or_raise(check_local(args));
    let scope = or_raise(local_scope(row, partition));
    let target = or_raise(docs_local::resolve(tbl, row, args.field)).scoped(scope);
    if k == 0 {
        return Vec::new();
    }
    docs_local::search(&target, query, k)
}

/// Epoch seconds like the documents carry them: a `date` and a `timestamp` count as UTC, a
/// `timestamptz` string without an offset follows the session time zone.
fn epoch_seconds(kind: MetaKind, text: &str) -> Option<f64> {
    let (type_name, instant) = match kind {
        MetaKind::Timestamptz => ("timestamptz", "$1::timestamptz"),
        MetaKind::Timestamp => ("timestamp", "$1::timestamp"),
        MetaKind::Date => ("date", "$1::date::timestamp"),
        _ => return None,
    };
    let sql = format!(
        "SELECT CASE WHEN pg_catalog.pg_input_is_valid($1, '{type_name}') \
         THEN pg_catalog.date_part('epoch', {instant}) END"
    );
    Spi::get_one_with_args::<f64>(&sql, &[text.into()])
        .expect("pg_pinecone: epoch seconds query failed")
        .filter(|seconds| seconds.is_finite())
}

fn declared_from(row: &DocRegistryRow, display: String) -> Declared {
    let text_columns = row
        .text_columns
        .as_object()
        .map(|columns| columns.keys().cloned().collect())
        .unwrap_or_default();
    let types: BTreeMap<String, String> = catalog::all_columns(row.relid)
        .into_iter()
        .map(|column| (column.name, column.type_name))
        .collect();
    let metadata = row
        .metadata_columns
        .iter()
        .filter_map(|name| {
            let kind = docs_schema::metadata_kind(types.get(name)?)?;
            Some((name.clone(), kind))
        })
        .collect();
    Declared {
        table: display,
        index: row.index_name.clone(),
        text_columns,
        metadata,
        vector: row.vector_column.clone().zip(row.vector_dimension),
    }
}

/// A rejected query gets the Lucene hint only in `query_string` mode, where the text is parsed.
fn explain(err: ClientError, mode: Mode) -> Result<SearchError, ClientError> {
    match err {
        ClientError::BadRequest { message } => {
            let hint = match mode {
                Mode::QueryString => QUERY_STRING_HINT,
                Mode::Text => TEXT_QUERY_HINT,
            };
            Ok(invalid(format!("Pinecone rejected the query: {message}")).hint(hint))
        }
        ClientError::Api { status, message } => {
            let text = format!("Pinecone returned {status}: {message}");
            match backend::classify_429(status, &message) {
                Some(limit) => Ok(SearchError::new(limit.sqlstate(), text).hint(limit.hint())),
                None if status >= 500 => {
                    Ok(SearchError::new(Code::ERRCODE_IO_ERROR, text).hint(OUTAGE_HINT))
                }
                None => Err(ClientError::Api { status, message }),
            }
        }
        ClientError::Timeout { ms } => Ok(SearchError::new(
            Code::ERRCODE_IO_ERROR,
            format!(
                "request timed out after {ms}ms — raise pinecone.timeout_ms if the index is \
                 slow to respond"
            ),
        )
        .hint(LOCAL_ENGINE_HINT)),
        ClientError::Transport { url, reason } => Ok(SearchError::new(
            Code::ERRCODE_IO_ERROR,
            format!(
                "could not reach {url}: {reason} — check pinecone.control_plane_url and \
                 network connectivity"
            ),
        )
        .hint(LOCAL_ENGINE_HINT)),
        other => Err(other),
    }
}

fn fail(err: ClientError, mode: Mode) -> ! {
    match explain(err, mode) {
        Ok(error) => error.raise(),
        Err(other) => backend::raise(other),
    }
}

/// What a remote search says while the first backfill is still delivering.
fn backfilling_notice(table: &str, done: i64, total: Option<i64>) -> String {
    let delivered = match total {
        Some(total) => format!("{done} of {total} rows delivered"),
        None => format!("{done} rows delivered"),
    };
    format!("{SEARCH}: {table} is still backfilling: {delivered}; results are partial")
}

/// Once per call, only while `row` is backfilling: Pinecone holds part of the table. Not for the
/// local engine, which searches the heap.
fn note_if_backfilling(table: &str, row: &DocRegistryRow) {
    if row.state == DocState::Backfilling {
        let notice = backfilling_notice(table, row.backfill_done, row.backfill_total);
        pgrx::notice!("{notice}");
    }
}

/// The `partition` argument: a value of any type, `anycompatible` in SQL. A `text` parameter
/// would refuse `partition => 42` (no implicit cast from an integer) and `anyelement` refuses the
/// untyped literal in `partition => 'acme'`; `anycompatible` takes both, and an omitted argument
/// is a NULL text.
pub(crate) struct PartitionArg(AnyElement);

// SAFETY: delegates to `AnyElement`, which unboxes the argument with its own type.
unsafe impl<'fcx> ArgAbi<'fcx> for PartitionArg {
    unsafe fn unbox_arg_unchecked(arg: Arg<'_, 'fcx>) -> Self {
        Self(unsafe { arg.unbox_unchecked::<AnyElement>() })
    }
}

pgrx::impl_sql_translatable!(PartitionArg, arg_only = "anycompatible");

impl PartitionArg {
    /// The value as text, for the types a partition column holds (`varchar` reads as text); the
    /// oid of any other type is the error.
    pub(crate) fn read(&self) -> Result<Option<String>, pg_sys::Oid> {
        let element = &self.0;
        // SAFETY: `oid()` is the type of the datum.
        unsafe {
            match element.oid() {
                pg_sys::INT2OID => Ok(element.into::<i16>().map(|value| value.to_string())),
                pg_sys::INT4OID => Ok(element.into::<i32>().map(|value| value.to_string())),
                pg_sys::INT8OID => Ok(element.into::<i64>().map(|value| value.to_string())),
                pg_sys::TEXTOID | pg_sys::VARCHAROID => Ok(element.into::<String>()),
                pg_sys::UUIDOID => Ok(element.into::<pgrx::Uuid>().map(|value| value.to_string())),
                other => Err(other),
            }
        }
    }

    fn text(&self) -> Result<String, SearchError> {
        match self.read() {
            Ok(Some(text)) => Ok(text),
            Ok(None) => Err(null_error("partition")),
            Err(oid) => Err(unsupported_partition_type(oid)),
        }
    }
}

pub(crate) fn type_name(oid: pg_sys::Oid) -> String {
    Spi::get_one_with_args::<String>(
        "SELECT $1::pg_catalog.oid::pg_catalog.regtype::pg_catalog.text",
        &[oid.into()],
    )
    .ok()
    .flatten()
    .unwrap_or_else(|| format!("type {oid:?}"))
}

fn unsupported_partition_type(oid: pg_sys::Oid) -> SearchError {
    SearchError::new(
        Code::ERRCODE_DATATYPE_MISMATCH,
        format!(
            "partition is of type {}, which no partition column holds",
            type_name(oid)
        ),
    )
    .hint("pass an integer, a text or a uuid value")
}

/// A filter operand as the text of a partition value: a string, or an integer.
pub(crate) fn operand_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => number
            .as_i64()
            .map(|n| n.to_string())
            .or_else(|| number.as_u64().map(|n| n.to_string())),
        _ => None,
    }
}

/// `text` as the canonical text of a `kind` value (what `col::text` gives), `None` if it reads as
/// no such value. Two statements: the planner folds a cast of a constant even in a `CASE` branch
/// that is never reached, and `int8in` would raise there.
pub(crate) fn canonical(text: &str, kind: PartitionKind) -> Option<String> {
    if kind == PartitionKind::Text {
        return Some(text.to_string());
    }
    let ty = kind.as_str();
    let valid = Spi::get_one_with_args::<bool>(
        &format!("SELECT pg_catalog.pg_input_is_valid($1, '{ty}')"),
        &[text.into()],
    )
    .expect("pg_pinecone: partition value check failed");
    if valid != Some(true) {
        return None;
    }
    Spi::get_one_with_args::<String>(
        &format!("SELECT ($1::pg_catalog.{ty})::pg_catalog.text"),
        &[text.into()],
    )
    .expect("pg_pinecone: partition value query failed")
}

fn unbound_partition(table: &str, column: &str) -> SearchError {
    invalid(format!(
        "{table} is partitioned by \"{column}\": a search needs one partition"
    ))
    .hint(format!(
        "pass partition => <value> (or filter => '{{\"{column}\": {{\"$eq\": <value>}}}}' with \
         \"{column}\" in metadata_columns); BM25 scores of different partitions cannot be merged"
    ))
}

/// The canonical text of `text` as a value of `spec`'s column, or the refusal.
fn partition_value(spec: &PartitionSpec, text: &str) -> Result<String, SearchError> {
    canonical(text, spec.kind).ok_or_else(|| {
        invalid(format!(
            "partition {text:?} is not a valid {} value for \"{}\"",
            spec.kind.as_str(),
            spec.column
        ))
    })
}

/// The local engine reads the heap, so `partition` is a condition on the column (`AND col =
/// value`) where the remote engine searches one namespace. A table without a partition column
/// has nothing to scope by and, as before, ignores the argument here.
fn local_scope(
    row: Option<&DocRegistryRow>,
    partition: Option<&PartitionArg>,
) -> Result<Option<docs_local::Scope>, SearchError> {
    let (Some(partition), Some(spec)) = (partition, row.and_then(|row| row.partition.as_ref()))
    else {
        return Ok(None);
    };
    let value = partition_value(spec, &partition.text()?)?;
    Ok(Some(docs_local::Scope::new(spec, value)))
}

/// The Pinecone namespace a search of `row` goes to: its own, or with a partition column the one
/// of the value `partition` names (or, failing that, the filter's `$eq` on the column).
fn search_namespace(
    row: &DocRegistryRow,
    table: &str,
    partition: Option<&PartitionArg>,
    filter: Option<&Value>,
) -> Result<String, SearchError> {
    let Some(spec) = &row.partition else {
        return match partition {
            None => Ok(row.namespace.clone()),
            Some(_) => Err(invalid(format!(
                "{table} has no partition column, so partition => does not apply"
            ))
            .hint("omit partition, or track the table again with partition_column => '<col>'")),
        };
    };
    let text = match partition {
        Some(arg) => arg.text()?,
        None => filter
            .and_then(|filter| filter::equality_on(filter, &spec.column))
            .and_then(operand_text)
            .ok_or_else(|| unbound_partition(table, &spec.column))?,
    };
    let canon = partition_value(spec, &text)?;
    let namespace = partition::namespace_for(&row.namespace, &canon);
    partition::fits_pinecone(&namespace).map_err(|error| {
        let code = match error {
            NamespaceError::TooLong { .. } => Code::ERRCODE_PROGRAM_LIMIT_EXCEEDED,
            _ => Code::ERRCODE_INVALID_PARAMETER_VALUE,
        };
        SearchError::new(
            code,
            format!("the namespace of this partition is unusable: {error}"),
        )
    })?;
    Ok(namespace)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn search_docs(
    tbl: PgRelation,
    query: Option<&str>,
    mode: Option<&str>,
    field: Option<&str>,
    must: Option<JsonB>,
    filter: Option<JsonB>,
    vector: Option<Vec<Option<f32>>>,
    engine: Option<&str>,
    k: Option<i32>,
    partition: Option<PartitionArg>,
) -> Vec<(String, f32, i32)> {
    let relid = tbl.oid();
    let display = catalog::display_name(&tbl);
    let mode = or_raise(parse_mode(
        mode.unwrap_or_else(|| null_error("mode").raise()),
    ));
    let engine = or_raise(parse_engine(
        engine.unwrap_or_else(|| null_error("engine").raise()),
    ));
    let k = or_raise(check_k(k.unwrap_or_else(|| null_error("k").raise())));
    let vector = or_raise(vector_values(vector));

    let args = Args {
        query,
        mode,
        field,
        must: must.as_ref().map(|json| &json.0),
        filter: filter.as_ref().map(|json| &json.0),
        vector: vector.as_deref(),
    };
    // The caller's right comes first: it decides whether the registry is read for them at all,
    // so an untracked table and a tracked one look the same to a role without SELECT.
    docs_outbox::require_select(relid, tbl.name());
    let row = docs_registry::as_owner(|| docs_registry::lookup(relid));
    if engine == Engine::Local {
        return search_local(&tbl, row.as_ref(), &args, k, partition.as_ref());
    }
    let row = row.unwrap_or_else(|| untracked(&display));

    let declared = declared_from(&row, display);
    let body = or_raise(plan(&declared, &args, &mut epoch_seconds));
    let namespace = or_raise(search_namespace(
        &row,
        &declared.table,
        partition.as_ref(),
        args.filter,
    ));
    if k == 0 {
        return Vec::new();
    }
    note_if_backfilling(&declared.table, &row);

    let request = SearchDocsRequest {
        index: row.index_name,
        namespace: Some(namespace),
        include_fields: Vec::new(),
        filter: body.filter,
        score_by: Value::Array(vec![body.score_by]),
        top_k: k,
    };
    let matches = backend::client_for_effective_key()
        .search_docs(&request)
        .unwrap_or_else(|err| fail(err, mode));
    matches
        .into_iter()
        .zip(1..)
        .map(|(found, rank)| (found.id, found.score, rank))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_filter_binds_the_partition_by_an_equality_at_the_top_or_inside_and() {
        let org = "org";
        let cases = [
            (json!({"org": {"$eq": 2}}), Some(json!(2))),
            (json!({"org": 2}), Some(json!(2))),
            (json!({"org": "acme"}), Some(json!("acme"))),
            (
                json!({"org": {"$eq": "a"}, "author": "ann"}),
                Some(json!("a")),
            ),
            (
                json!({"$and": [{"author": "ann"}, {"org": {"$eq": 7}}]}),
                Some(json!(7)),
            ),
            (json!({"$and": [{"$and": [{"org": 9}]}]}), Some(json!(9))),
            (json!({"$and": "nope", "org": 3}), Some(json!(3))),
            (json!({"$or": [{"org": 1}, {"org": 2}]}), None),
            (json!({"org": {"$in": [1, 2]}}), None),
            (json!({"org": {"$ne": 1}}), None),
            (json!({"org": [1]}), None),
            (json!({"org": null}), None),
            (json!({"author": "ann"}), None),
            (json!([{"org": 1}]), None),
            (json!({"$and": {"org": 1}}), None),
        ];
        for (filter, want) in cases {
            assert_eq!(filter::equality_on(&filter, org).cloned(), want, "{filter}");
        }
    }

    #[test]
    fn a_partition_operand_is_a_string_or_an_integer() {
        assert_eq!(operand_text(&json!("acme")).as_deref(), Some("acme"));
        assert_eq!(operand_text(&json!("")).as_deref(), Some(""));
        assert_eq!(operand_text(&json!(42)).as_deref(), Some("42"));
        assert_eq!(operand_text(&json!(-7)).as_deref(), Some("-7"));
        assert_eq!(
            operand_text(&json!(18_446_744_073_709_551_615u64)).as_deref(),
            Some("18446744073709551615")
        );
        for other in [
            json!(1.5),
            json!(true),
            json!(null),
            json!([1]),
            json!({"a": 1}),
        ] {
            assert_eq!(operand_text(&other), None, "{other}");
        }
    }

    fn declared() -> Declared {
        let metadata = [
            ("author", MetaKind::Text),
            ("published_at", MetaKind::Timestamptz),
            ("tags", MetaKind::TextArray),
            ("year", MetaKind::Int),
        ];
        Declared {
            table: "public.articles".to_string(),
            index: "pd-public-articles".to_string(),
            text_columns: ["body", "sku", "title"].map(String::from).to_vec(),
            metadata: metadata
                .into_iter()
                .map(|(name, kind)| (name.to_string(), kind))
                .collect(),
            vector: Some(("embedding".to_string(), 3)),
        }
    }

    #[test]
    fn the_backfilling_notice_names_the_table_and_the_progress() {
        assert_eq!(
            backfilling_notice("public.articles", 1_000, Some(2_500)),
            "pinecone.search_docs: public.articles is still backfilling: 1000 of 2500 rows \
             delivered; results are partial"
        );
        let unknown = backfilling_notice("public.articles", 7, None);
        assert!(
            unknown.contains("7 rows delivered; results are partial"),
            "{unknown}"
        );
    }

    fn epoch(_kind: MetaKind, text: &str) -> Option<f64> {
        match text {
            "2024-01-01T00:00:00Z" => Some(1_704_067_200.0),
            "2024-02-01T00:00:00Z" => Some(1_706_745_600.0),
            _ => None,
        }
    }

    fn args() -> Args<'static> {
        Args {
            query: Some("q"),
            mode: Mode::Text,
            field: None,
            must: None,
            filter: None,
            vector: None,
        }
    }

    fn run(args: &Args) -> Result<Body, SearchError> {
        plan(&declared(), args, &mut epoch)
    }

    fn ok(args: &Args) -> Body {
        run(args).unwrap_or_else(|e| panic!("{e:?}"))
    }

    fn refused(args: &Args, code: Code, parts: &[&str]) -> SearchError {
        let error = run(args).err().expect("the arguments must be refused");
        assert_eq!(error.code, code, "{error:?}");
        assert!(
            error.message.starts_with("pinecone.search_docs: "),
            "{error:?}"
        );
        for part in parts {
            assert!(error.message.contains(part), "{part:?} not in {error:?}");
        }
        error
    }

    const INVALID: Code = Code::ERRCODE_INVALID_PARAMETER_VALUE;

    #[test]
    fn a_text_query_is_one_clause_naming_every_declared_column() {
        let body = ok(&args());
        let clause = json!({"type": "text", "fields": ["body", "sku", "title"], "query": "q"});
        assert_eq!(body.score_by, clause);
        assert_eq!(body.filter, None);
    }

    #[test]
    fn field_narrows_the_text_clause_to_that_column() {
        let body = ok(&Args {
            field: Some("body"),
            ..args()
        });
        let clause = json!({"type": "text", "fields": ["body"], "query": "q"});
        assert_eq!(body.score_by, clause);
    }

    #[test]
    fn a_query_string_clause_carries_the_query_unchanged_and_no_fields() {
        let lucene = r#"title:(postgres AND "vector search"^2) -body:old"#;
        let body = ok(&Args {
            query: Some(lucene),
            mode: Mode::QueryString,
            ..args()
        });
        assert_eq!(
            body.score_by,
            json!({"type": "query_string", "query": lucene})
        );
    }

    #[test]
    fn a_vector_clause_names_the_vector_column_and_carries_the_values() {
        let values = [0.5, -0.25, 2.0];
        let body = ok(&Args {
            query: None,
            vector: Some(&values),
            ..args()
        });
        let clause = json!({"type": "dense_vector", "fields": ["embedding"], "values": values});
        assert_eq!(body.score_by, clause);
        assert_eq!(body.filter, None);
    }

    #[test]
    fn must_and_filter_are_and_ed_with_must_first_and_unwrapped_when_alone() {
        let must = json!({"body": {"$match_all": "TensorFlow"}});
        let filter = json!({"author": {"$eq": "ann"}});
        let both = Args {
            must: Some(&must),
            filter: Some(&filter),
            ..args()
        };
        assert_eq!(ok(&both).filter, Some(json!({"$and": [must, filter]})));
        let only_must = Args {
            must: Some(&must),
            ..args()
        };
        assert_eq!(ok(&only_must).filter, Some(must.clone()));
        let only_filter = Args {
            filter: Some(&filter),
            ..args()
        };
        assert_eq!(ok(&only_filter).filter, Some(filter));
    }

    #[test]
    fn a_vector_with_a_query_makes_the_query_a_match_all_requirement_on_field() {
        let values = [1.0, 0.0, 0.0];
        let with_field = Args {
            field: Some("body"),
            vector: Some(&values),
            ..args()
        };
        let body = ok(&with_field);
        assert_eq!(body.score_by["type"], json!("dense_vector"));
        assert_eq!(body.filter, Some(json!({"body": {"$match_all": "q"}})));

        let must = json!({"title": {"$match_phrase": "growling bear"}});
        let filter = json!({"year": {"$gte": 2020}});
        let all = Args {
            must: Some(&must),
            filter: Some(&filter),
            ..with_field
        };
        let expected = json!({"$and": [must, {"body": {"$match_all": "q"}}, filter]});
        assert_eq!(ok(&all).filter, Some(expected));
    }

    #[test]
    fn a_vector_with_must_filters_lexically_and_ranks_densely() {
        let must = json!({"body": {"$match_all": "TensorFlow"}});
        let values = [1.0, 0.0, 0.0];
        let body = ok(&Args {
            query: None,
            must: Some(&must),
            vector: Some(&values),
            ..args()
        });
        assert_eq!(body.score_by["type"], json!("dense_vector"));
        assert_eq!(body.filter, Some(must));
    }

    #[test]
    fn argument_combinations_that_cannot_be_one_request_are_refused() {
        let values = [1.0, 0.0, 0.0];
        let nothing = Args {
            query: None,
            ..args()
        };
        refused(&nothing, INVALID, &["nothing to rank by"]);
        let blank = Args {
            query: Some("  "),
            ..args()
        };
        refused(&blank, INVALID, &["query must not be empty"]);
        let no_field = Args {
            vector: Some(&values),
            ..args()
        };
        refused(
            &no_field,
            INVALID,
            &["vector => with query => needs field =>"],
        );
        let lucene_vector = Args {
            mode: Mode::QueryString,
            vector: Some(&values),
            ..args()
        };
        let error = refused(
            &lucene_vector,
            INVALID,
            &["query_string", "one scoring type"],
        );
        assert!(
            error
                .hint
                .is_some_and(|hint| hint.contains("mode => 'text'"))
        );
        let lucene_field = Args {
            mode: Mode::QueryString,
            field: Some("body"),
            ..args()
        };
        refused(&lucene_field, INVALID, &["field =>", "query_string"]);
    }

    #[test]
    fn field_must_be_a_declared_text_column() {
        for field in ["nope", "author", "embedding"] {
            let args = Args {
                field: Some(field),
                ..args()
            };
            let error = refused(
                &args,
                Code::ERRCODE_UNDEFINED_COLUMN,
                &[&format!("\"{field}\""), "public.articles"],
            );
            let hint = error.hint.expect("lists the text columns");
            assert!(hint.contains("body, sku, title"), "{hint}");
        }
    }

    #[test]
    fn must_takes_text_columns_with_text_match_operators_and_strings() {
        let cases = [
            (json!(["body"]), "non-empty JSON object"),
            (json!({}), "non-empty JSON object"),
            (
                json!({"author": {"$match_all": "a"}}),
                "\"author\" is not a declared text column",
            ),
            (
                json!({"nope": {"$match_all": "a"}}),
                "\"nope\" is not a declared text column",
            ),
            (json!({"body": "a"}), "needs an operator object"),
            (json!({"body": {}}), "needs an operator object"),
            (
                json!({"body": {"$eq": "a"}}),
                "operator \"$eq\" is not supported",
            ),
            (
                json!({"body": {"$match_all": 3}}),
                "needs a non-empty string",
            ),
            (
                json!({"body": {"$match_any": " "}}),
                "needs a non-empty string",
            ),
        ];
        for (must, part) in cases {
            let args = Args {
                must: Some(&must),
                ..args()
            };
            refused(&args, INVALID, &["must", part]);
        }
        let valid = json!({"body": {"$match_phrase": "a b"}, "title": {"$match_any": "x", "$match_all": "y"}});
        let args = Args {
            must: Some(&valid),
            ..args()
        };
        assert_eq!(ok(&args).filter, Some(valid));
    }

    #[test]
    fn filter_takes_metadata_columns_and_the_vector_paths_operators() {
        let accepted = [
            json!({"author": {"$eq": "ann"}}),
            json!({"author": {"$in": ["ann", "bob"]}, "year": {"$gte": 2020, "$lt": 2025}}),
            json!({"$or": [{"author": "ann"}, {"year": {"$exists": true}}]}),
            json!({"tags": {"$eq": "ml"}}),
        ];
        for filter in accepted {
            let args = Args {
                filter: Some(&filter),
                ..args()
            };
            assert_eq!(ok(&args).filter, Some(filter.clone()), "{filter}");
        }
        let refused_filters = [
            (json!([1]), "filter must be a JSON object"),
            (json!({}), "at least one key"),
            (
                json!({"body": {"$eq": "a"}}),
                "'body' is not a declared metadata column",
            ),
            (
                json!({"note": 1}),
                "'note' is not a declared metadata column",
            ),
            (
                json!({"author": {"$regex": "a"}}),
                "unknown operator '$regex'",
            ),
            (json!({"author": {"$gt": "a"}}), "range operators"),
            (json!({"year": {"$eq": 1.5}}), "not valid for an Int column"),
            (json!({"$not": {"author": "a"}}), "unknown operator '$not'"),
        ];
        for (filter, part) in refused_filters {
            let args = Args {
                filter: Some(&filter),
                ..args()
            };
            let error = refused(&args, INVALID, &["filter", part]);
            if part.contains("declared metadata column") {
                let hint = error.hint.expect("names the metadata columns");
                assert!(hint.contains("author, published_at, tags, year"), "{hint}");
                assert!(hint.contains("must =>"), "{hint}");
            }
        }
    }

    #[test]
    fn filter_instants_become_epoch_seconds_and_numbers_pass_through() {
        let filter = json!({"$and": [
            {"published_at": {"$gte": "2024-01-01T00:00:00Z", "$lt": 1_800_000_000}},
            {"$or": [{"published_at": {"$in": ["2024-02-01T00:00:00Z", 5]}},
                     {"published_at": "2024-01-01T00:00:00Z"}]},
            {"author": {"$eq": "2024-01-01T00:00:00Z"}},
        ]});
        let args = Args {
            filter: Some(&filter),
            ..args()
        };
        let typed = ok(&args).filter.expect("a filter");
        let number = |value: &Value| value.as_f64().expect("a number");
        let and = typed["$and"].as_array().expect("$and").clone();
        assert_eq!(number(&and[0]["published_at"]["$gte"]), 1_704_067_200.0);
        assert_eq!(number(&and[0]["published_at"]["$lt"]), 1_800_000_000.0);
        let in_list = &and[1]["$or"][0]["published_at"]["$in"];
        assert_eq!(number(&in_list[0]), 1_706_745_600.0);
        assert_eq!(number(&in_list[1]), 5.0);
        assert_eq!(number(&and[1]["$or"][1]["published_at"]), 1_704_067_200.0);
        assert_eq!(and[2], filter["$and"][2], "text columns are not touched");
    }

    #[test]
    fn an_instant_that_does_not_parse_names_the_value_and_the_column() {
        let filter = json!({"published_at": {"$gt": "next tuesday-ish"}});
        let args = Args {
            filter: Some(&filter),
            ..args()
        };
        refused(
            &args,
            INVALID,
            &[
                "filter",
                "\"next tuesday-ish\"",
                "timestamptz",
                "\"published_at\"",
            ],
        );
    }

    #[test]
    fn a_vector_needs_a_vector_column_the_right_size_and_finite_values() {
        let args = |values: &'static [f32]| Args {
            query: None,
            vector: Some(values),
            ..args()
        };
        let error = refused(
            &args(&[1.0, 2.0]),
            Code::ERRCODE_DATA_EXCEPTION,
            &["2 dimensions", "expects 3"],
        );
        assert!(error.message.contains("pd-public-articles"), "{error:?}");
        refused(
            &args(&[1.0, f32::NAN, 3.0]),
            INVALID,
            &["vector[2]", "finite"],
        );
        refused(&args(&[1.0, 2.0, f32::INFINITY]), INVALID, &["vector[3]"]);

        let mut no_vector = declared();
        no_vector.vector = None;
        let values = [1.0, 2.0, 3.0];
        let by_vector = Args {
            query: None,
            vector: Some(&values),
            ..self::args()
        };
        let error = plan(&no_vector, &by_vector, &mut epoch)
            .err()
            .expect("refused");
        assert_eq!(error.code, INVALID);
        assert!(
            error
                .message
                .contains("public.articles has no vector column")
        );
        assert!(error.hint.is_some_and(|hint| hint.contains("track_docs")));
    }

    #[test]
    fn k_is_between_zero_and_ten_thousand() {
        assert_eq!(check_k(0), Ok(0));
        assert_eq!(check_k(10), Ok(10));
        assert_eq!(check_k(10_000), Ok(10_000));
        for k in [10_001, 20_000, -1, i32::MIN, i32::MAX] {
            let error = check_k(k).expect_err("out of range");
            assert_eq!(error.code, INVALID);
            let message = format!("k must be between 0 and 10000 (got {k})");
            assert!(error.message.ends_with(&message), "{error:?}");
        }
    }

    #[test]
    fn mode_and_engine_accept_two_values_each() {
        assert_eq!(parse_mode("text"), Ok(Mode::Text));
        assert_eq!(parse_mode("query_string"), Ok(Mode::QueryString));
        assert_eq!(parse_engine("remote"), Ok(Engine::Remote));
        assert_eq!(parse_engine("local"), Ok(Engine::Local));
        for bad in ["", "Text", "fuzzy"] {
            assert_eq!(parse_mode(bad).expect_err(bad).code, INVALID);
        }
        for bad in ["", "Remote", "quantum"] {
            assert_eq!(parse_engine(bad).expect_err(bad).code, INVALID);
        }
    }

    #[test]
    fn a_null_vector_element_is_refused_with_its_position() {
        assert_eq!(vector_values(None), Ok(None));
        assert_eq!(
            vector_values(Some(vec![Some(1.0), Some(2.0)])),
            Ok(Some(vec![1.0, 2.0]))
        );
        let error = vector_values(Some(vec![Some(1.0), None])).expect_err("NULL element");
        assert_eq!(error.code, Code::ERRCODE_NULL_VALUE_NOT_ALLOWED);
        assert!(
            error.message.contains("vector[2] must not be NULL"),
            "{error:?}"
        );
    }

    fn api(status: u16, message: &str) -> ClientError {
        ClientError::Api {
            status,
            message: message.to_string(),
        }
    }

    fn explained(err: ClientError, mode: Mode) -> SearchError {
        explain(err, mode).expect("a mapped error")
    }

    #[test]
    fn a_rejected_query_surfaces_pinecones_message_and_a_mode_specific_hint() {
        let rejected = || ClientError::BadRequest {
            message: "unbalanced parenthesis".to_string(),
        };
        let lucene = explained(rejected(), Mode::QueryString);
        assert_eq!(lucene.code, INVALID);
        assert_eq!(
            lucene.message,
            "pinecone.search_docs: Pinecone rejected the query: unbalanced parenthesis"
        );
        assert!(
            lucene
                .hint
                .is_some_and(|hint| hint.contains("query_string"))
        );
        let text = explained(rejected(), Mode::Text);
        assert_eq!(text.code, INVALID);
        assert!(
            text.hint
                .is_some_and(|hint| hint.contains("must and filter"))
        );
    }

    #[test]
    fn a_429_is_a_quota_error_or_a_rate_error_by_its_message() {
        let quota = explained(
            api(
                429,
                "You've reached your read unit limit for the current month limit. Upgrade your plan.",
            ),
            Mode::Text,
        );
        assert_eq!(quota.code, Code::ERRCODE_CONFIGURATION_LIMIT_EXCEEDED);
        assert!(quota.message.contains("Pinecone returned 429"), "{quota:?}");
        assert!(
            quota
                .hint
                .is_some_and(|hint| hint.contains("billing period"))
        );
        let rate = explained(api(429, "Pace your query requests per second"), Mode::Text);
        assert_eq!(rate.code, Code::ERRCODE_IO_ERROR);
        assert!(rate.hint.is_some_and(|hint| hint.contains("transient")));
    }

    #[test]
    fn outages_are_io_errors_that_point_at_the_local_engine() {
        let server = explained(api(503, "unavailable"), Mode::Text);
        assert_eq!(server.code, Code::ERRCODE_IO_ERROR);
        assert!(
            server
                .message
                .contains("Pinecone returned 503: unavailable")
        );
        let hint = server.hint.expect("a hint");
        assert!(
            hint.contains("transient") && hint.contains("engine => 'local'"),
            "{hint}"
        );

        let timeout = explained(ClientError::Timeout { ms: 5000 }, Mode::Text);
        assert_eq!(timeout.code, Code::ERRCODE_IO_ERROR);
        assert!(timeout.message.contains("timed out after 5000ms"));
        assert!(
            timeout
                .hint
                .is_some_and(|hint| hint.contains("engine => 'local'"))
        );
        let transport = ClientError::Transport {
            url: "http://host".to_string(),
            reason: "refused".to_string(),
        };
        let unreachable = explained(transport, Mode::Text);
        assert_eq!(unreachable.code, Code::ERRCODE_IO_ERROR);
        assert!(
            unreachable
                .message
                .contains("could not reach http://host: refused")
        );
    }

    fn local(args: &Args) -> Result<String, SearchError> {
        check_local(args).map(str::to_string)
    }

    const UNSUPPORTED: Code = Code::ERRCODE_FEATURE_NOT_SUPPORTED;

    #[test]
    fn the_local_engine_takes_a_plain_query_and_optionally_field() {
        assert_eq!(local(&args()), Ok("q".to_string()));
        let narrowed = Args {
            field: Some("body"),
            ..args()
        };
        assert_eq!(local(&narrowed), Ok("q".to_string()));
    }

    #[test]
    fn the_local_engine_refuses_query_string_must_filter_and_vector() {
        let (must, filter) = (json!({"body": {"$match_all": "a"}}), json!({"year": 1}));
        let vector = [1.0, 0.0, 0.0];
        let cases = [
            (
                Args {
                    mode: Mode::QueryString,
                    ..args()
                },
                "query_string",
                "mode => 'text'",
            ),
            (
                Args {
                    must: Some(&must),
                    ..args()
                },
                "must/filter",
                "WHERE",
            ),
            (
                Args {
                    filter: Some(&filter),
                    ..args()
                },
                "must/filter",
                "WHERE",
            ),
            (
                Args {
                    vector: Some(&vector),
                    field: Some("body"),
                    ..args()
                },
                "vector",
                "ORDER BY",
            ),
        ];
        for (args, part, hint) in cases {
            let error = local(&args).expect_err("the local engine cannot do this");
            assert_eq!(error.code, UNSUPPORTED, "{error:?}");
            assert!(
                error.message.starts_with("pinecone.search_docs: "),
                "{error:?}"
            );
            assert!(error.message.contains(part), "{error:?}");
            assert!(error.message.contains("local engine"), "{error:?}");
            assert!(error.hint.is_some_and(|h| h.contains(hint)), "{part}");
        }
    }

    #[test]
    fn the_local_engine_needs_a_non_blank_query() {
        for (query, part) in [
            (None, "nothing to search"),
            (Some(""), "query must not be empty"),
            (Some("  \t"), "query must not be empty"),
        ] {
            let error = local(&Args { query, ..args() }).expect_err("no query");
            assert_eq!(error.code, INVALID, "{error:?}");
            assert!(error.message.contains(part), "{error:?}");
        }
    }

    #[test]
    fn an_undeclared_field_lists_the_declared_columns() {
        let declared = ["body", "title"].map(String::from);
        let error = undeclared_field("public.articles", "note", &declared);
        assert_eq!(error.code, Code::ERRCODE_UNDEFINED_COLUMN);
        assert_eq!(
            error.message,
            "pinecone.search_docs: field \"note\" is not a declared text column of \
             public.articles"
        );
        assert_eq!(
            error.hint.as_deref(),
            Some("declared text columns: body, title")
        );
    }

    #[test]
    fn auth_and_unexpected_statuses_keep_the_generic_messages() {
        assert!(matches!(
            explain(ClientError::Auth, Mode::Text),
            Err(ClientError::Auth)
        ));
        assert!(matches!(
            explain(api(404, "gone"), Mode::Text),
            Err(ClientError::Api { status: 404, .. })
        ));
    }
}
