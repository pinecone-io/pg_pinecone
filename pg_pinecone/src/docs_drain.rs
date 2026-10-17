//! The document lane's drain step: claim a batch of queue rows, fold them into one final state
//! per id (last op wins, `doc_delete_all` is a barrier), read the committed rows from the table
//! under a fresh snapshot, and deliver deletes and byte-packed upserts to Pinecone. A request
//! Pinecone rejects is bisected down to the offending documents, which become dead letters; only
//! lane-level failures count towards parking. Queue rows are deleted as soon as everything they
//! own is delivered, so a failure half way through a big batch never resends the delivered part.
//!
//! A table with a partition column keeps one namespace per value (`prefix:value`). An upsert
//! goes where the live row says (its partition column is read with the document), a delete where
//! its queue entry says (`{"id", "p"}`: the row is gone or has moved, so the row cannot tell), and
//! a `doc_delete_all` empties every namespace the registry lists. Each namespace is registered
//! (`pinecone.partitions`) when its first document is accepted.

use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
    panic::AssertUnwindSafe,
    time::{Duration, Instant},
};

use pg_pinecone_client::{Client, DeleteDocsRequest, DocDeleteSelector, Error as ClientError};
use pgrx::{JsonB, datum::DatumWithOid, pg_sys, prelude::*, spi::Spi};
use serde_json::{Map, Value};

use crate::{
    backend::{self, DrainCredentials},
    catalog, docs_backfill,
    docs_guard::{self, Caught},
    docs_outbox::{self, DocDrainOutcome, DocOp, DocOutboxRow},
    docs_registry::{self, DocRegistryRow},
    docs_schema::{
        self, MAX_FIELD_BYTES, MAX_FIELD_TOKENS, OnOversize, TRUNCATED_FIELD, exceeds_field_limits,
        oversize_message, text_size, truncate_text,
    },
    partition::{self, Lane as PartitionLane, PartitionKind},
    partition_lane, partition_ops, partition_registry, secret,
};

/// First key of the two-int4 advisory lock a drain holds per table (`b"PVDC"`). The vector lane
/// takes a single-bigint lock instead, so the two lanes of one table never block each other.
pub(crate) const DOC_LOCK_MAGIC: i32 = 0x5056_4443;
/// Per request: Pinecone allows 1,000 documents and 2 MB; the client budgets 1.5 MB.
const PACK_MAX_DOCS: usize = 1_000;
const PACK_MAX_BYTES: usize = 1_500_000;
/// Ids planned per read; the rows read together are bounded by [`SLICE_BYTES`].
const READ_SLICE: usize = 1_000;
/// What rows read together may weigh (estimated from column sizes before reading), so big rows
/// cannot take the worker's memory; a bigger row is read alone.
const SLICE_BYTES: u64 = 16 * 1024 * 1024;
/// How long a batch runs before handing the rest to the next round, so a rate-limited one cannot
/// hold the lane and `untrack_docs` for minutes. Checked between slices, after the first.
pub(crate) const BATCH_BUDGET: Duration = Duration::from_secs(10);

/// The time one batch has left.
pub(crate) struct Budget {
    started: Instant,
    limit: Duration,
    worked: Cell<bool>,
}

impl Budget {
    pub(crate) fn new(limit: Duration) -> Self {
        Self {
            started: Instant::now(),
            limit,
            worked: Cell::new(false),
        }
    }

    /// Notes that a unit of work went out: only then may the budget end the batch, so every step
    /// makes progress however slow Pinecone is.
    fn did_work(&self) {
        self.worked.set(true);
    }

    fn spent(&self) -> bool {
        self.worked.get() && self.started.elapsed() >= self.limit
    }
}

/// Test seams: what a unit test needs to see from inside a drain step. Compiled out of the
/// extension itself.
#[cfg(feature = "pg_test")]
pub(crate) mod hooks {
    use std::cell::{Cell, RefCell};

    thread_local! {
        static BEFORE_REQUEST: RefCell<Option<Box<dyn Fn()>>> = const { RefCell::new(None) };
        static TABLE_BUSY: Cell<bool> = const { Cell::new(false) };
    }

    /// Runs `hook` before every request to Pinecone.
    pub(crate) fn set_before_request(hook: Option<Box<dyn Fn()>>) {
        BEFORE_REQUEST.with(|slot| *slot.borrow_mut() = hook);
    }

    /// Makes every probe of the table lock fail, as if DDL held it.
    pub(crate) fn set_table_busy(busy: bool) {
        TABLE_BUSY.with(|flag| flag.set(busy));
    }

    pub(crate) fn before_request() {
        BEFORE_REQUEST.with(|slot| {
            if let Some(hook) = slot.borrow().as_ref() {
                hook();
            }
        });
    }

    pub(crate) fn table_busy() -> bool {
        TABLE_BUSY.with(Cell::get)
    }
}

#[cfg(not(feature = "pg_test"))]
mod hooks {
    pub(super) fn before_request() {}

    pub(super) fn table_busy() -> bool {
        false
    }
}

pub(crate) const NO_STORED_KEY_ERROR: &str =
    "no stored API key — run pinecone.set_api_key(...) before the worker can sync documents";

/// One id of a claimed row; a `{"id", "p"}` delete entry also names the partition the row leaves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Item {
    pub(crate) id: String,
    pub(crate) partition: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Claimed {
    pub(crate) row: i64,
    pub(crate) op: DocOp,
    pub(crate) items: Vec<Item>,
}

/// The entries of a `doc_upsert`/`doc_delete` payload: bare id strings and `{"id", "p"}` objects
/// (what a table with a partition column queues for its deletes; both can share one queue);
/// anything else is ignored.
pub(crate) fn payload_items(payload: Option<&Value>) -> Vec<Item> {
    let Some(Value::Array(entries)) = payload else {
        return Vec::new();
    };
    let item = |entry: &Value| match entry {
        Value::String(id) => Some(Item {
            id: id.clone(),
            partition: None,
        }),
        Value::Object(object) => Some(Item {
            id: object.get("id")?.as_str()?.to_string(),
            partition: match object.get("p") {
                Some(Value::String(partition)) => Some(partition.clone()),
                Some(Value::Number(number)) => Some(number.to_string()),
                _ => None,
            },
        }),
        _ => None,
    };
    entries.iter().filter_map(item).collect()
}

fn claimed_rows(rows: &[DocOutboxRow]) -> Vec<Claimed> {
    rows.iter()
        .map(|row| Claimed {
            row: row.id,
            op: row.op,
            items: payload_items(row.payload.as_ref().map(|json| &json.0)),
        })
        .collect()
}

/// An id to upsert (the row's own partition decides the namespace) and the last row that said so.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Upsert {
    pub(crate) id: String,
    pub(crate) row: i64,
}

/// An id to delete from one namespace: the table's own (`partition` is `None`) or a partition's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Delete {
    pub(crate) id: String,
    pub(crate) partition: Option<String>,
    pub(crate) row: i64,
}

impl Delete {
    fn key(&self) -> (Option<String>, String) {
        (self.partition.clone(), self.id.clone())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    /// The row of the last `doc_delete_all`: Pinecone's namespace is emptied before any id.
    pub(crate) delete_all: Option<i64>,
    /// Ids whose last operation is an upsert, in the order they were first seen.
    pub(crate) upserts: Vec<Upsert>,
    /// The deletes still owed, each namespace and id once, ids in the order they were first seen.
    pub(crate) deletes: Vec<Delete>,
}

/// What the claimed rows left of one id.
#[derive(Default)]
struct IdState {
    /// The row of the last upsert, unless a delete came after it.
    upsert: Option<i64>,
    /// The deletes owed, by partition (`None`: the table's own namespace), with their last row.
    deletes: Vec<(Option<String>, i64)>,
}

/// Folds claimed rows, oldest first, into the final state of each id. Without a partition column
/// the last operation wins: a delete discards an earlier upsert and an upsert an earlier delete.
/// With one, a delete names the partition it leaves and so survives a later upsert, which goes to
/// the row's own partition (a move is both); only a delete after the upsert cancels it, since the
/// row is gone (a bare delete names no partition: it empties the table's own namespace and says
/// nothing about the row). A `doc_delete_all` discards everything before it.
pub(crate) fn fold(rows: &[Claimed], partitioned: bool) -> Plan {
    let mut order: Vec<&str> = Vec::new();
    let mut states: HashMap<&str, IdState> = HashMap::new();
    let mut delete_all = None;
    for claimed in rows {
        if claimed.op == DocOp::DeleteAll {
            order.clear();
            states.clear();
            delete_all = Some(claimed.row);
            continue;
        }
        if !matches!(claimed.op, DocOp::Upsert | DocOp::Delete) {
            continue;
        }
        for item in &claimed.items {
            let state = states.entry(item.id.as_str()).or_insert_with(|| {
                order.push(item.id.as_str());
                IdState::default()
            });
            match claimed.op {
                DocOp::Upsert => {
                    state.upsert = Some(claimed.row);
                    if !partitioned {
                        state.deletes.clear();
                    }
                }
                _ => {
                    let partition = item.partition.clone().filter(|_| partitioned);
                    if partition.is_some() || !partitioned {
                        state.upsert = None;
                    }
                    match state.deletes.iter_mut().find(|(p, _)| *p == partition) {
                        Some(owed) => owed.1 = claimed.row,
                        None => state.deletes.push((partition, claimed.row)),
                    }
                }
            }
        }
    }
    let (mut upserts, mut deletes) = (Vec::new(), Vec::new());
    for id in order {
        let state = &states[id];
        if let Some(row) = state.upsert {
            upserts.push(Upsert {
                id: id.to_string(),
                row,
            });
        }
        for (partition, row) in &state.deletes {
            deletes.push(Delete {
                id: id.to_string(),
                partition: partition.clone(),
                row: *row,
            });
        }
    }
    Plan {
        delete_all,
        upserts,
        deletes,
    }
}

#[derive(Debug, Default)]
pub(crate) struct Progress {
    upserted: HashSet<String>,
    deleted: HashSet<(Option<String>, String)>,
    pub(crate) delete_all_done: bool,
}

impl Progress {
    /// Ids whose upsert is settled: sent, dead-lettered, or the row is gone.
    pub(crate) fn upserts_done(&mut self, ids: impl IntoIterator<Item = String>) {
        self.upserted.extend(ids);
    }

    /// Deletes (partition, id) that were sent or are not owed any more.
    fn deletes_done(&mut self, deletes: impl IntoIterator<Item = (Option<String>, String)>) {
        self.deleted.extend(deletes);
    }
}

/// The claimed rows whose work is finished: a row is complete when every id it mentions last is
/// delivered (a mention superseded by a later row, or discarded by a barrier, counts as done: the
/// row that supersedes it stays queued until its own work is), and a `doc_delete_all` row when its
/// request went through. The backfill job is never complete here.
pub(crate) fn complete_rows(rows: &[Claimed], plan: &Plan, progress: &Progress) -> Vec<i64> {
    let mut pending: HashSet<i64> = HashSet::new();
    for upsert in &plan.upserts {
        if !progress.upserted.contains(&upsert.id) {
            pending.insert(upsert.row);
        }
    }
    for delete in &plan.deletes {
        if !progress.deleted.contains(&delete.key()) {
            pending.insert(delete.row);
        }
    }
    if let Some(row) = plan.delete_all.filter(|_| !progress.delete_all_done) {
        pending.insert(row);
    }
    rows.iter()
        .filter(|claimed| !matches!(claimed.op, DocOp::Backfill | DocOp::Failed))
        .filter(|claimed| !pending.contains(&claimed.row))
        .map(|claimed| claimed.row)
        .collect()
}

/// A document with its serialised size, measured as the client does, and the canonical text of
/// its row's partition (`None` without a partition column).
#[derive(Debug, Clone)]
pub(crate) struct Doc {
    pub(crate) id: String,
    pub(crate) value: Value,
    pub(crate) bytes: usize,
    pub(crate) partition: Option<String>,
}

impl Doc {
    pub(crate) fn new(id: &str, value: Value) -> Self {
        let bytes = value.to_string().len();
        Self {
            id: id.to_string(),
            value,
            bytes,
            partition: None,
        }
    }

    pub(crate) fn in_partition(mut self, partition: Option<String>) -> Self {
        self.partition = partition;
        self
    }
}

/// Splits `docs` into requests of at most `max_docs` documents and `max_bytes` bytes; a document
/// over the byte budget travels alone.
pub(crate) fn pack(docs: Vec<Doc>, max_docs: usize, max_bytes: usize) -> Vec<Vec<Doc>> {
    let mut requests = Vec::new();
    let (mut current, mut bytes) = (Vec::new(), 0usize);
    for doc in docs {
        if !current.is_empty() && (current.len() >= max_docs || bytes + doc.bytes > max_bytes) {
            requests.push(std::mem::take(&mut current));
            bytes = 0;
        }
        bytes += doc.bytes;
        current.push(doc);
    }
    if !current.is_empty() {
        requests.push(current);
    }
    requests
}

/// Documents refused one after the other, with no request accepted in between, make a cause
/// outside the documents (the index schema drifted, the declaration no longer fits) likelier
/// than a hundred bad rows.
const SYSTEMIC_STREAK: usize = 100;
/// A step in which Pinecone accepts nothing and refuses at least this many documents has the
/// same cause; below it, bad rows are the likelier explanation and stay dead letters.
const SYSTEMIC_REFUSED: usize = 10;

#[derive(Debug, Default)]
pub(crate) struct UpsertOutcome {
    pub(crate) delivered: Vec<String>,
    /// `(id, Pinecone's message)` of documents it refused one by one.
    pub(crate) dead: Vec<(String, String)>,
    /// Documents refused since the last request Pinecone accepted.
    streak: usize,
    /// Documents accepted and refused over every call that shares this outcome.
    accepted: usize,
    refused: usize,
    /// The last message Pinecone gave for a refused document.
    last_message: String,
    /// The circuit breaker opened: the cause is not the documents, so `dead` is not recorded.
    pub(crate) tripped: bool,
}

impl UpsertOutcome {
    fn systemic(&mut self, what: String) -> ClientError {
        self.tripped = true;
        ClientError::BadRequest {
            message: format!(
                "Pinecone refused {what} — the index schema or the declaration has probably \
                 drifted: {}",
                self.last_message
            ),
        }
    }

    /// `Err` if the step as a whole was refused: nothing accepted, [`SYSTEMIC_REFUSED`] or more
    /// documents refused.
    pub(crate) fn check_all_refused(&mut self) -> Result<(), ClientError> {
        if self.accepted > 0 || self.refused < SYSTEMIC_REFUSED {
            return Ok(());
        }
        Err(self.systemic(format!("all {} documents of the batch", self.refused)))
    }
}

/// Rows the table could not render, counted like Pinecone's refusals (see [`UpsertOutcome`]): a
/// hundred in a row, or ten in a step in which nothing else could be read, say the cause is not
/// the rows.
#[derive(Debug, Default)]
pub(crate) struct Unreadable {
    streak: usize,
    total: usize,
    readable: usize,
    last_message: String,
}

impl Unreadable {
    /// `rows` were read: a run of failures ends.
    pub(crate) fn readable(&mut self, rows: usize) {
        if rows > 0 {
            self.streak = 0;
            self.readable += rows;
        }
    }

    /// One more row the table could not render; `Err` once that is systemic.
    pub(crate) fn refuse(&mut self, message: &str) -> Result<(), ClientError> {
        self.total += 1;
        self.streak += 1;
        self.last_message = message.to_string();
        match self.streak >= SYSTEMIC_STREAK {
            true => Err(self.systemic(format!("{} documents in a row", self.streak))),
            false => Ok(()),
        }
    }

    pub(crate) fn verdict(&self) -> Result<(), ClientError> {
        if self.readable > 0 || self.total < SYSTEMIC_REFUSED {
            return Ok(());
        }
        Err(self.systemic(format!("all {} documents of the batch", self.total)))
    }

    fn systemic(&self, what: String) -> ClientError {
        ClientError::BadRequest {
            message: format!(
                "the table could not render {what} — the SQL that reads its rows is failing, not \
                 the documents: {}",
                self.last_message
            ),
        }
    }
}

/// What a dead letter says about a row the table could not render.
fn unreadable_message(caught: &Caught) -> String {
    format!(
        "the row could not be read: {}",
        shorten(&caught.message, MAX_ERROR_TEXT)
    )
}

/// Sends `docs`; if Pinecone rejects the request (400), halves it until the culprits stand alone
/// and records them as dead letters. Anything else (auth, 429, 5xx, transport) aborts with the
/// error, keeping what was already delivered in `outcome`. Where the refusals look systemic
/// (see [`SYSTEMIC_STREAK`]) it stops and returns one lane error instead.
pub(crate) fn upsert_bisecting(
    send: &mut dyn FnMut(&[Doc]) -> Result<(), ClientError>,
    docs: &[Doc],
    outcome: &mut UpsertOutcome,
) -> Result<(), ClientError> {
    match send(docs) {
        Ok(()) => {
            outcome
                .delivered
                .extend(docs.iter().map(|doc| doc.id.clone()));
            outcome.accepted += docs.len();
            outcome.streak = 0;
            Ok(())
        }
        Err(ClientError::BadRequest { message })
            if message.contains(partition_lane::NEW_NAMESPACE_MARK) =>
        {
            Err(ClientError::BadRequest { message })
        }
        Err(ClientError::BadRequest { message }) => match docs {
            [single] => {
                outcome.refused += 1;
                outcome.streak += 1;
                outcome.last_message.clone_from(&message);
                outcome.dead.push((single.id.clone(), message));
                if outcome.streak >= SYSTEMIC_STREAK {
                    return Err(outcome.systemic(format!("{} documents in a row", outcome.streak)));
                }
                Ok(())
            }
            _ => {
                let (left, right) = docs.split_at(docs.len() / 2);
                upsert_bisecting(send, left, outcome)?;
                upsert_bisecting(send, right, outcome)
            }
        },
        Err(other) => Err(other),
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TextField {
    pub(crate) name: String,
    pub(crate) on_oversize: OnOversize,
}

/// The document for one row: `_id`, every non-NULL text column (cut under `truncate`, which
/// marks the document with `pg_pinecone_truncated`) and the already typed metadata and vector
/// fields. `Err` carries the dead-letter message of a value the `error` policy refuses, which
/// can only have reached the table before the trigger guarded it (or past it).
pub(crate) fn build_document(
    id: &str,
    fields: &[TextField],
    texts: &[Option<String>],
    meta: Option<Value>,
) -> Result<Value, String> {
    let mut doc = Map::new();
    doc.insert("_id".to_string(), Value::String(id.to_string()));
    let mut truncated = false;
    for (field, text) in fields.iter().zip(texts) {
        let Some(text) = text else { continue };
        let value = match field.on_oversize {
            OnOversize::Truncate => {
                let (cut, was_cut) = truncate_text(text, MAX_FIELD_TOKENS, MAX_FIELD_BYTES);
                truncated |= was_cut;
                cut
            }
            OnOversize::Error => {
                let size = text_size(text);
                if exceeds_field_limits(&size) {
                    return Err(oversize_message(&field.name, &size));
                }
                text.clone()
            }
        };
        doc.insert(field.name.clone(), Value::String(value));
    }
    if let Some(Value::Object(meta)) = meta {
        doc.extend(meta);
    }
    if truncated {
        doc.insert(TRUNCATED_FIELD.to_string(), Value::Bool(true));
    }
    Ok(Value::Object(doc))
}

/// The partition column of a table that has one, as its reads use it.
struct PartitionLayout {
    name: String,
    /// The quoted column.
    column: String,
    /// The SQL type a cursor value is cast to.
    cast: &'static str,
    /// The table's own namespace, which the partition's namespaces extend.
    prefix: String,
}

/// How the table is read: quoted identifiers and the SQL that renders a row as a document.
pub(crate) struct Layout {
    table: String,
    id_column: String,
    id_expr: String,
    id_array: &'static str,
    fields: Vec<TextField>,
    meta_expr: String,
    /// SQL for roughly what a row's document weighs, from the column sizes alone.
    size_expr: String,
    partition: Option<PartitionLayout>,
}

impl Layout {
    /// `, <partition column>::text AS part` when the table has one: the text a namespace is made of.
    fn partition_select(&self) -> String {
        match &self.partition {
            Some(partition) => format!(", {}::text AS part", partition.column),
            None => String::new(),
        }
    }
}

/// Pairs per `jsonb_build_object` call, which takes at most 100 arguments.
const PAIRS_PER_OBJECT: usize = 40;

fn drift(display: &str, message: impl std::fmt::Display) -> ClientError {
    ClientError::BadRequest {
        message: format!("{display}: {message} — untrack and re-track the table"),
    }
}

/// Checks the declared columns against the live table before any SQL is spliced from them: a
/// dropped or retyped column is a lane error (parks after `outbox_max_attempts`), never a
/// Postgres ERROR that would abort the whole worker round.
pub(crate) fn build_layout(
    relid: pg_sys::Oid,
    row: &DocRegistryRow,
) -> Result<Layout, ClientError> {
    use pgrx::spi::{quote_identifier, quote_literal, quote_qualified_identifier};
    let (schema, name) = catalog::resolve_table_name(relid)
        .ok_or_else(|| drift(&format!("relid {relid}"), "the table no longer exists"))?;
    let display = format!("{schema}.{name}");
    let columns = catalog::all_columns(relid);
    let type_of = |column: &str| {
        columns
            .iter()
            .find(|c| c.name == column)
            .map(|c| c.type_name.as_str())
    };
    let missing =
        |what: &str, column: &str| drift(&display, format!("{what} \"{column}\" is gone"));
    let id_type = type_of(&row.id_column).ok_or_else(|| missing("id column", &row.id_column))?;
    let id_array = catalog::DOC_ID_COLUMN_TYPES
        .iter()
        .copied()
        .find(|supported| *supported == id_type)
        .ok_or_else(|| {
            drift(
                &display,
                format!("id column \"{}\" has type {id_type}", row.id_column),
            )
        })?;
    let id_column = quote_identifier(&row.id_column);
    let id_expr = match id_array {
        "bpchar" => format!("pg_catalog.rtrim({id_column}::text)"),
        _ => format!("{id_column}::text"),
    };

    let declared = docs_schema::parse_text_columns(&row.text_columns)
        .map_err(|e| drift(&display, e.message))?;
    let mut fields = Vec::new();
    for column in declared {
        match type_of(&column.name) {
            Some("text" | "varchar" | "bpchar") => fields.push(TextField {
                name: column.name,
                on_oversize: column.on_oversize,
            }),
            Some(other) => {
                return Err(drift(
                    &display,
                    format!("\"{}\" has type {other}", column.name),
                ));
            }
            None => return Err(missing("text column", &column.name)),
        }
    }

    let mut pairs = Vec::new();
    for name in &row.metadata_columns {
        let ty = type_of(name).ok_or_else(|| missing("metadata column", name))?;
        let kind = docs_schema::metadata_kind(ty)
            .ok_or_else(|| drift(&display, format!("\"{name}\" has type {ty}")))?;
        let value = kind.value_expr(&quote_identifier(name));
        pairs.push(format!("{}, {value}", quote_literal(name)));
    }
    if let Some(vector) = &row.vector_column {
        match type_of(vector) {
            Some("vector" | "halfvec") => pairs.push(format!(
                "{}, pg_catalog.to_jsonb({}::real[])",
                quote_literal(vector),
                quote_identifier(vector)
            )),
            Some(other) => {
                return Err(drift(&display, format!("\"{vector}\" has type {other}")));
            }
            None => return Err(missing("vector column", vector)),
        }
    }
    let partition = match &row.partition {
        None => None,
        Some(spec) => {
            let ty =
                type_of(&spec.column).ok_or_else(|| missing("partition column", &spec.column))?;
            if PartitionKind::from_typname(ty) != Some(spec.kind) {
                return Err(drift(
                    &display,
                    format!("partition column \"{}\" has type {ty}", spec.column),
                ));
            }
            Some(PartitionLayout {
                name: spec.column.clone(),
                column: quote_identifier(&spec.column).to_string(),
                cast: spec.kind.as_str(),
                prefix: row.namespace.clone(),
            })
        }
    };
    let objects: Vec<String> = pairs
        .chunks(PAIRS_PER_OBJECT)
        .map(|chunk| format!("pg_catalog.jsonb_build_object({})", chunk.join(", ")))
        .collect();
    let meta_expr = match objects.is_empty() {
        true => "NULL::pg_catalog.jsonb".to_string(),
        false => format!("pg_catalog.jsonb_strip_nulls({})", objects.join(" || ")),
    };
    // Text by its raw length (`octet_length` does not detoast), everything else by its stored
    // size, a vector four times over for its text form in the document.
    let mut sizes: Vec<String> = fields
        .iter()
        .map(|field| {
            let column = quote_identifier(&field.name);
            format!("COALESCE(pg_catalog.octet_length({column}), 0)")
        })
        .collect();
    sizes.extend(row.metadata_columns.iter().map(|name| {
        format!(
            "COALESCE(pg_catalog.pg_column_size({}), 0)",
            quote_identifier(name)
        )
    }));
    if let Some(vector) = &row.vector_column {
        let column = quote_identifier(vector);
        sizes.push(format!(
            "4 * COALESCE(pg_catalog.pg_column_size({column}), 0)"
        ));
    }
    Ok(Layout {
        table: quote_qualified_identifier(&schema, &name),
        id_column,
        id_expr,
        id_array,
        fields,
        meta_expr,
        size_expr: sizes.join(" + "),
        partition,
    })
}

/// Splits `ids` (in their order) into groups whose estimated sizes add up to at most
/// `max_bytes`; a group never ends up empty, so an id bigger than that is read alone. An id with
/// no known size (its row is gone) weighs nothing.
fn group_ids(ids: &[String], sizes: &HashMap<String, u64>, max_bytes: u64) -> Vec<Vec<String>> {
    let mut groups: Vec<Vec<String>> = Vec::new();
    let mut weight = 0u64;
    for id in ids {
        let size = sizes.get(id).copied().unwrap_or(0);
        match groups.last_mut() {
            Some(group) if weight + size <= max_bytes => {
                group.push(id.clone());
                weight += size;
            }
            _ => {
                groups.push(vec![id.clone()]);
                weight = size;
            }
        }
    }
    groups
}

/// What one read of the table produced for a slice of ids.
#[derive(Default)]
struct Read {
    docs: Vec<Doc>,
    /// Upserts whose row is gone: they become deletes.
    missing: Vec<String>,
    /// Rows the `error` policy refuses: `(id, message)`, dead-lettered without a request.
    rejected: Vec<(String, String)>,
}

impl Read {
    fn absorb(&mut self, other: Read) {
        self.docs.extend(other.docs);
        self.missing.extend(other.missing);
        self.rejected.extend(other.rejected);
    }
}

/// A row as the table gave it: its id, the declared text columns, the typed metadata and the text
/// of its partition value (`None` without a partition column, and for a NULL one).
struct RawRow {
    id: String,
    texts: Vec<Option<String>>,
    meta: Option<Value>,
    partition: Option<String>,
}

/// The partition a row's document goes to: its canonical text, or `None` without a partition
/// column. A row the namespace cannot be made for is refused with the reason, which becomes its
/// dead letter.
fn route(layout: &Layout, partition: Option<String>) -> Result<Option<String>, String> {
    let Some(spec) = &layout.partition else {
        return Ok(None);
    };
    let Some(value) = partition else {
        return Err(format!(
            "the row has no value in its partition column \"{}\", so no namespace",
            spec.name
        ));
    };
    let namespace = partition::namespace_for(&spec.prefix, &value);
    match partition::fits_pinecone(&namespace) {
        Ok(()) => Ok(Some(value)),
        Err(error) => Err(format!(
            "the value of partition column \"{}\" makes an unusable namespace: {error}",
            spec.name
        )),
    }
}

/// Runs `SELECT <the document columns> FROM <table> <tail>` under a fresh snapshot (`update`
/// keeps SPI mutable, which takes a new READ COMMITTED snapshot instead of the transaction's
/// first one, so the newest committed text is what gets sent, not what the enqueuing statement
/// saw).
fn query_rows(layout: &Layout, tail: &str, args: &[DatumWithOid]) -> Vec<RawRow> {
    let texts: String = (0..layout.fields.len())
        .map(|at| {
            let column = pgrx::spi::quote_identifier(&layout.fields[at].name);
            format!(", {column}::text AS t{at}")
        })
        .collect();
    let sql = format!(
        "SELECT {} AS doc_id{texts}, {} AS meta{} FROM {} {tail}",
        layout.id_expr,
        layout.meta_expr,
        layout.partition_select(),
        layout.table
    );
    Spi::connect_mut(|client| {
        client
            .update(&sql, None, args)
            .expect("pg_pinecone: document read failed")
            .map(|row| RawRow {
                partition: layout.partition.as_ref().and_then(|_| {
                    row.get_by_name::<String, _>("part")
                        .expect("pg_pinecone: document partition read failed")
                }),
                id: row
                    .get_by_name::<String, _>("doc_id")
                    .expect("pg_pinecone: document id read failed")
                    .expect("pg_pinecone: document id unexpectedly NULL"),
                texts: (0..layout.fields.len())
                    .map(|at| {
                        row.get_by_name::<String, _>(&format!("t{at}"))
                            .expect("pg_pinecone: document text read failed")
                    })
                    .collect(),
                meta: row
                    .get_by_name::<JsonB, _>("meta")
                    .expect("pg_pinecone: document metadata read failed")
                    .map(|json| json.0),
            })
            .collect()
    })
}

/// Reads the committed rows of `ids` (see [`query_rows`] for the snapshot).
fn read_documents(layout: &Layout, ids: &[String]) -> Read {
    let tail = format!(
        "WHERE {} = ANY($1::text[]::{}[])",
        layout.id_column, layout.id_array
    );
    let rows = query_rows(layout, &tail, &[ids.to_vec().into()]);
    let mut read = Read::default();
    let mut found = HashSet::new();
    for row in rows {
        let document = route(layout, row.partition).and_then(|partition| {
            let value = build_document(&row.id, &layout.fields, &row.texts, row.meta)?;
            Ok(Doc::new(&row.id, value).in_partition(partition))
        });
        match document {
            Ok(doc) => read.docs.push(doc),
            Err(message) => read.rejected.push((row.id.clone(), message)),
        }
        found.insert(row.id);
    }
    read.missing = ids
        .iter()
        .filter(|id| !found.contains(*id))
        .cloned()
        .collect();
    read
}

/// A position in the backfill's order: the id, and with a partition column the canonical text of
/// the row's partition, since the backfill reads in `(partition, id)` order to keep a page mostly
/// inside one namespace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Cursor {
    pub(crate) partition: Option<String>,
    pub(crate) id: String,
}

/// One page of the backfill: the documents of the next rows after a cursor.
#[derive(Debug, Default)]
pub(crate) struct Page {
    pub(crate) docs: Vec<Doc>,
    /// Rows the `error` policy refuses: `(id, message)`, dead-lettered without a request.
    pub(crate) rejected: Vec<(String, String)>,
    /// Rows the table could not render: `(id, message)`, dead letters unless the step is systemic.
    pub(crate) unreadable: Vec<(String, String)>,
    /// Rows the cursor moves past: the documents plus the refused ones.
    pub(crate) rows: usize,
    /// The last of those rows, the next cursor; `None` for an empty page.
    pub(crate) last: Option<Cursor>,
}

/// A row of the next page, built into its document (`Err` = refused, with the reason).
pub(crate) struct Built {
    pub(crate) at: Cursor,
    pub(crate) document: Result<Doc, String>,
}

/// Cuts a page from rows already turned into documents at the byte budget: the row that would
/// cross it, and everything after, is left for the next page, but a page always takes at least
/// its first document.
pub(crate) fn assemble_page(built: impl IntoIterator<Item = Built>, max_bytes: usize) -> Page {
    let (mut page, mut bytes) = (Page::default(), 0usize);
    for Built { at, document } in built {
        match document {
            Ok(doc) => {
                if !page.docs.is_empty() && bytes + doc.bytes > max_bytes {
                    break;
                }
                bytes += doc.bytes;
                page.docs.push(doc);
            }
            Err(message) => page.rejected.push((at.id.clone(), message)),
        }
        page.rows += 1;
        page.last = Some(at);
    }
    page
}

/// A row as the planning queries see it: its id, the text of its partition value (with a
/// partition column) and what its document roughly weighs.
struct Weighed {
    at: Cursor,
    size: u64,
}

/// The ids and estimated sizes of the rows `SELECT ... WHERE <tail>` finds: what the planning
/// queries ask (`tail` may order and limit).
fn query_sizes(layout: &Layout, tail: &str, args: &[DatumWithOid]) -> Vec<Weighed> {
    let sql = format!(
        "SELECT {} AS doc_id, ({})::bigint AS sz{} FROM {} {tail}",
        layout.id_expr,
        layout.size_expr,
        layout.partition_select(),
        layout.table
    );
    // Read-write SPI, like [`query_rows`]: a fresh snapshot, not the one the transaction began with.
    Spi::connect_mut(|client| {
        client
            .update(&sql, None, args)
            .expect("pg_pinecone: document size query failed")
            .map(|row| {
                let id = row
                    .get_by_name::<String, _>("doc_id")
                    .expect("pg_pinecone: document id read failed")
                    .expect("pg_pinecone: document id unexpectedly NULL");
                let size = row
                    .get_by_name::<i64, _>("sz")
                    .expect("pg_pinecone: document size read failed")
                    .unwrap_or(0);
                let partition = layout.partition.as_ref().and_then(|_| {
                    row.get_by_name::<String, _>("part")
                        .expect("pg_pinecone: document partition read failed")
                });
                Weighed {
                    at: Cursor { partition, id },
                    size: size.max(0) as u64,
                }
            })
            .collect()
    })
}

/// Splits `ids` into groups of rows that may be read together (see [`SLICE_BYTES`]).
fn plan_groups(layout: &Layout, ids: &[String]) -> Vec<Vec<String>> {
    let tail = format!(
        "WHERE {} = ANY($1::text[]::{}[])",
        layout.id_column, layout.id_array
    );
    let sizes = query_sizes(layout, &tail, &[ids.to_vec().into()]);
    let sizes: HashMap<String, u64> = sizes.into_iter().map(|row| (row.at.id, row.size)).collect();
    group_ids(ids, &sizes, SLICE_BYTES)
}

/// A row of the next page as planned: its position and the position before it (or the cursor),
/// which bound the range that reads it.
struct PlannedRow {
    at: Cursor,
    lower: Option<Cursor>,
}

/// The positions and sizes of the next page: at most [`PACK_MAX_DOCS`] rows and [`SLICE_BYTES`]
/// of them, after `after` (`None` = from the start), in the id column's own order and type (a
/// text comparison would sort a bigint id lexicographically and skip rows), and with a partition
/// column in `(partition, id)` order, which keeps a namespace's rows together. A table whose
/// index is on `(partition, id)` (a composite primary key is one) pages in constant time; without
/// one every page reads and sorts the table.
fn plan_page(layout: &Layout, after: Option<&Cursor>) -> Vec<PlannedRow> {
    let (table, id, ty) = (&layout.table, &layout.id_column, layout.id_array);
    let limit = (PACK_MAX_DOCS as i64).into();
    let (tail, args): (String, Vec<DatumWithOid>) = match &layout.partition {
        None => (
            format!(
                "WHERE $1::text IS NULL OR {table}.{id} > $1::text::{ty} \
                 ORDER BY {table}.{id} LIMIT $2"
            ),
            vec![after.map(|at| at.id.clone()).into(), limit],
        ),
        Some(PartitionLayout { column, cast, .. }) => (
            format!(
                "WHERE {table}.{column} IS NOT NULL AND ($1::text IS NULL OR \
                 ({table}.{column}, {table}.{id}) > ($1::text::{cast}, $2::text::{ty})) \
                 ORDER BY {table}.{column}, {table}.{id} LIMIT $3"
            ),
            vec![
                after.and_then(|at| at.partition.clone()).into(),
                after.map(|at| at.id.clone()).into(),
                limit,
            ],
        ),
    };
    let (mut weight, mut lower) = (0u64, after.cloned());
    let mut rows = Vec::new();
    for row in query_sizes(layout, &tail, &args) {
        if weight != 0 && weight + row.size > SLICE_BYTES {
            break;
        }
        weight += row.size;
        rows.push(PlannedRow {
            at: row.at.clone(),
            lower: lower.replace(row.at),
        });
    }
    rows
}

/// The rows after `lower` up to and including `upper`, in the order of [`plan_page`]. `ORDER BY`
/// names the table on purpose: the select list aliases the id as `doc_id`, but a table column
/// named like an alias must never decide the order.
fn read_range(layout: &Layout, lower: Option<&Cursor>, upper: &Cursor) -> Vec<RawRow> {
    let (table, id, ty) = (&layout.table, &layout.id_column, layout.id_array);
    let (tail, args): (String, Vec<DatumWithOid>) = match &layout.partition {
        None => (
            format!(
                "WHERE ($1::text IS NULL OR {table}.{id} > $1::text::{ty}) \
                 AND {table}.{id} <= $2::text::{ty} ORDER BY {table}.{id}"
            ),
            vec![
                lower.map(|at| at.id.clone()).into(),
                upper.id.clone().into(),
            ],
        ),
        Some(PartitionLayout { column, cast, .. }) => (
            format!(
                "WHERE {table}.{column} IS NOT NULL AND ($1::text IS NULL OR \
                 ({table}.{column}, {table}.{id}) > ($1::text::{cast}, $2::text::{ty})) \
                 AND ({table}.{column}, {table}.{id}) <= ($3::text::{cast}, $4::text::{ty}) \
                 ORDER BY {table}.{column}, {table}.{id}"
            ),
            vec![
                lower.and_then(|at| at.partition.clone()).into(),
                lower.map(|at| at.id.clone()).into(),
                upper.partition.clone().into(),
                upper.id.clone().into(),
            ],
        ),
    };
    query_rows(layout, &tail, &args)
}

/// Reads the planned rows of a backfill page as the id range they cover, halving it where the
/// table cannot render a row.
struct PageReader<'a> {
    relid: pg_sys::Oid,
    layout: &'a Layout,
    unreadable: &'a mut Unreadable,
    built: Vec<Built>,
    refused: HashSet<String>,
}

impl Reading<PlannedRow> for PageReader<'_> {
    type Part = Vec<RawRow>;

    fn read(&mut self, rows: &[PlannedRow]) -> Option<Result<Vec<RawRow>, Caught>> {
        let (relid, layout) = (self.relid, self.layout);
        let (first, last) = (&rows[0], &rows[rows.len() - 1]);
        try_locked_read(relid, || read_range(layout, first.lower.as_ref(), &last.at))
    }

    fn accept(&mut self, rows: Vec<RawRow>) {
        self.unreadable.readable(rows.len());
        for row in rows {
            let at = Cursor {
                partition: row.partition.clone(),
                id: row.id.clone(),
            };
            let layout = self.layout;
            let document = route(layout, row.partition).and_then(|partition| {
                let value = build_document(&row.id, &layout.fields, &row.texts, row.meta)?;
                Ok(Doc::new(&row.id, value).in_partition(partition))
            });
            self.built.push(Built { at, document });
        }
    }

    fn refuse(&mut self, row: &PlannedRow, caught: &Caught) -> Result<(), Stop> {
        let message = unreadable_message(caught);
        self.refused.insert(row.at.id.clone());
        self.built.push(Built {
            at: row.at.clone(),
            document: Err(message.clone()),
        });
        self.unreadable.refuse(&message).map_err(Stop::from)
    }
}

/// Reads the next page of the backfill: the documents of the rows after `after` (`None` = from
/// the start). The page holds at most [`PACK_MAX_DOCS`] rows and about [`PACK_MAX_BYTES`] of
/// documents, so it is one request unless a document is huge. The rows are chosen by their
/// estimated size first ([`SLICE_BYTES`]) and read as the id range they cover; a row the table
/// cannot render ends up in `unreadable` and the cursor moves past it.
pub(crate) fn read_page(
    relid: pg_sys::Oid,
    layout: &Layout,
    after: Option<&Cursor>,
    unreadable: &mut Unreadable,
) -> Result<Page, Stop> {
    let planned = match try_locked_read(relid, || plan_page(layout, after)) {
        None => return Err(Stop::Busy),
        Some(Err(caught)) => return Err(Stop::Sql(caught)),
        Some(Ok(planned)) => planned,
    };
    let mut reader = PageReader {
        relid,
        layout,
        unreadable,
        built: Vec::new(),
        refused: HashSet::new(),
    };
    read_bisecting(&mut reader, &planned)?;
    let PageReader { built, refused, .. } = reader;
    let mut page = assemble_page(built, PACK_MAX_BYTES);
    let rejected = std::mem::take(&mut page.rejected);
    (page.unreadable, page.rejected) = rejected
        .into_iter()
        .partition(|(id, _)| refused.contains(id));
    Ok(page)
}

fn namespace_in(prefix: &str, partitioned: bool, partition: Option<&str>) -> String {
    match partition {
        Some(value) if partitioned => partition::namespace_for(prefix, value),
        _ => prefix.to_string(),
    }
}

fn route_docs(prefix: &str, partitioned: bool, docs: Vec<Doc>) -> Vec<(String, Vec<Doc>)> {
    let mut groups: Vec<(String, Vec<Doc>)> = Vec::new();
    let mut at: HashMap<String, usize> = HashMap::new();
    for doc in docs {
        let namespace = namespace_in(prefix, partitioned, doc.partition.as_deref());
        match at.get(&namespace) {
            Some(&group) => groups[group].1.push(doc),
            None => {
                at.insert(namespace.clone(), groups.len());
                groups.push((namespace, vec![doc]));
            }
        }
    }
    groups
}

/// Where a table's documents go: the client, the index and the table's namespace, which a
/// partition column extends to one namespace per value.
pub(crate) struct Lane {
    client: std::sync::Arc<Client>,
    relid: pg_sys::Oid,
    index: String,
    namespace: String,
    partitioned: bool,
}

impl Lane {
    pub(crate) fn new(client: std::sync::Arc<Client>, registry: &DocRegistryRow) -> Self {
        Self {
            client,
            relid: registry.relid,
            index: registry.index_name.clone(),
            namespace: registry.namespace.clone(),
            partitioned: registry.partition.is_some(),
        }
    }

    /// A partitioned lane's error with the namespace it came from, and a refusal of a namespace's
    /// first documents that names the plan's limit explained as such. A refused document stays
    /// the message Pinecone gave: it becomes a dead letter of its own.
    fn name(&self, namespace: &str, err: ClientError) -> ClientError {
        if !self.partitioned {
            return err;
        }
        let first = partition_lane::names_the_limit(&err)
            && !partition_registry::is_registered(self.relid, PartitionLane::Docs, namespace);
        let err = if first {
            partition_lane::explain_first_doc_write(err, namespace)
        } else {
            err
        };
        match err {
            ClientError::BadRequest { .. } => err,
            other => partition_lane::name_namespace(other, namespace),
        }
    }

    /// The namespace of the partition whose canonical text is `partition`; the table's own
    /// namespace for `None` (no partition column, or a delete queued before it had one).
    fn namespace_of(&self, partition: Option<&str>) -> String {
        namespace_in(&self.namespace, self.partitioned, partition)
    }

    /// Splits `docs` by namespace, namespaces and documents in the order they come.
    fn route(&self, docs: Vec<Doc>) -> Vec<(String, Vec<Doc>)> {
        route_docs(&self.namespace, self.partitioned, docs)
    }

    fn upsert(&self, namespace: &str, docs: &[Doc]) -> Result<(), ClientError> {
        hooks::before_request();
        let documents: Vec<&Value> = docs.iter().map(|doc| &doc.value).collect();
        self.client
            .upsert_documents(&self.index, Some(namespace), &documents)
            .map(|_| ())
            .map_err(|err| self.name(namespace, err))
    }

    fn delete(&self, namespace: &str, selector: DocDeleteSelector) -> Result<(), ClientError> {
        hooks::before_request();
        let request = DeleteDocsRequest {
            index: self.index.clone(),
            namespace: Some(namespace.to_string()),
            selector,
        };
        self.client
            .delete_docs(&request)
            .map_err(|err| self.name(namespace, err))
    }
}

/// Sends the documents of one drain step or backfill chunk: packs them into requests, bisects
/// what Pinecone refuses, and holds the dead letters back until the step is known not to be
/// systemic (see [`UpsertOutcome`]). Delivering a document also removes its dead letter.
pub(crate) struct Sender<'a> {
    lane: &'a Lane,
    relid: pg_sys::Oid,
    outcome: UpsertOutcome,
    unreadable: Unreadable,
    staged: Vec<(String, String)>,
    accepted: Vec<String>,
    sent: bool,
    /// Namespaces this sender has registered, so each is told to the registry once.
    registered: HashSet<String>,
}

impl<'a> Sender<'a> {
    pub(crate) fn new(lane: &'a Lane, relid: pg_sys::Oid) -> Self {
        Self {
            lane,
            relid,
            outcome: UpsertOutcome::default(),
            unreadable: Unreadable::default(),
            staged: Vec::new(),
            accepted: Vec::new(),
            registered: HashSet::new(),
            sent: false,
        }
    }

    /// Takes over rows the table could not render that were found, and counted, before the
    /// sender existed: they are held back like the dead letters Pinecone's refusals make.
    pub(crate) fn adopt(&mut self, counted: Unreadable, rows: Vec<(String, String)>) {
        self.unreadable = counted;
        self.staged.extend(rows);
    }

    /// `rows` were read from the table.
    fn rows_read(&mut self, rows: usize) {
        self.unreadable.readable(rows);
    }

    /// The table could not render row `id`: a dead letter, unless the step turns out systemic.
    fn unreadable_row(&mut self, id: &str, message: String) -> Result<(), ClientError> {
        self.staged.push((id.to_string(), message.clone()));
        let counted = self.unreadable.refuse(&message);
        self.outcome.tripped |= counted.is_err();
        counted
    }

    /// Sends `docs`, each namespace's documents together; on a lane-level error what got through
    /// stays in [`Self::take_accepted`].
    pub(crate) fn send(&mut self, docs: Vec<Doc>) -> Result<(), ClientError> {
        let lane = self.lane;
        for (namespace, group) in lane.route(docs) {
            for batch in pack(group, PACK_MAX_DOCS, PACK_MAX_BYTES) {
                let result = upsert_bisecting(
                    &mut |docs| lane.upsert(&namespace, docs),
                    &batch,
                    &mut self.outcome,
                );
                self.staged.append(&mut self.outcome.dead);
                let accepted = std::mem::take(&mut self.outcome.delivered);
                if !accepted.is_empty() {
                    self.sent = true;
                    self.register(&namespace, &batch);
                }
                docs_outbox::clear_failed(self.relid, &accepted);
                self.accepted.extend(accepted);
                result?;
            }
        }
        Ok(())
    }

    /// Tells the registry that `namespace` holds documents now, which a wipe needs to know (all
    /// of `batch` belong to one partition).
    fn register(&mut self, namespace: &str, batch: &[Doc]) {
        let Some(value) = batch.iter().find_map(|doc| doc.partition.as_deref()) else {
            return;
        };
        if self.registered.insert(namespace.to_string()) {
            partition_registry::register_partition(
                self.relid,
                PartitionLane::Docs,
                value,
                namespace,
            );
        }
    }

    /// `Err` if Pinecone refused, or the table could not render, every document of the step (see
    /// [`UpsertOutcome`] and [`Unreadable`]).
    pub(crate) fn verdict(&mut self) -> Result<(), ClientError> {
        let verdict = self.outcome.check_all_refused();
        let verdict = verdict.and_then(|()| self.unreadable.verdict());
        self.outcome.tripped |= verdict.is_err();
        verdict
    }

    /// The ids Pinecone accepted since the last call.
    pub(crate) fn take_accepted(&mut self) -> Vec<String> {
        std::mem::take(&mut self.accepted)
    }

    /// Whether any request was accepted.
    pub(crate) fn sent(&self) -> bool {
        self.sent
    }

    /// Records the dead letters, unless the breaker opened, and returns their ids.
    pub(crate) fn settle(self) -> Vec<String> {
        if self.outcome.tripped {
            return Vec::new();
        }
        for (id, message) in &self.staged {
            docs_outbox::enqueue_failed(self.relid, id, message);
        }
        self.staged.into_iter().map(|(id, _)| id).collect()
    }
}

const DELETE_CHUNK: usize = 1_000;

/// Why a batch ended before its plan was delivered; what it delivered up to then is kept.
#[derive(Debug)]
pub(crate) enum Stop {
    /// A lane-level failure, to record on the queue.
    Failed(ClientError),
    /// The SQL that reads the table failed in a way that is not about one row.
    Sql(Caught),
    /// DDL holds the table: the rest waits for the next round.
    Busy,
    /// The batch's time is spent: the rest waits for the next round.
    Budget,
}

impl From<ClientError> for Stop {
    fn from(error: ClientError) -> Self {
        Stop::Failed(error)
    }
}

/// Share-locks the table without waiting: blocking here, while holding the lane lock and the
/// claimed rows, could deadlock against DDL queued for an exclusive lock.
pub(crate) fn try_lock_table(relid: pg_sys::Oid) -> bool {
    const LOCKMODE: pg_sys::LOCKMODE = pg_sys::AccessShareLock as pg_sys::LOCKMODE;
    !hooks::table_busy() && unsafe { pg_sys::ConditionalLockRelationOid(relid, LOCKMODE) }
}

/// Runs the read `body` with the table share-locked, without waiting, and every lock it took
/// (indexes included) given back when it is done; `None` if the table is busy, `Err` if the read
/// raised. Held to the end of the transaction, the locks would span the requests that follow and
/// stall any DDL queued behind them, and every reader behind that.
fn try_locked_read<T>(relid: pg_sys::Oid, body: impl FnOnce() -> T) -> Option<Result<T, Caught>> {
    docs_guard::rolled_back_caught(AssertUnwindSafe(|| try_lock_table(relid).then(body)))
        .transpose()
}

/// One way of reading groups of `I` (ids, planned rows) from the table.
trait Reading<I> {
    type Part;

    /// Reads `items` together; `None` if the table is busy.
    fn read(&mut self, items: &[I]) -> Option<Result<Self::Part, Caught>>;

    fn accept(&mut self, part: Self::Part);

    /// `item` alone cannot be read.
    fn refuse(&mut self, item: &I, caught: &Caught) -> Result<(), Stop>;
}

/// Reads `items`; where the table raises a permanent ERROR for a group it halves the group until
/// the item the ERROR comes from stands alone, which `reader` then refuses. What reads is taken in
/// order. Any other failure ends the read: a transient ERROR says nothing about the rows, and
/// `reader` ends it when the refusals look systemic.
fn read_bisecting<I, R: Reading<I>>(reader: &mut R, items: &[I]) -> Result<(), Stop> {
    if items.is_empty() {
        return Ok(());
    }
    match reader.read(items) {
        None => Err(Stop::Busy),
        Some(Ok(part)) => {
            reader.accept(part);
            Ok(())
        }
        Some(Err(caught)) if caught.transient => Err(Stop::Sql(caught)),
        Some(Err(caught)) => match items {
            [single] => reader.refuse(single, &caught),
            _ => {
                let (left, right) = items.split_at(items.len() / 2);
                read_bisecting(reader, left)?;
                read_bisecting(reader, right)
            }
        },
    }
}

/// Reads the documents of a group of ids of the live path.
struct GroupReader<'a, 'b> {
    relid: pg_sys::Oid,
    layout: &'a Layout,
    sender: &'a mut Sender<'b>,
    read: Read,
}

impl Reading<String> for GroupReader<'_, '_> {
    type Part = Read;

    fn read(&mut self, ids: &[String]) -> Option<Result<Read, Caught>> {
        let (relid, layout) = (self.relid, self.layout);
        try_locked_read(relid, || read_documents(layout, ids))
    }

    fn accept(&mut self, part: Read) {
        self.sender.rows_read(part.docs.len() + part.rejected.len());
        self.read.absorb(part);
    }

    fn refuse(&mut self, id: &String, caught: &Caught) -> Result<(), Stop> {
        let message = unreadable_message(caught);
        self.sender.unreadable_row(id, message).map_err(Stop::from)
    }
}

/// What reading the upserts found, which the deletes then need.
#[derive(Default)]
struct Upserted {
    /// Ids whose row is gone.
    missing: Vec<String>,
    /// `(partition, id)` of every document read: a delete aimed at the same namespace is moot.
    placed: HashSet<(Option<String>, String)>,
}

/// Reads and sends the upserts group by group (see [`SLICE_BYTES`]). Rows the `error` policy
/// refuses become dead letters at once; rows the table cannot render are found by halving the
/// group (see [`read_bisecting`]) and become dead letters with the rest of the step's.
fn deliver_upserts(
    layout: &Layout,
    relid: pg_sys::Oid,
    upserts: &[String],
    found: &mut Upserted,
    progress: &mut Progress,
    sender: &mut Sender<'_>,
    budget: &Budget,
) -> Result<(), Stop> {
    for slice in upserts.chunks(READ_SLICE) {
        let groups = match try_locked_read(relid, || plan_groups(layout, slice)) {
            None => return Err(Stop::Busy),
            Some(Err(caught)) => return Err(Stop::Sql(caught)),
            Some(Ok(groups)) => groups,
        };
        for group in groups {
            if budget.spent() {
                return Err(Stop::Budget);
            }
            let mut reader = GroupReader {
                relid,
                layout,
                sender: &mut *sender,
                read: Read::default(),
            };
            read_bisecting(&mut reader, &group)?;
            let read = reader.read;
            for (id, message) in &read.rejected {
                docs_outbox::enqueue_failed(relid, id, message);
            }
            progress.upserts_done(read.rejected.into_iter().map(|(id, _)| id));
            found.missing.extend(read.missing);
            let read_at = read.docs.iter();
            found
                .placed
                .extend(read_at.map(|doc| (doc.partition.clone(), doc.id.clone())));
            sender.send(read.docs)?;
            budget.did_work();
        }
    }
    Ok(())
}

/// Empties what the table wrote (a `doc_delete_all`, queued as `barrier`): its namespace, or with
/// a partition column every namespace the registry lists. Each is forgotten as it is emptied, so a
/// step cut short by the budget or an error carries on with the rest.
///
/// The values the table holds name namespaces too, and they are registered first (once per
/// barrier): a drain that rolled back after Pinecone took a document of a new value lost that
/// value's registration, and nothing lists the namespaces of a document index.
fn wipe(
    lane: &Lane,
    relid: pg_sys::Oid,
    layout: &Layout,
    barrier: i64,
    sent: &mut bool,
    budget: &Budget,
) -> Result<(), Stop> {
    if !lane.partitioned {
        lane.delete(&lane.namespace, DocDeleteSelector::All)?;
        return Ok(());
    }
    if let Some(partition) = &layout.partition
        && !docs_outbox::wipe_adopted(barrier)
    {
        let (prefix, table) = (&partition.prefix, &layout.table);
        partition_ops::adopt_table_values(
            relid,
            PartitionLane::Docs,
            prefix,
            table,
            &partition.column,
        );
        docs_outbox::mark_wipe_adopted(barrier);
    }
    let namespaces = partition_registry::partitions_of(relid, PartitionLane::Docs);
    for (at, (_, namespace)) in namespaces.iter().enumerate() {
        lane.delete(namespace, DocDeleteSelector::All)?;
        partition_registry::forget_partition(relid, PartitionLane::Docs, namespace);
        *sent = true;
        budget.did_work();
        if at + 1 < namespaces.len() && budget.spent() {
            return Err(Stop::Budget);
        }
    }
    Ok(())
}

/// What a delete settles once it is sent: a delete the plan owes, or the upsert of a row that
/// vanished.
enum Settles {
    Delete(Option<String>),
    Upsert,
}

/// The ids to delete from each namespace, namespaces and ids in the order they come.
#[derive(Default)]
struct Owed {
    at: HashMap<String, usize>,
    by_namespace: Vec<(String, Vec<(String, Settles)>)>,
}

impl Owed {
    fn add(&mut self, namespace: String, id: String, settles: Settles) {
        match self.at.get(&namespace) {
            Some(&at) => self.by_namespace[at].1.push((id, settles)),
            None => {
                self.at.insert(namespace.clone(), self.by_namespace.len());
                self.by_namespace.push((namespace, vec![(id, settles)]));
            }
        }
    }
}

/// Sends the deletes the plan owes, and without a partition column those of the ids whose row
/// vanished (with one, their delete entries say where they were). A delete aimed at the namespace
/// that an upsert just filled with the same id is not sent: the document is the live one.
fn deliver_deletes(
    lane: &Lane,
    relid: pg_sys::Oid,
    plan: &Plan,
    found: Upserted,
    progress: &mut Progress,
    sent: &mut bool,
    budget: &Budget,
) -> Result<(), Stop> {
    let gone: HashSet<&str> = found.missing.iter().map(String::as_str).collect();
    // Rows that exist keep their dead letter: it is the upsert's, whatever stale copy goes.
    let live: HashSet<&str> = (plan.upserts.iter().map(|upsert| upsert.id.as_str()))
        .filter(|id| !gone.contains(id))
        .collect();
    let mut owed = Owed::default();
    for delete in &plan.deletes {
        if found.placed.contains(&delete.key()) {
            progress.deletes_done([delete.key()]);
            continue;
        }
        let namespace = lane.namespace_of(delete.partition.as_deref());
        let settles = Settles::Delete(delete.partition.clone());
        owed.add(namespace, delete.id.clone(), settles);
    }
    for id in &found.missing {
        match lane.partitioned {
            true => progress.upserts_done([id.clone()]),
            false => owed.add(lane.namespace.clone(), id.clone(), Settles::Upsert),
        }
    }
    for (namespace, entries) in &owed.by_namespace {
        for chunk in entries.chunks(DELETE_CHUNK) {
            if budget.spent() {
                return Err(Stop::Budget);
            }
            let ids: Vec<String> = chunk.iter().map(|(id, _)| id.clone()).collect();
            lane.delete(namespace, DocDeleteSelector::Ids(ids.clone()))?;
            *sent = true;
            let dead: Vec<String> = (ids.iter())
                .filter(|id| !live.contains(id.as_str()))
                .cloned()
                .collect();
            docs_outbox::clear_failed(relid, &dead);
            for (id, settles) in chunk {
                match settles {
                    Settles::Delete(partition) => {
                        progress.deletes_done([(partition.clone(), id.clone())]);
                    }
                    Settles::Upsert => progress.upserts_done([id.clone()]),
                }
            }
            budget.did_work();
        }
    }
    Ok(())
}

/// Sends the plan: the wipe first (the table's dead letters go with it), then the upserts group
/// by group (read, pack, send), then the deletes. Stops at the first lane-level error, a busy
/// table or a spent budget; `progress` and `sent` say what got through before it. Dead letters
/// are written at the end of the upserts, unless the whole step looks systemically refused.
fn deliver(
    lane: &Lane,
    relid: pg_sys::Oid,
    layout: &Layout,
    plan: &Plan,
    progress: &mut Progress,
    sent: &mut bool,
    budget: &Budget,
) -> Result<(), Stop> {
    if let Some(barrier) = plan.delete_all {
        wipe(lane, relid, layout, barrier, sent, budget)?;
        docs_outbox::clear_all_failed(relid);
        progress.delete_all_done = true;
        *sent = true;
        budget.did_work();
    }
    let upserts: Vec<String> = plan
        .upserts
        .iter()
        .map(|upsert| upsert.id.clone())
        .collect();
    let mut found = Upserted::default();
    let mut sender = Sender::new(lane, relid);
    let result = deliver_upserts(
        layout,
        relid,
        &upserts,
        &mut found,
        progress,
        &mut sender,
        budget,
    )
    .and_then(|()| sender.verdict().map_err(Stop::from));
    progress.upserts_done(sender.take_accepted());
    *sent |= sender.sent();
    progress.upserts_done(sender.settle());
    result?;
    deliver_deletes(lane, relid, plan, found, progress, sent, budget)
}

/// Records a lane-level failure on the oldest row still queued. A transient one (429, 5xx,
/// timeout) only sets `last_error`; a permanent one counts an attempt, and the lane parks once
/// `pinecone.outbox_max_attempts` is reached.
pub(crate) fn record_failure(relid: pg_sys::Oid, head: i64, error: ClientError) -> DocDrainOutcome {
    if backend::is_transient(&error) {
        docs_outbox::record_error(head, &backend::failure_text(&error));
        return DocDrainOutcome::Backoff(error);
    }
    let text = backend::failure_text(&error);
    permanent_failure(relid, head, &text, error)
}

/// Counts a permanent failure with `text` as its message and parks the lane when it was the
/// last attempt.
fn permanent_failure(
    relid: pg_sys::Oid,
    head: i64,
    text: &str,
    error: ClientError,
) -> DocDrainOutcome {
    // The row shows the attempt; the lane's own count decides, since the oldest row changes as
    // rows complete and a failure that follows a partial success would start from zero again.
    docs_outbox::bump_attempts(&[head], text);
    if docs_registry::count_lane_failure(relid) >= crate::OUTBOX_MAX_ATTEMPTS.get() {
        docs_registry::set_parked(relid, text);
        DocDrainOutcome::Parked(error)
    } else {
        DocDrainOutcome::Backoff(error)
    }
}

/// Postgres quotes the offending value in many messages; a lane's `last_error` is not the place
/// for a megabyte of it.
const MAX_ERROR_TEXT: usize = 500;

/// `text` cut to about `max` bytes on character boundaries, with an ellipsis in the middle if it
/// was cut: Postgres quotes the offending value at one end of its sentence and says what is
/// wrong at the other.
fn shorten(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let (mut head, mut tail) = (max / 2, text.len() - (max - max / 2));
    while !text.is_char_boundary(head) {
        head -= 1;
    }
    while !text.is_char_boundary(tail) {
        tail += 1;
    }
    format!("{}…{}", &text[..head], &text[tail..])
}

/// Records the ERROR `caught` in a lane's step like any lane failure: the message goes on the
/// queue row, a permanent one counts towards parking, a transient one only backs off. After a
/// rollback the lane lock is gone; the bookkeeping try-locks it again.
pub(crate) fn record_sql_failure(
    relid: pg_sys::Oid,
    caught: Caught,
    job_first: bool,
) -> DocDrainOutcome {
    let display = catalog::display_name_of(relid);
    let text = format!(
        "{display}: the document lane's SQL failed — {}",
        shorten(&caught.message, MAX_ERROR_TEXT)
    );
    let head = if try_lock_lane(relid) {
        docs_outbox::head_row(relid, job_first)
    } else {
        None
    };
    let error = match caught.transient {
        true => ClientError::Transport {
            url: display,
            reason: caught.message,
        },
        false => ClientError::BadRequest {
            message: text.clone(),
        },
    };
    match head {
        None => DocDrainOutcome::Backoff(error),
        Some(head) if caught.transient => {
            docs_outbox::record_error(head, &text);
            DocDrainOutcome::Backoff(error)
        }
        Some(head) => permanent_failure(relid, head, &text, error),
    }
}

/// [`drain_doc_step`] for the worker's round: an ERROR raised in it, by the registry or queue
/// SQL (the table's own reads are handled where they run), is the lane's failure (see
/// [`record_sql_failure`]), not an abort of the transaction every table's round shares.
pub(crate) fn drain_isolated(relid: pg_sys::Oid) -> DocDrainOutcome {
    match docs_guard::isolated(AssertUnwindSafe(|| drain_doc_step(relid))) {
        Ok(outcome) => outcome,
        Err(caught) => record_sql_failure(relid, caught, false),
    }
}

/// Drops everything queued for a table that is no longer tracked or no longer exists.
pub(crate) fn purge(relid: pg_sys::Oid, why: &str) -> DocDrainOutcome {
    let purged = docs_outbox::clear_relid(relid);
    warning!(
        "pg_pinecone: document drain: relid {relid} {why} — purged {purged} queued row(s) of \
         pinecone.doc_outbox"
    );
    DocDrainOutcome::Processed(purged.max(0) as u64)
}

pub(crate) fn lock_lane(relid: pg_sys::Oid) {
    Spi::run_with_args(
        "SELECT pg_catalog.pg_advisory_xact_lock($1, $2)",
        &[DOC_LOCK_MAGIC.into(), (relid.to_u32() as i32).into()],
    )
    .expect("pg_pinecone: document drain advisory lock failed");
}

/// [`lock_lane`] for the verbs that change a lane: they say so when a drain holds it (a batch is
/// bounded by [`BATCH_BUDGET`]).
pub(crate) fn lock_lane_for_verb(relid: pg_sys::Oid, display: &str) {
    if !try_lock_lane(relid) {
        notice!("pg_pinecone: waiting for the running drain of {display} to finish its batch");
        lock_lane(relid);
    }
}

/// [`lock_lane`] without waiting: `false` if someone else holds the lane.
pub(crate) fn try_lock_lane(relid: pg_sys::Oid) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT pg_catalog.pg_try_advisory_xact_lock($1, $2)",
        &[DOC_LOCK_MAGIC.into(), (relid.to_u32() as i32).into()],
    )
    .expect("pg_pinecone: document drain advisory try-lock failed")
    .expect("pg_pinecone: pg_try_advisory_xact_lock unexpectedly NULL")
}

fn drain_claimed(
    relid: pg_sys::Oid,
    registry: &DocRegistryRow,
    rows: &[DocOutboxRow],
    budget: &Budget,
) -> DocDrainOutcome {
    let head = rows[0].id;
    let layout = match build_layout(relid, registry) {
        Ok(layout) => layout,
        Err(error) => return record_failure(relid, head, error),
    };
    let claimed = claimed_rows(rows);
    let plan = fold(&claimed, registry.partition.is_some());
    let lane = Lane::new(
        backend::client_for_drain(DrainCredentials::Worker),
        registry,
    );
    let (mut progress, mut sent) = (Progress::default(), false);
    let result = deliver(
        &lane,
        relid,
        &layout,
        &plan,
        &mut progress,
        &mut sent,
        budget,
    );
    let complete = complete_rows(&claimed, &plan, &progress);
    docs_outbox::delete_rows(&complete);
    if sent {
        docs_registry::touch_delivered(relid);
        let waiting: Vec<i64> = (rows.iter())
            .map(|row| row.id)
            .filter(|id| !complete.contains(id))
            .collect();
        docs_outbox::clear_errors(&waiting);
    }
    match result {
        Ok(()) => {
            docs_registry::reset_lane_failures(relid);
            DocDrainOutcome::Processed(complete.len() as u64)
        }
        // The rest stays queued for the next round, nothing is wrong.
        Err(Stop::Busy | Stop::Budget) => {
            if sent || !complete.is_empty() {
                docs_registry::reset_lane_failures(relid);
            }
            DocDrainOutcome::Processed(complete.len() as u64)
        }
        Err(Stop::Failed(error)) => {
            // The oldest row still queued carries the failure.
            let oldest = rows
                .iter()
                .map(|row| row.id)
                .find(|id| !complete.contains(id));
            record_failure(relid, oldest.unwrap_or(head), error)
        }
        Err(Stop::Sql(caught)) => record_sql_failure(relid, caught, false),
    }
}

/// One drain step for `relid`: claims its next batch of live queue rows and delivers it, for at
/// most [`BATCH_BUDGET`]. The unit of work of the worker's document round and of
/// `pinecone._docs_round`. The backfill job is claimed with them (it is queue order's first row)
/// but left alone: `docs_backfill.rs` runs it, after the live rows. A lane another session holds
/// (`backfill_docs`, `untrack_docs`) is skipped, not waited for.
pub(crate) fn drain_doc_step(relid: pg_sys::Oid) -> DocDrainOutcome {
    drain_doc_step_with(relid, Budget::new(BATCH_BUDGET))
}

/// [`drain_doc_step`] with its own time budget.
pub(crate) fn drain_doc_step_with(relid: pg_sys::Oid, budget: Budget) -> DocDrainOutcome {
    if !try_lock_lane(relid) {
        return DocDrainOutcome::Processed(0);
    }
    let rows = docs_outbox::fetch_batch(relid);
    if rows.is_empty() {
        return DocDrainOutcome::Processed(0);
    }
    let Some(registry) = docs_registry::lookup(relid) else {
        return purge(relid, "is no longer tracked in pinecone.doc_tables");
    };
    if catalog::resolve_table_name(relid).is_none() {
        docs_registry::delete(relid);
        return purge(relid, "no longer resolves to a table");
    }
    let live: Vec<DocOutboxRow> = rows
        .into_iter()
        .filter(|row| row.op != DocOp::Backfill)
        .collect();
    if live.is_empty() {
        return DocDrainOutcome::Processed(0);
    }
    // Checked before building a client: stored_credentials() raises, which would abort the round.
    if !secret::has_stored_api_key() {
        docs_outbox::record_error(live[0].id, NO_STORED_KEY_ERROR);
        return DocDrainOutcome::Backoff(ClientError::MissingApiKey);
    }
    drain_claimed(relid, &registry, &live, &budget)
}

/// `pinecone._docs_round`: one worker round for `relid` in the caller's transaction, ignoring
/// scheduler backoff windows: the live rows first, then backfill chunks back to back until
/// `pinecone.backfill_round_ms` has elapsed (at least one, whatever live rows are still queued;
/// only a queued wipe holds them back). Reported as `idle` (nothing queued, no backfill
/// pending), `processed`, `backoff` or `parked`.
pub(crate) fn docs_round(relid: pg_sys::Oid) -> &'static str {
    match docs_registry::lookup(relid) {
        None => backend::error_with_code(
            PgSqlErrorCode::ERRCODE_UNDEFINED_OBJECT,
            format!(
                "pinecone._docs_round: {} is not tracked by pinecone.track_docs",
                catalog::display_name_of(relid)
            ),
            None,
            None,
        ),
        Some(row) if row.queue_parked => return "parked",
        Some(_) => {}
    }
    let mut progressed = match drain_doc_step(relid) {
        DocDrainOutcome::Processed(rows) => rows > 0,
        DocDrainOutcome::Backoff(_) => return "backoff",
        DocDrainOutcome::Parked(_) => return "parked",
    };
    if !docs_outbox::wipe_queued(relid) {
        let report = docs_backfill::run_in_caller_transaction(relid);
        match report.failure {
            Some(DocDrainOutcome::Backoff(_)) => return "backoff",
            Some(DocDrainOutcome::Parked(_)) => return "parked",
            _ => {}
        }
        progressed |= report.progressed > 0;
    }
    if progressed || docs_outbox::live_backlog(relid) > 0 || docs_backfill::job_queued(relid) {
        "processed"
    } else {
        "idle"
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn ids(list: &[&str]) -> Vec<String> {
        list.iter().map(|id| id.to_string()).collect()
    }

    fn item(id: &str, partition: Option<&str>) -> Item {
        Item {
            id: id.to_string(),
            partition: partition.map(str::to_string),
        }
    }

    fn row(row: i64, op: DocOp, list: &[&str]) -> Claimed {
        Claimed {
            row,
            op,
            items: list.iter().map(|id| item(id, None)).collect(),
        }
    }

    /// A claimed row of `{"id", "p"}` entries.
    fn rows_of(row: i64, op: DocOp, list: &[(&str, &str)]) -> Claimed {
        Claimed {
            row,
            op,
            items: list.iter().map(|(id, p)| item(id, Some(p))).collect(),
        }
    }

    fn upsert(id: &str, row: i64) -> Upsert {
        Upsert {
            id: id.to_string(),
            row,
        }
    }

    fn delete(id: &str, partition: Option<&str>, row: i64) -> Delete {
        Delete {
            id: id.to_string(),
            partition: partition.map(str::to_string),
            row,
        }
    }

    fn doc(id: &str, bytes: usize) -> Doc {
        let value = json!({"_id": id, "pad": "x".repeat(bytes)});
        Doc::new(id, value)
    }

    // ---- error texts ----

    #[test]
    fn a_long_error_text_keeps_both_ends_cut_on_character_boundaries() {
        assert_eq!(shorten("short", 500), "short");
        assert_eq!(shorten("abcdef", 4), "ab…ef");
        assert_eq!(
            shorten("aébécé", 4),
            "a…é",
            "never inside a two-byte character"
        );
        let long = format!(
            "\"{}\" is out of range for type double precision",
            "9".repeat(2_000)
        );
        let cut = shorten(&long, MAX_ERROR_TEXT);
        assert!(
            cut.starts_with("\"9999") && cut.ends_with("out of range for type double precision")
        );
        assert!(
            cut.len() <= MAX_ERROR_TEXT + '…'.len_utf8() + 3,
            "{} bytes",
            cut.len()
        );
    }

    // ---- reading groups that the table cannot always render ----

    fn caught(transient: bool) -> Caught {
        Caught {
            message: "value out of range".to_string(),
            transient,
            code: PgSqlErrorCode::ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
        }
    }

    /// Items `0..n`; the table raises for any group holding a poison item.
    struct Table<'a> {
        poison: &'a [usize],
        transient: Option<usize>,
        busy: bool,
        reads: usize,
        read: Vec<usize>,
        refused: Vec<usize>,
    }

    impl<'a> Table<'a> {
        fn new(poison: &'a [usize]) -> Self {
            Self {
                poison,
                transient: None,
                busy: false,
                reads: 0,
                read: Vec::new(),
                refused: Vec::new(),
            }
        }
    }

    impl Reading<usize> for Table<'_> {
        type Part = Vec<usize>;

        fn read(&mut self, items: &[usize]) -> Option<Result<Vec<usize>, Caught>> {
            self.reads += 1;
            if self.busy {
                return None;
            }
            if self.transient.is_some_and(|item| items.contains(&item)) {
                return Some(Err(caught(true)));
            }
            match items.iter().any(|item| self.poison.contains(item)) {
                true => Some(Err(caught(false))),
                false => Some(Ok(items.to_vec())),
            }
        }

        fn accept(&mut self, part: Vec<usize>) {
            self.read.extend(part);
        }

        fn refuse(&mut self, item: &usize, _: &Caught) -> Result<(), Stop> {
            self.refused.push(*item);
            Ok(())
        }
    }

    #[test]
    fn a_poison_item_anywhere_in_twenty_thousand_is_found_alone_and_the_rest_read_in_order() {
        let items: Vec<usize> = (0..20_000).collect();
        for at in [0, 1, 999, 1_000, 9_999, 10_000, 12_000, 19_998, 19_999] {
            let poison = [at];
            let mut table = Table::new(&poison);
            read_bisecting(&mut table, &items).expect("read");
            assert_eq!(table.refused, [at]);
            let rest: Vec<usize> = items.iter().copied().filter(|item| *item != at).collect();
            assert_eq!(table.read, rest, "poison at {at}");
            assert!(
                table.reads <= 2 * 15 + 1,
                "{} reads for poison at {at}",
                table.reads
            );
        }
    }

    #[test]
    fn several_poison_items_are_each_found_and_a_clean_group_is_read_once() {
        let items: Vec<usize> = (0..1_000).collect();
        let mut table = Table::new(&[3, 4, 700, 999]);
        read_bisecting(&mut table, &items).expect("read");
        assert_eq!(table.refused, [3, 4, 700, 999]);
        assert_eq!(table.read.len(), 996);
        assert!(
            table.read.windows(2).all(|pair| pair[0] < pair[1]),
            "in order"
        );

        let mut clean = Table::new(&[]);
        read_bisecting(&mut clean, &items).expect("read");
        assert_eq!((clean.reads, clean.refused.len()), (1, 0));
    }

    #[test]
    fn a_transient_error_or_a_busy_table_ends_the_read_without_refusing_anything() {
        let items: Vec<usize> = (0..100).collect();
        let mut table = Table::new(&[]);
        table.transient = Some(60);
        let stop = read_bisecting(&mut table, &items).expect_err("transient");
        assert!(
            matches!(stop, Stop::Sql(ref caught) if caught.transient),
            "{stop:?}"
        );
        assert!(table.refused.is_empty());

        let mut busy = Table::new(&[5]);
        busy.busy = true;
        let stop = read_bisecting(&mut busy, &items).expect_err("busy");
        assert!(matches!(stop, Stop::Busy), "{stop:?}");
        assert_eq!(busy.reads, 1);
        assert!(read_bisecting(&mut Table::new(&[]), &[] as &[usize]).is_ok());
    }

    #[test]
    fn a_hundred_unreadable_rows_in_a_row_are_systemic_a_read_row_in_between_is_not() {
        let mut unreadable = Unreadable::default();
        for _ in 0..99 {
            unreadable
                .refuse("value out of range")
                .expect("below the streak");
        }
        unreadable.readable(1);
        for _ in 0..99 {
            unreadable
                .refuse("value out of range")
                .expect("the streak started over");
        }
        let error = unreadable
            .refuse("value out of range")
            .expect_err("100 in a row");
        let message = error.to_string();
        assert!(message.contains("100 documents in a row"), "{message}");
        assert!(message.contains("value out of range"), "{message}");
    }

    #[test]
    fn a_step_in_which_nothing_reads_is_systemic_from_ten_unreadable_rows_on() {
        for (count, readable, systemic) in
            [(9, 0, false), (10, 0, true), (40, 0, true), (40, 1, false)]
        {
            let mut unreadable = Unreadable::default();
            unreadable.readable(readable);
            for _ in 0..count {
                unreadable.refuse("boom").expect("below the streak");
            }
            assert_eq!(
                unreadable.verdict().is_err(),
                systemic,
                "{count} unreadable, {readable} read"
            );
        }
    }

    // ---- read groups and the batch budget ----

    fn weights(list: &[(&str, u64)]) -> HashMap<String, u64> {
        list.iter()
            .map(|(id, size)| (id.to_string(), *size))
            .collect()
    }

    #[test]
    fn ids_are_grouped_in_order_by_their_estimated_size() {
        let sizes = weights(&[("a", 4), ("b", 4), ("c", 4), ("d", 1)]);
        let groups = group_ids(&ids(&["a", "b", "c", "d"]), &sizes, 8);
        assert_eq!(groups, vec![ids(&["a", "b"]), ids(&["c", "d"])]);
        let one = group_ids(&ids(&["a", "b", "c", "d"]), &sizes, 100);
        assert_eq!(one, vec![ids(&["a", "b", "c", "d"])]);
    }

    #[test]
    fn an_id_bigger_than_the_limit_is_read_alone_and_never_in_an_empty_group() {
        let sizes = weights(&[("a", 1), ("big", 50), ("c", 1)]);
        let groups = group_ids(&ids(&["a", "big", "c"]), &sizes, 8);
        assert_eq!(groups, vec![ids(&["a"]), ids(&["big"]), ids(&["c"])]);
        assert!(group_ids(&[], &sizes, 8).is_empty());
    }

    #[test]
    fn an_id_without_a_row_weighs_nothing_but_is_still_read() {
        let sizes = weights(&[("a", 8)]);
        let groups = group_ids(&ids(&["gone", "a", "also-gone"]), &sizes, 8);
        assert_eq!(groups, vec![ids(&["gone", "a", "also-gone"])]);
        let all: Vec<String> = groups.into_iter().flatten().collect();
        assert_eq!(
            all,
            ids(&["gone", "a", "also-gone"]),
            "every id is in a group"
        );
    }

    #[test]
    fn a_budget_ends_a_batch_only_once_something_went_out() {
        let spent = Budget::new(Duration::ZERO);
        assert!(!spent.spent(), "every step makes progress first");
        spent.did_work();
        assert!(spent.spent());
        let roomy = Budget::new(Duration::from_secs(3_600));
        roomy.did_work();
        assert!(!roomy.spent());
    }

    // ---- folding claimed rows ----

    #[test]
    fn the_last_op_per_id_wins_and_first_seen_order_is_kept() {
        let rows = [
            row(1, DocOp::Upsert, &["a", "b"]),
            row(2, DocOp::Delete, &["b"]),
            row(3, DocOp::Upsert, &["c", "a"]),
            row(4, DocOp::Delete, &["a"]),
        ];
        let plan = fold(&rows, false);
        assert_eq!(plan.delete_all, None);
        assert_eq!(plan.upserts, vec![upsert("c", 3)]);
        assert_eq!(
            plan.deletes,
            vec![delete("a", None, 4), delete("b", None, 2)]
        );
    }

    #[test]
    fn a_delete_all_is_a_barrier_that_discards_earlier_ids_only() {
        let rows = [
            row(1, DocOp::Upsert, &["a", "b"]),
            row(2, DocOp::DeleteAll, &[]),
            row(3, DocOp::Upsert, &["b"]),
            row(4, DocOp::DeleteAll, &[]),
            row(5, DocOp::Upsert, &["c"]),
        ];
        for partitioned in [false, true] {
            let plan = fold(&rows, partitioned);
            assert_eq!(plan.delete_all, Some(4), "the last one carries the barrier");
            assert_eq!(plan.upserts, vec![upsert("c", 5)]);
            assert!(plan.deletes.is_empty());
        }
    }

    #[test]
    fn backfill_jobs_and_non_array_payloads_contribute_nothing() {
        let rows = [
            row(1, DocOp::Backfill, &[]),
            row(2, DocOp::Upsert, &[]),
            row(3, DocOp::Upsert, &["a"]),
        ];
        assert_eq!(fold(&rows, false).upserts, vec![upsert("a", 3)]);
        assert_eq!(fold(&rows, true).upserts, vec![upsert("a", 3)]);
    }

    #[test]
    fn payload_items_keep_ids_and_id_partition_objects_and_ignore_everything_else() {
        let payload = json!([
            "1", 2, null, "x", ["y"], {"id": "8", "p": "2"}, {"id": "9"}, {"id": "10", "p": null},
            {"id": 5, "p": "1"}, {"p": "1"}, {"id": "11", "p": 3}
        ]);
        assert_eq!(
            payload_items(Some(&payload)),
            vec![
                item("1", None),
                item("x", None),
                item("8", Some("2")),
                item("9", None),
                item("10", None),
                item("11", Some("3")),
            ]
        );
        assert!(payload_items(Some(&json!({"id": "1"}))).is_empty());
        assert!(payload_items(Some(&Value::Null)).is_empty());
        assert!(payload_items(None).is_empty());
    }

    #[test]
    fn without_a_partition_column_a_delete_entry_is_a_plain_delete_of_its_id() {
        let rows = [
            rows_of(1, DocOp::Delete, &[("a", "1")]),
            row(2, DocOp::Upsert, &["a"]),
            rows_of(3, DocOp::Delete, &[("b", "2")]),
        ];
        let plan = fold(&rows, false);
        assert_eq!(plan.upserts, vec![upsert("a", 2)]);
        assert_eq!(plan.deletes, vec![delete("b", None, 3)]);
    }

    #[test]
    fn a_move_keeps_the_delete_from_the_old_partition_beside_the_upsert() {
        let rows = [
            rows_of(1, DocOp::Delete, &[("a", "1")]),
            row(2, DocOp::Upsert, &["a"]),
        ];
        let plan = fold(&rows, true);
        assert_eq!(plan.upserts, vec![upsert("a", 2)]);
        assert_eq!(plan.deletes, vec![delete("a", Some("1"), 1)]);
    }

    #[test]
    fn a_delete_after_the_upsert_cancels_it_because_the_row_is_gone() {
        let rows = [
            row(1, DocOp::Upsert, &["a"]),
            rows_of(2, DocOp::Delete, &[("a", "1")]),
        ];
        let plan = fold(&rows, true);
        assert!(plan.upserts.is_empty());
        assert_eq!(plan.deletes, vec![delete("a", Some("1"), 2)]);
    }

    #[test]
    fn moves_back_and_forth_in_one_batch_owe_each_old_partition_once() {
        let rows = [
            rows_of(1, DocOp::Delete, &[("a", "1")]),
            row(2, DocOp::Upsert, &["a"]),
            rows_of(3, DocOp::Delete, &[("a", "2"), ("b", "1")]),
            row(4, DocOp::Upsert, &["a"]),
            rows_of(5, DocOp::Delete, &[("a", "1")]),
            row(6, DocOp::Upsert, &["a"]),
        ];
        let plan = fold(&rows, true);
        assert_eq!(plan.upserts, vec![upsert("a", 6)]);
        assert_eq!(
            plan.deletes,
            vec![
                delete("a", Some("1"), 5),
                delete("a", Some("2"), 3),
                delete("b", Some("1"), 3),
            ]
        );
    }

    #[test]
    fn a_bare_delete_of_a_partitioned_table_aims_at_the_tables_own_namespace() {
        let rows = [
            row(1, DocOp::Delete, &["7"]),
            rows_of(2, DocOp::Delete, &[("8", "2")]),
            row(3, DocOp::Upsert, &["9"]),
            row(4, DocOp::Delete, &["9"]),
        ];
        let plan = fold(&rows, true);
        assert_eq!(
            plan.deletes,
            vec![
                delete("7", None, 1),
                delete("8", Some("2"), 2),
                delete("9", None, 4)
            ]
        );
        assert_eq!(
            plan.upserts,
            vec![upsert("9", 3)],
            "it says nothing about the row"
        );
    }

    // ---- which rows may be deleted ----

    /// Every id of `done` settled, as an upsert and as a delete of the table's own namespace.
    fn progress(done: &[&str], delete_all: bool) -> Progress {
        let mut progress = Progress::default();
        progress.upserts_done(done.iter().map(|id| id.to_string()));
        progress.deletes_done(done.iter().map(|id| (None, id.to_string())));
        progress.delete_all_done = delete_all;
        progress
    }

    #[test]
    fn a_row_is_complete_once_every_id_it_owns_is_delivered() {
        let rows = [
            row(1, DocOp::Upsert, &["a", "b"]),
            row(2, DocOp::Upsert, &["c"]),
        ];
        let plan = fold(&rows, false);
        assert!(complete_rows(&rows, &plan, &progress(&[], false)).is_empty());
        assert_eq!(
            complete_rows(&rows, &plan, &progress(&["a", "b"], false)),
            [1]
        );
        assert_eq!(
            complete_rows(&rows, &plan, &progress(&["a", "b", "c"], false)),
            [1, 2]
        );
        assert_eq!(
            complete_rows(&rows, &plan, &progress(&["a", "c"], false)),
            [2]
        );
    }

    #[test]
    fn a_superseded_mention_is_complete_at_once_but_its_successor_is_not() {
        let rows = [row(1, DocOp::Upsert, &["a"]), row(2, DocOp::Delete, &["a"])];
        let plan = fold(&rows, false);
        assert_eq!(complete_rows(&rows, &plan, &progress(&[], false)), [1]);
        assert_eq!(
            complete_rows(&rows, &plan, &progress(&["a"], false)),
            [1, 2]
        );
    }

    #[test]
    fn rows_before_a_barrier_are_complete_and_the_barrier_waits_for_its_request() {
        let rows = [
            row(1, DocOp::Upsert, &["a"]),
            row(2, DocOp::DeleteAll, &[]),
            row(3, DocOp::Upsert, &["b"]),
        ];
        let plan = fold(&rows, false);
        assert_eq!(complete_rows(&rows, &plan, &progress(&[], false)), [1]);
        assert_eq!(complete_rows(&rows, &plan, &progress(&[], true)), [1, 2]);
        assert_eq!(
            complete_rows(&rows, &plan, &progress(&["b"], true)),
            [1, 2, 3]
        );
    }

    #[test]
    fn the_backfill_job_is_never_complete_on_its_own() {
        let rows = [row(1, DocOp::Backfill, &[]), row(2, DocOp::Upsert, &["a"])];
        let plan = fold(&rows, false);
        assert_eq!(complete_rows(&rows, &plan, &progress(&["a"], false)), [2]);
    }

    #[test]
    fn a_move_is_complete_when_both_its_delete_and_its_upsert_are_delivered() {
        let rows = [
            rows_of(1, DocOp::Delete, &[("a", "1")]),
            row(2, DocOp::Upsert, &["a"]),
        ];
        let plan = fold(&rows, true);
        let mut progress = Progress::default();
        assert!(complete_rows(&rows, &plan, &progress).is_empty());
        progress.upserts_done(["a".to_string()]);
        assert_eq!(complete_rows(&rows, &plan, &progress), [2]);
        progress.deletes_done([(Some("2".to_string()), "a".to_string())]);
        assert_eq!(
            complete_rows(&rows, &plan, &progress),
            [2],
            "another partition's"
        );
        progress.deletes_done([(Some("1".to_string()), "a".to_string())]);
        assert_eq!(complete_rows(&rows, &plan, &progress), [1, 2]);
    }

    // ---- packing ----

    #[test]
    fn packing_respects_the_document_count_and_the_byte_budget() {
        let docs: Vec<Doc> = (0..5).map(|n| doc(&n.to_string(), 100)).collect();
        let one = docs[0].bytes;
        let by_count = pack(docs, 2, usize::MAX);
        let sizes: Vec<usize> = by_count.iter().map(Vec::len).collect();
        assert_eq!(sizes, [2, 2, 1]);

        let docs: Vec<Doc> = (0..5).map(|n| doc(&n.to_string(), 100)).collect();
        let by_bytes = pack(docs, 1000, one * 2 + one / 2);
        let sizes: Vec<usize> = by_bytes.iter().map(Vec::len).collect();
        assert_eq!(sizes, [2, 2, 1]);
        let order: Vec<&str> = by_bytes.iter().flatten().map(|d| d.id.as_str()).collect();
        assert_eq!(order, ["0", "1", "2", "3", "4"]);
    }

    #[test]
    fn a_document_over_the_byte_budget_travels_alone() {
        let docs = vec![doc("a", 10), doc("big", 5_000), doc("b", 10)];
        let packed = pack(docs, 1000, 1_000);
        let sizes: Vec<usize> = packed.iter().map(Vec::len).collect();
        assert_eq!(sizes, [1, 1, 1]);
        assert!(pack(Vec::new(), 10, 10).is_empty());
    }

    // ---- bisecting a rejected request ----

    fn bad(message: &str) -> ClientError {
        ClientError::BadRequest {
            message: message.to_string(),
        }
    }

    /// Rejects every request holding a document whose id is in `poison`.
    fn sender<'a>(
        poison: &'a [&'a str],
        requests: &'a mut Vec<Vec<String>>,
    ) -> impl FnMut(&[Doc]) -> Result<(), ClientError> + 'a {
        move |docs: &[Doc]| {
            requests.push(docs.iter().map(|d| d.id.clone()).collect());
            match docs.iter().find(|d| poison.contains(&d.id.as_str())) {
                Some(culprit) => Err(bad(&format!("document {} is invalid", culprit.id))),
                None => Ok(()),
            }
        }
    }

    #[test]
    fn a_healthy_request_is_sent_once() {
        let docs: Vec<Doc> = ["1", "2", "3"].iter().map(|id| doc(id, 1)).collect();
        let (mut requests, mut outcome) = (Vec::new(), UpsertOutcome::default());
        upsert_bisecting(&mut sender(&[], &mut requests), &docs, &mut outcome).expect("ok");
        assert_eq!(requests.len(), 1);
        assert_eq!(outcome.delivered, ids(&["1", "2", "3"]));
        assert!(outcome.dead.is_empty());
    }

    #[test]
    fn one_poison_document_is_dead_lettered_and_the_rest_delivered() {
        let docs: Vec<Doc> = ["1", "2", "3", "4", "5"]
            .iter()
            .map(|id| doc(id, 1))
            .collect();
        let (mut requests, mut outcome) = (Vec::new(), UpsertOutcome::default());
        upsert_bisecting(&mut sender(&["4"], &mut requests), &docs, &mut outcome).expect("ok");
        let mut delivered = outcome.delivered.clone();
        delivered.sort();
        assert_eq!(delivered, ids(&["1", "2", "3", "5"]));
        assert_eq!(
            outcome.dead,
            vec![("4".to_string(), "document 4 is invalid".to_string())]
        );
        assert!(
            requests.len() <= 9,
            "bisecting, not one by one: {requests:?}"
        );
    }

    #[test]
    fn a_single_rejected_document_is_a_dead_letter_not_an_error() {
        let docs = vec![doc("9", 1)];
        let (mut requests, mut outcome) = (Vec::new(), UpsertOutcome::default());
        upsert_bisecting(&mut sender(&["9"], &mut requests), &docs, &mut outcome).expect("ok");
        assert_eq!(requests.len(), 1);
        assert_eq!(outcome.dead.len(), 1);
        assert!(outcome.delivered.is_empty());
    }

    #[test]
    fn every_document_rejected_ends_with_one_dead_letter_each() {
        let docs: Vec<Doc> = ["1", "2", "3"].iter().map(|id| doc(id, 1)).collect();
        let (mut requests, mut outcome) = (Vec::new(), UpsertOutcome::default());
        upsert_bisecting(
            &mut sender(&["1", "2", "3"], &mut requests),
            &docs,
            &mut outcome,
        )
        .expect("ok");
        assert_eq!(outcome.dead.len(), 3);
        assert!(outcome.delivered.is_empty());
    }

    #[test]
    fn a_lane_error_stops_the_bisection_and_keeps_what_was_delivered() {
        let docs: Vec<Doc> = ["1", "2", "3", "4"].iter().map(|id| doc(id, 1)).collect();
        let mut calls = 0;
        let mut send = |batch: &[Doc]| -> Result<(), ClientError> {
            calls += 1;
            match (calls, batch.len()) {
                (1, _) => Err(bad("batch rejected")),
                (2, _) => Ok(()),
                _ => Err(ClientError::Auth),
            }
        };
        let mut outcome = UpsertOutcome::default();
        let error = upsert_bisecting(&mut send, &docs, &mut outcome).expect_err("auth");
        assert!(matches!(error, ClientError::Auth), "{error:?}");
        assert_eq!(outcome.delivered, ids(&["1", "2"]));
        assert!(outcome.dead.is_empty());
    }

    #[test]
    fn a_transient_error_is_returned_untouched() {
        let docs = vec![doc("1", 1), doc("2", 1)];
        let mut send = |_: &[Doc]| -> Result<(), ClientError> {
            Err(ClientError::Api {
                status: 429,
                message: "slow down".to_string(),
            })
        };
        let mut outcome = UpsertOutcome::default();
        let error = upsert_bisecting(&mut send, &docs, &mut outcome).expect_err("429");
        assert!(
            matches!(error, ClientError::Api { status: 429, .. }),
            "{error:?}"
        );
        assert!(outcome.delivered.is_empty() && outcome.dead.is_empty());
    }

    // ---- the circuit breaker ----

    const SCHEMA_DRIFT: &str = "field 'body' does not match the index schema";

    /// Refuses every request.
    fn refuse_all(requests: &mut Vec<usize>) -> impl FnMut(&[Doc]) -> Result<(), ClientError> + '_ {
        move |docs: &[Doc]| {
            requests.push(docs.len());
            Err(bad(SCHEMA_DRIFT))
        }
    }

    #[test]
    fn a_hundred_refusals_in_a_row_open_the_breaker_with_one_lane_error() {
        let docs: Vec<Doc> = (0..300).map(|n| doc(&n.to_string(), 1)).collect();
        let (mut requests, mut outcome) = (Vec::new(), UpsertOutcome::default());
        let error = upsert_bisecting(&mut refuse_all(&mut requests), &docs, &mut outcome)
            .expect_err("systemic");
        assert!(outcome.tripped);
        assert!(
            matches!(error, ClientError::BadRequest { .. }),
            "a permanent failure, so it counts towards parking: {error:?}"
        );
        let message = error.to_string();
        assert!(message.contains("100 documents in a row"), "{message}");
        assert!(message.contains(SCHEMA_DRIFT), "{message}");
        assert_eq!(outcome.dead.len(), 100, "held back for the caller to drop");
        assert!(requests.len() < 300, "{} requests", requests.len());
    }

    #[test]
    fn refusals_a_success_apart_never_reach_the_streak() {
        let docs: Vec<Doc> = (0..199).map(|n| doc(&n.to_string(), 1)).collect();
        let poison: Vec<String> = (0..199)
            .filter(|n| *n != 99)
            .map(|n| n.to_string())
            .collect();
        let poison: Vec<&str> = poison.iter().map(String::as_str).collect();
        let (mut requests, mut outcome) = (Vec::new(), UpsertOutcome::default());
        upsert_bisecting(&mut sender(&poison, &mut requests), &docs, &mut outcome)
            .expect("99 + accepted + 99 is not systemic");
        assert!(!outcome.tripped);
        assert_eq!(outcome.dead.len(), 198);
        assert_eq!(outcome.delivered, ids(&["99"]));
        assert!(outcome.check_all_refused().is_ok(), "one was accepted");
    }

    #[test]
    fn a_step_refused_as_a_whole_is_systemic_from_ten_documents_on() {
        for (count, systemic) in [(1usize, false), (9, false), (10, true), (40, true)] {
            let docs: Vec<Doc> = (0..count).map(|n| doc(&n.to_string(), 1)).collect();
            let (mut requests, mut outcome) = (Vec::new(), UpsertOutcome::default());
            upsert_bisecting(&mut refuse_all(&mut requests), &docs, &mut outcome)
                .expect("below the streak");
            let verdict = outcome.check_all_refused();
            assert_eq!(verdict.is_err(), systemic, "{count} refused");
            assert_eq!(outcome.tripped, systemic, "{count} refused");
            if let Err(error) = verdict {
                let message = error.to_string();
                assert!(
                    message.contains(&format!("all {count} documents")),
                    "{message}"
                );
            }
        }
    }

    #[test]
    fn the_counts_carry_over_between_the_requests_of_one_step() {
        let (mut requests, mut outcome) = (Vec::new(), UpsertOutcome::default());
        let first: Vec<Doc> = (0..6).map(|n| doc(&n.to_string(), 1)).collect();
        let second: Vec<Doc> = (6..12).map(|n| doc(&n.to_string(), 1)).collect();
        for batch in [&first, &second] {
            upsert_bisecting(&mut refuse_all(&mut requests), batch, &mut outcome).expect("ok");
        }
        assert!(
            outcome.check_all_refused().is_err(),
            "6 + 6 refused, none accepted"
        );
    }

    // ---- routing documents to namespaces ----

    fn doc_in(id: &str, partition: Option<&str>) -> Doc {
        doc(id, 1).in_partition(partition.map(str::to_string))
    }

    fn group_ids_of(groups: &[(String, Vec<Doc>)]) -> Vec<(String, Vec<String>)> {
        let ids = |docs: &Vec<Doc>| docs.iter().map(|d| d.id.clone()).collect();
        groups
            .iter()
            .map(|(ns, docs)| (ns.clone(), ids(docs)))
            .collect()
    }

    #[test]
    fn a_table_without_a_partition_column_sends_everything_to_its_namespace() {
        let docs = vec![
            doc_in("a", None),
            doc_in("b", Some("ignored")),
            doc_in("c", None),
        ];
        let groups = route_docs("public.t", false, docs);
        assert_eq!(
            group_ids_of(&groups),
            [("public.t".to_string(), ids(&["a", "b", "c"]))]
        );
    }

    #[test]
    fn documents_are_grouped_by_namespace_in_the_order_they_come() {
        let docs = vec![
            doc_in("a", Some("2")),
            doc_in("b", Some("Acme Corp")),
            doc_in("c", Some("2")),
            doc_in("d", Some("Acme Corp")),
            doc_in("e", Some("1")),
        ];
        let groups = route_docs("public.t", true, docs);
        assert_eq!(
            group_ids_of(&groups),
            [
                ("public.t:2".to_string(), ids(&["a", "c"])),
                ("public.t:Acme%20Corp".to_string(), ids(&["b", "d"])),
                ("public.t:1".to_string(), ids(&["e"])),
            ]
        );
    }

    #[test]
    fn a_namespace_is_the_prefix_plus_the_encoded_value_and_a_bare_delete_keeps_the_prefix() {
        assert_eq!(
            namespace_in("public.t", true, Some("a:b/c")),
            "public.t:a%3Ab%2Fc"
        );
        assert_eq!(namespace_in("public.t", true, Some("")), "public.t:");
        assert_eq!(namespace_in("public.t", true, None), "public.t");
        assert_eq!(namespace_in("public.t", false, Some("7")), "public.t");
    }

    // ---- cutting a backfill page ----

    fn at(id: &str) -> Cursor {
        Cursor {
            partition: None,
            id: id.to_string(),
        }
    }

    fn built(id: &str, bytes: usize) -> Built {
        let value = json!({"_id": id, "pad": "x".repeat(bytes)});
        Built {
            at: at(id),
            document: Ok(Doc::new(id, value)),
        }
    }

    fn refused(id: &str) -> Built {
        Built {
            at: at(id),
            document: Err("column \"body\" is too big".to_string()),
        }
    }

    fn page_ids(page: &Page) -> Vec<&str> {
        page.docs.iter().map(|d| d.id.as_str()).collect()
    }

    #[test]
    fn a_page_holds_every_row_that_fits_the_byte_budget() {
        let page = assemble_page([built("a", 10), built("b", 10), built("c", 10)], 10_000);
        assert_eq!(page_ids(&page), ["a", "b", "c"]);
        assert_eq!((page.rows, page.last), (3, Some(at("c"))));
        assert!(page.rejected.is_empty());
    }

    #[test]
    fn the_row_that_would_cross_the_budget_starts_the_next_page() {
        let one = Doc::new("a", json!({"_id": "a", "pad": "x".repeat(100)})).bytes;
        let rows = [built("a", 100), built("b", 100), built("c", 100)];
        let page = assemble_page(rows, one * 2 + 1);
        assert_eq!(page_ids(&page), ["a", "b"]);
        assert_eq!((page.rows, page.last), (2, Some(at("b"))));
    }

    #[test]
    fn a_page_always_takes_its_first_document_however_big() {
        let page = assemble_page([built("a", 5_000), built("b", 10)], 100);
        assert_eq!(page_ids(&page), ["a"]);
        assert_eq!(page.last, Some(at("a")));
    }

    #[test]
    fn refused_rows_are_passed_by_the_cursor_and_cost_no_budget() {
        let rows = [refused("a"), built("b", 10), refused("c"), built("d", 10)];
        let page = assemble_page(rows, 10_000);
        assert_eq!(page_ids(&page), ["b", "d"]);
        let rejected: Vec<&str> = page.rejected.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(rejected, ["a", "c"]);
        assert_eq!((page.rows, page.last), (4, Some(at("d"))));
    }

    #[test]
    fn a_page_of_only_refused_rows_still_moves_the_cursor() {
        let page = assemble_page([refused("x"), refused("y")], 100);
        assert!(page.docs.is_empty());
        assert_eq!((page.rows, page.last), (2, Some(at("y"))));
    }

    #[test]
    fn an_empty_page_has_no_cursor_and_ends_the_job() {
        let page = assemble_page(Vec::<Built>::new(), 100);
        assert_eq!((page.rows, page.last), (0, None));
    }

    #[test]
    fn the_cursor_of_a_partitioned_page_carries_its_last_partition() {
        let last = Cursor {
            partition: Some("Acme Corp".to_string()),
            id: "7".to_string(),
        };
        let doc = Doc::new("7", json!({"_id": "7"})).in_partition(Some("Acme Corp".to_string()));
        let rows = [Built {
            at: last.clone(),
            document: Ok(doc),
        }];
        let page = assemble_page(rows, 100);
        assert_eq!(page.last, Some(last));
        assert_eq!(page.docs[0].partition.as_deref(), Some("Acme Corp"));
    }

    // ---- building a document from a row ----

    fn field(name: &str, on_oversize: OnOversize) -> TextField {
        TextField {
            name: name.to_string(),
            on_oversize,
        }
    }

    #[test]
    fn a_document_holds_the_id_the_present_text_and_the_typed_metadata() {
        let fields = [
            field("title", OnOversize::Truncate),
            field("body", OnOversize::Truncate),
        ];
        let texts = [Some("a title".to_string()), None];
        let meta = Some(json!({"author": "ann", "embedding": [0.5, 1.0]}));
        let built = build_document("7", &fields, &texts, meta).expect("document");
        assert_eq!(
            built,
            json!({"_id": "7", "title": "a title", "author": "ann", "embedding": [0.5, 1.0]})
        );
    }

    #[test]
    fn oversized_text_is_cut_and_the_document_is_marked() {
        let fields = [
            field("body", OnOversize::Truncate),
            field("title", OnOversize::Truncate),
        ];
        let texts = [Some("ab ".repeat(22_000)), Some("short".to_string())];
        let built = build_document("1", &fields, &texts, None).expect("document");
        let body = built["body"].as_str().expect("body");
        assert!(body.split_whitespace().count() <= 9_000, "{}", body.len());
        assert_eq!(built["pg_pinecone_truncated"], json!(true));
        assert_eq!(built["title"], json!("short"));
    }

    #[test]
    fn an_untouched_document_carries_no_marker() {
        let fields = [field("body", OnOversize::Truncate)];
        let built = build_document("1", &fields, &[Some("fine".to_string())], None).expect("doc");
        assert!(built.get("pg_pinecone_truncated").is_none(), "{built}");
    }

    #[test]
    fn the_error_policy_refuses_what_truncation_would_cut() {
        let fields = [field("body", OnOversize::Error)];
        let texts = [Some("ab ".repeat(22_000))];
        let message = build_document("42", &fields, &texts, None).expect_err("refused");
        assert!(message.contains("column \"body\""), "{message}");
        assert!(message.contains("66000 bytes"), "{message}");
        assert!(message.contains("100 KB"), "{message}");
        let fits = [Some("fits".to_string())];
        assert!(build_document("42", &fields, &fits, None).is_ok());
    }

    #[test]
    fn the_error_policy_blames_a_long_word_for_what_it_is() {
        let fields = [field("body", OnOversize::Error)];
        let texts = [Some(format!("see https://x/{} here", "u".repeat(300)))];
        let message = build_document("42", &fields, &texts, None).expect_err("refused");
        assert!(message.contains("contains a 310-byte token"), "{message}");
        assert!(message.contains("256 bytes per token"), "{message}");
        assert!(!message.contains("100 KB"), "{message}");
    }
}
