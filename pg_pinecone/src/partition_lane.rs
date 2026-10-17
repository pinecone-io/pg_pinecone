//! The partition column of the vector trigger lane: the declaration `promote` validates, the
//! routing of queued writes to namespaces and the registrations that follow a first write.

use std::collections::{HashMap, HashSet};

use pg_pinecone_client::Error as ClientError;
use pgrx::{
    PgRelation, pg_sys,
    prelude::*,
    spi::{Spi, quote_identifier, quote_literal, quote_qualified_identifier},
};
use serde_json::Value;

use crate::{
    backend::{self, error_with_code},
    catalog,
    docs_registry::as_owner,
    partition::{self, Lane, PartitionKind, PartitionSpec},
    partition_registry,
    registry::RegistryRow,
};

const NOTICE_AT: i64 = 100;
const WARN_AT: i64 = 1000;
const PLAN_LIMITS: &str = "from 100 (Starter) to 1,000,000 (Enterprise) namespaces per index";

/// The refusal of a text partition column with a nondeterministic collation, shared by the lanes.
pub(crate) const NONDETERMINISTIC: &str = "has a nondeterministic collation: values it calls equal \
     ('acme' and 'ACME' under a case-insensitive one) would be one tenant to a WHERE clause and \
     two namespaces to Pinecone";
pub(crate) const NONDETERMINISTIC_HINT: &str =
    "use a deterministic collation: the default one, or COLLATE \"C\"";

/// The partition column `promote` stores for `tbl`: the requested one, else the one its registry
/// row holds, validated before anything is created. `None` for an unpartitioned table. `prefix`
/// is the namespace the values are appended to.
pub(crate) fn declare(
    tbl: &PgRelation,
    row: &RegistryRow,
    requested: Option<&str>,
    prefix: &str,
    func: &str,
) -> Option<PartitionSpec> {
    let column = requested
        .map(str::to_string)
        .or_else(|| row.partition.as_ref().map(|p| p.column.clone()))?;
    let relid = tbl.oid();
    let display = catalog::display_name(tbl);

    let kind = check_column(relid, &display, &column, func, true);
    if column == row.id_column {
        error_with_code(
            PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
            format!("{func}: partition column \"{column}\" of {display} is its id column"),
            None,
            Some("partition by the tenant or group column, not by the row id"),
        )
    }
    if !catalog::has_unique_single_column_index(relid, &row.id_column) {
        error_with_code(
            PgSqlErrorCode::ERRCODE_FEATURE_NOT_SUPPORTED,
            format!(
                "{func}: id column \"{}\" of {display} is not unique by itself, and the rows of \
                 every partition share one id space",
                row.id_column
            ),
            None,
            Some(
                "add a table-wide unique column (an identity column, or a UNIQUE index on the id) \
                 or wait for per-partition ids",
            ),
        )
    }
    let scan = Scan {
        table_ident: &quote_qualified_identifier(tbl.namespace(), tbl.name()),
        display: &display,
        prefix,
        id_expr: &quote_identifier(&row.id_column),
        id_label: "id",
    };
    announce_namespaces(&scan, &column, kind, func);
    Some(PartitionSpec { column, kind })
}

/// What any lane asks of a partition column: it exists, has a type with a canonical namespace
/// name (`uuid` only where `uuid_ok`: the access method's filter opclasses have none), is
/// NOT NULL, since a NULL has no namespace, and, as text, has a deterministic collation: Pinecone
/// compares namespace names byte for byte. Returns its kind.
pub(crate) fn check_column(
    relid: pg_sys::Oid,
    display: &str,
    column: &str,
    func: &str,
    uuid_ok: bool,
) -> PartitionKind {
    let Some(facts) = column_facts(relid, column) else {
        error_with_code(
            PgSqlErrorCode::ERRCODE_UNDEFINED_COLUMN,
            format!("{func}: partition column \"{column}\" does not exist on {display}"),
            None,
            None,
        )
    };
    let kind = PartitionKind::from_typname(&facts.typname)
        .filter(|kind| uuid_ok || *kind != PartitionKind::Uuid);
    let Some(kind) = kind else {
        let (tail, hint) = if uuid_ok {
            (
                "which has no canonical namespace name",
                "partition_column takes a smallint, integer, bigint, text, varchar or uuid column",
            )
        } else {
            (
                "which the access method cannot partition on",
                "a pg_pinecone index partitions on a smallint, integer, bigint, text or varchar \
                 column (uuid works with pinecone.promote and pinecone.track_docs)",
            )
        };
        error_with_code(
            PgSqlErrorCode::ERRCODE_FEATURE_NOT_SUPPORTED,
            format!(
                "{func}: partition column \"{column}\" of {display} has type {}, {tail}",
                facts.described
            ),
            None,
            Some(hint),
        )
    };
    if !facts.not_null {
        error_with_code(
            PgSqlErrorCode::ERRCODE_INVALID_TABLE_DEFINITION,
            format!(
                "{func}: partition column \"{column}\" of {display} must be NOT NULL: a NULL has \
                 no namespace"
            ),
            None,
            Some(&format!(
                "ALTER TABLE {display} ALTER COLUMN \"{column}\" SET NOT NULL"
            )),
        )
    }
    if kind == PartitionKind::Text && !catalog::column_is_deterministic(relid, column) {
        error_with_code(
            PgSqlErrorCode::ERRCODE_FEATURE_NOT_SUPPORTED,
            format!("{func}: partition column \"{column}\" of {display} {NONDETERMINISTIC}"),
            None,
            Some(NONDETERMINISTIC_HINT),
        )
    }
    kind
}

struct ColumnFacts {
    typname: String,
    described: String,
    not_null: bool,
}

fn column_facts(relid: pg_sys::Oid, column: &str) -> Option<ColumnFacts> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT t.typname::text AS typname, \
                        pg_catalog.format_type(a.atttypid, a.atttypmod) AS described, \
                        a.attnotnull AS not_null \
                 FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_type t ON t.oid = a.atttypid \
                 WHERE a.attrelid = $1::oid AND a.attname = $2 AND a.attnum > 0 \
                   AND NOT a.attisdropped",
                None,
                &[relid.into(), column.into()],
            )
            .expect("pg_pinecone: partition column catalog query failed")
            .next()
            .map(|row| ColumnFacts {
                typname: read(&row, "typname"),
                described: read(&row, "described"),
                not_null: read(&row, "not_null"),
            })
    })
}

fn read<T: FromDatum + IntoDatum>(row: &pgrx::spi::SpiHeapTupleData<'_>, name: &str) -> T {
    row.get_by_name::<T, _>(name)
        .unwrap_or_else(|e| panic!("pg_pinecone: partition column catalog {name} failed: {e}"))
        .unwrap_or_else(|| panic!("pg_pinecone: partition column catalog {name} unexpectedly NULL"))
}

/// What a declaration reads of the table before it creates anything.
pub(crate) struct Scan<'a> {
    /// The table, quoted and qualified.
    pub(crate) table_ident: &'a str,
    /// The table's name in messages.
    pub(crate) display: &'a str,
    /// The namespace the values are appended to.
    pub(crate) prefix: &'a str,
    /// A row's id as an SQL expression and what a message calls it: the quoted id column (`id`),
    /// or `ctid` (`row`) for the access method, whose ids are TIDs.
    pub(crate) id_expr: &'a str,
    pub(crate) id_label: &'a str,
}

/// Tells how many namespaces `column` of the table will make: a NOTICE with the number above 100,
/// and above 1,000 a WARNING with the plan limits. In the same scan a text value whose namespace
/// passes 512 bytes is refused (54000, naming a row that holds it): its first backfill chunk
/// would fail for good, hours later and with no row named.
pub(crate) fn announce_namespaces(scan: &Scan, column: &str, kind: PartitionKind, func: &str) {
    let ident = quote_identifier(column);
    let (values, longest) = read_values(scan, &ident, kind);
    if longest > partition::MAX_NAMESPACE_BYTES {
        refuse_long_value(scan, column, &ident, func);
    }
    let display = scan.display;
    if values > NOTICE_AT {
        notice!(
            "{func}: \"{column}\" of {display} has {values} distinct values: each gets its own \
             Pinecone namespace"
        );
    }
    if values > WARN_AT {
        warning!(
            "{func}: {values} partition values make {values} namespaces; Pinecone's plans hold \
             {PLAN_LIMITS}"
        );
    }
}

/// How many distinct values the column holds and, for text, the longest namespace they make
/// (measured once per distinct value; integers and uuids cannot pass the limit).
fn read_values(scan: &Scan, ident: &str, kind: PartitionKind) -> (i64, usize) {
    let table = scan.table_ident;
    if kind != PartitionKind::Text {
        let sql = format!("SELECT pg_catalog.count(DISTINCT {ident}) FROM {table}");
        let values = Spi::get_one::<i64>(&sql)
            .expect("pg_pinecone: partition value count query failed")
            .unwrap_or(0);
        return (values, 0);
    }
    let sql = format!(
        "SELECT pg_catalog.count(*) AS n, \
                pg_catalog.max(pinecone._partition_namespace_bytes($1, d.v::pg_catalog.text)) AS longest \
         FROM (SELECT DISTINCT {ident} AS v FROM {table}) d"
    );
    Spi::connect(|client| {
        let mut rows = client
            .select(&sql, Some(1), &[scan.prefix.into()])
            .expect("pg_pinecone: partition value count query failed");
        let row = rows
            .next()
            .expect("pg_pinecone: an aggregate returns a row");
        let values = read_optional::<i64>(&row, "n").unwrap_or(0);
        let longest = read_optional::<i32>(&row, "longest").unwrap_or(0);
        (values, usize::try_from(longest).unwrap_or(0))
    })
}

fn read_optional<T: FromDatum + IntoDatum>(
    row: &pgrx::spi::SpiHeapTupleData<'_>,
    name: &str,
) -> Option<T> {
    row.get_by_name::<T, _>(name)
        .unwrap_or_else(|e| panic!("pg_pinecone: partition value query {name} failed: {e}"))
}

/// Raises 54000 naming the first row whose value makes a namespace over the limit.
fn refuse_long_value(scan: &Scan, column: &str, ident: &str, func: &str) -> ! {
    let limit = partition::MAX_NAMESPACE_BYTES;
    let sql = format!(
        "SELECT {}::pg_catalog.text AS id, \
                pinecone._partition_namespace_bytes($1, {ident}::pg_catalog.text) AS bytes \
         FROM {} WHERE pinecone._partition_namespace_bytes($1, {ident}::pg_catalog.text) > {limit} \
         LIMIT 1",
        scan.id_expr, scan.table_ident
    );
    let found = Spi::connect(|client| {
        let mut rows = client
            .select(&sql, Some(1), &[scan.prefix.into()])
            .expect("pg_pinecone: partition value length query failed");
        rows.next().map(|row| {
            (
                read_optional::<String>(&row, "id").unwrap_or_default(),
                read_optional::<i32>(&row, "bytes").unwrap_or(0),
            )
        })
    });
    let display = scan.display;
    let which = match found {
        Some((id, bytes)) => format!(
            "{} \"{id}\" of {display} has a \"{column}\" value that makes a namespace of {bytes} bytes",
            scan.id_label
        ),
        None => format!(
            "a row of {display} has a \"{column}\" value that makes a namespace over {limit} bytes"
        ),
    };
    error_with_code(
        PgSqlErrorCode::ERRCODE_PROGRAM_LIMIT_EXCEEDED,
        format!("{func}: {which}; Pinecone allows {limit}"),
        None,
        Some(&format!(
            "shorten the value of \"{column}\" in that row and any other over the limit (every \
             byte outside A-Za-z0-9._- counts three in a namespace name)"
        )),
    )
}

/// A queued delete: a bare id (unpartitioned table, or queued before the table was partitioned)
/// or `{id, p}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeleteEntry {
    pub(crate) id: String,
    pub(crate) p: Option<String>,
}

#[derive(Debug)]
pub(crate) struct UpsertGroup {
    pub(crate) namespace: String,
    pub(crate) value: String,
    pub(crate) items: Vec<Value>,
}

/// Splits upsert items by the namespace of their partition value `p`, in order of first
/// appearance. A namespace Pinecone refuses cannot be written, so that is an error.
pub(crate) fn group_upserts(
    prefix: &str,
    items: Vec<Value>,
) -> Result<Vec<UpsertGroup>, ClientError> {
    let mut groups: Vec<UpsertGroup> = Vec::new();
    let mut at: HashMap<String, usize> = HashMap::new();
    for item in items {
        let Some(value) = item.get("p").and_then(Value::as_str).map(str::to_string) else {
            return Err(ClientError::BadRequest {
                message: format!(
                    "pg_pinecone: an upsert of a partitioned table lacks its partition value \"p\": \
                     {item}"
                ),
            });
        };
        let namespace = partition::namespace_for(prefix, &value);
        let slot = match at.get(&namespace) {
            Some(&slot) => slot,
            None => {
                partition::fits_pinecone(&namespace).map_err(|e| ClientError::BadRequest {
                    message: format!("pg_pinecone: the partition value makes a bad namespace: {e}"),
                })?;
                groups.push(UpsertGroup {
                    namespace: namespace.clone(),
                    value,
                    items: Vec::new(),
                });
                at.insert(namespace, groups.len() - 1);
                groups.len() - 1
            }
        };
        groups[slot].items.push(item);
    }
    Ok(groups)
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DeleteGroup {
    pub(crate) namespace: String,
    /// The partition value; `None` for the bare ids of the table's own namespace.
    pub(crate) value: Option<String>,
    pub(crate) ids: Vec<String>,
}

/// Splits deletes by namespace, in order of first appearance. A bare id goes to the table's own
/// namespace, where it was written before the table was partitioned; an entry for a namespace
/// Pinecone would refuse holds nothing and is dropped.
pub(crate) fn group_deletes(prefix: &str, entries: Vec<DeleteEntry>) -> Vec<DeleteGroup> {
    let mut groups: Vec<DeleteGroup> = Vec::new();
    for entry in entries {
        let namespace = match &entry.p {
            Some(p) => partition::namespace_for(prefix, p),
            None => prefix.to_string(),
        };
        if partition::fits_pinecone(&namespace).is_err() {
            continue;
        }
        match groups.iter_mut().find(|g| g.namespace == namespace) {
            Some(group) => group.ids.push(entry.id),
            None => groups.push(DeleteGroup {
                namespace,
                value: entry.p,
                ids: vec![entry.id],
            }),
        }
    }
    groups
}

/// Which partitions a drain step or a backfill job sends to: not the failed ones, and for a job
/// scoped to one partition (the rebuild of a retry) that one alone. The failed set is read from
/// the registry when first asked and kept current as the step marks or wipes.
///
/// A refusal that repeats what a failed partition of the table says is the table's, not a
/// tenant's (a dimension that does not fit the index, say): marking tenant after tenant would fail
/// the whole table while its queue looks healthy. The gate hands it back, so the row fails as a
/// table's does and parks after `pinecone.outbox_max_attempts`.
///
/// Only the worker's drain isolates a refusal. A `sync_mode = 'sync'` writer is the statement that
/// wrote the row, with no shared queue to park, and the mode exists so that a refused write is
/// visible: its gate neither marks nor skips, and the refusal reaches the writer.
pub(crate) struct Gate {
    isolate: bool,
    only: Option<String>,
    failed: Option<HashSet<String>>,
    /// What the failed partitions say ([`reason`]), read when a refusal first asks.
    reasons: Option<HashSet<String>>,
}

impl Gate {
    pub(crate) fn new(isolate: bool) -> Self {
        Self {
            isolate,
            only: None,
            failed: None,
            reasons: None,
        }
    }

    /// Limits the gate to `value` until called again with `None`.
    pub(crate) fn scope(&mut self, value: Option<String>) {
        self.only = value;
    }

    pub(crate) fn scoped_to(&self) -> Option<&str> {
        self.only.as_deref()
    }

    pub(crate) fn skips(&mut self, relid: pg_sys::Oid, value: &str) -> bool {
        self.skips_with(value, || {
            partition_registry::failed_values(relid, Lane::Vector)
        })
    }

    fn skips_with(&mut self, value: &str, load: impl FnOnce() -> HashSet<String>) -> bool {
        if self.only.as_deref().is_some_and(|only| only != value) {
            return true;
        }
        if !self.isolate {
            return false;
        }
        self.failed.get_or_insert_with(load).contains(value)
    }

    /// Settles a failed request for `namespace`: a refusal of that namespace alone marks the
    /// partition failed, warns once and lets the lane go on; any other error is handed back, and
    /// so is a refusal that repeats what a failed partition of the table says.
    pub(crate) fn refused(
        &mut self,
        relid: pg_sys::Oid,
        value: &str,
        namespace: &str,
        err: ClientError,
    ) -> Result<(), ClientError> {
        if !self.isolate || !backend::fails_one_partition(&err) {
            return Err(err);
        }
        let text = backend::failure_text(&err);
        let said = reason(namespace, &text);
        let repeated = self.repeats_with(&text, &said, || {
            let failed = partition_registry::failed_of(relid, Lane::Vector);
            let says = failed.iter().map(|f| reason(&f.namespace, &f.last_error));
            says.collect()
        });
        if repeated {
            return Err(err);
        }
        partition_registry::mark_failed(relid, Lane::Vector, value, namespace, &text);
        self.remember(value, said);
        let table = catalog::display_name_of(relid);
        warning!(
            "pg_pinecone: partition {} ({namespace}) of {table} failed for good and is skipped until \
             it is retried: {text} — pinecone.retry_failed_partition({}, {}) rebuilds it from the table",
            quote_literal(value),
            quote_literal(&table),
            quote_literal(value)
        );
        Ok(())
    }

    /// Whether a refusal repeats a failed partition's. The plan's limit of namespaces never does:
    /// each new tenant meets it on its own.
    fn repeats_with(
        &mut self,
        text: &str,
        said: &str,
        load: impl FnOnce() -> HashSet<String>,
    ) -> bool {
        if text.contains(NEW_NAMESPACE_MARK) {
            return false;
        }
        self.reasons.get_or_insert_with(load).contains(said)
    }

    fn remember(&mut self, value: &str, said: String) {
        if let Some(failed) = &mut self.failed {
            failed.insert(value.to_string());
        }
        if let Some(reasons) = &mut self.reasons {
            reasons.insert(said);
        }
    }

    /// Every partition of the table was wiped and forgotten, the failed ones with them.
    pub(crate) fn wiped(&mut self) {
        if self.isolate {
            self.failed = Some(HashSet::new());
            self.reasons = Some(HashSet::new());
        }
    }
}

/// `text` with the namespace it names left out: two partitions refused for one cause read alike.
fn reason(namespace: &str, text: &str) -> String {
    text.replace(namespace, "<namespace>")
}

/// Registers `namespace` once something was written to it, explains a refusal of its first
/// write, names the namespace in any other error and hands the failure to the gate, which marks
/// the partition failed when only that namespace was refused (`wrote` says whether any request
/// succeeded, also when a later one failed).
pub(crate) fn settle_write(
    relid: pg_sys::Oid,
    value: &str,
    namespace: &str,
    wrote: bool,
    result: Result<(), ClientError>,
    gate: &mut Gate,
) -> Result<(), ClientError> {
    if wrote {
        partition_registry::register_partition(relid, Lane::Vector, value, namespace);
    }
    let Err(err) = result else {
        return Ok(());
    };
    let first = !wrote && !partition_registry::is_registered(relid, Lane::Vector, namespace);
    let err = if first {
        explain_first_write(err, namespace)
    } else {
        err
    };
    gate.refused(relid, value, namespace, name_namespace(err, namespace))
}

/// `err` with the namespace it came from, so a `last_error` (a failed partition's, or the table's
/// under a 429) names the tenant. Errors that say nothing about a namespace (credentials, a
/// timeout) stay as they are.
pub(crate) fn name_namespace(err: ClientError, namespace: &str) -> ClientError {
    let note = |text: String| {
        if text.contains(namespace) {
            text
        } else {
            format!("{text} — namespace '{namespace}'")
        }
    };
    match err {
        ClientError::Api { status, message } => ClientError::Api {
            status,
            message: note(message),
        },
        ClientError::BadRequest { message } => ClientError::BadRequest {
            message: note(message),
        },
        ClientError::Transport { url, reason } => ClientError::Transport {
            url,
            reason: note(reason),
        },
        other => other,
    }
}

pub(crate) fn count_registered(relid: pg_sys::Oid, lane: Lane) -> i64 {
    as_owner(|| {
        Spi::get_one_with_args::<i64>(
            "SELECT count(*) FROM pinecone.partitions WHERE relid = $1::oid::regclass AND lane = $2",
            &[relid.into(), lane.as_str().into()],
        )
    })
    .expect("pg_pinecone: partition registry count failed")
    .unwrap_or(0)
}

/// Text that `backend::raise` and `backend::failure_text` turn into [`NAMESPACE_LIMIT_HINT`].
pub(crate) const NEW_NAMESPACE_MARK: &str = "while writing the first";

pub(crate) const NAMESPACE_LIMIT_HINT: &str = "if this is Pinecone's plan limit on namespaces per \
     index (100 on Starter up to 1,000,000 on Enterprise), delete namespaces you no longer need \
     or upgrade the plan; the namespaces already written keep working";

/// Marks a refusal of a namespace's first write that reads like a limit.
pub(crate) fn explain_first_write(err: ClientError, namespace: &str) -> ClientError {
    mark_first_write(err, namespace, "vector", reads_like_a_limit)
}

/// [`explain_first_write`] for a document: the message has to name the namespace limit, since a
/// refused document is a 400 too.
pub(crate) fn explain_first_doc_write(err: ClientError, namespace: &str) -> ClientError {
    mark_first_write(
        err,
        namespace,
        "document",
        pg_pinecone_client::names_namespace_limit,
    )
}

/// Whether `err` is a refusal that names the namespace limit.
pub(crate) fn names_the_limit(err: &ClientError) -> bool {
    match err {
        ClientError::BadRequest { message } | ClientError::Api { message, .. } => {
            pg_pinecone_client::names_namespace_limit(message)
        }
        _ => false,
    }
}

fn mark_first_write(
    err: ClientError,
    namespace: &str,
    item: &str,
    limit: fn(&str) -> bool,
) -> ClientError {
    let note = |message: String| {
        format!("{message} — {NEW_NAMESPACE_MARK} {item} of partition namespace '{namespace}'")
    };
    match err {
        ClientError::BadRequest { message } if limit(&message) => ClientError::BadRequest {
            message: note(message),
        },
        ClientError::Api { status, message }
            if (400..500).contains(&status) && status != 429 && limit(&message) =>
        {
            ClientError::Api {
                status,
                message: note(message),
            }
        }
        other => other,
    }
}

fn reads_like_a_limit(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    ["namespace", "limit", "quota", "exceed"]
        .iter()
        .any(|word| message.contains(word))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const PREFIX: &str = "public.docs";

    fn upsert(id: &str, p: &str) -> Value {
        json!({"id": id, "values": [1.0], "p": p})
    }

    fn entry(id: &str, p: Option<&str>) -> DeleteEntry {
        DeleteEntry {
            id: id.to_string(),
            p: p.map(str::to_string),
        }
    }

    fn is_marked(err: &ClientError) -> bool {
        match err {
            ClientError::BadRequest { message } | ClientError::Api { message, .. } => {
                message.contains(NEW_NAMESPACE_MARK)
            }
            _ => false,
        }
    }

    #[test]
    fn upserts_group_by_namespace_in_first_seen_order_and_keep_their_own_order() {
        let items = vec![
            upsert("1", "b"),
            upsert("2", "a"),
            upsert("3", "b"),
            upsert("4", "a b"),
        ];

        let groups = group_upserts(PREFIX, items).expect("groups");

        let shape: Vec<(&str, &str, Vec<&str>)> = groups
            .iter()
            .map(|g| {
                let ids = g.items.iter().filter_map(|i| i["id"].as_str());
                (g.namespace.as_str(), g.value.as_str(), ids.collect())
            })
            .collect();
        assert_eq!(
            shape,
            [
                ("public.docs:b", "b", vec!["1", "3"]),
                ("public.docs:a", "a", vec!["2"]),
                ("public.docs:a%20b", "a b", vec!["4"]),
            ]
        );
    }

    #[test]
    fn an_upsert_without_a_partition_value_is_refused() {
        for item in [
            json!({"id": "1"}),
            json!({"id": "1", "p": 7}),
            json!({"p": null}),
        ] {
            let err = group_upserts(PREFIX, vec![item]).expect_err("no usable p");
            assert!(
                matches!(&err, ClientError::BadRequest { message } if message.contains("\"p\"")),
                "{err:?}"
            );
        }
    }

    #[test]
    fn an_upsert_whose_namespace_is_too_long_is_refused_with_the_limit() {
        let fits = "x".repeat(512 - PREFIX.len() - 1);
        assert!(group_upserts(PREFIX, vec![upsert("1", &fits)]).is_ok());

        let err = group_upserts(PREFIX, vec![upsert("1", &format!("{fits}x"))]).expect_err("513");
        assert!(
            matches!(&err, ClientError::BadRequest { message } if message.contains("512")),
            "{err:?}"
        );
    }

    #[test]
    fn deletes_group_by_namespace_and_send_a_bare_id_to_the_table_namespace() {
        let entries = vec![
            entry("1", Some("1")),
            entry("7", None),
            entry("2", Some("2")),
            entry("3", Some("1")),
        ];

        let groups = group_deletes(PREFIX, entries);

        let group = |namespace: &str, value: Option<&str>, ids: &[&str]| DeleteGroup {
            namespace: namespace.to_string(),
            value: value.map(str::to_string),
            ids: ids.iter().map(|id| id.to_string()).collect(),
        };
        assert_eq!(
            groups,
            [
                group("public.docs:1", Some("1"), &["1", "3"]),
                group("public.docs", None, &["7"]),
                group("public.docs:2", Some("2"), &["2"]),
            ]
        );
    }

    #[test]
    fn a_delete_into_a_namespace_pinecone_would_refuse_is_dropped() {
        let too_long = "x".repeat(600);
        let entries = vec![entry("1", Some(&too_long)), entry("2", Some("ok"))];

        let groups = group_deletes(PREFIX, entries);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].namespace, "public.docs:ok");
    }

    #[test]
    fn a_refused_first_write_is_marked_only_when_the_message_reads_like_a_limit() {
        let bad = |m: &str| ClientError::BadRequest {
            message: m.to_string(),
        };
        let api = |status: u16, m: &str| ClientError::Api {
            status,
            message: m.to_string(),
        };

        assert!(is_marked(&explain_first_write(
            bad("Namespace limit reached"),
            "ns"
        )));
        assert!(is_marked(&explain_first_write(
            api(403, "quota exceeded"),
            "ns"
        )));
        let dimension = bad("Vector dimension 3 does not match the dimension of the index 2");
        assert!(!is_marked(&explain_first_write(dimension, "ns")));
        assert!(!is_marked(&explain_first_write(
            api(503, "namespace unavailable"),
            "ns"
        )));
        assert!(!is_marked(&explain_first_write(ClientError::Auth, "ns")));
    }

    #[test]
    fn a_refused_first_document_is_marked_only_when_it_names_the_namespace_limit() {
        let bad = |m: &str| ClientError::BadRequest {
            message: m.to_string(),
        };
        let api = |status: u16, m: &str| ClientError::Api {
            status,
            message: m.to_string(),
        };
        let limit = "You've reached the maximum number of namespaces for this index (100).";

        assert!(is_marked(&explain_first_doc_write(bad(limit), "ns")));
        assert!(is_marked(&explain_first_doc_write(api(403, limit), "ns")));
        let marked = explain_first_doc_write(bad(limit), "public.t:801");
        assert!(names_the_limit(&marked));
        assert!(
            format!("{marked}")
                .contains("the first document of partition namespace 'public.t:801'")
        );
        assert!(!is_marked(&explain_first_doc_write(
            bad("Field body exceeds the limit of 100 KB"),
            "ns"
        )));
        assert!(!is_marked(&explain_first_doc_write(api(503, limit), "ns")));
    }

    fn failed(values: &[&str]) -> HashSet<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    #[test]
    fn a_gate_that_does_not_isolate_skips_nothing_and_reads_nothing() {
        let mut gate = Gate::new(false);

        let skipped = gate.skips_with("2", || panic!("no registry read"));

        assert!(!skipped);
    }

    #[test]
    fn an_isolating_gate_skips_the_failed_partitions_and_reads_them_once() {
        let mut gate = Gate::new(true);
        let mut reads = 0;
        let mut load = |values: &[&str]| {
            reads += 1;
            failed(values)
        };

        assert!(gate.skips_with("2", || load(&["2", "7"])));
        assert!(gate.skips_with("7", || panic!("read twice")));
        assert!(!gate.skips_with("3", || panic!("read twice")));
        assert_eq!(reads, 1);
    }

    #[test]
    fn a_wipe_forgets_every_failed_partition_without_a_new_read() {
        let mut gate = Gate::new(true);
        assert!(gate.skips_with("2", || failed(&["2"])));

        gate.wiped();

        assert!(!gate.skips_with("2", || panic!("a wipe leaves nothing to read")));
    }

    #[test]
    fn a_scoped_gate_skips_every_partition_but_its_own() {
        let mut gate = Gate::new(true);
        gate.scope(Some("2".to_string()));

        assert_eq!(gate.scoped_to(), Some("2"));
        assert!(gate.skips_with("3", || panic!("another partition is out before any read")));
        assert!(!gate.skips_with("2", || failed(&[])));

        gate.scope(None);
        assert_eq!(gate.scoped_to(), None);
        assert!(!gate.skips_with("3", || failed(&[])));
    }

    #[test]
    fn a_scoped_gate_still_skips_its_own_partition_when_that_has_failed_again() {
        let mut gate = Gate::new(true);
        gate.scope(Some("2".to_string()));

        assert!(gate.skips_with("2", || failed(&["2"])));
    }

    const DIMENSION: &str = "Vector dimension 8 does not match the dimension of the index 2";

    fn says(namespace: &str, cause: &str) -> String {
        let text = format!("bad request: {cause} — namespace '{namespace}'");
        reason(namespace, &text)
    }

    #[test]
    fn two_namespaces_refused_for_one_cause_read_alike() {
        assert_eq!(says("public.t:2", DIMENSION), says("public.t:3", DIMENSION));
        let refused = says("public.t:3", "tenant 3 is refused");
        assert_ne!(says("public.t:2", DIMENSION), refused);
        let own = |namespace: &str| {
            reason(
                namespace,
                &format!("bad request: namespace {namespace} is no good"),
            )
        };
        assert_eq!(
            own("public.t:2"),
            own("public.t:3"),
            "Pinecone names it itself"
        );
    }

    #[test]
    fn a_refusal_that_a_failed_partition_already_says_is_the_tables() {
        let mut gate = Gate::new(true);
        let mut reads = 0;
        let same = format!("bad request: {DIMENSION} — namespace 'public.t:3'");

        let repeated = gate.repeats_with(&same, &says("public.t:3", DIMENSION), || {
            reads += 1;
            HashSet::from([says("public.t:2", DIMENSION)])
        });

        assert!(repeated);
        let other = "bad request: tenant 3 is refused — namespace 'public.t:3'";
        let said = says("public.t:3", "tenant 3 is refused");
        assert!(!gate.repeats_with(other, &said, || panic!("read twice")));
        assert_eq!(reads, 1);
    }

    #[test]
    fn the_first_refusal_of_a_table_is_a_partitions() {
        let mut gate = Gate::new(true);
        let text = format!("bad request: {DIMENSION} — namespace 'public.t:2'");

        assert!(!gate.repeats_with(&text, &says("public.t:2", DIMENSION), HashSet::new));
    }

    #[test]
    fn the_plans_namespace_limit_is_each_new_tenants_own() {
        let mut gate = Gate::new(true);
        let limit = format!(
            "bad request: You've reached the maximum number of namespaces — {NEW_NAMESPACE_MARK} \
             vector of partition namespace 'public.t:801' — {NAMESPACE_LIMIT_HINT}"
        );
        let said = reason("public.t:801", &limit);

        assert!(!gate.repeats_with(&limit, &said, || panic!("a limit needs no set")));
    }

    #[test]
    fn a_marked_refusal_counts_from_then_on_and_a_wipe_forgets_it() {
        let mut gate = Gate::new(true);
        let text = format!("bad request: {DIMENSION} — namespace 'public.t:3'");
        let said = says("public.t:3", DIMENSION);
        assert!(!gate.repeats_with(&text, &said, HashSet::new));
        assert!(!gate.skips_with("2", HashSet::new));

        gate.remember("2", says("public.t:2", DIMENSION));
        assert!(gate.repeats_with(&text, &said, || panic!("read twice")));
        assert!(gate.skips_with("2", || panic!("the mark is remembered")));

        gate.wiped();
        assert!(!gate.repeats_with(&text, &said, || panic!("a wipe leaves nothing to read")));
    }

    #[test]
    fn an_error_names_its_namespace_once() {
        let named = |err: ClientError| format!("{}", name_namespace(err, "public.t:2"));

        let api = ClientError::Api {
            status: 503,
            message: "unavailable".to_string(),
        };
        assert_eq!(
            named(api),
            "index API returned unexpected status 503: unavailable — namespace 'public.t:2'"
        );
        let bad = ClientError::BadRequest {
            message: "bad id — while writing the first vector of partition namespace 'public.t:2'"
                .to_string(),
        };
        assert!(named(bad).matches("public.t:2").count() == 1);
        let transport = ClientError::Transport {
            url: "https://host".to_string(),
            reason: "connection reset".to_string(),
        };
        assert!(named(transport).ends_with("connection reset — namespace 'public.t:2'"));
        for unchanged in [ClientError::Auth, ClientError::Timeout { ms: 5 }] {
            let text = format!("{unchanged}");
            assert_eq!(named(unchanged), text);
        }
    }
}
