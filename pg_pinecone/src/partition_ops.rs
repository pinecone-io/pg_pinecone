//! What Pinecone holds under a partition family against what `pinecone.partitions` lists: the
//! orphan report of `pinecone.sync`, the family check of `pinecone.repoint`, the namespaces of
//! the values a table holds, and `pinecone.delete_partition`.

use std::collections::BTreeSet;

use pg_pinecone_client::{Client, DeleteDocsRequest, DocDeleteSelector, Error as ClientError};
use pgrx::{
    PgRelation, pg_sys,
    prelude::*,
    spi::{Spi, quote_literal},
};

use crate::{
    backend::{self, error_with_code},
    catalog, docs_drain, docs_outbox,
    docs_registry::{self, as_owner},
    docs_search::{self, PartitionArg},
    docs_track, outbox,
    partition::{self, Lane, NamespaceError, PartitionKind, PartitionSpec},
    partition_registry,
    registry::{self, RegistryRow},
};

/// Namespaces a report names one by one; the rest are counted.
const NAMED_AT_MOST: usize = 20;

/// The namespaces under a family prefix, sorted by what the registry knows of them.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Family {
    /// Held, not registered, and the encoding of a value of the column: `(value, namespace)`.
    pub(crate) unregistered: Vec<(String, String)>,
    /// Held, not registered, and no write of the table makes such a name.
    pub(crate) foreign: Vec<String>,
    /// Registered and not held.
    pub(crate) missing: Vec<String>,
    /// Held and registered.
    pub(crate) found: usize,
    /// Held under the family prefix, however registered.
    pub(crate) held: usize,
}

/// Sorts the `held` names under `prefix:` against the `registered` ones. A name outside the family
/// (a neighbour table's, or the prefix itself) is not looked at; one inside it is `unregistered`
/// when `kind` could have produced it and `foreign` when not.
pub(crate) fn classify(
    prefix: &str,
    kind: PartitionKind,
    held: &[String],
    registered: &[String],
) -> Family {
    let family = partition::family_prefix(prefix);
    let held: BTreeSet<&str> = held
        .iter()
        .map(String::as_str)
        .filter(|name| name.starts_with(&family))
        .collect();
    let known: BTreeSet<&str> = registered.iter().map(String::as_str).collect();
    let mut out = Family {
        held: held.len(),
        ..Family::default()
    };
    for name in &held {
        if known.contains(name) {
            out.found += 1;
            continue;
        }
        match partition::parse_value(prefix, name).filter(|value| kind.is_canonical(value)) {
            Some(value) => out.unregistered.push((value, (*name).to_string())),
            None => out.foreign.push((*name).to_string()),
        }
    }
    let missing = registered
        .iter()
        .filter(|name| !held.contains(name.as_str()));
    out.missing = missing.cloned().collect();
    out
}

/// Every namespace of `index` under `prefix:`, as Pinecone lists them.
pub(crate) fn held_under(
    client: &Client,
    index: &str,
    prefix: &str,
) -> Result<Vec<String>, ClientError> {
    let family = partition::family_prefix(prefix);
    let all = client.list_all_namespaces(index, Some(&family))?;
    Ok(all.into_iter().map(|info| info.name).collect())
}

fn registered_names(relid: pg_sys::Oid, lane: Lane) -> Vec<String> {
    let partitions = partition_registry::partitions_of(relid, lane);
    partitions
        .into_iter()
        .map(|(_, namespace)| namespace)
        .collect()
}

fn index_and_prefix(row: &RegistryRow) -> (&str, &str) {
    let index = row.index_name.as_deref();
    let prefix = row.namespace.as_deref();
    (
        index.expect("pg_pinecone: a partitioned row has an index"),
        prefix.expect("pg_pinecone: a partitioned row has a namespace"),
    )
}

/// Warns about the names under the family that no write of the table makes: they are not
/// touched, since nothing says whose they are.
fn warn_foreign(func: &str, display: &str, prefix: &str, spec: &PartitionSpec, foreign: &[String]) {
    for namespace in foreign.iter().take(NAMED_AT_MOST) {
        warning!(
            "{func}: namespace '{namespace}' is under '{prefix}:' but is not a name {display} writes \
             for \"{}\"; left alone",
            spec.column
        );
    }
    if foreign.len() > NAMED_AT_MOST {
        warning!(
            "{func}: {} more namespaces under '{prefix}:' are not names {display} writes; left alone",
            foreign.len() - NAMED_AT_MOST
        );
    }
}

/// The orphans of `tbl`'s family: namespaces Pinecone holds under `prefix:` that the registry
/// does not list (a write that rolled back after Pinecone took it, the leftovers of a dropped
/// table, or a registry that was lost). Each is named in a WARNING and registered, so the wipe
/// that follows (`sync`, a fresh `promote`) empties it with the rest; the load that follows
/// brings it back if the table has rows with its value. A failed listing raises: the wipe would
/// not know what to empty.
pub(crate) fn adopt_orphans(
    func: &str,
    tbl: &PgRelation,
    spec: &PartitionSpec,
    index: &str,
    prefix: &str,
    client: &Client,
) {
    try_adopt_orphans(func, tbl.oid(), spec, index, prefix, client)
        .unwrap_or_else(|e| backend::raise(e));
}

/// [`adopt_orphans`] for a caller that handles a failed listing itself: the queue's wipe returns
/// it, so the drain retries before anything is emptied.
pub(crate) fn try_adopt_orphans(
    func: &str,
    relid: pg_sys::Oid,
    spec: &PartitionSpec,
    index: &str,
    prefix: &str,
    client: &Client,
) -> Result<(), ClientError> {
    let display = catalog::display_name_of(relid);
    let held = held_under(client, index, prefix)?;
    let registered = registered_names(relid, Lane::Vector);
    let family = classify(prefix, spec.kind, &held, &registered);
    for (value, namespace) in family.unregistered.iter().take(NAMED_AT_MOST) {
        warning!(
            "{func}: orphan namespace '{namespace}' of {display}: Pinecone holds it and \
             pinecone.partitions does not; it is emptied with the rest and reloaded if rows with \
             \"{}\" = {} exist",
            spec.column,
            quote_literal(value)
        );
    }
    if family.unregistered.len() > NAMED_AT_MOST {
        warning!(
            "{func}: {} more orphan namespaces of {display}, handled the same way",
            family.unregistered.len() - NAMED_AT_MOST
        );
    }
    warn_foreign(func, &display, prefix, spec, &family.foreign);
    partition_registry::register_partitions(relid, Lane::Vector, &family.unregistered);
    Ok(())
}

/// What a restored `index` holds of `tbl`'s family, as the clause the repoint notice carries. It
/// warns about registered namespaces the index lacks (recovery finds nothing in them) and about
/// names pg_pinecone never writes, and registers the namespaces of the family the registry lacks, so
/// recovery and every wipe reach them. A failed listing is a warning, not an error: the registered
/// namespaces are recovered as they are.
pub(crate) fn check_family(
    func: &str,
    tbl: &PgRelation,
    row: &RegistryRow,
    spec: &PartitionSpec,
    index: &str,
    client: &Client,
) -> String {
    let relid = tbl.oid();
    let display = catalog::display_name(tbl);
    let (_, prefix) = index_and_prefix(row);
    let registered = registered_names(relid, Lane::Vector);
    let held = match held_under(client, index, prefix) {
        Ok(held) => held,
        Err(error) => {
            warning!(
                "{func}: could not list the namespaces of index '{index}' ({error}); the {} \
                 registered namespaces of {display} are recovered as they are",
                registered.len()
            );
            return format!(
                "; its {} registered namespaces not checked",
                registered.len()
            );
        }
    };
    let family = classify(prefix, spec.kind, &held, &registered);
    if !family.missing.is_empty() {
        let shown: Vec<&str> = family.missing.iter().take(5).map(String::as_str).collect();
        let more = family.missing.len().saturating_sub(shown.len());
        warning!(
            "{func}: index '{index}' lacks {} of the {} registered namespaces of {display} ({}{}): \
             recover_from_remote finds nothing in them",
            family.missing.len(),
            registered.len(),
            shown.join(", "),
            if more > 0 {
                format!(", and {more} more")
            } else {
                String::new()
            }
        );
    }
    warn_foreign(func, &display, prefix, spec, &family.foreign);
    partition_registry::register_partitions(relid, Lane::Vector, &family.unregistered);
    let mut clause = format!(
        "; {} of {} registered namespaces found there",
        family.found,
        registered.len()
    );
    if !family.unregistered.is_empty() {
        clause.push_str(&format!(", adopted {}", family.unregistered.len()));
    }
    clause
}

/// Registers the namespaces of the values `table` holds (`table` and `column` are quoted). A wipe
/// walks the registry, and nothing lists the namespaces of a document index: a drain that rolled
/// back after Pinecone took a document of a new value lost that value's registration, and the
/// table is the one other place that names it.
pub(crate) fn adopt_table_values(
    relid: pg_sys::Oid,
    lane: Lane,
    prefix: &str,
    table: &str,
    column: &str,
) {
    let sql = format!("SELECT DISTINCT ({column}::text COLLATE \"C\") AS value FROM {table}");
    let values: Vec<String> = Spi::connect(|client| {
        client
            .select(&sql, None, &[])
            .expect("pg_pinecone: partition value query failed")
            .filter_map(|row| row.get_by_name::<String, _>("value").ok().flatten())
            .collect()
    });
    let mut pairs: Vec<(String, String)> = values
        .into_iter()
        .map(|value| {
            let namespace = partition::namespace_for(prefix, &value);
            (value, namespace)
        })
        .filter(|(_, namespace)| partition::fits_pinecone(namespace).is_ok())
        .collect();
    pairs.sort_by(|a, b| a.1.cmp(&b.1));
    partition_registry::register_partitions(relid, lane, &pairs);
}

/// Implements `pinecone.partitions_of`: the namespaces `tbl` wrote, lane by lane, for its owner.
/// `pinecone.partitions` itself is closed to every role but the registry's owner, so this is how
/// the owner of a table reads their own rows.
pub(crate) fn partitions_of(tbl: &PgRelation) -> Vec<partition_registry::Listed> {
    docs_track::require_owner(
        tbl.oid(),
        &catalog::display_name(tbl),
        "pinecone.partitions_of",
    );
    partition_registry::listed(tbl.oid())
}

/// Whether `lane` of `relid` has anything queued: delivered after a delete, a write of the value
/// (or a backfill) creates its namespace again.
fn has_queue(relid: pg_sys::Oid, lane: Lane) -> bool {
    as_owner(|| match lane {
        Lane::Vector => outbox::backlog_for(relid).unwrap_or(0) > 0,
        Lane::Docs => docs_outbox::backlog(relid) > 0,
    })
}

/// One lane of a table that has a partition column.
pub(crate) struct Target {
    pub(crate) lane: Lane,
    pub(crate) index: String,
    pub(crate) prefix: String,
    pub(crate) spec: PartitionSpec,
}

fn targets_of(
    vector: Option<RegistryRow>,
    docs: Option<docs_registry::DocRegistryRow>,
) -> Vec<Target> {
    let mut targets = Vec::new();
    if let Some(row) = vector
        && let (Some(spec), Some(index), Some(prefix)) =
            (row.partition, row.index_name, row.namespace)
    {
        targets.push(Target {
            lane: Lane::Vector,
            index,
            prefix,
            spec,
        });
    }
    if let Some(row) = docs
        && let Some(spec) = row.partition
    {
        targets.push(Target {
            lane: Lane::Docs,
            index: row.index_name,
            prefix: row.namespace,
            spec,
        });
    }
    targets
}

/// The partition value as text; NULL and a type no partition column holds are refused.
pub(crate) fn value_text(func: &str, value: Option<PartitionArg>) -> String {
    match value.as_ref().map(PartitionArg::read) {
        Some(Ok(Some(text))) => text,
        Some(Err(oid)) => error_with_code(
            PgSqlErrorCode::ERRCODE_DATATYPE_MISMATCH,
            format!(
                "{func}: value is of type {}, which no partition column holds",
                docs_search::type_name(oid)
            ),
            None,
            Some("pass an integer, a text or a uuid value"),
        ),
        Some(Ok(None)) | None => error_with_code(
            PgSqlErrorCode::ERRCODE_NULL_VALUE_NOT_ALLOWED,
            format!("{func}: value must not be NULL"),
            None,
            None,
        ),
    }
}

/// The namespace of `text` in `target`, refused when it is no value of the column or Pinecone
/// would refuse the name.
pub(crate) fn namespace_of(func: &str, target: &Target, text: &str) -> String {
    let spec = &target.spec;
    let Some(canon) = docs_search::canonical(text, spec.kind) else {
        error_with_code(
            PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
            format!(
                "{func}: {text:?} is not a valid {} value for \"{}\"",
                spec.kind.as_str(),
                spec.column
            ),
            None,
            None,
        )
    };
    let namespace = partition::namespace_for(&target.prefix, &canon);
    if let Err(error) = partition::fits_pinecone(&namespace) {
        let code = match error {
            NamespaceError::TooLong { .. } => PgSqlErrorCode::ERRCODE_PROGRAM_LIMIT_EXCEEDED,
            _ => PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
        };
        error_with_code(
            code,
            format!("{func}: the namespace of this partition is unusable: {error}"),
            None,
            None,
        );
    }
    namespace
}

/// Implements `pinecone.delete_partition`: deletes the Pinecone namespace of one partition value
/// (vectors: the namespace; documents: every document in it) and forgets it in
/// `pinecone.partitions`, for each lane of `tbl` that has a partition column. The table's rows are
/// not touched: the next write of a row with the value writes it again.
pub(crate) fn delete_partition(tbl: PgRelation, value: Option<PartitionArg>) {
    const FUNC: &str = "pinecone.delete_partition";
    let relid = tbl.oid();
    let display = catalog::display_name(&tbl);
    docs_track::require_owner(relid, &display, FUNC);
    let text = value_text(FUNC, value);

    let vector = as_owner(|| registry::lookup(relid));
    let docs = as_owner(|| docs_registry::lookup(relid));
    let tracked = vector.is_some() || docs.is_some();
    let targets = targets_of(vector, docs);
    if targets.is_empty() {
        if tracked {
            error_with_code(
                PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
                format!("{FUNC}: {display} has no partition column, so it has one namespace"),
                None,
                Some("pinecone.demote or pinecone.untrack_docs removes everything it wrote"),
            );
        }
        error_with_code(
            PgSqlErrorCode::ERRCODE_UNDEFINED_OBJECT,
            format!("{FUNC}: {display} is not tracked by pinecone.promote or pinecone.track_docs"),
            None,
            None,
        );
    }
    let namespaces: Vec<String> = targets
        .iter()
        .map(|target| namespace_of(FUNC, target, &text))
        .collect();

    let client = backend::client_for_effective_key();
    for (target, namespace) in targets.iter().zip(&namespaces) {
        let deleted = match target.lane {
            Lane::Vector => {
                outbox::lock_relid(relid);
                client.delete_namespace(&target.index, namespace)
            }
            Lane::Docs => {
                docs_drain::lock_lane_for_verb(relid, &display);
                client.delete_docs(&DeleteDocsRequest {
                    index: target.index.clone(),
                    namespace: Some(namespace.clone()),
                    selector: DocDeleteSelector::All,
                })
            }
        };
        deleted.unwrap_or_else(|e| backend::raise(e));
        let registered = partition_registry::is_registered(relid, target.lane, namespace);
        partition_registry::forget_partition(relid, target.lane, namespace);
        notice!(
            "{FUNC}: deleted namespace '{namespace}' of {display}{}; its rows with \"{}\" = {} stay \
             in the table, and the next write of one registers the namespace again",
            if registered {
                ""
            } else {
                " (it was not registered)"
            },
            target.spec.column,
            quote_literal(&text)
        );
        if has_queue(relid, target.lane) {
            notice!(
                "{FUNC}: {display} still has queued writes (or a backfill): one for \"{}\" = {} \
                 creates namespace '{namespace}' again when it is delivered",
                target.spec.column,
                quote_literal(&text)
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PREFIX: &str = "public.docs";

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|name| name.to_string()).collect()
    }

    fn pair(value: &str, namespace: &str) -> (String, String) {
        (value.to_string(), namespace.to_string())
    }

    #[test]
    fn registered_and_held_names_are_found_and_the_rest_is_sorted_out() {
        let held = names(&[
            "public.docs:3",
            "public.docs:1",
            "public.docs:9",
            "public.docs:abc",
            "public.docs:%2f",
        ]);
        let registered = names(&["public.docs:1", "public.docs:2", "public.docs:3"]);

        let family = classify(PREFIX, PartitionKind::Int8, &held, &registered);

        assert_eq!(family.found, 2);
        assert_eq!(family.held, 5);
        assert_eq!(family.unregistered, [pair("9", "public.docs:9")]);
        assert_eq!(family.foreign, ["public.docs:%2f", "public.docs:abc"]);
        assert_eq!(family.missing, ["public.docs:2"]);
    }

    #[test]
    fn names_outside_the_family_are_not_looked_at() {
        let held = names(&[
            "public.docs",
            "public.docs2:1",
            "public.docsx",
            "other:1",
            "__default__",
        ]);

        let family = classify(PREFIX, PartitionKind::Int8, &held, &[]);

        assert_eq!(family, Family::default());
    }

    #[test]
    fn what_a_kind_could_not_have_written_is_foreign() {
        let held = names(&[
            "public.docs:007",
            "public.docs:+7",
            "public.docs:7",
            "public.docs:a%20b",
        ]);

        let ints = classify(PREFIX, PartitionKind::Int4, &held, &[]);
        assert_eq!(ints.unregistered, [pair("7", "public.docs:7")]);
        assert_eq!(ints.foreign.len(), 3);

        let text = classify(PREFIX, PartitionKind::Text, &held, &[]);
        assert_eq!(text.unregistered.len(), 3);
        assert!(
            text.unregistered
                .contains(&pair("a b", "public.docs:a%20b"))
        );
        assert_eq!(text.foreign, ["public.docs:+7"], "enc escapes a plus sign");
    }

    #[test]
    fn unregistered_names_come_out_in_namespace_order_whatever_the_listing_order() {
        let held = names(&["public.docs:3", "public.docs:10", "public.docs:2"]);

        let family = classify(PREFIX, PartitionKind::Int8, &held, &[]);

        let order: Vec<&str> = family
            .unregistered
            .iter()
            .map(|(_, ns)| ns.as_str())
            .collect();
        assert_eq!(order, ["public.docs:10", "public.docs:2", "public.docs:3"]);
    }

    #[test]
    fn a_family_with_a_colon_in_its_prefix_keeps_its_own_names_only() {
        let held = names(&["a:b:1", "a:b", "a:c:1", "a:b:x%3Ay"]);

        let family = classify("a:b", PartitionKind::Text, &held, &[]);

        assert_eq!(family.held, 2);
        assert_eq!(
            family.unregistered,
            [pair("1", "a:b:1"), pair("x:y", "a:b:x%3Ay")]
        );
    }
}
