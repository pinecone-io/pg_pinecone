//! Implementation of `pinecone.topk`. `#[pg_schema]` only attaches to an inline
//! `mod {}` (see the comment on `pg_pinecone` in `lib.rs`), so its `#[pg_extern]`
//! wrapper is kept thin and delegates here.
//!
//! This is `pg_pinecone`'s one mode dispatch point (see [`topk`]): everything below
//! it is either pure-local SQL ([`topk_local`]) or, pure-remote client
//! calls — no scattered `if remote` checks elsewhere.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use pg_pinecone_client::QueryRequest;
use pgrx::{
    JsonB, PgRelation,
    datum::DatumWithOid,
    prelude::*,
    spi::{Spi, quote_identifier, quote_literal, quote_qualified_identifier},
};
use serde_json::{Value, json};

use crate::{
    backend::{self, error_with_code, require_non_negative_u32, require_real_vector},
    catalog,
    docs_search::{self, PartitionArg},
    filter::{self, ColKind},
    partition::{self, Lane, PartitionKind, PartitionSpec},
    partition_registry,
    registry::{self, RegistryRow},
};

/// Finds the `k` nearest rows to `query` in a tracked table. Query/dimension/
/// `k` are validated once here (rather than duplicated in each mode's
/// function) since both modes need identical checks; `k == 0` short-circuits
/// to an empty result *before* dispatching to either mode — LIMIT-0
/// semantics, with no local SQL execution and no remote HTTP call.
pub(crate) fn topk(
    tbl: PgRelation,
    query: Option<Vec<Option<f32>>>,
    k: i32,
    filter: Option<JsonB>,
    partition: Option<PartitionArg>,
) -> Vec<(String, f64)> {
    let relid = tbl.oid();
    let display = catalog::display_name(&tbl);

    let row = registry::lookup_for_reader(relid, &display).unwrap_or_else(|| {
        error!("pinecone.topk: table {display} is not tracked — run pinecone.track(...) first")
    });
    registry::require_not_am(&row, "pinecone.topk", &display);

    let query = require_real_vector("pinecone.topk", "query", query);
    if query.len() as i32 != row.dimension {
        error!(
            "pinecone.topk: query has {} dimensions, but {display} is tracked with dimension {}",
            query.len(),
            row.dimension
        );
    }
    let k = require_non_negative_u32("pinecone.topk", "k", k);
    let bound = bind_partition(
        &row,
        &display,
        filter.as_ref().map(|f| &f.0),
        partition.as_ref(),
    );

    if k == 0 {
        return Vec::new();
    }

    match row.mode.as_str() {
        "local" => topk_local(&tbl, &row, query, k, filter, bound.as_deref()),
        "remote" => topk_remote(&tbl, &row, query, k, filter, bound.as_deref()),
        // An async promote's backfill job hasn't exhausted yet — the remote
        // namespace may still be only partially populated (or still hold
        // whatever an earlier promote/demote cycle left in it), so this must
        // never serve remote results: search locally,
        // exactly like `'local'` mode, until the worker flips the row to
        // `'remote'` — never serve a partial namespace.
        "promoting" => topk_local(&tbl, &row, query, k, filter, bound.as_deref()),
        other => error!("pinecone.topk: table {display} has unknown registry mode '{other}'"),
    }
}

/// The local search path: pgvector operators via SPI, index-scan friendly
/// (`ORDER BY` the same operator expression used for `distance`). `query`/`k`
/// are already validated by [`topk`].
fn topk_local(
    tbl: &PgRelation,
    row: &RegistryRow,
    query: Vec<f32>,
    k: u32,
    filter_arg: Option<JsonB>,
    bound: Option<&str>,
) -> Vec<(String, f64)> {
    // READ-PATH ENFORCEMENT: a detached table's local
    // vector column is remote-only (NULL by definition) — [`topk`]'s own
    // dispatch never routes a `mode = 'remote'` row here, but a
    // `mode = 'promoting'` row (which DOES route here, to avoid serving a
    // partial namespace) could still be detached via a direct test/registry
    // manipulation (the real `detach_storage` requires `mode = 'remote'`
    // exactly, so this is normally unreachable in practice) — a clear error
    // naming `attach_storage` beats a silently-empty/wrong result.
    if row.storage == "detached" {
        error!(
            "pinecone.topk: {} is detached (storage = 'detached') — its local vector column is \
             remote-only; run pinecone.attach_storage(...) first",
            catalog::display_name(tbl)
        );
    }

    let vector_ns = catalog::pgvector_namespace().unwrap_or_else(|e| error!("pinecone.topk: {e}"));
    let op = metric_operator(&row.metric);
    let operator = format!("OPERATOR({}.{op})", quote_identifier(&vector_ns));
    let vector_type = quote_qualified_identifier(vector_ns.as_str(), "vector");
    let vcol = quote_identifier(&row.vector_column);
    let id_col = quote_identifier(&row.id_column);
    let table = quote_qualified_identifier(tbl.namespace(), tbl.name());

    // The `IS NOT NULL` guard is not just an optimization: pgvector's distance
    // operators return NULL for a NULL operand, and ORDER BY sorts NULLs last
    // by default — but "last" still means "returned" once `k` reaches the
    // non-NULL row count, which would surface a NULL `distance` that the
    // `.get_by_name` calls below aren't prepared to handle.
    let mut sql = format!(
        "SELECT {id_col}::text AS id, ({vcol} {operator} $1::{vector_type})::float8 AS distance \
         FROM {table} WHERE {vcol} IS NOT NULL"
    );

    let filter_value = match (&row.partition, bound) {
        (Some(spec), Some(value)) => Some(JsonB(scoped_to_partition(
            filter_arg.map(|f| f.0),
            spec,
            value,
        ))),
        _ => filter_arg,
    };
    if let Some(filter_value) = filter_value {
        let columns = filter_columns(tbl.oid());
        let where_clause = filter::translate(&filter_value.0, &columns)
            .unwrap_or_else(|e| error!("pinecone.topk: {e}"));
        sql.push_str(" AND ");
        sql.push_str(&where_clause);
    }

    sql.push_str(&format!(
        " ORDER BY {vcol} {operator} $1::{vector_type} LIMIT {k}"
    ));

    Spi::connect(|client| {
        client
            .select(&sql, None, &[DatumWithOid::from(query)])
            .unwrap_or_else(|e| error!("pinecone.topk: {e}"))
            .map(|r| {
                let id = r
                    .get_by_name::<String, _>("id")
                    .unwrap_or_else(|e| error!("pinecone.topk: failed to read id: {e}"))
                    .unwrap_or_else(|| {
                        error!(
                            "pinecone.topk: id unexpectedly NULL — should be unreachable, id_column \
                             is the table's primary key"
                        )
                    });
                let distance = r
                    .get_by_name::<f64, _>("distance")
                    .unwrap_or_else(|e| error!("pinecone.topk: failed to read distance: {e}"))
                    .unwrap_or_else(|| {
                        error!(
                            "pinecone.topk: distance unexpectedly NULL — should be unreachable, the \
                             query excludes rows with a NULL {vcol}"
                        )
                    });
                (id, distance)
            })
            .collect()
    })
}

/// The remote search path: Pinecone `/query` via the client, filter
/// passthrough (validated against `metadata_columns` first), and
/// score→distance conversion so `pinecone.topk` returns identical semantics
/// regardless of mode. `query`/`k` are already validated by [`topk`].
fn topk_remote(
    tbl: &PgRelation,
    row: &RegistryRow,
    query: Vec<f32>,
    k: u32,
    filter_arg: Option<JsonB>,
    bound: Option<&str>,
) -> Vec<(String, f64)> {
    let filter_json = filter_arg.map(|f| {
        validate_remote_filter(tbl, &f.0, row).unwrap_or_else(|e| error!("pinecone.topk: {e}"));
        f.0
    });

    let index_name = row
        .index_name
        .clone()
        .expect("pinecone.topk: remote row always has index_name");
    let namespace = match bound {
        Some(value) => {
            let prefix = row
                .namespace
                .as_deref()
                .expect("pinecone.topk: remote row always has namespace");
            Some(partition_namespace(prefix, value))
        }
        None => row.namespace.clone(),
    };

    let (scan_factor, max_candidates) = backend::drn_query_tuning();
    let request = QueryRequest {
        index: index_name,
        vector: query,
        top_k: k,
        namespace,
        filter: filter_json,
        include_metadata: false,
        scan_factor,
        max_candidates,
    };

    let matches = backend::client_for_effective_key()
        .query(&request)
        .unwrap_or_else(|e| backend::raise(e));

    let found: Vec<(String, f64)> = matches
        .into_iter()
        .map(|m| (m.id, score_to_distance(&row.metric, m.score)))
        .collect();
    match (&row.partition, bound, request.namespace.as_deref()) {
        (Some(spec), Some(value), Some(namespace)) => {
            recheck_failed(tbl, row, spec, value, namespace, found)
        }
        _ => found,
    }
}

/// A failed partition's namespace is frozen where it failed: rows that moved to another
/// partition or were deleted since are still in it, and a search would hand their ids out. So its
/// result is checked against the table, and says so.
fn recheck_failed(
    tbl: &PgRelation,
    row: &RegistryRow,
    spec: &PartitionSpec,
    value: &str,
    namespace: &str,
    found: Vec<(String, f64)>,
) -> Vec<(String, f64)> {
    let Some(since) = partition_registry::failed_since(row.relid, Lane::Vector, namespace) else {
        return found;
    };
    let display = catalog::display_name(tbl);
    notice!(
        "pinecone.topk: partition {} of {display} is failed since {since}: results may be stale \
         (rows that left it are left out); pinecone.failed_partitions({}) says why",
        quote_literal(value),
        quote_literal(&display)
    );
    if found.is_empty() {
        return found;
    }
    let ids = found.iter().map(|(id, _)| id.clone()).collect();
    let present = ids_in_partition(tbl, &display, row, spec, value, ids);
    found
        .into_iter()
        .filter(|(id, _)| present.contains(id))
        .collect()
}

/// The ids of `ids` whose row is in partition `value` of the table now.
fn ids_in_partition(
    tbl: &PgRelation,
    display: &str,
    row: &RegistryRow,
    spec: &PartitionSpec,
    value: &str,
    ids: Vec<String>,
) -> HashSet<String> {
    let id_type = catalog::column_type(row.relid, &row.id_column).unwrap_or_else(|| {
        error!(
            "pinecone.topk: id column \"{}\" of {display} no longer exists",
            row.id_column
        )
    });
    let partition_type = catalog::column_type(row.relid, &spec.column)
        .filter(|typname| PartitionKind::from_typname(typname) == Some(spec.kind))
        .unwrap_or_else(|| {
            error!(
                "pinecone.topk: partition column \"{}\" of {display} no longer exists or has \
                 another type",
                spec.column
            )
        });
    let sql = recheck_sql(
        &quote_qualified_identifier(tbl.namespace(), tbl.name()),
        &quote_identifier(&row.id_column),
        &id_type,
        &quote_identifier(&spec.column),
        &partition_type,
    );
    Spi::connect(|client| {
        client
            .select(&sql, None, &[ids.into(), value.into()])
            .unwrap_or_else(|e| error!("pinecone.topk: {e}"))
            .map(|r| {
                r.get_by_name::<String, _>("id")
                    .unwrap_or_else(|e| error!("pinecone.topk: failed to read id: {e}"))
                    .unwrap_or_else(|| error!("pinecone.topk: id unexpectedly NULL"))
            })
            .collect()
    })
}

/// The statement behind [`ids_in_partition`]: ids as `$1`, the partition's canonical text as
/// `$2`. The id is compared in its own type, so the primary key serves it; a type outside the
/// supported id types (changed since promote) is compared as text instead of being spliced in.
fn recheck_sql(
    table: &str,
    id_column: &str,
    id_type: &str,
    partition_column: &str,
    partition_type: &str,
) -> String {
    let id_match = if catalog::ID_COLUMN_TYPES.contains(&id_type) {
        format!("{id_column} = ANY($1::text[]::{id_type}[])")
    } else {
        format!("{id_column}::text = ANY($1::text[])")
    };
    format!(
        "SELECT {id_column}::text AS id FROM {table} \
         WHERE {id_match} AND {partition_column} = $2::text::{partition_type}"
    )
}

/// The partition `topk` is bound to, as canonical text: `partition => v`, or an equality on the
/// partition column in `filter`. A partitioned table is never searched across partitions, and an
/// unpartitioned one is never given a partition it would silently ignore.
fn bind_partition(
    row: &RegistryRow,
    display: &str,
    filter: Option<&Value>,
    arg: Option<&PartitionArg>,
) -> Option<String> {
    let Some(spec) = &row.partition else {
        if arg.is_some() {
            error_with_code(
                PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
                format!(
                    "pinecone.topk: {display} has no partition column, so partition => does not \
                     apply"
                ),
                None,
                Some("omit partition, or promote the table with partition_column => '<col>'"),
            );
        }
        return None;
    };
    let column = &spec.column;
    let text = match arg {
        Some(arg) => argument_text(arg),
        None => filter
            .and_then(|f| filter::equality_on(f, column))
            .and_then(docs_search::operand_text)
            .unwrap_or_else(|| {
                error_with_code(
                    PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
                    format!(
                        "pinecone.topk: {display} is partitioned by \"{column}\": a search \
                         needs one partition"
                    ),
                    None,
                    Some(&format!(
                        "pass partition => <value>, or filter => '{{\"{column}\": {{\"$eq\": \
                         <value>}}}}'"
                    )),
                )
            }),
    };
    let canonical = docs_search::canonical(&text, spec.kind).unwrap_or_else(|| {
        error_with_code(
            PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
            format!(
                "pinecone.topk: partition {text:?} is not a valid {} value for \"{column}\"",
                spec.kind.as_str()
            ),
            None,
            None,
        )
    });
    Some(canonical)
}

fn argument_text(arg: &PartitionArg) -> String {
    match arg.read() {
        Ok(Some(text)) => text,
        Ok(None) => error_with_code(
            PgSqlErrorCode::ERRCODE_NULL_VALUE_NOT_ALLOWED,
            "pinecone.topk: partition must not be NULL",
            None,
            None,
        ),
        Err(oid) => error_with_code(
            PgSqlErrorCode::ERRCODE_DATATYPE_MISMATCH,
            format!(
                "pinecone.topk: partition is of type {}, which no partition column holds",
                docs_search::type_name(oid)
            ),
            None,
            Some("pass an integer, a text or a uuid value"),
        ),
    }
}

/// The namespace of partition `value`, refused when Pinecone would refuse it.
fn partition_namespace(prefix: &str, value: &str) -> String {
    let namespace = partition::namespace_for(prefix, value);
    if let Err(e) = partition::fits_pinecone(&namespace) {
        let code = match e {
            partition::NamespaceError::TooLong { .. } => {
                PgSqlErrorCode::ERRCODE_PROGRAM_LIMIT_EXCEEDED
            }
            _ => PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
        };
        error_with_code(
            code,
            format!("pinecone.topk: the namespace of this partition is unusable: {e}"),
            None,
            None,
        );
    }
    namespace
}

/// `filter` and an equality on the partition column: how a local search honours the partition.
fn scoped_to_partition(filter: Option<Value>, spec: &PartitionSpec, value: &str) -> Value {
    let operand = match spec.kind {
        PartitionKind::Int2 | PartitionKind::Int4 | PartitionKind::Int8 => value
            .parse::<i64>()
            .map_or_else(|_| json!(value), |number| json!(number)),
        PartitionKind::Text | PartitionKind::Uuid => json!(value),
    };
    let pinned = json!({ spec.column.clone(): { "$eq": operand } });
    match filter {
        Some(filter) => json!({ "$and": [filter, pinned] }),
        None => pinned,
    }
}

/// Converts a Pinecone match `score` to a `pinecone.topk` `distance`:
/// cosine `1 − score`; l2 `sqrt(max(score, 0))` (Pinecone's l2 score is
/// already squared Euclidean distance; the `max` guards a score that's `-0.0` or a hair
/// negative from floating-point rounding); ip `−score`.
pub(crate) fn score_to_distance(metric: &str, score: f32) -> f64 {
    match metric {
        "cosine" => 1.0 - score as f64,
        "l2" => (score.max(0.0) as f64).sqrt(),
        "ip" => -(score as f64),
        other => error!("pinecone.topk: table has unknown registry metric '{other}'"),
    }
}

/// Validates a remote-mode filter's columns against `row.metadata_columns`
/// (unlike local mode's [`filter_columns`], which allows any table column) by
/// reusing [`filter::translate`]'s own column/operator validation — its SQL
/// text is discarded here; Pinecone gets the original filter `jsonb`
/// verbatim (filter passthrough). Reusing the same translator is what keeps
/// operator errors identical between modes (the parity suite's error-parity
/// case); an unregistered-column error gets an extra hint appended, since
/// that failure mode is remote-only.
fn validate_remote_filter(
    tbl: &PgRelation,
    filter: &Value,
    row: &RegistryRow,
) -> Result<(), String> {
    let columns = remote_filter_columns(tbl, row, filter)?;
    filter::translate(filter, &columns)
        .map(|_| ())
        .map_err(|e| {
            if e.starts_with("unknown column '") {
                format!("{e} — add it to metadata_columns and re-promote")
            } else {
                e
            }
        })
}

/// Builds the filter translator's column map from the subset of
/// `row.metadata_columns` that `filter` actually references (via
/// [`filter::referenced_columns`]) — remote mode restricts filters to
/// registered metadata columns (`plan.md`'s filter-translation spec), unlike
/// local mode's [`filter_columns`] (any live column). Resolving only the
/// referenced columns (rather than eagerly resolving every registered one)
/// means a *different*, unreferenced, metadata column that was dropped from
/// the table (e.g. via `ALTER TABLE ... DROP COLUMN`, after `track`/`promote`
/// registered it) can't break a query that never mentions it; one that *is*
/// referenced and has been dropped is a clear, user-facing `Err` naming the
/// column and the remedy — never a panic.
fn remote_filter_columns(
    tbl: &PgRelation,
    row: &RegistryRow,
    filter: &Value,
) -> Result<BTreeMap<String, ColKind>, String> {
    let mut referenced = BTreeSet::new();
    filter::referenced_columns(filter, &mut referenced);

    let partition = row.partition.iter().map(|spec| spec.column.as_str());
    let metadata: BTreeSet<&str> = row
        .metadata_columns
        .iter()
        .map(String::as_str)
        .chain(partition)
        .collect();
    let mut columns = BTreeMap::new();
    for name in referenced {
        if !metadata.contains(name.as_str()) {
            // Not a registered metadata column at all; `translate()`'s own
            // "unknown column" error (with the caller's hint) covers it.
            continue;
        }
        let type_name = catalog::column_type(row.relid, &name).ok_or_else(|| {
            format!(
                "metadata column '{name}' is registered for {} but no longer exists — re-track \
                 or re-promote with corrected metadata_columns",
                catalog::display_name(tbl)
            )
        })?;
        columns.insert(name, catalog::col_kind(&type_name));
    }
    Ok(columns)
}

/// Maps a registered metric to its pgvector operator symbol. The registry's
/// `CHECK (metric IN ('cosine', 'l2', 'ip'))` makes the `other` arm unreachable
/// in practice.
fn metric_operator(metric: &str) -> &'static str {
    match metric {
        "cosine" => "<=>",
        "l2" => "<->",
        "ip" => "<#>",
        other => error!("pinecone.topk: table has unknown registry metric '{other}'"),
    }
}

/// Builds the filter translator's column map from every live column of
/// `relid` — local mode allows filtering on any table column, not just
/// `metadata_columns` (unlike remote mode). A column with a type
/// [`catalog::col_kind`] doesn't recognize is included as
/// [`ColKind::Unsupported`], so the translator produces a clear, type-named
/// error if the filter references one, rather than "unknown column".
fn filter_columns(relid: pg_sys::Oid) -> BTreeMap<String, ColKind> {
    catalog::all_columns(relid)
        .into_iter()
        .map(|c| (c.name, catalog::col_kind(&c.type_name)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_recheck_compares_an_id_in_its_own_type_and_the_partition_as_text() {
        assert_eq!(
            recheck_sql("public.t", "id", "int8", "tenant_id", "int4"),
            "SELECT id::text AS id FROM public.t \
             WHERE id = ANY($1::text[]::int8[]) AND tenant_id = $2::text::int4"
        );
    }

    #[test]
    fn the_recheck_never_splices_an_id_type_it_does_not_know() {
        let sql = recheck_sql("public.t", "id", "int8); DROP TABLE t; --", "org", "uuid");
        assert!(!sql.contains("DROP"), "{sql}");
        assert!(sql.contains("id::text = ANY($1::text[])"), "{sql}");
    }

    fn spec(column: &str, kind: PartitionKind) -> PartitionSpec {
        PartitionSpec {
            column: column.to_string(),
            kind,
        }
    }

    #[test]
    fn a_local_search_pins_the_partition_with_an_equality_next_to_the_callers_filter() {
        let ints = spec("tenant_id", PartitionKind::Int8);
        assert_eq!(
            scoped_to_partition(None, &ints, "42"),
            json!({"tenant_id": {"$eq": 42}})
        );
        assert_eq!(
            scoped_to_partition(Some(json!({"category": "a"})), &ints, "42"),
            json!({"$and": [{"category": "a"}, {"tenant_id": {"$eq": 42}}]})
        );
        let texts = spec("org", PartitionKind::Text);
        assert_eq!(
            scoped_to_partition(None, &texts, "42"),
            json!({"org": {"$eq": "42"}})
        );
        let uuids = spec("org", PartitionKind::Uuid);
        let value = "123e4567-e89b-12d3-a456-426614174000";
        assert_eq!(
            scoped_to_partition(None, &uuids, value),
            json!({"org": {"$eq": value}})
        );
    }
}
