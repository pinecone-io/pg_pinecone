//! Partitioned remote index: `USING pinecone (vec, tenant_id)` with `tenant_id` as the registry's
//! partition column, one Pinecone namespace per value.

use std::{collections::HashMap, ffi::c_void};

use pg_pinecone_client::{Error as ClientError, Vector};
use pgrx::{PgMemoryContexts, pg_sys, prelude::*, spi::quote_qualified_identifier};
use serde_json::Value;

use crate::{
    am_filter::EqKey,
    backend::error_with_code,
    catalog,
    partition::{self, Claim, NamespaceError, PartitionSpec},
    partition_lane, partition_registry, registry,
};

/// No uuid: the access method's filter opclasses have none.
pub(crate) fn declare(
    relid: pg_sys::Oid,
    schema: &str,
    table: &str,
    column: &str,
    index_name: &str,
    func: &str,
) -> PartitionSpec {
    let display = format!("{schema}.{table}");
    let kind = partition_lane::check_column(relid, &display, column, func, false);
    let claim = Claim::Family(&display);
    if let Some(owner) = registry::claim_owner(index_name, claim, relid) {
        partition_registry::refuse_family_clash(func, index_name, claim, &owner);
    }
    let scan = partition_lane::Scan {
        table_ident: &quote_qualified_identifier(schema, table),
        display: &display,
        prefix: &display,
        id_expr: "ctid",
        id_label: "row",
    };
    partition_lane::announce_namespaces(&scan, column, kind, func);
    PartitionSpec {
        column: column.to_string(),
        kind,
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum RouteError {
    Null,
    Namespace(NamespaceError),
}

pub(crate) fn route(prefix: &str, key: Option<&EqKey>) -> Result<(String, String), RouteError> {
    let canon = key.ok_or(RouteError::Null)?.canon();
    let namespace = partition::namespace_for(prefix, &canon);
    partition::fits_pinecone(&namespace).map_err(RouteError::Namespace)?;
    Ok((canon, namespace))
}

pub(crate) fn scan_namespace(prefix: &str, key: &EqKey) -> Result<String, NamespaceError> {
    let namespace = partition::namespace_for(prefix, &key.canon());
    partition::fits_pinecone(&namespace)?;
    Ok(namespace)
}

fn namespace_error_code(error: &NamespaceError) -> PgSqlErrorCode {
    match error {
        NamespaceError::TooLong { .. } => PgSqlErrorCode::ERRCODE_PROGRAM_LIMIT_EXCEEDED,
        _ => PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
    }
}

pub(crate) fn raise_scan_namespace(relid: pg_sys::Oid, error: &NamespaceError) -> ! {
    error_with_code(
        namespace_error_code(error),
        format!(
            "pg_pinecone: the namespace of this partition of {} is unusable: {error}",
            catalog::display_name_of(relid)
        ),
        None,
        None,
    )
}

fn raise_route(relid: pg_sys::Oid, spec: &PartitionSpec, error: RouteError) -> ! {
    let display = catalog::display_name_of(relid);
    let column = &spec.column;
    match error {
        RouteError::Null => error_with_code(
            PgSqlErrorCode::ERRCODE_NOT_NULL_VIOLATION,
            format!(
                "pg_pinecone: partition column \"{column}\" of {display} is NULL: every row of a \
                 partitioned index needs a value"
            ),
            None,
            Some(&format!(
                "give \"{column}\" a value: the index is partitioned by it"
            )),
        ),
        RouteError::Namespace(error) => error_with_code(
            namespace_error_code(&error),
            format!(
                "pg_pinecone: the partition value of \"{column}\" of {display} makes an unusable \
                 namespace: {error}"
            ),
            None,
            None,
        ),
    }
}

/// Loaded once per statement into its `IndexInfo`: `aminsert` runs per tuple, the registry
/// cannot change under one statement.
pub(crate) struct WriterCache {
    target: Option<(String, PartitionSpec)>,
}

impl WriterCache {
    fn load(relid: pg_sys::Oid) -> Self {
        let target = registry::lookup(relid).and_then(|row| Some((row.namespace?, row.partition?)));
        Self { target }
    }

    /// The tuple's canonical partition value, `None` for an unpartitioned index. It writes no
    /// registry row: the drain registers the namespace once Pinecone took a write of it, so
    /// writers never wait on (or deadlock over) each other's uncommitted rows.
    pub(crate) fn route(&self, relid: pg_sys::Oid, key: Option<&EqKey>) -> Option<String> {
        let (prefix, spec) = self.target.as_ref()?;
        let (canon, _) = route(prefix, key).unwrap_or_else(|e| raise_route(relid, spec, e));
        Some(canon)
    }
}

/// # Safety
/// `index_info` is null or the live `IndexInfo` of the running `aminsert`, and the caller runs
/// as the registry's owner.
pub(crate) unsafe fn with_writer_cache<T>(
    index_info: *mut pg_sys::IndexInfo,
    relid: pg_sys::Oid,
    f: impl FnOnce(&WriterCache) -> T,
) -> T {
    unsafe {
        if index_info.is_null() {
            return f(&WriterCache::load(relid));
        }
        if (*index_info).ii_AmCache.is_null() {
            let cache = WriterCache::load(relid);
            let kept =
                PgMemoryContexts::For((*index_info).ii_Context).leak_and_drop_on_delete(cache);
            (*index_info).ii_AmCache = kept as *mut c_void;
        }
        f(&*((*index_info).ii_AmCache as *const WriterCache))
    }
}

pub(crate) fn require_key(relid: pg_sys::Oid) {
    let Some(spec) = registry::lookup(relid).and_then(|row| row.partition) else {
        return;
    };
    let column = &spec.column;
    error_with_code(
        PgSqlErrorCode::ERRCODE_FEATURE_NOT_SUPPORTED,
        format!(
            "pg_pinecone: a query on a partitioned remote index needs WHERE {column} = <value> \
             ({} is partitioned by \"{column}\")",
            catalog::display_name_of(relid)
        ),
        None,
        Some(&format!(
            "each value of \"{column}\" is its own Pinecone namespace and a query searches one: \
             add WHERE {column} = <value>"
        )),
    )
}

#[derive(Debug)]
pub(crate) struct VectorGroup {
    pub(crate) namespace: String,
    pub(crate) value: String,
    pub(crate) vectors: Vec<Vector>,
}

/// The canonical partition value of a vector, from the `{column: v}` metadata `aminsert` wrote.
pub(crate) fn value_of(vector: &Vector, column: &str) -> Option<String> {
    key_canon(vector.metadata.as_ref(), column)
}

/// The canonical partition value in a `{column: v}` metadata object.
pub(crate) fn key_canon(metadata: Option<&Value>, column: &str) -> Option<String> {
    match metadata?.get(column)? {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) if number.is_i64() => Some(number.to_string()),
        _ => None,
    }
}

/// Splits `vectors` per namespace in order of first appearance. One without a value, or with a
/// namespace Pinecone would refuse, stops the chunk.
pub(crate) fn group_vectors(
    prefix: &str,
    column: &str,
    vectors: Vec<Vector>,
) -> Result<Vec<VectorGroup>, ClientError> {
    let mut groups: Vec<VectorGroup> = Vec::new();
    let mut at: HashMap<String, usize> = HashMap::new();
    for vector in vectors {
        let Some(canon) = value_of(&vector, column) else {
            return Err(ClientError::BadRequest {
                message: format!(
                    "pg_pinecone: partition column \"{column}\" is NULL for the row of vector {}: a \
                     NULL has no namespace",
                    vector.id
                ),
            });
        };
        let namespace = partition::namespace_for(prefix, &canon);
        match at.get(&namespace) {
            Some(&group) => groups[group].vectors.push(vector),
            None => {
                partition::fits_pinecone(&namespace).map_err(|e| ClientError::BadRequest {
                    message: format!("pg_pinecone: the partition value makes a bad namespace: {e}"),
                })?;
                at.insert(namespace.clone(), groups.len());
                groups.push(VectorGroup {
                    namespace,
                    value: canon,
                    vectors: vec![vector],
                });
            }
        }
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const PREFIX: &str = "public.docs";

    #[test]
    fn a_key_routes_to_its_value_and_namespace() {
        assert_eq!(
            route(PREFIX, Some(&EqKey::Int(42))),
            Ok(("42".to_string(), "public.docs:42".to_string()))
        );
        assert_eq!(
            route(PREFIX, Some(&EqKey::Text("Acme Corp".into()))),
            Ok((
                "Acme Corp".to_string(),
                "public.docs:Acme%20Corp".to_string()
            ))
        );
    }

    #[test]
    fn a_null_has_no_namespace() {
        assert_eq!(route(PREFIX, None), Err(RouteError::Null));
    }

    #[test]
    fn a_partition_value_is_read_from_the_key_metadata() {
        let key = |metadata: Value| key_canon(Some(&metadata), "tenant_id");

        assert_eq!(key(json!({"tenant_id": 42})), Some("42".to_string()));
        assert_eq!(key(json!({"tenant_id": "Acme"})), Some("Acme".to_string()));
        assert_eq!(key(json!({"other": 42})), None);
        assert_eq!(key(json!({"tenant_id": null})), None);
        assert_eq!(key(json!({"tenant_id": 1.5})), None);
        assert_eq!(key_canon(None, "tenant_id"), None);
    }

    #[test]
    fn a_namespace_over_512_bytes_is_refused_and_512_is_not() {
        let room = partition::MAX_NAMESPACE_BYTES - PREFIX.len() - 1;
        let fits = EqKey::Text("x".repeat(room));
        assert!(route(PREFIX, Some(&fits)).is_ok());
        assert!(scan_namespace(PREFIX, &fits).is_ok());
        let over = EqKey::Text("x".repeat(room + 1));
        assert_eq!(
            route(PREFIX, Some(&over)),
            Err(RouteError::Namespace(NamespaceError::TooLong {
                bytes: 513
            }))
        );
        assert_eq!(
            scan_namespace(PREFIX, &over),
            Err(NamespaceError::TooLong { bytes: 513 })
        );
    }

    #[test]
    fn a_scan_reads_the_namespace_a_writer_used() {
        for key in [
            EqKey::Int(-7),
            EqKey::Text(String::new()),
            EqKey::Text("a:b%c".into()),
        ] {
            let (_, written) = route(PREFIX, Some(&key)).expect("route");
            assert_eq!(scan_namespace(PREFIX, &key), Ok(written));
        }
    }

    fn vector(id: &str, metadata: Option<Value>) -> Vector {
        Vector {
            id: id.to_string(),
            values: vec![1.0],
            metadata,
        }
    }

    #[test]
    fn a_chunk_splits_per_namespace_in_order_of_first_appearance() {
        let vectors = vec![
            vector("1:1", Some(json!({"tenant_id": 2}))),
            vector("1:2", Some(json!({"tenant_id": 1}))),
            vector("1:3", Some(json!({"tenant_id": 2}))),
        ];
        let groups = group_vectors(PREFIX, "tenant_id", vectors).expect("group");
        let shape: Vec<(&str, &str, Vec<&str>)> = groups
            .iter()
            .map(|g| {
                let ids = g.vectors.iter().map(|v| v.id.as_str()).collect();
                (g.namespace.as_str(), g.value.as_str(), ids)
            })
            .collect();
        assert_eq!(
            shape,
            [
                ("public.docs:2", "2", vec!["1:1", "1:3"]),
                ("public.docs:1", "1", vec!["1:2"]),
            ]
        );
    }

    #[test]
    fn text_values_group_by_their_bytes() {
        let vectors = vec![
            vector("1:1", Some(json!({"t": "a b"}))),
            vector("1:2", Some(json!({"t": "a"}))),
            vector("1:3", Some(json!({"t": "a b"}))),
        ];
        let groups = group_vectors(PREFIX, "t", vectors).expect("group");
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].namespace, "public.docs:a%20b");
        assert_eq!(groups[0].vectors.len(), 2);
    }

    #[test]
    fn a_vector_without_its_value_stops_the_chunk() {
        let cases = [
            None,
            Some(json!({})),
            Some(json!({"other": 1})),
            Some(json!({"t": 1.5})),
        ];
        for metadata in cases {
            let error = group_vectors(PREFIX, "t", vec![vector("1:1", metadata)])
                .expect_err("no namespace");
            assert!(error.to_string().contains("NULL"), "{error}");
        }
    }

    #[test]
    fn a_value_that_makes_a_bad_namespace_stops_the_chunk() {
        let long = "x".repeat(600);
        let error = group_vectors(PREFIX, "t", vec![vector("1:1", Some(json!({"t": long})))])
            .expect_err("too long");
        assert!(error.to_string().contains("512"), "{error}");
    }
}
