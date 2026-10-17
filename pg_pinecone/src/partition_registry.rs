//! `pinecone.partitions`: the namespaces a table's partition column has produced, and the
//! ownership checks over namespace families. Reached as the registry's owner, like `doc_tables`.
use std::collections::HashSet;

use pg_pinecone_client::{Client, DeleteDocsRequest, DocDeleteSelector, Error as ClientError};
use pgrx::{pg_sys, prelude::*, spi::Spi};

use crate::{
    backend, catalog,
    docs_registry::as_owner,
    partition::{self, Claim, Lane},
    partition_lane,
};

extension_sql!(
    r#"
    CREATE TABLE pinecone.partitions (
        relid regclass NOT NULL,
        lane text NOT NULL CHECK (lane IN ('vector', 'docs')),
        -- The canonical text of the partition value (`partition::canon`).
        value text NOT NULL,
        namespace text NOT NULL,
        first_seen timestamptz NOT NULL DEFAULT now(),
        -- Set while the lane skips this namespace because Pinecone refused it for good.
        failed_at timestamptz,
        last_error text,
        PRIMARY KEY (relid, lane, namespace),
        CHECK ((failed_at IS NULL) = (last_error IS NULL)),
        CHECK (failed_at IS NULL OR lane = 'vector')
    );
    CREATE INDEX pg_pinecone_partitions_failed_idx ON pinecone.partitions (relid) WHERE failed_at IS NOT NULL;
    REVOKE ALL ON pinecone.partitions FROM PUBLIC;
    SELECT pg_catalog.pg_extension_config_dump('pinecone.partitions', '');
    "#,
    name = "pg_pinecone_partitions",
    requires = [pinecone],
);

/// Notes that `namespace` holds `value`'s rows; a repeat changes nothing.
pub(crate) fn register_partition(relid: pg_sys::Oid, lane: Lane, value: &str, namespace: &str) {
    as_owner(|| {
        Spi::run_with_args(
            "INSERT INTO pinecone.partitions (relid, lane, value, namespace) \
             VALUES ($1::oid::regclass, $2, $3, $4) \
             ON CONFLICT (relid, lane, namespace) DO NOTHING",
            &[
                relid.into(),
                lane.as_str().into(),
                value.into(),
                namespace.into(),
            ],
        )
    })
    .expect("pg_pinecone: partition registry insert failed");
}

/// [`register_partition`] for many `(value, namespace)` pairs in one statement, inserted in the
/// order given (callers pass them sorted by namespace, the order writers register in).
pub(crate) fn register_partitions(relid: pg_sys::Oid, lane: Lane, partitions: &[(String, String)]) {
    if partitions.is_empty() {
        return;
    }
    let values: Vec<&str> = partitions.iter().map(|(value, _)| value.as_str()).collect();
    let namespaces: Vec<&str> = partitions.iter().map(|(_, ns)| ns.as_str()).collect();
    as_owner(|| {
        Spi::run_with_args(
            "INSERT INTO pinecone.partitions (relid, lane, value, namespace) \
             SELECT $1::oid::regclass, $2, u.value, u.namespace \
             FROM unnest($3::text[], $4::text[]) WITH ORDINALITY AS u(value, namespace, n) \
             ORDER BY u.n \
             ON CONFLICT (relid, lane, namespace) DO NOTHING",
            &[
                relid.into(),
                lane.as_str().into(),
                values.into(),
                namespaces.into(),
            ],
        )
    })
    .expect("pg_pinecone: partition registry insert failed");
}

/// The first registered `(value, namespace)` after `after` in namespace order (the first of all
/// for `None`): what a walk over the namespaces asks for next.
pub(crate) fn next_partition(
    relid: pg_sys::Oid,
    lane: Lane,
    after: Option<&str>,
) -> Option<(String, String)> {
    as_owner(|| {
        Spi::connect(|client| {
            client
                .select(
                    "SELECT value, namespace FROM pinecone.partitions \
                     WHERE relid = $1::oid::regclass AND lane = $2 \
                       AND ($3::text IS NULL OR namespace COLLATE \"C\" > $3::text COLLATE \"C\") \
                     ORDER BY namespace COLLATE \"C\" LIMIT 1",
                    Some(1),
                    &[relid.into(), lane.as_str().into(), after.into()],
                )
                .expect("pg_pinecone: partition registry next query failed")
                .next()
                .map(|row| (text(&row, "value"), text(&row, "namespace")))
        })
    })
}

/// `(value, namespace)` pairs in namespace order.
pub(crate) fn partitions_of(relid: pg_sys::Oid, lane: Lane) -> Vec<(String, String)> {
    as_owner(|| {
        Spi::connect(|client| {
            client
                .select(
                    "SELECT value, namespace FROM pinecone.partitions \
                     WHERE relid = $1::oid::regclass AND lane = $2 \
                     ORDER BY namespace COLLATE \"C\"",
                    None,
                    &[relid.into(), lane.as_str().into()],
                )
                .expect("pg_pinecone: partition registry list query failed")
                .map(|row| (text(&row, "value"), text(&row, "namespace")))
                .collect()
        })
    })
}

/// One row of `pinecone.partitions`.
pub(crate) struct Listed {
    pub(crate) lane: String,
    pub(crate) value: String,
    pub(crate) namespace: String,
    pub(crate) failed_at: Option<TimestampWithTimeZone>,
    pub(crate) last_error: Option<String>,
}

/// Every row of `relid`, by lane and namespace.
pub(crate) fn listed(relid: pg_sys::Oid) -> Vec<Listed> {
    as_owner(|| {
        Spi::connect(|client| {
            client
                .select(
                    "SELECT lane, value, namespace, failed_at, last_error FROM pinecone.partitions \
                     WHERE relid = $1::oid::regclass \
                     ORDER BY lane, namespace COLLATE \"C\"",
                    None,
                    &[relid.into()],
                )
                .expect("pg_pinecone: partition registry list query failed")
                .map(|row| Listed {
                    lane: text(&row, "lane"),
                    value: text(&row, "value"),
                    namespace: text(&row, "namespace"),
                    failed_at: row
                        .get_by_name::<TimestampWithTimeZone, _>("failed_at")
                        .expect("pg_pinecone: partition registry failed_at query failed"),
                    last_error: row
                        .get_by_name::<String, _>("last_error")
                        .expect("pg_pinecone: partition registry last_error query failed"),
                })
                .collect()
        })
    })
}

/// A partition the lanes skip: Pinecone refused it for good.
pub(crate) struct Failed {
    pub(crate) value: String,
    pub(crate) namespace: String,
    pub(crate) failed_at: TimestampWithTimeZone,
    pub(crate) last_error: String,
}

/// Marks `namespace` failed with `error`, registering it when the refusal was its first write; a
/// partition that fails again keeps the newest error.
pub(crate) fn mark_failed(
    relid: pg_sys::Oid,
    lane: Lane,
    value: &str,
    namespace: &str,
    error: &str,
) {
    as_owner(|| {
        Spi::run_with_args(
            "INSERT INTO pinecone.partitions (relid, lane, value, namespace, failed_at, last_error) \
             VALUES ($1::oid::regclass, $2, $3, $4, now(), $5) \
             ON CONFLICT (relid, lane, namespace) DO UPDATE \
             SET failed_at = now(), last_error = EXCLUDED.last_error",
            &[
                relid.into(),
                lane.as_str().into(),
                value.into(),
                namespace.into(),
                error.into(),
            ],
        )
    })
    .expect("pg_pinecone: partition registry mark failed");
}

/// The values of the failed partitions of `relid` in `lane`.
pub(crate) fn failed_values(relid: pg_sys::Oid, lane: Lane) -> HashSet<String> {
    as_owner(|| {
        Spi::connect(|client| {
            client
                .select(
                    "SELECT value FROM pinecone.partitions \
                     WHERE relid = $1::oid::regclass AND lane = $2 AND failed_at IS NOT NULL",
                    None,
                    &[relid.into(), lane.as_str().into()],
                )
                .expect("pg_pinecone: partition registry failed query failed")
                .map(|row| text(&row, "value"))
                .collect()
        })
    })
}

/// The failed partitions of `relid` in `lane`, by namespace.
pub(crate) fn failed_of(relid: pg_sys::Oid, lane: Lane) -> Vec<Failed> {
    as_owner(|| {
        Spi::connect(|client| {
            client
                .select(
                    "SELECT value, namespace, failed_at, last_error FROM pinecone.partitions \
                     WHERE relid = $1::oid::regclass AND lane = $2 AND failed_at IS NOT NULL \
                     ORDER BY namespace COLLATE \"C\"",
                    None,
                    &[relid.into(), lane.as_str().into()],
                )
                .expect("pg_pinecone: partition registry failed query failed")
                .map(|row| Failed {
                    value: text(&row, "value"),
                    namespace: text(&row, "namespace"),
                    failed_at: row
                        .get_by_name::<TimestampWithTimeZone, _>("failed_at")
                        .expect("pg_pinecone: partition registry failed_at query failed")
                        .expect("pg_pinecone: a failed partition has failed_at"),
                    last_error: text(&row, "last_error"),
                })
                .collect()
        })
    })
}

/// When `namespace` failed, as the session writes timestamps; `None` while it is healthy.
pub(crate) fn failed_since(relid: pg_sys::Oid, lane: Lane, namespace: &str) -> Option<String> {
    as_owner(|| {
        Spi::get_one_with_args::<String>(
            "SELECT (SELECT failed_at::text FROM pinecone.partitions \
                     WHERE relid = $1::oid::regclass AND lane = $2 AND namespace = $3)",
            &[relid.into(), lane.as_str().into(), namespace.into()],
        )
    })
    .expect("pg_pinecone: partition registry failed-since query failed")
}

pub(crate) fn clear_failed(relid: pg_sys::Oid, lane: Lane, namespace: &str) {
    as_owner(|| {
        Spi::run_with_args(
            "UPDATE pinecone.partitions SET failed_at = NULL, last_error = NULL \
             WHERE relid = $1::oid::regclass AND lane = $2 AND namespace = $3",
            &[relid.into(), lane.as_str().into(), namespace.into()],
        )
    })
    .expect("pg_pinecone: partition registry clear failed");
}

pub(crate) fn count_failed(relid: pg_sys::Oid, lane: Lane) -> i64 {
    as_owner(|| {
        Spi::get_one_with_args::<i64>(
            "SELECT count(*) FROM pinecone.partitions \
             WHERE relid = $1::oid::regclass AND lane = $2 AND failed_at IS NOT NULL",
            &[relid.into(), lane.as_str().into()],
        )
    })
    .expect("pg_pinecone: partition registry failed count failed")
    .unwrap_or(0)
}

/// Whether `namespace` is registered for `relid` in `lane`.
pub(crate) fn is_registered(relid: pg_sys::Oid, lane: Lane, namespace: &str) -> bool {
    as_owner(|| {
        Spi::get_one_with_args::<bool>(
            "SELECT EXISTS (SELECT 1 FROM pinecone.partitions \
             WHERE relid = $1::oid::regclass AND lane = $2 AND namespace = $3)",
            &[relid.into(), lane.as_str().into(), namespace.into()],
        )
    })
    .expect("pg_pinecone: partition registry lookup failed")
    .expect("pg_pinecone: EXISTS unexpectedly NULL")
}

/// Drops one namespace from the registry: it was emptied, so a later wipe need not visit it.
pub(crate) fn forget_partition(relid: pg_sys::Oid, lane: Lane, namespace: &str) {
    as_owner(|| {
        Spi::run_with_args(
            "DELETE FROM pinecone.partitions \
             WHERE relid = $1::oid::regclass AND lane = $2 AND namespace = $3",
            &[relid.into(), lane.as_str().into(), namespace.into()],
        )
    })
    .expect("pg_pinecone: partition registry delete failed");
}

pub(crate) fn forget_partitions(relid: pg_sys::Oid, lane: Lane) {
    as_owner(|| {
        Spi::run_with_args(
            "DELETE FROM pinecone.partitions WHERE relid = $1::oid::regclass AND lane = $2",
            &[relid.into(), lane.as_str().into()],
        )
    })
    .expect("pg_pinecone: partition registry delete failed");
}

fn text(row: &pgrx::spi::SpiHeapTupleData<'_>, name: &str) -> String {
    row.get_by_name::<String, _>(name)
        .unwrap_or_else(|e| panic!("pg_pinecone: partition registry row {name} query failed: {e}"))
        .unwrap_or_else(|| panic!("pg_pinecone: partition registry row {name} unexpectedly NULL"))
}

/// Empties every namespace registered for `relid` in `lane`, stopping at the first refusal, and
/// returns how many it emptied. Vector namespaces are deleted outright; document namespaces
/// are emptied with `delete_all`, since namespace operations on document indexes are not
/// verified against live Pinecone. The registry keeps its rows: the caller forgets them once
/// its own step is done, so a retry covers the same namespaces.
pub(crate) fn wipe_partitions(
    relid: pg_sys::Oid,
    lane: Lane,
    client: &Client,
    index: &str,
) -> Result<usize, ClientError> {
    let partitions = partitions_of(relid, lane);
    for (_, namespace) in &partitions {
        pgrx::check_for_interrupts!();
        match lane {
            Lane::Vector => client.delete_namespace(index, namespace),
            Lane::Docs => client.delete_docs(&DeleteDocsRequest {
                index: index.to_string(),
                namespace: Some(namespace.clone()),
                selector: DocDeleteSelector::All,
            }),
        }
        .map_err(|e| partition_lane::name_namespace(e, namespace))?;
    }
    Ok(partitions.len())
}

/// The registration that holds a claim in an index.
pub(crate) struct Owner {
    pub(crate) relid: pg_sys::Oid,
    pub(crate) namespace: String,
    pub(crate) partitioned: bool,
}

/// The `WHERE` condition over a registry's `namespace` and `partition_column` that selects the
/// rows clashing with `claim`, with the claim's namespace as `$2`. It mirrors
/// [`Claim::conflicts_with`], which the tests hold it to.
pub(crate) fn overlap(claim: Claim<'_>) -> &'static str {
    match claim {
        Claim::Exact(_) => {
            "(namespace = $2 OR (partition_column IS NOT NULL \
              AND pg_catalog.starts_with($2, namespace || ':')))"
        }
        Claim::Family(_) => {
            "(namespace = $2 OR pg_catalog.starts_with(namespace, $2 || ':') \
              OR (partition_column IS NOT NULL AND pg_catalog.starts_with($2, namespace || ':')))"
        }
    }
}

/// Raises 42710 when `claim` clashes with `owner` through a partition family; a clash of two
/// exact namespaces is left to the caller's own message.
pub(crate) fn refuse_family_clash(func: &str, index: &str, claim: Claim<'_>, owner: &Owner) {
    let held = Claim::of(&owner.namespace, owner.partitioned);
    if matches!((claim, held), (Claim::Exact(_), Claim::Exact(_))) {
        return;
    }
    let table = catalog::display_name_of(owner.relid);
    backend::error_with_code(
        PgSqlErrorCode::ERRCODE_DUPLICATE_OBJECT,
        partition::clash_message(func, index, claim, &table, held),
        None,
        Some(partition::CLASH_HINT),
    )
}
