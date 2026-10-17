//! The verbs over failed partitions: the list, and the retry that rebuilds one of them, or all,
//! from the table's rows. A partition fails when Pinecone refuses its namespace for good (see
//! `partition_lane::Gate`); the vector lane skips it until it is retried.

use pgrx::{PgRelation, pg_sys, prelude::*, spi::quote_literal};

use crate::{
    backend::error_with_code,
    catalog,
    docs_registry::{self, as_owner},
    docs_search::PartitionArg,
    docs_track, outbox,
    partition::Lane,
    partition_ops::{self, Target},
    partition_registry::{self, Failed},
    registry,
};

struct VectorLane {
    target: Target,
    /// Whether its ids are TIDs (an access-method index).
    tid: bool,
    /// Why the queue is parked, if it is.
    parked: Option<String>,
}

/// The vector lane of `relid`, refused when the table has no vector lane, no partition column or
/// is not promoted.
fn vector_lane(func: &str, relid: pg_sys::Oid, display: &str) -> VectorLane {
    let row = as_owner(|| registry::lookup(relid));
    let Some(row) = row else {
        if as_owner(|| docs_registry::lookup(relid)).is_some() {
            error_with_code(
                PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
                format!(
                    "{func}: {display} has no vector lane: a document lane dead-letters per document"
                ),
                None,
                Some("pinecone.failed_docs lists them and pinecone.retry_failed_docs retries them"),
            );
        }
        error_with_code(
            PgSqlErrorCode::ERRCODE_UNDEFINED_OBJECT,
            format!(
                "{func}: {display} is not tracked by pinecone.promote or pinecone.create_index"
            ),
            None,
            None,
        );
    };
    let tid = row.synced_by == "am";
    let syncs = row.syncs();
    let (Some(spec), Some(index), Some(prefix)) = (row.partition, row.index_name, row.namespace)
    else {
        error_with_code(
            PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
            format!("{func}: {display} has no partition column, so it has no partitions to fail"),
            None,
            None,
        );
    };
    if !syncs {
        error_with_code(
            PgSqlErrorCode::ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE,
            format!(
                "{func}: {display} is not promoted (mode = '{}') — run pinecone.promote(...) first",
                row.mode
            ),
            None,
            Some(
                "a bare demote keeps the failed partitions; a promote loads every partition again and clears them",
            ),
        );
    }
    let parked = row
        .queue_parked
        .then(|| row.parked_reason.unwrap_or_default());
    VectorLane {
        target: Target {
            lane: Lane::Vector,
            index,
            prefix,
            spec,
        },
        tid,
        parked,
    }
}

/// Implements `pinecone.failed_partitions`: the partitions the vector lane skips, with when and
/// why they failed, for the table's owner; none for a table that is not synced (a bare `demote`
/// keeps the state).
pub(crate) fn failed_partitions(tbl: &PgRelation) -> Vec<Failed> {
    let display = catalog::display_name(tbl);
    docs_track::require_owner(tbl.oid(), &display, "pinecone.failed_partitions");
    let syncs = as_owner(|| registry::lookup(tbl.oid())).is_some_and(|row| row.syncs());
    if !syncs {
        return Vec::new();
    }
    partition_registry::failed_of(tbl.oid(), Lane::Vector)
}

/// Clears the failures and queues one `rebuild_partition` job per partition, through the owner
/// path like every registry write here. Nothing is sent to Pinecone: the worker deletes each
/// namespace and loads the rows, so the table's lock is held for queries only and the caller
/// needs neither a key nor a grant on the queue.
fn retry(func: &str, display: &str, relid: pg_sys::Oid, lane: &VectorLane, partitions: &[&Failed]) {
    for partition in partitions {
        partition_registry::clear_failed(relid, Lane::Vector, &partition.namespace);
        as_owner(|| outbox::enqueue_partition_rebuild(relid, &partition.value, lane.tid));
        notice!(
            "{func}: partition {} ({}) of {display} is queued for a rebuild: the worker deletes its \
             namespace and loads the table's rows, and a search of it sees only what was written \
             since until the load has finished",
            quote_literal(&partition.value),
            partition.namespace
        );
    }
    if let Some(reason) = &lane.parked {
        notice!(
            "{func}: {display} is parked ({reason}): the worker skips it, so nothing is rebuilt \
             until pinecone.outbox_retry({}) unparks it",
            quote_literal(display)
        );
    }
}

/// Implements `pinecone.retry_failed_partition`: rebuilds one failed partition from the table.
pub(crate) fn retry_failed_partition(tbl: PgRelation, value: Option<PartitionArg>) {
    const FUNC: &str = "pinecone.retry_failed_partition";
    let relid = tbl.oid();
    let display = catalog::display_name(&tbl);
    docs_track::require_owner(relid, &display, FUNC);
    let text = partition_ops::value_text(FUNC, value);

    // The lock first: a `demote` that holds it changes the mode this reads.
    outbox::lock_relid(relid);
    let lane = vector_lane(FUNC, relid, &display);
    let namespace = partition_ops::namespace_of(FUNC, &lane.target, &text);
    let failed = partition_registry::failed_of(relid, Lane::Vector);
    let Some(chosen) = failed.iter().find(|f| f.namespace == namespace) else {
        let hint = if failed.is_empty() {
            format!("no partition of {display} is failed")
        } else {
            let values: Vec<&str> = failed.iter().map(|f| f.value.as_str()).collect();
            format!("failed partitions: {}", values.join(", "))
        };
        error_with_code(
            PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
            format!(
                "{FUNC}: partition {} of {display} is not failed",
                quote_literal(&text)
            ),
            None,
            Some(&hint),
        );
    };
    retry(FUNC, &display, relid, &lane, &[chosen]);
}

/// Implements `pinecone.retry_failed_partitions`: rebuilds every failed partition and says how
/// many it put back.
pub(crate) fn retry_failed_partitions(tbl: PgRelation) -> i32 {
    const FUNC: &str = "pinecone.retry_failed_partitions";
    let relid = tbl.oid();
    let display = catalog::display_name(&tbl);
    docs_track::require_owner(relid, &display, FUNC);

    outbox::lock_relid(relid);
    let lane = vector_lane(FUNC, relid, &display);
    let failed = partition_registry::failed_of(relid, Lane::Vector);
    if failed.is_empty() {
        notice!("{FUNC}: no partition of {display} is failed");
        return 0;
    }
    retry(
        FUNC,
        &display,
        relid,
        &lane,
        &failed.iter().collect::<Vec<_>>(),
    );
    i32::try_from(failed.len()).unwrap_or(i32::MAX)
}
