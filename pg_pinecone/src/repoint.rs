//! Implementation of `pinecone.repoint`: re-points a
//! promoted table's registry row at a different Pinecone index, same
//! namespace — the second step of a restore, after a Pinecone backup has
//! been restored to a brand-new index (Pinecone restores are ALWAYS to a
//! new index, never in place). Updates a registry ROW only (`UPDATE
//! pinecone.tables`), never its schema.
//!
//! A table with a partition column moves as a family: the prefix stays, the
//! index changes, and [`partition_ops::check_family`] lists the prefix on the new
//! index, registers the namespaces the registry lacks and warns about the registered ones
//! it lacks. An access-method table is still refused (`promote::remote_row`).

use pg_pinecone_client::Error as ClientError;
use pgrx::{PgRelation, prelude::*};

use crate::{
    backend, catalog, outbox, partition::Claim, partition_ops, partition_registry, promote,
    registry,
};

/// Validates `new_index`'s dimension AND metric against `tbl`'s registration
/// before updating `pinecone.tables.index_name` — refuses a mismatch with a
/// clear error rather than silently repointing at an incompatible index
/// (which would make every subsequent query/sync/recover fail confusingly
/// downstream instead). `tbl` must already be promoted (`mode = 'remote'`
/// exactly — not `'promoting'`: repointing a table whose async backfill
/// hasn't exhausted yet would leave that job writing against the OLD
/// index_name/namespace snapshot it captured at dispatch time). Also refuses
/// while a `'recover'` job is queued: the job's persisted
/// pagination token — and any ids it already listed — belongs to the CURRENT
/// index; repointing under it would replay a stale cursor against the new one.
pub(crate) fn repoint(tbl: PgRelation, new_index: Option<&str>) {
    let relid = tbl.oid();
    let display = catalog::display_name(&tbl);
    let new_index = backend::require_text("pinecone.repoint", "new_index", new_index);
    let row = promote::remote_row(&tbl, "pinecone.repoint");

    // Same uniform advisory-lock-first order every worker-round-adjacent
    // caller uses (`outbox::lock_relid`'s rustdoc) — also serializes this
    // check against a concurrent worker round mid-recover-chunk.
    outbox::lock_relid(relid);
    if outbox::has_queued_op(relid, "recover") {
        error!(
            "pinecone.repoint: {display} has a recover job queued or in progress — its persisted \
             pagination cursor belongs to the current index and would replay against the new \
             one; wait for it to finish (see pinecone.status()), or pinecone.outbox_discard(...) it \
             first"
        );
    }

    let namespace = row
        .namespace
        .as_deref()
        .expect("pinecone.repoint: remote row always has namespace");
    let claim = Claim::of(namespace, row.partition.is_some());
    if let Some(owner) = registry::claim_owner(&new_index, claim, relid) {
        partition_registry::refuse_family_clash("pinecone.repoint", &new_index, claim, &owner);
        let owner_display = catalog::display_name_of(owner.relid);
        error!(
            "pinecone.repoint: index '{new_index}' namespace '{namespace}' is already owned by \
             {owner_display} — repointing {display} there would collide"
        );
    }

    let client = backend::client_for_effective_key();
    let info = client
        .describe_index(&new_index)
        .unwrap_or_else(|e| match e {
            ClientError::IndexNotFound { .. } => error!(
                "pinecone.repoint: index '{new_index}' not found — restore the backup to a new index \
             first (see docs/disaster-recovery.md), then repoint at its real name"
            ),
            other => backend::raise(other),
        });

    let mapped_metric = promote::map_metric(&row.metric);
    if info.dimension != row.dimension as u32 {
        error!(
            "pinecone.repoint: index '{new_index}' has dimension {}, but {display} is registered \
             with dimension {} — refusing a dimension mismatch",
            info.dimension, row.dimension
        );
    }
    if info.metric != mapped_metric {
        error!(
            "pinecone.repoint: index '{new_index}' has metric '{}', but {display} is registered \
             with metric '{}' (maps to Pinecone metric '{mapped_metric}') — refusing a metric \
             mismatch",
            info.metric, row.metric
        );
    }
    if info.vector_type != "dense" {
        error!(
            "pinecone.repoint: index '{new_index}' has vector_type '{}' — pg_pinecone only supports \
             dense indexes",
            info.vector_type
        );
    }

    // A partitioned table moves as a family: the prefix stays and the index changes, and the new
    // index is checked for the registered namespaces (recovery walks those).
    let unchanged = match &row.partition {
        Some(spec) => {
            let found = partition_ops::check_family(
                "pinecone.repoint",
                &tbl,
                &row,
                spec,
                &new_index,
                &client,
            );
            format!("partition family '{namespace}:*' unchanged{found}")
        }
        None => format!("namespace '{namespace}' unchanged"),
    };
    registry::repoint_index(relid, &new_index);
    notice!(
        "pinecone.repoint: {display} now points at index '{new_index}' ({unchanged}) — run \
         pinecone.recover_from_remote('{display}') to restore vectors from it"
    );
}
