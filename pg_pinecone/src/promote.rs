//! Implementation of `pinecone.promote` / `pinecone.demote` / `pinecone.sync`.
//! `#[pg_schema]` only attaches to an inline `mod {}` (see the comment on
//! `pg_pinecone` in `lib.rs`), so its `#[pg_extern]` wrappers are kept thin and
//! delegate here.
//!
//! [`topk::topk`](crate::topk::topk) is `pg_pinecone`'s mode *read* dispatch
//! point; this module is the mode *write* dispatch point — [`promote`] is the
//! only place a row's `mode` flips `'local' -> 'remote'`, and [`demote`] the
//! only place it flips back.

use std::time::Duration;

use pg_pinecone_client::{
    Client, CreateIndexRequest, DeleteVectorsRequest, Error as ClientError, UPSERT_MAX_BYTES,
    UPSERT_MAX_RECORDS, UpsertVectorsRequest, Vector,
};
use pgrx::{JsonB, PgRelation, prelude::*, spi::quote_qualified_identifier};
use serde_json::{Value, json};

use crate::{
    am_filter, am_partition,
    backend::{self, require_text},
    catalog, docs_registry,
    filter::ColKind,
    local::metapage::FilterKind,
    outbox,
    partition::{self, Claim, Lane, PartitionKind, PartitionSpec},
    partition_lane::{self, Gate},
    partition_ops, partition_registry,
    registry::{self, RegistryRow},
    secret, track,
};

/// How long `pinecone.promote` waits for a newly created (or already-existing)
/// index to report `ready` before giving up.
const WAIT_READY_DEADLINE: Duration = Duration::from_secs(120);
/// How often `pinecone.promote` polls `describe_index` while waiting.
const WAIT_READY_POLL: Duration = Duration::from_secs(2);

/// Maximum index-name length Pinecone accepts; `default_index_name` truncates
/// to this after sanitizing.
const MAX_INDEX_NAME_LEN: usize = 45;

/// Rows of one backfill request, which is also the most a chunk reads. `pub(crate)` so tests can
/// build a fixture one row past it.
pub(crate) const BACKFILL_MAX_ROWS: usize = UPSERT_MAX_RECORDS;

/// Rows a chunk reads: what fits a request at 4 bytes a value, so that the byte budget, not this
/// limit, ends the chunk, and no more rows are read than a request can hold.
fn chunk_rows(dimension: i32) -> usize {
    let value_bytes = 4 * dimension.max(1) as usize;
    (UPSERT_MAX_BYTES / value_bytes).clamp(1, BACKFILL_MAX_ROWS)
}

/// Promotes `tbl` to `'remote'` mode: auto-tracks it if untracked (reusing
/// `pinecone.track`'s validation and defaults), creates (or adopts) a matching
/// serverless index, installs the sync triggers, and backfills every current
/// row. See the `pinecone.promote` rustdoc in `lib.rs` for the full contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn promote(
    tbl: PgRelation,
    index_name: Option<&str>,
    namespace: Option<&str>,
    cloud: Option<&str>,
    region: Option<&str>,
    spec: Option<JsonB>,
    vector_column: Option<&str>,
    id_column: Option<&str>,
    metric: Option<&str>,
    metadata_columns: Option<Vec<Option<String>>>,
    sync_mode: Option<&str>,
    partition_column: Option<&str>,
) {
    let relid = tbl.oid();
    let display = catalog::display_name(&tbl);

    // Auto-tracks if untracked, updates in place if already local, and raises
    // `pinecone.track`'s own clear error if already `'remote'` — exactly the
    // "already-remote -> error" behavior `pinecone.promote` needs, for free.
    track::track(
        tbl.clone(),
        vector_column,
        id_column,
        metric,
        metadata_columns,
    );
    let mut row = registry::lookup(relid).expect("pinecone.promote: just tracked, row must exist");

    let cloud = require_text("pinecone.promote", "cloud", cloud);
    let region = require_text("pinecone.promote", "region", region);

    // Priority: explicit arg > the registry row's own preserved value (set by
    // an earlier promote, surviving a bare `demote()`) > the computed
    // `pv-<schema>-<table>`/`<schema>.<table>` default — a bare re-promote
    // must reuse whatever it was promoted under before, not silently
    // re-derive a name that may no longer match the live remote index.
    let index_name = index_name
        .map(str::to_string)
        .or_else(|| row.index_name.clone())
        .unwrap_or_else(|| default_index_name(tbl.namespace(), tbl.name()));
    let namespace = namespace
        .map(str::to_string)
        .or_else(|| row.namespace.clone())
        .unwrap_or_else(|| format!("{}.{}", tbl.namespace(), tbl.name()));

    // Same early-validation spirit as `require_text` above: an invalid
    // `sync_mode` should fail before any network call, not after.
    let sync_mode = resolve_sync_mode(sync_mode, &row);

    let partition =
        partition_lane::declare(&tbl, &row, partition_column, &namespace, "pinecone.promote");
    let claim = match partition {
        Some(_) => Claim::Family(&namespace),
        None => Claim::Exact(&namespace),
    };
    if let Some(owner) = registry::claim_owner(&index_name, claim, relid) {
        partition_registry::refuse_family_clash("pinecone.promote", &index_name, claim, &owner);
        let owner_display = catalog::display_name_of(owner.relid);
        error!(
            "pinecone.promote: index '{index_name}' namespace '{namespace}' is already owned by \
             {owner_display} — pass a different index_name/namespace"
        );
    }

    registry::set_partition(relid, partition.as_ref());
    let unpartitioned_namespace = unpartitioned_namespace_to_clear(&row, &index_name, &partition);
    row.partition = partition;

    let mapped_metric = map_metric(&row.metric);
    let spec_value = spec
        .map(|j| j.0)
        .unwrap_or_else(|| json!({"serverless": {"cloud": cloud, "region": region}}));

    let client = backend::client_for_effective_key();

    // A session-only key (the GUC alone, no stored secret) would let this
    // promote succeed while stranding every *other* session's sync trigger —
    // those run `SECURITY DEFINER` and fall back to the stored secret, which
    // wouldn't exist. This check is independent of how `client` above
    // resolved its own credentials: it always asks whether a key is stored,
    // never merely whether the current session can authenticate.
    if !secret::has_stored_api_key() {
        error!(
            "pinecone.promote: writers' sync triggers need pinecone.set_api_key(...) — a \
             session-only key would break other sessions' writes"
        );
    }

    let read_capacity = create_or_adopt_index(
        &client,
        &index_name,
        row.dimension as u32,
        mapped_metric,
        spec_value,
        &display,
    );
    registry::set_read_capacity(relid, read_capacity.as_deref());
    wait_ready(&client, &index_name);
    if let Some(stale) = unpartitioned_namespace {
        delete_all_in(&client, &index_name, &stale);
    }

    if row.metadata_columns.is_empty() {
        warning!(
            "pinecone.promote: {display} has no metadata_columns — remote filters will be \
             unavailable until you set some and re-promote"
        );
    }

    // Order matters from here: installing the triggers first means no write
    // lands between "index ready" and "triggers armed".
    //
    // `sync_mode = 'sync'` then rebuilds the namespace inline (deleteAll +
    // backfill) and flips straight to `'remote'`, same as 0.2 — correct even
    // on a re-promote of a previously-demoted table.
    //
    // `sync_mode = 'async'` instead flips to `'promoting'` and *enqueues* a
    // `delete_all` + `backfill` job: both writes
    // land in this same, fast transaction as `install_triggers`, so the
    // `CREATE TRIGGER`s' `AccessExclusiveLock` window collapses to
    // milliseconds regardless of table size, instead of being held for the
    // whole (potentially very long) backfill. `outbox`'s own backfill
    // dispatch is the only place that later flips `'promoting'` ->
    // `'remote'`, once it exhausts; `pinecone.topk` treats `'promoting'` as
    // local in the meantime (never serving a partial namespace).
    install_triggers(&tbl);
    if sync_mode == "async" {
        registry::promote_promoting(relid, &index_name, &namespace, &sync_mode);
        outbox::enqueue_rebuild(relid);
    } else {
        delete_all_and_backfill(
            "pinecone.promote",
            &tbl,
            &row,
            &client,
            &index_name,
            &namespace,
        );
        registry::promote_remote(relid, &index_name, &namespace, &sync_mode);
    }

    // `sync_mode = 'async'` writes only ever drain via the background
    // worker — with no launcher alive-and-enabled, they'd queue in
    // `pinecone.outbox` forever and never reach the remote index.
    // Launcher liveness, not the per-database worker's own lock:
    // the worker idle-exits between backlogs by design (see
    // `outbox::launcher_lock_held`), so probing it here would warn
    // spuriously on every promote landing in an idle window.
    if sync_mode == "async" && !outbox::launcher_lock_held() {
        warning!(
            "pinecone.promote: {display} is sync_mode = 'async' but no pg_pinecone worker launcher \
             is running in this cluster (preload absent or pinecone.worker_enabled off) — writes \
             will queue in pinecone.outbox but nothing will drain them until one starts (see \
             pinecone.status())"
        );
    }
}

/// The namespace a table wrote before it was partitioned, which no partitioned query or wipe
/// reaches: `Some` when this promote declares the partition on a table that held one namespace in
/// this very index.
fn unpartitioned_namespace_to_clear(
    row: &RegistryRow,
    index_name: &str,
    declared: &Option<PartitionSpec>,
) -> Option<String> {
    if declared.is_none() || row.partition.is_some() {
        return None;
    }
    if row.index_name.as_deref() != Some(index_name) {
        return None;
    }
    row.namespace.clone()
}

fn delete_all_in(client: &Client, index_name: &str, namespace: &str) {
    client
        .delete_vectors(&DeleteVectorsRequest {
            index: index_name.to_string(),
            namespace: Some(namespace.to_string()),
            ids: None,
            delete_all: true,
            filter: None,
        })
        .unwrap_or_else(|e| backend::raise(e));
}

/// Resolves `sync_mode`: the explicit value if given (validated against
/// `'async'`/`'sync'`); else, on a genuinely new promotion — `row.index_name`
/// is `None`, meaning this table has never been promoted before — the
/// `pinecone.sync_mode` GUC's current value; else (a bare re-promote,
/// including one after a `demote()` that kept `index_name`/`namespace`) the
/// row's own already-stored `sync_mode`, unchanged — the same
/// preserve-registration convention `index_name`/`namespace` use just above,
/// reusing the same "has this row ever been promoted" signal.
fn resolve_sync_mode(explicit: Option<&str>, row: &RegistryRow) -> String {
    if let Some(mode) = explicit {
        if mode != "async" && mode != "sync" {
            error!(
                "pinecone.promote: sync_mode '{mode}' is not supported — must be 'async' or 'sync'"
            );
        }
        return mode.to_string();
    }
    if row.index_name.is_some() {
        return row.sync_mode.clone();
    }
    crate::SYNC_MODE.get().as_str().to_string()
}

/// Demotes `tbl` back to `'local'` mode: drops the sync triggers and, if
/// `drop_remote_index`, deletes the remote index too (refusing if another
/// tracked table still uses it). The registry row (and, unless
/// `drop_remote_index`, its `index_name`/`namespace`) survives, so a bare
/// `pinecone.promote(tbl)` re-promote reuses them.
///
/// Also clears every `pinecone.outbox` row still queued for `relid`, in the
/// same transaction as the trigger drop: a demoted table stops enqueuing new
/// writes, but a stale backlog (e.g. from `sync_mode = 'async'`) would
/// otherwise sit untouched until a later re-promote's first `sync_mode =
/// 'sync'` write drains it — replaying pre-demote payloads over whatever
/// `pinecone.promote`'s own backfill just wrote from the *current* local rows.
///
/// Takes `relid`'s advisory transaction lock ([`outbox::lock_relid`]) FIRST —
/// before `drop_triggers`'s `DROP TRIGGER` (an `AccessExclusiveLock` on
/// `tbl`) or any registry/outbox write (uniform lock
/// order: advisory -> table locks -> registry/outbox rows). A worker round
/// takes the very same lock before it ever requests its own weaker
/// `AccessShareLock` read of the table (`outbox::dispatch_backfill`); taking
/// the two locks in a *consistent* order everywhere is what rules out an ABBA
/// deadlock between this call and a worker round — the previous claim that
/// the outbox rows' own `FOR UPDATE` locking sufficed was wrong: that only
/// serializes the `DELETE` below against a concurrent *inline* drain (see
/// `outbox::drain_step`'s own comment on why `Inline` skips the advisory
/// lock), not against a worker round, which can be genuinely concurrent with
/// this call and blocks on `tbl`'s lock rather than on any outbox row.
///
/// Works on a `mode = 'promoting'` row too: an async
/// promote whose backfill job hasn't drained yet still has real triggers/an
/// `index_name` installed, and demoting it must drop the triggers and purge
/// the outbox (backfill job included) just like a fully `'remote'` row —
/// see [`demotable_row`].
pub(crate) fn demote(tbl: PgRelation, drop_remote_index: bool) {
    let relid = tbl.oid();
    outbox::lock_relid(relid);
    let row = demotable_row(&tbl, "pinecone.demote");
    // Demoting a detached table would silently abandon
    // Pinecone's only copy of the vector column (dropping the triggers with
    // no further sync at all, and — with `drop_remote_index => true` —
    // deleting the index outright). The only sanctioned way back to
    // `storage = 'full'` is `pinecone.attach_storage(tbl)`.
    registry::require_not_detached(&row, "pinecone.demote", &catalog::display_name(&tbl));

    // The `other_row_uses_index` refusal must be checked *before* dropping
    // anything: otherwise a refused `drop_remote_index => true` call would
    // still have torn down the sync triggers (leaving the registry row
    // claiming `mode = 'remote'` with no triggers actually installed).
    if drop_remote_index {
        let index_name = row
            .index_name
            .as_deref()
            .expect("pinecone.demote: remote row always has index_name");
        if registry::other_row_uses_index(index_name, relid) {
            error!(
                "pinecone.demote: index '{index_name}' is also used by another tracked table — \
                 demote that table first (or drop the index manually) rather than \
                 drop_remote_index => true"
            );
        }
        drop_triggers(&tbl);
        backend::client_for_effective_key()
            .delete_index(index_name)
            .unwrap_or_else(|e| backend::raise(e));
        partition_registry::forget_partitions(relid, Lane::Vector);
        registry::demote_local(relid, true);
    } else {
        drop_triggers(&tbl);
        registry::demote_local(relid, false);
    }
    outbox::clear_relid(relid);
    // A stale parked flag must never survive into a re-promote that hasn't
    // even run yet — outbox::clear_relid just emptied the queue it was
    // parked over.
    registry::unpark(relid);
}

/// Re-syncs `tbl`'s remote namespace from its current local rows.
/// `sync_mode = 'sync'` tables rebuild inline (`deleteAll(namespace)` then
/// backfill — the same repair `pinecone.promote` performs, factored out so
/// both go through one code path); `sync_mode = 'async'` tables instead flip
/// to `'promoting'` and enqueue a `delete_all` + `backfill` job
/// — the worker rebuilds the namespace the same way
/// `pinecone.promote`'s own async path does, and `pinecone.topk` serves local
/// in the meantime (see [`registry::flip_remote_to_promoting`]) rather than a
/// namespace the coalesced `delete_all` may already be mid-emptying.
///
/// Takes `relid`'s advisory transaction lock ([`outbox::lock_relid`]) FIRST,
/// for its whole duration — before any registry write, and (for the
/// `sync_mode = 'sync'` branch) before the inline `deleteAll`+backfill HTTP
/// calls below, which touch the remote namespace without going through
/// `pinecone.outbox` at all. See [`demote`]'s rustdoc for the uniform lock
/// order this and every other worker-round-adjacent caller now share
/// and the ABBA cycle it rules out.
///
/// Ownership-gated FIRST THING (mirrors `lifecycle::require_table_owner` / `maintain::
/// require_table_owner`): before this gate, the only thing standing between
/// any EXECUTE-holding role (PUBLIC by default) and a `sync_mode = 'sync'`
/// table's inline `deleteAll(namespace)` was the incidental `pinecone.tables`
/// SELECT inside [`remote_row`] — a role granted registry read access (e.g.
/// for `pinecone.status()` visibility) would have sailed straight into
/// destroying and rebuilding someone else's namespace against the stored
/// account. Checked before the registry read (authorization before
/// diagnosis) AND before the advisory lock, so a non-owner can't even
/// contend the worker's lock order.
pub(crate) fn sync(tbl: PgRelation) {
    let relid = tbl.oid();
    require_table_owner(relid, &catalog::display_name(&tbl), "pinecone.sync");
    outbox::lock_relid(relid);
    let row = remote_row(&tbl, "pinecone.sync");
    // sync() is `deleteAll(namespace)` + a backfill
    // that skips every NULL-vector row — on a detached table (whose vector
    // column is NULL by definition) that's a total remote wipe with nothing
    // to refill it. The only sanctioned way back to `storage = 'full'` is
    // `pinecone.attach_storage(tbl)`.
    registry::require_not_detached(&row, "pinecone.sync", &catalog::display_name(&tbl));
    let index_name = row
        .index_name
        .clone()
        .expect("pinecone.sync: remote row always has index_name");
    let namespace = row
        .namespace
        .clone()
        .expect("pinecone.sync: remote row always has namespace");
    if row.sync_mode == "async" {
        // The queued wipe adopts the orphans of the family too; doing it here as well puts the
        // WARNINGs in front of the caller and stops a failed listing before anything is queued.
        if let Some(spec) = &row.partition {
            let client = backend::client_for_effective_key();
            partition_ops::adopt_orphans(
                "pinecone.sync",
                &tbl,
                spec,
                &index_name,
                &namespace,
                &client,
            );
        }
        registry::flip_remote_to_promoting(relid);
        outbox::enqueue_rebuild(relid);
        return;
    }
    let client = backend::client_for_effective_key();
    delete_all_and_backfill(
        "pinecone.sync",
        &tbl,
        &row,
        &client,
        &index_name,
        &namespace,
    );
}

/// Requires the calling role to own the TABLE `heap_relid` — or be a member
/// of the owning role, or superuser. A mirror of
/// `lifecycle::require_table_owner` for this
/// module's verbs, with a message that fits them (no "engine" wording);
/// see [`sync`]'s rustdoc for why the incidental `pinecone.tables` ACL is not
/// a substitute for this gate.
fn require_table_owner(heap_relid: pg_sys::Oid, display: &str, func: &str) {
    let caller = unsafe { pg_sys::GetUserId() };
    let owns = unsafe { pg_sys::object_ownercheck(pg_sys::RelationRelationId, heap_relid, caller) };
    if !owns {
        error!("{func}: must own {display} (or be superuser)");
    }
}

/// Looks up `tbl`'s registry row, requiring it to exist and be `'remote'`
/// (`pinecone.sync`'s own precondition — unlike `pinecone.demote`, which also
/// accepts `'promoting'`; see [`demotable_row`]). `func` names the calling
/// SQL function in the error message. `pub(crate)` so `repoint::repoint`
/// (a different module, same precondition — repointing a still-mid-backfill
/// row would confuse its in-flight job) can reuse it.
pub(crate) fn remote_row(tbl: &PgRelation, func: &str) -> RegistryRow {
    let relid = tbl.oid();
    let display = catalog::display_name(tbl);
    let row = registry::lookup(relid).unwrap_or_else(|| {
        error!(
            "{func}: {display} is not tracked — run pinecone.track(...) or pinecone.promote(...) first"
        )
    });
    registry::require_not_am(&row, func, &display);
    if row.mode != "remote" {
        error!(
            "{func}: {display} is not promoted (mode = '{}') — run pinecone.promote(...) first",
            row.mode
        );
    }
    row
}

/// Like [`remote_row`], but also accepts `mode = 'promoting'` —
/// `pinecone.demote`'s own precondition: it must work mid-flight, on an async
/// promote whose backfill job hasn't drained yet. Not
/// reused by `recover::recover_from_remote` despite the same mode
/// acceptance: this rejects `synced_by = 'am'` rows (via
/// `registry::require_not_am`), and `pinecone.recover_from_remote` is
/// explicitly meant to work on those too — see
/// `recover::recoverable_row`, which duplicates just the mode check.
fn demotable_row(tbl: &PgRelation, func: &str) -> RegistryRow {
    let relid = tbl.oid();
    let display = catalog::display_name(tbl);
    let row = registry::lookup(relid).unwrap_or_else(|| {
        error!(
            "{func}: {display} is not tracked — run pinecone.track(...) or pinecone.promote(...) first"
        )
    });
    registry::require_not_am(&row, func, &display);
    if row.mode != "remote" && row.mode != "promoting" {
        error!(
            "{func}: {display} is not promoted (mode = '{}') — run pinecone.promote(...) first",
            row.mode
        );
    }
    row
}

/// `deleteAll(namespace)` then backfill every current local row — the shared
/// tail of `promote` and `sync` (`func` names the caller in warnings). A partitioned table
/// empties its whole family: the registered namespaces and the orphans under the prefix, adopted
/// first. A listing that fails raises before anything is wiped.
fn delete_all_and_backfill(
    func: &str,
    tbl: &PgRelation,
    row: &RegistryRow,
    client: &Client,
    index_name: &str,
    namespace: &str,
) {
    if let Some(spec) = &row.partition {
        partition_ops::adopt_orphans(func, tbl, spec, index_name, namespace, client);
        partition_registry::wipe_partitions(tbl.oid(), Lane::Vector, client, index_name)
            .unwrap_or_else(|e| backend::raise(e));
        partition_registry::forget_partitions(tbl.oid(), Lane::Vector);
    } else {
        delete_all_in(client, index_name, namespace);
    }
    backfill(tbl, row, client, index_name, namespace);
}

/// The message for a `wait_index_ready` timeout: names the deadline and the
/// index, and deliberately never mentions `pinecone.timeout_ms` (that GUC
/// bounds a single HTTP request, not this multi-minute readiness wait).
/// Factored out (rather than inlined at the one call site) so it has a pure,
/// pgrx-free unit test.
fn wait_ready_timeout_message(index_name: &str) -> String {
    format!(
        "pinecone.promote: index '{index_name}' did not become ready within the {}s readiness \
         deadline — check the index status in the Pinecone console and retry",
        WAIT_READY_DEADLINE.as_secs()
    )
}

/// A name a `track_docs` table owns is a document index, never a vector index: erroring here
/// beats the dimension/metric mismatch Pinecone's schema-only index would produce.
fn require_not_doc_index(index_name: &str, display: &str) {
    let owner = docs_registry::readable()
        .then(|| docs_registry::index_owner(index_name))
        .flatten();
    if let Some(owner) = owner {
        backend::error_with_code(
            PgSqlErrorCode::ERRCODE_WRONG_OBJECT_TYPE,
            format!(
                "pg_pinecone: index '{index_name}' is the document index of {} — {display} needs a \
                 vector index",
                catalog::display_name_of(owner)
            ),
            None,
            Some(
                "pass a different index_name: a document index and a vector index cannot share a name",
            ),
        );
    }
}

/// Creates `index_name` (or, on a 409, adopts the pre-existing one — erroring
/// if its dimension/metric/vector-type don't match). Shared by
/// `pinecone.promote` and the `pg_pinecone` access method's `ambuild` (`am.rs`),
/// which both need this same "create or adopt" semantics against Pinecone.
pub(crate) fn create_or_adopt_index(
    client: &Client,
    index_name: &str,
    dimension: u32,
    mapped_metric: &'static str,
    spec: Value,
    display: &str,
) -> Option<String> {
    require_not_doc_index(index_name, display);
    let create_req = CreateIndexRequest {
        name: index_name.to_string(),
        dimension,
        metric: mapped_metric.to_string(),
        spec,
    };
    match client.create_index(&create_req) {
        Ok(info) => info.read_capacity.as_ref().map(format_read_capacity),
        Err(ClientError::Conflict { .. }) => {
            let info = client
                .describe_index(index_name)
                .unwrap_or_else(|e| backend::raise(e));
            if info.dimension != dimension
                || info.metric != mapped_metric
                || info.vector_type != "dense"
            {
                error!(
                    "pg_pinecone: existing index '{index_name}' has dimension={}, metric={}, \
                     vector_type={} but {display} needs dimension={dimension}, \
                     metric={mapped_metric}, vector_type=dense",
                    info.dimension, info.metric, info.vector_type
                );
            }
            info.read_capacity.as_ref().map(format_read_capacity)
        }
        Err(e) => backend::raise(e),
    }
}

/// Validates `pinecone_index` against `dimension`/`mapped_metric` and
/// (unless `attach_unchecked`) `namespace`'s live vector count against
/// `local_row_count`, then returns its formatted read capacity — never
/// creates or adopts anything (the index is assumed to already exist and
/// already hold the table's data). For a filtered index (`filter_column`)
/// the vectors must also carry `{filter_column: v}` metadata: nothing here can
/// check that (attach runs no backfill), so it says so.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attach_index(
    client: &Client,
    pinecone_index: &str,
    dimension: u32,
    mapped_metric: &'static str,
    namespace: &str,
    local_row_count: i64,
    attach_unchecked: bool,
    display: &str,
    filter_column: Option<&str>,
) -> Option<String> {
    require_not_doc_index(pinecone_index, display);
    let info = client
        .describe_index(pinecone_index)
        .unwrap_or_else(|e| backend::raise(e));
    if info.dimension != dimension || info.metric != mapped_metric || info.vector_type != "dense" {
        ereport!(
            ERROR,
            PgSqlErrorCode::ERRCODE_INTERNAL_ERROR,
            format!(
                "pg_pinecone: index '{pinecone_index}' does not match {display}'s column/opclass"
            ),
            format!(
                "index '{pinecone_index}': dimension={}, metric={}, vector_type={}; {display} \
                 needs: dimension={dimension}, metric={mapped_metric}, vector_type=dense",
                info.dimension, info.metric, info.vector_type
            )
        );
    }

    if !attach_unchecked {
        let stats = client
            .describe_index_stats(pinecone_index)
            .unwrap_or_else(|e| backend::raise(e));
        let remote_count = stats
            .namespaces
            .get(namespace)
            .map(|n| n.vector_count)
            .unwrap_or(0) as i64;
        if remote_count != local_row_count {
            let metadata = filter_column.map_or_else(String::new, |column| {
                format!(" (and that every vector carries {{\"{column}\": <value>}} metadata)")
            });
            backend::error_with_hint(
                format!(
                    "pg_pinecone: index '{pinecone_index}' namespace '{namespace}' does not look \
                     like it holds {display}'s current rows"
                ),
                format!(
                    "namespace '{namespace}' has {remote_count} vectors, {display} has \
                     {local_row_count} live rows — pass attach_unchecked => true to skip this \
                     check if you're sure they already match{metadata}"
                ),
            );
        }
    }
    if let Some(column) = filter_column {
        notice!(
            "pg_pinecone: {display} is filtered by \"{column}\": the attached namespace must already \
             carry {{\"{column}\": <value>}} metadata on every vector (attach runs no backfill), or \
             filtered reads miss rows"
        );
    }

    info.read_capacity.as_ref().map(format_read_capacity)
}

/// Renders a `ReadCapacity` for `pinecone.tables.read_capacity`/`status()` —
/// e.g. `"OnDemand"` or `"Dedicated b1 2x1 Ready"`. Parts Pinecone didn't
/// report are simply omitted rather than shown as a placeholder.
fn format_read_capacity(rc: &pg_pinecone_client::ReadCapacity) -> String {
    if rc.mode != "Dedicated" {
        return rc.mode.clone();
    }
    let mut parts = vec![rc.mode.clone()];
    parts.extend(rc.node_type.clone());
    if let (Some(shards), Some(replicas)) = (rc.shards, rc.replicas) {
        parts.push(format!("{shards}x{replicas}"));
    }
    parts.extend(rc.state.clone());
    parts.join(" ")
}

/// Waits for `index_name` to report `ready`, translating a deadline timeout
/// into [`wait_ready_timeout_message`] rather than `backend::raise`'s generic
/// (and here misleading) `pinecone.timeout_ms` text. Shared by
/// `pinecone.promote` and `am::ambuild`.
pub(crate) fn wait_ready(client: &Client, index_name: &str) {
    client
        .wait_index_ready(index_name, WAIT_READY_DEADLINE, WAIT_READY_POLL)
        .unwrap_or_else(|e| match e {
            ClientError::Timeout { .. } => error!("{}", wait_ready_timeout_message(index_name)),
            other => backend::raise(other),
        });
}

/// Maps a registered local metric to the Pinecone metric name. The registry's
/// `CHECK (metric IN ('cosine', 'l2', 'ip'))` makes the `other` arm
/// unreachable in practice.
pub(crate) fn map_metric(metric: &str) -> &'static str {
    match metric {
        "cosine" => "cosine",
        "l2" => "euclidean",
        "ip" => "dotproduct",
        other => error!("pinecone.promote: table has unknown registry metric '{other}'"),
    }
}

/// Builds the default `index_name`: `pv-<schema>-<table>`, sanitized to
/// `[a-z0-9-]` (everything else collapses to a single `-`), leading/trailing
/// `-` trimmed, truncated to [`MAX_INDEX_NAME_LEN`] (re-trimmed afterward, in
/// case truncation lands on a `-`).
pub(crate) fn default_index_name(schema: &str, table: &str) -> String {
    sanitize_index_name(&format!("pv-{schema}-{table}"))
}

pub(crate) fn sanitize_index_name(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut prev_dash = false;
    for c in raw.chars() {
        let c = c.to_ascii_lowercase();
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            out.push(c);
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    let truncated: String = out
        .trim_matches('-')
        .chars()
        .take(MAX_INDEX_NAME_LEN)
        .collect();
    truncated.trim_matches('-').to_string()
}

/// Installs the four statement-level sync triggers (see `sync_trigger.rs` for
/// the function they call). All identifiers here are either the fixed trigger
/// names/transition-table aliases or `tbl`'s own catalog-validated
/// schema-qualified name — no user SQL in the trigger bodies themselves.
fn install_triggers(tbl: &PgRelation) {
    let table = quote_qualified_identifier(tbl.namespace(), tbl.name());
    for stmt in [
        format!(
            "CREATE TRIGGER pg_pinecone_sync_ins AFTER INSERT ON {table} \
             REFERENCING NEW TABLE AS new_table \
             FOR EACH STATEMENT EXECUTE FUNCTION pinecone._sync_trigger()"
        ),
        format!(
            "CREATE TRIGGER pg_pinecone_sync_upd AFTER UPDATE ON {table} \
             REFERENCING NEW TABLE AS new_table OLD TABLE AS old_table \
             FOR EACH STATEMENT EXECUTE FUNCTION pinecone._sync_trigger()"
        ),
        format!(
            "CREATE TRIGGER pg_pinecone_sync_del AFTER DELETE ON {table} \
             REFERENCING OLD TABLE AS old_table \
             FOR EACH STATEMENT EXECUTE FUNCTION pinecone._sync_trigger()"
        ),
        format!(
            "CREATE TRIGGER pg_pinecone_sync_trunc AFTER TRUNCATE ON {table} \
             FOR EACH STATEMENT EXECUTE FUNCTION pinecone._sync_trigger()"
        ),
    ] {
        Spi::run(&stmt).unwrap_or_else(|e| error!("pinecone.promote: {e}"));
    }
}

/// Drops the four triggers [`install_triggers`] installs. Not `IF EXISTS`: a
/// `'remote'` registry row implies they exist, so a missing one is a real bug.
fn drop_triggers(tbl: &PgRelation) {
    let table = quote_qualified_identifier(tbl.namespace(), tbl.name());
    for name in [
        "pg_pinecone_sync_ins",
        "pg_pinecone_sync_upd",
        "pg_pinecone_sync_del",
        "pg_pinecone_sync_trunc",
    ] {
        Spi::run(&format!("DROP TRIGGER {name} ON {table}"))
            .unwrap_or_else(|e| error!("pinecone.demote: {e}"));
    }
}

/// The synchronous, whole-table backfill `sync_mode = 'sync'` promotes and
/// `pinecone.sync` still use: [`backfill_chunk`], looped to exhaustion,
/// within the caller's own transaction — unlike the async `'backfill'` job
/// (`outbox.rs`'s dispatch), whose chunks each commit separately, this runs
/// start to finish before returning.
fn backfill(
    tbl: &PgRelation,
    row: &RegistryRow,
    client: &Client,
    index_name: &str,
    namespace: &str,
) {
    if row.partition.is_some() {
        let mut after: Option<(String, String)> = None;
        let mut gate = Gate::new(false);
        loop {
            let cursor = after.as_ref().map(|(p, id)| (p.as_str(), id.as_str()));
            let chunk = try_backfill_chunk_partitioned(
                tbl, row, client, index_name, namespace, cursor, &mut gate,
            );
            match chunk.unwrap_or_else(|e| backend::raise(e)) {
                Some(last) => after = Some(last),
                None => return,
            }
        }
    }
    let mut after_id: Option<String> = None;
    while let Some(last_id) =
        backfill_chunk(tbl, row, client, index_name, namespace, after_id.as_deref())
    {
        after_id = Some(last_id);
    }
}

/// Reads and pushes one backfill "chunk": at most [`BACKFILL_MAX_ROWS`] rows
/// (further bounded to [`UPSERT_MAX_BYTES`] of request), in ascending
/// `id_column` order, strictly after `after_id` — compared/cast to the id
/// column's *native* type (a bare text comparison
/// would sort e.g. a bigint id column lexicographically, not numerically).
/// A `NULL` vector is skipped (nothing to push, but the cursor still
/// advances past it); a `NULL` metadata value is omitted from that row's
/// metadata object (Pinecone rejects null metadata values).
///
/// Returns the id (as text) of the last row this call examined — the
/// caller's next `after_id` — or `None` if the table has no row at all past
/// `after_id` (the backfill is done). Shared by [`backfill`] (looped to
/// exhaustion, above) and, one call per drain round so its own transaction
/// can commit the persisted `pinecone.outbox.payload.after_id` alongside it,
/// `outbox`'s `'backfill'` job dispatch.
pub(crate) fn backfill_chunk(
    tbl: &PgRelation,
    row: &RegistryRow,
    client: &Client,
    index_name: &str,
    namespace: &str,
    after_id: Option<&str>,
) -> Option<String> {
    try_backfill_chunk(tbl, row, client, index_name, namespace, after_id)
        .unwrap_or_else(|e| backend::raise(e))
}

/// Core of [`backfill_chunk`]; see `sync_trigger::try_push_rows`'s rustdoc
/// for why this returns a `Result` (and takes `client` as a parameter)
/// instead of raising directly — `outbox::drain_step` needs the client error
/// back, to record it on the `'backfill'` job row before deciding the
/// batch's outcome.
pub(crate) fn try_backfill_chunk(
    tbl: &PgRelation,
    row: &RegistryRow,
    client: &Client,
    index_name: &str,
    namespace: &str,
    after_id: Option<&str>,
) -> Result<Option<String>, ClientError> {
    let ChunkSource {
        id_col,
        vcol,
        table,
        id_type,
        metadata_kinds,
    } = chunk_source(tbl, row)?;
    let mut select_cols = vec![
        format!("{id_col}::text AS id"),
        format!("{vcol}::real[] AS vector"),
    ];
    for (name, kind) in &metadata_kinds {
        let ident = pgrx::spi::quote_identifier(name);
        select_cols.push(format!("{ident}::{} AS {ident}", metadata_cast(kind)));
    }
    // `$1::text IS NULL` (rather than a plain `$1 IS NULL`) short-circuits
    // the very first chunk (`after_id = None`) without needing a second,
    // differently-typed comparison branch — `id_col > $1::text::{id_type}`
    // only ever runs once `$1` is non-NULL, at which point casting it to the
    // id column's own native type gives the native-type comparison
    // `backfill_chunk`'s rustdoc requires.
    //
    // `ORDER BY {table}.{id_col}` is table-qualified on purpose: the select
    // list aliases `{id_col}::text AS id`, and Postgres resolves a *bare*
    // `ORDER BY` identifier against output-column aliases first — so for the
    // very common case where the id column is literally named `id`, a bare
    // `ORDER BY {id_col}` silently sorted by the TEXT cast while the cursor
    // comparison above stayed native-typed. Mixed semantics like that can
    // *skip rows entirely* (e.g. ids {2, 19}: text order puts "19" first, so
    // a 1-row chunk persists after_id = 19 and the native `> 19` filter never
    // sees 2). Qualifying the reference forces the real table column — caught
    // by `multi_chunk_backfill_advances_cursor_and_flips_once_at_exhaustion`.
    let sql = format!(
        "SELECT {} FROM {table} WHERE $1::text IS NULL OR {table}.{id_col} > $1::text::{id_type} \
         ORDER BY {table}.{id_col} LIMIT $2",
        select_cols.join(", ")
    );

    Spi::connect(|spi_client| {
        let rows = spi_client
            .select(
                &sql,
                None,
                &[
                    after_id.map(str::to_string).into(),
                    (chunk_rows(row.dimension) as i64).into(),
                ],
            )
            .unwrap_or_else(|e| error!("pg_pinecone: backfill chunk read failed: {e}"));

        let mut batch: Vec<Vector> = Vec::new();
        let mut batch_bytes: usize = 0;
        let mut last_id: Option<String> = None;
        let mut any_row = false;

        for spi_row in rows {
            any_row = true;
            let id = get_opt::<String>(&spi_row, "id").unwrap_or_else(|| {
                panic!("pg_pinecone: backfill id unexpectedly NULL — id_column is a primary key")
            });
            // A `Result`, not `backend::require_id_len`'s raised `error!`,
            // for the same worker-crash-loop reason as the column-existence
            // checks above — a permanent-class failure
            // here parks the relid instead.
            if id.len() > backend::MAX_ID_LEN {
                return Err(backend::id_len_error(&id));
            }

            let Some(raw_values) = get_opt::<Vec<Option<f32>>>(&spi_row, "vector") else {
                last_id = Some(id); // NULL vector: skip, but the cursor still advances past it.
                continue;
            };
            let values: Vec<f32> = raw_values
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    v.unwrap_or_else(|| {
                        panic!("pg_pinecone: backfill vector[{}] unexpectedly NULL", i + 1)
                    })
                })
                .collect();

            let metadata = read_metadata(&spi_row, &metadata_kinds);
            let vector = Vector {
                id: id.clone(),
                values,
                metadata,
            };
            let size = vector.record_bytes();

            if !batch.is_empty() && batch_bytes + size > UPSERT_MAX_BYTES {
                // This row (and anything left in this LIMIT page) is
                // reconsidered on the *next* call — `last_id` (not yet
                // advanced past it) is what the caller persists.
                break;
            }
            batch_bytes += size;
            batch.push(vector);
            last_id = Some(id);
        }

        if !any_row {
            return Ok(None);
        }
        if !batch.is_empty() {
            client.upsert_vectors(&UpsertVectorsRequest {
                index: index_name.to_string(),
                namespace: Some(namespace.to_string()),
                vectors: batch,
            })?;
        }
        Ok(last_id)
    })
}

/// What both PK-ordered chunk readers need from the registry row, checked against the live table.
struct ChunkSource {
    id_col: String,
    vcol: String,
    table: String,
    id_type: String,
    metadata_kinds: Vec<(String, ColKind)>,
}

fn chunk_source(tbl: &PgRelation, row: &RegistryRow) -> Result<ChunkSource, ClientError> {
    let relid = tbl.oid();
    let id_col = pgrx::spi::quote_identifier(&row.id_column);
    let vcol = pgrx::spi::quote_identifier(&row.vector_column);
    let table = quote_qualified_identifier(tbl.namespace(), tbl.name());

    // Every failure below is returned, never raised directly (`error!`):
    // this runs both inline (`backfill_chunk` re-raises via `backend::raise`,
    // fine for the writer's own session) and from the background worker's
    // own transaction (`outbox::dispatch_backfill`), where an uncaught
    // `error!` would abort the worker process outright — a permanent,
    // dropped/mismatched-column condition would then crash-loop it every
    // round instead of parking the relid's queue with a clear reason.
    let id_type =
        catalog::column_type(relid, &row.id_column).ok_or_else(|| ClientError::BadRequest {
            message: format!(
                "id column \"{}\" is registered for {table} but no longer exists — re-track or \
             re-promote with a corrected id_column",
                row.id_column
            ),
        })?;
    // `id_type` is spliced UNQUOTED into the chunk-read SQL text
    // below (`{id_col} > $1::text::{id_type}`), since a cast target must be a
    // bare type name, not a quotable literal/identifier. `pinecone.track`/
    // `pinecone.promote` only validate `id_column`'s type against
    // `catalog::ID_COLUMN_TYPES` at registration time — a later `ALTER TABLE
    // ... ALTER COLUMN <id_column> TYPE <hostile-named domain>` changes
    // `pg_type.typname` (this query's `id_type`) to whatever SQL an attacker
    // named that domain, which would otherwise execute as this process
    // (`pg_pinecone`'s background worker, running with its own stored
    // credentials — not the writer session that ran the `ALTER TABLE`).
    // Re-checking the same whitelist here, immediately before it would ever
    // be interpolated, closes that off: a mismatch is a permanent-class
    // failure (parks the relid's queue with a clear reason), and the chunk
    // SQL is never built, let alone executed.
    if !catalog::ID_COLUMN_TYPES.contains(&id_type.as_str()) {
        return Err(ClientError::BadRequest {
            message: format!(
                "id column \"{}\" of {table} has type \"{id_type}\", which is not one of \
                 pg_pinecone's supported id column types ({}) — its type changed since this table \
                 was tracked/promoted; restore a supported type or re-track with a corrected \
                 id_column",
                row.id_column,
                catalog::ID_COLUMN_TYPES.join(", ")
            ),
        });
    }

    catalog::column_type(relid, &row.vector_column).ok_or_else(|| ClientError::BadRequest {
        message: format!(
            "vector column \"{}\" is registered for {table} but no longer exists — re-track or \
             re-promote with a corrected vector_column",
            row.vector_column
        ),
    })?;

    let mut metadata_kinds: Vec<(String, ColKind)> = Vec::new();
    for name in &row.metadata_columns {
        let type_name =
            catalog::column_type(relid, name).ok_or_else(|| ClientError::BadRequest {
                message: format!(
                    "metadata column \"{name}\" is registered for {table} but no longer exists — \
                 re-track or re-promote with corrected metadata_columns"
                ),
            })?;
        metadata_kinds.push((name.clone(), catalog::col_kind(&type_name)));
    }
    Ok(ChunkSource {
        id_col,
        vcol,
        table,
        id_type,
        metadata_kinds,
    })
}

/// Namespaces one backfill chunk may write: bounds the requests, hence the time, of a drain round.
pub(crate) const PARTITIONS_PER_CHUNK: usize = 100;

/// One request's vectors of a partitioned chunk: consecutive rows of one partition value.
struct PartitionRun {
    value: String,
    vectors: Vec<Vector>,
    bytes: usize,
}

/// [`try_backfill_chunk`] for a partitioned table: rows after the cursor `after = (partition, id)`
/// in `(partition, id)` order, sent one namespace per request. Rows of a partition the gate skips
/// are passed over (all but one partition, for a job scoped to it), and a refusal of one
/// namespace goes to the gate. Returns the cursor of the last row examined, `None` once the table
/// is exhausted.
pub(crate) fn try_backfill_chunk_partitioned(
    tbl: &PgRelation,
    row: &RegistryRow,
    client: &Client,
    index_name: &str,
    prefix: &str,
    after: Option<(&str, &str)>,
    gate: &mut Gate,
) -> Result<Option<(String, String)>, ClientError> {
    let relid = tbl.oid();
    let spec = row
        .partition
        .as_ref()
        .expect("pg_pinecone: a partitioned backfill needs the registry's partition column");
    let source = chunk_source(tbl, row)?;
    let ChunkSource {
        id_col,
        vcol,
        table,
        id_type,
        metadata_kinds,
    } = source;
    let partition_type = catalog::column_type(relid, &spec.column)
        .filter(|typname| PartitionKind::from_typname(typname) == Some(spec.kind))
        .ok_or_else(|| ClientError::BadRequest {
            message: format!(
                "partition column \"{}\" is registered for {table} but no longer exists or has \
                 another type — re-promote with a corrected partition_column",
                spec.column
            ),
        })?;
    let pcol = pgrx::spi::quote_identifier(&spec.column);

    let mut select_cols = vec![
        format!("{id_col}::text AS id"),
        format!("{vcol}::real[] AS vector"),
        format!("{table}.{pcol}::text AS pg_pinecone_partition_value"),
    ];
    for (name, kind) in &metadata_kinds {
        let ident = pgrx::spi::quote_identifier(name);
        select_cols.push(format!("{ident}::{} AS {ident}", metadata_cast(kind)));
    }
    let sql = format!(
        "SELECT {} FROM {table} WHERE ($1::text IS NULL OR ({table}.{pcol}, {table}.{id_col}) > \
         ($1::text::{partition_type}, $2::text::{id_type})) \
         AND ($4::text IS NULL OR {table}.{pcol} = $4::text::{partition_type}) \
         ORDER BY {table}.{pcol}, {table}.{id_col} LIMIT $3",
        select_cols.join(", ")
    );
    let only = gate.scoped_to().map(str::to_string);

    Spi::connect(|spi_client| {
        let rows = spi_client
            .select(
                &sql,
                None,
                &[
                    after.map(|(p, _)| p.to_string()).into(),
                    after.map(|(_, id)| id.to_string()).into(),
                    (BACKFILL_MAX_ROWS as i64).into(),
                    only.into(),
                ],
            )
            .unwrap_or_else(|e| error!("pg_pinecone: backfill chunk read failed: {e}"));

        let mut runs: Vec<PartitionRun> = Vec::new();
        let mut last: Option<(String, String)> = None;
        let mut any_row = false;

        for spi_row in rows {
            any_row = true;
            let id = get_opt::<String>(&spi_row, "id").unwrap_or_else(|| {
                panic!("pg_pinecone: backfill id unexpectedly NULL — id_column is a primary key")
            });
            if id.len() > backend::MAX_ID_LEN {
                return Err(backend::id_len_error(&id));
            }
            let value =
                get_opt::<String>(&spi_row, "pg_pinecone_partition_value").ok_or_else(|| {
                    ClientError::BadRequest {
                        message: format!(
                            "partition column \"{}\" of {table} is NULL for id {id} — a NULL has \
                             no namespace",
                            spec.column
                        ),
                    }
                })?;

            if gate.skips(relid, &value) {
                last = Some((value, id));
                continue;
            }
            let Some(raw_values) = get_opt::<Vec<Option<f32>>>(&spi_row, "vector") else {
                last = Some((value, id));
                continue;
            };
            let values: Vec<f32> = raw_values
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    v.unwrap_or_else(|| {
                        panic!("pg_pinecone: backfill vector[{}] unexpectedly NULL", i + 1)
                    })
                })
                .collect();
            let vector = Vector {
                id: id.clone(),
                values,
                metadata: partition_metadata(
                    read_metadata(&spi_row, &metadata_kinds),
                    spec,
                    &value,
                ),
            };
            let size = vector.record_bytes();

            let continues = runs.last().is_some_and(|run| run.value == value);
            if !continues && runs.len() >= PARTITIONS_PER_CHUNK {
                break;
            }
            if continues
                && runs
                    .last()
                    .is_some_and(|r| r.bytes + size > UPSERT_MAX_BYTES)
            {
                break;
            }
            if !continues {
                runs.push(PartitionRun {
                    value: value.clone(),
                    vectors: Vec::new(),
                    bytes: 0,
                });
            }
            let run = runs.last_mut().expect("a run was just pushed or continues");
            run.bytes += size;
            run.vectors.push(vector);
            last = Some((value, id));
        }

        if !any_row {
            return Ok(None);
        }
        for run in runs {
            let namespace = partition::namespace_for(prefix, &run.value);
            partition::fits_pinecone(&namespace).map_err(|e| ClientError::BadRequest {
                message: format!("pg_pinecone: the partition value makes a bad namespace: {e}"),
            })?;
            let result = client
                .upsert_vectors(&UpsertVectorsRequest {
                    index: index_name.to_string(),
                    namespace: Some(namespace.clone()),
                    vectors: run.vectors,
                })
                .map(|_| ());
            let wrote = result.is_ok();
            partition_lane::settle_write(relid, &run.value, &namespace, wrote, result, gate)?;
        }
        Ok(last)
    })
}

/// `metadata` with the partition column added as Pinecone metadata (a number for the integer
/// kinds, else a string), as the trigger writes it.
fn partition_metadata(metadata: Option<Value>, spec: &PartitionSpec, value: &str) -> Option<Value> {
    let mut map = match metadata {
        Some(Value::Object(map)) => map,
        _ => serde_json::Map::new(),
    };
    let json_value = match spec.kind {
        PartitionKind::Int2 | PartitionKind::Int4 | PartitionKind::Int8 => value
            .parse::<i64>()
            .map_or_else(|_| json!(value), |number| json!(number)),
        PartitionKind::Text | PartitionKind::Uuid => json!(value),
    };
    map.insert(spec.column.clone(), json_value);
    Some(Value::Object(map))
}

/// Number of heap blocks [`try_tid_backfill_chunk`] scans per chunk —
/// bounds one drain round's worst-case page-lock/I/O latency regardless of
/// how sparse (mostly-dead, or all-visible-but-empty) the scanned range
/// turns out to be, the same role [`BACKFILL_MAX_ROWS`]/[`UPSERT_MAX_BYTES`]
/// play for the PK-keyed path (which this can't reuse directly: a block may
/// hold zero live rows, so bounding only by row/byte count could otherwise
/// force one chunk to scan the entire relation).
const TID_BACKFILL_BLOCKS_PER_CHUNK: pg_sys::BlockNumber = 2048;

/// A partitioned index's chunk reads this many rows (or [`TID_WINDOW_BYTES`] of vectors) before it
/// ends at a block boundary, then sends one request per namespace: a heap that alternates tenants
/// row by row still fills requests. Requests per chunk are bounded by the rows, not by the
/// namespaces.
pub(crate) const TID_WINDOW_ROWS: usize = 5_000;
const TID_WINDOW_BYTES: usize = 64 * 1024 * 1024;

/// TID-semantics counterpart to [`try_backfill_chunk`] — dispatched by
/// `outbox::dispatch_backfill` for `payload.id_kind = "tid"` jobs, i.e. every
/// backfill `am::ambuild` enqueues: a pg_pinecone access-method index's remote
/// ids are heap TIDs (`"block:offset"`, `am::tid_to_id`'s own format), never
/// the table's primary key. The only metadata is the index's filter key
/// (`row.filter_column`, as `am::aminsert` syncs it): `{col: v}`, none for a NULL key.
///
/// **BLOCK-RANGE chunking, not `ORDER BY ctid`** (this used to read the
/// live table `ORDER BY ctid`, persisting the last row's own `ctid` as the
/// cursor). Two bugs that design
/// had: (1) it pushed the TID of whatever tuple currently sits at each heap
/// position — for a HOT-updated row that's the CURRENT chain member, not
/// the chain's ROOT, i.e. not the id any index (or this AM's own remote
/// store) can keep referencing once HOT pruning collapses the chain: a
/// pruned root becomes an `LP_REDIRECT` line pointer, stable for the row's
/// life, while a non-root member's line pointer is free to be reclaimed and
/// reused by an unrelated future tuple — pushing the latter risks a phantom
/// remote id that silently outlives (or collides with) the row it once
/// named; (2) a HOT update can relocate a row to a LOWER offset on the SAME
/// page (reusing a just-pruned, lower-numbered line pointer) — once the
/// per-row `ctid` cursor had advanced past that offset, the relocated row
/// was permanently skipped, with no later chunk ever revisiting it.
///
/// Both are fixed by making the cursor a **block number**, not a `ctid`:
/// blocks never move or get reused by ordinary (non-`VACUUM FULL`) heap
/// activity — lazy `VACUUM` only ever truncates trailing all-empty blocks,
/// which can't happen below this call's own `nblocks` snapshot while it
/// holds (at least) `AccessShareLock` — and, for each block, this walks the
/// WHOLE page in one buffer-share-locked pass ([`scan_block_for_live_roots`]),
/// so there is no intra-page cursor at all for a same-page HOT relocation to
/// slip under. [`pg_sys::heap_get_root_tuples`] — the exact function
/// `heapam_index_build_range_scan` uses to attribute HOT-chain values to
/// their root line pointer when building a real index — resolves every live
/// tuple's id to its chain's ROOT offset, the same offset `am::aminsert`
/// itself always used to push that logical row (see the CORRECTNESS
/// ARGUMENT below), so the two paths converge on identical ids for the same
/// logical row.
///
/// A partitioned index's chunk is a window ([`TID_WINDOW_ROWS`] rows or [`TID_WINDOW_BYTES`] of
/// vectors, ended at a block boundary) that `push_partitioned` groups per namespace, so a heap
/// that alternates tenants row by row still sends one request per tenant, not one per row.
///
/// `after_block`/the returned cursor are a plain decimal block-number string
/// (not the previous `ctid` text) — an internal `payload.after_id` shape
/// change scoped to `id_kind = "tid"` jobs only; `dispatch_backfill`'s
/// contract (`Some(cursor)` = more chunks may remain, `None` = table
/// exhausted) is unchanged, and is now cheaper to decide: reaching the end
/// needs no extra "did that read return zero rows" round trip, since
/// `start_block >= nblocks` answers it without touching a single page.
///
/// ## CORRECTNESS ARGUMENT
/// (a) **Roots are what `aminsert` also emits.** `am::aminsert` runs once
///     per newly-inserted tuple version (a plain `INSERT`, or the new-tuple
///     half of a non-HOT `UPDATE`), and Postgres hands it that tuple's OWN
///     `(block, offset)` as `heap_tid` — which, at the moment of insertion,
///     is *always* the root of its own (as yet single-member) chain, since
///     nothing else references it yet. `am::tid_to_id` of that same pair is
///     exactly what this scan reconstructs later via `heap_get_root_tuples`,
///     for as long as the row lives: a HOT update only ever *extends* a
///     chain from that same root, never changes which offset the root is.
/// (b) **Post-cursor inserts converge.** By the time this backfill job is
///     even enqueued, `CREATE INDEX ... USING pinecone` has already
///     committed — `aminsert` is live for every subsequent write, on every
///     block, including ones this scan's cursor has already passed
///     (Postgres can reuse free space in an earlier block, via the FSM, for
///     a brand-new row). Such a row is entirely `aminsert`'s problem, not
///     this scan's: it didn't exist at `ambuild` time, so this backfill
///     owning it too would be redundant, not required.
/// (c) **HOT updates can't change the backfilled value.** The vector column
///     (and the filter key, if any) is, by definition, indexed (this is the
///     AM's own index), and
///     Postgres never classifies an `UPDATE` that changes an indexed
///     column's value as HOT (`RelationGetIndexAttrBitmap`'s check,
///     refreshed by the relcache invalidation `CREATE INDEX` itself sends).
///     So any write that changes the vector or the key fires `aminsert` for
///     a fresh root (case (a)) instead of extending the old chain — meaning
///     a chain this scan reads (possibly concurrently with a write to it,
///     since this only holds `AccessShareLock`) can only ever be
///     HOT-extended by updates to *other* columns, which leave the vector
///     and key values this scan reads identical to what a fresh `aminsert`
///     would have pushed for the same logical row.
pub(crate) fn try_tid_backfill_chunk(
    tbl: &PgRelation,
    row: &RegistryRow,
    client: &Client,
    index_name: &str,
    namespace: &str,
    after_block: Option<&str>,
    gate: &mut Gate,
) -> Result<Option<String>, ClientError> {
    let relid = tbl.oid();
    let table = quote_qualified_identifier(tbl.namespace(), tbl.name());

    let vector_attnum = catalog::all_columns(relid)
        .into_iter()
        .find(|c| c.name == row.vector_column)
        .map(|c| c.attnum)
        .ok_or_else(|| ClientError::BadRequest {
            message: format!(
                "vector column \"{}\" is registered for {table} but no longer exists — re-track \
                 or re-promote with a corrected vector_column",
                row.vector_column
            ),
        })?;

    let filter = match row.filter_column.as_deref() {
        None => None,
        Some(name) => Some(backfill_filter(relid, name, &table, gate.scoped_to())?),
    };

    let start_block: pg_sys::BlockNumber = match after_block {
        None => 0,
        Some(s) => {
            let last: pg_sys::BlockNumber = s.parse().map_err(|_| ClientError::BadRequest {
                message: format!(
                    "'backfill' row for {table} has a corrupt after_id block cursor '{s}'"
                ),
            })?;
            last.checked_add(1).ok_or_else(|| ClientError::BadRequest {
                message: format!("pg_pinecone: TID backfill block cursor overflowed for {table}"),
            })?
        }
    };

    let heap = tbl.as_ptr();
    // # Safety: `tbl` is a live, at-least-`AccessShareLock`-held relation for
    // the duration of this call (its `PgRelation` guard outlives it).
    let nblocks =
        unsafe { pg_sys::RelationGetNumberOfBlocksInFork(heap, pg_sys::ForkNumber::MAIN_FORKNUM) };
    if start_block >= nblocks {
        return Ok(None);
    }
    let end_block = (start_block + TID_BACKFILL_BLOCKS_PER_CHUNK).min(nblocks);
    let tupdesc = unsafe { (*heap).rd_att };
    let snapshot = RegisteredSnapshot::of_transaction();

    let partition = row.partition.as_ref();
    let (max_rows, max_bytes) = match partition {
        Some(_) => (TID_WINDOW_ROWS, TID_WINDOW_BYTES),
        None => (BACKFILL_MAX_ROWS, UPSERT_MAX_BYTES),
    };
    let mut batch: Vec<Vector> = Vec::new();
    let mut batch_bytes: usize = 0;
    let mut last_block = start_block;
    for block in start_block..end_block {
        let block_rows = scan_block_for_live_roots(
            heap,
            relid,
            tupdesc,
            block,
            vector_attnum,
            filter.as_ref(),
            snapshot.0,
        )?;
        let block_bytes: usize = block_rows.iter().map(Vector::record_bytes).sum();
        // An unpartitioned chunk is one request: it ends before the block that would overflow it,
        // which opens the next chunk, rather than spilling into a second, small request. A
        // partitioned window is split per namespace anyway.
        let overflows =
            batch.len() + block_rows.len() > max_rows || batch_bytes + block_bytes > max_bytes;
        if partition.is_none() && !batch.is_empty() && overflows {
            break;
        }
        batch_bytes += block_bytes;
        batch.extend(block_rows);
        last_block = block;
        if batch.len() >= max_rows || batch_bytes >= max_bytes {
            break;
        }
    }

    if !batch.is_empty() {
        match partition {
            None => push_in_bounded_batches(client, index_name, namespace, batch)?,
            Some(spec) => {
                push_partitioned(relid, client, index_name, namespace, spec, batch, gate)?
            }
        }
    }
    Ok(Some(last_block.to_string()))
}

/// The transaction snapshot, registered until drop: PG18 asserts that visibility checks use a
/// registered or active snapshot.
struct RegisteredSnapshot(pg_sys::Snapshot);

impl RegisteredSnapshot {
    fn of_transaction() -> Self {
        Self(unsafe { pg_sys::RegisterSnapshot(pg_sys::GetTransactionSnapshot()) })
    }
}

impl Drop for RegisteredSnapshot {
    fn drop(&mut self) {
        unsafe { pg_sys::UnregisterSnapshot(self.0) }
    }
}

/// One namespace at a time, each registered once it holds a vector; the gate leaves out the
/// partitions it skips and takes the refusal of one namespace.
fn push_partitioned(
    relid: pg_sys::Oid,
    client: &Client,
    index_name: &str,
    prefix: &str,
    spec: &PartitionSpec,
    batch: Vec<Vector>,
    gate: &mut Gate,
) -> Result<(), ClientError> {
    for group in am_partition::group_vectors(prefix, &spec.column, batch)? {
        if gate.skips(relid, &group.value) {
            continue;
        }
        let result = push_in_bounded_batches(client, index_name, &group.namespace, group.vectors);
        let wrote = result.is_ok();
        let (value, namespace) = (&group.value, &group.namespace);
        partition_lane::settle_write(relid, value, namespace, wrote, result, gate)?;
    }
    Ok(())
}

/// The filter key of a TID backfill: which heap column to read into each vector's metadata, and
/// for a job scoped to one partition the value a row's key must have to be read at all.
struct BackfillFilter {
    name: String,
    attnum: i32,
    kind: FilterKind,
    only: Option<String>,
}

/// Resolves the registered filter column `name` of `relid` to its heap attribute and kind; a
/// column that was renamed or retyped since the build parks the job with the remedy.
fn backfill_filter(
    relid: pg_sys::Oid,
    name: &str,
    table: &str,
    only: Option<&str>,
) -> Result<BackfillFilter, ClientError> {
    let column = catalog::all_columns(relid)
        .into_iter()
        .find(|c| c.name == name)
        .ok_or_else(|| ClientError::BadRequest {
            message: format!(
                "filter column \"{name}\" is registered for {table} but no longer exists — \
                 REINDEX the index to follow a rename"
            ),
        })?;
    let kind = FilterKind::from_typname(&column.type_name).ok_or_else(|| ClientError::BadRequest {
        message: format!(
            "filter column \"{name}\" of {table} has type {} — a pg_pinecone filter column must be \
             smallint, integer, bigint, text, or varchar",
            column.type_name
        ),
    })?;
    Ok(BackfillFilter {
        name: name.to_string(),
        attnum: i32::from(column.attnum),
        kind,
        only: only.map(str::to_string),
    })
}

/// Sends `batch` as one or more `upsert_vectors` calls, each within the request limits
/// ([`UPSERT_MAX_RECORDS`], [`UPSERT_MAX_BYTES`]). A TID chunk must not split a block, so a block
/// alone may overshoot them; a failure partway leaves the earlier requests sent, and the chunk's
/// unpersisted cursor replays them (an upsert of the same `(id, values)` is idempotent).
fn push_in_bounded_batches(
    client: &Client,
    index_name: &str,
    namespace: &str,
    batch: Vec<Vector>,
) -> Result<(), ClientError> {
    let mut current: Vec<Vector> = Vec::new();
    let mut current_bytes = 0usize;
    for vector in batch {
        let size = vector.record_bytes();
        if !current.is_empty()
            && (current.len() >= UPSERT_MAX_RECORDS || current_bytes + size > UPSERT_MAX_BYTES)
        {
            client.upsert_vectors(&UpsertVectorsRequest {
                index: index_name.to_string(),
                namespace: Some(namespace.to_string()),
                vectors: std::mem::take(&mut current),
            })?;
            current_bytes = 0;
        }
        current_bytes += size;
        current.push(vector);
    }
    if !current.is_empty() {
        client.upsert_vectors(&UpsertVectorsRequest {
            index: index_name.to_string(),
            namespace: Some(namespace.to_string()),
            vectors: current,
        })?;
    }
    Ok(())
}

/// Opens, share-locks, scans, and releases heap block `block` of `heap` —
/// see [`try_tid_backfill_chunk`]'s rustdoc for why a whole-page pass under
/// one buffer lock (rather than a per-row cursor) is what makes this safe
/// against a concurrent same-page HOT update. Returns one `Vector` (id =
/// the tuple's HOT-chain ROOT tid, per [`pg_sys::heap_get_root_tuples`]) for
/// every tuple on the page currently visible to `snapshot`.
fn scan_block_for_live_roots(
    heap: pg_sys::Relation,
    relid: pg_sys::Oid,
    tupdesc: pg_sys::TupleDesc,
    block: pg_sys::BlockNumber,
    vector_attnum: i16,
    filter: Option<&BackfillFilter>,
    snapshot: pg_sys::Snapshot,
) -> Result<Vec<Vector>, ClientError> {
    // A closure (rather than an early-return-friendly loop inlined here)
    // guarantees `UnlockReleaseBuffer` always runs before this function
    // returns, on every `?`/`Err` exit alike, without needing a bespoke RAII
    // guard type for one call site — mirrors `Spi::connect`'s own
    // closure-scoped-cleanup shape used throughout this module.
    unsafe {
        let buffer = pg_sys::ReadBuffer(heap, block);
        pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_SHARE as i32);
        let result = scan_locked_page(
            buffer,
            relid,
            block,
            tupdesc,
            vector_attnum,
            filter,
            snapshot,
        );
        pg_sys::UnlockReleaseBuffer(buffer);
        result
    }
}

/// # Safety
/// `buffer` must be pinned and content-locked (at least `BUFFER_LOCK_SHARE`)
/// for `block` of the relation `tupdesc` belongs to, for this whole call.
unsafe fn scan_locked_page(
    buffer: pg_sys::Buffer,
    relid: pg_sys::Oid,
    block: pg_sys::BlockNumber,
    tupdesc: pg_sys::TupleDesc,
    vector_attnum: i16,
    filter: Option<&BackfillFilter>,
    snapshot: pg_sys::Snapshot,
) -> Result<Vec<Vector>, ClientError> {
    unsafe {
        let page = pg_sys::BufferGetPage(buffer);
        let maxoff = pg_sys::PageGetMaxOffsetNumber(page);
        // `heap_get_root_tuples` unconditionally `MemSet`s exactly
        // `MaxHeapTuplesPerPage` entries — this buffer must be at least that
        // large regardless of `maxoff` (a page can never legally hold more
        // line pointers than that bound) or the C call below overruns it.
        let mut root_offsets = vec![pg_sys::InvalidOffsetNumber; max_heap_tuples_per_page()];
        pg_sys::heap_get_root_tuples(page, root_offsets.as_mut_ptr());

        let mut out = Vec::new();
        let mut offnum = pg_sys::FirstOffsetNumber;
        while offnum <= maxoff {
            let lp = pg_sys::PageGetItemId(page, offnum);
            if (*lp).lp_flags() != pg_sys::LP_NORMAL {
                offnum += 1;
                continue; // unused, dead, or a redirect — no tuple data to read.
            }
            let mut tid = pg_sys::ItemPointerData::default();
            pgrx::itemptr::item_pointer_set_all(&mut tid, block, offnum);
            let mut tuple = pg_sys::HeapTupleData {
                t_len: (*lp).lp_len(),
                t_self: tid,
                t_tableOid: relid,
                t_data: pg_sys::PageGetItem(page, lp) as pg_sys::HeapTupleHeader,
            };
            if !pg_sys::HeapTupleSatisfiesVisibility(&mut tuple, snapshot, buffer) {
                offnum += 1;
                continue;
            }

            let root_offnum = root_offsets[(offnum - 1) as usize];
            if root_offnum == pg_sys::InvalidOffsetNumber {
                // Should be unreachable: `heap_get_root_tuples` gives every
                // live, non-heap-only-orphaned normal tuple a root entry —
                // returned (not panicked) so a hypothetical corrupt page
                // parks this relid's queue with a clear reason instead of
                // crash-looping the worker.
                return Err(ClientError::BadRequest {
                    message: format!(
                        "pg_pinecone: TID backfill found a live tuple at ({block},{offnum}) with no \
                         discoverable HOT-chain root — should be unreachable"
                    ),
                });
            }

            let mut is_null = false;
            let datum =
                pg_sys::fastgetattr(&mut tuple, vector_attnum as i32, tupdesc, &mut is_null);
            if is_null {
                offnum += 1;
                continue; // NULL vector: nothing to push for this row.
            }
            let metadata = filter.and_then(|filter| {
                let mut key_is_null = false;
                let key = pg_sys::fastgetattr(&mut tuple, filter.attnum, tupdesc, &mut key_is_null);
                // A NULL key carries no metadata: it never matches `$eq`.
                (!key_is_null)
                    .then(|| am_filter::metadata(&filter.name, filter.kind.datum_to_json(key)))
            });
            if let Some(filter) = filter
                && let Some(only) = &filter.only
                && am_partition::key_canon(metadata.as_ref(), &filter.name).as_deref()
                    != Some(only.as_str())
            {
                offnum += 1;
                continue;
            }
            let mut root_tid = pg_sys::ItemPointerData::default();
            pgrx::itemptr::item_pointer_set_all(&mut root_tid, block, root_offnum);
            out.push(Vector {
                id: crate::am::tid_to_id(root_tid),
                values: crate::am::parse_vector(datum),
                metadata,
            });
            offnum += 1;
        }
        Ok(out)
    }
}

/// `MaxHeapTuplesPerPage` — the maximum number of line pointers any heap
/// page can hold, computed the same way the C macro does
/// (`(BLCKSZ - SizeOfPageHeaderData) / (MAXALIGN(SizeofHeapTupleHeader) +
/// sizeof(ItemIdData))`) but from the bindgen'd struct layouts themselves
/// (`offset_of!`) rather than hand-derived constants, so it stays correct
/// across platforms/Postgres versions without needing to track the C
/// header's own definition by hand. [`scan_locked_page`]'s
/// `heap_get_root_tuples` buffer must be at least this large.
fn max_heap_tuples_per_page() -> usize {
    let page_header_size = std::mem::offset_of!(pg_sys::PageHeaderData, pd_linp);
    let tuple_header_size = std::mem::offset_of!(pg_sys::HeapTupleHeaderData, t_bits);
    let item_id_size = std::mem::size_of::<pg_sys::ItemIdData>();
    let align = pg_sys::MAXIMUM_ALIGNOF as usize;
    let aligned_tuple_header = tuple_header_size.div_ceil(align) * align;
    (pg_sys::BLCKSZ as usize - page_header_size) / (aligned_tuple_header + item_id_size)
}

/// The SQL cast applied to a metadata column of this [`ColKind`] so it can be
/// read back with a single, kind-specific Rust type regardless of the
/// column's exact underlying width (e.g. `int2` vs `int8`).
fn metadata_cast(kind: &ColKind) -> &'static str {
    match kind {
        ColKind::Text => "text",
        ColKind::Int => "bigint",
        ColKind::Float | ColKind::Numeric => "double precision",
        ColKind::Bool => "boolean",
        ColKind::TextArray => "text[]",
        // Unreachable: track()/promote() validate metadata_columns via
        // catalog::col_kind before they ever reach the registry.
        ColKind::Unsupported(t) => {
            panic!("pg_pinecone: backfill metadata column has unsupported type {t}")
        }
    }
}

fn read_metadata(
    row: &pgrx::spi::SpiHeapTupleData<'_>,
    kinds: &[(String, ColKind)],
) -> Option<Value> {
    let mut map = serde_json::Map::new();
    for (name, kind) in kinds {
        let value = match kind {
            ColKind::Text => get_opt::<String>(row, name).map(Value::String),
            ColKind::Int => get_opt::<i64>(row, name).map(|v| json!(v)),
            ColKind::Float | ColKind::Numeric => get_opt::<f64>(row, name).map(|v| json!(v)),
            ColKind::Bool => get_opt::<bool>(row, name).map(Value::Bool),
            ColKind::TextArray => get_opt::<Vec<Option<String>>>(row, name).map(|elems| {
                Value::Array(elems.into_iter().flatten().map(Value::String).collect())
            }),
            ColKind::Unsupported(_) => None, // unreachable, see metadata_cast
        };
        if let Some(v) = value {
            map.insert(name.clone(), v);
        }
    }
    if map.is_empty() {
        None
    } else {
        Some(Value::Object(map))
    }
}

/// Reads a nullable column by name, panicking (rather than silently treating
/// it as absent) if the query this module built itself no longer matches —
/// i.e. a bug here, not a user-facing error.
fn get_opt<T: FromDatum + IntoDatum>(
    row: &pgrx::spi::SpiHeapTupleData<'_>,
    name: &str,
) -> Option<T> {
    row.get_by_name::<T, _>(name)
        .unwrap_or_else(|e| panic!("pg_pinecone: backfill read \"{name}\" failed: {e}"))
}

// Pure-Rust (no pgrx/pg dependency), unlike this crate's `#[pg_test]`s — runs
// under plain `cargo test`/`cargo pgrx test` alike.
#[cfg(test)]
mod tests {
    use super::*;

    fn spec(kind: PartitionKind) -> PartitionSpec {
        PartitionSpec {
            column: "tenant_id".to_string(),
            kind,
        }
    }

    #[test]
    fn a_backfilled_vector_carries_the_partition_value_in_its_own_type() {
        let ints = [
            PartitionKind::Int2,
            PartitionKind::Int4,
            PartitionKind::Int8,
        ];
        for kind in ints {
            let metadata = partition_metadata(None, &spec(kind), "42");
            assert_eq!(metadata, Some(json!({"tenant_id": 42})), "{kind:?}");
        }
        for kind in [PartitionKind::Text, PartitionKind::Uuid] {
            let metadata = partition_metadata(None, &spec(kind), "42");
            assert_eq!(metadata, Some(json!({"tenant_id": "42"})), "{kind:?}");
        }
    }

    #[test]
    fn the_partition_value_joins_the_metadata_the_table_declares() {
        let declared = Some(json!({"category": "a"}));

        let metadata = partition_metadata(declared, &spec(PartitionKind::Int8), "7");

        assert_eq!(metadata, Some(json!({"category": "a", "tenant_id": 7})));
    }

    #[test]
    fn wait_ready_timeout_message_names_deadline_and_index_not_timeout_ms_guc() {
        let message = wait_ready_timeout_message("pv-public-docs");
        assert!(message.contains("pv-public-docs"), "message = {message}");
        assert!(message.contains("120s"), "message = {message}");
        assert!(
            !message.contains("pinecone.timeout_ms"),
            "message must not point at the per-request HTTP timeout GUC: {message}"
        );
    }
}
