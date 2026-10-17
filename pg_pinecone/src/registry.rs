//! `pinecone.tables`: the registry of tables `pinecone.track`/`pinecone.promote`
//! have registered, and the CRUD helpers `track`/`untrack`/`topk`/`status`
//! use to read and write it. Declared here rather than inside the `pg_pinecone`
//! `#[pg_schema]` module in `lib.rs` (which can't hold an `extension_sql!` with
//! an explicit `requires`, only implicit schema-membership) — the `requires`
//! below is what pins this table's `CREATE TABLE` after the `pg_pinecone` schema's
//! `CREATE SCHEMA` in the generated `pg_pinecone--<version>.sql`.

use pgrx::{prelude::*, spi::Spi};

use crate::{
    backend, docs_registry,
    partition::{Claim, PartitionSpec},
    partition_registry::{self, Owner},
};

extension_sql!(
    r#"
    CREATE TABLE pinecone.tables (
        relid regclass PRIMARY KEY,
        vector_column text NOT NULL,
        id_column text NOT NULL,
        metric text NOT NULL CHECK (metric IN ('cosine', 'l2', 'ip')),
        dimension int NOT NULL,
        metadata_columns text[] NOT NULL DEFAULT '{}',
        mode text NOT NULL DEFAULT 'local' CHECK (mode IN ('local', 'remote', 'promoting')),
        sync_mode text NOT NULL DEFAULT 'async' CHECK (sync_mode IN ('async', 'sync')),
        -- 'engine': a row created/managed by `pinecone.create_index`/
        -- `set_engine` (`lifecycle.rs`) for an engine with no OTHER sync
        -- mechanism of its own — `'pgvector'` (no sync at all: a plain local
        -- index) or a `'native'` (local-mode AM) index that `create_index`
        -- wraps for later `set_engine` lookups. A `'remote'`-engine row
        -- created by `create_index`/`promote_index` instead reuses
        -- `synced_by = 'am'` (the AM's own sync mechanism already applies) —
        -- see `registry.engine`'s own doc comment for how the two columns
        -- combine.
        synced_by text NOT NULL DEFAULT 'trigger' CHECK (synced_by IN ('trigger', 'am', 'engine')),
        queue_parked boolean NOT NULL DEFAULT false,
        parked_reason text,
        index_name text,
        namespace text,
        -- The pg_pinecone-AM index relation's OWN oid (`am::ambuild`'s
        -- `(*index).rd_id`) — `NULL` for `synced_by = 'trigger'` rows.
        -- `am.rs`'s `pinecone._on_sql_drop`/`pg_pinecone_on_sql_drop` event
        -- trigger matches `pg_event_trigger_dropped_objects().objid` against
        -- this column directly (no `relam`/`pg_class` lookup needed — and
        -- none would be possible: the dropped index's catalog row is
        -- already gone by the time `sql_drop` fires), so a `DROP INDEX` (or
        -- a `DROP TABLE` that drops it implicitly) can purge this row
        -- without depending on anything that no longer exists.
        am_index_relid oid,
        -- The most recent `am::ambulkdelete` (VACUUM) failure's message, if
        -- any — set by `record_vacuum_error` right before the failing HTTP
        -- call's error is raised, so `status()` can fold it into `last_error`
        -- alongside the outbox's own, so an operator can tell a VACUUM
        -- failure from an outage. Best-effort only: a write immediately followed
        -- by a raised Postgres ERROR is rolled back with the rest of that
        -- VACUUM transaction (Postgres has no autonomous-transaction
        -- primitive), so this persists only when the caller of `ambulkdelete`
        -- doesn't itself abort the same transaction afterward — see
        -- `record_vacuum_error`'s own rustdoc.
        last_vacuum_error text,
        -- Lifecycle surface (`pinecone.create_index`/`set_engine`/
        -- `promote_index`/`demote_index`): which PHYSICAL index backs this
        -- column right now. NULLABLE and NULL by default — populated only by
        -- that surface, never by the older `pinecone.track`/`promote`
        -- (synced_by='trigger') or a plain `CREATE INDEX ... USING pinecone`
        -- (synced_by='am') path, both of which predate this column and keep
        -- working unchanged with engine left NULL ("legacy row, not managed
        -- by the engine surface"). See `lifecycle.rs`'s module docs. Adding
        -- this column needs an upgrade-script ADD COLUMN against an
        -- already-installed 0.3.0-era pinecone.tables.
        engine text CHECK (engine IN ('pgvector', 'native', 'remote')),
        -- Whether this table's vector column is a local,
        -- full copy (`'full'`, the default — the invariant every pre-0.6
        -- row already satisfies) or has been shed to Pinecone-only
        -- (`'detached'` — `pinecone.detach_storage`/`attach_storage`; the
        -- correctness guards also hold for a direct `UPDATE pinecone.tables
        -- SET storage = 'detached'`, which is how tests force the state
        -- without the verbs). `mode` deliberately stays `'remote'` while detached —
        -- see `sync_trigger.rs`'s module rustdoc and `topk.rs`'s dispatch,
        -- both of which already panic/misbehave on an unrecognized mode
        -- value, not a storage one.
        storage text NOT NULL DEFAULT 'full' CHECK (storage IN ('full', 'detached')),
        -- Detach audit trail, written by `detach_storage`'s
        -- registry flip and cleared by `attach_storage`'s completion hook
        -- (`outbox::dispatch_recover`'s `attach => true` arm) — NULL for
        -- every `storage = 'full'` row, always. `backup_gate` is a STATIC
        -- point-in-time snapshot of the backup gate's outcome at detach
        -- time ('covered' = every check passed; 'forced' = `force => true`
        -- bypassed at least one failing check) — deliberately NOT the
        -- dynamic 'forced' -> 'covered' re-check: `pinecone.status()` makes no
        -- HTTP call at all, so it can only ever display what was true when
        -- `detach_storage` ran;
        -- `pinecone.backup_readiness(tbl)` remains the live check any time.
        -- `detach_rpo_secs` is the schedule-interval-derived worst-case RPO
        -- (`interval + 15 min`) computed once at detach time from the same
        -- backup-gate HTTP calls, so `status()` can format it with zero
        -- network access of its own.
        detached_at timestamptz,
        detach_backup_id text,
        backup_gate text CHECK (backup_gate IN ('covered', 'forced')),
        detach_rpo_secs int,
        -- Last-known `read_capacity` from a create/describe/adopt call,
        -- pre-rendered for `status()` (which makes no HTTP call of its own)
        -- — e.g. 'OnDemand' or 'Dedicated b1 2x1 Ready'. NULL for a
        -- local/pgvector-engine row, or a remote one never (re)describe'd
        -- since this column shipped.
        read_capacity text,
        -- True for a row registered via `pinecone.create_index`/`set_engine`'s
        -- attach path (an existing Pinecone index adopted in place, no backfill) —
        -- surfaced by `status()` so an operator can tell an attached table apart
        -- from an ordinarily-provisioned one.
        attached boolean NOT NULL DEFAULT false,
        -- A table whose values each get their own namespace, `<namespace>:<value>`:
        -- `namespace` then holds the prefix. NULL (both columns) = one namespace.
        partition_column text,
        partition_kind text CHECK (partition_kind IN ('int2', 'int4', 'int8', 'text', 'uuid')),
        -- The pg_pinecone-AM index's filter key (its second column): synced to Pinecone as the
        -- metadata `{filter_column: v}` and pushed down by scans as `$eq`. NULL = unfiltered. It
        -- is no partition column (that one picks a namespace).
        filter_column text,
        created_at timestamptz NOT NULL DEFAULT now(),
        UNIQUE (index_name, namespace),
        CONSTRAINT tables_partition_pair_check
            CHECK ((partition_column IS NULL) = (partition_kind IS NULL)),
        CONSTRAINT tables_filter_column_am_check
            CHECK (filter_column IS NULL OR synced_by = 'am'),
        -- The trigger's detached branches enqueue ids without a partition value.
        CONSTRAINT tables_partition_storage_check
            CHECK (partition_column IS NULL OR storage = 'full'),
        -- HARD BLOCK (AM-table detach is a non-goal):
        -- `synced_by = 'am'`/`'engine'` rows can NEVER be detached, even via a
        -- direct superuser `UPDATE pinecone.tables` — the AM's
        -- NULLing rewrite re-mints every TID (`amgettuple` would return dead
        -- ones) and `aminsert` no-ops on NULL, so nothing re-keys and the
        -- next VACUUM's `ambulkdelete` remote-deletes every old TID: the
        -- only copy. Enforcing this as a table CHECK (rather than only in
        -- application code) means even a hand-crafted `UPDATE` from a
        -- superuser session — exactly how tests force the
        -- `'detached'` state without the verbs —
        -- can't create the one row-shape `detach_storage` must never see.
        CONSTRAINT tables_storage_synced_by_check CHECK (storage = 'full' OR synced_by = 'trigger')
    );
    SELECT pg_catalog.pg_extension_config_dump('pinecone.tables', '');
    "#,
    name = "pg_pinecone_tables",
    requires = [pinecone],
);

/// One row of `pinecone.tables`.
pub(crate) struct RegistryRow {
    pub(crate) relid: pg_sys::Oid,
    pub(crate) vector_column: String,
    pub(crate) id_column: String,
    pub(crate) metric: String,
    pub(crate) dimension: i32,
    pub(crate) metadata_columns: Vec<String>,
    pub(crate) mode: String,
    pub(crate) sync_mode: String,
    /// Which write path owns this table's remote sync: `'trigger'`
    /// (`pinecone.promote`'s sync triggers, the only path before 0.3's AM) or
    /// `'am'` (a `CREATE INDEX ... USING pinecone` — see `am.rs`). The two are
    /// mutually exclusive per registry row: `am::ambuild` refuses a table
    /// already registered any other way, and `pinecone.track`/`promote` are
    /// not taught about `'am'` rows (out of scope — see `am.rs`).
    pub(crate) synced_by: String,
    pub(crate) queue_parked: bool,
    pub(crate) parked_reason: Option<String>,
    pub(crate) index_name: Option<String>,
    pub(crate) namespace: Option<String>,
    /// The pg_pinecone-AM index relation's own oid — `Some` only for
    /// `synced_by = 'am'` rows; see the column's own comment in this
    /// module's `extension_sql!` block.
    pub(crate) am_index_relid: Option<pg_sys::Oid>,
    /// The most recent `ambulkdelete` (VACUUM) failure's message, if any —
    /// see the column's own comment in this module's `extension_sql!` block.
    pub(crate) last_vacuum_error: Option<String>,
    /// `'pgvector' | 'native' | 'remote'`, or `None` for a row the engine
    /// surface (`lifecycle.rs`) has never touched — see the column's own
    /// comment in this module's `extension_sql!` block.
    pub(crate) engine: Option<String>,
    /// `'full'` (the default) or `'detached'` — see the column's own
    /// comment in this module's `extension_sql!` block. The table's own
    /// `CHECK` guarantees `'detached'` never coexists with `synced_by !=
    /// 'trigger'`.
    pub(crate) storage: String,
    /// Detach audit columns — see their own comments in
    /// this module's `extension_sql!` block. Always `None` together for a
    /// `storage = 'full'` row.
    pub(crate) detached_at: Option<TimestampWithTimeZone>,
    pub(crate) detach_backup_id: Option<String>,
    pub(crate) backup_gate: Option<String>,
    pub(crate) detach_rpo_secs: Option<i32>,
    pub(crate) read_capacity: Option<String>,
    pub(crate) attached: bool,
    /// The column whose values each get their own Pinecone namespace (`namespace` is then the
    /// prefix of the family); `None` for a table with one namespace.
    pub(crate) partition: Option<PartitionSpec>,
    /// The filter key (second column) of a `synced_by = 'am'` index: synced to Pinecone as
    /// metadata and pushed down by scans. `None` for an unfiltered index and every other row.
    pub(crate) filter_column: Option<String>,
}

impl RegistryRow {
    /// Whether the table syncs to Pinecone: `remote`, or `promoting` while the load runs. A bare
    /// `demote` keeps its index and partition state but syncs nothing.
    pub(crate) fn syncs(&self) -> bool {
        matches!(self.mode.as_str(), "remote" | "promoting")
    }
}

/// Looks up the registry row for `relid`, if any.
pub(crate) fn lookup(relid: pg_sys::Oid) -> Option<RegistryRow> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT relid::oid AS relid, vector_column, id_column, metric, dimension, \
                 metadata_columns, mode, sync_mode, synced_by, queue_parked, parked_reason, \
                 index_name, namespace, am_index_relid, last_vacuum_error, engine, storage, \
                 detached_at, detach_backup_id, backup_gate, detach_rpo_secs, read_capacity, \
                 attached, partition_column, partition_kind, filter_column \
                 FROM pinecone.tables WHERE relid::oid = $1::oid",
                None,
                &[relid.into()],
            )
            .expect("pg_pinecone: registry lookup query failed")
            .next()
            .map(|row| row_from_tuple(&row))
    })
}

/// Whether the caller may read `pinecone.tables` itself.
fn caller_reads_registry() -> bool {
    Spi::get_one::<bool>(
        "SELECT pg_catalog.has_table_privilege(current_user, 'pinecone.tables', 'SELECT')",
    )
    .expect("pg_pinecone: registry privilege query failed")
    .expect("pg_pinecone: registry privilege unexpectedly NULL")
}

/// [`lookup`] for a verb that only reads (`topk`): a caller without a grant on `pinecone.tables`
/// gets the row through the registry's owner, for a table they may `SELECT` and nothing else.
pub(crate) fn lookup_for_reader(relid: pg_sys::Oid, display: &str) -> Option<RegistryRow> {
    if caller_reads_registry() {
        return lookup(relid);
    }
    if !docs_registry::can_see(relid) {
        backend::error_with_code(
            PgSqlErrorCode::ERRCODE_INSUFFICIENT_PRIVILEGE,
            format!("permission denied for table {display}"),
            None,
            None,
        );
    }
    docs_registry::as_owner(|| lookup(relid))
}

/// [`list`] for `status()`: without a grant on `pinecone.tables`, the rows of the tables the
/// caller may `SELECT`, read through the registry's owner.
pub(crate) fn list_for_reader() -> Vec<RegistryRow> {
    if caller_reads_registry() {
        return list();
    }
    let rows = docs_registry::as_owner(list);
    rows.into_iter()
        .filter(|row| docs_registry::can_see(row.relid))
        .collect()
}

/// All registry rows, ordered by `relid` (a stable, if arbitrary, order).
pub(crate) fn list() -> Vec<RegistryRow> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT relid::oid AS relid, vector_column, id_column, metric, dimension, \
                 metadata_columns, mode, sync_mode, synced_by, queue_parked, parked_reason, \
                 index_name, namespace, am_index_relid, last_vacuum_error, engine, storage, \
                 detached_at, detach_backup_id, backup_gate, detach_rpo_secs, read_capacity, \
                 attached, partition_column, partition_kind, filter_column \
                 FROM pinecone.tables ORDER BY relid::oid",
                None,
                &[],
            )
            .expect("pg_pinecone: registry list query failed")
            .map(|row| row_from_tuple(&row))
            .collect()
    })
}

/// Inserts a new registry row. Panics (via a failed `UNIQUE`/`CHECK`
/// constraint) if `row.relid` is already tracked — callers must check
/// [`lookup`] first.
pub(crate) fn insert(row: &RegistryRow) {
    Spi::run_with_args(
        "INSERT INTO pinecone.tables \
         (relid, vector_column, id_column, metric, dimension, metadata_columns, mode, \
          sync_mode, synced_by, index_name, namespace, am_index_relid, engine, read_capacity, \
          attached, partition_column, partition_kind, filter_column) \
         VALUES ($1::oid::regclass, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, \
                 $16, $17, $18)",
        &[
            row.relid.into(),
            row.vector_column.clone().into(),
            row.id_column.clone().into(),
            row.metric.clone().into(),
            row.dimension.into(),
            row.metadata_columns.clone().into(),
            row.mode.clone().into(),
            row.sync_mode.clone().into(),
            row.synced_by.clone().into(),
            row.index_name.clone().into(),
            row.namespace.clone().into(),
            row.am_index_relid.into(),
            row.engine.clone().into(),
            row.read_capacity.clone().into(),
            row.attached.into(),
            row.partition.as_ref().map(|p| p.column.clone()).into(),
            row.partition.as_ref().map(|p| p.kind.as_str()).into(),
            row.filter_column.clone().into(),
        ],
    )
    .expect("pg_pinecone: registry insert failed");
}

/// Sets (or clears) the `read_capacity` display string on an EXISTING row —
/// `promote::promote`'s create-or-adopt step, right after it has a fresh one
/// from Pinecone (a fresh insert stamps it directly instead; see
/// `am::ambuild`/`lifecycle::promote_index`).
pub(crate) fn set_read_capacity(relid: pg_sys::Oid, read_capacity: Option<&str>) {
    Spi::run_with_args(
        "UPDATE pinecone.tables SET read_capacity = $2 WHERE relid::oid = $1::oid",
        &[relid.into(), read_capacity.into()],
    )
    .expect("pg_pinecone: registry set_read_capacity failed");
}

/// Sets (or clears, `engine = None`) the engine-surface column on an
/// EXISTING row — `lifecycle::create_index`/`set_engine`/`promote_index`/
/// `demote_index` call this after they've already ensured a row exists
/// (inserting one first via [`insert`] if none did). Never touches
/// `mode`/`synced_by`/anything else — this is purely the lifecycle surface's
/// own bookkeeping label, orthogonal to whichever of `synced_by`'s values
/// governs the row's actual sync mechanics.
pub(crate) fn set_engine(relid: pg_sys::Oid, engine: Option<&str>) {
    Spi::run_with_args(
        "UPDATE pinecone.tables SET engine = $2 WHERE relid::oid = $1::oid",
        &[relid.into(), engine.into()],
    )
    .expect("pg_pinecone: registry set_engine failed");
}

/// Sets (or clears) the partition declaration of an EXISTING row: `promote` stores what it
/// validated before it creates or fills anything.
pub(crate) fn set_partition(relid: pg_sys::Oid, partition: Option<&PartitionSpec>) {
    Spi::run_with_args(
        "UPDATE pinecone.tables SET partition_column = $2, partition_kind = $3 \
         WHERE relid::oid = $1::oid",
        &[
            relid.into(),
            partition.map(|p| p.column.clone()).into(),
            partition.map(|p| p.kind.as_str()).into(),
        ],
    )
    .expect("pg_pinecone: registry set_partition failed");
}

/// Updates the re-trackable columns of an existing (`mode = 'local'`) row.
/// `mode`/`index_name`/`namespace` are left untouched — those only change via
/// `pinecone.promote`/`demote`.
pub(crate) fn update_local(row: &RegistryRow) {
    Spi::run_with_args(
        "UPDATE pinecone.tables SET vector_column = $2, id_column = $3, metric = $4, \
         dimension = $5, metadata_columns = $6 \
         WHERE relid::oid = $1::oid",
        &[
            row.relid.into(),
            row.vector_column.clone().into(),
            row.id_column.clone().into(),
            row.metric.clone().into(),
            row.dimension.into(),
            row.metadata_columns.clone().into(),
        ],
    )
    .expect("pg_pinecone: registry update failed");
}

/// Refreshes an existing `synced_by = 'am'` row's `vector_column`/
/// `id_column`/`metric`/`dimension`/`filter_column`/`am_index_relid` in place —
/// `am::ambuild`'s REINDEX/VACUUM FULL/CLUSTER rebuild path (re-running
/// `ambuild` on an already-registered pg_pinecone access-method index).
/// `mode`/`sync_mode`/`synced_by`/`index_name`/`namespace` are left
/// untouched: rebuilding the *local* index relation never changes the
/// Pinecone index's own identity. `am_index_relid` still needs refreshing
/// even though a plain `REINDEX` keeps the same index oid: `am::ambuild`'s
/// other rebuild case — a fresh `CREATE INDEX` superseding a registration
/// whose recorded index has since gone *invalid* (e.g. a `CREATE INDEX
/// CONCURRENTLY` that failed after its `ambuild` transaction committed) —
/// lands here with a genuinely new oid; if this row kept pointing at the
/// stale one, the `sql_drop` event trigger firing when the user eventually
/// drops that invalid leftover would wrongly purge the registry row that's
/// actively serving the replacement. (`REINDEX CONCURRENTLY`, by contrast,
/// does NOT reach here anymore: its new-copy build runs while the OLD index
/// is still valid under a different oid, which is indistinguishable inside
/// `ambuild` from a user creating a second identical index — `am::ambuild`'s
/// one-index-per-table guard now errors on both; plain
/// `REINDEX` is the supported rebuild path.)
pub(crate) fn rebuild_am(row: &RegistryRow) {
    Spi::run_with_args(
        "UPDATE pinecone.tables SET vector_column = $2, id_column = $3, metric = $4, \
         dimension = $5, am_index_relid = $6, filter_column = $7 WHERE relid::oid = $1::oid",
        &[
            row.relid.into(),
            row.vector_column.clone().into(),
            row.id_column.clone().into(),
            row.metric.clone().into(),
            row.dimension.into(),
            row.am_index_relid.into(),
            row.filter_column.clone().into(),
        ],
    )
    .expect("pg_pinecone: registry am-rebuild update failed");
}

/// Deletes the registry row for `relid`. Callers must check [`lookup`] first
/// (to raise a clear "not tracked" error rather than silently no-op deleting).
pub(crate) fn delete(relid: pg_sys::Oid) {
    Spi::run_with_args(
        "DELETE FROM pinecone.tables WHERE relid::oid = $1::oid",
        &[relid.into()],
    )
    .expect("pg_pinecone: registry delete failed");
}

/// The `relid` of another registry row already holding the exact namespace `namespace` in
/// `index_name`, if any: see [`claim_owner`], which also knows partition families.
#[allow(
    dead_code,
    reason = "shorthand for an exact claim; the verbs call claim_owner"
)]
pub(crate) fn index_namespace_owner(
    index_name: &str,
    namespace: &str,
    relid: pg_sys::Oid,
) -> Option<pg_sys::Oid> {
    claim_owner(index_name, Claim::Exact(namespace), relid).map(|owner| owner.relid)
}

/// Another registration (never `relid` itself) whose claim in `index_name` clashes with `claim`:
/// the same namespace, or a partition family covering it or covered by it. Rows of
/// `pinecone.doc_tables` count too (when the caller can read them) and never exempt `relid`: a
/// document index can't also be the same table's vector index.
pub(crate) fn claim_owner(index_name: &str, claim: Claim<'_>, relid: pg_sys::Oid) -> Option<Owner> {
    let held = Spi::connect(|client| {
        client
            .select(
                &format!(
                    "SELECT relid::oid AS relid, namespace, partition_column IS NOT NULL AS partitioned \
                     FROM pinecone.tables \
                     WHERE index_name = $1 AND relid::oid <> $3::oid AND {} \
                     ORDER BY relid::oid LIMIT 1",
                    partition_registry::overlap(claim)
                ),
                None,
                &[index_name.into(), claim.namespace().into(), relid.into()],
            )
            .expect("pg_pinecone: registry claim owner query failed")
            .next()
            .map(|row| Owner {
                relid: get(&row, "relid"),
                namespace: get(&row, "namespace"),
                partitioned: get(&row, "partitioned"),
            })
    });
    held.or_else(|| {
        docs_registry::readable()
            .then(|| docs_registry::claim_owner(index_name, claim))
            .flatten()
    })
}

/// Whether any other registry row (any namespace) uses `index_name`. Used by
/// `pinecone.demote`'s `drop_remote_index` refusal: dropping the index out from
/// under another tracked table would strand its `'remote'` registration. Any
/// readable `pinecone.doc_tables` row counts (see [`index_namespace_owner`]).
pub(crate) fn other_row_uses_index(index_name: &str, relid: pg_sys::Oid) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT EXISTS (SELECT 1 FROM pinecone.tables WHERE index_name = $1 AND relid::oid <> $2::oid)",
        &[index_name.into(), relid.into()],
    )
    .expect("pg_pinecone: registry shared-index query failed")
    .expect("pg_pinecone: EXISTS unexpectedly NULL")
        || doc_index_visible(index_name)
}

/// Whether ANY registry row currently uses `index_name` — no relid
/// exclusion, unlike [`other_row_uses_index`]: `lifecycle::remote_cleanup`'s
/// re-check runs AFTER the demoted row was deleted (in an earlier, already-
/// committed transaction), so every remaining row is by definition someone
/// else's claim on the index. Includes readable
/// `pinecone.doc_tables` rows.
pub(crate) fn any_row_uses_index(index_name: &str) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT EXISTS (SELECT 1 FROM pinecone.tables WHERE index_name = $1)",
        &[index_name.into()],
    )
    .expect("pg_pinecone: registry any-row-uses-index query failed")
    .expect("pg_pinecone: EXISTS unexpectedly NULL")
        || doc_index_visible(index_name)
}

/// Whether a document table the caller may see uses `index_name`.
fn doc_index_visible(index_name: &str) -> bool {
    docs_registry::readable() && docs_registry::index_owner(index_name).is_some()
}

/// Flips a tracked row to `'remote'` mode with the given
/// `index_name`/`namespace`/`sync_mode` (`pinecone.promote`, after the index
/// is ready and backfilled — `sync_mode` itself is resolved by
/// `promote::resolve_sync_mode` before this call).
pub(crate) fn promote_remote(
    relid: pg_sys::Oid,
    index_name: &str,
    namespace: &str,
    sync_mode: &str,
) {
    Spi::run_with_args(
        "UPDATE pinecone.tables SET mode = 'remote', index_name = $2, namespace = $3, \
         sync_mode = $4 WHERE relid::oid = $1::oid",
        &[
            relid.into(),
            index_name.into(),
            namespace.into(),
            sync_mode.into(),
        ],
    )
    .expect("pg_pinecone: registry promote update failed");
}

/// Flips a tracked row to `'promoting'` mode:
/// `pinecone.promote`'s async path installs the triggers and writes this row
/// in the same fast transaction as enqueuing the `delete_all`+`backfill`
/// jobs — no inline HTTP backfill happens here. The worker's own backfill
/// dispatch ([`flip_promoting_to_remote`]) is the only place this flips to
/// `'remote'`; `pinecone.topk` treats `'promoting'` as local in the meantime
/// (never serving a partial namespace).
pub(crate) fn promote_promoting(
    relid: pg_sys::Oid,
    index_name: &str,
    namespace: &str,
    sync_mode: &str,
) {
    Spi::run_with_args(
        "UPDATE pinecone.tables SET mode = 'promoting', index_name = $2, namespace = $3, \
         sync_mode = $4 WHERE relid::oid = $1::oid",
        &[
            relid.into(),
            index_name.into(),
            namespace.into(),
            sync_mode.into(),
        ],
    )
    .expect("pg_pinecone: registry promote_promoting update failed");
}

/// Flips a `'remote'` row back to `'promoting'` — `pinecone.sync` on an
/// async table, in the SAME transaction as its `outbox::enqueue_rebuild`:
/// the enqueued `delete_all` may empty
/// the namespace long before the `backfill` refills it, so `pinecone.topk`
/// must serve local until the rebuild actually drains. Unlike
/// [`promote_promoting`], never touches `index_name`/`namespace`/`sync_mode`
/// — a synced table keeps its existing registration; only the mode flips.
/// Guarded by `AND mode = 'remote'` so a `'promoting'` row (a still-draining
/// earlier rebuild, about to be coalesced away) stays as-is.
pub(crate) fn flip_remote_to_promoting(relid: pg_sys::Oid) {
    Spi::run_with_args(
        "UPDATE pinecone.tables SET mode = 'promoting' WHERE relid::oid = $1::oid \
         AND mode = 'remote'",
        &[relid.into()],
    )
    .expect("pg_pinecone: registry flip remote->promoting failed");
}

/// Flips a `'promoting'` row to `'remote'` — called by `outbox`'s backfill
/// dispatch, in the same transaction as deleting the now-exhausted
/// `'backfill'` job row, once its live, PK-ordered chunk read comes back
/// empty. Guarded by `AND mode = 'promoting'` so a redundant call (e.g. a
/// concurrent demote already flipped the row back to `'local'`) is a silent
/// no-op rather than clobbering a state this call didn't expect to see.
pub(crate) fn flip_promoting_to_remote(relid: pg_sys::Oid) {
    Spi::run_with_args(
        "UPDATE pinecone.tables SET mode = 'remote' WHERE relid::oid = $1::oid \
         AND mode = 'promoting'",
        &[relid.into()],
    )
    .expect("pg_pinecone: registry flip promoting->remote failed");
}

/// Re-points a `'remote'` row's `index_name` at a different Pinecone index —
/// `pinecone.repoint`, after its caller has already
/// validated dimension/metric via `describe_index`. `namespace`/`mode`/
/// `sync_mode` are left untouched: a restore-to-new-index preserves
/// namespaces, and repointing doesn't change how writes drain. A registry
/// ROW update only — never touches `pinecone.tables`'s schema.
pub(crate) fn repoint_index(relid: pg_sys::Oid, new_index_name: &str) {
    Spi::run_with_args(
        "UPDATE pinecone.tables SET index_name = $2 WHERE relid::oid = $1::oid",
        &[relid.into(), new_index_name.into()],
    )
    .expect("pg_pinecone: registry repoint update failed");
}

/// Flips a `'remote'` row back to `'local'` (`pinecone.demote`). `clear_index`
/// also blanks `index_name`/`namespace` and the partition declaration — set only when `demote`
/// actually deleted the remote index; otherwise they're kept so a bare `pinecone.promote`
/// re-promote reuses them.
pub(crate) fn demote_local(relid: pg_sys::Oid, clear_index: bool) {
    let result = if clear_index {
        Spi::run_with_args(
            "UPDATE pinecone.tables SET mode = 'local', index_name = NULL, namespace = NULL, \
             partition_column = NULL, partition_kind = NULL WHERE relid::oid = $1::oid",
            &[relid.into()],
        )
    } else {
        Spi::run_with_args(
            "UPDATE pinecone.tables SET mode = 'local' WHERE relid::oid = $1::oid",
            &[relid.into()],
        )
    };
    result.expect("pg_pinecone: registry demote update failed");
}

/// Parks `relid`'s outbox queue: `outbox::drain_step` calls this once a
/// permanent-class error's `attempts` reaches `pinecone.outbox_max_attempts`
/// (the poison escape) — the worker/scheduler skips a
/// parked relid until [`unpark`] (`pinecone.outbox_retry`) runs.
pub(crate) fn set_parked(relid: pg_sys::Oid, reason: &str) {
    Spi::run_with_args(
        "UPDATE pinecone.tables SET queue_parked = true, parked_reason = $2 \
         WHERE relid::oid = $1::oid",
        &[relid.into(), reason.into()],
    )
    .expect("pg_pinecone: registry set_parked failed");
}

/// Clears [`set_parked`]'s flag/reason — `pinecone.outbox_retry`, and
/// `pinecone.demote` (so a stale parked flag never survives a demote/re-promote
/// cycle into a table that hasn't even been re-promoted yet).
pub(crate) fn unpark(relid: pg_sys::Oid) {
    Spi::run_with_args(
        "UPDATE pinecone.tables SET queue_parked = false, parked_reason = NULL \
         WHERE relid::oid = $1::oid",
        &[relid.into()],
    )
    .expect("pg_pinecone: registry unpark failed");
}

/// Records `am::ambulkdelete`'s most recent failure message on `relid`'s
/// registry row: called right
/// before `backend::raise` turns the same `ClientError` into a Postgres
/// `ERROR`, so `pinecone.status()` can fold it into its own `last_error`
/// column and let an operator tell a genuine VACUUM/Pinecone-outage failure
/// apart from an ordinary outbox backoff.
///
/// **Best-effort, not durable against the very failure it records**: this
/// `UPDATE` runs in the *same* transaction `ambulkdelete` is about to abort
/// by raising — Postgres has no autonomous-transaction primitive an
/// extension can reach for here, so when the caller (VACUUM) lets that
/// `ERROR` abort the transaction, this write is rolled back right along with
/// it, same as any other write earlier in a transaction that ends in
/// `ERROR`. It still runs (rather than being skipped as pointless): a caller
/// that catches/logs the error without aborting (or a future out-of-process
/// harness reading a separate, already-committed connection) does see it.
/// Closing the gap for the common, real-VACUUM case needs a genuinely
/// separate connection (`outbox.last_error` has the same durability gap) —
/// left as a documented limitation, not built here.
pub(crate) fn record_vacuum_error(relid: pg_sys::Oid, message: &str) {
    Spi::run_with_args(
        "UPDATE pinecone.tables SET last_vacuum_error = $2 WHERE relid::oid = $1::oid",
        &[relid.into(), message.into()],
    )
    .expect("pg_pinecone: registry record_vacuum_error failed");
}

/// Errors clearly if `row` is managed by the pg_pinecone access method
/// (`synced_by = 'am'`) — `pinecone.track`/`untrack`/`promote`/`demote`/
/// `sync`/`topk` do not apply to an AM-managed table: there are no sync
/// triggers to install/drop/resync (the AM's own `aminsert`/`ambulkdelete`
/// own its remote sync), and it isn't tracked toward a `pinecone.topk` call
/// either — queries against it should go through a plain `ORDER BY
/// <vector_column> <=> <value> LIMIT k`, served transparently by the AM's
/// own `amgettuple` (`am.rs`). `DROP INDEX` (see `am::install_sql_drop_event_trigger`)
/// is the only supported way to detach it from pg_pinecone.
///
/// Called by every one of those functions' own row-lookup helper, right
/// after `lookup`/`remote_row`/`demotable_row` confirms a row exists —
/// before any mode check, so an AM row (always `mode = 'remote'`) gets this
/// message instead of a generic "not promoted"/"run demote() first" one that
/// would send the caller toward a code path (`DROP TRIGGER`) that doesn't
/// exist for it (`demote()` used to raise a confusing "trigger does not
/// exist" error here).
pub(crate) fn require_not_am(row: &RegistryRow, func: &str, display: &str) {
    if row.synced_by == "am" {
        error!(
            "{func}: {display} is managed by the pg_pinecone index — DROP INDEX to detach; query \
             with ORDER BY {} <=> <value> instead",
            row.vector_column
        );
    }
}

fn row_from_tuple(row: &pgrx::spi::SpiHeapTupleData<'_>) -> RegistryRow {
    RegistryRow {
        relid: get(row, "relid"),
        vector_column: get(row, "vector_column"),
        id_column: get(row, "id_column"),
        metric: get(row, "metric"),
        dimension: get(row, "dimension"),
        metadata_columns: get(row, "metadata_columns"),
        mode: get(row, "mode"),
        sync_mode: get(row, "sync_mode"),
        synced_by: get(row, "synced_by"),
        queue_parked: get(row, "queue_parked"),
        parked_reason: row
            .get_by_name::<String, _>("parked_reason")
            .expect("pg_pinecone: registry row parked_reason query failed"),
        index_name: row
            .get_by_name::<String, _>("index_name")
            .expect("pg_pinecone: registry row index_name query failed"),
        namespace: row
            .get_by_name::<String, _>("namespace")
            .expect("pg_pinecone: registry row namespace query failed"),
        am_index_relid: row
            .get_by_name::<pg_sys::Oid, _>("am_index_relid")
            .expect("pg_pinecone: registry row am_index_relid query failed"),
        last_vacuum_error: row
            .get_by_name::<String, _>("last_vacuum_error")
            .expect("pg_pinecone: registry row last_vacuum_error query failed"),
        engine: row
            .get_by_name::<String, _>("engine")
            .expect("pg_pinecone: registry row engine query failed"),
        storage: get(row, "storage"),
        detached_at: row
            .get_by_name::<TimestampWithTimeZone, _>("detached_at")
            .expect("pg_pinecone: registry row detached_at query failed"),
        detach_backup_id: row
            .get_by_name::<String, _>("detach_backup_id")
            .expect("pg_pinecone: registry row detach_backup_id query failed"),
        backup_gate: row
            .get_by_name::<String, _>("backup_gate")
            .expect("pg_pinecone: registry row backup_gate query failed"),
        detach_rpo_secs: row
            .get_by_name::<i32, _>("detach_rpo_secs")
            .expect("pg_pinecone: registry row detach_rpo_secs query failed"),
        read_capacity: row
            .get_by_name::<String, _>("read_capacity")
            .expect("pg_pinecone: registry row read_capacity query failed"),
        attached: get(row, "attached"),
        partition: PartitionSpec::from_columns(
            row.get_by_name::<String, _>("partition_column")
                .expect("pg_pinecone: registry row partition_column query failed"),
            row.get_by_name::<String, _>("partition_kind")
                .expect("pg_pinecone: registry row partition_kind query failed"),
        )
        .unwrap_or_else(|e| panic!("pg_pinecone: registry row {e}")),
        filter_column: row
            .get_by_name::<String, _>("filter_column")
            .expect("pg_pinecone: registry row filter_column query failed"),
    }
}

/// Errors clearly if `row.storage` is `'detached'` — `pinecone.sync`/`pinecone.demote`
/// call this (`sync()` while detached is `deleteAll` +
/// a backfill that skips every NULL-vector row, i.e. a total remote wipe;
/// `demote()` while detached would silently abandon Pinecone's only copy with
/// no further sync at all). The only sanctioned way back to `storage =
/// 'full'` is `pinecone.attach_storage(tbl)`: it recovers real
/// values from Pinecone via `pinecone.recover_from_remote` before flipping the
/// flag back, unlike either of these ops.
pub(crate) fn require_not_detached(row: &RegistryRow, func: &str, display: &str) {
    if row.storage == "detached" {
        error!(
            "{func}: {display} is detached (storage = 'detached') — Pinecone holds the only \
             full copy of its vector column; run pinecone.attach_storage(...) first"
        );
    }
}

/// Flips a `'full'` row to `'detached'` and stamps the detach audit columns —
/// `detach::detach_storage`'s registry-write step, same transaction as
/// `sync_trigger::install_detached_id_guard` and the batched NULLing pass
/// (see that module for the full ordering rationale). `backup_id`/`gate`/
/// `rpo_secs` come from `backup_readiness::detach_gate`'s single HTTP round
/// at gate time — never recomputed later (`pinecone.status()`'s "no HTTP" rule).
pub(crate) fn mark_detached(
    relid: pg_sys::Oid,
    backup_id: Option<&str>,
    gate: &str,
    rpo_secs: Option<i32>,
) {
    Spi::run_with_args(
        "UPDATE pinecone.tables SET storage = 'detached', detached_at = now(), \
         detach_backup_id = $2, backup_gate = $3, detach_rpo_secs = $4 \
         WHERE relid::oid = $1::oid",
        &[relid.into(), backup_id.into(), gate.into(), rpo_secs.into()],
    )
    .expect("pg_pinecone: registry mark_detached failed");
}

/// Flips a `'detached'` row back to `'full'` and clears every detach audit
/// column — `outbox::dispatch_recover`'s `attach => true` completion hook,
/// once `pinecone.attach_storage`'s recover job exhausts (same ambient
/// transaction as that job row's own deletion).
pub(crate) fn mark_attached(relid: pg_sys::Oid) {
    Spi::run_with_args(
        "UPDATE pinecone.tables SET storage = 'full', detached_at = NULL, \
         detach_backup_id = NULL, backup_gate = NULL, detach_rpo_secs = NULL \
         WHERE relid::oid = $1::oid",
        &[relid.into()],
    )
    .expect("pg_pinecone: registry mark_attached failed");
}

/// Extracts a required, non-NULL column value, panicking (rather than
/// returning a wrong/empty result) if this module's own SQL shape ever stops
/// matching the table it created.
fn get<T: FromDatum + IntoDatum>(row: &pgrx::spi::SpiHeapTupleData<'_>, name: &str) -> T {
    row.get_by_name::<T, _>(name)
        .unwrap_or_else(|e| panic!("pg_pinecone: registry row {name} query failed: {e}"))
        .unwrap_or_else(|| panic!("pg_pinecone: registry row {name} unexpectedly NULL"))
}
