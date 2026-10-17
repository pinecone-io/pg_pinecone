//! Implementation of `pinecone.track` / `pinecone.untrack`. `#[pg_schema]` only
//! attaches to an inline `mod {}` (see the comment on `pg_pinecone` in `lib.rs`), so
//! its `#[pg_extern]` wrappers are kept thin and delegate here.

use pgrx::{PgRelation, prelude::*};

use crate::{
    catalog::{self, ColumnCatalog},
    filter::ColKind,
    partition::Lane,
    partition_registry,
    registry::{self, RegistryRow},
};

const VECTOR_TYPE: &str = "vector";
const METRICS: &[&str] = &["cosine", "l2", "ip"];

/// Registers (or re-registers) `tbl` for local search. See the `pinecone.track`
/// rustdoc in `lib.rs` for the autodetection rules.
pub(crate) fn track(
    tbl: PgRelation,
    vector_column: Option<&str>,
    id_column: Option<&str>,
    metric: Option<&str>,
    metadata_columns: Option<Vec<Option<String>>>,
) {
    let relid = tbl.oid();
    let display = catalog::display_name(&tbl);

    let existing = registry::lookup(relid);
    if let Some(row) = &existing
        && row.mode != "local"
    {
        registry::require_not_am(row, "pinecone.track", &display);
        error!(
            "pinecone.track: {display} is tracked in '{}' mode — run pinecone.demote(...) first",
            row.mode
        );
    }

    // Fail fast (and clearly) if pgvector itself isn't installed, before any
    // column-shape validation below.
    catalog::pgvector_namespace().unwrap_or_else(|e| error!("pinecone.track: {e}"));

    let columns = catalog::all_columns(relid);

    // NULL args mean "keep the existing registration" whenever one exists
    // (a re-`track()`/bare re-`promote()` on an already-tracked table) —
    // only an *explicit* arg autodetects/overrides. This must be checked
    // per-arg, not by skipping resolution entirely when `existing.is_some()`:
    // a caller may override just one of the four independently.
    let vector_column = resolve_vector_column(&columns, vector_column, existing.as_ref(), &display);
    let dimension = catalog::vector_dimension(relid, &vector_column.name);
    if dimension <= 0 {
        error!(
            "pinecone.track: column \"{}\" of {display} has an undeclared vector dimension — \
             declare it as vector(n), e.g. vector(1536)",
            vector_column.name
        );
    }

    let id_column = resolve_id_column(relid, id_column, existing.as_ref(), &display);
    let metric = resolve_metric(
        metric,
        relid,
        vector_column.attnum,
        existing.as_ref(),
        &display,
    );
    let metadata_columns =
        resolve_metadata_columns(&columns, metadata_columns, existing.as_ref(), &display);

    let row = RegistryRow {
        relid,
        vector_column: vector_column.name,
        id_column,
        metric,
        dimension,
        metadata_columns,
        mode: "local".to_string(),
        // Irrelevant for `update_local` (which never touches this column) —
        // only `promote::resolve_sync_mode` gives it real meaning, the first
        // time this row is ever promoted.
        sync_mode: "async".to_string(),
        // Irrelevant for `update_local` too (same reason as `sync_mode`
        // above) — `pinecone.track`/`untrack` never touch `synced_by` rows
        // created by the AM path (`am::ambuild` registers those directly),
        // so a table reaching `track()` always owns its sync via the trigger
        // path.
        synced_by: "trigger".to_string(),
        // Irrelevant for `update_local` too (same reason as `sync_mode`
        // above) — `insert` doesn't write these columns either, relying on
        // their table `DEFAULT`s (`false`/`NULL`); a genuinely new row is
        // never parked.
        queue_parked: false,
        parked_reason: None,
        index_name: None,
        namespace: None,
        // Irrelevant for `update_local` too (same reason as `sync_mode`
        // above) — see `synced_by`'s comment: `track()`/`untrack()` never
        // touch an AM-owned row, so a row reaching `insert` here always has
        // no AM index of its own.
        am_index_relid: None,
        // Irrelevant for `update_local` too (same reason as `sync_mode`
        // above) — only `am::ambulkdelete` ever writes this column.
        last_vacuum_error: None,
        // `pinecone.track`/`untrack` predate the engine surface and never
        // touch it — see the `engine` column's own doc comment in
        // `registry.rs`.
        engine: None,
        // Irrelevant for `update_local` too (same reason as `sync_mode`
        // above) — `storage` only ever flips via
        // `pinecone.detach_storage`/`attach_storage`; a genuinely new or
        // re-tracked row always starts (or stays) `'full'`.
        storage: "full".to_string(),
        detached_at: None,
        detach_backup_id: None,
        backup_gate: None,
        detach_rpo_secs: None,
        // `track`/`untrack` never touch Pinecone — only `promote`/`am::ambuild`/
        // `lifecycle::promote_index` ever populate this.
        read_capacity: None,
        attached: false,
        partition: None,
        filter_column: None,
    };

    if existing.is_some() {
        registry::update_local(&row);
    } else {
        registry::insert(&row);
    }
}

/// Removes `tbl` from the registry.
pub(crate) fn untrack(tbl: PgRelation) {
    let relid = tbl.oid();
    let display = catalog::display_name(&tbl);

    let Some(existing) = registry::lookup(relid) else {
        error!("pinecone.untrack: {display} is not tracked");
    };
    if existing.mode != "local" {
        registry::require_not_am(&existing, "pinecone.untrack", &display);
        error!(
            "pinecone.untrack: {display} is tracked in '{}' mode — run pinecone.demote(...) first",
            existing.mode
        );
    }
    partition_registry::forget_partitions(relid, Lane::Vector);
    registry::delete(relid);
}

/// Resolves `vector_column`: the named column if explicit; else the existing
/// row's own value, if re-tracking; else the table's single `vector`-typed
/// column.
fn resolve_vector_column(
    columns: &[ColumnCatalog],
    explicit: Option<&str>,
    existing: Option<&RegistryRow>,
    display: &str,
) -> ColumnCatalog {
    if let Some(name) = explicit {
        let col = find_column(columns, name, display);
        if col.type_name != VECTOR_TYPE {
            error!(
                "pinecone.track: column \"{name}\" of {display} has type {}, not vector",
                col.type_name
            );
        }
        return col.clone();
    }
    if let Some(row) = existing {
        return find_column(columns, &row.vector_column, display).clone();
    }

    let candidates: Vec<&ColumnCatalog> = columns
        .iter()
        .filter(|c| c.type_name == VECTOR_TYPE)
        .collect();
    match candidates.as_slice() {
        [single] => (*single).clone(),
        [] => error!(
            "pinecone.track: {display} has no vector column — add one (e.g. \"embedding \
             vector(1536)\") or pass vector_column explicitly"
        ),
        multiple => {
            let names = multiple
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            error!(
                "pinecone.track: {display} has multiple vector columns ({names}) — \
                 specify vector_column"
            );
        }
    }
}

/// Resolves `id_column`: the named column if explicit; else the existing
/// row's own value, if re-tracking; else the table's single primary-key
/// column. Either way its type must be in [`catalog::ID_COLUMN_TYPES`].
fn resolve_id_column(
    relid: pg_sys::Oid,
    explicit: Option<&str>,
    existing: Option<&RegistryRow>,
    display: &str,
) -> String {
    let (name, type_name) = match explicit {
        Some(name) => {
            let type_name = catalog::column_type(relid, name).unwrap_or_else(|| {
                error!("pinecone.track: column \"{name}\" does not exist on {display}")
            });
            (name.to_string(), type_name)
        }
        None => match existing {
            Some(row) => {
                let type_name = catalog::column_type(relid, &row.id_column).unwrap_or_else(|| {
                    error!(
                        "pinecone.track: previously-registered id_column \"{}\" of {display} no \
                         longer exists — pass id_column explicitly",
                        row.id_column
                    )
                });
                (row.id_column.clone(), type_name)
            }
            None => {
                let pk = catalog::primary_key_columns(relid);
                match pk.as_slice() {
                    [single] => (single.name.clone(), single.type_name.clone()),
                    [] => error!(
                        "pinecone.track: {display} has no primary key — pass id_column explicitly"
                    ),
                    _ => error!(
                        "pinecone.track: {display} has a composite primary key — pass id_column \
                         explicitly"
                    ),
                }
            }
        },
    };

    if !catalog::ID_COLUMN_TYPES.contains(&type_name.as_str()) {
        error!(
            "pinecone.track: column \"{name}\" of {display} has type {type_name}, which \
             pinecone.track does not support for id_column — allowed types: {} (values must have \
             a canonical, GUC-independent text representation)",
            catalog::ID_COLUMN_TYPES.join(", ")
        );
    }
    name
}

/// Resolves `metric`: the explicit value if given (validated against
/// [`METRICS`]); else the existing row's own value, if re-tracking; else
/// autodetected from an existing pgvector index opclass on the vector column,
/// else `'cosine'`. Autodetection keys purely on `pg_opclass.opcname` (via
/// [`catalog::index_opclass_names`]) — it never looks at the index's access
/// method (`pg_am`), so a `hnsw`/`ivfflat` index and a future AM (e.g.
/// pgvectorscale's `diskann`) using the same opclass name are detected
/// identically. If more than one *distinct* recognized opclass is found (e.g.
/// one index per metric), autodetection is ambiguous and errors rather than
/// silently picking one.
fn resolve_metric(
    explicit: Option<&str>,
    relid: pg_sys::Oid,
    vector_column_attnum: i16,
    existing: Option<&RegistryRow>,
    display: &str,
) -> String {
    if let Some(metric) = explicit {
        if !METRICS.contains(&metric) {
            error!(
                "pinecone.track: metric '{metric}' is not supported — must be one of {}",
                METRICS.join(", ")
            );
        }
        return metric.to_string();
    }
    if let Some(row) = existing {
        return row.metric.clone();
    }

    let opclasses = catalog::index_opclass_names(relid, vector_column_attnum);
    let found: Vec<(&str, &str)> = [
        ("vector_cosine_ops", "cosine"),
        ("vector_l2_ops", "l2"),
        ("vector_ip_ops", "ip"),
    ]
    .into_iter()
    .filter(|(opclass, _)| opclasses.iter().any(|o| o == opclass))
    .collect();

    match found.as_slice() {
        [] => "cosine".to_string(),
        [(_, metric)] => metric.to_string(),
        multiple => {
            let names = multiple
                .iter()
                .map(|(opclass, _)| *opclass)
                .collect::<Vec<_>>()
                .join(", ");
            error!(
                "pinecone.track: {display} has indexes with multiple distinct vector opclasses \
                 ({names}) — autodetection is ambiguous, pass metric explicitly (e.g. \
                 metric => 'cosine')"
            );
        }
    }
}

/// Resolves `metadata_columns`: an explicit (possibly empty) array always
/// wins; `NULL` means "keep the existing row's value", if re-tracking, else
/// none. Each named column must exist with a type the filter translator
/// supports ([`catalog::col_kind`]).
fn resolve_metadata_columns(
    columns: &[ColumnCatalog],
    explicit: Option<Vec<Option<String>>>,
    existing: Option<&RegistryRow>,
    display: &str,
) -> Vec<String> {
    let Some(explicit) = explicit else {
        return existing
            .map(|row| row.metadata_columns.clone())
            .unwrap_or_default();
    };

    explicit
        .into_iter()
        .enumerate()
        .map(|(i, name)| {
            let name = name
                .unwrap_or_else(|| error!("pinecone.track: metadata_columns[{}] is NULL", i + 1));
            let col = find_column(columns, &name, display);
            if let ColKind::Unsupported(_) = catalog::col_kind(&col.type_name) {
                error!(
                    "pinecone.track: metadata column \"{name}\" of {display} has unsupported type \
                     {} — pg_pinecone's filter translator does not support it",
                    col.type_name
                );
            }
            name
        })
        .collect()
}

fn find_column<'a>(columns: &'a [ColumnCatalog], name: &str, display: &str) -> &'a ColumnCatalog {
    columns
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| error!("pinecone.track: column \"{name}\" does not exist on {display}"))
}
