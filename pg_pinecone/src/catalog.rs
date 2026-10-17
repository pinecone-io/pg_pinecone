//! Postgres-catalog introspection (via SPI, never raw `pg_sys` catalog access)
//! shared by `pinecone.track`, `pinecone.topk`, and `pinecone.status`. Every
//! identifier this module hands back (schema/table/column names) came straight
//! out of `pg_catalog`, so callers may safely splice it into a dynamic SQL
//! string via `pgrx::spi::quote_identifier`/`quote_qualified_identifier`.

use pgrx::{PgRelation, prelude::*, spi::Spi};

use crate::filter::ColKind;

/// One live (non-dropped) column of a table, as seen by `pg_attribute`/`pg_type`.
#[derive(Clone)]
pub(crate) struct ColumnCatalog {
    pub(crate) name: String,
    pub(crate) attnum: i16,
    pub(crate) type_name: String,
}

/// `id_column` type whitelist: types with a canonical, GUC-independent text
/// representation — unlike e.g. `timestamptz`, whose text output depends on
/// the `TimeZone`/`DateStyle` session GUCs, these always round-trip the same way.
pub(crate) const ID_COLUMN_TYPES: &[&str] = &["text", "varchar", "uuid", "int2", "int4", "int8"];

/// Document-lane id types: [`ID_COLUMN_TYPES`] plus `bpchar`. `char(n)::text` strips the
/// blank padding, which is what `char(n)` equality ignores too, so text distinctness
/// still equals value distinctness; the vector path keeps its narrower list.
pub(crate) const DOC_ID_COLUMN_TYPES: &[&str] =
    &["text", "varchar", "bpchar", "uuid", "int2", "int4", "int8"];

/// Maps a `pg_type.typname` to the [`ColKind`] the filter translator uses.
/// Every type outside `pg_pinecone`'s supported set maps to
/// [`ColKind::Unsupported`] (carrying the type name) rather than being
/// omitted, so `pinecone.topk`'s filter translator can name the type in a
/// clear error if a filter ever references such a column, instead of it
/// reading as an unknown column. Doubles as the `metadata_columns` type
/// whitelist for `pinecone.track` (a column only makes sense as metadata if
/// `pinecone.topk`'s filter translator can target it).
pub(crate) fn col_kind(type_name: &str) -> ColKind {
    match type_name {
        "text" | "varchar" => ColKind::Text,
        "int2" | "int4" | "int8" => ColKind::Int,
        "float4" | "float8" => ColKind::Float,
        "numeric" => ColKind::Numeric,
        "bool" => ColKind::Bool,
        "_text" => ColKind::TextArray,
        other => ColKind::Unsupported(other.to_string()),
    }
}

/// A `schema.table` display name for error messages. Only meaningful for a
/// still-open `PgRelation` — `pinecone.status` instead uses
/// [`resolve_table_name`], which tolerates a since-dropped relid.
pub(crate) fn display_name(tbl: &PgRelation) -> String {
    format!("{}.{}", tbl.namespace(), tbl.name())
}

/// Returns the namespace pgvector's `vector` type is installed into (e.g.
/// `"public"`), so callers can schema-qualify the type and its operators.
pub(crate) fn pgvector_namespace() -> Result<String, String> {
    // Scalar subquery so the statement yields one row (NULL when pgvector is
    // absent) rather than zero rows, which `Spi::get_one` reports as an
    // `InvalidPosition` error — same pattern as `backend::effective_credentials`.
    Spi::get_one::<String>(
        "SELECT (SELECT n.nspname::text FROM pg_catalog.pg_extension e \
         JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace \
         WHERE e.extname = 'vector')",
    )
    .expect("pg_pinecone: pgvector namespace query failed")
    .ok_or_else(|| {
        "pgvector is not installed in this database — run: CREATE EXTENSION vector".to_string()
    })
}

/// The installed version of extension `name` in the connected database (from
/// `pg_extension.extversion`), or `None` if it isn't installed there at all.
/// Used by the background worker (`worker.rs`) both to skip a database that
/// never got `CREATE EXTENSION pg_pinecone` and to detect version skew between a
/// preloaded `.so` and a since-`ALTER EXTENSION UPDATE`d installed version.
pub(crate) fn extension_version(name: &str) -> Option<String> {
    Spi::get_one_with_args::<String>(
        "SELECT (SELECT e.extversion::text FROM pg_catalog.pg_extension e \
         WHERE e.extname = $1)",
        &[name.into()],
    )
    .expect("pg_pinecone: extension version query failed")
}

/// Every live column of `relid`, in `attnum` order.
pub(crate) fn all_columns(relid: pg_sys::Oid) -> Vec<ColumnCatalog> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT a.attname::text AS name, a.attnum, t.typname::text AS type_name \
                 FROM pg_catalog.pg_attribute a \
                 JOIN pg_catalog.pg_type t ON t.oid = a.atttypid \
                 WHERE a.attrelid = $1::oid AND a.attnum > 0 AND NOT a.attisdropped \
                 ORDER BY a.attnum",
                None,
                &[relid.into()],
            )
            .expect("pg_pinecone: column catalog query failed")
            .map(|row| ColumnCatalog {
                name: text(&row, "name"),
                attnum: required(&row, "attnum"),
                type_name: text(&row, "type_name"),
            })
            .collect()
    })
}

/// The type of a single named live column, or `None` if `relid` has no such
/// column. A scalar subquery (rather than a bare `SELECT ... WHERE ...`)
/// always yields exactly one row (NULL on no match) — `Spi::get_one_with_args`
/// requires exactly one result row and errors (`InvalidPosition`) on zero,
/// which every "does this column still exist" caller here relies on *not*
/// happening (see `registry::index_namespace_owner`'s rustdoc for the same
/// pattern applied there).
pub(crate) fn column_type(relid: pg_sys::Oid, column: &str) -> Option<String> {
    Spi::get_one_with_args::<String>(
        "SELECT (SELECT t.typname::text FROM pg_catalog.pg_attribute a \
                 JOIN pg_catalog.pg_type t ON t.oid = a.atttypid \
                 WHERE a.attrelid = $1::oid AND a.attname = $2 AND a.attnum > 0 \
                 AND NOT a.attisdropped)",
        &[relid.into(), column.into()],
    )
    .expect("pg_pinecone: column type query failed")
}

/// `column`'s `attstorage` letter: `p` plain, `e` external, `m` main, `x` extended.
pub(crate) fn column_storage(relid: pg_sys::Oid, column: &str) -> Option<char> {
    Spi::get_one_with_args::<String>(
        "SELECT (SELECT a.attstorage::text FROM pg_catalog.pg_attribute a \
                 WHERE a.attrelid = $1::oid AND a.attname = $2 AND a.attnum > 0 \
                 AND NOT a.attisdropped)",
        &[relid.into(), column.into()],
    )
    .expect("pg_pinecone: column storage query failed")
    .and_then(|storage| storage.chars().next())
}

/// Whether `column`'s collation is deterministic, i.e. strings it calls equal are byte-identical
/// (a case-insensitive ICU collation is not). A type that takes no collation counts as
/// deterministic.
pub(crate) fn column_is_deterministic(relid: pg_sys::Oid, column: &str) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT coalesce((SELECT c.collisdeterministic FROM pg_catalog.pg_attribute a \
                          JOIN pg_catalog.pg_collation c ON c.oid = a.attcollation \
                          WHERE a.attrelid = $1::oid AND a.attname = $2 AND a.attnum > 0 \
                            AND NOT a.attisdropped), true)",
        &[relid.into(), column.into()],
    )
    .expect("pg_pinecone: column collation query failed")
    .unwrap_or(true)
}

/// The declared `atttypmod` of `column`, i.e. its `vector` dimension (`-1` when
/// undeclared, e.g. a bare `vector` column with no `(n)`). Panics if `column`
/// doesn't exist — callers must already have validated that via [`all_columns`]
/// or [`column_type`].
pub(crate) fn vector_dimension(relid: pg_sys::Oid, column: &str) -> i32 {
    Spi::get_one_with_args::<i32>(
        "SELECT a.atttypmod FROM pg_catalog.pg_attribute a \
         WHERE a.attrelid = $1::oid AND a.attname = $2 AND a.attnum > 0 AND NOT a.attisdropped",
        &[relid.into(), column.into()],
    )
    .expect("pg_pinecone: vector dimension query failed")
    .expect("pg_pinecone: column disappeared between its catalog lookup and the dimension check")
}

/// The table's primary-key columns, in key order (empty = no PK; more than one
/// element = a composite PK — both are `pinecone.track` autodetect errors).
pub(crate) fn primary_key_columns(relid: pg_sys::Oid) -> Vec<ColumnCatalog> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT a.attname::text AS name, a.attnum, t.typname::text AS type_name \
                 FROM pg_catalog.pg_constraint c \
                 JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS ck(attnum, ord) ON true \
                 JOIN pg_catalog.pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ck.attnum \
                 JOIN pg_catalog.pg_type t ON t.oid = a.atttypid \
                 WHERE c.conrelid = $1::oid AND c.contype = 'p' \
                 ORDER BY ck.ord",
                None,
                &[relid.into()],
            )
            .expect("pg_pinecone: primary key catalog query failed")
            .map(|row| ColumnCatalog {
                name: text(&row, "name"),
                attnum: required(&row, "attnum"),
                type_name: text(&row, "type_name"),
            })
            .collect()
    })
}

/// Whether `column` is covered by a single-column `UNIQUE` index or PRIMARY
/// KEY on `relid` — `pinecone.detach_storage`'s id-uniqueness gate
/// (never `force`-bypassable: duplicate ids collapse to one
/// remote vector, and a later `pinecone.attach_storage`/`recover_from_remote`
/// would fabricate data trying to write one remote value back to every
/// duplicate row). Every `PRIMARY KEY` is itself a unique index, so a single
/// `pg_index` query covers both without a separate `pg_constraint` lookup.
/// Deliberately excludes a multi-column unique index that merely *includes*
/// `column` (uniqueness of the tuple doesn't imply uniqueness of one column
/// alone) and a partial index (`indpred IS NOT NULL` — doesn't cover every
/// row). `(i.indkey::int2[])[0]` keeps `indkey`'s 0-based lower bound after
/// the cast, same convention as [`index_first_opclass_name`].
pub(crate) fn has_unique_single_column_index(relid: pg_sys::Oid, column: &str) -> bool {
    unique_single_column_index(relid, column, false)
}

/// [`has_unique_single_column_index`] for the document lane, whose ids must be unique within a
/// statement, not merely at commit: a `DEFERRABLE` constraint (`indimmediate` is false for any
/// of them) can leave duplicates in a statement's transition table.
pub(crate) fn has_immediate_unique_single_column_index(relid: pg_sys::Oid, column: &str) -> bool {
    unique_single_column_index(relid, column, true)
}

fn unique_single_column_index(relid: pg_sys::Oid, column: &str, immediate: bool) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT EXISTS ( \
             SELECT 1 FROM pg_catalog.pg_index i \
             JOIN pg_catalog.pg_attribute a \
               ON a.attrelid = i.indrelid AND a.attnum = (i.indkey::int2[])[0] \
             WHERE i.indrelid = $1::oid AND i.indisunique AND i.indisvalid \
               AND i.indpred IS NULL AND cardinality(i.indkey::int2[]) = 1 \
               AND (i.indimmediate OR NOT $3) \
               AND a.attname = $2)",
        &[relid.into(), column.into(), immediate.into()],
    )
    .expect("pg_pinecone: unique-index catalog query failed")
    .expect("pg_pinecone: EXISTS unexpectedly NULL")
}

/// The triggers of `relid` (user-defined, not internal) whose names are among `names`.
pub(crate) fn trigger_names_among(relid: pg_sys::Oid, names: &[&str]) -> Vec<String> {
    let names: Vec<String> = names.iter().map(|name| name.to_string()).collect();
    Spi::connect(|client| {
        client
            .select(
                "SELECT tgname::text AS tgname FROM pg_catalog.pg_trigger \
                 WHERE tgrelid = $1::oid AND NOT tgisinternal AND tgname::text = ANY($2) \
                 ORDER BY tgname",
                None,
                &[relid.into(), names.into()],
            )
            .expect("pg_pinecone: trigger catalog query failed")
            .map(|row| text(&row, "tgname"))
            .collect()
    })
}

/// Every distinct pgvector operator-class name attached to `attnum` by any index
/// on `relid` — across any access method (hnsw, ivfflat, pgvectorscale's
/// diskann, …). Opclass *names* (`vector_cosine_ops`/`vector_l2_ops`/
/// `vector_ip_ops`) are shared across AMs, unlike AM names, so this
/// deliberately doesn't filter by `pg_am`.
pub(crate) fn index_opclass_names(relid: pg_sys::Oid, attnum: i16) -> Vec<String> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT DISTINCT opc.opcname::text AS opcname \
                 FROM pg_catalog.pg_index idx \
                 JOIN LATERAL generate_subscripts(idx.indkey::int2[], 1) AS s ON true \
                 JOIN pg_catalog.pg_opclass opc ON opc.oid = (idx.indclass::oid[])[s] \
                 WHERE idx.indrelid = $1::oid AND (idx.indkey::int2[])[s] = $2::int2",
                None,
                &[relid.into(), attnum.into()],
            )
            .expect("pg_pinecone: index opclass catalog query failed")
            .map(|row| text(&row, "opcname"))
            .collect()
    })
}

/// The opclass name attached to `index_relid`'s first (and, for `pg_pinecone`
/// access-method indexes, only) key column — used by `am::ambuild` to derive
/// the metric from the exact opclass the `CREATE INDEX ... USING pinecone (col
/// <opclass>)` statement specified. Unlike [`index_opclass_names`] (heap/
/// attnum keyed, so ambiguous across multiple indexes on the same column),
/// this is keyed by the index's own oid, so it's unambiguous for the one
/// index `ambuild` is actually building.
pub(crate) fn index_first_opclass_name(index_relid: pg_sys::Oid) -> Option<String> {
    Spi::get_one_with_args::<String>(
        // `indclass` (an `oidvector`) keeps its 0-based lower bound even after
        // casting to a plain `oid[]` (same quirk `index_opclass_names`' own
        // `generate_subscripts(idx.indkey::int2[], 1)` works around) — so the
        // first column is subscript `[0]`, not `[1]`.
        "SELECT (SELECT opc.opcname::text FROM pg_catalog.pg_index i \
                 JOIN pg_catalog.pg_opclass opc ON opc.oid = (i.indclass::oid[])[0] \
                 WHERE i.indexrelid = $1::oid)",
        &[index_relid.into()],
    )
    .expect("pg_pinecone: index opclass query failed")
}

/// Whether `index_relid` currently names a valid (`pg_index.indisvalid`)
/// index — `false` both for a genuinely invalid one (e.g. a `CREATE INDEX
/// CONCURRENTLY` build that failed partway, left behind for the owner to drop
/// manually) and for an oid that isn't in `pg_index` at all anymore. Used by
/// `am::ambuild`'s second-index guard:
/// a *different*, still-valid pg_pinecone-AM index on the same table means a
/// genuine second index, not a rebuild; a different oid that's since gone
/// invalid means the opposite — a stale, abandoned registration a fresh
/// `CREATE INDEX` is entitled to take over.
pub(crate) fn index_is_valid(index_relid: pg_sys::Oid) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT COALESCE((SELECT indisvalid FROM pg_catalog.pg_index \
                 WHERE indexrelid = $1::oid), false)",
        &[index_relid.into()],
    )
    .expect("pg_pinecone: index validity query failed")
    .expect("pg_pinecone: COALESCE(..., false) is never NULL")
}

/// `schema.table` display text for a bare relid — [`display_name`]'s
/// counterpart for callers holding an oid rather than an open `PgRelation`
/// (falls back to `<dropped:OID>`, `status`'s own convention).
pub(crate) fn display_name_of(relid: pg_sys::Oid) -> String {
    resolve_table_name(relid)
        .map(|(schema, table)| format!("{schema}.{table}"))
        .unwrap_or_else(|| format!("<dropped:{relid}>"))
}

/// The heap table `index_relid` belongs to (`pg_index.indrelid`), or `None`
/// if it isn't a valid index oid at all — used by `lifecycle::promote_index`/
/// `demote_index`, which are handed the INDEX's own regclass and need to get
/// back to its table for registry/outbox bookkeeping.
pub(crate) fn index_heap_relid(index_relid: pg_sys::Oid) -> Option<pg_sys::Oid> {
    Spi::get_one_with_args::<pg_sys::Oid>(
        "SELECT (SELECT indrelid::oid FROM pg_catalog.pg_index WHERE indexrelid = $1::oid)",
        &[index_relid.into()],
    )
    .expect("pg_pinecone: index heap-relid query failed")
}

/// The oid of `relid`'s pg_pinecone access-method index, if it has one — used by
/// `lifecycle::set_engine` to find the physical index currently backing a
/// table's vector column, regardless of whether the caller handed it the
/// table or the index itself.
pub(crate) fn pg_pinecone_am_index_for_table(relid: pg_sys::Oid) -> Option<pg_sys::Oid> {
    Spi::get_one_with_args::<pg_sys::Oid>(
        "SELECT (SELECT c.oid::oid FROM pg_catalog.pg_index i \
                 JOIN pg_catalog.pg_class c ON c.oid = i.indexrelid \
                 JOIN pg_catalog.pg_am a ON a.oid = c.relam \
                 WHERE i.indrelid = $1::oid AND a.amname = 'pinecone')",
        &[relid.into()],
    )
    .expect("pg_pinecone: pg_pinecone-AM-index-for-table query failed")
}

/// Resolves a possibly-stale `relid` (the table may have been dropped since it
/// was tracked) to its current `(schema, table)` name, or `None` if it no
/// longer resolves. Never fails — used by `pinecone.status`, which must not either.
pub(crate) fn resolve_table_name(relid: pg_sys::Oid) -> Option<(String, String)> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT n.nspname::text AS schema_name, c.relname::text AS table_name \
                 FROM pg_catalog.pg_class c \
                 JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace \
                 WHERE c.oid = $1::oid",
                None,
                &[relid.into()],
            )
            .expect("pg_pinecone: relation name query failed")
            .next()
            .map(|row| (text(&row, "schema_name"), text(&row, "table_name")))
    })
}

/// Whether `current_user` has been granted `SELECT` on `relid`. Used by
/// `pinecone.status` to decide whether `count(*)` is safe to run — a registry
/// row survives a `REVOKE` on its table, and `pinecone.status` must never fail
/// (or error out) just because the current session no longer has access.
pub(crate) fn has_select_privilege(relid: pg_sys::Oid) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT pg_catalog.has_table_privilege(current_user, $1::oid, 'SELECT')",
        &[relid.into()],
    )
    .expect("pg_pinecone: has_table_privilege query failed")
    .expect("pg_pinecone: has_table_privilege unexpectedly NULL")
}

/// `pg_class.relkind` of `relid` (`'r'` table, `'p'` partitioned table, `'v'` view, ...), or
/// `None` if no such relation exists.
pub(crate) fn relkind(relid: pg_sys::Oid) -> Option<char> {
    Spi::get_one_with_args::<String>(
        "SELECT (SELECT relkind::text FROM pg_catalog.pg_class WHERE oid = $1::oid)",
        &[relid.into()],
    )
    .expect("pg_pinecone: relkind query failed")
    .and_then(|kind| kind.chars().next())
}

/// Whether `relid` has inheritance children or partitions (`pg_inherits`).
pub(crate) fn has_inheritance_children(relid: pg_sys::Oid) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_inherits WHERE inhparent = $1::oid)",
        &[relid.into()],
    )
    .expect("pg_pinecone: inheritance children query failed")
    .expect("pg_pinecone: EXISTS unexpectedly NULL")
}

/// Whether a declared document text column may have this type (`text`, `varchar`, `char(n)`).
pub(crate) fn is_text_column_type(typoid: pg_sys::Oid) -> bool {
    [pg_sys::TEXTOID, pg_sys::VARCHAROID, pg_sys::BPCHAROID].contains(&typoid)
}

/// Extracts a required, non-NULL column value, panicking (rather than
/// returning a wrong/empty result) if the query shape ever stops matching —
/// a bug in this module's own SQL, not a user-facing error.
fn required<T: FromDatum + IntoDatum>(row: &pgrx::spi::SpiHeapTupleData<'_>, name: &str) -> T {
    row.get_by_name::<T, _>(name)
        .unwrap_or_else(|e| panic!("pg_pinecone: catalog column {name} query failed: {e}"))
        .unwrap_or_else(|| panic!("pg_pinecone: catalog column {name} unexpectedly NULL"))
}

fn text(row: &pgrx::spi::SpiHeapTupleData<'_>, name: &str) -> String {
    required(row, name)
}
