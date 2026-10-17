//! `pg_pinecone` the access method: `CREATE INDEX ... USING pinecone (col
//! vector_cosine_ops)` makes a promoted-at-DDL-time table's remote index
//! transparently participate in `ORDER BY col <=> $1 LIMIT k` plans, no
//! `pinecone.promote`/sync-trigger involved.
//!
//! ## Division of labor
//! - [`ambuild`]: on a genuinely new pg_pinecone-AM table, registers it in
//!   `pinecone.tables` (`mode = 'remote'`, `synced_by = 'am'`), creates/adopts
//!   the matching Pinecone index (cloud/region from the `pinecone.am_cloud`/
//!   `pinecone.am_region` GUCs), and enqueues one `'backfill'` `pinecone.outbox`
//!   row (the background worker processes it). On a REINDEX/VACUUM FULL/CLUSTER re-run
//!   against an already-registered pg_pinecone-AM table, it instead refreshes the
//!   registry row in place and enqueues `'delete_all'` then `'backfill'` — the
//!   Pinecone index's identity never changes across a rebuild of the *local*
//!   index relation.
//! - [`aminsert`]: one `'upsert'` `pinecone.outbox` row per inserted tuple,
//!   `{id: "block:offset", values}` (plus `metadata: {col: v}` for an index with a
//!   filter column and a non-NULL key, see [`crate::am_filter`]) — enqueued as the AM handler function's
//!   *owner* (a `SetUserIdAndSecContext` switch, mirroring how
//!   `pinecone._sync_trigger`'s `SECURITY DEFINER` lets an unprivileged writer
//!   push under the stored key without a direct grant on `pinecone.outbox`).
//! - [`amgettuple`]: resolves the registry row, runs Pinecone `/query` with a
//!   growing `topK` (no `LIMIT` signal reaches the AM) and
//!   the scan's equality key on the filter column, if any, as the metadata filter
//!   `{col: {"$eq": v}}`, and
//!   returns matches in remote order. `xs_recheckorderby` is always `false`:
//!   the hard rule is that a remote score can never be reported as a
//!   lower bound of the true (locally recomputed) distance, so passing one
//!   through as `xs_orderbyvals` breaks `nodeIndexscan.c`'s reorder-queue
//!   invariant ("index returned tuples in wrong order"). Remote order is
//!   trusted outright, exactly like pgvector's own AMs.
//! - [`ambulkdelete`]: VACUUM's dead-tuple purge. With no local tuple
//!   store, dead TIDs can't be found by scanning this index's own relation —
//!   instead every remote id (`Client::list_vector_ids`) plus every
//!   still-queued `'upsert'` outbox id (a TID can die before its own push
//!   ever drains) is tested against the callback Postgres hands in; the dead
//!   ones are purged from `pinecone.outbox` and (the remotely-known ones)
//!   synchronously deleted from the remote index (`advisory_xact_lock`
//!   first, unconditionally).
//! - The `pg_pinecone_on_sql_drop` event trigger (below the opclass DDL):
//!   `DROP INDEX` (or a `DROP TABLE` that drops one implicitly) has no
//!   `IndexAmRoutine` teardown callback to hook, so this cleans up the
//!   orphaned `pinecone.tables`/`pinecone.outbox` rows a different way.
//! - A partitioned index (`pinecone.tables.partition_column`, see [`crate::am_partition`]) is one
//!   namespace per value of its second key: `aminsert` adds `"p"` (the drain registers the
//!   namespace once Pinecone took a write of it), a scan reads the namespace of its key (none:
//!   0A000) and rechecks every row, `ambulkdelete` walks the registered namespaces.

use std::ffi::{CString, c_void};

use pg_pinecone_client::{Client, DeleteVectorsRequest, ListVectorsRequest, QueryRequest};
use pgrx::{
    JsonB, PgBox, PgMemoryContexts,
    guc::{GucContext, GucFlags, GucRegistry, GucSetting},
    itemptr::item_pointer_get_both,
    pg_catalog::pg_proc::PgProc,
    prelude::*,
    spi::{Spi, quote_identifier, quote_qualified_identifier},
};
use serde_json::json;

use crate::{
    am_filter::{self, EqKey, KeyDecode},
    am_partition, backend, catalog, outbox, partition, partition_registry, promote, registry,
    secret,
};

/// `SHOW pinecone.am_enabled`'s default — the AM planner-visibility kill
/// switch: when off, [`amcostestimate`] reports
/// [`pg_sys::disable_cost`] so the planner never picks a `pg_pinecone`-AM index.
static AM_ENABLED: GucSetting<bool> = GucSetting::<bool>::new(true);
/// `SHOW pinecone.am_min_topk`'s default — the smallest `topK` [`amgettuple`]
/// ever sends Pinecone, and the size of its very first `/query` per scan. No `LIMIT` reaches an
/// index access method (`ExecSetTupleBound` stops above an index scan, `IndexScanDesc` has no
/// field for it), so this is the one number that decides whether `LIMIT 100` is one request or
/// two: 100 serves any `LIMIT` up to it in one, at about 0.3 to 0.9 ms more than 50 per query.
static AM_MIN_TOPK: GucSetting<i32> = GucSetting::<i32>::new(100);
/// `SHOW pinecone.am_max_topk`'s default — [`amgettuple`]'s over-fetch cap:
/// the last regrow is clamped to it, and a scan that needs more errors
/// (`iterative_scan = off`, which a remote index reads in `auto`) or ends
/// quietly (`relaxed_order`, `strict_order`).
static AM_MAX_TOPK: GucSetting<i32> = GucSetting::<i32>::new(10_000);

/// `pinecone.iterative_scan`: keep fetching past the first batch while the
/// executor wants more rows (native: the next lists, `local::scan`; remote: a
/// larger `topK`, [`amgettuple_remote`]). `strict_order` drops rows that would
/// arrive out of distance order (so it may return fewer rows), `relaxed_order`
/// keeps them.
///
/// `auto` (the boot value) resolves per engine, since one GUC serves both: a native scan returns
/// `off`'s first batch and goes on as `relaxed_order` only if the executor asks for more
/// (`local::scan`); a remote scan runs `off` ([`IterativeScan::for_remote`]), as before.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PostgresGucEnum)]
pub(crate) enum IterativeScan {
    #[name = c"auto"]
    Auto,
    #[name = c"off"]
    Off,
    #[name = c"relaxed_order"]
    RelaxedOrder,
    #[name = c"strict_order"]
    StrictOrder,
}
static ITERATIVE_SCAN: GucSetting<IterativeScan> =
    GucSetting::<IterativeScan>::new(IterativeScan::Auto);

impl IterativeScan {
    /// What a remote scan runs: `auto` is `off` there, so the remote default is unchanged.
    pub(crate) fn for_remote(self) -> IterativeScan {
        match self {
            IterativeScan::Auto => IterativeScan::Off,
            explicit => explicit,
        }
    }
}

/// `SHOW pinecone.max_scan_tuples`'s default — the most rows an iterative scan
/// fetches before it ends quietly (what counts differs per engine: see
/// `local::scan` and [`amgettuple_remote`]).
static MAX_SCAN_TUPLES: GucSetting<i32> = GucSetting::<i32>::new(20_000);

pub(crate) fn iterative_scan() -> IterativeScan {
    ITERATIVE_SCAN.get()
}

/// `pinecone.max_scan_tuples`, always >= 1.
pub(crate) fn max_scan_tuples() -> u64 {
    MAX_SCAN_TUPLES.get() as u64
}

/// `SHOW pinecone.am_cloud`'s default — the cloud provider [`ambuild`] passes
/// to `promote::create_or_adopt_index` for a fresh (not adopted) serverless
/// index, same role as `pinecone.promote`'s own `cloud` argument but as a GUC:
/// `CREATE INDEX ... USING pinecone` has no argument list of its own to carry one.
static AM_CLOUD: GucSetting<Option<CString>> = GucSetting::<Option<CString>>::new(Some(c"aws"));
/// `SHOW pinecone.am_region`'s default — see [`AM_CLOUD`].
static AM_REGION: GucSetting<Option<CString>> =
    GucSetting::<Option<CString>>::new(Some(c"us-east-1"));

/// `(pinecone.am_cloud, pinecone.am_region)` — the same GUCs `ambuild`'s remote
/// branch uses for a fresh serverless index, exposed for
/// `lifecycle::promote_index`, which creates a serverless index the
/// exact same way outside of `ambuild` itself.
pub(crate) fn am_cloud_region() -> (String, String) {
    (
        backend::guc_string(&AM_CLOUD),
        backend::guc_string(&AM_REGION),
    )
}

/// Registers this module's GUCs — called from the crate's `_PG_init`.
pub(crate) fn init_gucs() {
    GucRegistry::define_bool_guc(
        c"pinecone.am_enabled",
        c"Whether the planner may pick a pg_pinecone access-method index. When off, \
          amcostestimate reports disable_cost so plans always fall back to a seqscan.",
        c"",
        &AM_ENABLED,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"pinecone.am_min_topk",
        c"Smallest topK a pg_pinecone access-method index scan ever sends Pinecone \
          (also its first query's topK): a LIMIT up to it is one request.",
        c"",
        &AM_MIN_TOPK,
        1,
        1_000_000,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"pinecone.am_max_topk",
        c"Largest topK a remote-mode pg_pinecone index scan may request while \
          over-fetching past a selective WHERE filter: beyond it the scan errors \
          (iterative_scan = off, which a remote index reads in auto) or ends quietly \
          (relaxed_order, strict_order).",
        c"",
        &AM_MAX_TOPK,
        1,
        1_000_000,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_enum_guc(
        c"pinecone.iterative_scan",
        c"Whether a pg_pinecone index scan keeps fetching past its first batch while the \
          executor still wants rows: auto (a native index returns off's first batch with the \
          early stop and then goes on as relaxed_order; a remote index runs off), off (one batch), \
          relaxed_order (ordered per batch) or strict_order (ordered, but may return fewer \
          rows). Not used by engine => 'pgvector' indexes.",
        c"Prefer relaxed_order and re-sort its few rows in a CTE for exact order: strict_order \
          drops every row closer than the last one it returned and can rescore up to \
          max_scan_tuples rows for a handful of results.",
        &ITERATIVE_SCAN,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"pinecone.max_scan_tuples",
        c"Rows an iterative pg_pinecone index scan may fetch before it ends quietly: exact-rescored \
          rows (native engine: the first batch is never cut, a later slice is cut to what the \
          cap leaves) or rows requested from Pinecone (remote engine).",
        c"",
        &MAX_SCAN_TUPLES,
        1,
        i32::MAX,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_string_guc(
        c"pinecone.am_cloud",
        c"Cloud provider for the serverless Pinecone index a fresh CREATE INDEX ... \
          USING pinecone build creates.",
        c"",
        &AM_CLOUD,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_string_guc(
        c"pinecone.am_region",
        c"Region for the serverless Pinecone index a fresh CREATE INDEX ... USING \
          pg_pinecone build creates.",
        c"",
        &AM_REGION,
        GucContext::Userset,
        GucFlags::default(),
    );

    crate::local::options::init();
    crate::local::scan::init_gucs();
}

// ---------------------------------------------------------------------------
// Handler + DDL
// ---------------------------------------------------------------------------

// The handler takes a bare `internal` argument pgrx can't derive SQL for, so
// its CREATE FUNCTION/CREATE ACCESS METHOD text is supplied verbatim; the
// function is declared directly under the
// `pg_pinecone` schema (no `#[pg_schema]` nesting needed — that's a Rust-side
// convenience for auto-qualifying `#[pg_extern]`s, not a requirement here).
#[pg_extern(
    sql = r#"
CREATE FUNCTION pinecone._am_handler(internal) RETURNS index_am_handler
PARALLEL SAFE IMMUTABLE STRICT LANGUAGE c AS 'MODULE_PATHNAME', '@FUNCTION_NAME@';
CREATE ACCESS METHOD pinecone TYPE INDEX HANDLER pinecone._am_handler;
"#,
    requires = [pinecone]
)]
fn am_handler(_fcinfo: pg_sys::FunctionCallInfo) -> PgBox<pg_sys::IndexAmRoutine> {
    let mut r =
        unsafe { PgBox::<pg_sys::IndexAmRoutine>::alloc_node(pg_sys::NodeTag::T_IndexAmRoutine) };

    // Flags:
    // `amcanorderbyop` + `amoptionalkey` is what lets a bare `ORDER BY col
    // <=> $1 LIMIT k` (no WHERE at all) pick this AM up, first try.
    // `amcanmulticol = true`: the composite filter key — `CREATE INDEX
    // ... USING pinecone (embedding <opclass>, tenant_id)`; `ambuild` validates
    // the shape (vector column first, at most ONE filter column, of a
    // supported type). `amstrategies` stays 0 ("flexible" per the AM docs);
    // the filter opclasses in [`install_opclasses`] define strategy 1 =
    // equality, the only search strategy the AM understands.
    r.amstrategies = 0;
    r.amsupport = 0;
    r.amoptsprocnum = 0;
    r.amcanorder = false;
    r.amcanorderbyop = true;
    r.amcanbackward = false;
    r.amcanunique = false;
    r.amcanmulticol = true;
    r.amoptionalkey = true;
    r.amsearcharray = false;
    r.amsearchnulls = false;
    r.amstorage = false;
    r.amclusterable = false;
    r.ampredlocks = false;
    r.amcanparallel = false;
    r.amcanbuildparallel = false;
    r.amcaninclude = false;
    r.amusemaintenanceworkmem = false;
    r.amsummarizing = false;
    r.amparallelvacuumoptions = 0;
    r.amkeytype = pg_sys::InvalidOid;

    r.ambuild = Some(ambuild);
    r.ambuildempty = Some(ambuildempty);
    r.aminsert = Some(aminsert);
    r.aminsertcleanup = None;
    r.ambulkdelete = Some(ambulkdelete);
    r.amvacuumcleanup = Some(amvacuumcleanup);
    r.amcanreturn = None; // no index-only scans
    r.amcostestimate = Some(amcostestimate);
    r.amoptions = Some(amoptions);
    r.amproperty = None;
    r.ambuildphasename = None;
    r.amvalidate = Some(amvalidate);
    r.amadjustmembers = None;
    r.ambeginscan = Some(ambeginscan);
    r.amrescan = Some(amrescan);
    r.amgettuple = Some(amgettuple);
    r.amgetbitmap = None; // no bitmap scans
    r.amendscan = Some(amendscan);
    r.ammarkpos = None;
    r.amrestrpos = None;
    r.amestimateparallelscan = None;
    r.aminitparallelscan = None;
    r.amparallelrescan = None;

    r.into_pg_boxed()
}

/// Creates the AM-scoped `vector_{cosine,l2,ip}_ops` opclasses, plus
/// the DEFAULT equality opclasses for the one optional FILTER key column a
/// composite `USING pinecone (embedding <opclass>, tenant_id)` index carries:
/// `int2_ops`/`int4_ops`/`int8_ops` (one shared `integer_ops` operator
/// family, with the six cross-type `=` operators added loose — btree's own
/// pattern — so `int8col = 5` (an `int4` literal) still matches an index
/// qual) and `text_ops` (`varchar` columns resolve to it too, via binary
/// coercibility, exactly as they do to btree's `text_ops`). All are DEFAULT
/// for their types under this AM, so the filter column needs no explicit
/// opclass in the `CREATE INDEX`. Strategy 1 = equality is the only member
/// — range strategies are out of scope and the scan errors loudly on
/// any other strategy number reaching it.
///
/// The vector opclasses' `FOR TYPE`/operator references must be
/// schema-qualified with wherever pgvector is actually installed
/// (`catalog::pgvector_namespace`, same as `topk::topk_local` resolves it) —
/// unknown until install time, so this runs as a plain SQL statement invoked
/// from `extension_sql!` (below) rather than being baked into a static DDL
/// string.
///
/// Schema-qualified via custom `sql` (like [`am_handler`]) rather than
/// nesting inside the `pg_pinecone` `#[pg_schema] mod {}` in `lib.rs`: that
/// attribute only applies to an inline `mod {}` physically present at its
/// declaration site (a rustc limitation — see the `pg_pinecone` module's own
/// rustdoc), which this file, being its own module, isn't.
///
/// pgvector is an optional dependency everywhere else in pg_pinecone (every
/// other call site checks for it lazily, at call time — see
/// `catalog::pgvector_namespace`'s callers) and this `extension_sql!`-invoked
/// call runs unconditionally during `CREATE EXTENSION pg_pinecone` itself, so it
/// must keep that invariant: a missing pgvector degrades to a `WARNING` and a
/// no-op (skipping opclass creation), never an `ERROR` — otherwise installing
/// `pg_pinecone` before `pg_pinecone vector` would newly become impossible, for every
/// user, regardless of whether they ever intend to use the access method.
/// The remedy (`CREATE EXTENSION vector; SELECT
/// pinecone._install_opclasses();`) is repeatable on demand once vector is
/// installed, and an event trigger runs it when `vector` is created after
/// pg_pinecone (the order `pg_restore` creates them in). Only what is missing is
/// created, so a second call is a no-op. What a call creates outside the
/// extension's own script joins the extension: a dump leaves it out and a
/// restore creates it again.
#[pg_extern(
    sql = "CREATE FUNCTION pinecone._install_opclasses() RETURNS void LANGUAGE c \
           AS 'MODULE_PATHNAME', '@FUNCTION_NAME@';",
    requires = [pinecone, am_handler]
)]
fn install_opclasses() {
    let mut made: Vec<&str> = Vec::new();
    let Ok(ns) = catalog::pgvector_namespace() else {
        warning!(
            "pg_pinecone: pgvector is not installed — the pg_pinecone access method's opclasses were \
             not created; once it is, run CREATE EXTENSION vector; SELECT \
             pinecone._install_opclasses(); before CREATE INDEX ... USING pinecone"
        );
        return;
    };
    let vector_type = quote_qualified_identifier(ns.as_str(), "vector");
    let ns_ident = quote_identifier(&ns);

    for (opclass, op) in [
        ("vector_cosine_ops", "<=>"),
        ("vector_l2_ops", "<->"),
        ("vector_ip_ops", "<#>"),
    ] {
        if opclass_exists(opclass) {
            continue;
        }
        Spi::run(&format!(
            "CREATE OPERATOR CLASS {opclass} FOR TYPE {vector_type} USING pinecone AS \
             OPERATOR 1 {ns_ident}.{op} ({vector_type}, {vector_type}) FOR ORDER BY float_ops"
        ))
        .unwrap_or_else(|e| error!("pg_pinecone: failed to create operator class {opclass}: {e}"));
        made.push(opclass);
    }

    // Filter-column opclasses — pg_catalog types/operators only, no
    // pgvector qualification needed. One family for the three int widths so
    // the cross-type `=` operators below are usable as index quals.
    let new_family = !opfamily_exists("integer_ops");
    if new_family {
        Spi::run("CREATE OPERATOR FAMILY integer_ops USING pinecone").unwrap_or_else(|e| {
            error!("pg_pinecone: failed to create operator family integer_ops: {e}")
        });
        made.push("integer_ops");
    }
    for (opclass, ty) in [
        ("int2_ops", "int2"),
        ("int4_ops", "int4"),
        ("int8_ops", "int8"),
    ] {
        if opclass_exists(opclass) {
            continue;
        }
        Spi::run(&format!(
            "CREATE OPERATOR CLASS {opclass} DEFAULT FOR TYPE {ty} USING pinecone \
             FAMILY integer_ops AS OPERATOR 1 = ({ty}, {ty})"
        ))
        .unwrap_or_else(|e| error!("pg_pinecone: failed to create operator class {opclass}: {e}"));
        made.push(opclass);
    }
    if new_family {
        Spi::run(
            "ALTER OPERATOR FAMILY integer_ops USING pinecone ADD \
             OPERATOR 1 = (int2, int4), OPERATOR 1 = (int2, int8), \
             OPERATOR 1 = (int4, int2), OPERATOR 1 = (int4, int8), \
             OPERATOR 1 = (int8, int2), OPERATOR 1 = (int8, int4)",
        )
        .unwrap_or_else(|e| error!("pg_pinecone: failed to add cross-type integer operators: {e}"));
    }
    if !opclass_exists("text_ops") {
        Spi::run(
            "CREATE OPERATOR CLASS text_ops DEFAULT FOR TYPE text USING pinecone AS \
             OPERATOR 1 = (text, text)",
        )
        .unwrap_or_else(|e| error!("pg_pinecone: failed to create operator class text_ops: {e}"));
        made.push("text_ops");
    }
    join_extension(&made);
}

fn opclass_exists(name: &str) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_opclass c \
         JOIN pg_catalog.pg_am a ON a.oid = c.opcmethod \
         WHERE a.amname = 'pinecone' AND c.opcname = $1 \
           AND pg_catalog.pg_opclass_is_visible(c.oid))",
        &[name.into()],
    )
    .expect("pg_pinecone: operator class lookup failed")
    .unwrap_or(false)
}

fn opfamily_exists(name: &str) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_opfamily f \
         JOIN pg_catalog.pg_am a ON a.oid = f.opfmethod \
         WHERE a.amname = 'pinecone' AND f.opfname = $1 \
           AND pg_catalog.pg_opfamily_is_visible(f.oid))",
        &[name.into()],
    )
    .expect("pg_pinecone: operator family lookup failed")
    .unwrap_or(false)
}

/// Makes the operator classes and families named `names` members of the extension. The
/// extension's own script does that for what it creates; a call from anywhere else (the event
/// trigger, the remedy by hand) has to, or a dump would carry them and a restore would meet
/// them twice.
fn join_extension(names: &[&str]) {
    if names.is_empty() || unsafe { pg_sys::creating_extension } {
        return;
    }
    let names: Vec<String> = names.iter().map(|name| name.to_string()).collect();
    let loose: Vec<(String, String, String)> = Spi::connect(|client| {
        client
            .select(
                "SELECT 'OPERATOR CLASS' AS kind, n.nspname::text AS schema, c.opcname::text AS name \
                 FROM pg_catalog.pg_opclass c \
                 JOIN pg_catalog.pg_am a ON a.oid = c.opcmethod \
                 JOIN pg_catalog.pg_namespace n ON n.oid = c.opcnamespace \
                 WHERE a.amname = 'pinecone' AND c.opcname = ANY($1) \
                   AND pg_catalog.pg_opclass_is_visible(c.oid) \
                   AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_depend d \
                                   WHERE d.classid = 'pg_catalog.pg_opclass'::regclass \
                                     AND d.objid = c.oid AND d.deptype = 'e') \
                 UNION ALL \
                 SELECT 'OPERATOR FAMILY', n.nspname::text, f.opfname::text \
                 FROM pg_catalog.pg_opfamily f \
                 JOIN pg_catalog.pg_am a ON a.oid = f.opfmethod \
                 JOIN pg_catalog.pg_namespace n ON n.oid = f.opfnamespace \
                 WHERE a.amname = 'pinecone' AND f.opfname = ANY($1) \
                   AND pg_catalog.pg_opfamily_is_visible(f.oid) \
                   AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_depend d \
                                   WHERE d.classid = 'pg_catalog.pg_opfamily'::regclass \
                                     AND d.objid = f.oid AND d.deptype = 'e')",
                None,
                &[names.into()],
            )
            .expect("pg_pinecone: operator class membership query failed")
            .map(|row| {
                let text = |column: &str| {
                    row.get_by_name::<String, _>(column)
                        .expect("pg_pinecone: operator class membership column")
                        .expect("pg_pinecone: operator class membership column is NOT NULL")
                };
                (text("kind"), text("schema"), text("name"))
            })
            .collect()
    });
    for (kind, schema, name) in loose {
        let object = quote_qualified_identifier(&schema, &name);
        Spi::run(&format!(
            "ALTER EXTENSION pg_pinecone ADD {kind} {object} USING pinecone"
        ))
        .unwrap_or_else(|e| {
            error!("pg_pinecone: failed to add {kind} {object} to the extension: {e}")
        });
    }
}

// An event trigger runs the opclass installation when `vector` is created after pg_pinecone, the
// order `pg_restore` uses (extensions in name order): the dump leaves the opclasses out (they
// are extension members) and the restore of an index that names one fails without them.
// `SECURITY DEFINER` because `CREATE EXTENSION vector` is open to a database owner and
// `CREATE OPERATOR CLASS` is not; a failure is a WARNING, never a failed `CREATE EXTENSION
// vector`. The opclasses go where `CREATE EXTENSION pg_pinecone` put them: its own schema.
extension_sql!(
    r#"
    CREATE FUNCTION pinecone._on_create_vector() RETURNS event_trigger
    LANGUAGE plpgsql
    SECURITY DEFINER
    SET search_path = pg_catalog, pg_temp
    AS $$
    DECLARE
        v_schema text;
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_event_trigger_ddl_commands()
                       WHERE command_tag = 'CREATE EXTENSION' AND object_identity = 'vector') THEN
            RETURN;
        END IF;
        SELECT n.nspname INTO v_schema FROM pg_extension e
            JOIN pg_namespace n ON n.oid = e.extnamespace WHERE e.extname = 'pg_pinecone';
        PERFORM set_config('search_path', quote_ident(v_schema) || ', pg_temp', true);
        BEGIN
            PERFORM pinecone._install_opclasses();
        EXCEPTION WHEN others THEN
            RAISE WARNING 'pg_pinecone: could not create the access method opclasses (%) - run SELECT pinecone._install_opclasses() as a superuser', SQLERRM;
        END;
    END;
    $$;
    REVOKE EXECUTE ON FUNCTION pinecone._on_create_vector() FROM PUBLIC;
    CREATE EVENT TRIGGER pg_pinecone_on_create_vector ON ddl_command_end WHEN TAG IN ('CREATE EXTENSION')
        EXECUTE FUNCTION pinecone._on_create_vector();
    "#,
    name = "pg_pinecone_am_create_vector",
    requires = [pinecone, install_opclasses],
);

extension_sql!(
    "SELECT pinecone._install_opclasses();",
    name = "pg_pinecone_am_opclasses",
    requires = [pinecone, am_handler, install_opclasses],
);

// ---------------------------------------------------------------------------
// DROP INDEX / DROP TABLE cleanup
// ---------------------------------------------------------------------------

// `pinecone._on_sql_drop`/`pg_pinecone_on_sql_drop`: no `am`-specific "index
// destroyed" callback exists in `IndexAmRoutine` (there's `ambuild`/
// `ambuildempty` for creation, nothing symmetric for teardown) — a
// `DROP INDEX` on a pg_pinecone-AM index (or a `DROP TABLE` that drops one
// implicitly) would otherwise leave its `pinecone.tables` row and any
// still-queued `pinecone.outbox` rows behind forever, silently orphaned.
// A `sql_drop` event trigger is Postgres's own mechanism for exactly this:
// it fires once per statement, after the drop, with
// `pg_event_trigger_dropped_objects()` listing every object actually
// removed — including ones dropped only as a consequence of the command
// (verified empirically: `DROP TABLE` on an AM table reports its index as
// its own `object_type = 'index'` row, identical in shape to a bare
// `DROP INDEX`), so one handler covers both without special-casing either.
//
// By the time `sql_drop` fires, the dropped index's `pg_class` row is
// already gone — there is no `relam` left to look up, which is why
// `registry.rs`'s `am_index_relid` column exists at all: `ambuild` stores
// the AM index's own oid there at creation time, so this handler only needs
// `pg_event_trigger_dropped_objects().objid` (which Postgres reports
// regardless of whether the catalog row survives), never a live catalog
// lookup, to find the matching registry row.
//
// **`SECURITY DEFINER`, like `pinecone._sync_trigger`** (this used to
// be plain `SECURITY INVOKER`, which broke on empirical testing: an event
// trigger fires for *every* `sql_drop` in the database, so an unprivileged
// role dropping its own, completely unrelated index/table hit this function's
// `SELECT`/`DELETE`s against `pinecone.tables`/`pinecone.outbox` too — and
// since neither grants `PUBLIC`
// anything, that role got a bare "permission denied for table tables" trying
// to drop objects it fully owns and that have nothing to do with pg_pinecone).
// `SECURITY DEFINER` fixes that by running this function's body as its
// owner (whoever installed the extension) regardless of caller — safe
// because, as `SET search_path` below already keeps true for the
// identifier-injection angle, the body is static SQL with no dynamic
// identifiers, and the only thing it acts on
// (`pg_event_trigger_dropped_objects().objid`) is system-supplied, not
// caller-controlled: an unrelated drop's `objid` simply matches no
// `am_index_relid` row (`IF FOUND` false), a pure no-op, so a non-owner
// triggering this on their own objects can't use it to read or delete
// another table's registry/outbox rows. `EXECUTE` is still `REVOKE`d from
// `PUBLIC` below (event-trigger functions can't be called directly via SQL
// regardless — `event_trigger` is a non-callable pseudo-type — but this
// matches `_sync_trigger`'s own belt-and-suspenders `REVOKE`).
//
// Runs on every `sql_drop` in the database, not just ones touching a
// pg_pinecone-AM index (event triggers have no narrower granularity than an
// event name) — one extra `SELECT ... FOR EACH dropped object` per
// unrelated `DROP` statement in the whole cluster is the accepted, inherent
// cost of this mechanism.
extension_sql!(
    r#"
    CREATE FUNCTION pinecone._on_sql_drop() RETURNS event_trigger
    LANGUAGE plpgsql
    SECURITY DEFINER
    SET search_path = pg_catalog, pg_temp
    AS $$
    DECLARE
        obj record;
        v_relid oid;
    BEGIN
        -- A dropped table's partition namespaces are forgotten (Pinecone is left alone), whichever
        -- lane synced it; the AM index below takes its registry row with it. A trigger-lane row stays
        -- as `<dropped:OID>` for `pinecone.status()`, unless the table was partitioned: its row
        -- claims a namespace family that nothing could release, and the namespaces are already gone.
        FOR obj IN SELECT * FROM pg_event_trigger_dropped_objects() WHERE object_type = 'table' LOOP
            DELETE FROM pinecone.partitions WHERE relid::oid = obj.objid AND lane = 'vector';
            IF EXISTS (SELECT 1 FROM pinecone.tables WHERE relid::oid = obj.objid
                       AND synced_by = 'trigger' AND partition_column IS NOT NULL) THEN
                DELETE FROM pinecone.outbox WHERE relid::oid = obj.objid;
                DELETE FROM pinecone.tables WHERE relid::oid = obj.objid;
            END IF;
        END LOOP;
        FOR obj IN SELECT * FROM pg_event_trigger_dropped_objects() WHERE object_type = 'index' LOOP
            SELECT relid::oid INTO v_relid FROM pinecone.tables WHERE am_index_relid = obj.objid;
            IF FOUND THEN
                DELETE FROM pinecone.outbox WHERE relid::oid = v_relid;
                DELETE FROM pinecone.partitions WHERE relid::oid = v_relid AND lane = 'vector';
                DELETE FROM pinecone.tables WHERE am_index_relid = obj.objid;
            END IF;
            -- `pinecone._am_engine` (engine stickiness across
            -- REINDEX): keyed by the index's own oid directly, unconditional
            -- (no matching `pinecone.tables` row needed — local-mode indexes
            -- never have one), a no-op DELETE for anything not tracked here.
            DELETE FROM pinecone._am_engine WHERE index_relid = obj.objid;
        END LOOP;
    END;
    $$;
    REVOKE EXECUTE ON FUNCTION pinecone._on_sql_drop() FROM PUBLIC;
    CREATE EVENT TRIGGER pg_pinecone_on_sql_drop ON sql_drop
        EXECUTE FUNCTION pinecone._on_sql_drop();
    "#,
    name = "pg_pinecone_am_sql_drop",
    requires = [
        pinecone,
        "pg_pinecone_tables",
        "pg_pinecone_outbox",
        "pg_pinecone_am_engine",
        "pg_pinecone_partitions"
    ],
);

// ---------------------------------------------------------------------------
// pgvector `vector` datum <-> Vec<f32>
// ---------------------------------------------------------------------------

/// Parses a pgvector `vector` datum: varlena header (4B), int16 dim, int16
/// unused, then `dim` `f32`s.
pub(crate) unsafe fn parse_vector(d: pg_sys::Datum) -> Vec<f32> {
    unsafe {
        let v = pg_sys::pg_detoast_datum(d.cast_mut_ptr::<pg_sys::varlena>());
        let base = v as *const u8;
        let dim = *(base.add(4) as *const i16) as usize;
        let data = base.add(8) as *const f32;
        (0..dim).map(|i| *data.add(i)).collect()
    }
}

// ---------------------------------------------------------------------------
// TID <-> Pinecone id ("block:offset")
// ---------------------------------------------------------------------------

pub(crate) fn tid_to_id(tid: pg_sys::ItemPointerData) -> String {
    let (block, offset) = item_pointer_get_both(tid);
    format!("{block}:{offset}")
}

/// Parses a remote match id back into an `ItemPointerData`, rejecting
/// anything that isn't exactly `"<u32 block>:<u16 offset>"` — a malformed id
/// can only mean the remote index holds data pg_pinecone itself didn't write.
///
/// Beyond the parse itself, `block`/`offset` are range-checked against the
/// two sentinel values Postgres gives special meaning to: `block ==
/// InvalidBlockNumber` (`0xFFFFFFFF`, i.e. `P_NEW`) is `ReadBufferExtended`'s
/// "extend the relation" signal — handing that block number down to the
/// executor's own tuple fetch (via `xs_heaptid`) would silently *grow the
/// heap by one page* while looking for a tuple that was never there, instead
/// of erroring; `offset == InvalidOffsetNumber` (`0`) or `offset >
/// MaxOffsetNumber` can never be a real tuple's line pointer on any page.
fn id_to_tid(id: &str) -> Result<pg_sys::ItemPointerData, String> {
    let (block_str, offset_str) = id
        .split_once(':')
        .ok_or_else(|| format!("malformed remote id '{id}' — expected 'block:offset'"))?;
    let block: pg_sys::BlockNumber = block_str
        .parse()
        .map_err(|_| format!("malformed remote id '{id}' — bad block number"))?;
    let offset: pg_sys::OffsetNumber = offset_str
        .parse()
        .map_err(|_| format!("malformed remote id '{id}' — bad offset"))?;
    if block == pg_sys::InvalidBlockNumber {
        return Err(format!("malformed remote id '{id}' — bad block number"));
    }
    if offset == pg_sys::InvalidOffsetNumber || offset > pg_sys::MaxOffsetNumber {
        return Err(format!("malformed remote id '{id}' — bad offset"));
    }
    let mut tid = pg_sys::ItemPointerData {
        ip_blkid: pg_sys::BlockIdData { bi_hi: 0, bi_lo: 0 },
        ip_posid: 0,
    };
    pgrx::itemptr::item_pointer_set_all(&mut tid, block, offset);
    Ok(tid)
}

// ---------------------------------------------------------------------------
// SECURITY DEFINER-style privilege switch for aminsert's enqueue
// ---------------------------------------------------------------------------

/// Switches the executing role to `owner` and restricts `search_path` to
/// `pg_catalog, pg_temp` (via [`pg_sys::RestrictSearchPath`]) for as long as
/// this guard lives — the same two things `fmgr_security_definer` does for a
/// SQL `SECURITY DEFINER` function with a `SET search_path` clause, needed
/// here because `aminsert` is a bare C callback, not an SQL-callable
/// function of its own that could declare either. Without the `search_path`
/// half, a writer with only `INSERT` on the table (no grant on
/// `pinecone.outbox`/`pinecone.tables` at all) could plant a same-named
/// `oid`/`regclass`-shadowing type in a schema of their own ahead of
/// `pg_catalog` on their session's `search_path`, and have it resolve during
/// the elevated statements below instead of the real system catalog type —
/// the classic `SECURITY DEFINER` search_path attack. `RestrictSearchPath`
/// pushes the new value with `GUC_ACTION_SAVE`, so — regardless of whether
/// [`Drop`] later calls `AtEOXact_GUC` with `isCommit = true` or `false` —
/// Postgres always restores the prior value; using `false` uniformly (rather
/// than mirroring `fmgr_security_definer`'s own `true`-on-success-only call)
/// means one code path serves both a normal return and a Rust-panic unwind.
///
/// Restoring in `Drop` (rather than an explicit "restore" call at the end of
/// [`enqueue_upsert`]) means a Postgres `ERROR` raised while switched — which
/// pgrx surfaces to Rust as a stack-unwinding panic, not a `longjmp` that
/// skips destructors — still restores the caller's real identity and
/// `search_path`.
struct SecurityDefinerGuard {
    saved_uid: pg_sys::Oid,
    saved_sec_context: i32,
    guc_nestlevel: i32,
}

impl SecurityDefinerGuard {
    /// # Safety
    /// Must be called from within a `#[pg_guard]`-wrapped callback running on
    /// a live backend (i.e. exactly `aminsert`'s calling context).
    unsafe fn switch_to(owner: pg_sys::Oid) -> Self {
        unsafe {
            let mut saved_uid = pg_sys::Oid::INVALID;
            let mut saved_sec_context = 0i32;
            pg_sys::GetUserIdAndSecContext(&mut saved_uid, &mut saved_sec_context);
            pg_sys::SetUserIdAndSecContext(
                owner,
                saved_sec_context | pg_sys::SECURITY_LOCAL_USERID_CHANGE as i32,
            );
            let guc_nestlevel = pg_sys::NewGUCNestLevel();
            pg_sys::RestrictSearchPath();
            Self {
                saved_uid,
                saved_sec_context,
                guc_nestlevel,
            }
        }
    }
}

impl Drop for SecurityDefinerGuard {
    fn drop(&mut self) {
        unsafe {
            pg_sys::AtEOXact_GUC(false, self.guc_nestlevel);
            pg_sys::SetUserIdAndSecContext(self.saved_uid, self.saved_sec_context);
        }
    }
}

/// The owning role of `index`'s access-method handler function, resolved
/// straight from the open relcache/catalog caches — `index`'s own `rd_rel
/// ->relam` (the pg_pinecone access method's oid) -> `pg_am.amhandler` (via
/// `SearchSysCache1`) -> `pg_proc.proowner` — rather than the SPI query this
/// used to be (`'pinecone._am_handler(internal)'::regprocedure`). That text
/// cast has to parse an *unqualified* `internal` type name to interpret the
/// signature in parens, which — like any unqualified type-name reference —
/// is resolved against the caller's own `search_path`; a malicious writer
/// could shadow it and, depending on what the parser picks, misresolve which
/// function's owner gets impersonated below. Reading `relam` out of the
/// already-open index relation and following it through `SearchSysCache1`
/// (a lookup by oid, not by name) needs no parsing, and hence no
/// `search_path`, at all.
unsafe fn am_handler_owner(index: pg_sys::Relation) -> pg_sys::Oid {
    unsafe {
        let relam = (*(*index).rd_rel).relam;
        let am_tup = pg_sys::SearchSysCache1(
            pg_sys::SysCacheIdentifier::AMOID as i32,
            relam.into_datum().expect("pg_sys::Oid is never SQL NULL"),
        );
        if am_tup.is_null() {
            error!(
                "pg_pinecone: no pg_am entry for access method {relam:?} — should be unreachable \
                 from inside that access method's own aminsert"
            );
        }
        let amhandler =
            (*pg_sys::heap_tuple_get_struct::<pg_sys::FormData_pg_am>(am_tup)).amhandler;
        pg_sys::ReleaseSysCache(am_tup);

        PgProc::new(amhandler)
            .unwrap_or_else(|| {
                error!(
                    "pg_pinecone: no pg_proc entry for am handler {amhandler:?} — should be \
                     unreachable"
                )
            })
            .proowner()
    }
}

/// Inserts one `pinecone.outbox` row. `relid`'s bound parameter is typed
/// `oid` (`IntoDatum`'s default for `pg_sys::Oid`), so `pinecone.outbox
/// .relid`'s `regclass` target column is reached through Postgres's own
/// implicit assignment-cast machinery — a direct `pg_cast` lookup keyed by
/// the two types' oids — rather than the previous `$1::oid::regclass`: an
/// unqualified *textual* cast, which (like [`am_handler_owner`]'s old SPI
/// query) is resolved by *type name* via the caller's `search_path`.
/// Matters for [`enqueue_upsert`], which runs this under
/// [`SecurityDefinerGuard`]'s elevated identity: a malicious writer could
/// otherwise shadow `oid`/`regclass` with same-named types in a schema of
/// their own ahead of `pg_catalog` on search_path and have the cast
/// expression resolve there instead. `payload = None` stores SQL `NULL`
/// (matching `pinecone._sync_trigger`'s own `'delete_all'` convention), not a
/// JSON `null` value.
pub(crate) fn insert_outbox_row(relid: pg_sys::Oid, op: &str, payload: Option<JsonB>) {
    Spi::run_with_args(
        "INSERT INTO pinecone.outbox (relid, op, payload) VALUES ($1, $2, $3)",
        &[relid.into(), op.into(), payload.into()],
    )
    .expect("pg_pinecone: access-method outbox enqueue failed");
}

/// Enqueues one `'upsert'` `pinecone.outbox` row for `(relid, id, values)` — `partition` is the
/// canonical value of a partitioned index's key, which the drain turns into the namespace. The
/// caller holds the [`SecurityDefinerGuard`] (the same "structural provenance, no SQL-visible
/// enqueue function" model: `relid` comes from the heap
/// relation `aminsert` was called for, never user input).
fn enqueue_upsert(
    relid: pg_sys::Oid,
    id: &str,
    values: &[f32],
    metadata: Option<serde_json::Value>,
    partition: Option<&str>,
) {
    let mut item = json!({ "id": id, "values": values });
    if let Some(metadata) = metadata {
        item["metadata"] = metadata;
    }
    if let Some(partition) = partition {
        item["p"] = json!(partition);
    }
    insert_outbox_row(relid, "upsert", Some(JsonB(json!([item]))));
}

// ---------------------------------------------------------------------------
// ambuild / ambuildempty
// ---------------------------------------------------------------------------

/// Maps the exact opclass a `CREATE INDEX ... USING pinecone (col <opclass>)`
/// statement named to a registry metric — unlike `track::resolve_metric`
/// (heap/attnum keyed, ambiguous across multiple indexes), `ambuild` always
/// knows precisely which opclass its own index uses
/// (`catalog::index_first_opclass_name`).
fn metric_for_opclass(opclass: &str) -> String {
    match opclass {
        "vector_cosine_ops" => "cosine",
        "vector_l2_ops" => "l2",
        "vector_ip_ops" => "ip",
        other => error!(
            "pg_pinecone: operator class '{other}' is not supported by the pg_pinecone access method — \
             use vector_cosine_ops, vector_l2_ops, or vector_ip_ops"
        ),
    }
    .to_string()
}

// ---------------------------------------------------------------------------
// Engine stickiness — `pinecone._am_engine`
// ---------------------------------------------------------------------------

// Durably records which engine `mode=auto` resolved a given pg_pinecone-AM index
// oid to, so a later `REINDEX`/`VACUUM FULL`/`CLUSTER` can keep it instead
// of blindly re-running `resolve_mode`'s heuristic against whatever the
// DB's global state (`secret::has_stored_api_key`/`registry::lookup`)
// happens to be *at rebuild time* — the "engine switch = ONE
// explicit verb" invariant, which a plain re-evaluation silently violates
// (a stored key appearing/disappearing DB-wide would otherwise flip the
// DEFAULT for every `mode=auto` table, local-mode index included).
//
// This can't just be the index's own metapage: by the time `ambuild` runs
// for a `REINDEX`, Postgres has already called `RelationSetNewRelfilenumber`
// on the index, so the OLD metapage bytes are gone before this AM ever sees
// the rebuild (`index.c`'s `reindex_index`) — there is no "read the
// existing index's own pages" available at that point, for either mode.
// A tiny side table keyed by the index's own oid (stable across a same-oid
// rebuild; a `CREATE INDEX CONCURRENTLY`/`REINDEX CONCURRENTLY` builds a
// genuinely new oid instead, but those are already rejected outright for
// local mode — see `local::vacuum`'s module docstring — and were already
// rejected for remote mode before this fix, per `ambuild`'s own
// "second, DIFFERENT-oid index" rustdoc) is the simplest durable anchor
// that survives exactly the rebuilds this needs to survive.
//
// Deliberately NOT folded into `pinecone.tables`: that table's `mode='local'`
// CHECK value already exists but is unused by design (`am_auto_mode_
// defaults_to_local_with_no_api_key`'s own test asserts a local-mode index
// creates no `pinecone.tables` row at all) — `pinecone.status()`/the outbox
// worker/`pinecone.promote` all read that table assuming every row is a real,
// remote-tracked table, and auditing every one of those call sites to
// tolerate an inert local-mode row was more risk than a second, single-
// purpose table with no other readers at all.
extension_sql!(
    "CREATE TABLE pinecone._am_engine (
        index_relid oid PRIMARY KEY,
        mode text NOT NULL CHECK (mode IN ('local', 'remote'))
    );",
    name = "pg_pinecone_am_engine",
    requires = [pinecone],
);

fn engine_mode_str(mode: crate::local::metapage::Mode) -> &'static str {
    match mode {
        crate::local::metapage::Mode::Local => "local",
        crate::local::metapage::Mode::Remote => "remote",
        crate::local::metapage::Mode::Promoting
        | crate::local::metapage::Mode::Detached
        | crate::local::metapage::Mode::Demoting => {
            error!(
                "pg_pinecone: cannot persist a non-local/remote engine mode — should be unreachable"
            )
        }
    }
}

/// Upserts `index_relid`'s resolved engine — called every time `resolve_mode`
/// settles on a value, explicit or auto-first-resolution alike, so an
/// explicit `WITH (mode=...)` (or `ALTER INDEX ... SET (mode=...)` followed
/// by `REINDEX`) always overwrites whatever was stored before, matching "an
/// explicit ask always wins, including as the new sticky value going
/// forward".
pub(crate) fn persist_engine_mode(index_relid: pg_sys::Oid, mode: crate::local::metapage::Mode) {
    Spi::run_with_args(
        "INSERT INTO pinecone._am_engine (index_relid, mode) VALUES ($1, $2) \
         ON CONFLICT (index_relid) DO UPDATE SET mode = EXCLUDED.mode",
        &[index_relid.into(), engine_mode_str(mode).into()],
    )
    .expect("pg_pinecone: persisting resolved engine mode failed");
}

/// Reads back `index_relid`'s previously-persisted engine, if any (`None`
/// for a brand-new index oid this has never resolved before).
fn lookup_engine_mode(index_relid: pg_sys::Oid) -> Option<crate::local::metapage::Mode> {
    // A scalar subquery (NULL on no match), not a bare `SELECT ... WHERE
    // ...` — `Spi::get_one_with_args` requires exactly one result row and
    // errors (`InvalidPosition`) on zero, which a never-before-seen index
    // oid (the common case for a fresh `CREATE INDEX`) always is; see
    // `catalog::column_type`'s identical rustdoc for this same pattern.
    let mode: Option<String> = Spi::get_one_with_args(
        "SELECT (SELECT mode FROM pinecone._am_engine WHERE index_relid = $1)",
        &[index_relid.into()],
    )
    .expect("pg_pinecone: engine-mode lookup failed");
    mode.map(|m| match m.as_str() {
        "local" => crate::local::metapage::Mode::Local,
        "remote" => crate::local::metapage::Mode::Remote,
        other => error!("pg_pinecone: corrupt pinecone._am_engine row: mode = '{other}'"),
    })
}

/// Resolves the `WITH (mode=...)` reloption to a concrete engine
/// (the least-surprising rule) — **and makes that resolution sticky across
/// rebuilds** (see [`persist_engine_mode`]'s own rustdoc for why a rebuild
/// can't just re-run the heuristic below). An explicit `local`/`remote` reloption value
/// always wins outright, persisting itself as the new sticky choice; `auto`
/// (the reloption default — `local::options::Mode::Auto`) first consults
/// `index_relid`'s stored resolution and, if one exists, keeps it — logging
/// a `WARNING` (never silently switching) if the heuristic below would now
/// disagree, so a drifted `auto` input is visible without ever being acted
/// on by surprise. Only a genuinely first-ever resolution (no stored row —
/// a fresh `CREATE INDEX`, or a physically new oid from a `CONCURRENTLY`
/// rebuild that got past the CIC/RIC rejections noted above) falls through
/// to the "least-surprising" rule itself: remote if this table already has
/// *any* sign of prior remote configuration — an existing `pinecone.tables` row
/// (any `synced_by`/mode: a table already tracked via `pinecone.track`/
/// `promote`, or already carrying an earlier pg_pinecone-AM index) or a stored API
/// key (`pinecone.set_api_key`, the same prerequisite the pre-0.5 remote-only
/// path already required) — and local otherwise. This is what keeps 0.3's
/// regression suite green unchanged: every existing AM test calls `pg_pinecone.
/// set_api_key`/points at a mock backend before `CREATE INDEX ... USING
/// pg_pinecone` with no `WITH (...)` clause at all, so `has_stored_api_key()` is
/// true and this resolves to remote exactly as before. A genuinely fresh
/// install (no key, no prior tracking) instead gets local — pg_pinecone's "exact
/// until your data needs approximation" first-touch story needs no Pinecone
/// account at all to get a working index.
fn resolve_mode(
    heap_relid: pg_sys::Oid,
    index_relid: pg_sys::Oid,
    requested: crate::local::options::Mode,
) -> crate::local::metapage::Mode {
    use crate::local::{metapage, options};
    let resolved = match requested {
        options::Mode::Local => metapage::Mode::Local,
        options::Mode::Remote => metapage::Mode::Remote,
        options::Mode::Auto => {
            let heuristic =
                if registry::lookup(heap_relid).is_some() || secret::has_stored_api_key() {
                    metapage::Mode::Remote
                } else {
                    metapage::Mode::Local
                };
            match lookup_engine_mode(index_relid) {
                Some(sticky) => {
                    if sticky != heuristic {
                        warning!(
                            "pg_pinecone: index {index_relid:?} has mode='auto', and its inputs would \
                             now resolve to '{}' — but it stays '{}' (its originally-resolved \
                             engine); REINDEX never silently switches engines. To switch \
                             deliberately: ALTER INDEX ... SET (mode = '{}'); REINDEX ...;",
                            engine_mode_str(heuristic),
                            engine_mode_str(sticky),
                            engine_mode_str(heuristic),
                        );
                    }
                    sticky
                }
                None => heuristic,
            }
        }
    };
    persist_engine_mode(index_relid, resolved);
    resolved
}

pub(crate) fn single_primary_key_column(relid: pg_sys::Oid, display: &str) -> String {
    let pk = catalog::primary_key_columns(relid);
    match pk.as_slice() {
        [single] => single.name.clone(),
        [] => {
            error!(
                "pg_pinecone: {display} has no primary key — the pg_pinecone access method requires one"
            )
        }
        _ => error!(
            "pg_pinecone: {display} has a composite primary key, which the pg_pinecone access method \
             does not support"
        ),
    }
}

/// Rejects a text/varchar filter column whose collation is
/// nondeterministic (`is_deterministic = false`, from
/// `pg_sys::get_collation_isdeterministic`) — see the call site in
/// [`ambuild`] for why. Split out as a plain, pg_sys-free function so a
/// `#[pg_test]` can drive both outcomes directly with a synthetic bool: this
/// machine's Postgres build is configured `--without-icu`
/// (`pg_config --configure`), and nondeterministic collations are an
/// ICU-only feature (confirmed by hand: `CREATE COLLATION ... (provider =
/// libc, deterministic = false)` raises "nondeterministic collations not
/// supported with this provider" on this build) — so no real nondeterministic
/// collation can be created here to exercise `get_collation_isdeterministic`
/// end to end. The DDL-time code path that calls this (`ambuild`) is instead
/// covered end to end for the deterministic (default-collation) case, which
/// every existing text-filter test already exercises.
pub(crate) fn reject_nondeterministic_filter_collation(
    is_deterministic: bool,
    column: &str,
    display: &str,
) {
    if !is_deterministic {
        error!(
            "pg_pinecone: filter column \"{column}\" of {display} uses a nondeterministic collation \
             — a pg_pinecone filter column's stored value is a hash of its exact bytes, and \
             nondeterministic collation equality (\"equal but not byte-identical\", e.g. ICU \
             case/accent-insensitive) can never be honored by a byte hash; use a deterministic \
             collation (the default, or any COLLATE without ICU's deterministic=false)"
        );
    }
}

/// A registry filter column for the "doesn't match" message.
fn show_filter_column(column: Option<&str>) -> String {
    column.map_or_else(|| "none".to_string(), |name| format!("\"{name}\""))
}

/// The optional filter key (index column 2) of the index being built, validated at DDL time with
/// the exact failure named: a plain column (no expression) of a supported equality-filterable
/// type. Shared by the local and the remote build; returns its spec and the column's name.
unsafe fn validate_filter_column(
    heap: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
    columns: &[catalog::ColumnCatalog],
    display: &str,
) -> Option<(crate::local::build::FilterSpec, String)> {
    unsafe {
        if (*index_info).ii_NumIndexKeyAttrs != 2 {
            return None;
        }
        let f_attnum = (*index_info).ii_IndexAttrNumbers[1];
        if f_attnum <= 0 {
            error!(
                "pg_pinecone: a pg_pinecone filter column must be a plain column, not an expression"
            );
        }
        let fcol = columns
            .iter()
            .find(|c| c.attnum == f_attnum)
            .unwrap_or_else(|| error!("pg_pinecone: filter column not found on {display}"));
        let kind = match fcol.type_name.as_str() {
            "int2" => crate::local::metapage::FilterKind::Int2,
            "int4" => crate::local::metapage::FilterKind::Int4,
            "int8" => crate::local::metapage::FilterKind::Int8,
            "text" | "varchar" => {
                // A pg_pinecone filter column's stored value
                // is a hash of its exact bytes (`pages::fnv1a64`),
                // checked for equality against another hash. A
                // *nondeterministic* collation (ICU's `deterministic
                // = false`, e.g. case/accent-insensitive) can make
                // SQL `=` true for two byte-different strings —
                // semantics no byte hash can ever honor — so mode L's
                // "recall 1.0 by construction" guarantee and mode
                // H's in-loop filter would both silently drop rows
                // Postgres considers matches. Rejected here, at
                // CREATE INDEX time, rather than discovered as a
                // missing row later. The remote engine has the same
                // limit: Pinecone's `$eq` is a byte comparison too.
                let tupdesc = (*heap).rd_att;
                let atts = tupdesc_attrs(tupdesc);
                let attcollation = atts[(f_attnum - 1) as usize].attcollation;
                let is_deterministic = pg_sys::get_collation_isdeterministic(attcollation);
                reject_nondeterministic_filter_collation(is_deterministic, &fcol.name, display);
                crate::local::metapage::FilterKind::Text
            }
            other => error!(
                "pg_pinecone: filter column \"{}\" of {display} has type {other} — a pg_pinecone \
                     filter column must be smallint, integer, bigint, text, or varchar",
                fcol.name
            ),
        };
        Some((
            crate::local::build::FilterSpec {
                heap_attnum: f_attnum,
                kind,
            },
            fcol.name.clone(),
        ))
    }
}

/// Counts live tuples scanned during [`ambuild`]'s (otherwise inert)
/// `table_index_build_scan` call — see that function's rustdoc for why the
/// scan itself does nothing beyond producing an accurate `heap_tuples`/
/// `index_tuples` count for `pg_class` statistics.
#[pg_guard]
unsafe extern "C-unwind" fn count_build_callback(
    _index: pg_sys::Relation,
    _tid: pg_sys::ItemPointer,
    _values: *mut pg_sys::Datum,
    _isnull: *mut bool,
    _tuple_is_alive: bool,
    state: *mut c_void,
) {
    unsafe {
        let count = &mut *(state as *mut i64);
        *count += 1;
    }
}

const VECTOR_DATUM_HEADER_BYTES: usize = 8;
const PLAIN_ROW_BUDGET_BYTES: usize = 8_000;

/// The storage name to put in the notice when `STORAGE PLAIN` would keep a `vector(dim)` column
/// inline: the column is allowed out of line (EXTENDED or EXTERNAL, pgvector's default), the datum is
/// wide enough to be moved there, and a PLAIN row still fits one page. MAIN stays inline already.
fn plain_storage_advice(storage: char, dim: usize) -> Option<&'static str> {
    let name = match storage {
        'x' => "EXTENDED",
        'e' => "EXTERNAL",
        _ => return None,
    };
    let datum_bytes = VECTOR_DATUM_HEADER_BYTES + 4 * dim;
    (dim > crate::local::cost::TOAST_DIM_THRESHOLD as usize && datum_bytes < PLAIN_ROW_BUDGET_BYTES)
        .then_some(name)
}

fn notice_if_plain_storage_would_help(
    relid: pg_sys::Oid,
    schema: &str,
    table: &str,
    column: &str,
    dimension: usize,
) {
    let Some(storage) = catalog::column_storage(relid, column)
        .and_then(|storage| plain_storage_advice(storage, dimension))
    else {
        return;
    };
    let table = pgrx::spi::quote_qualified_identifier(schema, table);
    let column = pgrx::spi::quote_identifier(column);
    pgrx::pg_sys::panic::ErrorReport::new(
        PgSqlErrorCode::ERRCODE_SUCCESSFUL_COMPLETION,
        format!(
            "pg_pinecone: {table}.{column} is stored out of line (STORAGE {storage}), so every rescored \
             row costs a TOAST fetch; keep the {}-byte vector inline with: ALTER TABLE {table} ALTER \
             COLUMN {column} SET STORAGE PLAIN",
            VECTOR_DATUM_HEADER_BYTES + 4 * dimension
        ),
        pgrx::function_name!(),
    )
    .set_detail(
        "measured on an r6id.4xlarge at 1M x 768d: about 0.65 ms less per 200-row rescore \
         (2.96 to 2.29 ms per query at 20 lists)",
    )
    .set_hint(format!(
        "existing rows move inline when the table is rewritten: VACUUM FULL {table}. A row must fit \
         one 8 KB page: about 2,000 dimensions at most, fewer next to other wide columns."
    ))
    .report(pgrx::PgLogLevel::NOTICE);
}

/// Builds a `pg_pinecone`-AM index: registers `heapRelation` in `pinecone.tables`
/// (`mode = 'remote'`, `synced_by = 'am'`), creates or adopts the matching
/// Pinecone index, enqueues a `'backfill'` outbox row (the background worker
/// walks existing rows), and writes a single `GenericXLog`'d
/// metapage so the index relation isn't zero-page (matters for pg_dump/
/// restore and `REINDEX`, neither exercised by the AM's own logic yet).
///
/// `table_index_build_scan` is still called — Postgres expects it for MVCC
/// snapshot handling during a plain (non-concurrent) `CREATE INDEX` — but
/// with a no-op callback that only counts tuples: this AM stores no index
/// data of its own (Pinecone is the only copy), so there's nothing to insert
/// per existing tuple here; that's the backfill job's job.
#[pg_guard]
unsafe extern "C-unwind" fn ambuild(
    heap: pg_sys::Relation,
    index: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    unsafe {
        let heap_relid = (*heap).rd_id;
        let index_relid = (*index).rd_id;

        // Composite-key shape: one vector column, then AT MOST one
        // filter column (`amcanmulticol`). `ii_NumIndexAttrs` equals
        // `ii_NumIndexKeyAttrs` here (`amcaninclude = false` rejects INCLUDE
        // columns before ambuild ever runs).
        let n_key_atts = (*index_info).ii_NumIndexKeyAttrs;
        if !(1..=2).contains(&n_key_atts) {
            error!(
                "pg_pinecone: the pg_pinecone access method supports one vector column plus at most one \
                 filter column — CREATE INDEX ... USING pinecone (embedding <opclass>[, \
                 filter_column]) — got {n_key_atts} columns"
            );
        }
        let attnum = (*index_info).ii_IndexAttrNumbers[0];
        if attnum <= 0 {
            error!(
                "pg_pinecone: the pg_pinecone access method only supports a plain indexed column, not an \
                 expression"
            );
        }
        // Partial indexes: neither the local
        // nor the remote engine ever evaluates `ii_Predicate` anywhere in
        // this crate — `ambuild`'s own collect callbacks and `aminsert` index
        // (and, for local mode, cluster/store) every row regardless of any
        // `WHERE` clause a caller wrote. A `CREATE INDEX ... WHERE ...` would
        // therefore silently behave as a full index while *looking* like a
        // partial one — a caller relying on the predicate to keep excluded
        // rows out of scan/maintenance-surface results (or out of what a
        // REINDEX-driven rebuild touches) would be wrong. Rejected outright,
        // at DDL time, rather than discovered later as an unenforced
        // assumption.
        if !(*index_info).ii_Predicate.is_null() {
            error!(
                "pg_pinecone: partial indexes (CREATE INDEX ... WHERE ...) are not supported by the \
                 pg_pinecone access method — the predicate would never actually be evaluated"
            );
        }

        let (schema, table) = catalog::resolve_table_name(heap_relid)
            .unwrap_or_else(|| error!("pg_pinecone: could not resolve the indexed table's name"));
        let display = format!("{schema}.{table}");

        let columns = catalog::all_columns(heap_relid);
        let column = columns
            .iter()
            .find(|c| c.attnum == attnum)
            .unwrap_or_else(|| error!("pg_pinecone: indexed column not found on {display}"));
        if column.type_name != "vector" {
            if n_key_atts == 2 {
                error!(
                    "pg_pinecone: column \"{}\" of {display} has type {}, not vector — a pg_pinecone \
                     index's FIRST column must be the vector column; the filter column comes \
                     second",
                    column.name, column.type_name
                );
            }
            error!(
                "pg_pinecone: column \"{}\" of {display} has type {}, not vector",
                column.name, column.type_name
            );
        }
        let dimension = catalog::vector_dimension(heap_relid, &column.name);
        if dimension <= 0 {
            error!(
                "pg_pinecone: column \"{}\" of {display} has an undeclared vector dimension — \
                 declare it as vector(n), e.g. vector(1536)",
                column.name
            );
        }

        let opclass = catalog::index_first_opclass_name(index_relid)
            .unwrap_or_else(|| error!("pg_pinecone: could not resolve {display}'s operator class"));
        let metric = metric_for_opclass(&opclass);
        let metric_enum = crate::local::metric::Metric::from_opclass_metric(&metric);

        // Mode resolution:
        // an explicit `WITH (mode=...)` reloption wins outright; `auto` (the
        // reloption default) defers to `resolve_mode`'s "least-surprising"
        // rule below. Local mode branches off completely here — it needs
        // none of the registry-row/Pinecone-index machinery the rest of this
        // function exists for (no `pinecone.tables` row, no primary key
        // requirement, no outbox) — see `local::build::ambuild_local`.
        let opts = crate::local::options::effective(index);
        let mode = resolve_mode(heap_relid, index_relid, opts.mode);
        if mode == crate::local::metapage::Mode::Local {
            // `CREATE INDEX CONCURRENTLY`/`REINDEX ... CONCURRENTLY` stays
            // rejected even now that local aminsert is real — see
            // `local::vacuum`'s module docstring ("the CIC/RIC decision")
            // for the full verdict; briefly: (1) a FLAT index stores no
            // TIDs, so `validate_index`'s enumerate-and-diff protocol
            // degenerates — it would see zero index TIDs, believe every
            // heap row is missing, and "re-insert" all of them (correct
            // results only by accident of the flat scan reading the heap,
            // with the staleness counter inflated to the whole table);
            // (2) a CONCURRENTLY build creates a fresh index oid mid-flight,
            // and `pinecone._am_engine`'s stickiness is keyed by oid — a
            // `mode = auto` index could silently flip engines across the
            // rebuild (the exact drift `resolve_mode`'s WARNING exists to
            // prevent); (3) the snapshot-window interplay between the
            // k-means training set and the catch-up inserts is unvalidated.
            // Honest rejection beats subtle corruption; revisit with the
            // lifecycle work.
            if (*index_info).ii_Concurrent {
                error!(
                    "pg_pinecone: CREATE INDEX CONCURRENTLY / REINDEX CONCURRENTLY is not supported \
                     for a local-mode pg_pinecone index — use a plain CREATE INDEX/REINDEX instead"
                );
            }
            // The optional filter key column: validated at DDL time.
            let filter = validate_filter_column(heap, index_info, &columns, &display)
                .map(|(spec, _name)| spec);
            notice_if_plain_storage_would_help(
                heap_relid,
                &schema,
                &table,
                &column.name,
                dimension as usize,
            );
            let (heap_tuples, index_tuples) = crate::local::build::ambuild_local(
                heap,
                index,
                index_info,
                attnum,
                dimension as u32,
                metric_enum,
                &opts,
                filter,
            );
            let res = pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBuildResult>())
                as *mut pg_sys::IndexBuildResult;
            (*res).heap_tuples = heap_tuples;
            (*res).index_tuples = index_tuples;
            return res;
        }

        // A remote index's filter key (PR3, step A) is synced to Pinecone as
        // metadata and pushed down as a `$eq` filter ([`crate::am_filter`]).
        // Same DDL-time validation as the local engine's.
        let filter = validate_filter_column(heap, index_info, &columns, &display);
        let filter_column = filter.as_ref().map(|(_, name)| name.clone());

        let id_column = single_primary_key_column(heap_relid, &display);

        match registry::lookup(heap_relid) {
            Some(existing) if existing.synced_by == "trigger" => {
                error!(
                    "pg_pinecone: {display} is promote()-managed (synced_by = 'trigger') — run \
                     pinecone.demote() on it before CREATE INDEX ... USING pinecone"
                );
            }
            // A same-oid rebuild (REINDEX) can only see another filter-column NAME
            // when the column was renamed: it follows the rename (REBUILD arm).
            Some(existing)
                if existing.vector_column != column.name
                    || existing.metric != metric
                    || existing.dimension != dimension
                    || (existing.filter_column != filter_column
                        && existing.am_index_relid != Some(index_relid)) =>
            {
                error!(
                    "pg_pinecone: {display} already has a pg_pinecone access-method index \
                     (vector_column=\"{}\", metric={}, dimension={}, filter_column={}) that \
                     doesn't match this one (vector_column=\"{}\", metric={metric}, \
                     dimension={dimension}, filter_column={}) — a second, different pg_pinecone \
                     access-method index on the same table is not yet supported",
                    existing.vector_column,
                    existing.metric,
                    existing.dimension,
                    show_filter_column(existing.filter_column.as_deref()),
                    column.name,
                    show_filter_column(filter_column.as_deref()),
                );
            }
            // Second, DIFFERENT-oid index of matching shape: without this
            // arm, a second `CREATE INDEX ... USING pinecone` on the same column/metric/dimension
            // fell into the REBUILD arm below, which clobbers the registry's
            // `am_index_relid` to point at the NEW index while the FIRST one
            // stays fully alive — both now independently enqueue their own
            // `aminsert` upserts (double-enqueue on every write), and
            // dropping the orphaned one no longer matches
            // `pg_pinecone_on_sql_drop`'s `am_index_relid = obj.objid` lookup (it
            // was overwritten), so that drop leaks its registry/outbox state
            // forever. `existing.am_index_relid`'s own validity (rather than
            // its mere presence) is what tells a genuine second index apart
            // from a stale, abandoned one: a same-oid rebuild (plain
            // `REINDEX`/`VACUUM FULL`/`CLUSTER`) and a fresh `CREATE INDEX`
            // replacing a `CREATE INDEX CONCURRENTLY` that failed partway
            // (leaving an invalid index behind) both fall through to the
            // REBUILD arm instead, unchanged. Known casualty: `REINDEX
            // CONCURRENTLY` now errors here too — its new-copy build runs
            // while the OLD index is still valid under a different oid,
            // which is indistinguishable inside `ambuild` from a genuine
            // second index (both present as "new invalid index built beside
            // an existing valid one"). Erroring on both is the safe side of
            // that ambiguity: silent double-enqueue corruption is worse than
            // requiring plain `REINDEX` for rebuilds.
            Some(existing)
                if existing.am_index_relid != Some(index_relid)
                    && existing.am_index_relid.is_some_and(catalog::index_is_valid) =>
            {
                error!(
                    "pg_pinecone: {display} already has a pg_pinecone access-method index (oid {}) — \
                     only one pg_pinecone access-method index per table is supported; DROP INDEX it \
                     first",
                    existing
                        .am_index_relid
                        .expect("checked Some by this arm's own guard")
                );
            }
            // Attach: `lifecycle::create_index` pre-inserts exactly this row
            // shape (an `am`-owned row with no local index yet) right before
            // issuing this `CREATE INDEX` — nothing else produces it, since the
            // `None` arm below always stamps `am_index_relid` in the same insert.
            Some(existing) if existing.synced_by == "am" && existing.am_index_relid.is_none() => {
                registry::rebuild_am(&registry::RegistryRow {
                    am_index_relid: Some(index_relid),
                    ..existing
                });
            }
            Some(existing) => {
                // REBUILD: REINDEX/VACUUM FULL/CLUSTER re-running `ambuild`
                // on an already-registered pg_pinecone access-method index. The
                // Pinecone index itself is untouched (same index_name/
                // namespace as `existing`) — only the *local* index relation
                // is rebuilt, so the registry row is refreshed in place and
                // the remote side is fully repopulated via delete_all then
                // backfill (mirrors `pinecone.promote`'s own repair path).
                registry::rebuild_am(&registry::RegistryRow {
                    vector_column: column.name.clone(),
                    id_column,
                    metric,
                    dimension,
                    // Explicit, not inherited via `..existing` (see
                    // `registry::rebuild_am`'s rustdoc): a `REINDEX
                    // CONCURRENTLY` builds this as a genuinely new index
                    // relation before dropping the old one, so this row must
                    // track the CURRENT `index_relid`, not whatever `existing`
                    // still held from before this rebuild.
                    am_index_relid: Some(index_relid),
                    // Follows a rename of the filter column (see the mismatch arm).
                    filter_column,
                    ..existing
                });
                enqueue_delete_all(heap_relid);
                enqueue_backfill(heap_relid);
            }
            None => {
                let index_name = promote::default_index_name(&schema, &table);
                let namespace = format!("{schema}.{table}");

                if !secret::has_stored_api_key() {
                    error!(
                        "pg_pinecone: a pg_pinecone access-method index needs a stored API key — run \
                         pinecone.set_api_key(...) before CREATE INDEX ... USING pinecone"
                    );
                }

                let client = backend::client_for_effective_key();
                let mapped_metric = promote::map_metric(&metric);
                let read_capacity = promote::create_or_adopt_index(
                    &client,
                    &index_name,
                    dimension as u32,
                    mapped_metric,
                    json!({
                        "serverless": {
                            "cloud": backend::guc_string(&AM_CLOUD),
                            "region": backend::guc_string(&AM_REGION),
                        }
                    }),
                    &display,
                );
                promote::wait_ready(&client, &index_name);

                registry::insert(&registry::RegistryRow {
                    relid: heap_relid,
                    vector_column: column.name.clone(),
                    id_column,
                    metric,
                    dimension,
                    metadata_columns: Vec::new(),
                    mode: "remote".to_string(),
                    sync_mode: "async".to_string(),
                    synced_by: "am".to_string(),
                    // A genuinely new row is never parked — same convention as
                    // `track::track`'s own insert.
                    queue_parked: false,
                    parked_reason: None,
                    index_name: Some(index_name),
                    namespace: Some(namespace),
                    am_index_relid: Some(index_relid),
                    // A genuinely new row has no prior VACUUM history.
                    last_vacuum_error: None,
                    // A plain `CREATE INDEX ... USING pinecone` predates the
                    // engine surface — only `lifecycle::promote_index`/
                    // `create_index`/`set_engine` populate `engine`.
                    engine: None,
                    // An AM row (`synced_by = 'am'`) can never be detached —
                    // enforced at the schema level too (`registry.rs`'s
                    // `tables_storage_synced_by_check`).
                    storage: "full".to_string(),
                    detached_at: None,
                    detach_backup_id: None,
                    backup_gate: None,
                    detach_rpo_secs: None,
                    read_capacity,
                    attached: false,
                    partition: None,
                    filter_column,
                });

                enqueue_backfill(heap_relid);
            }
        }

        // Every write through this AM (`aminsert`, and the backfill/
        // delete_all just enqueued above — new index or REBUILD) only ever
        // drains via the background worker: the AM path has no `sync_mode`
        // choice, unlike `pinecone.promote` — it's always async. With no
        // launcher alive-and-enabled, those rows would queue in
        // `pinecone.outbox` forever — same launcher-liveness check as
        // `promote::promote`'s own, mirrored here so
        // `CREATE INDEX ... USING pinecone` warns just as loudly.
        if !outbox::launcher_lock_held() {
            warning!(
                "pg_pinecone: {display} has a pg_pinecone access-method index but no pg_pinecone worker \
                 launcher is running in this cluster (preload absent or \
                 pinecone.worker_enabled off) — writes will queue in pinecone.outbox but nothing \
                 will drain them until one starts (see pinecone.status())"
            );
        }

        crate::local::metapage::write(
            index,
            &crate::local::metapage::MetaPage {
                mode: crate::local::metapage::Mode::Remote,
                metric: metric_enum,
                is_flat: false,
                heap_attnum: attnum,
                dim: dimension as u32,
                n_lists: 0,
                format: crate::local::metapage::FORMAT_VERSION,
                layout: crate::local::metapage::LAYOUT_VERSION,
                // A remote-mode index has no codes: no rotation, no query bits, no code width.
                rotation_rounds: 0,
                bq_bits: 0,
                code_bits: 0,
                rescore_multiplier: 0,
                small_n_threshold: 0,
                calib_nprobe: 0,
                calib_pool: 0,
                calib_samples: 0,
                calib_reached: false,
                calib_recall: 0.0,
                calib_queries: crate::local::metapage::QueriesSource::Rows,
                target_recall: 0.0,
                total_vectors: 0,
                seed: 0,
                centroid_dir_start: crate::local::metapage::INVALID_BLOCK,
                filter_kind: filter
                    .as_ref()
                    .map_or(crate::local::metapage::FilterKind::None, |(spec, _)| {
                        spec.kind
                    }),
                filter_heap_attnum: filter.as_ref().map_or(0, |(spec, _)| spec.heap_attnum),
                stats_start: crate::local::metapage::INVALID_BLOCK,
                inserted_since_build: 0,
            },
        );

        let mut build_count: i64 = 0;
        pg_sys::table_index_build_scan(
            heap,
            index,
            index_info,
            true,
            true,
            Some(count_build_callback),
            &mut build_count as *mut i64 as *mut c_void,
            std::ptr::null_mut(),
        );

        let res = pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBuildResult>())
            as *mut pg_sys::IndexBuildResult;
        (*res).heap_tuples = build_count as f64;
        (*res).index_tuples = build_count as f64;
        res
    }
}

/// Postgres calls `ambuildempty` only for `UNLOGGED` indexes, to populate
/// their `INIT_FORKNUM` (which it copies over `MAIN_FORKNUM` to reset the
/// index after a crash restart) — `index.c`'s `index_build` already
/// `smgrcreate`s/`log_smgrcreate`s that fork before calling this, exactly
/// mirroring how it drives `ambuild`'s own `MAIN_FORKNUM` write. Writing to
/// `MAIN_FORKNUM` here (as this used to, unconditionally) would leave an
/// unlogged pg_pinecone-AM index with an empty `INIT_FORKNUM`, so `CREATE INDEX`
/// itself would look fine but the index would silently revert to zero pages
/// on the next crash restart. Matches contrib/bloom's own `blbuildempty` ->
/// `BloomInitMetapage(index, INIT_FORKNUM)`.
///
/// Goes through [`crate::local::metapage::write_init_fork_direct`], **never**
/// [`crate::local::metapage::write`] — that function's own docstring is the
/// fix for a real bug this used to have: routing this write through
/// `GenericXLog` (as `write` does, for `MAIN_FORKNUM`) silently drops its WAL
/// record for exactly the unlogged relations `ambuildempty` exists to serve,
/// which corrupts every such index across the very first crash after it's
/// created.
#[pg_guard]
unsafe extern "C-unwind" fn ambuildempty(index: pg_sys::Relation) {
    unsafe {
        // Mirrors whatever `ambuild` just wrote to `MAIN_FORKNUM` (mode,
        // metric, dimension, heap attnum — and the filter column's
        // identity: the reset image must remember WHICH column filters this
        // index even though, being flat, it stores no per-record data) but
        // reset to empty — an unlogged relation's `INIT_FORKNUM` is what a
        // crash restart copies back over `MAIN_FORKNUM`, so it only needs to
        // look like "freshly built, empty", never to carry real cluster/list
        // data of its own.
        let meta = crate::local::metapage::read(index);
        crate::local::metapage::write_init_fork_direct(
            index,
            &crate::local::metapage::MetaPage {
                is_flat: true, // reset to the trivial empty state
                n_lists: 0,
                rescore_multiplier: 0,
                calib_nprobe: 0,
                calib_pool: 0,
                calib_samples: 0,
                calib_reached: false,
                calib_recall: 0.0,
                calib_queries: crate::local::metapage::QueriesSource::Rows,
                target_recall: 0.0,
                total_vectors: 0,
                seed: 0,
                centroid_dir_start: crate::local::metapage::INVALID_BLOCK,
                stats_start: crate::local::metapage::INVALID_BLOCK,
                inserted_since_build: 0,
                ..meta
            },
        );
    }
}

/// Enqueues a `'backfill'` outbox row. `id_kind: "tid"` self-describes the
/// AM's own backfills as TID-keyed (`aminsert`'s own upsert ids are
/// `"block:offset"`, never the table's primary key) — unlike a
/// `pinecone.promote`-driven backfill, which is PK-keyed and so stays
/// `{"after_id": null}` with no `id_kind` at all. Nothing dispatches on
/// this yet (the worker will), but the AM's own rows need to be
/// self-describing from day one — retrofitting `id_kind` onto already
/// in-flight rows once a worker exists would be a migration, not a one-line
/// change.
pub(crate) fn enqueue_backfill(relid: pg_sys::Oid) {
    insert_outbox_row(
        relid,
        "backfill",
        Some(JsonB(json!({"after_id": null, "id_kind": "tid"}))),
    );
}

/// Enqueues a `'delete_all'` outbox row — `am::ambuild`'s REINDEX/VACUUM
/// FULL/CLUSTER rebuild path pairs this with [`enqueue_backfill`] to fully
/// repopulate the remote namespace, mirroring `pinecone.promote`'s own
/// `delete_all_and_backfill` repair path.
fn enqueue_delete_all(relid: pg_sys::Oid) {
    insert_outbox_row(relid, "delete_all", None);
}

// ---------------------------------------------------------------------------
// Metapage v2 read/write lives in `local::metapage` now — see that module's
// own docstring for the version-dispatch rules (0.3's original placeholder
// format, `version == 1`, is a permanent, read-only fossil; every build from
// 0.5 onward writes v2, local or remote mode alike).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// aminsert
// ---------------------------------------------------------------------------

/// Remote mode: enqueues one `'upsert'` outbox row per inserted tuple (one
/// row per tuple, never batched, so `ambulkdelete` can later purge by TID).
/// Local mode: appends the tuple to the
/// index's own pages — `local::insert::aminsert_local` (nearest-centroid
/// append for a clustered index; counter-only for the flat bootstrap, whose
/// scan path reads the heap directly). A `NULL` vector is a pure no-op in
/// both modes: nothing to push/store, and nothing to delete either (there
/// was never an index record for this TID to begin with) — matching the
/// build paths' own NULL handling. Remote-mode routing is unchanged from
/// 0.3: `local::metapage::read` maps a pre-0.5 `version == 1` metapage to
/// `Mode::Remote` unconditionally, so the 0.3 regression suite observes
/// nothing new.
#[pg_guard]
#[allow(clippy::too_many_arguments)] // Postgres's aminsert_function C signature, not ours to shrink
unsafe extern "C-unwind" fn aminsert(
    index: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    heap: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    index_info: *mut pg_sys::IndexInfo,
) -> bool {
    unsafe {
        if *isnull {
            return false;
        }
        let meta = crate::local::metapage::read(index);
        // A native index below format 3 holds codes this build cannot extend: writes to its table
        // fail until REINDEX (runbook: "native indexes are offline from the restart until REINDEX").
        if meta.needs_reindex() {
            crate::local::metapage::refuse_old_format(index, meta.stored());
        }
        // `Promoting` (`pinecone.promote_index`): DUAL-write. The index is
        // still serving LOCAL scans throughout the promoting window
        // (never-partial-namespace), so a row inserted mid-promotion must
        // still land in the local list/centroid pages — `aminsert_local` has
        // no mode check of its own, it just uses the metapage's structural
        // fields (n_lists/k_bits/centroid_dir_start/...), all untouched by a
        // mode flip. It's ALSO enqueued for remote, exactly like plain
        // remote-mode writes: `promote::try_tid_backfill_chunk`'s own
        // correctness argument ("post-cursor inserts converge") already
        // assumes aminsert keeps enqueueing during the backfill — this is
        // the first mode that argument actually applies to.
        match meta.mode {
            crate::local::metapage::Mode::Local => {
                crate::local::insert::aminsert_local(index, values, isnull, heap_tid, &meta);
                return false;
            }
            crate::local::metapage::Mode::Promoting => {
                crate::local::insert::aminsert_local(index, values, isnull, heap_tid, &meta);
            }
            // Remote, Demoting (still served/kept in sync from remote until
            // the rebuild REINDEX swaps it), Detached: enqueue only.
            _ => {}
        }
        let relid = (*heap).rd_id;
        let id = tid_to_id(*heap_tid);
        let vector = parse_vector(*values);
        let key = filter_key(&meta, values, isnull);
        let metadata = key.as_ref().map(|key| {
            let column = heap_column_name(heap, meta.filter_heap_attnum);
            am_filter::metadata(&column, key.to_json())
        });
        let owner = am_handler_owner(index);
        let _guard = SecurityDefinerGuard::switch_to(owner);
        let partition = if meta.has_filter() {
            am_partition::with_writer_cache(index_info, relid, |cache| {
                cache.route(relid, key.as_ref())
            })
        } else {
            None
        };
        enqueue_upsert(relid, &id, &vector, metadata, partition.as_deref());
        false
    }
}

/// `tupdesc`'s `FormData_pg_attribute` array; behind `compact_attrs` since PG18.
unsafe fn tupdesc_attrs<'a>(tupdesc: pg_sys::TupleDesc) -> &'a [pg_sys::FormData_pg_attribute] {
    unsafe {
        let natts = (*tupdesc).natts as usize;
        #[cfg(not(feature = "pg18"))]
        let attrs = (*tupdesc).attrs.as_slice(natts);
        #[cfg(feature = "pg18")]
        let attrs =
            std::slice::from_raw_parts((*tupdesc).compact_attrs.as_ptr().add(natts).cast(), natts);
        attrs
    }
}

/// The current name of heap column `attnum`.
unsafe fn heap_column_name(heap: pg_sys::Relation, attnum: i16) -> String {
    unsafe {
        let tupdesc = (*heap).rd_att;
        let attrs = tupdesc_attrs(tupdesc);
        std::ffi::CStr::from_ptr(attrs[(attnum - 1) as usize].attname.data.as_ptr())
            .to_string_lossy()
            .into_owned()
    }
}

/// The filter key of the tuple being inserted; `None` for an unfiltered index and for a NULL key
/// (which never matches `$eq`).
unsafe fn filter_key(
    meta: &crate::local::metapage::MetaPage,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
) -> Option<EqKey> {
    unsafe {
        if !meta.has_filter() || *isnull.add(1) {
            return None;
        }
        Some(EqKey::of_datum(meta.filter_kind, *values.add(1)))
    }
}

// ---------------------------------------------------------------------------
// ambulkdelete / amvacuumcleanup
// ---------------------------------------------------------------------------

/// Real dead-tuple purge. With no local tuple store, this AM cannot enumerate
/// dead TIDs by scanning its own index relation the way a normal AM's
/// `ambulkdelete` would — there is nothing of its own to scan. Instead it
/// supplies the *candidates*: every id currently in the remote namespace
/// (`Client::list_vector_ids`, paginated — one `/vectors/list` walk per
/// VACUUM, the documented cost of this design) **UNION** `relid`'s own
/// still-queued `'upsert'` outbox ids ([`queued_upsert_ids`] —
/// a TID whose `aminsert` upsert is still queued (worker down/backlogged)
/// when the row dies — a `DELETE`, or
/// a non-HOT `UPDATE` that supersedes it — never appears in the remote list
/// at all, since it was never pushed; a remote-only candidate set silently
/// misses it, leaving a stale queued upsert to push a phantom vector once a
/// worker eventually resumes — for a Postgres row that no longer exists, and
/// whose TID VACUUM may since have told the heap it's free to recycle for an
/// unrelated future row). Each candidate is parsed back to a TID
/// (`id_to_tid`) and tested against `callback` — Postgres's own answer to
/// "is this TID dead" (via the heap's visibility map, exactly what a normal
/// AM's callback loop also relies on).
///
/// `pg_advisory_xact_lock(relid)` (`outbox::lock_relid`) is taken FIRST,
/// unconditionally, before the registry lookup or either remote call
/// (this used to be taken only once
/// `dead_ids` was known to be non-empty, and after both HTTP calls) and held
/// for this whole call's duration. The horizon that establishes: no
/// concurrent worker drain round for this relid (`drain_step`'s `Worker`
/// path takes this same lock first) can run between the moment this call
/// captures its candidate list and the moment it purges/deletes them —
/// closing a race where such a drain could otherwise remove, or add to, the
/// very outbox rows this call is mid-deciding about.
///
/// Then: purge this relid's still-queued `'upsert'` outbox rows for every
/// dead id (a stale queued upsert winning a race against the remote delete
/// below would resurrect a value VACUUM just decided was dead) -> a
/// SYNCHRONOUS remote delete of whichever dead ids are *also* present in the
/// remote list (deleting a never-pushed id remotely would be harmless —
/// Pinecone no-ops a delete of a nonexistent id — but there's nothing to
/// clean up by sending it, so this skips those for cleanliness). The remote
/// delete is never handed to the outbox: a failure here is raised as a
/// Postgres `ERROR` (bounded by the client's own retry policy first) rather
/// than queued for a later drain — the hard rule is that a stale TID
/// must never be reachable again once VACUUM believes it's gone (a later
/// page truncation would turn a stale match into a hard error on heap
/// fetch, not just a wrong answer), so autovacuum retrying the whole
/// operation later is the only acceptable failure mode. Both HTTP failure
/// points record [`registry::record_vacuum_error`] before raising
/// (see its own rustdoc for the
/// durability caveat).
/// Uses the stored API key (`backend::client_for_drain`'s `Worker` path,
/// same as the background worker), not the vacuuming session's own GUC/
/// effective key: VACUUM can run as any role with privileges on the table
/// (autovacuum included), and this must behave identically regardless of
/// who/what triggers it — the same reasoning as the worker's own credential
/// resolution.
#[pg_guard]
unsafe extern "C-unwind" fn ambulkdelete(
    info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
    callback: pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    unsafe {
        let stats = ensure_stats(stats);
        let heap_relid = (*(*info).heaprel).rd_id;

        // Local-mode dead-TID purging: this was once a documented no-op,
        // but it is a real hazard, not just a missing optimization — see
        // `local::vacuum`'s module docstring for the full "stale TID + heap
        // truncation -> permanent ERROR" story this closes.
        // `local::vacuum::ambulkdelete_local` does the actual
        // list-page sweep; this just wraps Postgres's raw callback as the
        // `FnMut(ItemPointerData) -> bool` it expects (same shape
        // `ambulkdelete_core`'s remote-mode closure below already uses).
        let index = (*info).index;
        let meta = crate::local::metapage::read(index);
        // A native index below format 3 is skipped, with a WARNING: autovacuum must not fail on it,
        // and REINDEX rebuilds it from the heap. A promoting one still sweeps its remote half.
        let stale_local = meta.needs_reindex();
        if stale_local {
            crate::local::metapage::warn_old_format(index, meta.stored());
            if meta.mode == crate::local::metapage::Mode::Local {
                return stats;
            }
        }
        // Should be unreachable in practice (Postgres always supplies a real
        // callback to `ambulkdelete` — only `amvacuumcleanup`'s
        // cleanup-only pass ever omits one) — handled defensively rather
        // than asserted, since there is nothing this AM could purge without
        // one anyway, in any mode.
        let Some(callback_fn) = callback else {
            return stats;
        };

        // LOCAL half — `Local` and `Promoting` alike: dual-write means
        // local storage has real, VACUUM-able per-record data throughout the
        // promoting window, so the list-page sweep must run exactly as for
        // genuine local mode. `Demoting`, by contrast, has no local pages
        // yet (that's what the pending REINDEX creates) — remote half only.
        let is_local_half = meta.mode.is_native() && !stale_local;
        if is_local_half {
            let sweep = crate::local::vacuum::ambulkdelete_local(index, &meta, |tid| {
                let mut tid = tid;
                callback_fn(&mut tid, callback_state)
            });
            (*stats).num_index_tuples = sweep.live as f64;
            (*stats).tuples_removed = sweep.dead as f64;
            if meta.mode == crate::local::metapage::Mode::Local {
                return stats;
            }
            // `Promoting` falls through to the REMOTE half below:
            // during promoting, deletes must
            // dual-sweep exactly as writes dual-write — a TID VACUUM just
            // reclaimed locally may already have been backfilled/enqueued
            // remotely, and leaving it there (or leaving its queued upsert)
            // resurrects it the moment the flip lands. Stats stay the LOCAL
            // sweep's numbers: local is the serving store while promoting.
        }

        // REMOTE half — `Remote`/`Demoting`/`Detached`, and (per the above)
        // the remote side of `Promoting`.
        // Taken first, before the registry read or either remote call — see
        // this function's own rustdoc.
        outbox::lock_relid(heap_relid);

        let client = backend::client_for_drain(backend::DrainCredentials::Worker);
        let (remote_count, dead_remote_count) = ambulkdelete_core(heap_relid, &client, |id| {
            let mut tid = id_to_tid(id).unwrap_or_else(|e| error!("pg_pinecone: {e}"));
            callback_fn(&mut tid, callback_state)
        })
        .unwrap_or_else(|e| {
            registry::record_vacuum_error(heap_relid, &e.to_string());
            backend::raise(e)
        });

        if !is_local_half {
            (*stats).num_index_tuples = (remote_count - dead_remote_count) as f64;
            (*stats).tuples_removed = dead_remote_count as f64;
        }
        stats
    }
}

/// [`ambulkdelete`]'s testable core, factored out so a `#[pg_test]` can
/// exercise the dispatch/purge/delete logic with a fake `is_dead` predicate
/// instead of a real Postgres `IndexBulkDeleteCallback` — sidestepping the
/// real-`VACUUM` harness limitation `am_ambulkdelete_purges_dead_tuple_via_
/// real_vacuum`'s own rustdoc documents (a `#[pg_test]`'s enclosing
/// transaction pins a snapshot that keeps a row deleted *inside* that same
/// test from ever looking reclaimable to a `VACUUM` triggered from inside
/// it, so `dead_ids` can never come back non-empty there). The real
/// `ambulkdelete` wraps Postgres's own callback (via `id_to_tid`) as
/// `is_dead`; tests substitute a plain closure over the id strings.
///
/// Candidate set = remote ids ([`Client::list_vector_ids`]) UNION `relid`'s
/// still-queued `'upsert'` outbox ids ([`queued_upsert_ids`] — a TID whose
/// `aminsert` upsert is still queued (worker down/backlogged) when the row
/// dies never appears in the remote list at all, since it was never pushed; a
/// remote-only candidate set silently misses it, leaving a stale queued upsert
/// to push a phantom
/// vector once a worker eventually resumes — for a Postgres row that no
/// longer exists, and whose TID VACUUM may since have told the heap it's
/// free to recycle for an unrelated future row). Every dead id (remote or
/// queued-only) gets its queued upsert purged; only dead ids *also* known to
/// the remote index get an actual remote delete call (a never-pushed id has
/// nothing to delete there — Pinecone would no-op it harmlessly, but
/// there's no reason to send it). Returns `(remote ids known, of those how
/// many were dead)` — what [`ambulkdelete`] needs for
/// `IndexBulkDeleteResult`'s stats; a queued-but-never-pushed dead id isn't
/// counted here, since it was never part of the remote index's own size.
pub(crate) fn ambulkdelete_core(
    heap_relid: pg_sys::Oid,
    client: &Client,
    mut is_dead: impl FnMut(&str) -> bool,
) -> Result<(usize, usize), pg_pinecone_client::Error> {
    let row = registry::lookup(heap_relid).unwrap_or_else(|| {
        error!(
            "pg_pinecone: VACUUM found no pinecone.tables row for relid {heap_relid:?} — should be \
             unreachable, ambuild always inserts one"
        )
    });
    let index_name = row
        .index_name
        .clone()
        .expect("pg_pinecone: an access-method row always has index_name");
    let partitioned = row.partition.is_some();
    // A partitioned table's vectors sit in the namespaces it registered (the drain registers one
    // once Pinecone took a write of it, and writes of a namespace still queued are purged below
    // whatever their namespace); an unregistered one is never visited.
    let namespaces: Vec<Option<String>> = if partitioned {
        partition_registry::partitions_of(heap_relid, partition::Lane::Vector)
            .into_iter()
            .map(|(_, namespace)| Some(namespace))
            .collect()
    } else {
        vec![row.namespace.clone()]
    };

    let mut listed: Vec<(Option<String>, Vec<String>)> = Vec::with_capacity(namespaces.len());
    for namespace in namespaces {
        let ids = client.list_vector_ids(&ListVectorsRequest {
            index: index_name.clone(),
            namespace: namespace.clone(),
        })?;
        listed.push((namespace, ids));
    }
    let remote_set: std::collections::HashSet<&str> = listed
        .iter()
        .flat_map(|(_, ids)| ids.iter().map(String::as_str))
        .collect();

    let mut candidates: Vec<String> = listed
        .iter()
        .flat_map(|(_, ids)| ids.iter().cloned())
        .collect();
    for id in queued_upsert_ids(heap_relid) {
        if !remote_set.contains(id.as_str()) {
            candidates.push(id);
        }
    }

    let dead_ids: Vec<String> = candidates.into_iter().filter(|id| is_dead(id)).collect();
    if !dead_ids.is_empty() {
        purge_am_outbox_rows(heap_relid, &dead_ids);
    }

    // Stats (and the remote deletes) reflect what the remote index holds: a still-queued,
    // never-pushed candidate was never part of it to begin with.
    let dead_set: std::collections::HashSet<&str> = dead_ids.iter().map(String::as_str).collect();
    let (mut known, mut purged) = (0, 0);
    for (namespace, ids) in &listed {
        known += ids.len();
        let dead_here: Vec<String> = ids
            .iter()
            .filter(|id| dead_set.contains(id.as_str()))
            .cloned()
            .collect();
        purged += dead_here.len();
        let batch = if partitioned {
            crate::sync_trigger::DELETE_MAX_IDS
        } else {
            dead_here.len().max(1)
        };
        for ids in dead_here.chunks(batch) {
            client.delete_vectors(&DeleteVectorsRequest {
                index: index_name.clone(),
                namespace: namespace.clone(),
                ids: Some(ids.to_vec()),
                delete_all: false,
                filter: None,
            })?;
        }
    }

    Ok((known, purged))
}

/// Deletes `relid`'s still-queued `'upsert'` outbox rows whose payload
/// references one of `dead_ids` — `am::aminsert` enqueues exactly one such
/// row per tuple, `payload = [{"id": ..., "values": [...]}]` (a one-element
/// jsonb array; see `enqueue_upsert`), so `payload->0->>'id'` is that row's
/// TID. Scoped to `op = 'upsert'` (the only op AM rows ever enqueue) and to
/// `relid` (already narrowed by `pg_pinecone_outbox_relid_id_idx`), so this scan
/// is bounded by that one relid's own backlog, not the whole outbox table.
fn purge_am_outbox_rows(relid: pg_sys::Oid, dead_ids: &[String]) {
    Spi::run_with_args(
        "DELETE FROM pinecone.outbox WHERE relid::oid = $1::oid AND op = 'upsert' \
         AND payload->0->>'id' = ANY($2::text[])",
        &[relid.into(), dead_ids.to_vec().into()],
    )
    .expect("pg_pinecone: ambulkdelete outbox purge failed");
}

/// `relid`'s still-queued `'upsert'` outbox ids — every TID `am::aminsert`
/// has enqueued a push for that a worker (or an inline drain — though AM
/// rows are always async, so that path never applies here) hasn't drained
/// yet. Folded into [`ambulkdelete`]'s dead-TID candidate set alongside
/// [`ListVectorsRequest`]'s remote-known ids — see that function's own
/// rustdoc for why a remote-only candidate set misses a TID that died before
/// its own queued upsert ever reached the remote index.
fn queued_upsert_ids(relid: pg_sys::Oid) -> Vec<String> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT payload->0->>'id' AS id FROM pinecone.outbox \
                 WHERE relid::oid = $1::oid AND op = 'upsert'",
                None,
                &[relid.into()],
            )
            .expect("pg_pinecone: ambulkdelete queued-upsert scan failed")
            .map(|row| {
                row.get_by_name::<String, _>("id")
                    .expect("pg_pinecone: queued-upsert id query failed")
                    .expect("pg_pinecone: queued-upsert id unexpectedly NULL")
            })
            .collect()
    })
}

#[pg_guard]
unsafe extern "C-unwind" fn amvacuumcleanup(
    info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    unsafe {
        // Cleanup does nothing for any index but must say, once per VACUUM, that it skips a native
        // index below format 3 (never raising: `peek` reads block 0 without refusing).
        if let Some(format) = crate::local::metapage::stale_native_format((*info).index) {
            crate::local::metapage::warn_old_format((*info).index, format);
        }
        ensure_stats(stats)
    }
}

unsafe fn ensure_stats(
    stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    if stats.is_null() {
        unsafe {
            pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBulkDeleteResult>())
                as *mut pg_sys::IndexBulkDeleteResult
        }
    } else {
        stats
    }
}

// ---------------------------------------------------------------------------
// amcostestimate / amoptions / amvalidate
// ---------------------------------------------------------------------------

/// The planner's WARNING for a native index below format 3: once per backend and index (the
/// planner asks for a cost many times per statement).
unsafe fn warn_stale_index_once(index: pg_sys::Relation, stored: crate::local::metapage::Stored) {
    use std::{cell::RefCell, collections::HashSet};
    thread_local! {
        static WARNED: RefCell<HashSet<u32>> = RefCell::new(HashSet::new());
    }
    let first = WARNED.with_borrow_mut(|warned| warned.insert(unsafe { (*index).rd_id }.to_u32()));
    if first {
        unsafe { crate::local::metapage::warn_old_format(index, stored) };
    }
}

/// Mode-aware cost estimate — dispatches on the index's own metapage
/// mode (`local::metapage::read`, `rd_amcache`-cached) and prices each mode
/// per `local::cost`'s model: LOCAL from the measured cost structure
/// (probe pages + near-free Hamming + the dominant rescore-pool heap
/// fetches, Mackert–Lohman-estimated); REMOTE with the 0.3 shape (high
/// constant startup ≈ one HTTP round trip — the earlier constant, unchanged —
/// plus a network-bound per-tuple term). `Promoting` prices as LOCAL and
/// `Demoting` as REMOTE — cost follows the mode a scan would actually SERVE
/// (`ambeginscan`'s dispatch). Flat (small-N) local indexes are deliberately
/// priced ABOVE the equivalent seqscan (see `local::cost`'s module docs) so
/// tiny tables keep seqscanning at `enable_seqscan = on` — the correct plan —
/// while a realistic clustered index now wins the comparison on its own,
/// with no `enable_seqscan = off` crutch.
///
/// `pinecone.am_enabled = off` still reports [`pg_sys::disable_cost`] (the
/// planner's standard "never pick this path" signal), and so does a path
/// with NO order-by operator: with `amcanmulticol` the planner
/// legitimately builds index paths whose only payload is the filter column's
/// equality clause — a pure `WHERE tenant_id = 5` with no `ORDER BY` at all.
/// This AM cannot serve that plan correctly (rows whose VECTOR column is
/// NULL are not in the index at all, so a keys-only index scan would
/// silently drop them — a wrong-results bug, not a performance choice; a
/// btree on the filter column is the right tool). The `ambeginscan`
/// orderby-count check remains as the loud backstop if something (e.g.
/// every other plan also disabled) picks it anyway. A zero-block index
/// relation (nothing valid to read a metapage from) is disabled too, rather
/// than guessed at.
///
/// Opens the index relation by oid to reach its metapage — the same pattern
/// pgvector's own `ivfflatcostestimate` uses to read its `lists` count;
/// cheap after the first call per relcache lifetime (`rd_amcache`).
#[pg_guard]
#[allow(clippy::too_many_arguments)] // Postgres's amcostestimate_function C signature, not ours to shrink
unsafe extern "C-unwind" fn amcostestimate(
    root: *mut pg_sys::PlannerInfo,
    path: *mut pg_sys::IndexPath,
    _loop_count: f64,
    index_startup_cost: *mut pg_sys::Cost,
    index_total_cost: *mut pg_sys::Cost,
    index_selectivity: *mut pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    unsafe {
        *index_selectivity = 1.0;
        *index_correlation = 0.0;
        *index_pages = 0.0;

        let disable = |startup: *mut pg_sys::Cost, total: *mut pg_sys::Cost| {
            *startup = pg_sys::disable_cost;
            *total = pg_sys::disable_cost;
            // PG18 no longer adds `disable_cost` for `enable_* = off`: it counts disabled nodes
            // ahead of cost, so only this keeps the path behind a disabled seqscan.
            #[cfg(feature = "pg18")]
            {
                (*path).path.disabled_nodes += 1;
            }
        };
        if !AM_ENABLED.get() || (*path).indexorderbys.is_null() {
            disable(index_startup_cost, index_total_cost);
            return;
        }

        let indexinfo = (*path).indexinfo;
        let index_relid = (*indexinfo).indexoid;
        let index_rel =
            pg_sys::index_open(index_relid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        let index_blocks =
            pg_sys::RelationGetNumberOfBlocksInFork(index_rel, pg_sys::ForkNumber::MAIN_FORKNUM);
        if index_blocks == 0 {
            // No metapage to read (e.g. a mid-DDL state) — never guess.
            pg_sys::index_close(index_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
            disable(index_startup_cost, index_total_cost);
            return;
        }
        let meta = crate::local::metapage::read(index_rel);
        // A native index below format 3 cannot serve a scan: the disabled cost sends the planner to
        // another path (a seqscan), with one WARNING per backend and index. Only the cost is
        // disabled, not a PG18 `disabled_nodes` count, so `enable_seqscan = off` still reaches
        // `ambeginscan` and its `55000`, as on PG17.
        if meta.needs_reindex() {
            warn_stale_index_once(index_rel, meta.stored());
            pg_sys::index_close(index_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
            *index_startup_cost = pg_sys::disable_cost;
            *index_total_cost = pg_sys::disable_cost;
            return;
        }
        pg_sys::index_close(index_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);

        let costs = crate::local::cost::PlannerCosts::current();
        use crate::local::metapage::Mode;
        match meta.mode {
            Mode::Local | Mode::Promoting => {
                let rel = (*indexinfo).rel;
                let planner_rows = (*rel).tuples;
                let heap_pages = (*rel).pages as f64;
                let rows = if planner_rows > 0.0 {
                    planner_rows
                } else {
                    meta.total_vectors as f64 + meta.inserted_since_build as f64
                };
                // A pushed key (an `indexclauses` entry) scans with the 0.9.0 pool and the ramped lists.
                let pushed_key = !(*path).indexclauses.is_null();
                let (pool_rows, nprobe) =
                    crate::local::scan::planned_pool_and_nprobe(&meta, pushed_key);
                let pool = crate::local::cost::rescore_pool(pool_rows, rows);
                // The planner's own Mackert–Lohman estimator — how many
                // physical page reads `pool` random heap fetches actually
                // cost, given cache effects (the same call
                // genericcostestimate makes).
                let rescore_pages_fetched = pg_sys::index_pages_fetched(
                    pool,
                    (*rel).pages,
                    (*indexinfo).pages as f64,
                    root,
                );
                let inputs = crate::local::cost::local_inputs(
                    &meta,
                    planner_rows,
                    heap_pages,
                    rescore_pages_fetched,
                    pool_rows,
                    nprobe,
                );
                let (startup, total, pages) = crate::local::cost::local_cost(&inputs, &costs);
                *index_startup_cost = startup;
                *index_total_cost = total;
                *index_pages = pages;
            }
            Mode::Remote | Mode::Demoting | Mode::Detached => {
                let (startup, total) =
                    crate::local::cost::remote_cost(AM_MIN_TOPK.get() as f64, &costs);
                *index_startup_cost = startup;
                *index_total_cost = total;
            }
        }
    }
}

#[pg_guard]
unsafe extern "C-unwind" fn amoptions(
    reloptions: pg_sys::Datum,
    validate: bool,
) -> *mut pg_sys::bytea {
    unsafe { crate::local::options::amoptions(reloptions, validate) }
}

#[pg_guard]
unsafe extern "C-unwind" fn amvalidate(_opclassoid: pg_sys::Oid) -> bool {
    true
}

// ---------------------------------------------------------------------------
// Scan: ambeginscan / amrescan / amgettuple / amendscan
// ---------------------------------------------------------------------------

/// The remote-mode scan state — unchanged from 0.3, just renamed (it used to
/// be the only variant of a struct called `ScanState`; see [`ScanState`]).
struct RemoteScanState {
    /// `None` until the first `amgettuple` call issues the initial `/query`.
    /// `(tid, Pinecone score)` in remote order.
    matches: Option<Vec<(pg_sys::ItemPointerData, f32)>>,
    cursor: usize,
    /// `topK` of the most recent `/query` (0 before the first one).
    current_topk: u32,
    /// Whether the most recent `/query` returned fewer matches than its
    /// `topK` — i.e. the remote namespace is exhausted, so a drained
    /// `matches` means the scan is over, not "grow and re-query".
    exhausted: bool,
    /// Every TID already handed back this scan — a growing re-query
    /// (re-query topK×4, dedupe client-side) overlaps
    /// entirely with what a smaller `topK` already returned, so the larger
    /// result must be filtered against this set before being served, or
    /// already-returned rows would come back a second time.
    seen: std::collections::HashSet<(pg_sys::BlockNumber, pg_sys::OffsetNumber)>,
    /// `pinecone.iterative_scan` as of the scan's first `/query`.
    mode: IterativeScan,
    /// The table's registry metric (`cosine`/`l2`/`ip`): how `strict_order`
    /// reads a score.
    metric: String,
    /// `strict_order`: distance of the last row served.
    last_dist: f64,
    /// The index's filter key kind and heap column (`FilterKind::None`: unfiltered).
    filter_kind: crate::local::metapage::FilterKind,
    filter_attnum: i16,
    /// The equality key of this scan, decoded once per [`amrescan`].
    key: KeyDecode,
    /// Whether the executor must recheck every row against the original quals (see
    /// [`EqKey::inexact_in_pinecone`]).
    recheck: bool,
    /// The index is partitioned: every row is rechecked, since a stale entry whose TID VACUUM
    /// recycled for another tenant's row must never reach the wrong tenant. Known after the
    /// first `/query`.
    partitioned: bool,
}

/// A scan's opaque state, mode-dispatched once at [`ambeginscan`] (from the
/// metapage — [`crate::local::metapage::read`]) and never re-dispatched for
/// the life of the scan; [`amrescan`] resets whichever variant is active in
/// place rather than re-reading the metapage (a rescan changes the query
/// vector/orderby, never the index's own mode).
enum ScanState {
    Remote(RemoteScanState),
    Local(crate::local::scan::LocalScanState),
}

#[pg_guard]
unsafe extern "C-unwind" fn ambeginscan(
    index: pg_sys::Relation,
    nkeys: std::ffi::c_int,
    norderbys: std::ffi::c_int,
) -> pg_sys::IndexScanDesc {
    unsafe {
        // Exactly one ORDER BY operator, always — the vector opclasses
        // define a single ordering operator, and `amcostestimate` disables
        // every orderby-less path (see its rustdoc: a keys-only scan would
        // silently drop NULL-vector rows), so the planner should never build
        // a scan that trips this. Checked here, once, at scan setup, rather
        // than lazily on the first `amgettuple` call — the loud backstop if
        // a plan with every alternative disabled picks this index anyway.
        if norderbys != 1 {
            error!(
                "pg_pinecone: the pg_pinecone access method only supports plans with exactly one ORDER BY \
                 <column> <op> <vector> clause, got {norderbys} — for a filtered query keep the \
                 ORDER BY and add the equality condition on the index's filter column"
            );
        }
        let scan = pg_sys::RelationGetIndexScan(index, nkeys, norderbys);
        let meta = crate::local::metapage::read(index);
        // Forced or chosen, a scan of a native index below format 3 is an error, not a wrong answer.
        if meta.needs_reindex() {
            crate::local::metapage::refuse_old_format(index, meta.stored());
        }
        let mode = meta.mode;
        // A local-mode index has no registry row, hence no partition; every other mode (the local
        // scan of a promoting index included) is bound to one namespace per query.
        if nkeys == 0 && meta.has_filter() && mode != crate::local::metapage::Mode::Local {
            am_partition::require_key((*(*index).rd_index).indrelid);
        }
        // `Promoting` serves LOCAL (never-partial-namespace: the backfill to
        // remote hasn't finished, so remote is incomplete); `Demoting` serves
        // REMOTE (never-unavailable: the local rebuild hasn't finished, so
        // local doesn't exist/isn't authoritative yet) — see `am.rs`'s module
        // docs and `lifecycle.rs` for the full promote_index/demote_index
        // contract these dispatch decisions implement.
        let state = match mode {
            crate::local::metapage::Mode::Local | crate::local::metapage::Mode::Promoting => {
                ScanState::Local(crate::local::scan::LocalScanState::new())
            }
            crate::local::metapage::Mode::Remote
            | crate::local::metapage::Mode::Demoting
            | crate::local::metapage::Mode::Detached => ScanState::Remote(RemoteScanState {
                matches: None,
                cursor: 0,
                current_topk: 0,
                exhausted: false,
                seen: std::collections::HashSet::new(),
                mode: IterativeScan::Off,
                metric: String::new(),
                last_dist: f64::NEG_INFINITY,
                filter_kind: meta.filter_kind,
                filter_attnum: meta.filter_heap_attnum,
                key: KeyDecode::Unkeyed,
                recheck: false,
                partitioned: false,
            }),
        };
        // A plain `Box::into_raw` (this allocation's previous form) is
        // invisible to Postgres's memory-context machinery: if the scan
        // errors out mid-way (e.g. `amgettuple`'s malformed-remote-id or
        // am_max_topk-exceeded errors) rather than reaching a normal
        // `amendscan`, that raw heap pointer — and the `HashSet`'s own
        // separate heap allocation inside it — would leak for the rest of
        // the backend's life. `leak_and_drop_on_delete` instead registers a
        // reset callback on the context active right now (whatever
        // `ambeginscan` is invoked under, which is always at most
        // query/portal-lifetime) that runs `ScanState`'s `Drop` — freeing
        // the `HashSet` too — the moment that context is reset or deleted,
        // on *any* exit path, not just a clean `amendscan`. See `amendscan`
        // below for why it must not also free this.
        (*scan).opaque =
            PgMemoryContexts::CurrentMemoryContext.leak_and_drop_on_delete(state) as *mut c_void;
        scan
    }
}

#[pg_guard]
unsafe extern "C-unwind" fn amrescan(
    scan: pg_sys::IndexScanDesc,
    keys: pg_sys::ScanKey,
    _nkeys: std::ffi::c_int,
    orderbys: pg_sys::ScanKey,
    norderbys: std::ffi::c_int,
) {
    unsafe {
        // The filter-column equality scankey arrives here — copied
        // into the scan's own keyData exactly like the orderbys below, so a
        // nested-loop rescan with fresh parameter values re-filters
        // correctly; `local::scan::extract_filter` decodes it on the next
        // amgettuple. A remote scan decodes it right below, once per rescan
        // ([`crate::am_filter::decode_keys`]).
        if !keys.is_null() && (*scan).numberOfKeys > 0 {
            std::ptr::copy(keys, (*scan).keyData, (*scan).numberOfKeys as usize);
        }
        if !orderbys.is_null() && (*scan).numberOfOrderBys > 0 {
            std::ptr::copy(
                orderbys,
                (*scan).orderByData,
                (*scan).numberOfOrderBys as usize,
            );
        }
        let _ = norderbys;
        let st = &mut *((*scan).opaque as *mut ScanState);
        match st {
            ScanState::Remote(r) => {
                r.matches = None;
                r.cursor = 0;
                r.current_topk = 0;
                r.exhausted = false;
                r.seen.clear();
                r.last_dist = f64::NEG_INFINITY;
                r.key = crate::am_filter::decode_keys(scan, r.filter_kind);
                r.recheck = matches!(&r.key, KeyDecode::Eq(key) if key.inexact_in_pinecone());
            }
            ScanState::Local(l) => l.reset(),
        }
    }
}

/// What one `/query` of a scan came back with.
struct RemoteAnswer {
    metric: String,
    partitioned: bool,
    matches: Vec<(pg_sys::ItemPointerData, f32)>,
}

/// Runs one `/query` against the registry row for `heap_relid`'s Pinecone
/// index/namespace, translating matches to `(ItemPointerData, score)` (item 4:
/// reject a malformed id with a clear error, never silently drop or panic on
/// it) and returning the table's metric alongside. A partitioned row reads the
/// namespace of the scan's key (and still sends it as the `$eq` filter).
fn remote_query(
    heap_relid: pg_sys::Oid,
    vector: Vec<f32>,
    top_k: u32,
    filter: Option<(&EqKey, &str)>,
) -> RemoteAnswer {
    let row = registry::lookup(heap_relid).unwrap_or_else(|| {
        error!(
            "pg_pinecone: index scan found no pinecone.tables row for relid {heap_relid:?} — should \
             be unreachable, ambuild always inserts one"
        )
    });
    let index_name = row
        .index_name
        .clone()
        .expect("pg_pinecone: an access-method row always has index_name");
    let partitioned = row.partition.is_some();
    let namespace = if partitioned {
        let prefix = row
            .namespace
            .as_deref()
            .expect("pg_pinecone: a partitioned row always has its namespace prefix");
        let Some((key, _)) = filter else {
            error!(
                "pg_pinecone: a scan of the partitioned index of relid {heap_relid:?} has no key — \
                 should be unreachable, ambeginscan refuses it"
            )
        };
        let namespace = am_partition::scan_namespace(prefix, key)
            .unwrap_or_else(|e| am_partition::raise_scan_namespace(heap_relid, &e));
        Some(namespace)
    } else {
        row.namespace.clone()
    };
    let metric = row.metric.clone();
    // The key goes to Pinecone under the name the writers synced it with. A rename since the
    // build leaves the metadata under the old name until a REINDEX: say so, don't miss rows.
    let filter = filter.map(|(key, current)| {
        let registered = row.filter_column.as_deref().unwrap_or_else(|| {
            error!(
                "pg_pinecone: index scan has a filter key but pinecone.tables has no filter column \
                 for relid {heap_relid:?} — should be unreachable"
            )
        });
        if registered != current {
            backend::error_with_hint(
                format!(
                    "pg_pinecone: filter column \"{current}\" was renamed — the index was built on \
                     \"{registered}\" and Pinecone holds its metadata under that name"
                ),
                "REINDEX the index to resync the metadata under the new name".to_string(),
            );
        }
        am_filter::eq_filter(registered, key)
    });

    let (scan_factor, max_candidates) = backend::drn_query_tuning();
    let request = QueryRequest {
        index: index_name,
        vector,
        top_k,
        namespace,
        filter,
        include_metadata: false,
        scan_factor,
        max_candidates,
    };
    let matches = backend::client_for_effective_key()
        .query(&request)
        .unwrap_or_else(|e| backend::raise(e));

    let matches = matches
        .into_iter()
        .map(|m| {
            let tid = id_to_tid(&m.id).unwrap_or_else(|e| error!("pg_pinecone: {e}"));
            (tid, m.score)
        })
        .collect();
    RemoteAnswer {
        metric,
        partitioned,
        matches,
    }
}

#[pg_guard]
unsafe extern "C-unwind" fn amgettuple(
    scan: pg_sys::IndexScanDesc,
    _direction: pg_sys::ScanDirection::Type,
) -> bool {
    unsafe {
        let st = &mut *((*scan).opaque as *mut ScanState);
        match st {
            ScanState::Remote(r) => amgettuple_remote(scan, r),
            ScanState::Local(l) => crate::local::scan::amgettuple(scan, l),
        }
    }
}

/// The remote-mode `amgettuple` body: serve `matches`, and when the executor
/// has drained them re-query from scratch with `topK` x4 (Pinecone has no
/// cursor), dropping the rows already served.
///
/// `pinecone.iterative_scan` decides what happens at the cap. `off` (the
/// legacy loop) clamps the last regrow to `pinecone.am_max_topk` and errors
/// once even that was not enough. `relaxed_order`/`strict_order` end the scan
/// quietly instead, at `pinecone.max_scan_tuples` rows fetched (`am_max_topk`
/// still bounds a single request: Pinecone rejects a larger `topK`), and
/// `strict_order` skips a re-queried row that ranks before the last one served
/// — Pinecone ranks server-side, so that rarely fires. The first `/query` is
/// never cut by `max_scan_tuples`, as on the native engine.
unsafe fn amgettuple_remote(scan: pg_sys::IndexScanDesc, st: &mut RemoteScanState) -> bool {
    unsafe {
        // A NULL or contradictory key matches no row: nothing to ask Pinecone.
        if st.key == KeyDecode::Nothing {
            return false;
        }
        loop {
            if let Some(matches) = &st.matches {
                while st.cursor < matches.len() {
                    let (tid, score) = matches[st.cursor];
                    st.cursor += 1;
                    if st.mode == IterativeScan::StrictOrder {
                        let dist = crate::topk::score_to_distance(&st.metric, score);
                        if dist < st.last_dist {
                            // Dropped for good: a larger query would rank it
                            // the same way.
                            st.seen.insert(item_pointer_get_both(tid));
                            continue;
                        }
                        st.last_dist = dist;
                    }
                    st.seen.insert(item_pointer_get_both(tid));
                    (*scan).xs_heaptid = tid;
                    (*scan).xs_recheck = st.recheck || st.partitioned;
                    (*scan).xs_recheckorderby = false;
                    return true;
                }
                if st.exhausted {
                    return false;
                }
                let max_topk = AM_MAX_TOPK.get() as u32;
                if st.mode == IterativeScan::Off {
                    if st.current_topk >= max_topk {
                        error!(
                            "pg_pinecone: index scan needed more than pinecone.am_max_topk ({max_topk}) \
                             remote matches to satisfy this query — raise pinecone.am_max_topk if this \
                             is expected (e.g. a very selective WHERE filter)"
                        );
                    }
                    st.current_topk = st.current_topk.saturating_mul(4).min(max_topk);
                } else {
                    let cap = (max_scan_tuples().min(u64::from(max_topk))) as u32;
                    if st.current_topk >= cap {
                        crate::local::scan::record_scan_capped();
                        pgrx::debug1!(
                            "pg_pinecone: iterative remote scan stopped at {} rows (pinecone.max_scan_tuples \
                             / pinecone.am_max_topk)",
                            st.current_topk
                        );
                        return false;
                    }
                    st.current_topk = st.current_topk.saturating_mul(4).min(cap);
                }
            } else {
                // Clamped against `am_max_topk` too (not just later regrow
                // attempts): a misconfiguration with `am_min_topk >
                // am_max_topk` must still issue a satisfiable first query
                // rather than immediately exceed the cap it was just about
                // to be checked against.
                st.current_topk = (AM_MIN_TOPK.get() as u32).min(AM_MAX_TOPK.get() as u32);
                st.mode = iterative_scan().for_remote();
                crate::local::scan::reset_scan_stats();
            }

            // `ambeginscan` already rejects anything but exactly one ORDER
            // BY operator for the life of this scan.
            let ob = &*(*scan).orderByData;
            if ob.sk_flags & pg_sys::SK_ISNULL as i32 != 0 {
                return false;
            }
            let query_vector = parse_vector(ob.sk_argument);
            let heap_relid = (*(*scan).heapRelation).rd_id;

            let filter = match &st.key {
                KeyDecode::Eq(key) => {
                    let heap = (*scan).heapRelation;
                    Some((key, heap_column_name(heap, st.filter_attnum)))
                }
                _ => None,
            };
            let RemoteAnswer {
                metric,
                partitioned,
                matches: results,
            } = remote_query(
                heap_relid,
                query_vector,
                st.current_topk,
                filter.as_ref().map(|(key, name)| (*key, name.as_str())),
            );
            st.metric = metric;
            st.partitioned = partitioned;
            crate::local::scan::record_remote_query(u64::from(st.current_topk));
            // Exhaustion is judged against the raw (pre-dedup) count: a
            // remote namespace with fewer live vectors than `current_topk`
            // is exhausted regardless of how many of those happen to be
            // rows we've already returned.
            st.exhausted = (results.len() as u32) < st.current_topk;
            let fresh: Vec<(pg_sys::ItemPointerData, f32)> = results
                .into_iter()
                .filter(|(tid, _)| !st.seen.contains(&item_pointer_get_both(*tid)))
                .collect();
            st.matches = Some(fresh);
            st.cursor = 0;
        }
    }
}

/// Deliberately does nothing: `ambeginscan` allocates `ScanState` via
/// `PgMemoryContexts::leak_and_drop_on_delete`, which already drops it (thus
/// freeing its `HashSet`'s own heap allocation) when the memory context it
/// was allocated in is reset or deleted — including on a mid-scan `ERROR`
/// that skips this function entirely. Also `Box::from_raw`-ing and dropping
/// `(*scan).opaque` here, on top of that, would double-free it once that
/// same reset callback later fires on the ordinary (non-error) path too.
#[pg_guard]
unsafe extern "C-unwind" fn amendscan(_scan: pg_sys::IndexScanDesc) {}

#[cfg(test)]
mod tests {
    use super::{IterativeScan, plain_storage_advice};

    #[test]
    fn auto_is_off_for_a_remote_index_and_explicit_modes_stay_as_set() {
        assert_eq!(IterativeScan::Auto.for_remote(), IterativeScan::Off);
        for explicit in [
            IterativeScan::Off,
            IterativeScan::RelaxedOrder,
            IterativeScan::StrictOrder,
        ] {
            assert_eq!(explicit.for_remote(), explicit);
        }
    }

    #[test]
    fn external_and_extended_vectors_of_toasted_width_get_advice() {
        assert_eq!(plain_storage_advice('e', 768), Some("EXTERNAL"));
        assert_eq!(plain_storage_advice('x', 1536), Some("EXTENDED"));
    }

    #[test]
    fn plain_and_main_columns_get_none() {
        assert_eq!(plain_storage_advice('p', 768), None);
        assert_eq!(plain_storage_advice('m', 768), None);
    }

    #[test]
    fn a_vector_below_the_toast_threshold_is_inline_anyway() {
        assert_eq!(plain_storage_advice('e', 128), None);
        assert_eq!(plain_storage_advice('e', 500), None);
        assert_eq!(plain_storage_advice('e', 501), Some("EXTERNAL"));
    }

    #[test]
    fn a_vector_that_cannot_share_a_page_with_a_row_gets_none() {
        assert_eq!(plain_storage_advice('e', 1997), Some("EXTERNAL"));
        assert_eq!(plain_storage_advice('e', 1998), None);
        assert_eq!(plain_storage_advice('e', 3072), None);
    }
}
