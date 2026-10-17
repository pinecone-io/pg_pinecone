//! `ambuild`'s local-mode path. Block layout is fixed and computed entirely
//! up front, before any page is written, so every page is written exactly
//! once, in strictly increasing block order:
//! - block 0: the metapage ([`metapage::write`]).
//! - block 1 (FILTERED clustered builds only): the filter-stats page
//!   ([`super::stats`]).
//! - next `D` blocks: the `D`-page centroid directory
//!   ([`pages::dir_page_count`]).
//! - remaining blocks: each list's page chain, lists in order `0..n_lists`,
//!   empty lists contributing zero pages.
//!
//! This relies on `ReadBufferExtended(index, forknum, P_NEW, ...)` handing
//! out strictly sequential block numbers for a relation with no concurrent
//! writers — true here: `ambuild` runs under the exclusive lock `CREATE
//! INDEX`/`REINDEX` already holds on a brand-new (zero-page) index relation.
//! That determinism is what lets the centroid directory's `list_head`
//! pointers be computed once, arithmetically, before any list page exists —
//! no second "patch up the directory" pass needed.
//!
//! Every row is encoded (RaBitQ, `rabitq::encode_into`) against the centroid it is assigned to; a
//! cosine index encodes the unit vector `x / |x|` (k-means is spherical already). Codes depend on
//! nothing but `(seed, row, centroid)`, so any `build_parallelism` writes the same bytes.
//!
//! Each page is its own `GenericXLogStart`/`Finish` pair (one page per WAL
//! record — trivially inside `MAX_GENERIC_XLOG_PAGES`'s 4-page/record cap),
//! the same pattern `metapage::write`/0.3's original `write_metapage` already
//! used for the single-page case. A future optimization could batch large builds the way
//! core btree/gin avoid per-page WAL entirely under `wal_level = minimal`
//! (`RelationNeedsWAL`) — deferred; correctness, not throughput, is the bar here.

use std::borrow::Cow;
use std::{ffi::c_void, thread};

use pgrx::prelude::*;

use super::{
    calibrate,
    kmeans::{self, KMeans},
    metapage::{self, FilterKind, INVALID_BLOCK, MetaPage, Mode, QueriesSource},
    metric::{Metric, normalise_into},
    options::Options,
    pages::{self, ListBody, ListRecord},
    queries::{self, Draw, Drawn},
    rabitq::{self, Code, EncodeScratch},
    rotation::Rotation,
    scan::ASSUMED_TOP_K,
    stats::FilterStats,
};

const P_NEW: pg_sys::BlockNumber = 0xFFFFFFFF;

/// The one optional filter key column a composite index carries
/// (`CREATE INDEX ... USING pinecone (embedding <opclass>, tenant_id)`) —
/// resolved and validated by `am::ambuild` before this module runs.
pub(crate) struct FilterSpec {
    /// The filter column's HEAP attribute number.
    pub(crate) heap_attnum: i16,
    /// Never [`FilterKind::None`] — a `FilterSpec` only exists for a real
    /// filter column.
    pub(crate) kind: FilterKind,
}

/// One heap tuple as collected by [`collect_build_callback`]: TID, vector,
/// and — when the index has a filter column — that column's stored value
/// (`0` + `fnull = true` for SQL NULL; see `pages.rs`'s filter-array docs).
struct CollectedRow {
    tid: pg_sys::ItemPointerData,
    vector: Vec<f32>,
    fval: i64,
    fnull: bool,
}

struct CollectState {
    rows: Vec<CollectedRow>,
    filter_kind: Option<FilterKind>,
}

#[pg_guard]
unsafe extern "C-unwind" fn collect_build_callback(
    _index: pg_sys::Relation,
    tid: pg_sys::ItemPointer,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    _tuple_is_alive: bool,
    state: *mut c_void,
) {
    unsafe {
        // NULL vectors are skipped entirely — nothing to cluster or score
        // them against (matches `am::aminsert`'s own NULL handling). A NULL
        // FILTER value is the opposite: the row is fully indexed (it must
        // appear in unfiltered scans) and merely never matches an equality
        // scankey (`fnull` below).
        if *isnull {
            return;
        }
        let state = &mut *(state as *mut CollectState);
        let vector = crate::am::parse_vector(*values);
        reject_non_finite(&vector, *tid);
        let (fval, fnull) = match state.filter_kind {
            None => (0, false),
            Some(kind) => {
                if *isnull.add(1) {
                    (0, true)
                } else {
                    (kind.datum_to_value(*values.add(1)), false)
                }
            }
        };
        state.rows.push(CollectedRow {
            tid: *tid,
            vector,
            fval,
            fnull,
        });
    }
}

/// Raises for a vector with a NaN or infinite component, which the native engine cannot encode:
/// pgvector stores finite components only. (A finite vector whose square overflows `f32`, 1e20 per
/// component, is stored: its code has an infinite `rnorm` and a zero `dot`, its keys are NaN, and
/// `total_cmp` sorts them last, so it is never a candidate and never an error.)
fn reject_non_finite(vector: &[f32], tid: pg_sys::ItemPointerData) {
    if let Some(at) = super::metric::first_non_finite(vector) {
        let (block, offset) = pgrx::itemptr::item_pointer_get_both(tid);
        error!(
            "pg_pinecone: the vector at heap tuple ({block},{offset}) has a non-finite value (NaN or \
             Infinity) in component {}: the native engine cannot encode it",
            at + 1
        );
    }
}

/// A pseudo-random, index-specific default for `seed` when the `seed`
/// reloption is left at its `0` ("auto") default — combines the wall clock
/// with the new index's own oid so two indexes built back-to-back (even in
/// the same test run) don't silently share a rotation. Not
/// cryptographically meaningful and not required to be: any fixed draw
/// satisfies the random-rotation guarantees equally — this only needs to
/// *vary*, not to be secret.
fn derive_seed(index_relid: pg_sys::Oid) -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ (index_relid.to_u32() as u64).wrapping_mul(0x9E3779B97F4A7C15)
}

/// The local-mode `ambuild` path — see this module's docstring for the block
/// layout. Returns `(heap_tuples, index_tuples)` for `IndexBuildResult`.
#[allow(clippy::too_many_arguments)] // one arg per ambuild-resolved build input
pub(crate) unsafe fn ambuild_local(
    heap: pg_sys::Relation,
    index: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
    heap_attnum: i16,
    dim: u32,
    metric: Metric,
    opts: &Options,
    filter: Option<FilterSpec>,
) -> (f64, f64) {
    // Dim-cap validation (Part A quick win #3) — REGARDLESS of row count,
    // unlike the pre-existing check this replaces (which only ran once the
    // build had already decided NOT to go flat, i.e. only for tables at or
    // above `small_n_threshold`). A small table that starts flat today can
    // still cross into IVF mode on a later `REINDEX` once it grows past that
    // threshold, and that future build would hit this exact page-layout
    // constraint then — surfacing it here, at CREATE INDEX time, on every
    // build regardless of size, means the error is discoverable up front
    // instead of on some future maintenance REINDEX. Checked before the
    // (potentially large) heap scan below, too, so a doomed build fails
    // without paying for it. See `pages::max_centroid_dimension`'s own docs
    // for the exact page-format constraint this enforces.
    if pages::dir_entries_per_page(dim as usize) == 0 {
        error!(
            "pg_pinecone: local mode supports at most {} dimensions (a centroid must fit one \
             {}-byte page); column has {dim}",
            pages::max_centroid_dimension(),
            pg_sys::BLCKSZ,
        );
    }

    let (rotation_rounds, bq_bits, code_bits) = metapage::native_codec(dim);
    if opts.k_bits != 0 {
        notice!(
            "pg_pinecone: the k_bits reloption ({}) is ignored: codes are D bits, the dimension padded \
             up to a multiple of 64 ({code_bits} for this column)",
            opts.k_bits
        );
    }

    // REINDEX, VACUUM FULL, CLUSTER and TRUNCATE rebuild through `reindex_index`, which marks the index;
    // CREATE INDEX does not (nor does ALTER TABLE .. TYPE, which re-creates indexes through it, nor a
    // TRUNCATE of a table created in the same transaction), and the concurrent forms are refused for a
    // native index before this runs.
    let rebuild = unsafe { pg_sys::ReindexIsProcessingIndex((*index).rd_id) };

    let mut state = CollectState {
        rows: Vec::new(),
        filter_kind: filter.as_ref().map(|f| f.kind),
    };
    unsafe {
        pg_sys::table_index_build_scan(
            heap,
            index,
            index_info,
            true,
            true,
            Some(collect_build_callback),
            &mut state as *mut CollectState as *mut c_void,
            std::ptr::null_mut(),
        );
    }
    let collected = state.rows;

    let total = collected.len();
    if u32::try_from(total).is_err() {
        error!(
            "pg_pinecone: a native index holds at most {} rows, the table has {total}",
            u32::MAX
        );
    }
    let is_flat = total == 0 || total < opts.small_n_threshold as usize;
    let (filter_kind, filter_heap_attnum) = match &filter {
        Some(f) => (f.kind, f.heap_attnum),
        None => (FilterKind::None, 0),
    };

    if is_flat {
        if opts.calibration_queries.is_some() && !rebuild {
            notice!(
                "pg_pinecone: calibration_queries is not used: the index is flat ({total} rows, \
                 fewer than small_n_threshold = {})",
                opts.small_n_threshold
            );
        }
        // A flat (small-N bootstrap) index stores no per-record pages at all
        // — filtered or not — so it needs no stats page either: its scan
        // path reads the filter column straight off live heap tuples
        // (`scan.rs`'s flat path), exact by construction.
        unsafe {
            metapage::write(
                index,
                &MetaPage {
                    mode: Mode::Local,
                    metric,
                    is_flat: true,
                    heap_attnum,
                    dim,
                    n_lists: 0,
                    format: metapage::FORMAT_VERSION,
                    layout: metapage::LAYOUT_VERSION,
                    rotation_rounds,
                    bq_bits,
                    code_bits,
                    rescore_multiplier: opts.rescore_multiplier,
                    small_n_threshold: opts.small_n_threshold,
                    calib_nprobe: 0,
                    calib_pool: 0,
                    calib_samples: 0,
                    calib_reached: false,
                    calib_recall: 0.0,
                    calib_queries: QueriesSource::Rows,
                    target_recall: 0.0,
                    total_vectors: total as u32,
                    seed: 0,
                    centroid_dir_start: INVALID_BLOCK,
                    filter_kind,
                    filter_heap_attnum,
                    stats_start: INVALID_BLOCK,
                    inserted_since_build: 0,
                },
            );
        }
        return (total as f64, total as f64);
    }

    let seed = if opts.seed == 0 {
        derive_seed(unsafe { (*index).rd_id })
    } else {
        opts.seed as u64
    };
    // Before k-means, so a table of queries that cannot be used fails a CREATE INDEX early.
    let (drawn, queries_source) = if total < calibrate::MIN_ROWS {
        if opts.calibration_queries.is_some() && !rebuild {
            notice!(
                "pg_pinecone: calibration_queries is not used: calibrating needs at least {} rows, \
                 the table has {total}",
                calibrate::MIN_ROWS
            );
        }
        (None, QueriesSource::Rows)
    } else {
        match queries::draw(
            opts.calibration_queries.as_deref(),
            heap,
            heap_attnum,
            dim,
            seed,
            calibrate::query_sample_size(total),
            rebuild,
        ) {
            Draw::Unset => (None, QueriesSource::Rows),
            Draw::Fallback => (None, QueriesSource::Fallback),
            Draw::Drawn(drawn) => (Some(drawn), QueriesSource::Table),
        }
    };
    let n_lists_requested = if opts.n_lists == 0 {
        kmeans::default_n_lists(total)
    } else {
        opts.n_lists
    };
    // C7 parbuild: worker-thread budget for the two O(N * n_lists * dim)-ish
    // phases below (k-means training + the final per-row assign/encode
    // pass) — a pure CPU-budget knob, resolved once per build; see
    // `options::build_parallelism`'s own rustdoc for the default/bounds and
    // why it's a session GUC, not a persisted reloption.
    let parallelism = crate::local::options::build_parallelism();

    // Borrowed slices into `collected`'s own `Vec<f32>`s — NOT a clone of
    // the underlying float data (this used to be
    // `collected.iter().map(|(_, v)| v.clone()).collect()`, an O(N *
    // dim) copy of the entire build's vectors on every `ambuild`; see
    // `KMeans::train`'s own rustdoc for why this is now generic enough to
    // accept borrows directly).
    let vector_refs: Vec<&[f32]> = collected.iter().map(|r| r.vector.as_slice()).collect();
    let km = KMeans::train(
        &vector_refs,
        dim as usize,
        n_lists_requested,
        metric,
        seed,
        parallelism,
    );
    let n_lists = km.centroids.len();

    let rotation = Rotation::new(seed, dim as usize);
    let code_bytes = pages::code_bytes(dim as usize);
    let has_filter = filter.is_some();

    // Assign every collected vector (not just the k-means sample) to its
    // nearest final centroid and encode it against that centroid, in parallel
    // (`assign_rows_parallel`) — each row's (cluster, code, rnorm, dot, rc) is
    // a pure function of that row's own vector plus the now-fixed
    // `km`/`rotation` state, so this needs none of `local::kmeans::update_step`'s
    // fixed-chunk-order care: computing it out of order is fine, since nothing
    // is summed across rows here. What DOES have to stay in order is which row
    // lands at which OFFSET within its list's page chain — `scan.rs`'s rescore
    // and `vacuum.rs`'s sweep don't care about intra-list order, but two builds
    // of the same data must still land identical bytes on disk (this module's
    // own determinism contract) — so the per-list row lists below are filled by
    // a single, ordinary sequential pass over `collected` in its original order.
    let assigned = assign_rows_parallel(&collected, &km, &rotation, metric, parallelism);
    let mut lists: Vec<Vec<u32>> = (0..n_lists).map(|_| Vec::new()).collect();
    for (row, &cluster) in assigned.cluster.iter().enumerate() {
        lists[cluster as usize].push(row as u32);
    }
    let list_counts: Vec<usize> = lists.iter().map(Vec::len).collect();

    // Plan the block layout up front (see module docstring): the optional
    // stats page sits at block 1, before the directory, so the directory's
    // start (and every list head) shifts by exactly `stats_pages`.
    let stats_pages = has_filter as usize;
    let dir_start = 1 + stats_pages as u32;
    let d_pages = pages::dir_page_count(n_lists, dim as usize);
    let mut list_heads = vec![INVALID_BLOCK; n_lists];
    let mut next_block: u32 = dir_start + d_pages as u32;
    for (i, &count) in list_counts.iter().enumerate() {
        let pc = pages::list_page_count(count, code_bytes, has_filter);
        if pc > 0 {
            list_heads[i] = next_block;
            next_block += pc as u32;
        }
    }

    let calibration = calibrate_defaults(
        opts,
        metric,
        seed,
        &vector_refs,
        &km,
        &lists,
        &assigned,
        parallelism,
        drawn.as_ref(),
    );
    let (calib_nprobe, calib_pool, calib_samples, calib_reached, calib_recall, calib_queries) =
        match &calibration {
            Some(c) => (
                c.nprobe,
                c.pool,
                c.samples,
                c.reached,
                c.recall as f32,
                queries_source,
            ),
            None => (0, 0, 0, false, 0.0, QueriesSource::Rows),
        };

    unsafe {
        metapage::write(
            index,
            &MetaPage {
                mode: Mode::Local,
                metric,
                is_flat: false,
                heap_attnum,
                dim,
                n_lists: n_lists as u32,
                format: metapage::FORMAT_VERSION,
                layout: metapage::LAYOUT_VERSION,
                rotation_rounds,
                bq_bits,
                code_bits,
                rescore_multiplier: opts.rescore_multiplier,
                small_n_threshold: opts.small_n_threshold,
                calib_nprobe,
                calib_pool,
                calib_samples,
                calib_reached,
                calib_recall,
                calib_queries,
                target_recall: opts.target_recall as f32,
                total_vectors: total as u32,
                seed,
                centroid_dir_start: dir_start,
                filter_kind,
                filter_heap_attnum,
                stats_start: if has_filter { 1 } else { INVALID_BLOCK },
                inserted_since_build: 0,
            },
        );

        if has_filter {
            let stats = FilterStats::compute(collected.iter().map(|r| (r.fval, r.fnull)));
            write_page(index, &stats.page_body());
        }

        write_dir_pages(
            index,
            dim as usize,
            &km.centroids,
            &list_heads,
            &list_counts,
            d_pages,
            dir_start,
        );

        let source = ListSource {
            collected: &collected,
            assigned: &assigned,
            code_bytes,
            has_filter,
        };
        for (i, rows) in lists.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            write_list_pages(index, &source, list_heads[i], rows);
        }
    }

    (total as f64, total as f64)
}

/// The sample the defaults are calibrated on (the `drawn` queries, else table rows), and what it
/// found: `None` when the table is too small to calibrate (`calibrate::MIN_ROWS`). Says what it chose
/// in a NOTICE.
#[allow(clippy::too_many_arguments)] // one arg per build-resolved input
fn calibrate_defaults(
    opts: &Options,
    metric: Metric,
    seed: u64,
    vectors: &[&[f32]],
    km: &KMeans,
    lists: &[Vec<u32>],
    assigned: &Assigned,
    parallelism: usize,
    drawn: Option<&Drawn>,
) -> Option<calibrate::Outcome> {
    let started = std::time::Instant::now();
    let built = calibrate::Built {
        metric,
        seed,
        kernel: super::scan::kernel_for_scan(),
        vectors,
        centroids: &km.centroids,
        lists,
        codes: &assigned.codes,
        rnorm: &assigned.rnorm,
        dot: &assigned.dot,
        rc: &assigned.rc,
    };
    let pinned_pool =
        (opts.rescore_multiplier > 0).then(|| opts.rescore_multiplier * ASSUMED_TOP_K as u32);
    let settings = calibrate::Settings {
        target: opts.target_recall,
        pinned_pool,
        workers: parallelism,
    };
    let interrupt = || {
        pg_sys::check_for_interrupts!();
    };
    let outcome = match drawn {
        Some(drawn) => {
            let queries: Vec<&[f32]> = drawn.vectors.iter().map(Vec::as_slice).collect();
            calibrate::calibrate_on(&built, &settings, &queries, &interrupt)
        }
        None => calibrate::calibrate(&built, &settings, &interrupt),
    }?;
    let seconds = started.elapsed().as_secs_f64();
    let (target, n_lists) = (opts.target_recall, km.centroids.len());
    let sample = match drawn {
        Some(drawn) => format!("{} queries of {}", outcome.samples, drawn.table),
        None => format!("{} sample queries", outcome.samples),
    };
    if outcome.reached {
        notice!(
            "pg_pinecone: calibrated for recall@10 {target}: probing {} of {n_lists} lists and rescoring \
             {} rows reached {:.3} on {sample} ({seconds:.2} s)",
            outcome.nprobe,
            outcome.pool,
            outcome.recall
        );
    } else {
        let hint = if pinned_pool.is_some() {
            "rescore_multiplier pins the pool: unset it to let the build raise the pool, or lower \
             target_recall"
        } else {
            "lower target_recall, or set pinecone.scan_fraction and pinecone.rescore_multiplier \
             by hand"
        };
        pgrx::pg_sys::panic::ErrorReport::new(
            PgSqlErrorCode::ERRCODE_SUCCESSFUL_COMPLETION,
            format!(
                "pg_pinecone: recall@10 {target} not reached on {sample}: the best pair, probing {} of \
                 {n_lists} lists and rescoring {} rows, reached {:.3} ({seconds:.2} s)",
                outcome.nprobe, outcome.pool, outcome.recall
            ),
            pgrx::function_name!(),
        )
        .set_hint(hint)
        .report(PgLogLevel::NOTICE);
    }
    Some(outcome)
}

/// Every collected row's final-centroid assignment and RaBitQ code, in `collected`'s order.
struct Assigned {
    cluster: Vec<u32>,
    /// `code_bytes` per row: the code's words, little-endian (the on-disk form).
    codes: Vec<u8>,
    rnorm: Vec<f32>,
    dot: Vec<f32>,
    rc: Vec<f32>,
}

impl Assigned {
    fn with_capacity(rows: usize, code_bytes: usize) -> Assigned {
        Assigned {
            cluster: Vec::with_capacity(rows),
            codes: Vec::with_capacity(rows * code_bytes),
            rnorm: Vec::with_capacity(rows),
            dot: Vec::with_capacity(rows),
            rc: Vec::with_capacity(rows),
        }
    }

    fn append(&mut self, mut other: Assigned) {
        self.cluster.append(&mut other.cluster);
        self.codes.append(&mut other.codes);
        self.rnorm.append(&mut other.rnorm);
        self.dot.append(&mut other.dot);
        self.rc.append(&mut other.rc);
    }
}

/// One worker's share: `rows` assigned to their nearest centroid and encoded against it, with
/// one [`EncodeScratch`] and one [`Code`] for the whole chunk. A cosine index encodes the unit
/// vector (`x / |x|`); the assignment itself uses the stored vector, as the scan's ranking does.
fn assign_chunk(
    rows: &[CollectedRow],
    km: &KMeans,
    rotation: &Rotation,
    metric: Metric,
) -> Assigned {
    let mut out = Assigned::with_capacity(rows.len(), pages::code_bytes(rotation.dim()));
    let mut scratch = EncodeScratch::default();
    let mut code = Code {
        bits: Vec::new(),
        rnorm: 0.0,
        dot: 1.0,
        rc: 0.0,
    };
    let mut unit = Vec::new();
    for row in rows {
        let cluster = km.assign(&row.vector, metric);
        let x: &[f32] = if metric == Metric::Cosine {
            normalise_into(&row.vector, &mut unit);
            &unit
        } else {
            &row.vector
        };
        rabitq::encode_into(rotation, x, &km.centroids[cluster], &mut scratch, &mut code);
        out.cluster.push(cluster as u32);
        for word in &code.bits {
            out.codes.extend_from_slice(&word.to_le_bytes());
        }
        out.rnorm.push(code.rnorm);
        out.dot.push(code.dot);
        out.rc.push(code.rc);
    }
    out
}

/// Assigns and encodes every collected row — the O(N * n_lists * dim)-ish pass that (at
/// production scale) dominates alongside k-means training itself, per the C7 build-time gate
/// this module exists to close. Embarrassingly parallel with NO reduction
/// and hence none of `kmeans::update_step`'s fixed-chunk-order machinery:
/// each row's result depends ONLY on that row's own vector plus `km`/
/// `rotation`, which are both already-finalized, read-only snapshots by
/// this point — so however rows are chunked or scheduled across threads,
/// row `i`'s computed value is identical to what a single-threaded pass
/// would have produced for it. Returned in the SAME order as `collected`
/// (`.chunks(..)` + a same-order append below) purely so the
/// caller's own list-population loop can zip the two positionally;
/// this function only computes each row's assignment; it never itself
/// influences which physical page offset a row lands on (that's the
/// caller's sequential pass over the rows, which is what actually
/// has to run in a fixed order for the on-disk bytes to be
/// parallelism-invariant — see the call site's own comment).
fn assign_rows_parallel(
    collected: &[CollectedRow],
    km: &KMeans,
    rotation: &Rotation,
    metric: Metric,
    parallelism: usize,
) -> Assigned {
    if parallelism <= 1 || collected.is_empty() {
        return assign_chunk(collected, km, rotation, metric);
    }

    let workers = parallelism.min(collected.len()).max(1);
    let chunk_len = collected.len().div_ceil(workers).max(1);
    thread::scope(|scope| {
        let handles: Vec<_> = collected
            .chunks(chunk_len)
            .map(|chunk| scope.spawn(move || assign_chunk(chunk, km, rotation, metric)))
            .collect();
        // Ascending chunk order (`.chunks` + a same-order `.map` above) —
        // reproduces `collected`'s own row order exactly, regardless of
        // `parallelism`/thread scheduling.
        let mut all = Assigned::with_capacity(collected.len(), pages::code_bytes(rotation.dim()));
        for handle in handles {
            all.append(
                handle
                    .join()
                    .expect("pg_pinecone: build-assign worker thread panicked"),
            );
        }
        all
    })
}

/// What the page writer reads a list's records from.
struct ListSource<'a> {
    collected: &'a [CollectedRow],
    assigned: &'a Assigned,
    code_bytes: usize,
    has_filter: bool,
}

impl ListSource<'_> {
    fn record(&self, row: usize) -> ListRecord<'_> {
        let a = self.assigned;
        let r = &self.collected[row];
        ListRecord {
            code: Cow::Borrowed(&a.codes[row * self.code_bytes..(row + 1) * self.code_bytes]),
            rnorm: a.rnorm[row],
            dot: a.dot[row],
            rc: a.rc[row],
            tid: pgrx::itemptr::item_pointer_get_both(r.tid),
            fval: r.fval,
            fnull: r.fnull,
        }
    }
}

/// Writes the `d_pages`-page centroid directory (blocks
/// `dir_start..dir_start + d_pages`).
unsafe fn write_dir_pages(
    index: pg_sys::Relation,
    dim: usize,
    centroids: &[Vec<f32>],
    list_heads: &[u32],
    list_counts: &[usize],
    d_pages: usize,
    dir_start: u32,
) {
    let capacity = pages::dir_entries_per_page(dim);
    let n_lists = centroids.len();

    for page_idx in 0..d_pages {
        let start = page_idx * capacity;
        let end = (start + capacity).min(n_lists);
        let next = if page_idx + 1 < d_pages {
            dir_start + page_idx as u32 + 1
        } else {
            INVALID_BLOCK
        };

        let mut body =
            Vec::with_capacity(pages::DIR_HEADER_SIZE + (end - start) * pages::dir_entry_size(dim));
        body.extend_from_slice(&next.to_le_bytes());
        body.extend_from_slice(&((end - start) as u32).to_le_bytes());
        for i in start..end {
            for &f in &centroids[i] {
                body.extend_from_slice(&f.to_le_bytes());
            }
            body.extend_from_slice(&list_heads[i].to_le_bytes());
            body.extend_from_slice(&(list_counts[i] as u32).to_le_bytes());
        }

        unsafe {
            write_page(index, &body);
        }
    }
}

/// Writes one list's page chain (`rows` are its records' indexes into `source`, in build order;
/// columnar codes/scalars/tids — plus fvals/fnulls for a filtered index — per page; see
/// `pages.rs`'s documented layout).
unsafe fn write_list_pages(
    index: pg_sys::Relation,
    source: &ListSource<'_>,
    head_block: pg_sys::BlockNumber,
    rows: &[u32],
) {
    let capacity = pages::list_entries_per_page(source.code_bytes, source.has_filter);
    let n_pages = pages::list_page_count(rows.len(), source.code_bytes, source.has_filter);

    for page_idx in 0..n_pages {
        let start = page_idx * capacity;
        let end = (start + capacity).min(rows.len());
        // This list's own pages are allocated sequentially starting at
        // `head_block` (guaranteed by `ambuild_local`'s block-layout plan
        // and P_NEW's sequential allocation — see module docstring), so the
        // next page's block number is pure arithmetic, computed before
        // either page is actually written.
        let next = if page_idx + 1 < n_pages {
            head_block + page_idx as u32 + 1
        } else {
            INVALID_BLOCK
        };

        let mut body = ListBody::with_capacity(source.has_filter, source.code_bytes, end - start);
        for &row in &rows[start..end] {
            body.push(&source.record(row as usize));
        }
        unsafe {
            write_page(index, &body.finish(next));
        }
    }
}

/// Allocates a fresh page (`P_NEW`) and writes `body` starting at
/// `PageGetContents` — the same `ReadBufferExtended -> GenericXLogStart ->
/// RegisterBuffer -> PageInit -> memcpy -> Finish -> UnlockReleaseBuffer`
/// sequence `metapage::write` uses. Chained-page `next_block` fields are
/// baked directly into `body` from the up-front block plan (blocks are
/// handed out sequentially, so a page's true "next" block is known before
/// writing it) — no page is ever written twice.
unsafe fn write_page(index: pg_sys::Relation, body: &[u8]) {
    unsafe {
        let buffer = pg_sys::ReadBufferExtended(
            index,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            P_NEW,
            pg_sys::ReadBufferMode::RBM_NORMAL,
            std::ptr::null_mut(),
        );
        pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);

        let state = pg_sys::GenericXLogStart(index);
        let page = pg_sys::GenericXLogRegisterBuffer(
            state,
            buffer,
            pg_sys::GENERIC_XLOG_FULL_IMAGE as i32,
        );
        pg_sys::PageInit(page, pg_sys::BLCKSZ as usize, 0);

        let dest = pg_sys::PageGetContents(page) as *mut u8;
        std::ptr::copy_nonoverlapping(body.as_ptr(), dest, body.len());
        let header = page as *mut pg_sys::PageHeaderData;
        (*header).pd_lower += body.len() as u16;

        pg_sys::GenericXLogFinish(state);
        pg_sys::UnlockReleaseBuffer(buffer);
    }
}

/// TEST-ONLY directory surgery (`#[cfg(feature = "pg_test")]` — never in a
/// release build): overwrites directory entry `entry_idx`'s centroid and
/// list head in place, then queues a relcache invalidation so no backend
/// serves a stale `rd_amcache` copy of the old entry. Exists to synthesize
/// the one directory state a black-box fixture can't deterministically
/// produce: an EMPTY list (`list_head == INVALID_BLOCK`) whose centroid is
/// STRICTLY nearest to a chosen query — k-means only leaves a centroid
/// memberless on degenerate/tie-heavy inputs, and ties rank the non-empty
/// twin first, which never exercises the empty-slot path. Setting `head =
/// INVALID_BLOCK` orphans the entry's former records (unreachable, not
/// corrupt) — callers must not assert on rows that lived in that list.
#[cfg(feature = "pg_test")]
pub(crate) unsafe fn test_overwrite_directory_entry(
    index: pg_sys::Relation,
    meta: &MetaPage,
    entry_idx: usize,
    centroid: &[f32],
    list_head: u32,
) {
    unsafe {
        assert!(!meta.is_flat, "directory surgery needs a clustered index");
        assert_eq!(centroid.len(), meta.dim as usize, "centroid dim mismatch");
        let dim = meta.dim as usize;
        let capacity = pages::dir_entries_per_page(dim);
        let block = meta.centroid_dir_start + (entry_idx / capacity) as u32;
        let offset = pages::DIR_HEADER_SIZE + (entry_idx % capacity) * pages::dir_entry_size(dim);

        let buffer = pg_sys::ReadBufferExtended(
            index,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            block,
            pg_sys::ReadBufferMode::RBM_NORMAL,
            std::ptr::null_mut(),
        );
        pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
        let state = pg_sys::GenericXLogStart(index);
        let page = pg_sys::GenericXLogRegisterBuffer(state, buffer, 0);
        let contents = pg_sys::PageGetContents(page) as *mut u8;
        for (i, &f) in centroid.iter().enumerate() {
            std::ptr::copy_nonoverlapping(
                f.to_le_bytes().as_ptr(),
                contents.add(offset + i * 4),
                4,
            );
        }
        std::ptr::copy_nonoverlapping(
            list_head.to_le_bytes().as_ptr(),
            contents.add(offset + dim * 4),
            4,
        );
        pg_sys::GenericXLogFinish(state);
        pg_sys::UnlockReleaseBuffer(buffer);

        pg_sys::CacheInvalidateRelcacheByRelid((*index).rd_id);
    }
}
