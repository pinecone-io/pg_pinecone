//! `amgettuple`'s local-mode path (the IVF probe scan, the small-N exact
//! bootstrap, and the two-mode filtered scan). Every sub-path computes the
//! *entire* answer on the scan's first `amgettuple` call (there is no
//! per-tuple incremental work beyond serving a precomputed,
//! already-sorted `Vec` through a cursor) — see [`amgettuple`]'s own
//! docstring for why that one-shot shape is the actual correctness
//! mechanism, not just a simplification. (Exception: under
//! `pinecone.iterative_scan` (`auto`, the default, `relaxed_order`, `strict_order`) a mode-H scan
//! computes more batches on demand, "Iterative scan" below.)
//!
//! ## Design decision: in-AM full-pool rescore, not `xs_recheckorderby`
//!
//! `IndexScanDesc::xs_recheckorderby` + `xs_orderbyvals` is Postgres's
//! built-in "reorder queue" (`nodeIndexscan.c`): an AM returns *approximate*
//! order-by values, and the executor merges results across index pages, but
//! **only correctly** if every returned value is a valid, provable *lower
//! bound* of the true final distance — that's what lets the executor's
//! priority queue know it's safe to emit a tuple (nothing not-yet-seen can
//! possibly beat it once its lower bound exceeds the queue's current worst
//! candidate). pgvectorscale's SBQ index (the precedent)
//! makes this sound via a real statistical/algebraic bound on its quantized
//! distance. The RaBitQ estimate this module uses (`rabitq::estimate_bits`)
//! is an **unbiased estimator** with a 1.9-sigma error bound, not a sound lower
//! bound — it can land on either side of the true distance — so plumbing it
//! through as `xs_orderbyvals` would violate the reorder queue's invariant
//! and risk **silently wrong `ORDER BY` output**, not just lower recall.
//! The bound only decides when to *stop rescoring* ("Early stop" below), never the
//! order. The documented fallback stands: **rescore the candidate pool
//! with exact fp32 distance before returning anything**, sort those exact
//! distances, and serve the result
//! with `xs_recheckorderby = false` (Postgres trusts our order outright —
//! same as the remote AM's own choice, `am.rs`'s `amgettuple` rustdoc).
//!
//! ## The `LIMIT`-blindness problem, and why this design never grows a batch
//!
//! `amgettuple` has no visibility into the query's `LIMIT`. The remote AM
//! copes by *growing* `topK` and re-querying (safe there because a trusted
//! remote order never needs revising). Growing anything *locally* — probing
//! more lists, or exact-rescoring further down an already-ranked pool —
//! risks discovering a candidate whose *true* distance undercuts a row
//! already handed to the executor, silently breaking the "results actually
//! sorted by true distance" guarantee (once a tuple is returned, it can
//! never be un-returned). The only way to guarantee that never happens is to
//! fix the candidate pool *once* (this module's `nprobe`, resolved at build
//! time and stored in the metapage — never widened mid-scan) and fully
//! resolve true distances for the portion of it this scan will ever return
//! before returning the first row.
//!
//! Concretely: `nprobe` lists are probed once; every candidate in them is
//! scored by its RaBitQ estimate (one rotation of the query, one 4-bit prep per probed
//! list, table lookups per record in blocks; `rabitq.rs` has the formulas); the best `rescore_multiplier *
//! ASSUMED_TOP_K` of those (§ below), fewer in `auto`'s first batch once the early stop fires, are
//! exact-rescored from the heap; the result is sorted by exact distance and streamed out. If the
//! query's real `LIMIT` exceeds that fixed pool, a scan under `iterative_scan = off` simply returns
//! fewer rows than asked — the same accepted, documented behavior as pgvector's own
//! `ivfflat` (too few `probes`) or `hnsw` (`ef_search` too small): a smaller
//! result than `LIMIT`, never a wrongly-ordered one. `ASSUMED_TOP_K` is a
//! fixed module constant, not a real `LIMIT` — see its own doc comment for
//! why, and for the investigation this leaves on the table.
//! The default, `auto`, goes on past the pool ("Iterative scan" below).
//!
//! ## Early stop (`pinecone.scan_prune`, on by default)
//!
//! The pool is a cap, not a quota. In the first batch of an `auto` scan, candidates are rescored in
//! estimate order, 32 at a time, and the walk ends when no unseen candidate's lower bound (its key at
//! estimate + 2 x err) is below the k-th best exact key so far, k = max(10, pool / 10). The
//! estimator's own 1.9-sigma bound is too tight for this (it loses true neighbours), hence
//! the wider margin. The walk goes on until `min(pool, ASSUMED_TOP_K)` rows are found, so a
//! `LIMIT` up to 100 is still filled, but only the pool's exact top k is certified: the ranks after
//! it are the rows fetched, in exact order. Rows returned are rows rescored. Only that batch stops early
//! ([`early_stop`]): an explicit `off` rescores the whole pool (the 0.9.0 / P1 behaviour: one batch,
//! with a residual `WHERE` the index cannot see filtering what it returns), as does `auto` with
//! `scan_prune` off. `relaxed_order`/`strict_order` never stop early, nor does what `auto` goes on
//! with: a batch returned short would hand the candidates it left out over after farther rows.
//! [`super::blockscan`] has the loop.
//!
//! ## Iterative scan (`pinecone.iterative_scan`)
//!
//! A residual `WHERE` (anything the index cannot push) or a `LIMIT` above the
//! pool leaves the executor asking for more after the first batch.
//!
//! `auto` (the default) serves `off`'s first batch (the calibrated lists, the bounded buffer) with the
//! early stop, so a query it satisfies reads `off`'s top `pool / 10` rows at its cost or less, and keeps
//! the query, the pushed key and the candidates it fetched. If the executor drains the batch and asks
//! for more, [`restart_relaxed`] starts a `relaxed_order` scan from the nearest lists, leaving out
//! every candidate the batch fetched (the rows it returned and those whose row was gone). The batch kept
//! only its best `pool` candidates, so the lists are probed again: that is what a continuation
//! costs over `relaxed_order` from the start. A batch that probed every list and fetched every
//! record they hold has nothing left to find, and starts none.
//!
//! With `relaxed_order`/`strict_order` a mode-H scan answers: [`IvfCont`] keeps the
//! lists ranked on the first call and the estimate-ranked candidates of the
//! lists probed so far. Each time the executor drains a batch `amgettuple`
//! rescores the next pool-sized slice of those candidates (deepen: the
//! residual runs after the pool cut, so the rows it keeps can sit below it)
//! and only when none are left probes the next `nprobe` lists (widen; lists
//! are disjoint, so no visited set). The first batch is the whole pool `off`
//! rescores over the same lists, so any query it satisfies is unchanged when both probe the same
//! lists; the iterative modes default to the 0.9.0 list count (`effective_nprobe`), `off` to the
//! calibrated one.
//!
//! Order holds per batch only (`relaxed_order`: a later batch can hand out a
//! closer row after a farther one); `strict_order` drops the rows closer than
//! the last one handed out — ordered but lossy, like pgvector's; an exact
//! version needs per-list distance bounds in the directory (a format change).
//! `pinecone.max_scan_tuples` caps the exact-rescored rows: the first batch is
//! never cut, a later slice is cut to what the cap leaves. Hitting it, or
//! running out of candidates and lists, ends the scan quietly
//! (`pinecone.last_scan_stats()` says which). Mode L and the flat path already
//! return every qualifying row and ignore all of this.
//!
//! ## The two-mode FILTERED scan
//!
//! When `amrescan` hands this scan an equality scankey on the index's filter
//! column (`am.rs` copies it into `keyData`; `extract_filter` below decodes
//! it), a clustered index picks one of two strategies by estimated
//! selectivity (the build-time [`super::stats`] sketch) against the
//! `pinecone.mode_l_threshold` GUC — overridable outright via
//! `pinecone.force_scan_mode` (the documented estimator escape hatch):
//!
//! - **Mode L** (estimated qualifying fraction <= threshold): walk EVERY
//!   list page collecting the TIDs whose *stored* filter value matches, then
//!   exact-rescore all of them from the heap. Recall 1.0 by construction —
//!   the candidate set IS the qualifying set (modulo text-hash collisions,
//!   which the executor recheck removes; see below). This reads all codes
//!   pages but touches no heap tuple except matches — cheap relative to
//!   scoring, per the plan's exact-scan-dominates-at-1% evidence.
//! - **Mode H** (above threshold): the normal IVF probe path, but each
//!   record's stored filter value is tested BEFORE scoring — the
//!   intersection happens in the codes loop itself — and survivors rescore
//!   from the heap exactly as an unfiltered scan would.
//!
//! Text filter columns store a 64-bit hash (`pages::fnv1a64`), so hash
//! equality only *nominates* a row — every text-filtered scan sets
//! `xs_recheck = true` and the executor re-evaluates the original qual
//! against the fetched heap tuple, discarding collisions. Int filters are
//! exact (`xs_recheck = false`). Rows whose filter column is NULL never
//! match an equality scankey (SQL semantics), in every mode and in the flat
//! bootstrap path alike.
//!
//! The mode actually run is observable: `elog DEBUG1` per filtered scan plus
//! the `pinecone.last_scan_mode()` SQL helper (backend-local; see
//! [`last_scan_mode_name`]) so tests/benchmarks can assert which mode served
//! a query without parsing logs.
//!
//! Scans WITHOUT a filter key — on filtered and unfiltered indexes alike —
//! run the exact unfiltered paths: the per-record filter arrays are never read (the
//! layout math skips over them), so the plain path's behavior is unchanged.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};

use pgrx::{
    guc::{GucContext, GucFlags, GucRegistry, GucSetting},
    prelude::*,
};

use crate::am::{IterativeScan, iterative_scan, max_scan_tuples};

use super::{
    blockscan::{self, CandBuf, Candidate, Plan, ScanCounters, Scoring, nan_last},
    kernel::{self, Kernel},
    metapage::{self, FilterKind, INVALID_BLOCK, MetaPage},
    metric::{
        Metric, exact_distance, exact_key, first_non_finite, key_to_distance, normalise_into,
    },
    options::{FALLBACK_POOL, legacy_nprobe},
    pages::{self, Geometry, ListPage},
    rabitq::{self, QueryPrep},
    rotation::{self, Rotation},
    stats,
};

/// `SHOW pinecone.rescore_multiplier`'s default (`0` = no override) — scan-time
/// overrides for the two knobs the build fixes (the pool and the probe count).
/// Both default to a sentinel meaning "no override, use what the build put in
/// the metapage/reloptions"; setting either lets `SET` (session-, or even
/// query-scoped via `SET LOCAL`) sweep the knob across a whole benchmark run
/// with zero `REINDEX`s. Not a substitute for real `LIMIT` propagation (no such
/// `IndexAmRoutine` hook exists — see `ASSUMED_TOP_K`'s own rustdoc).
static RESCORE_MULTIPLIER_OVERRIDE: GucSetting<i32> = GucSetting::<i32>::new(0);
/// `SHOW pinecone.scan_fraction`'s default — overrides the probe count the index
/// calibrated at build (`MetaPage::default_nprobe`), at SCAN time: `0.0` means
/// "use the index's own", any value in `(0.0, 1.0]` recomputes `nprobe =
/// clamp(ceil(fraction * n_lists), 1, n_lists)` fresh, per scan, from the
/// CURRENT directory size — letting a sweep vary this without rebuilding the
/// list pages the fraction has no other effect on.
static SCAN_FRACTION_OVERRIDE: GucSetting<f64> = GucSetting::<f64>::new(0.0);
/// `SHOW pinecone.scan_prune`'s default: whether the first batch of an `auto` native scan stops
/// rescoring once the error bound shows that no remaining candidate can enter the top rows
/// (`blockscan::rescore_progressive`).
static SCAN_PRUNE: GucSetting<bool> = GucSetting::<bool>::new(true);
/// `SHOW pinecone.mode_l_threshold`'s default — the two-mode filtered scan's
/// selectivity boundary (module docstring): estimated qualifying fraction at
/// or below this runs mode L (filter-first exact), above it mode H (IVF
/// probe + in-loop filter). 0.03 sits inside a 2-5% starting band;
/// benchmark sweeps tune it against the three corpora.
static MODE_L_THRESHOLD: GucSetting<f64> = GucSetting::<f64>::new(0.03);
/// `SHOW pinecone.force_scan_mode`'s default — `'auto'` (use the estimator +
/// threshold), `'l'`, or `'h'`. The documented fallback if the stats sketch
/// proves unsound for some corpus (see `super::stats`'s accuracy-limits
/// section), and the lever the mode-boundary sweeps drive directly.
static FORCE_SCAN_MODE: GucSetting<Option<std::ffi::CString>> =
    GucSetting::<Option<std::ffi::CString>>::new(Some(c"auto"));

/// Registers this module's scan-time GUCs — called from `am::init_gucs`.
pub(crate) fn init_gucs() {
    GucRegistry::define_int_guc(
        c"pinecone.rescore_multiplier",
        c"Overrides every local-mode index's rescore pool (this times 100 rows) for scans in this \
          session (0 = no override: each index's own rescore_multiplier reloption, else the pool \
          its build calibrated).",
        c"",
        &RESCORE_MULTIPLIER_OVERRIDE,
        0,
        1000,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_float_guc(
        c"pinecone.scan_fraction",
        c"Overrides every local-mode index's IVF probe fraction (nprobe / n_lists) for scans \
          in this session (0.0 = no override: unfiltered scans use each index's own \
          calibrated nprobe, iterative scans 4 / sqrt(n_lists), and FILTERED mode-H scans an \
          automatically ramped fraction — min(0.5, 16 / sqrt(n_lists)), already 0.5 up to \
          about 1000 lists, so a pushed key gains here only above 0.5 — since in-loop \
          filtering makes extra probed lists nearly free; any explicit value here wins).",
        c"",
        &SCAN_FRACTION_OVERRIDE,
        0.0,
        1.0,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_bool_guc(
        c"pinecone.scan_prune",
        c"Whether the first batch of an auto native scan stops rescoring once the estimator's error \
          bound shows that no remaining candidate can enter the top rows of the pool (on); off \
          rescores the whole pool (rescore_multiplier x 100 rows), as 0.9 did. An explicit \
          iterative_scan = off always rescores the whole pool.",
        c"",
        &SCAN_PRUNE,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_float_guc(
        c"pinecone.mode_l_threshold",
        c"Estimated filter selectivity at or below which a filtered local-mode scan uses \
          mode L (filter-first exact scan, recall 1.0) instead of mode H (IVF probe with \
          in-loop filtering).",
        c"",
        &MODE_L_THRESHOLD,
        0.0,
        1.0,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_string_guc(
        c"pinecone.force_scan_mode",
        c"Forces every filtered local-mode scan's mode: 'auto' (selectivity estimate vs \
          pinecone.mode_l_threshold), 'l', or 'h'.",
        c"",
        &FORCE_SCAN_MODE,
        GucContext::Userset,
        GucFlags::default(),
    );
}

pub(crate) fn scan_prune() -> bool {
    SCAN_PRUNE.get()
}

/// Whether a batch stops its rescore early ([`blockscan::rescore_progressive`]): the first batch of an
/// `auto` scan, with `pinecone.scan_prune` on. An explicit `off` is the whole rescored pool and nothing
/// more (the index cannot see a residual `WHERE` that filters it), and an iterative mode rescores
/// every batch whole.
fn early_stop(setting: IterativeScan, scan_prune: bool) -> bool {
    scan_prune && setting == IterativeScan::Auto
}

/// Rows the scan rescores. A session `rescore_multiplier` wins; a scan with a pushed key keeps the
/// 0.9.0 pool unless the reloption pinned one, because the calibration measured unfiltered scans:
/// with the key in the scan loop the pool is cut over candidates that all qualify, and a 300-row pool
/// there left `iterative_scan = off` with 3 of 10 rows. Otherwise the index's default pool.
fn pool_for(
    session_multiplier: i32,
    reloption_multiplier: u32,
    default_pool: u32,
    pushed_key: bool,
) -> usize {
    if session_multiplier > 0 {
        session_multiplier as usize * ASSUMED_TOP_K
    } else if pushed_key && reloption_multiplier == 0 {
        FALLBACK_POOL as usize
    } else {
        default_pool as usize
    }
}

fn effective_pool(meta: &MetaPage, pushed_key: bool) -> usize {
    pool_for(
        RESCORE_MULTIPLIER_OVERRIDE.get(),
        meta.rescore_multiplier,
        meta.default_pool(),
        pushed_key,
    )
}

/// The index's default probe count (calibrated, else the 0.9.0 rule), unless [`SCAN_FRACTION_OVERRIDE`] is set for this session — in which case it's
/// recomputed from the override fraction and the directory's ACTUAL size (`n_lists_now`), not the
/// metapage's stored `n_lists`.
///
/// A mode-H FILTERED scan (`is_filtered`) intersects the filter in the same codes loop
/// (`scan_list`), so an unqualifying record costs one extra compare and probing more lists is
/// nearly free: its default fraction ramps to `min(0.5, 16 / sqrt(n_lists))` (measured: raising
/// the fraction to 0.5 on the 1% correlated cross-tenant fixture delivered 0.967 recall @ 9.7ms
/// where 4/sqrt(n_lists) plateaued at 0.833). The ramp is its own constant, not derived from the
/// unfiltered default, which was cut to `0.8 / sqrt(n_lists)`.
///
/// An iterative scan (`relaxed_order`, `strict_order`, what `auto` goes on with) keeps the 0.9.0
/// `4 / sqrt(n_lists)`: it fills a `LIMIT` behind a residual `WHERE`, and the candidates of the
/// first lists decide which rows it returns (recall@100 behind a 10% residual: 0.63 at 48 lists,
/// 0.53 at 25). The pool needs no such care: the scan deepens until the executor is satisfied. A
/// [`SCAN_FRACTION_OVERRIDE`] wins outright, filtered or not.
fn effective_nprobe(
    meta: &MetaPage,
    n_lists_now: usize,
    is_filtered: bool,
    iterative: bool,
) -> usize {
    nprobe_for(
        meta.default_nprobe(),
        n_lists_now,
        SCAN_FRACTION_OVERRIDE.get(),
        is_filtered,
        iterative,
    )
}

/// What `amcostestimate` prices a first batch at: [`effective_pool`] and [`effective_nprobe`] without
/// the session overrides, so that a `SET` sweep does not move plans.
pub(crate) fn planned_pool_and_nprobe(meta: &MetaPage, pushed_key: bool) -> (u32, u32) {
    let pool = pool_for(0, meta.rescore_multiplier, meta.default_pool(), pushed_key);
    let lists = meta.n_lists as usize;
    let nprobe = nprobe_for(meta.default_nprobe(), lists, 0.0, pushed_key, false);
    (pool as u32, nprobe as u32)
}

fn nprobe_for(
    stored: u32,
    n_lists_now: usize,
    session_fraction: f64,
    is_filtered: bool,
    iterative: bool,
) -> usize {
    if session_fraction > 0.0 {
        return ((session_fraction * n_lists_now as f64).ceil() as usize)
            .clamp(1, n_lists_now.max(1));
    }
    if is_filtered {
        let unfiltered_fraction = 4.0 / (n_lists_now.max(1) as f64).sqrt();
        let ramped_fraction = (unfiltered_fraction * 4.0).min(0.5);
        return ((ramped_fraction * n_lists_now as f64).ceil() as usize)
            .clamp(1, n_lists_now.max(1));
    }
    if iterative {
        return legacy_nprobe(n_lists_now as u32) as usize;
    }
    (stored as usize).min(n_lists_now)
}

/// The scan has no way to learn the query's real `LIMIT` (see module docstring),
/// so the exact-rescore pool size is `rescore_multiplier * ASSUMED_TOP_K`
/// rather than `rescore_multiplier * k` — a fixed stand-in for "the largest
/// `LIMIT` this index is tuned to answer with full pool resolution". A
/// `LIMIT` at or below this is filled from the pool, in exact order among the rows rescored: under
/// `off` that is the whole pool, under `auto`'s first batch the early stop certifies only the pool's
/// exact top `k = max(10, pool / 10)` and the ranks after it are the rows fetched. Above the pool a
/// scan under `off` under-returns rather than misordering (`auto` goes on).
/// Propagating the planner's actual `LIMIT` into the AM (there's no existing
/// `IndexAmRoutine` hook for it — it would need a new support function or a
/// planner-side hint) is the natural follow-up.
pub(crate) const ASSUMED_TOP_K: usize = 100;

// ---------------------------------------------------------------------------
// Scan-mode observability
// ---------------------------------------------------------------------------

const MODE_NONE: u8 = 0;
const MODE_FLAT: u8 = 1;
const MODE_H: u8 = 2;
const MODE_L: u8 = 3;

/// Which path the most recent local-mode scan in THIS backend ran — a plain
/// backend-local flag (Postgres backends are single-threaded; `AtomicU8`
/// only to satisfy Rust's `static` rules). Read by `pinecone.last_scan_mode()`
/// (`lib.rs`'s `pg_pinecone` schema module) so tests/benchmarks can assert the
/// mode without parsing DEBUG logs. Never reset between scans — it reports
/// the LAST completed candidate computation, `"none"` only before the first.
static LAST_SCAN_MODE: AtomicU8 = AtomicU8::new(MODE_NONE);

fn record_scan_mode(mode: u8) {
    LAST_SCAN_MODE.store(mode, Ordering::Relaxed);
}

pub(crate) fn last_scan_mode_name() -> &'static str {
    match LAST_SCAN_MODE.load(Ordering::Relaxed) {
        MODE_NONE => "none",
        MODE_FLAT => "flat",
        MODE_H => "h",
        MODE_L => "l",
        _ => unreachable!("pg_pinecone: only this module stores LAST_SCAN_MODE"),
    }
}

/// Batches computed, rows exact-rescored (native) or fetched from Pinecone
/// (remote), whether `pinecone.max_scan_tuples` ended the most recent scan in
/// THIS backend, native batches that went deeper into lists already probed,
/// and lists probed. Reset when a scan computes its first batch; read by
/// `pinecone.last_scan_stats()`.
static LAST_SCAN_BATCHES: AtomicU32 = AtomicU32::new(0);
static LAST_SCAN_RESCORED: AtomicU64 = AtomicU64::new(0);
static LAST_SCAN_CAPPED: AtomicBool = AtomicBool::new(false);
static LAST_SCAN_DEEPENED: AtomicU32 = AtomicU32::new(0);
static LAST_SCAN_LISTS: AtomicU32 = AtomicU32::new(0);
/// Records scored, records that entered the candidate buffer and blocks that gave it none
/// (`blockscan::ScanCounters`), summed over a scan's probe steps.
static LAST_SCAN_SCANNED: AtomicU64 = AtomicU64::new(0);
static LAST_SCAN_ADMITTED: AtomicU64 = AtomicU64::new(0);
static LAST_SCAN_BLOCKS_SKIPPED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn reset_scan_stats() {
    LAST_SCAN_BATCHES.store(0, Ordering::Relaxed);
    LAST_SCAN_RESCORED.store(0, Ordering::Relaxed);
    LAST_SCAN_CAPPED.store(false, Ordering::Relaxed);
    LAST_SCAN_DEEPENED.store(0, Ordering::Relaxed);
    LAST_SCAN_LISTS.store(0, Ordering::Relaxed);
    LAST_SCAN_SCANNED.store(0, Ordering::Relaxed);
    LAST_SCAN_ADMITTED.store(0, Ordering::Relaxed);
    LAST_SCAN_BLOCKS_SKIPPED.store(0, Ordering::Relaxed);
}

fn record_scan_counters(c: &ScanCounters) {
    LAST_SCAN_SCANNED.fetch_add(c.scanned, Ordering::Relaxed);
    LAST_SCAN_ADMITTED.fetch_add(c.admitted, Ordering::Relaxed);
    LAST_SCAN_BLOCKS_SKIPPED.fetch_add(c.blocks_skipped, Ordering::Relaxed);
}

/// One more native batch; `rescored` rows join the running total.
fn record_scan_batch(rescored: u64) {
    LAST_SCAN_BATCHES.fetch_add(1, Ordering::Relaxed);
    LAST_SCAN_RESCORED.fetch_add(rescored, Ordering::Relaxed);
}

fn record_scan_deepened() {
    LAST_SCAN_DEEPENED.fetch_add(1, Ordering::Relaxed);
}

fn record_lists_probed(n: usize) {
    LAST_SCAN_LISTS.fetch_add(n as u32, Ordering::Relaxed);
}

/// One more remote `/query`. Each re-query starts from scratch, so the running
/// total is the largest `topK` asked for, not a sum.
pub(crate) fn record_remote_query(top_k: u64) {
    LAST_SCAN_BATCHES.fetch_add(1, Ordering::Relaxed);
    LAST_SCAN_RESCORED.store(top_k, Ordering::Relaxed);
}

/// The scan stopped at `pinecone.max_scan_tuples` with the executor still asking.
pub(crate) fn record_scan_capped() {
    LAST_SCAN_CAPPED.store(true, Ordering::Relaxed);
}

pub(crate) struct ScanStats {
    pub(crate) batches: u32,
    pub(crate) rescored: u64,
    pub(crate) capped: bool,
    pub(crate) deepened: u32,
    pub(crate) lists_probed: u32,
    pub(crate) scanned: u64,
    pub(crate) admitted: u64,
    pub(crate) blocks_skipped: u64,
}

pub(crate) fn last_scan_stats() -> ScanStats {
    ScanStats {
        batches: LAST_SCAN_BATCHES.load(Ordering::Relaxed),
        rescored: LAST_SCAN_RESCORED.load(Ordering::Relaxed),
        capped: LAST_SCAN_CAPPED.load(Ordering::Relaxed),
        deepened: LAST_SCAN_DEEPENED.load(Ordering::Relaxed),
        lists_probed: LAST_SCAN_LISTS.load(Ordering::Relaxed),
        scanned: LAST_SCAN_SCANNED.load(Ordering::Relaxed),
        admitted: LAST_SCAN_ADMITTED.load(Ordering::Relaxed),
        blocks_skipped: LAST_SCAN_BLOCKS_SKIPPED.load(Ordering::Relaxed),
    }
}

// ---------------------------------------------------------------------------
// Filter scankey extraction
// ---------------------------------------------------------------------------

/// Inclusive `i64` interval over a record's stored filter value; equality is
/// `lo == hi`. `recheck`: the stored value is a text hash, so the executor
/// must re-evaluate the original qual on each fetched tuple.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Predicate {
    lo: i64,
    hi: i64,
    recheck: bool,
}

impl Predicate {
    fn eq(value: i64, recheck: bool) -> Self {
        Predicate {
            lo: value,
            hi: value,
            recheck,
        }
    }

    /// `None` when `lo > hi`: nothing can match.
    fn between(lo: i64, hi: i64, recheck: bool) -> Option<Self> {
        (lo <= hi).then_some(Predicate { lo, hi, recheck })
    }

    /// `None` at `i64::MAX`: `> MAX` matches nothing.
    #[allow(dead_code)] // groundwork: no range scan key is pushed yet
    fn greater_than(value: i64, recheck: bool) -> Option<Self> {
        Predicate::between(value.checked_add(1)?, i64::MAX, recheck)
    }

    /// `None` at `i64::MIN`: `< MIN` matches nothing.
    #[allow(dead_code)] // see `greater_than`
    fn less_than(value: i64, recheck: bool) -> Option<Self> {
        Predicate::between(i64::MIN, value.checked_sub(1)?, recheck)
    }

    #[inline]
    fn matches(&self, v: i64) -> bool {
        self.lo <= v && v <= self.hi
    }

    /// The single value an equality matches (what the selectivity sketch
    /// can estimate), `None` for a wider interval.
    fn point(&self) -> Option<i64> {
        (self.lo == self.hi).then_some(self.lo)
    }

    /// Both predicates at once; `None` when nothing can match.
    fn intersect(self, other: Self) -> Option<Self> {
        Predicate::between(
            self.lo.max(other.lo),
            self.hi.min(other.hi),
            self.recheck || other.recheck,
        )
    }
}

/// The decoded per-scan filter, from `amrescan`'s scankeys.
enum FilterState {
    /// No scankey — an unfiltered scan (the only possibility on an index
    /// with no filter column).
    None,
    /// `filter_col = <value>`: match records whose stored filter value
    /// satisfies the predicate (`recheck`: see [`Predicate`]).
    Match(Predicate),
    /// Structurally unsatisfiable keys (`= NULL`, or two conflicting
    /// equalities) — return nothing, exactly.
    MatchNothing,
}

/// Maps a scankey's right-hand type to the conversion its datum needs. The
/// operator family's cross-type members mean an int8 column's scankey can
/// legitimately carry an int2/int4 right-hand side (e.g. `int8col = 5` parses
/// as `=(int8, int4)`); every int width sign-extends into the same stored
/// `i64` domain, so equality survives the width mix.
pub(crate) fn kind_for_subtype(subtype: pg_sys::Oid, column_kind: FilterKind) -> FilterKind {
    if subtype == pg_sys::Oid::INVALID {
        return column_kind;
    }
    match subtype {
        pg_sys::INT2OID => FilterKind::Int2,
        pg_sys::INT4OID => FilterKind::Int4,
        pg_sys::INT8OID => FilterKind::Int8,
        pg_sys::TEXTOID | pg_sys::VARCHAROID => FilterKind::Text,
        other => error!(
            "pg_pinecone: unsupported scankey right-hand type (oid {}) for a pg_pinecone filter column",
            other.to_u32()
        ),
    }
}

/// Decodes the scan's filter scankeys (if any) into a [`FilterState`]. The
/// AM only ever advertises equality (strategy 1) on key column 2, so
/// anything else here is a planner-contract violation worth a loud error,
/// not a silent skip.
unsafe fn extract_filter(scan: pg_sys::IndexScanDesc, meta: &MetaPage) -> FilterState {
    unsafe {
        let nkeys = (*scan).numberOfKeys as usize;
        if nkeys == 0 {
            return FilterState::None;
        }
        if !meta.has_filter() {
            error!(
                "pg_pinecone: scan keys supplied for a pg_pinecone index with no filter column — should be \
                 unreachable"
            );
        }
        let keys = std::slice::from_raw_parts((*scan).keyData, nkeys);
        let mut pred: Option<Predicate> = None;
        for key in keys {
            if key.sk_attno != 2 || key.sk_strategy != 1 {
                error!(
                    "pg_pinecone: unexpected scan key (attno {}, strategy {}) — the pg_pinecone access \
                     method only supports equality on its filter column",
                    key.sk_attno, key.sk_strategy
                );
            }
            if key.sk_flags & pg_sys::SK_ISNULL as i32 != 0 {
                // `col = NULL` is never true — normally folded away by the
                // planner before it ever reaches an AM, handled here for the
                // paths (e.g. prepared-statement params) that aren't.
                return FilterState::MatchNothing;
            }
            let kind = kind_for_subtype(key.sk_subtype, meta.filter_kind);
            let v = kind.datum_to_value(key.sk_argument);
            let key_pred = Predicate::eq(v, meta.filter_kind == FilterKind::Text);
            pred = match pred {
                None => Some(key_pred),
                // Two equality keys with different values (`x = 1 AND x = 2`)
                // can both be pushed; their conjunction matches nothing.
                Some(prev) => match prev.intersect(key_pred) {
                    Some(both) => Some(both),
                    None => return FilterState::MatchNothing,
                },
            };
        }
        FilterState::Match(pred.expect("nkeys > 0 always sets a predicate"))
    }
}

// ---------------------------------------------------------------------------
// Scan state + amgettuple
// ---------------------------------------------------------------------------

pub(crate) struct LocalScanState {
    /// `(heap TID, exact distance)`, sorted ascending — the current batch:
    /// the entire answer (every path but an iterative mode-H scan), else the
    /// latest of its batches. First computed by the first `amgettuple` call
    /// of each rescan.
    results: Option<Vec<(pg_sys::ItemPointerData, f32)>>,
    cursor: usize,
    /// Whether every tuple this scan returns must be executor-rechecked
    /// against the original quals — true exactly for text-hash filters
    /// (module docstring).
    recheck: bool,
    /// What an iterative mode-H scan needs to probe the lists its first batch
    /// left unprobed; `None` under `off`, for the complete paths (mode L,
    /// flat) and once the scan is over. Boxed: it dwarfs the rest of the state.
    cont: Option<Box<IvfCont>>,
    /// `auto`: what the scan needs to go on as `relaxed_order` once the executor drains the first
    /// batch (`off`'s); `None` under every other setting, on the complete paths and once started.
    restart: Option<Box<Restart>>,
}

/// What an `auto` scan keeps after its first batch: the query and pushed key to restart from, and
/// the candidates the batch fetched, found or not (the restart leaves them out; they count against
/// `pinecone.max_scan_tuples`).
struct Restart {
    query: Vec<f32>,
    pred: Option<Predicate>,
    fetched: Vec<Candidate>,
}

impl LocalScanState {
    pub(crate) fn new() -> Self {
        LocalScanState {
            results: None,
            cursor: 0,
            recheck: false,
            cont: None,
            restart: None,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.results = None;
        self.cursor = 0;
        self.recheck = false;
        self.cont = None;
        self.restart = None;
    }
}

/// The local-mode `amgettuple` body — see module docstring for the
/// compute-then-stream design, the two-mode filtered dispatch and the
/// iterative continuation.
pub(crate) unsafe fn amgettuple(scan: pg_sys::IndexScanDesc, state: &mut LocalScanState) -> bool {
    unsafe {
        if state.results.is_none() {
            state.cont = None;
            state.restart = None;
            reset_scan_stats();
            let ob = &*(*scan).orderByData;
            let results = if ob.sk_flags & pg_sys::SK_ISNULL as i32 != 0 {
                // A NULL query vector matches nothing — mirrors the remote
                // AM's own handling in `am::amgettuple`.
                Vec::new()
            } else {
                let query = crate::am::parse_vector(ob.sk_argument);
                let index = (*scan).indexRelation;
                let heap = (*scan).heapRelation;
                let snapshot = (*scan).xs_snapshot;
                let meta = metapage::read(index);
                // Every path below indexes into per-record data
                // sized by `meta.dim` (the rotation's padding, the exact-
                // distance loops, the flat bootstrap's heap vectors) with no
                // bounds check of its own — in a release build,
                // the distance kernels' `debug_assert_eq!(a.len(), b.len())`
                // compiles out entirely, so a mismatched query would either
                // panic (debug) or silently truncate to the shorter
                // length and score garbage (release), neither of which is an
                // error a caller could act on. Checked once, here, before any
                // scan path runs — mirrors pgvector's own `vector_l2_ops`
                // etc. distance functions, which raise exactly this message
                // (`vector.c`) for a query vector of the wrong dimension.
                if query.len() != meta.dim as usize {
                    error!(
                        "pg_pinecone: different vector dimensions {} and {}",
                        meta.dim,
                        query.len()
                    );
                }
                let filter = extract_filter(scan, &meta);
                state.recheck = matches!(filter, FilterState::Match(p) if p.recheck);

                match (&filter, meta.is_flat) {
                    (_, true) => {
                        record_scan_mode(MODE_FLAT);
                        let rows = exact_scan_all(heap, snapshot, &meta, &query, &filter);
                        record_scan_batch(rows.len() as u64);
                        rows
                    }
                    (FilterState::None, false) => {
                        record_scan_mode(MODE_H);
                        ivf_first_batch(state, index, heap, snapshot, &meta, query, None)
                    }
                    (FilterState::MatchNothing, false) => {
                        // Trivially exact: nothing can match. Recorded as L
                        // (the exact mode) for the observability helper.
                        record_scan_mode(MODE_L);
                        Vec::new()
                    }
                    (FilterState::Match(pred), false) => {
                        match choose_filtered_mode(index, &meta, pred) {
                            ScanMode::L => {
                                record_scan_mode(MODE_L);
                                let rows = mode_l_scan(index, heap, snapshot, &meta, &query, pred);
                                record_scan_batch(rows.len() as u64);
                                rows
                            }
                            ScanMode::H => {
                                record_scan_mode(MODE_H);
                                let pred = Some(*pred);
                                ivf_first_batch(state, index, heap, snapshot, &meta, query, pred)
                            }
                        }
                    }
                }
            };
            state.results = Some(results);
            state.cursor = 0;
        }

        loop {
            let results = state.results.as_ref().expect("just populated above");
            if state.cursor < results.len() {
                let (tid, _dist) = results[state.cursor];
                state.cursor += 1;
                (*scan).xs_heaptid = tid;
                // Text-hash filters are candidate tests, not proofs — the
                // executor rechecks the original qual per tuple (module
                // docstring); int filters and unfiltered scans are exact.
                (*scan).xs_recheck = state.recheck;
                // See module docstring: local mode never hands Postgres an
                // order-by value to reorder against — our own sort is already
                // exact for whatever subset we return.
                (*scan).xs_recheckorderby = false;
                return true;
            }
            // The executor drained the batch and still wants rows. An
            // iterative scan probes more lists (a batch may come back empty:
            // keep going); anything else is over.
            let Some(batch) = next_iterative_batch(scan, state) else {
                return false;
            };
            state.results = Some(batch);
            state.cursor = 0;
        }
    }
}

/// The next batch of an iterative scan, or `None` when it is over: every
/// candidate rescored and every list probed, or `pinecone.max_scan_tuples`
/// rescored rows reached (a quiet end, recorded for
/// `pinecone.last_scan_stats()`). The first call of an `auto` scan starts
/// the `relaxed_order` scan it goes on as ([`restart_relaxed`]).
unsafe fn next_iterative_batch(
    scan: pg_sys::IndexScanDesc,
    state: &mut LocalScanState,
) -> Option<Vec<(pg_sys::ItemPointerData, f32)>> {
    unsafe {
        // `auto`: the executor drained the first batch (`off`'s) and wants more.
        if state.cont.is_none()
            && let Some(restart) = state.restart.take()
        {
            state.cont = Some(Box::new(restart_relaxed((*scan).indexRelation, *restart)));
        }
        let cont = state.cont.as_mut()?;
        if cont.exhausted() {
            state.cont = None;
            return None;
        }
        let left = cont.cands.len() - cont.rescored_upto;
        if cont.rescored >= cont.cap {
            record_scan_capped();
            pgrx::debug1!(
                "pg_pinecone: iterative scan stopped by pinecone.max_scan_tuples ({}): {} rows \
                 rescored, {left} candidates and {} of {} lists unprobed",
                cont.cap,
                cont.rescored,
                cont.heads.len() - cont.next,
                cont.heads.len()
            );
            state.cont = None;
            return None;
        }
        pg_sys::check_for_interrupts!();
        let step = cont.nprobe.max(1);
        let (index, heap, snapshot) = (
            (*scan).indexRelation,
            (*scan).heapRelation,
            (*scan).xs_snapshot,
        );
        let mut batch = cont.next_batch(index, heap, snapshot, step);
        cont.settle(&mut batch);
        pgrx::debug1!(
            "pg_pinecone: iterative scan batch: {} rows, {} rescored so far, {left} candidates \
             were left, {} of {} lists probed",
            batch.len(),
            cont.rescored,
            cont.next,
            cont.heads.len()
        );
        Some(batch)
    }
}

// ---------------------------------------------------------------------------
// Mode selection
// ---------------------------------------------------------------------------

enum ScanMode {
    L,
    H,
}

/// `pinecone.force_scan_mode`, parsed. Registered as a plain string GUC (pgrx
/// exposes no check-hook surface), so an invalid value errors here at scan
/// time rather than at `SET` time — with the valid values named.
fn forced_scan_mode() -> Option<ScanMode> {
    let value = FORCE_SCAN_MODE.get();
    let value = value.as_deref()?;
    match value.to_str() {
        Ok("auto") | Ok("") => None,
        Ok("l") | Ok("L") => Some(ScanMode::L),
        Ok("h") | Ok("H") => Some(ScanMode::H),
        _ => error!(
            "pg_pinecone: invalid pinecone.force_scan_mode value {value:?} — valid values are 'auto', \
             'l', and 'h'"
        ),
    }
}

/// The mode decision (module docstring): forced mode if set, else the
/// stats sketch's estimated qualifying fraction vs `pinecone.mode_l_threshold`.
/// Always logs the decision at DEBUG1 so a plan run under
/// `client_min_messages = debug1` shows which mode served it.
fn choose_filtered_mode(index: pg_sys::Relation, meta: &MetaPage, pred: &Predicate) -> ScanMode {
    if let Some(forced) = forced_scan_mode() {
        let name = match forced {
            ScanMode::L => "L",
            ScanMode::H => "H",
        };
        pgrx::debug1!(
            "pg_pinecone: filtered scan mode={name} (forced by pinecone.force_scan_mode)"
        );
        return forced;
    }
    let est = if meta.stats_start == INVALID_BLOCK {
        // No stats page (shouldn't happen for a filtered clustered index —
        // ambuild always writes one) — assume unselective, the mode that
        // degrades gracefully.
        1.0
    } else if let Some(value) = pred.point() {
        // The CURRENT on-disk insert counter, not the rd_amcache'd copy —
        // a long-lived backend's cache never learns about other backends'
        // (or its own) inserts, and the staleness correction must be
        // monotone in the true insert count to keep its toward-mode-L
        // guarantee (`stats.rs`'s insert-staleness bound). One extra
        // buffered block-0 read per FILTERED scan only, alongside the
        // stats-page read this branch already pays.
        let inserted = unsafe { metapage::read_inserted_count_fresh(index) };
        unsafe { stats::read(index, meta.stats_start) }.estimate_fraction(value, inserted)
    } else {
        // The sketch only knows equality: a wider interval is assumed
        // unselective, like a missing stats page.
        1.0
    };
    let threshold = MODE_L_THRESHOLD.get();
    let mode = if est <= threshold {
        ScanMode::L
    } else {
        ScanMode::H
    };
    pgrx::debug1!(
        "pg_pinecone: filtered scan mode={} estimated_fraction={est:.6} mode_l_threshold={threshold}",
        match mode {
            ScanMode::L => "L",
            ScanMode::H => "H",
        }
    );
    mode
}

// ---------------------------------------------------------------------------
// Flat (small-N exact bootstrap) scan
// ---------------------------------------------------------------------------

/// Small-N exact bootstrap: a plain forward table scan, computing the true
/// distance for every live, non-NULL-vector tuple and sorting — recall 1.0
/// by construction, the correctness floor below `small_n_threshold`. A
/// filter scankey is applied right here against the LIVE heap value of the
/// filter column (a flat index stores no per-record data at all): NULL
/// filter values never match; text values compare by hash, with the
/// executor recheck discarding any collision (module docstring).
unsafe fn exact_scan_all(
    heap: pg_sys::Relation,
    snapshot: pg_sys::Snapshot,
    meta: &MetaPage,
    query: &[f32],
    filter: &FilterState,
) -> Vec<(pg_sys::ItemPointerData, f32)> {
    unsafe {
        if matches!(filter, FilterState::MatchNothing) {
            return Vec::new();
        }
        let sscan = pg_sys::table_beginscan(heap, snapshot, 0, std::ptr::null_mut());
        let slot = pg_sys::table_slot_create(heap, std::ptr::null_mut());

        let mut out = Vec::new();
        while pg_sys::table_scan_getnextslot(
            sscan,
            pg_sys::ScanDirection::ForwardScanDirection,
            slot,
        ) {
            if let FilterState::Match(pred) = filter {
                let mut fisnull = false;
                let fdatum =
                    pg_sys::slot_getattr(slot, meta.filter_heap_attnum as i32, &mut fisnull);
                if fisnull || !pred.matches(meta.filter_kind.datum_to_value(fdatum)) {
                    continue;
                }
            }
            let mut isnull = false;
            let datum = pg_sys::slot_getattr(slot, meta.heap_attnum as i32, &mut isnull);
            if !isnull {
                let vector = crate::am::parse_vector(datum);
                let dist = exact_distance(meta.metric, query, &vector);
                out.push(((*slot).tts_tid, dist));
            }
        }

        pg_sys::ExecDropSingleTupleTableSlot(slot);
        pg_sys::table_endscan(sscan);

        // `total_cmp`, not `partial_cmp().expect(...)`: a NaN distance —
        // reachable from an engineered query vector containing `Inf`/`-Inf`
        // (finite `f32` inputs can still combine into a NaN distance, e.g.
        // `Inf - Inf` inside a squared-L2 sum) — must never abort the whole
        // scan. `total_cmp` gives every `f32` a real total order (NaN sorts as
        // the single largest value,
        // per IEEE 754's totalOrder predicate), so a NaN-scored row simply
        // sorts last — never higher-ranked than a genuinely-scored one, and
        // trivially dropped by any `LIMIT`/rescore-pool cut same as an
        // ordinary low-ranked candidate.
        out.sort_by(|a, b| a.1.total_cmp(&b.1));
        out
    }
}

// ---------------------------------------------------------------------------
// Clustered scans (mode H / mode L)
// ---------------------------------------------------------------------------

/// The flat per-entry stride of the cached directory payload
/// (`metapage::AmCache`'s trailing bytes): `dim` centroid f32s + a `u32`
/// list-head block number. Deliberately EXCLUDES the on-page entry's
/// trailing `list_count` (`pages::dir_entry_size` = this + 4), which no
/// scan path reads — and the on-page layout leads with exactly these
/// `stride` bytes, so the fill path copies each entry contiguously.
fn cached_dir_stride(dim: usize) -> usize {
    dim * 4 + 4
}

/// A borrowed view over the `rd_amcache` flat directory payload — see
/// [`metapage::AmCache`] for the layout and the staleness-safety argument.
/// The pointer stays valid for the current scan (or the current
/// `aminsert` call — `local::insert` assigns new records against this same
/// cache): `rd_amcache` only resets on relcache invalidation, which cannot
/// happen mid-call under the caller's own relation lock.
pub(crate) struct CachedDirectory {
    base: *const u8,
    /// The rotated centroids: `n_lists` slots of `padded` f32, the first `n_entries` filled.
    rotated: *const f32,
    pub(crate) n_entries: usize,
    dim: usize,
    padded: usize,
}

impl CachedDirectory {
    /// Entry `i`'s centroid. Alignment: the payload base is MAXALIGN'd
    /// (`MemoryContextAlloc`) plus `size_of::<AmCache>()` (a multiple of 8 —
    /// `MetaPage` holds a `u64`), and the stride is a multiple of 4, so
    /// every centroid span is f32-aligned.
    pub(crate) unsafe fn centroid(&self, i: usize) -> &[f32] {
        debug_assert!(i < self.n_entries);
        unsafe {
            std::slice::from_raw_parts(
                self.base.add(i * cached_dir_stride(self.dim)) as *const f32,
                self.dim,
            )
        }
    }

    /// Entry `i`'s rotated centroid `P^T c`, `D` floats: what the per-list query prep subtracts.
    pub(crate) unsafe fn rotated(&self, i: usize) -> &[f32] {
        debug_assert!(i < self.n_entries);
        unsafe { std::slice::from_raw_parts(self.rotated.add(i * self.padded), self.padded) }
    }

    pub(crate) unsafe fn list_head(&self, i: usize) -> pg_sys::BlockNumber {
        unsafe {
            std::ptr::read_unaligned(
                self.base
                    .add(i * cached_dir_stride(self.dim) + self.dim * 4)
                    as *const u32,
            )
        }
    }
}

/// Returns the centroid directory as a flat, backend-cached view — reading
/// and parsing the directory's page chain only on the FIRST clustered scan
/// per relcache lifetime (this used to happen on EVERY query —
/// at 1M x 768d that's ~3MB of locked page reads plus one `Vec` allocation
/// per centroid, per query). The fill path replaces the metapage-only
/// `rd_amcache` payload `metapage::read` installed with a combined
/// metapage+directory allocation (same `rd_indexcxt` context) and frees the
/// old one.
pub(crate) unsafe fn cached_directory(index: pg_sys::Relation, meta: &MetaPage) -> CachedDirectory {
    unsafe {
        let dim = meta.dim as usize;
        let padded = rotation::padded_dim(dim);
        let header = std::mem::size_of::<metapage::AmCache>();
        let stride = cached_dir_stride(dim);
        // checked_mul/checked_add, not a plain `*`/`+`: `meta.n_lists` is
        // bounded by `metapage::read`'s own validation by the time it reaches
        // here, but this allocation is the one place a bound miss anywhere
        // upstream would turn straight into
        // an out-of-bounds/wildly-oversized `palloc` — fail with a clear
        // corruption error instead of wrapping into an undersized buffer (or
        // panicking with a generic overflow message) on any platform/build.
        // The rotated centroids sit after the directory entries, at a fixed offset.
        let overflow = || -> ! {
            error!(
                "pg_pinecone: corrupt metapage — directory allocation size overflows (n_lists {}, \
                 dim {dim})",
                meta.n_lists
            )
        };
        let n_lists = meta.n_lists as usize;
        let rotated_off = n_lists
            .checked_mul(stride)
            .and_then(|body| body.checked_add(header))
            .unwrap_or_else(|| overflow());
        let total = n_lists
            .checked_mul(padded * 4)
            .and_then(|rotated| rotated.checked_add(rotated_off))
            .unwrap_or_else(|| overflow());

        let cache = (*index).rd_amcache as *mut metapage::AmCache;
        // `metapage::read` always runs before any scan path, so the cache
        // pointer itself can't be NULL here.
        assert!(
            !cache.is_null(),
            "pg_pinecone: metapage cache missing before directory read"
        );
        if (*cache).dir_entries > 0 {
            return CachedDirectory {
                base: (cache as *const u8).add(header),
                rotated: (cache as *const u8).add(rotated_off) as *const f32,
                n_entries: (*cache).dir_entries as usize,
                dim,
                padded,
            };
        }

        // Huge: the rotated centroids make the payload `n_lists * (dim + D)` floats, which a very
        // large directory can push past `MaxAllocSize` (1 GB).
        let fresh =
            pg_sys::MemoryContextAllocHuge((*index).rd_indexcxt, total) as *mut metapage::AmCache;
        let mut out = (fresh as *mut u8).add(header);
        let mut n_entries: usize = 0;

        let max_hops = metapage::max_chain_hops(index);
        let mut hops = 0u64;
        let mut block = meta.centroid_dir_start;
        while block != INVALID_BLOCK {
            pg_sys::check_for_interrupts!();
            hops += 1;
            if hops > max_hops {
                error!(
                    "pg_pinecone: centroid directory chain exceeds the relation's own block count — \
                     cyclic or corrupt chain"
                );
            }
            let buffer = pg_sys::ReadBufferExtended(
                index,
                pg_sys::ForkNumber::MAIN_FORKNUM,
                block,
                pg_sys::ReadBufferMode::RBM_NORMAL,
                std::ptr::null_mut(),
            );
            pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_SHARE as i32);
            let page = pg_sys::BufferGetPage(buffer);
            let contents = pg_sys::PageGetContents(page) as *const u8;
            let buf = std::slice::from_raw_parts(contents, metapage::usable_page_bytes());

            let dir_page = match pages::DirPage::parse(buf, dim) {
                Ok(dir_page) => dir_page,
                Err(problem) => {
                    // Unlock/release BEFORE erroring (matches `metapage::read`'s own unknown-version
                    // branch): an `error!()` raised while holding this buffer's lock/pin leaves it
                    // for the resource owner to force-release at abort instead.
                    pg_sys::UnlockReleaseBuffer(buffer);
                    error!("pg_pinecone: corrupt directory page at block {block} — {problem}");
                }
            };
            let next_block = dir_page.next_block;
            for i in 0..dir_page.n_entries {
                if n_entries >= meta.n_lists as usize {
                    pg_sys::UnlockReleaseBuffer(buffer);
                    error!(
                        "pg_pinecone: corrupt centroid directory — more entries across its pages than \
                         the metapage's n_lists ({})",
                        meta.n_lists
                    );
                }
                // The on-page entry leads with exactly the flat layout's bytes (centroid f32s +
                // list head) — one contiguous copy.
                std::ptr::copy_nonoverlapping(dir_page.entry(i).as_ptr(), out, stride);
                out = out.add(stride);
                n_entries += 1;
            }

            pg_sys::UnlockReleaseBuffer(buffer);
            block = next_block;
        }

        // `P^T c` for every entry, once per relcache lifetime: a rebuild with another seed is a new
        // relfilenode, which resets this cache (module docs of `metapage::AmCache`). The region is
        // zeroed first so every f32 slice handed out below is initialised.
        let rotated = (fresh as *mut u8).add(rotated_off) as *mut f32;
        std::ptr::write_bytes(rotated, 0, n_lists * padded);
        let rotation = Rotation::new(meta.seed, dim);
        let entries = (fresh as *const u8).add(header);
        for i in 0..n_entries {
            let centroid = std::slice::from_raw_parts(entries.add(i * stride) as *const f32, dim);
            let target = std::slice::from_raw_parts_mut(rotated.add(i * padded), padded);
            rotation.apply_into(centroid, target);
        }

        std::ptr::write(
            fresh,
            metapage::AmCache {
                meta: (*cache).meta,
                dir_entries: n_entries as u32,
            },
        );
        pg_sys::pfree(cache as *mut std::ffi::c_void);
        (*index).rd_amcache = fresh as *mut std::ffi::c_void;

        CachedDirectory {
            base: (fresh as *const u8).add(header),
            rotated: rotated as *const f32,
            n_entries,
            dim,
            padded,
        }
    }
}

/// The estimator's per-scan state: the query as the metric scores it, its rotation, the kernel
/// and one reusable [`QueryPrep`]. Built once per scan; [`prepare`](Self::prepare) refills the
/// prep for each probed list. Owned (nothing points into the relcache), so an iterative scan can
/// keep it between `amgettuple` calls.
pub(crate) struct Scorer {
    metric: Metric,
    seed: u64,
    kernel: Kernel,
    /// `q`, or `q / |q|` for cosine (the unit vector the codes are scored against).
    q: Vec<f32>,
    /// `P^T q`, once per scan.
    ptq: Vec<f32>,
    prep: QueryPrep,
}

impl Scorer {
    fn new(meta: &MetaPage, query: &[f32]) -> Scorer {
        if let Some(at) = first_non_finite(query) {
            error!(
                "pg_pinecone: the query vector has a non-finite value (NaN or Infinity) in component \
                 {}",
                at + 1
            );
        }
        let rotation = Rotation::new(meta.seed, meta.dim as usize);
        Scorer::build(meta.metric, meta.seed, kernel_for_scan(), &rotation, query)
    }

    /// The estimator state for a finite `query` against an index of this `metric` and `seed`, whose
    /// `rotation` the caller built once. It reads no Postgres state, so the build's calibration
    /// scores its sample queries with it on worker threads: the same arithmetic as a scan's.
    pub(crate) fn build(
        metric: Metric,
        seed: u64,
        kernel: Kernel,
        rotation: &Rotation,
        query: &[f32],
    ) -> Scorer {
        let q = if metric == Metric::Cosine {
            let mut unit = Vec::new();
            normalise_into(query, &mut unit);
            unit
        } else {
            query.to_vec()
        };
        let ptq = rotation.apply(&q);
        Scorer {
            metric,
            seed,
            kernel,
            q,
            ptq,
            prep: QueryPrep::new(rotation.padded()),
        }
    }

    /// Fills the prep for list number `list`, whose centroid is `c` (rotated: `ptc`): the form
    /// with the smaller error scale for IP and cosine, the centred one for L2.
    pub(crate) fn prepare(&mut self, list: u32, c: &[f32], ptc: &[f32]) {
        rabitq::prepare_query_into(
            &mut self.prep,
            self.metric,
            &self.q,
            &self.ptq,
            c,
            ptc,
            self.seed,
            list,
        );
    }

    /// How a page's blocks are scored against the list last [`prepare`](Self::prepare)d; `with_lo`
    /// computes the pruning bound too.
    pub(crate) fn scoring(&self, with_lo: bool) -> Scoring<'_> {
        Scoring {
            kernel: self.kernel,
            prep: &self.prep,
            metric: self.metric,
            dp: self.ptq.len(),
            with_lo,
        }
    }

    /// The key (estimated distance, ascending is better) of one stored record, against the list
    /// last [`prepare`](Self::prepare)d: `code` is its `D / 64` words, `dot`, `rnorm` and `rc` its
    /// stored scalars. What a scan ranks its candidates by.
    #[inline]
    pub(crate) fn key(&self, code: &[u64], dot: f32, rnorm: f32, rc: f32) -> f32 {
        let (est, _err) = rabitq::estimate_bits(self.kernel, &self.prep, code, dot);
        rabitq::key_from_parts(self.metric, &self.prep, rnorm, rc, est)
    }
}

/// The popcount kernel of this backend, reporting once through the server log any path the
/// startup check rejected (none on healthy hardware).
pub(crate) fn kernel_for_scan() -> Kernel {
    static REPORTED: AtomicBool = AtomicBool::new(false);
    let kernel = Kernel::detect();
    if !REPORTED.swap(true, Ordering::Relaxed) {
        for isa in kernel::rejected() {
            pgrx::log!(
                "pg_pinecone: the {isa:?} popcount kernel disagrees with the scalar path and is not \
                 used"
            );
        }
    }
    kernel
}

/// Walks one list's page chain and hands each page to [`blockscan::scan_page`], which scores its
/// blocks against the query prepared for this list (`scoring`, from [`Scorer::scoring`] after
/// [`Scorer::prepare`]) and admits what `buf` takes.
///
/// `filter` is mode H's in-loop intersection (module docstring): the inclusive range of the pushed
/// key; a record whose stored filter value is NULL or outside it is skipped BEFORE it is scored (a
/// block none of whose records passes is not scored at all). When `None`, the filter arrays are
/// never read (an unfiltered scan pays nothing for them, whether or not they exist).
unsafe fn scan_list(
    index: pg_sys::Relation,
    head_block: pg_sys::BlockNumber,
    meta: &MetaPage,
    scoring: &Scoring<'_>,
    filter: Option<(i64, i64)>,
    buf: &mut CandBuf,
    counters: &mut ScanCounters,
) {
    unsafe {
        // Worked out once per list: every page of an index has the same geometry. A width the
        // metapage validation let through cannot fail it.
        let geometry = match Geometry::checked(meta.code_bytes(), meta.has_filter()) {
            Ok(geometry) => geometry,
            Err(problem) => error!("pg_pinecone: corrupt metapage — {problem}"),
        };
        let max_hops = metapage::max_chain_hops(index);
        let mut hops = 0u64;
        let mut block = head_block;
        while block != INVALID_BLOCK {
            // A corrupt/cyclic chain must
            // ERROR, not spin the backend forever inside an uninterruptible
            // loop — `check_for_interrupts!` keeps a long (bounded, but
            // possibly slow against a huge index) walk cancellable, and the
            // hop count itself is the hard backstop for an actually-cyclic
            // chain.
            pg_sys::check_for_interrupts!();
            hops += 1;
            if hops > max_hops {
                error!(
                    "pg_pinecone: list page chain exceeds the relation's own block count — cyclic or \
                     corrupt chain"
                );
            }
            let buffer = pg_sys::ReadBufferExtended(
                index,
                pg_sys::ForkNumber::MAIN_FORKNUM,
                block,
                pg_sys::ReadBufferMode::RBM_NORMAL,
                std::ptr::null_mut(),
            );
            pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_SHARE as i32);
            let page = pg_sys::BufferGetPage(buffer);
            let contents = pg_sys::PageGetContents(page) as *const u8;
            let bytes = std::slice::from_raw_parts(contents, metapage::usable_page_bytes());

            let list_page = match ListPage::parse_with(bytes, &geometry) {
                Ok(list_page) => list_page,
                Err(problem) => {
                    pg_sys::UnlockReleaseBuffer(buffer);
                    error!("pg_pinecone: corrupt list page at block {block} — {problem}");
                }
            };
            let next_block = list_page.next_block;
            blockscan::scan_page(&list_page, scoring, filter, buf, counters);
            pg_sys::UnlockReleaseBuffer(buffer);
            block = next_block;
        }
    }
}

/// A stored `(block, offset)` as the executor's TID.
fn tid_of((block, offset): (u32, u16)) -> pg_sys::ItemPointerData {
    let mut tid = pg_sys::ItemPointerData {
        ip_blkid: pg_sys::BlockIdData { bi_hi: 0, bi_lo: 0 },
        ip_posid: 0,
    };
    pgrx::itemptr::item_pointer_set_all(&mut tid, block, offset);
    tid
}

/// Fetches the indexed vector column of the heap tuple at `tid` (its
/// currently-visible version under `snapshot`) into `out`, returning `false`
/// if it's gone (concurrently deleted, or a stale/dead candidate — a stale
/// index entry is simply dropped from the result set, exactly how a live
/// index's own visibility check would behave).
///
/// The rescore loop is the scan's dominant cost (measured:
/// ~27.6us per fetched candidate at 768d, linear in the pool size, dwarfing
/// the code scan) — so this path allocates NOTHING per fetch: the vector
/// is copied into the caller's reusable `out` scratch (replacing a fresh
/// 3KB `Vec` per candidate via `parse_vector`), and a detoasted copy (768d
/// vectors exceed the TOAST threshold, so most fetches decompress into a
/// fresh palloc) is `pfree`d immediately instead of accumulating
/// pool-size x 3KB of dead memory per query in the scan's context.
unsafe fn fetch_vector_into(
    fetch: *mut pg_sys::IndexFetchTableData,
    slot: *mut pg_sys::TupleTableSlot,
    snapshot: pg_sys::Snapshot,
    tid: pg_sys::ItemPointerData,
    heap_attnum: i16,
    out: &mut [f32],
) -> bool {
    unsafe {
        let mut tid_mut = tid;
        let mut call_again = false;
        let mut all_dead = false;
        let found = pg_sys::table_index_fetch_tuple(
            fetch,
            &mut tid_mut,
            snapshot,
            slot,
            &mut call_again,
            &mut all_dead,
        );
        if !found {
            return false;
        }
        let mut isnull = false;
        let datum = pg_sys::slot_getattr(slot, heap_attnum as i32, &mut isnull);
        if isnull {
            return false;
        }
        let orig = datum.cast_mut_ptr::<pg_sys::varlena>();
        let v = pg_sys::pg_detoast_datum(orig);
        let base = v as *const u8;
        // Same pgvector `vector` layout as `am::parse_vector`: varlena
        // header (4B), int16 dim, int16 unused, then dim f32s.
        let vdim = std::ptr::read_unaligned(base.add(4) as *const i16) as usize;
        if vdim != out.len() {
            // A same-column heap value can't legitimately differ from the
            // index's dimension (typmod-enforced) — corrupt data deserves an
            // error, not a silent skip that reads as "deleted row".
            error!(
                "pg_pinecone: heap tuple vector dimension {vdim} != index dimension {} — corrupt data?",
                out.len()
            );
        }
        std::ptr::copy_nonoverlapping(base.add(8), out.as_mut_ptr() as *mut u8, out.len() * 4);
        if !std::ptr::eq(v, orig) {
            pg_sys::pfree(v as *mut std::ffi::c_void);
        }
        true
    }
}

/// Exact-rescores `tids` from the heap (visibility check included — a
/// vanished tuple just drops out) and returns them sorted by true distance.
unsafe fn rescore_and_sort(
    heap: pg_sys::Relation,
    snapshot: pg_sys::Snapshot,
    meta: &MetaPage,
    query: &[f32],
    tids: impl Iterator<Item = pg_sys::ItemPointerData>,
) -> Vec<(pg_sys::ItemPointerData, f32)> {
    unsafe {
        let fetch = pg_sys::table_index_fetch_begin(heap);
        let slot = pg_sys::table_slot_create(heap, std::ptr::null_mut());
        // One scratch for the whole rescore (see `fetch_vector_into`).
        let mut scratch = vec![0f32; meta.dim as usize];
        let mut rescored = Vec::new();
        for tid in tids {
            // A batch is up to a pool of heap fetches, and an iterative scan
            // runs several per `amgettuple`: stay cancellable.
            pg_sys::check_for_interrupts!();
            if fetch_vector_into(fetch, slot, snapshot, tid, meta.heap_attnum, &mut scratch) {
                let dist = exact_distance(meta.metric, query, &scratch);
                rescored.push((tid, dist));
            }
        }
        pg_sys::ExecDropSingleTupleTableSlot(slot);
        pg_sys::table_index_fetch_end(fetch);

        // See `exact_scan_all`'s own comment: `total_cmp`, never
        // `partial_cmp().expect(...)` — a NaN exact distance sorts last, not
        // an aborted scan.
        rescored.sort_by(|a, b| a.1.total_cmp(&b.1));
        rescored
    }
}

/// The progressive rescore of `cands` (sorted by estimate): the heap behind
/// [`blockscan::rescore_progressive`]. Returns what it fetched (indices into `cands`) and the rows it
/// found with their exact distances, nearest first. The early stop compares keys, which for L2 are
/// squared distances (`exact_key`), so `tau` and a candidate's bound are in one unit.
unsafe fn rescore_candidates(
    heap: pg_sys::Relation,
    snapshot: pg_sys::Snapshot,
    meta: &MetaPage,
    query: &[f32],
    cands: &[Candidate],
    plan: &Plan,
    prune: bool,
) -> (blockscan::Outcome, Vec<(pg_sys::ItemPointerData, f32)>) {
    unsafe {
        let fetch = pg_sys::table_index_fetch_begin(heap);
        let slot = pg_sys::table_slot_create(heap, std::ptr::null_mut());
        // One scratch for the whole rescore (see `fetch_vector_into`).
        let mut scratch = vec![0f32; meta.dim as usize];
        let outcome = blockscan::rescore_progressive(cands, plan, prune, &mut |c| {
            // A batch is up to a pool of heap fetches, and an iterative scan runs several per
            // `amgettuple`: stay cancellable.
            pg_sys::check_for_interrupts!();
            fetch_vector_into(
                fetch,
                slot,
                snapshot,
                tid_of(c.tid),
                meta.heap_attnum,
                &mut scratch,
            )
            .then(|| exact_key(meta.metric, query, &scratch))
        });
        pg_sys::ExecDropSingleTupleTableSlot(slot);
        pg_sys::table_index_fetch_end(fetch);

        let mut rows: Vec<(pg_sys::ItemPointerData, f32)> = outcome
            .found
            .iter()
            .map(|&(i, key)| (tid_of(cands[i].tid), key_to_distance(meta.metric, key)))
            .collect();
        // `nan_last`, never `partial_cmp().expect(...)`: a NaN exact distance sorts last (whatever its
        // sign), not an aborted scan. Equal distances (f32 repeats at this scale) go by TID, so the
        // order does not depend on the order the candidates were fetched in (the early stop changes it).
        rows.sort_by(|a, b| {
            nan_last(a.1, b.1).then_with(|| {
                pgrx::itemptr::item_pointer_get_both(a.0)
                    .cmp(&pgrx::itemptr::item_pointer_get_both(b.0))
            })
        });
        (outcome, rows)
    }
}

/// What an iterative mode-H scan carries between `amgettuple` calls: the
/// lists ranked once on the first call, the position of the next unprobed
/// one, the probed lists' candidates not yet rescored, and everything a batch
/// needs. All OWNED: nothing points into the relcache directory (`rd_amcache`
/// can be reset between calls) and no buffer stays pinned (`scan_list`
/// releases per page).
struct IvfCont {
    meta: MetaPage,
    query: Vec<f32>,
    scorer: Scorer,
    /// `(list number, head block)` of the non-empty lists, nearest centroid first. Lists are
    /// disjoint, so a list probed once never comes back: no visited set, no
    /// duplicates.
    heads: Vec<(u32, pg_sys::BlockNumber)>,
    /// Position in `heads` of the next unprobed list.
    next: usize,
    /// The non-empty lists of the index, `heads.len()` but for `off`, which keeps only `nprobe`.
    n_lists: usize,
    /// The last probe kept every record it scanned (the bounded buffer dropped none).
    kept_all: bool,
    /// Lists in the first batch (`effective_nprobe`); a continuation batch
    /// probes as many again.
    nprobe: usize,
    pred: Option<Predicate>,
    /// Rows exact-rescored per batch ([`effective_pool`]).
    pool: usize,
    /// `pinecone.max_scan_tuples`: no batch starts once this many rows have
    /// been rescored, and a slice is cut to what is left.
    cap: u64,
    rescored: u64,
    /// A batch was served: every later slice is cut to what the cap leaves, even when the first
    /// batch rescored nothing.
    past_first: bool,
    /// Estimated candidates of the lists probed last, `cands[..rescored_upto]`
    /// already rescored; the rest is partitioned slice by slice as needed.
    cands: Vec<Candidate>,
    rescored_upto: usize,
    /// `strict_order`: rows closer than `last_dist`, the last distance handed
    /// out, are dropped from later batches.
    strict: bool,
    last_dist: f32,
    /// A scan under `iterative_scan = off` (and `auto`'s first batch) keeps the best `pool` candidates
    /// by estimate while it scans; an iterative one keeps them all, since it deepens past any pool,
    /// and rescores every batch whole: the early stop leaves out candidates a batch cannot serve in
    /// order, and a later batch would hand them out after farther rows.
    bounded: bool,
    /// The batch stops its rescore early ([`early_stop`]): `auto`'s first batch only.
    prune: bool,
    /// Sorted heap TIDs of the candidates an `auto` scan's first batch fetched; the restart leaves
    /// them out. Empty for every other scan.
    skip: Vec<(u32, u16)>,
}

impl IvfCont {
    /// Mode-H IVF probe setup (module docstring has the design): rotates the
    /// query and ranks the lists, ready for [`IvfCont::next_batch`]. `prune` is [`early_stop`] for
    /// the setting `mode` was resolved from.
    unsafe fn begin(
        index: pg_sys::Relation,
        meta: &MetaPage,
        query: Vec<f32>,
        pred: Option<Predicate>,
        mode: IterativeScan,
        prune: bool,
    ) -> Self {
        unsafe {
            debug_assert_ne!(mode, IterativeScan::Auto, "the caller resolves `auto`");
            // The query's rotation (`P^T q`), the kernel and the prep buffers: once per scan —
            // never recomputed per candidate or per list.
            let scorer = Scorer::new(meta, &query);

            let directory = cached_directory(index, meta);

            // Score ALL centroids exactly — cheap: n_lists is
            // O(sqrt(row count)), never the full dataset. Empty lists
            // (`list_head == INVALID_BLOCK`, from build-time k-means leaving a
            // centroid memberless) are filtered out BEFORE ranking: they used
            // to be ranked and could consume `take(nprobe)` slots while
            // contributing zero candidates — at nprobe=1 an empty list whose
            // centroid happened to rank first would eat the entire
            // probe budget and return NOTHING. Filtering first means every
            // probe slot goes to a list that actually holds records — strictly
            // better coverage at identical cost. (`nprobe` itself stays a
            // fraction of the TOTAL directory size, so the probed FRACTION of
            // real data can only grow.)
            let mut ranked_lists: Vec<(usize, f32)> = (0..directory.n_entries)
                .filter(|&i| directory.list_head(i) != INVALID_BLOCK)
                .map(|i| {
                    (
                        i,
                        exact_distance(meta.metric, &query, directory.centroid(i)),
                    )
                })
                .collect();
            // `total_cmp` (see `exact_scan_all`'s comment): a NaN centroid
            // distance ranks that centroid last for probing rather than
            // aborting the scan outright.
            ranked_lists.sort_by(|a, b| a.1.total_cmp(&b.1));

            let nprobe = effective_nprobe(
                meta,
                directory.n_entries,
                pred.is_some(),
                mode != IterativeScan::Off,
            );
            // `off` never probes past the first `nprobe` lists.
            let keep = if mode == IterativeScan::Off {
                nprobe
            } else {
                ranked_lists.len()
            };
            let heads = ranked_lists
                .iter()
                .take(keep)
                .map(|&(i, _)| (i as u32, directory.list_head(i)))
                .collect();
            IvfCont {
                meta: *meta,
                query,
                scorer,
                heads,
                next: 0,
                n_lists: ranked_lists.len(),
                kept_all: true,
                nprobe,
                pred,
                pool: effective_pool(meta, pred.is_some()),
                cap: max_scan_tuples(),
                rescored: 0,
                past_first: false,
                cands: Vec::new(),
                rescored_upto: 0,
                strict: mode == IterativeScan::StrictOrder,
                last_dist: f32::NEG_INFINITY,
                bounded: mode == IterativeScan::Off,
                prune,
                skip: Vec::new(),
            }
        }
    }

    /// Probes the next `n_lists` lists (RaBitQ-estimated candidates, with
    /// the optional in-loop filter intersection) into `cands`. The directory view is fetched
    /// afresh here (a later `amgettuple` may find `rd_amcache` rebuilt): each list's prep needs
    /// its centroid and the cached `P^T c`.
    unsafe fn probe(&mut self, index: pg_sys::Relation, n_lists: usize) {
        unsafe {
            debug_assert!(self.cands.is_empty() && self.rescored_upto == 0);
            let end = self.next.saturating_add(n_lists).min(self.heads.len());
            // Refills `rd_amcache` first if a relcache reset dropped it.
            metapage::read(index);
            let directory = cached_directory(index, &self.meta);
            let mut buf = if self.bounded {
                CandBuf::best(self.pool)
            } else {
                CandBuf::unbounded()
            };
            let mut counters = ScanCounters::default();
            let filter = self.pred.map(|p| (p.lo, p.hi));
            for &(list, head) in &self.heads[self.next..end] {
                debug_assert_ne!(head, INVALID_BLOCK, "empty lists filtered before ranking");
                let entry = list as usize;
                self.scorer
                    .prepare(list, directory.centroid(entry), directory.rotated(entry));
                let scoring = self.scorer.scoring(self.prune && self.bounded);
                scan_list(
                    index,
                    head,
                    &self.meta,
                    &scoring,
                    filter,
                    &mut buf,
                    &mut counters,
                );
            }
            self.cands = buf.into_candidates();
            self.kept_all = self.cands.len() as u64 == counters.scanned;
            drop_seen(&mut self.cands, &self.skip);
            record_scan_counters(&counters);
            record_lists_probed(end - self.next);
            self.next = end;
        }
    }

    /// Every candidate rescored and every list probed.
    fn exhausted(&self) -> bool {
        self.rescored_upto >= self.cands.len() && self.next >= self.heads.len()
    }

    /// The next batch, exact-rescored, best first: the next `pool` candidates
    /// of the probed lists by sketch distance (deepen), or once none are left
    /// the best `pool` of the next `n_lists` lists (widen).
    unsafe fn next_batch(
        &mut self,
        index: pg_sys::Relation,
        heap: pg_sys::Relation,
        snapshot: pg_sys::Snapshot,
        n_lists: usize,
    ) -> Vec<(pg_sys::ItemPointerData, f32)> {
        unsafe {
            if self.rescored_upto < self.cands.len() {
                record_scan_deepened();
            } else {
                self.probe(index, n_lists);
            }

            let from = self.rescored_upto;
            let left = self.cands.len() - from;
            let mut take = left.min(self.pool);
            // The first batch is never cut by the cap; a later slice stops at it.
            if self.past_first {
                take = take.min(self.cap.saturating_sub(self.rescored) as usize);
            }
            self.past_first = true;
            // Only the slice's membership matters (it is rescored and re-sorted
            // below), so partition in O(n) rather than sort. `nan_last`: a
            // NaN estimate lands on the worse side instead of panicking.
            if take > 0 && take < left {
                self.cands[from..].select_nth_unstable_by(take - 1, |a, b| nan_last(a.est, b.est));
            }
            // `auto`'s first batch stops its rescore early (module docs of `blockscan`); `off` and an
            // iterative scan rescore every slice whole.
            let prune = self.prune && self.bounded;
            // An empty probe step (a pushed key rejected every record) opens no heap fetch.
            let rows = if take == 0 {
                record_scan_batch(0);
                Vec::new()
            } else {
                let slice = &mut self.cands[from..from + take];
                // The early stop walks the slice best estimate first; a whole slice needs no order.
                if prune {
                    slice.sort_by(|a, b| nan_last(a.est, b.est));
                }
                let plan = Plan::for_pool(self.pool, ASSUMED_TOP_K);
                let (outcome, rows) = rescore_candidates(
                    heap,
                    snapshot,
                    &self.meta,
                    &self.query,
                    slice,
                    &plan,
                    prune,
                );
                // The fetched candidates go to the front, in order: what is left of the slice is
                // what a deeper batch rescores first.
                for (to, &at) in outcome.fetched.iter().enumerate() {
                    slice.swap(to, at);
                }
                self.rescored_upto = from + outcome.fetched.len();
                self.rescored += outcome.fetched.len() as u64;
                record_scan_batch(outcome.fetched.len() as u64);
                rows
            };
            // A bounded (`off`) batch is the first and only one, and an `auto` scan carries the
            // candidates it fetched over to its restart.
            if !self.bounded && self.rescored_upto == self.cands.len() {
                self.cands = Vec::new();
                self.rescored_upto = 0;
            }
            rows
        }
    }

    /// `strict_order`: drops the rows of a freshly sorted batch that are
    /// closer than the last distance handed out, then remembers this batch's
    /// last one (never a NaN). Without it every batch is served as it is.
    fn settle(&mut self, batch: &mut Vec<(pg_sys::ItemPointerData, f32)>) {
        if !self.strict {
            return;
        }
        let dropped = batch.partition_point(|r| nan_last(r.1, self.last_dist).is_lt());
        batch.drain(..dropped);
        if let Some(&(_, dist)) = batch.last()
            && !dist.is_nan()
        {
            self.last_dist = dist;
        }
    }
}

/// Mode-H first batch: the `nprobe` nearest lists' exact-rescored pool, which
/// is all `off` returns. Under an iterative mode the state for more batches
/// stays in `state.cont` (unless nothing is left); under `off` it lives for
/// this call only (the query is moved in, not copied, and only the first
/// `nprobe` lists are kept).
///
/// `auto` returns `off`'s batch with the early stop and keeps a [`Restart`]; one that drains it
/// starts the `relaxed_order` scan in [`next_iterative_batch`], unless the batch fetched every
/// record of every list.
unsafe fn ivf_first_batch(
    state: &mut LocalScanState,
    index: pg_sys::Relation,
    heap: pg_sys::Relation,
    snapshot: pg_sys::Snapshot,
    meta: &MetaPage,
    query: Vec<f32>,
    pred: Option<Predicate>,
) -> Vec<(pg_sys::ItemPointerData, f32)> {
    unsafe {
        let setting = iterative_scan();
        let mode = match setting {
            IterativeScan::Auto => IterativeScan::Off,
            explicit => explicit,
        };
        let prune = early_stop(setting, scan_prune());
        let mut cont = IvfCont::begin(index, meta, query, pred, mode, prune);
        let mut first = cont.next_batch(index, heap, snapshot, cont.nprobe);
        if setting == IterativeScan::Auto {
            // Every record of every list fetched: a restart would find only the rows returned.
            let whole = cont.next >= cont.n_lists
                && cont.kept_all
                && cont.rescored_upto == cont.cands.len();
            if !whole {
                // The candidates the batch fetched sit first in `cands`.
                cont.cands.truncate(cont.rescored_upto);
                state.restart = Some(Box::new(Restart {
                    query: cont.query,
                    pred,
                    fetched: cont.cands,
                }));
            }
        } else if mode != IterativeScan::Off && !cont.exhausted() {
            cont.settle(&mut first);
            state.cont = Some(Box::new(cont));
        }
        first
    }
}

/// The `relaxed_order` scan an `auto` scan goes on as once the executor drained its first batch
/// and wants more. It probes from the nearest lists again (the first batch kept only its best
/// candidates), leaves out the candidates the batch fetched, and counts them against
/// `pinecone.max_scan_tuples`.
unsafe fn restart_relaxed(index: pg_sys::Relation, restart: Restart) -> IvfCont {
    unsafe {
        let meta = metapage::read(index);
        let mut cont = IvfCont::begin(
            index,
            &meta,
            restart.query,
            restart.pred,
            IterativeScan::RelaxedOrder,
            false,
        );
        cont.rescored = restart.fetched.len() as u64;
        cont.past_first = true;
        cont.skip = restart.fetched.iter().map(|c| c.tid).collect();
        cont.skip.sort_unstable();
        cont
    }
}

/// Removes the candidates whose TID is in `seen` (sorted).
fn drop_seen(cands: &mut Vec<Candidate>, seen: &[(u32, u16)]) {
    if !seen.is_empty() {
        cands.retain(|c| seen.binary_search(&c.tid).is_err());
    }
}

/// Mode L (module docstring): walk EVERY list page collecting the TIDs whose
/// stored filter value matches, then exact-rescore all of them from the heap
/// — recall 1.0 by construction (the candidate set IS the qualifying set;
/// text-hash collisions are removed by the executor recheck). No code
/// scoring, no rotation, no rescore-pool cap: the match set is small by the
/// very selectivity estimate that routed the scan here, and capping it would
/// forfeit the mode's exactness guarantee.
unsafe fn mode_l_scan(
    index: pg_sys::Relation,
    heap: pg_sys::Relation,
    snapshot: pg_sys::Snapshot,
    meta: &MetaPage,
    query: &[f32],
    pred: &Predicate,
) -> Vec<(pg_sys::ItemPointerData, f32)> {
    unsafe {
        let code_bytes = meta.code_bytes();
        let mut matches: Vec<pg_sys::ItemPointerData> = Vec::new();
        let max_hops = metapage::max_chain_hops(index);

        let heads = super::vacuum::read_list_heads(index, meta);
        record_lists_probed(heads.iter().filter(|&&h| h != INVALID_BLOCK).count());
        for head in heads {
            let mut block = head;
            let mut hops = 0u64;
            while block != INVALID_BLOCK {
                // See `scan_list`'s identical guard for the rationale.
                pg_sys::check_for_interrupts!();
                hops += 1;
                if hops > max_hops {
                    error!(
                        "pg_pinecone: list page chain exceeds the relation's own block count — cyclic \
                         or corrupt chain"
                    );
                }
                let buffer = pg_sys::ReadBufferExtended(
                    index,
                    pg_sys::ForkNumber::MAIN_FORKNUM,
                    block,
                    pg_sys::ReadBufferMode::RBM_NORMAL,
                    std::ptr::null_mut(),
                );
                pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_SHARE as i32);
                let page = pg_sys::BufferGetPage(buffer);
                let contents = pg_sys::PageGetContents(page) as *const u8;
                let buf = std::slice::from_raw_parts(contents, metapage::usable_page_bytes());

                let list_page = match ListPage::parse(buf, code_bytes, meta.has_filter()) {
                    Ok(list_page) => list_page,
                    Err(problem) => {
                        pg_sys::UnlockReleaseBuffer(buffer);
                        error!("pg_pinecone: corrupt list page at block {block} — {problem}");
                    }
                };
                for i in 0..list_page.n {
                    if !list_page.fnull(i) && pred.matches(list_page.fval(i)) {
                        matches.push(tid_of(list_page.tid(i)));
                    }
                }
                let next_block = list_page.next_block;

                pg_sys::UnlockReleaseBuffer(buffer);
                block = next_block;
            }
        }

        rescore_and_sort(heap, snapshot, meta, query, matches.into_iter())
    }
}

/// TEST-ONLY (`#[cfg(feature = "pg_test")]` — never in a release build):
/// overwrites directory page `block`'s on-page `n_entries` header field
/// directly, bypassing every code path that would otherwise validate it —
/// crafts the exact corruption [`cached_directory`]'s own bounds checks
/// exist to catch, the same "poke raw page
/// bytes, then invalidate relcache" technique
/// `build::test_overwrite_directory_entry` already uses for a different
/// field of the same page kind.
#[cfg(feature = "pg_test")]
pub(crate) unsafe fn test_corrupt_directory_n_entries(
    index: pg_sys::Relation,
    block: pg_sys::BlockNumber,
    n_entries: u32,
) {
    unsafe {
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
        // `n_entries` is the directory page header's second `u32` (right
        // after `next_block`) — see `pages::DIR_HEADER_SIZE`/module docs.
        std::ptr::copy_nonoverlapping(n_entries.to_le_bytes().as_ptr(), contents.add(4), 4);
        pg_sys::GenericXLogFinish(state);
        pg_sys::UnlockReleaseBuffer(buffer);

        pg_sys::CacheInvalidateRelcacheByRelid((*index).rd_id);
    }
}

#[cfg(test)]
mod tests {
    use super::{Candidate, Predicate, drop_seen, early_stop, nprobe_for, pool_for};
    use crate::am::IterativeScan;
    use crate::local::options::legacy_nprobe;

    fn candidate(block: u32, offset: u16) -> Candidate {
        Candidate {
            tid: (block, offset),
            est: 0.0,
            lo: f32::NEG_INFINITY,
        }
    }

    fn tids(cands: &[Candidate]) -> Vec<(u32, u16)> {
        cands.iter().map(|c| c.tid).collect()
    }

    #[test]
    fn drop_seen_removes_exactly_the_tids_it_is_given() {
        let mut cands: Vec<Candidate> = [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)]
            .into_iter()
            .map(|(block, offset)| candidate(block, offset))
            .collect();
        drop_seen(&mut cands, &[(0, 2), (1, 1), (2, 2)]);
        assert_eq!(tids(&cands), [(0, 1), (1, 2), (2, 1)]);
    }

    #[test]
    fn drop_seen_with_nothing_seen_or_nothing_in_common_changes_nothing() {
        let mut cands = vec![candidate(3, 1), candidate(1, 7)];
        drop_seen(&mut cands, &[]);
        assert_eq!(tids(&cands), [(3, 1), (1, 7)]);
        drop_seen(&mut cands, &[(0, 1), (3, 2), (9, 9)]);
        assert_eq!(
            tids(&cands),
            [(3, 1), (1, 7)],
            "a block and an offset are both compared"
        );
    }

    #[test]
    fn only_the_first_batch_of_auto_stops_early_and_scan_prune_off_stops_it_everywhere() {
        use IterativeScan::{Auto, Off, RelaxedOrder, StrictOrder};
        for (setting, scan_prune, want) in [
            (Auto, true, true),
            (Auto, false, false),
            (Off, true, false),
            (Off, false, false),
            (RelaxedOrder, true, false),
            (StrictOrder, true, false),
        ] {
            assert_eq!(
                early_stop(setting, scan_prune),
                want,
                "{setting:?}, scan_prune {scan_prune}"
            );
        }
    }

    #[test]
    fn drop_seen_can_empty_the_candidates() {
        let mut cands = vec![candidate(5, 5), candidate(6, 6)];
        drop_seen(&mut cands, &[(5, 5), (6, 6)]);
        assert!(cands.is_empty());
    }

    #[test]
    fn unfiltered_scans_use_the_stored_default_probe_count() {
        assert_eq!(nprobe_for(26, 1000, 0.0, false, false), 26);
        assert_eq!(nprobe_for(127, 1000, 0.0, false, false), 127);
        assert_eq!(nprobe_for(500, 300, 0.0, false, false), 300);
    }

    #[test]
    fn iterative_scans_keep_the_090_probe_count_whatever_was_calibrated() {
        for n in [1usize, 3, 100, 142, 1000, 31_623] {
            for stored in [1, 8, 25, 127] {
                assert_eq!(
                    nprobe_for(stored, n, 0.0, false, true),
                    legacy_nprobe(n as u32) as usize
                );
            }
        }
        assert_eq!(nprobe_for(26, 1000, 0.0, false, true), 127);
    }

    #[test]
    fn the_pool_is_the_session_setting_then_the_index_default_and_a_pushed_key_keeps_the_090_pool()
    {
        assert_eq!(pool_for(0, 0, 200, false), 200);
        assert_eq!(pool_for(0, 0, 2000, false), 2000);
        assert_eq!(
            pool_for(0, 0, 200, true),
            1000,
            "a pushed key: the 0.9.0 pool"
        );
        assert_eq!(pool_for(0, 0, 100, true), 1000);
        assert_eq!(pool_for(0, 3, 300, false), 300);
        assert_eq!(
            pool_for(0, 3, 300, true),
            300,
            "a pinned pool applies to a pushed key too"
        );
        for pushed_key in [false, true] {
            assert_eq!(
                pool_for(7, 3, 300, pushed_key),
                700,
                "the session setting wins"
            );
            assert_eq!(pool_for(1000, 0, 200, pushed_key), 100_000);
        }
    }

    #[test]
    fn the_filtered_ramp_is_independent_of_the_unfiltered_default() {
        for stored in [1, 26, 127] {
            assert_eq!(nprobe_for(stored, 1000, 0.0, true, false), 500);
            assert_eq!(nprobe_for(stored, 1000, 0.0, true, true), 500);
            assert_eq!(nprobe_for(stored, 100, 0.0, true, false), 50);
        }
        assert_eq!(nprobe_for(26, 1_000_000, 0.0, true, false), 16_000);
    }

    #[test]
    fn a_session_fraction_wins_on_every_path() {
        for (filtered, iterative) in [(false, false), (true, false), (false, true), (true, true)] {
            assert_eq!(nprobe_for(26, 1000, 0.025, filtered, iterative), 25);
            assert_eq!(nprobe_for(26, 1000, 1.0, filtered, iterative), 1000);
            assert_eq!(nprobe_for(26, 1000, 1e-9, filtered, iterative), 1);
        }
    }

    #[test]
    fn equality_matches_only_its_value() {
        let p = Predicate::eq(7, false);
        assert!(p.matches(7));
        assert!(!p.matches(6) && !p.matches(8));
        assert_eq!(p.point(), Some(7));
        assert!(!p.recheck);
        assert!(Predicate::eq(7, true).recheck);
    }

    #[test]
    fn equality_holds_at_the_i64_extremes() {
        for v in [i64::MIN, -1, 0, 1, i64::MAX] {
            let p = Predicate::eq(v, false);
            assert!(p.matches(v), "{v}");
            assert_eq!(p.point(), Some(v));
        }
        assert!(!Predicate::eq(i64::MAX, false).matches(i64::MIN));
    }

    #[test]
    fn bounds_are_inclusive() {
        let p = Predicate::between(-3, 5, false).unwrap();
        for v in [-3, 0, 5] {
            assert!(p.matches(v), "{v}");
        }
        for v in [-4, 6, i64::MIN, i64::MAX] {
            assert!(!p.matches(v), "{v}");
        }
        assert_eq!(p.point(), None);
        assert_eq!(
            Predicate::between(5, 5, false),
            Some(Predicate::eq(5, false))
        );
        assert_eq!(Predicate::between(6, 5, false), None);
        let all = Predicate::between(i64::MIN, i64::MAX, false).unwrap();
        assert!(all.matches(i64::MIN) && all.matches(0) && all.matches(i64::MAX));
    }

    #[test]
    fn strict_bounds_saturate_to_nothing_at_the_extremes() {
        assert_eq!(Predicate::greater_than(i64::MAX, false), None);
        assert_eq!(Predicate::less_than(i64::MIN, false), None);
        let gt = Predicate::greater_than(i64::MAX - 1, false).unwrap();
        assert!(gt.matches(i64::MAX) && !gt.matches(i64::MAX - 1));
        let lt = Predicate::less_than(i64::MIN + 1, false).unwrap();
        assert!(lt.matches(i64::MIN) && !lt.matches(i64::MIN + 1));
        let gt0 = Predicate::greater_than(0, false).unwrap();
        assert!(gt0.matches(1) && !gt0.matches(0));
    }

    #[test]
    fn intersection_narrows_or_proves_nothing_matches() {
        let a = Predicate::between(0, 10, false).unwrap();
        let b = Predicate::between(5, 20, false).unwrap();
        assert_eq!(a.intersect(b), Predicate::between(5, 10, false));
        assert_eq!(a.intersect(b), b.intersect(a));
        // `x = 1 AND x = 2`, and disjoint ranges: nothing can match.
        assert_eq!(
            Predicate::eq(1, false).intersect(Predicate::eq(2, false)),
            None
        );
        assert_eq!(
            a.intersect(Predicate::between(11, 12, false).unwrap()),
            None
        );
        assert_eq!(
            Predicate::eq(1, false).intersect(Predicate::eq(1, false)),
            Some(Predicate::eq(1, false))
        );
        // The recheck flag survives a merge.
        let hashed = Predicate::eq(4, true);
        assert!(hashed.intersect(Predicate::eq(4, false)).unwrap().recheck);
        assert!(Predicate::eq(4, false).intersect(hashed).unwrap().recheck);
    }
}
