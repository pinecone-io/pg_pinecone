//! Mode-aware `amcostestimate` — replaces `am.rs`'s original
//! mode-blind constant (a flat `10_000 * cpu_operator_cost`, which priced a
//! LOCAL index scan as if it were one remote HTTP round trip, forcing every
//! local-mode test/benchmark to lean on the `enable_seqscan = off` crutch).
//!
//! ## The model (local, clustered)
//! Derived from the measured 768-d cost structure (`p50 ≈ 1ms + 27.6µs ×
//! rescore_pool`, the code pass cheap, heap-rescore TOAST fetches dominant), but
//! PARAMETERIZED in the planner's own cost units rather than transcribing
//! those wall-clock constants — the measured numbers came from a
//! `--enable-cassert` build (all absolutes inflated; relative structure
//! valid), and the planner's job is a *relative* comparison against its own
//! seqscan estimate anyway. Term by term:
//! - centroid-directory + probed-list-page reads: sequential-ish page reads
//!   (`seq_page_cost` each) — sizes computed from the metapage's own
//!   `n_lists`/`dim`/`code_bits`, the same layout math `pages.rs` uses, over the lists the scan
//!   probes (the calibrated count; the ramped one behind a pushed key).
//! - Code pass (RaBitQ estimate): `probed_tuples * cpu_operator_cost *`
//!   [`CODE_OP_FACTOR`] — cheap relative to a full distance eval, hence the
//!   < 1 factor.
//! - exact rescore: the dominant term, exactly as measured. `pool` (the rows the scan rescores: the
//!   index's pool, the 0.9.0 pool behind a pushed key) random heap fetches — priced via the
//!   planner's own Mackert–Lohman estimator ([`pg_sys::index_pages_fetched`],
//!   the same function `genericcostestimate` uses), times `random_page_cost`,
//!   times a TOAST multiplier ([`TOAST_FETCH_PAGES`]) when the vector column
//!   is wide enough to be stored out of line (pgvector's
//!   `typstorage = EXTERNAL` sends >~2KB vectors through toast-index lookup +
//!   chunk pages + reassembly per candidate).
//! - per-candidate CPU: `pool × (cpu_operator_cost + cpu_tuple_cost)`.
//!
//! ## The model (local, flat / small-N bootstrap)
//! A flat scan reads every live heap tuple and computes exact distances —
//! the same work as the seqscan+top-N-sort plan, minus the sort, plus AM
//! dispatch overhead, with no early-exit advantage. There is no honest
//! reason for the planner to prefer it, so it's priced at
//! [`FLAT_OVERHEAD_MULT`] × the seqscan-equivalent work: **seqscan wins on
//! tiny/flat tables by design** (the task's own "correct behavior" case);
//! `enable_seqscan = off` remains the explicit forcing lever tests use to
//! exercise the flat path itself.
//!
//! ## The model (remote / demoting / detached / v1 metapages)
//! Unchanged 0.3 shape: high constant startup approximating one HTTP round
//! trip ([`REMOTE_STARTUP_OPS`] × `cpu_operator_cost` — the exact constant
//! `am.rs` used before, so every existing remote-mode planner interaction is
//! byte-identical) plus a network-bound per-returned-tuple term. Tiny tables
//! still seqscan; big ones use the index.
//!
//! `Promoting` prices as LOCAL and `Demoting` as REMOTE — cost must follow
//! the mode a scan would actually SERVE (`am::ambeginscan`'s dispatch), not
//! the transition's target.
//!
//! ## Insert-path pricing
//! A clustered local `aminsert` currently costs **2 GenericXLog WAL records**
//! (the `inserted_since_build` bump on block 0, then the list append) plus
//! **one exclusive block-0 buffer lock acquisition per insert** — block 0 is
//! a cluster-wide serialization point under concurrent insert load (see
//! `local::insert`'s module docstring, "Price of the separate bump record").
//! **Measured baseline: ~5k inserts/s at 2 concurrent backends** (the
//! crash-drill matrix's concurrent-insert leg, cassert build — relative
//! baseline, not a publishable absolute). The recorded alternative — folding
//! the bump into the append's own record (block 0 + tail [+ fresh page] ≤ 3
//! registered pages, inside `MAX_GENERIC_XLOG_PAGES` = 4) — would halve the
//! WAL record count and close the crash-overcount window, at the cost of
//! holding block 0's exclusive lock across the whole chain-append critical
//! section plus a multi-buffer lock-order proof. **Fold decision: NOT taken** —
//! no new evidence was gathered beyond that baseline, and the call should
//! rest on evidence; the locking discipline built for the metapage mode
//! flip (`metapage::flip_mode` — single-record
//! block-0 write under the buffer's exclusive lock) is where the fold would
//! live if a churn-benchmark rerun ever shows the bump record dominating.
//! This module's cost outputs are scan-side only; insert cost is not part of
//! `amcostestimate`'s contract.

use pgrx::prelude::*;

use super::{metapage::MetaPage, pages::Geometry};

/// Fraction of one `cpu_operator_cost` a single code estimate costs relative
/// to a full-precision distance eval — cheap (the heap rescore dominates),
/// priced > 0 so probing more of a huge index still registers as work.
const CODE_OP_FACTOR: f64 = 0.25;

/// Heap pages per rescore candidate when the vector column is TOASTed
/// (typstorage = EXTERNAL, datum > ~2KB — dims above ~500): toast-index
/// lookup + ~2 chunk pages. Inline vectors
/// cost 1.0 (the tuple's own heap page).
const TOAST_FETCH_PAGES: f64 = 3.0;

/// Postgres's `TOAST_TUPLE_THRESHOLD` is ~2KB; a `vector(d)` datum is
/// `8 + 4d` bytes. Above this dim the rescore fetch pays the TOAST path.
pub(crate) const TOAST_DIM_THRESHOLD: u32 = 500;

/// Flat-mode multiplier over the seqscan-equivalent work — the flat scan is
/// the same heap read + exact distance evals a seqscan does, with AM
/// dispatch overhead and no sort-avoidance advantage, so it must lose the
/// cost comparison (module docstring).
const FLAT_OVERHEAD_MULT: f64 = 2.0;

/// The remote path's startup constant, in `cpu_operator_cost` units —
/// exactly the value `am.rs`'s earlier `amcostestimate` used for everything,
/// kept byte-identical for the remote path so no existing remote-mode
/// planner behavior shifts.
const REMOTE_STARTUP_OPS: f64 = 10_000.0;

/// Per-returned-tuple network cost for the remote path, in
/// `cpu_operator_cost` units — the 0.3 shape's "network-bound per-tuple"
/// term (response parsing + TID translation per match, dwarfed by startup).
const REMOTE_TUPLE_OPS: f64 = 10.0;

/// The planner cost GUCs, captured once per estimate — a plain struct so the
/// arithmetic below is pure and unit-testable without a backend.
pub(crate) struct PlannerCosts {
    pub(crate) seq_page: f64,
    pub(crate) random_page: f64,
    pub(crate) cpu_tuple: f64,
    pub(crate) cpu_index_tuple: f64,
    pub(crate) cpu_operator: f64,
}

impl PlannerCosts {
    /// Reads the live GUC values — only callable inside a backend.
    pub(crate) fn current() -> Self {
        unsafe {
            Self {
                seq_page: pg_sys::seq_page_cost,
                random_page: pg_sys::random_page_cost,
                cpu_tuple: pg_sys::cpu_tuple_cost,
                cpu_index_tuple: pg_sys::cpu_index_tuple_cost,
                cpu_operator: pg_sys::cpu_operator_cost,
            }
        }
    }
}

/// Everything the pure local cost math needs, pre-resolved by the caller
/// (`am::amcostestimate`) from the metapage + planner state.
pub(crate) struct LocalCostInputs {
    /// Planner's row estimate for the heap (falls back to the metapage's
    /// `total_vectors` when the planner has none).
    pub(crate) rows: f64,
    /// Heap relation size in pages (planner's `RelOptInfo.pages`).
    pub(crate) heap_pages: f64,
    pub(crate) is_flat: bool,
    pub(crate) dim: u32,
    /// The width of a stored code in bits (`D`).
    pub(crate) code_bits: u32,
    /// The index has a filter column: its list records carry the value and a NULL flag.
    pub(crate) has_filter: bool,
    pub(crate) n_lists: u32,
    pub(crate) nprobe: u32,
    /// Rows the scan rescores (`scan::planned_pool_and_nprobe`).
    pub(crate) pool: u32,
    /// Mackert–Lohman page-fetch estimate for `pool` random heap fetches —
    /// computed by the caller via [`pg_sys::index_pages_fetched`] (needs the
    /// live `PlannerInfo`), passed in so this stays pure.
    pub(crate) rescore_pages_fetched: f64,
    /// Usable bytes per index page (`metapage::usable_page_bytes()`).
    pub(crate) usable_page_bytes: f64,
}

/// The exact-rescore pool the scan will build, capped by the row count: the pool can't exceed the data.
pub(crate) fn rescore_pool(pool: u32, rows: f64) -> f64 {
    f64::from(pool).min(rows.max(0.0))
}

/// Records on one list page: the layout's own capacity (blocks of codes, then the per-record
/// arrays); a bytes-per-record quotient overstated it by 12 to 17 %.
fn records_per_page(inputs: &LocalCostInputs) -> f64 {
    let code_bytes = inputs.code_bits as usize / 8;
    let usable = inputs.usable_page_bytes as usize;
    Geometry::within(code_bytes, inputs.has_filter, usable)
        .capacity
        .max(1) as f64
}

/// `(startup_cost, total_cost, index_pages)` for a LOCAL-mode scan — see the
/// module docstring for the model, term by term.
pub(crate) fn local_cost(inputs: &LocalCostInputs, costs: &PlannerCosts) -> (f64, f64, f64) {
    if inputs.is_flat {
        // Seqscan-equivalent work × overhead — always loses to the real
        // seqscan, by design (module docstring).
        let seq_equivalent = inputs.heap_pages * costs.seq_page
            + inputs.rows * (costs.cpu_operator + costs.cpu_tuple);
        let total = FLAT_OVERHEAD_MULT * seq_equivalent;
        // The whole candidate set is computed before the first tuple returns
        // (top-k over the full heap), so startup ≈ total.
        return (total, total, 0.0);
    }

    let n_lists = inputs.n_lists.max(1) as f64;
    let probe_frac = (inputs.nprobe as f64 / n_lists).clamp(0.0, 1.0);
    let probed_tuples = inputs.rows * probe_frac;

    // Directory: one entry per list = dim*4 centroid bytes + 4-byte head.
    let dir_entry_bytes = (inputs.dim as f64) * 4.0 + 4.0;
    let dir_pages = (n_lists * dir_entry_bytes / inputs.usable_page_bytes).ceil();

    let probed_list_pages = (probed_tuples / records_per_page(inputs)).ceil();

    let pool = rescore_pool(inputs.pool, inputs.rows);
    let toast_mult = if inputs.dim > TOAST_DIM_THRESHOLD {
        TOAST_FETCH_PAGES
    } else {
        1.0
    };

    let startup = dir_pages * costs.seq_page
        + probed_list_pages * costs.seq_page
        + probed_tuples * costs.cpu_operator * CODE_OP_FACTOR
        + inputs.rescore_pages_fetched * toast_mult * costs.random_page
        + pool * (costs.cpu_operator + costs.cpu_tuple);
    let total = startup + pool * costs.cpu_index_tuple;
    (startup, total, dir_pages + probed_list_pages)
}

/// `(startup_cost, total_cost)` for a REMOTE-mode scan (also `Demoting`/
/// `Detached`/v1 metapages) — the 0.3 shape, startup byte-identical to the
/// earlier constant.
pub(crate) fn remote_cost(min_topk: f64, costs: &PlannerCosts) -> (f64, f64) {
    let startup = REMOTE_STARTUP_OPS * costs.cpu_operator;
    let total = startup + min_topk * REMOTE_TUPLE_OPS * costs.cpu_operator;
    (startup, total)
}

/// Resolves [`LocalCostInputs`] from the metapage + planner numbers — thin,
/// kept beside the math so the two field lists can't drift. `pool` and `nprobe` are what the scan
/// will use ([`super::scan::planned_pool_and_nprobe`]): they depend on a pushed key.
pub(crate) fn local_inputs(
    meta: &MetaPage,
    planner_rows: f64,
    heap_pages: f64,
    rescore_pages_fetched: f64,
    pool: u32,
    nprobe: u32,
) -> LocalCostInputs {
    let rows = if planner_rows > 0.0 {
        planner_rows
    } else {
        meta.total_vectors as f64 + meta.inserted_since_build as f64
    };
    LocalCostInputs {
        rows,
        heap_pages,
        is_flat: meta.is_flat,
        dim: meta.dim,
        code_bits: meta.code_bits,
        has_filter: meta.has_filter(),
        n_lists: meta.n_lists,
        nprobe,
        pool,
        rescore_pages_fetched,
        usable_page_bytes: super::metapage::usable_page_bytes() as f64,
    }
}

// Pure-Rust (no pgrx/pg dependency) — runs under plain `cargo test`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::local::options::{FALLBACK_POOL, legacy_nprobe};
    use crate::local::pages::list_entries_per_page;
    use crate::local::rotation::padded_dim;

    fn default_costs() -> PlannerCosts {
        PlannerCosts {
            seq_page: 1.0,
            random_page: 4.0,
            cpu_tuple: 0.01,
            cpu_index_tuple: 0.005,
            cpu_operator: 0.0025,
        }
    }

    /// Seqscan + top-N sort estimate in the same units, for the relative
    /// assertions below — mirrors the planner's own structure closely enough
    /// for a coarse cross-check (pages + per-tuple CPU + distance evals +
    /// top-N comparison term).
    fn seqscan_equivalent(rows: f64, heap_pages: f64, costs: &PlannerCosts) -> f64 {
        heap_pages * costs.seq_page
            + rows * (costs.cpu_tuple + costs.cpu_operator)
            + 2.0 * costs.cpu_operator * rows * (2.0f64 * 10.0).log2() // top-10 sort comparisons
    }

    #[test]
    fn clustered_index_beats_seqscan_on_a_realistic_table() {
        // 50k rows x 16d, rm=2 — the planner test's own fixture shape.
        let costs = default_costs();
        let inputs = LocalCostInputs {
            rows: 50_000.0,
            heap_pages: 620.0,
            is_flat: false,
            dim: 16,
            code_bits: 512,
            has_filter: false,
            n_lists: 224,
            nprobe: 16,
            pool: 200,
            rescore_pages_fetched: 180.0, // Mackert–Lohman for 200 fetches over 620 pages
            usable_page_bytes: 8168.0,
        };
        let (_startup, total, _pages) = local_cost(&inputs, &costs);
        let seq = seqscan_equivalent(inputs.rows, inputs.heap_pages, &costs);
        assert!(
            total < seq * 0.8,
            "clustered index must clearly beat seqscan at 50k rows: index={total:.0} seq={seq:.0}"
        );
    }

    #[test]
    fn flat_index_loses_to_seqscan_by_design() {
        let costs = default_costs();
        let inputs = LocalCostInputs {
            rows: 500.0,
            heap_pages: 7.0,
            is_flat: true,
            dim: 16,
            code_bits: 512,
            has_filter: false,
            n_lists: 0,
            nprobe: 0,
            pool: 1000,
            rescore_pages_fetched: 0.0,
            usable_page_bytes: 8168.0,
        };
        let (_startup, total, _pages) = local_cost(&inputs, &costs);
        let seq = seqscan_equivalent(inputs.rows, inputs.heap_pages, &costs);
        assert!(
            total > seq,
            "flat scan must never be priced under the seqscan it duplicates: \
             index={total:.0} seq={seq:.0}"
        );
    }

    #[test]
    fn toast_multiplier_prices_wide_dims_higher() {
        let costs = default_costs();
        let mk = |dim: u32| LocalCostInputs {
            rows: 100_000.0,
            heap_pages: 5_000.0,
            is_flat: false,
            dim,
            code_bits: 1024,
            has_filter: false,
            n_lists: 317,
            nprobe: 25,
            pool: 300,
            rescore_pages_fetched: 900.0,
            usable_page_bytes: 8168.0,
        };
        let (_, narrow, _) = local_cost(&mk(128), &costs);
        let (_, wide, _) = local_cost(&mk(768), &costs);
        assert!(
            wide > narrow,
            "a TOASTed (768d) rescore pool must cost more than an inline (128d) one: \
             wide={wide:.0} narrow={narrow:.0}"
        );
    }

    #[test]
    fn remote_startup_matches_the_pre_q3_constant() {
        let costs = default_costs();
        let (startup, total) = remote_cost(50.0, &costs);
        assert_eq!(startup, 10_000.0 * 0.0025); // = 25.0, the 0.3 constant
        assert!(total > startup);
    }

    fn million_by_768(nprobe: u32, pool: u32) -> LocalCostInputs {
        let pooled = rescore_pool(pool, 1_000_000.0);
        LocalCostInputs {
            rows: 1_000_000.0,
            heap_pages: 7_600.0,
            is_flat: false,
            dim: 768,
            code_bits: 1024,
            has_filter: false,
            n_lists: 1000,
            nprobe,
            pool,
            rescore_pages_fetched: pooled * 0.9,
            usable_page_bytes: 8168.0,
        }
    }

    #[test]
    fn a_calibrated_pair_prices_below_the_090_constants_at_1m_x_768d() {
        let costs = default_costs();
        let new = million_by_768(16, 200);
        let old = million_by_768(legacy_nprobe(1000), FALLBACK_POOL);
        let (new_startup, new_total, new_pages) = local_cost(&new, &costs);
        let (old_startup, old_total, old_pages) = local_cost(&old, &costs);
        assert!(
            new_total < 0.5 * old_total,
            "new={new_total:.0} old={old_total:.0}"
        );
        assert!(new_startup < old_startup);
        assert!(new_pages < old_pages, "new={new_pages} old={old_pages}");
    }

    #[test]
    fn cost_grows_with_the_probe_count_and_with_the_pool() {
        let costs = default_costs();
        let at = |nprobe, pool| local_cost(&million_by_768(nprobe, pool), &costs).1;
        assert!(at(26, 300) < at(52, 300));
        assert!(at(26, 300) < at(26, 600));
    }

    #[test]
    fn rescore_pool_caps_at_row_count() {
        assert_eq!(rescore_pool(1000, 50.0), 50.0);
        assert_eq!(rescore_pool(1000, 1e9), 1000.0);
    }

    /// A list page holds what the layout says (blocks of codes, then the per-record arrays), not
    /// a bytes-per-record quotient, which counted 12 to 17 % too many records per page.
    #[test]
    fn a_page_holds_what_the_list_layout_holds_at_768_and_1024_dimensions() {
        for dim in [768u32, 1024] {
            let code_bits = padded_dim(dim as usize) as u32;
            for has_filter in [false, true] {
                let inputs = LocalCostInputs {
                    dim,
                    code_bits,
                    has_filter,
                    ..million_by_768(16, 200)
                };
                let layout = list_entries_per_page(code_bits as usize / 8, has_filter);
                assert_eq!(
                    records_per_page(&inputs),
                    layout as f64,
                    "{dim}d, filter {has_filter}"
                );

                // The probed pages the estimate reports are the layout's, too.
                let (_, _, pages) = local_cost(&inputs, &default_costs());
                let probed = inputs.rows * f64::from(inputs.nprobe) / f64::from(inputs.n_lists);
                let dir = (f64::from(inputs.n_lists) * (f64::from(dim) * 4.0 + 4.0)
                    / inputs.usable_page_bytes)
                    .ceil();
                assert_eq!(
                    pages,
                    dir + (probed / layout as f64).ceil(),
                    "{dim}d, filter {has_filter}"
                );
            }
        }
    }
}
