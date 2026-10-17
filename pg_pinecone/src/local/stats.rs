//! Per-index filter-value statistics — the cardinality-estimate half of
//! the two-mode filtered scan (mode select on
//! bitmap cardinality). Built once at `ambuild` (which sees every record and
//! can count exactly), stored on one dedicated page whose block number the
//! v3 metapage records (`MetaPage::stats_start`), and read at scan time to
//! estimate what fraction of the index a given equality filter value
//! qualifies — that estimate against the `pinecone.mode_l_threshold` GUC is the
//! whole mode-L/mode-H decision.
//!
//! ## Shape: top-N exact counts + a residual estimate
//! The simplest honest v1: the [`MAX_ENTRIES`] most
//! frequent values get their EXACT build-time counts stored; everything else
//! collapses into two numbers — `residual_count` (how many records hold a
//! value outside the top-N) and `residual_distinct` (how many distinct such
//! values there were), whose ratio is the "average residual value" estimate.
//!
//! ## Accuracy limits (documented, not hidden)
//! - Counts are frozen at build time; both churn directions can drift them.
//!   **Inserts**: every local `aminsert` bumps the metapage's
//!   `inserted_since_build` counter, and [`FilterStats::estimate_fraction`]
//!   divides by `total + uncounted_inserts` — the true fraction's LOWER
//!   bound (an inserted row can only ever RAISE a value's true count, never
//!   the frozen numerator) — so insert staleness biases every estimate
//!   toward mode L (exact), never away from it. The drift bound: an
//!   estimate can understate a value's true fraction by at most the factor
//!   `(total + inserted) / total`, and understating costs time only (an
//!   L-routed common value pays a bigger exact scan), never recall. The
//!   counter can also PERMANENTLY overcount by aborted transactions'
//!   inserts (GenericXLog page writes are non-transactional — a ROLLBACK'd
//!   INSERT's bump survives; `local::insert`'s module docstring) — same
//!   conservative direction, and a REINDEX reconciles it to zero.
//!   **Deletes**: VACUUM can shrink the data, so the stored fraction can
//!   drift from the table's CURRENT true fraction in either direction —
//!   which way depends on the delete pattern (e.g. disproportionately
//!   deleting one value's own rows drops its true fraction below the frozen
//!   estimate; deleting mostly OTHER values' rows raises it), and nothing
//!   here assumes deletes are uniform across filter values. That drift is
//!   harmless for recall regardless of direction: mode L is exact by
//!   construction whether or not its selectivity estimate was accurate, and
//!   the one direction that WOULD cost recall — a genuinely rare value
//!   misestimated as common enough to route to mode H — is bounded below by
//!   the build-time residual-skew argument, not by any claim about which way
//!   VACUUM drift goes.
//! - A value absent at build time estimates as the residual average (or 0.0
//!   if there was no residual) — for a genuinely absent value either mode
//!   returns the correct empty-ish result, so a wrong mode pick costs only
//!   time.
//! - Residual skew: one huge value hiding below the top-N cutoff drags the
//!   average UP, not down (a right-skewed outlier pulls the mean above the
//!   typical residual value — that's exactly why the bound below has to hold
//!   for every residual value, not just the "huge" one). With
//!   `MAX_ENTRIES = 256`, any single residual value's true fraction is
//!   bounded at `1/257` (~0.4%): the 256 top-N entries all outrank it by
//!   count, so their combined count alone is >= 256x its own, and the
//!   residual average — itself an average of individually-bounded values —
//!   inherits the same bound. This makes the misestimate unable to flip an
//!   L-worthy (genuinely rare) query to H **only while
//!   `pinecone.mode_l_threshold >= ~1/257`**; the documented default (0.03) and
//!   the plan's 2-5% sweep band both clear it, but this is a property of
//!   that GUC's value, not an unconditional guarantee — a threshold set
//!   below ~0.4% (it's a `Userset` GUC, range 0.0-1.0) moves outside it.
//! - Text values are counted by their 64-bit hash; a build-time collision
//!   merges two values' counts (overestimating each — again erring toward
//!   mode L / exactness).
//!
//! If benchmark sweeps show even this sketch misleads, `pinecone.force_scan_mode`
//! (`scan.rs`) overrides the estimator outright — the documented fallback.
//!
//! ## Page layout (block `stats_start`, one page, same raw-bytes style as
//! `pages.rs`)
//! `n_entries: u32 | null_count: u32 | residual_distinct: u32 | total: u32 |
//! residual_count: u64 | entries: [value: i64, count: u32] * n_entries`.
//! Capacity: 24-byte fixed head + 12 bytes/entry; 256 entries = 3096 bytes,
//! always a single page.

use std::collections::HashMap;

use pgrx::prelude::*;

use super::{metapage, pages};

/// Top-N cutoff. See the module docstring's residual-skew bound for why 256
/// is enough for mode selection (not a general-purpose histogram).
pub(crate) const MAX_ENTRIES: usize = 256;

const HEADER_SIZE: usize = 24;

const ENTRY_SIZE: usize = 12;

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct FilterStats {
    /// `(stored filter value, exact build-time count)` — the top-N by count.
    pub(crate) entries: Vec<(i64, u32)>,
    /// Records whose filter column was SQL NULL (never match an equality
    /// scankey; counted so `total` still covers every indexed record).
    pub(crate) null_count: u32,
    /// Records holding a non-null value outside `entries`.
    pub(crate) residual_count: u64,
    /// Distinct non-null values outside `entries`.
    pub(crate) residual_distinct: u32,
    /// Every indexed record (matches the metapage's `total_vectors`).
    pub(crate) total: u32,
}

impl FilterStats {
    /// Exact per-value counting over the build's collected `(value, is_null)`
    /// pairs, then top-N selection. O(rows) time, O(distinct) memory — fine
    /// at `ambuild`, which already holds every vector in memory.
    pub(crate) fn compute(values: impl Iterator<Item = (i64, bool)>) -> Self {
        let mut counts: HashMap<i64, u32> = HashMap::new();
        let mut null_count = 0u32;
        let mut total = 0u32;
        for (value, is_null) in values {
            total += 1;
            if is_null {
                null_count += 1;
            } else {
                *counts.entry(value).or_insert(0) += 1;
            }
        }

        let mut all: Vec<(i64, u32)> = counts.into_iter().collect();
        // Count-descending; value ascending as a deterministic tiebreak so
        // two builds of the same data store identical stats bytes.
        all.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let residual: Vec<(i64, u32)> = all.split_off(all.len().min(MAX_ENTRIES));

        FilterStats {
            entries: all,
            null_count,
            residual_count: residual.iter().map(|&(_, c)| c as u64).sum(),
            residual_distinct: residual.len() as u32,
            total,
        }
    }

    /// Estimated fraction of the index's records matching `value` — see the
    /// module docstring for the estimator and its documented limits.
    /// `uncounted_inserts` is the metapage's `inserted_since_build` counter:
    /// rows appended since the build, invisible to these frozen
    /// counts. Folding them into the DENOMINATOR yields the true fraction's
    /// lower bound, which biases mode selection toward L (exact) — the
    /// conservative direction (module docstring's insert-staleness bound).
    pub(crate) fn estimate_fraction(&self, value: i64, uncounted_inserts: u32) -> f64 {
        let denominator = self.total as f64 + uncounted_inserts as f64;
        if denominator == 0.0 {
            return 0.0;
        }
        if let Some(&(_, count)) = self.entries.iter().find(|&&(v, _)| v == value) {
            return count as f64 / denominator;
        }
        if self.residual_distinct == 0 {
            return 0.0;
        }
        (self.residual_count as f64 / self.residual_distinct as f64) / denominator
    }

    /// The stats page's body bytes (everything after `PageInit`'s header) —
    /// written by `build.rs` through its ordinary `write_page` helper.
    pub(crate) fn page_body(&self) -> Vec<u8> {
        assert!(
            self.entries.len() <= MAX_ENTRIES,
            "pg_pinecone: filter stats exceed one page — MAX_ENTRIES regression"
        );
        let mut body = Vec::with_capacity(HEADER_SIZE + self.entries.len() * ENTRY_SIZE);
        body.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        body.extend_from_slice(&self.null_count.to_le_bytes());
        body.extend_from_slice(&self.residual_distinct.to_le_bytes());
        body.extend_from_slice(&self.total.to_le_bytes());
        body.extend_from_slice(&self.residual_count.to_le_bytes());
        for &(value, count) in &self.entries {
            body.extend_from_slice(&value.to_le_bytes());
            body.extend_from_slice(&count.to_le_bytes());
        }
        body
    }

    /// The page contents back into stats, the inverse of [`page_body`](Self::page_body). The count
    /// is checked before anything is sized from it, so a corrupt page can neither make the scan
    /// allocate gigabytes nor read past its buffer.
    pub(crate) fn parse(buf: &[u8]) -> Result<FilterStats, String> {
        if buf.len() < HEADER_SIZE {
            return Err(format!("a {}-byte page has no stats header", buf.len()));
        }
        let n_entries = pages::read_u32(buf, 0) as usize;
        if n_entries > MAX_ENTRIES {
            return Err(format!("n_entries {n_entries} exceeds {MAX_ENTRIES}"));
        }
        if HEADER_SIZE + n_entries * ENTRY_SIZE > buf.len() {
            return Err(format!(
                "n_entries {n_entries} runs past the {}-byte page",
                buf.len()
            ));
        }
        let null_count = pages::read_u32(buf, 4);
        let residual_distinct = pages::read_u32(buf, 8);
        let total = pages::read_u32(buf, 12);
        let residual_count = pages::read_u64(buf, 16);
        let mut entries = Vec::with_capacity(n_entries);
        for i in 0..n_entries {
            let off = HEADER_SIZE + i * ENTRY_SIZE;
            entries.push((pages::read_i64(buf, off), pages::read_u32(buf, off + 8)));
        }
        Ok(FilterStats {
            entries,
            null_count,
            residual_count,
            residual_distinct,
            total,
        })
    }
}

/// Reads the stats page back — `scan.rs`'s mode selection, once per filtered
/// scan (a single buffered page read; not worth an `rd_amcache`-style cache
/// until profiling says otherwise).
pub(crate) unsafe fn read(index: pg_sys::Relation, block: pg_sys::BlockNumber) -> FilterStats {
    unsafe {
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

        let parsed = FilterStats::parse(buf);
        // Released before erroring (like the directory and list readers): an `error!()` raised
        // while holding the lock would leave the buffer for the resource owner to release at abort.
        pg_sys::UnlockReleaseBuffer(buffer);
        match parsed {
            Ok(stats) => stats,
            Err(problem) => error!("pg_pinecone: corrupt stats page at block {block} — {problem}"),
        }
    }
}

/// TEST-ONLY (`#[cfg(feature = "pg_test")]`): overwrites the stats page's `n_entries` word, the
/// corruption [`read`]'s bound exists to catch (same technique as
/// `scan::test_corrupt_directory_n_entries`).
#[cfg(feature = "pg_test")]
pub(crate) unsafe fn test_corrupt_n_entries(
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
        std::ptr::copy_nonoverlapping(n_entries.to_le_bytes().as_ptr(), contents, 4);
        pg_sys::GenericXLogFinish(state);
        pg_sys::UnlockReleaseBuffer(buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local::testkit::Rng;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[test]
    fn exact_counts_when_under_cap() {
        let values: Vec<(i64, bool)> = (0..100).map(|i| ((i % 4) as i64, false)).collect();
        let stats = FilterStats::compute(values.into_iter());
        assert_eq!(stats.total, 100);
        assert_eq!(stats.residual_distinct, 0);
        assert_eq!(stats.residual_count, 0);
        for v in 0..4 {
            let f = stats.estimate_fraction(v, 0);
            assert!((f - 0.25).abs() < 1e-9, "value {v}: {f}");
        }
        // Absent value with no residual: definitively 0.
        assert_eq!(stats.estimate_fraction(99, 0), 0.0);
    }

    #[test]
    fn nulls_counted_but_never_match() {
        let mut values: Vec<(i64, bool)> = (0..90).map(|_| (7, false)).collect();
        values.extend((0..10).map(|_| (0, true)));
        let stats = FilterStats::compute(values.into_iter());
        assert_eq!(stats.total, 100);
        assert_eq!(stats.null_count, 10);
        assert!((stats.estimate_fraction(7, 0) - 0.9).abs() < 1e-9);
    }

    #[test]
    fn uncounted_inserts_shrink_the_estimate_monotonically() {
        // The insert-staleness bound (module docstring): folding
        // uncounted inserts into the denominator can only SHRINK an
        // estimate — the conservative, toward-mode-L direction.
        let values: Vec<(i64, bool)> = (0..100).map(|i| ((i % 4) as i64, false)).collect();
        let stats = FilterStats::compute(values.into_iter());
        assert!((stats.estimate_fraction(1, 0) - 0.25).abs() < 1e-9);
        assert!((stats.estimate_fraction(1, 100) - 0.125).abs() < 1e-9);
        let mut prev = f64::MAX;
        for inserted in [0u32, 1, 10, 100, 10_000] {
            let est = stats.estimate_fraction(1, inserted);
            assert!(est <= prev, "estimate must never grow with inserts");
            prev = est;
        }
        // A totally-empty build that has since received inserts still
        // estimates 0 (routes to mode L — exact — safe).
        let empty = FilterStats::compute(std::iter::empty());
        assert_eq!(empty.estimate_fraction(1, 50), 0.0);
    }

    #[test]
    fn residual_average_for_values_past_the_cap() {
        // MAX_ENTRIES frequent values (10 records each) + 100 rare distinct
        // values (1 each): the rare ones fall in the residual and estimate
        // as its average.
        let mut values: Vec<(i64, bool)> = Vec::new();
        for v in 0..MAX_ENTRIES as i64 {
            values.extend(std::iter::repeat_n((v, false), 10));
        }
        for v in 0..100i64 {
            values.push((1_000_000 + v, false));
        }
        let stats = FilterStats::compute(values.into_iter());
        let total = (MAX_ENTRIES * 10 + 100) as f64;
        assert_eq!(stats.residual_distinct, 100);
        assert_eq!(stats.residual_count, 100);
        // A top-N value: exact.
        assert!((stats.estimate_fraction(3, 0) - 10.0 / total).abs() < 1e-9);
        // A residual value: the residual average (1/total each here).
        assert!((stats.estimate_fraction(1_000_005, 0) - 1.0 / total).abs() < 1e-9);
        // An absent value: also the residual average — documented limit.
        assert!((stats.estimate_fraction(-42, 0) - 1.0 / total).abs() < 1e-9);
    }

    #[test]
    fn page_body_round_trips_through_manual_decode() {
        let values: Vec<(i64, bool)> =
            vec![(5, false), (5, false), (9, false), (0, true), (5, false)];
        let stats = FilterStats::compute(values.into_iter());
        let body = stats.page_body();
        // Decode with the same offsets `read` uses (sans the page header).
        assert_eq!(pages::read_u32(&body, 0), stats.entries.len() as u32);
        assert_eq!(pages::read_u32(&body, 4), 1); // null_count
        assert_eq!(pages::read_u32(&body, 12), 5); // total
        assert_eq!(pages::read_i64(&body, HEADER_SIZE), 5); // top value
        assert_eq!(pages::read_u32(&body, HEADER_SIZE + 8), 3); // its count
    }

    #[test]
    fn empty_input_estimates_zero() {
        let stats = FilterStats::compute(std::iter::empty());
        assert_eq!(stats.estimate_fraction(1, 0), 0.0);
    }

    fn stats_of(entries: usize, rng: &mut Rng) -> FilterStats {
        FilterStats {
            entries: (0..entries)
                .map(|_| (rng.u64() as i64, rng.u64() as u32))
                .collect(),
            null_count: rng.u64() as u32,
            residual_count: rng.u64(),
            residual_distinct: rng.u64() as u32,
            total: rng.u64() as u32,
        }
    }

    fn page_of(body: &[u8], len: usize) -> Vec<u8> {
        let mut page = vec![0u8; len];
        let n = body.len().min(len);
        page[..n].copy_from_slice(&body[..n]);
        page
    }

    #[test]
    fn parse_gives_back_what_the_writer_stores() {
        let mut rng = Rng::new(0x57A7);
        let usable = metapage::usable_page_bytes();
        for entries in [0, 1, 2, 100, MAX_ENTRIES - 1, MAX_ENTRIES] {
            let stats = stats_of(entries, &mut rng);
            let page = page_of(&stats.page_body(), usable);
            assert_eq!(FilterStats::parse(&page).as_ref(), Ok(&stats), "{entries}");
        }
        let computed =
            FilterStats::compute((0..5_000).map(|i| ((i * i % 997) as i64, i % 41 == 0)));
        let page = page_of(&computed.page_body(), usable);
        assert_eq!(FilterStats::parse(&page).as_ref(), Ok(&computed));
    }

    fn refusal(buf: &[u8]) -> String {
        match FilterStats::parse(buf) {
            Err(problem) => problem,
            Ok(stats) => panic!("accepted {} entries", stats.entries.len()),
        }
    }

    /// The count is checked before anything is sized from it: one the writer never stores
    /// (`MAX_ENTRIES`), or more than the buffer holds, is refused.
    #[test]
    fn parse_refuses_a_count_the_writer_never_stores_or_the_page_cannot_hold() {
        let mut rng = Rng::new(0xB0_0D);
        let usable = metapage::usable_page_bytes();
        let mut with_count = |n: u32, len: usize| {
            let mut page = page_of(&stats_of(MAX_ENTRIES, &mut rng).page_body(), len);
            page[0..4].copy_from_slice(&n.to_le_bytes());
            page
        };
        for n in [MAX_ENTRIES as u32 + 1, 10_000, u32::MAX / 2, u32::MAX] {
            let problem = refusal(&with_count(n, usable));
            assert!(
                problem.contains(&n.to_string()) && problem.contains(&MAX_ENTRIES.to_string()),
                "{problem}"
            );
        }
        assert!(FilterStats::parse(&with_count(MAX_ENTRIES as u32, usable)).is_ok());

        // Within MAX_ENTRIES but past a short buffer: the last entry's last byte decides.
        let end = |n: usize| HEADER_SIZE + n * ENTRY_SIZE;
        assert!(FilterStats::parse(&with_count(10, end(10))).is_ok());
        let problem = refusal(&with_count(10, end(10) - 1));
        assert!(
            problem.contains("10") && problem.contains("page"),
            "{problem}"
        );
        refusal(&with_count(1, HEADER_SIZE));

        // No header at all.
        for len in [0, 1, 4, HEADER_SIZE - 1] {
            refusal(&vec![0xFF; len]);
        }
        assert_eq!(
            FilterStats::parse(&[0; HEADER_SIZE]).as_ref(),
            Ok(&FilterStats::compute(std::iter::empty()))
        );
    }

    /// Random pages, valid pages with flipped bytes and rewritten counts, and truncated buffers:
    /// the parser answers `Err` or stats whose entries are what the header counts and the page
    /// holds, never panics, and allocates no more than a page's worth.
    #[test]
    fn parse_survives_random_and_mutated_pages() {
        let mut rng = Rng::new(0x57A7_F022);
        let usable = metapage::usable_page_bytes();
        let (mut parsed, mut refused) = (0usize, 0usize);
        for round in 0..10_000 {
            let mut len = usable;
            let mut page = match round % 5 {
                0 => (0..usable).map(|_| rng.u64() as u8).collect(),
                1 | 2 => {
                    let stats = stats_of(rng.below(MAX_ENTRIES + 1), &mut rng);
                    let mut page = page_of(&stats.page_body(), usable);
                    for _ in 0..=rng.below(8) {
                        let at = if round % 5 == 1 {
                            rng.below(usable)
                        } else {
                            rng.below(HEADER_SIZE)
                        };
                        page[at] ^= 1 << rng.below(8);
                    }
                    page
                }
                3 => {
                    let n = [
                        0,
                        1,
                        MAX_ENTRIES as u32 - 1,
                        MAX_ENTRIES as u32,
                        MAX_ENTRIES as u32 + 1,
                        u32::MAX,
                    ][rng.below(6)];
                    let mut page = page_of(&stats_of(MAX_ENTRIES, &mut rng).page_body(), usable);
                    page[0..4].copy_from_slice(&n.to_le_bytes());
                    page
                }
                _ => {
                    len = rng.below(usable + 1);
                    page_of(
                        &stats_of(rng.below(MAX_ENTRIES + 1), &mut rng).page_body(),
                        usable,
                    )
                }
            };
            page.truncate(len);
            let outcome = catch_unwind(AssertUnwindSafe(|| FilterStats::parse(&page)));
            let Ok(result) = outcome else {
                panic!(
                    "FilterStats::parse panicked on round {round} (kind {})",
                    round % 5
                );
            };
            let claimed = (page.len() >= 4).then(|| pages::read_u32(&page, 0) as usize);
            let consistent = page.len() >= HEADER_SIZE
                && claimed.is_some_and(|n| {
                    n <= MAX_ENTRIES && HEADER_SIZE + n * ENTRY_SIZE <= page.len()
                });
            match result {
                Ok(stats) => {
                    assert!(
                        consistent,
                        "round {round}: accepted a count the page cannot hold"
                    );
                    assert_eq!(Some(stats.entries.len()), claimed, "round {round}");
                    assert!(stats.entries.capacity() <= MAX_ENTRIES, "round {round}");
                    parsed += 1;
                }
                Err(_) => {
                    assert!(!consistent, "round {round}: refused a consistent page");
                    refused += 1;
                }
            }
        }
        assert!(
            parsed > 1_000 && refused > 1_000,
            "{parsed} parsed, {refused} refused"
        );
    }
}
