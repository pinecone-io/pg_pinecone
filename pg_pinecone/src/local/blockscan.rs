//! The block scan of a list page and the error-bound early stop of the rescore, as plain functions
//! over page bytes and numbers (no Postgres: `scan.rs` supplies the buffers and the heap).
//!
//! Papers: André et al., Cache locality is not enough: high-performance nearest neighbor search with
//! product quantization fast scan, VLDB 2015 (blocks of codes scanned with table lookups, no per-record
//! call); Gao & Long, RaBitQ, SIGMOD 2024 (rerank in estimate order and skip a candidate whose error
//! bound cannot beat the current k-th exact distance). Clean-room from the papers.
//!
//! **Scan**: [`scan_page`] runs the kernel over each block of a page, scores the live lanes
//! (`rabitq::score_block`) and admits those whose key beats the admission bound `theta` of the
//! candidate buffer; a block with no admitted lane costs no candidate work, and a block no record of
//! which passes the pushed filter is not scored at all.
//!
//! **Rescore**: [`rescore_progressive`] walks the candidates best estimate first, in batches, keeps the
//! k-th best exact distance `tau` found so far and skips a candidate whose bound `lo` is not below it.
//! The bound is the estimator's `err` times [`rabitq::PRUNE_MARGIN`]: a skipped candidate would have
//! had to beat `tau` with an estimate off by more than 3.8 sigma. That certifies the pool's exact top
//! `k` (`max(10, pool / 10)`), not the `LIMIT`: the walk goes on until `floor` rows are found, so a
//! `LIMIT` up to that many rows is filled, but the ranks after `k` are the rows it fetched, in exact
//! order. It ends at the latest when the buffer is spent (the cap).

use super::{
    kernel::Kernel,
    metric::Metric,
    pages::{BlockScalars, ListPage},
    rabitq::{self, BlockInput, BlockScores, QueryPrep},
};

/// Candidates fetched between two updates of `tau`.
pub(crate) const RESCORE_BATCH: usize = 32;

/// Ascending order with a NaN last whatever its sign: `total_cmp` puts a NaN with the sign bit set
/// (what x86 arithmetic makes of `inf - inf`) before every number.
pub(crate) fn nan_last(a: f32, b: f32) -> std::cmp::Ordering {
    a.is_nan().cmp(&b.is_nan()).then_with(|| a.total_cmp(&b))
}

/// One scored record.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Candidate {
    pub(crate) tid: (u32, u16),
    /// The key from the estimate (ascending is better: squared L2, negated inner product, cosine
    /// distance).
    pub(crate) est: f32,
    /// The key at the pruning bound, no more than the exact key if the bound holds (an L2 bound is
    /// clamped at 0, so it can exceed a negative `est`); `-inf` when the scan does not prune.
    pub(crate) lo: f32,
}

/// What a scan sums up for `pinecone.last_scan_stats()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ScanCounters {
    /// Records scored: live, and passing the pushed filter.
    pub(crate) scanned: u64,
    /// Records that entered the candidate buffer.
    pub(crate) admitted: u64,
    /// Blocks no record of which was admitted.
    pub(crate) blocks_skipped: u64,
}

/// The candidates of the probed lists: all of them, or the best `cap` by estimate.
pub(crate) struct CandBuf {
    cands: Vec<Candidate>,
    cap: usize,
    theta: f32,
}

impl CandBuf {
    /// Keeps every candidate: an iterative scan deepens past any pool.
    pub(crate) fn unbounded() -> CandBuf {
        CandBuf {
            cands: Vec::new(),
            cap: 0,
            theta: f32::INFINITY,
        }
    }

    /// Keeps the best `cap` by estimate (`cap` at least 1).
    pub(crate) fn best(cap: usize) -> CandBuf {
        CandBuf {
            cands: Vec::new(),
            cap: cap.max(1),
            theta: f32::INFINITY,
        }
    }

    /// A record is admitted when its estimate is below this (or a NaN): infinite until the buffer
    /// first holds twice its cap, then the `cap`-th best estimate seen at the last trim.
    pub(crate) fn theta(&self) -> f32 {
        self.theta
    }

    pub(crate) fn push(&mut self, c: Candidate) {
        self.cands.push(c);
    }

    /// Cuts a buffer of twice its cap back to the cap and tightens `theta`.
    pub(crate) fn trim_if_full(&mut self) {
        if self.cap > 0 && self.cands.len() >= 2 * self.cap {
            self.cands
                .select_nth_unstable_by(self.cap - 1, |a, b| nan_last(a.est, b.est));
            self.cands.truncate(self.cap);
            // A NaN at the cap-th place (fewer real keys than the cap) bounds nothing.
            let worst = self.cands[self.cap - 1].est;
            self.theta = if worst.is_nan() { f32::INFINITY } else { worst };
        }
    }

    #[cfg_attr(not(test), allow(dead_code))] // tests
    pub(crate) fn len(&self) -> usize {
        self.cands.len()
    }

    /// The best `cap` candidates, unsorted, for a bounded buffer; everything for an unbounded one.
    pub(crate) fn into_candidates(mut self) -> Vec<Candidate> {
        if self.cap > 0 && self.cands.len() > self.cap {
            self.cands
                .select_nth_unstable_by(self.cap - 1, |a, b| nan_last(a.est, b.est));
            self.cands.truncate(self.cap);
        }
        self.cands
    }
}

/// How a page's blocks are scored.
pub(crate) struct Scoring<'a> {
    pub(crate) kernel: Kernel,
    pub(crate) prep: &'a QueryPrep,
    pub(crate) metric: Metric,
    /// `D`, the code width in bits.
    pub(crate) dp: usize,
    /// Compute the pruning bound (else candidates carry `lo = -inf`).
    pub(crate) with_lo: bool,
}

/// Whether a key passes the admission bound: below `theta`, or a NaN (an overflowing vector: the exact
/// rescore decides it). `theta` itself is never NaN (see [`CandBuf::trim_if_full`]).
#[inline(always)]
fn admitted(est: f32, theta: f32) -> bool {
    est < theta || est.is_nan()
}

/// Whether any key of `est` is admitted: an OR over the lanes, which the compiler turns into vector
/// compares.
fn any_admitted(est: &[f32], theta: f32) -> bool {
    est.iter()
        .fold(0u32, |any, e| any | u32::from(admitted(*e, theta)))
        != 0
}

/// Scores every block of `page` and admits what `buf` takes. `filter` is the pushed key as an
/// inclusive range of stored filter values (a NULL never matches); only a filtered index's pages
/// have values to test.
pub(crate) fn scan_page(
    page: &ListPage<'_>,
    scoring: &Scoring<'_>,
    filter: Option<(i64, i64)>,
    buf: &mut CandBuf,
    counters: &mut ScanCounters,
) {
    let lanes = page.lanes();
    let mut acc = [0u32; 64];
    let mut scalars = BlockScalars::new();
    let mut scores = BlockScores::new();
    let mut lo = [0f32; 64];
    for b in 0..page.n_blocks() {
        let live = page.block_live(b);
        let live_mask = if live == 64 {
            u64::MAX
        } else {
            (1u64 << live) - 1
        };
        let mask = match filter {
            Some((low, high)) => live_mask & page.block_filter_mask(b, low, high),
            None => live_mask,
        };
        if mask == 0 {
            counters.blocks_skipped += 1;
            continue;
        }
        counters.scanned += u64::from(mask.count_ones());
        scoring
            .kernel
            .acc_block(&scoring.prep.lut, page.block_rows(b), lanes, &mut acc);
        page.block_scalars(b, &mut scalars);
        let input = BlockInput {
            acc: &acc[..live],
            pb: &scalars.pb[..live],
            dot: &scalars.dot[..live],
            rnorm: &scalars.rnorm[..live],
            rc: &scalars.rc[..live],
        };
        rabitq::score_block_keys(
            scoring.metric,
            scoring.prep,
            scoring.dp,
            &input,
            &mut scores,
        );
        let est = &scores.est[..live];
        let theta = buf.theta();
        // After the first `cap` candidates nearly every block ends here, on one vector compare.
        if !any_admitted(est, theta) {
            counters.blocks_skipped += 1;
            continue;
        }
        let mut admit = 0u64;
        for (r, e) in est.iter().enumerate() {
            admit |= u64::from(mask >> r & 1 == 1 && admitted(*e, theta)) << r;
        }
        if admit == 0 {
            counters.blocks_skipped += 1;
            continue;
        }
        counters.admitted += u64::from(admit.count_ones());
        if scoring.with_lo {
            rabitq::score_block_bounds(
                scoring.metric,
                scoring.prep,
                rabitq::PRUNE_MARGIN,
                &input,
                &scores,
                &mut lo[..live],
            );
        }
        let first = b * lanes;
        let mut rest = admit;
        while rest != 0 {
            let r = rest.trailing_zeros() as usize;
            rest &= rest - 1;
            buf.push(Candidate {
                tid: page.tid(first + r),
                est: est[r],
                lo: if scoring.with_lo {
                    lo[r]
                } else {
                    f32::NEG_INFINITY
                },
            });
        }
        buf.trim_if_full();
    }
}

/// When the progressive rescore may stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    /// The rank whose exact distance bounds the rest: a candidate is fetched only while its `lo`
    /// is below the `k`-th best exact distance found.
    pub(crate) k: usize,
    /// Rows found whatever the bound says (best estimates first; a candidate whose row is gone does
    /// not count).
    pub(crate) floor: usize,
    pub(crate) batch: usize,
}

impl Plan {
    /// For a rescore pool of `cap` rows and a floor of `floor_rows` (the `LIMIT` a scan always
    /// fills): `k`, the rank the walk certifies, is a tenth of the pool, at least 10
    /// (`rescore_multiplier = 10` is how a `LIMIT 100` is certified to its last rank).
    pub(crate) fn for_pool(cap: usize, floor_rows: usize) -> Plan {
        Plan {
            k: (cap / 10).max(10),
            floor: floor_rows.min(cap),
            batch: RESCORE_BATCH,
        }
    }
}

/// What a progressive rescore did, in terms of indices into the candidate slice.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Outcome {
    /// Candidates a fetch was issued for, in fetch order (best estimate first).
    pub(crate) fetched: Vec<usize>,
    /// `(index, exact key)` of those whose row was found, in fetch order.
    pub(crate) found: Vec<(usize, f32)>,
}

/// Fetches the candidates of `cands` (sorted ascending by `est`) best estimate first, in batches of
/// `plan.batch`, `exact` giving a candidate's exact key (`None`: the row is gone). With `prune` off
/// every candidate is fetched. With it on, once `plan.floor` rows are found, a candidate is fetched
/// only if its `lo` is below `tau`, the `plan.k`-th best exact key so far (infinite while fewer were
/// found), and the walk ends when no remaining candidate has a `lo` below it. A NaN `lo` is never
/// pruned and a NaN exact key bounds nothing, whatever its sign.
pub(crate) fn rescore_progressive(
    cands: &[Candidate],
    plan: &Plan,
    prune: bool,
    exact: &mut impl FnMut(&Candidate) -> Option<f32>,
) -> Outcome {
    let n = cands.len();
    let lo_of = |c: &Candidate| {
        if c.lo.is_nan() {
            f32::NEG_INFINITY
        } else {
            c.lo
        }
    };
    let mut tail_lo = vec![f32::INFINITY; n + 1];
    for i in (0..n).rev() {
        tail_lo[i] = tail_lo[i + 1].min(lo_of(&cands[i]));
    }
    let k = plan.k.max(1);
    let mut best: Vec<f32> = Vec::with_capacity(k + plan.batch);
    let mut tau = f32::INFINITY;
    let mut out = Outcome::default();
    let mut i = 0;
    while i < n {
        if prune && out.found.len() >= plan.floor && tail_lo[i] >= tau {
            break;
        }
        let mut taken = 0;
        while i < n && taken < plan.batch.max(1) {
            let c = &cands[i];
            if !prune || out.found.len() < plan.floor || lo_of(c) < tau {
                out.fetched.push(i);
                taken += 1;
                if let Some(key) = exact(c) {
                    out.found.push((i, key));
                    // A NaN key (an overflowing vector) is no rank: it must not take one.
                    if !key.is_nan() {
                        best.push(key);
                    }
                }
            }
            i += 1;
        }
        if best.len() >= k {
            best.sort_by(f32::total_cmp);
            best.truncate(k);
            tau = best[k - 1];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local::{
        kernel::Kernel,
        pages::{Geometry, ListBody, ListRecord, SCALAR_SIZE},
        rabitq::{Code, encode, estimate_bits, key_from_parts, lo_from_parts, prepare_query},
        rotation::{Rotation, padded_dim},
        testkit::Rng,
    };
    use std::borrow::Cow;
    use std::collections::HashSet;

    const SEED: u64 = 0x5EED_0004;

    fn cand(i: usize, est: f32, lo: f32) -> Candidate {
        Candidate {
            tid: (i as u32, 1),
            est,
            lo,
        }
    }

    /// `n` candidates with exact distances `d`, estimates off by up to `noise` and a bound that is valid
    /// by construction (`lo <= exact`), sorted by estimate.
    fn synthetic(rng: &mut Rng, n: usize, noise: f64, slack: f64) -> (Vec<Candidate>, Vec<f32>) {
        let mut rows: Vec<(f32, f32, f32)> = (0..n)
            .map(|_| {
                let d = 1.0 + rng.normal() * 0.2;
                let est = d + noise * rng.normal();
                let lo = d - slack * rng.uniform();
                (est as f32, lo as f32, d as f32)
            })
            .collect();
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        let cands = rows
            .iter()
            .enumerate()
            .map(|(i, r)| cand(i, r.0, r.1))
            .collect();
        (cands, rows.iter().map(|r| r.2).collect())
    }

    fn top_ids(found: &[(usize, f32)], k: usize) -> Vec<usize> {
        let mut v = found.to_vec();
        v.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut ids: Vec<usize> = v.iter().take(k).map(|x| x.0).collect();
        ids.sort_unstable();
        ids
    }

    #[test]
    fn the_plan_follows_the_pool_and_never_fetches_more_than_it_holds() {
        for (cap, floor_rows, k, floor) in [
            (100, 100, 10, 100),
            (200, 100, 20, 100),
            (300, 100, 30, 100),
            (1000, 100, 100, 100),
            (50, 100, 10, 50),
            (5000, 100, 500, 100),
        ] {
            let plan = Plan::for_pool(cap, floor_rows);
            assert_eq!(
                (plan.k, plan.floor, plan.batch),
                (k, floor, RESCORE_BATCH),
                "pool {cap}"
            );
        }
    }

    #[test]
    fn with_pruning_off_every_candidate_is_fetched_in_estimate_order() {
        let mut rng = Rng::new(1);
        let (cands, exact) = synthetic(&mut rng, 150, 0.1, 0.05);
        let out = rescore_progressive(&cands, &Plan::for_pool(150, 100), false, &mut |c| {
            Some(exact[c.tid.0 as usize])
        });
        assert_eq!(out.fetched, (0..150).collect::<Vec<_>>());
        assert_eq!(out.found.len(), 150);
    }

    /// The soundness argument, executed: when `lo <= exact` for every candidate, a candidate left
    /// unfetched has `lo >= tau` and so cannot be among the best k, whatever the estimates did.
    #[test]
    fn a_valid_bound_never_loses_a_top_k_candidate() {
        let mut rng = Rng::new(2);
        let mut pruned_somewhere = false;
        for round in 0..400 {
            let n = [1, 5, 50, 120, 300][round % 5];
            let (noise, slack) = [(0.02, 0.01), (0.1, 0.05), (0.3, 0.2)][round % 3];
            let (cands, exact) = synthetic(&mut rng, n, noise, slack);
            let plan = Plan {
                k: [1, 3, 10, 30][round % 4],
                floor: [0, 5, 100][round % 3],
                batch: [1, 7, 32][round % 3],
            };
            let out =
                rescore_progressive(&cands, &plan, true, &mut |c| Some(exact[c.tid.0 as usize]));
            let all: Vec<(usize, f32)> = exact.iter().copied().enumerate().collect();
            assert_eq!(
                top_ids(&out.found, plan.k),
                top_ids(&all, plan.k),
                "round {round}, n {n}, {plan:?}"
            );
            assert!(
                out.fetched.len() >= plan.floor.min(n),
                "round {round}: the floor is fetched"
            );
            assert!(
                out.fetched.windows(2).all(|w| w[0] < w[1]),
                "estimate order"
            );
            pruned_somewhere |= out.fetched.len() < n;
        }
        assert!(
            pruned_somewhere,
            "the instances must exercise the early stop"
        );
    }

    #[test]
    fn a_tight_bound_stops_right_after_the_floor() {
        let cands: Vec<Candidate> = (0..300)
            .map(|i| cand(i, i as f32, i as f32 - 0.5))
            .collect();
        let plan = Plan {
            k: 10,
            floor: 20,
            batch: 8,
        };
        let out = rescore_progressive(&cands, &plan, true, &mut |c| Some(c.est));
        assert_eq!(out.fetched, (0..20).collect::<Vec<_>>());
        let plan = Plan {
            k: 10,
            floor: 0,
            batch: 8,
        };
        let out = rescore_progressive(&cands, &plan, true, &mut |c| Some(c.est));
        assert_eq!(
            out.fetched.len(),
            16,
            "tau is known after two batches, 16 >= k"
        );
        assert_eq!(top_ids(&out.found, 10), (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn fewer_than_k_exact_keys_leave_tau_infinite_and_everything_is_fetched() {
        let mut rng = Rng::new(3);
        let (cands, exact) = synthetic(&mut rng, 40, 0.05, 0.02);
        let plan = Plan {
            k: 50,
            floor: 0,
            batch: 8,
        };
        let out = rescore_progressive(&cands, &plan, true, &mut |c| Some(exact[c.tid.0 as usize]));
        assert_eq!(out.fetched.len(), 40);
        let plan = Plan {
            k: 10,
            floor: 0,
            batch: 8,
        };
        let gone: HashSet<u32> = (0..40).step_by(2).collect();
        let out = rescore_progressive(&cands, &plan, true, &mut |c| {
            (!gone.contains(&c.tid.0)).then(|| exact[c.tid.0 as usize])
        });
        assert!(
            out.found
                .iter()
                .all(|(i, _)| !gone.contains(&cands[*i].tid.0))
        );
        assert!(out.fetched.len() >= out.found.len() && out.found.len() >= 10);
    }

    #[test]
    fn a_nan_bound_is_never_pruned_and_a_nan_exact_key_never_tightens_tau() {
        let mut cands: Vec<Candidate> = (0..200)
            .map(|i| cand(i, i as f32, i as f32 - 0.5))
            .collect();
        cands[150].lo = f32::NAN;
        let plan = Plan {
            k: 10,
            floor: 0,
            batch: 8,
        };
        let out = rescore_progressive(&cands, &plan, true, &mut |c| Some(c.est));
        assert!(out.fetched.contains(&150), "the NaN bound is fetched");
        let out = rescore_progressive(&cands, &plan, true, &mut |c| {
            Some(if c.tid.0 < 5 { f32::NAN } else { c.est })
        });
        assert_eq!(
            top_ids(&out.found, 10),
            (5..15).collect::<Vec<_>>(),
            "NaN keys sort last"
        );
    }

    #[test]
    fn a_nan_among_the_k_best_leaves_tau_infinite_so_the_walk_goes_on() {
        let cands: Vec<Candidate> = (0..300)
            .map(|i| cand(i, i as f32, i as f32 - 0.5))
            .collect();
        let plan = Plan {
            k: 10,
            floor: 0,
            batch: 32,
        };
        let out = rescore_progressive(&cands, &plan, true, &mut |c| {
            Some(if c.tid.0 < 295 { f32::NAN } else { c.est })
        });
        assert_eq!(
            out.fetched.len(),
            300,
            "fewer than k real keys bound nothing"
        );
        assert_eq!(top_ids(&out.found, 5), vec![295, 296, 297, 298, 299]);
    }

    /// x86 arithmetic makes a NaN with the sign bit set (`inf - inf`, `0 * inf`), which `total_cmp`
    /// orders before every number; Rust leaves the sign of a NaN unspecified.
    const NEGATIVE_NAN: u32 = 0xFFC0_0000;

    #[test]
    fn a_nan_exact_key_of_either_sign_bounds_nothing() {
        for nan in [f32::NAN, f32::from_bits(NEGATIVE_NAN)] {
            assert!(nan.is_nan());
            let cands: Vec<Candidate> = (0..200)
                .map(|i| cand(i, i as f32, i as f32 - 0.5))
                .collect();
            let plan = Plan {
                k: 10,
                floor: 0,
                batch: 1,
            };
            let out = rescore_progressive(&cands, &plan, true, &mut |c| {
                Some(if c.tid.0 < 5 { nan } else { c.est })
            });
            let real: Vec<usize> = out
                .found
                .iter()
                .filter(|(_, key)| !key.is_nan())
                .map(|(i, _)| *i)
                .collect();
            assert_eq!(
                real,
                (5..15).collect::<Vec<_>>(),
                "the ten best real keys, NaN sign bit {}",
                nan.is_sign_negative()
            );
        }
    }

    #[test]
    fn a_nan_estimate_of_either_sign_is_the_first_candidate_to_go() {
        for nan in [f32::NAN, f32::from_bits(NEGATIVE_NAN)] {
            let mut buf = CandBuf::best(2);
            for (i, est) in [nan, nan, 1.0, 2.0].into_iter().enumerate() {
                buf.push(cand(i, est, 0.0));
            }
            buf.trim_if_full();
            let mut kept: Vec<u32> = buf.into_candidates().iter().map(|c| c.tid.0).collect();
            kept.sort_unstable();
            assert_eq!(kept, vec![2, 3], "NaN sign bit {}", nan.is_sign_negative());
        }
    }

    #[test]
    fn nan_last_puts_a_nan_of_either_sign_after_infinity() {
        let mut v = [
            f32::NAN,
            2.0,
            f32::from_bits(NEGATIVE_NAN),
            f32::INFINITY,
            -1.0,
            f32::NEG_INFINITY,
        ];
        v.sort_by(|a, b| nan_last(*a, *b));
        assert_eq!(&v[..4], &[f32::NEG_INFINITY, -1.0, 2.0, f32::INFINITY]);
        assert!(v[4].is_nan() && v[5].is_nan());
    }

    /// The floor is rows found, not candidates fetched: dead heap rows among the first candidates
    /// must not end the walk short of the rows a `LIMIT` of up to 100 was promised.
    #[test]
    fn the_floor_counts_rows_found_not_candidates_fetched() {
        let cands: Vec<Candidate> = (0..300)
            .map(|i| cand(i, i as f32, i as f32 - 0.5))
            .collect();
        let plan = Plan::for_pool(300, 100);
        assert_eq!((plan.k, plan.floor), (30, 100));
        let out = rescore_progressive(&cands, &plan, true, &mut |c| {
            let i = c.tid.0 as usize;
            (i >= 100 || i.is_multiple_of(2)).then_some(c.est)
        });
        assert!(
            out.found.len() >= 100,
            "{} rows found, {} fetched",
            out.found.len(),
            out.fetched.len()
        );
    }

    #[test]
    fn an_empty_slice_and_a_zero_batch_are_handled() {
        let plan = Plan {
            k: 10,
            floor: 5,
            batch: 0,
        };
        assert_eq!(
            rescore_progressive(&[], &plan, true, &mut |_| Some(0.0)),
            Outcome::default()
        );
        let cands: Vec<Candidate> = (0..5).map(|i| cand(i, i as f32, 0.0)).collect();
        let out = rescore_progressive(&cands, &plan, true, &mut |c| Some(c.est));
        assert_eq!(out.fetched.len(), 5);
    }

    #[test]
    fn a_bounded_buffer_keeps_the_best_cap_by_estimate_whatever_the_arrival_order() {
        let mut rng = Rng::new(4);
        for cap in [1usize, 7, 100, 300] {
            let ests: Vec<f32> = (0..3_000).map(|_| rng.uniform() as f32).collect();
            let mut buf = CandBuf::best(cap);
            let mut theta_before = f32::INFINITY;
            for (chunk_no, chunk) in ests.chunks(64).enumerate() {
                for (j, e) in chunk.iter().enumerate() {
                    if admitted(*e, buf.theta()) {
                        buf.push(cand(chunk_no * 64 + j, *e, f32::NEG_INFINITY));
                    }
                }
                buf.trim_if_full();
                assert!(buf.theta() <= theta_before, "theta only tightens");
                theta_before = buf.theta();
                assert!(buf.len() < 2 * cap + 64, "the buffer stays near its cap");
            }
            let mut kept = buf.into_candidates();
            kept.sort_by(|a, b| a.est.total_cmp(&b.est));
            let mut all = ests.clone();
            all.sort_by(f32::total_cmp);
            assert_eq!(
                kept.iter().map(|c| c.est).collect::<Vec<_>>(),
                all[..cap.min(all.len())].to_vec(),
                "cap {cap}"
            );
        }
    }

    #[test]
    fn an_unbounded_buffer_keeps_everything_and_admits_every_estimate_including_nan() {
        let mut buf = CandBuf::unbounded();
        for i in 0..1_000 {
            let e = if i == 500 { f32::NAN } else { i as f32 };
            assert!(admitted(e, buf.theta()), "unbounded admits {e}");
            buf.push(cand(i, e, 0.0));
            buf.trim_if_full();
        }
        assert_eq!(buf.into_candidates().len(), 1_000);
        let mut bounded = CandBuf::best(2);
        for i in 0..4 {
            bounded.push(cand(i, i as f32, 0.0));
        }
        bounded.trim_if_full();
        assert_eq!(bounded.theta(), 1.0);
        let nan = f32::from_bits(0x7fc0_0000);
        assert!(
            nan.is_nan() && admitted(nan, bounded.theta()),
            "a NaN key is still admitted"
        );
    }

    struct Records {
        codes: Vec<Code>,
        tids: Vec<(u32, u16)>,
        fvals: Vec<i64>,
        fnulls: Vec<bool>,
    }

    fn records(rng: &mut Rng, rot: &Rotation, c: &[f32], n: usize) -> Records {
        let d = rot.dim();
        let codes = (0..n)
            .map(|_| {
                let x: Vec<f32> = c.iter().map(|v| v + 0.4 * rng.normal() as f32).collect();
                debug_assert_eq!(x.len(), d);
                encode(rot, &x, c)
            })
            .collect();
        Records {
            codes,
            tids: (0..n)
                .map(|i| (10_000 + i as u32 / 7, (i % 7) as u16 + 1))
                .collect(),
            fvals: (0..n).map(|i| (i % 10) as i64 - 3).collect(),
            fnulls: (0..n).map(|i| i % 13 == 5).collect(),
        }
    }

    #[repr(align(64))]
    struct Page([u8; 8192]);

    /// The pages of one list: `capacity` records each.
    fn pages_of(r: &Records, has_filter: bool) -> Vec<Box<Page>> {
        let code_bytes = r.codes[0].bits.len() * 8;
        let cap = Geometry::new(code_bytes, has_filter).capacity;
        (0..r.codes.len())
            .step_by(cap)
            .map(|start| {
                let end = (start + cap).min(r.codes.len());
                let mut body = ListBody::with_capacity(has_filter, code_bytes, end - start);
                for i in start..end {
                    let code = &r.codes[i];
                    body.push(&ListRecord {
                        code: Cow::Owned(code.bits.iter().flat_map(|w| w.to_le_bytes()).collect()),
                        rnorm: code.rnorm,
                        dot: code.dot,
                        rc: code.rc,
                        tid: r.tids[i],
                        fval: if has_filter { r.fvals[i] } else { 0 },
                        fnull: has_filter && r.fnulls[i],
                    });
                }
                let bytes = body.finish(0);
                let mut page = Box::new(Page([0; 8192]));
                page.0[..bytes.len()].copy_from_slice(&bytes);
                page
            })
            .collect()
    }

    struct Setup {
        r: Records,
        prep: QueryPrep,
        metric: Metric,
        dp: usize,
        rows: Vec<(f32, f32)>,
    }

    fn setup(rng: &mut Rng, d: usize, n: usize, metric: Metric) -> Setup {
        let rot = Rotation::new(SEED, d);
        let c: Vec<f32> = (0..d).map(|_| 3.0 * rng.normal() as f32).collect();
        let r = records(rng, &rot, &c, n);
        let q: Vec<f32> = c
            .iter()
            .map(|v| 0.9 * v + 0.2 * rng.normal() as f32)
            .collect();
        let prep = prepare_query(metric, &q, &rot.apply(&q), &c, &rot.apply(&c), SEED, 3);
        let kernel = Kernel::detect();
        let rows = r
            .codes
            .iter()
            .map(|code| {
                let (e, err) = estimate_bits(kernel, &prep, &code.bits, code.dot);
                (
                    key_from_parts(metric, &prep, code.rnorm, code.rc, e),
                    lo_from_parts(
                        metric,
                        &prep,
                        code.rnorm,
                        code.rc,
                        e,
                        rabitq::PRUNE_MARGIN * err,
                    ),
                )
            })
            .collect();
        Setup {
            r,
            prep,
            metric,
            dp: padded_dim(d),
            rows,
        }
    }

    fn scan_all(
        s: &Setup,
        has_filter: bool,
        filter: Option<(i64, i64)>,
        mut buf: CandBuf,
        with_lo: bool,
    ) -> (Vec<Candidate>, ScanCounters) {
        let code_bytes = s.dp / 8;
        let scoring = Scoring {
            kernel: Kernel::detect(),
            prep: &s.prep,
            metric: s.metric,
            dp: s.dp,
            with_lo,
        };
        let mut counters = ScanCounters::default();
        for page in pages_of(&s.r, has_filter) {
            let parsed = ListPage::parse(&page.0[..8168], code_bytes, has_filter).expect("page");
            scan_page(&parsed, &scoring, filter, &mut buf, &mut counters);
        }
        (buf.into_candidates(), counters)
    }

    use crate::local::pages::ListPage;
    use crate::local::rabitq;

    #[test]
    fn scanning_pages_scores_every_record_like_the_record_major_path() {
        let mut rng = Rng::new(5);
        for (d, metric) in [
            (100, Metric::L2),
            (768, Metric::Cosine),
            (200, Metric::Ip),
            (1536, Metric::L2),
        ] {
            let cap = Geometry::new(padded_dim(d) / 8, false).capacity;
            for n in [1, 31, 63, 64, 65, cap, cap + 1, 2 * cap + 5] {
                let s = setup(&mut rng, d, n, metric);
                let (cands, counters) = scan_all(&s, false, None, CandBuf::unbounded(), true);
                assert_eq!(cands.len(), n, "d = {d}, n = {n}");
                let mut by_tid = std::collections::HashMap::new();
                for c in &cands {
                    assert!(by_tid.insert(c.tid, *c).is_none(), "{:?} twice", c.tid);
                }
                for i in 0..n {
                    let got = by_tid[&s.r.tids[i]];
                    assert_eq!(
                        got.est.to_bits(),
                        s.rows[i].0.to_bits(),
                        "est of record {i}, d = {d}"
                    );
                    assert_eq!(
                        got.lo.to_bits(),
                        s.rows[i].1.to_bits(),
                        "lo of record {i}, d = {d}"
                    );
                }
                assert_eq!(
                    counters,
                    ScanCounters {
                        scanned: n as u64,
                        admitted: n as u64,
                        blocks_skipped: 0
                    }
                );
            }
        }
    }

    #[test]
    fn without_a_bound_the_candidates_carry_minus_infinity() {
        let mut rng = Rng::new(6);
        let s = setup(&mut rng, 200, 100, Metric::Ip);
        let (cands, _) = scan_all(&s, false, None, CandBuf::unbounded(), false);
        assert!(cands.iter().all(|c| c.lo == f32::NEG_INFINITY));
        assert_eq!(cands.len(), 100);
    }

    #[test]
    fn a_pushed_filter_scores_only_the_records_that_pass_it_and_skips_empty_blocks() {
        let mut rng = Rng::new(7);
        let s = setup(&mut rng, 768, 300, Metric::L2);
        for (lo, hi) in [(i64::MIN, i64::MAX), (0, 0), (-3, -2), (6, 99), (50, 60)] {
            let (cands, counters) = scan_all(&s, true, Some((lo, hi)), CandBuf::unbounded(), true);
            let want: HashSet<(u32, u16)> = (0..300)
                .filter(|&i| !s.r.fnulls[i] && lo <= s.r.fvals[i] && s.r.fvals[i] <= hi)
                .map(|i| s.r.tids[i])
                .collect();
            let got: HashSet<(u32, u16)> = cands.iter().map(|c| c.tid).collect();
            assert_eq!(got, want, "filter {lo}..={hi}");
            assert_eq!(counters.scanned, want.len() as u64);
            assert_eq!(counters.admitted, want.len() as u64);
            if want.is_empty() {
                let blocks: u64 = pages_of(&s.r, true)
                    .iter()
                    .map(|p| ListPage::parse(&p.0[..8168], 96, true).unwrap().n_blocks() as u64)
                    .sum();
                assert_eq!(counters.blocks_skipped, blocks, "no block is scored");
            }
        }
    }

    #[test]
    fn a_bounded_buffer_ends_with_the_best_estimates_of_everything_scanned() {
        let mut rng = Rng::new(8);
        for cap in [10usize, 64, 100] {
            let s = setup(&mut rng, 256, 2_500, Metric::Cosine);
            let (mut cands, counters) = scan_all(&s, false, None, CandBuf::best(cap), true);
            assert_eq!(cands.len(), cap);
            cands.sort_by(|a, b| a.est.total_cmp(&b.est));
            let mut all: Vec<(f32, (u32, u16))> = s
                .rows
                .iter()
                .zip(&s.r.tids)
                .map(|(r, t)| (r.0, *t))
                .collect();
            all.sort_by(|a, b| a.0.total_cmp(&b.0));
            assert_eq!(
                cands.iter().map(|c| c.tid).collect::<Vec<_>>(),
                all[..cap].iter().map(|x| x.1).collect::<Vec<_>>(),
                "cap {cap}"
            );
            assert_eq!(counters.scanned, 2_500);
            assert!(
                counters.admitted < counters.scanned && counters.blocks_skipped > 0,
                "{counters:?}: the admission bound must skip work"
            );
        }
    }

    #[test]
    fn the_scalar_arrays_are_native_values_per_block() {
        let mut rng = Rng::new(9);
        let s = setup(&mut rng, 256, 70, Metric::L2);
        let pages = pages_of(&s.r, false);
        let page = ListPage::parse(&pages[0].0[..8168], 32, false).unwrap();
        let mut scalars = BlockScalars::new();
        page.block_scalars(0, &mut scalars);
        for (i, code) in s.r.codes.iter().take(page.lanes().min(70)).enumerate() {
            assert_eq!(scalars.rnorm[i].to_bits(), code.rnorm.to_bits());
            assert_eq!(scalars.dot[i].to_bits(), code.dot.to_bits());
            assert_eq!(scalars.rc[i].to_bits(), code.rc.to_bits());
        }
        assert_eq!(SCALAR_SIZE, 4);
    }
}
