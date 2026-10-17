//! Build-time calibration of a clustered index's default probe count and rescore pool.
//!
//! The build holds every vector, centroid and code in memory, so it can search the
//! `(nprobe, pool)` grid for the cheapest pair whose recall@10 reaches `target_recall`. Sample rows
//! stand in for queries and are left out of their own search (a row is neither its own neighbour nor
//! a candidate); their exact neighbours come from the rescore's own `exact_distance`; each probe
//! count is scanned once with the scan's `Scorer`, which serves every pool because the rescore is
//! exact. A pair must reach the target plus [`CALIB_OFFSET`], and the winner's probe count is then
//! bisected toward the grid value below it. Counts stay integers until the last comparison, so the
//! result is the same for any worker count.

use std::{collections::BTreeSet, sync::OnceLock, thread};

use super::{
    blockscan::nan_last,
    kernel::Kernel,
    metric::{Metric, exact_distance},
    rng::{SplitMix64, domain},
    rotation::Rotation,
    scan::Scorer,
};

pub(crate) const TOP_K: usize = 10;
/// Nearest rows kept per sample query; ties with the tenth beyond it are ignored.
pub(crate) const TRUTH_KEPT: usize = 32;
/// Below this a sample of [`MIN_SAMPLES`] would be a tenth of the table.
pub(crate) const MIN_ROWS: usize = 500;
pub(crate) const MAX_SAMPLES: usize = 200;
pub(crate) const MIN_SAMPLES: usize = 50;
pub(crate) const SAMPLE_ROW_DIVISOR: usize = 50;
/// Sample queries times rows for the exact pass: 200 over a million, 1.7 s with 8 workers at 768-d.
pub(crate) const TRUTH_BUDGET: u64 = 200_000_000;
pub(crate) const NPROBE_GRID: [u32; 10] = [8, 12, 16, 25, 37, 50, 75, 100, 150, 200];
pub(crate) const POOL_GRID: [u32; 6] = [100, 200, 300, 500, 1000, 2000];
/// One rescored row in scanned codes; measured 55 (`STORAGE PLAIN`) to 170 (TOAST) at 768-d.
pub(crate) const RESCORE_COST: f64 = 100.0;
pub(crate) const BEST_EFFORT_SLACK: f64 = 0.005;
/// A pair reaches the target when the sample's recall, less this many standard errors, does: a fresh
/// query set then misses it about one time in forty.
pub(crate) const MARGIN_SIGMAS: f64 = 2.0;
/// Table rows read 0.01 to 0.04 above held-out queries on wiki_dpr, so a pair must reach
/// `target + CALIB_OFFSET` on the sample. One constant for every data set.
pub(crate) const CALIB_OFFSET: f64 = 0.01;
/// Probe counts tried between the last grid value that misses and the first that reaches.
pub(crate) const REFINE_EVALUATIONS: usize = 4;

/// Per-query hits (`0..=TOP_K`) summed and squared over the sample.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Stats {
    pub(crate) sum: u64,
    pub(crate) sum_sq: u64,
    pub(crate) queries: u32,
}

/// The target plus [`CALIB_OFFSET`], at most halfway to 1.0: a target of 0.99 asks for 0.995, not for
/// an unreachable 1.0.
pub(crate) fn goal(target: f64) -> f64 {
    (target + CALIB_OFFSET).min(f64::midpoint(target, 1.0))
}

/// The bars a pair is held to, against the sample's mean less [`MARGIN_SIGMAS`] standard errors: the
/// goal is preferred, the floor taken when no pair reaches the goal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Aim {
    pub(crate) goal: f64,
    pub(crate) floor: f64,
}

impl Aim {
    pub(crate) fn of(target: f64) -> Aim {
        Aim {
            goal: goal(target),
            floor: target,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Chosen {
    pub(crate) nprobe: u32,
    pub(crate) pool: u32,
    pub(crate) stats: Stats,
    pub(crate) reached: bool,
    pub(crate) levels_scanned: usize,
    /// Index of `nprobe` in the levels searched.
    pub(crate) level: usize,
    /// The bar the pair clears; `None` for a best effort.
    pub(crate) held_to: Option<f64>,
}

pub(crate) struct Built<'a> {
    pub(crate) metric: Metric,
    pub(crate) seed: u64,
    pub(crate) kernel: Kernel,
    pub(crate) vectors: &'a [&'a [f32]],
    pub(crate) centroids: &'a [Vec<f32>],
    pub(crate) lists: &'a [Vec<u32>],
    /// `code_bytes` per row, little-endian words, in row order.
    pub(crate) codes: &'a [u8],
    pub(crate) rnorm: &'a [f32],
    pub(crate) dot: &'a [f32],
    pub(crate) rc: &'a [f32],
}

pub(crate) struct Settings {
    pub(crate) target: f64,
    /// Search the probe count only, for this pool.
    pub(crate) pinned_pool: Option<u32>,
    pub(crate) workers: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Outcome {
    pub(crate) nprobe: u32,
    pub(crate) pool: u32,
    pub(crate) recall: f64,
    pub(crate) samples: u32,
    pub(crate) reached: bool,
    pub(crate) levels_scanned: usize,
    pub(crate) refine_probes: usize,
}

impl Stats {
    pub(crate) fn of_hits(hits: &[u8]) -> Stats {
        hits.iter().fold(Stats::default(), |s, &h| Stats {
            sum: s.sum + u64::from(h),
            sum_sq: s.sum_sq + u64::from(h) * u64::from(h),
            queries: s.queries + 1,
        })
    }

    pub(crate) fn mean(&self) -> f64 {
        if self.queries == 0 {
            return 0.0;
        }
        self.sum as f64 / (TOP_K as f64 * f64::from(self.queries))
    }

    pub(crate) fn std_err(&self) -> f64 {
        if self.queries < 2 {
            return f64::INFINITY;
        }
        let n = u128::from(self.queries);
        let spread = u128::from(self.sum_sq) * n - u128::from(self.sum) * u128::from(self.sum);
        let variance = spread as f64 / (n * (n - 1)) as f64 / (TOP_K * TOP_K) as f64;
        (variance / n as f64).sqrt()
    }

    /// The mean less [`MARGIN_SIGMAS`] standard errors: what the target is held against.
    pub(crate) fn lower_bound(&self) -> f64 {
        self.mean() - MARGIN_SIGMAS * self.std_err()
    }
}

pub(crate) fn sample_size(rows: usize) -> usize {
    let by_rows = rows / SAMPLE_ROW_DIVISOR;
    MAX_SAMPLES
        .min(by_rows)
        .min(budget_samples(rows))
        .max(MIN_SAMPLES)
}

/// Queries the exact pass can afford against `rows` rows (see [`TRUTH_BUDGET`]).
fn budget_samples(rows: usize) -> usize {
    usize::try_from(TRUTH_BUDGET / rows.max(1) as u64).unwrap_or(usize::MAX)
}

/// How many of the user's own queries (`calibration_queries`) a build draws for `rows` table rows:
/// as many as [`sample_size`] allows a table of any size, so a few hundred real queries are all used.
pub(crate) fn query_sample_size(rows: usize) -> usize {
    MAX_SAMPLES.min(budget_samples(rows)).max(MIN_SAMPLES)
}

/// Floyd's algorithm: `count` distinct rows without building a permutation of the table.
pub(crate) fn sample_rows(rows: usize, count: usize, seed: u64) -> Vec<u32> {
    let count = count.min(rows);
    let mut rng = SplitMix64::for_domain(seed, domain::CALIBRATION);
    let mut chosen = BTreeSet::new();
    for j in rows - count..rows {
        let t = rng.next_bounded(j as u32 + 1);
        if !chosen.insert(t) {
            chosen.insert(j as u32);
        }
    }
    chosen.into_iter().collect()
}

pub(crate) fn nprobe_levels(n_lists: usize) -> Vec<u32> {
    let n_lists = n_lists.max(1) as u64;
    let mut grid: Vec<u64> = NPROBE_GRID.iter().map(|&n| u64::from(n)).collect();
    let (mut at, mut half_step) = (200u64, true);
    while at < n_lists {
        at = if half_step { at * 3 / 2 } else { at * 4 / 3 };
        half_step = !half_step;
        grid.push(at);
    }
    let mut levels: Vec<u32> = grid
        .into_iter()
        .filter(|&n| n < n_lists)
        .map(|n| n as u32)
        .collect();
    levels.push(n_lists as u32);
    levels
}

pub(crate) fn pair_cost(nprobe: u32, pool: u32, rows_per_list: f64) -> f64 {
    f64::from(nprobe) * rows_per_list + RESCORE_COST * f64::from(pool)
}

struct Seen {
    cost: f64,
    nprobe: u32,
    pool: u32,
    stats: Stats,
    level: usize,
}

/// `evaluate(level index)` returns one [`Stats`] per pool; levels are scanned lazily. The cheapest pair
/// reaching the goal wins and stops the search at the first level that cannot cost less; failing that
/// the cheapest reaching the floor (every level scanned); failing that the cheapest near the best mean.
pub(crate) fn search(
    levels: &[u32],
    pools: &[u32],
    rows_per_list: f64,
    aim: Aim,
    mut evaluate: impl FnMut(usize) -> Vec<Stats>,
) -> Chosen {
    let cheapest_pool = pools.iter().copied().min().unwrap_or(0);
    let mut seen: Vec<Seen> = Vec::new();
    let mut goal_at: Option<usize> = None;
    let mut floor_at: Option<usize> = None;
    let mut levels_scanned = 0;
    for (level, &nprobe) in levels.iter().enumerate() {
        if let Some(h) = goal_at
            && pair_cost(nprobe, cheapest_pool, rows_per_list) >= seen[h].cost
        {
            break;
        }
        let stats = evaluate(level);
        levels_scanned += 1;
        for (&pool, stats) in pools.iter().zip(stats) {
            let cost = pair_cost(nprobe, pool, rows_per_list);
            let bound = stats.lower_bound();
            seen.push(Seen {
                cost,
                nprobe,
                pool,
                stats,
                level,
            });
            let at = seen.len() - 1;
            let cheaper = |held: Option<usize>| held.is_none_or(|h| cost < seen[h].cost);
            if bound >= aim.goal && cheaper(goal_at) {
                goal_at = Some(at);
            }
            if bound >= aim.floor && cheaper(floor_at) {
                floor_at = Some(at);
            }
        }
    }
    let (held_to, at) = match (goal_at, floor_at) {
        (Some(h), _) => (Some(aim.goal), h),
        (None, Some(f)) => (Some(aim.floor), f),
        (None, None) => {
            let best = seen
                .iter()
                .map(|e| e.stats.mean())
                .fold(f64::NEG_INFINITY, f64::max);
            let near = |e: &Seen| e.stats.mean() >= best - BEST_EFFORT_SLACK;
            let mut cheapest: Option<usize> = None;
            for (i, e) in seen.iter().enumerate() {
                if near(e) && cheapest.is_none_or(|c| e.cost < seen[c].cost) {
                    cheapest = Some(i);
                }
            }
            (None, cheapest.unwrap_or(0))
        }
    };
    let Seen {
        nprobe,
        pool,
        stats,
        level,
        ..
    } = seen[at];
    Chosen {
        nprobe,
        pool,
        stats,
        reached: held_to.is_some(),
        levels_scanned,
        level,
        held_to,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Refined {
    pub(crate) nprobe: u32,
    pub(crate) stats: Stats,
    pub(crate) evaluations: usize,
}

/// Bisects `(lo, hi]` for the smallest probe count reaching `bar` at the chosen pool; `lo` misses it,
/// `hi` reached it with `hi_stats`. At most [`REFINE_EVALUATIONS`] probes, each held to the grid's
/// margin, so the count returned has been seen to reach the bar (the smallest one when the bracket is
/// no wider than 2^[`REFINE_EVALUATIONS`] and recall rises with the probe count).
pub(crate) fn refine(
    lo: u32,
    hi: u32,
    hi_stats: Stats,
    bar: f64,
    mut probe: impl FnMut(u32) -> Stats,
) -> Refined {
    let (mut lo, mut hi, mut stats, mut evaluations) = (lo, hi, hi_stats, 0);
    while hi.saturating_sub(lo) > 1 && evaluations < REFINE_EVALUATIONS {
        let mid = lo + (hi - lo) / 2;
        let at_mid = probe(mid);
        evaluations += 1;
        if at_mid.lower_bound() >= bar {
            (hi, stats) = (mid, at_mid);
        } else {
            lo = mid;
        }
    }
    Refined {
        nprobe: hi,
        stats,
        evaluations,
    }
}

/// The truth's order, the scan's: ascending distance with a NaN last whatever its sign (`total_cmp`
/// puts one with the sign bit set first: x86 makes one of `inf - inf`, aarch64 of the negated
/// inner product), then the row.
fn order(a: (f32, u32), b: (f32, u32)) -> std::cmp::Ordering {
    nan_last(a.0, b.0).then(a.1.cmp(&b.1))
}

/// The [`TRUTH_KEPT`] nearest rows seen so far, ascending.
#[derive(Clone, Default)]
struct Nearest {
    kept: Vec<(f32, u32)>,
}

impl Nearest {
    fn offer(&mut self, dist: f32, row: u32) {
        let entry = (dist, row);
        if self.kept.len() == TRUTH_KEPT && order(entry, self.kept[TRUTH_KEPT - 1]).is_ge() {
            return;
        }
        let at = self.kept.partition_point(|&e| order(e, entry).is_lt());
        self.kept.insert(at, entry);
        self.kept.truncate(TRUTH_KEPT);
    }

    fn merge(&mut self, other: &Nearest) {
        for &(dist, row) in &other.kept {
            self.offer(dist, row);
        }
    }

    /// Every kept row within the tenth distance, ascending by row.
    fn neighbours(&self) -> Vec<u32> {
        let Some(&(tenth, _)) = self.kept.get(TOP_K - 1).or(self.kept.last()) else {
            return Vec::new();
        };
        let mut rows: Vec<u32> = self
            .kept
            .iter()
            .filter(|&&(d, _)| nan_last(d, tenth).is_le())
            .map(|&(_, r)| r)
            .collect();
        rows.sort_unstable();
        rows
    }
}

/// Rows per round of the exact pass: `interrupt` runs between rounds.
const TRUTH_ROUND_ROWS: usize = 65_536;

/// A calibration query; `leave_out` is the table row it copies, left out of its search as neighbour
/// and as candidate.
#[derive(Clone, Copy)]
pub(crate) struct Query<'a> {
    pub(crate) vector: &'a [f32],
    pub(crate) leave_out: Option<u32>,
}

pub(crate) fn sample_queries<'a>(vectors: &[&'a [f32]], sample: &[u32]) -> Vec<Query<'a>> {
    sample
        .iter()
        .map(|&row| Query {
            vector: vectors[row as usize],
            leave_out: Some(row),
        })
        .collect()
}

/// Per query: every row but its `leave_out` within its tenth-smallest distance (ties included),
/// ascending.
pub(crate) fn exact_neighbours_of(
    vectors: &[&[f32]],
    metric: Metric,
    queries: &[Query],
    workers: usize,
    interrupt: &dyn Fn(),
) -> Vec<Vec<u32>> {
    let scan = |rows: std::ops::Range<usize>| -> Vec<Nearest> {
        let mut local = vec![Nearest::default(); queries.len()];
        for row in rows {
            let v = vectors[row];
            for (nearest, query) in local.iter_mut().zip(queries) {
                if query.leave_out != Some(row as u32) {
                    nearest.offer(exact_distance(metric, query.vector, v), row as u32);
                }
            }
        }
        local
    };
    let mut total = vec![Nearest::default(); queries.len()];
    for start in (0..vectors.len()).step_by(TRUTH_ROUND_ROWS) {
        let end = (start + TRUTH_ROUND_ROWS).min(vectors.len());
        let parts: Vec<Vec<Nearest>> = if workers <= 1 {
            vec![scan(start..end)]
        } else {
            let share = (end - start).div_ceil(workers).max(1);
            thread::scope(|scope| {
                let handles: Vec<_> = (start..end)
                    .step_by(share)
                    .map(|from| {
                        let scan = &scan;
                        scope.spawn(move || scan(from..(from + share).min(end)))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("pg_pinecone: calibration worker panicked"))
                    .collect()
            })
        };
        for part in &parts {
            for (all, one) in total.iter_mut().zip(part) {
                all.merge(one);
            }
        }
        interrupt();
    }
    total.iter().map(Nearest::neighbours).collect()
}

/// What a scan reads, regrouped by list: the build's per-row arrays in list order, and the rotated
/// centroids, computed once per list when a query first reaches it.
struct Engine<'a> {
    built: &'a Built<'a>,
    rotation: Rotation,
    words_per_code: usize,
    list_start: Vec<usize>,
    rows: Vec<u32>,
    words: Vec<u64>,
    dot: Vec<f32>,
    rnorm: Vec<f32>,
    rc: Vec<f32>,
    rotated_centroids: Vec<OnceLock<Box<[f32]>>>,
}

impl<'a> Engine<'a> {
    fn new(built: &'a Built<'a>) -> Engine<'a> {
        let rotation = Rotation::new(built.seed, built.vectors[0].len());
        let words_per_code = rotation.padded() / 64;
        let total: usize = built.lists.iter().map(Vec::len).sum();
        let mut engine = Engine {
            built,
            words_per_code,
            list_start: Vec::with_capacity(built.lists.len() + 1),
            rows: Vec::with_capacity(total),
            words: Vec::with_capacity(total * words_per_code),
            dot: Vec::with_capacity(total),
            rnorm: Vec::with_capacity(total),
            rc: Vec::with_capacity(total),
            rotated_centroids: built.centroids.iter().map(|_| OnceLock::new()).collect(),
            rotation,
        };
        let code_bytes = words_per_code * 8;
        for list in built.lists {
            engine.list_start.push(engine.rows.len());
            for &row in list {
                let at = row as usize;
                engine.rows.push(row);
                let code = &built.codes[at * code_bytes..(at + 1) * code_bytes];
                engine.words.extend(
                    code.as_chunks::<8>()
                        .0
                        .iter()
                        .map(|w| u64::from_le_bytes(*w)),
                );
                engine.dot.push(built.dot[at]);
                engine.rnorm.push(built.rnorm[at]);
                engine.rc.push(built.rc[at]);
            }
        }
        engine.list_start.push(engine.rows.len());
        engine
    }

    fn rotated_centroid(&self, list: usize) -> &[f32] {
        self.rotated_centroids[list].get_or_init(|| {
            self.rotation
                .apply(&self.built.centroids[list])
                .into_boxed_slice()
        })
    }
}

/// One sample query's scan, resumable across probe counts.
struct QueryScan {
    leave_out: Option<u32>,
    scorer: Scorer,
    ranked: Vec<u32>,
    consumed: usize,
    best: Vec<(f32, u32)>,
    truth: Vec<u32>,
}

/// Pools beyond this are scored as this one: a pool that size rescores most of what is probed.
const MAX_TRACKED_POOL: usize = 20_000;

struct Mark {
    consumed: usize,
    best: Vec<(f32, u32)>,
}

impl QueryScan {
    fn new(engine: &Engine, query: Query, truth: Vec<u32>) -> QueryScan {
        let built = engine.built;
        let mut ranked: Vec<(u32, f32)> = (0..built.centroids.len())
            .filter(|&l| engine.list_start[l] < engine.list_start[l + 1])
            .map(|l| {
                (
                    l as u32,
                    exact_distance(built.metric, query.vector, &built.centroids[l]),
                )
            })
            .collect();
        ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
        QueryScan {
            leave_out: query.leave_out,
            scorer: Scorer::build(
                built.metric,
                built.seed,
                built.kernel,
                &engine.rotation,
                query.vector,
            ),
            ranked: ranked.into_iter().map(|(l, _)| l).collect(),
            consumed: 0,
            best: Vec::new(),
            truth,
        }
    }

    fn mark(&self) -> Mark {
        Mark {
            consumed: self.consumed,
            best: self.best.clone(),
        }
    }

    fn rewind(&mut self, mark: &Mark) {
        self.consumed = mark.consumed;
        self.best.clone_from(&mark.best);
    }

    fn prune(&mut self, keep: usize) {
        if self.best.len() > keep {
            self.best
                .select_nth_unstable_by(keep - 1, |a, b| order(*a, *b));
            self.best.truncate(keep);
        }
    }

    /// Scans lists up to `nprobe`, then counts the true neighbours among the best `pool` candidates
    /// for every pool.
    fn hits(&mut self, engine: &Engine, nprobe: usize, pools: &[u32], keep: usize) -> Vec<u8> {
        let built = engine.built;
        let upto = nprobe.min(self.ranked.len());
        while self.consumed < upto {
            let list = self.ranked[self.consumed] as usize;
            self.scorer.prepare(
                list as u32,
                &built.centroids[list],
                engine.rotated_centroid(list),
            );
            let words = engine.words_per_code;
            for slot in engine.list_start[list]..engine.list_start[list + 1] {
                let row = engine.rows[slot];
                if Some(row) == self.leave_out {
                    continue;
                }
                let code = &engine.words[slot * words..(slot + 1) * words];
                let key =
                    self.scorer
                        .key(code, engine.dot[slot], engine.rnorm[slot], engine.rc[slot]);
                self.best.push((key, row));
                if self.best.len() >= 2 * keep {
                    self.prune(keep);
                }
            }
            self.consumed += 1;
        }
        self.prune(keep);
        self.best.sort_by(|a, b| order(*a, *b));
        pools
            .iter()
            .map(|&pool| {
                let pool = (pool as usize).min(self.best.len());
                let found = self.best[..pool]
                    .iter()
                    .filter(|(_, r)| self.truth.binary_search(r).is_ok())
                    .count();
                found.min(TOP_K) as u8
            })
            .collect()
    }
}

fn evaluate_level(
    engine: &Engine,
    scans: &mut [QueryScan],
    nprobe: usize,
    pools: &[u32],
    keep: usize,
    workers: usize,
) -> Vec<Stats> {
    let run = |chunk: &mut [QueryScan]| -> Vec<Vec<u8>> {
        chunk
            .iter_mut()
            .map(|q| q.hits(engine, nprobe, pools, keep))
            .collect()
    };
    let per_query: Vec<Vec<u8>> = if workers <= 1 {
        run(scans)
    } else {
        let share = scans.len().div_ceil(workers).max(1);
        thread::scope(|scope| {
            let handles: Vec<_> = scans
                .chunks_mut(share)
                .map(|chunk| scope.spawn(|| run(chunk)))
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("pg_pinecone: calibration worker panicked"))
                .collect()
        })
    };
    (0..pools.len())
        .map(|p| Stats::of_hits(&per_query.iter().map(|h| h[p]).collect::<Vec<u8>>()))
        .collect()
}

/// The sample's scans, level by level. A scan only moves forward, so every query's state after each
/// level is kept: a probe count between two levels restarts from the lower one.
struct Sweep<'a> {
    engine: &'a Engine<'a>,
    scans: Vec<QueryScan>,
    after: Vec<Vec<Mark>>,
    levels: Vec<u32>,
    pools: Vec<u32>,
    keep: usize,
    workers: usize,
}

impl Sweep<'_> {
    fn level(&mut self, i: usize) -> Vec<Stats> {
        let stats = evaluate_level(
            self.engine,
            &mut self.scans,
            self.levels[i] as usize,
            &self.pools,
            self.keep,
            self.workers,
        );
        self.after
            .push(self.scans.iter().map(QueryScan::mark).collect());
        stats
    }

    /// `nprobe` lists for one pool, restarted from the state after level `below`.
    fn probe(&mut self, below: usize, nprobe: u32, pool: u32) -> Stats {
        for (scan, mark) in self.scans.iter_mut().zip(&self.after[below]) {
            scan.rewind(mark);
        }
        let stats = evaluate_level(
            self.engine,
            &mut self.scans,
            nprobe as usize,
            &[pool],
            self.keep,
            self.workers,
        );
        stats[0]
    }
}

/// `None` below [`MIN_ROWS`]. `interrupt` runs between phases on the calling thread.
pub(crate) fn calibrate(
    built: &Built,
    settings: &Settings,
    interrupt: &dyn Fn(),
) -> Option<Outcome> {
    let rows = built.vectors.len();
    if rows < MIN_ROWS || built.centroids.is_empty() {
        return None;
    }
    let sample = sample_rows(rows, sample_size(rows), built.seed);
    let queries = sample_queries(built.vectors, &sample);
    Some(calibrate_queries(built, settings, &queries, interrupt))
}

/// [`calibrate`] against the user's own `queries` instead of table rows (`calibration_queries`): vectors
/// of the table's dimension, none of them a row, so nothing is left out and the truth is the exact top
/// ten of the whole table. `None` below [`MIN_ROWS`] or without queries.
pub(crate) fn calibrate_on(
    built: &Built,
    settings: &Settings,
    queries: &[&[f32]],
    interrupt: &dyn Fn(),
) -> Option<Outcome> {
    if built.vectors.len() < MIN_ROWS || built.centroids.is_empty() || queries.is_empty() {
        return None;
    }
    let queries = external_queries(queries);
    Some(calibrate_queries(built, settings, &queries, interrupt))
}

/// The user's own queries: no table row behind them, so nothing is left out of any search.
pub(crate) fn external_queries<'a>(vectors: &[&'a [f32]]) -> Vec<Query<'a>> {
    vectors
        .iter()
        .map(|&vector| Query {
            vector,
            leave_out: None,
        })
        .collect()
}

fn calibrate_queries(
    built: &Built,
    settings: &Settings,
    queries: &[Query],
    interrupt: &dyn Fn(),
) -> Outcome {
    let rows = built.vectors.len();
    let neighbours = exact_neighbours_of(
        built.vectors,
        built.metric,
        queries,
        settings.workers,
        interrupt,
    );
    let engine = Engine::new(built);
    let scans: Vec<QueryScan> = queries
        .iter()
        .zip(neighbours)
        .map(|(&query, truth)| QueryScan::new(&engine, query, truth))
        .collect();
    let pools: Vec<u32> = match settings.pinned_pool {
        Some(pool) => vec![pool.min(MAX_TRACKED_POOL as u32)],
        None => POOL_GRID.to_vec(),
    };
    let levels = nprobe_levels(built.centroids.len());
    let rows_per_list = rows as f64 / built.centroids.len() as f64;
    let mut sweep = Sweep {
        engine: &engine,
        scans,
        after: Vec::new(),
        keep: pools.iter().copied().max().unwrap_or(0) as usize,
        levels: levels.clone(),
        pools: pools.clone(),
        workers: settings.workers,
    };
    let mut chosen = search(
        &levels,
        &pools,
        rows_per_list,
        Aim::of(settings.target),
        |i| {
            interrupt();
            sweep.level(i)
        },
    );
    let mut refine_probes = 0;
    if let (Some(bar), Some(below)) = (chosen.held_to, chosen.level.checked_sub(1)) {
        let refined = refine(levels[below], chosen.nprobe, chosen.stats, bar, |nprobe| {
            interrupt();
            sweep.probe(below, nprobe, chosen.pool)
        });
        chosen.nprobe = refined.nprobe;
        chosen.stats = refined.stats;
        refine_probes = refined.evaluations;
    }
    Outcome {
        nprobe: chosen.nprobe,
        pool: chosen.pool,
        recall: chosen.stats.mean(),
        samples: queries.len() as u32,
        reached: chosen.reached,
        levels_scanned: chosen.levels_scanned,
        refine_probes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local::{
        kmeans::KMeans,
        metric::{exact_distance, normalise_into},
        options::legacy_nprobe,
        rabitq::{self, Code, EncodeScratch},
        rotation::Rotation,
        testkit::{self as tk, MixtureSpec},
    };

    fn stats_at(permille: u32) -> Stats {
        let total = 10 * permille;
        let (base, extra) = ((total / 1000) as u8, (total % 1000) as usize);
        let mut hits = vec![base + 1; extra];
        hits.resize(1000, base);
        Stats::of_hits(&hits)
    }

    #[test]
    fn stats_are_exact_integer_sums_and_the_bound_is_the_mean_less_two_standard_errors() {
        let s = Stats::of_hits(&[10, 10, 9, 10]);
        assert_eq!((s.sum, s.sum_sq, s.queries), (39, 381, 4));
        assert!((s.mean() - 0.975).abs() < 1e-12);
        assert!((s.std_err() - 0.025).abs() < 1e-12);
        assert!((s.lower_bound() - 0.925).abs() < 1e-12);
        let flat = Stats::of_hits(&[9; 50]);
        assert_eq!(
            (flat.mean(), flat.std_err(), flat.lower_bound()),
            (0.9, 0.0, 0.9)
        );
        assert_eq!(Stats::default().mean(), 0.0);
        assert!(
            Stats::of_hits(&[10]).lower_bound() < 0.0,
            "one query proves nothing"
        );
    }

    #[test]
    fn the_sample_is_two_percent_of_the_rows_between_50_and_200_within_the_exact_pass_budget() {
        let table = [
            (500, 50),
            (2_500, 50),
            (5_000, 100),
            (9_999, 199),
            (10_000, 200),
            (20_000, 200),
            (300_000, 200),
            (1_000_000, 200),
            (1_500_000, 133),
            (2_000_000, 100),
            (4_000_000, 50),
            (10_000_000, 50),
            (100_000_000, 50),
        ];
        for (rows, want) in table {
            assert_eq!(sample_size(rows), want, "rows = {rows}");
        }
        let mut previous = usize::MAX;
        for rows in (MIN_ROWS..=20_000_000).step_by(7_919) {
            let s = sample_size(rows);
            assert!(
                (MIN_SAMPLES..=MAX_SAMPLES).contains(&s),
                "rows = {rows}: {s}"
            );
            if s > MIN_SAMPLES {
                assert!(s <= rows / SAMPLE_ROW_DIVISOR, "rows = {rows}: {s}");
                assert!(s as u64 * rows as u64 <= TRUTH_BUDGET, "rows = {rows}: {s}");
            }
            if rows >= 10_000 {
                assert!(s <= previous, "rows = {rows}: {s} after {previous}");
                previous = s;
            }
        }
    }

    #[test]
    fn sample_rows_are_distinct_ascending_in_range_and_a_function_of_the_seed() {
        for (rows, count) in [(500, 50), (1_000, 200), (60, 60), (10_000, 1)] {
            let sample = sample_rows(rows, count, 7);
            assert_eq!(sample.len(), count);
            assert!(
                sample.windows(2).all(|w| w[0] < w[1]),
                "ascending and distinct"
            );
            assert!(sample.iter().all(|&r| (r as usize) < rows));
            assert_eq!(sample, sample_rows(rows, count, 7));
        }
        assert_ne!(sample_rows(1_000, 50, 1), sample_rows(1_000, 50, 2));
        assert_eq!(sample_rows(60, 60, 3), (0..60).collect::<Vec<u32>>());
    }

    #[test]
    fn probe_counts_are_the_grid_then_steps_of_one_and_a_half_and_one_and_a_third_up_to_the_lists()
    {
        let to = |n: usize| nprobe_levels(n);
        assert_eq!(
            to(1000),
            [
                8, 12, 16, 25, 37, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000
            ]
        );
        assert_eq!(to(142), [8, 12, 16, 25, 37, 50, 75, 100, 142]);
        assert_eq!(to(25), [8, 12, 16, 25]);
        assert_eq!(to(20), [8, 12, 16, 20]);
        assert_eq!(to(200), [8, 12, 16, 25, 37, 50, 75, 100, 150, 200]);
        assert_eq!(to(201), [8, 12, 16, 25, 37, 50, 75, 100, 150, 200, 201]);
        assert_eq!((to(8), to(5), to(1)), (vec![8], vec![5], vec![1]));
        let big = to(1_000_000);
        assert_eq!(big.last(), Some(&1_000_000));
        assert!(
            big.windows(2).all(|w| w[0] < w[1]) && big.len() <= 40,
            "{big:?}"
        );
    }

    #[test]
    fn a_pair_costs_its_scanned_codes_plus_100_per_rescored_row() {
        assert_eq!(pair_cost(16, 200, 1000.0), 36_000.0);
        assert_eq!(pair_cost(25, 100, 1000.0), 35_000.0);
        assert_eq!(pair_cost(12, 300, 1000.0), 42_000.0);
        assert!(pair_cost(8, 100, 50.0) < pair_cost(8, 200, 50.0));
    }

    /// A bar without the offset: the goal is the target.
    fn plain(target: f64) -> Aim {
        Aim {
            goal: target,
            floor: target,
        }
    }

    fn run_search_for(
        aim: Aim,
        levels: &[u32],
        pools: &[u32],
        rows_per_list: f64,
        recall: impl Fn(u32, u32) -> u32,
    ) -> (Chosen, Vec<u32>) {
        let mut scanned = Vec::new();
        let chosen = search(levels, pools, rows_per_list, aim, |i| {
            scanned.push(levels[i]);
            pools
                .iter()
                .map(|&p| stats_at(recall(levels[i], p)))
                .collect()
        });
        (chosen, scanned)
    }

    fn run_search(
        levels: &[u32],
        pools: &[u32],
        rows_per_list: f64,
        recall: impl Fn(u32, u32) -> u32,
    ) -> (Chosen, Vec<u32>) {
        run_search_for(plain(0.95), levels, pools, rows_per_list, recall)
    }

    #[test]
    fn the_search_keeps_the_cheapest_pair_and_stops_at_a_level_that_cannot_beat_it() {
        let recall = |nprobe: u32, pool: u32| match (nprobe, pool) {
            (8, _) => 600,
            (12, _) => 700,
            (16, 100) => 900,
            (16, 200) => 920,
            (16, 300) => 940,
            (16, _) => 970,
            (25, 100) => 930,
            (25, _) => 970,
            _ => 990,
        };
        let pools = [100, 200, 300, 500, 1000, 2000];
        let (chosen, scanned) = run_search(&[8, 12, 16, 25, 37, 50], &pools, 1000.0, recall);
        assert_eq!(
            (chosen.nprobe, chosen.pool, chosen.reached),
            (25, 200, true)
        );
        assert_eq!(
            scanned,
            [8, 12, 16, 25],
            "37 lists cost at least 47,000 against 45,000"
        );
        assert_eq!(chosen.levels_scanned, 4);
    }

    #[test]
    fn a_target_no_pair_reaches_keeps_the_cheapest_pair_near_the_best_recall() {
        let recall = |nprobe: u32, pool: u32| match (nprobe, pool) {
            (25, 1000) => 897,
            (25, 2000) => 899,
            (37, 1000) => 898,
            (37, 2000) | (50, 2000) => 900,
            _ => 800,
        };
        let pools = [100, 200, 300, 500, 1000, 2000];
        let (chosen, scanned) = run_search(&[8, 12, 16, 25, 37, 50], &pools, 1000.0, recall);
        assert_eq!(
            (chosen.nprobe, chosen.pool, chosen.reached),
            (25, 1000, false)
        );
        assert_eq!(
            scanned.len(),
            6,
            "nothing reached the target: every level is scanned"
        );
        assert!((chosen.stats.mean() - 0.897).abs() < 1e-9);
    }

    #[test]
    fn the_target_is_held_against_the_mean_less_two_standard_errors() {
        let noisy = |hits_high: usize, low: u8| {
            let mut hits = vec![10u8; hits_high];
            hits.resize(100, low);
            Stats::of_hits(&hits)
        };
        let (wide, narrow) = (noisy(88, 6), noisy(97, 7));
        assert!(wide.mean() >= 0.95 && wide.lower_bound() < 0.95, "{wide:?}");
        assert!(narrow.lower_bound() >= 0.95, "{narrow:?}");
        let chosen = search(&[16, 25], &[100], 1000.0, plain(0.95), |i| {
            vec![if i == 0 { wide } else { narrow }]
        });
        assert_eq!((chosen.nprobe, chosen.reached), (25, true));
    }

    #[test]
    fn a_pinned_pool_searches_the_probe_count_alone() {
        let recall = |nprobe: u32, _pool: u32| if nprobe >= 25 { 970 } else { 940 };
        let (chosen, scanned) = run_search(&[8, 12, 16, 25, 37], &[300], 1000.0, recall);
        assert_eq!(
            (chosen.nprobe, chosen.pool, chosen.reached),
            (25, 300, true)
        );
        assert_eq!(scanned, [8, 12, 16, 25]);
    }

    #[test]
    fn equal_costs_keep_the_smaller_probe_count() {
        let recall = |nprobe: u32, pool: u32| {
            if nprobe == 50 && pool == 100 {
                800
            } else {
                970
            }
        };
        let (chosen, scanned) = run_search(&[50, 150], &[100, 200], 100.0, recall);
        assert_eq!((chosen.nprobe, chosen.pool), (50, 200));
        assert_eq!(scanned, [50], "(150, 100) costs the same 25,000");
    }

    struct Fixture {
        vectors: Vec<Vec<f32>>,
        queries: Vec<Vec<f32>>,
        centroids: Vec<Vec<f32>>,
        lists: Vec<Vec<u32>>,
        codes: Vec<u8>,
        rnorm: Vec<f32>,
        dot: Vec<f32>,
        rc: Vec<f32>,
        metric: Metric,
        seed: u64,
    }

    fn spec(dim: usize, components: usize) -> MixtureSpec {
        MixtureSpec {
            dim,
            components,
            centre_norm: 1.0,
            offset_norm: 0.0,
            offset_components: 0,
            rank: 8,
            factor_std: (0.5f64 / 8.0).sqrt(),
            noise_norm: 0.6,
            scale_range: None,
        }
    }

    fn fixture_of(
        vectors: Vec<Vec<f32>>,
        queries: Vec<Vec<f32>>,
        n_lists: u32,
        metric: Metric,
    ) -> Fixture {
        let seed = 0xFA57;
        let dim = vectors[0].len();
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        let km = KMeans::train(&refs, dim, n_lists, metric, seed, 0);
        let rotation = Rotation::new(seed, dim);
        let mut f = Fixture {
            vectors: Vec::new(),
            queries,
            centroids: km.centroids.clone(),
            lists: vec![Vec::new(); km.centroids.len()],
            codes: Vec::new(),
            rnorm: Vec::new(),
            dot: Vec::new(),
            rc: Vec::new(),
            metric,
            seed,
        };
        let (mut scratch, mut unit) = (EncodeScratch::default(), Vec::new());
        let mut code = Code {
            bits: Vec::new(),
            rnorm: 0.0,
            dot: 1.0,
            rc: 0.0,
        };
        for (row, v) in vectors.iter().enumerate() {
            let cluster = km.assign(v, metric);
            f.lists[cluster].push(row as u32);
            let x: &[f32] = if metric == Metric::Cosine {
                normalise_into(v, &mut unit);
                &unit
            } else {
                v
            };
            rabitq::encode_into(
                &rotation,
                x,
                &km.centroids[cluster],
                &mut scratch,
                &mut code,
            );
            code.bits
                .iter()
                .for_each(|w| f.codes.extend_from_slice(&w.to_le_bytes()));
            f.rnorm.push(code.rnorm);
            f.dot.push(code.dot);
            f.rc.push(code.rc);
        }
        f.vectors = vectors;
        f
    }

    fn mixture_fixture(dim: usize, rows: usize, n_lists: u32, metric: Metric) -> Fixture {
        let data = tk::mixture(&spec(dim, 20), rows, 100, 0xDA7A + dim as u64);
        fixture_of(data.vectors, data.queries, n_lists, metric)
    }

    fn calibrated(
        f: &Fixture,
        target: f64,
        pinned_pool: Option<u32>,
        workers: usize,
    ) -> Option<Outcome> {
        let refs: Vec<&[f32]> = f.vectors.iter().map(Vec::as_slice).collect();
        let built = Built {
            metric: f.metric,
            seed: f.seed,
            kernel: Kernel::detect(),
            vectors: &refs,
            centroids: &f.centroids,
            lists: &f.lists,
            codes: &f.codes,
            rnorm: &f.rnorm,
            dot: &f.dot,
            rc: &f.rc,
        };
        calibrate(
            &built,
            &Settings {
                target,
                pinned_pool,
                workers,
            },
            &|| {},
        )
    }

    fn brute_neighbours(
        vectors: &[Vec<f32>],
        metric: Metric,
        query: &[f32],
        skip: Option<u32>,
    ) -> Vec<u32> {
        let mut all: Vec<(f32, u32)> = vectors
            .iter()
            .enumerate()
            .filter(|(j, _)| Some(*j as u32) != skip)
            .map(|(j, v)| (exact_distance(metric, query, v), j as u32))
            .collect();
        all.sort_by(|a, b| nan_last(a.0, b.0).then(a.1.cmp(&b.1)));
        all.truncate(TRUTH_KEPT);
        let tenth = all[TOP_K - 1].0;
        let mut rows: Vec<u32> = all
            .iter()
            .filter(|(d, _)| nan_last(*d, tenth).is_le())
            .map(|&(_, j)| j)
            .collect();
        rows.sort_unstable();
        rows
    }

    fn naive_hits(
        f: &Fixture,
        query: &[f32],
        skip: Option<u32>,
        nprobe: usize,
        pool: usize,
    ) -> usize {
        let rotation = Rotation::new(f.seed, query.len());
        let mut unit = Vec::new();
        let q: &[f32] = if f.metric == Metric::Cosine {
            normalise_into(query, &mut unit);
            &unit
        } else {
            query
        };
        let ptq = rotation.apply(q);
        let mut ranked: Vec<(usize, f32)> = f
            .centroids
            .iter()
            .enumerate()
            .filter(|(l, _)| !f.lists[*l].is_empty())
            .map(|(l, c)| (l, exact_distance(f.metric, query, c)))
            .collect();
        ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
        let code_bytes = rotation.padded() / 8;
        let mut cands: Vec<(f32, u32)> = Vec::new();
        for &(list, _) in ranked.iter().take(nprobe) {
            let centroid = &f.centroids[list];
            let ptc = rotation.apply(centroid);
            let prep =
                rabitq::prepare_query(f.metric, q, &ptq, centroid, &ptc, f.seed, list as u32);
            for &row in &f.lists[list] {
                if Some(row) == skip {
                    continue;
                }
                let at = row as usize;
                let bits = f.codes[at * code_bytes..(at + 1) * code_bytes]
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|w| u64::from_le_bytes(*w))
                    .collect();
                let code = Code {
                    bits,
                    rnorm: f.rnorm[at],
                    dot: f.dot[at],
                    rc: f.rc[at],
                };
                let (est, _) = rabitq::estimate(&prep, &code);
                cands.push((rabitq::key(f.metric, &prep, &code, est), row));
            }
        }
        cands.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        cands.truncate(pool);
        let truth = brute_neighbours(&f.vectors, f.metric, query, skip);
        cands
            .iter()
            .filter(|(_, r)| truth.binary_search(r).is_ok())
            .count()
            .min(TOP_K)
    }

    #[test]
    fn exact_neighbours_equal_a_brute_force_ranking_with_ties_for_any_worker_count() {
        let mut rng = tk::Rng::new(7);
        for (rows, dims, values) in [(700usize, 5usize, 3usize), (600, 12, 40)] {
            let vectors: Vec<Vec<f32>> = (0..rows)
                .map(|_| (0..dims).map(|_| rng.below(values) as f32).collect())
                .collect();
            let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
            let sample = sample_rows(rows, 50, 3);
            for metric in [Metric::L2, Metric::Cosine, Metric::Ip] {
                let want: Vec<Vec<u32>> = sample
                    .iter()
                    .map(|&s| brute_neighbours(&vectors, metric, &vectors[s as usize], Some(s)))
                    .collect();
                for workers in [0, 1, 3, 8] {
                    let got = exact_neighbours(&refs, metric, &sample, workers, &|| {});
                    assert_eq!(got, want, "{metric:?}, {rows} rows, {workers} workers");
                }
            }
        }
    }

    #[test]
    fn the_calibrated_recall_is_what_a_naive_scan_of_the_same_sample_gets_through_the_public_estimator()
     {
        for metric in [Metric::L2, Metric::Cosine, Metric::Ip] {
            let f = mixture_fixture(32, 1500, 40, metric);
            let outcome = calibrated(&f, 0.95, None, 3).expect("1,500 rows are calibrated");
            assert_eq!(outcome.samples as usize, sample_size(1500));
            let sample = sample_rows(f.vectors.len(), outcome.samples as usize, f.seed);
            let (nprobe, pool) = (outcome.nprobe as usize, outcome.pool as usize);
            let hits: usize = sample
                .iter()
                .map(|&s| naive_hits(&f, &f.vectors[s as usize], Some(s), nprobe, pool))
                .sum();
            assert_eq!(
                outcome.recall,
                hits as f64 / (TOP_K * sample.len()) as f64,
                "{metric:?}: {outcome:?}"
            );
        }
    }

    #[test]
    fn a_resumed_scan_gets_the_hits_of_a_naive_scan_at_every_probe_count_and_pool() {
        let weak = MixtureSpec {
            noise_norm: 2.0,
            ..spec(32, 40)
        };
        let data = tk::mixture(&weak, 1_200, 0, 0x5CA7);
        let f = fixture_of(data.vectors, Vec::new(), 35, Metric::L2);
        let refs: Vec<&[f32]> = f.vectors.iter().map(Vec::as_slice).collect();
        let built = Built {
            metric: f.metric,
            seed: f.seed,
            kernel: Kernel::detect(),
            vectors: &refs,
            centroids: &f.centroids,
            lists: &f.lists,
            codes: &f.codes,
            rnorm: &f.rnorm,
            dot: &f.dot,
            rc: &f.rc,
        };
        let sample = sample_rows(refs.len(), MIN_SAMPLES, f.seed);
        let truth = exact_neighbours(&refs, f.metric, &sample, 2, &|| {});
        let engine = Engine::new(&built);
        let mut scans: Vec<QueryScan> = sample
            .iter()
            .zip(truth)
            .map(|(&row, t)| QueryScan::new(&engine, sample_query(&refs, row), t))
            .collect();
        let pools = [100u32, 200, 300];
        let mut imperfect = 0;
        for nprobe in [1usize, 3, 8, 20, 35] {
            for (scan, &row) in scans.iter_mut().zip(&sample) {
                let hits = scan.hits(&engine, nprobe, &pools, 300);
                for (&hit, &pool) in hits.iter().zip(&pools) {
                    let query = &f.vectors[row as usize];
                    let naive = naive_hits(&f, query, Some(row), nprobe, pool as usize);
                    assert_eq!(
                        usize::from(hit),
                        naive,
                        "row {row}, {nprobe} lists, pool {pool}"
                    );
                    imperfect += usize::from(naive < TOP_K);
                }
            }
        }
        assert!(
            imperfect > 10,
            "premise: the settings differ in recall ({imperfect} misses)"
        );
    }

    #[test]
    fn the_outcome_is_the_same_for_every_worker_count() {
        let f = mixture_fixture(32, 1200, 35, Metric::L2);
        let serial = calibrated(&f, 0.95, None, 0).expect("calibrated");
        for workers in [1, 2, 5, 16] {
            assert_eq!(
                calibrated(&f, 0.95, None, workers),
                Some(serial),
                "workers = {workers}"
            );
        }
        let pinned = calibrated(&f, 0.95, Some(300), 0).expect("calibrated");
        assert_eq!(pinned.pool, 300);
        assert_eq!(calibrated(&f, 0.95, Some(300), 4), Some(pinned));
    }

    #[test]
    fn a_table_below_the_minimum_is_not_calibrated() {
        let few = mixture_fixture(16, MIN_ROWS - 1, 12, Metric::L2);
        assert_eq!(calibrated(&few, 0.95, None, 0), None);
        let enough = mixture_fixture(16, MIN_ROWS, 12, Metric::L2);
        assert!(calibrated(&enough, 0.95, None, 0).is_some());
    }

    #[test]
    fn a_clustered_set_reaches_the_target_on_the_sample_and_on_fresh_queries_for_less_than_the_fallback()
     {
        for metric in [Metric::L2, Metric::Cosine] {
            let f = mixture_fixture(32, 2000, 45, metric);
            let outcome = calibrated(&f, 0.95, None, 2).expect("calibrated");
            assert!(
                outcome.reached && outcome.recall >= 0.95,
                "{metric:?}: {outcome:?}"
            );
            let per_list = 2000.0 / 45.0;
            let fallback = pair_cost(legacy_nprobe(45), 1000, per_list);
            assert!(pair_cost(outcome.nprobe, outcome.pool, per_list) < fallback);
            let (nprobe, pool) = (outcome.nprobe as usize, outcome.pool as usize);
            let hits: usize = f
                .queries
                .iter()
                .map(|q| naive_hits(&f, q, None, nprobe, pool))
                .sum();
            let fresh = hits as f64 / (TOP_K * f.queries.len()) as f64;
            assert!(
                fresh >= 0.93,
                "{metric:?}: fresh queries {fresh}, sample {}",
                outcome.recall
            );
        }
    }

    #[test]
    fn exact_duplicates_count_as_neighbours_instead_of_making_the_target_unreachable() {
        let base = tk::mixture(&spec(16, 5), 20, 0, 0xD0).vectors;
        let vectors: Vec<Vec<f32>> = (0..600).map(|i| base[i % 20].clone()).collect();
        let f = fixture_of(vectors, Vec::new(), 12, Metric::L2);
        let outcome = calibrated(&f, 0.95, None, 2).expect("calibrated");
        assert!(outcome.reached && outcome.recall == 1.0, "{outcome:?}");
    }

    #[test]
    fn vectors_whose_square_overflows_do_not_break_the_calibration() {
        let mut vectors = tk::mixture(&spec(16, 5), 700, 0, 0x0F).vectors;
        for v in vectors.iter_mut().step_by(25) {
            v.iter_mut().for_each(|x| *x *= 1e20);
        }
        let f = fixture_of(vectors, Vec::new(), 20, Metric::L2);
        assert!(calibrated(&f, 0.95, None, 2).is_some());
    }

    /// A NaN with its sign bit set (x86's default NaN) or without (aarch64's).
    fn nan(negative: bool) -> f32 {
        f32::from_bits(if negative { 0xFFC0_0000 } else { 0x7FC0_0000 })
    }

    /// `total_cmp` puts a NaN with the sign bit set before every number: the truth must rank it
    /// last, as the scan does, or an overflowing row is the true neighbour of every sample query.
    #[test]
    fn the_truth_ranks_a_nan_after_every_number_whatever_its_sign() {
        let mut nearest = Nearest::default();
        let finite = |row: u32| (row as f32, row);
        let mut offers: Vec<(f32, u32)> = (0..40).map(finite).collect();
        offers.push((f32::NEG_INFINITY, 40));
        offers.push((f32::INFINITY, 41));
        offers.extend((42..45).map(|row| (nan(true), row)));
        offers.extend((45..48).map(|row| (nan(false), row)));
        let mut rng = tk::Rng::new(0x4A4);
        for round in 0..20 {
            // Any order of arrival gives the same answer.
            for i in (1..offers.len()).rev() {
                offers.swap(i, rng.below(i + 1));
            }
            nearest = Nearest::default();
            offers.iter().for_each(|&(d, row)| nearest.offer(d, row));
            let rows: Vec<u32> = nearest.kept.iter().map(|e| e.1).collect();
            let want: Vec<u32> = std::iter::once(40).chain(0..=30).collect();
            assert_eq!(rows, want, "round {round}");
            assert!(nearest.kept.iter().all(|e| !e.0.is_nan()), "round {round}");
        }
        // The tenth is the 8.0 of row 8: the infinity below it and rows 0..=8 are the neighbours.
        assert_eq!(nearest.neighbours(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 40]);

        // With fewer numbers than the cap the NaNs follow them, a negative one before a positive.
        let mut few = Nearest::default();
        for (d, row) in [
            (nan(false), 3),
            (1.0, 0),
            (nan(true), 2),
            (f32::INFINITY, 1),
        ] {
            few.offer(d, row);
        }
        assert_eq!(
            few.kept.iter().map(|e| e.1).collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
    }

    /// Rows whose distance to every query is a NaN of either sign (appended, so the other rows
    /// keep their numbers) are nobody's neighbours: the truth is what it is without them, for
    /// every metric. The NaN comes from the components, so its sign is the one built here (the
    /// inner product's key is the negated dot product, which flips it).
    #[test]
    fn rows_with_a_nan_distance_are_nobodys_neighbours_whatever_its_sign() {
        let clean = tk::mixture(&spec(16, 5), 400, 0, 0x6C).vectors;
        let clean_refs: Vec<&[f32]> = clean.iter().map(Vec::as_slice).collect();
        let sample = sample_rows(clean.len(), 40, 3);
        let mut wrong = Vec::new();
        for metric in [Metric::L2, Metric::Cosine, Metric::Ip] {
            let want = exact_neighbours(&clean_refs, metric, &sample, 2, &|| {});
            for negative in [false, true] {
                let component = nan(negative != (metric == Metric::Ip));
                let mut vectors = clean.clone();
                vectors.extend((0..12).map(|_| vec![component; 16]));
                let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
                let d = exact_distance(metric, &vectors[0], &vectors[clean.len()]);
                assert!(
                    d.is_nan() && d.is_sign_negative() == negative,
                    "{metric:?}: premise, a distance that is a NaN of the sign asked for: {d:?}"
                );
                for workers in [0, 3] {
                    let got = exact_neighbours(&refs, metric, &sample, workers, &|| {});
                    if got != want {
                        wrong.push(format!(
                            "{metric:?}, negative NaN {negative}, {workers} workers"
                        ));
                    }
                }
            }
        }
        assert!(wrong.is_empty(), "NaN rows became neighbours: {wrong:?}");
    }

    #[test]
    fn the_interrupt_hook_runs_between_phases() {
        let f = mixture_fixture(16, 800, 28, Metric::L2);
        let refs: Vec<&[f32]> = f.vectors.iter().map(Vec::as_slice).collect();
        let built = Built {
            metric: f.metric,
            seed: f.seed,
            kernel: Kernel::detect(),
            vectors: &refs,
            centroids: &f.centroids,
            lists: &f.lists,
            codes: &f.codes,
            rnorm: &f.rnorm,
            dot: &f.dot,
            rc: &f.rc,
        };
        let calls = std::cell::Cell::new(0u32);
        let settings = Settings {
            target: 0.95,
            pinned_pool: None,
            workers: 2,
        };
        calibrate(&built, &settings, &|| calls.set(calls.get() + 1)).expect("calibrated");
        assert!(calls.get() >= 2, "{} calls", calls.get());
    }

    fn exact_neighbours(
        vectors: &[&[f32]],
        metric: Metric,
        sample: &[u32],
        workers: usize,
        interrupt: &dyn Fn(),
    ) -> Vec<Vec<u32>> {
        let queries = sample_queries(vectors, sample);
        exact_neighbours_of(vectors, metric, &queries, workers, interrupt)
    }

    fn sample_query<'a>(refs: &[&'a [f32]], row: u32) -> Query<'a> {
        Query {
            vector: refs[row as usize],
            leave_out: Some(row),
        }
    }

    fn refs_of(f: &Fixture) -> Vec<&[f32]> {
        f.vectors.iter().map(Vec::as_slice).collect()
    }

    fn built_of<'a>(f: &'a Fixture, refs: &'a [&'a [f32]]) -> Built<'a> {
        Built {
            metric: f.metric,
            seed: f.seed,
            kernel: Kernel::detect(),
            vectors: refs,
            centroids: &f.centroids,
            lists: &f.lists,
            codes: &f.codes,
            rnorm: &f.rnorm,
            dot: &f.dot,
            rc: &f.rc,
        }
    }

    /// `f` with `row` removed and the rows after it moved up by one.
    fn without_row(f: &Fixture, row: u32) -> Fixture {
        let code_bytes = f.codes.len() / f.vectors.len();
        let at = row as usize;
        let mut codes = f.codes.clone();
        codes.drain(at * code_bytes..(at + 1) * code_bytes);
        let drop_row = |v: &Vec<f32>| {
            let mut v = v.clone();
            v.remove(at);
            v
        };
        let mut vectors = f.vectors.clone();
        vectors.remove(at);
        Fixture {
            vectors,
            queries: Vec::new(),
            centroids: f.centroids.clone(),
            lists: f
                .lists
                .iter()
                .map(|list| {
                    list.iter()
                        .filter(|&&r| r != row)
                        .map(|&r| if r > row { r - 1 } else { r })
                        .collect()
                })
                .collect(),
            codes,
            rnorm: drop_row(&f.rnorm),
            dot: drop_row(&f.dot),
            rc: drop_row(&f.rc),
            metric: f.metric,
            seed: f.seed,
        }
    }

    #[test]
    fn a_sample_row_is_neither_its_own_neighbour_nor_a_candidate() {
        for metric in [Metric::L2, Metric::Cosine, Metric::Ip] {
            let f = mixture_fixture(32, 1200, 35, metric);
            let refs = refs_of(&f);
            let built = built_of(&f, &refs);
            let sample = sample_rows(refs.len(), MIN_SAMPLES, f.seed);
            let truth = exact_neighbours(&refs, metric, &sample, 2, &|| {});
            let engine = Engine::new(&built);
            for (&row, truth) in sample.iter().zip(truth) {
                assert!(
                    !truth.contains(&row),
                    "{metric:?}: row {row} in its own truth"
                );
                assert!(truth.len() >= TOP_K, "{metric:?}: row {row}: {truth:?}");
                let mut scan = QueryScan::new(&engine, sample_query(&refs, row), truth);
                scan.hits(&engine, f.centroids.len(), &[300], 2_000);
                assert_eq!(
                    scan.best.len(),
                    refs.len() - 1,
                    "{metric:?}: every list probed, every other row a candidate"
                );
                assert!(
                    scan.best.iter().all(|&(_, r)| r != row),
                    "{metric:?}: row {row} is a candidate of its own search"
                );
            }
        }
    }

    #[test]
    fn a_row_left_out_reads_what_the_same_vector_reads_against_a_table_without_the_row() {
        for metric in [Metric::L2, Metric::Cosine, Metric::Ip] {
            let f = mixture_fixture(32, 1200, 35, metric);
            let refs = refs_of(&f);
            let built = built_of(&f, &refs);
            let engine = Engine::new(&built);
            let sample = sample_rows(refs.len(), MIN_SAMPLES, f.seed);
            let truth = exact_neighbours(&refs, metric, &sample, 2, &|| {});
            // pools of 10 and 12 notice a row that stays among the candidates: it takes a slot
            let pools = [10u32, 12, 100, 300];
            let probe_counts = [1usize, 3, 8, 20, 35];
            let mut imperfect = 0;
            for (&row, truth_in) in sample.iter().zip(truth) {
                let held_out = without_row(&f, row);
                let held_refs = refs_of(&held_out);
                let held_built = built_of(&held_out, &held_refs);
                let outside = Query {
                    vector: &f.vectors[row as usize],
                    leave_out: None,
                };
                let truth_out = exact_neighbours_of(&held_refs, metric, &[outside], 2, &|| {});
                let truth_out = truth_out.into_iter().next().expect("one query");
                let shifted: Vec<u32> = truth_in
                    .iter()
                    .map(|&r| if r > row { r - 1 } else { r })
                    .collect();
                assert_eq!(shifted, truth_out, "{metric:?}: truth of row {row}");

                let held_engine = Engine::new(&held_built);
                let mut left_out = QueryScan::new(&engine, sample_query(&refs, row), truth_in);
                let mut outside = QueryScan::new(&held_engine, outside, truth_out);
                for nprobe in probe_counts {
                    let a = left_out.hits(&engine, nprobe, &pools, 300);
                    let b = outside.hits(&held_engine, nprobe, &pools, 300);
                    assert_eq!(a, b, "{metric:?}: row {row}, {nprobe} lists");
                    imperfect += a.iter().filter(|&&h| usize::from(h) < TOP_K).count();
                }
            }
            assert!(
                imperfect > 10,
                "{metric:?}: premise: the settings differ in recall ({imperfect} misses)"
            );
        }
    }

    #[test]
    fn the_goal_is_the_target_plus_the_offset_and_never_more_than_halfway_to_perfect_recall() {
        assert_eq!(CALIB_OFFSET, 0.01);
        for (target, want) in [
            (0.5, 0.51),
            (0.8, 0.81),
            (0.95, 0.96),
            (0.98, 0.99),
            (0.99, 0.995),
            (0.999, 0.9995),
        ] {
            assert!(
                (goal(target) - want).abs() < 1e-12,
                "{target}: {}",
                goal(target)
            );
        }
        for thousandth in 500..=999 {
            let target = f64::from(thousandth) / 1000.0;
            assert!(goal(target) > target && goal(target) < 1.0, "{target}");
        }
        assert_eq!(
            Aim::of(0.95),
            Aim {
                goal: goal(0.95),
                floor: 0.95
            }
        );
    }

    #[test]
    fn a_pair_that_reaches_the_target_but_not_the_offset_makes_way_for_one_that_does() {
        let recall = |nprobe: u32, _pool: u32| match nprobe {
            8 => 940,
            12 => 960,
            _ => 975,
        };
        let levels = [8, 12, 16, 25];
        let (without, _) = run_search_for(plain(0.95), &levels, &[100], 1000.0, recall);
        assert_eq!((without.nprobe, without.reached), (12, true));
        let (chosen, scanned) = run_search_for(Aim::of(0.95), &levels, &[100], 1000.0, recall);
        assert_eq!(
            (chosen.nprobe, chosen.level, chosen.reached, chosen.held_to),
            (16, 2, true, Some(goal(0.95)))
        );
        assert_eq!(scanned, [8, 12, 16], "25 lists cannot cost less than 16");
        assert!(chosen.stats.lower_bound() >= goal(0.95));
    }

    #[test]
    fn with_no_pair_at_the_goal_the_cheapest_at_the_target_wins_after_every_level() {
        let recall = |nprobe: u32, pool: u32| match (nprobe, pool) {
            (8, _) => 940,
            (12, 100) => 960,
            _ => 962,
        };
        let levels = [8, 12, 16, 25];
        let (chosen, scanned) = run_search_for(Aim::of(0.95), &levels, &[100, 200], 1000.0, recall);
        assert_eq!(
            (chosen.nprobe, chosen.pool, chosen.reached, chosen.held_to),
            (12, 100, true, Some(0.95))
        );
        assert_eq!(chosen.level, 1);
        assert_eq!(
            scanned, levels,
            "no pair at the goal: no level may be skipped"
        );
    }

    #[test]
    fn a_target_nothing_reaches_stays_a_best_effort_with_the_offset_too() {
        let recall = |nprobe: u32, _pool: u32| if nprobe >= 16 { 900 } else { 800 };
        let (chosen, scanned) =
            run_search_for(Aim::of(0.95), &[8, 12, 16, 25], &[100], 1000.0, recall);
        assert_eq!(
            (chosen.nprobe, chosen.reached, chosen.held_to),
            (16, false, None)
        );
        assert_eq!(scanned.len(), 4);
    }

    #[test]
    fn refinement_finds_the_smallest_reaching_probe_count_of_a_bracket_in_at_most_four_probes() {
        let (reaching, missing) = (stats_at(980), stats_at(900));
        for (lo, hi) in [(8u32, 12u32), (12, 16), (16, 25), (25, 37), (37, 50)] {
            for first in lo + 1..=hi {
                let mut probed = Vec::new();
                let refined = refine(lo, hi, reaching, 0.95, |m| {
                    probed.push(m);
                    if m >= first { reaching } else { missing }
                });
                assert_eq!(
                    refined.nprobe, first,
                    "({lo}, {hi}], first reaching {first}: {probed:?}"
                );
                assert_eq!(refined.evaluations, probed.len());
                assert!(probed.len() <= REFINE_EVALUATIONS, "{probed:?}");
                assert!(probed.iter().all(|&m| lo < m && m < hi), "{probed:?}");
                let mut distinct = probed.clone();
                distinct.sort_unstable();
                distinct.dedup();
                assert_eq!(distinct.len(), probed.len(), "{probed:?}");
            }
        }
    }

    #[test]
    fn a_bracket_too_wide_for_four_probes_ends_within_a_sixteenth_of_its_width_of_the_smallest() {
        let (reaching, missing) = (stats_at(980), stats_at(900));
        for (lo, hi) in [(50u32, 75u32), (100, 150), (200, 300)] {
            for first in lo + 1..=hi {
                let refined = refine(lo, hi, reaching, 0.95, |m| {
                    if m >= first { reaching } else { missing }
                });
                assert!(
                    refined.nprobe >= first,
                    "({lo}, {hi}], first {first}: {refined:?}"
                );
                assert!(
                    refined.nprobe - first <= (hi - lo).div_ceil(16),
                    "({lo}, {hi}], first {first}: {refined:?}"
                );
                assert!(refined.evaluations <= REFINE_EVALUATIONS);
            }
        }
    }

    #[test]
    fn refinement_returns_a_count_it_saw_reach_even_when_recall_wobbles() {
        let (reaching, missing) = (stats_at(980), stats_at(900));
        let wobble = |m: u32| (m * 7 + 3) % 5 < 2;
        for (lo, hi) in [(8u32, 12u32), (16, 25), (25, 37)] {
            let mut seen_reaching = vec![hi];
            let refined = refine(lo, hi, reaching, 0.95, |m| {
                if wobble(m) {
                    seen_reaching.push(m);
                    reaching
                } else {
                    missing
                }
            });
            assert!(refined.evaluations >= 1, "({lo}, {hi}]: nothing probed");
            assert!(
                seen_reaching.contains(&refined.nprobe),
                "({lo}, {hi}]: {refined:?}"
            );
            assert_eq!(refined.nprobe, *seen_reaching.iter().min().unwrap());
        }
    }

    #[test]
    fn refinement_holds_a_probe_count_to_the_margin_and_not_to_the_mean() {
        let noisy = {
            let mut hits = vec![10u8; 88];
            hits.resize(100, 6);
            Stats::of_hits(&hits)
        };
        assert!(noisy.mean() >= 0.95 && noisy.lower_bound() < 0.95);
        let refined = refine(8, 12, stats_at(980), 0.95, |_| noisy);
        assert_eq!(refined.nprobe, 12, "no probe passed the margin");
        assert_eq!(refined.stats, stats_at(980));
        assert_eq!(refined.evaluations, 2, "10 and 11 were probed and rejected");
    }

    #[test]
    fn a_bracket_without_room_costs_no_probes() {
        let reaching = stats_at(980);
        for (lo, hi) in [(11u32, 12u32), (12, 12), (13, 12)] {
            let refined = refine(lo, hi, reaching, 0.95, |m| panic!("probed {m}"));
            assert_eq!((refined.nprobe, refined.evaluations), (hi, 0));
        }
    }

    #[test]
    fn a_first_level_that_reaches_the_goal_is_kept_without_probes() {
        let f = mixture_fixture(32, 1500, 40, Metric::L2);
        let outcome = calibrated(&f, 0.5, None, 2).expect("calibrated");
        assert_eq!(
            (
                outcome.nprobe,
                outcome.refine_probes,
                outcome.levels_scanned
            ),
            (8, 0, 1),
            "{outcome:?}"
        );
    }

    #[test]
    fn the_calibrated_probe_count_is_the_smallest_in_its_bracket_that_a_naive_scan_reaches() {
        // weak clusters: recall climbs slowly with the probe count, so the grid steps cross the bar
        let weak = MixtureSpec {
            noise_norm: 2.0,
            ..spec(32, 40)
        };
        let data = tk::mixture(&weak, 1_200, 0, 0x5CA7);
        let f = fixture_of(data.vectors, Vec::new(), 35, Metric::L2);
        let sample = sample_rows(f.vectors.len(), sample_size(f.vectors.len()), f.seed);
        let naive = |nprobe: usize| -> Stats {
            let hits: Vec<u8> = sample
                .iter()
                .map(|&s| naive_hits(&f, &f.vectors[s as usize], Some(s), nprobe, 100) as u8)
                .collect();
            Stats::of_hits(&hits)
        };
        let levels = nprobe_levels(35);
        let bound: Vec<f64> = (0..=35)
            .map(|m| if m == 0 { 0.0 } else { naive(m).lower_bound() })
            .collect();
        // every grid step wide enough to bisect that gains at least 0.02 and rises steadily: a bar
        // in the middle of the gain is crossed inside the step, and no earlier level reaches it
        let mut brackets = 0;
        for pair in levels.windows(2) {
            let (lo, hi) = (pair[0] as usize, pair[1] as usize);
            let bar = (bound[lo] + bound[hi]) / 2.0;
            let rises = (lo..hi).all(|m| bound[m] <= bound[m + 1]);
            if hi - lo < 3 || bound[hi] - bound[lo] < 0.02 || bar > 0.98 || !rises {
                continue;
            }
            let smallest = (lo + 1..=hi)
                .find(|&m| bound[m] >= bar)
                .expect("hi reaches");
            let outcome = calibrated(&f, bar - CALIB_OFFSET, None, 2).expect("calibrated");
            assert_eq!(
                (outcome.nprobe as usize, outcome.pool, outcome.reached),
                (smallest, 100, true),
                "bracket ({lo}, {hi}], bar {bar}: {outcome:?}"
            );
            assert!(
                (usize::from(smallest < hi)..=REFINE_EVALUATIONS).contains(&outcome.refine_probes),
                "bracket ({lo}, {hi}]: {outcome:?}"
            );
            assert_eq!(outcome.recall, naive(smallest).mean());
            brackets += 1;
        }
        assert!(brackets >= 2, "premise: {brackets} brackets in {bound:?}");
    }

    #[test]
    fn a_calibrated_pair_reaches_the_target_plus_the_offset_on_the_sample() {
        for metric in [Metric::L2, Metric::Cosine] {
            let f = mixture_fixture(32, 2000, 45, metric);
            let outcome = calibrated(&f, 0.95, None, 2).expect("calibrated");
            let sample = sample_rows(f.vectors.len(), outcome.samples as usize, f.seed);
            let (nprobe, pool) = (outcome.nprobe as usize, outcome.pool as usize);
            let hits: Vec<u8> = sample
                .iter()
                .map(|&s| naive_hits(&f, &f.vectors[s as usize], Some(s), nprobe, pool) as u8)
                .collect();
            let stats = Stats::of_hits(&hits);
            assert!(
                outcome.reached && stats.lower_bound() >= 0.95 + CALIB_OFFSET,
                "{metric:?}: {outcome:?}, bound {}",
                stats.lower_bound()
            );
            assert_eq!(stats.mean(), outcome.recall);
        }
    }

    // ---- calibrating on the user's own queries ----

    fn calibrated_on(
        f: &Fixture,
        queries: &[Vec<f32>],
        target: f64,
        pinned_pool: Option<u32>,
        workers: usize,
    ) -> Option<Outcome> {
        let refs = refs_of(f);
        let built = built_of(f, &refs);
        let queries: Vec<&[f32]> = queries.iter().map(Vec::as_slice).collect();
        calibrate_on(
            &built,
            &Settings {
                target,
                pinned_pool,
                workers,
            },
            &queries,
            &|| {},
        )
    }

    /// Hits of `queries` at a pair, none of them left out: what a fresh query set would read.
    fn fresh_recall(f: &Fixture, queries: &[Vec<f32>], nprobe: u32, pool: u32) -> f64 {
        let hits: usize = queries
            .iter()
            .map(|q| naive_hits(f, q, None, nprobe as usize, pool as usize))
            .sum();
        hits as f64 / (TOP_K * queries.len()) as f64
    }

    #[test]
    fn external_queries_have_the_exact_top_ten_of_the_whole_table_as_truth() {
        let mut rng = tk::Rng::new(11);
        for (rows, dims, values) in [(700usize, 5usize, 3usize), (600, 12, 40)] {
            let vectors: Vec<Vec<f32>> = (0..rows)
                .map(|_| (0..dims).map(|_| rng.below(values) as f32).collect())
                .collect();
            let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
            let mut queries: Vec<Vec<f32>> = (0..40)
                .map(|_| (0..dims).map(|_| rng.below(values) as f32 + 0.5).collect())
                .collect();
            queries.extend(vectors[..10].iter().cloned());
            for metric in [Metric::L2, Metric::Cosine, Metric::Ip] {
                let want: Vec<Vec<u32>> = queries
                    .iter()
                    .map(|q| brute_neighbours(&vectors, metric, q, None))
                    .collect();
                let vector_refs: Vec<&[f32]> = queries.iter().map(Vec::as_slice).collect();
                let external = external_queries(&vector_refs);
                assert!(external.iter().all(|q| q.leave_out.is_none()));
                for workers in [0, 1, 4] {
                    let got = exact_neighbours_of(&refs, metric, &external, workers, &|| {});
                    assert_eq!(got, want, "{metric:?}, {rows} rows, {workers} workers");
                }
                if metric == Metric::L2 {
                    for (row, truth) in want[40..].iter().enumerate() {
                        assert!(
                            truth.contains(&(row as u32)),
                            "a query equal to row {row} finds it: {truth:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn calibrating_on_queries_is_what_a_naive_scan_of_those_queries_gets() {
        for metric in [Metric::L2, Metric::Cosine, Metric::Ip] {
            let f = mixture_fixture(32, 1500, 40, metric);
            // Fresh draws plus copies of table rows: a row-copy is its own nearest neighbour here.
            let mut queries: Vec<Vec<f32>> = f.queries.clone();
            queries.extend(f.vectors[..20].iter().cloned());
            let outcome = calibrated_on(&f, &queries, 0.95, None, 3).expect("calibrated");
            assert_eq!(outcome.samples as usize, queries.len());
            let hits: usize = queries
                .iter()
                .map(|q| naive_hits(&f, q, None, outcome.nprobe as usize, outcome.pool as usize))
                .sum();
            assert_eq!(
                outcome.recall,
                hits as f64 / (TOP_K * queries.len()) as f64,
                "{metric:?}: {outcome:?}"
            );
        }
    }

    #[test]
    fn calibrating_on_queries_gives_one_outcome_for_any_worker_count() {
        let f = mixture_fixture(32, 1200, 30, Metric::L2);
        let first = calibrated_on(&f, &f.queries, 0.95, None, 0).expect("calibrated");
        for workers in [1, 2, 5, 16] {
            assert_eq!(
                calibrated_on(&f, &f.queries, 0.95, None, workers),
                Some(first),
                "{workers} workers"
            );
        }
    }

    #[test]
    fn a_table_below_the_minimum_or_no_queries_is_not_calibrated_on_queries() {
        let small = mixture_fixture(16, MIN_ROWS - 1, 8, Metric::L2);
        assert_eq!(calibrated_on(&small, &small.queries, 0.95, None, 2), None);
        let f = mixture_fixture(16, MIN_ROWS, 8, Metric::L2);
        assert_eq!(calibrated_on(&f, &[], 0.95, None, 2), None);
        assert!(calibrated_on(&f, &f.queries, 0.95, None, 2).is_some());
    }

    #[test]
    fn the_query_sample_is_up_to_200_within_the_exact_pass_budget_and_never_below_50() {
        for (rows, want) in [
            (0, 200),
            (500, 200),
            (10_000, 200),
            (1_000_000, 200),
            (1_000_001, 199),
            (2_000_000, 100),
            (4_000_000, 50),
            (4_000_001, 50),
            (50_000_000, 50),
        ] {
            assert_eq!(query_sample_size(rows), want, "{rows} rows");
        }
        assert!(
            (500..5_000_000)
                .step_by(997)
                .all(|rows| query_sample_size(rows) >= sample_size(rows))
        );
        assert_eq!(
            [500, 10_000, 1_000_000].map(sample_size),
            [50, 200, 200],
            "the row sample's size is unchanged"
        );
    }

    /// Queries that are not like the rows (here every query is a fresh draw moved by one fixed
    /// offset, the way a question embedding sits apart from the passages it retrieves) are what the
    /// table rows cannot stand in for: the rows read 0.98 and fresh shifted queries 0.89.
    #[test]
    fn queries_unlike_the_rows_make_calibration_probe_more_lists_and_reach_the_target() {
        let (dim, rows, lists) = (32, 4_000, 100);
        let mut sp = spec(dim, 20);
        sp.noise_norm = 1.0;
        let data = tk::mixture(&sp, rows, 600, 1);
        let offset: Vec<f32> = tk::Rng::new(0x5416)
            .unit(dim)
            .iter()
            .map(|v| (v * 3.0) as f32)
            .collect();
        let shifted: Vec<Vec<f32>> = data
            .queries
            .iter()
            .map(|q| q.iter().zip(&offset).map(|(a, b)| a + b).collect())
            .collect();
        let (calibration, held_out) = shifted.split_at(200);
        let f = fixture_of(data.vectors, Vec::new(), lists, Metric::L2);

        let from_rows = calibrated(&f, 0.95, None, 3).expect("calibrated on rows");
        let from_queries = calibrated_on(&f, calibration, 0.95, None, 3).expect("calibrated");
        let fresh_rows = fresh_recall(&f, held_out, from_rows.nprobe, from_rows.pool);
        let fresh_queries = fresh_recall(&f, held_out, from_queries.nprobe, from_queries.pool);

        assert!(from_rows.reached && from_rows.recall >= 0.95 + CALIB_OFFSET);
        assert!(
            fresh_rows < 0.93,
            "premise: the row sample's pair {from_rows:?} reads {fresh_rows:.3} on shifted queries"
        );
        assert!(
            from_queries.nprobe > from_rows.nprobe,
            "rows {from_rows:?}, queries {from_queries:?}"
        );
        assert!(from_queries.reached && from_queries.recall >= 0.95 + CALIB_OFFSET);
        // 0.964 here; the bound leaves room for another CPU's kernels ranking a tie differently.
        assert!(
            fresh_queries >= 0.945 && fresh_queries > fresh_rows + 0.04,
            "{from_queries:?} reads {fresh_queries:.3} on held-out shifted queries ({fresh_rows:.3} for rows)"
        );
    }
}
