//! Sample-based k-means for the local engine's IVF partitioning. Defaults
//! cited to Jégou et al.'s IVF (as popularized by Faiss's
//! `IndexIVFFlat`/`Clustering` defaults):
//! - sample size = `min(100_000, N)` training points.
//! - `n_lists` default = `ceil(sqrt(N))` (Faiss's own rule of thumb — enough
//!   lists that each holds roughly `sqrt(N)` vectors, balancing per-list scan
//!   cost against the number of lists a query must probe).
//! - 10 Lloyd iterations, fixed (Faiss's own default `niter` is higher, but
//!   this build fixes a smaller constant — revisit as a tuning knob if
//!   under-converged clusters show up in recall numbers).
//!
//! Centroid initialization uses a plain random subset of the sample (not
//! k-means++): the simplest correct choice; it's still a standard, citable
//! Lloyd's-algorithm initialization (see e.g. Faiss's own `Clustering` with
//! `min_points_per_centroid` un-tuned); k-means++
//! is a reasonable follow-up if convergence quality becomes a limiting factor.
//!
//! For the cosine metric, centroids are re-normalized to unit length after
//! every update (standard "spherical k-means" — Dhillon & Modha 2001) since
//! plain arithmetic-mean centroids drift toward the origin as points spread
//! angularly, which would otherwise make the *magnitude* of a centroid
//! carry no information for a metric that's defined to ignore magnitude
//! entirely.

use std::thread;

use super::{
    metric::{Metric, exact_distance},
    rng::SplitMix64,
};

pub(crate) const MAX_SAMPLE_SIZE: usize = 100_000;
pub(crate) const MAX_ITERATIONS: usize = 10;

/// Fixed chunk size for the update phase's parallel reduction (`update_
/// step`) — a SIZE, not a per-run chunk COUNT, and independent of
/// `parallelism` entirely. This is the crux of the parbuild determinism
/// contract (`options::build_parallelism`'s own rustdoc; the byte-identical-
/// index test in `lib.rs`): floating-point addition isn't associative, so
/// summing the same points in a different grouping produces a different
/// (still correct, but bit-different) result. If the number/boundaries of
/// chunks depended on `parallelism`, a `parallelism=0` build (one giant
/// "chunk") and a `parallelism=4` build (four chunks) would sum in
/// different orders and diverge in their last few float bits — silently
/// breaking "same (seed, data) => same centroids regardless of worker
/// count". Keeping the chunk grid fixed by SIZE means `parallelism` only
/// ever changes how many OS threads pull from an identical, pre-determined
/// list of chunks — never which points land in which chunk, nor the order
/// chunk results are combined in (always ascending chunk index — see
/// `update_step`). 2048 is large enough that per-chunk thread-dispatch
/// overhead is negligible next to its `O(chunk_len * dim)` accumulation
/// work, while still yielding ~48 chunks at `MAX_SAMPLE_SIZE` — comfortably
/// more than any sane `build_parallelism` setting, so load-balancing has
/// headroom.
const UPDATE_CHUNK_SIZE: usize = 2048;

/// `ceil(sqrt(n))`, `n_lists`'s default per this module's docstring — always
/// at least 1 (an `n=0` build never reaches k-means at all; see `build.rs`'s
/// small-N bootstrap, which this guards defensively regardless).
pub(crate) fn default_n_lists(n: usize) -> u32 {
    ((n as f64).sqrt().ceil() as u32).max(1)
}

pub(crate) struct KMeans {
    pub(crate) centroids: Vec<Vec<f32>>,
}

/// `n_lists` (`u32`, straight from the `n_lists` reloption or `default_n_
/// lists`) clamped against `sample_size` (`vectors.len().min(MAX_SAMPLE_
/// SIZE)`), NOT `vectors.len()` — initialization below
/// picks `n_lists` DISTINCT points out of `sample_idx[..n_lists]`, which only
/// has `sample_size` entries. Clamping against `vectors.len()` alone (the
/// old code) let `n_lists` exceed `sample_idx.len()` whenever `vectors.len()`
/// exceeded `MAX_SAMPLE_SIZE` and the requested/derived `n_lists` fell
/// between the two — e.g. an explicit `n_lists` reloption above
/// `MAX_SAMPLE_SIZE` on a table with over 100k rows — panicking on the
/// out-of-bounds slice in [`KMeans::train`] instead of training a (still
/// statistically valid — fewer lists is always a legal Lloyd's-algorithm
/// input) index. A free function so this arithmetic is unit-testable at the
/// sizes that actually trigger it (k-means at over 100k points is far too
/// slow to run in a test).
fn clamp_n_lists(requested: u32, sample_size: usize) -> usize {
    (requested as usize).clamp(1, sample_size)
}

impl KMeans {
    /// Trains `n_lists` centroids over (at most [`MAX_SAMPLE_SIZE`] of)
    /// `vectors`, each of length `dim`, using `metric` for assignment.
    /// `n_lists` is clamped per [`clamp_n_lists`] — callers needing a
    /// small-N exact bootstrap instead of clustering at all should never
    /// reach this (see `build.rs`'s `small_n_threshold` gate).
    ///
    /// Generic over `V: AsRef<[f32]>` rather than requiring `&[Vec<f32>]`
    /// specifically: `build::ambuild_local` used to `.clone()` its entire
    /// `collected` row set into a fresh `Vec<Vec<f32>>` just to satisfy this
    /// signature — an O(N * dim) copy of the whole table on every build,
    /// wholly wasted, since [`KMeans::train`] never needed to OWN the
    /// vectors, only to borrow them. Its caller now passes a
    /// `&[&[f32]]` of borrows straight into `collected` instead — `&[f32]`
    /// trivially satisfies `AsRef<[f32]>` too, so `train`'s own tests
    /// (`&[Vec<f32>]`) keep working unchanged. Also requires `V: Sync` (new
    /// with `parallelism`): the assign/update phases below share `vectors`
    /// read-only across scoped worker threads (`options::build_parallelism`'s
    /// own rustdoc) — trivially satisfied by both `&[f32]` and `Vec<f32>`.
    ///
    /// `parallelism` (`0` = strictly serial; `N` = up to `N` scoped
    /// `std::thread` workers — see [`assign_step`]/[`update_step`]) is a
    /// pure CPU-budget knob: it must NEVER change the trained centroids for
    /// a fixed `(seed, vectors, dim, n_lists, metric)` — see
    /// `UPDATE_CHUNK_SIZE`'s own rustdoc for the mechanism that guarantees
    /// this (fixed-size chunking + fixed chunk-order reduction, independent
    /// of `parallelism` itself). Every Postgres call (`palloc`/`elog`/SPI)
    /// stays on the CALLER's thread; nothing in [`assign_step`]/
    /// [`update_step`]'s worker closures ever touches `pg_sys` — required
    /// for spawning bare OS threads safely from inside a Postgres backend
    /// (see `build.rs`'s module docs for why that's the hard constraint).
    pub(crate) fn train<V: AsRef<[f32]> + Sync>(
        vectors: &[V],
        dim: usize,
        n_lists: u32,
        metric: Metric,
        seed: u64,
        parallelism: usize,
    ) -> Self {
        assert!(
            !vectors.is_empty(),
            "pg_pinecone: k-means needs at least one vector"
        );
        // Sample (without replacement, via a partial Fisher-Yates over
        // indices) at most MAX_SAMPLE_SIZE training points.
        let sample_size = vectors.len().min(MAX_SAMPLE_SIZE);
        let mut indices: Vec<usize> = (0..vectors.len()).collect();
        // `for_domain`, not `new` — see `rng::domain`'s own rustdoc: this
        // shares the metapage's single `seed` field with `Projector::new`
        // and must not draw the identical sequence it would.
        let mut rng = SplitMix64::for_domain(seed, super::rng::domain::KMEANS);
        partial_shuffle(&mut indices, sample_size, &mut rng);
        let sample_idx = &indices[..sample_size];

        let n_lists = clamp_n_lists(n_lists, sample_size);

        // Random-subset initialization: pick n_lists distinct sample points.
        // (These `n_lists` — O(sqrt(N)) — copies are the only ones this
        // function makes; seeding centroids necessarily needs owned storage
        // to mutate in place below.)
        let mut centroids: Vec<Vec<f32>> = sample_idx[..n_lists]
            .iter()
            .map(|&i| vectors[i].as_ref().to_vec())
            .collect();
        if metric == Metric::Cosine {
            for c in &mut centroids {
                normalize_in_place(c);
            }
        }

        let mut assignment = vec![0usize; sample_size];
        for _ in 0..MAX_ITERATIONS {
            assign_step(
                &mut assignment,
                sample_idx,
                vectors,
                &centroids,
                metric.partition(),
                parallelism,
            );

            // Update: mean of assigned points per cluster — the
            // floating-point-sensitive reduction; see `update_step`'s own
            // rustdoc for the determinism mechanism.
            let (sums, counts) =
                update_step(&assignment, sample_idx, vectors, n_lists, dim, parallelism);
            for (c, (sum, &count)) in centroids.iter_mut().zip(sums.iter().zip(&counts)) {
                if count == 0 {
                    // Empty cluster: reinitialize to a fresh random sample
                    // point rather than leaving a dead, never-probeable
                    // centroid (standard Lloyd's-algorithm degeneracy fix).
                    // Always sequential, on the caller's own thread — this
                    // reads the RNG's single, already-serial stream and the
                    // just-merged (parallelism-invariant) `counts`, so it
                    // stays deterministic regardless of `parallelism` too.
                    let idx = sample_idx[rng.next_bounded(sample_size as u32) as usize];
                    c.copy_from_slice(vectors[idx].as_ref());
                } else {
                    for (ci, &s) in c.iter_mut().zip(sum) {
                        *ci = s / count as f32;
                    }
                }
                if metric == Metric::Cosine {
                    normalize_in_place(c);
                }
            }
        }

        KMeans { centroids }
    }

    /// The index of `vectors`'s nearest centroid, by `metric`'s [`Metric::partition`].
    pub(crate) fn assign(&self, vector: &[f32], metric: Metric) -> usize {
        nearest_centroid(vector, &self.centroids, metric.partition())
    }
}

fn nearest_centroid(v: &[f32], centroids: &[Vec<f32>], metric: Metric) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, exact_distance(metric, v, c)))
        // `total_cmp`, never `partial_cmp().expect` (the build-path
        // completion of the scan-path fix): a
        // large-magnitude — still finite, pgvector-accepted — vector can
        // overflow to Inf inside the distance arithmetic and produce NaN
        // (e.g. Inf - Inf in squared-L2), and a panic here aborts the whole
        // CREATE INDEX. IEEE totalOrder sorts NaN last, so a poisoned
        // distance never wins the assignment and never errors the build.
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .expect("pg_pinecone: k-means centroids must be non-empty")
        .0
}

fn normalize_in_place(v: &mut [f32]) {
    let norm = super::metric::l2_norm(v);
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Shuffles just enough of `indices` (via partial Fisher–Yates) that its
/// first `k` elements are a uniformly random `k`-subset in random order —
/// avoids paying for a full shuffle when `k` is far smaller than
/// `indices.len()` (`N` up to a full production table between builds).
fn partial_shuffle(indices: &mut [usize], k: usize, rng: &mut SplitMix64) {
    let n = indices.len();
    let k = k.min(n);
    for i in 0..k {
        let j = i + rng.next_bounded((n - i) as u32) as usize;
        indices.swap(i, j);
    }
}

/// One Lloyd's-algorithm iteration's assign phase: `assignment[i] =
/// nearest_centroid(vectors[sample_idx[i]], centroids)` for every `i`. Safe
/// to run in parallel with NO determinism precaution beyond "each output
/// slot is independent": unlike [`update_step`], nothing here is summed
/// across points — every `assignment[i]` is a pure function of exactly one
/// input vector and the (shared, read-only) `centroids` snapshot, so
/// however the work is chunked/scheduled across threads, each slot's value
/// is identical to what a single-threaded pass would have written there.
/// `parallelism <= 1` (including the `0` "serial fallback" case) skips
/// `thread::scope` entirely — not just an optimization: it's what makes the
/// serial path trivially, definitionally consistent with any parallel one
/// (same loop body, same order, just not handed to worker threads).
fn assign_step<V: AsRef<[f32]> + Sync>(
    assignment: &mut [usize],
    sample_idx: &[usize],
    vectors: &[V],
    centroids: &[Vec<f32>],
    metric: Metric,
    parallelism: usize,
) {
    if parallelism <= 1 {
        for (slot, &vi) in assignment.iter_mut().zip(sample_idx) {
            *slot = nearest_centroid(vectors[vi].as_ref(), centroids, metric);
        }
        return;
    }
    let workers = parallelism.min(assignment.len()).max(1);
    let chunk_len = assignment.len().div_ceil(workers).max(1);
    thread::scope(|scope| {
        for (a_chunk, i_chunk) in assignment
            .chunks_mut(chunk_len)
            .zip(sample_idx.chunks(chunk_len))
        {
            let vectors = &vectors;
            let centroids = &centroids;
            scope.spawn(move || {
                for (slot, &vi) in a_chunk.iter_mut().zip(i_chunk) {
                    *slot = nearest_centroid(vectors[vi].as_ref(), centroids, metric);
                }
            });
        }
    });
}

/// One fixed-size chunk's local partial sums/counts for the update phase —
/// see [`update_step`]'s own rustdoc for why the chunk this was computed
/// over must be a fixed-SIZE slice (`UPDATE_CHUNK_SIZE`), never a
/// per-`parallelism` fraction of the data.
struct ChunkAccum {
    sums: Vec<Vec<f32>>,
    counts: Vec<u32>,
}

fn accumulate_chunk<V: AsRef<[f32]>>(
    assignment_chunk: &[usize],
    sample_idx_chunk: &[usize],
    vectors: &[V],
    n_lists: usize,
    dim: usize,
) -> ChunkAccum {
    let mut sums = vec![vec![0f32; dim]; n_lists];
    let mut counts = vec![0u32; n_lists];
    for (&cluster, &vi) in assignment_chunk.iter().zip(sample_idx_chunk) {
        counts[cluster] += 1;
        for (s, x) in sums[cluster].iter_mut().zip(vectors[vi].as_ref()) {
            *s += x;
        }
    }
    ChunkAccum { sums, counts }
}

/// One Lloyd's-algorithm iteration's update phase: per-cluster sum/count of
/// every assigned point, mean-ready for the caller to divide through.
///
/// THE determinism-sensitive half of this module (`options::build_
/// parallelism`'s own rustdoc; `UPDATE_CHUNK_SIZE`'s rustdoc for the
/// mechanism in full): floating-point addition isn't associative, so the
/// grouping in which per-point contributions are summed can change the
/// result's last few bits. This function fixes that grouping independent of
/// `parallelism` — `assignment`/`sample_idx` are always split into
/// `UPDATE_CHUNK_SIZE`-sized chunks (a SIZE, so the chunk grid's shape is a
/// pure function of `assignment.len()`, never of `parallelism`), each
/// chunk's partial sums/counts are computed in isolation
/// ([`accumulate_chunk`]), and the chunks are always combined in ASCENDING
/// chunk-index order below — regardless of how many threads computed them
/// or in what order those threads happened to finish. `parallelism` only
/// ever changes which thread computes which chunk (round-robin over
/// contiguous chunk-index GROUPS, joined back in the same ascending
/// order), never the chunk boundaries or the final combine order — so
/// `parallelism = 0, 1, 2, 4, ...` all produce byte-identical `(sums,
/// counts)` for the same `(assignment, sample_idx, vectors)`.
fn update_step<V: AsRef<[f32]> + Sync>(
    assignment: &[usize],
    sample_idx: &[usize],
    vectors: &[V],
    n_lists: usize,
    dim: usize,
    parallelism: usize,
) -> (Vec<Vec<f32>>, Vec<u32>) {
    let a_chunks: Vec<&[usize]> = assignment.chunks(UPDATE_CHUNK_SIZE).collect();
    let i_chunks: Vec<&[usize]> = sample_idx.chunks(UPDATE_CHUNK_SIZE).collect();
    let num_chunks = a_chunks.len();

    let chunk_accums: Vec<ChunkAccum> = if parallelism <= 1 || num_chunks <= 1 {
        a_chunks
            .iter()
            .zip(&i_chunks)
            .map(|(&ac, &ic)| accumulate_chunk(ac, ic, vectors, n_lists, dim))
            .collect()
    } else {
        let workers = parallelism.min(num_chunks);
        let group_size = num_chunks.div_ceil(workers);
        thread::scope(|scope| {
            let handles: Vec<_> = (0..num_chunks)
                .step_by(group_size)
                .map(|start| {
                    let end = (start + group_size).min(num_chunks);
                    let a_chunks = &a_chunks;
                    let i_chunks = &i_chunks;
                    scope.spawn(move || {
                        (start..end)
                            .map(|idx| {
                                accumulate_chunk(
                                    a_chunks[idx],
                                    i_chunks[idx],
                                    vectors,
                                    n_lists,
                                    dim,
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            // Ascending group order (the `.map` above preserves the
            // `step_by` iteration order it was built from) — the fixed
            // combine order this whole function exists to guarantee.
            handles
                .into_iter()
                .flat_map(|h| {
                    h.join()
                        .expect("pg_pinecone: k-means update worker thread panicked")
                })
                .collect()
        })
    };

    let mut sums = vec![vec![0f32; dim]; n_lists];
    let mut counts = vec![0u32; n_lists];
    for chunk in &chunk_accums {
        for c in 0..n_lists {
            counts[c] += chunk.counts[c];
            for (s, x) in sums[c].iter_mut().zip(&chunk.sums[c]) {
                *s += x;
            }
        }
    }
    (sums, counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clustered_data(seed: u64) -> Vec<Vec<f32>> {
        let mut rng = SplitMix64::new(seed);
        let centers = [[0.0f32, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]];
        let mut data = vec![];
        for center in centers {
            for _ in 0..50 {
                let jitter = |r: &mut SplitMix64| r.next_f32() * 4.0 - 2.0;
                data.push(vec![
                    center[0] + jitter(&mut rng),
                    center[1] + jitter(&mut rng),
                ]);
            }
        }
        data
    }

    #[test]
    fn default_n_lists_matches_ceil_sqrt() {
        assert_eq!(default_n_lists(100), 10);
        assert_eq!(default_n_lists(10_000), 100);
        assert_eq!(default_n_lists(0), 1);
        assert_eq!(default_n_lists(2), 2); // ceil(sqrt(2)) = 2
    }

    #[test]
    fn recovers_well_separated_clusters() {
        let data = make_clustered_data(1);
        let km = KMeans::train(&data, 2, 4, Metric::L2, 7, 0);
        assert_eq!(km.centroids.len(), 4);

        // Every trained centroid should land near exactly one of the four
        // true centers (within jitter bound), and all four true centers
        // should be covered (no duplicate/collapsed centroids).
        let true_centers = [[0.0f32, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]];
        let mut covered = vec![false; 4];
        for c in &km.centroids {
            let (closest, dist) = true_centers
                .iter()
                .enumerate()
                .map(|(i, t)| (i, exact_distance(Metric::L2, c, t)))
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            assert!(
                dist < 5.0,
                "centroid {c:?} not near any true center (dist={dist})"
            );
            covered[closest] = true;
        }
        assert!(
            covered.iter().all(|&c| c),
            "not all clusters were recovered: {covered:?}"
        );
    }

    #[test]
    fn every_vector_gets_assigned_to_some_cluster() {
        let data = make_clustered_data(2);
        let km = KMeans::train(&data, 2, 4, Metric::L2, 3, 0);
        for v in &data {
            let c = km.assign(v, Metric::L2);
            assert!(c < 4);
        }
    }

    #[test]
    fn inner_product_partitions_offset_data_evenly() {
        let data: Vec<Vec<f32>> = make_clustered_data(4)
            .into_iter()
            .map(|v| v.iter().map(|x| x + 1000.0).collect())
            .collect();
        let km = KMeans::train(&data, 2, 4, Metric::Ip, 11, 0);
        let mut counts = [0usize; 4];
        for v in &data {
            counts[km.assign(v, Metric::Ip)] += 1;
        }
        assert_eq!(counts, [50; 4], "one list per blob");
    }

    #[test]
    fn cosine_centroids_are_unit_norm() {
        let data = make_clustered_data(5);
        let km = KMeans::train(&data, 2, 4, Metric::Cosine, 9, 0);
        for c in &km.centroids {
            let n = super::super::metric::l2_norm(c);
            assert!((n - 1.0).abs() < 1e-4, "centroid norm = {n}");
        }
    }

    #[test]
    fn n_lists_clamped_to_vector_count() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let km = KMeans::train(&data, 2, 100, Metric::L2, 1, 0);
        assert_eq!(km.centroids.len(), 2);
    }

    /// The bug this replaced, at the scale that actually
    /// triggers it: `vectors.len() > MAX_SAMPLE_SIZE` (so `sample_size <
    /// vectors.len()`) with a requested `n_lists` in between the two —
    /// running `KMeans::train` itself at this scale is far too slow for a
    /// test, so this exercises the clamp arithmetic directly instead (see
    /// `clamp_n_lists`'s own rustdoc).
    #[test]
    fn clamp_n_lists_bounds_to_sample_size_not_vector_count() {
        assert_eq!(clamp_n_lists(500_000, MAX_SAMPLE_SIZE), MAX_SAMPLE_SIZE);
        assert_eq!(clamp_n_lists(50, MAX_SAMPLE_SIZE), 50);
        assert_eq!(clamp_n_lists(0, MAX_SAMPLE_SIZE), 1);
    }

    #[test]
    fn deterministic_given_same_seed() {
        let data = make_clustered_data(11);
        let km1 = KMeans::train(&data, 2, 4, Metric::L2, 42, 0);
        let km2 = KMeans::train(&data, 2, 4, Metric::L2, 42, 0);
        assert_eq!(km1.centroids, km2.centroids);
    }

    /// A larger fixture than [`make_clustered_data`] — big enough (well
    /// past [`UPDATE_CHUNK_SIZE`]'s 2048) that `update_step` actually
    /// exercises multiple chunks and a real combine, not the trivial
    /// single-chunk case every other test in this file hits by construction
    /// (all tiny fixtures). Used only by the parallelism-determinism tests
    /// below — a plain, fast `#[test]` (no `#[pg_test]`/Postgres needed),
    /// the cheap first line of defense before `lib.rs`'s full end-to-end,
    /// byte-identical-index proof (`am_local_build_parallelism_is_byte_
    /// identical_to_serial`).
    fn make_larger_clustered_data(seed: u64) -> Vec<Vec<f32>> {
        let mut rng = SplitMix64::new(seed);
        let n_clusters = 16;
        let centers: Vec<[f32; 3]> = (0..n_clusters)
            .map(|_| {
                [
                    rng.next_f32() * 200.0 - 100.0,
                    rng.next_f32() * 200.0 - 100.0,
                    rng.next_f32() * 200.0 - 100.0,
                ]
            })
            .collect();
        (0..6_000)
            .map(|i| {
                let c = &centers[i % n_clusters];
                vec![
                    c[0] + rng.next_f32() * 2.0 - 1.0,
                    c[1] + rng.next_f32() * 2.0 - 1.0,
                    c[2] + rng.next_f32() * 2.0 - 1.0,
                ]
            })
            .collect()
    }

    /// **The core parbuild determinism contract, at the `KMeans::train`
    /// level** (`lib.rs`'s `am_local_build_parallelism_is_byte_identical_
    /// to_serial` re-proves this end-to-end through a real index build):
    /// `parallelism` must never change the trained centroids for the same
    /// `(seed, data)`. Compares serial (`0`) against several worker counts,
    /// including ones that don't evenly divide the chunk count (`3`) and
    /// one exceeding it entirely (`64` — `update_step`/`assign_step` must
    /// clamp, never panic on an over-large `parallelism`).
    #[test]
    fn parallelism_never_changes_trained_centroids() {
        let data = make_larger_clustered_data(2024);
        let serial = KMeans::train(&data, 3, 12, Metric::L2, 777, 0);
        for parallelism in [1, 2, 3, 4, 8, 64] {
            let parallel = KMeans::train(&data, 3, 12, Metric::L2, 777, parallelism);
            assert_eq!(
                serial.centroids, parallel.centroids,
                "parallelism={parallelism} diverged from the serial (0) result"
            );
        }
    }

    /// Same contract for the cosine metric's extra per-centroid
    /// normalization step (`normalize_in_place`), which runs after the
    /// parallel update/combine — make sure that doesn't reintroduce
    /// parallelism-dependent drift either.
    #[test]
    fn parallelism_never_changes_trained_centroids_cosine() {
        let data = make_larger_clustered_data(2025);
        let serial = KMeans::train(&data, 3, 9, Metric::Cosine, 555, 0);
        let parallel = KMeans::train(&data, 3, 9, Metric::Cosine, 555, 4);
        assert_eq!(serial.centroids, parallel.centroids);
    }

    /// `update_step`/`assign_step`'s own unit-level check, isolated from
    /// `KMeans::train`'s iteration loop: a fixed `assignment` reduced at
    /// `parallelism = 0` vs `4` must yield byte-identical `(sums, counts)` —
    /// the exact mechanism `UPDATE_CHUNK_SIZE`'s rustdoc describes, pinned
    /// directly rather than only observed transitively through converged
    /// centroids.
    #[test]
    fn update_step_reduction_is_parallelism_invariant() {
        let data = make_larger_clustered_data(99);
        let n_lists = 5;
        // A fixed, arbitrary (not converged) assignment — this test cares
        // about the reduction's own determinism, not clustering quality.
        let mut rng = SplitMix64::new(3);
        let assignment: Vec<usize> = (0..data.len())
            .map(|_| rng.next_bounded(n_lists as u32) as usize)
            .collect();
        let sample_idx: Vec<usize> = (0..data.len()).collect();

        let (serial_sums, serial_counts) =
            update_step(&assignment, &sample_idx, &data, n_lists, 3, 0);
        for parallelism in [1, 2, 3, 4, 7] {
            let (sums, counts) =
                update_step(&assignment, &sample_idx, &data, n_lists, 3, parallelism);
            assert_eq!(counts, serial_counts, "parallelism={parallelism}");
            assert_eq!(sums, serial_sums, "parallelism={parallelism}");
        }
    }
}
