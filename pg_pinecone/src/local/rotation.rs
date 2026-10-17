//! Random orthonormal rotation `P` of the RaBitQ quantiser.
//!
//! Papers: Gao & Long, RaBitQ, SIGMOD 2024; Fino & Algazi, Unified matrix treatment of the fast
//! Walsh-Hadamard transform, IEEE Trans. Computers 1976. Clean-room from the papers: no code from
//! any proprietary or third-party implementation.
//!
//! RaBitQ assumes a Haar-random rotation. This is the structured substitute of the JL literature:
//! random signs followed by a Walsh-Hadamard transform (Ailon & Chazelle, the fast
//! Johnson-Lindenstrauss transform, STOC 2006), repeated with fresh signs (the `HD1 HD2 HD3` blocks
//! of Yu et al., Orthogonal random features, NeurIPS 2016), with a permutation between rounds
//! (Le, Sarlos & Smola, Fastfood, ICML 2013) and `H_64` per 64-block so that `D` need not be a
//! power of two. Orthonormal by construction; that it spreads structured inputs is what the tests
//! check.
//!
//! The construction is on-disk format (never stored, regenerated from the metapage seed, pinned by
//! the golden tests): scalar f32, fixed evaluation order, no `mul_add`, no fast-math.
//! ```text
//! D = max(256, ceil(d / 64) * 64)   rng = SplitMix64::for_domain(seed, rng::domain::ROTATION)
//! per round r in 0..4: signs[r][i] = rng.next_sign() for i in 0..D, then perm[r] = identity(D)
//!                      shuffled by rng.shuffle
//! apply(v) = P^T v: x = v zero-padded to D; per round: x[i] *= signs[r][i]; every 64-block :=
//!                   H_64 * block / 8 (Sylvester order, H[i][j] = (-1)^popcount(i & j));
//!                   y[perm[r][i]] = x[i]; x = y
//! ```
//! The 1/8 is folded into the stored signs (a power of two, so the products are exact). The 64-point
//! transform runs its stages in the order `h = 1, 2, 4, 8, 16, 32`.
//!
//! Evaluation: the blocks are transformed side by side. Between rounds a value sits at `i * nb + b`
//! (coordinate `i` of block `b`, `nb = D / 64`), so a stage is one vector add and subtract over
//! contiguous floats, and the permutation tables point into that layout. The sums are the same, in
//! the same order, as the per-block definition above, so the results are bit-identical to it.

use std::cell::RefCell;

use super::rng::{SplitMix64, domain};

pub(crate) const BLOCK: usize = 64;
pub(crate) const ROUNDS: usize = 4;
/// The narrowest code: a column of fewer dimensions is zero-padded up to it.
pub(crate) const MIN_CODE_BITS: usize = 256;

/// Rounds ping-pong between two buffers and the last must land in the caller's.
const _: () = assert!(ROUNDS.is_multiple_of(2));

/// `1 / sqrt(BLOCK)`: a power of two, so folding it into the signs changes no rounding.
const BLOCK_SCALE: f32 = 0.125;

/// `D`: code width, estimator dimension and the `D - 1` of the error bound. The dimension padded up
/// to a whole 64-block and to at least [`MIN_CODE_BITS`]: the estimator's error shrinks with `sqrt(D)`
/// whatever the dimension of the data (the padding is zeros, which the rotation spreads), so a column
/// of 64 or 24 dimensions would rank worse than 0.9.0's 512-bit sketch on a 64-bit code.
pub(crate) fn padded_dim(d: usize) -> usize {
    (d.div_ceil(BLOCK) * BLOCK).max(MIN_CODE_BITS)
}

pub(crate) struct Rotation {
    d: usize,
    dp: usize,
    /// `ROUNDS * D` entries of `+-1/8`, each round in the layout of its input.
    signs: Vec<f32>,
    /// `ROUNDS * D` entries: where the value at each position of a round's layout goes next (a
    /// natural-order index after the last round).
    targets: Vec<u32>,
}

thread_local! {
    static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

impl Rotation {
    /// `seed` is the index's metapage seed.
    pub(crate) fn new(seed: u64, d: usize) -> Rotation {
        let dp = padded_dim(d);
        assert!(
            d > 0 && u32::try_from(dp).is_ok(),
            "rotation: dimension {d} out of range"
        );
        let nb = dp / BLOCK;
        let layout = |p: usize| (p % BLOCK) * nb + p / BLOCK;
        let mut rng = SplitMix64::for_domain(seed, domain::ROTATION);
        let mut signs = vec![0.0; ROUNDS * dp];
        let mut targets = vec![0; ROUNDS * dp];
        for round in 0..ROUNDS {
            let natural: Vec<f32> = (0..dp).map(|_| rng.next_sign() * BLOCK_SCALE).collect();
            let mut perm: Vec<u32> = (0..dp as u32).collect();
            rng.shuffle(&mut perm);
            for (p, (sign, target)) in natural.into_iter().zip(perm).enumerate() {
                let at = round * dp + layout(p);
                signs[at] = sign;
                targets[at] = if round + 1 < ROUNDS {
                    layout(target as usize) as u32
                } else {
                    target
                };
            }
        }
        Rotation {
            d,
            dp,
            signs,
            targets,
        }
    }

    pub(crate) fn dim(&self) -> usize {
        self.d
    }

    pub(crate) fn padded(&self) -> usize {
        self.dp
    }

    /// `v.len() == d`; the result has length `D`.
    pub(crate) fn apply(&self, v: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; self.dp];
        self.apply_into(v, &mut out);
        out
    }

    /// `out.len() == D`. Allocation-free after the thread's first call (thread-local scratch).
    pub(crate) fn apply_into(&self, v: &[f32], out: &mut [f32]) {
        SCRATCH.with_borrow_mut(|scratch| {
            scratch.resize(self.dp, 0.0);
            self.apply_into_with(v, out, scratch);
        });
    }

    /// [`apply_into`](Self::apply_into) with a caller-owned `scratch` of length `D`: never
    /// allocates.
    pub(crate) fn apply_into_with(&self, v: &[f32], out: &mut [f32], scratch: &mut [f32]) {
        assert_eq!(v.len(), self.d, "rotation: input length");
        assert_eq!(out.len(), self.dp, "rotation: output length");
        assert_eq!(scratch.len(), self.dp, "rotation: scratch length");
        let (dp, nb) = (self.dp, self.dp / BLOCK);
        let first_signs = &self.signs[..dp];
        for b in 0..nb {
            for i in 0..BLOCK {
                let at = i * nb + b;
                out[at] = v.get(b * BLOCK + i).map_or(0.0, |x| x * first_signs[at]);
            }
        }
        let (mut from, mut to) = (out, scratch);
        for (r, targets) in self.targets.chunks_exact(dp).enumerate() {
            hadamard_stages(from, nb);
            for (value, target) in from.iter().zip(targets) {
                to[*target as usize] = *value;
            }
            if let Some(next_signs) = self.signs.get((r + 1) * dp..(r + 2) * dp) {
                for (value, sign) in to.iter_mut().zip(next_signs) {
                    *value *= sign;
                }
            }
            std::mem::swap(&mut from, &mut to);
        }
    }
}

/// The six stages of `H_64` (unnormalised, Sylvester order) on `nb` blocks stored side by side,
/// `x.len() == 64 * nb`: stage `h` replaces the rows `i` and `i + h` (`nb` floats each) of every
/// aligned group of `2h` rows by their sum and difference.
fn hadamard_stages(x: &mut [f32], nb: usize) {
    let mut width = nb;
    while width < x.len() {
        for pair in x.chunks_exact_mut(2 * width) {
            let (lo, hi) = pair.split_at_mut(width);
            for (a, b) in lo.iter_mut().zip(hi) {
                let (u, v) = (*a, *b);
                *a = u + v;
                *b = u - v;
            }
        }
        width *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local::testkit::{
        Rng, dot64, norm64, oracle_rotation, structured_unit_vectors, to_f32, to_f64,
    };

    const SEED: u64 = 0x5EED_0001;
    const DIMS: [usize; 11] = [1, 3, 64, 65, 100, 128, 768, 800, 1024, 1536, 2000];

    fn rotated(rot: &Rotation, v: &[f64]) -> Vec<f64> {
        to_f64(&rot.apply(&to_f32(v)))
    }

    #[test]
    fn padded_dim_rounds_up_to_whole_blocks_with_a_floor_of_256() {
        let table = [
            (1, 256),
            (3, 256),
            (24, 256),
            (63, 256),
            (64, 256),
            (65, 256),
            (100, 256),
            (128, 256),
            (129, 256),
            (192, 256),
            (255, 256),
            (256, 256),
            (257, 320),
            (300, 320),
            (320, 320),
            (321, 384),
            (768, 768),
            (769, 832),
            (800, 832),
            (832, 832),
            (1024, 1024),
            (1536, 1536),
            (2000, 2048),
            (2038, 2048),
        ];
        for (d, dp) in table {
            assert_eq!(padded_dim(d), dp, "d = {d}");
        }
    }

    #[test]
    fn padded_dim_is_the_smallest_multiple_of_64_that_holds_the_column_and_the_floor() {
        assert_eq!(MIN_CODE_BITS, 256);
        for d in 1..=2048usize {
            let dp = padded_dim(d);
            assert!(
                dp >= d && dp >= 256 && dp.is_multiple_of(BLOCK),
                "d = {d}: {dp}"
            );
            assert!(
                dp - BLOCK < d.max(256),
                "d = {d}: {dp} is not the smallest such width"
            );
        }
    }

    #[test]
    fn rotation_reports_its_dimensions_and_pads_the_output() {
        for &d in &DIMS {
            let rot = Rotation::new(SEED, d);
            assert_eq!(rot.dim(), d);
            assert_eq!(rot.padded(), padded_dim(d));
            assert_eq!(rot.apply(&vec![0.5; d]).len(), padded_dim(d), "d = {d}");
        }
    }

    #[test]
    fn apply_into_equals_apply_and_overwrites_the_buffer() {
        let rot = Rotation::new(SEED, 100);
        let v = to_f32(&Rng::new(1).normals(100));
        let mut out = vec![f32::NAN; padded_dim(100)];
        rot.apply_into(&v, &mut out);
        assert_eq!(out, rot.apply(&v));
    }

    #[test]
    fn rotation_preserves_norms_and_inner_products() {
        for &d in &DIMS {
            let rot = Rotation::new(SEED, d);
            let mut rng = Rng::new(d as u64);
            let mut inputs: Vec<Vec<f64>> = (0..4).map(|_| rng.normals(d)).collect();
            inputs.extend(structured_unit_vectors(d, 7).into_iter().map(|(_, v)| v));
            let inputs: Vec<Vec<f64>> = inputs.iter().map(|v| to_f64(&to_f32(v))).collect();
            let outputs: Vec<Vec<f64>> = inputs.iter().map(|v| rotated(&rot, v)).collect();
            for (u, pu) in inputs.iter().zip(&outputs) {
                let (n, pn) = (norm64(u), norm64(pu));
                assert!(
                    (pn - n).abs() <= 1e-5 * n,
                    "d = {d}: |Pu| = {pn}, |u| = {n}"
                );
            }
            for i in 0..inputs.len() {
                for j in i + 1..inputs.len() {
                    let want = dot64(&inputs[i], &inputs[j]);
                    let got = dot64(&outputs[i], &outputs[j]);
                    let scale = norm64(&inputs[i]) * norm64(&inputs[j]);
                    assert!(
                        (got - want).abs() <= 1e-5 * scale,
                        "d = {d}, pair ({i}, {j}): {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn rotation_matrix_has_orthonormal_columns() {
        for d in [1usize, 3, 64, 100, 128, 192] {
            let rot = Rotation::new(SEED, d);
            let columns: Vec<Vec<f64>> = (0..d)
                .map(|j| {
                    let mut e = vec![0.0f32; d];
                    e[j] = 1.0;
                    to_f64(&rot.apply(&e))
                })
                .collect();
            for i in 0..d {
                for j in i..d {
                    let want = if i == j { 1.0 } else { 0.0 };
                    let got = dot64(&columns[i], &columns[j]);
                    assert!(
                        (got - want).abs() <= 1e-5,
                        "d = {d}: <P e{i}, P e{j}> = {got}"
                    );
                }
            }
        }
    }

    #[test]
    fn rotation_is_linear() {
        for d in [64usize, 100, 768] {
            let rot = Rotation::new(SEED, d);
            let mut rng = Rng::new(d as u64);
            let (u, v) = (to_f32(&rng.normals(d)), to_f32(&rng.normals(d)));
            let mixed: Vec<f32> = u.iter().zip(&v).map(|(a, b)| 2.5 * a - 0.75 * b).collect();
            let (pu, pv) = (rot.apply(&u), rot.apply(&v));
            let want: Vec<f64> = pu
                .iter()
                .zip(&pv)
                .map(|(a, b)| 2.5 * *a as f64 - 0.75 * *b as f64)
                .collect();
            let got = to_f64(&rot.apply(&mixed));
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(
                worst <= 1e-5 * (d as f64).sqrt(),
                "d = {d}: worst deviation {worst}"
            );
        }
    }

    #[test]
    fn rotation_is_a_pure_function_of_seed_and_dimension() {
        for d in [64usize, 100, 768] {
            let v = to_f32(&Rng::new(d as u64).normals(d));
            let first = Rotation::new(SEED, d).apply(&v);
            assert_eq!(first, Rotation::new(SEED, d).apply(&v), "d = {d}");
            let other = Rotation::new(SEED + 1, d).apply(&v);
            let differing = first.iter().zip(&other).filter(|(a, b)| a != b).count();
            assert!(
                differing > first.len() / 2,
                "d = {d}: {differing} coordinates differ"
            );
            let cosine = dot64(&to_f64(&first), &to_f64(&other))
                / (norm64(&to_f64(&first)) * norm64(&to_f64(&other)));
            assert!(
                cosine.abs() < 6.0 / (first.len() as f64).sqrt(),
                "d = {d}: seeds correlate, cos = {cosine}"
            );
        }
    }

    #[test]
    fn rotation_is_shareable_across_threads_with_identical_results() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Rotation>();
        let rot = Rotation::new(SEED, 768);
        let inputs: Vec<Vec<f32>> = (0..16).map(|i| to_f32(&Rng::new(i).normals(768))).collect();
        let serial: Vec<Vec<f32>> = inputs.iter().map(|v| rot.apply(v)).collect();
        let parallel: Vec<Vec<f32>> = std::thread::scope(|scope| {
            let handles: Vec<_> = inputs
                .chunks(4)
                .map(|chunk| {
                    let rot = &rot;
                    scope.spawn(move || chunk.iter().map(|v| rot.apply(v)).collect::<Vec<_>>())
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("worker"))
                .collect()
        });
        assert_eq!(serial, parallel);
    }

    #[test]
    fn rotation_matches_the_f64_oracle() {
        for d in [1usize, 3, 64, 65, 100, 128, 768, 800, 1024] {
            let (rot, oracle) = (Rotation::new(SEED, d), oracle_rotation(SEED, d));
            let mut inputs = vec![Rng::new(d as u64).normals(d)];
            inputs.extend(
                structured_unit_vectors(d, 3)
                    .into_iter()
                    .step_by(4)
                    .map(|(_, v)| v),
            );
            for v in inputs {
                let v = to_f64(&to_f32(&v));
                let got = rotated(&rot, &v);
                let want = oracle.apply(&v);
                let scale = norm64(&v).max(1.0);
                let worst = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                assert!(worst <= 5e-6 * scale, "d = {d}: worst deviation {worst}");
            }
        }
    }

    #[test]
    fn rotation_spreads_structured_inputs_so_the_code_keeps_their_mass() {
        // The columns below 256 dimensions are zero-padded to D = 256: the rotation must spread a
        // structured vector over all of it, not leave it in the padded block's corner.
        let floors = [
            (24usize, 0.65),
            (64, 0.65),
            (100, 0.7),
            (128, 0.7),
            (768, 0.72),
            (1024, 0.72),
        ];
        for (d, floor) in floors {
            for seed in [SEED, SEED + 1, 0xDEAD_BEEF, 42] {
                let rot = Rotation::new(seed, d);
                let sqrt_dp = (rot.padded() as f64).sqrt();
                for (name, v) in structured_unit_vectors(d, 11) {
                    let rotated = rotated(&rot, &v);
                    let dot = rotated.iter().map(|x| x.abs()).sum::<f64>() / sqrt_dp;
                    assert!(
                        dot >= floor,
                        "d = {d}, seed = {seed:#x}, {name}: |Pv|_1 / sqrt(D) = {dot:.3} < {floor}"
                    );
                }
            }
        }
    }

    fn pinned_input(d: usize, a: usize, b: usize, m: usize) -> Vec<f32> {
        (0..d)
            .map(|i| ((i * a + b) % m) as f32 - (m / 2) as f32)
            .collect()
    }

    struct Pinned {
        seed: u64,
        d: usize,
        input: (usize, usize, usize),
        first: [f64; 6],
        last: [f64; 2],
        l1: f64,
    }

    /// Derived from `testkit::oracle_rotation` (the f64 definition of the construction in this
    /// module's docs): integer inputs, so every value is a dyadic rational. On-disk format: change
    /// them only with a format bump.
    const PINNED: [Pinned; 2] = [
        Pinned {
            seed: 0xC0DE_0064,
            d: 64,
            input: (7, 3, 11),
            first: [
                1.555_908_203_125,
                0.683_105_468_75,
                -0.257_324_218_75,
                -0.208_007_812_5,
                -0.850_585_937_5,
                -1.642_333_984_375,
            ],
            last: [0.608_886_718_75, 3.105_468_75],
            l1: 322.520_507_812_5,
        },
        Pinned {
            seed: 0xFEED_0768,
            d: 768,
            input: (13, 5, 17),
            first: [
                -6.879_394_5,
                -1.606_201_2,
                -0.427_734_4,
                -1.291_748,
                -3.867_675_8,
                -3.476_806_6,
            ],
            last: [0.584_716_8, 2.771_240_2],
            l1: 3_033.205_078,
        },
    ];

    #[test]
    fn golden_rotation_outputs_are_pinned() {
        for p in &PINNED {
            let out = Rotation::new(p.seed, p.d)
                .apply(&pinned_input(p.d, p.input.0, p.input.1, p.input.2));
            let label = format!("seed {:#x}, d = {}", p.seed, p.d);
            for (i, want) in p.first.iter().enumerate() {
                assert!(
                    (out[i] as f64 - want).abs() <= 1e-6 * want.abs().max(1.0),
                    "{label}: out[{i}] = {} vs {want}",
                    out[i]
                );
            }
            for (k, want) in p.last.iter().enumerate() {
                let got = out[out.len() - 2 + k] as f64;
                assert!(
                    (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                    "{label}: tail {k} = {got} vs {want}"
                );
            }
            let l1: f64 = out.iter().map(|x| x.abs() as f64).sum();
            assert!(
                (l1 - p.l1).abs() <= 1e-6 * p.l1,
                "{label}: |Pv|_1 = {l1} vs {}",
                p.l1
            );
        }
    }

    #[test]
    fn oracle_reproduces_the_golden_rotation_outputs() {
        for p in &PINNED {
            let v = to_f64(&pinned_input(p.d, p.input.0, p.input.1, p.input.2));
            let out = oracle_rotation(p.seed, p.d).apply(&v);
            for (i, want) in p.first.iter().enumerate() {
                assert!(
                    (out[i] - want).abs() <= 1e-6 * want.abs().max(1.0),
                    "out[{i}] = {} vs {want}",
                    out[i]
                );
            }
            assert!((out.iter().map(|x| x.abs()).sum::<f64>() - p.l1).abs() <= 1e-6 * p.l1);
        }
    }
}

#[cfg(test)]
mod hot_path_tests {
    use super::*;
    use crate::local::testkit::{Rng, to_f32};
    use std::panic::catch_unwind;

    /// The construction of the module docs block by block, in f32 with the same stage order.
    fn per_block(seed: u64, d: usize, v: &[f32]) -> Vec<f32> {
        let dp = padded_dim(d);
        let mut rng = SplitMix64::for_domain(seed, domain::ROTATION);
        let mut x = v.to_vec();
        x.resize(dp, 0.0);
        for _ in 0..ROUNDS {
            let signs: Vec<f32> = (0..dp).map(|_| rng.next_sign() * BLOCK_SCALE).collect();
            let mut perm: Vec<usize> = (0..dp).collect();
            rng.shuffle(&mut perm);
            for (value, sign) in x.iter_mut().zip(&signs) {
                *value *= sign;
            }
            for block in x.as_chunks_mut::<BLOCK>().0 {
                let mut h = 1;
                while h < BLOCK {
                    for pair in block.chunks_exact_mut(2 * h) {
                        let (lo, hi) = pair.split_at_mut(h);
                        for (a, b) in lo.iter_mut().zip(hi) {
                            let (u, w) = (*a, *b);
                            (*a, *b) = (u + w, u - w);
                        }
                    }
                    h *= 2;
                }
            }
            let mut y = vec![0.0; dp];
            for (value, target) in x.iter().zip(&perm) {
                y[*target] = *value;
            }
            x = y;
        }
        x
    }

    #[test]
    fn side_by_side_evaluation_equals_the_block_by_block_definition() {
        for (n, d) in [1usize, 3, 63, 64, 65, 100, 128, 129, 768, 800, 1024, 2000]
            .into_iter()
            .enumerate()
        {
            let mut rng = Rng::new(n as u64);
            let mut inputs = vec![to_f32(&rng.normals(d)), vec![0.0; d], vec![1.0; d]];
            let mut one_hot = vec![0.0f32; d];
            one_hot[d / 2] = 3.5;
            inputs.push(one_hot);
            for seed in [7u64, 0xDEAD_BEEF] {
                let rot = Rotation::new(seed, d);
                for v in &inputs {
                    assert_eq!(
                        rot.apply(v),
                        per_block(seed, d, v),
                        "d = {d}, seed = {seed}"
                    );
                }
            }
        }
    }

    #[test]
    fn apply_into_with_ignores_what_the_buffers_held() {
        for d in [1usize, 63, 64, 65, 100, 768] {
            let rot = Rotation::new(7, d);
            let v = to_f32(&Rng::new(d as u64).normals(d));
            let want = rot.apply(&v);
            let mut out = vec![f32::NAN; padded_dim(d)];
            let mut scratch = vec![f32::INFINITY; padded_dim(d)];
            rot.apply_into_with(&v, &mut out, &mut scratch);
            assert_eq!(out, want, "d = {d}");
        }
    }

    #[test]
    fn the_thread_local_scratch_follows_the_dimension() {
        let (small, large) = (Rotation::new(3, 100), Rotation::new(3, 1024));
        let u = to_f32(&Rng::new(1).normals(100));
        let w = to_f32(&Rng::new(2).normals(1024));
        let (first_small, first_large) = (small.apply(&u), large.apply(&w));
        for _ in 0..3 {
            assert_eq!(small.apply(&u), first_small);
            assert_eq!(large.apply(&w), first_large);
        }
    }

    #[test]
    fn wrong_lengths_and_a_zero_dimension_panic() {
        let rot = Rotation::new(1, 100);
        let (v, mut out) = (vec![0.5f32; 100], vec![0.0f32; padded_dim(100)]);
        assert!(catch_unwind(|| rot.apply(&v[..99])).is_err(), "short input");
        assert!(
            catch_unwind(std::panic::AssertUnwindSafe(
                || rot.apply_into(&v, &mut out[..100])
            ))
            .is_err(),
            "short output"
        );
        assert!(
            catch_unwind(std::panic::AssertUnwindSafe(|| {
                rot.apply_into_with(&v, &mut out, &mut [0.0; 64])
            }))
            .is_err(),
            "short scratch"
        );
        assert!(catch_unwind(|| Rotation::new(1, 0)).is_err(), "d = 0");
    }
}
