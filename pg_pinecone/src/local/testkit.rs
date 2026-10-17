//! Test support for the RaBitQ quantiser: deterministic data, an f64 oracle written from the
//! formulas alone (it shares no code with `rotation.rs`, `rabitq.rs` or `kernel.rs`), and the
//! error statistics. Pure functions only: plain `#[test]`s link without Postgres.

#![allow(dead_code)]

use super::{
    metric::Metric,
    rng::{SplitMix64, domain},
};

pub(crate) struct Rng {
    inner: SplitMix64,
    spare: Option<f64>,
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Rng {
        Rng {
            inner: SplitMix64::new(seed),
            spare: None,
        }
    }

    pub(crate) fn u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    /// Uniform in [0, 1).
    pub(crate) fn uniform(&mut self) -> f64 {
        (self.inner.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.uniform() * n as f64) as usize
    }

    /// Standard normal (Box-Muller).
    pub(crate) fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }

    pub(crate) fn normals(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.normal()).collect()
    }

    pub(crate) fn unit(&mut self, n: usize) -> Vec<f64> {
        unit64(&self.normals(n))
    }

    pub(crate) fn log_uniform(&mut self, lo: f64, hi: f64) -> f64 {
        (lo.ln() + self.uniform() * (hi.ln() - lo.ln())).exp()
    }
}

pub(crate) fn dot64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn norm64(a: &[f64]) -> f64 {
    dot64(a, a).sqrt()
}

pub(crate) fn unit64(a: &[f64]) -> Vec<f64> {
    let n = norm64(a);
    a.iter().map(|x| x / n).collect()
}

/// Inner product of f32 vectors accumulated in f64, so rounding in the reference is negligible.
pub(crate) fn dotf(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

pub(crate) fn normf(a: &[f32]) -> f64 {
    dotf(a, a).sqrt()
}

pub(crate) fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|x| *x as f32).collect()
}

pub(crate) fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
}

pub(crate) fn unit_f32(v: &[f32]) -> Vec<f32> {
    let n = normf(v);
    v.iter().map(|x| (*x as f64 / n) as f32).collect()
}

pub(crate) fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

pub(crate) fn std_dev(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

pub(crate) fn padded(d: usize) -> usize {
    (d.div_ceil(64) * 64).max(256)
}

/// The rotation of `rotation.rs`'s construction note with the block transform evaluated from its
/// definition (`H[i][j] = (-1)^popcount(i & j)`), not by a butterfly, so the two cannot share a bug.
pub(crate) struct OracleRotation {
    pub(crate) d: usize,
    pub(crate) dp: usize,
    signs: Vec<Vec<f64>>,
    perms: Vec<Vec<usize>>,
}

pub(crate) fn oracle_rotation(seed: u64, d: usize) -> OracleRotation {
    let dp = padded(d);
    let mut rng = SplitMix64::for_domain(seed, domain::ROTATION);
    let (mut signs, mut perms) = (Vec::new(), Vec::new());
    for _ in 0..4 {
        signs.push((0..dp).map(|_| rng.next_sign() as f64).collect());
        let mut perm: Vec<usize> = (0..dp).collect();
        rng.shuffle(&mut perm);
        perms.push(perm);
    }
    OracleRotation {
        d,
        dp,
        signs,
        perms,
    }
}

impl OracleRotation {
    pub(crate) fn apply(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(v.len(), self.d);
        let mut x = vec![0.0; self.dp];
        x[..v.len()].copy_from_slice(v);
        for round in 0..4 {
            for (xi, s) in x.iter_mut().zip(&self.signs[round]) {
                *xi *= s;
            }
            let mut blocked = vec![0.0; self.dp];
            for block in 0..self.dp / 64 {
                for i in 0..64usize {
                    let mut sum = 0.0;
                    for j in 0..64usize {
                        let sign = if (i & j).count_ones() % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        sum += sign * x[block * 64 + j];
                    }
                    blocked[block * 64 + i] = sum / 8.0;
                }
            }
            for (i, value) in blocked.into_iter().enumerate() {
                x[self.perms[round][i]] = value;
            }
        }
        x
    }
}

pub(crate) struct OracleCode {
    pub(crate) bits: Vec<u64>,
    pub(crate) rnorm: f64,
    pub(crate) dot: f64,
    pub(crate) rc: f64,
    /// `min_i |o'_i| * sqrt(D)`: how close the nearest coordinate is to flipping its bit; a golden
    /// vector needs it well above f32 rounding.
    pub(crate) margin: f64,
    /// `o'` itself, so a comparison can skip coordinates too close to zero for an f32 rotation.
    pub(crate) rotated: Vec<f64>,
}

pub(crate) fn oracle_encode(rot: &OracleRotation, x: &[f32], c: &[f32]) -> OracleCode {
    let r: Vec<f64> = x
        .iter()
        .zip(c)
        .map(|(a, b)| *a as f64 - *b as f64)
        .collect();
    let rnorm = norm64(&r);
    let words = rot.dp / 64;
    if rnorm == 0.0 {
        return OracleCode {
            bits: vec![0; words],
            rnorm: 0.0,
            dot: 1.0,
            rc: 0.0,
            margin: f64::INFINITY,
            rotated: vec![0.0; rot.dp],
        };
    }
    let o: Vec<f64> = r.iter().map(|v| v / rnorm).collect();
    let rotated = rot.apply(&o);
    let mut bits = vec![0u64; words];
    for (i, v) in rotated.iter().enumerate() {
        if *v >= 0.0 {
            bits[i / 64] |= 1u64 << (i % 64);
        }
    }
    let sqrt_d = (rot.dp as f64).sqrt();
    OracleCode {
        bits,
        rnorm,
        dot: rotated.iter().map(|v| v.abs()).sum::<f64>() / sqrt_d,
        rc: r.iter().zip(c).map(|(a, b)| a * *b as f64).sum(),
        margin: rotated
            .iter()
            .map(|v| v.abs())
            .fold(f64::INFINITY, f64::min)
            * sqrt_d,
        rotated,
    }
}

pub(crate) fn oracle_bit(bits: &[u64], i: usize) -> bool {
    bits[i / 64] >> (i % 64) & 1 == 1
}

pub(crate) fn oracle_err(dot: f64, dp: usize) -> f64 {
    1.9 * ((1.0 - dot * dot) / (dot * dot * (dp as f64 - 1.0))).sqrt()
}

/// Whether IP and cosine take the centred form for this list: `|q - c| < |q|` (L2 is always centred).
pub(crate) fn oracle_centred(metric: Metric, c: &[f32], q: &[f32]) -> bool {
    match metric {
        Metric::L2 => true,
        Metric::Ip | Metric::Cosine => {
            let (mut diff, mut qq) = (0.0f64, 0.0f64);
            for (c, q) in c.iter().zip(q) {
                diff += (*q as f64 - *c as f64).powi(2);
                qq += (*q as f64).powi(2);
            }
            diff < qq
        }
    }
}

/// What `est` estimates: `<o, g> / |g|` with `o = (x - c) / |x - c|`, `g = q - c` when the list is
/// centred and `g = q` otherwise (0 if either vector is 0).
pub(crate) fn exact_est(metric: Metric, x: &[f32], c: &[f32], q: &[f32]) -> f64 {
    let centred = oracle_centred(metric, c, q);
    let (mut rg, mut rr, mut gg) = (0.0, 0.0, 0.0);
    for ((x, c), q) in x.iter().zip(c).zip(q) {
        let r = *x as f64 - *c as f64;
        let g = if centred {
            *q as f64 - *c as f64
        } else {
            *q as f64
        };
        rg += r * g;
        rr += r * r;
        gg += g * g;
    }
    if rr == 0.0 || gg == 0.0 {
        0.0
    } else {
        rg / (rr.sqrt() * gg.sqrt())
    }
}

/// The exact ascending key: squared L2, `-<x, q>`, and for cosine `1 - <x, q>` on unit vectors.
pub(crate) fn exact_key(metric: Metric, x: &[f32], q: &[f32]) -> f64 {
    match metric {
        Metric::L2 => x
            .iter()
            .zip(q)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum(),
        Metric::Ip => -dotf(x, q),
        Metric::Cosine => 1.0 - dotf(x, q),
    }
}

/// Accumulates `(est, err)` against the exact value over many pairs. `sigma` is the per-pair
/// predicted standard deviation `err / 1.9 * sqrt(1 - truth^2)`: the bound's sigma times the
/// component of the query orthogonal to the vector.
#[derive(Default)]
pub(crate) struct ErrorStats {
    pub(crate) n: usize,
    sum_e: f64,
    sum_e2: f64,
    sum_sigma2: f64,
    covered_19: usize,
    covered_3: usize,
    covered_err: usize,
    below_err: usize,
    sum_dot: f64,
    pub(crate) min_dot: f64,
}

impl ErrorStats {
    pub(crate) fn new() -> ErrorStats {
        ErrorStats {
            min_dot: f64::INFINITY,
            ..ErrorStats::default()
        }
    }

    pub(crate) fn add(&mut self, est: f64, err: f64, dot: f64, truth: f64) {
        let e = est - truth;
        let sigma = err / 1.9 * (1.0 - truth * truth).max(0.0).sqrt();
        self.n += 1;
        self.sum_e += e;
        self.sum_e2 += e * e;
        self.sum_sigma2 += sigma * sigma;
        self.covered_19 += (e.abs() <= 1.9 * sigma) as usize;
        self.covered_3 += (e.abs() <= 3.0 * sigma) as usize;
        self.covered_err += (e.abs() <= err) as usize;
        self.below_err += (e <= err) as usize;
        self.sum_dot += dot;
        self.min_dot = self.min_dot.min(dot);
    }

    pub(crate) fn mean_error(&self) -> f64 {
        self.sum_e / self.n as f64
    }

    pub(crate) fn mean_error_se(&self) -> f64 {
        self.rms_error() / (self.n as f64).sqrt()
    }

    pub(crate) fn rms_error(&self) -> f64 {
        (self.sum_e2 / self.n as f64).sqrt()
    }

    /// RMS error over the RMS predicted sigma: 1 when `err` is calibrated.
    pub(crate) fn sigma_ratio(&self) -> f64 {
        (self.sum_e2 / self.sum_sigma2).sqrt()
    }

    /// Fraction with `|e| <= 1.9 sigma` (0.943 for a Gaussian error).
    pub(crate) fn coverage_1_9(&self) -> f64 {
        self.covered_19 as f64 / self.n as f64
    }

    pub(crate) fn coverage_3(&self) -> f64 {
        self.covered_3 as f64 / self.n as f64
    }

    /// Fraction with `|e| <= err` as stored (no angle factor): at least [`coverage_1_9`](Self::coverage_1_9).
    pub(crate) fn coverage_err(&self) -> f64 {
        self.covered_err as f64 / self.n as f64
    }

    /// Fraction with `est - truth <= err`: the lower bound `est - err` holds.
    pub(crate) fn one_sided(&self) -> f64 {
        self.below_err as f64 / self.n as f64
    }

    pub(crate) fn mean_dot(&self) -> f64 {
        self.sum_dot / self.n as f64
    }
}

/// Named adversarial unit vectors of dimension `d`: axis-aligned, flat, sign patterns, Walsh rows
/// (`d` a power of two), sparse and low-rank.
pub(crate) fn structured_unit_vectors(d: usize, seed: u64) -> Vec<(String, Vec<f64>)> {
    let mut rng = Rng::new(seed);
    let mut out: Vec<(String, Vec<f64>)> = Vec::new();
    for j in [0, 1, d / 3, d / 2, d - 1] {
        let mut v = vec![0.0; d];
        v[j.min(d - 1)] = 1.0;
        out.push((format!("one-hot {j}"), v));
    }
    for (a, b) in [(0, 1), (0, d - 1), (d / 4, 3 * d / 4)] {
        if a == b || b >= d {
            continue;
        }
        let mut v = vec![0.0; d];
        v[a] = 1.0;
        v[b] = -1.0;
        out.push((format!("two-hot {a},{b}"), unit64(&v)));
    }
    let signs = |f: &dyn Fn(usize) -> bool| -> Vec<f64> {
        unit64(
            &(0..d)
                .map(|i| if f(i) { 1.0 } else { -1.0 })
                .collect::<Vec<_>>(),
        )
    };
    out.push(("constant".into(), unit64(&vec![1.0; d])));
    out.push(("alternating".into(), signs(&|i| i % 2 == 0)));
    out.push(("half and half".into(), signs(&|i| i < d / 2)));
    for k in 0..3 {
        let pattern: Vec<bool> = (0..d).map(|_| rng.u64() & 1 == 0).collect();
        out.push((format!("random signs {k}"), signs(&|i| pattern[i])));
    }
    for k in 0..3 {
        let mut v = vec![0.0; d];
        for _ in 0..8 {
            v[rng.below(d)] = rng.normal();
        }
        out.push((format!("8-sparse {k}"), unit64(&v)));
    }
    if d.is_power_of_two() {
        for row in [1, 2, d / 2, d - 1] {
            out.push((
                format!("walsh row {row}"),
                signs(&|i| (i & row).count_ones() % 2 == 0),
            ));
        }
    }
    let basis: Vec<Vec<f64>> = (0..3).map(|_| rng.normals(d)).collect();
    for k in 0..3 {
        let mut v = vec![0.0; d];
        for b in &basis {
            let weight = rng.normal() + 0.1 * k as f64;
            for (vi, bi) in v.iter_mut().zip(b) {
                *vi += weight * bi;
            }
        }
        out.push((format!("rank-3 {k}"), unit64(&v)));
    }
    out
}

/// A Gaussian mixture: components with a centre, `rank` latent factors and isotropic noise, an
/// optional offset (added to the first `offset_components` components, `0` meaning all of them) and
/// an optional per-vector scale.
#[derive(Clone)]
pub(crate) struct MixtureSpec {
    pub(crate) dim: usize,
    pub(crate) components: usize,
    pub(crate) centre_norm: f64,
    pub(crate) offset_norm: f64,
    pub(crate) offset_components: usize,
    pub(crate) rank: usize,
    pub(crate) factor_std: f64,
    pub(crate) noise_norm: f64,
    pub(crate) scale_range: Option<(f64, f64)>,
}

pub(crate) struct Mixture {
    pub(crate) vectors: Vec<Vec<f32>>,
    pub(crate) queries: Vec<Vec<f32>>,
    pub(crate) vector_component: Vec<usize>,
    pub(crate) query_component: Vec<usize>,
}

/// `n` rows and `n_queries` data-like queries (the same sampler, fresh draws), deterministic in `seed`.
pub(crate) fn mixture(spec: &MixtureSpec, n: usize, n_queries: usize, seed: u64) -> Mixture {
    let mut rng = Rng::new(seed);
    let dim = spec.dim;
    let offset: Vec<f64> = if spec.offset_norm > 0.0 {
        rng.unit(dim).iter().map(|v| v * spec.offset_norm).collect()
    } else {
        vec![0.0; dim]
    };
    let offset_components = if spec.offset_components == 0 {
        spec.components
    } else {
        spec.offset_components
    };
    let centres: Vec<Vec<f64>> = (0..spec.components)
        .map(|k| {
            let carries_offset = k < offset_components;
            rng.unit(dim)
                .iter()
                .zip(&offset)
                .map(|(v, o)| v * spec.centre_norm + if carries_offset { *o } else { 0.0 })
                .collect()
        })
        .collect();
    let factors: Vec<Vec<Vec<f64>>> = (0..spec.components)
        .map(|_| (0..spec.rank).map(|_| rng.unit(dim)).collect())
        .collect();
    let noise_std = spec.noise_norm / (dim as f64).sqrt();
    let sample = |rng: &mut Rng| -> (Vec<f32>, usize) {
        let k = rng.below(spec.components);
        let mut v = centres[k].clone();
        for f in &factors[k] {
            let w = rng.normal() * spec.factor_std;
            for (vi, fi) in v.iter_mut().zip(f) {
                *vi += w * fi;
            }
        }
        for vi in v.iter_mut() {
            *vi += rng.normal() * noise_std;
        }
        if let Some((lo, hi)) = spec.scale_range {
            let s = rng.log_uniform(lo, hi);
            v.iter_mut().for_each(|x| *x *= s);
        }
        (to_f32(&v), k)
    };
    let (vectors, vector_component) = (0..n).map(|_| sample(&mut rng)).unzip();
    let (queries, query_component) = (0..n_queries).map(|_| sample(&mut rng)).unzip();
    Mixture {
        vectors,
        queries,
        vector_component,
        query_component,
    }
}
