//! RaBitQ codes, the unbiased distance estimator and its error bound.
//!
//! Papers: Gao & Long, RaBitQ, SIGMOD 2024; Gao, Vu, Xu, Long, extended RaBitQ, 2025. Clean-room
//! from the papers: no code from any proprietary or third-party implementation.
//!
//! All in the rotated, padded frame (`D` from [`super::rotation::padded_dim`]: the dimension padded to a
//! multiple of 64 and to at least 256, vectors zero-padded):
//! ```text
//! encode(x, c):  r = x - c, rnorm = |r|, o' = P^T (r / rnorm), bit i = [o'_i >= 0] (word i/64, bit i%64),
//!                dot = |o'|_1 / sqrt(D), rc = <r, c>;  rnorm = 0 -> code 0, dot = 1, rc = 0
//! prepare:       the error of est scales with rnorm * gnorm, so each list takes the smaller scale:
//!                centred   (always for L2; IP and cosine when |q - c| < |q|):
//!                          g = q - c, gnorm = |g|, q' = (P^T q - P^T c) / gnorm
//!                uncentred (IP and cosine otherwise):
//!                          g = q,     gnorm = |q|, q' = P^T q / gnorm       (one rotation per query)
//!                cq = <c, q>; v_l = min q', v_r = max q', delta = (v_r - v_l) / 15, s_q = sum qbar,
//!                qbar_i = min(15, floor((q'_i - v_l) / delta + u_i)), u_i in [0, 1) seeded from (seed, list),
//!                plane_j = bit j of qbar
//! estimate:      acc = sum_j 2^j popcnt(b & plane_j), pb = popcnt(b)
//!                ip = (delta * (2 acc - s_q) + v_l * (2 pb - D)) / sqrt(D), est = ip / dot  (~ <o, g> / |g|)
//!                err = 1.9 * sqrt((1 - dot^2) / (dot^2 * (D - 1)))
//! keys (ascending = best, no sqrt):
//!   L2      gnorm^2 + rnorm^2 - 2 gnorm rnorm est
//!   IP      -(cq + rnorm gnorm est + rc) centred,  -(cq + rnorm gnorm est) uncentred
//!   cosine  1 - (the same sum), on x^ = x/|x| and q^ = q/|q| chosen by the caller
//! lo = the key with est + err (L2 clamped at 0)
//! ```
//! `err` is a 1.9 sigma statement for a query orthogonal to the vector (about 94 % coverage).
//! Block form: the same `acc` as one table lookup per nibble of a code (`lut`, 16 entries per 4 coordinates:
//! entry `v` of group `g` sums the 4-bit levels of the coordinates of `g` whose bit is set in `v`; see
//! `kernel.rs`), over blocks of codes stored byte-interleaved, with `pb` stored per record. [`score_block`]
//! evaluates `est`, the key and the pruning bound for a whole block with the arithmetic of [`estimate_bits`].
//! Guards: `gnorm = 0` gives est 0 (keys stay finite); `delta = 0` (a constant q') keeps the `v_l`
//! term, which is exact for a constant q'; `dot < 1e-3` gives est 0 and err = +inf (never pruned).
//!
//! Arithmetic is f32 with fixed accumulation orders (eight lanes, then a fixed tree), so results do
//! not depend on the platform or on auto-vectorisation. f64 is used only for `cq`, an inner product
//! that cancels (the golden tests pin it to 1e-5 absolute; eight f32 lanes reach 5e-6 there). The
//! rounding offsets `u_i` take one byte each, `(byte + 1/2) / 256`: unbiased on average, and never
//! 0 or 1, so the minimum always maps to level 0 and the maximum to 15.

use super::{
    kernel::Kernel,
    metric::Metric,
    rng::{SplitMix64, domain},
    rotation::Rotation,
};

pub(crate) const EPSILON: f32 = 1.9;
/// The scan prunes with the bound at this many times [`EPSILON`]'s `err` (3.8 sigma): a candidate that
/// belongs in the top rows is dropped only if its estimate is off by more than that (about 7 in
/// 100,000, against 3 in 100 at 1.9 sigma, where 6 to 14 % of the queries of the test sets lose a
/// neighbour).
pub(crate) const PRUNE_MARGIN: f32 = 2.0;
/// Bits per query coordinate; the metapage's `bq_bits`.
pub(crate) const QUERY_BITS: u32 = 4;
pub(crate) const QUERY_LEVELS: u32 = (1 << QUERY_BITS) - 1;

/// A `dot` below this is a degenerate code: no estimate, no bound.
const MIN_DOT: f32 = 1e-3;
const LANES: usize = 8;

/// One record's stored quantisation: `D / 64` words, plus three f32 (the on-disk record).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Code {
    pub(crate) bits: Vec<u64>,
    pub(crate) rnorm: f32,
    pub(crate) dot: f32,
    pub(crate) rc: f32,
}

/// The 4-bit query and the constants of one probed list.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct QueryPrep {
    pub(crate) planes: [Vec<u64>; 4],
    /// The same levels as 16-entry tables, `4 D` bytes (module docs); every entry is at most 60.
    pub(crate) lut: Vec<u8>,
    pub(crate) v_l: f32,
    pub(crate) delta: f32,
    pub(crate) s_q: u32,
    pub(crate) gnorm: f32,
    pub(crate) cq: f32,
    /// The form this list uses: `true` = centred (`g = q - c`), `false` = uncentred (`g = q`).
    pub(crate) centred: bool,
    /// `1 / sqrt(D)`.
    pub(crate) inv_sqrt_d: f32,
    /// `EPSILON / sqrt(D - 1)`.
    pub(crate) err_scale: f32,
}

impl QueryPrep {
    /// An empty prep for padded dimension `dp`, to be filled by [`prepare_query_into`].
    pub(crate) fn new(dp: usize) -> QueryPrep {
        QueryPrep {
            planes: std::array::from_fn(|_| vec![0; dp / 64]),
            lut: vec![0; dp * 4],
            v_l: 0.0,
            delta: 0.0,
            s_q: 0,
            gnorm: 0.0,
            cq: 0.0,
            centred: true,
            inv_sqrt_d: 0.0,
            err_scale: 0.0,
        }
    }
}

/// The eight lane sums of a reduction, combined in a fixed tree.
fn fold(lanes: [f32; LANES]) -> f32 {
    ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5]))
        + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]))
}

/// `sum_i f(a_i, b_i)` in eight lanes plus a sequential tail.
#[inline(always)]
fn lane_sum(a: &[f32], b: &[f32], f: impl Fn(f32, f32) -> f32) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut lanes = [0.0f32; LANES];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<LANES>(), b.as_chunks::<LANES>());
    for (x, y) in ca.iter().zip(cb) {
        for ((lane, p), q) in lanes.iter_mut().zip(x).zip(y) {
            *lane += f(*p, *q);
        }
    }
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| f(*x, *y)).sum();
    fold(lanes) + tail
}

fn sum_sq(a: &[f32]) -> f32 {
    lane_sum(a, a, |x, _| x * x)
}

fn sum_sq_diff(a: &[f32], b: &[f32]) -> f32 {
    lane_sum(a, b, |x, y| (x - y) * (x - y))
}

fn sum_abs(a: &[f32]) -> f32 {
    lane_sum(a, a, |x, _| x.abs())
}

/// `<a, b>` in f64: the cancelling reduction (`cq`).
fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    let mut lanes = [0.0f64; LANES];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<LANES>(), b.as_chunks::<LANES>());
    for (x, y) in ca.iter().zip(cb) {
        for ((lane, p), q) in lanes.iter_mut().zip(x).zip(y) {
            *lane += f64::from(*p) * f64::from(*q);
        }
    }
    let tail: f64 = ra
        .iter()
        .zip(rb)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5]))
        + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]))
        + tail
}

/// Minimum and maximum of `a`; `f32::min` and `max` skip NaNs and are one NEON/SSE instruction.
fn min_max(a: &[f32]) -> (f32, f32) {
    let (chunks, tail) = a.as_chunks::<LANES>();
    let (mut lo, mut hi) = ([f32::INFINITY; LANES], [f32::NEG_INFINITY; LANES]);
    for chunk in chunks {
        for ((l, h), v) in lo.iter_mut().zip(&mut hi).zip(chunk) {
            *l = l.min(*v);
            *h = h.max(*v);
        }
    }
    finish_min_max(lo, hi, tail.iter().copied())
}

/// [`min_max`] of `a - b`.
fn min_max_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    debug_assert_eq!(a.len(), b.len());
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<LANES>(), b.as_chunks::<LANES>());
    let (mut lo, mut hi) = ([f32::INFINITY; LANES], [f32::NEG_INFINITY; LANES]);
    for (x, y) in ca.iter().zip(cb) {
        for (((l, h), p), q) in lo.iter_mut().zip(&mut hi).zip(x).zip(y) {
            let v = p - q;
            *l = l.min(v);
            *h = h.max(v);
        }
    }
    finish_min_max(lo, hi, ra.iter().zip(rb).map(|(x, y)| x - y))
}

fn finish_min_max(
    lo: [f32; LANES],
    hi: [f32; LANES],
    tail: impl Iterator<Item = f32>,
) -> (f32, f32) {
    let lo_all = lo.into_iter().fold(f32::INFINITY, f32::min);
    let hi_all = hi.into_iter().fold(f32::NEG_INFINITY, f32::max);
    tail.fold((lo_all, hi_all), |(l, h), v| (l.min(v), h.max(v)))
}

/// Reusable buffers of [`encode_into`]: the unit residual, its rotation and the rotation's scratch.
#[derive(Default)]
pub(crate) struct EncodeScratch(Vec<f32>);

/// `x` and `c` have length `rot.dim()`; cosine callers pass `x / |x|`.
pub(crate) fn encode(rot: &Rotation, x: &[f32], c: &[f32]) -> Code {
    let mut code = Code {
        bits: Vec::new(),
        rnorm: 0.0,
        dot: 1.0,
        rc: 0.0,
    };
    encode_into(rot, x, c, &mut EncodeScratch::default(), &mut code);
    code
}

/// [`encode`] into `code`, reusing its `bits` and `scratch`: no allocation after the first call.
pub(crate) fn encode_into(
    rot: &Rotation,
    x: &[f32],
    c: &[f32],
    scratch: &mut EncodeScratch,
    code: &mut Code,
) {
    let (d, dp) = (rot.dim(), rot.padded());
    assert_eq!(x.len(), d, "encode: vector length");
    assert_eq!(c.len(), d, "encode: centroid length");
    code.bits.clear();
    code.bits.resize(dp / 64, 0);
    let rr = lane_sum(x, c, |a, b| (a - b) * (a - b));
    if rr == 0.0 {
        (code.rnorm, code.dot, code.rc) = (0.0, 1.0, 0.0);
        return;
    }
    let rnorm = rr.sqrt();
    scratch.0.resize(d + 2 * dp, 0.0);
    let (unit, rest) = scratch.0.split_at_mut(d);
    let (rotated, spare) = rest.split_at_mut(dp);
    for ((u, a), b) in unit.iter_mut().zip(x).zip(c) {
        *u = (a - b) / rnorm;
    }
    rot.apply_into_with(unit, rotated, spare);
    for (word, chunk) in code.bits.iter_mut().zip(rotated.as_chunks::<64>().0) {
        *word = chunk
            .iter()
            .enumerate()
            .fold(0, |w, (i, v)| w | u64::from(*v >= 0.0) << i);
    }
    code.rnorm = rnorm;
    code.dot = sum_abs(rotated) / (dp as f32).sqrt();
    code.rc = lane_sum(x, c, |a, b| (a - b) * b);
}

/// `ptq = rot.apply(q)` once per query and `ptc = rot.apply(c)` once per centroid; cosine callers
/// pass `q / |q|`. IP and cosine pick the centred form only when `|q - c| < |q|` (a tie is
/// uncentred); L2 is always centred. `seed` is the index seed and `list` the list number: the
/// rounding offsets `u_i` are a pure function of both.
#[cfg_attr(not(test), allow(dead_code))] // tests; the scan refills a `QueryPrep` with `prepare_query_into`
pub(crate) fn prepare_query(
    metric: Metric,
    q: &[f32],
    ptq: &[f32],
    c: &[f32],
    ptc: &[f32],
    seed: u64,
    list: u32,
) -> QueryPrep {
    let mut prep = QueryPrep::new(ptq.len());
    prepare_query_into(&mut prep, metric, q, ptq, c, ptc, seed, list);
    prep
}

/// [`prepare_query`] into `prep`, reusing its planes: no allocation once `prep` has the right size.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_query_into(
    prep: &mut QueryPrep,
    metric: Metric,
    q: &[f32],
    ptq: &[f32],
    c: &[f32],
    ptc: &[f32],
    seed: u64,
    list: u32,
) {
    let dp = ptq.len();
    assert!(
        dp > 0 && dp.is_multiple_of(64),
        "prepare_query: padded length {dp}"
    );
    assert_eq!(ptc.len(), dp, "prepare_query: rotated centroid length");
    assert_eq!(c.len(), q.len(), "prepare_query: centroid length");
    let diff_sq = sum_sq_diff(q, c);
    let (centred, plain_sq) = match metric {
        Metric::L2 => (true, 0.0),
        Metric::Ip | Metric::Cosine => {
            let plain_sq = sum_sq(q);
            (diff_sq < plain_sq, plain_sq)
        }
    };
    prep.centred = centred;
    let scale_sq = if centred { diff_sq } else { plain_sq };
    prep.gnorm = scale_sq.sqrt();
    prep.cq = match metric {
        Metric::L2 => 0.0,
        Metric::Ip | Metric::Cosine => dot_f64(c, q) as f32,
    };
    prep.inv_sqrt_d = 1.0 / (dp as f32).sqrt();
    prep.err_scale = EPSILON / ((dp - 1) as f32).sqrt();
    (prep.v_l, prep.delta, prep.s_q) = (0.0, 0.0, 0);
    for plane in &mut prep.planes {
        plane.clear();
        plane.resize(dp / 64, 0);
    }
    prep.lut.clear();
    prep.lut.resize(dp * 4, 0);
    if prep.gnorm > 0.0 {
        quantise(prep, ptq, ptc, seed, list);
    }
}

/// The start of list `list`'s rounding stream: the index seed in its own domain, with the list
/// number folded in through one SplitMix64 step, so no two (seed, list) pairs share a stream.
fn rounding_seed(seed: u64, list: u32) -> u64 {
    let base = SplitMix64::for_domain(seed, domain::ROUNDING).next_u64();
    SplitMix64::new(base.wrapping_add(u64::from(list))).next_u64()
}

/// Bit `j` of each of the eight bytes of `x`, as the eight bits of the result (byte `k` becomes
/// bit `k`).
fn gather_bit(x: u64, j: u32) -> u64 {
    ((x >> j) & 0x0101_0101_0101_0101).wrapping_mul(0x0102_0408_1020_4080) >> 56
}

/// The 4-bit planes of `q' = (ptq - ptc) / gnorm` (`ptq / gnorm` when uncentred), with `v_l`,
/// `delta` and `s_q`. Needs `gnorm > 0`; a constant `q'` has no step and stays at level 0.
fn quantise(prep: &mut QueryPrep, ptq: &[f32], ptc: &[f32], seed: u64, list: u32) {
    if prep.centred {
        quantise_form::<true>(prep, ptq, ptc, seed, list);
    } else {
        quantise_form::<false>(prep, ptq, ptc, seed, list);
    }
}

/// One loop per form, so the choice is not re-tested per element.
fn quantise_form<const CENTRED: bool>(
    prep: &mut QueryPrep,
    ptq: &[f32],
    ptc: &[f32],
    seed: u64,
    list: u32,
) {
    let (lo, hi) = if CENTRED {
        min_max_diff(ptq, ptc)
    } else {
        min_max(ptq)
    };
    let range = hi - lo;
    let scale = QUERY_LEVELS as f32 / range;
    prep.v_l = lo / prep.gnorm;
    if !(range > 0.0 && scale.is_finite()) {
        return;
    }
    prep.delta = range / (prep.gnorm * QUERY_LEVELS as f32);
    let mut rng = SplitMix64::new(rounding_seed(seed, list));
    let mut s_q = 0;
    let [p0, p1, p2, p3] = &mut prep.planes;
    let words = ptq.as_chunks::<64>().0.iter().zip(ptc.as_chunks::<64>().0);
    let planes = p0.iter_mut().zip(p1).zip(p2).zip(p3);
    let luts = prep.lut.as_chunks_mut::<256>().0.iter_mut();
    for (((x, y), (((w0, w1), w2), w3)), lut) in words.zip(planes).zip(luts) {
        let offsets = rounding_bytes(&mut rng);
        let levels = word_levels::<CENTRED>(x, y, lo, scale, &offsets);
        build_lut_word(&levels, lut);
        let [a0, a1, a2, a3] = pack_planes(&levels);
        (*w0, *w1, *w2, *w3) = (a0, a1, a2, a3);
        s_q += a0.count_ones() + 2 * a1.count_ones() + 4 * a2.count_ones() + 8 * a3.count_ones();
    }
    prep.s_q = s_q;
}

/// The rounding bytes `u_i` of one 64-coordinate word: eight per draw.
fn rounding_bytes(rng: &mut SplitMix64) -> [u8; 64] {
    let mut bytes = [0u8; 64];
    for eight in bytes.as_chunks_mut::<8>().0 {
        *eight = rng.next_u64().to_le_bytes();
    }
    bytes
}

/// The 4-bit levels of one 64-coordinate word of `q'` (before the division by `gnorm`).
fn word_levels<const CENTRED: bool>(
    x: &[f32; 64],
    y: &[f32; 64],
    lo: f32,
    scale: f32,
    offsets: &[u8; 64],
) -> [u8; 64] {
    let mut levels = [0u8; 64];
    for (((level, offset), a), b) in levels.iter_mut().zip(offsets).zip(x).zip(y) {
        let value = if CENTRED { a - b } else { *a };
        let rounding = (f32::from(*offset) + 0.5) * (1.0 / 256.0);
        *level = ((value - lo) * scale + rounding).min(QUERY_LEVELS as f32) as u8;
    }
    levels
}

/// The 16 tables of one 64-coordinate word: group `g` (coordinates `4 g ..= 4 g + 3`, the bits of its
/// nibble in that order) is `out[16 g ..][..16]`, entry `v` the sum of the levels whose bit is set.
fn build_lut_word(levels: &[u8; 64], out: &mut [u8; 256]) {
    for (group, table) in levels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<16>().0)
    {
        let [a, b, c, d] = *group;
        let low = [0, a, b, a + b];
        let high = [0, c, d, c + d];
        for (h, quad) in high.iter().zip(table.as_chunks_mut::<4>().0) {
            for (l, slot) in low.iter().zip(quad) {
                *slot = l + h;
            }
        }
    }
}

/// Bit `j` of every level as one word per plane, coordinate `k` at bit `k`.
fn pack_planes(levels: &[u8; 64]) -> [u64; 4] {
    let mut planes = [0u64; 4];
    for (g, group) in levels.as_chunks::<8>().0.iter().enumerate() {
        let eight = u64::from_le_bytes(*group);
        for (j, plane) in planes.iter_mut().enumerate() {
            *plane |= gather_bit(eight, j as u32) << (8 * g);
        }
    }
    planes
}

/// `(est, err)`.
#[cfg_attr(not(test), allow(dead_code))] // tests; the scan calls `estimate_bits` with its `Kernel`
pub(crate) fn estimate(prep: &QueryPrep, code: &Code) -> (f32, f32) {
    estimate_bits(Kernel::detect(), prep, &code.bits, code.dot)
}

/// [`estimate`] on a stored code's words and `dot`, without a [`Code`]. The kernel is a parameter
/// so that [`Kernel::detect`] is resolved once, not per record.
pub(crate) fn estimate_bits(
    kernel: Kernel,
    prep: &QueryPrep,
    bits: &[u64],
    dot: f32,
) -> (f32, f32) {
    debug_assert_eq!(bits.len(), prep.planes[0].len());
    if dot.is_nan() || dot < MIN_DOT {
        return (0.0, f32::INFINITY);
    }
    let [p0, p1, p2, p3] = &prep.planes;
    let counts = kernel.and_popcount_4(bits, &[p0, p1, p2, p3]);
    let acc = counts[0] + 2 * counts[1] + 4 * counts[2] + 8 * counts[3];
    let pb = kernel.popcount(bits);
    estimate_from_counts(prep, acc, pb, bits.len() as i32 * 64, dot)
}

/// `(est, err)` from a record's `acc` and `pb`: the one body of the estimator, shared by the
/// record-major path and the block path so their results are the same bits. `dp` is `D`.
#[inline(always)]
fn estimate_from_counts(prep: &QueryPrep, acc: u32, pb: u32, dp: i32, dot: f32) -> (f32, f32) {
    let (est, inv_dot) = estimate_of(prep, acc, pb, dp, dot);
    (est, error_of(prep, dot, inv_dot))
}

/// `(est, 1 / dot)`; the estimate is 0 for an unusable `dot`. No early return: a block loop over this
/// compiles to vector selects (a NaN `dot` fails `>=`).
#[inline(always)]
fn estimate_of(prep: &QueryPrep, acc: u32, pb: u32, dp: i32, dot: f32) -> (f32, f32) {
    let usable = dot >= MIN_DOT;
    let signed_acc = 2 * acc as i32 - prep.s_q as i32;
    let signed_pb = 2 * pb as i32 - dp;
    let ip = (prep.delta * signed_acc as f32 + prep.v_l * signed_pb as f32) * prep.inv_sqrt_d;
    let inv_dot = 1.0 / dot;
    (if usable { ip * inv_dot } else { 0.0 }, inv_dot)
}

/// The error bound of a record, `+inf` for an unusable `dot`; `inv_dot` is `1 / dot`.
#[inline(always)]
fn error_of(prep: &QueryPrep, dot: f32, inv_dot: f32) -> f32 {
    let err = prep.err_scale * (1.0 - dot * dot).max(0.0).sqrt() * inv_dot;
    if dot >= MIN_DOT { err } else { f32::INFINITY }
}

/// `rnorm * gnorm * est`, 0 when either factor is (so an infinite `est` cannot make a NaN).
fn cross(prep: &QueryPrep, rnorm: f32, est: f32) -> f32 {
    let scale = prep.gnorm * rnorm;
    if scale == 0.0 { 0.0 } else { scale * est }
}

/// `<x, q>` up to the sign: the IP key is `-inner`, the cosine key `1 - inner`.
fn inner(prep: &QueryPrep, rc: f32, cross: f32) -> f32 {
    prep.cq + cross + if prep.centred { rc } else { 0.0 }
}

#[cfg_attr(not(test), allow(dead_code))] // tests; the scan calls `key_from_parts`
pub(crate) fn key(metric: Metric, prep: &QueryPrep, code: &Code, est: f32) -> f32 {
    key_from_parts(metric, prep, code.rnorm, code.rc, est)
}

/// [`key`] from a record's stored `rnorm` and `rc`, without a [`Code`]: what the scan evaluates
/// per record, next to [`estimate_bits`].
pub(crate) fn key_from_parts(
    metric: Metric,
    prep: &QueryPrep,
    rnorm: f32,
    rc: f32,
    est: f32,
) -> f32 {
    let cross = cross(prep, rnorm, est);
    match metric {
        Metric::L2 => prep.gnorm * prep.gnorm + rnorm * rnorm - 2.0 * cross,
        Metric::Ip => -inner(prep, rc, cross),
        Metric::Cosine => 1.0 - inner(prep, rc, cross),
    }
}

#[cfg_attr(not(test), allow(dead_code))] // tests; the scan calls `lo_from_parts`
pub(crate) fn lo(metric: Metric, prep: &QueryPrep, code: &Code, est: f32, err: f32) -> f32 {
    lo_from_parts(metric, prep, code.rnorm, code.rc, est, err)
}

/// [`lo`] from a record's stored `rnorm` and `rc`: the key at `est + err`, the best distance the
/// record could have if the estimate is off by `err` (clamped at 0 for L2).
#[inline(always)]
pub(crate) fn lo_from_parts(
    metric: Metric,
    prep: &QueryPrep,
    rnorm: f32,
    rc: f32,
    est: f32,
    err: f32,
) -> f32 {
    let optimistic = key_from_parts(metric, prep, rnorm, rc, est + err);
    match metric {
        Metric::L2 => optimistic.max(0.0),
        Metric::Ip | Metric::Cosine => optimistic,
    }
}

/// A block's records as the scan reads them off a page: for every live lane the kernel's `acc`
/// and the stored `pb`, `dot`, `rnorm` and `rc`, all as long as the number of live lanes.
pub(crate) struct BlockInput<'a> {
    pub(crate) acc: &'a [u32],
    pub(crate) pb: &'a [u16],
    pub(crate) dot: &'a [f32],
    pub(crate) rnorm: &'a [f32],
    pub(crate) rc: &'a [f32],
}

/// The first pass over a block: `est` (the key, ascending is better) of every live lane, with the
/// estimate and error it came from kept for [`score_block_bounds`]. Written as short passes over the
/// lanes, each of which the compiler turns into vector code; the arithmetic per lane is that of the
/// record-major path ([`estimate_bits`], [`key_from_parts`]), operation for operation, so the results
/// are the same bits.
pub(crate) struct BlockScores {
    pub(crate) est: [f32; 64],
    e: [f32; 64],
    inv_dot: [f32; 64],
}

impl BlockScores {
    pub(crate) fn new() -> BlockScores {
        BlockScores {
            est: [0.0; 64],
            e: [0.0; 64],
            inv_dot: [0.0; 64],
        }
    }
}

/// Fills `scores.est[..n]` for the `n` live lanes of `input` (`dp` is `D`).
pub(crate) fn score_block_keys(
    metric: Metric,
    prep: &QueryPrep,
    dp: usize,
    input: &BlockInput<'_>,
    scores: &mut BlockScores,
) {
    let n = input.acc.len();
    let (acc, pb, dot) = (&input.acc[..n], &input.pb[..n], &input.dot[..n]);
    let (rnorm, rc) = (&input.rnorm[..n], &input.rc[..n]);
    let dp = dp as i32;
    let (e, inv_dot, est) = (
        &mut scores.e[..n],
        &mut scores.inv_dot[..n],
        &mut scores.est[..n],
    );
    for i in 0..n {
        (e[i], inv_dot[i]) = estimate_of(prep, acc[i], u32::from(pb[i]), dp, dot[i]);
    }
    match metric {
        Metric::L2 => keys::<L2>(prep, rnorm, rc, e, est),
        Metric::Ip => keys::<IP>(prep, rnorm, rc, e, est),
        Metric::Cosine => keys::<COSINE>(prep, rnorm, rc, e, est),
    }
}

/// The second pass: `lo` (the key at `est + margin * err`, see [`PRUNE_MARGIN`]) of the lanes
/// [`score_block_keys`] scored, into `lo[..n]`.
pub(crate) fn score_block_bounds(
    metric: Metric,
    prep: &QueryPrep,
    margin: f32,
    input: &BlockInput<'_>,
    scores: &BlockScores,
    lo: &mut [f32],
) {
    let n = input.acc.len();
    let (rnorm, rc, dot, lo) = (
        &input.rnorm[..n],
        &input.rc[..n],
        &input.dot[..n],
        &mut lo[..n],
    );
    let (e, inv_dot) = (&scores.e[..n], &scores.inv_dot[..n]);
    let mut err = [0f32; 64];
    let err = &mut err[..n];
    for i in 0..n {
        err[i] = error_of(prep, dot[i], inv_dot[i]);
    }
    match metric {
        Metric::L2 => bounds::<L2>(prep, margin, rnorm, rc, e, err, lo),
        Metric::Ip => bounds::<IP>(prep, margin, rnorm, rc, e, err, lo),
        Metric::Cosine => bounds::<COSINE>(prep, margin, rnorm, rc, e, err, lo),
    }
}

/// Both passes: `est` of every live lane, and `lo` when asked for: exactly the values
/// [`estimate_bits`] and [`key_from_parts`] / [`lo_from_parts`] give the record on its own.
#[cfg_attr(not(test), allow(dead_code))] // tests; the scan runs the two passes itself
pub(crate) fn score_block(
    metric: Metric,
    prep: &QueryPrep,
    dp: usize,
    margin: f32,
    input: &BlockInput<'_>,
    est: &mut [f32],
    lo: Option<&mut [f32]>,
) {
    let n = input.acc.len();
    let mut scores = BlockScores::new();
    score_block_keys(metric, prep, dp, input, &mut scores);
    est[..n].copy_from_slice(&scores.est[..n]);
    if let Some(lo) = lo {
        score_block_bounds(metric, prep, margin, input, &scores, lo);
    }
}

const L2: u8 = 0;
const IP: u8 = 1;
const COSINE: u8 = 2;

/// [`key_from_parts`] for one metric fixed at compile time.
#[inline(always)]
fn key_lane<const M: u8>(prep: &QueryPrep, rnorm: f32, rc: f32, est: f32) -> f32 {
    let cross = cross(prep, rnorm, est);
    match M {
        L2 => prep.gnorm * prep.gnorm + rnorm * rnorm - 2.0 * cross,
        IP => -inner(prep, rc, cross),
        _ => 1.0 - inner(prep, rc, cross),
    }
}

#[inline(always)]
fn keys<const M: u8>(prep: &QueryPrep, rnorm: &[f32], rc: &[f32], e: &[f32], est: &mut [f32]) {
    for i in 0..e.len() {
        est[i] = key_lane::<M>(prep, rnorm[i], rc[i], e[i]);
    }
}

#[inline(always)]
fn bounds<const M: u8>(
    prep: &QueryPrep,
    margin: f32,
    rnorm: &[f32],
    rc: &[f32],
    e: &[f32],
    err: &[f32],
    lo: &mut [f32],
) {
    for i in 0..e.len() {
        let optimistic = key_lane::<M>(prep, rnorm[i], rc[i], e[i] + margin * err[i]);
        lo[i] = if M == L2 {
            optimistic.max(0.0)
        } else {
            optimistic
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local::{
        kernel,
        rotation::{Rotation, padded_dim},
        testkit::{
            ErrorStats, Rng, exact_est, exact_key, oracle_bit, oracle_centred, oracle_encode,
            oracle_err, oracle_rotation, structured_unit_vectors, to_f32, to_f64, unit64,
        },
    };

    const SEED: u64 = 0x5EED_0002;

    /// A metric and the geometry of its queries: `far` queries are much shorter than the distance
    /// to the centroid, so IP and cosine take the uncentred form; otherwise they sit near the centroid.
    #[derive(Clone, Copy, Debug)]
    struct Case {
        metric: Metric,
        far: bool,
    }

    const L2: Case = Case {
        metric: Metric::L2,
        far: false,
    };
    const IP_NEAR: Case = Case {
        metric: Metric::Ip,
        far: false,
    };
    const IP_FAR: Case = Case {
        metric: Metric::Ip,
        far: true,
    };
    const COSINE_NEAR: Case = Case {
        metric: Metric::Cosine,
        far: false,
    };
    const COSINE_FAR: Case = Case {
        metric: Metric::Cosine,
        far: true,
    };
    const CASES: [Case; 3] = [L2, IP_NEAR, IP_FAR];
    const ALL_CASES: [Case; 5] = [L2, IP_NEAR, IP_FAR, COSINE_NEAR, COSINE_FAR];

    impl Case {
        fn centred(self) -> bool {
            self.metric == Metric::L2 || !self.far
        }

        fn centre_norm(self) -> f64 {
            if self.metric == Metric::Cosine {
                1.0
            } else {
                10.0
            }
        }
    }

    fn random_vec(rng: &mut Rng, d: usize, norm: f64) -> Vec<f32> {
        let v = rng.unit(d);
        to_f32(&v.iter().map(|x| x * norm).collect::<Vec<_>>())
    }

    fn bit(bits: &[u64], i: usize) -> bool {
        bits[i / 64] >> (i % 64) & 1 == 1
    }

    fn prep_for(
        metric: Metric,
        rot: &Rotation,
        q: &[f32],
        c: &[f32],
        ptc: &[f32],
        list: u32,
    ) -> QueryPrep {
        prepare_query(metric, q, &rot.apply(q), c, ptc, SEED, list)
    }

    fn quantised_levels(prep: &QueryPrep) -> Vec<u32> {
        let dp = prep.planes[0].len() * 64;
        (0..dp)
            .map(|i| (0..4).map(|j| (bit(&prep.planes[j], i) as u32) << j).sum())
            .collect()
    }

    /// `c` (norm `case.centre_norm()`), a vector `x` near it and a query `q` of the case's geometry;
    /// cosine inputs are unit vectors. The geometry is checked against the oracle's rule.
    fn fixture(case: Case, rng: &mut Rng, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let cn = case.centre_norm();
        let c = random_vec(rng, d, cn);
        let x: Vec<f32> = c
            .iter()
            .map(|v| *v + rng.normal() as f32 * 0.2 * cn as f32 / (d as f32).sqrt())
            .collect();
        let q: Vec<f32> = if case.far {
            random_vec(rng, d, 0.1 * cn)
        } else {
            c.iter()
                .map(|v| 0.8 * *v + rng.normal() as f32 * 0.05 * cn as f32 / (d as f32).sqrt())
                .collect()
        };
        let (x, q) = if case.metric == Metric::Cosine {
            (to_f32(&unit64(&to_f64(&x))), to_f32(&unit64(&to_f64(&q))))
        } else {
            (x, q)
        };
        assert_eq!(
            oracle_centred(case.metric, &c, &q),
            case.centred(),
            "{case:?}: the fixture geometry must be unambiguous"
        );
        (c, x, q)
    }

    #[test]
    fn err_is_1_9_sigma_with_the_padded_dimension() {
        assert_eq!(EPSILON, 1.9);
        for d in [3usize, 65, 100, 128, 768, 800] {
            let dp = padded_dim(d);
            let rot = Rotation::new(SEED, d);
            let mut rng = Rng::new(d as u64);
            let (c, q) = (random_vec(&mut rng, d, 1.0), random_vec(&mut rng, d, 1.0));
            let prep = prep_for(Metric::L2, &rot, &q, &c, &rot.apply(&c), 0);
            for dot in [0.2f32, 0.5, 0.7, 0.8, 0.95] {
                let code = Code {
                    bits: vec![0; dp / 64],
                    rnorm: 1.0,
                    dot,
                    rc: 0.0,
                };
                let (_, err) = estimate(&prep, &code);
                let want = oracle_err(dot as f64, dp);
                assert!(
                    (err as f64 - want).abs() <= 1e-5 * want,
                    "d = {d}, dot = {dot}: err {err} vs {want}"
                );
            }
            let exact = Code {
                bits: vec![0; dp / 64],
                rnorm: 1.0,
                dot: 1.0,
                rc: 0.0,
            };
            assert!(
                estimate(&prep, &exact).1.abs() <= 1e-6,
                "d = {d}: dot = 1 has no error"
            );
        }
    }

    #[test]
    fn kernel_ip_equals_the_float_inner_product_of_the_code_and_the_quantised_query() {
        for case in CASES {
            for d in [64usize, 65, 100, 128, 768, 800, 1024] {
                let dp = padded_dim(d);
                let rot = Rotation::new(SEED, d);
                let mut rng = Rng::new(d as u64 + 1);
                for trial in 0..6u32 {
                    let (c, _, q) = fixture(case, &mut rng, d);
                    let prep = prep_for(case.metric, &rot, &q, &c, &rot.apply(&c), trial);
                    let levels = quantised_levels(&prep);
                    for _ in 0..12 {
                        let bits: Vec<u64> = (0..dp / 64).map(|_| rng.u64()).collect();
                        let dot = 0.5 + 0.5 * rng.uniform() as f32;
                        let code = Code {
                            bits,
                            rnorm: 1.0,
                            dot,
                            rc: 0.0,
                        };
                        let (est, _) = estimate(&prep, &code);
                        let reference: f64 = (0..dp)
                            .map(|i| {
                                let xbar = if bit(&code.bits, i) { 1.0 } else { -1.0 }
                                    / (dp as f64).sqrt();
                                xbar * (prep.v_l as f64 + prep.delta as f64 * levels[i] as f64)
                            })
                            .sum();
                        let got = est as f64 * dot as f64;
                        assert!(
                            (got - reference).abs() <= 1e-5,
                            "{case:?}, d = {d}: kernel ip {got} vs float {reference}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn kernel_counts_compose_into_the_estimator_inputs() {
        let d = 256;
        let rot = Rotation::new(SEED, d);
        let mut rng = Rng::new(5);
        let (c, q) = (random_vec(&mut rng, d, 1.0), random_vec(&mut rng, d, 1.0));
        let prep = prep_for(Metric::L2, &rot, &q, &c, &rot.apply(&c), 3);
        let levels = quantised_levels(&prep);
        let bits: Vec<u64> = (0..d / 64).map(|_| rng.u64()).collect();
        let planes: Vec<&[u64]> = prep.planes.iter().map(Vec::as_slice).collect();
        let counts = kernel::and_popcount_4(&bits, &planes);
        let acc: u32 = (0..4).map(|j| counts[j] << j).sum();
        let want: u32 = (0..d).filter(|&i| bit(&bits, i)).map(|i| levels[i]).sum();
        assert_eq!(acc, want);
        assert_eq!(
            kernel::popcount(&bits),
            (0..d).filter(|&i| bit(&bits, i)).count() as u32
        );
    }

    #[test]
    fn query_quantisation_is_four_bit_within_one_step_and_reproducible() {
        for case in CASES {
            for d in [64usize, 100, 768] {
                let rot = Rotation::new(SEED, d);
                let dp = padded_dim(d);
                let mut rng = Rng::new(d as u64 + 9);
                let (c, _, q) = fixture(case, &mut rng, d);
                let (ptc, ptq) = (rot.apply(&c), rot.apply(&q));
                let prep = prepare_query(case.metric, &q, &ptq, &c, &ptc, SEED, 7);
                assert_eq!(
                    prep,
                    prepare_query(case.metric, &q, &ptq, &c, &ptc, SEED, 7),
                    "{case:?}: same (seed, list)"
                );
                assert_eq!(prep.centred, case.centred(), "{case:?}: form");

                let g: Vec<f64> = q
                    .iter()
                    .zip(&c)
                    .map(|(a, b)| *a as f64 - if prep.centred { *b as f64 } else { 0.0 })
                    .collect();
                let gnorm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
                assert!(
                    (prep.gnorm as f64 - gnorm).abs() <= 1e-5 * gnorm,
                    "{case:?}: gnorm {} vs {gnorm}",
                    prep.gnorm
                );
                if case.metric != Metric::L2 {
                    let cq: f64 = c.iter().zip(&q).map(|(a, b)| *a as f64 * *b as f64).sum();
                    assert!(
                        (prep.cq as f64 - cq).abs() <= 1e-5 * (1.0 + cq.abs()),
                        "{case:?}: cq"
                    );
                }

                let qp: Vec<f64> = ptq
                    .iter()
                    .zip(&ptc)
                    .map(|(a, b)| (*a as f64 - if prep.centred { *b as f64 } else { 0.0 }) / gnorm)
                    .collect();
                let (lo_v, hi_v) = qp
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), v| {
                        (l.min(*v), h.max(*v))
                    });
                let delta = (hi_v - lo_v) / 15.0;
                assert!((prep.v_l as f64 - lo_v).abs() <= 1e-5 * delta * 15.0, "v_l");
                assert!((prep.delta as f64 - delta).abs() <= 1e-4 * delta, "delta");

                let levels = quantised_levels(&prep);
                assert_eq!(levels.len(), dp);
                assert_eq!(
                    prep.s_q,
                    levels.iter().sum::<u32>(),
                    "s_q is the sum of the levels"
                );
                for (i, (&level, &v)) in levels.iter().zip(&qp).enumerate() {
                    assert!(level <= 15);
                    let dequantised = prep.v_l as f64 + prep.delta as f64 * level as f64;
                    assert!(
                        (dequantised - v).abs() <= delta * 1.001,
                        "{case:?}, d = {d}, coordinate {i}: {dequantised} vs {v}"
                    );
                }
                let argmin = qp.iter().position(|v| *v == lo_v).expect("min");
                let argmax = qp.iter().position(|v| *v == hi_v).expect("max");
                assert_eq!(levels[argmin], 0, "the minimum maps to level 0");
                assert_eq!(levels[argmax], 15, "the maximum maps to level 15");
            }
        }
    }

    #[test]
    fn each_list_takes_the_form_with_the_smaller_scale() {
        let d = 100;
        let rot = Rotation::new(SEED, d);
        let mut rng = Rng::new(61);
        let c_hat = rng.unit(d);
        let centre = |norm: f64| to_f32(&c_hat.iter().map(|v| v * norm).collect::<Vec<_>>());
        let orthogonal = {
            let z = rng.normals(d);
            let along: f64 = z.iter().zip(&c_hat).map(|(a, b)| a * b).sum();
            unit64(
                &z.iter()
                    .zip(&c_hat)
                    .map(|(a, b)| a - along * b)
                    .collect::<Vec<_>>(),
            )
        };
        let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let along_and_across = |a: f64, b: f64| -> Vec<f32> {
            to_f32(
                &c_hat
                    .iter()
                    .zip(&orthogonal)
                    .map(|(u, w)| a * u + b * w)
                    .collect::<Vec<_>>(),
            )
        };
        for metric in [Metric::Ip, Metric::Cosine] {
            let cn = if metric == Metric::Cosine { 1.0 } else { 10.0 };
            let c = centre(cn);
            let ptc = rot.apply(&c);
            let other = random_vec(&mut rng, d, cn);
            let mut uncentred_planes = None;
            for fraction in [0.1, 0.45, 0.55, 0.9, 1.0, 2.0] {
                let q = along_and_across(fraction * cn, 0.2 * cn);
                let prep = prepare_query(metric, &q, &rot.apply(&q), &c, &ptc, SEED, 5);
                let diff = q.iter().zip(&c).map(|(a, b)| a - b).collect::<Vec<f32>>();
                let (scale_centred, scale_plain) = (norm(&diff), norm(&q));
                let want_centred = scale_centred < scale_plain;
                assert_eq!(
                    prep.centred, want_centred,
                    "{metric:?}: |q - c| = {scale_centred:.3}, |q| = {scale_plain:.3}"
                );
                let want = scale_centred.min(scale_plain);
                assert!(
                    (prep.gnorm as f64 - want).abs() <= 1e-5 * want,
                    "{metric:?}: gnorm {} is the smaller scale {want}",
                    prep.gnorm
                );
                if !prep.centred {
                    uncentred_planes = Some((q.clone(), prep.planes.clone()));
                }
            }
            let (q, planes) = uncentred_planes.expect("an uncentred query was swept");
            let flipped = prepare_query(
                metric,
                &q,
                &rot.apply(&q),
                &other,
                &rot.apply(&other),
                SEED,
                5,
            );
            assert_eq!(
                flipped.planes, planes,
                "{metric:?}: an uncentred list's planes do not depend on the centroid"
            );
        }

        let c = centre(10.0);
        let far = along_and_across(1.0, 0.5);
        let prep = prepare_query(
            Metric::L2,
            &far,
            &rot.apply(&far),
            &c,
            &rot.apply(&c),
            SEED,
            0,
        );
        let diff = far.iter().zip(&c).map(|(a, b)| a - b).collect::<Vec<f32>>();
        assert!(prep.centred, "L2 is always centred");
        assert!((prep.gnorm as f64 - norm(&diff)).abs() <= 1e-4 * norm(&diff));
    }

    #[test]
    fn encode_matches_the_f64_oracle_bit_for_bit() {
        for d in [3usize, 64, 65, 100, 128, 768, 800] {
            let (rot, oracle) = (Rotation::new(SEED, d), oracle_rotation(SEED, d));
            let mut rng = Rng::new(d as u64 + 2);
            let c = random_vec(&mut rng, d, 2.0);
            for _ in 0..6 {
                let x: Vec<f32> = c
                    .iter()
                    .map(|v| *v + rng.normal() as f32 * 0.7 / (d as f32).sqrt())
                    .collect();
                let code = encode(&rot, &x, &c);
                let want = oracle_encode(&oracle, &x, &c);
                assert_eq!(code.bits.len(), padded_dim(d) / 64, "d = {d}");
                let sqrt_d = (padded_dim(d) as f64).sqrt();
                let mut checked = 0;
                for i in 0..padded_dim(d) {
                    if want.rotated[i].abs() * sqrt_d >= 1e-3 {
                        checked += 1;
                        assert_eq!(
                            bit(&code.bits, i),
                            oracle_bit(&want.bits, i),
                            "d = {d}, bit {i}"
                        );
                    }
                }
                assert!(
                    checked + 8 >= padded_dim(d),
                    "d = {d}: only {checked} bits were far enough from zero"
                );
                assert!(
                    (code.rnorm as f64 - want.rnorm).abs() <= 1e-5 * want.rnorm,
                    "rnorm"
                );
                assert!(
                    (code.dot as f64 - want.dot).abs() <= 1e-5,
                    "d = {d}: dot {} vs {}",
                    code.dot,
                    want.dot
                );
                assert!(
                    (code.rc as f64 - want.rc).abs() <= 1e-5 * (1.0 + want.rc.abs()),
                    "rc {} vs {}",
                    code.rc,
                    want.rc
                );
            }
        }
    }

    #[test]
    fn encode_depends_on_the_direction_of_the_residual_only() {
        let d = 100;
        let rot = Rotation::new(SEED, d);
        let mut rng = Rng::new(31);
        let c = random_vec(&mut rng, d, 4.0);
        let r = random_vec(&mut rng, d, 1.0);
        let along =
            |scale: f32| -> Vec<f32> { c.iter().zip(&r).map(|(a, b)| a + scale * b).collect() };
        let base = encode(&rot, &along(1.0), &c);
        let scaled = encode(&rot, &along(3.0), &c);
        assert_eq!(scaled.bits, base.bits);
        assert!((scaled.dot - base.dot).abs() <= 1e-5);
        assert!((scaled.rnorm / base.rnorm - 3.0).abs() <= 1e-5);
        assert!((scaled.rc / base.rc - 3.0).abs() <= 1e-4);
        let opposite = encode(&rot, &along(-1.0), &c);
        for i in 0..padded_dim(d) {
            assert_ne!(
                bit(&opposite.bits, i),
                bit(&base.bits, i),
                "bit {i} flips with the sign of r"
            );
        }
        assert!((opposite.dot - base.dot).abs() <= 1e-5);
        assert!((opposite.rnorm - base.rnorm).abs() <= 1e-5);
        assert!((opposite.rc + base.rc).abs() <= 1e-4 * base.rc.abs());
    }

    #[test]
    fn encode_keeps_the_mass_of_axis_aligned_residuals() {
        for (d, floor) in [(64usize, 0.65), (128, 0.7), (768, 0.72), (1024, 0.72)] {
            for seed in [SEED, 0xBEEF, 7] {
                let rot = Rotation::new(seed, d);
                let c = vec![0.25f32; d];
                for (name, o) in structured_unit_vectors(d, 5) {
                    let x: Vec<f32> = c.iter().zip(&o).map(|(a, b)| a + 2.0 * *b as f32).collect();
                    let code = encode(&rot, &x, &c);
                    assert!(
                        code.dot >= floor,
                        "d = {d}, seed = {seed:#x}, {name}: dot = {} < {floor}",
                        code.dot
                    );
                }
            }
        }
    }

    #[test]
    fn keys_are_the_exact_distances_when_est_is_exact() {
        for d in [65usize, 128, 200] {
            let rot = Rotation::new(SEED, d);
            let mut rng = Rng::new(d as u64 + 4);
            for case in ALL_CASES {
                for trial in 0..8u32 {
                    let (c, x, q) = fixture(case, &mut rng, d);
                    let code = encode(&rot, &x, &c);
                    let prep = prep_for(case.metric, &rot, &q, &c, &rot.apply(&c), trial);
                    assert_eq!(prep.centred, case.centred(), "{case:?}: form");
                    let est = exact_est(case.metric, &x, &c, &q) as f32;
                    let got = key(case.metric, &prep, &code, est) as f64;
                    let want = exact_key(case.metric, &x, &q);
                    let scale = 1.0
                        + (prep.cq.abs()
                            + code.rc.abs()
                            + code.rnorm * prep.gnorm
                            + prep.gnorm * prep.gnorm
                            + code.rnorm * code.rnorm) as f64;
                    assert!(
                        (got - want).abs() <= 1e-5 * scale,
                        "{case:?}, d = {d}: key {got} vs exact {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn lo_is_the_key_at_est_plus_err_clamped_for_l2() {
        let d = 128;
        let rot = Rotation::new(SEED, d);
        let mut rng = Rng::new(77);
        let mut clamped = 0;
        for trial in 0..20u32 {
            for case in ALL_CASES {
                let (c, x, mut q) = fixture(case, &mut rng, d);
                if case.metric == Metric::L2 && trial % 2 == 1 {
                    q = x.clone();
                }
                let code = encode(&rot, &x, &c);
                let prep = prep_for(case.metric, &rot, &q, &c, &rot.apply(&c), trial);
                let (est, err) = estimate(&prep, &code);
                assert!(err > 0.0);
                let k = key(case.metric, &prep, &code, est);
                let l = lo(case.metric, &prep, &code, est, err);
                let mut want = key(case.metric, &prep, &code, est + err);
                if case.metric == Metric::L2 {
                    want = want.max(0.0);
                }
                assert!(
                    (l - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "{case:?}: lo {l} vs {want}"
                );
                let ceiling = if case.metric == Metric::L2 {
                    k.max(0.0)
                } else {
                    k
                };
                assert!(
                    l <= ceiling + 1e-5 * (1.0 + k.abs()),
                    "{case:?}: lo {l} above key {k}"
                );
                if case.metric == Metric::L2 {
                    assert!(l >= 0.0);
                    clamped += (want == 0.0) as usize;
                }
            }
        }
        assert!(
            clamped >= 3,
            "the fixture must reach the L2 clamp, reached it {clamped} times"
        );
    }

    #[derive(Clone, Copy)]
    enum Shape {
        Isotropic,
        LowRank,
        Sparse,
        Far,
    }

    fn pairs_for(
        shape: Shape,
        rng: &mut Rng,
        d: usize,
        clusters: usize,
        per_cluster: usize,
    ) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let mut out = Vec::new();
        for _ in 0..clusters {
            let c = random_vec(
                rng,
                d,
                if matches!(shape, Shape::Far) {
                    10.0
                } else {
                    3.0
                },
            );
            let basis: Vec<Vec<f64>> = (0..6).map(|_| rng.unit(d)).collect();
            let support: Vec<usize> = (0..24).map(|_| rng.below(d)).collect();
            for _ in 0..per_cluster {
                let draw = |rng: &mut Rng| -> Vec<f32> {
                    let mut v: Vec<f64> = to_f64(&c);
                    match shape {
                        Shape::Isotropic | Shape::Far => {
                            for vi in v.iter_mut() {
                                *vi += rng.normal() / (d as f64).sqrt();
                            }
                        }
                        Shape::LowRank => {
                            for b in &basis {
                                let w = rng.normal() / 6f64.sqrt();
                                for (vi, bi) in v.iter_mut().zip(b) {
                                    *vi += w * bi;
                                }
                            }
                            for vi in v.iter_mut() {
                                *vi += 0.05 * rng.normal() / (d as f64).sqrt();
                            }
                        }
                        Shape::Sparse => {
                            for &j in &support {
                                v[j] += rng.normal() / 24f64.sqrt();
                            }
                        }
                    }
                    to_f32(&v)
                };
                let x = draw(rng);
                let q = if matches!(shape, Shape::Far) {
                    random_vec(rng, d, 1.0)
                } else {
                    draw(rng)
                };
                out.push((c.clone(), x, q));
            }
        }
        out
    }

    /// Estimates every pair with the form the oracle's rule picks for it, asserting the
    /// implementation picked the same one.
    fn measure(
        metric: Metric,
        rot: &Rotation,
        pairs: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
        expect_centred: bool,
    ) -> ErrorStats {
        let mut stats = ErrorStats::new();
        let mut last: Option<(Vec<f32>, Vec<f32>)> = None;
        for (i, (c, x, q)) in pairs.iter().enumerate() {
            let ptc = match &last {
                Some((lc, p)) if lc == c => p.clone(),
                _ => rot.apply(c),
            };
            last = Some((c.clone(), ptc.clone()));
            let code = encode(rot, x, c);
            let prep = prep_for(metric, rot, q, c, &ptc, i as u32);
            assert_eq!(
                prep.centred, expect_centred,
                "{metric:?}: pair {i} took the wrong form"
            );
            let (est, err) = estimate(&prep, &code);
            stats.add(
                est as f64,
                err as f64,
                code.dot as f64,
                exact_est(metric, x, c, q),
            );
        }
        stats
    }

    struct Bounds {
        ratio: (f64, f64),
        coverage: (f64, f64),
        coverage_3: f64,
        err_coverage: (f64, f64),
    }

    const STANDARD: Bounds = Bounds {
        ratio: (0.8, 1.2),
        coverage: (0.90, 0.98),
        coverage_3: 0.985,
        err_coverage: (0.90, 0.98),
    };

    /// At small angles the 4-bit query noise, which does not shrink with the angle, is a larger share of the error.
    const SHALLOW: Bounds = Bounds {
        ratio: (0.8, 1.3),
        coverage: (0.87, 0.98),
        coverage_3: 0.975,
        err_coverage: (0.90, 1.0),
    };

    fn assert_calibrated(label: &str, stats: &ErrorStats, bounds: Bounds) {
        let summary = format!(
            "{label}: n = {}, mean error = {:+.5} (se {:.5}), sigma ratio = {:.3}, coverage(1.9 sigma) = {:.4}, \
             coverage(3 sigma) = {:.4}, coverage(err) = {:.4}, one-sided = {:.4}, mean dot = {:.4}, min dot = {:.4}",
            stats.n,
            stats.mean_error(),
            stats.mean_error_se(),
            stats.sigma_ratio(),
            stats.coverage_1_9(),
            stats.coverage_3(),
            stats.coverage_err(),
            stats.one_sided(),
            stats.mean_dot(),
            stats.min_dot
        );
        eprintln!("{summary}");
        assert!(
            stats.mean_error_se() <= 0.0007,
            "population too small to test the mean: {summary}"
        );
        assert!(stats.mean_error().abs() <= 0.002, "biased: {summary}");
        assert!(
            (bounds.ratio.0..=bounds.ratio.1).contains(&stats.sigma_ratio()),
            "err is not 1.9 sigma: {summary}"
        );
        assert!(
            (bounds.coverage.0..=bounds.coverage.1).contains(&stats.coverage_1_9()),
            "coverage of 1.9 sigma: {summary}"
        );
        assert!(
            (bounds.err_coverage.0..=bounds.err_coverage.1).contains(&stats.coverage_err()),
            "coverage of +-err as stored: {summary}"
        );
        assert!(
            stats.coverage_3() >= bounds.coverage_3,
            "coverage of 3 sigma: {summary}"
        );
        assert!(stats.one_sided() >= 0.95, "lower bound coverage: {summary}");
        assert!(
            (0.76..=0.83).contains(&stats.mean_dot()),
            "dot is |o'|_1 / sqrt(D) ~ 0.8: {summary}"
        );
        assert!(stats.min_dot >= 0.55, "a code with little mass: {summary}");
    }

    fn pairs_needed(d: usize) -> usize {
        match padded_dim(d) {
            64..=128 => 20_000,
            129..=512 => 8_000,
            _ => 5_000,
        }
    }

    /// The centred form (L2; IP with queries near the centroids computes the same estimate) on `shape`;
    /// `Far` queries are also calibrated as the uncentred IP form.
    fn calibrate(label: &str, shape: Shape, dims: &[usize], clusters: usize, seed: u64) {
        let forms: &[(Metric, bool)] = if matches!(shape, Shape::Far) {
            &[(Metric::L2, true), (Metric::Ip, false)]
        } else {
            &[(Metric::L2, true)]
        };
        for &(metric, centred) in forms {
            for &d in dims {
                let rot = Rotation::new(SEED, d);
                let mut rng = Rng::new(d as u64 + seed);
                let n = pairs_needed(d);
                let pairs = pairs_for(shape, &mut rng, d, clusters, n / clusters);
                assert_calibrated(
                    &format!("{label} {metric:?} (centred: {centred}) d = {d}"),
                    &measure(metric, &rot, &pairs, centred),
                    STANDARD,
                );
            }
        }
    }

    #[test]
    fn estimator_is_calibrated_on_isotropic_pairs() {
        calibrate(
            "isotropic",
            Shape::Isotropic,
            &[65, 128, 768, 800, 1024],
            4,
            100,
        );
    }

    #[test]
    fn estimator_is_calibrated_on_low_rank_clusters() {
        calibrate("low-rank", Shape::LowRank, &[128, 768, 832, 1024], 8, 200);
    }

    #[test]
    fn estimator_is_calibrated_on_axis_aligned_sparse_clusters() {
        calibrate("sparse", Shape::Sparse, &[128, 768, 832, 1024], 8, 300);
    }

    #[test]
    fn estimator_is_calibrated_for_queries_far_from_a_centroid_of_norm_10() {
        calibrate("far", Shape::Far, &[128, 768, 1024], 4, 400);
    }

    #[test]
    fn estimator_is_calibrated_for_structured_vectors_at_fixed_angles() {
        for case in [L2, IP_FAR] {
            for (d, reps) in [(128usize, 1000usize), (1024, 150)] {
                let rot = Rotation::new(SEED, d);
                for degrees in [90.0f64, 60.0, 30.0] {
                    let (sin, cos) = degrees.to_radians().sin_cos();
                    let mut rng = Rng::new(d as u64 + degrees as u64);
                    let c = random_vec(&mut rng, d, 2.0);
                    let ptc = rot.apply(&c);
                    let mut stats = ErrorStats::new();
                    let mut pair = 0u32;
                    for (_, o) in structured_unit_vectors(d, 13) {
                        for _ in 0..reps {
                            let z = rng.normals(d);
                            let along: f64 = z.iter().zip(&o).map(|(a, b)| a * b).sum();
                            let w = unit64(
                                &z.iter()
                                    .zip(&o)
                                    .map(|(a, b)| a - along * b)
                                    .collect::<Vec<_>>(),
                            );
                            let direction: Vec<f64> =
                                o.iter().zip(&w).map(|(b, wv)| cos * b + sin * wv).collect();
                            let x: Vec<f32> =
                                c.iter().zip(&o).map(|(a, b)| a + 1.3 * *b as f32).collect();
                            let q: Vec<f32> = if case.centred() {
                                c.iter()
                                    .zip(&direction)
                                    .map(|(a, g)| a + 0.8 * *g as f32)
                                    .collect()
                            } else {
                                direction.iter().map(|g| 0.3 * *g as f32).collect()
                            };
                            let code = encode(&rot, &x, &c);
                            let prep = prep_for(case.metric, &rot, &q, &c, &ptc, pair);
                            pair += 1;
                            assert_eq!(prep.centred, case.centred(), "{case:?}: form");
                            let (est, err) = estimate(&prep, &code);
                            stats.add(
                                est as f64,
                                err as f64,
                                code.dot as f64,
                                exact_est(case.metric, &x, &c, &q),
                            );
                        }
                    }
                    let bounds = if degrees < 45.0 { SHALLOW } else { STANDARD };
                    assert_calibrated(
                        &format!("fixed angle {degrees} deg {case:?}, d = {d}"),
                        &stats,
                        bounds,
                    );
                }
            }
        }
    }

    #[test]
    fn keys_rank_clear_gaps_correctly() {
        for d in [65usize, 768] {
            let rot = Rotation::new(SEED, d);
            let mut rng = Rng::new(d as u64 + 400);
            let c = random_vec(&mut rng, d, 3.0);
            let ptc = rot.apply(&c);
            let q: Vec<f32> = c
                .iter()
                .map(|v| *v + rng.normal() as f32 * 1.0 / (d as f32).sqrt())
                .collect();
            let prep = prep_for(Metric::L2, &rot, &q, &c, &ptc, 0);
            let pool: Vec<(f64, f32)> = (0..300)
                .map(|_| {
                    let spread = 0.3 + 2.0 * rng.uniform();
                    let x: Vec<f32> = c
                        .iter()
                        .map(|v| *v + rng.normal() as f32 * spread as f32 / (d as f32).sqrt())
                        .collect();
                    let code = encode(&rot, &x, &c);
                    let (est, _) = estimate(&prep, &code);
                    (
                        exact_key(Metric::L2, &x, &q),
                        key(Metric::L2, &prep, &code, est),
                    )
                })
                .collect();
            let (mut agree, mut total) = (0usize, 0usize);
            for i in 0..pool.len() {
                for j in 0..i {
                    if (pool[i].0 / pool[j].0).ln().abs() > 0.4 {
                        total += 1;
                        agree += ((pool[i].0 < pool[j].0) == (pool[i].1 < pool[j].1)) as usize;
                    }
                }
            }
            let rate = agree as f64 / total as f64;
            assert!(
                total > 5_000 && rate >= 0.99,
                "d = {d}: {agree}/{total} clear gaps ranked right"
            );
        }
    }

    fn golden_vector(d: usize, salt: u64) -> Vec<f32> {
        (0..d)
            .map(|i| {
                let h = (i as u64 + 1)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(salt.wrapping_mul(0xBF58_476D_1CE4_E5B9));
                let h = (h ^ (h >> 29)).wrapping_mul(0x94D0_49BB_1331_11EB);
                ((h >> 40) % 2001) as f32 / 500.0 - 2.0
            })
            .collect()
    }

    struct GoldenCode {
        salt: u64,
        bits: &'static [u64],
        rnorm: f32,
        dot: f32,
        rc: f32,
    }

    /// `l2` is `(gnorm, v_l, delta)` of the centred L2 query; `ip_uncentred` and `ip_centred` are
    /// `(gnorm, cq, v_l, delta)` of the IP query `golden_vector(d, query_salt)` (uncentred: it is shorter
    /// than its distance to the centre) and of `c + 0.2 * golden_vector(d, query_salt + 3)` (centred).
    struct GoldenCase {
        seed: u64,
        d: usize,
        centre_salt: u64,
        query_salt: u64,
        codes: &'static [GoldenCode],
        l2: [f32; 3],
        ip_uncentred: [f32; 4],
        ip_centred: [f32; 4],
    }

    /// Derived from the f64 oracle in `testkit` (`oracle_encode`, `oracle_rotation`): vectors are
    /// `golden_vector(d, salt)`, the centre is `0.5 * golden_vector(d, centre_salt)`; the bits are
    /// word `i / 64`, bit `i % 64`. The codes are on-disk format: change them only with a format bump.
    const GOLDEN: [GoldenCase; 3] = [
        GoldenCase {
            seed: 0xC0DE_0064,
            d: 64,
            centre_salt: 11,
            query_salt: 31,
            codes: &[
                GoldenCode {
                    salt: 21,
                    bits: &[
                        0xc448_d336_377b_c704,
                        0x3c67_1e01_055d_7831,
                        0xfc30_9816_bcc8_1449,
                        0x856c_28c4_ebc7_8e04,
                    ],
                    rnorm: 10.743_924,
                    dot: 0.790_418_3,
                    rc: -25.587_784,
                },
                GoldenCode {
                    salt: 22,
                    bits: &[
                        0x0a2f_63b4_ddea_3103,
                        0x04c7_2e10_ac1d_3391,
                        0xec3f_69f0_7628_927b,
                        0xc81c_8450_418b_3838,
                    ],
                    rnorm: 10.069_814,
                    dot: 0.817_827_05,
                    rc: -20.288_637,
                },
            ],
            l2: [9.893_124, -0.167_976_92, 0.022_149_224],
            ip_uncentred: [9.117_147, 3.332_119_5, -0.144_208_12, 0.021_643_14],
            ip_centred: [1.860_881_8, 20.210_49, -0.174_304_47, 0.022_796_38],
        },
        GoldenCase {
            seed: 20_260_715,
            d: 100,
            centre_salt: 12,
            query_salt: 32,
            codes: &[
                GoldenCode {
                    salt: 23,
                    bits: &[
                        0xef47_96cb_8086_8a3e,
                        0x5e01_84fc_d578_c6a4,
                        0x2c66_e49b_9638_b77c,
                        0x46a4_2ddc_51ce_301f,
                    ],
                    rnorm: 13.960_836,
                    dot: 0.819_717_47,
                    rc: -47.170_46,
                },
                GoldenCode {
                    salt: 24,
                    bits: &[
                        0x6a43_470f_408e_c93e,
                        0x8393_851c_d47b_9241,
                        0x79c0_7e5b_4671_b2e7,
                        0x7cfc_77f3_73c0_3899,
                    ],
                    rnorm: 13.204_473,
                    dot: 0.818_810_5,
                    rc: -26.450_104,
                },
            ],
            l2: [12.590_054, -0.211_025_8, 0.026_839_973],
            ip_uncentred: [11.026_303, -2.137_048_2, -0.182_624_65, 0.023_586_966],
            ip_centred: [2.318_687_2, 33.799_5, -0.200_987_79, 0.025_486_95],
        },
        GoldenCase {
            seed: 0xFEED_0768,
            d: 768,
            centre_salt: 13,
            query_salt: 33,
            codes: &[
                GoldenCode {
                    salt: 25,
                    bits: &[
                        0xe026_3bf9_3004_ccc8,
                        0x6323_2385_1713_e0d4,
                        0xb3c4_e92a_dbd7_592b,
                        0xd863_61ab_25cb_fa25,
                        0x97ce_a607_ad20_7626,
                        0x81cf_baf3_e808_d8ae,
                        0x92e8_2f62_cf93_d02c,
                        0xd009_9d7b_f504_0339,
                        0xcbc2_631c_25fa_371a,
                        0xaf24_7af6_3767_b1ab,
                        0x9e30_cdc0_94f5_f38c,
                        0xd7a4_9ac6_01d6_f710,
                    ],
                    rnorm: 35.833_984,
                    dot: 0.792_292_4,
                    rc: -264.630_58,
                },
                GoldenCode {
                    salt: 26,
                    bits: &[
                        0xa2d6_5bea_b386_5e8b,
                        0x7ce1_389e_1723_ab6d,
                        0x9bc3_6772_3748_090b,
                        0x212c_ba1a_1147_f851,
                        0xac1c_3e73_2bdc_8b3e,
                        0x22f6_250e_0f3a_f428,
                        0x62c1_0f41_5dbb_fd7c,
                        0x3340_ddea_99e2_0637,
                        0x46f0_8957_401b_7430,
                        0x9645_50f1_5d93_ab70,
                        0x4775_8cce_3e29_d1c1,
                        0x9073_263f_0b1c_37fc,
                    ],
                    rnorm: 36.001_116,
                    dot: 0.796_293_6,
                    rc: -269.404_7,
                },
            ],
            l2: [35.770_93, -0.121_649_3, 0.016_361_395],
            ip_uncentred: [31.865_463, 0.083_027_54, -0.099_746_34, 0.013_759_513],
            ip_centred: [6.443_048, 260.650_12, -0.116_588_44, 0.014_312_566],
        },
    ];

    fn golden_centre(case: &GoldenCase) -> Vec<f32> {
        golden_vector(case.d, case.centre_salt)
            .iter()
            .map(|v| v * 0.5)
            .collect()
    }

    fn assert_close(what: &str, got: f32, want: f32, rel: f32) {
        assert!(
            (got - want).abs() <= rel * want.abs().max(1.0),
            "{what}: {got} vs pinned {want}"
        );
    }

    #[test]
    fn golden_codes_are_pinned() {
        for case in &GOLDEN {
            let rot = Rotation::new(case.seed, case.d);
            let c = golden_centre(case);
            for pinned in case.codes {
                let code = encode(&rot, &golden_vector(case.d, pinned.salt), &c);
                let label = format!(
                    "seed {:#x}, d = {}, vector {}",
                    case.seed, case.d, pinned.salt
                );
                assert_eq!(code.bits, pinned.bits, "{label}: code words");
                assert_close(&format!("{label}: rnorm"), code.rnorm, pinned.rnorm, 1e-5);
                assert_close(&format!("{label}: dot"), code.dot, pinned.dot, 1e-5);
                assert_close(&format!("{label}: rc"), code.rc, pinned.rc, 1e-5);
            }
            let q = golden_vector(case.d, case.query_salt);
            let q_near: Vec<f32> = c
                .iter()
                .zip(golden_vector(case.d, case.query_salt + 3))
                .map(|(a, b)| a + 0.2 * b)
                .collect();
            let label = format!("seed {:#x}, d = {}, query", case.seed, case.d);
            let ptc = rot.apply(&c);
            let l2 = prep_for(Metric::L2, &rot, &q, &c, &ptc, 0);
            assert!(l2.centred);
            assert_close(&format!("{label}: L2 gnorm"), l2.gnorm, case.l2[0], 1e-5);
            assert_close(&format!("{label}: L2 v_l"), l2.v_l, case.l2[1], 1e-5);
            assert_close(&format!("{label}: L2 delta"), l2.delta, case.l2[2], 1e-5);
            for metric in [Metric::Ip, Metric::Cosine] {
                for (q, centred, pinned) in [
                    (&q, false, case.ip_uncentred),
                    (&q_near, true, case.ip_centred),
                ] {
                    let prep = prep_for(metric, &rot, q, &c, &ptc, 0);
                    let label = format!("{label}: {metric:?} (centred: {centred})");
                    assert_eq!(prep.centred, centred, "{label}: form");
                    assert_close(&format!("{label}: gnorm"), prep.gnorm, pinned[0], 1e-5);
                    assert_close(&format!("{label}: cq"), prep.cq, pinned[1], 1e-5);
                    assert_close(&format!("{label}: v_l"), prep.v_l, pinned[2], 1e-5);
                    assert_close(&format!("{label}: delta"), prep.delta, pinned[3], 1e-5);
                }
            }
        }
    }

    #[test]
    fn oracle_reproduces_the_golden_constants() {
        for case in &GOLDEN {
            let oracle = oracle_rotation(case.seed, case.d);
            let c = golden_centre(case);
            for pinned in case.codes {
                let want = oracle_encode(&oracle, &golden_vector(case.d, pinned.salt), &c);
                assert!(
                    want.margin > 1e-4,
                    "golden vector too close to a bit flip: {}",
                    want.margin
                );
                assert_eq!(want.bits, pinned.bits);
                for (got, pinned) in [
                    (want.rnorm, pinned.rnorm),
                    (want.dot, pinned.dot),
                    (want.rc, pinned.rc),
                ] {
                    assert!(
                        (got - pinned as f64).abs() <= 1e-6 * got.abs().max(1.0),
                        "{got} vs {pinned}"
                    );
                }
            }
            let q = golden_vector(case.d, case.query_salt);
            assert!(!oracle_centred(Metric::Ip, &c, &q));
        }
    }

    #[test]
    fn oracle_rotation_is_orthonormal() {
        for d in [64usize, 100] {
            let oracle = oracle_rotation(SEED, d);
            let mut rng = Rng::new(d as u64);
            let (u, v) = (rng.normals(d), rng.normals(d));
            let (pu, pv) = (oracle.apply(&u), oracle.apply(&v));
            let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
            assert!((dot(&pu, &pu) - dot(&u, &u)).abs() <= 1e-9 * dot(&u, &u));
            assert!(
                (dot(&pu, &pv) - dot(&u, &v)).abs()
                    <= 1e-9 * dot(&u, &u).sqrt() * dot(&v, &v).sqrt()
            );
        }
    }

    #[test]
    fn a_vector_equal_to_the_centroid_encodes_to_the_zero_code() {
        for d in [3usize, 64, 65, 768] {
            let rot = Rotation::new(SEED, d);
            let mut rng = Rng::new(d as u64);
            for c in [vec![0.0f32; d], random_vec(&mut rng, d, 5.0)] {
                let code = encode(&rot, &c, &c);
                assert_eq!(code.bits, vec![0u64; padded_dim(d) / 64], "d = {d}");
                assert_eq!((code.rnorm, code.dot, code.rc), (0.0, 1.0, 0.0), "d = {d}");
            }
        }
    }

    #[test]
    fn the_zero_vector_against_a_centroid_is_an_ordinary_code() {
        let d = 65;
        let rot = Rotation::new(SEED, d);
        let c = random_vec(&mut Rng::new(3), d, 4.0);
        let code = encode(&rot, &vec![0.0; d], &c);
        assert!((code.rnorm - 4.0).abs() <= 1e-5);
        assert!(
            (code.rc + 16.0).abs() <= 1e-4,
            "rc = <-c, c> = -|c|^2, got {}",
            code.rc
        );
        assert!((0.6..=1.0).contains(&code.dot), "dot = {}", code.dot);
        assert!(code.bits.iter().any(|w| *w != 0));
    }

    #[test]
    fn a_query_equal_to_the_centroid_scores_every_record_at_its_own_distance() {
        for case in [L2, IP_NEAR, COSINE_NEAR] {
            for d in [65usize, 128] {
                let rot = Rotation::new(SEED, d);
                let mut rng = Rng::new(d as u64 + 6);
                let c = random_vec(&mut rng, d, case.centre_norm());
                let prep = prep_for(case.metric, &rot, &c, &c, &rot.apply(&c), 0);
                assert_eq!(prep.gnorm, 0.0, "{case:?}");
                assert!(prep.centred, "{case:?}: |q - c| = 0 is the smaller scale");
                for _ in 0..10 {
                    let x: Vec<f32> = c
                        .iter()
                        .map(|v| *v + rng.normal() as f32 / (d as f32).sqrt())
                        .collect();
                    let x = if case.metric == Metric::Cosine {
                        to_f32(&unit64(&to_f64(&x)))
                    } else {
                        x
                    };
                    let code = encode(&rot, &x, &c);
                    let (est, err) = estimate(&prep, &code);
                    assert!(
                        est.is_finite() && err.is_finite(),
                        "{case:?}, d = {d}: est {est}, err {err}"
                    );
                    let (k, l) = (
                        key(case.metric, &prep, &code, est) as f64,
                        lo(case.metric, &prep, &code, est, err) as f64,
                    );
                    let want = exact_key(case.metric, &x, &c);
                    assert!(
                        (k - want).abs() <= 1e-4 * (1.0 + want.abs()),
                        "{case:?}: key {k} vs exact {want}"
                    );
                    assert!(
                        (l - k).abs() <= 1e-4 * (1.0 + k.abs()),
                        "{case:?}: with gnorm = 0 the bound is the key"
                    );
                }
            }
        }
    }

    #[test]
    fn a_zero_query_scores_every_record_at_its_list_constant() {
        for d in [65usize, 128] {
            let rot = Rotation::new(SEED, d);
            let mut rng = Rng::new(d as u64 + 7);
            let c = random_vec(&mut rng, d, 6.0);
            let zero = vec![0.0f32; d];
            for (metric, want_key) in [(Metric::Ip, 0.0f32), (Metric::Cosine, 1.0)] {
                let prep = prep_for(metric, &rot, &zero, &c, &rot.apply(&c), 0);
                assert_eq!((prep.gnorm, prep.cq), (0.0, 0.0), "{metric:?}");
                assert!(!prep.centred, "{metric:?}: |q| = 0 is the smaller scale");
                for _ in 0..10 {
                    let x: Vec<f32> = c
                        .iter()
                        .map(|v| *v + rng.normal() as f32 / (d as f32).sqrt())
                        .collect();
                    let code = encode(&rot, &x, &c);
                    let (est, err) = estimate(&prep, &code);
                    assert!(
                        est.is_finite() && err.is_finite(),
                        "{metric:?}: est {est}, err {err}"
                    );
                    let (k, l) = (
                        key(metric, &prep, &code, est),
                        lo(metric, &prep, &code, est, err),
                    );
                    assert!(
                        (k - want_key).abs() <= 1e-6 && (l - want_key).abs() <= 1e-6,
                        "{metric:?}: key {k}, lo {l}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_constant_rotated_query_has_no_quantisation_step_and_still_estimates() {
        let d = 64;
        let q = vec![3.0f32; d];
        let zero = vec![0.0f32; d];
        for metric in [Metric::L2, Metric::Ip, Metric::Cosine] {
            let prep = prepare_query(metric, &q, &q, &zero, &zero, SEED, 0);
            let levels = quantised_levels(&prep);
            assert!(
                levels.iter().all(|l| *l == 0) && prep.s_q == 0,
                "{metric:?}: no step, no levels"
            );
            let mut rng = Rng::new(9);
            for _ in 0..8 {
                let bits = vec![rng.u64()];
                let code = Code {
                    bits,
                    rnorm: 1.0,
                    dot: 0.8,
                    rc: 0.0,
                };
                let (est, err) = estimate(&prep, &code);
                assert!(est.is_finite() && err.is_finite());
                let pb = code.bits[0].count_ones() as f64;
                let want = 0.125 * (2.0 * pb - 64.0) / 8.0 / 0.8;
                assert!(
                    (est as f64 - want).abs() <= 1e-5,
                    "{metric:?}: est {est} vs {want}"
                );
            }
        }
    }

    #[test]
    fn a_degenerate_dot_gives_zero_est_an_infinite_bound_and_no_nan() {
        let d = 64;
        let rot = Rotation::new(SEED, d);
        let mut rng = Rng::new(21);
        let mut cases = Vec::new();
        for case in CASES {
            let (c, _, q) = fixture(case, &mut rng, d);
            let ptc = rot.apply(&c);
            cases.push((case.metric, prep_for(case.metric, &rot, &q, &c, &ptc, 0)));
            cases.push((case.metric, prep_for(case.metric, &rot, &c, &c, &ptc, 0)));
            cases.push((
                case.metric,
                prep_for(case.metric, &rot, &vec![0.0; d], &c, &ptc, 0),
            ));
        }
        for (metric, prep) in cases {
            let code = Code {
                bits: (0..padded_dim(d) / 64).map(|_| rng.u64()).collect(),
                rnorm: 2.0,
                dot: 0.0,
                rc: 1.5,
            };
            let (est, err) = estimate(&prep, &code);
            assert_eq!(est, 0.0);
            assert_eq!(err, f32::INFINITY);
            let k = key(metric, &prep, &code, est);
            let l = lo(metric, &prep, &code, est, err);
            assert!(k.is_finite(), "{metric:?}: key {k}");
            assert!(
                !l.is_nan() && l <= k + 1e-4 * (1.0 + k.abs()),
                "{metric:?}: lo {l} vs key {k}"
            );
            if metric == Metric::L2 {
                assert!(l >= 0.0);
            }
        }
    }

    #[test]
    fn padding_dimensions_run_the_whole_pipeline_with_finite_values() {
        for case in CASES {
            for d in [1usize, 3, 65, 768] {
                let dp = padded_dim(d);
                let rot = Rotation::new(SEED, d);
                let mut rng = Rng::new(d as u64 + 8);
                for trial in 0..40u32 {
                    let (c, x, q) = fixture(case, &mut rng, d);
                    let code = encode(&rot, &x, &c);
                    assert_eq!(code.bits.len(), dp / 64, "d = {d}");
                    let floor = 1.0 / (dp as f32).sqrt() - 1e-6;
                    assert!(
                        code.dot >= floor && code.dot <= 1.0 + 1e-6,
                        "d = {d}: dot {}",
                        code.dot
                    );
                    let prep = prep_for(case.metric, &rot, &q, &c, &rot.apply(&c), trial);
                    let (est, err) = estimate(&prep, &code);
                    assert!(
                        est.is_finite() && err.is_finite() && err > 0.0,
                        "{case:?}, d = {d}: est {est}, err {err}"
                    );
                    let truth = exact_est(case.metric, &x, &c, &q);
                    assert!(
                        (est as f64 - truth).abs() <= 8.0 * err as f64 + 1e-3,
                        "{case:?}, d = {d}: est {est} vs exact {truth}, err {err}"
                    );
                    for key_metric in [case.metric, Metric::Cosine] {
                        assert!(key(key_metric, &prep, &code, est).is_finite());
                        assert!(lo(key_metric, &prep, &code, est, err).is_finite());
                    }
                }
            }
        }
    }

    #[test]
    fn encoding_and_query_prep_are_deterministic_across_threads() {
        let d = 768;
        let rot = Rotation::new(SEED, d);
        let mut rng = Rng::new(17);
        let c = random_vec(&mut rng, d, 10.0);
        let ptc = rot.apply(&c);
        let near: Vec<Vec<f32>> = (0..6)
            .map(|_| {
                c.iter()
                    .map(|v| 0.8 * v + rng.normal() as f32 * 0.05)
                    .collect()
            })
            .collect();
        let far: Vec<Vec<f32>> = (0..6).map(|_| random_vec(&mut rng, d, 1.0)).collect();
        let vectors: Vec<Vec<f32>> = near.into_iter().chain(far).collect();
        let work = |chunk: &[Vec<f32>]| -> Vec<(Code, QueryPrep, QueryPrep)> {
            chunk
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    (
                        encode(&rot, v, &c),
                        prep_for(Metric::L2, &rot, v, &c, &ptc, i as u32),
                        prep_for(Metric::Ip, &rot, v, &c, &ptc, i as u32),
                    )
                })
                .collect()
        };
        let serial: Vec<_> = vectors.chunks(3).flat_map(&work).collect();
        assert!(
            serial.iter().any(|(_, _, p)| p.centred) && serial.iter().any(|(_, _, p)| !p.centred),
            "both forms are exercised"
        );
        let parallel: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = vectors
                .chunks(3)
                .map(|chunk| scope.spawn(|| work(chunk)))
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("worker"))
                .collect()
        });
        assert_eq!(serial, parallel);
    }
}

#[cfg(test)]
mod hot_path_tests {
    use super::*;
    use crate::local::{
        kernel::Isa,
        rotation::{Rotation, padded_dim},
        testkit::{Rng, norm64, to_f32, to_f64},
    };

    const SEED: u64 = 0x5EED_0003;

    fn vector(rng: &mut Rng, d: usize, norm: f64) -> Vec<f32> {
        to_f32(&rng.unit(d).iter().map(|x| x * norm).collect::<Vec<_>>())
    }

    fn levels(prep: &QueryPrep) -> Vec<u32> {
        let bit = |plane: &[u64], i: usize| (plane[i / 64] >> (i % 64) & 1) as u32;
        (0..prep.planes[0].len() * 64)
            .map(|i| (0..4).map(|j| bit(&prep.planes[j], i) << j).sum())
            .collect()
    }

    #[test]
    fn prepare_query_into_on_a_dirty_prep_equals_a_fresh_one() {
        let mut rng = Rng::new(11);
        let mut reused = QueryPrep::new(64);
        for (round, d) in [100usize, 768, 65, 100, 1024].into_iter().enumerate() {
            let rot = Rotation::new(SEED, d);
            let c = vector(&mut rng, d, 6.0);
            let ptc = rot.apply(&c);
            let near: Vec<f32> = c
                .iter()
                .map(|v| 0.8 * v + 0.05 * rng.normal() as f32)
                .collect();
            let far = vector(&mut rng, d, 0.3);
            let constant = vec![2.0f32; d];
            let queries = [&near, &far, &c, &vec![0.0f32; d], &constant];
            for (n, q) in queries.into_iter().enumerate() {
                for metric in [Metric::L2, Metric::Ip, Metric::Cosine] {
                    let ptq = if n == 4 {
                        vec![0.25f32; padded_dim(d)]
                    } else {
                        rot.apply(q)
                    };
                    let list = (round * 10 + n) as u32;
                    prepare_query_into(&mut reused, metric, q, &ptq, &c, &ptc, SEED, list);
                    let fresh = prepare_query(metric, q, &ptq, &c, &ptc, SEED, list);
                    assert_eq!(reused, fresh, "d = {d}, query {n}, {metric:?}");
                }
            }
        }
    }

    #[test]
    fn encode_into_with_reused_buffers_equals_encode() {
        let mut rng = Rng::new(12);
        let mut scratch = EncodeScratch::default();
        let mut code = Code {
            bits: Vec::new(),
            rnorm: f32::NAN,
            dot: f32::NAN,
            rc: f32::NAN,
        };
        for d in [100usize, 768, 65, 100] {
            let rot = Rotation::new(SEED, d);
            let c = vector(&mut rng, d, 3.0);
            let near: Vec<f32> = c.iter().map(|v| v + 0.3 * rng.normal() as f32).collect();
            for x in [&near, &c, &vector(&mut rng, d, 2.0)] {
                encode_into(&rot, x, &c, &mut scratch, &mut code);
                assert_eq!(code, encode(&rot, x, &c), "d = {d}");
            }
        }
    }

    #[test]
    fn estimate_bits_on_every_supported_path_equals_estimate() {
        let mut rng = Rng::new(13);
        let d = 200;
        let rot = Rotation::new(SEED, d);
        let c = vector(&mut rng, d, 4.0);
        let q: Vec<f32> = c
            .iter()
            .map(|v| 0.9 * v + 0.1 * rng.normal() as f32)
            .collect();
        let prep = prepare_query(Metric::Ip, &q, &rot.apply(&q), &c, &rot.apply(&c), SEED, 3);
        for _ in 0..50 {
            let x: Vec<f32> = c.iter().map(|v| v + 0.4 * rng.normal() as f32).collect();
            let code = encode(&rot, &x, &c);
            let want = estimate(&prep, &code);
            for kernel in [Isa::Scalar, Isa::Avx2, Isa::Avx512, Isa::Neon]
                .into_iter()
                .filter_map(Kernel::new)
            {
                assert_eq!(
                    estimate_bits(kernel, &prep, &code.bits, code.dot),
                    want,
                    "{kernel:?}"
                );
            }
        }
    }

    #[test]
    fn key_from_parts_is_key_without_the_code() {
        let mut rng = Rng::new(15);
        for d in [100usize, 768] {
            let rot = Rotation::new(SEED, d);
            let c = vector(&mut rng, d, 4.0);
            let near: Vec<f32> = c
                .iter()
                .map(|v| 0.9 * v + 0.1 * rng.normal() as f32)
                .collect();
            let far = vector(&mut rng, d, 0.4);
            for q in [&near, &far] {
                for metric in [Metric::L2, Metric::Ip, Metric::Cosine] {
                    let prep = prepare_query(metric, q, &rot.apply(q), &c, &rot.apply(&c), SEED, 5);
                    for _ in 0..20 {
                        let x: Vec<f32> = c.iter().map(|v| v + 0.4 * rng.normal() as f32).collect();
                        let code = encode(&rot, &x, &c);
                        let (est, err) = estimate(&prep, &code);
                        for e in [est, est + err, 0.0] {
                            assert_eq!(
                                key(metric, &prep, &code, e),
                                key_from_parts(metric, &prep, code.rnorm, code.rc, e),
                                "d = {d}, {metric:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// `rows[j * lanes + r]` = byte `j` of record `r`'s code; dead lanes stay zero.
    fn interleave(codes: &[&Code], lanes: usize) -> Vec<u8> {
        let code_bytes = codes[0].bits.len() * 8;
        let mut rows = vec![0u8; code_bytes * lanes];
        for (r, code) in codes.iter().enumerate() {
            for j in 0..code_bytes {
                rows[j * lanes + r] = code.bits[j / 8].to_le_bytes()[j % 8];
            }
        }
        rows
    }

    struct Fixture {
        rot: Rotation,
        c: Vec<f32>,
        codes: Vec<Code>,
        queries: Vec<Vec<f32>>,
    }

    fn fixture(rng: &mut Rng, d: usize, n: usize) -> Fixture {
        let rot = Rotation::new(SEED, d);
        let c = vector(rng, d, 4.0);
        let codes = (0..n)
            .map(|_| {
                let x: Vec<f32> = c.iter().map(|v| v + 0.4 * rng.normal() as f32).collect();
                encode(&rot, &x, &c)
            })
            .collect();
        let near: Vec<f32> = c
            .iter()
            .map(|v| 0.9 * v + 0.1 * rng.normal() as f32)
            .collect();
        let queries = vec![
            near,
            vector(rng, d, 0.4),
            vec![0.0; d],
            vec![1.5; d],
            c.clone(),
        ];
        Fixture {
            rot,
            c,
            codes,
            queries,
        }
    }

    #[test]
    fn the_tables_give_the_plane_counts_for_every_metric_width_and_query() {
        let mut rng = Rng::new(31);
        let kernel = Kernel::detect();
        for d in [100usize, 200, 768, 1024] {
            let f = fixture(&mut rng, d, 70);
            let ptc = f.rot.apply(&f.c);
            for q in &f.queries {
                for metric in [Metric::L2, Metric::Ip, Metric::Cosine] {
                    let prep = prepare_query(metric, q, &f.rot.apply(q), &f.c, &ptc, SEED, 9);
                    assert_eq!(prep.lut.len(), 4 * padded_dim(d));
                    assert!(
                        prep.lut.iter().all(|e| *e <= 60),
                        "four levels of at most 15 per entry"
                    );
                    let [p0, p1, p2, p3] = &prep.planes;
                    for lanes in [16usize, 32, 64] {
                        for chunk in f.codes.chunks(lanes) {
                            let refs: Vec<&Code> = chunk.iter().collect();
                            let rows = interleave(&refs, lanes);
                            let mut acc = [0u32; 64];
                            kernel.acc_block(&prep.lut, &rows, lanes, &mut acc);
                            for (r, code) in chunk.iter().enumerate() {
                                let c = kernel.and_popcount_4(&code.bits, &[p0, p1, p2, p3]);
                                let want = c[0] + 2 * c[1] + 4 * c[2] + 8 * c[3];
                                assert_eq!(
                                    acc[r], want,
                                    "d = {d}, {metric:?}, {lanes} lanes, lane {r}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn score_block_is_bit_identical_to_the_record_major_path() {
        let mut rng = Rng::new(32);
        let kernel = Kernel::detect();
        let degenerate = [0.0f32, f32::NAN, 1e-4, 1.0, 0.5, 2.0];
        for d in [100usize, 768] {
            let f = fixture(&mut rng, d, 64);
            let ptc = f.rot.apply(&f.c);
            let dp = padded_dim(d);
            for q in &f.queries {
                for metric in [Metric::L2, Metric::Ip, Metric::Cosine] {
                    let prep = prepare_query(metric, q, &f.rot.apply(q), &f.c, &ptc, SEED, 4);
                    let lanes = 64;
                    let refs: Vec<&Code> = f.codes.iter().collect();
                    let rows = interleave(&refs, lanes);
                    let mut acc = [0u32; 64];
                    kernel.acc_block(&prep.lut, &rows, lanes, &mut acc);
                    let live = 61;
                    let pb: Vec<u16> = f.codes[..live]
                        .iter()
                        .map(|c| c.bits.iter().map(|w| w.count_ones()).sum::<u32>() as u16)
                        .collect();
                    let mut dot: Vec<f32> = f.codes[..live].iter().map(|c| c.dot).collect();
                    let mut rnorm: Vec<f32> = f.codes[..live].iter().map(|c| c.rnorm).collect();
                    let rc: Vec<f32> = f.codes[..live].iter().map(|c| c.rc).collect();
                    for (i, v) in degenerate.iter().enumerate() {
                        dot[7 * i + 3] = *v;
                    }
                    rnorm[1] = 0.0;
                    for margin in [1.0, PRUNE_MARGIN] {
                        let (mut est, mut lo) = (vec![0.0f32; live], vec![0.0f32; live]);
                        let input = BlockInput {
                            acc: &acc[..live],
                            pb: &pb,
                            dot: &dot,
                            rnorm: &rnorm,
                            rc: &rc,
                        };
                        score_block(metric, &prep, dp, margin, &input, &mut est, Some(&mut lo));
                        let mut est_only = vec![0.0f32; live];
                        score_block(metric, &prep, dp, margin, &input, &mut est_only, None);
                        for i in 0..live {
                            let (e, err) = estimate_bits(kernel, &prep, &f.codes[i].bits, dot[i]);
                            let key = key_from_parts(metric, &prep, rnorm[i], rc[i], e);
                            let bound =
                                lo_from_parts(metric, &prep, rnorm[i], rc[i], e, margin * err);
                            let label = format!(
                                "d = {d}, {metric:?}, margin {margin}, lane {i}, dot {}",
                                dot[i]
                            );
                            assert_eq!(est[i].to_bits(), key.to_bits(), "est: {label}");
                            assert_eq!(lo[i].to_bits(), bound.to_bits(), "lo: {label}");
                            assert_eq!(est_only[i].to_bits(), key.to_bits(), "est only: {label}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_wider_margin_never_raises_the_bound_and_the_bound_never_exceeds_the_key() {
        let mut rng = Rng::new(33);
        let kernel = Kernel::detect();
        let f = fixture(&mut rng, 200, 64);
        let ptc = f.rot.apply(&f.c);
        for q in &f.queries {
            for metric in [Metric::L2, Metric::Ip, Metric::Cosine] {
                let prep = prepare_query(metric, q, &f.rot.apply(q), &f.c, &ptc, SEED, 2);
                for code in &f.codes {
                    let (e, err) = estimate_bits(kernel, &prep, &code.bits, code.dot);
                    let key = key(metric, &prep, code, e);
                    let narrow = lo(metric, &prep, code, e, err);
                    let wide = lo(metric, &prep, code, e, PRUNE_MARGIN * err);
                    assert!(
                        wide <= narrow && narrow <= key,
                        "{metric:?}: {wide} {narrow} {key}"
                    );
                    if metric == Metric::L2 {
                        assert!(wide >= 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn query_rounding_is_unbiased_over_lists() {
        let (d, lists) = (128usize, 2_000u32);
        let mut rng = Rng::new(14);
        let rot = Rotation::new(SEED, d);
        let c = vector(&mut rng, d, 1.0);
        let q = vector(&mut rng, d, 1.0);
        let (ptq, ptc) = (rot.apply(&q), rot.apply(&c));
        let g: Vec<f64> = q
            .iter()
            .zip(&c)
            .map(|(a, b)| f64::from(*a) - f64::from(*b))
            .collect();
        let gnorm = norm64(&g);
        let target: Vec<f64> = to_f64(&ptq)
            .iter()
            .zip(to_f64(&ptc))
            .map(|(a, b)| (a - b) / gnorm)
            .collect();
        let mut sums = vec![0.0f64; target.len()];
        let mut delta = 0.0;
        for list in 0..lists {
            let prep = prepare_query(Metric::L2, &q, &ptq, &c, &ptc, SEED, list);
            delta = f64::from(prep.delta);
            for (sum, level) in sums.iter_mut().zip(levels(&prep)) {
                *sum += f64::from(prep.v_l) + delta * f64::from(level);
            }
        }
        let errors: Vec<f64> = sums
            .iter()
            .zip(&target)
            .map(|(sum, t)| (sum / f64::from(lists) - t) / delta)
            .collect();
        let worst = errors.iter().fold(0.0f64, |m, e| m.max(e.abs()));
        let mean = errors.iter().sum::<f64>() / errors.len() as f64;
        assert!(worst <= 0.06, "a coordinate is biased by {worst} steps");
        assert!(mean.abs() <= 0.006, "the mean bias is {mean} steps");
    }
}
