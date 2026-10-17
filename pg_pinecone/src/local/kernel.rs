//! AND+popcount kernels of the RaBitQ scan.
//!
//! Papers: Gao & Long, RaBitQ, SIGMOD 2024 (the inner product of a 1-bit code with a 4-bit query as
//! four AND+popcount passes); Mula, Kurz, Lemire, Faster population counts using AVX2 instructions,
//! The Computer Journal 2018 (the byte-lookup popcount). Clean-room from the papers: no code from
//! any proprietary or third-party implementation.
//!
//! A code is `D / 64` words, bit `i` at word `i / 64`, bit `i % 64`; a query is four bit-planes in
//! the same packing. Every path returns `[popcnt(code & plane_j) for j in 0..4]` for any equal
//! slice length (tails included); integer arithmetic, so all paths equal [`Isa::Scalar`] exactly.
//!
//! Paths: scalar `count_ones`, AVX2, AVX-512 `vpopcntq` and NEON `vcntq_u8`. The AVX paths read
//! tails with masked loads and NEON counts an odd last word in scalar: no read goes past a slice.

//! Block kernels (FastScan, André et al., VLDB 2015, applied to the 1-bit codes): a page stores
//! its codes in blocks of `lanes` records, byte-interleaved (`rows[j * lanes + r]` is byte `j` of
//! record `r`), and the query carries one 16-entry table per 4 coordinates (`lut`, built from the
//! same 4-bit levels as the planes: entry `v` of group `g` sums the levels of the coordinates of
//! `g` whose bit is set in `v`). `acc` of a record is then one table lookup per nibble:
//! `acc = sum_g lut[16 g + nibble_g]`, equal to `sum_j 2^j popcnt(code & plane_j)`.
//!
//! The table of the low nibble of row `j` is `lut[32 j ..][..16]`, that of the high nibble
//! `lut[32 j + 16 ..][..16]`; every entry is at most 63, so four lookups fit a byte.
//!
//! The vector block paths take rows in pairs: four lookups add up in a byte (at most 252) that
//! widens into 16-bit sums, so one call sums at most 512 rows and longer blocks go in segments.

use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Isa {
    Scalar,
    Avx2,
    /// `avx2` + `avx512f` + `avx512bw` + `avx512vpopcntdq`.
    Avx512,
    Neon,
}

/// Most records a page block holds (and `acc` values one block call returns).
pub(crate) const BLOCK_MAX_LANES: usize = 64;

/// Block widths a page can have: the codes of one block fill `lanes * D / 8` bytes of a page.
pub(crate) fn valid_block_lanes(lanes: usize) -> bool {
    matches!(lanes, 16 | 32 | 64)
}

/// Runtime feature detection; [`Isa::Scalar`] is always supported.
pub(crate) fn supported(isa: Isa) -> bool {
    match isa {
        Isa::Scalar => true,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => is_x86_feature_detected!("avx2"),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => {
            is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vpopcntdq")
        }
        // NEON is part of the aarch64 baseline.
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => true,
        _ => false,
    }
}

struct Detected {
    isa: Isa,
    rejected: Vec<Isa>,
}

fn detected() -> &'static Detected {
    static DETECTED: OnceLock<Detected> = OnceLock::new();
    DETECTED.get_or_init(|| {
        let mut rejected = Vec::new();
        for isa in [Isa::Avx512, Isa::Neon, Isa::Avx2] {
            if !supported(isa) {
                continue;
            }
            if agrees_with_scalar(isa) {
                return Detected { isa, rejected };
            }
            rejected.push(isa);
        }
        Detected {
            isa: Isa::Scalar,
            rejected,
        }
    })
}

/// The fastest supported path, resolved once per process: AVX-512 if supported, else AVX2 or
/// scalar on x86_64, NEON on aarch64. A path that disagrees with the scalar one on the startup
/// check is skipped, see [`rejected`].
pub(crate) fn detect() -> Isa {
    detected().isa
}

/// The paths [`detect`] skipped for disagreeing with the scalar one (none on healthy hardware):
/// the scan reports them once per backend through the server log, so this module needs no
/// Postgres to be tested.
pub(crate) fn rejected() -> &'static [Isa] {
    &detected().rejected
}

/// Runs `isa` against the scalar path on lengths that reach every loop stage and tail, and on
/// all-ones words that fill the byte accumulators of the longest blocks; then the block kernel,
/// see [`block_agrees`].
fn agrees_with_scalar(isa: Isa) -> bool {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..=40).chain([124, 125, 253]).all(|len| {
        let ones = vec![u64::MAX; len];
        let code: Vec<u64> = if len > 100 {
            ones.clone()
        } else {
            (0..len).map(|_| next()).collect()
        };
        let random: Vec<Vec<u64>> = (0..3).map(|_| (0..len).map(|_| next()).collect()).collect();
        let planes: [&[u64]; 4] = [&ones, &random[0], &random[1], &random[2]];
        and_popcount_4_on(isa, &code, planes) == scalar::and_popcount_4(&code, planes)
            && popcount_on(isa, &code) == scalar::popcount(&code)
    }) && block_agrees(&mut next, |lut, rows, lanes, out| {
        acc_block_on(isa, lut, rows, lanes, out);
    })
}

/// A block `candidate` against the scalar one at every width: random content from one row pair up,
/// and the [`extreme_cases`]. `out` past the lanes must stay untouched.
fn block_agrees(
    next: &mut impl FnMut() -> u64,
    candidate: impl Fn(&[u8], &[u8], usize, &mut [u32; BLOCK_MAX_LANES]),
) -> bool {
    for lanes in [16usize, 32, 64] {
        let mut cases = Vec::new();
        for rows in [2usize, 6, 96] {
            let lut: Vec<u8> = (0..rows * 32).map(|_| (next() % 64) as u8).collect();
            let bytes: Vec<u8> = (0..rows * lanes).map(|_| next() as u8).collect();
            cases.push((lut, bytes));
        }
        cases.extend(extreme_cases(lanes));
        for (lut, bytes) in cases {
            let mut want = [u32::MAX; BLOCK_MAX_LANES];
            scalar::acc_block(&lut, &bytes, lanes, &mut want);
            let mut got = [u32::MAX; BLOCK_MAX_LANES];
            candidate(&lut, &bytes, lanes, &mut got);
            if got != want {
                return false;
            }
        }
    }
    true
}

/// A segment of rows that fills the byte sums (252) and nearly the 16-bit ones (64,512): every table
/// holds 0 at entry 0 and 63 elsewhere, so a 0xFF byte adds 126 and a 0x00 byte nothing, and the
/// bytes are 0xFF in every lane, only the even ones or only the odd ones. The last two make the
/// even and odd lanes sum unequally, which the 16-bit accumulators of the AVX2 path split by parity.
fn extreme_cases(lanes: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    let lut: Vec<u8> = (0..SEGMENT_ROWS * 32)
        .map(|i| if i % 16 == 0 { 0 } else { 63 })
        .collect();
    [[0xFFu8, 0xFF], [0xFF, 0], [0, 0xFF]]
        .iter()
        .map(|fill| {
            let bytes = (0..SEGMENT_ROWS * lanes)
                .map(|i| fill[i % lanes % 2])
                .collect();
            (lut.clone(), bytes)
        })
        .collect()
}

/// `planes.len() == 4`, each as long as `code`; panics on an `isa` this CPU lacks.
#[cfg_attr(not(test), allow(dead_code))] // tests name the path; the scan holds a `Kernel`
pub(crate) fn and_popcount_4_with(isa: Isa, code: &[u64], planes: &[&[u64]]) -> [u32; 4] {
    assert!(
        supported(isa),
        "kernel: {isa:?} is not supported on this CPU"
    );
    and_popcount_4_on(isa, code, plane_array(code, planes))
}

#[cfg_attr(not(test), allow(dead_code))] // tests; the scan holds a `Kernel`
pub(crate) fn and_popcount_4(code: &[u64], planes: &[&[u64]]) -> [u32; 4] {
    and_popcount_4_on(detect(), code, plane_array(code, planes))
}

/// `pb` of the estimator, on the [`detect`]ed path.
#[cfg_attr(not(test), allow(dead_code))] // tests; the scan holds a `Kernel`
pub(crate) fn popcount(code: &[u64]) -> u32 {
    popcount_on(detect(), code)
}

/// [`popcount`] on a named path; panics on an `isa` this CPU lacks.
#[cfg_attr(not(test), allow(dead_code))] // tests name the path
pub(crate) fn popcount_with(isa: Isa, code: &[u64]) -> u32 {
    assert!(
        supported(isa),
        "kernel: {isa:?} is not supported on this CPU"
    );
    popcount_on(isa, code)
}

/// [`Kernel::acc_block`] on a named path; panics on an `isa` this CPU lacks.
#[cfg_attr(not(test), allow(dead_code))] // tests name the path
pub(crate) fn acc_block_with(
    isa: Isa,
    lut: &[u8],
    rows: &[u8],
    lanes: usize,
    out: &mut [u32; BLOCK_MAX_LANES],
) {
    assert!(
        supported(isa),
        "kernel: {isa:?} is not supported on this CPU"
    );
    check_block(lut, rows, lanes);
    acc_block_on(isa, lut, rows, lanes, out);
}

/// A path this CPU supports: building one is the support check, so its calls skip it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Kernel(Isa);

impl Kernel {
    #[cfg_attr(not(test), allow(dead_code))] // tests name the path
    pub(crate) fn new(isa: Isa) -> Option<Kernel> {
        supported(isa).then_some(Kernel(isa))
    }

    /// The [`detect`]ed path.
    pub(crate) fn detect() -> Kernel {
        Kernel(detect())
    }

    #[cfg_attr(not(test), allow(dead_code))] // tests
    pub(crate) fn isa(self) -> Isa {
        self.0
    }

    /// `planes.len() == 4`, each as long as `code`.
    pub(crate) fn and_popcount_4(self, code: &[u64], planes: &[&[u64]]) -> [u32; 4] {
        and_popcount_4_on(self.0, code, plane_array(code, planes))
    }

    pub(crate) fn popcount(self, code: &[u64]) -> u32 {
        popcount_on(self.0, code)
    }

    /// `acc` of the first `lanes` records of a block (module docs for the layouts): `out[..lanes]`
    /// is overwritten, the rest untouched. Panics unless `lanes` is 16, 32 or 64, `lut.len()` is a
    /// multiple of 64 (an even number of rows) and `rows.len() == lut.len() / 32 * lanes`.
    pub(crate) fn acc_block(
        self,
        lut: &[u8],
        rows: &[u8],
        lanes: usize,
        out: &mut [u32; BLOCK_MAX_LANES],
    ) {
        check_block(lut, rows, lanes);
        acc_block_on(self.0, lut, rows, lanes, out);
    }
}

/// What the vector block paths rely on for memory safety.
fn check_block(lut: &[u8], rows: &[u8], lanes: usize) {
    assert!(valid_block_lanes(lanes), "kernel: a block of {lanes} lanes");
    assert!(
        lut.len().is_multiple_of(64),
        "kernel: a table of {} bytes is not an even number of rows",
        lut.len()
    );
    assert_eq!(
        rows.len(),
        lut.len() / 32 * lanes,
        "kernel: block rows do not match the table"
    );
}

/// Most rows one vector call sums: 512 rows of two entries of at most 63 fill its 16-bit sums.
const SEGMENT_ROWS: usize = 512;

/// `isa` must be [`supported`]; the arguments passed [`check_block`].
fn acc_block_on(isa: Isa, lut: &[u8], rows: &[u8], lanes: usize, out: &mut [u32; BLOCK_MAX_LANES]) {
    if isa == Isa::Scalar || lut.len() <= SEGMENT_ROWS * 32 {
        return acc_segment(isa, lut, rows, lanes, out);
    }
    // Past D = 4,096: sum the segments in 32 bits.
    let (mut sum, mut part) = ([0u32; BLOCK_MAX_LANES], [0u32; BLOCK_MAX_LANES]);
    let segments = lut
        .chunks(SEGMENT_ROWS * 32)
        .zip(rows.chunks(SEGMENT_ROWS * lanes));
    for (lut, rows) in segments {
        acc_segment(isa, lut, rows, lanes, &mut part);
        for (sum, part) in sum.iter_mut().zip(&part[..lanes]) {
            *sum += part;
        }
    }
    out[..lanes].copy_from_slice(&sum[..lanes]);
}

/// [`acc_block_on`] with at most [`SEGMENT_ROWS`] rows.
fn acc_segment(isa: Isa, lut: &[u8], rows: &[u8], lanes: usize, out: &mut [u32; BLOCK_MAX_LANES]) {
    match (isa, lanes) {
        (Isa::Scalar, _) => scalar::acc_block(lut, rows, lanes, out),
        // SAFETY: `check_block` checked the lengths against `lanes`, the callers `supported(isa)`
        // and `acc_block_on` the segment length, which is all the kernels require.
        #[cfg(target_arch = "x86_64")]
        (Isa::Avx2, 64) => unsafe { x86::acc_block_avx2::<2>(lut, rows, out) },
        // SAFETY: as above.
        #[cfg(target_arch = "x86_64")]
        (Isa::Avx2 | Isa::Avx512, 32) => unsafe { x86::acc_block_avx2::<1>(lut, rows, out) },
        // SAFETY: as above.
        #[cfg(target_arch = "x86_64")]
        (Isa::Avx2 | Isa::Avx512, 16) => unsafe { x86::acc_block_avx2_16(lut, rows, out) },
        // SAFETY: as above.
        #[cfg(target_arch = "x86_64")]
        (Isa::Avx512, 64) => unsafe { x86::acc_block_avx512(lut, rows, out) },
        // SAFETY: as above.
        #[cfg(target_arch = "aarch64")]
        (Isa::Neon, 64) => unsafe { neon::acc_block::<4>(lut, rows, out) },
        // SAFETY: as above.
        #[cfg(target_arch = "aarch64")]
        (Isa::Neon, 32) => unsafe { neon::acc_block::<2>(lut, rows, out) },
        // SAFETY: as above.
        #[cfg(target_arch = "aarch64")]
        (Isa::Neon, 16) => unsafe { neon::acc_block::<1>(lut, rows, out) },
        _ => unreachable!("kernel: {isa:?} with {lanes} lanes"),
    }
}

/// The four planes as an array, after checking what the vector paths rely on for memory safety.
fn plane_array<'a>(code: &[u64], planes: &[&'a [u64]]) -> [&'a [u64]; 4] {
    let &[p0, p1, p2, p3] = planes else {
        panic!("kernel: expected 4 planes, got {}", planes.len());
    };
    let n = code.len();
    assert!(
        [p0, p1, p2, p3].iter().all(|p| p.len() >= n),
        "kernel: a plane is shorter than the code"
    );
    [p0, p1, p2, p3]
}

/// `isa` must be [`supported`] and the planes at least as long as `code`.
fn and_popcount_4_on(isa: Isa, code: &[u64], planes: [&[u64]; 4]) -> [u32; 4] {
    match isa {
        Isa::Scalar => scalar::and_popcount_4(code, planes),
        // SAFETY: the callers checked `supported(isa)` (or took it from `detect`) and `plane_array`
        // checked the plane lengths, which is all the kernels require.
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { x86::and_popcount_4_avx2(code, planes) },
        // SAFETY: as above.
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => unsafe { x86::and_popcount_4_avx512(code, planes) },
        // SAFETY: as above.
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => unsafe { neon::and_popcount_4(code, planes) },
        _ => unreachable!("kernel: {isa:?} is not compiled for this architecture"),
    }
}

/// `isa` must be [`supported`].
fn popcount_on(isa: Isa, code: &[u64]) -> u32 {
    match isa {
        Isa::Scalar => scalar::popcount(code),
        // SAFETY: the callers checked `supported(isa)` (or took it from `detect`).
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { x86::popcount_avx2(code) },
        // SAFETY: as above.
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => unsafe { x86::popcount_avx512(code) },
        // SAFETY: as above.
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => unsafe { neon::popcount(code) },
        _ => unreachable!("kernel: {isa:?} is not compiled for this architecture"),
    }
}

mod scalar {
    pub(super) fn and_popcount_4(code: &[u64], planes: [&[u64]; 4]) -> [u32; 4] {
        let mut counts = [0u32; 4];
        for (count, plane) in counts.iter_mut().zip(planes) {
            *count = code
                .iter()
                .zip(plane)
                .map(|(c, p)| (c & p).count_ones())
                .sum();
        }
        counts
    }

    pub(super) fn popcount(code: &[u64]) -> u32 {
        code.iter().map(|w| w.count_ones()).sum()
    }

    pub(super) fn acc_block(
        lut: &[u8],
        rows: &[u8],
        lanes: usize,
        out: &mut [u32; super::BLOCK_MAX_LANES],
    ) {
        out[..lanes].fill(0);
        for (row, tables) in rows.chunks_exact(lanes).zip(lut.as_chunks::<32>().0) {
            let (low, high) = tables.split_at(16);
            for (acc, &byte) in out.iter_mut().zip(row) {
                *acc += u32::from(low[usize::from(byte & 15)])
                    + u32::from(high[usize::from(byte >> 4)]);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// A byte accumulator takes at most 8 per chunk, so 31 chunks stay below 256.
    const BYTE_CHUNKS: usize = 31;

    /// Per-byte popcounts of `v`: a 16-entry nibble table through `vpshufb`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn byte_counts(v: __m256i) -> __m256i {
        let table = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );
        let low = _mm256_set1_epi8(0x0f);
        let lo = _mm256_and_si256(v, low);
        let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v), low);
        _mm256_add_epi8(
            _mm256_shuffle_epi8(table, lo),
            _mm256_shuffle_epi8(table, hi),
        )
    }

    /// Lanes `< live` of a 4-word load are selected (`vpmaskmovq` faults on none of the others).
    #[inline]
    #[target_feature(enable = "avx2")]
    fn tail_mask(live: usize) -> __m256i {
        _mm256_cmpgt_epi64(
            _mm256_set1_epi64x(live as i64),
            _mm256_setr_epi64x(0, 1, 2, 3),
        )
    }

    /// Sums the four 64-bit lanes of each of `lanes` (every lane below 2^32) into one `u32` each.
    ///
    /// # Safety
    /// The CPU must support AVX2.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sum_lanes(lanes: [__m256i; 4]) -> [u32; 4] {
        unsafe {
            let pairs_01 = _mm256_hadd_epi32(lanes[0], lanes[1]);
            let pairs_23 = _mm256_hadd_epi32(lanes[2], lanes[3]);
            let halves = _mm256_hadd_epi32(pairs_01, pairs_23);
            let sums = _mm_add_epi32(
                _mm256_castsi256_si128(halves),
                _mm256_extracti128_si256::<1>(halves),
            );
            let mut out = [0u32; 4];
            _mm_storeu_si128(out.as_mut_ptr().cast(), sums);
            out
        }
    }

    /// # Safety
    /// The CPU must support AVX2 and every plane must be at least as long as `code`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn and_popcount_4_avx2(code: &[u64], planes: [&[u64]; 4]) -> [u32; 4] {
        unsafe {
            let zero = _mm256_setzero_si256();
            let mut totals = [zero; 4];
            let full = code.len() / 4 * 4;
            let mut i = 0;
            while i < full {
                let stop = full.min(i + 4 * BYTE_CHUNKS);
                let mut bytes = [zero; 4];
                while i < stop {
                    let c = _mm256_loadu_si256(code.as_ptr().add(i).cast());
                    for (acc, plane) in bytes.iter_mut().zip(&planes) {
                        let p = _mm256_loadu_si256(plane.as_ptr().add(i).cast());
                        *acc = _mm256_add_epi8(*acc, byte_counts(_mm256_and_si256(c, p)));
                    }
                    i += 4;
                }
                for (total, acc) in totals.iter_mut().zip(bytes) {
                    *total = _mm256_add_epi64(*total, _mm256_sad_epu8(acc, zero));
                }
            }
            if full < code.len() {
                let mask = tail_mask(code.len() - full);
                let c = _mm256_maskload_epi64(code.as_ptr().add(full).cast(), mask);
                for (total, plane) in totals.iter_mut().zip(&planes) {
                    let p = _mm256_maskload_epi64(plane.as_ptr().add(full).cast(), mask);
                    let counts = byte_counts(_mm256_and_si256(c, p));
                    *total = _mm256_add_epi64(*total, _mm256_sad_epu8(counts, zero));
                }
            }
            sum_lanes(totals)
        }
    }

    /// # Safety
    /// The CPU must support AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn popcount_avx2(code: &[u64]) -> u32 {
        unsafe {
            let zero = _mm256_setzero_si256();
            let mut total = zero;
            let full = code.len() / 4 * 4;
            let mut i = 0;
            while i < full {
                let stop = full.min(i + 4 * BYTE_CHUNKS);
                let mut bytes = zero;
                while i < stop {
                    let c = _mm256_loadu_si256(code.as_ptr().add(i).cast());
                    bytes = _mm256_add_epi8(bytes, byte_counts(c));
                    i += 4;
                }
                total = _mm256_add_epi64(total, _mm256_sad_epu8(bytes, zero));
            }
            if full < code.len() {
                let mask = tail_mask(code.len() - full);
                let c = _mm256_maskload_epi64(code.as_ptr().add(full).cast(), mask);
                total = _mm256_add_epi64(total, _mm256_sad_epu8(byte_counts(c), zero));
            }
            let pair = _mm_add_epi64(
                _mm256_castsi256_si128(total),
                _mm256_extracti128_si256::<1>(total),
            );
            let sum = _mm_add_epi64(pair, _mm_unpackhi_epi64(pair, pair));
            _mm_cvtsi128_si64(sum) as u32
        }
    }

    /// Lanes `< live` of an 8-word load.
    #[inline]
    fn word_mask(live: usize) -> __mmask8 {
        ((1u16 << live) - 1) as __mmask8
    }

    /// Folds the eight 64-bit lanes of `v` to four.
    #[inline]
    #[target_feature(enable = "avx2,avx512f")]
    fn fold_wide(v: __m512i) -> __m256i {
        _mm256_add_epi64(_mm512_castsi512_si256(v), _mm512_extracti64x4_epi64::<1>(v))
    }

    /// [`sum_lanes`] for eight-lane accumulators.
    ///
    /// # Safety
    /// The CPU must support AVX2 and `avx512f`.
    #[inline]
    #[target_feature(enable = "avx2,avx512f")]
    unsafe fn sum_wide_lanes(wide: [__m512i; 4]) -> [u32; 4] {
        unsafe {
            sum_lanes([
                fold_wide(wide[0]),
                fold_wide(wide[1]),
                fold_wide(wide[2]),
                fold_wide(wide[3]),
            ])
        }
    }

    /// # Safety
    /// The CPU must support `avx512f` and `avx512vpopcntdq`, and every plane must be at least as
    /// long as `code`.
    #[target_feature(enable = "avx2,avx512f,avx512vpopcntdq")]
    pub(super) unsafe fn and_popcount_4_avx512(code: &[u64], planes: [&[u64]; 4]) -> [u32; 4] {
        unsafe {
            let mut wide = [_mm512_setzero_si512(); 4];
            let n = code.len();
            let mut i = 0;
            while i + 8 <= n {
                let c = _mm512_loadu_si512(code.as_ptr().add(i).cast());
                for (acc, plane) in wide.iter_mut().zip(&planes) {
                    let p = _mm512_loadu_si512(plane.as_ptr().add(i).cast());
                    let counts = _mm512_popcnt_epi64(_mm512_and_si512(c, p));
                    *acc = _mm512_add_epi64(*acc, counts);
                }
                i += 8;
            }
            if i < n {
                let mask = word_mask(n - i);
                let c = _mm512_maskz_loadu_epi64(mask, code.as_ptr().add(i).cast());
                for (acc, plane) in wide.iter_mut().zip(&planes) {
                    let p = _mm512_maskz_loadu_epi64(mask, plane.as_ptr().add(i).cast());
                    let counts = _mm512_popcnt_epi64(_mm512_and_si512(c, p));
                    *acc = _mm512_add_epi64(*acc, counts);
                }
            }
            sum_wide_lanes(wide)
        }
    }

    /// # Safety
    /// The CPU must support `avx512f` and `avx512vpopcntdq`.
    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    pub(super) unsafe fn popcount_avx512(code: &[u64]) -> u32 {
        unsafe {
            let mut total = _mm512_setzero_si512();
            let n = code.len();
            let mut i = 0;
            while i + 8 <= n {
                let c = _mm512_loadu_si512(code.as_ptr().add(i).cast());
                total = _mm512_add_epi64(total, _mm512_popcnt_epi64(c));
                i += 8;
            }
            if i < n {
                let c = _mm512_maskz_loadu_epi64(word_mask(n - i), code.as_ptr().add(i).cast());
                total = _mm512_add_epi64(total, _mm512_popcnt_epi64(c));
            }
            _mm512_reduce_add_epi64(total) as u32
        }
    }

    /// The 16-byte table at `p` in both 128-bit lanes.
    ///
    /// # Safety
    /// `p` must be valid for 16 bytes and the CPU must support AVX2.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn table_avx2(p: *const u8) -> __m256i {
        unsafe { _mm256_broadcastsi128_si256(_mm_loadu_si128(p.cast())) }
    }

    /// Both nibble lookups of every byte of `v`, added per byte.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn lookup_avx2(v: __m256i, low: __m256i, high: __m256i, mask: __m256i) -> __m256i {
        let lo = _mm256_and_si256(v, mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v), mask);
        _mm256_add_epi8(_mm256_shuffle_epi8(low, lo), _mm256_shuffle_epi8(high, hi))
    }

    /// Stores the 32 sums of one 32-lane half as `u32`: `all` holds the 16-bit sums of every byte
    /// pair (even byte + 256 * odd byte, mod 2^16), `odd` those of the odd bytes alone.
    ///
    /// # Safety
    /// The CPU must support AVX2 and `out` must be valid for 32 `u32`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store_sums_avx2(all: __m256i, odd: __m256i, out: *mut u32) {
        unsafe {
            let even = _mm256_sub_epi16(all, _mm256_slli_epi16::<8>(odd));
            // Records 0..8 and 16..24, then 8..16 and 24..32, one group per 128-bit lane.
            let first = _mm256_unpacklo_epi16(even, odd);
            let second = _mm256_unpackhi_epi16(even, odd);
            let wide = |v: __m128i| _mm256_cvtepu16_epi32(v);
            _mm256_storeu_si256(out.cast(), wide(_mm256_castsi256_si128(first)));
            _mm256_storeu_si256(out.add(8).cast(), wide(_mm256_castsi256_si128(second)));
            _mm256_storeu_si256(
                out.add(16).cast(),
                wide(_mm256_extracti128_si256::<1>(first)),
            );
            _mm256_storeu_si256(
                out.add(24).cast(),
                wide(_mm256_extracti128_si256::<1>(second)),
            );
        }
    }

    /// `acc` of a block of `32 * V` lanes. `all` adds each row pair's byte sums as 16-bit lanes
    /// (even byte + 256 * odd byte, mod 2^16) and `odd` their odd bytes, so the even sums are
    /// `all - 256 * odd` and a widening costs a shift and two adds, not an AND more.
    ///
    /// # Safety
    /// The CPU must support AVX2; `lut.len()` must be a multiple of 64 and at most 512 rows, and
    /// `rows.len()` must be `lut.len() / 32 * 32 * V`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn acc_block_avx2<const V: usize>(
        lut: &[u8],
        rows: &[u8],
        out: &mut [u32; 64],
    ) {
        const { assert!(V == 1 || V == 2) };
        unsafe {
            let lanes = 32 * V;
            let mask = _mm256_set1_epi8(0x0f);
            let mut all = [_mm256_setzero_si256(); V];
            let mut odd = all;
            let (mut t, mut r) = (lut.as_ptr(), rows.as_ptr());
            let end = t.add(lut.len());
            while t < end {
                let (t0, t1) = (
                    [table_avx2(t), table_avx2(t.add(16))],
                    [table_avx2(t.add(32)), table_avx2(t.add(48))],
                );
                for (h, (all, odd)) in all.iter_mut().zip(&mut odd).enumerate() {
                    let a = _mm256_loadu_si256(r.add(32 * h).cast());
                    let b = _mm256_loadu_si256(r.add(lanes + 32 * h).cast());
                    let s = _mm256_add_epi8(
                        lookup_avx2(a, t0[0], t0[1], mask),
                        lookup_avx2(b, t1[0], t1[1], mask),
                    );
                    *all = _mm256_add_epi16(*all, s);
                    *odd = _mm256_add_epi16(*odd, _mm256_srli_epi16::<8>(s));
                }
                t = t.add(64);
                r = r.add(2 * lanes);
            }
            for (h, (all, odd)) in all.iter().zip(&odd).enumerate() {
                store_sums_avx2(*all, *odd, out.as_mut_ptr().add(32 * h));
            }
        }
    }

    /// `acc` of a 16-lane block, summed like [`acc_block_avx2`]. A row's 16 bytes sit in both
    /// 128-bit lanes, shifted by 0 and by 4, so one `vpshufb` against its two tables as stored
    /// (`lut[32 j ..][..32]`) does both nibbles: 3 vector ops per row, not 8. The low lane sums
    /// the low-nibble entries and the high lane the others (at most 63 per row, so four rows
    /// fill a byte); they fold at the end.
    ///
    /// # Safety
    /// The CPU must support AVX2; `lut.len()` must be a multiple of 64 and at most 512 rows, and
    /// `rows.len()` must be `lut.len() / 32 * 16`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn acc_block_avx2_16(lut: &[u8], rows: &[u8], out: &mut [u32; 64]) {
        unsafe {
            let mask = _mm256_set1_epi8(0x0f);
            let shifts = _mm256_setr_epi32(0, 0, 0, 0, 4, 4, 4, 4);
            let row = |tables: *const u8, bytes: *const u8| {
                let bytes = _mm256_broadcastsi128_si256(_mm_loadu_si128(bytes.cast()));
                let nibbles = _mm256_and_si256(_mm256_srlv_epi32(bytes, shifts), mask);
                _mm256_shuffle_epi8(_mm256_loadu_si256(tables.cast()), nibbles)
            };
            let (mut all, mut odd) = (_mm256_setzero_si256(), _mm256_setzero_si256());
            let pairs = lut.len() / 64;
            let mut p = 0;
            while p < pairs {
                let (t, r) = (lut.as_ptr().add(64 * p), rows.as_ptr().add(32 * p));
                let mut s = _mm256_add_epi8(row(t, r), row(t.add(32), r.add(16)));
                if p + 1 < pairs {
                    let (t, r) = (t.add(64), r.add(32));
                    s = _mm256_add_epi8(s, _mm256_add_epi8(row(t, r), row(t.add(32), r.add(16))));
                    p += 1;
                }
                all = _mm256_add_epi16(all, s);
                odd = _mm256_add_epi16(odd, _mm256_srli_epi16::<8>(s));
                p += 1;
            }
            let fold = |v: __m256i| {
                _mm_add_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v))
            };
            let (all, odd) = (fold(all), fold(odd));
            let even = _mm_sub_epi16(all, _mm_slli_epi16::<8>(odd));
            _mm256_storeu_si256(
                out.as_mut_ptr().cast(),
                _mm256_cvtepu16_epi32(_mm_unpacklo_epi16(even, odd)),
            );
            _mm256_storeu_si256(
                out.as_mut_ptr().add(8).cast(),
                _mm256_cvtepu16_epi32(_mm_unpackhi_epi16(even, odd)),
            );
        }
    }

    /// The 16-byte table at `p` in all four 128-bit lanes.
    ///
    /// # Safety
    /// `p` must be valid for 16 bytes and the CPU must support `avx512f`.
    #[inline]
    #[target_feature(enable = "avx2,avx512f")]
    unsafe fn table_avx512(p: *const u8) -> __m512i {
        unsafe { _mm512_broadcast_i32x4(_mm_loadu_si128(p.cast())) }
    }

    /// [`lookup_avx2`] on 64 bytes.
    #[inline]
    #[target_feature(enable = "avx2,avx512f,avx512bw")]
    fn lookup_avx512(v: __m512i, low: __m512i, high: __m512i, mask: __m512i) -> __m512i {
        let lo = _mm512_and_si512(v, mask);
        let hi = _mm512_and_si512(_mm512_srli_epi16::<4>(v), mask);
        _mm512_add_epi8(_mm512_shuffle_epi8(low, lo), _mm512_shuffle_epi8(high, hi))
    }

    /// [`acc_block_avx2`] for 64 lanes: one 512-bit vector per row.
    ///
    /// # Safety
    /// The CPU must support `avx512f` and `avx512bw`; `lut.len()` must be a multiple of 64 and at
    /// most 512 rows, and `rows.len()` must be `lut.len() / 32 * 64`.
    #[target_feature(enable = "avx2,avx512f,avx512bw")]
    pub(super) unsafe fn acc_block_avx512(lut: &[u8], rows: &[u8], out: &mut [u32; 64]) {
        unsafe {
            let mask = _mm512_set1_epi8(0x0f);
            let (mut all, mut odd) = (_mm512_setzero_si512(), _mm512_setzero_si512());
            let (mut t, mut r) = (lut.as_ptr(), rows.as_ptr());
            let end = t.add(lut.len());
            while t < end {
                let a = _mm512_loadu_si512(r.cast());
                let b = _mm512_loadu_si512(r.add(64).cast());
                let s = _mm512_add_epi8(
                    lookup_avx512(a, table_avx512(t), table_avx512(t.add(16)), mask),
                    lookup_avx512(b, table_avx512(t.add(32)), table_avx512(t.add(48)), mask),
                );
                all = _mm512_add_epi16(all, s);
                odd = _mm512_add_epi16(odd, _mm512_srli_epi16::<8>(s));
                t = t.add(64);
                r = r.add(128);
            }
            let even = _mm512_sub_epi16(all, _mm512_slli_epi16::<8>(odd));
            // Records 0..8, 16..24, 32..40, 48..56 and 8..16, 24..32, 40..48, 56..64: one group
            // per 128-bit lane; two permutes put them in order, 32 records each.
            let first = _mm512_unpacklo_epi16(even, odd);
            let second = _mm512_unpackhi_epi16(even, odd);
            let low = _mm512_permutex2var_epi64(
                first,
                _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0),
                second,
            );
            let high = _mm512_permutex2var_epi64(
                first,
                _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4),
                second,
            );
            let at = out.as_mut_ptr();
            _mm512_storeu_si512(
                at.cast(),
                _mm512_cvtepu16_epi32(_mm512_castsi512_si256(low)),
            );
            _mm512_storeu_si512(
                at.add(16).cast(),
                _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64::<1>(low)),
            );
            _mm512_storeu_si512(
                at.add(32).cast(),
                _mm512_cvtepu16_epi32(_mm512_castsi512_si256(high)),
            );
            _mm512_storeu_si512(
                at.add(48).cast(),
                _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64::<1>(high)),
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// A byte accumulator takes at most 8 per chunk, so 31 chunks stay below 256.
    const BYTE_CHUNKS: usize = 31;

    /// # Safety
    /// Every plane must be at least as long as `code`.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn and_popcount_4(code: &[u64], planes: [&[u64]; 4]) -> [u32; 4] {
        unsafe {
            let mut counts = [0u32; 4];
            let chunks = code.len() / 2;
            let mut i = 0;
            while i < chunks {
                let stop = chunks.min(i + BYTE_CHUNKS);
                let mut bytes = [vdupq_n_u8(0); 4];
                while i < stop {
                    let c = vld1q_u8(code.as_ptr().add(2 * i).cast());
                    for (acc, plane) in bytes.iter_mut().zip(&planes) {
                        let p = vld1q_u8(plane.as_ptr().add(2 * i).cast());
                        *acc = vaddq_u8(*acc, vcntq_u8(vandq_u8(c, p)));
                    }
                    i += 1;
                }
                for (count, acc) in counts.iter_mut().zip(bytes) {
                    *count += u32::from(vaddlvq_u8(acc));
                }
            }
            if code.len() % 2 == 1 {
                let last = code.len() - 1;
                for (count, plane) in counts.iter_mut().zip(&planes) {
                    *count += (code[last] & plane[last]).count_ones();
                }
            }
            counts
        }
    }

    /// # Safety
    /// Always sound on aarch64 (NEON is baseline); `unsafe` for the raw loads.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn popcount(code: &[u64]) -> u32 {
        unsafe {
            let mut total = 0u32;
            let chunks = code.len() / 2;
            let mut i = 0;
            while i < chunks {
                let stop = chunks.min(i + BYTE_CHUNKS);
                let mut bytes = vdupq_n_u8(0);
                while i < stop {
                    let c = vld1q_u8(code.as_ptr().add(2 * i).cast());
                    bytes = vaddq_u8(bytes, vcntq_u8(c));
                    i += 1;
                }
                total += u32::from(vaddlvq_u8(bytes));
            }
            if code.len() % 2 == 1 {
                total += code[code.len() - 1].count_ones();
            }
            total
        }
    }

    /// `acc` of a block of `16 * CHUNKS` lanes; `vaddw` widens the byte sums in record order.
    ///
    /// # Safety
    /// `lut.len()` must be a multiple of 64 and at most 512 rows, and `rows.len()` must be
    /// `lut.len() / 32 * 16 * CHUNKS`.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn acc_block<const CHUNKS: usize>(
        lut: &[u8],
        rows: &[u8],
        out: &mut [u32; 64],
    ) {
        const { assert!(CHUNKS >= 1 && CHUNKS <= 4) };
        unsafe {
            let lanes = 16 * CHUNKS;
            let mask = vdupq_n_u8(0x0f);
            let mut sums = [vdupq_n_u16(0); 8];
            let (mut t, mut r) = (lut.as_ptr(), rows.as_ptr());
            let end = t.add(lut.len());
            while t < end {
                let tables = vld1q_u8_x4(t);
                for (c, pair) in sums
                    .as_chunks_mut::<2>()
                    .0
                    .iter_mut()
                    .take(CHUNKS)
                    .enumerate()
                {
                    let a = vld1q_u8(r.add(16 * c));
                    let b = vld1q_u8(r.add(lanes + 16 * c));
                    let s = vaddq_u8(
                        vaddq_u8(
                            vqtbl1q_u8(tables.0, vandq_u8(a, mask)),
                            vqtbl1q_u8(tables.1, vshrq_n_u8::<4>(a)),
                        ),
                        vaddq_u8(
                            vqtbl1q_u8(tables.2, vandq_u8(b, mask)),
                            vqtbl1q_u8(tables.3, vshrq_n_u8::<4>(b)),
                        ),
                    );
                    pair[0] = vaddw_u8(pair[0], vget_low_u8(s));
                    pair[1] = vaddw_high_u8(pair[1], s);
                }
                t = t.add(64);
                r = r.add(2 * lanes);
            }
            for (k, sum) in sums.iter().take(2 * CHUNKS).enumerate() {
                let at = out.as_mut_ptr().add(8 * k);
                vst1q_u32(at, vmovl_u16(vget_low_u16(*sum)));
                vst1q_u32(at.add(4), vmovl_high_u16(*sum));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local::testkit::Rng;

    const ALL: [Isa; 4] = [Isa::Scalar, Isa::Avx2, Isa::Avx512, Isa::Neon];

    #[cfg(target_arch = "x86_64")]
    fn host_supports(isa: Isa) -> bool {
        match isa {
            Isa::Scalar => true,
            Isa::Avx2 => is_x86_feature_detected!("avx2"),
            Isa::Avx512 => {
                is_x86_feature_detected!("avx2")
                    && is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512bw")
                    && is_x86_feature_detected!("avx512vpopcntdq")
            }
            Isa::Neon => false,
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn host_supports(isa: Isa) -> bool {
        matches!(isa, Isa::Scalar | Isa::Neon)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn host_supports(isa: Isa) -> bool {
        isa == Isa::Scalar
    }

    fn supported_paths() -> Vec<Isa> {
        ALL.into_iter().filter(|isa| host_supports(*isa)).collect()
    }

    fn naive(code: &[u64], planes: &[&[u64]]) -> [u32; 4] {
        let mut out = [0u32; 4];
        for (j, plane) in planes.iter().enumerate() {
            for (w, p) in code.iter().zip(*plane) {
                for bit in 0..64 {
                    out[j] += ((w >> bit) & (p >> bit) & 1) as u32;
                }
            }
        }
        out
    }

    fn random_words(rng: &mut Rng, n: usize) -> Vec<u64> {
        (0..n)
            .map(|_| match rng.below(8) {
                0 => 0,
                1 => u64::MAX,
                2 => 0xAAAA_AAAA_AAAA_AAAA,
                3 => 0x8000_0000_0000_0001,
                _ => rng.u64(),
            })
            .collect()
    }

    #[test]
    fn supported_never_claims_a_path_the_cpu_lacks_and_knows_the_named_ones() {
        for isa in ALL {
            if supported(isa) {
                assert!(host_supports(isa), "{isa:?} claimed but not available");
            }
            if !host_supports(isa) {
                eprintln!(
                    "kernel: {isa:?} is not supported on this CPU: its equivalence tests skip it"
                );
            }
        }
        assert!(supported(Isa::Scalar));
        assert_eq!(
            supported(Isa::Avx512),
            host_supports(Isa::Avx512),
            "avx512f and avx512vpopcntdq"
        );
        assert_eq!(
            supported(Isa::Neon),
            host_supports(Isa::Neon),
            "NEON is baseline on aarch64"
        );
    }

    #[test]
    fn detect_returns_a_supported_path_and_the_fastest_known_one() {
        let isa = detect();
        assert!(host_supports(isa), "{isa:?} is not supported here");
        assert_eq!(isa, detect(), "resolved once per process");
        if host_supports(Isa::Avx512) {
            assert_eq!(isa, Isa::Avx512);
        } else if cfg!(target_arch = "aarch64") {
            assert_eq!(isa, Isa::Neon);
        } else {
            assert!(matches!(isa, Isa::Scalar | Isa::Avx2), "{isa:?}");
        }
    }

    #[test]
    fn known_patterns_have_known_counts() {
        let code = [u64::MAX, 0xFFFF_0000_FFFF_0000, 1];
        let planes: [&[u64]; 4] = [
            &[u64::MAX, u64::MAX, u64::MAX],
            &[0, 0x00FF_00FF_00FF_00FF, 1],
            &[0xAAAA_AAAA_AAAA_AAAA, 0, 0],
            &[1, 1, 0],
        ];
        for isa in supported_paths() {
            assert_eq!(
                and_popcount_4_with(isa, &code, &planes),
                [64 + 32 + 1, 16 + 1, 32, 1],
                "{isa:?}"
            );
        }
        assert_eq!(popcount(&code), 64 + 32 + 1);
        assert_eq!(popcount(&[]), 0);
        assert_eq!(and_popcount_4(&[], &[&[], &[], &[], &[]]), [0; 4]);
    }

    #[test]
    fn every_supported_path_equals_the_naive_count_on_random_words() {
        let mut rng = Rng::new(0xC0FFEE);
        let paths = supported_paths();
        let mut words = 0usize;
        for round in 0..2_000 {
            let len = [
                0usize, 1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 16, 17, 24, 31, 32, 33, 47,
            ][round % 18];
            let code = random_words(&mut rng, len);
            let planes: Vec<Vec<u64>> = (0..4).map(|_| random_words(&mut rng, len)).collect();
            let views: Vec<&[u64]> = planes.iter().map(Vec::as_slice).collect();
            let want = naive(&code, &views);
            for &isa in &paths {
                assert_eq!(
                    and_popcount_4_with(isa, &code, &views),
                    want,
                    "{isa:?}, len = {len}"
                );
            }
            assert_eq!(and_popcount_4(&code, &views), want, "dispatch, len = {len}");
            words += len * 5;
        }
        assert!(words >= 100_000, "{words} random words compared");
    }

    #[test]
    fn paths_agree_at_the_code_widths_in_use_and_at_unaligned_offsets() {
        let mut rng = Rng::new(7);
        let paths = supported_paths();
        for dim in [64usize, 128, 192, 768, 832, 1024, 1536, 2048] {
            let len = dim / 64;
            for offset in 0..4 {
                let code = random_words(&mut rng, len + offset);
                let planes: Vec<Vec<u64>> = (0..4)
                    .map(|_| random_words(&mut rng, len + offset))
                    .collect();
                let views: Vec<&[u64]> = planes.iter().map(|p| &p[offset..]).collect();
                let want = naive(&code[offset..], &views);
                for &isa in &paths {
                    assert_eq!(
                        and_popcount_4_with(isa, &code[offset..], &views),
                        want,
                        "{isa:?}, D = {dim}, offset {offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn popcount_counts_the_set_bits_at_every_length() {
        let mut rng = Rng::new(11);
        for len in 0..70 {
            let code = random_words(&mut rng, len);
            let want: u32 = code.iter().map(|w| w.count_ones()).sum();
            assert_eq!(popcount(&code), want, "len = {len}");
        }
    }
}

#[cfg(test)]
mod hot_path_tests {
    use super::*;
    use crate::local::testkit::Rng;
    use std::panic::catch_unwind;

    const ALL: [Isa; 4] = [Isa::Scalar, Isa::Avx2, Isa::Avx512, Isa::Neon];

    fn words(rng: &mut Rng, n: usize) -> Vec<u64> {
        (0..n).map(|_| rng.u64()).collect()
    }

    #[test]
    fn popcount_with_every_supported_path_equals_the_scalar_count() {
        let mut rng = Rng::new(21);
        for len in 0..140 {
            let code = words(&mut rng, len);
            let want: u32 = code.iter().map(|w| w.count_ones()).sum();
            for isa in ALL.into_iter().filter(|isa| supported(*isa)) {
                assert_eq!(popcount_with(isa, &code), want, "{isa:?}, len = {len}");
            }
        }
    }

    #[test]
    fn a_kernel_exists_for_exactly_the_supported_paths() {
        for isa in ALL {
            let kernel = Kernel::new(isa);
            assert_eq!(kernel.is_some(), supported(isa), "{isa:?}");
            assert_eq!(kernel.map(Kernel::isa), supported(isa).then_some(isa));
        }
        assert_eq!(Kernel::detect().isa(), detect());
        let code = [3u64, 5, 7];
        let ones = [u64::MAX; 3];
        let kernel = Kernel::detect();
        assert_eq!(
            kernel.and_popcount_4(&code, &[&ones, &ones, &ones, &ones]),
            [7, 7, 7, 7]
        );
        assert_eq!(kernel.popcount(&code), 7);
    }

    #[test]
    fn every_supported_path_passes_the_startup_check() {
        for isa in ALL.into_iter().filter(|isa| supported(*isa)) {
            assert!(agrees_with_scalar(isa), "{isa:?}");
        }
    }

    #[test]
    fn paths_agree_beyond_one_byte_accumulator_block() {
        let mut rng = Rng::new(22);
        for len in [248usize, 249, 250, 251, 500, 997] {
            let code = vec![u64::MAX; len];
            let ones = vec![u64::MAX; len];
            let alternating: Vec<u64> = (0..len).map(|_| 0xAAAA_AAAA_AAAA_AAAA).collect();
            let random = words(&mut rng, len);
            let planes: [&[u64]; 4] = [&ones, &alternating, &random, &code];
            let want = scalar::and_popcount_4(&code, planes);
            for isa in ALL.into_iter().filter(|isa| supported(*isa)) {
                assert_eq!(and_popcount_4_with(isa, &code, &planes), want, "{isa:?}");
                assert_eq!(popcount_with(isa, &code), 64 * len as u32, "{isa:?}");
            }
        }
    }

    #[test]
    fn unsupported_paths_and_malformed_planes_panic_instead_of_reading_blindly() {
        let code = [1u64, 2, 3];
        let short = [1u64, 2];
        let full = [1u64, 2, 3];
        for isa in ALL.into_iter().filter(|isa| !supported(*isa)) {
            assert!(
                catch_unwind(|| and_popcount_4_with(isa, &code, &[&full, &full, &full, &full]))
                    .is_err(),
                "{isa:?}"
            );
            assert!(
                catch_unwind(|| popcount_with(isa, &code)).is_err(),
                "{isa:?}"
            );
        }
        assert!(
            catch_unwind(|| and_popcount_4(&code, &[&full, &full, &short, &full])).is_err(),
            "a short plane"
        );
        assert!(
            catch_unwind(|| and_popcount_4(&code, &[&full, &full, &full])).is_err(),
            "three planes"
        );
        let longer = [1u64, 2, 3, 4];
        assert_eq!(
            and_popcount_4(&code, &[&longer, &longer, &longer, &longer]),
            and_popcount_4(&code, &[&full, &full, &full, &full]),
            "extra plane words are ignored"
        );
    }
}

#[cfg(test)]
mod block_tests {
    use super::*;
    use crate::local::testkit::Rng;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    const ALL: [Isa; 4] = [Isa::Scalar, Isa::Avx2, Isa::Avx512, Isa::Neon];

    type BlockFn = fn(&[u8], &[u8], usize, &mut [u32; BLOCK_MAX_LANES]);

    fn levels(rng: &mut Rng, d: usize) -> Vec<u8> {
        (0..d).map(|_| rng.below(16) as u8).collect()
    }

    /// One table per group of four coordinates: entry `v` sums the levels of the coordinates whose
    /// bit is set in `v`.
    fn lut_of(levels: &[u8]) -> Vec<u8> {
        let mut lut = Vec::new();
        for group in levels.as_chunks::<4>().0 {
            for v in 0..16usize {
                lut.push(
                    (0..4)
                        .map(|t| if v >> t & 1 == 1 { group[t] } else { 0 })
                        .sum(),
                );
            }
        }
        lut
    }

    fn planes_of(levels: &[u8]) -> [Vec<u64>; 4] {
        std::array::from_fn(|j| {
            levels
                .as_chunks::<64>()
                .0
                .iter()
                .map(|word| {
                    word.iter()
                        .enumerate()
                        .fold(0u64, |w, (i, l)| w | u64::from(l >> j & 1) << i)
                })
                .collect()
        })
    }

    fn plane_acc(code: &[u64], planes: &[Vec<u64>; 4]) -> u32 {
        let views: [&[u64]; 4] = std::array::from_fn(|j| planes[j].as_slice());
        let counts = scalar::and_popcount_4(code, views);
        counts[0] + 2 * counts[1] + 4 * counts[2] + 8 * counts[3]
    }

    /// `rows[j * lanes + r]` = byte `j` of record `r`; lanes past `codes.len()` are zero.
    fn interleave(codes: &[Vec<u64>], lanes: usize, code_bytes: usize) -> Vec<u8> {
        let mut rows = vec![0u8; code_bytes * lanes];
        for (r, code) in codes.iter().enumerate() {
            for j in 0..code_bytes {
                rows[j * lanes + r] = code[j / 8].to_le_bytes()[j % 8];
            }
        }
        rows
    }

    fn supported_paths() -> Vec<Isa> {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            for isa in ALL.into_iter().filter(|isa| !supported(*isa)) {
                eprintln!("kernel: {isa:?} is not supported on this CPU: the block tests skip it");
            }
        });
        ALL.into_iter().filter(|isa| supported(*isa)).collect()
    }

    /// The definition of `acc`, independent of `scalar::acc_block`.
    fn naive(lut: &[u8], rows: &[u8], lanes: usize) -> Vec<u32> {
        (0..lanes)
            .map(|r| {
                (0..lut.len() / 32)
                    .map(|j| {
                        let b = usize::from(rows[j * lanes + r]);
                        u32::from(lut[32 * j + (b & 15)]) + u32::from(lut[32 * j + 16 + (b >> 4)])
                    })
                    .sum()
            })
            .collect()
    }

    /// Bytes with the nibble and sign-bit edges over-represented.
    fn edge_bytes(rng: &mut Rng, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| match rng.below(8) {
                0 => 0x00,
                1 => 0xFF,
                2 => 0x0F,
                3 => 0xF0,
                4 => 0x80,
                5 => 0x08,
                _ => rng.u64() as u8,
            })
            .collect()
    }

    /// Table entries up to 63 (not only sums of four levels), the ends over-represented.
    fn edge_entries(rng: &mut Rng, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| match rng.below(6) {
                0 => 0,
                1 => 63,
                _ => rng.below(64) as u8,
            })
            .collect()
    }

    fn run(isa: Isa, lut: &[u8], rows: &[u8], lanes: usize) -> [u32; BLOCK_MAX_LANES] {
        let mut out = [u32::MAX; BLOCK_MAX_LANES];
        acc_block_with(isa, lut, rows, lanes, &mut out);
        out
    }

    #[test]
    fn block_acc_equals_the_plane_popcounts_on_every_supported_path() {
        let mut rng = Rng::new(0xB10C);
        for d in [256usize, 512, 768, 832, 1536, 2048] {
            let (words, code_bytes) = (d / 64, d / 8);
            for lanes in [16usize, 32, 64] {
                for live in [1, lanes / 2 + 1, lanes] {
                    let levels = levels(&mut rng, d);
                    let (lut, planes) = (lut_of(&levels), planes_of(&levels));
                    let codes: Vec<Vec<u64>> = (0..live)
                        .map(|_| (0..words).map(|_| rng.u64()).collect())
                        .collect();
                    let rows = interleave(&codes, lanes, code_bytes);
                    let want: Vec<u32> = codes.iter().map(|c| plane_acc(c, &planes)).collect();
                    for isa in supported_paths() {
                        let mut out = [u32::MAX; BLOCK_MAX_LANES];
                        Kernel::new(isa)
                            .unwrap()
                            .acc_block(&lut, &rows, lanes, &mut out);
                        assert_eq!(&out[..live], &want[..], "{isa:?}, D = {d}, {lanes} lanes");
                        assert!(
                            out[live..lanes].iter().all(|a| *a == 0),
                            "{isa:?}: padding lanes hold the zero code"
                        );
                        assert!(
                            out[lanes..].iter().all(|a| *a == u32::MAX),
                            "{isa:?}: lanes past the block are untouched"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn every_supported_path_equals_the_definition_on_random_tables() {
        let mut rng = Rng::new(0xFA57);
        let paths = supported_paths();
        eprintln!("kernel: block paths run: {paths:?}");
        for lanes in [16usize, 32, 64] {
            for rows in [0usize, 2, 4, 6, 14, 32, 34, 96, 104, 192, 256, 258] {
                for round in 0..4 {
                    let lut = edge_entries(&mut rng, rows * 32);
                    let bytes = edge_bytes(&mut rng, rows * lanes);
                    let want = naive(&lut, &bytes, lanes);
                    for &isa in &paths {
                        let out = run(isa, &lut, &bytes, lanes);
                        assert_eq!(
                            &out[..lanes],
                            &want[..],
                            "{isa:?}, {rows} rows, {lanes} lanes, round {round}"
                        );
                        assert!(out[lanes..].iter().all(|a| *a == u32::MAX), "{isa:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_extremes_fill_the_byte_and_the_16_bit_sums_and_longer_blocks_split_cleanly() {
        let paths = supported_paths();
        for lanes in [16usize, 32, 64] {
            for rows in [2usize, 32, 256, 510, 512, 514, 1030, 1536] {
                // Entry 0 of every table is 0, the others 63: a zero byte adds nothing, so the lanes
                // that hold none sum to 0 next to the ones that hold only 0xFF.
                let lut: Vec<u8> = (0..rows * 32)
                    .map(|i| if i % 16 == 0 { 0 } else { 63 })
                    .collect();
                for pattern in 0..3 {
                    let name = ["all bytes", "even lanes", "odd lanes"][pattern];
                    let byte = |r: usize| {
                        if pattern == 0 || r % 2 == pattern - 1 {
                            0xFF
                        } else {
                            0
                        }
                    };
                    let bytes: Vec<u8> = (0..rows * lanes).map(|i| byte(i % lanes)).collect();
                    let want = naive(&lut, &bytes, lanes);
                    for &isa in &paths {
                        let out = run(isa, &lut, &bytes, lanes);
                        assert_eq!(&out[..lanes], &want[..], "{isa:?}, {name}, {rows} rows");
                    }
                    for (lane, sum) in want.iter().enumerate() {
                        let full = pattern == 0 || lane % 2 == pattern - 1;
                        assert_eq!(
                            *sum,
                            if full { 126 * rows as u32 } else { 0 },
                            "{name}, {rows} rows, lane {lane}"
                        );
                    }
                }
            }
        }
        // The widest block a page holds (D = 2,048) and one a 16-bit sum could not.
        assert_eq!(
            run(Isa::Scalar, &[63; 256 * 32], &[0xFF; 256 * 16], 16)[0],
            32_256
        );
        assert_eq!(
            run(Isa::Scalar, &[63; 1030 * 32], &[0xFF; 1030 * 16], 16)[15],
            129_780
        );
    }

    #[test]
    fn unaligned_rows_and_tables_give_the_same_sums() {
        let mut rng = Rng::new(0xA11);
        let paths = supported_paths();
        for lanes in [16usize, 32, 64] {
            for rows in [2usize, 32, 96] {
                let lut = edge_entries(&mut rng, rows * 32);
                let bytes = edge_bytes(&mut rng, rows * lanes);
                let want = naive(&lut, &bytes, lanes);
                for offset in 1..8 {
                    let mut big_lut = vec![0xA5u8; offset];
                    big_lut.extend_from_slice(&lut);
                    big_lut.extend_from_slice(&[0xA5; 64]);
                    let mut big_rows = vec![0x5Au8; 8 - offset];
                    big_rows.extend_from_slice(&bytes);
                    big_rows.extend_from_slice(&[0x5A; 64]);
                    let lut_at = &big_lut[offset..offset + lut.len()];
                    let rows_at = &big_rows[8 - offset..8 - offset + bytes.len()];
                    for &isa in &paths {
                        let out = run(isa, lut_at, rows_at, lanes);
                        assert_eq!(
                            &out[..lanes],
                            &want[..],
                            "{isa:?}, {rows} rows, {lanes} lanes, offset {offset}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn repeated_calls_into_one_out_leak_no_state() {
        let mut rng = Rng::new(0x5EED);
        for isa in supported_paths() {
            let kernel = Kernel::new(isa).unwrap();
            let mut out = [7u32; BLOCK_MAX_LANES];
            let big = (vec![63u8; 96 * 32], vec![0xFFu8; 96 * 64]);
            kernel.acc_block(&big.0, &big.1, 64, &mut out);
            assert!(out.iter().all(|a| *a == 126 * 96), "{isa:?}");
            for lanes in [64usize, 16, 32, 16, 64] {
                let before = out;
                let lut = edge_entries(&mut rng, 32 * 32);
                let bytes = edge_bytes(&mut rng, 32 * lanes);
                kernel.acc_block(&lut, &bytes, lanes, &mut out);
                assert_eq!(&out[..lanes], &naive(&lut, &bytes, lanes)[..], "{isa:?}");
                assert_eq!(out[lanes..], before[lanes..], "{isa:?}: past the block");
            }
            kernel.acc_block(&[], &[], 32, &mut out);
            assert!(out[..32].iter().all(|a| *a == 0), "{isa:?}: no rows");
        }
    }

    #[test]
    fn malformed_blocks_and_unsupported_paths_panic_instead_of_reading_blindly() {
        let lut = vec![1u8; 4 * 32];
        let rows16 = vec![1u8; 4 * 16];
        for isa in supported_paths() {
            let kernel = Kernel::new(isa).unwrap();
            let mut out = [u32::MAX; BLOCK_MAX_LANES];
            let bad = |lut: &[u8], rows: &[u8], lanes: usize| {
                let mut scratch = out;
                let call = catch_unwind(AssertUnwindSafe(|| {
                    kernel.acc_block(lut, rows, lanes, &mut scratch)
                }));
                if call.is_err() {
                    assert_eq!(scratch, out, "{isa:?}: a refused call wrote");
                }
                call.is_err()
            };
            for lanes in [0usize, 8, 17, 24, 48, 128] {
                assert!(
                    bad(&lut, &vec![1u8; 4 * lanes], lanes),
                    "{isa:?}, {lanes} lanes"
                );
            }
            for len in [1usize, 32, 65, 96, 4 * 32 + 32] {
                let rows = vec![1u8; len / 32 * 16];
                assert!(
                    bad(&vec![1u8; len], &rows, 16),
                    "{isa:?}, a table of {len} bytes"
                );
            }
            assert!(bad(&lut, &rows16[1..], 16), "{isa:?}: one row byte short");
            assert!(
                bad(&lut, &[1u8; 4 * 16 + 1], 16),
                "{isa:?}: one row byte long"
            );
            assert!(
                bad(&lut, &[1u8; 4 * 32], 16),
                "{isa:?}: rows of a wider block"
            );
            assert!(bad(&lut, &[], 16), "{isa:?}: no rows");
            assert!(bad(&[], &rows16, 16), "{isa:?}: no table");
            assert!(!bad(&lut, &rows16, 16), "{isa:?}: the well-formed call");
            kernel.acc_block(&lut, &rows16, 16, &mut out);
            assert!(
                out[..16].iter().all(|a| *a == 8),
                "{isa:?}: 4 rows of 1 + 1"
            );
        }
        for isa in ALL.into_iter().filter(|isa| !supported(*isa)) {
            assert!(Kernel::new(isa).is_none());
            let mut out = [0u32; BLOCK_MAX_LANES];
            let call = catch_unwind(AssertUnwindSafe(|| {
                acc_block_with(isa, &lut, &rows16, 16, &mut out)
            }));
            assert!(call.is_err(), "{isa:?}");
        }
    }

    #[test]
    fn the_startup_check_rejects_a_kernel_that_overflows_or_swaps_lanes() {
        let mut state = 0x1234_5678_9ABC_DEF1u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for isa in supported_paths() {
            assert!(
                block_agrees(&mut next, |l, r, n, o| acc_block_on(isa, l, r, n, o)),
                "{isa:?}"
            );
        }
        let broken: [(&str, BlockFn); 3] = [
            ("byte sums", |lut, rows, lanes, out| {
                out[..lanes].fill(0);
                for (row, tables) in rows.chunks_exact(lanes).zip(lut.as_chunks::<32>().0) {
                    for (acc, &b) in out.iter_mut().zip(row) {
                        let sum = tables[usize::from(b & 15)]
                            .wrapping_add(tables[16 + usize::from(b >> 4)]);
                        *acc = u32::from((*acc as u8).wrapping_add(sum));
                    }
                }
            }),
            ("swapped lanes", |lut, rows, lanes, out| {
                scalar::acc_block(lut, rows, lanes, out);
                out.swap(0, 1);
            }),
            ("a dropped last pair", |lut, rows, lanes, out| {
                let n = lut.len().saturating_sub(64);
                scalar::acc_block(&lut[..n], &rows[..n / 32 * lanes], lanes, out);
            }),
        ];
        for (name, candidate) in broken {
            assert!(!block_agrees(&mut next, candidate), "{name}");
        }
    }

    /// The startup check's extreme cases must differ between even and odd lanes, or they do not
    /// test what they are for (under all-63 tables every fill summed alike).
    #[test]
    fn the_startup_extremes_are_lopsided_between_even_and_odd_lanes() {
        let full = (SEGMENT_ROWS * 126) as u32;
        for lanes in [16usize, 32, 64] {
            let cases = extreme_cases(lanes);
            assert_eq!(cases.len(), 3);
            for (case, (lut, bytes)) in cases.iter().enumerate() {
                let mut out = [0u32; BLOCK_MAX_LANES];
                scalar::acc_block(lut, bytes, lanes, &mut out);
                for (lane, sum) in out[..lanes].iter().enumerate() {
                    let want = if case == 0 || lane % 2 == case - 1 {
                        full
                    } else {
                        0
                    };
                    assert_eq!(*sum, want, "{lanes} lanes, case {case}, lane {lane}");
                }
            }
        }
    }

    /// Random shapes through the checked entry point of the detected kernel, past the segment
    /// length, equal the definition; a slice of the wrong length is refused before any write.
    #[test]
    fn random_shapes_through_the_checked_entry_point_match_and_a_wrong_length_is_refused() {
        let mut rng = Rng::new(0xF00D);
        let kernel = Kernel::detect();
        for round in 0..120 {
            let lanes = [16usize, 32, 64][rng.below(3)];
            let rows = 2 * rng.below(700);
            let lut: Vec<u8> = (0..rows * 32).map(|_| rng.below(64) as u8).collect();
            let bytes: Vec<u8> = (0..rows * lanes).map(|_| rng.u64() as u8).collect();
            let mut out = [7u32; BLOCK_MAX_LANES];
            kernel.acc_block(&lut, &bytes, lanes, &mut out);
            assert_eq!(
                &out[..lanes],
                &naive(&lut, &bytes, lanes)[..],
                "round {round}: {lanes} lanes, {rows} rows"
            );
            assert!(out[lanes..].iter().all(|a| *a == 7), "round {round}");
            if !bytes.is_empty() {
                let mut untouched = [9u32; BLOCK_MAX_LANES];
                let refused = catch_unwind(AssertUnwindSafe(|| {
                    kernel.acc_block(&lut, &bytes[..bytes.len() - 1], lanes, &mut untouched);
                }));
                assert!(refused.is_err(), "round {round}: a short row slice");
                assert_eq!(untouched, [9u32; BLOCK_MAX_LANES], "round {round}");
            }
        }
    }

    #[test]
    fn the_detected_kernel_dispatches_the_block_call() {
        let mut rng = Rng::new(3);
        let lut = edge_entries(&mut rng, 96 * 32);
        let bytes = edge_bytes(&mut rng, 96 * 64);
        let mut out = [0u32; BLOCK_MAX_LANES];
        Kernel::detect().acc_block(&lut, &bytes, 64, &mut out);
        assert_eq!(&out[..], &naive(&lut, &bytes, 64)[..]);
    }
}
