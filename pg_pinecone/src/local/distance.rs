//! Exact f32 kernels behind `metric.rs`: squared L2, dot product and cosine distance, dispatched
//! once per process to AVX-512F, AVX2+FMA, NEON or a portable path. Every path keeps at least 16
//! independent accumulator lanes, so no add waits on the previous one.

#![cfg_attr(not(test), allow(dead_code))]

use std::sync::OnceLock;

const LANES: usize = 16;
// 157 floats run every loop stage of every path: body, single step and tail.
const SELF_CHECK_DIM: usize = 157;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kernel {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

pub(crate) fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: `best` only returns a kernel this CPU supports.
    unsafe { Kernel::best().l2_sq_unchecked(a, b) }
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: `best` only returns a kernel this CPU supports.
    unsafe { Kernel::best().dot_unchecked(a, b) }
}

/// Cosine distance `1 - cos`; 1.0 when either vector has zero norm.
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: `best` only returns a kernel this CPU supports.
    unsafe { Kernel::best().cosine_unchecked(a, b) }
}

impl Kernel {
    /// Narrowest path first.
    #[cfg(target_arch = "x86_64")]
    pub(crate) const ALL: &'static [Kernel] = &[Kernel::Scalar, Kernel::Avx2, Kernel::Avx512];
    #[cfg(target_arch = "aarch64")]
    pub(crate) const ALL: &'static [Kernel] = &[Kernel::Scalar, Kernel::Neon];
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) const ALL: &'static [Kernel] = &[Kernel::Scalar];

    pub(crate) fn best() -> Kernel {
        static BEST: OnceLock<Kernel> = OnceLock::new();
        *BEST.get_or_init(detect)
    }

    fn agrees_with_scalar(self) -> bool {
        let a: Vec<f32> = (0..SELF_CHECK_DIM)
            .map(|i| 0.25 + (i % 7) as f32 * 0.5)
            .collect();
        let b: Vec<f32> = (0..SELF_CHECK_DIM)
            .map(|i| 1.5 - (i % 5) as f32 * 0.25)
            .collect();
        let close = |x: f32, y: f32| (x - y).abs() <= 1e-4 * y.abs().max(1.0);
        // SAFETY: callers pass a supported kernel; scalar is always supported.
        unsafe {
            close(
                self.l2_sq_unchecked(&a, &b),
                Kernel::Scalar.l2_sq_unchecked(&a, &b),
            ) && close(
                self.dot_unchecked(&a, &b),
                Kernel::Scalar.dot_unchecked(&a, &b),
            ) && close(
                self.cosine_unchecked(&a, &b),
                Kernel::Scalar.cosine_unchecked(&a, &b),
            )
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Kernel::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => "avx2+fma",
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => "avx512f",
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => "neon",
        }
    }

    pub(crate) fn supported(self) -> bool {
        match self {
            Kernel::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => is_x86_feature_detected!("avx512f"),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => std::arch::is_aarch64_feature_detected!("neon"),
        }
    }

    // Skylake and Cascade Lake lose more clock to 512-bit FMA than the kernel saves; Ice Lake and
    // Zen 4 (the parts with VPOPCNTDQ, like the Hamming kernel's gate) do not.
    fn preferred(self) -> bool {
        match self {
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => self.supported() && is_x86_feature_detected!("avx512vpopcntdq"),
            _ => self.supported(),
        }
    }

    fn require_supported(self) {
        assert!(
            self.supported(),
            "{} is not supported by this CPU",
            self.name()
        );
    }

    pub(crate) fn l2_sq(self, a: &[f32], b: &[f32]) -> f32 {
        self.require_supported();
        // SAFETY: just checked.
        unsafe { self.l2_sq_unchecked(a, b) }
    }

    pub(crate) fn dot(self, a: &[f32], b: &[f32]) -> f32 {
        self.require_supported();
        // SAFETY: just checked.
        unsafe { self.dot_unchecked(a, b) }
    }

    pub(crate) fn cosine(self, a: &[f32], b: &[f32]) -> f32 {
        self.require_supported();
        // SAFETY: just checked.
        unsafe { self.cosine_unchecked(a, b) }
    }

    /// The caller must have checked [`Kernel::supported`]. A shorter `b` truncates, never over-reads.
    pub(crate) unsafe fn l2_sq_unchecked(self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        match self {
            Kernel::Scalar => scalar::sum::<true>(a, b),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => unsafe { x86::sum_avx2::<true>(a, b) },
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe { x86::sum_avx512::<true>(a, b) },
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => unsafe { neon::sum::<true>(a, b) },
        }
    }

    /// See [`Kernel::l2_sq_unchecked`].
    pub(crate) unsafe fn dot_unchecked(self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        match self {
            Kernel::Scalar => scalar::sum::<false>(a, b),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => unsafe { x86::sum_avx2::<false>(a, b) },
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe { x86::sum_avx512::<false>(a, b) },
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => unsafe { neon::sum::<false>(a, b) },
        }
    }

    /// See [`Kernel::l2_sq_unchecked`].
    pub(crate) unsafe fn cosine_unchecked(self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let (ab, aa, bb) = match self {
            Kernel::Scalar => scalar::dot3(a, b),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => unsafe { x86::dot3_avx2(a, b) },
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe { x86::dot3_avx512(a, b) },
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => unsafe { neon::dot3(a, b) },
        };
        let denom = aa.sqrt() * bb.sqrt();
        if denom == 0.0 { 1.0 } else { 1.0 - ab / denom }
    }
}

fn detect() -> Kernel {
    for &kernel in Kernel::ALL.iter().rev() {
        if !kernel.preferred() {
            continue;
        }
        if kernel.agrees_with_scalar() {
            return kernel;
        }
        eprintln!(
            "pg_pinecone: distance kernel {} disagrees with the scalar path, not using it",
            kernel.name()
        );
    }
    Kernel::Scalar
}

#[inline(always)]
fn term<const L2: bool>(p: f32, q: f32) -> f32 {
    if L2 {
        let d = p - q;
        d * d
    } else {
        p * q
    }
}

mod scalar {
    use super::{LANES, term};

    fn reduce(v: [f32; LANES]) -> f32 {
        let h: [f32; 8] = std::array::from_fn(|i| v[i] + v[i + 8]);
        let q: [f32; 4] = std::array::from_fn(|i| h[i] + h[i + 4]);
        (q[0] + q[2]) + (q[1] + q[3])
    }

    pub(super) fn sum<const L2: bool>(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (ca, ta) = a[..n].as_chunks::<LANES>();
        let (cb, tb) = b[..n].as_chunks::<LANES>();
        let mut acc = [0f32; LANES];
        for (x, y) in ca.iter().zip(cb) {
            for ((s, &p), &q) in acc.iter_mut().zip(x).zip(y) {
                *s += term::<L2>(p, q);
            }
        }
        let mut total = reduce(acc);
        for (&p, &q) in ta.iter().zip(tb) {
            total += term::<L2>(p, q);
        }
        total
    }

    pub(super) fn dot3(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        let n = a.len().min(b.len());
        let (ca, ta) = a[..n].as_chunks::<LANES>();
        let (cb, tb) = b[..n].as_chunks::<LANES>();
        let (mut ab, mut aa, mut bb) = ([0f32; LANES], [0f32; LANES], [0f32; LANES]);
        for (x, y) in ca.iter().zip(cb) {
            for (i, (&p, &q)) in x.iter().zip(y).enumerate() {
                ab[i] += p * q;
                aa[i] += p * p;
                bb[i] += q * q;
            }
        }
        let (mut sab, mut saa, mut sbb) = (reduce(ab), reduce(aa), reduce(bb));
        for (&p, &q) in ta.iter().zip(tb) {
            sab += p * q;
            saa += p * p;
            sbb += q * q;
        }
        (sab, saa, sbb)
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::term;
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn step256<const L2: bool>(x: __m256, y: __m256, acc: __m256) -> __m256 {
        if L2 {
            let d = _mm256_sub_ps(x, y);
            _mm256_fmadd_ps(d, d, acc)
        } else {
            _mm256_fmadd_ps(x, y, acc)
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn hsum256(v: __m256) -> f32 {
        let q = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let d = _mm_add_ps(q, _mm_movehl_ps(q, q));
        _mm_cvtss_f32(_mm_add_ss(d, _mm_shuffle_ps::<1>(d, d)))
    }

    #[derive(Clone, Copy)]
    struct Acc256 {
        ab: __m256,
        aa: __m256,
        bb: __m256,
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn step256x3(x: __m256, y: __m256, acc: Acc256) -> Acc256 {
        Acc256 {
            ab: _mm256_fmadd_ps(x, y, acc.ab),
            aa: _mm256_fmadd_ps(x, x, acc.aa),
            bb: _mm256_fmadd_ps(y, y, acc.bb),
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn sum_avx2<const L2: bool>(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let n = a.len().min(b.len());
            let (pa, pb) = (a.as_ptr(), b.as_ptr());
            let mut acc = [_mm256_setzero_ps(); 4];
            let mut i = 0;
            while i + 32 <= n {
                for (j, s) in acc.iter_mut().enumerate() {
                    let x = _mm256_loadu_ps(pa.add(i + 8 * j));
                    let y = _mm256_loadu_ps(pb.add(i + 8 * j));
                    *s = step256::<L2>(x, y, *s);
                }
                i += 32;
            }
            while i + 8 <= n {
                let x = _mm256_loadu_ps(pa.add(i));
                let y = _mm256_loadu_ps(pb.add(i));
                acc[0] = step256::<L2>(x, y, acc[0]);
                i += 8;
            }
            let pairs = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
            let mut total = hsum256(pairs);
            while i < n {
                total += term::<L2>(*pa.add(i), *pb.add(i));
                i += 1;
            }
            total
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot3_avx2(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        unsafe {
            let n = a.len().min(b.len());
            let (pa, pb) = (a.as_ptr(), b.as_ptr());
            let zero = _mm256_setzero_ps();
            let mut acc = [Acc256 {
                ab: zero,
                aa: zero,
                bb: zero,
            }; 2];
            let mut i = 0;
            while i + 16 <= n {
                for (j, s) in acc.iter_mut().enumerate() {
                    let x = _mm256_loadu_ps(pa.add(i + 8 * j));
                    let y = _mm256_loadu_ps(pb.add(i + 8 * j));
                    *s = step256x3(x, y, *s);
                }
                i += 16;
            }
            while i + 8 <= n {
                let x = _mm256_loadu_ps(pa.add(i));
                let y = _mm256_loadu_ps(pb.add(i));
                acc[0] = step256x3(x, y, acc[0]);
                i += 8;
            }
            let mut ab = hsum256(_mm256_add_ps(acc[0].ab, acc[1].ab));
            let mut aa = hsum256(_mm256_add_ps(acc[0].aa, acc[1].aa));
            let mut bb = hsum256(_mm256_add_ps(acc[0].bb, acc[1].bb));
            while i < n {
                let (p, q) = (*pa.add(i), *pb.add(i));
                ab += p * q;
                aa += p * p;
                bb += q * q;
                i += 1;
            }
            (ab, aa, bb)
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    fn step512<const L2: bool>(x: __m512, y: __m512, acc: __m512) -> __m512 {
        if L2 {
            let d = _mm512_sub_ps(x, y);
            _mm512_fmadd_ps(d, d, acc)
        } else {
            _mm512_fmadd_ps(x, y, acc)
        }
    }

    #[derive(Clone, Copy)]
    struct Acc512 {
        ab: __m512,
        aa: __m512,
        bb: __m512,
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    fn step512x3(x: __m512, y: __m512, acc: Acc512) -> Acc512 {
        Acc512 {
            ab: _mm512_fmadd_ps(x, y, acc.ab),
            aa: _mm512_fmadd_ps(x, x, acc.aa),
            bb: _mm512_fmadd_ps(y, y, acc.bb),
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn sum_avx512<const L2: bool>(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let n = a.len().min(b.len());
            let (pa, pb) = (a.as_ptr(), b.as_ptr());
            let mut acc = [_mm512_setzero_ps(); 4];
            let mut i = 0;
            while i + 64 <= n {
                for (j, s) in acc.iter_mut().enumerate() {
                    let x = _mm512_loadu_ps(pa.add(i + 16 * j));
                    let y = _mm512_loadu_ps(pb.add(i + 16 * j));
                    *s = step512::<L2>(x, y, *s);
                }
                i += 64;
            }
            while i + 16 <= n {
                let x = _mm512_loadu_ps(pa.add(i));
                let y = _mm512_loadu_ps(pb.add(i));
                acc[0] = step512::<L2>(x, y, acc[0]);
                i += 16;
            }
            if i < n {
                let mask = ((1u32 << (n - i)) - 1) as u16;
                let x = _mm512_maskz_loadu_ps(mask, pa.add(i));
                let y = _mm512_maskz_loadu_ps(mask, pb.add(i));
                acc[0] = step512::<L2>(x, y, acc[0]);
            }
            _mm512_reduce_add_ps(_mm512_add_ps(
                _mm512_add_ps(acc[0], acc[1]),
                _mm512_add_ps(acc[2], acc[3]),
            ))
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn dot3_avx512(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        unsafe {
            let n = a.len().min(b.len());
            let (pa, pb) = (a.as_ptr(), b.as_ptr());
            let zero = _mm512_setzero_ps();
            let mut acc = [Acc512 {
                ab: zero,
                aa: zero,
                bb: zero,
            }; 2];
            let mut i = 0;
            while i + 32 <= n {
                for (j, s) in acc.iter_mut().enumerate() {
                    let x = _mm512_loadu_ps(pa.add(i + 16 * j));
                    let y = _mm512_loadu_ps(pb.add(i + 16 * j));
                    *s = step512x3(x, y, *s);
                }
                i += 32;
            }
            while i + 16 <= n {
                let x = _mm512_loadu_ps(pa.add(i));
                let y = _mm512_loadu_ps(pb.add(i));
                acc[0] = step512x3(x, y, acc[0]);
                i += 16;
            }
            if i < n {
                let mask = ((1u32 << (n - i)) - 1) as u16;
                let x = _mm512_maskz_loadu_ps(mask, pa.add(i));
                let y = _mm512_maskz_loadu_ps(mask, pb.add(i));
                acc[0] = step512x3(x, y, acc[0]);
            }
            (
                _mm512_reduce_add_ps(_mm512_add_ps(acc[0].ab, acc[1].ab)),
                _mm512_reduce_add_ps(_mm512_add_ps(acc[0].aa, acc[1].aa)),
                _mm512_reduce_add_ps(_mm512_add_ps(acc[0].bb, acc[1].bb)),
            )
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::term;
    use std::arch::aarch64::*;

    #[inline]
    #[target_feature(enable = "neon")]
    fn step<const L2: bool>(x: float32x4_t, y: float32x4_t, acc: float32x4_t) -> float32x4_t {
        if L2 {
            let d = vsubq_f32(x, y);
            vfmaq_f32(acc, d, d)
        } else {
            vfmaq_f32(acc, x, y)
        }
    }

    #[derive(Clone, Copy)]
    struct Acc {
        ab: float32x4_t,
        aa: float32x4_t,
        bb: float32x4_t,
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn step3(x: float32x4_t, y: float32x4_t, acc: Acc) -> Acc {
        Acc {
            ab: vfmaq_f32(acc.ab, x, y),
            aa: vfmaq_f32(acc.aa, x, x),
            bb: vfmaq_f32(acc.bb, y, y),
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn sum<const L2: bool>(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let n = a.len().min(b.len());
            let (pa, pb) = (a.as_ptr(), b.as_ptr());
            let mut acc = [vdupq_n_f32(0.0); 8];
            let mut i = 0;
            while i + 32 <= n {
                for (j, s) in acc.iter_mut().enumerate() {
                    let x = vld1q_f32(pa.add(i + 4 * j));
                    let y = vld1q_f32(pb.add(i + 4 * j));
                    *s = step::<L2>(x, y, *s);
                }
                i += 32;
            }
            while i + 4 <= n {
                let x = vld1q_f32(pa.add(i));
                let y = vld1q_f32(pb.add(i));
                acc[0] = step::<L2>(x, y, acc[0]);
                i += 4;
            }
            let lo = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
            let hi = vaddq_f32(vaddq_f32(acc[4], acc[5]), vaddq_f32(acc[6], acc[7]));
            let mut total = vaddvq_f32(vaddq_f32(lo, hi));
            while i < n {
                total += term::<L2>(*pa.add(i), *pb.add(i));
                i += 1;
            }
            total
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn dot3(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        unsafe {
            let n = a.len().min(b.len());
            let (pa, pb) = (a.as_ptr(), b.as_ptr());
            let zero = vdupq_n_f32(0.0);
            let mut acc = [Acc {
                ab: zero,
                aa: zero,
                bb: zero,
            }; 4];
            let mut i = 0;
            while i + 16 <= n {
                for (j, s) in acc.iter_mut().enumerate() {
                    let x = vld1q_f32(pa.add(i + 4 * j));
                    let y = vld1q_f32(pb.add(i + 4 * j));
                    *s = step3(x, y, *s);
                }
                i += 16;
            }
            while i + 4 <= n {
                let x = vld1q_f32(pa.add(i));
                let y = vld1q_f32(pb.add(i));
                acc[0] = step3(x, y, acc[0]);
                i += 4;
            }
            let fold = |pick: fn(&Acc) -> float32x4_t| {
                let lo = vaddq_f32(pick(&acc[0]), pick(&acc[1]));
                let hi = vaddq_f32(pick(&acc[2]), pick(&acc[3]));
                vaddvq_f32(vaddq_f32(lo, hi))
            };
            let (mut ab, mut aa, mut bb) = (fold(|s| s.ab), fold(|s| s.aa), fold(|s| s.bb));
            while i < n {
                let (p, q) = (*pa.add(i), *pb.add(i));
                ab += p * q;
                aa += p * p;
                bb += q * q;
                i += 1;
            }
            (ab, aa, bb)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }

        fn unit(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Shape {
        MixedSigns,
        AllPositive,
        MagnitudesFarApart,
    }

    fn vector(rng: &mut Rng, n: usize, shape: Shape, side: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let u = rng.unit();
                match shape {
                    Shape::MixedSigns => u,
                    Shape::AllPositive => u + 1.0,
                    Shape::MagnitudesFarApart => u * if side == 0 { 1e4 } else { 1e-4 },
                }
            })
            .collect()
    }

    fn dot64(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
    }

    fn abs_dot64(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x as f64 * y as f64).abs())
            .sum()
    }

    fn l2_sq64(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
            .sum()
    }

    fn cosine64(a: &[f32], b: &[f32]) -> f64 {
        let denom = dot64(a, a).sqrt() * dot64(b, b).sqrt();
        if denom == 0.0 {
            1.0
        } else {
            1.0 - dot64(a, b) / denom
        }
    }

    const TOLERANCE: f64 = 1e-5;
    const DIMS: [usize; 5] = [3, 64, 768, 1536, 2000];
    const TAIL_DIMS: [usize; 22] = [
        0, 1, 2, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 65, 100, 127, 129, 255, 257, 1000, 1999,
    ];

    fn supported_kernels() -> impl Iterator<Item = Kernel> {
        Kernel::ALL.iter().copied().filter(|k| k.supported())
    }

    // Mixed-sign dot products cancel, so their error is judged against sum |a_i * b_i|.
    fn check(kernel: Kernel, a: &[f32], b: &[f32], context: &str) {
        let name = kernel.name();
        let got = kernel.l2_sq(a, b) as f64;
        let want = l2_sq64(a, b);
        assert!(
            (got - want).abs() <= TOLERANCE * want,
            "{name} l2_sq {context}: got {got}, want {want}"
        );
        let got = kernel.dot(a, b) as f64;
        let want = dot64(a, b);
        let scale = abs_dot64(a, b);
        assert!(
            (got - want).abs() <= TOLERANCE * scale,
            "{name} dot {context}: got {got}, want {want}, scale {scale}"
        );
        let got = kernel.cosine(a, b) as f64;
        let want = cosine64(a, b);
        assert!(
            (got - want).abs() <= TOLERANCE,
            "{name} cosine {context}: got {got}, want {want}"
        );
    }

    #[test]
    fn every_supported_path_matches_the_f64_oracle() {
        let mut rng = Rng(0x5EED_0001);
        let (mut tested, mut skipped) = (Vec::new(), Vec::new());
        for &kernel in Kernel::ALL {
            if !kernel.supported() {
                skipped.push(kernel.name());
                continue;
            }
            tested.push(kernel.name());
            for shape in [
                Shape::MixedSigns,
                Shape::AllPositive,
                Shape::MagnitudesFarApart,
            ] {
                for &dim in &DIMS {
                    for trial in 0..24 {
                        let a = vector(&mut rng, dim, shape, 0);
                        let b = vector(&mut rng, dim, shape, 1);
                        check(
                            kernel,
                            &a,
                            &b,
                            &format!("{shape:?} dim {dim} trial {trial}"),
                        );
                    }
                }
                for &dim in &TAIL_DIMS {
                    let a = vector(&mut rng, dim, shape, 0);
                    let b = vector(&mut rng, dim, shape, 1);
                    check(kernel, &a, &b, &format!("{shape:?} dim {dim}"));
                }
            }
        }
        eprintln!(
            "distance kernels: checked {tested:?}, skipped (CPU lacks the feature) {skipped:?}"
        );
        assert!(tested.contains(&"scalar"));
    }

    #[test]
    fn every_length_is_exact_on_small_integers() {
        let mut rng = Rng(0x5EED_0002);
        for kernel in supported_kernels() {
            for n in 0..=300 {
                let a: Vec<f32> = (0..n).map(|_| (rng.unit() * 8.0).round()).collect();
                let b: Vec<f32> = (0..n).map(|_| (rng.unit() * 8.0).round()).collect();
                let name = kernel.name();
                assert_eq!(kernel.dot(&a, &b) as f64, dot64(&a, &b), "{name} dot n={n}");
                assert_eq!(
                    kernel.l2_sq(&a, &b) as f64,
                    l2_sq64(&a, &b),
                    "{name} l2_sq n={n}"
                );
            }
        }
    }

    #[test]
    fn results_do_not_depend_on_alignment() {
        let mut rng = Rng(0x5EED_0003);
        let a: Vec<f32> = (0..2100).map(|_| rng.unit()).collect();
        let b: Vec<f32> = (0..2100).map(|_| rng.unit()).collect();
        for kernel in supported_kernels() {
            for offset in 0..16 {
                let x = &a[offset..offset + 2000];
                let y = &b[16 - offset..16 - offset + 2000];
                check(kernel, x, y, &format!("offset {offset}"));
            }
        }
    }

    #[test]
    fn cosine_edge_cases() {
        for kernel in supported_kernels() {
            let name = kernel.name();
            let v: Vec<f32> = (1..=70).map(|i| i as f32 * 0.25 - 9.0).collect();
            let zero = vec![0.0f32; 70];
            assert_eq!(kernel.cosine(&zero, &v), 1.0, "{name}: zero vs v");
            assert_eq!(kernel.cosine(&v, &zero), 1.0, "{name}: v vs zero");
            assert_eq!(kernel.cosine(&zero, &zero), 1.0, "{name}: zero vs zero");
            assert_eq!(kernel.cosine(&[], &[]), 1.0, "{name}: empty");
            assert!(kernel.cosine(&v, &v).abs() < 1e-6, "{name}: same");
            let scaled: Vec<f32> = v.iter().map(|x| x * 3.5).collect();
            assert!(kernel.cosine(&v, &scaled).abs() < 1e-6, "{name}: scaled");
            let negated: Vec<f32> = v.iter().map(|x| -x).collect();
            assert!(
                (kernel.cosine(&v, &negated) - 2.0).abs() < 1e-6,
                "{name}: opposite"
            );
            assert!(
                (kernel.cosine(&[1.0, 0.0], &[0.0, 1.0]) - 1.0).abs() < 1e-6,
                "{name}: orthogonal"
            );
        }
    }

    #[test]
    fn non_finite_inputs_propagate_from_body_and_tail() {
        for kernel in supported_kernels() {
            let name = kernel.name();
            for dim in [3usize, 16, 70, 768, 2000] {
                for at in [0, dim / 2, dim - 1] {
                    let mut a = vec![1.0f32; dim];
                    let b = vec![0.5f32; dim];
                    a[at] = f32::NAN;
                    assert!(
                        kernel.l2_sq(&a, &b).is_nan(),
                        "{name} l2 nan dim {dim} at {at}"
                    );
                    assert!(
                        kernel.dot(&a, &b).is_nan(),
                        "{name} dot nan dim {dim} at {at}"
                    );
                    assert!(
                        kernel.cosine(&a, &b).is_nan(),
                        "{name} cos nan dim {dim} at {at}"
                    );
                    a[at] = f32::INFINITY;
                    assert_eq!(
                        kernel.l2_sq(&a, &b),
                        f32::INFINITY,
                        "{name} l2 inf dim {dim}"
                    );
                    assert_eq!(
                        kernel.dot(&a, &b),
                        f32::INFINITY,
                        "{name} dot inf dim {dim}"
                    );
                }
            }
        }
    }

    #[test]
    fn dispatch_picks_the_widest_preferred_path() {
        #[cfg(target_arch = "x86_64")]
        let want =
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq") {
                "avx512f"
            } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                "avx2+fma"
            } else {
                "scalar"
            };
        #[cfg(target_arch = "aarch64")]
        let want = "neon";
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let want = "scalar";
        assert_eq!(Kernel::best().name(), want);
    }

    #[test]
    fn every_supported_path_passes_the_dispatch_self_check() {
        for kernel in supported_kernels() {
            assert!(kernel.agrees_with_scalar(), "{}", kernel.name());
        }
    }

    #[test]
    fn entry_points_are_the_best_kernel() {
        let mut rng = Rng(0x5EED_0004);
        let a: Vec<f32> = (0..768).map(|_| rng.unit()).collect();
        let b: Vec<f32> = (0..768).map(|_| rng.unit()).collect();
        let best = Kernel::best();
        assert_eq!(l2_sq(&a, &b).to_bits(), best.l2_sq(&a, &b).to_bits());
        assert_eq!(dot(&a, &b).to_bits(), best.dot(&a, &b).to_bits());
        assert_eq!(cosine(&a, &b).to_bits(), best.cosine(&a, &b).to_bits());
    }
}
