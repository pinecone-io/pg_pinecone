//! Standalone kernel microbench. `cargo run --release -- [--cpu N] [--quick]` prints Markdown.
//! A: Hamming over 1,024-bit sketches as pg_pinecone 0.9.0 scans them. B: 1-bit code x 4-bit query planes at 768-d (scalar,
//! AVX2, AVX-512; record-major and 64-code blocks). C: read roofline and a core-frequency probe.
//! Every variant is checked against the scalar reference before it is timed.

use std::arch::asm;
use std::arch::x86_64::*;
use std::hint::black_box;
use std::time::Instant;

const SKETCH_BYTES: usize = 128;
const SKETCH_WORDS: usize = 16;
const CODE_WORDS: usize = 12;
const CODE_BYTES: usize = 96;
const PLANES: usize = 4;
const PAGE_USABLE_BYTES: usize = 8168;
const LIST_HEADER_BYTES: usize = 8;
const TID_BYTES: usize = 6;

extern "C" {
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
}

fn pin(cpu: usize) {
    let mut mask = [0u64; 16];
    mask[cpu / 64] |= 1 << (cpu % 64);
    assert_eq!(
        unsafe { sched_setaffinity(0, 128, mask.as_ptr()) },
        0,
        "sched_setaffinity({cpu}) failed"
    );
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
}

struct Aligned {
    buf: Vec<u64>,
    skip: usize,
    len: usize,
}

impl Aligned {
    fn new(len: usize) -> Self {
        let buf = vec![0u64; len + 8];
        let skip = ((64 - (buf.as_ptr() as usize % 64)) % 64) / 8;
        Aligned { buf, skip, len }
    }

    fn words(&self) -> &[u64] {
        &self.buf[self.skip..self.skip + self.len]
    }

    fn words_mut(&mut self) -> &mut [u64] {
        let (skip, len) = (self.skip, self.len);
        &mut self.buf[skip..skip + len]
    }

    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.words().as_ptr() as *const u8, self.len * 8) }
    }
}

// Inline asm keeps LLVM from auto-vectorizing the "scalar" variants.
fn popcnt(x: u64) -> u64 {
    let r: u64;
    unsafe { asm!("popcnt {0}, {1}", out(reg) r, in(reg) x, options(pure, nomem, nostack)) };
    r
}

fn has_avx512() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vpopcntdq")
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

struct Cfg {
    quick: bool,
}

struct Timing {
    ns_per_item: f64,
    gb_per_s: f64,
}

fn measure<F: FnMut() -> u64>(cfg: &Cfg, items: usize, bytes_per_item: usize, mut f: F) -> Timing {
    let mut sink = f();
    let (min_ms, trials) = if cfg.quick { (5, 3) } else { (30, 7) };
    let mut reps = 1usize;
    loop {
        let t = Instant::now();
        for _ in 0..reps {
            sink ^= black_box(f());
        }
        if t.elapsed().as_millis() >= min_ms {
            break;
        }
        reps *= 2;
    }
    let samples = (0..trials)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..reps {
                sink ^= black_box(f());
            }
            t.elapsed().as_nanos() as f64 / (reps * items) as f64
        })
        .collect();
    black_box(sink);
    let ns_per_item = median(samples);
    Timing {
        ns_per_item,
        gb_per_s: bytes_per_item as f64 / ns_per_item,
    }
}

mod pg_pinecone_simd_verbatim {
    use std::arch::x86_64::*;
    use std::sync::OnceLock;

    type HammingFn = fn(&[u8], &[u8]) -> u32;

    pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
        static DISPATCH: OnceLock<HammingFn> = OnceLock::new();
        (DISPATCH.get_or_init(detect))(a, b)
    }

    pub fn scalar(a: &[u8], b: &[u8]) -> u32 {
        a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
    }

    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    unsafe fn hamming_avx512(a: &[u8], b: &[u8]) -> u32 {
        unsafe {
            let mut acc = _mm512_setzero_si512();
            let (a_blocks, a_rem) = a.as_chunks::<64>();
            let (b_blocks, b_rem) = b.as_chunks::<64>();
            for (ca, cb) in a_blocks.iter().zip(b_blocks) {
                let va = std::ptr::read_unaligned(ca.as_ptr() as *const __m512i);
                let vb = std::ptr::read_unaligned(cb.as_ptr() as *const __m512i);
                let x = _mm512_xor_si512(va, vb);
                let pc = _mm512_popcnt_epi64(x);
                acc = _mm512_add_epi64(acc, pc);
            }
            let mut total = _mm512_reduce_add_epi64(acc) as u64;
            total += scalar(a_rem, b_rem) as u64;
            total as u32
        }
    }

    pub fn avx512_selected() -> bool {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq")
    }

    fn detect() -> HammingFn {
        if avx512_selected() {
            |a, b| unsafe { hamming_avx512(a, b) }
        } else {
            scalar
        }
    }
}

use pg_pinecone_simd_verbatim as shipped;

fn hamming_shipped(sketches: &[u8], n: usize, query: &[u8], out: &mut [u32]) {
    for i in 0..n {
        out[i] =
            shipped::hamming_distance(&sketches[i * SKETCH_BYTES..(i + 1) * SKETCH_BYTES], query);
    }
}

fn hamming_naive(sketches: &[u8], n: usize, query: &[u8], out: &mut [u32]) {
    for i in 0..n {
        out[i] = shipped::scalar(&sketches[i * SKETCH_BYTES..(i + 1) * SKETCH_BYTES], query);
    }
}

fn hamming_u64(sketches: &[u64], n: usize, query: &[u64], out: &mut [u32]) {
    for i in 0..n {
        let mut s = 0;
        for w in 0..SKETCH_WORDS {
            s += popcnt(sketches[i * SKETCH_WORDS + w] ^ query[w]);
        }
        out[i] = s as u32;
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vpopcntdq")]
unsafe fn hamming_avx512_inline(sketches: &[u8], n: usize, query: &[u8], out: &mut [u32]) {
    unsafe {
        let q0 = _mm512_loadu_si512(query.as_ptr() as *const _);
        let q1 = _mm512_loadu_si512(query.as_ptr().add(64) as *const _);
        for i in 0..n {
            let p = sketches.as_ptr().add(i * SKETCH_BYTES);
            let a = _mm512_popcnt_epi64(_mm512_xor_si512(_mm512_loadu_si512(p as *const _), q0));
            let b = _mm512_popcnt_epi64(_mm512_xor_si512(
                _mm512_loadu_si512(p.add(64) as *const _),
                q1,
            ));
            out[i] = _mm512_reduce_add_epi64(_mm512_add_epi64(a, b)) as u32;
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vpopcntdq")]
unsafe fn hamming_blocked_avx512(blocks: &[u64], nblocks: usize, query: &[u64], out: &mut [u32]) {
    unsafe {
        for b in 0..nblocks {
            let base = blocks.as_ptr().add(b * SKETCH_WORDS * 64) as *const i64;
            let mut acc = [_mm512_setzero_si512(); 8];
            for w in 0..SKETCH_WORDS {
                let q = _mm512_set1_epi64(query[w] as i64);
                for t in 0..8 {
                    let v = _mm512_loadu_si512(base.add(w * 64 + t * 8) as *const _);
                    acc[t] = _mm512_add_epi64(acc[t], _mm512_popcnt_epi64(_mm512_xor_si512(v, q)));
                }
            }
            for t in 0..8 {
                _mm256_storeu_si256(
                    out.as_mut_ptr().add(b * 64 + t * 8) as *mut __m256i,
                    _mm512_cvtepi64_epi32(acc[t]),
                );
            }
        }
    }
}

fn nibble_lut() -> __m256i {
    unsafe {
        _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        )
    }
}

#[inline]
unsafe fn popcnt_bytes_avx2(v: __m256i, lut: __m256i, low_nibble: __m256i) -> __m256i {
    unsafe {
        let lo = _mm256_and_si256(v, low_nibble);
        let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_nibble);
        _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi))
    }
}

#[target_feature(enable = "avx2")]
unsafe fn hamming_blocked_avx2(blocks: &[u64], nblocks: usize, query: &[u64], out: &mut [u32]) {
    unsafe {
        let (lut, low, zero) = (nibble_lut(), _mm256_set1_epi8(0x0f), _mm256_setzero_si256());
        for b in 0..nblocks {
            let base = blocks.as_ptr().add(b * SKETCH_WORDS * 64);
            for sub in 0..4 {
                let mut acc = [zero; 4];
                for w in 0..SKETCH_WORDS {
                    let q = _mm256_set1_epi64x(query[w] as i64);
                    let col = base.add(w * 64 + sub * 16) as *const __m256i;
                    for t in 0..4 {
                        let v = _mm256_xor_si256(_mm256_loadu_si256(col.add(t)), q);
                        acc[t] = _mm256_add_epi8(acc[t], popcnt_bytes_avx2(v, lut, low));
                    }
                }
                let mut sums = [0u64; 16];
                for t in 0..4 {
                    _mm256_storeu_si256(
                        sums.as_mut_ptr().add(t * 4) as *mut __m256i,
                        _mm256_sad_epu8(acc[t], zero),
                    );
                }
                for c in 0..16 {
                    out[b * 64 + sub * 16 + c] = sums[c] as u32;
                }
            }
        }
    }
}

struct ListLayout {
    codes: usize,
    norms: usize,
    tids: usize,
}

fn list_layout(n: usize) -> ListLayout {
    let codes = LIST_HEADER_BYTES;
    let norms = codes + n * SKETCH_BYTES;
    ListLayout {
        codes,
        norms,
        tids: norms + n * 4,
    }
}

fn read_u32(buf: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(buf[at..at + 4].try_into().expect("page read out of bounds"))
}

fn read_u16(buf: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(buf[at..at + 2].try_into().expect("page read out of bounds"))
}

fn read_f32(buf: &[u8], at: usize) -> f32 {
    f32::from_le_bytes(buf[at..at + 4].try_into().expect("page read out of bounds"))
}

#[derive(Clone, Copy)]
struct Candidate {
    tid: [u16; 3],
    est_dist: f32,
}

fn scan_list_like(
    pages: &[Vec<u8>],
    query: &[u8],
    cosines: &[f32],
    query_norm: f32,
    out: &mut Vec<Candidate>,
) {
    for buf in pages {
        let n = read_u32(buf, 4) as usize;
        let l = list_layout(n);
        for i in 0..n {
            let code = &buf[l.codes + i * SKETCH_BYTES..l.codes + (i + 1) * SKETCH_BYTES];
            let est_cos = cosines[shipped::hamming_distance(code, query) as usize];
            let norm_v = read_f32(buf, l.norms + i * 4);
            let sq =
                query_norm * query_norm + norm_v * norm_v - 2.0 * query_norm * norm_v * est_cos;
            let tid_block = read_u32(buf, l.tids + i * TID_BYTES);
            let tid_offset = read_u16(buf, l.tids + i * TID_BYTES + 4);
            out.push(Candidate {
                tid: [(tid_block >> 16) as u16, tid_block as u16, tid_offset],
                est_dist: sq.max(0.0).sqrt(),
            });
        }
    }
}

struct Row {
    variant: String,
    layout: &'static str,
    n: usize,
    t: Timing,
}

fn part_a(cfg: &Cfg, rows: &mut Vec<Row>) {
    let n = if cfg.quick { 10_048 } else { 100_032 };
    let nblocks = n / 64;
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut sk = Aligned::new(n * SKETCH_WORDS);
    sk.words_mut().iter_mut().for_each(|x| *x = rng.next());
    let mut q = Aligned::new(SKETCH_WORDS);
    q.words_mut().iter_mut().for_each(|x| *x = rng.next());
    let mut blocks = Aligned::new(n * SKETCH_WORDS);
    for b in 0..nblocks {
        for w in 0..SKETCH_WORDS {
            for c in 0..64 {
                let v = sk.words()[(b * 64 + c) * SKETCH_WORDS + w];
                blocks.words_mut()[(b * SKETCH_WORDS + w) * 64 + c] = v;
            }
        }
    }
    let (sk8, q8) = (sk.bytes(), q.bytes());
    let mut reference = vec![0u32; n];
    hamming_naive(sk8, n, q8, &mut reference);
    let mut out = vec![0u32; n];
    let push = |rows: &mut Vec<Row>, variant: &str, layout: &'static str, t: Timing| {
        rows.push(Row {
            variant: variant.to_string(),
            layout,
            n,
            t,
        })
    };

    let t = measure(cfg, n, SKETCH_BYTES, || {
        hamming_naive(sk8, n, q8, &mut out);
        out[n / 2] as u64
    });
    push(rows, "naive byte loop (`scalar()`)", "record", t);
    hamming_u64(sk.words(), n, q.words(), &mut out);
    assert_eq!(out, reference);
    let t = measure(cfg, n, SKETCH_BYTES, || {
        hamming_u64(sk.words(), n, q.words(), &mut out);
        out[n / 2] as u64
    });
    push(rows, "u64 popcnt", "record", t);
    hamming_shipped(sk8, n, q8, &mut out);
    assert_eq!(out, reference);
    let t = measure(cfg, n, SKETCH_BYTES, || {
        hamming_shipped(sk8, n, q8, &mut out);
        out[n / 2] as u64
    });
    let name = format!(
        "shipped dispatch, fn pointer per record ({})",
        if shipped::avx512_selected() {
            "AVX-512"
        } else {
            "scalar"
        }
    );
    push(rows, &name, "record", t);
    if has_avx512() {
        unsafe { hamming_avx512_inline(sk8, n, q8, &mut out) };
        assert_eq!(out, reference);
        let t = measure(cfg, n, SKETCH_BYTES, || {
            unsafe { hamming_avx512_inline(sk8, n, q8, &mut out) };
            out[n / 2] as u64
        });
        push(
            rows,
            "AVX-512 inlined, 2 zmm + reduce per record",
            "record",
            t,
        );
        unsafe { hamming_blocked_avx512(blocks.words(), nblocks, q.words(), &mut out) };
        assert_eq!(out, reference);
        let t = measure(cfg, n, SKETCH_BYTES, || {
            unsafe { hamming_blocked_avx512(blocks.words(), nblocks, q.words(), &mut out) };
            out[n / 2] as u64
        });
        push(rows, "AVX-512 blocked, one pass", "block64", t);
    }
    unsafe { hamming_blocked_avx2(blocks.words(), nblocks, q.words(), &mut out) };
    assert_eq!(out, reference);
    let t = measure(cfg, n, SKETCH_BYTES, || {
        unsafe { hamming_blocked_avx2(blocks.words(), nblocks, q.words(), &mut out) };
        out[n / 2] as u64
    });
    push(rows, "AVX2 blocked, pshufb popcount", "block64", t);

    let per_page = (PAGE_USABLE_BYTES - LIST_HEADER_BYTES) / (SKETCH_BYTES + 4 + TID_BYTES);
    let mut pages: Vec<Vec<u8>> = Vec::new();
    for first in (0..n).step_by(per_page) {
        let m = per_page.min(n - first);
        let l = list_layout(m);
        let mut buf = vec![0u8; PAGE_USABLE_BYTES];
        buf[0..4].copy_from_slice(&u32::MAX.to_le_bytes());
        buf[4..8].copy_from_slice(&(m as u32).to_le_bytes());
        for i in 0..m {
            buf[l.codes + i * SKETCH_BYTES..l.codes + (i + 1) * SKETCH_BYTES]
                .copy_from_slice(&sk8[(first + i) * SKETCH_BYTES..(first + i + 1) * SKETCH_BYTES]);
            buf[l.norms + i * 4..l.norms + i * 4 + 4]
                .copy_from_slice(&(1.0f32 + (i % 7) as f32 * 0.1).to_le_bytes());
            buf[l.tids + i * TID_BYTES..l.tids + i * TID_BYTES + 4]
                .copy_from_slice(&((first + i) as u32).to_le_bytes());
            buf[l.tids + i * TID_BYTES + 4..l.tids + i * TID_BYTES + 6]
                .copy_from_slice(&1u16.to_le_bytes());
        }
        pages.push(buf);
    }
    let cosines: Vec<f32> = (0..=1024)
        .map(|h| (std::f64::consts::PI * h as f64 / 1024.0).cos() as f32)
        .collect();
    let t = measure(cfg, n, SKETCH_BYTES + 10, || {
        let mut out = Vec::new();
        scan_list_like(&pages, q8, &cosines, 1.0, &mut out);
        let c: Candidate = out[n / 3];
        out.len() as u64 ^ c.est_dist.to_bits() as u64 ^ (c.tid[0] ^ c.tid[1] ^ c.tid[2]) as u64
    });
    push(
        rows,
        &format!("`scan_list` per-record loop as shipped ({per_page} records per page)"),
        "pages",
        t,
    );
}

type Planes = [[u64; CODE_WORDS]; PLANES];

fn codes_scalar(codes: &[u64], n: usize, q: &Planes, out: &mut [u32]) {
    for i in 0..n {
        let mut acc = 0;
        for j in 0..PLANES {
            let mut s = 0;
            for w in 0..CODE_WORDS {
                s += popcnt(codes[i * CODE_WORDS + w] & q[j][w]);
            }
            acc += s << j;
        }
        out[i] = acc as u32;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn codes_avx2(codes: &[u64], n: usize, q: &Planes, out: &mut [u32]) {
    unsafe {
        let (lut, low, zero) = (nibble_lut(), _mm256_set1_epi8(0x0f), _mm256_setzero_si256());
        let mut qv = [[zero; 3]; PLANES];
        for j in 0..PLANES {
            for t in 0..3 {
                qv[j][t] = _mm256_loadu_si256((q[j].as_ptr() as *const __m256i).add(t));
            }
        }
        let base = codes.as_ptr() as *const u8;
        for i in 0..n {
            let p = base.add(i * CODE_BYTES) as *const __m256i;
            let (c0, c1, c2) = (
                _mm256_loadu_si256(p),
                _mm256_loadu_si256(p.add(1)),
                _mm256_loadu_si256(p.add(2)),
            );
            let mut total = zero;
            for j in 0..PLANES {
                let bytes = _mm256_add_epi8(
                    _mm256_add_epi8(
                        popcnt_bytes_avx2(_mm256_and_si256(c0, qv[j][0]), lut, low),
                        popcnt_bytes_avx2(_mm256_and_si256(c1, qv[j][1]), lut, low),
                    ),
                    popcnt_bytes_avx2(_mm256_and_si256(c2, qv[j][2]), lut, low),
                );
                let sums = _mm256_sad_epu8(bytes, zero);
                total =
                    _mm256_add_epi64(total, _mm256_sll_epi64(sums, _mm_cvtsi32_si128(j as i32)));
            }
            let two = _mm_add_epi64(
                _mm256_castsi256_si128(total),
                _mm256_extracti128_si256(total, 1),
            );
            out[i] = _mm_cvtsi128_si64(_mm_add_epi64(two, _mm_unpackhi_epi64(two, two))) as u32;
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vpopcntdq")]
unsafe fn codes_avx512(codes: &[u64], n: usize, q: &Planes, out: &mut [u32]) {
    unsafe {
        let mut q_lo = [_mm512_setzero_si512(); PLANES];
        let mut q_hi = [_mm512_setzero_si512(); PLANES];
        for j in 0..PLANES {
            q_lo[j] = _mm512_loadu_si512(q[j].as_ptr() as *const _);
            q_hi[j] = _mm512_maskz_loadu_epi64(0x0F, q[j].as_ptr().add(8) as *const i64);
        }
        let base = codes.as_ptr() as *const u8;
        for i in 0..n {
            let p = base.add(i * CODE_BYTES);
            let c_lo = _mm512_loadu_si512(p as *const _);
            let c_hi = _mm512_maskz_loadu_epi64(0x0F, p.add(64) as *const i64);
            let mut total = _mm512_setzero_si512();
            for j in 0..PLANES {
                let a = _mm512_popcnt_epi64(_mm512_and_si512(c_lo, q_lo[j]));
                let b = _mm512_popcnt_epi64(_mm512_and_si512(c_hi, q_hi[j]));
                total = _mm512_add_epi64(
                    total,
                    _mm512_sll_epi64(_mm512_add_epi64(a, b), _mm_cvtsi32_si128(j as i32)),
                );
            }
            out[i] = _mm512_reduce_add_epi64(total) as u32;
        }
    }
}

fn blocks_scalar(blocks: &[u64], nblocks: usize, q: &Planes, out: &mut [u32]) {
    for b in 0..nblocks {
        let mut acc = [0u32; 64];
        for w in 0..CODE_WORDS {
            let col = &blocks[(b * CODE_WORDS + w) * 64..(b * CODE_WORDS + w) * 64 + 64];
            for j in 0..PLANES {
                for c in 0..64 {
                    acc[c] += (popcnt(col[c] & q[j][w]) as u32) << j;
                }
            }
        }
        out[b * 64..b * 64 + 64].copy_from_slice(&acc);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn blocks_avx2(blocks: &[u64], nblocks: usize, q: &Planes, out: &mut [u32]) {
    unsafe {
        let (lut, low, zero) = (nibble_lut(), _mm256_set1_epi8(0x0f), _mm256_setzero_si256());
        for b in 0..nblocks {
            let base = blocks.as_ptr().add(b * CODE_WORDS * 64);
            for sub in 0..4 {
                let mut total = [zero; 4];
                for j in 0..PLANES {
                    let mut acc = [zero; 4];
                    for w in 0..CODE_WORDS {
                        let qv = _mm256_set1_epi64x(q[j][w] as i64);
                        let col = base.add(w * 64 + sub * 16) as *const __m256i;
                        for t in 0..4 {
                            let v = _mm256_and_si256(_mm256_loadu_si256(col.add(t)), qv);
                            acc[t] = _mm256_add_epi8(acc[t], popcnt_bytes_avx2(v, lut, low));
                        }
                    }
                    for t in 0..4 {
                        total[t] = _mm256_add_epi64(
                            total[t],
                            _mm256_sll_epi64(
                                _mm256_sad_epu8(acc[t], zero),
                                _mm_cvtsi32_si128(j as i32),
                            ),
                        );
                    }
                }
                let mut sums = [0u64; 16];
                for t in 0..4 {
                    _mm256_storeu_si256(sums.as_mut_ptr().add(t * 4) as *mut __m256i, total[t]);
                }
                for c in 0..16 {
                    out[b * 64 + sub * 16 + c] = sums[c] as u32;
                }
            }
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vpopcntdq")]
unsafe fn blocks_avx512(blocks: &[u64], nblocks: usize, q: &Planes, out: &mut [u32]) {
    unsafe {
        for b in 0..nblocks {
            let base = blocks.as_ptr().add(b * CODE_WORDS * 64) as *const i64;
            let mut total = [_mm512_setzero_si512(); 8];
            for j in 0..PLANES {
                let mut acc = [_mm512_setzero_si512(); 8];
                for w in 0..CODE_WORDS {
                    let qv = _mm512_set1_epi64(q[j][w] as i64);
                    for t in 0..8 {
                        let v = _mm512_loadu_si512(base.add(w * 64 + t * 8) as *const _);
                        acc[t] =
                            _mm512_add_epi64(acc[t], _mm512_popcnt_epi64(_mm512_and_si512(v, qv)));
                    }
                }
                for t in 0..8 {
                    total[t] = _mm512_add_epi64(
                        total[t],
                        _mm512_sll_epi64(acc[t], _mm_cvtsi32_si128(j as i32)),
                    );
                }
            }
            for t in 0..8 {
                _mm256_storeu_si256(
                    out.as_mut_ptr().add(b * 64 + t * 8) as *mut __m256i,
                    _mm512_cvtepi64_epi32(total[t]),
                );
            }
        }
    }
}

fn part_b(cfg: &Cfg, n: usize, rows: &mut Vec<Row>) {
    let nblocks = n / 64;
    let mut rng = Rng(0xD1B54A32D192ED03 ^ n as u64);
    let mut records = Aligned::new(n * CODE_WORDS);
    records.words_mut().iter_mut().for_each(|x| *x = rng.next());
    let mut q: Planes = [[0; CODE_WORDS]; PLANES];
    q.iter_mut()
        .for_each(|plane| plane.iter_mut().for_each(|x| *x = rng.next()));
    let mut blocks = Aligned::new(n * CODE_WORDS);
    for b in 0..nblocks {
        for w in 0..CODE_WORDS {
            for c in 0..64 {
                let v = records.words()[(b * 64 + c) * CODE_WORDS + w];
                blocks.words_mut()[(b * CODE_WORDS + w) * 64 + c] = v;
            }
        }
    }
    let mut reference = vec![0u32; n];
    codes_scalar(records.words(), n, &q, &mut reference);
    let mut out = vec![0u32; n];
    let push = |rows: &mut Vec<Row>, variant: &str, layout: &'static str, t: Timing| {
        rows.push(Row {
            variant: variant.to_string(),
            layout,
            n,
            t,
        })
    };

    let t = measure(cfg, n, CODE_BYTES, || {
        codes_scalar(records.words(), n, &q, &mut out);
        out[n / 2] as u64
    });
    push(rows, "scalar popcnt", "record", t);
    unsafe { codes_avx2(records.words(), n, &q, &mut out) };
    assert_eq!(out, reference);
    let t = measure(cfg, n, CODE_BYTES, || {
        unsafe { codes_avx2(records.words(), n, &q, &mut out) };
        out[n / 2] as u64
    });
    push(rows, "AVX2 pshufb", "record", t);
    if has_avx512() {
        unsafe { codes_avx512(records.words(), n, &q, &mut out) };
        assert_eq!(out, reference);
        let t = measure(cfg, n, CODE_BYTES, || {
            unsafe { codes_avx512(records.words(), n, &q, &mut out) };
            out[n / 2] as u64
        });
        push(rows, "AVX-512 VPOPCNTDQ", "record", t);
    }
    blocks_scalar(blocks.words(), nblocks, &q, &mut out);
    assert_eq!(out, reference);
    let t = measure(cfg, n, CODE_BYTES, || {
        blocks_scalar(blocks.words(), nblocks, &q, &mut out);
        out[n / 2] as u64
    });
    push(rows, "scalar popcnt", "block64", t);
    unsafe { blocks_avx2(blocks.words(), nblocks, &q, &mut out) };
    assert_eq!(out, reference);
    let t = measure(cfg, n, CODE_BYTES, || {
        unsafe { blocks_avx2(blocks.words(), nblocks, &q, &mut out) };
        out[n / 2] as u64
    });
    push(rows, "AVX2 pshufb", "block64", t);
    if has_avx512() {
        unsafe { blocks_avx512(blocks.words(), nblocks, &q, &mut out) };
        assert_eq!(out, reference);
        let t = measure(cfg, n, CODE_BYTES, || {
            unsafe { blocks_avx512(blocks.words(), nblocks, &q, &mut out) };
            out[n / 2] as u64
        });
        push(rows, "AVX-512 VPOPCNTDQ", "block64", t);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn read_sum_avx512(words: &[u64]) -> u64 {
    unsafe {
        let p = words.as_ptr() as *const i64;
        let mut acc = [_mm512_setzero_si512(); 8];
        for i in (0..words.len() / 64 * 64).step_by(64) {
            for t in 0..8 {
                acc[t] = _mm512_add_epi64(acc[t], _mm512_loadu_si512(p.add(i + t * 8) as *const _));
            }
        }
        let mut s = acc[0];
        for t in 1..8 {
            s = _mm512_add_epi64(s, acc[t]);
        }
        _mm512_reduce_add_epi64(s) as u64
    }
}

#[target_feature(enable = "avx2")]
unsafe fn read_sum_avx2(words: &[u64]) -> u64 {
    unsafe {
        let p = words.as_ptr() as *const __m256i;
        let mut acc = [_mm256_setzero_si256(); 8];
        for i in (0..words.len() / 32 * 32).step_by(32) {
            for t in 0..8 {
                acc[t] = _mm256_add_epi64(acc[t], _mm256_loadu_si256(p.add(i / 4 + t)));
            }
        }
        let mut s = acc[0];
        for t in 1..8 {
            s = _mm256_add_epi64(s, acc[t]);
        }
        let two = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
        _mm_cvtsi128_si64(_mm_add_epi64(two, _mm_unpackhi_epi64(two, two))) as u64
    }
}

fn read_roofline(cfg: &Cfg, bytes: usize) -> (f64, f64) {
    let words = bytes / 8 / 64 * 64;
    let mut buf = Aligned::new(words);
    buf.words_mut()
        .iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = i as u64);
    let avx2 = measure(cfg, words, 8, || unsafe { read_sum_avx2(buf.words()) }).gb_per_s;
    let avx512 = if has_avx512() {
        measure(cfg, words, 8, || unsafe { read_sum_avx512(buf.words()) }).gb_per_s
    } else {
        0.0
    };
    (avx2, avx512)
}

// Eight dependent adds per iteration at one cycle each (Intel and AMD cores), so GHz = 8 * iterations / ns.
fn add_chain_ghz(iters: u64) -> f64 {
    let t = Instant::now();
    let mut x = 0u64;
    let mut n = iters;
    unsafe {
        asm!(
            "2:",
            "add {x}, 1", "add {x}, 1", "add {x}, 1", "add {x}, 1", "add {x}, 1", "add {x}, 1", "add {x}, 1", "add {x}, 1",
            "dec {n}",
            "jnz 2b",
            x = inout(reg) x,
            n = inout(reg) n,
            options(nomem, nostack)
        );
    }
    black_box((x, n));
    8.0 * iters as f64 / t.elapsed().as_nanos() as f64
}

fn frequency_probe(cfg: &Cfg) -> Vec<(String, f64)> {
    let n = 4096;
    let mut rng = Rng(7);
    let mut records = Aligned::new(n * CODE_WORDS);
    records.words_mut().iter_mut().for_each(|x| *x = rng.next());
    let mut q: Planes = [[0; CODE_WORDS]; PLANES];
    q.iter_mut()
        .for_each(|plane| plane.iter_mut().for_each(|x| *x = rng.next()));
    let mut out = vec![0u32; n];
    let rounds = if cfg.quick { 15 } else { 60 };
    let run = |kind: &str, out: &mut [u32], codes: usize| match kind {
        "AVX2" => unsafe { codes_avx2(records.words(), codes, &q, out) },
        _ => unsafe { codes_avx512(records.words(), codes, &q, out) },
    };
    add_chain_ghz(50_000_000);
    let mut res = vec![(
        "dependent add chain, idle core".to_string(),
        median((0..rounds).map(|_| add_chain_ghz(2_000_000)).collect()),
    )];
    for kind in ["AVX2", "AVX-512"] {
        if kind == "AVX-512" && !has_avx512() {
            continue;
        }
        let after = (0..rounds)
            .map(|_| {
                let t = Instant::now();
                while t.elapsed().as_millis() < 20 {
                    run(kind, &mut out, n);
                }
                add_chain_ghz(200_000)
            })
            .collect();
        res.push((
            format!("chain right after a 20 ms {kind} kernel burst"),
            median(after),
        ));
        let interleaved = (0..rounds)
            .map(|_| {
                let t = Instant::now();
                while t.elapsed().as_micros() < 150 {
                    run(kind, &mut out, 256);
                }
                add_chain_ghz(100_000)
            })
            .collect();
        res.push((
            format!("chain interleaved with 0.15 ms {kind} bursts"),
            median(interleaved),
        ));
    }
    res.push((
        "dependent add chain, idle core again".to_string(),
        median((0..rounds).map(|_| add_chain_ghz(2_000_000)).collect()),
    ));
    black_box(&out);
    res
}

fn cache_sizes() -> String {
    let mut parts = Vec::new();
    for i in 0..5 {
        let dir = format!("/sys/devices/system/cpu/cpu0/cache/index{i}");
        let (Ok(level), Ok(kind), Ok(size)) = (
            std::fs::read_to_string(format!("{dir}/level")),
            std::fs::read_to_string(format!("{dir}/type")),
            std::fs::read_to_string(format!("{dir}/size")),
        ) else {
            continue;
        };
        parts.push(format!(
            "L{}{} {}",
            level.trim(),
            if kind.trim() == "Instruction" {
                "i"
            } else {
                ""
            },
            size.trim()
        ));
    }
    parts.join(", ")
}

fn cpu_model() -> String {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    cpuinfo
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1))
        .unwrap_or("")
        .trim()
        .to_string()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cfg = Cfg {
        quick: args.iter().any(|a| a == "--quick"),
    };
    if let Some(i) = args.iter().position(|a| a == "--cpu") {
        pin(args[i + 1].parse().unwrap());
    }
    println!("## Kernel microbench\n");
    println!("- CPU: {}", cpu_model());
    println!("- caches (cpu0): {}", cache_sizes());
    println!(
        "- features: popcnt {} avx2 {} avx512f {} avx512bw {} avx512vpopcntdq {}",
        is_x86_feature_detected!("popcnt"),
        is_x86_feature_detected!("avx2"),
        is_x86_feature_detected!("avx512f"),
        is_x86_feature_detected!("avx512bw"),
        is_x86_feature_detected!("avx512vpopcntdq")
    );
    println!("- built with target-cpu=x86-64-v3 like the release tarball; AVX-512 paths sit behind #[target_feature] and runtime detection; {} mode", if cfg.quick { "quick" } else { "full" });

    let freq = frequency_probe(&cfg);
    let base_ghz = freq[0].1;
    println!("\n### Core frequency probe\n\n| state | GHz |\n|---|---|");
    for (state, ghz) in &freq {
        println!("| {state} | {ghz:.3} |");
    }

    let mut a = Vec::new();
    part_a(&cfg, &mut a);
    let sketch_bytes = a[0].n * SKETCH_BYTES;
    let (r2, r5) = read_roofline(&cfg, sketch_bytes);
    println!("\n### A. Hamming over 1,024-bit sketches, {} sketches ({} MB), read roofline {:.1} GB/s (AVX2) / {:.1} GB/s (AVX-512)\n", a[0].n, sketch_bytes / 1_000_000, r2, r5);
    println!("| variant | layout | ns / sketch | GB/s | cycles / sketch at {base_ghz:.2} GHz |\n|---|---|---|---|---|");
    for r in &a {
        println!(
            "| {} | {} | {:.2} | {:.1} | {:.1} |",
            r.variant,
            r.layout,
            r.t.ns_per_item,
            r.t.gb_per_s,
            r.t.ns_per_item * base_ghz
        );
    }

    let sizes: &[usize] = if cfg.quick {
        &[1_024, 20_032]
    } else {
        &[1_024, 100_032, 1_000_000]
    };
    let mut per_size = Vec::new();
    for &n in sizes {
        let mut b = Vec::new();
        part_b(&cfg, n, &mut b);
        per_size.push((n, read_roofline(&cfg, n * CODE_BYTES), b));
    }
    println!("\n### B. 1-bit code x 4-bit query planes, 768-d, 96 B per code\n");
    print!("| variant | layout |");
    for (n, _, _) in &per_size {
        print!(
            " ns / code ({n} codes, {} KB) | GB/s |",
            n * CODE_BYTES / 1000
        );
    }
    println!(" cycles / code at {base_ghz:.2} GHz (smallest set) | bound at the middle size |");
    println!("|---|---|{}---|---|", "---|---|".repeat(per_size.len()));
    for i in 0..per_size[0].2.len() {
        let first = &per_size[0].2[i];
        print!("| {} | {} |", first.variant, first.layout);
        for (_, _, rows) in &per_size {
            print!(
                " {:.2} | {:.1} |",
                rows[i].t.ns_per_item, rows[i].t.gb_per_s
            );
        }
        let mid = per_size.len() / 2;
        let roofline = per_size[mid].1 .0.max(per_size[mid].1 .1);
        let gbps = per_size[mid].2[i].t.gb_per_s;
        println!(
            " {:.1} | {} |",
            first.t.ns_per_item * base_ghz,
            if gbps >= 0.8 * roofline {
                "memory"
            } else {
                "compute"
            }
        );
    }
    println!("\n### C. Single-thread read roofline (sum of u64 loads)\n\n| working set | AVX2 GB/s | AVX-512 GB/s |\n|---|---|---|");
    for (n, (r2, r5), _) in &per_size {
        println!(
            "| {} codes, {} KB | {r2:.1} | {r5:.1} |",
            n,
            n * CODE_BYTES / 1000
        );
    }
    println!(
        "| sketches, {} KB | {:.1} | {:.1} |",
        sketch_bytes / 1000,
        r2,
        r5
    );
}
