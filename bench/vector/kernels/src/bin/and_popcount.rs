//! RaBitQ kernels of pg_pinecone (`pg_pinecone/src/local/{kernel,rotation,rabitq}.rs`, included verbatim):
//! ns per code of `and_popcount_4` on every path this CPU has, record-major, and of `acc_block` on
//! the block layout beside it, then the per-code estimate, the per-list query prep, the rotation
//! and the encoder.
//! `cargo run --release --bin and_popcount -- [--dim 768] [--cpu N] [--quick]` prints Markdown.

// The path attribute on the module keeps the nested ones relative to src/bin: an inline module
// would add its own name as a directory that does not exist.
#[allow(dead_code)]
#[path = "."]
mod local {
    #[path = "../../../../../pg_pinecone/src/local/distance.rs"]
    pub mod distance;
    #[path = "../../../../../pg_pinecone/src/local/kernel.rs"]
    pub mod kernel;
    #[path = "../../../../../pg_pinecone/src/local/metric.rs"]
    pub mod metric;
    #[path = "../../../../../pg_pinecone/src/local/rabitq.rs"]
    pub mod rabitq;
    #[path = "../../../../../pg_pinecone/src/local/rng.rs"]
    pub mod rng;
    #[path = "../../../../../pg_pinecone/src/local/rotation.rs"]
    pub mod rotation;
}

use local::kernel::{self, Isa, Kernel};
use local::metric::Metric;
use local::rabitq::{self, Code, EncodeScratch, QueryPrep};
use local::rotation::{padded_dim, Rotation};
use std::hint::black_box;
use std::time::Instant;

const SEED: u64 = 42;
const ISAS: [(Isa, &str); 4] = [
    (Isa::Scalar, "scalar `count_ones`"),
    (Isa::Avx2, "AVX2 byte lookup"),
    (Isa::Avx512, "AVX-512 VPOPCNTDQ"),
    (Isa::Neon, "NEON `vcntq_u8`"),
];
// Same order as `ISAS`.
const BLOCK_NAMES: [&str; 4] = [
    "scalar table lookups",
    "AVX2 `vpshufb`",
    "AVX-512 `vpshufb`",
    "NEON `tbl`",
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as u32) as f32 / (1u32 << 23) as f32 - 1.0
    }

    fn vector(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.unit() * scale).collect()
    }
}

struct Cfg {
    quick: bool,
    dim: usize,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Median over trials of at least 30 ms (5 ms with `--quick`) of ns per item.
fn measure(cfg: &Cfg, items: usize, mut f: impl FnMut() -> u64) -> f64 {
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
    median(samples)
}

#[cfg(target_os = "linux")]
fn pin(cpu: usize) {
    extern "C" {
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
    }
    let mut mask = [0u64; 16];
    mask[cpu / 64] |= 1 << (cpu % 64);
    assert_eq!(
        unsafe { sched_setaffinity(0, 128, mask.as_ptr()) },
        0,
        "sched_setaffinity({cpu}) failed"
    );
}

#[cfg(not(target_os = "linux"))]
fn pin(cpu: usize) {
    eprintln!("--cpu {cpu}: pinning is Linux-only, ignored");
}

fn cpu_model() -> String {
    let linux = std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|s| {
        s.lines()
            .find(|l| l.starts_with("model name"))
            .and_then(|l| l.split(':').nth(1))
            .map(|m| m.trim().to_string())
    });
    let mac = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    linux.or(mac).unwrap_or_else(|| "unknown".into())
}

fn codes_for(rng: &mut Rng, n: usize, words: usize) -> Vec<u64> {
    (0..n * words).map(|_| rng.next()).collect()
}

fn paths() -> Vec<(Isa, &'static str)> {
    ISAS.into_iter()
        .filter(|(isa, _)| kernel::supported(*isa))
        .collect()
}

fn planes_of(prep: &QueryPrep) -> [&[u64]; 4] {
    [
        &prep.planes[0],
        &prep.planes[1],
        &prep.planes[2],
        &prep.planes[3],
    ]
}

fn kernel_rows(cfg: &Cfg, rng: &mut Rng, prep: &QueryPrep) {
    let words = cfg.dim.div_ceil(64);
    let planes = planes_of(prep);
    let sizes = [1_024usize, 100_032];
    let sets: Vec<Vec<u64>> = sizes.iter().map(|n| codes_for(rng, *n, words)).collect();
    println!(
        "\n### 1. `and_popcount_4`, record-major, D = {} ({} B per code): ns per code\n",
        words * 64,
        words * 8
    );
    println!(
        "| path | {} codes ({} KB) | GB/s | {} codes ({:.1} MB) | `popcount` alone, {} codes |",
        sizes[0],
        sizes[0] * words * 8 / 1024,
        sizes[1],
        (sizes[1] * words * 8) as f64 / 1e6,
        sizes[0]
    );
    println!("|---|---|---|---|---|");
    for (isa, name) in paths() {
        let kernel = Kernel::new(isa).expect("a supported path");
        let mut cells = Vec::new();
        for codes in &sets {
            for code in codes.chunks_exact(words).take(2_000) {
                let want = kernel::and_popcount_4_with(Isa::Scalar, code, &planes);
                assert_eq!(
                    kernel::and_popcount_4_with(isa, code, &planes),
                    want,
                    "{isa:?}"
                );
                assert_eq!(
                    kernel::popcount_with(isa, code),
                    kernel::popcount_with(Isa::Scalar, code),
                    "{isa:?}"
                );
            }
            let n = codes.len() / words;
            cells.push(measure(cfg, n, || {
                let mut sum = 0u64;
                for code in codes.chunks_exact(words) {
                    let c = kernel.and_popcount_4(code, &planes);
                    sum += u64::from(c[0] + 2 * c[1] + 4 * c[2] + 8 * c[3]);
                }
                sum
            }));
        }
        let pb = measure(cfg, sizes[0], || {
            sets[0]
                .chunks_exact(words)
                .map(|code| u64::from(kernel.popcount(code)))
                .sum()
        });
        println!(
            "| {name} | {:.2} | {:.1} | {:.2} | {:.2} |",
            cells[0],
            (words * 8) as f64 / cells[0],
            cells[1],
            pb
        );
    }
}

/// A page's block code area starts on a cache line.
struct Aligned {
    buf: Vec<u8>,
    skip: usize,
    len: usize,
}

impl Aligned {
    /// `len` random bytes at `shift` bytes past a 64-byte boundary.
    fn random(rng: &mut Rng, len: usize, shift: usize) -> Aligned {
        let mut buf = vec![0u8; len + 64 + shift];
        for chunk in buf.chunks_mut(8) {
            let word = rng.next().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
        let skip = (64 - buf.as_ptr() as usize % 64) % 64 + shift;
        Aligned { buf, skip, len }
    }

    fn bytes(&self) -> &[u8] {
        &self.buf[self.skip..self.skip + self.len]
    }
}

/// Nibble tables of random 4-bit levels: entry `v` of group `g` sums the levels of the coordinates
/// of `g` whose bit is set in `v`.
fn block_lut(rng: &mut Rng, d: usize) -> Vec<u8> {
    let levels: Vec<u8> = (0..d).map(|_| (rng.next() % 16) as u8).collect();
    levels
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|group| {
            (0..16usize).map(|v| {
                (0..4)
                    .map(|t| if v >> t & 1 == 1 { group[t] } else { 0 })
                    .sum()
            })
        })
        .collect()
}

fn block_rows(cfg: &Cfg, rng: &mut Rng) {
    let sizes = [1_024usize, 100_032];
    println!(
        "\n### 1b. `acc_block` (block layout) beside `and_popcount_4` + `popcount` (record-major): ns per record\n"
    );
    println!(
        "Random codes and tables of random 4-bit levels; blocks start on a cache line; `+8` starts the same blocks 8 bytes past one (a page buffer is only 8-byte aligned). Record-major is the pair of calls the estimator makes per code.\n"
    );
    for (d, lanes) in [(768usize, 64usize), (1536, 32), (256, 32), (768, 16)] {
        let (words, code_bytes) = (d / 64, d / 8);
        let planes: Vec<Vec<u64>> = (0..4).map(|_| codes_for(rng, 1, words)).collect();
        let planes = [&planes[0][..], &planes[1], &planes[2], &planes[3]];
        let lut = block_lut(rng, d);
        let block_bytes = code_bytes * lanes;
        println!(
            "D = {d}, {lanes} lanes: {code_bytes} rows, {block_bytes} B per block, {} B of tables\n",
            lut.len()
        );
        println!(
            "| path | record-major, {} rec | block, {} rec | block, +8 | speedup | record-major, {} rec | block, {} rec | speedup |",
            sizes[0], sizes[0], sizes[1], sizes[1]
        );
        println!("|---|---|---|---|---|---|---|---|");
        let sets: Vec<(Vec<u64>, Aligned, Aligned)> = sizes
            .iter()
            .map(|&n| {
                (
                    codes_for(rng, n, words),
                    Aligned::random(rng, n / lanes * block_bytes, 0),
                    Aligned::random(rng, n / lanes * block_bytes, 8),
                )
            })
            .collect();
        for (isa, _) in paths() {
            let kernel = Kernel::new(isa).expect("a supported path");
            let name = ISAS
                .iter()
                .position(|(i, _)| *i == isa)
                .map_or("", |i| BLOCK_NAMES[i]);
            let mut out = [0u32; kernel::BLOCK_MAX_LANES];
            let mut want = [0u32; kernel::BLOCK_MAX_LANES];
            let mut cells = Vec::new();
            for (codes, aligned, shifted) in &sets {
                let n = codes.len() / words;
                for block in aligned.bytes().chunks_exact(block_bytes).take(200) {
                    kernel::acc_block_with(Isa::Scalar, &lut, block, lanes, &mut want);
                    kernel.acc_block(&lut, block, lanes, &mut out);
                    assert_eq!(out[..lanes], want[..lanes], "{isa:?}, D = {d}");
                }
                let record_major = measure(cfg, n, || {
                    let mut sum = 0u64;
                    for code in codes.chunks_exact(words) {
                        let c = kernel.and_popcount_4(code, &planes);
                        sum += u64::from(
                            c[0] + 2 * c[1] + 4 * c[2] + 8 * c[3] + kernel.popcount(code),
                        );
                    }
                    sum
                });
                let mut run = |bytes: &[u8]| {
                    measure(cfg, n, || {
                        let mut sum = 0u64;
                        for block in bytes.chunks_exact(block_bytes) {
                            kernel.acc_block(&lut, block, lanes, &mut out);
                            sum += u64::from(out[0]) + u64::from(out[lanes - 1]);
                            black_box(&out);
                        }
                        sum
                    })
                };
                let block = run(aligned.bytes());
                let unaligned = if cells.is_empty() {
                    run(shifted.bytes())
                } else {
                    f64::NAN
                };
                cells.push((record_major, block, unaligned));
            }
            let shifted = cells[0].2;
            println!(
                "| {name} | {:.2} | {:.2} | {shifted:.2} | {:.1}x | {:.2} | {:.2} | {:.1}x |",
                cells[0].0,
                cells[0].1,
                cells[0].0 / cells[0].1,
                cells[1].0,
                cells[1].1,
                cells[1].0 / cells[1].1
            );
        }
        println!();
    }
}

fn estimate_rows(cfg: &Cfg, rng: &mut Rng, prep: &QueryPrep) {
    let words = cfg.dim.div_ceil(64);
    let n = 1_024;
    let codes = codes_for(rng, n, words);
    let dots: Vec<f32> = (0..n).map(|_| 0.75 + 0.1 * rng.unit().abs()).collect();
    println!("\n### 2. `estimate_bits` (kernel, `popcount` and the estimator arithmetic): ns per code, {n} codes\n");
    println!("| path | est and err | with `key` and `lo` (L2) |");
    println!("|---|---|---|");
    for (isa, name) in paths() {
        let kernel = Kernel::new(isa).expect("a supported path");
        let est = measure(cfg, n, || {
            let mut sum = 0u32;
            for (code, dot) in codes.chunks_exact(words).zip(&dots) {
                let (e, r) = rabitq::estimate_bits(kernel, prep, code, *dot);
                sum ^= e.to_bits() ^ r.to_bits();
            }
            u64::from(sum)
        });
        let full = measure(cfg, n, || {
            let mut sum = 0u32;
            for (code, dot) in codes.chunks_exact(words).zip(&dots) {
                let (e, r) = rabitq::estimate_bits(kernel, prep, code, *dot);
                let stored = Code {
                    bits: Vec::new(),
                    rnorm: 1.5,
                    dot: *dot,
                    rc: 0.25,
                };
                sum ^= rabitq::key(Metric::L2, prep, &stored, e).to_bits()
                    ^ rabitq::lo(Metric::L2, prep, &stored, e, r).to_bits();
            }
            u64::from(sum)
        });
        println!("| {name} | {est:.2} | {full:.2} |");
    }
}

fn prep_rows(cfg: &Cfg, rng: &mut Rng, rot: &Rotation) {
    let d = cfg.dim;
    let c = rng.vector(d, 1.0);
    let ptc = rot.apply(&c);
    let near: Vec<f32> = c.iter().map(|v| 0.9 * v + 0.05 * rng.unit()).collect();
    let far = rng.vector(d, 0.05);
    println!(
        "\n### 3. per-list query prep, D = {}: us per list\n",
        padded_dim(d)
    );
    println!(
        "| form | `prepare_query_into` (planes reused) | `prepare_query` (allocates the planes) |"
    );
    println!("|---|---|---|");
    let forms: [(&str, Metric, &[f32]); 3] = [
        ("L2, centred", Metric::L2, &near),
        ("IP, centred", Metric::Ip, &near),
        ("IP, uncentred", Metric::Ip, &far),
    ];
    for (name, metric, q) in forms {
        let ptq = rot.apply(q);
        let mut prep = QueryPrep::new(padded_dim(d));
        let mut list = 0u32;
        let reused = measure(cfg, 1, || {
            list = list.wrapping_add(1);
            rabitq::prepare_query_into(&mut prep, metric, q, &ptq, &c, &ptc, SEED, list);
            u64::from(prep.s_q) ^ u64::from(prep.delta.to_bits())
        });
        let fresh = measure(cfg, 1, || {
            list = list.wrapping_add(1);
            let prep = rabitq::prepare_query(metric, q, &ptq, &c, &ptc, SEED, list);
            u64::from(prep.s_q) ^ u64::from(prep.delta.to_bits())
        });
        println!("| {name} | {:.3} | {:.3} |", reused / 1e3, fresh / 1e3);
    }
}

fn rotation_rows(cfg: &Cfg, rng: &mut Rng) {
    let d = cfg.dim;
    let rot = Rotation::new(SEED, d);
    let v = rng.vector(d, 1.0);
    let c = rng.vector(d, 1.0);
    let x: Vec<f32> = c.iter().map(|a| a + 0.3 * rng.unit()).collect();
    let mut out = vec![0.0f32; padded_dim(d)];
    let mut spare = vec![0.0f32; padded_dim(d)];
    let mut scratch = EncodeScratch::default();
    let mut code = Code {
        bits: Vec::new(),
        rnorm: 0.0,
        dot: 1.0,
        rc: 0.0,
    };
    println!("\n### 4. rotation and encoder, d = {d}: us per vector\n");
    println!("| operation | us |");
    println!("|---|---|");
    let new = measure(cfg, 1, || Rotation::new(SEED, d).padded() as u64);
    let apply = measure(cfg, 1, || {
        rot.apply_into(&v, &mut out);
        u64::from(out[1].to_bits())
    });
    let with = measure(cfg, 1, || {
        rot.apply_into_with(&v, &mut out, &mut spare);
        u64::from(out[1].to_bits())
    });
    let encode = measure(cfg, 1, || {
        rabitq::encode_into(&rot, &x, &c, &mut scratch, &mut code);
        code.bits[0] ^ u64::from(code.dot.to_bits())
    });
    println!(
        "| `Rotation::new` (once per scan or build) | {:.2} |",
        new / 1e3
    );
    println!(
        "| `apply_into` (thread-local scratch) | {:.3} |",
        apply / 1e3
    );
    println!("| `apply_into_with` (caller scratch) | {:.3} |", with / 1e3);
    println!(
        "| `encode_into` (rotation, sign bits, three f32) | {:.3} |",
        encode / 1e3
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().position(|a| a == name);
    let value = |name: &str| {
        flag(name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
    };
    let cfg = Cfg {
        quick: flag("--quick").is_some(),
        dim: value("--dim").unwrap_or(768),
    };
    if let Some(cpu) = value("--cpu") {
        pin(cpu);
    }
    let mut rng = Rng(SEED);
    let rot = Rotation::new(SEED, cfg.dim);
    let c = rng.vector(cfg.dim, 1.0);
    let q: Vec<f32> = c.iter().map(|v| 0.9 * v + 0.05 * rng.unit()).collect();
    let prep = rabitq::prepare_query(Metric::L2, &q, &rot.apply(&q), &c, &rot.apply(&c), SEED, 0);
    println!("## RaBitQ kernel microbench\n");
    println!("CPU: {}; paths: {}", cpu_model(), {
        let names: Vec<&str> = paths().into_iter().map(|(_, n)| n).collect();
        names.join(", ")
    });
    println!(
        "Detected: {:?}. Medians of {} trials, input in cache as noted; every path is checked against the scalar one first.",
        kernel::detect(),
        if cfg.quick { 3 } else { 7 }
    );
    kernel_rows(&cfg, &mut rng, &prep);
    block_rows(&cfg, &mut rng);
    estimate_rows(&cfg, &mut rng, &prep);
    prep_rows(&cfg, &mut rng, &rot);
    rotation_rows(&cfg, &mut rng);
}
