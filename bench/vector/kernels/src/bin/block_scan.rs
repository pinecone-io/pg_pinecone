//! The scan phase of one query, record-major (the page layout before blocks: one `estimate_bits` call
//! and one candidate push per record) against the blocked scan (`pg_pinecone/src/local/{pages, blockscan,
//! kernel, rabitq}.rs`, included verbatim): ns per record over a probed set of lists.
//! `cargo run --release --bin block_scan -- [--cpu N] [--quick]` prints Markdown.

#[allow(dead_code)]
#[path = "."]
mod local {
    pub mod metapage {
        pub(crate) fn usable_page_bytes() -> usize {
            8168
        }
        pub(crate) const INVALID_BLOCK: u32 = u32::MAX;
    }
    #[path = "../../../../../pg_pinecone/src/local/blockscan.rs"]
    pub mod blockscan;
    #[path = "../../../../../pg_pinecone/src/local/distance.rs"]
    pub mod distance;
    #[path = "../../../../../pg_pinecone/src/local/kernel.rs"]
    pub mod kernel;
    #[path = "../../../../../pg_pinecone/src/local/metric.rs"]
    pub mod metric;
    #[path = "../../../../../pg_pinecone/src/local/pages.rs"]
    pub mod pages;
    #[path = "../../../../../pg_pinecone/src/local/rabitq.rs"]
    pub mod rabitq;
    #[path = "../../../../../pg_pinecone/src/local/rng.rs"]
    pub mod rng;
    #[path = "../../../../../pg_pinecone/src/local/rotation.rs"]
    pub mod rotation;
}

use local::blockscan::{CandBuf, Candidate, ScanCounters, Scoring, scan_page};
use local::kernel::{Isa, Kernel};
use local::metric::Metric;
use local::pages::{Geometry, ListBody, ListPage, ListRecord};
use local::rabitq::{self, QueryPrep};
use local::rotation::{Rotation, padded_dim};
use std::borrow::Cow;
use std::hint::black_box;
use std::time::Instant;

const SEED: u64 = 42;

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
}

struct Cfg {
    quick: bool,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn measure(cfg: &Cfg, items: usize, mut f: impl FnMut() -> u64) -> f64 {
    let mut sink = f();
    let (min_ms, trials) = if cfg.quick { (5, 3) } else { (40, 7) };
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

/// `n` random records of `dim` dimensions: codes, and scalars in the ranges real data has.
struct Records {
    words: usize,
    codes: Vec<u64>,
    rnorm: Vec<f32>,
    dot: Vec<f32>,
    rc: Vec<f32>,
}

fn records(rng: &mut Rng, n: usize, dim: usize) -> Records {
    let words = padded_dim(dim) / 64;
    Records {
        words,
        codes: (0..n * words).map(|_| rng.next()).collect(),
        rnorm: (0..n).map(|_| 0.5 + 0.5 * (rng.unit() + 1.0)).collect(),
        dot: (0..n).map(|_| 0.72 + 0.1 * (rng.unit() + 1.0) * 0.5).collect(),
        rc: (0..n).map(|_| rng.unit()).collect(),
    }
}

fn pages_of(r: &Records, has_filter: bool) -> Vec<Vec<u8>> {
    let code_bytes = r.words * 8;
    let cap = Geometry::new(code_bytes, has_filter).capacity;
    let n = r.rnorm.len();
    (0..n)
        .step_by(cap)
        .map(|start| {
            let end = (start + cap).min(n);
            let mut body = ListBody::with_capacity(has_filter, code_bytes, end - start);
            for i in start..end {
                let code: Vec<u8> = r.codes[i * r.words..(i + 1) * r.words]
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect();
                body.push(&ListRecord {
                    code: Cow::Owned(code),
                    rnorm: r.rnorm[i],
                    dot: r.dot[i],
                    rc: r.rc[i],
                    tid: (i as u32 / 100, (i % 100) as u16 + 1),
                    fval: (i % 100) as i64,
                    fnull: false,
                });
            }
            let mut page = body.finish(0);
            page.resize(8168, 0);
            page
        })
        .collect()
}

fn prep_for(rng: &mut Rng, dim: usize, metric: Metric) -> (QueryPrep, usize) {
    let rot = Rotation::new(SEED, dim);
    let c: Vec<f32> = (0..dim).map(|_| 3.0 * rng.unit()).collect();
    let q: Vec<f32> = c.iter().map(|v| 0.9 * v + 0.2 * rng.unit()).collect();
    (
        rabitq::prepare_query(metric, &q, &rot.apply(&q), &c, &rot.apply(&c), SEED, 3),
        padded_dim(dim),
    )
}

fn main() {
    let mut cfg = Cfg { quick: false };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--quick" => cfg.quick = true,
            "--cpu" => pin(args.next().expect("--cpu N").parse().expect("a CPU number")),
            other => panic!("unknown argument {other}"),
        }
    }
    let kernel = Kernel::detect();
    println!("# Scan phase, record-major against blocked\n");
    println!(
        "CPU: {}; kernel path: {:?}. One probed set of 26 lists at 1,000 records each (26,000 records) per dimension; \
         the blocked scan with the early stop's bound on and off, a bounded buffer of 300 candidates; \
         the record-major path pushes every record's candidate and partitions the best 300 at the end, as the \
         page layout before blocks did.\n",
        cpu_model(),
        kernel.isa()
    );
    println!("| D | record-major ns/record | blocked, no bound | blocked, with bound | speed-up (with bound) | lanes | records/page |");
    println!("|---|---|---|---|---|---|---|");
    let mut rng = Rng(7);
    for dim in [128usize, 768, 1536] {
        let metric = Metric::L2;
        let n = 26_000;
        let recs = records(&mut rng, n, dim);
        let (prep, dp) = prep_for(&mut rng, dim, metric);
        let code_bytes = dp / 8;
        let geo = Geometry::new(code_bytes, false);
        let pages = pages_of(&recs, false);

        let record_major = measure(&cfg, n, || {
            let mut cands: Vec<(u32, f32)> = Vec::with_capacity(n);
            for i in 0..n {
                let (est, _) = rabitq::estimate_bits(
                    kernel,
                    &prep,
                    &recs.codes[i * recs.words..(i + 1) * recs.words],
                    recs.dot[i],
                );
                let key = rabitq::key_from_parts(metric, &prep, recs.rnorm[i], recs.rc[i], est);
                cands.push((i as u32, key));
            }
            cands.select_nth_unstable_by(299, |a, b| a.1.total_cmp(&b.1));
            cands.truncate(300);
            cands.iter().map(|c| u64::from(c.0)).sum::<u64>()
        });

        let blocked = |with_lo: bool| {
            measure(&cfg, n, || {
                let scoring = Scoring {
                    kernel,
                    prep: &prep,
                    metric,
                    dp,
                    with_lo,
                };
                let mut buf = CandBuf::best(300);
                let mut counters = ScanCounters::default();
                for page in &pages {
                    let parsed = ListPage::parse(page, code_bytes, false).expect("a page");
                    scan_page(&parsed, &scoring, None, &mut buf, &mut counters);
                }
                let cands: Vec<Candidate> = buf.into_candidates();
                cands.iter().map(|c| u64::from(c.tid.0)).sum::<u64>() ^ counters.scanned
            })
        };
        let (no_bound, with_bound) = (blocked(false), blocked(true));
        println!(
            "| {dp} | {record_major:.1} | {no_bound:.1} | {with_bound:.1} | {:.1}x | {} | {} |",
            record_major / with_bound,
            geo.lanes,
            geo.capacity
        );
    }
    println!();
    let _ = Isa::Scalar;
}
