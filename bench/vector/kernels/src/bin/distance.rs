//! Exact f32 distance kernels of pg_pinecone (`pg_pinecone/src/local/distance.rs`, included verbatim):
//! ns per distance for every path this CPU has, against the 0.9.0 sequential loops.
//! `cargo run --release --bin distance -- [--dim 768] [--cpu N] [--quick]` prints Markdown.

#[path = "../../../../../pg_pinecone/src/local/distance.rs"]
mod distance;

use distance::Kernel;
use std::hint::black_box;
use std::time::Instant;

const CENTROIDS: usize = 1_000;

struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        (((z ^ (z >> 31)) >> 40) as u32) as f32 / (1u32 << 23) as f32 - 1.0
    }

    fn vector(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.unit()).collect()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Op {
    L2Sq,
    Dot,
    Cosine,
}

impl Op {
    const ALL: [Op; 3] = [Op::L2Sq, Op::Dot, Op::Cosine];

    fn name(self) -> &'static str {
        match self {
            Op::L2Sq => "l2_sq",
            Op::Dot => "dot",
            Op::Cosine => "cosine",
        }
    }
}

fn sequential_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn sequential(op: Op, a: &[f32], b: &[f32]) -> f32 {
    match op {
        Op::L2Sq => a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum(),
        Op::Dot => sequential_dot(a, b),
        Op::Cosine => {
            let denom = sequential_dot(a, a).sqrt() * sequential_dot(b, b).sqrt();
            if denom == 0.0 {
                1.0
            } else {
                1.0 - sequential_dot(a, b) / denom
            }
        }
    }
}

fn oracle(op: Op, a: &[f32], b: &[f32]) -> f64 {
    let dot =
        |x: &[f32], y: &[f32]| -> f64 { x.iter().zip(y).map(|(&p, &q)| p as f64 * q as f64).sum() };
    match op {
        Op::L2Sq => a
            .iter()
            .zip(b)
            .map(|(&p, &q)| (p as f64 - q as f64).powi(2))
            .sum(),
        Op::Dot => dot(a, b),
        Op::Cosine => 1.0 - dot(a, b) / (dot(a, a).sqrt() * dot(b, b).sqrt()),
    }
}

fn kernel_call(kernel: Kernel, op: Op, a: &[f32], b: &[f32]) -> f32 {
    match op {
        Op::L2Sq => kernel.l2_sq(a, b),
        Op::Dot => kernel.dot(a, b),
        Op::Cosine => kernel.cosine(a, b),
    }
}

fn dispatched(op: Op, a: &[f32], b: &[f32]) -> f32 {
    match op {
        Op::L2Sq => distance::l2_sq(a, b),
        Op::Dot => distance::dot(a, b),
        Op::Cosine => distance::cosine(a, b),
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn ns_per_distance(quick: bool, per_rep: usize, mut f: impl FnMut() -> f32) -> f64 {
    let (min_ms, trials) = if quick { (5, 3) } else { (30, 7) };
    let mut sink = f();
    let mut reps = 1usize;
    loop {
        let t = Instant::now();
        for _ in 0..reps {
            sink += black_box(f());
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
                sink += black_box(f());
            }
            t.elapsed().as_nanos() as f64 / (reps * per_rep) as f64
        })
        .collect();
    black_box(sink);
    median(samples)
}

struct Inputs {
    a: Vec<f32>,
    b: Vec<f32>,
    centroids: Vec<f32>,
    dim: usize,
}

impl Inputs {
    fn l1(&self, f: impl Fn(&[f32], &[f32]) -> f32, quick: bool) -> f64 {
        ns_per_distance(quick, 1, || f(black_box(&self.a), black_box(&self.b)))
    }

    fn stream(&self, f: impl Fn(&[f32], &[f32]) -> f32, quick: bool) -> f64 {
        ns_per_distance(quick, CENTROIDS, || {
            let q = black_box(&self.a);
            let mut sum = 0.0;
            for c in self.centroids.chunks_exact(self.dim) {
                sum += f(q, c);
            }
            sum
        })
    }
}

fn check(op: Op, got: f32, a: &[f32], b: &[f32], what: &str) {
    let want = oracle(op, a, b);
    let scale = if op == Op::Cosine {
        1.0
    } else {
        want.abs().max(1.0)
    };
    assert!(
        (got as f64 - want).abs() <= 1e-4 * scale,
        "{what} {}: got {got}, want {want}",
        op.name()
    );
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
fn pin(_cpu: usize) {
    eprintln!("--cpu is ignored: no thread pinning on this OS");
}

fn cpu_model() -> String {
    if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
        if let Some(line) = text.lines().find(|l| l.starts_with("model name")) {
            return line.split(':').nth(1).unwrap_or("").trim().to_string();
        }
    }
    std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
    };
    let quick = flag("--quick");
    let dim: usize = value("--dim").map_or(768, |v| v.parse().expect("--dim takes a number"));
    if let Some(cpu) = value("--cpu") {
        pin(cpu.parse().expect("--cpu takes a core number"));
    }

    let mut rng = Rng(0xD157_A9CE);
    let inputs = Inputs {
        a: rng.vector(dim),
        b: rng.vector(dim),
        centroids: rng.vector(dim * CENTROIDS),
        dim,
    };

    let (supported, missing): (Vec<Kernel>, Vec<Kernel>) =
        Kernel::ALL.iter().partition(|k| k.supported());
    println!(
        "## Exact f32 distance, {dim}-d: ns per distance (median of {} trials)\n",
        if quick { 3 } else { 7 }
    );
    println!(
        "CPU: {}; paths here: {}; best: {}",
        cpu_model(),
        supported
            .iter()
            .map(|k| k.name())
            .collect::<Vec<_>>()
            .join(", "),
        Kernel::best().name()
    );
    if !missing.is_empty() {
        println!(
            "Paths compiled for this architecture but not supported by this CPU (skipped): {}",
            missing
                .iter()
                .map(|k| k.name())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!(
        "\nEvery variant is checked against an f64 oracle before it is timed. `L1 pair`: one {dim}-d pair, back to back; `{CENTROIDS} vectors`: one query against {CENTROIDS} vectors ({:.1} MB, the centroid ranking of one query).\n",
        (CENTROIDS * dim * 4) as f64 / 1e6
    );
    println!("| path | op | L1 pair | {CENTROIDS} vectors | speedup (L1 pair) |");
    println!("|---|---|---|---|---|");

    for op in Op::ALL {
        let base = inputs.l1(|a, b| sequential(op, a, b), quick);
        check(
            op,
            sequential(op, &inputs.a, &inputs.b),
            &inputs.a,
            &inputs.b,
            "sequential",
        );
        println!(
            "| 0.9.0 sequential f32 | {} | {base:.1} | {:.1} | 1.0x |",
            op.name(),
            inputs.stream(|a, b| sequential(op, a, b), quick)
        );
        for &kernel in &supported {
            check(
                op,
                kernel_call(kernel, op, &inputs.a, &inputs.b),
                &inputs.a,
                &inputs.b,
                kernel.name(),
            );
            let l1 = inputs.l1(|a, b| kernel_call(kernel, op, a, b), quick);
            let stream = inputs.stream(|a, b| kernel_call(kernel, op, a, b), quick);
            println!(
                "| {} | {} | {l1:.1} | {stream:.1} | {:.1}x |",
                kernel.name(),
                op.name(),
                base / l1
            );
        }
        let l1 = inputs.l1(|a, b| dispatched(op, a, b), quick);
        println!(
            "| dispatched (`distance::{}`) | {} | {l1:.1} | {:.1} | {:.1}x |",
            match op {
                Op::L2Sq => "l2_sq",
                Op::Dot => "dot",
                Op::Cosine => "cosine",
            },
            op.name(),
            inputs.stream(|a, b| dispatched(op, a, b), quick),
            base / l1
        );
    }
}
