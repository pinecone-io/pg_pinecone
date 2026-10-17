//! `CREATE INDEX ... USING pinecone (...) WITH (...)` reloptions — the
//! `mode`/engine-param surface. Goes through Postgres's own reloptions catalog
//! (`add_reloption_kind`/`add_enum_reloption`/`add_int_reloption` +
//! `build_reloptions`, registered once from `_PG_init` — same mechanism
//! contrib/bloom and pgvector's own `ivfflat`/`hnsw` use for their reloptions)
//! rather than hand-parsing the raw `WITH (...)` text: Postgres validates
//! each field's type/bounds and produces the clear "unrecognized parameter"/
//! "value ... out of range" errors itself (this module's job is just to
//! register the fields and their bounds).
//!
//! `mode` is a three-way enum, not the two-way `local`/`remote` split the
//! rest of this module talks about — see [`Mode`]'s own docs for why `auto`
//! has to be a real, distinct value rather than merely "the default local/
//! remote resolves to when omitted".
//!
//! One knob here is a GUC, not a reloption: [`build_parallelism`]'s backing
//! `pinecone.build_parallelism` (registered in [`init`] alongside the
//! reloptions, since that's the one local-engine init hook `am::init_gucs`
//! already calls — see [`init`]'s own rustdoc for why the GUC lives here
//! rather than beside `am.rs`'s other GUCs). Unlike `n_lists`/`k_bits`/
//! `seed`, a build's thread count leaves no trace in the on-disk format or
//! query behavior at all — it's a pure CPU-budget knob for the one
//! statement currently running, the same role `pinecone.rescore_multiplier`
//! (`scan.rs`) plays for its own build-time-fixed counterpart — so a session
//! `SET` (no `REINDEX`, no permanent reloption) is the right surface,
//! including for the parallel-vs-serial determinism cross-check this knob
//! exists to enable (`build.rs`'s parallel k-means/assign phases).

use std::sync::OnceLock;

use pgrx::{
    guc::{GucContext, GucFlags, GucRegistry, GucSetting},
    prelude::*,
};

/// `mode`'s three reloption values. `Auto` (the default) defers the local-
/// vs-remote choice to `am::resolve_mode`'s "least-surprising" rule (an
/// existing registry row, or a stored API key, means remote; otherwise
/// local) — it has to be a distinct enum member, not just "whatever `mode`
/// defaults to", because Postgres's reloptions machinery always hands
/// `amoptions` a fully-defaulted struct (there is no "the user didn't
/// specify this field at all" signal once `build_reloptions` returns) — see
/// this module's docstring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Auto = 0,
    Local = 1,
    Remote = 2,
}

impl Mode {
    fn from_i32(v: i32) -> Self {
        match v {
            0 => Mode::Auto,
            1 => Mode::Local,
            2 => Mode::Remote,
            other => error!("pg_pinecone: corrupt reloptions — unknown mode code {other}"),
        }
    }
}

/// The parsed, defaulted reloptions view every local-mode code path reads —
/// [`effective`] is the only public entry point; [`PgPineconeOptionsRaw`] is
/// purely the on-disk/`rd_options` wire format.
pub(crate) struct Options {
    pub(crate) mode: Mode,
    /// `0` means "auto: `ceil(sqrt(row_count))`" — resolved by
    /// `local::build`, which is the only place that knows `row_count`.
    pub(crate) n_lists: u32,
    /// Accepted and ignored since 0.10 (the codes are `D` bits, the dimension padded to a multiple of
    /// 64): kept so existing `WITH (k_bits = ...)` clauses and scripts still run. `local::build` says
    /// so with a NOTICE when it is non-zero.
    pub(crate) k_bits: u32,
    /// `0` = not set: the build calibrates the pool (`local::calibrate`).
    pub(crate) rescore_multiplier: u32,
    pub(crate) small_n_threshold: u32,
    /// `0` means "auto-derive at build time" — see `local::build`'s seed
    /// resolution.
    pub(crate) seed: u32,
    /// The recall@10 the build calibrates the probe count and pool for.
    pub(crate) target_recall: f64,
    /// `schema.table` or `schema.table(column)`: real queries to calibrate on instead of table rows
    /// (`local::queries`); `None` when not set.
    pub(crate) calibration_queries: Option<String>,
}

/// `build_reloptions`'s wire struct — `repr(C)`, `vl_len_` first (the same
/// varlena-header-first convention `rd_options: *mut bytea` itself requires;
/// matches pgvector's own `IvfflatOptions`/`HnswOptions` shape). Every field
/// here is a plain `int`/enum (`RELOPT_TYPE_INT`/`RELOPT_TYPE_ENUM`) rather
/// than `RELOPT_TYPE_STRING` specifically to avoid `RELOPT_TYPE_STRING`'s
/// offset-into-trailing-bytes indirection. `calibration_queries` is the one text option: its
/// field holds the offset of the string (0 = not set), which `build_reloptions` appends after the
/// struct ([`string_option`]).
#[repr(C)]
#[derive(Clone, Copy)]
struct PgPineconeOptionsRaw {
    vl_len_: i32,
    mode: i32,
    n_lists: i32,
    k_bits: i32,
    rescore_multiplier: i32,
    small_n_threshold: i32,
    seed: i32,
    calibration_queries: i32,
    target_recall: f64,
}

const DEFAULT_N_LISTS: i32 = 0; // auto
/// `0` = not given: `k_bits` has no effect since 0.10 (see [`Options::k_bits`]).
const DEFAULT_K_BITS: i32 = 0;
/// `0` = not set: the pool is calibrated at build time, or [`FALLBACK_POOL`] when it is not.
const DEFAULT_RESCORE_MULTIPLIER: i32 = 0;
/// The 0.9.0 pool (`rescore_multiplier` 10), kept for the scans calibration does not cover and for
/// indexes it skipped.
pub(crate) const FALLBACK_POOL: u32 = 1_000;
pub(crate) const DEFAULT_TARGET_RECALL: f64 = 0.95;
pub(crate) const MIN_TARGET_RECALL: f64 = 0.5;
pub(crate) const MAX_TARGET_RECALL: f64 = 0.999;
const DEFAULT_SMALL_N_THRESHOLD: i32 = 10_000;
const DEFAULT_SEED: i32 = 0; // auto-derive

/// `n_lists`'s reloption ceiling — shared with `metapage::validate_untrusted_
/// fields`'s own defense-in-depth bound so the
/// two can never silently drift apart: a corrupt on-disk metapage claiming
/// more lists than any `CREATE INDEX` could ever legitimately request is
/// exactly the signal that bound exists to catch.
pub(crate) const MAX_N_LISTS: u32 = 1_000_000;
/// `k_bits`'s reloption ceiling — and the loose bound the metapage check keeps on the code width of
/// an index below format 3 (same sharing rationale as [`MAX_N_LISTS`]).
pub(crate) const MAX_K_BITS: u32 = 4096;

/// The 0.9.0 policy, `ceil(4 * sqrt(n_lists))` lists: what iterative scans probe and what an index
/// without a calibration falls back to.
pub(crate) fn legacy_nprobe(n_lists: u32) -> u32 {
    probe_count(4.0, n_lists)
}

fn probe_count(coefficient: f64, n_lists: u32) -> u32 {
    let n = f64::from(n_lists);
    (((coefficient / n.sqrt()) * n).ceil() as u32).clamp(1, n_lists.max(1))
}

/// `SHOW pinecone.build_parallelism`'s default — `-1` ("auto": `min(8, cores /
/// 2)`, resolved lazily by [`build_parallelism`] since `cores` isn't a
/// compile-time constant a `static` initializer can compute). `cores / 2`
/// leaves the other half of the box for concurrent backends/autovacuum
/// during a foreground `CREATE INDEX`/`REINDEX` (this is NOT a
/// `maintenance_work_mem`-style background job with a dedicated window).
/// The `8` cap is MEASURED, not guessed (ad-hoc release-mode timings, 200k x
/// 768d, this development machine — see the parbuild report): speedup over
/// serial was 1.86x/3.39x/4.63x/5.95x/7.40x at 2/4/6/8/12 threads
/// respectively (efficiency 93%/85%/77%/74%/62%) — clearly still a net win
/// at 8 (not "marginal" — a real ~6x), with the knee bending noticeably
/// harder past that point as the parallel assign/update loops' memory-
/// bandwidth-bound nature (streaming reads over the collected `Vec<f32>`
/// data) starts to dominate over added threads. `8` sits just past that
/// knee: enough to make a credible dent in the 1M x 768d build-time bar
/// this GUC exists to close, without defaulting to a thread count this
/// module has no measurements past (raise it explicitly via `SET` for a
/// dedicated maintenance window — measurements above 12 threads on a bigger
/// box are exactly the kind of thing worth re-sweeping there). `0` is its
/// own distinct, explicit value ("serial fallback"), never reused as the
/// auto sentinel: `local::kmeans`/`local::build`'s determinism contract
/// depends on a caller being able to force strictly-serial execution on
/// demand (the byte-identical-index cross-check `am_local_build_parallelism_
/// is_byte_identical_to_serial` in `lib.rs` relies on exactly this), so `0`
/// must always mean that, never "whatever auto happens to resolve to".
static BUILD_PARALLELISM: GucSetting<i32> = GucSetting::<i32>::new(-1);

static RELOPT_KIND: OnceLock<pg_sys::relopt_kind::Type> = OnceLock::new();

/// Registers `pg_pinecone`'s AM reloptions kind and fields — called once from
/// `am::init_gucs` (in turn called from `_PG_init`), matching the
/// shared_preload_libraries-time requirement `add_reloption_kind`/
/// `add_*_reloption` share with GUC registration (`GucRegistry::define_*`).
pub(crate) fn init() {
    unsafe {
        let kind = pg_sys::add_reloption_kind();

        // Leaked once, for the process's lifetime — Postgres retains this
        // pointer in its reloptions catalog and consults it on every
        // subsequent `CREATE INDEX`/relcache-load `amoptions` call, so it
        // must outlive all of them; a `'static` leak is the standard way C
        // extensions provide that (this mirrors how the GUC statics above
        // are similarly leaked for the process's life).
        let members: &'static mut [pg_sys::relopt_enum_elt_def] = Box::leak(Box::new([
            pg_sys::relopt_enum_elt_def {
                string_val: c"auto".as_ptr(),
                symbol_val: Mode::Auto as i32,
            },
            pg_sys::relopt_enum_elt_def {
                string_val: c"local".as_ptr(),
                symbol_val: Mode::Local as i32,
            },
            pg_sys::relopt_enum_elt_def {
                string_val: c"remote".as_ptr(),
                symbol_val: Mode::Remote as i32,
            },
            // Sentinel: reloptions.c's enum parser walks this array until
            // `string_val == NULL`.
            pg_sys::relopt_enum_elt_def {
                string_val: std::ptr::null(),
                symbol_val: 0,
            },
        ]));

        pg_sys::add_enum_reloption(
            kind,
            c"mode".as_ptr(),
            c"Storage engine this index uses: 'local' (the native in-index engine), \
              'remote' (Pinecone-backed, 0.3's original access-method mode), or 'auto' \
              (defers to whether this table already has remote credentials/registration)."
                .as_ptr(),
            members.as_mut_ptr(),
            Mode::Auto as i32,
            c"Valid values are \"auto\", \"local\", and \"remote\".".as_ptr(),
            pg_sys::AccessExclusiveLock as i32,
        );
        pg_sys::add_int_reloption(
            kind,
            c"n_lists".as_ptr(),
            c"Number of IVF lists (0 = auto: ceil(sqrt(row count)) at build time).".as_ptr(),
            DEFAULT_N_LISTS,
            0,
            MAX_N_LISTS as i32,
            pg_sys::AccessExclusiveLock as i32,
        );
        pg_sys::add_int_reloption(
            kind,
            c"k_bits".as_ptr(),
            c"Ignored since 0.10 (codes are D bits, the dimension padded to a multiple of 64); \
              accepted so existing indexes and scripts keep working. Must be a multiple of 8."
                .as_ptr(),
            DEFAULT_K_BITS,
            0,
            MAX_K_BITS as i32,
            pg_sys::AccessExclusiveLock as i32,
        );
        pg_sys::add_int_reloption(
            kind,
            c"rescore_multiplier".as_ptr(),
            c"Exact-rescore candidate pool size, as a multiple of the assumed top-k \
              (0 = calibrate it at build time for target_recall)."
                .as_ptr(),
            DEFAULT_RESCORE_MULTIPLIER,
            0,
            1000,
            pg_sys::AccessExclusiveLock as i32,
        );
        pg_sys::add_real_reloption(
            kind,
            c"target_recall".as_ptr(),
            c"recall@10 the build calibrates the default probe count and rescore pool for."
                .as_ptr(),
            DEFAULT_TARGET_RECALL,
            MIN_TARGET_RECALL,
            MAX_TARGET_RECALL,
            pg_sys::AccessExclusiveLock as i32,
        );
        pg_sys::add_int_reloption(
            kind,
            c"small_n_threshold".as_ptr(),
            c"Row-count threshold below which the index bootstraps as an exact flat scan \
              instead of clustering."
                .as_ptr(),
            DEFAULT_SMALL_N_THRESHOLD,
            0,
            100_000_000,
            pg_sys::AccessExclusiveLock as i32,
        );
        pg_sys::add_int_reloption(
            kind,
            c"seed".as_ptr(),
            c"Random seed of the RaBitQ rotation and of k-means (0 = auto-derive at build time)."
                .as_ptr(),
            DEFAULT_SEED,
            0,
            i32::MAX,
            pg_sys::AccessExclusiveLock as i32,
        );
        pg_sys::add_string_reloption(
            kind,
            c"calibration_queries".as_ptr(),
            c"Table of real queries ('schema.table' or 'schema.table(column)') the build calibrates \
              the probe count and rescore pool on, instead of table rows; read at the next \
              CREATE INDEX or REINDEX."
                .as_ptr(),
            std::ptr::null(),
            Some(validate_calibration_queries),
            pg_sys::AccessExclusiveLock as i32,
        );

        RELOPT_KIND
            .set(kind)
            .unwrap_or_else(|_| error!("pg_pinecone: local::options::init called more than once"));
    }

    GucRegistry::define_int_guc(
        c"pinecone.build_parallelism",
        c"Worker-thread count for a local-mode ambuild's k-means training and final \
          centroid-assignment phases (-1 = auto: min(8, cores/2); 0 = fully serial, \
          for determinism cross-checks; all values, including auto's resolved count, \
          produce byte-identical index pages for the same (seed, data)).",
        c"",
        &BUILD_PARALLELISM,
        -1,
        1024,
        GucContext::Userset,
        GucFlags::default(),
    );
}

/// The resolved worker-thread count [`crate::local::build`]/[`crate::local::
/// kmeans`] should use for one ambuild — never `-1` (the raw GUC's own auto
/// sentinel; resolved here, not stored back into the GUC, since `SHOW` should
/// keep surfacing "auto" rather than whatever core count happened to be
/// detected). `0` passes through unchanged — see [`BUILD_PARALLELISM`]'s own
/// rustdoc for why that value is never auto-resolved away.
pub(crate) fn build_parallelism() -> usize {
    let configured = BUILD_PARALLELISM.get();
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    if configured < 0 {
        (cores / 2).min(8)
    } else {
        // Cap an EXPLICIT setting at the actual core count too: the GUC's
        // upper bound is a sanity rail, not a promise that 64 scoped threads
        // help on a 10-core box — oversubscribing the memory-bandwidth-bound
        // assign/update loops only adds scheduler
        // churn to a foreground CREATE INDEX. Output is unaffected either
        // way: the parbuild design is byte-identical across ANY thread count
        // (the determinism matrix in lib.rs), so this cap is purely a
        // resource guard. `0` (the explicit serial fallback) passes through
        // unchanged (`0 < cores` always).
        (configured as usize).min(cores)
    }
}

/// `calibration_queries`'s check at DDL time (`CREATE INDEX ... WITH`, `ALTER INDEX ... SET`): the
/// shape only, as the table may not exist yet and `SET` takes effect at the next rebuild. Postgres
/// also calls it with the default, `NULL`, when the option is registered.
#[pg_guard]
unsafe extern "C-unwind" fn validate_calibration_queries(value: *const std::ffi::c_char) {
    if value.is_null() {
        return;
    }
    let text = unsafe { std::ffi::CStr::from_ptr(value) }.to_string_lossy();
    if !text.trim().is_empty()
        && let Err(problem) = super::queries::parse(&text)
    {
        problem.raise(false);
    }
}

/// The text option stored `offset` bytes into the options struct at `base` (0 = not set); blank is
/// not set either.
unsafe fn string_option(base: *const u8, offset: i32) -> Option<String> {
    if offset <= 0 {
        return None;
    }
    let text = unsafe { std::ffi::CStr::from_ptr(base.add(offset as usize).cast()) }
        .to_string_lossy()
        .trim()
        .to_string();
    (!text.is_empty()).then_some(text)
}

/// `amoptions`'s entire body — see `am::amoptions`, which just forwards here.
/// Always returns a non-null, fully-defaulted `bytea*` (never `NULL`, even
/// with no `WITH (...)` clause at all) — matching `build_reloptions`'s own
/// behavior of unconditionally producing a defaults-filled struct, so
/// `index.rd_options` is never null for a `pg_pinecone`-AM index and every reader
/// can skip a null check... except [`effective`] still defends against it,
/// since that invariant lives in Postgres's relcache plumbing, not in this
/// module's control.
pub(crate) unsafe fn amoptions(reloptions: pg_sys::Datum, validate: bool) -> *mut pg_sys::bytea {
    const TAB: &[pg_sys::relopt_parse_elt] = &[
        pg_sys::relopt_parse_elt {
            optname: c"mode".as_ptr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_ENUM,
            offset: std::mem::offset_of!(PgPineconeOptionsRaw, mode) as i32,
            #[cfg(feature = "pg18")]
            isset_offset: 0,
        },
        pg_sys::relopt_parse_elt {
            optname: c"n_lists".as_ptr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset: std::mem::offset_of!(PgPineconeOptionsRaw, n_lists) as i32,
            #[cfg(feature = "pg18")]
            isset_offset: 0,
        },
        pg_sys::relopt_parse_elt {
            optname: c"k_bits".as_ptr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset: std::mem::offset_of!(PgPineconeOptionsRaw, k_bits) as i32,
            #[cfg(feature = "pg18")]
            isset_offset: 0,
        },
        pg_sys::relopt_parse_elt {
            optname: c"rescore_multiplier".as_ptr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset: std::mem::offset_of!(PgPineconeOptionsRaw, rescore_multiplier) as i32,
            #[cfg(feature = "pg18")]
            isset_offset: 0,
        },
        pg_sys::relopt_parse_elt {
            optname: c"small_n_threshold".as_ptr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset: std::mem::offset_of!(PgPineconeOptionsRaw, small_n_threshold) as i32,
            #[cfg(feature = "pg18")]
            isset_offset: 0,
        },
        pg_sys::relopt_parse_elt {
            optname: c"seed".as_ptr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset: std::mem::offset_of!(PgPineconeOptionsRaw, seed) as i32,
            #[cfg(feature = "pg18")]
            isset_offset: 0,
        },
        pg_sys::relopt_parse_elt {
            optname: c"calibration_queries".as_ptr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_STRING,
            offset: std::mem::offset_of!(PgPineconeOptionsRaw, calibration_queries) as i32,
            #[cfg(feature = "pg18")]
            isset_offset: 0,
        },
        pg_sys::relopt_parse_elt {
            optname: c"target_recall".as_ptr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_REAL,
            offset: std::mem::offset_of!(PgPineconeOptionsRaw, target_recall) as i32,
            #[cfg(feature = "pg18")]
            isset_offset: 0,
        },
    ];

    unsafe {
        let kind = *RELOPT_KIND
            .get()
            .expect("pg_pinecone: local::options::init not called");
        let result = pg_sys::build_reloptions(
            reloptions,
            validate,
            kind,
            std::mem::size_of::<PgPineconeOptionsRaw>(),
            TAB.as_ptr(),
            TAB.len() as i32,
        ) as *mut pg_sys::bytea;

        // `k_bits % 8 == 0` isn't expressible as an `add_int_reloption` min/max bound, so it's
        // checked here instead, on the `validate = true` pass DDL time always
        // makes (`am::amoptions`'s own rustdoc). It stays although the value is ignored: a
        // `k_bits` that was never valid is still a typo worth refusing. `0` (not given)
        // trivially passes this check (`0 % 8 == 0`) without special-casing it.
        if validate && !result.is_null() {
            let raw = std::ptr::read_unaligned(result as *const PgPineconeOptionsRaw);
            if raw.k_bits % 8 != 0 {
                error!(
                    "pg_pinecone: k_bits ({}) must be a multiple of 8 (the reloption is ignored since \
                     0.10, but a value that was never valid is still refused)",
                    raw.k_bits
                );
            }
        }
        result
    }
}

/// Reads `index`'s already-parsed reloptions (`rd_options`, populated by
/// [`amoptions`] at relcache-load time — see this module's docstring) into
/// an [`Options`]. Falls back to compiled-in defaults if `rd_options` is
/// somehow null (defensive: [`amoptions`] never itself returns null, but
/// this doesn't otherwise depend on that holding).
pub(crate) fn effective(index: pg_sys::Relation) -> Options {
    unsafe {
        let ptr = (*index).rd_options as *const PgPineconeOptionsRaw;
        if ptr.is_null() {
            return Options {
                mode: Mode::Auto,
                n_lists: DEFAULT_N_LISTS as u32,
                k_bits: DEFAULT_K_BITS as u32,
                rescore_multiplier: DEFAULT_RESCORE_MULTIPLIER as u32,
                small_n_threshold: DEFAULT_SMALL_N_THRESHOLD as u32,
                seed: DEFAULT_SEED as u32,
                target_recall: DEFAULT_TARGET_RECALL,
                calibration_queries: None,
            };
        }
        let raw = std::ptr::read_unaligned(ptr);
        Options {
            mode: Mode::from_i32(raw.mode),
            n_lists: raw.n_lists as u32,
            k_bits: raw.k_bits as u32,
            rescore_multiplier: raw.rescore_multiplier as u32,
            small_n_threshold: raw.small_n_threshold as u32,
            seed: raw.seed as u32,
            target_recall: raw.target_recall,
            calibration_queries: string_option(ptr.cast(), raw.calibration_queries),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local::scan::ASSUMED_TOP_K;

    #[test]
    fn legacy_probe_count_is_the_090_expression_bit_for_bit() {
        for n in 1..=20_000u32 {
            let was = (((4.0 / f64::from(n).sqrt()) * f64::from(n)).ceil() as u32).clamp(1, n);
            assert_eq!(legacy_nprobe(n), was, "n_lists {n}");
        }
        assert_eq!(legacy_nprobe(100), 40);
        assert_eq!(legacy_nprobe(1000), 127);
    }

    #[test]
    fn the_fallback_pool_is_the_090_pool() {
        assert_eq!(FALLBACK_POOL as usize, 10 * ASSUMED_TOP_K);
    }

    #[test]
    fn the_pool_is_calibrated_unless_the_reloption_is_set() {
        assert_eq!(DEFAULT_RESCORE_MULTIPLIER, 0);
    }

    #[test]
    fn the_target_recall_range_holds_its_default() {
        assert_eq!(
            (MIN_TARGET_RECALL, DEFAULT_TARGET_RECALL, MAX_TARGET_RECALL),
            (0.5, 0.95, 0.999)
        );
    }
}
