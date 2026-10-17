//! Metapage v2/v3: block 0 of every pg_pinecone-AM index relation, extended from
//! 0.3's placeholder (`am.rs`'s old `PgPineconeMetaPageData { version, reserved }`)
//! to carry the mode discriminator + engine params, and (v3) the
//! filter-column metadata the two-mode filtered scan needs.
//!
//! **Version read-path validation is the whole compatibility story**: a
//! metapage's first 4 bytes are always a `u32` version tag, checked before
//! anything else is interpreted.
//! - `version == 1`: a pre-0.5 (0.3-built) index — [`MetaPageV1`] is
//!   `{version, reserved}` and carries no mode field at all, because 0.3 had
//!   only one mode. By design (v1 metapages map to mode=remote
//!   FOREVER, `ALTER EXTENSION UPDATE` never forces `REINDEX`), this maps
//!   unconditionally to [`Mode::Remote`] — permanently, not just until the
//!   next rebuild: an `ALTER EXTENSION pg_pinecone UPDATE` from 0.3 rewrites no
//!   index files, so any physical index built under 0.3 keeps these exact
//!   bytes on disk until its owner runs `REINDEX` (which calls [`ambuild`]
//!   again and writes a fresh metapage) — a v1-tagged index must keep
//!   working with zero forced maintenance.
//! - `version == 2`: [`MetaPageV2Raw`] below, read in full. **Still the live
//!   write format for every UNFILTERED index** (local or remote mode) — v2
//!   is not a fossil: an index with no filter column gets the same bytes
//!   earlier builds wrote (bar the format word, below), metapage and list pages alike, which is
//!   what makes the "unfiltered scans stay byte-identical" claim
//!   literal rather than aspirational, and keeps every earlier physical
//!   index readable with zero REINDEX.
//! - `version == 3`: [`MetaPageV3Raw`] — v2's fields plus the filter
//!   column's heap attnum, its [`FilterKind`], and the stats page's block
//!   ([`MetaPage::stats_start`]). Written if and only if the index has a
//!   filter column; a v3 index's LIST pages also carry the two extra
//!   per-record filter arrays (`pages.rs`'s module docstring), which is why
//!   this is a real version bump and not more `_reserved` scavenging — a
//!   binary that didn't know v3 would misread those pages.
//! - anything else: a clear `ERROR` rather than a guess — a newer/corrupt metapage this build
//!   doesn't understand must never be silently misinterpreted.
//!
//! ## Format version (0.10)
//!
//! The tag above picks the struct and never changes meaning. Separately, every v2/v3 payload ends in a
//! `format_version` word ([`FORMAT_OFFSET`], the same offset in both) that versions the whole index,
//! list pages included: a later on-disk change bumps it. Zero (a 0.9 page, where the word sits in
//! `_reserved`) reads as format 1; this build writes [`FORMAT_VERSION`] (3: RaBitQ codes, `pages.rs`).
//! [`read`] refuses a *newer* format with `55000` and "REINDEX it" before it interprets a field. An
//! *older* one matters only where list pages are read: a remote-mode (or `Demoting`) index carries no
//! codes and keeps working at format 1 or 2, while a native (`Local`/`Promoting`) one holds codes this
//! build cannot score. [`read`] still returns it ([`MetaPage::needs_reindex`]) and each entry point
//! decides: the planner walks away, scans and inserts raise `55000`, VACUUM and the maintenance
//! readers warn ([`refuse_old_format`], [`warn_old_format`]), until REINDEX writes format 3. Only a
//! build (`CREATE INDEX`, `REINDEX`) stamps the word: the in-place page updates ([`flip_mode`],
//! [`bump_inserted_on_page`]) keep what they read, so an upgrade rewrites no index. No separate
//! magic: the tag, checked first, is the page's type tag.
//!
//! Within format 3 the word's second byte is the **list page layout** ([`LAYOUT_VERSION`]): 0 is the
//! record-major pages of the 0.10 snapshots before page blocks, 1 the byte-interleaved code blocks
//! (`pages.rs`). Format 3 stays the format the release ships; a native index of an older layout is
//! refused exactly like an older format (REINDEX), a newer one like a newer format.
//!
//! Format 3 adds the codec fields: `code_bits` (in the slot `k_bits` had: the width of a code, now
//! always `D`, the dimension padded to 64 and to at least 256), `rotation_rounds` (4) and `bq_bits` (4, bits per query
//! coordinate), carved out of `_reserved`. Zero in a remote-mode page, which has no codes.
//!
//! It also carries the build's calibration (`local::calibrate`): the probe count in the slot `nprobe` had
//! (a page below format 3 still holds 0.9.0's probe count there, which [`read`] ignores), and from the
//! reserved words the pool, the sample count, whether the target was reached, where the sample's queries
//! came from ([`QueriesSource`]), the sample recall and the target. Zero when nothing was calibrated (a flat index, a table below `calibrate::MIN_ROWS`).
//!
//! [`ambuild`]: crate::am::ambuild

use pgrx::prelude::*;

use super::metric::Metric;

/// One byte on disk (metapage v2's `mode` field) — `Detached` is RESERVED
/// and unused by any code path; it exists purely so
/// a future version never has to widen this field. `Promoting`/`Demoting`
/// (`pinecone.promote_index`/`demote_index`) are the two in-flight lifecycle
/// states: a `Promoting` index is a local-mode index whose backfill to
/// remote is in progress (scans/dual-writes still serve/maintain LOCAL —
/// never-partial-namespace); a `Demoting` index is a remote-mode index whose
/// local rebuild is scheduled/in progress (scans/writes still serve REMOTE —
/// never-unavailable). Neither needs a metapage version bump — same
/// padding-scavenged-value technique `Promoting`/`Detached` themselves used
/// (one more of 256 possible byte values).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Local = 0,
    Remote = 1,
    Promoting = 2,
    #[allow(dead_code)]
    Detached = 3,
    Demoting = 4,
}

impl Mode {
    /// `None` for a byte this build never wrote (what [`peek`] reports without raising).
    fn try_from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Mode::Local),
            1 => Some(Mode::Remote),
            2 => Some(Mode::Promoting),
            3 => Some(Mode::Detached),
            4 => Some(Mode::Demoting),
            _ => None,
        }
    }

    fn from_u8(v: u8) -> Self {
        Self::try_from_u8(v)
            .unwrap_or_else(|| error!("pg_pinecone: corrupt metapage — unknown mode code {v}"))
    }

    /// Whether scans, inserts and sweeps in this mode read the list pages: `Local`, and
    /// `Promoting` (still serving local scans). The remote modes keep no codes.
    pub(crate) fn is_native(self) -> bool {
        matches!(self, Mode::Local | Mode::Promoting)
    }
}

/// One byte on disk (metapage v3's `filter_kind` field): what a filtered
/// index's second key column stores per record. The int kinds all store the
/// raw value sign-extended to `i64` (so cross-width equality is one integer
/// compare), but the WIDTH is still recorded — the small-N/flat scan path
/// reads the filter column back out of live heap tuples and must know how to
/// decode that datum. `Text` stores `pages::fnv1a64` of the column's bytes;
/// hash equality is a candidate test, not a proof, so text-filtered scans
/// set `xs_recheck` (see `scan.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FilterKind {
    None = 0,
    Int2 = 1,
    Int4 = 2,
    Int8 = 3,
    Text = 4,
}

impl FilterKind {
    pub(crate) fn from_u8(v: u8) -> Self {
        match v {
            0 => FilterKind::None,
            1 => FilterKind::Int2,
            2 => FilterKind::Int4,
            3 => FilterKind::Int8,
            4 => FilterKind::Text,
            other => error!("pg_pinecone: corrupt metapage — unknown filter kind {other}"),
        }
    }

    pub(crate) fn as_u8(self) -> u8 {
        self as u8
    }

    /// The kind of a column of this `pg_type.typname`; `None` for a type with no filter opclass.
    pub(crate) fn from_typname(typname: &str) -> Option<Self> {
        match typname {
            "int2" => Some(FilterKind::Int2),
            "int4" => Some(FilterKind::Int4),
            "int8" => Some(FilterKind::Int8),
            "text" | "varchar" => Some(FilterKind::Text),
            _ => None,
        }
    }

    /// The text of one non-NULL `text`/`varchar` datum, as Pinecone filters on it (exact bytes).
    ///
    /// # Safety
    /// `d` must be a valid, non-NULL `text` or `varchar` datum.
    pub(crate) unsafe fn datum_to_text(d: pg_sys::Datum) -> String {
        unsafe {
            String::from_datum(d, false)
                .expect("pg_pinecone: non-null text filter datum failed to convert")
        }
    }

    /// The Pinecone metadata value of one non-NULL datum OF THIS KIND'S OWN TYPE: an integer is
    /// a JSON number (any width, exact), text its string. What `aminsert` and the backfill sync,
    /// and what a scan's `$eq` compares against.
    ///
    /// # Safety
    /// `d` must be a valid, non-NULL datum of the Postgres type this kind was derived from.
    pub(crate) unsafe fn datum_to_json(self, d: pg_sys::Datum) -> serde_json::Value {
        unsafe {
            match self {
                FilterKind::None => {
                    error!(
                        "pg_pinecone: filter value requested with no filter column — unreachable"
                    )
                }
                FilterKind::Text => serde_json::Value::String(Self::datum_to_text(d)),
                _ => serde_json::Value::from(self.datum_to_value(d)),
            }
        }
    }

    /// Converts one non-NULL datum OF THIS KIND'S OWN TYPE to the stored
    /// `i64` filter value — used at build time (`build.rs`'s collect
    /// callback) and by the flat scan path's live-heap filter check; the
    /// scankey side (whose right-hand type can differ cross-type — e.g.
    /// `int8col = 5::int4`) goes through `scan.rs`'s own subtype-aware
    /// conversion instead.
    ///
    /// # Safety
    /// `d` must be a valid, non-NULL datum of the Postgres type this kind
    /// was derived from (`am::ambuild`'s filter-column validation).
    pub(crate) unsafe fn datum_to_value(self, d: pg_sys::Datum) -> i64 {
        unsafe {
            match self {
                FilterKind::None => {
                    error!(
                        "pg_pinecone: filter value requested with no filter column — unreachable"
                    )
                }
                FilterKind::Int2 => i16::from_datum(d, false)
                    .expect("pg_pinecone: non-null int2 filter datum failed to convert")
                    as i64,
                FilterKind::Int4 => i32::from_datum(d, false)
                    .expect("pg_pinecone: non-null int4 filter datum failed to convert")
                    as i64,
                FilterKind::Int8 => i64::from_datum(d, false)
                    .expect("pg_pinecone: non-null int8 filter datum failed to convert"),
                FilterKind::Text => hash_text_datum(d),
            }
        }
    }
}

/// `pages::fnv1a64` over a text/varchar datum's (detoasted) bytes. Postgres
/// `text` cannot contain a NUL byte, so `text_to_cstring`'s C-string round
/// trip is lossless; the palloc'd copy is freed immediately rather than
/// left for context reset (a large build hashes one of these per row).
unsafe fn hash_text_datum(d: pg_sys::Datum) -> i64 {
    unsafe {
        let detoasted = pg_sys::pg_detoast_datum(d.cast_mut_ptr::<pg_sys::varlena>());
        let cstr = pg_sys::text_to_cstring(detoasted as *const pg_sys::text);
        let hash = super::pages::fnv1a64(std::ffi::CStr::from_ptr(cstr).to_bytes());
        pg_sys::pfree(cstr as *mut std::ffi::c_void);
        // If detoasting made a fresh copy, free that too (matches how core
        // code pairs PG_DETOAST_DATUM with PG_FREE_IF_COPY).
        if detoasted != d.cast_mut_ptr::<pg_sys::varlena>() {
            pg_sys::pfree(detoasted as *mut std::ffi::c_void);
        }
        hash
    }
}

/// Postgres's `ReadBufferExtended` "extend the file" sentinel
/// (`P_NEW`/`InvalidBlockNumber`) — see `am.rs`'s own `P_NEW` for why this is
/// redefined locally rather than relying on bindgen's macro resolution.
const P_NEW: pg_sys::BlockNumber = 0xFFFFFFFF;
/// Matches `pg_sys::InvalidBlockNumber`'s value — used as the on-disk
/// "no block" sentinel for `centroid_dir_start`/`stats_start`/list-chain
/// `next_block`.
pub(crate) const INVALID_BLOCK: u32 = 0xFFFFFFFF;

const META_VERSION_V1: u32 = 1;
const META_VERSION_V2: u32 = 2;
const META_VERSION_V3: u32 = 3;

/// The index format this build writes and the newest it reads (module docs, "Format version").
pub(crate) const FORMAT_VERSION: u32 = 3;
/// Offset of the `format_version` word in the metapage payload: the last of its 88 bytes, in v2 and v3.
pub(crate) const FORMAT_OFFSET: usize = 84;
/// The list page layout this build writes and reads within format 3 (module docs): the second byte of
/// the format word.
pub(crate) const LAYOUT_VERSION: u32 = 1;

/// What the format word of block 0 says: the format version and, within it, the list page layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Stored {
    pub(crate) format: u32,
    pub(crate) layout: u32,
}

impl Stored {
    pub(crate) const CURRENT: Stored = Stored {
        format: FORMAT_VERSION,
        layout: LAYOUT_VERSION,
    };

    /// Zero (a 0.9 page) reads as format 1. A word above 16 bits is none this build wrote: it reads as
    /// that format number, which is newer than any this build knows.
    pub(crate) fn from_word(word: u32) -> Stored {
        if word > 0xFFFF {
            Stored {
                format: word,
                layout: 0,
            }
        } else {
            Stored {
                format: (word & 0xFF).max(1),
                layout: word >> 8,
            }
        }
    }

    pub(crate) fn word(self) -> u32 {
        self.format | self.layout << 8
    }

    /// Older than this build reads: an older format, or the format's own older page layout.
    pub(crate) fn is_older(self) -> bool {
        self.format < FORMAT_VERSION
            || (self.format == FORMAT_VERSION && self.layout < LAYOUT_VERSION)
    }

    /// Newer than this build reads.
    pub(crate) fn is_newer(self) -> bool {
        self.format > FORMAT_VERSION
            || (self.format == FORMAT_VERSION && self.layout > LAYOUT_VERSION)
    }
}

/// 0.3's original metapage payload — read-only from 0.5 onward (see this
/// module's docstring). Byte-for-byte identical to `am.rs`'s old
/// `PgPineconeMetaPageData`.
#[repr(C)]
struct MetaPageV1 {
    version: u32,
    #[allow(dead_code)]
    reserved: u32,
}

/// The v2 on-disk payload, `repr(C)` for a stable byte layout — still the
/// live write format for unfiltered indexes (module docstring). `_reserved`
/// gives headroom to add fields without a version bump as long as they fit
/// in the padding AND change no other page's layout; filter support
/// fails the second condition (list pages grow per-record arrays), hence v3.
/// `inserted_since_build` is exactly such a padding-scavenged addition:
/// one `u32` taken out of `_reserved`, which every earlier build wrote as
/// zero — so an earlier physical index reads back as "zero inserts since
/// build", which is true, with no version bump and no REINDEX.
#[repr(C)]
#[derive(Clone, Copy)]
struct MetaPageV2Raw {
    version: u32,
    mode: u8,
    metric: u8,
    is_flat: u8,
    _pad0: u8,
    /// Which `heap`'s attribute number holds the indexed vector column —
    /// needed at scan time (exact-rescore, small-N bootstrap) to pull the
    /// raw vector back out of a heap tuple; the remote path never needed this
    /// since it has no rescore/bootstrap step of its own, but it costs
    /// nothing to record uniformly for both modes.
    heap_attnum: i32,
    dim: u32,
    n_lists: u32,
    /// Formats 1 and 2: the FJLT code width (the `k_bits` reloption, resolved). Format 3: `D`, the
    /// RaBitQ code width (the dimension padded to 64, at least 256); zero in a remote-mode page.
    code_bits: u32,
    rescore_multiplier: u32,
    small_n_threshold: u32,
    /// Format 3: the probe count the build calibrated (`local::calibrate`), 0 when it did not.
    calib_nprobe: u32,
    total_vectors: u32,
    seed: u64,
    /// First block of the centroid directory page chain, or
    /// [`INVALID_BLOCK`] when `is_flat` (no clustering at all).
    centroid_dir_start: u32,
    /// Rows appended by local `aminsert` since the last `ambuild` — the
    /// stats-staleness counter (the "appended-since-build" drift
    /// metric). Updated in place via [`bump_inserted_on_page`], read fresh
    /// via [`read_inserted_count_fresh`] — never maintained through the
    /// `rd_amcache`d view, which goes stale the moment anyone inserts.
    inserted_since_build: u32,
    /// Format 3: the RaBitQ rotation's round count (0 below format 3 and in remote mode).
    rotation_rounds: u32,
    /// Format 3: bits per query coordinate of the estimator (0 below format 3 and in remote mode).
    bq_bits: u32,
    /// Format 3, from `calib_pool_samples` on: the calibration's pool (low 16 bits), sample count (the
    /// next 13), queries source (the two above, [`QueriesSource`]; the top bit is set when the target
    /// was not reached), the sample recall and the target as `f32` bits; zero when there is none.
    calib_pool_samples: u32,
    calib_recall_bits: u32,
    target_recall_bits: u32,
    /// Sized so the struct has NO tail padding (56 param bytes, 28 reserved
    /// and the 4-byte `format_version` make 88, a multiple of the u64-driven
    /// 8-byte alignment): `write` copies the whole struct into a WAL-logged
    /// page, and tail padding would be uninitialized stack bytes leaking to
    /// disk.
    _reserved: [u32; 2],
    /// The index format (module docs): zero in a 0.9 page, which reads as 1.
    format_version: u32,
}

/// The v3 on-disk payload: v2's fields plus the filter column's
/// metadata. Field-for-field identical to [`MetaPageV2Raw`] through
/// `centroid_dir_start` except that v2's `_pad0` byte now carries
/// `filter_kind` (safe: v2 always wrote it as `0`, which is
/// `FilterKind::None`); the three new trailing fields take 12 bytes out of
/// v2's reserved block, keeping the same 88-byte, tail-padding-free total.
#[repr(C)]
#[derive(Clone, Copy)]
struct MetaPageV3Raw {
    version: u32,
    mode: u8,
    metric: u8,
    is_flat: u8,
    filter_kind: u8,
    heap_attnum: i32,
    dim: u32,
    n_lists: u32,
    /// As [`MetaPageV2Raw::code_bits`].
    code_bits: u32,
    rescore_multiplier: u32,
    small_n_threshold: u32,
    calib_nprobe: u32,
    total_vectors: u32,
    seed: u64,
    centroid_dir_start: u32,
    /// The filter column's HEAP attribute number (index key column 2).
    filter_heap_attnum: i32,
    /// Block of the single filter-stats page ([`crate::local::stats`]), or
    /// [`INVALID_BLOCK`] for a flat (small-N bootstrap) filtered index,
    /// which has no list pages to mode-select over and therefore no stats.
    stats_start: u32,
    /// See [`MetaPageV2Raw::inserted_since_build`] — same reserved-scavenged
    /// addition (earlier v3 builds wrote this word as zero).
    inserted_since_build: u32,
    /// As [`MetaPageV2Raw::rotation_rounds`].
    rotation_rounds: u32,
    /// As [`MetaPageV2Raw::bq_bits`].
    bq_bits: u32,
    /// As [`MetaPageV2Raw::calib_pool_samples`] and the two words after it.
    calib_pool_samples: u32,
    calib_recall_bits: u32,
    target_recall_bits: u32,
    /// As [`MetaPageV2Raw::format_version`], at the same offset.
    format_version: u32,
}

const _: () = {
    assert!(std::mem::offset_of!(MetaPageV2Raw, format_version) == FORMAT_OFFSET);
    assert!(std::mem::offset_of!(MetaPageV3Raw, format_version) == FORMAT_OFFSET);
    assert!(std::mem::size_of::<MetaPageV2Raw>() == FORMAT_OFFSET + 4);
    assert!(std::mem::size_of::<MetaPageV3Raw>() == FORMAT_OFFSET + 4);
};

/// The parsed, mode-aware view [`read`] hands back — what every local-mode
/// (and remote-mode, minimally) code path actually works with. `Copy` (every
/// field already is) so [`read`]'s `rd_amcache` fast path can return a plain
/// value copy instead of a borrow tied to the cache's lifetime. Also the
/// argument [`write`]/[`write_init_fork_direct`] take — one struct, one
/// field list, no 14-positional-argument drift between writers and readers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MetaPage {
    pub(crate) mode: Mode,
    pub(crate) metric: Metric,
    pub(crate) is_flat: bool,
    pub(crate) heap_attnum: i16,
    pub(crate) dim: u32,
    pub(crate) n_lists: u32,
    /// The index's on-disk format word as read (1 for a 0.9 page); a writer's value is ignored:
    /// every build stamps [`FORMAT_VERSION`].
    pub(crate) format: u32,
    /// The list page layout within the format (the format word's second byte; module docs).
    pub(crate) layout: u32,
    /// RaBitQ codec of a format-3 native index: `rotation_rounds` (4), `bq_bits` (4) and
    /// `code_bits` (`D`). Zero in a remote-mode page; below format 3, `code_bits` is the old
    /// FJLT width and the other two are zero.
    pub(crate) rotation_rounds: u32,
    pub(crate) bq_bits: u32,
    pub(crate) code_bits: u32,
    /// The `rescore_multiplier` reloption as given: `0` = not set.
    pub(crate) rescore_multiplier: u32,
    pub(crate) small_n_threshold: u32,
    /// What the build's calibration found (`local::calibrate`): the probe count, the pool in rows,
    /// the number of sample queries and their recall@10, and the `target_recall` it aimed at. All
    /// zero for a flat index, a table too small to calibrate and a format below 3.
    pub(crate) calib_nprobe: u32,
    pub(crate) calib_pool: u32,
    pub(crate) calib_samples: u32,
    /// Whether a pair reached the target (see `calibrate::MARGIN_SIGMAS`): `false` for a best effort
    /// and when nothing was measured.
    pub(crate) calib_reached: bool,
    pub(crate) calib_recall: f32,
    /// Where the calibration's queries came from; [`QueriesSource::Rows`] when nothing was calibrated.
    pub(crate) calib_queries: QueriesSource,
    pub(crate) target_recall: f32,
    /// Recorded for future introspection/drift-tracking and reported by the
    /// flat VACUUM path.
    pub(crate) total_vectors: u32,
    pub(crate) seed: u64,
    pub(crate) centroid_dir_start: pg_sys::BlockNumber,
    /// [`FilterKind::None`] for an unfiltered index (v1/v2 metapages always
    /// read back as `None`).
    pub(crate) filter_kind: FilterKind,
    pub(crate) filter_heap_attnum: i16,
    pub(crate) stats_start: pg_sys::BlockNumber,
    /// Rows appended by local `aminsert` since the last `ambuild`.
    /// **Advisory through this struct**: the `rd_amcache`d copy is frozen at
    /// cache-fill time and inserts never invalidate the relcache, so readers
    /// needing the current value ([`crate::local::scan`]'s mode selection,
    /// the flat-mode transition warning) must go through
    /// [`read_inserted_count_fresh`]/[`bump_inserted_on_page`]'s return
    /// value instead.
    pub(crate) inserted_since_build: u32,
}

impl MetaPage {
    pub(crate) fn has_filter(&self) -> bool {
        self.filter_kind != FilterKind::None
    }

    /// A native index (`Local`, or `Promoting`, which still serves local scans) built below
    /// [`FORMAT_VERSION`]: its list pages hold codes this build cannot score, so it is offline until
    /// REINDEX. A remote-mode page carries no codes and never needs it (module docs).
    pub(crate) fn needs_reindex(&self) -> bool {
        format_needs_reindex(self.stored(), Some(self.mode))
    }

    pub(crate) fn stored(&self) -> Stored {
        Stored {
            format: self.format,
            layout: self.layout,
        }
    }

    /// `D / 8`: bytes of one stored code.
    pub(crate) fn code_bytes(&self) -> usize {
        super::pages::code_bytes(self.dim as usize)
    }
}

/// Where a build's calibration queries came from (`calibration_queries`; two bits of the first
/// calibration word).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QueriesSource {
    /// Rows of the indexed table (the default).
    Rows = 0,
    /// A sample of the table named by `calibration_queries`.
    Table = 1,
    /// `calibration_queries` was set but could not be read at a rebuild: rows were used.
    Fallback = 2,
}

impl QueriesSource {
    fn from_bits(bits: u32) -> QueriesSource {
        match bits {
            1 => QueriesSource::Table,
            2 => QueriesSource::Fallback,
            _ => QueriesSource::Rows,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            QueriesSource::Rows => "rows",
            QueriesSource::Table => "table",
            QueriesSource::Fallback => "fallback",
        }
    }
}

const SAMPLES_SHIFT: u32 = 16;
/// The sample count's width: [`QUERIES_SHIFT`] sits above it (a sample is at most
/// `calibrate::MAX_SAMPLES`, 200).
const SAMPLES_MASK: u32 = 0x1FFF;
const QUERIES_SHIFT: u32 = 29;
const NOT_REACHED_BIT: u32 = 1 << 31;
const _: () = assert!(super::calibrate::MAX_SAMPLES as u32 <= SAMPLES_MASK);

/// The calibration's words as the metapage stores them: the pool in the low 16 bits of the first,
/// the sample count in the next 13, the queries source in the two above them (zero, rows, until
/// [`with_queries_source`]) and a set top bit for a target not reached, then the sample recall and the
/// target as `f32` bits.
pub(crate) fn pack_calibration(
    pool: u32,
    samples: u32,
    reached: bool,
    recall: f32,
    target: f32,
) -> [u32; 3] {
    debug_assert!(pool <= 0xFFFF && samples <= SAMPLES_MASK);
    [
        pool.min(0xFFFF)
            | samples.min(SAMPLES_MASK) << SAMPLES_SHIFT
            | if !reached && samples > 0 {
                NOT_REACHED_BIT
            } else {
                0
            },
        recall.to_bits(),
        target.to_bits(),
    ]
}

/// `words` with the queries `source` set.
pub(crate) fn with_queries_source(mut words: [u32; 3], source: QueriesSource) -> [u32; 3] {
    words[0] = words[0] & !(0b11 << QUERIES_SHIFT) | (source as u32) << QUERIES_SHIFT;
    words
}

/// [`pack_calibration`] backwards: `(pool, samples, reached, recall, target)`; `reached` is false
/// when nothing was measured.
pub(crate) fn unpack_calibration(words: [u32; 3]) -> (u32, u32, bool, f32, f32) {
    let samples = words[0] >> SAMPLES_SHIFT & SAMPLES_MASK;
    (
        words[0] & 0xFFFF,
        samples,
        samples > 0 && words[0] & NOT_REACHED_BIT == 0,
        f32::from_bits(words[1]),
        f32::from_bits(words[2]),
    )
}

/// The queries source of a page of this `format`: [`QueriesSource::Rows`] below 3, where the bits
/// were never written.
fn queries_source_in(format: u32, words: [u32; 3]) -> QueriesSource {
    if format < 3 {
        QueriesSource::Rows
    } else {
        QueriesSource::from_bits(words[0] >> QUERIES_SHIFT & 0b11)
    }
}

/// `(nprobe, pool, samples, reached, recall, target)` of a page of this `format`: below 3 there is
/// none, whatever the slot 0.9.0 kept its probe count in holds; neither is a probe count in that slot
/// of a format-3 page that recorded no sample (a build of the unreleased format 3 kept its default there).
fn calibration_of(
    format: u32,
    nprobe_slot: u32,
    words: [u32; 3],
) -> (u32, u32, u32, bool, f32, f32) {
    if format < 3 {
        return (0, 0, 0, false, 0.0, 0.0);
    }
    let (pool, samples, reached, recall, target) = unpack_calibration(words);
    let nprobe = if samples == 0 { 0 } else { nprobe_slot };
    (nprobe, pool, samples, reached, recall, target)
}

impl MetaPage {
    /// Whether a build calibrated the defaults (a flat index, a table below
    /// [`super::calibrate::MIN_ROWS`] and a page below format 3 have none).
    pub(crate) fn calibrated(&self) -> bool {
        self.calib_samples > 0 && self.calib_nprobe > 0
    }

    /// Lists an unfiltered scan probes when no session setting says otherwise; `0` for a flat index.
    pub(crate) fn default_nprobe(&self) -> u32 {
        if self.n_lists == 0 {
            0
        } else if self.calibrated() {
            self.calib_nprobe
        } else {
            super::options::legacy_nprobe(self.n_lists)
        }
    }

    /// Rows an unfiltered scan rescores when no session setting says otherwise: the reloption, else
    /// the calibrated pool, else the 0.9.0 pool.
    pub(crate) fn default_pool(&self) -> u32 {
        if self.rescore_multiplier > 0 {
            self.rescore_multiplier * super::scan::ASSUMED_TOP_K as u32
        } else if self.calibrated() && self.calib_pool > 0 {
            self.calib_pool
        } else {
            super::options::FALLBACK_POOL
        }
    }
}

/// What `pinecone.index_calibration` reports. `nprobe` and `pool` are the defaults a scan uses (`None`
/// where none apply: a flat index, a remote one, a native one below format 3); the `calib_` fields are
/// what the build measured, `None` and a zero sample count when it measured nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CalibrationReport {
    pub(crate) nprobe: Option<i32>,
    pub(crate) pool: Option<i32>,
    pub(crate) calib_nprobe: Option<i32>,
    pub(crate) calib_pool: Option<i32>,
    pub(crate) calib_recall: Option<f32>,
    pub(crate) calib_samples: i32,
    pub(crate) target_recall: Option<f32>,
    pub(crate) calib_reached: Option<bool>,
    /// `rows`, `table` or `fallback` (see [`QueriesSource`]); `None` when nothing was measured.
    pub(crate) queries_source: Option<&'static str>,
    /// Queries drawn from the `calibration_queries` table: the sample when `queries_source` is
    /// `table`, else 0.
    pub(crate) queries_count: i32,
}

impl MetaPage {
    pub(crate) fn calibration_report(&self) -> CalibrationReport {
        let serves_defaults = self.mode.is_native() && !self.is_flat && !self.needs_reindex();
        let measured = serves_defaults && self.calibrated();
        CalibrationReport {
            nprobe: serves_defaults.then(|| self.default_nprobe() as i32),
            pool: serves_defaults.then(|| self.default_pool() as i32),
            calib_nprobe: measured.then_some(self.calib_nprobe as i32),
            calib_pool: measured.then_some(self.calib_pool as i32),
            calib_recall: measured.then_some(self.calib_recall),
            calib_samples: if measured {
                self.calib_samples as i32
            } else {
                0
            },
            target_recall: (serves_defaults && self.target_recall > 0.0)
                .then_some(self.target_recall),
            calib_reached: measured.then_some(self.calib_reached),
            queries_source: measured.then_some(self.calib_queries.as_str()),
            queries_count: if measured && self.calib_queries == QueriesSource::Table {
                self.calib_samples as i32
            } else {
                0
            },
        }
    }
}

/// The codec fields of a format-3 native index over `dim` dimensions: `(rotation_rounds, bq_bits,
/// code_bits)`.
pub(crate) fn native_codec(dim: u32) -> (u32, u32, u32) {
    (
        super::rotation::ROUNDS as u32,
        super::rabitq::QUERY_BITS,
        super::pages::code_bits(dim as usize) as u32,
    )
}

/// Whether an index of this `format` and `mode` is a native one below the current format. An
/// unknown mode byte is not classifiable (`None`) and counts as needing it.
fn format_needs_reindex(stored: Stored, mode: Option<Mode>) -> bool {
    stored.is_older() && mode.is_none_or(Mode::is_native)
}

/// The `rd_amcache` payload: the parsed metapage plus, once a clustered scan
/// has read it, a flat copy of the centroid directory (at 1M x
/// 768d the directory is ~3MB across ~375 pages, and `ivf_scan` used to
/// re-read and re-parse ALL of it — page locks, per-entry `Vec` allocations
/// — on EVERY query; it is immutable for a given built index, so one flat
/// copy per relcache lifetime serves every scan).
///
/// Layout: this struct, then `dir_entries` packed entries of
/// `dim * 4` centroid-f32 bytes followed by a `u32` list-head block number
/// (stride `dim * 4 + 4`), then, at offset `n_lists * stride` after the header, `dir_entries`
/// rotated centroids `P^T c` of `D` f32 each (the RaBitQ rotation of the entry's centroid, filled
/// with the directory and gone with it). `repr(C)` with `meta` first so the whole payload
/// is reachable from the single `rd_amcache` pointer; the trailing bytes'
/// base is `size_of::<AmCache>()`-aligned (a multiple of 8 — `MetaPage`
/// holds a `u64`), which keeps every entry's f32 span 4-byte aligned (both regions' strides are
/// multiples of 4).
///
/// Staleness safety is exactly [`read`]'s own argument: the directory can
/// only change through a full rebuild (list-page VACUUM sweeps rewrite
/// records in place, never centroids or list heads; local `aminsert`
/// appends to existing list chains and NEVER touches the directory, by
/// design: see `local::insert`'s module docstring for the empty-list rule
/// that keeps this true), and every rebuild swaps the relfilenode, which
/// invalidates the relcache entry, which resets `rd_amcache` to NULL. The
/// one metapage field inserts DO change (`inserted_since_build`) is
/// explicitly advisory through this cache — see its field docs. The rotated centroids depend on
/// nothing but the seed and the directory, so the same argument covers them: a reseed is a rebuild.
#[repr(C)]
pub(crate) struct AmCache {
    pub(crate) meta: MetaPage,
    /// Entries in the flat directory payload trailing this struct
    /// (`0` = directory not cached yet — [`read`] always starts it at 0;
    /// `scan::cached_directory` fills it on the first clustered scan).
    pub(crate) dir_entries: u32,
}

/// Reads and validates block 0 of `index`'s `MAIN_FORKNUM` — see this
/// module's docstring for the version-dispatch rules. Cached in `index->
/// rd_amcache` after the first call per relcache lifetime (every
/// `amgettuple`/`aminsert`/`ambulkdelete` call used to re-read and re-lock
/// block 0 from scratch, once per call, for data that
/// never changes without a rebuild). Safe because `rd_amcache` is reset to
/// `NULL` by Postgres itself (`RelationInvalidateRelation`, relcache.c)
/// whenever this relation's relcache entry is invalidated — which a
/// `REINDEX`'s relfilenode swap always triggers — so this can never serve a
/// stale metapage across a rebuild; the same assumption core btree/gin/gist
/// already rely on for their own `rd_amcache` build-time caches.
pub(crate) fn read(index: pg_sys::Relation) -> MetaPage {
    unsafe {
        if !(*index).rd_amcache.is_null() {
            return (*((*index).rd_amcache as *const AmCache)).meta;
        }

        let buffer = pg_sys::ReadBufferExtended(
            index,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            0,
            pg_sys::ReadBufferMode::RBM_NORMAL,
            std::ptr::null_mut(),
        );
        pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_SHARE as i32);
        let page = pg_sys::BufferGetPage(buffer);
        let contents = pg_sys::PageGetContents(page);
        // `read_unaligned` throughout (matching `write`'s `write_unaligned`)
        // rather than a direct dereference: nothing here actually depends on
        // `contents` being 8-byte aligned (it always is, in practice, since
        // `PageHeaderData` is 24 bytes and buffers are MAXALIGN'd), but nor
        // should correctness silently ride on that fact holding forever.
        let version = std::ptr::read_unaligned(contents as *const u32);

        // A newer format is refused before any other field is interpreted.
        let stored = stored_of(contents as *const u8, version).unwrap_or(Stored::from_word(0));
        let format = stored.format;
        if format > FORMAT_VERSION {
            pg_sys::UnlockReleaseBuffer(buffer);
            refuse_format(index, stored);
        }

        let result = match version {
            META_VERSION_V1 => {
                let _v1 = std::ptr::read_unaligned(contents as *const MetaPageV1);
                MetaPage {
                    mode: Mode::Remote,
                    // Never read for a remote-mode index (see `metric`'s own
                    // field docs) — `Metric::L2` is an arbitrary, inert
                    // placeholder, not a semantically meaningful default.
                    metric: Metric::L2,
                    is_flat: false,
                    heap_attnum: 0,
                    dim: 0,
                    n_lists: 0,
                    format,
                    layout: stored.layout,
                    rotation_rounds: 0,
                    bq_bits: 0,
                    code_bits: 0,
                    rescore_multiplier: 0,
                    small_n_threshold: 0,
                    calib_nprobe: 0,
                    calib_pool: 0,
                    calib_samples: 0,
                    calib_reached: false,
                    calib_recall: 0.0,
                    calib_queries: QueriesSource::Rows,
                    target_recall: 0.0,
                    total_vectors: 0,
                    seed: 0,
                    centroid_dir_start: INVALID_BLOCK,
                    filter_kind: FilterKind::None,
                    filter_heap_attnum: 0,
                    stats_start: INVALID_BLOCK,
                    inserted_since_build: 0,
                }
            }
            META_VERSION_V2 => {
                let raw = std::ptr::read_unaligned(contents as *const MetaPageV2Raw);
                let (
                    calib_nprobe,
                    calib_pool,
                    calib_samples,
                    calib_reached,
                    calib_recall,
                    target_recall,
                ) = calibration_of(
                    format,
                    raw.calib_nprobe,
                    [
                        raw.calib_pool_samples,
                        raw.calib_recall_bits,
                        raw.target_recall_bits,
                    ],
                );
                MetaPage {
                    mode: Mode::from_u8(raw.mode),
                    metric: Metric::from_u8(raw.metric),
                    is_flat: raw.is_flat != 0,
                    heap_attnum: raw.heap_attnum as i16,
                    dim: raw.dim,
                    n_lists: raw.n_lists,
                    format,
                    layout: stored.layout,
                    rotation_rounds: raw.rotation_rounds,
                    bq_bits: raw.bq_bits,
                    code_bits: raw.code_bits,
                    rescore_multiplier: raw.rescore_multiplier,
                    small_n_threshold: raw.small_n_threshold,
                    calib_nprobe,
                    calib_pool,
                    calib_samples,
                    calib_reached,
                    calib_recall,
                    calib_queries: queries_source_in(
                        format,
                        [
                            raw.calib_pool_samples,
                            raw.calib_recall_bits,
                            raw.target_recall_bits,
                        ],
                    ),
                    target_recall,
                    total_vectors: raw.total_vectors,
                    seed: raw.seed,
                    centroid_dir_start: raw.centroid_dir_start,
                    filter_kind: FilterKind::None,
                    filter_heap_attnum: 0,
                    stats_start: INVALID_BLOCK,
                    inserted_since_build: raw.inserted_since_build,
                }
            }
            META_VERSION_V3 => {
                let raw = std::ptr::read_unaligned(contents as *const MetaPageV3Raw);
                let (
                    calib_nprobe,
                    calib_pool,
                    calib_samples,
                    calib_reached,
                    calib_recall,
                    target_recall,
                ) = calibration_of(
                    format,
                    raw.calib_nprobe,
                    [
                        raw.calib_pool_samples,
                        raw.calib_recall_bits,
                        raw.target_recall_bits,
                    ],
                );
                MetaPage {
                    mode: Mode::from_u8(raw.mode),
                    metric: Metric::from_u8(raw.metric),
                    is_flat: raw.is_flat != 0,
                    heap_attnum: raw.heap_attnum as i16,
                    dim: raw.dim,
                    n_lists: raw.n_lists,
                    format,
                    layout: stored.layout,
                    rotation_rounds: raw.rotation_rounds,
                    bq_bits: raw.bq_bits,
                    code_bits: raw.code_bits,
                    rescore_multiplier: raw.rescore_multiplier,
                    small_n_threshold: raw.small_n_threshold,
                    calib_nprobe,
                    calib_pool,
                    calib_samples,
                    calib_reached,
                    calib_recall,
                    calib_queries: queries_source_in(
                        format,
                        [
                            raw.calib_pool_samples,
                            raw.calib_recall_bits,
                            raw.target_recall_bits,
                        ],
                    ),
                    target_recall,
                    total_vectors: raw.total_vectors,
                    seed: raw.seed,
                    centroid_dir_start: raw.centroid_dir_start,
                    filter_kind: FilterKind::from_u8(raw.filter_kind),
                    filter_heap_attnum: raw.filter_heap_attnum as i16,
                    stats_start: raw.stats_start,
                    inserted_since_build: raw.inserted_since_build,
                }
            }
            other => {
                pg_sys::UnlockReleaseBuffer(buffer);
                error!(
                    "pg_pinecone: index metapage has unknown version {other} — this index was built \
                     by an incompatible pg_pinecone version; REINDEX it"
                );
            }
        };
        pg_sys::UnlockReleaseBuffer(buffer);
        if result.mode.is_native() && stored.is_newer() {
            refuse_format(index, stored);
        }

        // Defense-in-depth against a corrupt on-disk metapage: every
        // downstream consumer of `dim`/`n_lists`/`code_bits`
        // (`scan::cached_directory`'s allocation and copy loop,
        // `vacuum::read_list_heads`, the page readers' code-byte sizing,
        // etc.) trusts these fields to be exactly what `ambuild`
        // would have written — a bit-flipped or otherwise corrupted page
        // could claim, say, `n_lists = u32::MAX`, which would then drive a
        // wildly out-of-bounds allocation or copy somewhere downstream
        // instead of erroring HERE, at the one place every reader already
        // funnels through. Bounded by the same ceilings the `n_lists`
        // reloption enforces at CREATE INDEX time (see
        // `options::MAX_N_LISTS`'s own rustdoc), by
        // `pages::max_centroid_dimension`'s page-format-derived cap and, for
        // a format-3 native index, by the codec this build writes — a
        // value outside these was never legitimately written by this build,
        // regardless of version.
        validate_untrusted_fields(&result);

        // `rd_indexcxt`: the index relcache entry's own long-lived context,
        // specifically earmarked for exactly this kind of AM-private
        // build-time cache (and, like `rd_amcache` itself, cleared whenever
        // the entry is invalidated — see this function's own rustdoc).
        // Directory starts uncached (`dir_entries = 0`);
        // `scan::cached_directory` upgrades the payload in place on the
        // first clustered scan.
        let cached =
            pg_sys::MemoryContextAlloc((*index).rd_indexcxt, std::mem::size_of::<AmCache>())
                as *mut AmCache;
        std::ptr::write(
            cached,
            AmCache {
                meta: result,
                dir_entries: 0,
            },
        );
        (*index).rd_amcache = cached as *mut std::ffi::c_void;

        result
    }
}

/// [`read`]'s own defense-in-depth bound check — see its call site's
/// comment for why this exists. `dim == 0`/`n_lists == 0` (a v1 remote-mode
/// metapage, or any legitimately degenerate value) trivially pass every
/// bound here, so this never rejects a genuinely-valid metapage of any
/// version.
fn validate_untrusted_fields(meta: &MetaPage) {
    if let Some(problem) = untrusted_field_problem(meta) {
        error!("pg_pinecone: corrupt metapage — {problem}");
    }
}

/// What is wrong with `meta`'s bounded fields, if anything.
fn untrusted_field_problem(meta: &MetaPage) -> Option<String> {
    let max_dim = super::pages::max_centroid_dimension() as u32;
    if meta.dim > max_dim {
        return Some(format!(
            "dim {} exceeds the maximum centroid dimension {max_dim}",
            meta.dim
        ));
    }
    if meta.n_lists > super::options::MAX_N_LISTS {
        return Some(format!(
            "n_lists {} exceeds the maximum {}",
            meta.n_lists,
            super::options::MAX_N_LISTS
        ));
    }
    // A format-3 native index has the codec this build writes: anything else would have the page
    // readers slice codes of the wrong width.
    if meta.format >= 3 && meta.mode.is_native() {
        let want = native_codec(meta.dim);
        let have = (meta.rotation_rounds, meta.bq_bits, meta.code_bits);
        if meta.dim == 0 || have != want {
            return Some(format!(
                "codec fields (rotation_rounds, bq_bits, code_bits) are {have:?} for dim {}, \
                 this build writes {want:?}",
                meta.dim
            ));
        }
    } else if meta.code_bits > super::options::MAX_K_BITS || !meta.code_bits.is_multiple_of(8) {
        return Some(format!(
            "code_bits {} is invalid (a multiple of 8, at most {})",
            meta.code_bits,
            super::options::MAX_K_BITS
        ));
    }
    calibration_problem(meta)
}

/// What is wrong with the calibration words: they are zero together or describe a pair this build
/// could have found (a probe count within the lists, a pool, a recall and a target in `0..=1`).
fn calibration_problem(meta: &MetaPage) -> Option<String> {
    let unit = |x: f32| x.is_finite() && (0.0..=1.0).contains(&x);
    if !unit(meta.calib_recall) || !unit(meta.target_recall) {
        return Some(format!(
            "calibration recall {} or target {} is not within 0..=1",
            meta.calib_recall, meta.target_recall
        ));
    }
    if meta.calib_nprobe > meta.n_lists {
        return Some(format!(
            "calibrated probe count {} exceeds n_lists {}",
            meta.calib_nprobe, meta.n_lists
        ));
    }
    let none = meta.calib_nprobe == 0;
    if none != (meta.calib_pool == 0)
        || none != (meta.calib_samples == 0)
        || (none && (meta.calib_reached || meta.calib_queries != QueriesSource::Rows))
    {
        return Some(format!(
            "calibration (probes {}, pool {}, samples {}) is only partly set",
            meta.calib_nprobe, meta.calib_pool, meta.calib_samples
        ));
    }
    None
}

/// The format and page layout of a metapage whose layout tag is `tag` and payload `contents`: a v1
/// (0.3) page has no format word and is format 1, a v2/v3 page's word is [`Stored::from_word`].
/// `None` for a tag this build does not know.
unsafe fn stored_of(contents: *const u8, tag: u32) -> Option<Stored> {
    match tag {
        META_VERSION_V1 => Some(Stored::from_word(1)),
        META_VERSION_V2 | META_VERSION_V3 => {
            let word =
                unsafe { std::ptr::read_unaligned(contents.add(FORMAT_OFFSET) as *const u32) };
            Some(Stored::from_word(word))
        }
        _ => None,
    }
}

#[cfg(test)]
unsafe fn format_of(contents: *const u8, tag: u32) -> Option<u32> {
    unsafe { stored_of(contents, tag) }.map(|stored| stored.format)
}

/// The relation's name, for a refusal message.
unsafe fn relation_name(index: pg_sys::Relation) -> String {
    unsafe { std::ffi::CStr::from_ptr((*(*index).rd_rel).relname.data.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Raises the `55000` refusal of an index in a format or page layout newer than this build's.
unsafe fn refuse_format(index: pg_sys::Relation, stored: Stored) -> ! {
    let name = unsafe { relation_name(index) };
    let what = if stored.format > FORMAT_VERSION {
        format!(
            "format version {} (this build reads up to {FORMAT_VERSION})",
            stored.format
        )
    } else {
        format!(
            "format version {} with page layout {} (this build reads up to layout {LAYOUT_VERSION})",
            stored.format, stored.layout
        )
    };
    crate::backend::error_with_code(
        PgSqlErrorCode::ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE,
        format!("pg_pinecone index '{name}' has {what}: REINDEX it"),
        None,
        None,
    )
}

/// The refusal text of a native index older than this build reads (format or page layout), as an
/// error and as a warning.
pub(crate) fn old_format_message(name: &str, stored: Stored) -> String {
    if stored.format < FORMAT_VERSION {
        format!(
            "pg_pinecone index '{name}' has format version {} (this build reads \
             {FORMAT_VERSION}): REINDEX it",
            stored.format
        )
    } else {
        format!(
            "pg_pinecone index '{name}' has format version {} with page layout {} (this build reads \
             layout {LAYOUT_VERSION}): REINDEX it",
            stored.format, stored.layout
        )
    }
}

const OLD_FORMAT_HINT: &str = "native indexes built by an older format or page layout are offline \
    until REINDEX; pinecone.reindex_native_indexes() rebuilds them all";

/// Raises `55000` for a native index older than this build reads: its codes are in a layout this
/// build cannot score (scans, inserts and the engine switch refuse it). `stored` is
/// [`MetaPage::stored`].
pub(crate) unsafe fn refuse_old_format(index: pg_sys::Relation, stored: Stored) -> ! {
    let name = unsafe { relation_name(index) };
    crate::backend::error_with_code(
        PgSqlErrorCode::ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE,
        old_format_message(&name, stored),
        None,
        Some(OLD_FORMAT_HINT),
    )
}

/// The same as a `WARNING`, for the callers that skip the index instead of failing (VACUUM, the
/// maintenance readers, the planner).
pub(crate) unsafe fn warn_old_format(index: pg_sys::Relation, stored: Stored) {
    let name = unsafe { relation_name(index) };
    pgrx::pg_sys::panic::ErrorReport::new(
        PgSqlErrorCode::ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE,
        old_format_message(&name, stored),
        pgrx::function_name!(),
    )
    .set_hint(OLD_FORMAT_HINT)
    .report(PgLogLevel::WARNING);
}

/// The stored format and layout of `index` if block 0 says it is a native index older than this
/// build reads, else `None`, without raising (what VACUUM's cleanup pass asks, which must never fail
/// on an index it skips).
pub(crate) unsafe fn stale_native_format(index: pg_sys::Relation) -> Option<Stored> {
    let (stored, mode) = unsafe { peek(index) }?;
    format_needs_reindex(stored, mode).then_some(stored)
}

/// What block 0 of `index` says about its format, read fresh: no `rd_amcache`, no refusal (what
/// `pinecone.index_format` reports): `(stored, mode)`, the mode `None` for a byte this build never
/// wrote. `None`: no block 0, or a layout tag this build does not know.
pub(crate) unsafe fn peek(index: pg_sys::Relation) -> Option<(Stored, Option<Mode>)> {
    unsafe {
        if pg_sys::RelationGetNumberOfBlocksInFork(index, pg_sys::ForkNumber::MAIN_FORKNUM) == 0 {
            return None;
        }
        let buffer = pg_sys::ReadBufferExtended(
            index,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            0,
            pg_sys::ReadBufferMode::RBM_NORMAL,
            std::ptr::null_mut(),
        );
        pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_SHARE as i32);
        let contents = pg_sys::PageGetContents(pg_sys::BufferGetPage(buffer));
        let tag = std::ptr::read_unaligned(contents as *const u32);
        let peeked = stored_of(contents as *const u8, tag).map(|stored| {
            let mode = match tag {
                META_VERSION_V1 => Some(Mode::Remote),
                // The mode byte sits at offset 4 of both layouts.
                _ => Mode::try_from_u8(std::ptr::read_unaligned(contents.cast::<u8>().add(4))),
            };
            (stored, mode)
        });
        pg_sys::UnlockReleaseBuffer(buffer);
        peeked
    }
}

/// `(format_version, readable)` as `pinecone.index_format` reports a [`peek`] result: readable is
/// false for a newer format, for a native index older than this build reads (format or page
/// layout), for a native one of a newer page layout, and for a mode byte this build does not know.
pub(crate) fn describe_format(peeked: Option<(Stored, Option<Mode>)>) -> (Option<i32>, bool) {
    match peeked {
        Some((stored, mode)) => (
            Some(i32::try_from(stored.format).unwrap_or(i32::MAX)),
            stored.format <= FORMAT_VERSION
                && mode.is_some()
                && !format_needs_reindex(stored, mode)
                && !(stored.is_newer() && mode.is_some_and(Mode::is_native)),
        ),
        None => (None, false),
    }
}

/// An upper bound on how many blocks a well-formed page chain (directory or
/// list) inside `index` could possibly have — the relation's own current
/// block count plus one. Used as a hop-count ceiling by every chain walker
/// in `scan`/`vacuum`/`insert`: a
/// corrupt/cyclic `next_block` chain must raise a clear `ERROR`, not spin
/// forever, and no acyclic chain built by this crate can ever legitimately
/// have more links than the relation has blocks.
pub(crate) unsafe fn max_chain_hops(index: pg_sys::Relation) -> u64 {
    unsafe {
        pg_sys::RelationGetNumberOfBlocksInFork(index, pg_sys::ForkNumber::MAIN_FORKNUM) as u64 + 1
    }
}

/// Builds a fresh metapage's raw bytes (`PageInit` + payload) directly into
/// `page` — a pointer to any BLCKSZ-sized page buffer, whether a
/// buffer-cache page ([`write`]) or a bare `palloc`'d region
/// ([`write_init_fork_direct`]). Factored out so both writers construct
/// byte-identical payloads and can't drift. **Version choice lives here**:
/// v2 for an unfiltered `meta` (byte-identical to what earlier builds wrote —
/// the module docstring's compatibility claim), v3 when a filter column
/// exists.
unsafe fn init_meta_page(page: pg_sys::Page, meta: &MetaPage) {
    unsafe {
        pg_sys::PageInit(page, pg_sys::BLCKSZ as usize, 0);
        let contents = pg_sys::PageGetContents(page);
        let header = page as *mut pg_sys::PageHeaderData;
        let calibration = with_queries_source(
            pack_calibration(
                meta.calib_pool,
                meta.calib_samples,
                meta.calib_reached,
                meta.calib_recall,
                meta.target_recall,
            ),
            meta.calib_queries,
        );

        if meta.has_filter() {
            let raw = MetaPageV3Raw {
                version: META_VERSION_V3,
                mode: meta.mode as u8,
                metric: meta.metric.as_u8(),
                is_flat: meta.is_flat as u8,
                filter_kind: meta.filter_kind.as_u8(),
                heap_attnum: meta.heap_attnum as i32,
                dim: meta.dim,
                n_lists: meta.n_lists,
                code_bits: meta.code_bits,
                rescore_multiplier: meta.rescore_multiplier,
                small_n_threshold: meta.small_n_threshold,
                calib_nprobe: meta.calib_nprobe,
                total_vectors: meta.total_vectors,
                seed: meta.seed,
                centroid_dir_start: meta.centroid_dir_start,
                filter_heap_attnum: meta.filter_heap_attnum as i32,
                stats_start: meta.stats_start,
                inserted_since_build: meta.inserted_since_build,
                rotation_rounds: meta.rotation_rounds,
                bq_bits: meta.bq_bits,
                calib_pool_samples: calibration[0],
                calib_recall_bits: calibration[1],
                target_recall_bits: calibration[2],
                format_version: Stored::CURRENT.word(),
            };
            std::ptr::write_unaligned(contents as *mut MetaPageV3Raw, raw);
            (*header).pd_lower += std::mem::size_of::<MetaPageV3Raw>() as u16;
        } else {
            let raw = MetaPageV2Raw {
                version: META_VERSION_V2,
                mode: meta.mode as u8,
                metric: meta.metric.as_u8(),
                is_flat: meta.is_flat as u8,
                _pad0: 0,
                heap_attnum: meta.heap_attnum as i32,
                dim: meta.dim,
                n_lists: meta.n_lists,
                code_bits: meta.code_bits,
                rescore_multiplier: meta.rescore_multiplier,
                small_n_threshold: meta.small_n_threshold,
                calib_nprobe: meta.calib_nprobe,
                total_vectors: meta.total_vectors,
                seed: meta.seed,
                centroid_dir_start: meta.centroid_dir_start,
                inserted_since_build: meta.inserted_since_build,
                rotation_rounds: meta.rotation_rounds,
                bq_bits: meta.bq_bits,
                calib_pool_samples: calibration[0],
                calib_recall_bits: calibration[1],
                target_recall_bits: calibration[2],
                _reserved: [0; 2],
                format_version: Stored::CURRENT.word(),
            };
            std::ptr::write_unaligned(contents as *mut MetaPageV2Raw, raw);
            (*header).pd_lower += std::mem::size_of::<MetaPageV2Raw>() as u16;
        }
    }
}

/// Writes a fresh metapage to block 0 of `index`'s `MAIN_FORKNUM` — the
/// first page any local- or remote-mode `ambuild` writes (see `am.rs`'s
/// `write_metapage`'s old rustdoc for the `GenericXLogStart -> RegisterBuffer
/// -> PageInit -> GenericXLogFinish` sequence this follows, contrib/bloom's
/// own `BloomInitMetapage` pattern). Always allocates via `P_NEW`, so callers
/// must invoke this as the *first* page write of a fresh build — see
/// `local::build`'s module docstring for why that determinism (metapage =
/// block 0, later pages in write order) matters.
///
/// **`MAIN_FORKNUM` only** — `INIT_FORKNUM` (unlogged-relation reset images)
/// must go through [`write_init_fork_direct`] instead, never this function.
/// See that function's docstring for why: routing an `INIT_FORKNUM` write
/// through `GenericXLog` silently drops its WAL record for any unlogged
/// relation, which is exactly the fork whose durability an unlogged table's
/// crash-reset semantics depend on.
pub(crate) unsafe fn write(index: pg_sys::Relation, meta: &MetaPage) {
    unsafe {
        let buffer = pg_sys::ReadBufferExtended(
            index,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            P_NEW,
            pg_sys::ReadBufferMode::RBM_NORMAL,
            std::ptr::null_mut(),
        );
        pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);

        let state = pg_sys::GenericXLogStart(index);
        let page = pg_sys::GenericXLogRegisterBuffer(
            state,
            buffer,
            pg_sys::GENERIC_XLOG_FULL_IMAGE as i32,
        );
        init_meta_page(page, meta);

        pg_sys::GenericXLogFinish(state);
        pg_sys::UnlockReleaseBuffer(buffer);
    }
}

/// Atomically flips an EXISTING metapage's `mode` byte to `new_mode` — the
/// mechanics behind `pinecone.promote_index`/`demote_index`: local ->
/// promoting, promoting -> remote (worker, on backfill exhaustion), remote ->
/// demoting, demoting -> local (worker, after the rebuild `REINDEX`
/// commits — though in practice that REINDEX's own `ambuild` writes a whole
/// fresh metapage via [`write`] rather than reaching this function at all;
/// this function's real callers are the two flips that do NOT go through a
/// full rebuild). ONE `GenericXLog` record, delta mode (only the mode byte
/// changes) — the crash-drill invariant this exists to make true: a crash at
/// any point during this call either replays the OLD mode (record never
/// flushed) or the NEW one (replayed), never a torn mix, since a single WAL
/// record is atomic by construction.
///
/// Does **not** itself verify the on-disk mode was some particular "expected"
/// value before flipping — callers are responsible for that (via a
/// [`read`]/`rd_amcache`-fresh check under whatever serialization they already
/// hold, e.g. an advisory lock on the index's own heap relid — see
/// `lifecycle::promote_index`/`demote_index`), which keeps this primitive
/// itself simple and side-effect-free beyond the one byte it's asked to
/// write.
///
/// **Known gap, documented rather than solved**: for an
/// `UNLOGGED` index relation, this updates `MAIN_FORKNUM` only — the
/// `INIT_FORKNUM` reset image `ambuildempty`/[`write_init_fork_direct`] wrote
/// at CREATE INDEX time still shows whatever mode was current THEN. A crash
/// restart's `ResetUnloggedRelations` recopies that stale image over
/// `MAIN_FORKNUM`, silently reverting an unlogged pg_pinecone index's lifecycle
/// state to what it was at build time. `lifecycle::promote_index`/
/// `demote_index` warn (not refuse) on an unlogged target index for exactly
/// this reason; a full fix would extend this function to also
/// read-modify-write `INIT_FORKNUM` when `relpersistence` is unlogged,
/// mirroring [`write_init_fork_direct`]'s own reasoning — left for a future
/// pass since UNLOGGED + promote_index/demote_index is expected to be rare.
pub(crate) unsafe fn flip_mode(index: pg_sys::Relation, new_mode: Mode) {
    unsafe {
        let buffer = pg_sys::ReadBufferExtended(
            index,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            0,
            pg_sys::ReadBufferMode::RBM_NORMAL,
            std::ptr::null_mut(),
        );
        pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);

        let state = pg_sys::GenericXLogStart(index);
        // Flag `0` (delta mode), matching `bump_inserted_on_page`: only the
        // mode byte changes, so a full-image copy would be wasted WAL.
        let page = pg_sys::GenericXLogRegisterBuffer(state, buffer, 0);
        let contents = pg_sys::PageGetContents(page);
        let version = std::ptr::read_unaligned(contents as *const u32);
        match version {
            META_VERSION_V2 => {
                let mut raw = std::ptr::read_unaligned(contents as *const MetaPageV2Raw);
                raw.mode = new_mode as u8;
                std::ptr::write_unaligned(contents as *mut MetaPageV2Raw, raw);
            }
            META_VERSION_V3 => {
                let mut raw = std::ptr::read_unaligned(contents as *const MetaPageV3Raw);
                raw.mode = new_mode as u8;
                std::ptr::write_unaligned(contents as *mut MetaPageV3Raw, raw);
            }
            other => {
                pg_sys::UnlockReleaseBuffer(buffer);
                error!(
                    "pg_pinecone: cannot flip mode on metapage version {other} — should be unreachable"
                );
            }
        }
        pg_sys::GenericXLogFinish(state);
        pg_sys::UnlockReleaseBuffer(buffer);

        // Keep `rd_amcache` in sync — unlike `inserted_since_build`
        // (deliberately left stale through the cache; see
        // `MetaPage::inserted_since_build`'s own field docs for why that one
        // specific field is advisory-only), `mode` drives EVERY scan/insert/
        // vacuum dispatch decision in `am.rs`, all of which go through
        // [`read`]'s cached fast path. Postgres's relcache is shared by every
        // handle to the same relation oid within one backend (a fresh
        // `RelationIdGetRelation`/`PgRelation::with_lock` call on the SAME
        // oid returns the SAME underlying struct, `rd_amcache` included) and
        // is *not* invalidated by an ordinary page write — only a relfilenode
        // swap (REINDEX) or an unrelated cache-invalidating operation resets
        // it. Without this, this backend's own very next `aminsert`/
        // `ambeginscan` on this index (and any later call in the SAME
        // backend, indefinitely — confirmed by hand: a stale read survived
        // until process exit in earlier testing of this function) would keep
        // dispatching against the OLD mode despite the on-disk byte already
        // being correct. `AmCache` places `meta` as its first field
        // specifically so this cast is exactly the same value [`read`]'s own
        // fast path dereferences.
        if !(*index).rd_amcache.is_null() {
            (*((*index).rd_amcache as *mut AmCache)).meta.mode = new_mode;
        }

        // Cross-backend half: the in-place
        // patch above only fixes THIS backend's cached copy — every OTHER
        // backend that has ever scanned this index holds its own
        // `rd_amcache`d mode, and nothing about a GenericXLog page write
        // invalidates relcaches. Without this, a backend that scanned before
        // a promote finished keeps dispatching on the STALE mode
        // indefinitely (it never re-reads block 0) — serving local scans
        // against an index whose contract is now remote, or vice versa.
        // `CacheInvalidateRelcacheByRelid` registers a relcache invalidation
        // in the current transaction, broadcast at COMMIT — exactly the
        // right delivery point: the mode flip's WAL record and its
        // registry-row companion commit together, and other backends process
        // the queued invalidation at their next command boundary, resetting
        // `rd_amcache` so their next dispatch re-reads the flipped metapage.
        // (Locally it also fires at the next CommandCounterIncrement, which
        // is harmless — the in-place patch above already covers same-command
        // reads, and a post-CCI re-read just reloads the same new mode.)
        pg_sys::CacheInvalidateRelcacheByRelid((*index).rd_id);
    }
}

/// Writes a fresh metapage to block 0 of `index`'s `INIT_FORKNUM` —
/// `ambuildempty`'s only caller, for exactly one reason: **`GenericXLog`
/// (i.e. [`write`]) must never touch `INIT_FORKNUM`.**
///
/// `GenericXLogFinish` decides whether to emit a WAL record at all via
/// `RelationNeedsWAL(relation)`, which checks only `relpersistence` — it has
/// no idea which *fork* is being written. For an `UNLOGGED` relation (the
/// only relation `ambuildempty` is ever called for — see this module's
/// caller), `RelationNeedsWAL` is unconditionally `false`, so a
/// `GenericXLog`'d `INIT_FORKNUM` write silently skips WAL entirely. That's
/// correct — desired, even — for `MAIN_FORKNUM` (an unlogged relation's main
/// fork is never replayed after a crash; `ResetUnloggedRelations` just
/// recopies `INIT_FORKNUM` over it instead). It's catastrophic for
/// `INIT_FORKNUM` itself: that recopy is the *only* mechanism restoring an
/// unlogged index after a crash, so `INIT_FORKNUM`'s own bytes must survive
/// the crash regardless of the relation's persistence — the buffer manager
/// alone (no WAL, no guaranteed flush before a crash) cannot promise that.
/// Skip the WAL record here and a crash between `CREATE INDEX ... UNLOGGED`
/// and the next checkpoint leaves `INIT_FORKNUM` as a zero/garbage page;
/// `ResetUnloggedRelations` then copies that over `MAIN_FORKNUM` at the next
/// restart, and every subsequent `metapage::read` on the reset index `ERROR`s
/// on an unrecognized version tag forever, with no `REINDEX` able to fix it
/// (the corruption happens *before* any session gets a chance to).
///
/// The fix goes through the ORDINARY buffer manager (unlike the old-style
/// `smgrextend`/`smgrimmedsync` bypass contrib/bloom and core btree used
/// before PG17's `smgr_bulk_write` API existed) but forces a WAL record
/// despite `RelationNeedsWAL(index)` being false — matching core GIN's own
/// `ginbuildempty` exactly (`src/backend/access/gin/gininsert.c`, still
/// written this way in this same PG17 tree; PG17's `smgr_bulk_write` API
/// isn't in pgrx 0.19's binding allowlist, so this follows the still-current
/// non-bulk reference implementation instead of adding a custom cshim for a
/// single page): `ExtendBufferedRel` allocates the new page already
/// exclusively locked (`EB_LOCK_FIRST`; `EB_SKIP_EXTENSION_LOCK` because
/// `ambuildempty` already runs under `CREATE INDEX`'s own exclusive lock on
/// the relation, so no concurrent extender could conflict), `PageInit` +
/// the payload write happen directly in the real shared buffer (no
/// GenericXLog shadow copy involved), then `MarkBufferDirty` +
/// `log_newpage_buffer` (which, unlike `GenericXLogFinish`, has no
/// `RelationNeedsWAL` gate at all — it unconditionally inserts a WAL record
/// for whatever the buffer currently holds) inside a critical section
/// (`log_newpage_buffer` itself `Assert`s `CritSectionCount > 0` — this
/// crashed the backend outright when first tried without one, confirming
/// the assert is load-bearing, not decorative). No manual `smgrimmedsync`/
/// checksum call is needed: a normal dirty shared buffer is exactly what
/// the checkpointer/eviction path already knows how to flush (computing the
/// page checksum then, same as any other page in the buffer pool) — the
/// raw-`smgr` bypass's very reason to `smgrimmedsync` (the write never
/// touched shared_buffers, so nothing else would ever flush it) doesn't
/// apply here.
pub(crate) unsafe fn write_init_fork_direct(index: pg_sys::Relation, meta: &MetaPage) {
    unsafe {
        let bmr = pg_sys::BufferManagerRelation {
            rel: index,
            smgr: std::ptr::null_mut(),
            relpersistence: 0,
        };
        let buffer = pg_sys::ExtendBufferedRel(
            bmr,
            pg_sys::ForkNumber::INIT_FORKNUM,
            std::ptr::null_mut(),
            pg_sys::ExtendBufferedFlags::EB_LOCK_FIRST
                | pg_sys::ExtendBufferedFlags::EB_SKIP_EXTENSION_LOCK,
        );
        let page = pg_sys::BufferGetPage(buffer);
        init_meta_page(page, meta);

        // Matches `ginbuildempty`'s own `START_CRIT_SECTION()`/
        // `END_CRIT_SECTION()` pair — no callable wrapper for those macros
        // exists in pgrx's bindings, so this pokes the same backend global
        // they do directly (both macros ultimately just increment/decrement
        // `CritSectionCount`; see miscadmin.h).
        pg_sys::CritSectionCount += 1;
        pg_sys::MarkBufferDirty(buffer);
        pg_sys::log_newpage_buffer(buffer, true);
        pg_sys::CritSectionCount -= 1;

        pg_sys::UnlockReleaseBuffer(buffer);
    }
}

/// Usable bytes per page after `PageInit(page, BLCKSZ, 0)` — every page
/// layout (metapage, centroid directory, stats, list pages) budgets against
/// this.
pub(crate) fn usable_page_bytes() -> usize {
    pg_sys::BLCKSZ as usize - std::mem::size_of::<pg_sys::PageHeaderData>()
}

/// TEST-ONLY (`#[cfg(feature = "pg_test")]`): overwrites an UNFILTERED
/// (v2-format) metapage's on-disk `n_lists` field directly — bypassing
/// `ambuild`/the `n_lists` reloption's own bound entirely — to craft the
/// exact corruption [`validate_untrusted_fields`] exists to catch.
/// `std::mem::offset_of!` against the real raw struct (rather than a
/// hand-computed byte offset) so this can't silently drift from the actual
/// on-disk layout if [`MetaPageV2Raw`] ever changes.
#[cfg(feature = "pg_test")]
pub(crate) unsafe fn test_corrupt_n_lists(index: pg_sys::Relation, value: u32) {
    unsafe {
        let offset = std::mem::offset_of!(MetaPageV2Raw, n_lists);
        let buffer = pg_sys::ReadBufferExtended(
            index,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            0,
            pg_sys::ReadBufferMode::RBM_NORMAL,
            std::ptr::null_mut(),
        );
        pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
        let state = pg_sys::GenericXLogStart(index);
        let page = pg_sys::GenericXLogRegisterBuffer(state, buffer, 0);
        let contents = pg_sys::PageGetContents(page) as *mut u8;
        std::ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), contents.add(offset), 4);
        pg_sys::GenericXLogFinish(state);
        pg_sys::UnlockReleaseBuffer(buffer);

        pg_sys::CacheInvalidateRelcacheByRelid((*index).rd_id);
    }
}

/// TEST-ONLY (`#[cfg(feature = "pg_test")]`): overwrites the `u32` at `offset` of the metapage payload
/// (`FORMAT_OFFSET` for the format word, 0 for the layout tag) and invalidates the relcache so the
/// next read sees it.
#[cfg(feature = "pg_test")]
pub(crate) unsafe fn test_write_u32(index: pg_sys::Relation, offset: usize, value: u32) {
    unsafe {
        let buffer = pg_sys::ReadBufferExtended(
            index,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            0,
            pg_sys::ReadBufferMode::RBM_NORMAL,
            std::ptr::null_mut(),
        );
        pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
        let state = pg_sys::GenericXLogStart(index);
        let page = pg_sys::GenericXLogRegisterBuffer(state, buffer, 0);
        let contents = pg_sys::PageGetContents(page) as *mut u8;
        std::ptr::copy_nonoverlapping(value.to_ne_bytes().as_ptr(), contents.add(offset), 4);
        pg_sys::GenericXLogFinish(state);
        pg_sys::UnlockReleaseBuffer(buffer);

        pg_sys::CacheInvalidateRelcacheByRelid((*index).rd_id);
    }
}

// ---------------------------------------------------------------------------
// inserted_since_build maintenance (local aminsert's staleness counter)
// ---------------------------------------------------------------------------

/// Increments `inserted_since_build` directly on `page` — a **GenericXLog
/// working copy** the caller has already registered (flag `0`, delta mode:
/// only this one word changes) while holding the metapage buffer's exclusive
/// lock. Read-modify-write on the page bytes themselves, NEVER through the
/// `rd_amcache`d [`MetaPage`]: the cache is per-backend and frozen at fill
/// time, so counting through it would silently collapse any number of
/// inserts into "cached value + 1". The exclusive buffer lock is what makes
/// concurrent inserters' increments serialize correctly. Returns the NEW
/// count (the caller's flat-mode transition warning needs it).
///
/// Version-dispatched like [`read`]: v2 and v3 store the field at different
/// offsets (the raw structs above), and a v1 page can never reach here (v1
/// maps to remote mode forever, and only local-mode `aminsert` bumps this).
pub(crate) unsafe fn bump_inserted_on_page(page: pg_sys::Page) -> u32 {
    unsafe {
        let contents = pg_sys::PageGetContents(page);
        let version = std::ptr::read_unaligned(contents as *const u32);
        match version {
            META_VERSION_V2 => {
                let mut raw = std::ptr::read_unaligned(contents as *const MetaPageV2Raw);
                raw.inserted_since_build = raw.inserted_since_build.saturating_add(1);
                std::ptr::write_unaligned(contents as *mut MetaPageV2Raw, raw);
                raw.inserted_since_build
            }
            META_VERSION_V3 => {
                let mut raw = std::ptr::read_unaligned(contents as *const MetaPageV3Raw);
                raw.inserted_since_build = raw.inserted_since_build.saturating_add(1);
                std::ptr::write_unaligned(contents as *mut MetaPageV3Raw, raw);
                raw.inserted_since_build
            }
            other => error!(
                "pg_pinecone: cannot bump insert counter on metapage version {other} — \
                 should be unreachable from local-mode aminsert"
            ),
        }
    }
}

/// The CURRENT on-disk `inserted_since_build` — one buffered block-0 read
/// (share lock), deliberately bypassing `rd_amcache` (see
/// [`MetaPage::inserted_since_build`]'s own docs for why the cached copy
/// can't serve this). Costs one page read per call; its only caller is the
/// filtered scan's mode selection (`scan::choose_filtered_mode`), which
/// already pays an uncached stats-page read per filtered scan — unfiltered
/// scans never call this.
pub(crate) unsafe fn read_inserted_count_fresh(index: pg_sys::Relation) -> u32 {
    unsafe {
        let buffer = pg_sys::ReadBufferExtended(
            index,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            0,
            pg_sys::ReadBufferMode::RBM_NORMAL,
            std::ptr::null_mut(),
        );
        pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_SHARE as i32);
        let contents = pg_sys::PageGetContents(pg_sys::BufferGetPage(buffer));
        let version = std::ptr::read_unaligned(contents as *const u32);
        let count = match version {
            META_VERSION_V2 => {
                std::ptr::read_unaligned(contents as *const MetaPageV2Raw).inserted_since_build
            }
            META_VERSION_V3 => {
                std::ptr::read_unaligned(contents as *const MetaPageV3Raw).inserted_since_build
            }
            // v1 (remote-forever) has no counter; anything else would have
            // already errored in `read` before any caller got this far.
            _ => 0,
        };
        pg_sys::UnlockReleaseBuffer(buffer);
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(tag: u32, format_word: u32) -> [u8; FORMAT_OFFSET + 4] {
        let mut page = [0u8; FORMAT_OFFSET + 4];
        page[..4].copy_from_slice(&tag.to_ne_bytes());
        page[FORMAT_OFFSET..].copy_from_slice(&format_word.to_ne_bytes());
        page
    }

    #[test]
    fn a_zero_format_word_reads_as_format_1_for_both_layouts() {
        for tag in [2, 3] {
            for (word, format) in [
                (0, 1),
                (1, 1),
                (2, 2),
                (3, 3),
                (0x103, 3),
                (u32::MAX, u32::MAX),
            ] {
                let page = payload(tag, word);
                assert_eq!(unsafe { format_of(page.as_ptr(), tag) }, Some(format));
            }
        }
    }

    #[test]
    fn a_v1_page_is_format_1_whatever_follows_and_an_unknown_tag_has_none() {
        let page = payload(1, 7);
        assert_eq!(unsafe { format_of(page.as_ptr(), 1) }, Some(1));
        for tag in [0, 4, u32::MAX] {
            let page = payload(tag, 2);
            assert_eq!(unsafe { format_of(page.as_ptr(), tag) }, None);
        }
    }

    #[test]
    fn this_build_writes_format_3() {
        assert_eq!(FORMAT_VERSION, 3);
    }

    fn at(format: u32, layout: u32) -> Stored {
        Stored { format, layout }
    }

    #[test]
    fn only_a_native_index_below_this_format_needs_a_reindex() {
        for mode in [Mode::Local, Mode::Promoting] {
            assert!(format_needs_reindex(at(1, 0), Some(mode)), "{mode:?} at 1");
            assert!(format_needs_reindex(at(2, 0), Some(mode)), "{mode:?} at 2");
            assert!(
                !format_needs_reindex(Stored::CURRENT, Some(mode)),
                "{mode:?} at 3"
            );
        }
        for mode in [Mode::Remote, Mode::Demoting, Mode::Detached] {
            for stored in [at(1, 0), at(2, 0), at(3, 0), Stored::CURRENT] {
                assert!(
                    !format_needs_reindex(stored, Some(mode)),
                    "{mode:?} at {stored:?}"
                );
            }
        }
        assert!(
            format_needs_reindex(at(2, 0), None),
            "an unknown mode is not served"
        );
    }

    #[test]
    fn the_layout_rides_in_the_second_byte_of_the_format_word() {
        assert_eq!(Stored::CURRENT, at(3, LAYOUT_VERSION));
        assert_eq!(Stored::CURRENT.word(), 3 | 1 << 8);
        for (word, format, layout) in [
            (0, 1, 0),
            (1, 1, 0),
            (2, 2, 0),
            (3, 3, 0),
            (0x103, 3, 1),
            (0x203, 3, 2),
            (0x1_0003, 0x1_0003, 0),
            (u32::MAX, u32::MAX, 0),
        ] {
            assert_eq!(Stored::from_word(word), at(format, layout), "{word:#x}");
        }
        for stored in [Stored::CURRENT, at(3, 0), at(2, 0), at(3, 2)] {
            assert_eq!(Stored::from_word(stored.word()), stored);
        }
    }

    #[test]
    fn a_native_index_of_the_pre_block_page_layout_needs_a_reindex_a_newer_one_is_not_readable() {
        let old_layout = at(3, 0);
        assert!(old_layout.is_older() && !old_layout.is_newer());
        for mode in [Mode::Local, Mode::Promoting] {
            assert!(format_needs_reindex(old_layout, Some(mode)), "{mode:?}");
        }
        for mode in [Mode::Remote, Mode::Demoting] {
            assert!(!format_needs_reindex(old_layout, Some(mode)), "{mode:?}");
        }
        assert!(at(3, 2).is_newer() && !at(3, 2).is_older());
        assert!(at(4, 0).is_newer() && !at(3, 1).is_newer() && !at(3, 1).is_older());
        assert!(!meta(Mode::Local, 3, 8).needs_reindex());
        assert!(
            MetaPage {
                layout: 0,
                ..meta(Mode::Local, 3, 8)
            }
            .needs_reindex()
        );
        assert!(
            !MetaPage {
                layout: 0,
                ..meta(Mode::Remote, 3, 8)
            }
            .needs_reindex()
        );
    }

    #[test]
    fn describe_format_reports_native_indexes_below_3_and_newer_ones_unreadable() {
        let native = Some(Mode::Local);
        let remote = Some(Mode::Remote);
        let cur = Stored::CURRENT;
        assert_eq!(describe_format(Some((at(1, 0), native))), (Some(1), false));
        assert_eq!(describe_format(Some((at(2, 0), native))), (Some(2), false));
        assert_eq!(describe_format(Some((cur, native))), (Some(3), true));
        assert_eq!(describe_format(Some((at(1, 0), remote))), (Some(1), true));
        assert_eq!(describe_format(Some((at(2, 0), remote))), (Some(2), true));
        assert_eq!(describe_format(Some((cur, remote))), (Some(3), true));
        assert_eq!(describe_format(Some((at(4, 0), native))), (Some(4), false));
        assert_eq!(describe_format(Some((at(4, 0), remote))), (Some(4), false));
        assert_eq!(describe_format(Some((cur, None))), (Some(3), false));
        assert_eq!(
            describe_format(Some((at(u32::MAX, 0), remote))),
            (Some(i32::MAX), false)
        );
        assert_eq!(describe_format(None), (None, false));
        assert_eq!(describe_format(Some((at(3, 0), native))), (Some(3), false));
        assert_eq!(describe_format(Some((at(3, 2), native))), (Some(3), false));
        assert_eq!(describe_format(Some((at(3, 0), remote))), (Some(3), true));
    }

    #[test]
    fn the_mode_byte_round_trips_and_unknown_bytes_are_none() {
        for mode in [
            Mode::Local,
            Mode::Remote,
            Mode::Promoting,
            Mode::Detached,
            Mode::Demoting,
        ] {
            assert_eq!(
                Mode::try_from_u8(mode as u8).map(|m| m as u8),
                Some(mode as u8)
            );
        }
        assert!(Mode::try_from_u8(5).is_none() && Mode::try_from_u8(255).is_none());
    }

    #[test]
    fn the_codec_fields_sit_in_reserved_space_and_the_old_code_width_slot() {
        // Both layouts keep their size and every old field's offset; `code_bits` is the old
        // `k_bits` slot (offset 20 in both), the new rounds/bq words are the first reserved ones.
        assert_eq!(std::mem::size_of::<MetaPageV2Raw>(), 88);
        assert_eq!(std::mem::size_of::<MetaPageV3Raw>(), 88);
        assert_eq!(std::mem::offset_of!(MetaPageV2Raw, code_bits), 20);
        assert_eq!(std::mem::offset_of!(MetaPageV3Raw, code_bits), 20);
        assert_eq!(
            std::mem::offset_of!(MetaPageV2Raw, inserted_since_build),
            52
        );
        assert_eq!(std::mem::offset_of!(MetaPageV2Raw, rotation_rounds), 56);
        assert_eq!(std::mem::offset_of!(MetaPageV2Raw, bq_bits), 60);
        assert_eq!(
            std::mem::offset_of!(MetaPageV3Raw, inserted_since_build),
            60
        );
        assert_eq!(std::mem::offset_of!(MetaPageV3Raw, rotation_rounds), 64);
        assert_eq!(std::mem::offset_of!(MetaPageV3Raw, bq_bits), 68);
        assert_eq!(
            std::mem::offset_of!(MetaPageV2Raw, format_version),
            FORMAT_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(MetaPageV3Raw, format_version),
            FORMAT_OFFSET
        );
    }

    #[test]
    fn the_calibration_sits_in_the_old_probe_count_slot_and_the_reserved_words() {
        assert_eq!(std::mem::offset_of!(MetaPageV2Raw, calib_nprobe), 32);
        assert_eq!(std::mem::offset_of!(MetaPageV3Raw, calib_nprobe), 32);
        assert_eq!(std::mem::offset_of!(MetaPageV2Raw, calib_pool_samples), 64);
        assert_eq!(std::mem::offset_of!(MetaPageV2Raw, calib_recall_bits), 68);
        assert_eq!(std::mem::offset_of!(MetaPageV2Raw, target_recall_bits), 72);
        assert_eq!(std::mem::offset_of!(MetaPageV3Raw, calib_pool_samples), 72);
        assert_eq!(std::mem::offset_of!(MetaPageV3Raw, calib_recall_bits), 76);
        assert_eq!(std::mem::offset_of!(MetaPageV3Raw, target_recall_bits), 80);
    }

    #[test]
    fn calibration_words_round_trip() {
        for (pool, samples, reached, recall, target) in [
            (100, 50, true, 0.95_f32, 0.95_f32),
            (2000, 200, false, 0.9873, 0.999),
            (65_535, 8_191, true, 1.0, 0.5),
            (300, 133, false, 0.0, 0.0),
        ] {
            let words = pack_calibration(pool, samples, reached, recall, target);
            assert_eq!(
                unpack_calibration(words),
                (pool, samples, reached, recall, target)
            );
        }
        assert_eq!(pack_calibration(100, 50, true, 0.0, 0.0)[0], 100 | 50 << 16);
        assert_eq!(
            pack_calibration(100, 50, false, 0.0, 0.0)[0],
            100 | 50 << 16 | 1 << 31
        );
        assert_eq!(pack_calibration(0, 0, false, 0.0, 0.0), [0, 0, 0]);
        assert_eq!(unpack_calibration([0; 3]), (0, 0, false, 0.0, 0.0));
    }

    #[test]
    fn the_queries_source_rides_in_two_bits_above_the_sample_count() {
        let base = pack_calibration(100, 200, true, 0.97, 0.95);
        assert_eq!(base[0], 100 | 200 << 16, "no source reads as rows");
        assert_eq!(queries_source_in(3, base), QueriesSource::Rows);
        for (source, bits) in [
            (QueriesSource::Rows, 0),
            (QueriesSource::Table, 1),
            (QueriesSource::Fallback, 2),
        ] {
            let words = with_queries_source(base, source);
            assert_eq!(words[0], base[0] | bits << 29, "{source:?}");
            assert_eq!(words[1..], base[1..]);
            assert_eq!(queries_source_in(3, words), source);
            assert_eq!(queries_source_in(2, words), QueriesSource::Rows);
            assert_eq!(unpack_calibration(words), (100, 200, true, 0.97, 0.95));
            assert_eq!(with_queries_source(words, QueriesSource::Rows), base);
        }
        let not_reached = with_queries_source(
            pack_calibration(2000, 8_191, false, 0.9, 0.95),
            QueriesSource::Table,
        );
        assert_eq!(
            unpack_calibration(not_reached),
            (2000, 8_191, false, 0.9, 0.95)
        );
        assert_eq!(queries_source_in(3, not_reached), QueriesSource::Table);
        assert_eq!(
            queries_source_in(3, [0b11 << 29, 0, 0]),
            QueriesSource::Rows,
            "an unknown source reads as rows"
        );
        assert_eq!(
            [
                QueriesSource::Rows,
                QueriesSource::Table,
                QueriesSource::Fallback
            ]
            .map(QueriesSource::as_str),
            ["rows", "table", "fallback"]
        );
    }

    #[test]
    fn a_page_below_format_3_reads_no_calibration_whatever_its_old_probe_slot_holds() {
        let words = pack_calibration(200, 100, true, 0.96, 0.95);
        assert_eq!(calibration_of(1, 127, words), (0, 0, 0, false, 0.0, 0.0));
        assert_eq!(calibration_of(2, 127, words), (0, 0, 0, false, 0.0, 0.0));
        assert_eq!(
            calibration_of(3, 12, words),
            (12, 200, 100, true, 0.96, 0.95)
        );
        assert_eq!(calibration_of(3, 0, [0; 3]), (0, 0, 0, false, 0.0, 0.0));
        let skipped = pack_calibration(0, 0, false, 0.0, 0.95);
        assert_eq!(
            calibration_of(3, 26, skipped),
            (0, 0, 0, false, 0.0, 0.95),
            "a probe count with no sample behind it is not a calibration"
        );
    }

    #[test]
    fn the_defaults_are_the_calibrated_pair_the_reloption_or_the_090_constants() {
        let calibrated = meta(Mode::Local, 3, 128);
        assert!(calibrated.calibrated());
        assert_eq!(
            (calibrated.default_nprobe(), calibrated.default_pool()),
            (5, 200)
        );

        let pinned = MetaPage {
            rescore_multiplier: 5,
            ..calibrated
        };
        assert_eq!((pinned.default_nprobe(), pinned.default_pool()), (5, 500));

        let uncalibrated = MetaPage {
            calib_nprobe: 0,
            calib_pool: 0,
            calib_samples: 0,
            calib_reached: false,
            calib_recall: 0.0,
            ..calibrated
        };
        assert!(!uncalibrated.calibrated());
        assert_eq!(
            uncalibrated.default_nprobe(),
            10,
            "every list of a 10-list index"
        );
        let wide = MetaPage {
            n_lists: 100,
            ..uncalibrated
        };
        assert_eq!(wide.default_nprobe(), 40, "ceil(4 sqrt(100)) lists");
        assert_eq!(uncalibrated.default_pool(), 1000);
        let pinned_uncalibrated = MetaPage {
            rescore_multiplier: 2,
            ..uncalibrated
        };
        assert_eq!(pinned_uncalibrated.default_pool(), 200);

        let flat = MetaPage {
            n_lists: 0,
            ..uncalibrated
        };
        assert_eq!(flat.default_nprobe(), 0);
    }

    #[test]
    fn a_corrupt_calibration_is_refused_and_every_calibration_this_build_writes_passes() {
        let good = meta(Mode::Local, 3, 128);
        assert_eq!(untrusted_field_problem(&good), None);
        let none = MetaPage {
            calib_nprobe: 0,
            calib_pool: 0,
            calib_samples: 0,
            calib_reached: false,
            calib_recall: 0.0,
            ..good
        };
        assert_eq!(untrusted_field_problem(&none), None);
        for tweak in [
            |m: &mut MetaPage| m.calib_nprobe = 11,
            |m: &mut MetaPage| m.calib_pool = 0,
            |m: &mut MetaPage| m.calib_samples = 0,
            |m: &mut MetaPage| m.calib_recall = 1.5,
            |m: &mut MetaPage| m.calib_recall = f32::NAN,
            |m: &mut MetaPage| m.target_recall = -0.1,
            |m: &mut MetaPage| m.target_recall = f32::INFINITY,
        ] {
            let mut bad = good;
            tweak(&mut bad);
            assert!(untrusted_field_problem(&bad).is_some(), "{bad:?}");
        }
        for tweak in [
            |m: &mut MetaPage| m.calib_nprobe = 5,
            |m: &mut MetaPage| m.calib_pool = 100,
            |m: &mut MetaPage| m.calib_samples = 7,
            |m: &mut MetaPage| m.calib_reached = true,
        ] {
            let mut bad = none;
            tweak(&mut bad);
            assert!(untrusted_field_problem(&bad).is_some(), "{bad:?}");
        }
    }

    fn meta(mode: Mode, format: u32, dim: u32) -> MetaPage {
        let (rotation_rounds, bq_bits, code_bits) = native_codec(dim);
        MetaPage {
            mode,
            metric: Metric::L2,
            is_flat: false,
            heap_attnum: 1,
            dim,
            n_lists: 10,
            format,
            layout: if format >= FORMAT_VERSION {
                LAYOUT_VERSION
            } else {
                0
            },
            rotation_rounds,
            bq_bits,
            code_bits,
            rescore_multiplier: 0,
            small_n_threshold: 100,
            calib_nprobe: 5,
            calib_pool: 200,
            calib_samples: 100,
            calib_reached: true,
            calib_recall: 0.96,
            calib_queries: QueriesSource::Rows,
            target_recall: 0.95,
            total_vectors: 1000,
            seed: 7,
            centroid_dir_start: 1,
            filter_kind: FilterKind::None,
            filter_heap_attnum: 0,
            stats_start: INVALID_BLOCK,
            inserted_since_build: 0,
        }
    }

    #[test]
    fn a_format_3_native_page_must_carry_this_builds_codec() {
        for dim in [1, 8, 64, 65, 768, 2037] {
            let good = meta(Mode::Local, 3, dim);
            assert_eq!(untrusted_field_problem(&good), None, "d = {dim}");
            assert_eq!(good.code_bytes() * 8, good.code_bits as usize);
            for tweak in [
                |m: &mut MetaPage| m.rotation_rounds = 3,
                |m: &mut MetaPage| m.bq_bits = 1,
                |m: &mut MetaPage| m.code_bits += 64,
                |m: &mut MetaPage| m.code_bits = 0,
            ] {
                let mut bad = good;
                tweak(&mut bad);
                let problem = untrusted_field_problem(&bad).expect("a codec mismatch is refused");
                assert!(problem.contains("codec fields"), "{problem}");
            }
        }
        let mut no_dim = meta(Mode::Local, 3, 8);
        no_dim.dim = 0;
        no_dim.code_bits = 0;
        assert!(untrusted_field_problem(&no_dim).is_some());
    }

    #[test]
    fn older_and_remote_pages_keep_the_loose_code_width_bound() {
        let mut old = meta(Mode::Local, 2, 128);
        (old.rotation_rounds, old.bq_bits, old.code_bits) = (0, 0, 512);
        assert_eq!(untrusted_field_problem(&old), None);
        old.code_bits = 100;
        assert!(
            untrusted_field_problem(&old).is_some(),
            "not a multiple of 8"
        );
        old.code_bits = 8192;
        assert!(
            untrusted_field_problem(&old).is_some(),
            "above the old maximum"
        );
        let mut remote = meta(Mode::Remote, 3, 16);
        (remote.rotation_rounds, remote.bq_bits, remote.code_bits) = (0, 0, 0);
        assert_eq!(untrusted_field_problem(&remote), None);
    }

    #[test]
    fn dimension_and_list_bounds_still_apply() {
        let mut m = meta(Mode::Local, 3, 8);
        m.n_lists = super::super::options::MAX_N_LISTS + 1;
        assert!(untrusted_field_problem(&m).unwrap().contains("n_lists"));
        let mut m = meta(Mode::Remote, 3, 8);
        m.dim = super::super::pages::max_centroid_dimension() as u32 + 1;
        assert!(untrusted_field_problem(&m).unwrap().contains("dim"));
    }

    #[test]
    fn the_refusal_text_names_the_index_the_format_and_reindex() {
        let text = old_format_message("docs_idx", at(2, 0));
        assert_eq!(
            text,
            "pg_pinecone index 'docs_idx' has format version 2 (this build reads 3): REINDEX it"
        );
        assert_eq!(
            old_format_message("docs_idx", at(3, 0)),
            "pg_pinecone index 'docs_idx' has format version 3 with page layout 0 (this build reads \
             layout 1): REINDEX it"
        );
    }

    #[test]
    fn needs_reindex_follows_the_mode_and_the_stored_format() {
        assert!(meta(Mode::Local, 2, 8).needs_reindex());
        assert!(meta(Mode::Promoting, 1, 8).needs_reindex());
        assert!(!meta(Mode::Local, 3, 8).needs_reindex());
        assert!(!meta(Mode::Remote, 1, 8).needs_reindex());
        assert!(!meta(Mode::Demoting, 2, 8).needs_reindex());
    }
    #[test]
    fn the_report_shows_the_defaults_in_force_and_what_the_build_measured() {
        let calibrated = meta(Mode::Local, 3, 128);
        assert_eq!(
            calibrated.calibration_report(),
            CalibrationReport {
                nprobe: Some(5),
                pool: Some(200),
                calib_nprobe: Some(5),
                calib_pool: Some(200),
                calib_recall: Some(0.96),
                calib_samples: 100,
                target_recall: Some(0.95),
                calib_reached: Some(true),
                queries_source: Some("rows"),
                queries_count: 0,
            }
        );

        let skipped = MetaPage {
            calib_nprobe: 0,
            calib_pool: 0,
            calib_samples: 0,
            calib_reached: false,
            calib_recall: 0.0,
            ..calibrated
        };
        let report = MetaPage {
            n_lists: 100,
            ..skipped
        }
        .calibration_report();
        assert_eq!(
            (report.nprobe, report.pool),
            (Some(40), Some(1000)),
            "the 0.9.0 constants"
        );
        assert_eq!(
            (
                report.calib_nprobe,
                report.calib_pool,
                report.calib_recall,
                report.calib_samples
            ),
            (None, None, None, 0)
        );
        assert_eq!(report.target_recall, Some(0.95));

        let pinned = MetaPage {
            rescore_multiplier: 5,
            ..calibrated
        };
        assert_eq!(pinned.calibration_report().pool, Some(500));
        assert_eq!(pinned.calibration_report().calib_pool, Some(200));

        let best_effort = MetaPage {
            calib_reached: false,
            calib_recall: 0.9,
            ..calibrated
        };
        assert_eq!(best_effort.calibration_report().calib_reached, Some(false));
        assert_eq!(skipped.calibration_report().calib_reached, None);
    }

    #[test]
    fn the_report_names_where_the_queries_came_from_and_how_many_were_drawn() {
        let calibrated = meta(Mode::Local, 3, 128);
        for (source, label, count) in [
            (QueriesSource::Rows, "rows", 0),
            (QueriesSource::Table, "table", 100),
            (QueriesSource::Fallback, "fallback", 0),
        ] {
            let report = MetaPage {
                calib_queries: source,
                ..calibrated
            }
            .calibration_report();
            assert_eq!(
                (report.queries_source, report.queries_count),
                (Some(label), count),
                "{source:?}"
            );
        }
        let unmeasured = MetaPage {
            calib_nprobe: 0,
            calib_pool: 0,
            calib_samples: 0,
            calib_reached: false,
            calib_recall: 0.0,
            ..calibrated
        };
        let report = unmeasured.calibration_report();
        assert_eq!((report.queries_source, report.queries_count), (None, 0));
    }

    #[test]
    fn a_queries_source_without_a_calibration_is_a_partly_set_one() {
        let none = MetaPage {
            calib_nprobe: 0,
            calib_pool: 0,
            calib_samples: 0,
            calib_reached: false,
            calib_recall: 0.0,
            ..meta(Mode::Local, 3, 128)
        };
        assert_eq!(untrusted_field_problem(&none), None);
        for source in [QueriesSource::Table, QueriesSource::Fallback] {
            let bad = MetaPage {
                calib_queries: source,
                ..none
            };
            assert!(untrusted_field_problem(&bad).is_some(), "{source:?}");
        }
        let with_table = MetaPage {
            calib_queries: QueriesSource::Table,
            ..meta(Mode::Local, 3, 128)
        };
        assert_eq!(untrusted_field_problem(&with_table), None);
    }

    #[test]
    fn the_report_is_empty_for_a_flat_a_remote_and_an_old_native_index() {
        let calibrated = meta(Mode::Local, 3, 128);
        let empty = CalibrationReport {
            nprobe: None,
            pool: None,
            calib_nprobe: None,
            calib_pool: None,
            calib_recall: None,
            calib_samples: 0,
            target_recall: None,
            calib_reached: None,
            queries_source: None,
            queries_count: 0,
        };
        let flat = MetaPage {
            is_flat: true,
            n_lists: 0,
            ..calibrated
        };
        let remote = MetaPage {
            mode: Mode::Remote,
            ..calibrated
        };
        let old = MetaPage {
            format: 2,
            ..calibrated
        };
        for page in [flat, remote, old] {
            assert_eq!(page.calibration_report(), empty, "{page:?}");
        }
    }
}
