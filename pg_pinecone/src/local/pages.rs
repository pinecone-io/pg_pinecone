//! On-disk layout math for the local engine's page kinds, shared by
//! `build.rs` (writer), `insert.rs` and `vacuum.rs` (rewriters), and `scan.rs` (reader) so they can
//! never disagree about offsets. No page kind uses Postgres's line-pointer/
//! item mechanism (`PageAddItem`) — all are simple fixed-layout byte regions
//! written directly into `PageGetContents`, the same style `metapage.rs`
//! uses for block 0. `metapage::usable_page_bytes()` is the shared "how much
//! room after `PageInit`" budget every capacity computation below starts
//! from.
//!
//! **Centroid directory page**: `DirPageHeader` (`next_block`, `n_entries`)
//! followed by `n_entries` fixed-size entries, each `[f32; dim]` (the
//! centroid) + `list_head: u32` (that list's first list-page block, or
//! [`crate::local::metapage::INVALID_BLOCK`] for an empty list) +
//! `list_count: u32` (informational — the list's total live vector count).
//!
//! **List page** (format 3, page layout 1; one list's records, chained across as many pages as needed):
//! `ListPageHeader` (`next_block: u32`, `n_in_page: u32`, 8 bytes), padding up to [`CODES_OFFSET`], then
//! page-local arrays of `n_in_page` elements (the codes in whole blocks), in this order:
//! ```text
//! codes   blocks * lanes * D/8 bytes   RaBitQ codes in blocks of `lanes` records (below)
//! rnorm   n * f32                      |x - c|, the residual norm
//! dot     n * f32                      |o'|_1 / sqrt(D), the estimator's correction factor
//! rc      n * f32                      <x - c, c>, the centroid term of the inner-product keys
//! pb      n * u16                      popcount of the code (the estimator's `pb`)
//! tids    n * 6 bytes                  heap block u32 + offset u16
//! fvals   n * i64                      filtered indexes only: the filter value (int kinds raw, text its FNV hash)
//! fnulls  n * u8                       filtered indexes only: 1 = the filter column was NULL (then fval is 0)
//! ```
//! A block holds `lanes` records (64, 32 or 16: [`Geometry`] picks the width that fits the most records
//! on a page; 64 up to 832 dimensions) and is byte-interleaved: byte `j` of record `r` of block `b` is at
//! `codes[b * lanes * D/8 + j * lanes + r]`, so one vector load reads byte `j` of every record of the
//! block (`kernel.rs`, FastScan). The last block is zero-padded: its dead lanes are not counted in
//! `n_in_page`, which counts live records only, and hold the zero code. The codes start at
//! [`CODES_OFFSET`], where every row of a 64-lane block is one cache line of a page-aligned buffer.
//! VACUUM compacts every array together and rebuilds the blocks; an insert rewrites the page.
//!
//! The filter value lives *alongside* the code on the same page — not in a sidecar relation/fork —
//! so mode H can skip non-qualifying records before scoring them and mode L can collect
//! qualifying TIDs from the list pages alone, without touching the heap. An UNFILTERED index's
//! list pages carry neither array (`has_filter = false` everywhere below); the per-record cost of
//! a filter column is [`FILTER_VALUE_SIZE`]` + `[`FILTER_FLAG_SIZE`] = 9 bytes.

use std::borrow::Cow;

use super::{metapage, rotation::padded_dim};

pub(crate) const DIR_HEADER_SIZE: usize = 8; // next_block: u32, n_entries: u32
pub(crate) const LIST_HEADER_SIZE: usize = 8; // next_block: u32, n_in_page: u32

/// `PageGetContents`'s offset in a page, `MAXALIGN(sizeof(PageHeaderData))` (a test pins it).
pub(crate) const CONTENTS_OFFSET: usize = 24;
/// Where the code blocks start in the page contents: with the 24-byte page header and a 64-byte
/// aligned buffer, every 64-byte row of a block is one cache line.
pub(crate) const CODES_OFFSET: usize = 40;
const _: () = assert!(
    CODES_OFFSET >= LIST_HEADER_SIZE && (CONTENTS_OFFSET + CODES_OFFSET).is_multiple_of(64)
);

/// One TID packed as `block: u32` + `offset: u16` — matches
/// `pgrx::itemptr::item_pointer_get_both`'s own `(BlockNumber, OffsetNumber)`
/// pair, just flattened to fixed byte width for on-page storage.
pub(crate) const TID_SIZE: usize = 6;

/// `rnorm`, `dot` and `rc`: one little-endian `f32` each per record.
pub(crate) const SCALARS_PER_RECORD: usize = 3;
pub(crate) const SCALAR_SIZE: usize = 4;

/// One stored filter value: `i64`, little-endian (the raw integer for
/// int2/int4/int8 filter columns — every width sign-extends losslessly — or
/// the [`fnv1a64`] hash for text; see the module docstring).
pub(crate) const FILTER_VALUE_SIZE: usize = 8;
/// One per-record filter NULL flag (`1` = NULL).
pub(crate) const FILTER_FLAG_SIZE: usize = 1;
/// `pb`: one little-endian `u16` per record, the popcount of its code.
pub(crate) const PB_SIZE: usize = 2;

/// Record counts of a page block, widest first (the order breaks ties).
const BLOCK_LANES: [usize; 3] = [64, 32, 16];

/// `D`, the code width in bits: the column dimension padded up to a multiple of 64 and to at least 256.
pub(crate) fn code_bits(dim: usize) -> usize {
    padded_dim(dim)
}

/// Bytes of one stored code, `D / 8` (a multiple of 8).
pub(crate) fn code_bytes(dim: usize) -> usize {
    padded_dim(dim) / 8
}

pub(crate) fn dir_entry_size(dim: usize) -> usize {
    dim * 4 + 4 + 4 // centroid f32s + list_head + list_count
}

pub(crate) fn dir_entries_per_page(dim: usize) -> usize {
    (metapage::usable_page_bytes() - DIR_HEADER_SIZE) / dir_entry_size(dim)
}

/// The largest vector dimension whose centroid directory entry still fits at
/// least once on a page (`dir_entries_per_page(dim) >= 1`) — the real,
/// page-format-derived local-mode dimension cap (Part A quick win #3), not a
/// hardcoded "~2000" folklore number (which this happens to sit close to
/// only because pgvector's own ivfflat cap is governed by a similarly-shaped
/// one-page constraint over a similarly-sized `BLCKSZ`). Derived by solving
/// `dir_entry_size(dim) <= usable_page_bytes() - DIR_HEADER_SIZE` for the
/// largest integer `dim`, so it tracks `BLCKSZ`/header sizes automatically
/// rather than needing a hand-updated constant — see [`dir_entries_per_page`]
/// and [`dir_entry_size`] for the exact layout this mirrors.
pub(crate) fn max_centroid_dimension() -> usize {
    (metapage::usable_page_bytes() - DIR_HEADER_SIZE - 8) / 4
}

/// `ceil(n_lists / dir_entries_per_page(dim))`, at least 1 (a build with
/// `n_lists >= 1` always writes at least one directory page).
pub(crate) fn dir_page_count(n_lists: usize, dim: usize) -> usize {
    n_lists.div_ceil(dir_entries_per_page(dim)).max(1)
}

/// Bytes a record takes outside its code: three scalars, `pb`, the TID and the filter pair.
pub(crate) fn record_overhead(has_filter: bool) -> usize {
    let filter = if has_filter {
        FILTER_VALUE_SIZE + FILTER_FLAG_SIZE
    } else {
        0
    };
    SCALARS_PER_RECORD * SCALAR_SIZE + PB_SIZE + TID_SIZE + filter
}

/// Most records a page of `usable` bytes holds with blocks of `lanes` records (0 when not even one fits).
fn capacity_with(lanes: usize, code_bytes: usize, has_filter: bool, usable: usize) -> usize {
    let (per, block) = (
        record_overhead(has_filter),
        lanes.saturating_mul(code_bytes),
    );
    let Some(room) = usable.checked_sub(CODES_OFFSET) else {
        return 0;
    };
    let mut best = 0;
    for blocks in 1..=room / block.max(1) {
        let Some(left) = room.checked_sub(blocks * block) else {
            break;
        };
        let n = (left / per).min(blocks * lanes);
        if n <= (blocks - 1) * lanes {
            break;
        }
        best = best.max(n);
    }
    best
}

/// The block width and record capacity of the list pages of one index (a function of the code width
/// and whether it has a filter column, like every other offset of a page).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub(crate) code_bytes: usize,
    pub(crate) has_filter: bool,
    /// Records per block: the widest of 64, 32 and 16 that fits the most records on a page.
    pub(crate) lanes: usize,
    /// Records per page.
    pub(crate) capacity: usize,
}

impl Geometry {
    pub(crate) fn new(code_bytes: usize, has_filter: bool) -> Geometry {
        Geometry::within(code_bytes, has_filter, metapage::usable_page_bytes())
    }

    /// [`new`](Self::new) for a code width a page can hold: whole words, at least one.
    pub(crate) fn checked(code_bytes: usize, has_filter: bool) -> Result<Geometry, String> {
        if code_bytes == 0
            || !code_bytes.is_multiple_of(8)
            || code_bytes > metapage::usable_page_bytes()
        {
            return Err(format!(
                "code width of {code_bytes} bytes is not a whole number of words that fits a page"
            ));
        }
        Ok(Geometry::new(code_bytes, has_filter))
    }

    /// For a page of `usable` contents bytes.
    pub(crate) fn within(code_bytes: usize, has_filter: bool, usable: usize) -> Geometry {
        let mut best = Geometry {
            code_bytes,
            has_filter,
            lanes: BLOCK_LANES[BLOCK_LANES.len() - 1],
            capacity: 0,
        };
        for lanes in BLOCK_LANES {
            let capacity = capacity_with(lanes, code_bytes, has_filter, usable);
            if capacity > best.capacity {
                best = Geometry {
                    lanes,
                    capacity,
                    ..best
                };
            }
        }
        best
    }

    /// Bytes of one block's codes.
    pub(crate) fn block_bytes(&self) -> usize {
        self.lanes * self.code_bytes
    }

    pub(crate) fn blocks(&self, n: usize) -> usize {
        n.div_ceil(self.lanes)
    }

    /// Byte offsets of the arrays of a page holding `n` records.
    pub(crate) fn layout(&self, n: usize) -> ListLayout {
        let codes_off = CODES_OFFSET;
        let rnorm_off = codes_off + self.blocks(n) * self.block_bytes();
        let dot_off = rnorm_off + n * SCALAR_SIZE;
        let rc_off = dot_off + n * SCALAR_SIZE;
        let pb_off = rc_off + n * SCALAR_SIZE;
        let tids_off = pb_off + n * PB_SIZE;
        let fvals_off = tids_off + n * TID_SIZE;
        let fnulls_off = fvals_off + n * FILTER_VALUE_SIZE;
        let end = if n == 0 {
            LIST_HEADER_SIZE
        } else if self.has_filter {
            fnulls_off + n * FILTER_FLAG_SIZE
        } else {
            fvals_off
        };
        ListLayout {
            codes_off,
            rnorm_off,
            dot_off,
            rc_off,
            pb_off,
            tids_off,
            fvals_off,
            fnulls_off,
            end,
        }
    }
}

/// Records per list page of an index with these codes.
pub(crate) fn list_entries_per_page(code_bytes: usize, has_filter: bool) -> usize {
    Geometry::new(code_bytes, has_filter).capacity
}

/// How many list pages a list of `count` records needs (0 for an empty list
/// — callers must special-case `count == 0` to skip writing/traversing any
/// pages for it at all, using [`metapage::INVALID_BLOCK`] as that list's head).
pub(crate) fn list_page_count(count: usize, code_bytes: usize, has_filter: bool) -> usize {
    if count == 0 {
        0
    } else {
        count.div_ceil(list_entries_per_page(code_bytes, has_filter))
    }
}

/// Byte offsets of a list page's arrays for a page holding `n_in_page` records, relative to
/// `PageGetContents` — the single source of truth the readers and writers derive from.
/// `fvals_off`/`fnulls_off` are computed unconditionally but only backed by real bytes when the
/// index has a filter column; `end` is the first byte past the last array that exists.
pub(crate) struct ListLayout {
    pub(crate) codes_off: usize,
    pub(crate) rnorm_off: usize,
    pub(crate) dot_off: usize,
    pub(crate) rc_off: usize,
    pub(crate) pb_off: usize,
    pub(crate) tids_off: usize,
    pub(crate) fvals_off: usize,
    pub(crate) fnulls_off: usize,
    pub(crate) end: usize,
}

#[cfg_attr(not(any(test, feature = "pg_test")), allow(dead_code))] // test seams
pub(crate) fn list_layout(n_in_page: usize, code_bytes: usize, has_filter: bool) -> ListLayout {
    Geometry::new(code_bytes, has_filter).layout(n_in_page)
}

/// One stored record, as the writers take it and [`ListPage::record`] hands it back.
pub(crate) struct ListRecord<'a> {
    /// `code_bytes` bytes: the code's `D / 64` words, little-endian.
    pub(crate) code: Cow<'a, [u8]>,
    pub(crate) rnorm: f32,
    pub(crate) dot: f32,
    pub(crate) rc: f32,
    /// Heap `(block, offset)`.
    pub(crate) tid: (u32, u16),
    /// `0` and `false` on an unfiltered index.
    pub(crate) fval: i64,
    pub(crate) fnull: bool,
}

/// A list page's body under construction: every array grows record by record, and
/// [`finish`](Self::finish) lays them out as the module docs say. One writer for the build, the
/// insert append and the VACUUM compaction, so the layout has no second implementation.
pub(crate) struct ListBody {
    geo: Geometry,
    n: u32,
    codes: Vec<u8>,
    rnorm: Vec<u8>,
    dot: Vec<u8>,
    rc: Vec<u8>,
    pb: Vec<u8>,
    tids: Vec<u8>,
    fvals: Vec<u8>,
    fnulls: Vec<u8>,
}

impl ListBody {
    pub(crate) fn with_capacity(has_filter: bool, code_bytes: usize, records: usize) -> ListBody {
        let geo = Geometry::new(code_bytes, has_filter);
        let (filter_values, filter_flags) = if has_filter {
            (records, records)
        } else {
            (0, 0)
        };
        ListBody {
            geo,
            n: 0,
            codes: Vec::with_capacity(geo.blocks(records) * geo.block_bytes()),
            rnorm: Vec::with_capacity(records * SCALAR_SIZE),
            dot: Vec::with_capacity(records * SCALAR_SIZE),
            rc: Vec::with_capacity(records * SCALAR_SIZE),
            pb: Vec::with_capacity(records * PB_SIZE),
            tids: Vec::with_capacity(records * TID_SIZE),
            fvals: Vec::with_capacity(filter_values * FILTER_VALUE_SIZE),
            fnulls: Vec::with_capacity(filter_flags * FILTER_FLAG_SIZE),
        }
    }

    pub(crate) fn push(&mut self, r: &ListRecord<'_>) {
        let (lanes, code_bytes) = (self.geo.lanes, self.geo.code_bytes);
        assert_eq!(r.code.len(), code_bytes, "pg_pinecone: record code width");
        let lane = self.n as usize % lanes;
        if lane == 0 {
            self.codes
                .resize(self.codes.len() + self.geo.block_bytes(), 0);
        }
        let block_start = self.codes.len() - self.geo.block_bytes();
        for (j, byte) in r.code.iter().enumerate() {
            self.codes[block_start + j * lanes + lane] = *byte;
        }
        self.n += 1;
        self.rnorm.extend_from_slice(&r.rnorm.to_le_bytes());
        self.dot.extend_from_slice(&r.dot.to_le_bytes());
        self.rc.extend_from_slice(&r.rc.to_le_bytes());
        let pb: u32 = r.code.iter().map(|b| b.count_ones()).sum();
        self.pb.extend_from_slice(&(pb as u16).to_le_bytes());
        self.tids.extend_from_slice(&r.tid.0.to_le_bytes());
        self.tids.extend_from_slice(&r.tid.1.to_le_bytes());
        if self.geo.has_filter {
            self.fvals.extend_from_slice(&r.fval.to_le_bytes());
            self.fnulls.push(r.fnull as u8);
        }
    }

    /// The bytes to write at `PageGetContents`: header, padding, then the arrays in order.
    pub(crate) fn finish(self, next_block: u32) -> Vec<u8> {
        let end = if self.n == 0 {
            LIST_HEADER_SIZE
        } else {
            self.geo.layout(self.n as usize).end
        };
        let mut body = Vec::with_capacity(end);
        body.extend_from_slice(&next_block.to_le_bytes());
        body.extend_from_slice(&self.n.to_le_bytes());
        if self.n > 0 {
            body.resize(CODES_OFFSET, 0);
            for array in [
                &self.codes,
                &self.rnorm,
                &self.dot,
                &self.rc,
                &self.pb,
                &self.tids,
                &self.fvals,
                &self.fnulls,
            ] {
                body.extend_from_slice(array);
            }
        }
        debug_assert_eq!(body.len(), end);
        body
    }
}

/// The scalars of one block, copied out of the page as native values.
pub(crate) struct BlockScalars {
    pub(crate) rnorm: [f32; 64],
    pub(crate) dot: [f32; 64],
    pub(crate) rc: [f32; 64],
    pub(crate) pb: [u16; 64],
}

impl BlockScalars {
    pub(crate) fn new() -> BlockScalars {
        BlockScalars {
            rnorm: [0.0; 64],
            dot: [0.0; 64],
            rc: [0.0; 64],
            pb: [0; 64],
        }
    }
}

/// A list page's contents, bounds-checked once by [`ListPage::parse`]: a header that claims more
/// records than fit the page is refused before any array is sliced, so a crafted or corrupt page
/// cannot send a reader past its buffer.
pub(crate) struct ListPage<'a> {
    pub(crate) next_block: u32,
    pub(crate) n: usize,
    geo: Geometry,
    codes: &'a [u8],
    rnorm: &'a [u8],
    dot: &'a [u8],
    rc: &'a [u8],
    pb: &'a [u8],
    tids: &'a [u8],
    fvals: &'a [u8],
    fnulls: &'a [u8],
}

/// `buf[off..off + len]`, or why not.
fn region(buf: &[u8], off: usize, len: usize) -> Result<&[u8], String> {
    off.checked_add(len)
        .and_then(|end| buf.get(off..end))
        .ok_or_else(|| format!("array of {len} bytes at offset {off} runs past the page"))
}

impl<'a> ListPage<'a> {
    /// `buf` is the page's contents (`usable_page_bytes()` long); `code_bytes` is `D / 8`.
    pub(crate) fn parse(
        buf: &'a [u8],
        code_bytes: usize,
        has_filter: bool,
    ) -> Result<ListPage<'a>, String> {
        if buf.len() < LIST_HEADER_SIZE {
            return Err(format!("a {}-byte page has no list header", buf.len()));
        }
        if code_bytes > buf.len() {
            return Err(format!(
                "code width of {code_bytes} bytes is not a whole number of words that fits a page"
            ));
        }
        ListPage::parse_with(buf, &Geometry::checked(code_bytes, has_filter)?)
    }

    /// [`parse`](Self::parse) with the index's [`Geometry`], worked out once (every page of a scan has
    /// the same).
    pub(crate) fn parse_with(buf: &'a [u8], geo: &Geometry) -> Result<ListPage<'a>, String> {
        let (geo, has_filter) = (*geo, geo.has_filter);
        if buf.len() < LIST_HEADER_SIZE {
            return Err(format!("a {}-byte page has no list header", buf.len()));
        }
        let next_block = read_u32(buf, 0);
        let n = read_u32(buf, 4) as usize;
        if n > geo.capacity {
            return Err(format!(
                "n_in_page {n} exceeds this page's capacity {}",
                geo.capacity
            ));
        }
        if n == 0 {
            return Ok(ListPage {
                next_block,
                n,
                geo,
                codes: &[],
                rnorm: &[],
                dot: &[],
                rc: &[],
                pb: &[],
                tids: &[],
                fvals: &[],
                fnulls: &[],
            });
        }
        let layout = geo.layout(n);
        // The filter arrays exist on a filtered index's pages only: their offsets mean nothing
        // (and may lie past the page) on an unfiltered one.
        let (fvals, fnulls): (&[u8], &[u8]) = if has_filter {
            (
                region(buf, layout.fvals_off, n * FILTER_VALUE_SIZE)?,
                region(buf, layout.fnulls_off, n * FILTER_FLAG_SIZE)?,
            )
        } else {
            (&[], &[])
        };
        Ok(ListPage {
            next_block,
            n,
            geo,
            codes: region(buf, layout.codes_off, geo.blocks(n) * geo.block_bytes())?,
            rnorm: region(buf, layout.rnorm_off, n * SCALAR_SIZE)?,
            dot: region(buf, layout.dot_off, n * SCALAR_SIZE)?,
            rc: region(buf, layout.rc_off, n * SCALAR_SIZE)?,
            pb: region(buf, layout.pb_off, n * PB_SIZE)?,
            tids: region(buf, layout.tids_off, n * TID_SIZE)?,
            fvals,
            fnulls,
        })
    }

    /// Records per block.
    pub(crate) fn lanes(&self) -> usize {
        self.geo.lanes
    }

    pub(crate) fn n_blocks(&self) -> usize {
        self.geo.blocks(self.n)
    }

    /// Live records of block `b`: all `lanes` but the last block's remainder.
    pub(crate) fn block_live(&self, b: usize) -> usize {
        self.n
            .saturating_sub(b * self.geo.lanes)
            .min(self.geo.lanes)
    }

    /// The interleaved codes of block `b`: `D / 8` rows of `lanes` bytes (borrowed from the page).
    pub(crate) fn block_rows(&self, b: usize) -> &'a [u8] {
        let size = self.geo.block_bytes();
        &self.codes[b * size..(b + 1) * size]
    }

    /// The scalars of the live records of block `b` into `out[..live]`.
    pub(crate) fn block_scalars(&self, b: usize, out: &mut BlockScalars) {
        let (first, live) = (b * self.geo.lanes, self.block_live(b));
        for (dst, src) in [
            (&mut out.rnorm, self.rnorm),
            (&mut out.dot, self.dot),
            (&mut out.rc, self.rc),
        ] {
            let bytes = &src[first * SCALAR_SIZE..(first + live) * SCALAR_SIZE];
            for (d, s) in dst.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *d = f32::from_le_bytes(*s);
            }
        }
        let bytes = &self.pb[first * PB_SIZE..(first + live) * PB_SIZE];
        for (d, s) in out.pb.iter_mut().zip(bytes.as_chunks::<2>().0) {
            *d = u16::from_le_bytes(*s);
        }
    }

    /// Bit `r` of the result: record `b * lanes + r` has a filter value within `lo..=hi` (a NULL
    /// never matches). Only on a filtered index's page.
    pub(crate) fn block_filter_mask(&self, b: usize, lo: i64, hi: i64) -> u64 {
        let first = b * self.geo.lanes;
        (0..self.block_live(b)).fold(0u64, |mask, r| {
            let i = first + r;
            let v = self.fval(i);
            mask | u64::from(!self.fnull(i) && lo <= v && v <= hi) << r
        })
    }

    pub(crate) fn rnorm(&self, i: usize) -> f32 {
        scalar_at(self.rnorm, i)
    }

    pub(crate) fn dot(&self, i: usize) -> f32 {
        scalar_at(self.dot, i)
    }

    pub(crate) fn rc(&self, i: usize) -> f32 {
        scalar_at(self.rc, i)
    }

    #[cfg_attr(not(test), allow(dead_code))] // tests; the scan reads `pb` per block
    pub(crate) fn pb(&self, i: usize) -> u16 {
        read_u16(self.pb, i * PB_SIZE)
    }

    pub(crate) fn tid(&self, i: usize) -> (u32, u16) {
        let at = i * TID_SIZE;
        (read_u32(self.tids, at), read_u16(self.tids, at + 4))
    }

    /// The stored filter value of record `i`; only on a filtered index's page.
    pub(crate) fn fval(&self, i: usize) -> i64 {
        read_i64(self.fvals, i * FILTER_VALUE_SIZE)
    }

    /// Whether record `i`'s filter column was NULL; only on a filtered index's page.
    pub(crate) fn fnull(&self, i: usize) -> bool {
        self.fnulls[i] != 0
    }

    /// Record `i`'s code as `D / 8` bytes (gathered from its block).
    pub(crate) fn code(&self, i: usize) -> Vec<u8> {
        let (lanes, code_bytes) = (self.geo.lanes, self.geo.code_bytes);
        let (block, lane) = (i / lanes, i % lanes);
        let rows = self.block_rows(block);
        (0..code_bytes).map(|j| rows[j * lanes + lane]).collect()
    }

    /// Record `i` whole, to copy it into a [`ListBody`].
    pub(crate) fn record(&self, i: usize) -> ListRecord<'a> {
        ListRecord {
            code: Cow::Owned(self.code(i)),
            rnorm: self.rnorm(i),
            dot: self.dot(i),
            rc: self.rc(i),
            tid: self.tid(i),
            fval: if self.geo.has_filter { self.fval(i) } else { 0 },
            fnull: self.geo.has_filter && self.fnull(i),
        }
    }
}

fn scalar_at(array: &[u8], i: usize) -> f32 {
    read_f32(array, i * SCALAR_SIZE)
}

/// A directory page's contents, bounds-checked once by [`DirPage::parse`] (the list-page reader's
/// counterpart: a header that claims more entries than fit is refused before any is sliced).
pub(crate) struct DirPage<'a> {
    pub(crate) next_block: u32,
    pub(crate) n_entries: usize,
    entry_size: usize,
    dim: usize,
    entries: &'a [u8],
}

impl<'a> DirPage<'a> {
    /// `buf` is the page's contents; `dim` the column dimension.
    pub(crate) fn parse(buf: &'a [u8], dim: usize) -> Result<DirPage<'a>, String> {
        if buf.len() < DIR_HEADER_SIZE {
            return Err(format!("a {}-byte page has no directory header", buf.len()));
        }
        let entry_size = dir_entry_size(dim);
        let capacity = dir_entries_per_page(dim);
        let n_entries = read_u32(buf, 4) as usize;
        if n_entries > capacity {
            return Err(format!(
                "n_entries {n_entries} exceeds this page's capacity {capacity}"
            ));
        }
        Ok(DirPage {
            next_block: read_u32(buf, 0),
            n_entries,
            entry_size,
            dim,
            entries: region(buf, DIR_HEADER_SIZE, n_entries * entry_size)?,
        })
    }

    /// Entry `i`: the centroid's `dim` little-endian `f32`s, then the list head.
    pub(crate) fn entry(&self, i: usize) -> &'a [u8] {
        &self.entries[i * self.entry_size..(i + 1) * self.entry_size]
    }

    /// The first block of entry `i`'s list, [`metapage::INVALID_BLOCK`] when it is empty.
    pub(crate) fn list_head(&self, i: usize) -> u32 {
        read_u32(self.entry(i), self.dim * 4)
    }
}

pub(crate) fn read_u32(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        buf[offset..offset + 4]
            .try_into()
            .expect("pg_pinecone: page read out of bounds"),
    )
}

pub(crate) fn read_u16(buf: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(
        buf[offset..offset + 2]
            .try_into()
            .expect("pg_pinecone: page read out of bounds"),
    )
}

pub(crate) fn read_f32(buf: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(
        buf[offset..offset + 4]
            .try_into()
            .expect("pg_pinecone: page read out of bounds"),
    )
}

pub(crate) fn read_i64(buf: &[u8], offset: usize) -> i64 {
    i64::from_le_bytes(
        buf[offset..offset + 8]
            .try_into()
            .expect("pg_pinecone: page read out of bounds"),
    )
}

pub(crate) fn read_u64(buf: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        buf[offset..offset + 8]
            .try_into()
            .expect("pg_pinecone: page read out of bounds"),
    )
}

/// FNV-1a 64 — the stable, platform-independent hash a TEXT filter column's
/// values are stored (and compared) as. **On-disk format**: never change
/// these constants once an index format version shipping them exists —
/// a query's scankey hash must match what `ambuild` wrote, forever. Not
/// cryptographic, and doesn't need to be: collisions merely widen the
/// candidate set, and the executor rechecks the original qual against the
/// heap tuple for every text-filtered match (`xs_recheck = true` — see
/// `scan.rs`), so a collision can never produce a wrong row, only a wasted
/// heap fetch.
pub(crate) fn fnv1a64(bytes: &[u8]) -> i64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local::testkit::Rng;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    const CONTENTS: usize = std::mem::size_of::<pgrx::pg_sys::PageHeaderData>();
    const USABLE: usize = 8168;

    #[test]
    fn dir_page_count_matches_naive_division() {
        let per_page = dir_entries_per_page(8);
        assert!(per_page > 0);
        assert_eq!(dir_page_count(0, 8), 1);
        assert_eq!(dir_page_count(per_page, 8), 1);
        assert_eq!(dir_page_count(per_page + 1, 8), 2);
        assert_eq!(dir_page_count(10 * per_page, 8), 10);
    }

    #[test]
    fn list_page_count_zero_for_empty_list() {
        assert_eq!(list_page_count(0, 64, false), 0);
        assert_eq!(list_page_count(0, 64, true), 0);
    }

    #[test]
    fn a_record_costs_its_code_and_20_bytes_29_with_a_filter() {
        assert_eq!(record_overhead(false), 12 + 2 + 6);
        assert_eq!(record_overhead(true), record_overhead(false) + 8 + 1);
        assert_eq!(metapage::usable_page_bytes(), USABLE);
    }

    #[test]
    fn the_code_is_the_padded_dimension() {
        for (dim, bits) in [
            (1, 256),
            (8, 256),
            (24, 256),
            (64, 256),
            (65, 256),
            (100, 256),
            (256, 256),
            (257, 320),
            (768, 768),
        ] {
            assert_eq!(code_bits(dim), bits, "d = {dim}");
            assert_eq!(code_bytes(dim), bits / 8, "d = {dim}");
        }
    }

    /// The largest `n` whose arrays fit `usable` bytes, found by trying every `n` against the layout
    /// arithmetic written out (blocks of `lanes` records of `code_bytes` rows, then the arrays).
    fn brute_capacity(lanes: usize, code_bytes: usize, has_filter: bool, usable: usize) -> usize {
        let per = 12 + 2 + 6 + if has_filter { 9 } else { 0 };
        (0..=usable)
            .rev()
            .find(|&n| n == 0 || 40 + n.div_ceil(lanes) * lanes * code_bytes + n * per <= usable)
            .unwrap_or(0)
    }

    #[test]
    fn the_block_width_is_the_one_that_fits_the_most_records_the_wider_on_a_tie() {
        for dim in (256..=2048usize).step_by(64) {
            for has_filter in [false, true] {
                let bytes = dim / 8;
                let caps: Vec<usize> = [64, 32, 16]
                    .iter()
                    .map(|&lanes| brute_capacity(lanes, bytes, has_filter, USABLE))
                    .collect();
                let best = *caps.iter().max().unwrap();
                let lanes = [64, 32, 16][caps.iter().position(|c| *c == best).unwrap()];
                let geo = Geometry::new(bytes, has_filter);
                assert_eq!(
                    (geo.lanes, geo.capacity),
                    (lanes, best),
                    "D = {dim}, filter {has_filter}: capacities {caps:?}"
                );
            }
        }
    }

    #[test]
    fn the_block_width_and_capacity_of_the_common_dimensions_are_pinned() {
        for (dim, has_filter, lanes, capacity) in [
            (768, false, 64, 64),
            (768, true, 64, 64),
            (256, false, 32, 150),
            (1536, false, 32, 32),
            (1024, false, 16, 48),
            (2048, false, 16, 16),
            (832, false, 64, 64),
            (896, false, 64, 48),
        ] {
            let geo = Geometry::new(dim / 8, has_filter);
            assert_eq!(
                (geo.lanes, geo.capacity),
                (lanes, capacity),
                "D = {dim}, filter {has_filter}"
            );
            assert_eq!(list_entries_per_page(dim / 8, has_filter), capacity);
        }
    }

    #[test]
    fn a_block_of_64_cannot_hold_a_1024_dimensional_code_in_one_page() {
        assert_eq!(capacity_with(64, 128, false, USABLE), 0);
        assert!(Geometry::new(128, false).lanes < 64);
    }

    #[test]
    fn the_code_blocks_start_on_a_cache_line_of_an_aligned_page() {
        assert_eq!(
            CONTENTS, CONTENTS_OFFSET,
            "MAXALIGN(sizeof(PageHeaderData))"
        );
        assert!((CONTENTS + CODES_OFFSET).is_multiple_of(64));
        for dim in (256..=2048usize).step_by(64) {
            let geo = Geometry::new(dim / 8, false);
            assert!(geo.block_bytes().is_multiple_of(64), "D = {dim}");
            let layout = geo.layout(geo.capacity);
            for b in 0..geo.blocks(geo.capacity) {
                assert!(
                    (CONTENTS + layout.codes_off + b * geo.block_bytes()).is_multiple_of(64),
                    "D = {dim}, block {b}"
                );
            }
            assert!((layout.rnorm_off + CONTENTS).is_multiple_of(4));
        }
    }

    #[test]
    fn layout_arrays_follow_each_other_without_gaps() {
        for (code_bytes, has_filter) in [(32, false), (96, true), (128, false), (256, true)] {
            let geo = Geometry::new(code_bytes, has_filter);
            for n in [1, 2, geo.lanes - 1, geo.lanes, geo.lanes + 1, geo.capacity] {
                let l = geo.layout(n);
                let blocks = n.div_ceil(geo.lanes);
                assert_eq!(l.codes_off, CODES_OFFSET);
                assert_eq!(l.rnorm_off, l.codes_off + blocks * geo.lanes * code_bytes);
                assert_eq!(l.dot_off, l.rnorm_off + 4 * n);
                assert_eq!(l.rc_off, l.dot_off + 4 * n);
                assert_eq!(l.pb_off, l.rc_off + 4 * n);
                assert_eq!(l.tids_off, l.pb_off + 2 * n);
                assert_eq!(l.fvals_off, l.tids_off + 6 * n);
                assert_eq!(l.fnulls_off, l.fvals_off + 8 * n);
                let end = if has_filter {
                    l.fnulls_off + n
                } else {
                    l.tids_off + 6 * n
                };
                assert_eq!(l.end, end, "n = {n}");
                assert_eq!(list_layout(n, code_bytes, has_filter).end, end);
            }
            assert_eq!(geo.layout(0).end, LIST_HEADER_SIZE);
        }
    }

    #[test]
    fn a_full_page_fits_and_one_more_record_does_not_at_every_supported_dimension() {
        for dim in (256..=2048usize).step_by(64) {
            for has_filter in [false, true] {
                let geo = Geometry::new(dim / 8, has_filter);
                assert!(geo.capacity >= 16, "D = {dim}: a block's worth must fit");
                assert!(geo.layout(geo.capacity).end <= USABLE, "D = {dim}");
                assert!(geo.layout(geo.capacity + 1).end > USABLE, "D = {dim}");
            }
        }
        let engine_max = padded_dim(max_centroid_dimension());
        assert!(
            engine_max <= 2048,
            "the largest padded dimension is {engine_max}"
        );
    }

    #[test]
    fn the_most_records_a_list_needs_pages_for_follow_the_capacity() {
        let (bytes, filter) = (code_bytes(768), false);
        let cap = list_entries_per_page(bytes, filter);
        assert_eq!(list_page_count(1, bytes, filter), 1);
        assert_eq!(list_page_count(cap, bytes, filter), 1);
        assert_eq!(list_page_count(cap + 1, bytes, filter), 2);
        assert_eq!(
            list_page_count(1000, bytes, filter),
            1000_usize.div_ceil(cap)
        );
    }

    fn sample(i: usize, code_bytes: usize) -> (Vec<u8>, f32, f32, f32, (u32, u16), i64, bool) {
        let code: Vec<u8> = (0..code_bytes).map(|b| (i * 31 + b * 7) as u8).collect();
        (
            code,
            i as f32 + 0.5,
            1.0 / (i as f32 + 2.0),
            -(i as f32),
            (1000 + i as u32, (i % 200) as u16 + 1),
            (i as i64 - 3) * 1_000_000_007,
            i.is_multiple_of(3),
        )
    }

    fn record_of(i: usize, code_bytes: usize, has_filter: bool) -> ListRecord<'static> {
        let (code, rnorm, dot, rc, tid, fval, fnull) = sample(i, code_bytes);
        ListRecord {
            code: Cow::Owned(code),
            rnorm,
            dot,
            rc,
            tid,
            fval: if has_filter { fval } else { 0 },
            fnull: has_filter && fnull,
        }
    }

    fn body_of(n: usize, code_bytes: usize, has_filter: bool, next: u32) -> Vec<u8> {
        let mut body = ListBody::with_capacity(has_filter, code_bytes, n);
        for i in 0..n {
            body.push(&record_of(i, code_bytes, has_filter));
        }
        body.finish(next)
    }

    /// A page-sized, 8-aligned buffer holding `body` at its start (the contents of a page).
    #[repr(align(8))]
    struct Page([u8; 8192]);

    fn page_with(body: &[u8]) -> Box<Page> {
        let mut page = Box::new(Page([0; 8192]));
        page.0[..body.len()].copy_from_slice(body);
        page
    }

    fn parse(page: &Page, code_bytes: usize, has_filter: bool) -> ListPage<'_> {
        ListPage::parse(&page.0[..USABLE], code_bytes, has_filter)
            .expect("a page this module wrote")
    }

    #[test]
    fn a_page_round_trips_through_the_writer_and_the_reader() {
        for code_bytes in [32, 96, 128, 256] {
            for has_filter in [false, true] {
                let geo = Geometry::new(code_bytes, has_filter);
                let lanes = geo.lanes;
                for n in [
                    0,
                    1,
                    lanes - 1,
                    lanes,
                    lanes + 1,
                    2 * lanes + 3,
                    geo.capacity,
                ] {
                    let n = n.min(geo.capacity);
                    let body = body_of(n, code_bytes, has_filter, 77);
                    assert_eq!(body.len(), geo.layout(n).end, "n = {n}");
                    let page = page_with(&body);
                    let parsed = parse(&page, code_bytes, has_filter);
                    assert_eq!((parsed.next_block, parsed.n), (77, n));
                    assert_eq!(parsed.n_blocks(), n.div_ceil(lanes));
                    for i in 0..n {
                        let (code, rnorm, dot, rc, tid, fval, fnull) = sample(i, code_bytes);
                        let r = parsed.record(i);
                        assert_eq!(&*r.code, &code[..], "record {i}");
                        assert_eq!((r.rnorm, r.dot, r.rc, r.tid), (rnorm, dot, rc, tid));
                        assert_eq!(r.fval, if has_filter { fval } else { 0 });
                        assert_eq!(r.fnull, has_filter && fnull);
                        let pb: u32 = code.iter().map(|b| b.count_ones()).sum();
                        assert_eq!(u32::from(parsed.pb(i)), pb);
                    }
                    for b in 0..parsed.n_blocks() {
                        let (rows, live) = (parsed.block_rows(b), parsed.block_live(b));
                        assert_eq!(rows.len(), code_bytes * lanes);
                        assert_eq!(live, (n - b * lanes).min(lanes));
                        for lane in 0..lanes {
                            for j in 0..code_bytes {
                                let want = if lane < live {
                                    sample(b * lanes + lane, code_bytes).0[j]
                                } else {
                                    0
                                };
                                assert_eq!(
                                    rows[j * lanes + lane],
                                    want,
                                    "block {b} row {j} lane {lane}"
                                );
                            }
                        }
                        let mut scalars = BlockScalars::new();
                        parsed.block_scalars(b, &mut scalars);
                        for lane in 0..live {
                            let i = b * lanes + lane;
                            assert_eq!(scalars.rnorm[lane], parsed.rnorm(i));
                            assert_eq!(scalars.dot[lane], parsed.dot(i));
                            assert_eq!(scalars.rc[lane], parsed.rc(i));
                            assert_eq!(scalars.pb[lane], parsed.pb(i));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_small_page_has_the_documented_bytes() {
        let (code_bytes, n) = (32usize, 3usize);
        let geo = Geometry::new(code_bytes, false);
        assert_eq!(geo.lanes, 32);
        let mut body = ListBody::with_capacity(false, code_bytes, n);
        for r in 0..n {
            body.push(&ListRecord {
                code: Cow::Owned((0..code_bytes).map(|j| (r * 16 + j) as u8).collect()),
                rnorm: 1.5 + r as f32,
                dot: 0.75,
                rc: -2.0,
                tid: (0x0A0B_0C0D + r as u32, 7 + r as u16),
                fval: 0,
                fnull: false,
            });
        }
        let bytes = body.finish(0x0102_0304);
        assert_eq!(bytes.len(), 40 + 32 * 32 + 3 * 12 + 3 * 2 + 3 * 6);
        assert_eq!(&bytes[0..4], &0x0102_0304u32.to_le_bytes());
        assert_eq!(&bytes[4..8], &3u32.to_le_bytes());
        assert!(
            bytes[8..40].iter().all(|b| *b == 0),
            "padding up to the first block"
        );
        for (r, j) in [(0, 0), (2, 5), (1, 31)] {
            assert_eq!(
                bytes[40 + j * 32 + r],
                (r * 16 + j) as u8,
                "row {j} lane {r}"
            );
        }
        assert!(
            bytes[40 + 3..40 + 32].iter().all(|b| *b == 0),
            "dead lanes of row 0 are zero"
        );
        let rnorm = 40 + 32 * 32;
        assert_eq!(&bytes[rnorm + 4..rnorm + 8], &2.5f32.to_le_bytes());
        let dot = rnorm + 12;
        assert_eq!(&bytes[dot..dot + 4], &0.75f32.to_le_bytes());
        let rc = dot + 12;
        assert_eq!(&bytes[rc + 8..rc + 12], &(-2.0f32).to_le_bytes());
        let pb = rc + 12;
        let pb0: u32 = (0..32u32).map(|j| (j as u8).count_ones()).sum();
        assert_eq!(&bytes[pb..pb + 2], &(pb0 as u16).to_le_bytes());
        let tids = pb + 6;
        assert_eq!(
            &bytes[tids + 6..tids + 10],
            &(0x0A0B_0C0D_u32 + 1).to_le_bytes()
        );
        assert_eq!(&bytes[tids + 10..tids + 12], &8u16.to_le_bytes());
        assert_eq!(bytes.len(), tids + 18);
    }

    /// The page image is a function of its record sequence alone: rebuilding it after an append or
    /// after dropping records gives the bytes of building the survivors from scratch.
    #[test]
    fn appending_and_compacting_give_the_bytes_of_a_fresh_build() {
        for (code_bytes, has_filter) in [(32, false), (96, true), (128, false)] {
            let geo = Geometry::new(code_bytes, has_filter);
            let n = (geo.lanes + 1).min(geo.capacity - 1);
            let page = page_with(&body_of(n, code_bytes, has_filter, 5));
            let parsed = parse(&page, code_bytes, has_filter);

            let mut appended = ListBody::with_capacity(has_filter, code_bytes, n + 1);
            for i in 0..n {
                appended.push(&parsed.record(i));
            }
            appended.push(&record_of(n, code_bytes, has_filter));
            assert_eq!(
                appended.finish(5),
                body_of(n + 1, code_bytes, has_filter, 5),
                "append at {n} records"
            );

            let keep: Vec<usize> = (0..n).filter(|i| i % 3 != 1).collect();
            let mut compacted = ListBody::with_capacity(has_filter, code_bytes, keep.len());
            for &i in &keep {
                compacted.push(&parsed.record(i));
            }
            let mut fresh = ListBody::with_capacity(has_filter, code_bytes, keep.len());
            for &i in &keep {
                fresh.push(&record_of(i, code_bytes, has_filter));
            }
            assert_eq!(
                compacted.finish(9),
                fresh.finish(9),
                "compaction of {n} records"
            );
        }
    }

    #[test]
    fn sixty_five_and_a_thousand_records_fill_pages_and_blocks_exactly() {
        for (code_bytes, has_filter, total) in
            [(32, false, 65), (96, false, 1000), (32, true, 1000)]
        {
            let geo = Geometry::new(code_bytes, has_filter);
            let pages = list_page_count(total, code_bytes, has_filter);
            let mut seen = 0usize;
            for p in 0..pages {
                let n = geo.capacity.min(total - seen);
                let mut body = ListBody::with_capacity(has_filter, code_bytes, n);
                for i in seen..seen + n {
                    body.push(&record_of(i, code_bytes, has_filter));
                }
                let page = page_with(&body.finish(p as u32 + 1));
                let parsed = parse(&page, code_bytes, has_filter);
                assert_eq!(parsed.n, n);
                assert_eq!(parsed.n_blocks(), n.div_ceil(geo.lanes));
                for k in 0..n {
                    assert_eq!(&*parsed.record(k).code, &sample(seen + k, code_bytes).0[..]);
                }
                seen += n;
            }
            assert_eq!(seen, total);
        }
    }

    #[test]
    fn filter_masks_select_the_non_null_values_inside_the_range() {
        let (code_bytes, n) = (32, 40);
        let page = page_with(&body_of(n, code_bytes, true, 0));
        let parsed = parse(&page, code_bytes, true);
        let lanes = parsed.lanes();
        for (lo, hi) in [(i64::MIN, i64::MAX), (0, 5_000_000_000), (7, 7), (1, 0)] {
            for b in 0..parsed.n_blocks() {
                let mask = parsed.block_filter_mask(b, lo, hi);
                for lane in 0..lanes {
                    let i = b * lanes + lane;
                    let want =
                        i < n && !parsed.fnull(i) && lo <= parsed.fval(i) && parsed.fval(i) <= hi;
                    assert_eq!(mask >> lane & 1 == 1, want, "range {lo}..={hi}, record {i}");
                }
            }
        }
    }

    #[test]
    fn a_page_claiming_more_records_than_fit_is_refused() {
        let mut body = body_of(2, 32, false, 0);
        let capacity = Geometry::new(32, false).capacity;
        body[4..8].copy_from_slice(&(capacity as u32 + 1).to_le_bytes());
        let page = page_with(&body);
        let error = ListPage::parse(&page.0[..USABLE], 32, false)
            .err()
            .expect("refused");
        assert!(error.contains("exceeds this page's capacity"), "{error}");
        body[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        let page = page_with(&body);
        assert!(ListPage::parse(&page.0[..USABLE], 32, false).is_err());
        body[4..8].copy_from_slice(&(capacity as u32).to_le_bytes());
        let page = page_with(&body);
        assert!(ListPage::parse(&page.0[..USABLE], 32, false).is_ok());
    }

    #[test]
    fn a_code_width_that_is_not_whole_words_or_does_not_fit_is_refused() {
        let buf = [0u8; 64];
        assert!(ListPage::parse(&buf, 12, false).is_err());
        assert!(ListPage::parse(&buf, 0, false).is_err());
        assert!(ListPage::parse(&buf[..4], 8, false).is_err());
        for width in [usize::MAX & !7, 1 << 62, 8 * 1024, USABLE + 8] {
            assert!(ListPage::parse(&buf, width, true).is_err(), "width {width}");
        }
    }

    #[test]
    fn a_buffer_shorter_than_the_arrays_it_claims_is_refused() {
        let body = body_of(100, 32, false, 0);
        let page = page_with(&body);
        for len in [LIST_HEADER_SIZE, 40, 500, body.len() - 1] {
            assert!(
                ListPage::parse(&page.0[..len], 32, false).is_err(),
                "a {len}-byte buffer cannot hold 100 records"
            );
        }
        assert!(ListPage::parse(&page.0[..body.len()], 32, false).is_ok());
    }

    /// Everything a reader may do with a page that parsed.
    fn exercise(page: &ListPage<'_>, code_bytes: usize, has_filter: bool) {
        assert!(
            page.n <= Geometry::new(code_bytes, has_filter).capacity,
            "a parsed page holds no more records than a page does"
        );
        let mut scalars = BlockScalars::new();
        for b in 0..page.n_blocks() {
            assert_eq!(page.block_rows(b).len(), page.lanes() * code_bytes);
            assert!(page.block_live(b) <= page.lanes());
            page.block_scalars(b, &mut scalars);
            if has_filter {
                let _ = page.block_filter_mask(b, i64::MIN, i64::MAX);
            }
        }
        for i in 0..page.n {
            let (_, _, _, _, _) = (
                page.rnorm(i),
                page.dot(i),
                page.rc(i),
                page.pb(i),
                page.tid(i),
            );
            let record = page.record(i);
            assert_eq!(record.code.len(), code_bytes);
            if has_filter {
                let _ = (page.fval(i), page.fnull(i));
            }
        }
    }

    /// Random pages, valid pages with flipped bytes and rewritten counts, and truncated buffers: the
    /// parser answers `Err` or hands out slices a reader can use, and never panics or reads outside.
    #[test]
    fn list_page_parse_survives_random_and_mutated_pages() {
        let mut rng = Rng::new(0xF022_5EED);
        let (mut parsed_ok, mut refused) = (0usize, 0usize);
        for round in 0..10_000 {
            let code_bytes = [8usize, 32, 96, 128, 192, 256][rng.below(6)];
            let has_filter = rng.below(2) == 1;
            let geo = Geometry::new(code_bytes, has_filter);
            let mut page = Box::new(Page([0; 8192]));
            let mut len = USABLE;
            match round % 5 {
                0 => page.0[..USABLE]
                    .iter_mut()
                    .for_each(|b| *b = rng.u64() as u8),
                1 | 2 => {
                    let n = rng.below(geo.capacity + 1);
                    let body = body_of(n, code_bytes, has_filter, rng.u64() as u32);
                    page.0[..body.len()].copy_from_slice(&body);
                    for _ in 0..=rng.below(8) {
                        let at = if round % 5 == 1 {
                            rng.below(USABLE)
                        } else {
                            rng.below(48)
                        };
                        page.0[at] ^= 1 << rng.below(8);
                    }
                }
                3 => {
                    let n = [
                        0,
                        1,
                        geo.capacity - 1,
                        geo.capacity,
                        geo.capacity + 1,
                        u32::MAX as usize,
                    ][rng.below(6)];
                    let body = body_of(n.min(geo.capacity), code_bytes, has_filter, 3);
                    page.0[..body.len()].copy_from_slice(&body);
                    page.0[4..8].copy_from_slice(&(n as u32).to_le_bytes());
                }
                _ => {
                    let n = rng.below(geo.capacity + 1);
                    let body = body_of(n, code_bytes, has_filter, 3);
                    page.0[..body.len()].copy_from_slice(&body);
                    len = rng.below(USABLE + 1);
                }
            }
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                ListPage::parse(&page.0[..len], code_bytes, has_filter).map(|p| {
                    exercise(&p, code_bytes, has_filter);
                })
            }));
            match outcome {
                Ok(Ok(())) => parsed_ok += 1,
                Ok(Err(_)) => refused += 1,
                Err(_) => panic!(
                    "ListPage::parse or a read panicked on round {round} (kind {})",
                    round % 5
                ),
            }
        }
        assert!(
            parsed_ok > 1_000 && refused > 1_000,
            "{parsed_ok} parsed, {refused} refused"
        );
    }

    #[test]
    fn list_page_parse_survives_odd_code_widths() {
        let mut rng = Rng::new(0x0DD);
        for _ in 0..2_000 {
            let width = [
                0usize,
                1,
                7,
                8,
                12,
                24,
                100,
                4096,
                8160,
                1 << 40,
                usize::MAX,
            ][rng.below(11)];
            let mut page = Box::new(Page([0; 8192]));
            page.0[..USABLE]
                .iter_mut()
                .for_each(|b| *b = rng.u64() as u8);
            let has_filter = rng.below(2) == 1;
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                let _ = ListPage::parse(&page.0[..USABLE], width, has_filter);
            }));
            assert!(outcome.is_ok(), "panic at code width {width}");
        }
    }

    #[test]
    fn dir_page_parse_survives_random_and_mutated_pages() {
        let mut rng = Rng::new(0xD1F);
        let (mut parsed_ok, mut refused) = (0usize, 0usize);
        for round in 0..10_000 {
            let dim = [1usize, 8, 128, 768, 2038][rng.below(5)];
            let entry = dir_entry_size(dim);
            let capacity = (USABLE - DIR_HEADER_SIZE) / entry;
            let mut page = Box::new(Page([0; 8192]));
            let mut len = USABLE;
            match round % 4 {
                0 => page.0[..USABLE]
                    .iter_mut()
                    .for_each(|b| *b = rng.u64() as u8),
                1 => {
                    let n = rng.below(capacity + 1);
                    page.0[..4].copy_from_slice(&(rng.u64() as u32).to_le_bytes());
                    page.0[4..8].copy_from_slice(&(n as u32).to_le_bytes());
                    page.0[8..USABLE]
                        .iter_mut()
                        .for_each(|b| *b = rng.u64() as u8);
                    page.0[rng.below(48)] ^= 1 << rng.below(8);
                }
                2 => {
                    let n = [0, 1, capacity, capacity + 1, u32::MAX as usize][rng.below(5)];
                    page.0[4..8].copy_from_slice(&(n as u32).to_le_bytes());
                }
                _ => {
                    let n = rng.below(capacity + 1);
                    page.0[4..8].copy_from_slice(&(n as u32).to_le_bytes());
                    len = rng.below(USABLE + 1);
                }
            }
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                DirPage::parse(&page.0[..len], dim).map(|p| {
                    assert!(p.n_entries <= capacity);
                    for i in 0..p.n_entries {
                        assert_eq!(p.entry(i).len(), entry);
                        let _ = p.list_head(i);
                    }
                })
            }));
            match outcome {
                Ok(Ok(())) => parsed_ok += 1,
                Ok(Err(_)) => refused += 1,
                Err(_) => panic!("DirPage::parse or a read panicked on round {round}"),
            }
        }
        assert!(
            parsed_ok > 1_000 && refused > 1_000,
            "{parsed_ok} parsed, {refused} refused"
        );
    }

    #[test]
    fn a_directory_page_claiming_more_entries_than_fit_is_refused() {
        let dim = 768;
        let capacity = (USABLE - DIR_HEADER_SIZE) / dir_entry_size(dim);
        let mut page = Box::new(Page([0; 8192]));
        page.0[4..8].copy_from_slice(&(capacity as u32).to_le_bytes());
        assert!(DirPage::parse(&page.0[..USABLE], dim).is_ok());
        page.0[4..8].copy_from_slice(&(capacity as u32 + 1).to_le_bytes());
        let error = DirPage::parse(&page.0[..USABLE], dim)
            .err()
            .expect("refused");
        assert!(error.contains("exceeds this page's capacity"), "{error}");
        assert!(DirPage::parse(&page.0[..4], dim).is_err());
    }

    #[test]
    fn fnv1a64_matches_published_test_vectors() {
        assert_eq!(fnv1a64(b"") as u64, 0xcbf29ce484222325);
        assert_eq!(fnv1a64(b"a") as u64, 0xaf63dc4c8601ec8c);
        assert_eq!(fnv1a64(b"foobar") as u64, 0x85944171f73967e8);
    }
}
