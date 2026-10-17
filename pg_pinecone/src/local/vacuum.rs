//! `ambulkdelete`'s local-mode path (maintenance = O(churn) by
//! construction). Before this module existed, local-mode `ambulkdelete` was a
//! documented no-op (`am.rs`'s old rustdoc: "a no-op is safe... stale
//! list-page entries just become
//! candidates [that a heap fetch] naturally drops"). That was wrong in one
//! specific, serious way: **VACUUM doesn't just mark heap tuples dead, it can
//! also physically truncate trailing all-dead heap pages** (`lazy_truncate_
//! heap` in vacuumlazy.c). A stale index-resident TID pointing at a
//! now-truncated-away block makes `table_index_fetch_tuple` — `local::scan`'s
//! `fetch_vector_by_tid` — raise a hard `ERROR` ("could not read block N in
//! file... it is out of range"), not just return "not found": once that
//! happens, EVERY subsequent scan that happens to probe the list containing
//! that stale TID breaks, permanently, until a `REINDEX` clears it. A no-op
//! `ambulkdelete` guarantees this eventually happens on any table with both
//! ongoing churn and enough deletes to make VACUUM's truncation path fire.
//!
//! The fix: an O(churn) **linear list-page sweep**. Every list page gets
//! visited exactly once per `VACUUM` (unavoidable — a dead TID could be in
//! any of them), but only pages that actually contain at least one now-dead
//! record get rewritten (one `GenericXLog`'d record per modified page,
//! comfortably inside `MAX_GENERIC_XLOG_PAGES`'s 4-page cap since this only
//! ever touches one page at a time) — so the WRITE volume (and WAL) this
//! produces is proportional to churn, not index size, which is the actual
//! "O(churn) maintenance" claim (for comparison, pgvector's ivfflat
//! `VACUUM` walks/rewrites its whole graph regardless of how many rows
//! actually died; this only rewrites pages that changed).
//!
//! Small-N exact-bootstrap ("flat") indexes need none of this: they store no
//! per-vector data of their own at all (`local::scan::exact_scan_all` reads
//! straight from the heap every time), so there is nothing for VACUUM to ever
//! make stale — [`ambulkdelete_local`] is a true no-op for them, correctly.
//!
//! ## `CREATE INDEX CONCURRENTLY` / `REINDEX CONCURRENTLY` — the decision
//!
//! **Verdict (re-affirmed once local `aminsert` became real): still
//! rejected.** Postgres's own `validate_index()` (`index.c`, the second
//! phase of a concurrent index build) enumerates every TID *currently in the
//! new index* by calling `index_bulk_delete` — i.e. this module's entry
//! point — with a callback that always returns `false` ("not dead") and just
//! records every TID it's shown into a tuplesort, to diff against the heap
//! and find rows the concurrent build's snapshot window missed.
//! [`ambulkdelete_local`]'s sweep already visits and reports every live TID
//! to its callback unconditionally, so a clustered (non-flat) local-mode
//! index's enumeration half of that protocol works for free, and local
//! `aminsert` could serve the catch-up inserts. Three holes keep the
//! rejection in place regardless:
//! 1. A FLAT (small-N bootstrap) index stores no TIDs at all, so the
//!    enumerate-and-diff protocol degenerates: `validate_index` sees zero
//!    index-resident TIDs, believes every heap row is "missing", and
//!    re-inserts the whole table — which "works" only by accident of the
//!    flat scan reading the heap directly, while inflating the
//!    `inserted_since_build` staleness counter to the entire table size and
//!    firing spurious transition warnings. An AM lying to `validate_index`
//!    about its contents is not a foundation, even when the lie happens to
//!    be harmless today.
//! 2. A CONCURRENTLY build creates a fresh index oid mid-flight, and
//!    `pinecone._am_engine`'s engine stickiness is keyed by oid — a
//!    `mode = auto` index could silently flip engines across the rebuild,
//!    the exact drift `am::resolve_mode`'s WARNING exists to prevent.
//! 3. The interplay between the k-means training snapshot and the catch-up
//!    insert window is unvalidated (inserts assigned against centroids
//!    trained on a snapshot that never saw them — plausibly fine, IVF
//!    assignment is approximate anyway, but "plausibly" isn't a crash
//!    matrix).
//!
//! `am::ambuild` therefore rejects `index_info->ii_Concurrent` for local
//! mode up front, with a clear error naming the remedy (plain `CREATE
//! INDEX`/`REINDEX`, which take the required exclusive lock instead).
//! Honest rejection beats subtle corruption; revisit with the lifecycle work
//! alongside promote/demote's own crash matrix.

// `pg_test`-only (its sole callers are `lib.rs`'s `#[pg_test]`s, which are
// themselves feature-gated) — `any(test, ...)` would leave it dead, and
// warned-on, under a plain `cargo clippy --all-targets` pass.
#[cfg(feature = "pg_test")]
use std::collections::HashSet;

use pgrx::prelude::*;

use super::{
    metapage::{self, INVALID_BLOCK, MetaPage, usable_page_bytes},
    pages::{self, ListBody, ListPage},
};

/// [`ambulkdelete_local`]'s result — folded into `IndexBulkDeleteResult` by
/// its caller (`am::ambulkdelete`).
pub(crate) struct SweepStats {
    /// Records that survived the sweep, across every list.
    pub(crate) live: u64,
    /// Records the callback reported dead (and this sweep dropped).
    pub(crate) dead: u64,
}

/// Local-mode `ambulkdelete`'s body — see module docstring for the sweep's
/// shape and the CIC/RIC decision this leans on. `is_dead` mirrors Postgres's
/// own `IndexBulkDeleteCallback` (`ItemPointer -> bool`), just as a safe Rust
/// closure — `am::ambulkdelete` wraps the real C callback; `#[pg_test]`s can
/// substitute a plain predicate over TIDs directly.
pub(crate) unsafe fn ambulkdelete_local(
    index: pg_sys::Relation,
    meta: &MetaPage,
    mut is_dead: impl FnMut(pg_sys::ItemPointerData) -> bool,
) -> SweepStats {
    unsafe {
        if meta.is_flat {
            // No list pages exist at all for a flat (small-N bootstrap)
            // index — nothing stored can ever go stale. `total_vectors` is
            // build-time bookkeeping, not re-derived here (see module
            // docstring); reported as-is, purely informational.
            return SweepStats {
                live: meta.total_vectors as u64,
                dead: 0,
            };
        }

        let code_bytes = meta.code_bytes();
        let has_filter = meta.has_filter();
        let list_heads = read_list_heads(index, meta);

        let mut live = 0u64;
        let mut dead = 0u64;
        let max_hops = metapage::max_chain_hops(index);
        for head in list_heads {
            let mut block = head;
            let mut hops = 0u64;
            while block != INVALID_BLOCK {
                // Same rationale as
                // `scan::scan_list`'s identical guard.
                pg_sys::check_for_interrupts!();
                // Between pages, with no buffer held: honour vacuum_cost_delay like core's index AMs.
                vacuum_delay();
                hops += 1;
                if hops > max_hops {
                    error!(
                        "pg_pinecone: list page chain exceeds the relation's own block count — cyclic \
                         or corrupt chain"
                    );
                }
                block = sweep_page(
                    index,
                    block,
                    code_bytes,
                    has_filter,
                    &mut is_dead,
                    &mut live,
                    &mut dead,
                );
            }
        }
        SweepStats { live, dead }
    }
}

/// `vacuum_delay_point()`: sleeps when `vacuum_cost_delay` is set and the cost of the pages read
/// and dirtied so far reaches `vacuum_cost_limit` (a no-op outside VACUUM).
fn vacuum_delay() {
    unsafe {
        #[cfg(not(feature = "pg18"))]
        pg_sys::vacuum_delay_point();
        #[cfg(feature = "pg18")]
        pg_sys::vacuum_delay_point(false);
    }
}

/// Reads just the `list_head` column of every centroid directory entry —
/// everything [`ambulkdelete_local`] (and `scan.rs`'s mode-L filtered scan,
/// which walks every list the same way) needs from the directory (unlike
/// `scan::read_directory`, no use for the centroids themselves, so this
/// skips allocating `n_lists * dim` floats the caller never looks at).
pub(crate) unsafe fn read_list_heads(
    index: pg_sys::Relation,
    meta: &MetaPage,
) -> Vec<pg_sys::BlockNumber> {
    unsafe {
        let dim = meta.dim as usize;
        let mut out = Vec::with_capacity(meta.n_lists as usize);
        let max_hops = metapage::max_chain_hops(index);
        let mut hops = 0u64;
        let mut block = meta.centroid_dir_start;
        while block != INVALID_BLOCK {
            // Same rationale as
            // `scan::scan_list`'s identical guard — this walks the SAME
            // centroid-directory page chain `scan::cached_directory` does.
            pg_sys::check_for_interrupts!();
            hops += 1;
            if hops > max_hops {
                error!(
                    "pg_pinecone: centroid directory chain exceeds the relation's own block count — \
                     cyclic or corrupt chain"
                );
            }
            let buffer = pg_sys::ReadBufferExtended(
                index,
                pg_sys::ForkNumber::MAIN_FORKNUM,
                block,
                pg_sys::ReadBufferMode::RBM_NORMAL,
                std::ptr::null_mut(),
            );
            pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_SHARE as i32);
            let page = pg_sys::BufferGetPage(buffer);
            let contents = pg_sys::PageGetContents(page) as *const u8;
            let buf = std::slice::from_raw_parts(contents, usable_page_bytes());

            let dir_page = match pages::DirPage::parse(buf, dim) {
                Ok(dir_page) => dir_page,
                Err(problem) => {
                    // Unlock/release before erroring — see `scan::cached_directory`'s identical
                    // comment.
                    pg_sys::UnlockReleaseBuffer(buffer);
                    error!("pg_pinecone: corrupt directory page at block {block} — {problem}");
                }
            };
            let next_block = dir_page.next_block;
            // Same total-count bound as `scan::cached_directory`'s identical
            // copy loop — this reads the SAME directory page format.
            for i in 0..dir_page.n_entries {
                if out.len() >= meta.n_lists as usize {
                    pg_sys::UnlockReleaseBuffer(buffer);
                    error!(
                        "pg_pinecone: corrupt centroid directory — more entries across its pages than \
                         the metapage's n_lists ({})",
                        meta.n_lists
                    );
                }
                out.push(dir_page.list_head(i));
            }

            pg_sys::UnlockReleaseBuffer(buffer);
            block = next_block;
        }
        out
    }
}

/// Sweeps one list page: decides every record's liveness via `is_dead`, and
/// — only if at least one record actually died — rewrites the page in place
/// (same `next_block`, fewer records) through one `GenericXLog` record.
/// Returns the page's `next_block`, so callers can walk the chain without
/// re-reading anything already extracted here. A filtered (format v3) page's
/// per-record filter value/null arrays are compacted right along with the
/// codes/norms/TIDs — a surviving record's filter value must survive the
/// sweep byte-identical, or every later filtered scan silently misfilters.
unsafe fn sweep_page(
    index: pg_sys::Relation,
    block: pg_sys::BlockNumber,
    code_bytes: usize,
    has_filter: bool,
    is_dead: &mut impl FnMut(pg_sys::ItemPointerData) -> bool,
    live: &mut u64,
    dead: &mut u64,
) -> pg_sys::BlockNumber {
    unsafe {
        let buffer = pg_sys::ReadBufferExtended(
            index,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            block,
            pg_sys::ReadBufferMode::RBM_NORMAL,
            std::ptr::null_mut(),
        );
        // Exclusive from the start (not share-then-upgrade): this may need
        // to rewrite the page it just read. Local `aminsert` is a
        // real concurrent writer (VACUUM's ShareUpdateExclusiveLock does NOT
        // block RowExclusiveLock inserts): the per-page exclusive buffer
        // lock is what serializes the two — each page rewrite (this sweep's
        // compaction, insert's tail append) is atomic under its lock, insert
        // never unlinks or relinks existing pages (chains are append-only:
        // `next_block` only ever goes INVALID -> valid, at the tail), and a
        // record appended behind the sweep's position is by definition a
        // freshly-inserted, live tuple — missed by THIS sweep, collected by
        // the next, exactly like any AM whose inserts race a VACUUM.
        pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
        let page = pg_sys::BufferGetPage(buffer);
        let contents = pg_sys::PageGetContents(page) as *const u8;
        let buf = std::slice::from_raw_parts(contents, usable_page_bytes());

        let list_page = match ListPage::parse(buf, code_bytes, has_filter) {
            Ok(list_page) => list_page,
            Err(problem) => {
                pg_sys::UnlockReleaseBuffer(buffer);
                error!("pg_pinecone: corrupt list page at block {block} — {problem}");
            }
        };
        let next_block = list_page.next_block;
        let n_in_page = list_page.n;

        let mut keep: Vec<usize> = Vec::with_capacity(n_in_page);
        let mut dropped = 0usize;
        for i in 0..n_in_page {
            let (tid_block, tid_offset) = list_page.tid(i);
            let mut tid = pg_sys::ItemPointerData {
                ip_blkid: pg_sys::BlockIdData { bi_hi: 0, bi_lo: 0 },
                ip_posid: 0,
            };
            pgrx::itemptr::item_pointer_set_all(&mut tid, tid_block, tid_offset);

            if is_dead(tid) {
                dropped += 1;
            } else {
                keep.push(i);
            }
        }

        if dropped == 0 {
            pg_sys::UnlockReleaseBuffer(buffer);
            *live += n_in_page as u64;
            return next_block;
        }

        // Build the compacted body from `buf` before touching the page
        // again — `GenericXLogRegisterBuffer` below hands back a working
        // copy distinct from `contents`/`buf`, but by this point everything
        // needed has already been copied out into `body`.
        let mut compacted = ListBody::with_capacity(has_filter, code_bytes, keep.len());
        for &i in &keep {
            compacted.push(&list_page.record(i));
        }
        let body = compacted.finish(next_block);

        let state = pg_sys::GenericXLogStart(index);
        let new_page = pg_sys::GenericXLogRegisterBuffer(
            state,
            buffer,
            pg_sys::GENERIC_XLOG_FULL_IMAGE as i32,
        );
        pg_sys::PageInit(new_page, pg_sys::BLCKSZ as usize, 0);
        let dest = pg_sys::PageGetContents(new_page) as *mut u8;
        std::ptr::copy_nonoverlapping(body.as_ptr(), dest, body.len());
        let header = new_page as *mut pg_sys::PageHeaderData;
        (*header).pd_lower += body.len() as u16;
        pg_sys::GenericXLogFinish(state);
        pg_sys::UnlockReleaseBuffer(buffer);

        *live += keep.len() as u64;
        *dead += dropped as u64;
        next_block
    }
}

/// Testable, TID-only core used by [`ambulkdelete_local`]'s `#[pg_test]`s to
/// build a `is_dead` predicate straight from a set of TIDs — same "test
/// against a plain closure over ids, not a real Postgres callback" pattern
/// `am::ambulkdelete_core`'s own rustdoc documents for the remote path.
#[cfg(feature = "pg_test")]
pub(crate) fn dead_set_predicate(
    dead_tids: HashSet<(pg_sys::BlockNumber, pg_sys::OffsetNumber)>,
) -> impl FnMut(pg_sys::ItemPointerData) -> bool {
    move |tid| dead_tids.contains(&pgrx::itemptr::item_pointer_get_both(tid))
}

/// One list page's identity + contents summary, as reported by
/// [`debug_list_pages`]: `(list_idx, blkno, next_blkno, n_records, tids)`.
/// `next_blkno` is `-1` for a chain tail (`INVALID_BLOCK` doesn't fit the
/// SQL-friendly `i64` sign convention); `tids` are `"block:offset"` strings,
/// the same rendering `am::tid_to_id` uses for remote ids.
pub(crate) type DebugListPage = (i32, i64, i64, i32, Vec<String>);

/// Walks EVERY list chain read-only (share locks, one page at a time) and
/// reports each page — the chain-integrity primitive behind
/// `pinecone._debug_local_list_pages` (added for the concurrent-insert
/// drill's assertions: no block in two chains, chain TIDs == heap TIDs).
/// Flat indexes report zero rows (they have no list pages at all).
pub(crate) unsafe fn debug_list_pages(
    index: pg_sys::Relation,
    meta: &MetaPage,
) -> Vec<DebugListPage> {
    unsafe {
        let mut out = Vec::new();
        if meta.is_flat {
            return out;
        }
        let code_bytes = meta.code_bytes();
        let max_hops = metapage::max_chain_hops(index);
        for (list_idx, head) in read_list_heads(index, meta).into_iter().enumerate() {
            let mut block = head;
            let mut hops: u64 = 0;
            while block != INVALID_BLOCK {
                // Defensive cycle guard: a corrupt chain must produce an
                // ERROR here, not an unkillable infinite loop inside a
                // diagnostic whose whole purpose is inspecting corruption —
                // bounded by the relation's own block count, not the much
                // looser `u64::from(INVALID_BLOCK)` this used to use, and
                // cancellable via `check_for_interrupts!` besides.
                pg_sys::check_for_interrupts!();
                hops += 1;
                if hops > max_hops {
                    error!(
                        "pg_pinecone: list {list_idx} chain exceeds the relation's own block count — \
                         cyclic or corrupt chain"
                    );
                }
                let buffer = pg_sys::ReadBufferExtended(
                    index,
                    pg_sys::ForkNumber::MAIN_FORKNUM,
                    block,
                    pg_sys::ReadBufferMode::RBM_NORMAL,
                    std::ptr::null_mut(),
                );
                pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_SHARE as i32);
                let page = pg_sys::BufferGetPage(buffer);
                let contents = pg_sys::PageGetContents(page) as *const u8;
                let buf = std::slice::from_raw_parts(contents, usable_page_bytes());

                let list_page = match ListPage::parse(buf, code_bytes, meta.has_filter()) {
                    Ok(list_page) => list_page,
                    Err(problem) => {
                        pg_sys::UnlockReleaseBuffer(buffer);
                        error!("pg_pinecone: corrupt list page at block {block} — {problem}");
                    }
                };
                let (next_block, n_in_page) = (list_page.next_block, list_page.n);
                let tids: Vec<String> = (0..n_in_page)
                    .map(|i| {
                        let (tid_block, tid_offset) = list_page.tid(i);
                        format!("({tid_block},{tid_offset})")
                    })
                    .collect();
                pg_sys::UnlockReleaseBuffer(buffer);

                out.push((
                    list_idx as i32,
                    block as i64,
                    if next_block == INVALID_BLOCK {
                        -1
                    } else {
                        next_block as i64
                    },
                    n_in_page as i32,
                    tids,
                ));
                block = next_block;
            }
        }
        out
    }
}

/// TEST-ONLY (`#[cfg(feature = "pg_test")]`): overwrites page `block`'s
/// `next_block` header field directly — the first `u32` of EITHER a
/// centroid-directory page or a list page (both share the same
/// `next_block`-then-count header shape, `pages::DIR_HEADER_SIZE`==
/// `LIST_HEADER_SIZE`==8), so this works on either page kind. Crafts a
/// cyclic chain (e.g. `test_set_next_block(index, head, head)` for a
/// one-page self-loop) to exercise every chain walker's hop-count guard —
/// the same "poke raw page bytes, then invalidate relcache" technique
/// `build::test_overwrite_directory_entry` uses for a different field.
#[cfg(feature = "pg_test")]
pub(crate) unsafe fn test_set_next_block(
    index: pg_sys::Relation,
    block: pg_sys::BlockNumber,
    next_block: pg_sys::BlockNumber,
) {
    unsafe {
        let buffer = pg_sys::ReadBufferExtended(
            index,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            block,
            pg_sys::ReadBufferMode::RBM_NORMAL,
            std::ptr::null_mut(),
        );
        pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
        let state = pg_sys::GenericXLogStart(index);
        let page = pg_sys::GenericXLogRegisterBuffer(state, buffer, 0);
        let contents = pg_sys::PageGetContents(page) as *mut u8;
        std::ptr::copy_nonoverlapping(next_block.to_le_bytes().as_ptr(), contents, 4);
        pg_sys::GenericXLogFinish(state);
        pg_sys::UnlockReleaseBuffer(buffer);

        pg_sys::CacheInvalidateRelcacheByRelid((*index).rd_id);
    }
}

/// TEST-ONLY (`#[cfg(feature = "pg_test")]`): rewrites the stored filter
/// value of the list record whose heap TID is `tid` (clearing its NULL flag),
/// returning whether it was found. Simulates a text-hash collision, which no
/// pair of real strings a test can write produces: the index then claims
/// another row's hash for this record and only the executor's recheck can
/// tell the difference.
#[cfg(feature = "pg_test")]
pub(crate) unsafe fn test_overwrite_filter_value(
    index: pg_sys::Relation,
    meta: &MetaPage,
    tid: pg_sys::ItemPointerData,
    value: i64,
) -> bool {
    unsafe {
        let (want_block, want_offset) = pgrx::itemptr::item_pointer_get_both(tid);
        let code_bytes = meta.code_bytes();
        for head in read_list_heads(index, meta) {
            let mut block = head;
            while block != INVALID_BLOCK {
                let buffer = pg_sys::ReadBufferExtended(
                    index,
                    pg_sys::ForkNumber::MAIN_FORKNUM,
                    block,
                    pg_sys::ReadBufferMode::RBM_NORMAL,
                    std::ptr::null_mut(),
                );
                pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
                let page = pg_sys::BufferGetPage(buffer);
                let contents = pg_sys::PageGetContents(page) as *const u8;
                let buf = std::slice::from_raw_parts(contents, usable_page_bytes());
                let next_block = pages::read_u32(buf, 0);
                let n_in_page = pages::read_u32(buf, 4) as usize;
                let layout = pages::list_layout(n_in_page, code_bytes, meta.has_filter());
                let found = (0..n_in_page).find(|&i| {
                    let at = layout.tids_off + i * pages::TID_SIZE;
                    pages::read_u32(buf, at) == want_block
                        && pages::read_u16(buf, at + 4) == want_offset
                });
                if let Some(i) = found {
                    let state = pg_sys::GenericXLogStart(index);
                    let page = pg_sys::GenericXLogRegisterBuffer(state, buffer, 0);
                    let contents = pg_sys::PageGetContents(page) as *mut u8;
                    let at = layout.fvals_off + i * pages::FILTER_VALUE_SIZE;
                    std::ptr::copy_nonoverlapping(
                        value.to_le_bytes().as_ptr(),
                        contents.add(at),
                        8,
                    );
                    *contents.add(layout.fnulls_off + i) = 0;
                    pg_sys::GenericXLogFinish(state);
                    pg_sys::UnlockReleaseBuffer(buffer);
                    return true;
                }
                pg_sys::UnlockReleaseBuffer(buffer);
                block = next_block;
            }
        }
        false
    }
}
