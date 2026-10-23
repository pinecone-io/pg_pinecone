//! `aminsert`'s local-mode path (maintenance = O(churn) by construction:
//! aminsert = nearest-centroid append, 1–2 pages/record).
//! Replaces the earlier stub `ERROR`.
//!
//! ## Clustered (IVF) insert
//! Assign the new tuple to the nearest centroid (the `rd_amcache`-cached directory — same cache
//! the scan path uses, `scan::cached_directory`) and encode it (RaBitQ, relative to THAT centroid;
//! the unit vector for cosine) plus its filter value, for a filtered index — exactly what
//! `build.rs` would have stored; append it to that list's LAST page,
//! or extend the chain with a fresh page when the last one is full. The
//! append is ONE GenericXLog record touching at most TWO pages (the plan's
//! design claim, verified here: in-place append registers only the last page
//! — full image, since the columnar arrays all shift; chain extension
//! registers the old last page — delta, only its `next_block` word changes —
//! plus the fresh page), comfortably inside `MAX_GENERIC_XLOG_PAGES` (4).
//!
//! **The empty-list rule (directory immutability).** A list that was empty
//! at build time has `list_head == INVALID_BLOCK` in the centroid directory.
//! Appending to it would mean updating the directory page — and the
//! directory is cached per-backend in `rd_amcache` with a staleness argument
//! that rests on "the directory can only change through a full rebuild"
//! (`metapage::AmCache`). Rather than invent cross-backend directory
//! invalidation, an insert whose nearest centroid heads an empty list goes
//! to the nearest centroid WITH a non-empty list instead. This is
//! recall-benign: IVF assignment is approximate by construction, the record
//! lands in a list whose centroid ranks immediately after (so probes that
//! would find the ideal list overwhelmingly probe this one too), and the
//! exact heap rescore fixes final ordering regardless. Empty-at-build lists
//! simply stay empty for the life of the physical index; a REINDEX
//! redistributes everything. (Lists never BECOME headless later: the VACUUM
//! sweep rewrites pages in place and keeps even fully-emptied pages in the
//! chain — `vacuum::sweep_page` — so a once-headed list keeps its head.)
//!
//! ## Concurrency (buffer locking discipline)
//! The chain walk locks ONE page at a time, exclusively, and re-checks
//! `next_block` under that lock before treating a page as the chain's tail:
//! `next_block` only ever transitions `INVALID -> valid` (append-only
//! chains; the sweep never unlinks pages), so a page observed mid-walk to
//! have grown a `next` simply forwards the walk. Two backends racing to
//! extend the same full tail serialize on that tail's exclusive lock; the
//! loser re-reads `next` and walks into the winner's fresh page. Two
//! backends extending DIFFERENT lists' tails serialize on the relation
//! extension lock inside `ExtendBufferedRel` (never `P_NEW`'s
//! skip-extension-lock compat path — see the fix comment at the call site),
//! so each is handed a distinct fresh block. Lock ordering vs the metapage:
//! the counter bump (below) locks block 0 alone and releases it BEFORE the
//! chain walk begins — no path in this module
//! (or scan/sweep, which lock one page at a time) ever holds two of these
//! locks at once except the tail+fresh-page pair, whose fresh page is
//! brand-new and thus unreachable by any other backend until the record
//! commits the link.
//!
//! ## Stats staleness (the drift story)
//! Every local insert bumps `inserted_since_build` in the metapage — its own
//! one-page GenericXLog record, deliberately ORDERED BEFORE the list append
//! (and after all validation, so an erroring insert bumps nothing): a crash
//! between bump and append leaves the counter one high with no record behind
//! it, which errs conservative (a higher uncounted-inserts figure biases
//! filtered mode selection toward mode L — exact — never away from it; see
//! `stats::estimate_fraction`). The bump is a read-modify-write on the page
//! bytes under the buffer's exclusive lock
//! (`metapage::bump_inserted_on_page`), never through the `rd_amcache`d
//! view. **Drift bound**: the build-time stats page's per-value counts can
//! understate a value's true count by at most `inserted_since_build` (every
//! insert is uncounted there), so mode selection divides by
//! `total + inserted_since_build` — the true fraction's LOWER bound —
//! routing borderline values to mode L (recall 1.0 by construction) rather
//! than mode H. Overstating a fraction (the direction that could cost
//! recall by routing a genuinely-rare value to H) is impossible from insert
//! staleness alone. **Aborted transactions overcount permanently**:
//! GenericXLog page writes are non-transactional, so a ROLLBACK'd INSERT's
//! bump (and its appended record — invisible to scans via the rescore's
//! MVCC heap fetch, reclaimed by the next VACUUM sweep) survives the abort.
//! Same conservative direction — toward L, costing time never recall — and
//! the counter resets to zero at every `ambuild`/REINDEX, which recounts
//! everything.
//!
//! ## Price of the separate bump record (measured, not
//! restructured)
//! Keeping the bump its OWN GenericXLog record costs, per clustered insert:
//! one extra exclusive block-0 buffer lock acquisition (brief, but every
//! concurrent inserter serializes through it — block 0 is a cluster-wide
//! point of contention under insert-heavy load) and one extra WAL record
//! (a one-u32 delta plus GenericXLog/XLog record headers, ~100 bytes) on
//! top of the append's own record — i.e. 2 WAL records/insert instead of 1.
//! The alternative — folding the bump into the append's record (block 0 +
//! tail [+ fresh] = at most 3 registered pages, still inside
//! `MAX_GENERIC_XLOG_PAGES` = 4) — would halve the record count and close
//! the crash-overcount window entirely (bump+append become atomic), at the
//! cost of holding block 0's exclusive lock for the whole chain-append
//! critical section instead of an instant, plus a multi-buffer lock-order
//! proof. Deliberately DEFERRED so mode-aware `amcostestimate` and any
//! churn-benchmark rerun can price insert overhead against this documented
//! baseline first.
//!
//! ## Flat (small-N bootstrap) insert, and the flat→IVF transition
//! A flat index stores NO per-record data — its scan path reads live heap
//! tuples directly (`scan::exact_scan_all`), so an inserted row is visible
//! to flat scans with zero index maintenance. Flat `aminsert` is therefore
//! just the counter bump plus the TRANSITION POLICY, which for 0.5 is the
//! honest v1 (per the funded plan): **keep appending flat and WARN advising
//! REINDEX** once the index outgrows its threshold — never auto-rebuild
//! inside `aminsert` (a multi-second k-means build firing mid-INSERT inside
//! a user's transaction, taking locks they never asked for, is worse than a
//! warning). The warning fires when `total_vectors + inserted_since_build`
//! reaches 2x `small_n_threshold` and again at every further threshold
//! multiple ([`should_warn_flat_growth`] — bounded spam: once per multiple,
//! cluster-wide, since the counter is serialized under the metapage lock).
//! Flat scans stay EXACT at any size — the cost of ignoring the warning is
//! query latency (a full heap scan per query), never recall.
//!
//! ## Cost note
//! `Rotation::new` (signs and permutations, about 6 us at 768 dimensions) is
//! rebuilt per insert, exactly as the scan path rebuilds it per scan — same
//! accepted cost. Amortizing it across a bulk INSERT (e.g. caching in
//! `rd_amcache` keyed by `(seed, dim)`) is a later optimization;
//! correctness first.

use std::borrow::Cow;

use pgrx::prelude::*;

use super::{
    metapage::{self, INVALID_BLOCK, MetaPage},
    metric::{Metric, first_non_finite, normalise_into},
    pages::{self, ListBody, ListPage, ListRecord},
    rabitq,
    rotation::Rotation,
    scan,
};

/// Local-mode `aminsert` (see module docstring). `values`/`isnull` are the
/// index-column arrays Postgres hands `aminsert` (0 = vector, 1 = optional
/// filter column); the vector is already known non-NULL (`am::aminsert`
/// skips NULL vectors before routing here, mirroring the build's collect
/// callback).
pub(crate) unsafe fn aminsert_local(
    index: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    meta: &MetaPage,
) {
    unsafe {
        if meta.is_flat {
            // Flat mode's whole insert IS the counter bump (module
            // docstring) — nothing to validate first.
            let effective_total = bump_inserted_count(index, meta);
            if should_warn_flat_growth(effective_total, meta.small_n_threshold) {
                warning!(
                    "pg_pinecone: index \"{}\" bootstrapped as a flat (exact) index below \
                     small_n_threshold = {} rows and has now grown to ~{} rows via inserts — \
                     scans remain exact but pay a full heap scan per query; REINDEX it to \
                     build the IVF structure",
                    relation_name(index),
                    meta.small_n_threshold,
                    effective_total,
                );
            }
            // A flat index stores no per-record data at all — the heap scan
            // path sees the new row with no index write (module docstring).
            return;
        }

        // Everything fallible/derivable runs BEFORE the counter bump: a
        // validation ERROR after the bump would leave a permanent +1 behind
        // (GenericXLog writes are non-transactional — see the module
        // docstring's overcount note),
        // so raise before touching any page. The bump still strictly
        // precedes the APPEND, which is the ordering the staleness bound
        // actually needs.
        let vector = crate::am::parse_vector(*values);
        // Defensive dim check, mirroring the scan path's check: the
        // column's typmod already enforces this for any column ambuild
        // accepted (it requires a declared vector(n)), but the page math
        // below would silently mis-size on a mismatch, so check loudly.
        if vector.len() != meta.dim as usize {
            error!(
                "pg_pinecone: different vector dimensions {} and {}",
                meta.dim,
                vector.len()
            );
        }

        if let Some(at) = first_non_finite(&vector) {
            error!(
                "pg_pinecone: cannot index a vector with a non-finite value (NaN or Infinity) in \
                 component {}: the native engine cannot encode it",
                at + 1
            );
        }

        let (fval, fnull) = if meta.has_filter() {
            if *isnull.add(1) {
                (0i64, true)
            } else {
                (meta.filter_kind.datum_to_value(*values.add(1)), false)
            }
        } else {
            (0i64, false)
        };

        // Nearest centroid WITH a non-empty list (the empty-list rule — module docstring).
        let directory = scan::cached_directory(index, meta);
        let mut best: Option<(usize, pg_sys::BlockNumber, f32)> = None;
        for i in 0..directory.n_entries {
            let head = directory.list_head(i);
            if head == INVALID_BLOCK {
                continue;
            }
            let dist = super::metric::exact_distance(
                meta.metric.partition(),
                &vector,
                directory.centroid(i),
            );
            // `total_cmp` ordering (same reasoning as
            // `kmeans::nearest_centroid`): a NaN distance (finite inputs can
            // overflow to Inf - Inf) must neither win the assignment (bare
            // `<` would let a first-slot NaN stick, since nothing compares
            // less than it) nor panic the INSERT.
            if best.is_none_or(|(_, _, d)| dist.total_cmp(&d) == std::cmp::Ordering::Less) {
                best = Some((i, head, dist));
            }
        }
        // Unreachable for any real clustered build: k-means only produces
        // centroids from actual data, so at least one list is non-empty, and
        // the sweep never removes a chain (module docstring).
        let Some((entry, head, _)) = best else {
            error!("pg_pinecone: clustered local index has no non-empty list to insert into");
        };

        // The code is relative to the centroid of the list the record lands in; a cosine index
        // encodes the unit vector, as the build does.
        let rotation = Rotation::new(meta.seed, meta.dim as usize);
        let mut unit = Vec::new();
        let x: &[f32] = if meta.metric == Metric::Cosine {
            normalise_into(&vector, &mut unit);
            &unit
        } else {
            &vector
        };
        let code = rabitq::encode(&rotation, x, directory.centroid(entry));
        let code_bytes: Vec<u8> = code.bits.iter().flat_map(|w| w.to_le_bytes()).collect();
        let record = ListRecord {
            code: Cow::Borrowed(&code_bytes),
            rnorm: code.rnorm,
            dot: code.dot,
            rc: code.rc,
            tid: pgrx::itemptr::item_pointer_get_both(*heap_tid),
            fval,
            fnull,
        };

        // Counter bump — after validation, before the append (module
        // docstring's staleness ordering).
        bump_inserted_count(index, meta);

        append_to_chain(index, head, meta.code_bytes(), meta.has_filter(), &record);
    }
}

/// Bumps `inserted_since_build` on block 0 through its own one-page
/// GenericXLog record (delta mode — only one word changes) and returns
/// `total_vectors + new_count`, the effective row-count estimate the
/// flat-mode transition warning compares against. `meta.total_vectors` is
/// build-frozen and safe to read from the cached view; the counter itself is
/// read-modify-written on the page under the exclusive buffer lock.
unsafe fn bump_inserted_count(index: pg_sys::Relation, meta: &MetaPage) -> u64 {
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
        let new_count = metapage::bump_inserted_on_page(page);
        pg_sys::GenericXLogFinish(state);
        pg_sys::UnlockReleaseBuffer(buffer);
        meta.total_vectors as u64 + new_count as u64
    }
}

/// The flat→IVF transition warning's trigger (module docstring): fires at
/// exactly 2x `small_n_threshold` effective rows and at every further
/// threshold multiple — each specific count value occurs exactly once
/// cluster-wide (the counter is serialized under the metapage buffer lock),
/// so each multiple warns exactly once. Pure function, unit-tested below.
fn should_warn_flat_growth(effective_total: u64, small_n_threshold: u32) -> bool {
    let threshold = small_n_threshold as u64;
    threshold > 0 && effective_total >= 2 * threshold && effective_total.is_multiple_of(threshold)
}

/// `index`'s relation name, for the transition warning's message.
unsafe fn relation_name(index: pg_sys::Relation) -> String {
    unsafe {
        let name = pg_sys::get_rel_name((*index).rd_id);
        if name.is_null() {
            format!("{:?}", (*index).rd_id)
        } else {
            std::ffi::CStr::from_ptr(name)
                .to_string_lossy()
                .into_owned()
        }
    }
}

/// Walks `head`'s chain to the current tail and appends one record there —
/// in place if the tail has room (one page, full image: every columnar array
/// shifts), else by extending the chain (two pages, one record: the old
/// tail's `next_block` word — delta — plus the fresh `P_NEW` page). See the
/// module docstring's concurrency section for the locking discipline.
unsafe fn append_to_chain(
    index: pg_sys::Relation,
    head: pg_sys::BlockNumber,
    code_bytes: usize,
    has_filter: bool,
    record: &ListRecord<'_>,
) {
    unsafe {
        let capacity = pages::list_entries_per_page(code_bytes, has_filter);
        let max_hops = metapage::max_chain_hops(index);
        let mut hops = 0u64;
        let mut block = head;
        loop {
            // Same rationale as
            // `scan::scan_list`'s identical guard — a corrupt/cyclic chain
            // must ERROR here, not hold this record's exclusive page locks
            // in an uninterruptible spin.
            pg_sys::check_for_interrupts!();
            hops += 1;
            if hops > max_hops {
                error!(
                    "pg_pinecone: list page chain exceeds the relation's own block count — cyclic or \
                     corrupt chain"
                );
            }
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
            let buf = std::slice::from_raw_parts(contents, metapage::usable_page_bytes());

            let list_page = match ListPage::parse(buf, code_bytes, has_filter) {
                Ok(list_page) => list_page,
                Err(problem) => {
                    pg_sys::UnlockReleaseBuffer(buffer);
                    error!("pg_pinecone: corrupt list page at block {block} — {problem}");
                }
            };
            // Re-checked under the exclusive lock — a concurrent extender
            // may have linked a fresh tail since this walk read the previous
            // page (next_block only ever goes INVALID -> valid).
            let next_block = list_page.next_block;
            if next_block != INVALID_BLOCK {
                pg_sys::UnlockReleaseBuffer(buffer);
                block = next_block;
                continue;
            }

            if list_page.n < capacity {
                // In-place append: rebuild the page body with n+1 records
                // (the columnar arrays all shift) — one page, full image,
                // same shape as `vacuum::sweep_page`'s compaction rewrite.
                let mut appended = ListBody::with_capacity(has_filter, code_bytes, list_page.n + 1);
                for i in 0..list_page.n {
                    appended.push(&list_page.record(i));
                }
                appended.push(record);
                let body = appended.finish(next_block);
                let state = pg_sys::GenericXLogStart(index);
                let new_page = pg_sys::GenericXLogRegisterBuffer(
                    state,
                    buffer,
                    pg_sys::GENERIC_XLOG_FULL_IMAGE as i32,
                );
                write_body(new_page, &body);
                pg_sys::GenericXLogFinish(state);
                pg_sys::UnlockReleaseBuffer(buffer);
                return;
            }

            // Tail is full: extend the chain. The fresh page is allocated
            // while still holding the tail's exclusive lock, so the
            // link (tail.next = fresh) and the fresh page's contents commit
            // as ONE record — a concurrent walker can never observe the link
            // without the linked page's contents (GenericXLogFinish applies
            // both working copies before releasing anything).
            //
            // **Allocation goes through `ExtendBufferedRel` WITHOUT
            // `EB_SKIP_EXTENSION_LOCK`**: the first
            // cut used `ReadBufferExtended(P_NEW)`, whose PG16+ compat path
            // passes `EB_SKIP_EXTENSION_LOCK` — preserving the OLD `P_NEW`
            // contract that the CALLER already holds the relation extension
            // lock (or is the relation's only possible extender: new rel,
            // AEL, recovery). `aminsert` runs under RowExclusiveLock, so
            // none of that holds: two backends extending DIFFERENT lists'
            // full tails could both compute the same new block number
            // (`smgrnblocks` raced), one full-image `PageInit` erasing the
            // other's just-committed record and both tails linking the same
            // block — cross-linked chains. Dropping the skip flag makes
            // `ExtendBufferedRel` take the relation extension heavyweight
            // lock around the block-number assignment itself (released
            // before it returns); `EB_LOCK_FIRST` hands the page back
            // already exclusively locked. Holding the TAIL's buffer lock
            // while acquiring the extension lock is nbtree's own
            // `_bt_split` -> `_bt_allocbuf` pattern and cannot deadlock
            // here: an extension-lock holder only ever touches its own
            // fresh (unshared) buffer, never another backend's tail.
            let bmr = pg_sys::BufferManagerRelation {
                rel: index,
                smgr: std::ptr::null_mut(),
                relpersistence: 0,
            };
            let fresh = pg_sys::ExtendBufferedRel(
                bmr,
                pg_sys::ForkNumber::MAIN_FORKNUM,
                std::ptr::null_mut(),
                pg_sys::ExtendBufferedFlags::EB_LOCK_FIRST,
            );
            let fresh_block = pg_sys::BufferGetBlockNumber(fresh);

            let mut single = ListBody::with_capacity(has_filter, code_bytes, 1);
            single.push(record);
            let body = single.finish(INVALID_BLOCK);

            let state = pg_sys::GenericXLogStart(index);
            // The old tail: delta mode — only its next_block word changes.
            let tail_page = pg_sys::GenericXLogRegisterBuffer(state, buffer, 0);
            let tail_contents = pg_sys::PageGetContents(tail_page) as *mut u8;
            std::ptr::copy_nonoverlapping(fresh_block.to_le_bytes().as_ptr(), tail_contents, 4);
            // The fresh page: full image.
            let fresh_page = pg_sys::GenericXLogRegisterBuffer(
                state,
                fresh,
                pg_sys::GENERIC_XLOG_FULL_IMAGE as i32,
            );
            write_body(fresh_page, &body);
            pg_sys::GenericXLogFinish(state);
            pg_sys::UnlockReleaseBuffer(fresh);
            pg_sys::UnlockReleaseBuffer(buffer);
            return;
        }
    }
}

/// `PageInit` + body copy into a GenericXLog working page — the same
/// sequence `build::write_page`/`vacuum::sweep_page` use, on an
/// already-registered working copy.
unsafe fn write_body(page: pg_sys::Page, body: &[u8]) {
    unsafe {
        pg_sys::PageInit(page, pg_sys::BLCKSZ as usize, 0);
        let dest = pg_sys::PageGetContents(page) as *mut u8;
        std::ptr::copy_nonoverlapping(body.as_ptr(), dest, body.len());
        let header = page as *mut pg_sys::PageHeaderData;
        (*header).pd_lower += body.len() as u16;
    }
}

#[cfg(test)]
mod tests {
    use super::should_warn_flat_growth;

    #[test]
    fn flat_growth_warning_fires_at_2x_then_every_threshold_multiple() {
        let t = 100;
        // Below 2x: never.
        assert!(!should_warn_flat_growth(100, t));
        assert!(!should_warn_flat_growth(199, t));
        // At 2x and each later multiple: exactly at the multiple.
        assert!(should_warn_flat_growth(200, t));
        assert!(!should_warn_flat_growth(201, t));
        assert!(!should_warn_flat_growth(299, t));
        assert!(should_warn_flat_growth(300, t));
        assert!(should_warn_flat_growth(1000, t));
        // A zero threshold (reloption min) never warns.
        assert!(!should_warn_flat_growth(1000, 0));
    }
}
