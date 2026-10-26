# Disaster recovery: restore -> repoint -> recover

For a table on the `remote` engine (`pinecone.promote`-managed, any `sync_mode`, or a `pg_pinecone`
access-method index — however it got there: `pinecone.create_index(engine => 'remote')`,
`pinecone.set_engine(..., 'remote')`, or the raw `CREATE INDEX ... USING pinecone`) whose Pinecone index
is gone, or whose local vector column was damaged independently of Pinecone (storage corruption, a
mistaken bulk update, a restored-from-an-old-dump database).

A document index (`pinecone.track_docs`) is the other way round: Postgres holds every document, so
nothing is recovered *from* Pinecone. After a restore or a lost index, `pinecone.backfill_docs(tbl,
clear => true)` rebuilds it from the table — see the
[runbook](runbook.md#backfill).

## The qualifiers (read these before you page anyone)

1. **RPO is never zero.** `pinecone.recover_from_remote` restores only what the remote index
   *currently* holds. If you got there by restoring a Pinecone backup first, your RPO is
   `backup-schedule interval + 15 minutes` — Pinecone backups only include vectors written at
   least 15 minutes before the backup was taken, and backups run on a schedule, not continuously.
   There is no path to zero data loss here; plan capacity/alerting around the schedule interval,
   not around this tool closing the gap for you.
2. **Restore is always to a new index.** Pinecone has no in-place restore — `create-index-from-backup`
   always creates a brand-new index. `pinecone.repoint` is the second step that re-points a table's
   registration at it; there is no third option.
3. **Pinecone documents no backup durability SLA — the gate verifies a backup exists and is
   fresh, not a contractual guarantee.** A backup existing, `Ready`, and recent is the strongest
   claim pg_pinecone (or Pinecone's own API) can make about it; treat backups as best-effort, and keep
   the local vector column populated wherever losing it would actually hurt.
4. **Recover only fixes damage that hasn't reached Pinecone yet.** If a local vector went `NULL`
   through the sync trigger (an `UPDATE ... SET vec = NULL`, a `DELETE`, mis-set metadata that
   triggered a delete), that already propagated as a delete to the remote index *before* you ever
   run `recover_from_remote` — by the time you recover, Pinecone may no longer have that row's
   vector either. Recovering after the fact restores only what survived that propagation, not what
   existed before it.

## The story, end to end

```sql
-- 1. Something is wrong with the live index (deleted, corrupted, whatever) and you have a backup.
--    Outside pg_pinecone, via the Pinecone console/API: restore the backup to a NEW index, e.g. "docs-restored".

-- 2. Re-point the table's registration at it. Validates dimension AND metric first; refuses a mismatch.
SELECT pinecone.repoint('docs', 'docs-restored');

-- 3. Pull vectors from the (now-repointed) index back into the local table.
SELECT pinecone.recover_from_remote('docs');

-- 4. Watch it to completion (async — one page per worker drain round, resumable across worker
--    restarts; a NOTICE reports final counts when it finishes).
SELECT table_name, backlog, queue_state FROM pinecone.status() WHERE table_name = 'docs';
```

`recover_from_remote` also stands alone, with no backup/repoint involved at all — e.g. local
storage lost some vectors independently, but the *live* Pinecone index (whatever it's currently
registered against) still has them. Skip straight to step 3.

## `pinecone.recover_from_remote(tbl, overwrite => false, ghosts => 'report')`

- **`overwrite => false`** (default): a row whose local vector is already non-`NULL` is left
  alone — protects a fresher local write racing this job on a table that's still being written to
  during the recovery. Pass `overwrite => true` to force every fetched vector to win instead
  (typical when you know the whole table's local data is suspect, e.g. right after a restore).
- **`ghosts => 'report'`** (default): a remote id with no matching local row is counted (see the
  completion NOTICE / the job's own `pinecone.outbox` row while in flight) but left alone.
  **`ghosts => 'delete'`**: additionally enqueues that id's removal from the remote index.
- Works on any promoted table, `synced_by = 'trigger'` or the pg_pinecone access method
  (`CREATE INDEX ... USING pinecone`); a table with a partition column is recovered
  [namespace by namespace](#tables-with-a-partition-column). The access-method path REQUIRES **application-level write
  quiescence** for the duration of the job — see the next section for why, what happens when it's
  violated, and how to get out.
- Progress is visible the same way any other outbox job is (`pinecone.status()`'s `backlog`/
  `queue_state`; the job's own `pinecone.outbox` row while queued). A `NOTICE` on completion reports
  `recovered` / `skipped (already non-NULL locally)` / `ghosts` / `misses` / `quarantined` counts
  (a `WARNING` instead when anything was quarantined).

## Access-method tables: misses, `on_miss => 'quarantine'`, and VACUUM

An access-method table's remote ids are heap TIDs (physical row positions). Any concurrent
**non-HOT** write — an update to any indexed column, on any index — silently relocates a row to a
new TID, so the job's fetched ids stop matching. That's a "miss", and misses while a row's local
vector is NULL are how data actually dies: **never `VACUUM` the table while an AM-path recover is
parked on misses** — the index's `ambulkdelete` remote-DELETEs every dead TID it reaps, and for a
relocated row whose local vector is still NULL, the remote copy under the old TID is the ONLY
copy. This is why AM-path recover requires application-level write quiescence in the first place.

**Do not take table locks to quiesce.** A `SHARE` (or stronger) lock blocks the recover job's own
write-back `UPDATE` too — the retry just freezes for as long as you hold it. Pause writes at the
application level instead (stop the writers; no lock).

`on_miss` picks what a chunk does with missed ids:

- `'park'` (default): fail the chunk; the job parks after `pinecone.outbox_max_attempts`. Safe — but
  if rows genuinely relocated, retrying is a DEAD END: the stale TIDs stay listed in the remote
  namespace and miss again on every retry, forever, while any writes made since queue behind the
  parked job.
- `'quarantine'`: copy each missed id's fetched vector (+ metadata) into
  `pinecone.recover_quarantine` — stored as `real[]` straight from the fetched f32s, byte-exact, no
  text round-trip — then ADVANCE past the chunk. The job completes with a `WARNING` and the
  quarantine count. Re-attaching the bytes to rows is yours to do: the quarantined `remote_id` is
  the STALE heap position, so only application knowledge can match bytes back to rows.

**The full remediation sequence when an AM-path recover parks on misses:**

```sql
-- 1. Quiesce application writes to the table (stop the writers — NO table locks).
-- 2. Switch the parked job to quarantine (updates the queued job in place, cursor preserved):
SELECT pinecone.recover_from_remote('my_table', on_miss => 'quarantine');
-- 3. Resume it:
SELECT pinecone.outbox_retry('my_table');
-- 4. It completes with a WARNING; residual misses are now in pinecone.recover_quarantine:
SELECT remote_id, quarantined_at FROM pinecone.recover_quarantine
WHERE relid = 'my_table'::regclass;
-- 5. Re-attach manually. The join key is application knowledge — e.g. a natural key you keep in
--    metadata, or (as here) a pre-loss snapshot of your own:
UPDATE my_table t
SET embedding = q.embedding::vector
FROM pinecone.recover_quarantine q, my_snapshot o
WHERE q.relid = 'my_table'::regclass
  AND t.id = o.id
  AND t.embedding IS NULL
  AND q.embedding = o.embedding::real[];
DELETE FROM pinecone.recover_quarantine WHERE relid = 'my_table'::regclass;
-- 6. Only NOW is VACUUM safe again (the job is complete and every byte is accounted for).
VACUUM my_table;
```

For access-method tables specifically, quarantined `metadata` is usually `NULL` (the AM pushes no
metadata), so step 5 genuinely depends on your own snapshot/keys — quarantine's guarantee is that
the bytes survive, not that re-attachment is automatic.

## `pinecone.repoint(tbl, new_index)`

Updates `pinecone.tables.index_name` for `tbl` — a registry ROW change, nothing else. Requires
`tbl` to be `mode = 'remote'` exactly (not mid-`'promoting'`), and validates `new_index`'s
dimension and metric against the registry row before touching anything; refuses a mismatch with a
clear error. Refuses while a `'recover'` job is queued for `tbl` (the job's persisted pagination
cursor belongs to the current index). The namespace is left untouched (Pinecone backup restores
preserve namespaces); so is a [partitioned table's](#tables-with-a-partition-column) family
`<namespace>:*`.

## Checking backup readiness — `pinecone.backup_readiness(tbl)`

A read-only diagnostic for a promoted table — it never restores, deletes, or otherwise touches
anything (no shed verbs, no storage state). Every Pinecone call it makes is read-only
(`describe_index_stats`, `list_backups`, `get_backup_schedule`), so it's safe to run with a
ControlPlaneViewer-scoped key. Returns three rows, one per check:

```sql
SELECT * FROM pinecone.backup_readiness('docs');
--   check   | status |                          detail
-- ----------+--------+------------------------------------------------------------------
--  sync     | ok     | live(remote namespace 'public.docs')=12000 local(non-NULL vector rows)=12000 diff=0 outbox_backlog=0
--  coverage | ok     | newest Ready backup 'bkp-1' created_at=... namespace_count=1 dimension=1536; this table's namespace is the ONLY namespace in the index
--  freshness| ok     | backup age=612s window=90000s (schedule interval=86400s + max(25%,1h)=21600s + 15min staleness allowance); backup created_at=...
```

1. **sync** — the remote index's live per-namespace vector count (`describe_index_stats`)
   against the local table's non-NULL-vector row count. A nonzero difference is `'warn'` if
   `pinecone.outbox` has a nonzero (or invisible-to-this-role) backlog that could explain it —
   in-flight writes haven't reached Pinecone yet — and `'fail'` only when the backlog is
   *visibly* zero (a genuine drift, not a race).
2. **coverage** — the newest `Ready` backup's `namespace_count` must be >= the index's live
   namespace count (a smaller one means the backup predates a namespace that exists now — a
   restore from it would silently omit that namespace), and its `dimension` must match this
   table's registered dimension. Backups are whole-index: `detail` also reports whether this
   table's namespace is the index's only one, or the index is shared (a restore brings back
   every tenant sharing it, not just this table's data).
3. **freshness** — the newest `Ready` backup's age against the window, verbatim:

   ```
   window = schedule_interval + max(25% of schedule_interval, 1 hour) + 15 minutes
   ```

   (the 15 minutes is Pinecone's own staleness allowance — backups only include vectors written
   at least that long before the backup was taken; see qualifier 1, above). **No configured
   schedule is itself a failing check** — absent a schedule, worst-case RPO is unbounded, and
   pg_pinecone has no schedule-creation SQL surface yet (configure one via the Pinecone console/API, or
   take manual backups before relying on backup-based DR).

A remote-call failure — a plan-gated 403 (backups are Standard/Enterprise-only), an auth failure,
a network error, or a schedule/timestamp shape pg_pinecone can't parse — degrades *only that row's*
`status` to `'unavailable (...)'`, quoting the server's own message in `detail`; it never aborts
the other two checks and never raises. The only thing that raises is `tbl` not being a promoted,
tracked table. Same ACL gate as `repoint`/`recover_from_remote` below (`SECURITY INVOKER`, gated
by `pinecone.tables`'s own privileges) — nothing extra to grant. For a table with a partition column
it counts the [whole family](#tables-with-a-partition-column).

## Tables with a partition column

A [partition column](reference.md#partition-column-on-a-vector-table) gives every value its own
namespace, so recovery is per namespace. The steps above are the same; they work on the family
`<namespace>:*`.

- **`repoint`** keeps the prefix and changes the index. It registers the namespaces the new index
  holds under the prefix and the registry lacks (`adopted N`), so a restore that lost
  `pinecone.partitions` still recovers every tenant.
- **`recover_from_remote`** walks the registered namespaces in order, a page at a time, and a worker
  round keeps walking until `pinecone.backfill_round_ms` is spent (many small tenants take a few
  rounds, not one each). The job keeps the namespace and the page token, so an unfinished recovery
  (a worker restart, a parked job retried with `outbox_retry`) **resumes at the cursor** instead of
  starting over. A
  vector is written back only to a row of that namespace's partition: an id (or an access-method
  TID) that belongs to another tenant's row by now is a ghost (or a miss, see `on_miss`), never a
  restore, and the partition column is never read from remote metadata. The completion NOTICE adds
  `namespaces N`.
- **`backup_readiness`** counts the whole family, names namespaces the registry does not list, and
  warns above 2,000 namespaces in the index: that per-backup figure is unverified, so restore a test
  copy before you rely on it.
- A tenant removed with `delete_partition` after the backup comes back with the restore: delete it
  again ([runbook](runbook.md#partitioned-tables)).

## Restoring the Postgres database (`pg_dump` / `pg_restore`)

`pg_restore` creates `pg_pinecone` before `vector` (extensions go in name order), so
`CREATE INDEX ... USING pinecone (embedding vector_cosine_ops)` used to find no opclass. An event
trigger now creates the opclasses when `vector` is created, and they belong to the extension, so a
dump leaves them out and the restore ends clean. The same membership means `DROP EXTENSION vector
CASCADE` drops `pg_pinecone` too: drop `pg_pinecone` first if you only mean to reinstall pgvector. `pinecone.secret` and the queues are not dumped:
store the key again, and `pinecone.sync` repairs what a queue held.

**A restored access-method index rewrites the family it points at.** The registry row is dumped and
the restore rebuilds the index, which queues a `delete_all` and a backfill for the index and namespace
the row names (the whole family of a partitioned table). They wait while the database has no key;
once it has one, the restored database empties that family and fills it with its own heap's TIDs,
and when the row names production's index that is production's data. Restore next to production only
into an index of its own, before the key goes in: `repoint` a trigger-lane table; for an access-method
table, which `repoint` refuses, `DROP INDEX` the restored one (it forgets the row, discards the queued
rebuild and leaves Pinecone alone) and `create_index(.., engine => 'remote', index_name => 'restored')`.

## Restoring both Postgres and Pinecone together

The two sections above each assume one side survived — [Pinecone gone, Postgres
live](#the-story-end-to-end), or [Postgres restored, Pinecone
live](#restoring-the-postgres-database-pg_dump--pg_restore). A full-cluster loss is neither: you
restore a `pg_dump`/basebackup **and** a Pinecone backup, taken at different times, and must land on
one consistent state. Order is load-bearing, and the default direction of reconciliation depends on
`storage`.

### Order (do not reorder)

A restored registration rewrites the family it names the instant a key is present (see
[above](#restoring-the-postgres-database-pg_dump--pg_restore)): the rebuilt index queues a
`delete_all` + backfill against whatever index its row names. If that is your freshly-restored
Pinecone index, you overwrite it with the *staler* dumped heap. So:

1. **Restore the Postgres DB with no key.** `pinecone.secret` is not dumped, so this is the default —
   just do not store it yet. Every queued rebuild waits.
2. **Sever the restored registration from the restored index before the key goes in.**
   - Trigger lane: `SELECT pinecone.repoint(tbl, 'restored');`
   - AM lane (`repoint` refuses): `DROP INDEX` the restored one — it forgets the row and discards the
     queued rebuild, Pinecone untouched (see above) — then
     `SELECT pinecone.create_index(.., engine => 'remote', index_name => 'restored');`.
3. **Store the key** — `SELECT pinecone.set_api_key('...');`.
4. **Reconcile in the direction set by `storage`** (next).
5. **Verify** (below).

### Direction of truth — by `storage`

| `storage` | Authoritative copy | Reconcile with |
| --- | --- | --- |
| `full` | the local heap column | `pinecone.sync(tbl)` — local -> remote |
| `detached` | the Pinecone index | `pinecone.attach_storage(tbl)` (or `recover_from_remote`) — remote -> local |

`full` means the heap column is a complete copy, self-consistent as of the dump; push it out and the
restored index matches it. `detached` means the local column is NULL — Pinecone is the only full
copy, so pull it back.

**Tie-break.** If the *other* backup is clearly newer (compare the dump's time against the backup's
`created_at`), invert the default: a `full` table whose Pinecone backup is newer ->
`recover_from_remote(tbl, overwrite => true)`. Pick ONE direction. **Running both clobbers whichever
copy was fresher**, and nothing restores writes made after the older of the two backups — worst-case
RPO is that older timestamp (plus Pinecone's 15-minute lag,
[qualifier 1](#the-qualifiers-read-these-before-you-page-anyone)).

### Verify

```sql
SELECT * FROM pinecone.backup_readiness(tbl);  -- the `sync` row must read diff=0, outbox_backlog=0
```

Then one real query per lane.

### Partitioned tables

Everything above holds per namespace for a [partition column](#tables-with-a-partition-column), with
two joint-restore specifics:

- `repoint` adopts the namespaces the restored index holds under the prefix (`adopted N`), so a dump
  that lost `pinecone.partitions` still re-registers every tenant before you reconcile.
- Reconcile walks the family `<namespace>:*` and resumes at its cursor; run `backup_readiness` for
  the whole-family counts and heed its >2,000-namespace warning before trusting a single backup.

## Who can run these

Every function on this page — including `detach_storage`/`attach_storage` below — runs as the
caller (`SECURITY INVOKER`) and reads/writes `pinecone.outbox` /
`pinecone.tables`, neither of which grants `PUBLIC` anything — so in practice they are
superuser/extension-owner operations (or a role explicitly granted those tables); there is no
separate table-owner check beyond that ACL gate. The recover job's local write-backs themselves
execute as the recovered table's OWNER (with `SECURITY_RESTRICTED_OPERATION`), so a table owner's
own triggers/constraints never gain the worker's privileges.

## Detached storage — `pinecone.detach_storage(tbl, force => false)` / `pinecone.attach_storage(tbl)`

`detach_storage` sheds a promoted table's local vector column (NULLs it in place; the rows and
every other column survive, so `JOIN` hydration keeps working) and flips it to
`storage = 'detached'` — Pinecone becomes the ONLY full copy of that column. `attach_storage`
reverses it, re-hydrating from Pinecone (the `recover_from_remote` machinery) and flipping back to
`storage = 'full'` once the recover job finishes.

**Read-only until attach.** While detached, both the id column and the vector column reject any
write that would change their value: an id change ERRORs unconditionally (the old id's remote
vector would be deleted with nothing local left to replace it); giving the vector column an actual
new non-`NULL` value ERRORs too (it's remote-only while detached — a local-only write can never
reach Pinecone's authoritative copy); a write that sets it to `NULL` (the shed pass itself, or any
other one) stays a silent no-op, never a remote delete. Everything else keeps working: a
metadata-only `UPDATE` on a `NULL`-vector row (`setMetadata`, merge-only), `DELETE`, and `TRUNCATE`
are all unaffected by `storage`. `attach_storage` lifts both guards in the same transaction its
recover job's last chunk commits.

> **Owner-tier caveat.** The id guard is a `BEFORE UPDATE ... FOR EACH ROW` trigger. A table
> owner's *own* `BEFORE UPDATE` row trigger that fires *after* `pg_pinecone_detached_id_guard` (Postgres
> fires per-row triggers in name order) can still mutate `NEW.<id>` *after* the guard has already
> inspected the old/new ids — re-opening the same "id changed while detached" hole the guard
> closes. This is the same trust tier as `ALTER TABLE ... DISABLE TRIGGER` or a direct
> `UPDATE pinecone.tables`: it requires table ownership, which Pinecone for Postgres does not defend against here.
> Don't add id-mutating row triggers to a detached table. (This could be hardened by
> re-checking the id in an `AFTER EACH ROW` guard that no owner trigger can run after; not built.)

Two preconditions worth calling out before you run detach:

- **The id column MUST have a single-column `UNIQUE` index or `PRIMARY KEY`** — `detach_storage`
  refuses without one, and this is never bypassable, not even with `force => true`. Without a
  continuously-enforced unique constraint, an `INSERT` of an existing id with a `NULL` vector
  would create two local rows sharing one id, and `attach_storage` would then write the single
  fetched remote vector onto BOTH rows — silently fabricating data. Add one first, e.g.
  `CREATE UNIQUE INDEX ON mytable (id);`. **Keep it in place for the whole detached period**:
  the id column is trigger-immutable while detached, but that trigger does not protect the
  *constraint* from DDL — `DROP`ping the UNIQUE index/PK after detaching re-opens exactly the
  duplicate-id fabrication path above (a dup-id `INSERT` with a `NULL` vector becomes possible
  again). Only drop it after `attach_storage` has completed.
- **Drop any local pgvector (HNSW/IVFFlat) index on the vector column first.** Once the column is
  NULL it indexes nothing, and leaving it in place makes every subsequent write churn a useless
  index. Pinecone for Postgres does not drop it for you (it will not silently drop a user's index); do it yourself
  before detaching.

The RPO qualifiers above apply verbatim: while detached, Pinecone is the only copy, so your
worst-case data loss is still `backup-schedule interval + 15 min`, never zero. `force => true`
bypasses only the backup-coverage gate (a custody choice) — never the id-uniqueness, mode, or
`synced_by` preconditions. A forced detach with no fresh backup can be permanently unrecoverable
if Pinecone loses the index.

**`backup_gate = 'forced'` is a permanent audit mark, not a live status.** `pinecone.status()` shows
`storage`, `backup_gate` (`'covered'` or `'forced'`), `detached_at`, `detach_backup_id`, and a
computed worst-case `rpo` for a detached table — all stamped once, at detach time, from that one
gate check, and never recomputed: `status()` makes no network call of its own, so a `'forced'`
mark stays `'forced'` even after a fresh backup later appears. Run `pinecone.backup_readiness(tbl)`
any time for the live answer to "would the gate pass today" — it never rewrites the stamped
`backup_gate` value.

## See also

- [`runbook.md`](runbook.md) — parked-queue recovery, `NO WORKER`, backlog monitoring; applies to
  a `'recover'` outbox job exactly like any other op.
- [`reference.md`](reference.md#function-x-mode-support) — API-key/mode requirements for every
  function above.
