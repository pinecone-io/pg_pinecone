# Contributing

## Schema changes and upgrade scripts

From 0.10.0 on, a release upgrades in place: install the new files, restart Postgres, then
`ALTER EXTENSION pg_pinecone UPDATE` ([runbook](runbook.md#upgrading-from-0100-and-later)). There is no path from
`pinegres` (renamed in 0.10.0: a reinstall), so v0.10.0 is the first base. Anything that changes
what `CREATE EXTENSION` builds (a table, column, constraint, index, event trigger, view, function
signature or body in SQL, an operator class) needs a line in the upgrade script of the cycle, or the
upgraded catalog stops matching a fresh install. CI says so (`upgrade-test`, below).

### Where the script lives

`pg_pinecone/sql/pg_pinecone--<prev>--<new>.sql`, one per consecutive release pair. `cargo pgrx install` and
`cargo pgrx package` copy every `sql/pg_pinecone--*--*.sql` next to the generated `pg_pinecone--<new>.sql`, so
the release tarballs carry them. Postgres chains hops (0.10.0 to 0.12.0 runs both scripts).

The first PR of a cycle bumps `pg_pinecone/Cargo.toml` and `pg_pinecone-client/Cargo.toml` (and the lockfile):
`default_version` in the control file follows the crate version, and main is that release's line from then
on. The same PR adds `pg_pinecone--<last release>--<new>.sql` (empty is fine): CI upgrades from the latest
release tag below the crate version (`scripts/upgrade/pick-base.py derive`) and fails, naming the file, when
that script is missing.

### Rules

The restart comes first, so **until `ALTER EXTENSION UPDATE` runs, the new library serves the old catalog**
(its worker idles meanwhile, see [the runbook](runbook.md#after-alter-extension-pg_pinecone-update)); the update
has to follow the restart immediately. Within a cycle:

- **Every function in the previous release's catalog keeps its C symbol and call shape.** pgrx derives the
  symbol from the Rust name (`<name>_wrapper`): don't rename a `#[pg_extern]`, don't change its arguments or
  `RETURNS TABLE` columns. A new SQL signature is a new Rust function (`#[pg_extern(name = "...")]`, an
  overload) next to the old one. `CREATE OR REPLACE` cannot change a result type, a parameter name or drop a
  default: the diff tool says when it needs `DROP` plus `CREATE`. It can change a default's value (it flipped
  `create_index`'s `engine`) and keeps the function's owner and grants, which `DROP` plus `CREATE` would lose.
- **A removed `#[pg_extern]` is dropped by the script** (`DROP FUNCTION IF EXISTS`). The script runs after the
  restart, so a call to it errors in between.
- **A new `#[pg_extern]` is a plain `CREATE FUNCTION` in the script.** Postgres resolves its symbol in the
  library the session has loaded, so the update cannot come first: with `pg_pinecone` preloaded the session still
  maps the old library and fails with `could not find function "..._wrapper"` (`UPGRADE_ORDER=update-first`
  shows it locally).
- **Registry and outbox changes are additive with defaults**, and what runs before the update (sync triggers,
  searches, scans, `status()`) must work without them. A new column goes last in the Rust `CREATE TABLE` (an
  upgraded table gets it by `ADD COLUMN`, which appends, and the catalog compare checks column order),
  nullable or with a `DEFAULT`. `CHECK` lists only grow.
- **Op kinds are additive**, and so are payload keys: the new worker drains what the old library queued. A
  new kind widens the `CHECK` in the script.
- **Dropped objects go through `DROP ... IF EXISTS` in the script**, never a rename. The compare ignores the
  stub of a dropped column, but the new library writes the old table until the update: the column must be
  nullable or have a default.
- **Data migrations are idempotent `DO $$ ... $$` blocks**: the script may run on a database that had
  part of it applied by hand.
- **An on-disk layout change bumps the index format version.** The access-method metapage carries
  `format_version` (`FORMAT_VERSION` in `local/metapage.rs`; 0.9 pages hold zero there, read as 1; 0.10
  writes 3). A change to the layout of any index page bumps it and adds a reader branch for the old
  format or a refusal, per entry point: a build meeting a newer format raises 55000 `REINDEX it` rather
  than misread it; an older native format (0.10 refuses 1 and 2: the sketch codes) is refused by
  `MetaPage::needs_reindex` at the planner, the scan, the insert, VACUUM and the maintenance readers, and
  `pinecone.index_format` reports it without raising. An update rewrites no index, so an index keeps its
  format until `REINDEX`; a change that refuses old indexes ships the way 0.10's did: a
  `reindex_native_indexes()`-style helper, a WARNING listing the affected indexes in the upgrade script (an
  extension script runs with `client_min_messages = warning`, so a NOTICE would not reach the operator) and
  a runbook step.
  While a format is unreleased, a change to its list pages bumps `LAYOUT_VERSION` instead, the second byte
  of the same word (0.10's format 3 has layout 1: byte-interleaved code blocks; layout 0 is the record-major
  pages of the development snapshots): the refusals are the older format's, with the layout in the message
  (`Stored::is_older`), and no format number is spent. Once a format ships, a layout change takes a new one.
- A changed operator class or access method is hand-written (`SELECT pinecone._install_opclasses();` is
  idempotent and adds missing opclasses, with the new library: the update runs after the restart). Existing
  indexes keep the opclasses they were built with. A member added to an existing operator class cannot pass
  the compare (`ALTER OPERATOR FAMILY ... ADD` dumps as a loose member, `CREATE OPERATOR CLASS` lists its
  members inline): add a new operator class.

### Authoring loop

1. Bump the version if this is the first change of the cycle.
2. Ask the diff tool what changed. The base schema needs no build, a release tarball carries it:

   ```sh
   (cd pg_pinecone && cargo pgrx schema pg17 --out /tmp/head.sql)
   gh release download v0.10.0 -p 'pg_pinecone-0.10.0-pg17-linux-x86_64.tar.gz' -D /tmp/pg_pinecone-upgrade-base/v0.10.0
   scripts/upgrade/diff-schema.py /tmp/pg_pinecone-upgrade-base/v0.10.0/pg_pinecone-0.10.0-pg17-linux-x86_64.tar.gz /tmp/head.sql
   ```

   (Use the `darwin-arm64` tarball on a Mac. To build the base from its tag instead:
   `git worktree add /tmp/pg_pinecone-base v0.10.0`, then the same `cargo pgrx schema` there.) The output is a
   proposal: SQL for functions, event triggers, views and indexes; `CHECK BY HAND` for tables; a list of
   operators, opclasses, types and the access method only a human can write. Nothing is applied for you.
3. Copy what applies into `pg_pinecone/sql/pg_pinecone--<prev>--<new>.sql`.
4. Run the drill: `scripts/upgrade-drill.sh v0.10.0 [pg_config]` (CI's base is
   `scripts/upgrade/pick-base.py derive`; `UPGRADE_BASES=<tag>` overrides it). It builds this tree with
   `cargo pgrx package`, replaces pg_pinecone files in that Postgres install (never aim it at a live server) and
   needs pgvector there (`scripts/install-pgvector.sh`). `UPGRADE_ORDER=update-first` runs the old order: it
   fails as soon as the script creates a C function.

### What `upgrade-test` checks

For Postgres 17 and 18, from the latest release below this version, on a scratch cluster against a fake
Pinecone:

- installs the released tarball (checked against its `.sha256`; a missing one fails the drill unless
  `UPGRADE_ALLOW_UNVERIFIED=1`, for local runs) and builds a fixture (a promoted table, a partition-column table, an
  access-method remote index, a full-text table, a clustered native index (600 rows over its
  `small_n_threshold` of 50: a flat one has no codes and no calibration), a table indexed by `create_index`
  with `engine` left out (hnsw under 0.10.0), an empty one whose `engine => 'pgvector'` index was dropped (0.10.0
  kept its registry row) and a `track()`ed local table whose `engine => 'pgvector'` index was dropped too; a few
  hundred rows each; drained);
- builds this tree, swaps its files in, restarts Postgres and runs `ALTER EXTENSION pg_pinecone UPDATE`;
- **catalog**: `pg_dump --schema-only` of the extension's objects (normalized: comments, OIDs, order, the
  stubs of dropped columns) is identical to a fresh `CREATE EXTENSION` in a second database;
- **per-table objects**: the fixture's own schema (`pg_dump --schema-only -n fx`: the triggers and indexes
  the lanes created on the user's tables) is identical before and after the update. An update that rewrites
  them has to say so in the runbook, and the drill's expectation changes with it;
- **function**: `status()`, `backup_readiness()`, every lane's search and `partitions_of()` return what they
  returned before, also in the window between restart and update (new library, old catalog);
  `worker_version()` shows `compiled` newer there and equal after the update; a write queued while the
  worker idles drains after it; new writes drain (a new partition included) and the fake holds exactly the
  tables' ids.
- **access-method indexes**: each keeps the format word the base wrote and reads through
  `pinecone.index_format()`. At this tree's format (3, from 0.10.0 on) the native index stays readable, scans,
  takes inserts and keeps its calibration; `pinecone.reindex_native_indexes()` finds nothing and the update warns
  of nothing. A new index writes format 3. (An older word would leave the native index refused until the helper
  rebuilds it: the drill keeps that branch, unexercised, for the next format change.)
- **default engine**: the index `create_index` built under the base with `engine` left out (hnsw under 0.10.0) is
  not rebuilt (same `relfilenode`). The function's default is the base's until the update (new library, old
  catalog) and this tree's (`native`) after: a new `create_index` without `engine` builds a native index the
  planner uses, `engine => 'pgvector'` an hnsw one. At the end `set_engine(.., 'native')` moves the base's hnsw
  index to a native one under the same name (the exact top 5, no registry row); the row of an hnsw index
  `create_index` built survives `REINDEX CONCURRENTLY` and goes with `DROP INDEX` (the update deletes the one the base
  kept); the `track()`ed table keeps its row.

CI runs restart-first only, on x86_64 Linux: macOS is drilled by hand, linux-arm64 (also a release tarball)
not at all.

## Live tests

The `live_*` pg tests and the `--ignored` client tests call a real Pinecone project: they run only with `PG_PINECONE_LIVE_TESTS=1` as well as `PINECONE_API_KEY` (set by `scripts/smoke.sh` and the two backup drills); without it the pg tests skip with a notice and the client tests fail.

## Provenance

The quantiser and scan algorithms are implemented from their published papers, not from anyone's code. A
module that implements a published algorithm names the paper in its header (`local/rotation.rs`,
`local/rabitq.rs` and `local/kernel.rs` cite Gao and Long's RaBitQ, SIGMOD 2024, the fast Walsh-Hadamard
literature and the byte-lookup popcount paper) and carries the line "Clean-room from the papers: no code
from any proprietary or third-party implementation". When you add such a module:

- Cite the paper (authors, venue, year) in the module header, and the formula or section you implement
  where it is not obvious.
- Do not copy or translate code from another implementation (proprietary or third-party, e.g. pgvector,
  pgvectorscale, FAISS). Reading a paper's pseudocode is fine; transcribing someone's source is not. If code from a
  third party is needed, review its licence first and attribute it in the file and in `LICENSE`'s notices.
- Test the math against an independent evaluation (the f64 oracle in `local/testkit.rs` is written from the
  formulas, with no code shared with the implementation), not against the implementation's own output.
