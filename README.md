# Pinecone for Postgres

**One vector index, three storage engines.** Keep your pgvector queries; pick `pgvector`,
`native`, or `remote` per table, and switch between them with a single call — in either
direction, without touching a query.

| Engine | What it is | Reach for it when |
|---|---|---|
| **`pgvector`** *(default)* | The HNSW/ivfflat you already run | Unfiltered, in-RAM similarity search |
| **`native`** | Pinecone's engine, in-process — no account, no API key, no network | **Filtered** search where matches cluster unevenly (e.g. per-tenant) |
| **`remote`** | Pinecone's serverless platform, behind the same index | A table that has **outgrown the node** |

All three back the same column. `ORDER BY embedding <=> $1 LIMIT k` reads identically no matter
which engine is live — Postgres stays the system of record throughout.

**Contents** — [Setup](#setup) · [Getting started](#getting-started) · [Examples](#examples) ·
[Which engine, when](#which-engine-when) · [The remote engine](#the-remote-engine) ·
[Native-engine maintenance](#native-engine-maintenance) ·
[Full-text search](#full-text-search-on-a-table) ·
[Reference & further reading](#reference--further-reading) · [Development](#development) ·
[Roadmap](#roadmap)

---

## Setup

**Requirements**

- **PostgreSQL 17 or 18.**
- **pgvector**, installed first — `CREATE EXTENSION pg_pinecone` builds its operator classes against
  it, and both the `pgvector` and `native` engines need it.
- **Only for the `remote` engine:** `shared_preload_libraries = 'pg_pinecone'` in `postgresql.conf`
  (then restart) and a stored API key. A `native`-only install needs neither.

> Installed `pg_pinecone` before pgvector (as `pg_restore` does)? A `WARNING` says so at `CREATE EXTENSION`
> time; `CREATE EXTENSION vector` then creates the operator classes by itself. If it warns that it could
> not, run `SELECT pinecone._install_opclasses();` as a superuser. The classes belong to `pg_pinecone`,
> so `DROP EXTENSION vector CASCADE` drops `pg_pinecone` with them.

**Install** — build from this repo:

```bash
cargo install cargo-pgrx --version 0.19.2 --locked
cargo pgrx init --pg17=download   # or --pg18=download; --pg17/--pg18 $(which pg_config) uses an existing Postgres
cd pg_pinecone
cargo pgrx install --release      # add --pg-config /path/to/pg_config for an existing server (17 or 18)
```

> **Deploying, not building?** For deployments, Pinecone for Postgres is distributed as **prebuilt
> binaries** (one per platform × Postgres major: `pg_pinecone-<version>-pg17-<os>-<arch>.tar.gz` and
> `...-pg18-...`, each smoke-tested on its Postgres before release) — internal-only, not published
> to any public download page or source-review marketplace (RDS, Cloud SQL, Supabase, Neon) today;
> the targets are self-hosted, containers, and Pinecone-managed deployments. The **x86_64 binary
> needs an x86-64-v3 CPU** (Haswell 2013+: POPCNT + AVX2 — every cloud x86 of the last decade);
> older CPUs use the source build above. arm64 runs on any aarch64. Licensed under the
> [Apache License 2.0](LICENSE).

> **Coming from `pinegres`?** The extension was renamed in 0.10.0 and Postgres cannot rename an installed
> one: drop `pinegres` and set `pg_pinecone` up as a new install
> ([runbook](docs/runbook.md#upgrading-from-pinegres-090-or-earlier)). From 0.10.0 on, upgrades are in
> place (`ALTER EXTENSION pg_pinecone UPDATE`).

**Enable the remote engine** (only if you'll use it):

```sql
-- after adding pg_pinecone to shared_preload_libraries and restarting
SELECT pinecone.set_api_key('your-api-key');
```

The key lives in `pinecone.secret` (`REVOKE ALL FROM PUBLIC` by default) — the worker and each
writer session read it there, so no session ever has to hold it.

## Getting started

A `native` index on a table you already have — in one call, no account, no network:

```sql
CREATE EXTENSION vector;      -- pgvector first: pg_pinecone builds on it
CREATE EXTENSION pg_pinecone;

CREATE TABLE docs (
    id        bigint PRIMARY KEY,
    body      text,
    embedding vector(768)
);

SELECT pinecone.create_index('docs', 'embedding', engine => 'native');
```

Your existing query is unchanged — and the planner uses the index:

```sql
EXPLAIN SELECT id, body FROM docs ORDER BY embedding <=> '[...]' LIMIT 10;
-- Index Scan using docs_embedding_pinecone_idx on docs
```

Leave out `engine` and you get `pgvector` (the default). That's the whole adoption cost of
`native`: one function call. `create_index` also takes `metric => 'cosine' | 'l2' | 'ip'` (default
`cosine`) and an explicit `index_name`.

## Examples

Copy-paste recipes. Every query against the vector column is plain SQL — the engine is a
lifecycle detail, not a query rewrite.

**1 · Filtered search** (`native`'s strong suit, and `remote` pushes it to Pinecone too — one filter
column per index). `filter_column` makes the index `(embedding, tenant_id)`:

```sql
SELECT pinecone.create_index('docs', 'embedding', engine => 'native', filter_column => 'tenant_id');
-- or the DDL it runs: CREATE INDEX ON docs USING pinecone (embedding vector_cosine_ops, tenant_id)
--                       WITH (mode = 'local');

SELECT id, body FROM docs
 WHERE tenant_id = $1
 ORDER BY embedding <=> $2
 LIMIT 10;
```

**2 · Scale out — promote a table to Pinecone** when it outgrows the node:

```sql
SELECT pinecone.set_engine('docs', 'remote');   -- backfills in the background; never a read/write gap
SELECT pinecone.set_engine('docs', 'native');   -- ...and back. Postgres was the source of truth all along.
```

**3 · Check sync state** (no HTTP call, never fails):

```sql
SELECT * FROM pinecone.status();   -- backlog, oldest_enqueued, queue_state per table; docs_* for full-text
```

**4 · Disaster recovery** — rebuild a promoted table's vectors from Pinecone after a restore:

```sql
SELECT pinecone.backup_readiness('docs');    -- read-only preflight: sync / coverage / freshness
SELECT pinecone.recover_from_remote('docs'); -- rehydrate the vector column from the remote index
```

**5 · Full-text search on a table** — declare the text columns once; Pinecone keeps a BM25
document index in sync, and `search_docs` queries it by table (more in
[Full-text search on a table](#full-text-search-on-a-table)):

```sql
SELECT pinecone.track_docs('articles', text_columns => '["title", "body"]');
SELECT id, score FROM pinecone.search_docs('articles', 'postgres vector search', k => 10);
```

See [`docs/reference.md`](docs/reference.md) for every function, mode, and GUC.

## Which engine, when

| | `pgvector` | `native` | `remote` |
|---|---|---|---|
| **What it is** | HNSW/ivfflat you already run | Pinecone's engine, in-process | Pinecone's serverless platform |
| **Default** | Yes | No — opt in | No — opt in |
| **Best for** | Unfiltered, in-RAM search | Correlated, cross-tenant filtered search | A table that's outgrown the node |
| **Needs** | pgvector | pgvector | pgvector + preload + API key |

`engine = 'pgvector'` is a real `USING hnsw` index the planner serves directly — zero `pg_pinecone`
overhead on the read path; `pg_pinecone` only adds lifecycle and the ability to switch it out later.

### Filtered search: native's case

Production vector search is rarely the clean unfiltered case — it's `WHERE tenant_id = $1` on top
of the similarity search. A *correlated* filter concentrates matching rows in a few regions of an
HNSW graph, and a cross-tenant query strands outside them. `native` scans by cluster instead — IVF
partitioning with RaBitQ binary codes and exact rescoring, implemented from the published papers
— so it doesn't strand.

Measured on a production Postgres build, 1M rows of real 768-dimensional embeddings, cross-tenant
queries against correlated filters (full methodology and every swept point: the
[2026-09 report](bench/vector/REPORT-2026-09-pcvec.md), written under the old `pcvec` name; the
residual-filter benchmark is [`bench/vector/REPORT-residual-filter.md`](bench/vector/REPORT-residual-filter.md)):

| Cross-tenant query, correlated filter | `native` | `pgvector` (best swept point) |
|---|---|---|
| 1% of rows match | Clears recall ≥ 0.95 with ~1.6× latency headroom | Never reaches 0.95 at any setting |
| 10% of rows match | Clears the bar with ~2.9× headroom | Never reaches 0.95 — fastest point is quicker in isolation but stays under the bar |
| Throughput @ 8 conns, 1% filter | ~4.5× the passing-recall bar | Never reaches 0.95 at any concurrency |
| Index size, 1M × 768d | ~26× smaller | Baseline |

> Figures are margins over a pre-committed recall bar, not absolute milliseconds (measured on a
> temporary VM, ahead of a bare-metal rerun). **Scope:** correlated filters queried *cross-tenant*.
> A tenant-scoped filtered query is pgvector's comfort zone too and isn't part of this claim.

**On `remote`.** The same key is stored as Pinecone metadata (`{"tenant_id": 7}`) and
`WHERE tenant_id = $1` is sent as the filter `{"tenant_id": {"$eq": 7}}`, so Pinecone narrows the
search itself instead of the scan over-fetching past a selective `WHERE`.
`create_index(..., engine => 'remote', filter_column => 'tenant_id')`; `set_engine` promotes a
filtered native index the same way. Limits: [reference](docs/reference.md#limits).

**One namespace per tenant.** `promote` can give every value of a NOT NULL column its own Pinecone
namespace (`public.docs:42`): a tenant's query pays for its own namespace, and it is deleted on its own.
`topk` then needs the tenant, as `partition => 42` (an integer, text or uuid) or an `$eq` on the column
in `filter`; ids stay unique across the table. More than 1,000 distinct values warn: Pinecone limits
namespaces per index by plan. `delete_partition` offboards one tenant, and recovery, `repoint` and
`backup_readiness` work namespace by namespace. A tenant Pinecone refuses for good is set aside
(`status().failed_partitions`) instead of parking the table, and `retry_failed_partition` rebuilds it
from the heap. Details:
[reference](docs/reference.md#partition-column-on-a-vector-table).

```sql
SELECT pinecone.promote('docs', partition_column => 'tenant_id');
SELECT * FROM pinecone.topk('docs', $1, k => 10, partition => 42);
SELECT pinecone.delete_partition('docs', 42);   -- the tenant leaves; its rows stay until you delete them
```

On the index itself, `create_index(..., engine => 'remote', partition_column => 'tenant_id')` builds
`USING pinecone (embedding, tenant_id)`: a plain `WHERE tenant_id = 42 ORDER BY embedding <=> $1`
reads namespace `public.docs:42`, and a query without the `WHERE` is an error, not a cross-tenant scan.
Details: [reference](docs/reference.md#partition-column-on-a-remote-index).

**Residual filters.** Only `=` on the one filter column is pushed into the index. Any other `WHERE`
(a range, `IN`, another column) runs after the index returns a batch, so a selective one can
return fewer than `LIMIT` rows unless the scan keeps going. A native index does, until the executor
has enough rows (`pinecone.iterative_scan = auto`, the default); a remote one needs
`SET pinecone.iterative_scan = 'relaxed_order'`. After the first batch the rows come back in batch
order, so re-sort the few rows in a CTE when the exact order matters. `strict_order` keeps the order
but may return fewer rows and can rescore up to `pinecone.max_scan_tuples` rows for a handful, so
prefer `relaxed_order` — see [Iterative scan](docs/reference.md#iterative-scan). For
`engine => 'pgvector'` indexes use pgvector's own `hnsw.iterative_scan`.

Measured on 1M × 768d wiki_dpr embeddings (release Postgres 17, `LIMIT 10`, 500 cross-tenant
queries): a 1% or 10% key the index pushes, plus a 0.3% or 1% residual it cannot.

| Key + residual | `off`: rows / recall | `relaxed_order`: rows / recall | Recall ≥ 0.95 at (p50, 1 / 8 clients) | Exact seqscan (1 / 8 clients) |
|---|---|---|---|---|
| 1% + 0.3% | 3.0 / 0.30 | 10 / 0.76 | `rescore_multiplier` 50 + `scan_fraction` 1.0 (22 / 23 ms) | 17 / 85 ms |
| 1% + 1% | 9.6 / 0.75 | 10 / 0.79 | not reached: 0.94 at `rescore_multiplier` 50 | 20 / 97 ms |
| 10% + 0.3% | 3.4 / 0.32 | 10 / 0.78 | `rescore_multiplier` 100 (34 / 43 ms) | 19 / 90 ms |
| 10% + 1% | 8.5 / 0.67 | 10 / 0.79 | `rescore_multiplier` 50 (21 / 22 ms) | 22 / 103 ms |

Both at `rescore_multiplier` 10, the 0.9.0 default, which 0.10 keeps for a scan with a pushed key. `relaxed_order` filled `LIMIT 10`
in every query (`off`: 0–76%) for 0–7 ms more. Where recall reaches 0.95 (three cells), pg_pinecone
takes 0.95–1.8× the exact sequential scan's time at one client and runs 2–5× faster at eight. That
scan is the plan pgvector's planner picked in three of the four cells; in the fourth its hnsw index
returned 0 rows at the default `hnsw.max_scan_tuples`. Every swept point:
[report](bench/vector/REPORT-residual-filter.md). Knobs and caveats:
[reference](docs/reference.md#measured-a-pushed-key-plus-a-residual-where-1m-x-768d).

### Where pgvector still wins

Unfiltered, in-RAM, the classic ANN-benchmark shape is pgvector's home turf. `native`'s unfiltered
latency at 0.95 recall is still ~1.7× behind pgvector's — short of the ship gate it needs to clear
before it could ever be the default. Until then, **`pgvector` stays the default**: pure similarity
search, no filter, fits in RAM → ask for `pgvector` (or nothing — that's what you get). `native`
earns its keep on filtered, churning workloads first.

## The remote engine

`remote` and the older `promote()`-managed tables write through the same **transactional outbox**
and keep Postgres as the system of record.

- **Consistency.** A committed write queues in `pinecone.outbox` and reaches Pinecone within about
  `pinecone.worker_naptime_ms`. There's no read-your-writes — a query right after a write may not
  see it yet. A rolled-back transaction's queued rows never existed; no orphans. `promote()`
  additionally offers `sync_mode => 'sync'` to drain inline (fail-visible, no lag).
- **Switching.** *Promote* (`native → remote`) never interrupts reads or writes — it serves
  natively and dual-writes until the backfill drains, then flips in one step. *Demote*
  (`remote → native`) returns immediately and queues the rebuild: never unavailable while queued,
  but an availability gap equal to the rebuild once it starts (no `REINDEX CONCURRENTLY` yet — plan
  it into a window). Details: [`docs/runbook.md`](docs/runbook.md#engine-lifecycle-pg_pineconecreate_index--set_engine--promote_index--demote_index).
- **Disaster recovery.** `recover_from_remote(tbl)` rebuilds the vector column from the registered
  index; `repoint(tbl, new_index)` re-points that registration (validating dimension/metric) after
  a backup restore; `backup_readiness(tbl)` is a read-only preflight. Recovery restores only what
  the remote currently holds — never zero-RPO (writes made since the last sync can be lost). See
  [`docs/disaster-recovery.md`](docs/disaster-recovery.md).
- **Detached storage.** `detach_storage(tbl)` NULLs the local vector column in place (every other
  column — and anything you `JOIN` against the table — still returns data) so Pinecone holds the
  only full copy — storage scales
  with Pinecone, not the node; `attach_storage(tbl)` re-hydrates. Requires a fresh verified backup.
  One honest line: **detaching makes Pinecone the only copy of that column — backups are your
  safety net.**
- **The explicit control path.** Want per-row control, arbitrary metadata filters (not just the
  one filter column an index can carry), or `pinecone.embed()`/`rerank()`?
  `track()`/`topk()`/`promote()`/`demote()` predate the engine surface and still work standalone —
  same outbox, same contract.

A backlog that won't drain is either **backing off** (transient, retried) or **parked**
(`outbox_max_attempts` permanent failures) — see [`docs/runbook.md`](docs/runbook.md) for
`outbox_retry`/`outbox_discard` and "no worker" diagnosis.

## Native-engine maintenance

`native` inserts are cheap (nearest-centroid append), but each one leaves the index a little
staler than a fresh build. `pinecone.index_drift(idx)` reports how stale; `schedule_maintenance(idx)`
registers a drift threshold the worker `REINDEX`es past — **only if `pinecone.auto_maintain = on`
(default `off`)**. There's no `REINDEX CONCURRENTLY` here, so a rebuild takes the table's locks for
its duration, and it buffers every row's vector in RAM (~4KB/row at 768d, peaking near double
during clustering) — size memory before enabling `auto_maintain`. A build tunes its probe count on
rows of the table; `WITH (calibration_queries = 'public.real_queries')` tunes it on a table of real
queries instead. Full detail:
[`docs/runbook.md`](docs/runbook.md#local-engine-mode--local-maintenance).

## Full-text search on a table

Say which columns are searchable, once. Pinecone keeps a BM25 document index of them in sync with
the table, and `pinecone.search_docs` searches it by table name. Postgres stays the system of
record. It needs the setup of the `remote` engine: a stored API key and `pg_pinecone` in
`shared_preload_libraries`.

**Freshness.** A committed row is delivered within about one worker interval and is searchable
once Pinecone has indexed it, up to about a minute later. There is no read-your-writes.
`pinecone.flush_docs` waits for delivery, not for indexing. `engine => 'local'` (below) has no delay.

### Declare

```sql
SELECT pinecone.track_docs('articles',
  text_columns     => '["title", "body"]',
  metadata_columns => ARRAY['author', 'published_at']);   -- creates the index, queues a backfill
```

A list of names takes the defaults (English, stemming and stop words on). Pass an object for
control per column, e.g. `'{"title": {"language": "en"}, "sku": {"stemming": false, "stop_words": false}}'`.
The document id is the table's single-column primary key. Only the declared columns leave
Postgres. You must own the table. The call returns once the index exists; existing rows follow in
the background:

```sql
SELECT table_name, docs_backfill_done, docs_backfill_total, docs_queue_state FROM pinecone.status();
SELECT pinecone.flush_docs('articles');   -- true once everything committed so far was delivered
```

### Search

```sql
SELECT a.id, a.title, s.score
FROM pinecone.search_docs('articles', 'postgres vector search', k => 10) s
JOIN articles a ON a.id = s.id::bigint
ORDER BY s.rank;
```

`search_docs` returns `(id text, score real, rank int)`: Pinecone's ids and scores, nothing else.
The join reads the rows under your own privileges and row-level security, and drops a row deleted
since the last sync. You need `SELECT` on the table (and the API key, as for `topk`).

### Filter

```sql
SELECT s.id, s.score
FROM pinecone.search_docs('articles', 'vector search',
       must   => '{"body": {"$match_all": "TensorFlow"}}',    -- text that must be there
       filter => '{"author": {"$eq": "ann"}}',                -- metadata columns
       k => 10) s;

-- Lucene syntax, passed through unchanged
SELECT s.id FROM pinecone.search_docs('articles',
  'title:(postgres AND "vector search"^2) -body:old', mode => 'query_string', k => 10) s;
```

`must` takes `$match_phrase`, `$match_all` and `$match_any` on text columns. Use it for identifiers
such as SKUs and error codes: declare the column with `stemming` and `stop_words` off, and require
`{"sku": {"$match_phrase": "SKU-1234-AB"}}`.

### Per tenant

`track_docs(..., partition_column => 'tenant_id')` gives every value of a NOT NULL column its own
Pinecone namespace (`public.articles:42`): a search ranks inside one tenant (BM25 statistics are per
namespace) and a tenant's documents live apart from the others'. Search with `partition => 42` (an
integer, text or uuid), which such a table requires; the local engine applies it as a condition on
the column. Ids stay unique across the table, and a column with more than 1,000 distinct values
warns: Pinecone limits namespaces per index by plan. `delete_partition('articles', 42)` removes one
tenant's documents. See the [reference](docs/reference.md#partition-column).

```sql
SELECT pinecone.track_docs('articles', text_columns => '["title", "body"]', partition_column => 'tenant_id');
SELECT s.id, s.score FROM pinecone.search_docs('articles', 'invoice overdue', partition => 42) s;
```

### Hybrid

Rank by meaning in the same request: declare a `vector_column`
(`track_docs(..., vector_column => 'embedding')`), then pass `vector` with a `query` and a `field`.
The text is a requirement on that column and the vector ranks what is left.

```sql
SELECT s.id FROM pinecone.search_docs('articles', 'tensorflow', field => 'body',
  vector => (SELECT embedding FROM articles WHERE id = 42), k => 10) s;
```

Or rank by text and by meaning separately and fuse the ranks (reciprocal rank fusion, k = 60).
The vector leg is plain SQL, so this works with any vector engine:

```sql
WITH lex AS (
  SELECT s.id::bigint AS id, s.rank
  FROM pinecone.search_docs('articles', 'growling bear', k => 20) s),
sem AS (
  SELECT id, rank() OVER (ORDER BY embedding <=> $1) AS rank   -- $1: the query embedding
  FROM articles ORDER BY embedding <=> $1 LIMIT 20)
SELECT coalesce(lex.id, sem.id) AS id,
       coalesce(1.0 / (60 + lex.rank), 0) + coalesce(1.0 / (60 + sem.rank), 0) AS score
FROM lex FULL OUTER JOIN sem ON lex.id = sem.id
ORDER BY score DESC LIMIT 5;
```

### Local

```sql
SELECT a.id, a.title, s.score
FROM pinecone.search_docs('articles', 'postgres "vector search"', engine => 'local', k => 10) s
JOIN articles a ON a.id = s.id::bigint
ORDER BY s.rank;
```

The same call, answered by Postgres full-text search: no Pinecone, no sync delay, and it works on a
table that isn't declared if you name the column (`field => 'body'`). The ranking is `ts_rank_cd`,
not BM25, so the order differs from the remote one. It builds no index; for a big table add one,
e.g. `CREATE INDEX ON articles USING gin (to_tsvector('english', coalesce(body, '')))` for
`field => 'body'` (the [reference](docs/reference.md#local-engine-postgres-fts-not-bm25) has the
multi-column form). `mode => 'query_string'`, `must`, `filter` and `vector` are refused here.

### Highlight

```sql
SELECT a.id, pinecone.headline('articles', 'body', a.body, 'postgres vector search') AS snippet
FROM pinecone.search_docs('articles', 'postgres vector search', k => 10) s
JOIN articles a ON a.id = s.id::bigint
ORDER BY s.rank;
```

`headline` is core `ts_headline` with the declared language of the column. It runs in Postgres on
the text you pass, so it works after either engine.

### Engines and full-text search

Which engine answers has nothing to do with the table's vector engine.

| Where the table stands | Engine | Ranking | Query features | Visibility |
|---|---|---|---|---|
| Declared with `track_docs`; `engine => 'remote'` (the default) | Pinecone document index | BM25 | Lucene `query_string`, `must`, `filter`, `vector` | Eventual: delivery, then up to a minute of indexing |
| A `pgvector` or `native` table, or any other, that is not declared; or any table with `engine => 'local'` | Postgres full-text search (`websearch_to_tsquery`) | `ts_rank_cd`, not BM25 | `query` and `field`; `query_string`, `must`, `filter` and `vector` are refused in 0.8 | Immediate |
| Planned | BM25 in Postgres, through `pg_textsearch` or RUM when installed | BM25 | The operator form, `ORDER BY body <@@> 'q'` | Immediate |
| Not scheduled | A BM25 index of our own in the native engine | BM25 | The same as the planned row | Immediate |

Hybrid works with every vector engine because the vector leg is plain `ORDER BY embedding <=> $1`.

### Operate

```sql
SELECT * FROM pinecone.failed_docs('articles');          -- documents Pinecone refused, and why
SELECT pinecone.retry_failed_docs('articles');           -- requeue them
SELECT pinecone.backfill_docs('articles', clear => true); -- re-send every row (after a restore)
SELECT pinecone.untrack_docs('articles', drop_index => true);
```

Declared columns can't be dropped, renamed or retyped until `untrack_docs`. A document Pinecone
refuses is set aside as a dead letter and never stops the table; Pinecone being down or
rate-limiting only delays delivery. The runbook covers
[dead letters, a parked lane, flush and backfill](docs/runbook.md#document-sync-track_docs).

### Ad-hoc indexes

`pinecone.search_doc_index` (called `search_docs` before 0.8) queries any document index directly,
no table needed:

```sql
SELECT pinecone.create_doc_index('articles',
  '{"fields": {"content": {"type": "string", "full_text_search": {"language": "en"}}}}');
SELECT pinecone.upsert_docs('articles',
  '[{"_id": "1", "content": "Postgres vector search with Pinecone", "year": 2026}]');
SELECT id, score FROM pinecone.search_doc_index('articles',
  '{"type": "text", "fields": ["content"], "query": "postgres vector search"}', k => 10);
```

`query` is one Pinecone `score_by` clause (`text`, `query_string`, `dense_vector` or `sparse_vector`)
or an array of them; `filter` takes metadata and text-match filters; `fields` picks the returned
document fields (default: all).

## Reference & further reading

**pg_pinecone docs**

- [`docs/tutorial.md`](docs/tutorial.md) — end-to-end walkthrough: data → `pinecone.embed` →
  index → search, all in SQL.
- [`docs/reference.md`](docs/reference.md) — every GUC and function (mode/engine support) + the
  error-message table.
- [`docs/runbook.md`](docs/runbook.md) — operator runbook: outbox backlog, engine lifecycle,
  maintenance.
- [`docs/disaster-recovery.md`](docs/disaster-recovery.md) — restore → repoint → recover.

**Pinecone documentation** (for the `remote` engine — your data lives in a Pinecone serverless
index)

- [Pinecone docs home](https://docs.pinecone.io) · [Create an index](https://docs.pinecone.io/guides/index-data/create-an-index) · [Indexing overview](https://docs.pinecone.io/guides/index-data/indexing-overview)
- [Filter by metadata](https://docs.pinecone.io/guides/search/filter-by-metadata) — how filter expressions map to the `remote` engine
- [Backups & restore](https://docs.pinecone.io/guides/manage-data/backups-overview) — the safety net behind DR and detached storage
- [API reference](https://docs.pinecone.io/reference/api) · [Rust SDK](https://docs.pinecone.io/reference/sdks/rust/overview) (the client `pg_pinecone` builds on)

## Development

```bash
cargo test -p pg_pinecone-client            # client tests, no Postgres needed
./scripts/install-pgvector.sh            # one-time: pgvector into the pgrx-managed Postgres (pass a pg_config for 18)
cd pg_pinecone && cargo pgrx test pg17      # extension tests, incl. e2e against a mock (pg18 likewise)
PINECONE_API_KEY=... ./scripts/smoke.sh  # live smoke: temp index on real Pinecone (skips without key)
```

## Roadmap

| Version | Status | What |
|---|---|---|
| 0.3 | ✅ Shipped | Transparent index access method (`CREATE INDEX ... USING pinecone`) + outbox/worker sync |
| 0.4 | ✅ Shipped | Disaster recovery (`recover_from_remote` + `repoint` + `backup_readiness`) |
| 0.5 | ✅ Shipped | Storage engines + the `native` engine (IVF, binary sketching, calibrated rescore, filter pushdown) |
| 0.6 | ✅ Shipped | Detached storage — drop the local vector column once a table's fully `remote` |
| 0.7 | ✅ Shipped | The "Pinecone for Postgres" launch — the 0.6.x engine under its product identity |
| 0.8 | ✅ Shipped | Full-text search on a table: BM25 through Pinecone document indexes (`track_docs`/`search_docs`), local tsvector engine |
| 0.9 | ✅ Shipped | Iterative scan for residual filters, filter column pushdown, partition column (one Pinecone namespace per tenant, failed-partition isolation), Postgres 18 |
| 0.10 | ✅ Shipped | Native engine on public algorithms (RaBitQ codes, blocked scan, calibrated defaults, `calibration_queries`), renamed to `pg_pinecone` (the last clean break; in-place upgrades from 0.10.0 on), backfill at Pinecone's request limits, Apache 2.0 |
| 0.11 | Planned | Incremental IVF maintenance; operator form for text search (`ORDER BY body <@@> 'q'`), BM25 locally via pg_textsearch/RUM when installed; precision & hybrid: `halfvec` end-to-end, merge-on-read, sparse + dense in one query |

*One index. Pick the engine that fits today. Change it the day that stops being true.*
