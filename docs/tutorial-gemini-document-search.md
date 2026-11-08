# Semantic search over a Postgres table with Gemini — in SQL

> *Nobody searches for a vector.* You search for a meaning, and something turns your
> words into a vector behind the scenes. This tutorial makes that "something" Pinecone's
> hosted **`gemini-embedding-2`** model, and the "behind the scenes" a single
> `pinecone.track_docs(...)` call — no embedding pipeline, no pgvector column, no vectors
> on the wire.

By the end you'll have an ordinary Postgres table that answers three kinds of search:

- **semantic** — "find rows that *mean* this", ranked by Google's `gemini-embedding-2`;
- **keyword** — classic BM25 full-text search;
- **hybrid** — semantic ranking narrowed by an exact filter.

All of it from SQL, with Pinecone embedding your text server-side on every write and every query.

## Prerequisites

- `pg_pinecone` installed and the Pinecone (`remote`) engine available — this feature is
  **remote-only** (more on that below).
- A Pinecone API key on a **Builder, Standard, or Enterprise** plan. `gemini-embedding-2` is
  not available on the Starter plan.

```sql
SELECT pinecone.set_api_key('pcsk_your_key');   -- stored: the background sync worker needs it
```

## Step 1 — a plain table

Nothing special: text columns and whatever metadata you want to filter on later.

```sql
CREATE TABLE notes (
  id       bigint PRIMARY KEY,
  body     text,
  category text
);

INSERT INTO notes (id, body, category) VALUES
  (1, 'PostgreSQL is a relational database that stores tables of rows and columns.', 'database'),
  (2, 'Pinecone is a managed vector database for semantic search over embeddings.',  'database'),
  (3, 'An embedding model turns a piece of text into a numeric vector that captures its meaning.', 'ml'),
  (4, 'BM25 ranks results by exact keyword overlap between the query and the document.', 'search'),
  (5, 'Gemini is a family of large language and embedding models built by Google.',    'ml'),
  (6, 'Hybrid search blends keyword matching with semantic similarity to improve recall.', 'search');
```

## Step 2 — track it with Gemini embedding

One call. The `embed` option on the `body` column is the whole trick: it tells Pinecone to embed
that column's text with `gemini-embedding-2`.

```sql
SELECT pinecone.track_docs('notes',
  text_columns     => '{"body": {"language": "en", "embed": {"model": "gemini-embedding-2"}}}',
  metadata_columns => ARRAY['category']);

SELECT pinecone.flush_docs('notes');   -- wait until the six rows are delivered
```

`track_docs` creates a Pinecone document index whose `body` field carries both a full-text index
*and* a hosted `gemini-embedding-2` embedding, registers triggers so future `INSERT`/`UPDATE`/
`DELETE`s stay in sync, and backfills the rows already there. You never compute or send a vector —
you send `body` text, and Pinecone embeds it.

> **Dimension, task type, etc.** `embed` is forwarded to Pinecone, so you can pin them if you want:
> `{"embed": {"model": "gemini-embedding-2", "dimension": 1024, "write_parameters": {"task_type":
> "retrieval_document"}}}`. Omit them and Pinecone picks sensible defaults (3072 dims, cosine,
> `retrieval_document` on write / `retrieval_query` on read).

## Step 3 — search by meaning

Pass text; Pinecone embeds your *query* with the same model and ranks by cosine similarity. Join the
results back to your table like any other query:

```sql
SELECT n.id, n.body, s.score
FROM pinecone.search_docs('notes', 'finding information by concept instead of the exact words typed',
                          mode => 'embed') s
JOIN notes n ON n.id = s.id::bigint
ORDER BY s.rank;
```

```
 id |                                   body                                    |  score
----+--------------------------------------------------------------------------+---------
  6 | Hybrid search blends keyword matching with semantic similarity ...        |  0.728
  4 | BM25 ranks results by exact keyword overlap ...                           |  0.704
  2 | Pinecone is a managed vector database for semantic search ...             |  0.665
  3 | An embedding model turns a piece of text into a numeric vector ...        |  0.625
```

Notice the query shares almost no *words* with the rows it found — it matched on meaning. (Scores
are illustrative; exact values from a real `gemini-embedding-2` run.)

## Step 4 — keyword search (BM25), same column

The `body` column still has a full-text index, so `mode => 'text'` does classic keyword ranking:

```sql
SELECT n.id, n.body, s.score
FROM pinecone.search_docs('notes', 'relational database rows columns', mode => 'text') s
JOIN notes n ON n.id = s.id::bigint
ORDER BY s.rank;
```

```
 id |                              body                               | score
----+----------------------------------------------------------------+-------
  1 | PostgreSQL is a relational database that stores tables ...      | 5.775
  2 | Pinecone is a managed vector database for semantic search ...   | 1.088
```

BM25 scores (here ~5.8) and semantic scores (~0.7) live on **different scales** — don't compare
them across searches.

## Step 5 — hybrid: rank by meaning, filter exactly

A single request ranks by one method, so blend by putting the *hard* requirement in `filter` and the
*ranking* in `mode => 'embed'`. Here: rank by meaning, but only within `category = 'ml'`:

```sql
SELECT n.id, n.body, s.score
FROM pinecone.search_docs('notes', 'models that understand human language',
                          mode => 'embed',
                          filter => '{"category": {"$eq": "ml"}}') s
JOIN notes n ON n.id = s.id::bigint
ORDER BY s.rank;
```

```
 id |                              body                               | score
----+----------------------------------------------------------------+-------
  3 | An embedding model turns a piece of text into a numeric vector ...| 0.716
  5 | Gemini is a family of large language and embedding models ...   | 0.701
```

You can also narrow with a text requirement instead of metadata — `must => '{"body":
{"$match_all": "search"}}'` keeps only rows containing *search*, still ranked by meaning. To rank by
*both* keyword and meaning, run the two searches and merge by rank client-side.

## The remote-only guard

Semantic search needs the hosted model, which only runs at Pinecone. So asking for it on the local
engine is refused — clearly, and before any call:

```sql
SELECT * FROM pinecone.search_docs('notes', 'anything', mode => 'embed', engine => 'local');
-- ERROR:  pinecone.search_docs: mode => 'embed' (semantic search) is not supported on the
--         local engine — it needs the index's hosted embedding model
-- HINT:   use engine => 'remote'
```

`engine => 'local'` (Postgres full-text search) still works for `mode => 'text'` on the same column —
you just can't run Gemini without Pinecone.

## Alternative: the ad-hoc document API

If your data doesn't live in a Postgres table, drive the document index directly — same `embed`
field, same `type: "embed"` query:

```sql
SELECT pinecone.create_doc_index('kb-gemini', '{"fields": {"chunk_text": {
  "type": "string",
  "embed": {"model": "gemini-embedding-2"},
  "full_text_search": {"language": "en"}
}}}');

SELECT pinecone.upsert_docs('kb-gemini',
  '[{"_id": "1", "chunk_text": "Pinecone is a vector database for semantic search."}]');

SELECT * FROM pinecone.search_doc_index('kb-gemini',
  '{"type": "embed", "fields": ["chunk_text"], "query": "a database for AI"}');
```

## Good to know

- **Freshness is eventual.** `track_docs` sync is asynchronous; `flush_docs` waits for delivery, and
  Pinecone then indexes for up to about a minute. There's no read-your-writes.
- **One dense field per index.** An `embed` column uses the index's single dense slot, so it can't be
  combined with a bring-your-own `vector_column`, and at most one text column may set `embed`.
- **Cleanup.** `SELECT pinecone.untrack_docs('notes', drop_index => true);` stops syncing and
  deletes the Pinecone index.
