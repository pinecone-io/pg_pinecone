# Tutorial: data → embedding → search

A full round trip in SQL: load some text, embed it with Pinecone's hosted models through
`pinecone.embed`, index it, and search — the query embedded the same way. No application code, no
separate embedding service. Uses the `native` engine, so after setup nothing leaves Postgres on the
read path.

Prerequisites: pgvector and `pg_pinecone` installed ([Setup](../README.md#setup)) and a Pinecone API
key for the embedding calls (`embed` calls Pinecone Inference; `native` search itself needs no
account).

## 1. Set the API key

`embed` is the only step that talks to Pinecone. Point it at a key for this session:

```sql
SET pinecone.api_key = 'your-api-key';
```

> Prefer a stored key — `SELECT pinecone.set_api_key('your-api-key');` — when you want it to
> outlive the session or be usable by the background worker. See [reference](reference.md#api-key).

## 2. A table with a vector column

`multilingual-e5-large` (the `embed` default) returns 1024 dimensions, so the column is
`vector(1024)`:

```sql
CREATE EXTENSION IF NOT EXISTS vector;    -- pgvector first: pg_pinecone builds on it
CREATE EXTENSION IF NOT EXISTS pg_pinecone;

CREATE TABLE docs (
    id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    body      text,
    embedding vector(1024)
);
```

## 3. Embed the data

`pinecone.embed(text)` returns `real[]`; cast it to `vector` to store it. Use
`input_type => 'passage'` for stored documents (asymmetric models encode documents and queries
differently):

```sql
INSERT INTO docs (body, embedding)
SELECT t.body, pinecone.embed(t.body, input_type => 'passage')::vector
FROM (VALUES
        ('Postgres is a relational database'),
        ('Pinecone is a vector database'),
        ('pgvector adds vector search to Postgres')
     ) AS t(body);
```

For a bulk load, one round trip per row is wasteful — `embed_batch` encodes an array and chunks the
calls for you, returning a 1-based `ordinal` to join back on:

```sql
WITH corpus(body, ordinal) AS (
    SELECT * FROM unnest(ARRAY[
        'Postgres is a relational database',
        'Pinecone is a vector database',
        'pgvector adds vector search to Postgres'
    ]) WITH ORDINALITY
)
INSERT INTO docs (body, embedding)
SELECT c.body, e.embedding::vector
FROM corpus c
JOIN pinecone.embed_batch(
        (SELECT array_agg(body ORDER BY ordinal) FROM corpus),
        input_type => 'passage'
     ) AS e ON e.ordinal = c.ordinal;
```

## 4. Build the index

One call turns the column into a `native` index. Searches are unchanged — the engine is a
lifecycle detail, not a query rewrite:

```sql
SELECT pinecone.create_index('docs', 'embedding', metric => 'cosine');
```

> A NOTICE may suggest `ALTER TABLE docs ALTER COLUMN embedding SET STORAGE PLAIN`: a 1024-d vector
> is stored out of line by default, and inline is faster to rescore. Optional here.
>
> `engine` defaults to `'native'`. Pass `engine => 'pgvector'` for a plain HNSW index, or `'remote'` to
> push the vectors to Pinecone serverless. The search in step 5 is identical for all three. See
> [Which engine, when](../README.md#which-engine-when).

## 5. Search

Embed the query — `input_type` now defaults to `'query'`, the right half of the asymmetric
pair — and order by cosine distance (`<=>`). Plain pgvector SQL:

```sql
SELECT id, body
FROM docs
ORDER BY embedding <=> pinecone.embed('which database stores vectors?')::vector
LIMIT 5;
```

```
 id |                   body
----+------------------------------------------
  2 | Pinecone is a vector database
  3 | pgvector adds vector search to Postgres
  1 | Postgres is a relational database
```

### Let the index serve the query

The one-liner above is exact and fine for getting started, but `pinecone.embed` is `VOLATILE` — so
with the call inline in `ORDER BY`, the planner does a full exact scan rather than the ANN index.
That's correct, just not what you want once the table is large. Embed the query *once* and pass the
vector as a constant (a bound `$1` from your app, or `\gset` in psql) so the index can order by it:

```sql
-- psql: embed once, capture the literal, then search
SELECT pinecone.embed('which database stores vectors?')::vector::text AS qv \gset
SELECT id, body FROM docs ORDER BY embedding <=> :'qv'::vector LIMIT 5;

EXPLAIN SELECT id, body FROM docs ORDER BY embedding <=> :'qv'::vector LIMIT 5;
-- Index Scan using docs_embedding_pinecone_idx on docs
```

> On a handful of rows the planner still prefers a sequential scan by cost — the index only wins
> once the table is big enough to earn it. `SET enable_seqscan = off` forces it for a quick check.

## Three things that trip people up

- **Cast the result.** `embed`/`embed_batch` return `real[]`; a `vector(n)` column and the `<=>`
  operator need `::vector`.
- **`input_type` is asymmetric.** `'passage'` for stored rows, `'query'` for the search text. The
  defaults already match (`embed_batch` → `'passage'`, `embed` → `'query'`), so the common case is
  free — but override it if you embed a query with `embed_batch` or a document with `embed`.
- **Embed the query as a constant to use the index.** `embed` is `VOLATILE`; inline in `ORDER BY`
  it forces an exact scan. Embed once, pass the vector as `$1` / a literal (above).

## Where next

- Filter, scale to `remote`, per-tenant namespaces — [Examples](../README.md#examples).
- Full-text (BM25) search on the same table — [`search_docs`](../README.md#full-text-search-on-a-table).
- Every function, GUC, and model option — [`docs/reference.md`](reference.md).
