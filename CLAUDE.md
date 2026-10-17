# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repo.

## What this is

Pinecone for Postgres (`pg_pinecone`): a Postgres extension that backs one pgvector column
with three interchangeable engines — `pgvector`, `native` (Pinecone's engine in-process),
and `remote` (Pinecone serverless). Rust + [pgrx](https://github.com/pgcentralfoundation/pgrx),
workspace of two crates: `pg_pinecone` (the extension) and `pg_pinecone-client` (pure Rust, no Postgres).

## Build & test

```bash
cargo test -p pg_pinecone-client            # client crate, no Postgres needed
./scripts/install-pgvector.sh            # one-time: pgvector into the pgrx-managed Postgres
cd pg_pinecone && cargo pgrx test pg17      # extension tests incl. e2e vs a mock (pg18 likewise)
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
```

CI required checks: `pgrx-tests-pg17/pg18`, `upgrade-test-pg17/pg18`, `lint`, `client-tests`.
Main requires green CI; don't merge red.

## Invariants that bite

- **Upgrades are in place from 0.10.0 on** (the rename from `pinegres` was the last clean break). Anything that changes what `CREATE EXTENSION`
  builds needs a line in `pg_pinecone/sql/pg_pinecone--<prev>--<new>.sql`, or `upgrade-test` fails.
  Full rules in [docs/contributing.md](docs/contributing.md).
- **Don't break a shipped `#[pg_extern]`.** Its C symbol and call shape must stay until an
  upgrade script drops it. New SQL signature = new Rust function, not a changed one.
- **The restart precedes the update**, so the new library serves the old catalog for a
  window. Code that runs before `ALTER EXTENSION UPDATE` must work without new columns/kinds.
- **On-disk layout changes bump the index format version** (`FORMAT_VERSION` in
  `local/metapage.rs`) and add a reader branch or a refusal.
- **Clean-room provenance.** Algorithm modules are implemented from published papers only —
  no code from any proprietary or third-party implementation. Cite the
  paper in the module header; test against an independent oracle, not another implementation.

## Conventions

- Keep comments and PR text short; state the why in as few words as possible.
- Match the surrounding code's idiom and comment density.
- See [docs/contributing.md](docs/contributing.md) for the upgrade-script authoring loop and
  the full schema-change checklist, and the [runbook](docs/runbook.md) for operational detail.
