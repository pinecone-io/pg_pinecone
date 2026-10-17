# Contributing to Pinecone for Postgres

Thanks for contributing. This page is the quick start; the detailed rules live in
[docs/contributing.md](docs/contributing.md).

## Setup

PostgreSQL 17 or 18, Rust, and [cargo-pgrx](https://github.com/pgcentralfoundation/pgrx) 0.19.2:

```bash
cargo install cargo-pgrx --version 0.19.2 --locked
cargo pgrx init --pg17=download          # or point --pg17/--pg18 at an existing pg_config
./scripts/install-pgvector.sh            # pgvector into the pgrx-managed Postgres
```

## Test before you push

```bash
cargo test -p pg_pinecone-client            # client crate, no Postgres
cd pg_pinecone && cargo pgrx test pg17      # extension tests (pg18 likewise)
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
```

## Opening a PR

- Target `main`. Keep the title and description short — state the why.
- Fill in the [PR checklist](.github/pull_request_template.md): schema changes need an
  upgrade script with `upgrade-test` green; a touched `#[pg_extern]` keeps its C symbol and
  call shape until a script drops it.
- CI must be green (`pgrx-tests`, `upgrade-test`, `lint`, `client-tests`) before merge.

## Two things to read first

- **Schema or on-disk changes** — the upgrade-script rules and authoring loop in
  [docs/contributing.md](docs/contributing.md).
- **Algorithm work** — the clean-room provenance rules in the same doc: implement from
  published papers, never from another codebase.

By contributing you agree your work is licensed under the [Apache License 2.0](LICENSE).
