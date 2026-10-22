# Contributing to pg_pinecone

Thanks for your interest in pg_pinecone. Pinecone maintains the extension and its roadmap, and
contributions start with an issue.

## Ask a question or share feedback

Join the [Pinecone Discord community](https://discord.gg/tJ8V62S3sH). Questions, ideas, and
feedback on pg_pinecone are welcome there.

## Report a bug

Open a [GitHub issue](https://github.com/pinecone-io/pg_pinecone/issues) and include:

- The pg_pinecone version (`SELECT extversion FROM pg_extension WHERE extname = 'pg_pinecone';`)
- The PostgreSQL version (`SELECT version();`) and your operating system
- The table's engine (`pgvector`, `native`, or `remote`)
- The steps to reproduce and the full error message

To report a security issue, don't open an issue. Follow [SECURITY.md](SECURITY.md) instead.

## Propose a change

1. Open an issue that describes the problem and the change you'd like to make.
2. A maintainer replies, and you agree on the approach in the issue.
3. If the change goes ahead, a maintainer either makes it or gives you access to open a pull
   request for it.

Maintainers review pull requests only for changes agreed in an issue. This keeps the roadmap and
the upgrade path consistent across releases.

## Set up a development environment

This section is for contributors a maintainer has asked to open a pull request. To install
pg_pinecone and use it, follow the [quickstart](https://docs.pinecone.io/guides/pg-pinecone/quickstart) instead.

You need PostgreSQL 17 or 18, Rust, and [cargo-pgrx](https://github.com/pgcentralfoundation/pgrx) 0.19.2:

```bash
cargo install cargo-pgrx --version 0.19.2 --locked
cargo pgrx init --pg17=download          # or point --pg17/--pg18 at an existing pg_config
./scripts/install-pgvector.sh            # pgvector into the pgrx-managed Postgres
```

Before you push:

```bash
cargo test -p pg_pinecone-client            # client crate, no Postgres
cd pg_pinecone && cargo pgrx test pg17      # extension tests (pg18 likewise)
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
```

When you open the pull request:

- Target `main`, link the issue, and keep the title and description short.
- Fill in the [PR checklist](.github/pull_request_template.md). Schema changes need an upgrade
  script with `upgrade-test` green, and a touched `#[pg_extern]` keeps its C symbol and call shape
  until a script drops it.
- CI must be green (`pgrx-tests`, `upgrade-test`, `lint`, `client-tests`) before merge.

Two kinds of changes have extra rules. If your change touches the schema or on-disk format, read the
[upgrade-script rules](docs/contributing.md#schema-changes-and-upgrade-scripts). If it touches
the search algorithms, read the [provenance rules](docs/contributing.md#provenance): implement from
published papers, never from another codebase.

## License

By contributing, you agree that your work is licensed under the [Apache License 2.0](LICENSE).
