## Checklist

- [ ] Schema change? `pg_pinecone--<prev>--<new>.sql` updated and `upgrade-test` green
- [ ] Touched a `#[pg_extern]`? Its C symbol and call shape stay until the script drops it (docs/contributing.md, Rules)
