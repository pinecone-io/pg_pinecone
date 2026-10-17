#!/usr/bin/env bash
# Runs the live tests against a real Pinecone project: pinecone.embed, the
# promote/topk/demote round trip and the document-index round trip
# (track_docs/search_docs/untrack_docs).
# Each skips itself (prints "SKIPPED") unless PG_PINECONE_LIVE_TESTS=1 (set below: running this script is
# the opt-in) and PINECONE_API_KEY are both set.
set -euo pipefail
export PG_PINECONE_LIVE_TESTS=1

if [ -z "${PINECONE_API_KEY:-}" ]; then
    echo "PINECONE_API_KEY is not set — the live tests will run but immediately skip." >&2
    echo "Set it to run them against a real Pinecone project." >&2
fi

cd "$(dirname "$0")/../pg_pinecone"
cargo pgrx test pg17 live_embed
cargo pgrx test pg17 live_promote_roundtrip
cargo pgrx test pg17 live_docs_roundtrip
# Each of these needs its own isolated invocation (fresh cluster, one test):
# pinecone.worker_enabled = on, which pg_test::postgresql_conf_options only
# sets cluster-wide when this variable is present — never for the shared
# deterministic suite, which would otherwise race a real worker.
PG_PINECONE_TEST_ENABLE_WORKER=1 cargo pgrx test pg17 live_worker_drain
PG_PINECONE_TEST_ENABLE_WORKER=1 cargo pgrx test pg17 live_am_roundtrip
