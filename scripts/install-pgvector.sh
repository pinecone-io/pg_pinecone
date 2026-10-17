#!/usr/bin/env bash
# Builds and installs pgvector (pinned tag) against a pg_config — the local-mode
# prerequisite for pg_pinecone's tests (`CREATE EXTENSION vector` inside `cargo pgrx
# test pg17`). Usage:
#   ./scripts/install-pgvector.sh [pg_config-path] [--force]
# Defaults to the pgrx-managed pg17's pg_config (same one `cargo pgrx test pg17`
# uses); pass an explicit path (e.g. `$(pg_config)`) to target another install,
# such as CI's PGDG Postgres. Idempotent: skips the build if vector.control is
# already present for the pinned version; pass --force to rebuild anyway.
set -euo pipefail

PGVECTOR_TAG=v0.8.6
PGVECTOR_VERSION="${PGVECTOR_TAG#v}"
DEFAULT_PG_CONFIG="$HOME/.pgrx/17.10/pgrx-install/bin/pg_config"

FORCE=0
PG_CONFIG=""
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        *) PG_CONFIG="$arg" ;;
    esac
done
: "${PG_CONFIG:=$DEFAULT_PG_CONFIG}"

control_file="$("$PG_CONFIG" --sharedir)/extension/vector.control"
if [ "$FORCE" -eq 0 ] && [ -f "$control_file" ] \
    && grep -q "default_version = '$PGVECTOR_VERSION'" "$control_file"; then
    echo "pgvector $PGVECTOR_VERSION already installed ($control_file) — skipping. Use --force to rebuild." >&2
    exit 0
fi

echo "Installing pgvector $PGVECTOR_TAG against $PG_CONFIG" >&2

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
git clone --branch "$PGVECTOR_TAG" --depth 1 https://github.com/pgvector/pgvector.git "$tmpdir/pgvector"
make -C "$tmpdir/pgvector" PG_CONFIG="$PG_CONFIG"
make -C "$tmpdir/pgvector" PG_CONFIG="$PG_CONFIG" install
