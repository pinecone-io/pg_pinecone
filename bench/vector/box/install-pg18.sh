#!/usr/bin/env bash
# usage (on the box; detached through step.sh): install-pg18.sh
# PostgreSQL 18 from PGDG next to the 17 of user-data.sh, pgvector for it and the
# pg_pinecone pg18 tarball (0.10.0 by default, PG_PINECONE_VERSION; sha256-checked by install-pg_pinecone.sh), so hnsw and native run on one server.
set -euo pipefail
PG_VER=18.6-1.pgdg24.04+2
PGVECTOR_VER=0.8.7-1.pgdg24.04+1
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARBALL=${TARBALL:-$HOME/pg_pinecone-${PG_PINECONE_VERSION:-0.10.0}-pg18-linux-x86_64.tar.gz}
STATE=${STATE:-/data/state}

sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=900 -y install \
    postgresql-18=$PG_VER postgresql-client-18=$PG_VER postgresql-server-dev-18=$PG_VER postgresql-18-pgvector=$PGVECTOR_VER
sudo systemctl disable --now postgresql || true
PG_MAJOR=18 "$HERE/install-pg_pinecone.sh" "$TARBALL"
dpkg-query -W -f '${Package}=${Version}\n' postgresql-18 postgresql-client-18 postgresql-server-dev-18 postgresql-18-pgvector > "$STATE/pg18-packages.txt"
cat "$STATE/pg18-packages.txt"
/usr/lib/postgresql/18/bin/postgres --version
sha256sum /usr/lib/postgresql/18/lib/pg_pinecone.so | tee "$STATE/pg_pinecone-pg18-so.sha256"
echo "install-pg18 done"
