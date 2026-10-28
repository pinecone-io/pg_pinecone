#!/usr/bin/env bash
# usage: install-pg_pinecone.sh [tarball]   (env: PG_MAJOR=17|18, default 17; PG_CONFIG; PG_PINECONE_VERSION, default 0.10.0; PG_PINECONE_REPO)
set -euo pipefail
VERSION=${PG_PINECONE_VERSION:-0.10.0}
PG_MAJOR=${PG_MAJOR:-17}
ASSET=pg_pinecone-$VERSION-pg$PG_MAJOR-linux-x86_64.tar.gz
case "$VERSION/$PG_MAJOR" in
    0.10.0/17) SHA256=52167cc7a2883fc14d480ce2f4a2564b87780f63efebf7b3d2c72f72bc8a5ec2 ;;
    0.10.0/18) SHA256=6c7589f7f34f0c0819e767187ed875e796cdb8e77d10e7bdc343a2966ecdcb09 ;;
    0.11.0/17) SHA256=2d0834297085303f66de3bdc3f41c2e5f68d9a60c35e0ad18bc84f717bf6bf86 ;;
    *) echo "no checksum for pg_pinecone $VERSION on PostgreSQL $PG_MAJOR (17 or 18)" >&2; exit 1 ;;
esac
REPO=${PG_PINECONE_REPO:-pinecone-io/pg_pinecone}
URL=https://github.com/$REPO/releases/download/v$VERSION/$ASSET
TARBALL=${1:-$HOME/$ASSET}
PG_CONFIG=${PG_CONFIG:-/usr/lib/postgresql/$PG_MAJOR/bin/pg_config}

if [ ! -f "$TARBALL" ]; then
    curl -fsSL "$URL" -o "$TARBALL"
fi
echo "$SHA256  $TARBALL" | sha256sum -c -

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
tar -xzf "$TARBALL" -C "$STAGE"
one() { local f; f=$(find "$STAGE" -type f -name "$1"); [ "$(printf '%s\n' "$f" | grep -c .)" = 1 ] || { echo "expected one $1 in the tarball" >&2; exit 1; }; echo "$f"; }
lib=$(one pg_pinecone.so)
control=$(one pg_pinecone.control)
sql=$(one "pg_pinecone--$VERSION.sql")
[ "$(sed -n "s/^default_version *= *'\(.*\)'.*/\1/p" "$control")" = "$VERSION" ] || { echo "control is not $VERSION" >&2; exit 1; }

PKGLIBDIR=$("$PG_CONFIG" --pkglibdir)
EXTDIR=$("$PG_CONFIG" --sharedir)/extension
sudo rm -f "$PKGLIBDIR/pg_pinecone.so"
sudo install -m 755 "$lib" "$PKGLIBDIR/"
sudo install -m 644 "$control" "$sql" "$EXTDIR/"
if ldd "$PKGLIBDIR/pg_pinecone.so" | grep -q 'not found'; then ldd "$PKGLIBDIR/pg_pinecone.so" >&2; exit 1; fi
echo "installed pg_pinecone $VERSION: $(sha256sum "$PKGLIBDIR/pg_pinecone.so" | cut -c1-16)... in $PKGLIBDIR, control and sql in $EXTDIR"
