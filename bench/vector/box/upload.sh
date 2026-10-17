#!/usr/bin/env bash
# usage: BOX_JSON=<file> [EE_ENV=<key file>] TARBALL=<release tarball> [UPLOAD_PATHS="bench/vector"] upload.sh   (run inside the repo checkout; committed paths only)
# EE_ENV is only for legs that call Pinecone; without it no key file goes to the box.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/conn.sh"
ROOT=$(git -C "$HERE" rev-parse --show-toplevel)
SHA=$(git -C "$ROOT" rev-parse HEAD)
EE_ENV=${EE_ENV:-}
TARBALL=${TARBALL:?path of the pg_pinecone release tarball}
PATHS_=${UPLOAD_PATHS:-bench/vector}
[ -z "$(git -C "$ROOT" status --porcelain -- $PATHS_)" ] || { echo "uncommitted changes in $PATHS_: commit first" >&2; exit 1; }

if [ -z "${FORCE:-}" ] && bssh 'for f in /data/state/step-*.json; do [ -f "$f" ] && [ "$(jq -r .status "$f")" = running ] && exit 0; done; exit 1' 2> /dev/null; then
    echo "a step is running on the box and would pick up the new code mid-chain: wait, or FORCE=1" >&2
    exit 1
fi
DIR=pg_pinecone-${SHA:0:9}
git -C "$ROOT" archive --format=tar "$SHA" $PATHS_ | bssh "mkdir -p ~/$DIR && tar -x -C ~/$DIR && for p in $PATHS_; do echo $SHA > ~/$DIR/\$p/HARNESS_SHA; done && ln -sfn $DIR ~/pg_pinecone"
bscp "$TARBALL" "ubuntu@$BOX_IP:"
if [ -n "$EE_ENV" ]; then
    bscp "$EE_ENV" "ubuntu@$BOX_IP:ee.env"
    bssh 'chmod 600 ~/ee.env'
fi
echo "uploaded $PATHS_ at $SHA to ~/$DIR (symlink ~/pg_pinecone)"
