#!/bin/bash
# Downloads the datasets the vector harness reads into $PG_PINECONE_BENCH_DATA/vector/datasets_cache (default bench/.data/...).
#   scripts/download-datasets.sh sift      SIFT1M, 128d: 168 MB download (ftp.irisa.fr) -> sift1m.npz, 521 MB.
#                                          Real-data sanity check (`sift1m:` block) of smoke, full, full_1m, sift_hard_pg_pinecone, followup_c5_sift_k512.
#   scripts/download-datasets.sh wikidpr   wiki_dpr (DPR), 768d: 4 shards, 5.3 GB download -> wikidpr_768d_1m.npz, 3.4 GB (1M base
#                                          + held-out queries). Needs ~9 GB free and ~12 GB RAM while converting. For baselines_exact_real*,
#                                          wikidpr_1m_pg_pinecone, followup_c1_wikidpr and residual-{c1,c2,ext,ext2}.
#   scripts/download-datasets.sh all
# The synthetic datasets (smoke*, residual-smoke, sweep_*, spotcheck_*, full*) are generated on first use: nothing to download.
# Files already present are kept; interrupted downloads resume. Shards stay in wikidpr_shards/ (rm -r to reclaim 5.3 GB).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)          # bench/vector
DATA=${PG_PINECONE_BENCH_DATA:-$HERE/../.data}
CACHE=$DATA/vector/datasets_cache
PY=${PY:-$HERE/.venv/bin/python}
WIKI=https://dl.fbaipublicfiles.com/dpr/data/wiki_encoded/single/nq
WIKI_BYTES=(1324693633 1324804738 1325065657 1325225044)       # wiki_passages_0..3, as served 2026-10-12
mkdir -p "$CACHE"

sift() {
  if [ -f "$CACHE/sift1m.npz" ]; then echo "sift1m.npz present, skipping"; return; fi
  echo "SIFT1M: 168 MB from ftp.irisa.fr (the harness's own downloader), then sift1m.npz (521 MB)"
  (cd "$HERE" && PG_PINECONE_BENCH_DATA="$DATA" "$PY" -c '
import logging; logging.basicConfig(level=logging.INFO)
from harness.sift import try_load_sift1m
d = try_load_sift1m()
raise SystemExit(0 if d is not None else "SIFT1M download or conversion failed")')
}

wikidpr() {
  if [ -f "$CACHE/wikidpr_768d_1m.npz" ]; then echo "wikidpr_768d_1m.npz present, skipping"; return; fi
  mkdir -p "$CACHE/wikidpr_shards"
  for i in 0 1 2 3; do
    f=$CACHE/wikidpr_shards/wiki_passages_$i
    have=$( [ -f "$f" ] && wc -c < "$f" | tr -d ' ' || echo 0 )
    if [ "$have" != "${WIKI_BYTES[$i]}" ]; then
      echo "wiki_dpr shard $i: $((${WIKI_BYTES[$i]} / 1000000)) MB"
      curl -fL --retry 3 -C - -o "$f" "$WIKI/wiki_passages_$i"
    fi
    have=$(wc -c < "$f" | tr -d ' ')
    [ "$have" = "${WIKI_BYTES[$i]}" ] || { echo "shard $i: $have bytes, expected ${WIKI_BYTES[$i]}"; exit 1; }
  done
  echo "converting 4 shards to wikidpr_768d_1m.npz (3.4 GB)"
  "$PY" "$HERE/scripts/convert_wikidpr.py" --shards "$CACHE/wikidpr_shards" --out "$CACHE/wikidpr_768d_1m.npz"
}

case "${1:-}" in
  sift) sift ;;
  wikidpr) wikidpr ;;
  all) sift; wikidpr ;;
  *) sed -n "2,10p" "$0"; exit 2 ;;
esac
