#!/bin/bash
# Downloads the datasets the vector harness reads into $PG_PINECONE_BENCH_DATA/vector/datasets_cache (default bench/.data/...).
#   scripts/download-datasets.sh sift       SIFT1M, 128d: 168 MB download (ftp.irisa.fr) -> sift1m.npz, 521 MB.
#                                           Real-data sanity check (`sift1m:` block) of smoke, full, full_1m, sift_hard_pg_pinecone, followup_c5_sift_k512.
#   scripts/download-datasets.sh wikidpr    wiki_dpr (DPR), 768d: 4 shards, 5.3 GB download -> wikidpr_768d_1m.npz, 3.4 GB (1M base
#                                           + held-out queries). Needs ~9 GB free and ~12 GB RAM while converting. For baselines_exact_real*,
#                                           wikidpr_1m_pg_pinecone, followup_c1_wikidpr and residual-{c1,c2,ext,ext2}.
#   scripts/download-datasets.sh wikidpr10m wiki_dpr 10M: shards 0..24 (~33 GB) -> wikidpr_768d_10m.vectors.npy (~31 GB)
#                                           + wikidpr_768d_10m.npz (holdout only). One shard in RAM. Needs ~70 GB free during convert.
#                                           For configs/local_10m.yaml. Not part of `all`.
#   scripts/download-datasets.sh cohere     Cohere wikipedia-22-12 1M (angular), 768d, as published by Tiger Data: 2.9 GB HDF5
#                                           (s3://vector-datasets/1M/..., public GetObject) -> cohere_768d_1m.npz.
#                                           Needs h5py. For residual-cohere-c1 and -c2.
#   scripts/download-datasets.sh cohere10m  Cohere 10M: 28.6 GB HDF5 -> cohere_768d_10m.npz. Put CACHE on
#                                           NVMe at this scale (root EBS is 100 GB). Needs ~70 GB RAM to convert.
#   scripts/download-datasets.sh all
# The synthetic datasets (smoke*, residual-smoke, sweep_*, spotcheck_*, full*) are generated on first use: nothing to download.
# Files already present are kept; interrupted downloads resume. Shards stay in wikidpr_shards/ (rm -r to reclaim 5.3 GB,
# or ~33 GB after wikidpr10m); the Cohere HDF5 stays in cohere/ (rm -r to reclaim 2.9 GB after the npz exists).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)          # bench/vector
DATA=${PG_PINECONE_BENCH_DATA:-$HERE/../.data}
CACHE=$DATA/vector/datasets_cache
PY=${PY:-$HERE/.venv/bin/python}
WIKI=https://dl.fbaipublicfiles.com/dpr/data/wiki_encoded/single/nq
WIKI_BYTES=(1324693633 1324804738 1325065657 1325225044)       # wiki_passages_0..3, as served 2026-10-12
# Shards 0..24: 25 files. Four shards are 1,681,224 vectors (~420k each); 25 are ~10.51M,
# a 10M base plus the 100,660-vector holdout. Shard 50 is not published (0..49 are).
WIKI10_LAST=24
# Public Cohere files (HeadObject 2026-10-24 / 25).
COHERE_URL=https://vector-datasets.s3.amazonaws.com/1M/cohere-wikipedia-22-12-1M-angular.hdf5
COHERE_BYTES=3118728192
COHERE10_URL=https://vector-datasets.s3.amazonaws.com/10M/cohere-wikipedia-22-12-10M-angular.hdf5
COHERE10_BYTES=30743368192
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

fetch_wiki_shard() {
  local i=$1
  local f=$CACHE/wikidpr_shards/wiki_passages_$i
  local url=$WIKI/wiki_passages_$i
  local expect have
  if [ "$i" -lt ${#WIKI_BYTES[@]} ]; then
    expect=${WIKI_BYTES[$i]}
  else
    expect=$(curl -fsI -L --retry 3 --max-time 60 "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:" {v=$2} END {print v}')
    [ -n "${expect:-}" ] || { echo "shard $i: no content-length from $url"; exit 1; }
  fi
  mkdir -p "$CACHE/wikidpr_shards"
  have=$( [ -f "$f" ] && wc -c < "$f" | tr -d ' ' || echo 0 )
  if [ "$have" != "$expect" ]; then
    if [ "$have" != 0 ] && [ "$have" -gt "$expect" ]; then rm -f "$f"; fi
    echo "wiki_dpr shard $i: $((expect / 1000000)) MB"
    curl -fL --retry 3 -C - -o "$f" "$url"
  fi
  have=$(wc -c < "$f" | tr -d ' ')
  [ "$have" = "$expect" ] || { echo "shard $i: $have bytes, expected $expect"; exit 1; }
}

wikidpr() {
  if [ -f "$CACHE/wikidpr_768d_1m.npz" ]; then echo "wikidpr_768d_1m.npz present, skipping"; return; fi
  local i
  for i in 0 1 2 3; do fetch_wiki_shard "$i"; done
  echo "converting 4 shards to wikidpr_768d_1m.npz (3.4 GB)"
  "$PY" "$HERE/scripts/convert_wikidpr.py" --shards "$CACHE/wikidpr_shards" --out "$CACHE/wikidpr_768d_1m.npz"
}

wikidpr10m() {
  local npz=$CACHE/wikidpr_768d_10m.npz
  local npy=$CACHE/wikidpr_768d_10m.vectors.npy
  if [ -f "$npz" ] && [ -f "$npy" ]; then echo "wikidpr 10m present, skipping"; return; fi
  local i
  for i in $(seq 0 "$WIKI10_LAST"); do fetch_wiki_shard "$i"; done
  echo "converting shards 0-$WIKI10_LAST to wikidpr_768d_10m (one shard in RAM)"
  "$PY" "$HERE/scripts/convert_wikidpr.py" --scale 10m --shards "$CACHE/wikidpr_shards" --out "$npz"
}

cohere() {
  if [ -f "$CACHE/cohere_768d_1m.npz" ]; then echo "cohere_768d_1m.npz present, skipping"; return; fi
  "$PY" -c "import h5py" 2>/dev/null || { echo "need h5py: $PY -m pip install h5py"; exit 1; }
  mkdir -p "$CACHE/cohere"
  f=$CACHE/cohere/cohere-wikipedia-22-12-1M-angular.hdf5
  have=$( [ -f "$f" ] && wc -c < "$f" | tr -d ' ' || echo 0 )
  if [ "$have" != "$COHERE_BYTES" ]; then
    echo "Cohere 1M HDF5: $((COHERE_BYTES / 1000000)) MB from vector-datasets (unsigned GetObject)"
    curl -fL --retry 3 -C - -o "$f" "$COHERE_URL"
  fi
  have=$(wc -c < "$f" | tr -d ' ')
  [ "$have" = "$COHERE_BYTES" ] || { echo "cohere hdf5: $have bytes, expected $COHERE_BYTES"; exit 1; }
  echo "converting HDF5 to cohere_768d_1m.npz"
  "$PY" "$HERE/scripts/convert_cohere.py" --scale 1m --hdf5 "$f" --out "$CACHE/cohere_768d_1m.npz"
}

cohere10m() {
  if [ -f "$CACHE/cohere_768d_10m.npz" ]; then echo "cohere_768d_10m.npz present, skipping"; return; fi
  "$PY" -c "import h5py" 2>/dev/null || { echo "need h5py: $PY -m pip install h5py"; exit 1; }
  mkdir -p "$CACHE/cohere"
  f=$CACHE/cohere/cohere-wikipedia-22-12-10M-angular.hdf5
  have=$( [ -f "$f" ] && wc -c < "$f" | tr -d ' ' || echo 0 )
  if [ "$have" != "$COHERE10_BYTES" ]; then
    echo "Cohere 10M HDF5: $((COHERE10_BYTES / 1000000)) MB from vector-datasets (unsigned GetObject)"
    curl -fL --retry 3 -C - -o "$f" "$COHERE10_URL"
  fi
  have=$(wc -c < "$f" | tr -d ' ')
  [ "$have" = "$COHERE10_BYTES" ] || { echo "cohere 10m hdf5: $have bytes, expected $COHERE10_BYTES"; exit 1; }
  echo "converting HDF5 to cohere_768d_10m.npz"
  "$PY" "$HERE/scripts/convert_cohere.py" --scale 10m --hdf5 "$f" --out "$CACHE/cohere_768d_10m.npz"
}

case "${1:-}" in
  sift) sift ;;
  wikidpr) wikidpr ;;
  wikidpr10m) wikidpr10m ;;
  cohere) cohere ;;
  cohere10m) cohere10m ;;
  all) sift; wikidpr; cohere ;;
  *) sed -n "2,16p" "$0"; exit 2 ;;
esac
