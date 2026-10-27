#!/usr/bin/env bash
# usage (on the box, detached through ./step.sh): phase.sh <phase>...
# phases: tools data prep smoke smoke_default smoke_tuned local_1m c1 c2 ext ext_model
#         remote_1m_build remote_1m remote_filtered lane_am_build lane_am lane_topk_build lane_topk partition_build partition cleanup
#         pg18_install   (PostgreSQL 18 and the pg18 tarball next to the 17 of user-data.sh)
#         data_cohere residual_cohere_c1_prep residual_cohere_c1 residual_cohere_c2_prep residual_cohere_c2   (residual cells on Cohere 1M, cosine; PostgreSQL 18 after pg18_install, no Pinecone)
#         hv_install hv_prep hv_smoke hv_1m   (the halfvec session: pg_pinecone built from main, no Pinecone, no key file)
#         hvf_prep hvf_smoke hvf_1m   (filtered halfvec: PostgreSQL 18 and the 0.10.0 pg18 tarball after pg18_install, no Pinecone)
#         overram_prep overram_smoke overram_1m   (search under a memory cap: box/overram-pg.sh, PostgreSQL 17, no Pinecone)
#         data_wikidpr10m local_10m_prep local_10m   (wiki_dpr 10M unfiltered and 1%/10% filtered: configs/local_10m.yaml)
#         settings   (settings-<label>.json of the run; BOX_LABEL required)
# local_1m, c1, c2 and ext run the configs/*-0.10.yaml variants; remote_filtered keeps its index (remote_filtered_keep.yaml).
# No key sits in any environment: remote legs read it from the file $EE_ENV. Finished phases leave $STATE/done-<phase>.
set -euo pipefail
source "$HOME/.bench-env"
V=$HOME/pg_pinecone/bench/vector
cd "$V"
export BENCH_RESULTS_DIR=$RAW/results
PY=$VENV/bin/python
mkdir -p "$BENCH_RESULTS_DIR"

run() { local name=$1 cfg=$2; shift 2; env -u PINECONE_API_KEY "$PY" run.py --config "configs/$cfg.yaml" --data-dir "$CL/pgdata-$name" -v "$@"; }
run18() { PG_PINECONE_BENCH_PG=/usr/lib/postgresql/18 run "$@"; }

phase() {
    case $1 in
        tools) "$V/box/tools.sh" ;;
        data)
            PY=$PY "$V/scripts/download-datasets.sh" wikidpr
            (cd "$PG_PINECONE_BENCH_DATA/vector/datasets_cache" && sha256sum wikidpr_shards/wiki_passages_{0..3} wikidpr_768d_1m.npz > "$STATE/data-sha256.txt")
            ;;
        prep)
            for c in local_1m_0.10 residual-c1-0.10 residual-c2-0.10 residual-ext-0.10 remote_1m remote_filtered_keep remote_partition remote_lanes_am_100k; do
                env -u PINECONE_API_KEY "$PY" run.py --config "configs/$c.yaml" --prep-only -v
            done
            ;;
        smoke) run smoke remote-smoke --allow-dirty ;;
        smoke_default) run smoke remote-smoke --allow-dirty --set postgres/pinecone.worker_naptime_ms=2000 ;;
        smoke_tuned) run smoke remote-smoke --allow-dirty --set postgres/pinecone.worker_naptime_ms=100 ;;
        local_1m) run local1m local_1m_0.10 ;;
        c1) run c1 residual-c1-0.10 ;;
        c2) run c2 residual-c2-0.10 ;;
        ext) run ext residual-ext-0.10 ;;
        ext_model) run extm residual-ext-model ;;
        remote_1m_build) run remote1m remote_1m --stage build ;;
        remote_1m) run remote1m remote_1m --stage measure ;;
        remote_filtered) run remote1m remote_filtered_keep --stage measure ;;
        lane_am_build) run laneam remote_lanes_am_100k --stage build ;;
        lane_am) run laneam remote_lanes_am_100k --stage measure ;;
        lane_topk_build) run lanetopk remote_lanes_topk_100k --stage build ;;
        lane_topk) run lanetopk remote_lanes_topk_100k --stage measure ;;
        partition_build) run part remote_partition --stage build ;;
        partition) run part remote_partition --stage measure ;;
        cleanup) "$PY" box/pc_indexes.py delete-all --yes ;;
        pg18_install) "$V/box/install-pg18.sh" ;;
        data_cohere)
            PY=$PY "$V/scripts/download-datasets.sh" cohere
            (cd "$PG_PINECONE_BENCH_DATA/vector/datasets_cache" && sha256sum cohere/cohere-wikipedia-22-12-1M-angular.hdf5 cohere_768d_1m.npz > "$STATE/data-cohere-sha256.txt")
            ;;
        residual_cohere_c1_prep) env -u PINECONE_API_KEY "$PY" run.py --config configs/residual-cohere-c1.yaml --prep-only -v ;;
        residual_cohere_c1) run18 rc1cohere residual-cohere-c1 ;;
        residual_cohere_c2_prep) env -u PINECONE_API_KEY "$PY" run.py --config configs/residual-cohere-c2.yaml --prep-only -v ;;
        residual_cohere_c2) run18 rc2cohere residual-cohere-c2 ;;
        hv_install)
            # pg_pinecone from main (~/pg_pinecone-src and its COMMIT, uploaded as box/README.md "Halfvec session" says) over the release
            # tarball: the workspace's cargo-pgrx pin under its own --root, x86-64-v3 as the release build; commit and sha256 to $STATE
            SRC=$HOME/pg_pinecone-src PGC=$PG_PINECONE_BENCH_PG/bin/pg_config
            PGRX=$(sed -n 's/^pgrx = "=\([0-9.]*\)".*/\1/p' "$SRC/pg_pinecone/Cargo.toml")
            CPGRX=$HOME/.cargo-pgrx-$PGRX/bin/cargo-pgrx
            sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=900 -y install clang libclang-dev pkg-config libssl-dev
            [ -x "$HOME/.cargo/bin/rustup" ] || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
            source "$HOME/.cargo/env"
            [ -x "$CPGRX" ] || cargo install cargo-pgrx --version "$PGRX" --locked --root "$HOME/.cargo-pgrx-$PGRX"
            "$CPGRX" pgrx init --pg17 "$PGC"
            sudo chown -R "$(whoami)" "$("$PGC" --pkglibdir)" "$("$PGC" --sharedir)/extension"
            (cd "$SRC/pg_pinecone" && RUSTFLAGS='-C target-cpu=x86-64-v3' "$CPGRX" pgrx install --release --pg-config "$PGC")
            echo "pg_pinecone $(cat "$SRC/COMMIT") $(sha256sum "$("$PGC" --pkglibdir)/pg_pinecone.so" | cut -d' ' -f1)" | tee "$STATE/pg_pinecone-build.txt"
            ;;
        hv_prep) env -u PINECONE_API_KEY "$PY" run.py --config configs/halfvec_1m.yaml --prep-only -v ;;
        hv_smoke) run hvsmoke smoke-halfvec-native ;;
        hv_1m) run hv1m halfvec_1m ;;
        hvf_prep) env -u PINECONE_API_KEY "$PY" run.py --config configs/halfvec_filtered_1m_0.10.yaml --prep-only -v ;;
        hvf_smoke) run18 hvfsmoke smoke-halfvec-filtered ;;
        hvf_1m) run18 hvf1m halfvec_filtered_1m_0.10 ;;
        overram_prep) env -u PINECONE_API_KEY "$PY" run.py --config configs/overram_1m.yaml --prep-only -v ;;
        overram_smoke) OVERRAM_SHARED_BUFFERS=64MB "$V/box/overram-pg.sh" start && env -u PINECONE_API_KEY "$PY" run.py --config configs/smoke-overram.yaml -v ;;
        overram_1m)
            for cap in infinity 4G; do
                "$V/box/overram-pg.sh" start
                OVERRAM_CAP=$cap env -u PINECONE_API_KEY "$PY" run.py --config configs/overram_1m.yaml --set "name=overram_1m_cap_$cap" -v
            done
            "$V/box/overram-pg.sh" stop
            ;;
        data_wikidpr10m)
            PY=$PY "$V/scripts/download-datasets.sh" wikidpr10m
            (cd "$PG_PINECONE_BENCH_DATA/vector/datasets_cache" && sha256sum wikidpr_768d_10m.vectors.npy wikidpr_768d_10m.npz > "$STATE/data-wikidpr10m-sha256.txt")
            ;;
        local_10m_prep) env -u PINECONE_API_KEY "$PY" run.py --config configs/local_10m.yaml --prep-only -v ;;
        local_10m) run local10m local_10m ;;
        settings) "$PY" "$V/box/settings.py" "${BOX_LABEL:?set BOX_LABEL}" ;;
        *) echo "unknown phase $1" >&2; return 2 ;;
    esac
}

for p in "$@"; do
    mark=$STATE/done-$p
    if [ -f "$mark" ]; then echo "== $p: done earlier, skipped"; continue; fi
    echo "== $(date '+%F %T') $p"
    phase "$p"
    date '+%F %T' > "$mark"
done
echo "== phase.sh finished: $*"
