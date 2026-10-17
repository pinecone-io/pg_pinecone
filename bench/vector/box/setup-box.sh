#!/usr/bin/env bash
# usage (on the box, after upload.sh and install-pg_pinecone.sh): setup-box.sh
set -euo pipefail
V=${V:-$HOME/pg_pinecone/bench/vector}
cat > "$HOME/.bench-env" <<ENV
export PG_PINECONE_BENCH_PG=/usr/lib/postgresql/17
export PG_PINECONE_BENCH_DATA=\$HOME/bench-data
export BENCH_TUNED_CONF=$V/pg/tuned.linux-aws.conf
export LOGS=/data/logs RAW=/data/raw STATE=/data/state CL=/data/cl
export EE_ENV=\$HOME/ee.env
export PYTHONUNBUFFERED=1
export VENV=\$HOME/bench-data/vector/venv
export PATH="\$VENV/bin:\$PATH"
ENV
source "$HOME/.bench-env"
mkdir -p "$PG_PINECONE_BENCH_DATA/vector" "$CL" "$LOGS" "$STATE" "$RAW"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -r "$V/requirements.txt"
"$VENV/bin/pip" freeze > "$STATE/pip-freeze.txt"
[ ! -f "$EE_ENV" ] || chmod 600 "$EE_ENV"
"$PG_PINECONE_BENCH_PG/bin/postgres" --version
ls "$("$PG_PINECONE_BENCH_PG/bin/pg_config" --pkglibdir)"/{vector,pg_pinecone}.so
pgrep -x sar > /dev/null || (command -v sar > /dev/null && setsid nohup sar -u -r 10 > "$LOGS/sar.log" 2>&1 < /dev/null &) || true
echo "setup-box done"
