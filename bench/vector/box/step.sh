#!/usr/bin/env bash
# usage (on the box): step.sh <name> <command...>   detached, log $LOGS/<name>.log, state $STATE/step-<name>.json
set -euo pipefail
[ ! -f "$HOME/.bench-env" ] || source "$HOME/.bench-env"
LOGS=${LOGS:-/data/logs}; STATE=${STATE:-/data/state}
name=$1; shift
f=$STATE/step-$name.json
if [ -f "$f" ] && [ "$(jq -r .status "$f")" = running ] && kill -0 "$(jq -r .pid "$f")" 2> /dev/null; then
    echo "$name: already running (pid $(jq -r .pid "$f"))"
    exit 0
fi
setsid nohup "$(dirname "${BASH_SOURCE[0]}")/step-run.sh" "$name" "$@" > "$LOGS/$name.log" 2>&1 < /dev/null &
sleep 1
echo "$name: started, log $LOGS/$name.log"
