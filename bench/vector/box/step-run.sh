#!/usr/bin/env bash
[ ! -f "$HOME/.bench-env" ] || source "$HOME/.bench-env"
LOGS=${LOGS:-/data/logs}; STATE=${STATE:-/data/state}
name=$1; shift
cmd="$*"
f=$STATE/step-$name.json
started=$(date -u +%FT%TZ)
record() { jq -n --arg name "$name" --arg status "$1" --arg started "$started" --arg ended "$2" --arg rc "$3" --argjson pid $$ --arg cmd "$cmd" \
    '{name:$name, status:$status, pid:$pid, started:$started, ended:$ended, rc:$rc, cmd:$cmd}' > "$f.tmp" && mv "$f.tmp" "$f"; }
record running "" ""
nice -n 10 "$@"
rc=$?
status=failed
[ $rc = 0 ] && status='done'
record "$status" "$(date -u +%FT%TZ)" "$rc"
exit $rc
