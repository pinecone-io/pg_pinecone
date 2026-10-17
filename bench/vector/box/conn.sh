#!/usr/bin/env bash
BOX_JSON=${BOX_JSON:?set BOX_JSON to the file provision.sh wrote}
BOX_IP=$(jq -r .public_ip "$BOX_JSON")
BOX_KEY=$(jq -r .key_path "$BOX_JSON")
BOX_KNOWN=$(jq -r .known_hosts "$BOX_JSON")
SSH_OPTS=(-i "$BOX_KEY" -o IdentitiesOnly=yes -o UserKnownHostsFile="$BOX_KNOWN" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30)
bssh() { ssh "${SSH_OPTS[@]}" "ubuntu@$BOX_IP" "$@"; }
bscp() { scp -q "${SSH_OPTS[@]}" "$@"; }
