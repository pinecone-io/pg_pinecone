#!/usr/bin/env bash
# usage (on the box; phase `tools`): perf for the phase split, a Rust toolchain (kernels microbench), build tools; MOTD scripts off;
# machine facts (lscpu, kernel, perf) into $STATE. Idempotent.
set -euo pipefail
STATE=${STATE:-/data/state}
APT="sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=900 -y"
sudo chmod -x /etc/update-motd.d/* 2> /dev/null || true
$APT update > /dev/null
$APT install linux-tools-common "linux-tools-$(uname -r)" binutils elfutils numactl clang libclang-dev pkg-config libssl-dev build-essential git curl jq \
    libreadline-dev zlib1g-dev flex bison strace rsync bc python3-venv 2>&1 | tail -3
PERF=$(command -v perf || true)
[ -n "$PERF" ] || PERF=$(ls /usr/lib/linux-tools/*/perf | tail -1)
{ "$PERF" --version; sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0; } | tee "$STATE/perf-version.txt"
if [ ! -x "$HOME/.cargo/bin/rustup" ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
fi
source "$HOME/.cargo/env"
{ rustc --version; cargo --version; } | tee "$STATE/rustc-version.txt"
lscpu > "$STATE/lscpu.txt"; lscpu -e > "$STATE/lscpu-e.txt"; uname -a > "$STATE/uname.txt"
cp /etc/os-release "$STATE/os-release.txt"
dpkg-query -W -f '${Package}=${Version}\n' postgresql-17 postgresql-client-17 postgresql-server-dev-17 postgresql-17-pgvector postgresql-common > "$STATE/pg17-packages.txt" 2> /dev/null || true
echo "tools done"
