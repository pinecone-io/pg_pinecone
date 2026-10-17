#!/bin/bash
# Readable from instance metadata: no secrets here.
set -euxo pipefail
exec > >(tee -a /var/log/pg_pinecone-bench-userdata.log) 2>&1
export DEBIAN_FRONTEND=noninteractive

PG_VER=17.11-1.pgdg24.04+2
PGVECTOR_VER=0.8.7-1.pgdg24.04+1
PGCOMMON_VER=293.pgdg24.04+1
APT="apt-get -o DPkg::Lock::Timeout=900 -y"

systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer || true

install -d -m 755 /etc/apt/keyrings
curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc -o /etc/apt/keyrings/pgdg.asc
echo "deb [signed-by=/etc/apt/keyrings/pgdg.asc] https://apt.postgresql.org/pub/repos/apt noble-pgdg main" > /etc/apt/sources.list.d/pgdg.list
$APT update
$APT install postgresql-common=$PGCOMMON_VER postgresql-client-common=$PGCOMMON_VER
sed -i 's/^#\?create_main_cluster.*/create_main_cluster = false/' /etc/postgresql-common/createcluster.conf
$APT install postgresql-17=$PG_VER postgresql-client-17=$PG_VER postgresql-server-dev-17=$PG_VER postgresql-17-pgvector=$PGVECTOR_VER
$APT install build-essential python3-venv python3-dev git jq curl unzip rsync sysstat tmux htop
systemctl disable --now postgresql || true

cat > /etc/security/limits.d/bench.conf <<'EOF_LIMITS'
ubuntu soft nofile 65536
ubuntu hard nofile 65536
EOF_LIMITS

cat > /usr/local/sbin/bench-nvme.sh <<'EOF_NVME'
#!/bin/bash
# The instance store is empty after every stop/start, so format on demand; logs, state and raw results live on EBS.
set -eu
dev=$(ls /dev/disk/by-id/nvme-Amazon_EC2_NVMe_Instance_Storage_* 2>/dev/null | grep -v -- '-part' | head -1 || true)
[ -n "$dev" ] || { echo "no instance-store NVMe" >&2; exit 1; }
dev=$(readlink -f "$dev")
mkdir -p /data
if ! mountpoint -q /data; then
    blkid "$dev" > /dev/null 2>&1 || mkfs.ext4 -F -q -m 0 "$dev"
    mount -o noatime "$dev" /data
fi
install -d -o ubuntu -g ubuntu /var/bench/logs /var/bench/state /var/bench/raw /data/cl
chown ubuntu:ubuntu /data
for d in logs state raw; do ln -sfn /var/bench/$d /data/$d; done
EOF_NVME
chmod 755 /usr/local/sbin/bench-nvme.sh
cat > /etc/systemd/system/bench-nvme.service <<'EOF_UNIT'
[Unit]
Description=Format and mount the instance-store NVMe at /data
After=local-fs.target
[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/bench-nvme.sh
[Install]
WantedBy=multi-user.target
EOF_UNIT
systemctl daemon-reload
systemctl enable --now bench-nvme.service

install -d /opt/pg_pinecone-bench
cat > /opt/pg_pinecone-bench/postgresql.bench.conf <<'EOF_CONF'
@@POSTGRESQL_BENCH_CONF@@
EOF_CONF

shutdown -h +4320 "pg_pinecone-bench: 72 h limit"
dpkg-query -W -f '${Package} ${Version}\n' postgresql-17 postgresql-client-17 postgresql-server-dev-17 postgresql-17-pgvector postgresql-common > /var/lib/pg_pinecone-bench-ready
date -u +%FT%TZ >> /var/lib/pg_pinecone-bench-ready
