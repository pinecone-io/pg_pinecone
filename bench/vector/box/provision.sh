#!/usr/bin/env bash
# usage: provision.sh   (env: AWS_PROFILE and BOX_OWNER required; BOX_DATE, AZ, AMI_ID=<id>|latest, BOX_JSON, KEY_PATH)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROFILE=${AWS_PROFILE:?set AWS_PROFILE}
REGION=us-east-1
AZ=${AZ:-us-east-1a}
DATE=${BOX_DATE:-$(date +%F)}
NAME=pg_pinecone-bench-$DATE
INSTANCE_TYPE=r6id.4xlarge
ROOT_GB=100
AMI_PINNED=ami-0fa5967347d08d2df
AMI_ID=${AMI_ID:-$AMI_PINNED}
AMI_SSM=/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id
KEY_PATH=${KEY_PATH:-$HOME/.ssh/$NAME.pem}
export BOX_JSON=${BOX_JSON:-$HOME/.pg_pinecone-bench/box-$DATE.json}
KNOWN_HOSTS=$HOME/.pg_pinecone-bench/known_hosts-$DATE
OWNER=${BOX_OWNER:?set BOX_OWNER (the Owner tag value)}
TAGS="{Key=Name,Value=$NAME},{Key=Project,Value=pg_pinecone-bench},{Key=Owner,Value=$OWNER},{Key=TTL,Value=3d}"

A() { aws --profile "$PROFILE" --region "$REGION" --no-cli-pager "$@"; }
mkdir -p "$(dirname "$BOX_JSON")" "$(dirname "$KNOWN_HOSTS")"

A sts get-caller-identity > /dev/null || { echo "AWS login for profile $PROFILE failed: aws sso login, then retry" >&2; exit 1; }
[ ! -e "$KEY_PATH" ] || { echo "$KEY_PATH exists: run teardown.sh first" >&2; exit 1; }

if [ "$AMI_ID" = latest ]; then
    AMI_ID=$(A ssm get-parameter --name "$AMI_SSM" --query Parameter.Value --output text)
fi
read -r AMI_NAME AMI_ARCH < <(A ec2 describe-images --image-ids "$AMI_ID" --query 'Images[0].[Name,Architecture]' --output text)
[ "$AMI_ARCH" = x86_64 ] || { echo "$AMI_ID is $AMI_ARCH, need x86_64" >&2; exit 1; }

VPC=$(A ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SUBNET=$(A ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC" "Name=availability-zone,Values=$AZ" Name=default-for-az,Values=true \
    --query 'Subnets[0].SubnetId' --output text)
MYIP=$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')

umask 077
A ec2 create-key-pair --key-name "$NAME" --key-type ed25519 --query KeyMaterial --output text > "$KEY_PATH" || { rm -f "$KEY_PATH"; exit 1; }
chmod 600 "$KEY_PATH"
SG=$(A ec2 create-security-group --group-name "$NAME" --description "pg_pinecone bench: SSH from one IP" --vpc-id "$VPC" \
    --tag-specifications "ResourceType=security-group,Tags=[$TAGS]" --query GroupId --output text)
A ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP/32" > /dev/null
jq -n --arg sg "$SG" --arg key "$NAME" --arg key_path "$KEY_PATH" --arg known "$KNOWN_HOSTS" \
    '{instance_id:null, sg_id:$sg, key_name:$key, key_path:$key_path, known_hosts:$known}' > "$BOX_JSON"

UD=$(mktemp)
awk -v conf="$HERE/postgresql.bench.conf" '/^@@POSTGRESQL_BENCH_CONF@@$/ { while ((getline l < conf) > 0) print l; close(conf); next } { print }' \
    "$HERE/user-data.sh" > "$UD"
if grep -q '@@POSTGRESQL_BENCH_CONF@@' "$UD"; then echo "conf marker not replaced" >&2; exit 1; fi

IID=$(A ec2 run-instances \
    --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" --count 1 \
    --key-name "$NAME" --security-group-ids "$SG" --subnet-id "$SUBNET" --associate-public-ip-address \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$ROOT_GB,VolumeType=gp3,DeleteOnTermination=true}" \
    --metadata-options HttpTokens=required,HttpEndpoint=enabled \
    --instance-initiated-shutdown-behavior stop \
    --user-data "file://$UD" \
    --tag-specifications "ResourceType=instance,Tags=[$TAGS]" "ResourceType=volume,Tags=[$TAGS]" \
    --query 'Instances[0].InstanceId' --output text)
rm -f "$UD"
jq --arg iid "$IID" '.instance_id = $iid' "$BOX_JSON" > "$BOX_JSON.tmp" && mv "$BOX_JSON.tmp" "$BOX_JSON"
echo "launched $IID ($INSTANCE_TYPE, $AMI_ID $AMI_NAME, $AZ)"

A ec2 wait instance-status-ok --instance-ids "$IID"
read -r IP LAUNCH < <(A ec2 describe-instances --instance-ids "$IID" --query 'Reservations[0].Instances[0].[PublicIpAddress,LaunchTime]' --output text)
jq -n --arg date "$DATE" --arg iid "$IID" --arg ip "$IP" --arg az "$AZ" --arg type "$INSTANCE_TYPE" --arg ami "$AMI_ID" --arg ami_name "$AMI_NAME" \
    --arg sg "$SG" --arg key "$NAME" --arg key_path "$KEY_PATH" --arg known "$KNOWN_HOSTS" --arg launch "$LAUNCH" --arg region "$REGION" \
    '{date:$date, region:$region, az:$az, instance_type:$type, instance_id:$iid, public_ip:$ip, ami_id:$ami, ami_name:$ami_name, sg_id:$sg,
      key_name:$key, key_path:$key_path, known_hosts:$known, launch_time:$launch}' > "$BOX_JSON"

source "$HERE/conn.sh"
for _ in $(seq 1 120); do
    bssh 'test -f /var/lib/pg_pinecone-bench-ready' 2> /dev/null && break
    sleep 10
done
bssh 'cat /var/lib/pg_pinecone-bench-ready' || { bssh 'tail -40 /var/log/pg_pinecone-bench-userdata.log' || true; exit 1; }
echo "box ready: $BOX_JSON"
