#!/usr/bin/env bash
# usage: BOX_JSON=<file> teardown.sh   terminates the instance, deletes the key pair and security group, lists leftovers
set -euo pipefail
PROFILE=${AWS_PROFILE:?set AWS_PROFILE}
REGION=us-east-1
A() { aws --profile "$PROFILE" --region "$REGION" --no-cli-pager "$@"; }
BOX_JSON=${BOX_JSON:?set BOX_JSON to the file provision.sh wrote}
IID=$(jq -r .instance_id "$BOX_JSON"); SG=$(jq -r .sg_id "$BOX_JSON"); KEY=$(jq -r .key_name "$BOX_JSON")
KEY_PATH=$(jq -r .key_path "$BOX_JSON"); KNOWN=$(jq -r .known_hosts "$BOX_JSON")

if [ "$IID" != null ]; then
    A ec2 terminate-instances --instance-ids "$IID" > /dev/null
    A ec2 wait instance-terminated --instance-ids "$IID"
    echo "instance $IID: $(A ec2 describe-instances --instance-ids "$IID" --query 'Reservations[0].Instances[0].State.Name' --output text)"
fi
A ec2 delete-key-pair --key-name "$KEY"
rm -f "$KEY_PATH" "$KNOWN"
for _ in $(seq 1 18); do
    A ec2 delete-security-group --group-id "$SG" 2> /dev/null && break
    sleep 10
done
echo "key pairs left: $(A ec2 describe-key-pairs --query "KeyPairs[?KeyName=='$KEY'].KeyName" --output text)"
echo "security groups left: $(A ec2 describe-security-groups --filters "Name=group-name,Values=$KEY" --query 'SecurityGroups[].GroupId' --output text)"
echo "volumes left: $(A ec2 describe-volumes --filters Name=tag:Project,Values=pg_pinecone-bench --query 'Volumes[].VolumeId' --output text)"
