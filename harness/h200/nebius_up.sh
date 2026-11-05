#!/bin/bash
# on a Nebius VM host (harness/h200/nebius.py up): the judge image as a long-running container (GPU, host network and IPC,
# unlimited memlock, nice), its /root on the boot disk (/data/root), sshd inside on port 2222 with /root/nebius_keys.txt; its
# command is the box's boot (nebius_boot.sh, from the bundle once shipped), so a VM boot brings the runner back by itself.
# Rerunnable: the container is made again, /data/root kept
set -e
IMG=lmsysorg/sglang:dev-muse-spark-open
mkdir -p /data/root/.ssh /data/root/out
cp /root/nebius_keys.txt /data/root/.ssh/authorized_keys; chmod 700 /data/root/.ssh; chmod 600 /data/root/.ssh/authorized_keys
docker pull -q $IMG > /dev/null
docker rm -f okc > /dev/null 2>&1 || true
docker run -d --name okc --restart unless-stopped --gpus all --network host --ipc host --ulimit memlock=-1:-1 \
  --cap-add SYS_NICE --cap-add SYS_PTRACE -v /data/root:/root -v /data:/data -w /root $IMG \
  bash -c '[ -f /root/okc/harness/h200/nebius_boot.sh ] && exec bash /root/okc/harness/h200/nebius_boot.sh; exec sleep infinity' > /dev/null
docker exec okc bash -c 'command -v sshd > /dev/null && command -v rsync > /dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server rsync > /dev/null); mkdir -p /run/sshd; ssh-keygen -A > /dev/null; /usr/sbin/sshd -p 2222 -o PermitRootLogin=prohibit-password -o PasswordAuthentication=no'
echo "container up"
