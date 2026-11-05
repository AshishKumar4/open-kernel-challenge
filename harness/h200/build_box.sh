#!/bin/bash
# build_box.sh <label> <GPU box label>: set up a build box from the registry (role "build") for a GPU box: the bench
# bundle under the same paths, an extension dir per GPU, and an ssh key the GPU box accepts (the build box copies built
# extensions and queues bench jobs there). Rerunnable.
set -eu
cd "$(dirname "$0")/../.."
read -r PORT IP GP GIP NG < <(python3 -c "import sys; sys.path.insert(0, 'harness'); from vast_hosts import boxes
b = boxes(); d, g = b['$1'], b['$2']; print(d['port'], d['ip'], g['port'], g['ip'], len(g['cpus']))")
SSH="ssh ${OKC_SSH_KEY:+-i $OKC_SSH_KEY} -o UserKnownHostsFile=${OKC_STATE:-$HOME/.okc}/known_hosts -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o IdentitiesOnly=yes"
bash harness/h200/bundle.sh > /dev/null
${SSH/ssh/scp} -P "$PORT" -q ${OKC_STATE:-$HOME/.okc}/okc_box.tgz "root@$IP:/root/"
PUB=$($SSH -p "$PORT" "root@$IP" "mkdir -p /root/okc /root/out && tar xzf /root/okc_box.tgz -C /root/okc && \
  mkdir -p /root/build && install -m 755 /root/okc/harness/h200/ccache_wrap.sh /usr/local/bin/ccache && \
  { [ -f /root/.ssh/xfer ] || ssh-keygen -q -t ed25519 -N '' -f /root/.ssh/xfer; } && cat /root/.ssh/xfer.pub" 2>/dev/null | tail -1)
$SSH -p "$GP" "root@$GIP" "grep -qF '$PUB' /root/.ssh/authorized_keys || { sed -i -e '\$a\\' /root/.ssh/authorized_keys; echo '$PUB' >> /root/.ssh/authorized_keys; }" 2>/dev/null   # (a last line without its newline would swallow the key)
$SSH -p "$PORT" "root@$IP" "ssh -i /root/.ssh/xfer -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $GP root@$GIP 'hostname; nproc'; \
  nproc; cat /sys/fs/cgroup/cpu.max; python3 -c 'import torch; print(torch.__version__)'; /usr/local/cuda/bin/nvcc --version | tail -2 | head -1" 2>&1 \
  | grep -v "Welcome to vast.ai\|Have fun"
