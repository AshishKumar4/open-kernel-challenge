#!/bin/bash
# On a slim build box (nvidia/cuda:13.0.1-devel-ubuntu24.04): the GPU boxes' build toolchain under their exact paths, so
# the .so files it builds load on them without a rebuild (torch writes these paths into build.ninja, and a different path
# makes the GPU box rebuild): python 3.12 in a venv at /opt/sglang, torch 2.14.1+cu130, apache-tvm-ffi 0.1.11, ninja
# 1.13.2, ccache at /usr/bin/ccache (torch prepends it when present), nvcc 13.0.88 at /usr/local/cuda. Rerunnable.
set -e
if ! /opt/sglang/bin/python3 -c "import torch, tvm_ffi" 2> /dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq python3.12 python3.12-venv python3.12-dev ccache rsync g++ > /dev/null
  python3.12 -m venv /opt/sglang
  /opt/sglang/bin/pip install -q --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1
  /opt/sglang/bin/pip install -q apache-tvm-ffi==0.1.11 ninja==1.13.2 numpy
fi
# wrappers, not symlinks: a venv is found from the path the interpreter is started by (a symlink elsewhere loses it)
for t in python3 pip ninja; do printf '#!/bin/sh\nexec /opt/sglang/bin/%s "$@"\n' $t > /usr/local/bin/$t; chmod +x /usr/local/bin/$t; done
/usr/local/cuda/bin/nvcc --version | tail -2 | head -1
python3 -c "import sys, torch, tvm_ffi; print(sys.version.split()[0], torch.__version__, torch.version.cuda, tvm_ffi.__version__)"
