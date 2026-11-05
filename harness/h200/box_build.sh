#!/bin/bash
# On a build box, started by the PC relay (harness/relay.py): build one study trial's extensions into the box's cache
# for the GPU index it will bench on (/root/ext<gpu>, the GPU box's path), with the whole machine at nice 19. The relay
# collects /root/out/built_<name>.rc (the build's exit status) and the extension dirs the build log names.
#   box_build.sh <name> <gpu>
N=$1; G=$2
echo $$ > /root/out/pgid_$N.txt   # (started under setsid: its process group, which a cancel kills whole)
cd /root/okc
mkdir -p /root/ext$G /root/.cache/tvm-ffi
: > /root/out/using_$N.txt
bash harness/h200/reap_locks.sh /root/ext$G   # (a killed build's lock, never a live one's)
# the box's effective threads (its cgroup quota, else nproc): every build on the box shares that many compile slots
# (cpu_prebuild.py), each extension job compiling serially, so the box never runs more compilers than threads
T=$(awk '$1 != "max" {print int($1 / $2)}' /sys/fs/cgroup/cpu.max 2>/dev/null)
[ -n "$T" ] || T=$(awk 'NR == 1 {q = $1} NR == 2 && q > 0 {print int(q / $1)}' /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null)
[ -n "$T" ] && [ "$T" -gt 0 ] || T=$(nproc)
T=${OKC_COMPILE_SLOTS:-$T}   # (preset by the relay for a GPU box building for itself: gpu1, a few cores beside its benches)
PIN=${OKC_BUILD_CPUS:+taskset -c $OKC_BUILD_CPUS}
[ -f /root/out/diag_env.sh ] && . /root/out/diag_env.sh   # (optional diagnostics for the next builds, e.g. ccache logs)
rm -f /root/out/tvm_$N.txt
env PYTHONHASHSEED=0 OKC_BUILD_FOR=h200 OKC_USING=/root/out/using_$N.txt OKC_TVM_USING=/root/out/tvm_$N.txt TORCH_EXTENSIONS_DIR=/root/ext$G TORCH_CUDA_ARCH_LIST=9.0a TVM_FFI_CUDA_ARCH_LIST=9.0a \
  OKC_COMPILE_SLOTS=$T MAX_JOBS=1 $PIN nice -n 19 timeout 1200 python3 harness/h200/cpu_prebuild.py candidates/layer_$N.py > /root/out/prebuild_$N.log 2>&1
rc=$?
for f in /root/lanes/$N/candidates/*.py; do   # a lane job's own candidate files: built the same way into the caches the relay
  [ $rc = 0 ] && [ -f "$f" ] || continue      # ships, so the job compiles nothing on the GPU box (ep_pair_layer_ba compiled
  env PYTHONHASHSEED=0 OKC_BUILD_FOR=h200 OKC_USING=/root/out/using_$N.txt OKC_TVM_USING=/root/out/tvm_$N.txt \
    TORCH_EXTENSIONS_DIR=/root/ext$G TORCH_CUDA_ARCH_LIST=9.0a TVM_FFI_CUDA_ARCH_LIST=9.0a OKC_COMPILE_SLOTS=$T MAX_JOBS=1 \
    $PIN nice -n 19 timeout 1200 python3 harness/h200/cpu_prebuild.py "$f" >> /root/out/prebuild_$N.log 2>&1 || rc=2
done                                          # its modules on spot1 under the GPU lock for 100+ s, 11-05 01:00 UTC)
if [ $rc = 0 ] && [ -f /root/out/build_$N.sh ]; then   # a lane job's own compile step, here and not under the GPU lock
  mkdir -p /root/lanes/$N && (cd /root/lanes/$N && nice -n 19 timeout 900 bash /root/out/build_$N.sh) >> /root/out/prebuild_$N.log 2>&1
  rc=$?; [ $rc = 0 ] || rc=2   # (a failed compile: reported as the job's result, never run)
fi
echo $rc > /root/out/built_$N.rc
