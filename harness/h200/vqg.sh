#!/bin/bash
# vast.ai box (inside the judge image): a runner of one GPU, popping /root/queue<gpu>.txt under its lock.
#   vqg.sh <gpu> [slot] [cores]      line forms: run p<level> <command...> | stop
# One queue per GPU for every job (screens, confirmations, lane jobs), filled by the relay (harness/submit.py): the line
# with the lowest level is taken first (a plain run line: level 3; stop: first), then the oldest. Each job carries its own
# limit (a bench request 170 s, a lane job 480 s); 900 s here is only the backstop. Two runners (slots a, b) may share a GPU: every job holds /root/gpu<gpu>.lock shared and takes it exclusive
# only while it times (OKC_GPU_LOCK: okc_harness and benchd do), so nothing overlaps a timed section and no job stalls the
# study by holding the GPU. Each runner and its jobs run pinned to their own cores (disjoint from the other
# runners, compile-ahead and lane builds) at nice 19 and idle IO; a bench raises its own priority only while it times
# (okc_harness gpu_timing). Log: /root/out/vq<gpu><slot>.log (slot a: vq<gpu>.log).
set -u
G=$1; S=${2:-a}; C=${3:-}
if [ -n "$C" ] && [ -z "${VQG_PINNED:-}" ]; then
  VQG_PINNED=1 exec taskset -c "$C" nice -n 19 ionice -c3 bash "$0" "$G" "$S" "$C"
fi
BACKSTOP=${VQG_BACKSTOP:-900}   # (s: any job's group is killed after this, whatever its own limit)
mkdir -p /dev/shm/okc_inputs   # (the bench servers' shared input cache: okc_harness Problem.inputs)
export CUDA_VISIBLE_DEVICES=$G TORCH_EXTENSIONS_DIR=/root/ext$G PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 OKC_GPU_LOCK=/root/gpu$G.lock \
  OKC_INPUT_CACHE=${OKC_INPUT_CACHE:-/root/okc_inputs} \
  OMP_NUM_THREADS=2 OKC_SLOT=$G$S   # (OKC_SLOT: a job's persistent bench server, /root/benchd$OKC_SLOT.sock)
cd /root/okc
while true; do
  line=$(flock /root/queue$G.lock sh -c "n=\$(awk '{l = \$1 == \"stop\" ? -1 : \$2 ~ /^p[0-9]\$/ ? substr(\$2, 2) + 0 : 3; \
    if (!b || l < bl) { b = NR; bl = l }} END { print b + 0 }' /root/queue$G.txt 2>/dev/null); \
    [ \"\${n:-0}\" -gt 0 ] && sed -n \"\${n}p\" /root/queue$G.txt && sed -i \"\${n}d\" /root/queue$G.txt")
  if [ -z "$line" ]; then sleep 5; continue; fi
  set -- $line
  kind=$1
  [ "$kind" = stop ] && { echo "== gpu$G$S stop $(date -u +%T)"; break; }
  echo "== gpu$G $line $(date -u +%T)"
  bash harness/h200/reap_locks.sh /root/ext$G   # (a killed build's lock, never a live one's: gpu1 builds here too)
  case $kind in
    run) shift
         [[ "${1:-}" =~ ^p[0-9]$ ]] && shift
         flock -s /root/gpu$G.lock true   # (its time limit starts once the GPU is not being timed)
         # the job in its own session: the session is everything it starts, orphans included, whatever their process group
         # (an inner limit's timeout kills only its own child: a python under it lived on, holding the GPU lock); bench
         # servers start their own sessions
         setsid bash -c "$*" > /root/out/vrun_$(date +%s)_$G$S.txt 2>&1 < /dev/null &
         job=$!
         setsid bash -c "sleep $BACKSTOP; pkill -TERM -s $job; sleep 10; pkill -KILL -s $job" > /dev/null 2>&1 < /dev/null &
         dog=$!
         wait $job; rc=$?
         { pkill -KILL -s $dog; wait $dog; } 2> /dev/null
         pkill -KILL -s $job   # (whatever the job left behind)
         echo "   gpu$G exit $rc $(date -u +%T)" ;;
  esac
done
