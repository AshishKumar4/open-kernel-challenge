#!/bin/bash
# On a build box, started by the PC relay: ship one built trial straight to the GPU box that benches it: the extension
# dirs its build log names, the TVM-FFI cache and the candidate, then its bench job into that GPU's queue. The build box
# keeps its copy (its cache). Prints "shipped <name>" once the job is queued. The GPU it is queued on (default: the one it
# was built for) is chosen by the relay at ship time: its extensions go to that GPU's dir, and its job line names it.
#   box_ship.sh <name> <gpu built for> <GPU box ssh port> <GPU box ip | local> [<gpu queued on>]
N=$1; G=$2; P=$3; IP=$4; T=${5:-$2}
JOB=$(sed "s#TORCH_EXTENSIONS_DIR=/root/ext$G #TORCH_EXTENSIONS_DIR=/root/ext$T #" /root/out/job_$N.txt 2>/dev/null)
case "$JOB" in run\ p[0-9]\ *) ;; *) echo "no job line for $N: not shipped (build it again)"; exit 3;; esac   # (eat_mst5:
# its job file gone, an empty line was queued and "shipped" reported: the job never ran)
DIRS=$(python3 -c "import ast, re; t = open('/root/out/prebuild_$N.log').read(); m = re.search(r'prebuilt \d+ extensions in \d+ s: (\[.*?\])\n', t, re.S)
print(' '.join(sorted({n for n, _ in ast.literal_eval(m.group(1)) if not n.startswith('muse_')})) if m else '')")   # (muse_: the clean
# layer's tvm-ffi modules, in the cache copied below)
if [ "$IP" = local ]; then   # (the GPU box built it itself: its extensions are in place for the GPU it was built for)
  if [ "$T" != "$G" ] && [ -n "$DIRS" ]; then cd /root/ext$G && rsync -aR --exclude lock $DIRS /root/ext$T/ || exit 1; fi
  echo "$JOB" | flock /root/queue$T.lock sh -c "cat >> /root/queue$T.txt" && echo "shipped $N"; exit
fi
X="ssh -i /root/.ssh/xfer -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $P"
# (none for the clean layer, whose modules are all tvm-ffi's: with no source an rsync lists its whole destination, the GPU
# box's ext dir of thousands of entries: 30-70 s of every ship, 19:00-19:25 UTC Nov 4)
[ -z "$DIRS" ] || { cd /root/ext$G && rsync -aR --exclude lock -e "$X" $DIRS root@$IP:/root/ext$T/; } || exit 1
# the job's own tvm-ffi modules (cpu_prebuild's list: each build's dir prefix), only those the GPU box lacks (a dir's name is
# its build's hash); none for a job that built none (a lane job's placeholder): never the whole cache, whose rsync scanned
# thousands of dirs per ship and stalled every ship for minutes
# (harness/tvm_publish.py: the GPU box's cache holds only complete, never rewritten modules: the missing ones are staged in
# .incoming/<job> and each published by one rename, never rsynced into a directory a bench may be loading or building)
if [ -s /root/out/tvm_$N.txt ]; then
  TD=$(cd /root/.cache/tvm-ffi && for p in $(cat /root/out/tvm_$N.txt); do for d in ${p}_*; do ls $d/*.so > /dev/null 2>&1 && echo $d; done; done)
  MISS=$([ -z "$TD" ] || echo $TD | $X root@$IP 'cd /root/.cache/tvm-ffi && for d in $(cat); do ls $d/*.so > /dev/null 2>&1 || echo $d; done')
  if [ -n "$MISS" ]; then
    (cd /root/.cache/tvm-ffi && rsync -aR --mkpath --exclude lock -e "$X" $MISS root@$IP:/root/.cache/tvm-ffi/.incoming/$N/) || exit 1
    echo $MISS | $X root@$IP "cd /root/.cache/tvm-ffi && for d in \$(cat); do mv -T .incoming/$N/\$d \$d 2> /dev/null || ls \$d/*.so > /dev/null 2>&1 || \
      { mkdir -p .trash && mv -T \$d .trash/\$d.\$\$.\$(date +%s%N) 2> /dev/null; mv -T .incoming/$N/\$d \$d; }; done; rm -rf .incoming/$N" || exit 1
  fi
fi
rsync -a -e "$X" /root/okc/candidates/layer_$N.py root@$IP:/root/okc/candidates/ || exit 1
[ ! -d /root/lanes/$N ] || rsync -a --mkpath -e "$X" /root/lanes/$N/ root@$IP:/root/lanes/$N/ || exit 1   # (a lane job's files)
echo "$JOB" | $X root@$IP "flock /root/queue$T.lock sh -c 'cat >> /root/queue$T.txt'" && echo "shipped $N"
