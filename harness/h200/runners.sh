#!/bin/bash
# On a vast box: run <n> queue runners (vqg.sh) on GPU <gpu>, replacing the ones there at a job boundary. One "stop"
# line per running runner goes to the head of the queue, and each finishes its current job before taking it. New
# runners start once the old ones are gone. Runs in the background (setsid); its log is /root/out/runners<gpu>.log.
#   runners.sh <gpu> <n>
G=$1; N=$2
old=$(pgrep -fc "vqg[.]sh $G( [a-z]( [0-9,-]+)?)?\$")
if [ "$old" -gt 0 ]; then
  flock /root/queue$G.lock sh -c "(for i in \$(seq $old); do echo stop; done; cat /root/queue$G.txt) > /root/queue$G.new && mv /root/queue$G.new /root/queue$G.txt"
fi
setsid bash -c "
  while pgrep -f 'vqg[.]sh $G( [a-z]( [0-9,-]+)?)?\$' > /dev/null; do sleep 2; done
  i=0
  for s in \$(echo a b c d | cut -d' ' -f1-$N); do   # 7 cores per runner from 64 + 16 G (28 for 4 runners): clear of builds
    c0=\$((${CORE0:-64 + 16 * $G} + 7 * i)); i=\$((i + 1))   # (CORE0: the first core on a small box, Nebius's 16)
    setsid nohup bash /root/okc/harness/h200/vqg.sh $G \$s \$c0-\$((c0 + 6)) >> /root/out/vq$G\$([ \$s = a ] || echo \$s).log 2>&1 < /dev/null &
  done
  echo \"\$(date -u +%T) gpu$G: $N runner(s) started\"
" >> /root/out/runners$G.log 2>&1 < /dev/null &
