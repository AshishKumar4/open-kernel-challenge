#!/bin/bash
# a Nebius box's container start (its command, nebius_up.sh; docker restarts it when the VM boots after a preemption):
# sshd on 2222, an empty queue (the relay queues what the stop lost again: relay.py "back"), two runners, and the scored
# inputs on the persistent disk (/root/okc_inputs: generated once, kept across preemptions); then it stays up.
#   nebius_boot.sh [now]   # now: the same, returning (harness/h200/nebius_recover.sh, a box set up before this script)
mkdir -p /run/sshd /root/out /root/okc_inputs
pgrep -x sshd > /dev/null || /usr/sbin/sshd -p 2222 -o PermitRootLogin=prohibit-password -o PasswordAuthentication=no
export OKC_INPUT_CACHE=/root/okc_inputs
: > /root/queue0.txt
for f in /root/out/vq*.log; do [ -f $f ] && echo "-- started $(date -u +%T): a stop ended any job above" >> $f; done
cd /root && CORE0=0 RUNNERS=2 bash /root/okc/harness/h200/box_up.sh 1 < /dev/null > /root/out/boot.log 2>&1
if [ ! -f /root/okc_inputs/.complete ] && ! pgrep -f "gen_inputs[.]py" > /dev/null; then
  (cd /root/okc && setsid nohup sh -c "python3 harness/h200/gen_inputs.py > /root/out/gen_inputs.log 2>&1 && touch /root/okc_inputs/.complete" < /dev/null > /dev/null 2>&1 &)
fi
[ "$1" = now ] && exit 0
exec sleep infinity
