#!/bin/bash
# On a vast box: (re)start the GPU telemetry, appending one line per GPU per second to /root/out/util.csv
# (timestamp, index, utilization.gpu, power.draw, clocks.sm). Rerunnable: an older writer is replaced.
pkill -f '^nvidia-smi --query-gpu'
setsid nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,clocks.sm --format=csv,noheader -l 1 \
  >> /root/out/util.csv 2> /dev/null < /dev/null &
