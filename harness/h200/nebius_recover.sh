#!/bin/bash
# on a Nebius VM host after a start (a preemption stops it) or a setup (harness/h200/nebius.py): the container (its command
# already ran nebius_boot.sh at a boot if the bundle was there), then the boot's work again, returning: sshd, an empty
# queue, the runner, the scored inputs on disk (all idempotent)
docker start okc > /dev/null
docker exec okc bash /root/okc/harness/h200/nebius_boot.sh now
echo recovered
