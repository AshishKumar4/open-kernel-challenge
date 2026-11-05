#!/bin/bash
# The code a box needs, as one small tarball (${OKC_STATE:-$HOME/.okc}/okc_box.tgz): the harness (the bench, its
# server and the box scripts), the layer problem's reference files and one scored profile, and reviewed production (the
# baseline every bench runs against). The code is main's committed files (git archive HEAD: every box runs one version,
# never a working tree's edits); reviewed production and its profile (kept out of git) come from the working tree.
# Candidates reach a box with their jobs (the relay ships each one).
set -eu
cd "$(dirname "$0")/../.."
# the scored cases with the seeds of production's scored run 58d2ec83 (the run calibration compares against)
PROFILE=reviewed/profiles/muse_decode_layer.58d2ec83252f4c0f.json
T=$(mktemp -d ${OKC_STATE:-$HOME/.okc}/bundle.XXXXXX)
git archive HEAD harness/okc_harness.py harness/benchd.py harness/tvm_publish.py harness/soak.py harness/h200 muse_decode_layer/reference.py \
  muse_decode_layer/checks.py muse_decode_layer/nvfp4.py muse_decode_layer/baseline.py muse_decode_layer/TASK.md | tar x -C $T
mkdir -p $T/candidates
cp --parents reviewed/muse_decode_layer.py "$PROFILE" $T/
tar czf ${OKC_STATE:-$HOME/.okc}/okc_box.tgz -C $T --transform "s#^\./$PROFILE#./muse_decode_layer/profiles/58d2ec83.json#" .
rm -rf $T
ls -la ${OKC_STATE:-$HOME/.okc}/okc_box.tgz
