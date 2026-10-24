#!/usr/bin/env bash
# Runs a command while holding one lock file per GPU, so our own jobs never
# share a GPU. Waits in line until every requested GPU is free of our locks,
# then sets HIP_VISIBLE_DEVICES and runs the command.
#
#   tools/gpulock.sh 0,1 -- ./run ...      # HIP indices
#   tools/gpulock.sh all -- ./run ...      # all 8
#
# The locks only coordinate our jobs; check tools/gpus.py for other users.
set -euo pipefail

if [[ $# -lt 3 || $2 != -- ]]; then
  echo "usage: $0 <hip ids|all> -- command ..." >&2
  exit 2
fi
ids=$1
shift 2
[[ $ids == all ]] && ids=0,1,2,3,4,5,6,7

dir=/tmp/gptoss-gpu-locks
mkdir -p "$dir"
for i in $(tr ',' '\n' <<<"$ids" | sort -n); do
  exec {fd}>"$dir/gpu$i.lock"
  if ! flock -n "$fd"; then
    echo "gpulock: waiting for GPU $i ..." >&2
    flock "$fd"
  fi
done
export HIP_VISIBLE_DEVICES=$ids
exec "$@"
