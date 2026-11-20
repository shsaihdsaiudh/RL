#!/usr/bin/env bash
# Usage: sweep_1gpu.sh "<ENV=.. ENV=..>" ["<...>" ...]  -> one 1-GPU multi_gpu_offload_check run per arg, summary lines only
D=/planning-users/yangyeyuan/nemo-rl-4166/scratch/cml
for envs in "$@"; do
  log=/tmp/sweep_$(echo "$envs" | tr ' =/,:' '_____').log
  env $envs TORCHDYNAMO_DISABLE=1 NPROC=1 EP_SIZES=${EP_SIZES:-1} timeout 1200 bash $D/run_multi_gpu_check.sh > "$log" 2>&1
  echo "== $envs"
  grep -m1 AC_WRAPPED "$log" | cut -c1-300
  python3 $D/summarize.py < "$log"
  grep -E "^SAME|^COPIES|^BUILD|^NAN_TRACE" "$log"
  grep -E "Traceback|Error" "$log" | head -3
done
echo SWEEP_DONE
