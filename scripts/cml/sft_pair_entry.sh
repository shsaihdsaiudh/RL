#!/usr/bin/env bash
# NeMo-RL #4166 cml job: SFT with activation offload off, then on, on the same node.
# Env: PAIR_NAME, plus MODEL / STEPS / MB_TOKENS for sft_offload_entry.sh.
# Results: /planning-users/yangyeyuan/nemo-rl-4166/logs/sft_offload/$PAIR_NAME-{off,on}/
set -uo pipefail
D=/planning-users/yangyeyuan/nemo-rl-4166/scratch/cml
PAIR_NAME=${PAIR_NAME:-pair-$(date +%m%d-%H%M%S)}
for offload in false true; do
    tag=$([[ $offload == true ]] && echo on || echo off)
    RUN_NAME=$PAIR_NAME-$tag RAY_TAG=s$tag OFFLOAD=$offload bash $D/sft_offload_entry.sh
    echo "[sft-pair] offload=$offload exit=$?"
    ray stop --force > /dev/null 2>&1 || true
    sleep 10
done
echo "[sft-pair] done"
