#!/usr/bin/env bash
# End-to-end check of the environment: colocated GRPO, Megatron policy + vLLM generation,
# Qwen2.5-0.5B on GSM8K, a few steps on the single H20. Extra args are Hydra overrides.
#   nohup env/smoke_grpo_megatron.sh > logs/smoke_grpo_megatron.log 2>&1 &
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$(readlink -f "$0")")/env.sh"
cd "$NRL_ROOT"
uv run --locked examples/run_grpo.py \
    --config examples/configs/grpo_math_1B_megatron.yaml \
    policy.model_name=Qwen/Qwen2.5-0.5B \
    data.train.dataset_name=gsm8k \
    grpo.max_num_steps=3 \
    grpo.num_prompts_per_step=8 \
    grpo.num_generations_per_prompt=4 \
    grpo.val_period=-1 \
    policy.train_global_batch_size=32 \
    policy.max_total_sequence_length=512 \
    checkpointing.enabled=false \
    logger.log_dir="$NRL_WS/logs/smoke_grpo_megatron" \
    "$@"
