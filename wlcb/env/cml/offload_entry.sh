#!/usr/bin/env bash
# Multi-GPU validation of the optimizer offloader for NVIDIA-NeMo/RL#3976, with the
# temporary probe nemo_rl/models/megatron/_offload_probe_3976.py in the tree.
#
# Env: RUN_NAME, ALGO (grpo|ppo), MODEL (Qwen/Qwen2.5-1.5B, or a local path), STEPS (12), NGPU (all visible), EXTRA_OVERRIDES,
# RAY_TAG (short per-job RAY_TMPDIR suffix), RAY_TMPDIR_BASE (dev machine: /dev/shm/...), NRL_OFFLOAD_LEGACY (1: old move_optimizer),
# NRL_OFFLOAD_CHECK (N: bit-exact check on the first N reloads).
# Results: /planning-users/yangyeyuan/nemo-rl-3976/logs/offload_8g/$RUN_NAME/
set -euo pipefail

WS=/planning-users/yangyeyuan/nemo-rl-3976
RUN_NAME=${RUN_NAME:-local-$(date +%m%d-%H%M%S)}
OUT=$WS/logs/offload_8g/$RUN_NAME
mkdir -p "$OUT"
exec > >(tee -a "$OUT/entry.log") 2>&1

# shellcheck disable=SC1091
source "$WS/env/env.sh"
# Per-job RAY_TMPDIR: init_ray first tries address="auto" through RAY_TMPDIR. Keep it short,
# Ray puts unix sockets under it.
export RAY_TMPDIR=${RAY_TMPDIR_BASE:-/planning-users/yangyeyuan/tmp/ray}${RAY_TAG:+/$RAY_TAG}
export TMPDIR=/planning-users/yangyeyuan/tmp/nrl-3976/$RUN_NAME
export PYTHONPYCACHEPREFIX=/planning-users/yangyeyuan/tmp/pycache-nrl
mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$PYTHONPYCACHEPREFIX"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

ALGO=${ALGO:-grpo}
STEPS=${STEPS:-12}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B}

{
    hostname
    nvidia-smi
    free -g
    nproc
} > "$OUT/sysinfo.txt" 2>&1 || true
env | grep -viE 'key|secret|token|passw|cred' | sort > "$OUT/env.txt"
md5sum "$NRL_ROOT"/nemo_rl/models/megatron/{optimizer_offload,_offload_probe_3976}.py \
    "$NRL_ROOT"/nemo_rl/models/policy/workers/megatron_policy_worker.py \
    "$NRL_ROOT"/nemo_rl/models/value/workers/megatron_value_worker.py > "$OUT/code.md5"

echo "[offload-entry] host=$(hostname) algo=$ALGO ngpu=$NGPU steps=$STEPS run=$RUN_NAME"
echo "[offload-entry] legacy=${NRL_OFFLOAD_LEGACY:-0} check=${NRL_OFFLOAD_CHECK:-0} extra=${EXTRA_OVERRIDES:-<none>}"
cd "$NRL_ROOT"
if [[ $ALGO == grpo ]]; then
    # shellcheck disable=SC2086
    uv run --locked examples/run_grpo.py \
        --config examples/configs/grpo_math_1B_megatron.yaml \
        policy.model_name="$MODEL" \
        data.train.dataset_name=gsm8k \
        cluster.gpus_per_node="$NGPU" \
        grpo.max_num_steps="$STEPS" \
        grpo.num_prompts_per_step=32 \
        grpo.num_generations_per_prompt=8 \
        grpo.val_period=-1 \
        policy.train_global_batch_size=256 \
        policy.max_total_sequence_length=512 \
        checkpointing.enabled=false \
        logger.log_dir="$OUT/train" \
        ${EXTRA_OVERRIDES:-}
else
    # shellcheck disable=SC2086
    uv run --locked examples/run_ppo.py \
        --config examples/configs/recipes/llm/ppo-qwen2.5-1.5b-gsm8k-1n8g-megatron-valuetp2sp-dynbatch.yaml \
        policy.model_name="$MODEL" \
        value.model_name="$MODEL" \
        cluster.gpus_per_node="$NGPU" \
        ppo.max_num_steps="$STEPS" \
        ppo.num_prompts_per_step=256 \
        ppo.val_period=0 \
        ppo.val_at_start=false \
        ppo.val_at_end=false \
        checkpointing.enabled=false \
        logger.wandb_enabled=false \
        logger.tensorboard_enabled=false \
        logger.log_dir="$OUT/train" \
        ${EXTRA_OVERRIDES:-}
fi
echo "[offload-entry] done"
