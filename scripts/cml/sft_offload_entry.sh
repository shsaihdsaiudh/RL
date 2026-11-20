#!/usr/bin/env bash
# NeMo-RL #4166: SFT with automodel activation checkpointing, activation offload on/off.
# Env: RUN_NAME, MODEL (Qwen/Qwen2.5-1.5B), STEPS (20), NGPU (all), MB_TOKENS (16384),
# OFFLOAD (true|false), EXTRA_OVERRIDES. Results: $WS/logs/sft_offload/$RUN_NAME/
set -euo pipefail
WS=/planning-users/yangyeyuan/nemo-rl-4166
RUN_NAME=${RUN_NAME:-local-$(date +%m%d-%H%M%S)}
OUT=$WS/logs/sft_offload/$RUN_NAME
mkdir -p "$OUT"
exec > >(tee -a "$OUT/entry.log") 2>&1
export NRL_WS=$WS
# shellcheck disable=SC1091
source /planning-users/yangyeyuan/nemo-rl-3976/env/env.sh
export RAY_TMPDIR=${RAY_TMPDIR_BASE:-/planning-users/yangyeyuan/tmp/ray}${RAY_TAG:+/$RAY_TAG}
export TMPDIR=/planning-users/yangyeyuan/tmp/nrl-4166/$RUN_NAME
export PYTHONPYCACHEPREFIX=/planning-users/yangyeyuan/tmp/pycache-nrl4166
mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$PYTHONPYCACHEPREFIX"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B}
STEPS=${STEPS:-20}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
MB_TOKENS=${MB_TOKENS:-16384}
OFFLOAD=${OFFLOAD:-false}
{ hostname; nvidia-smi; free -g; nproc; } > "$OUT/sysinfo.txt" 2>&1 || true
# Per-GPU memory.used every 2 s, for the peak with offload on and off.
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -l 2 > "$OUT/nvsmi.csv" 2>/dev/null &
NVSMI_PID=$!
trap 'kill $NVSMI_PID 2>/dev/null || true' EXIT
md5sum "$NRL_ROOT/nemo_rl/models/automodel/activation_offload.py" "$NRL_ROOT/nemo_rl/models/automodel/train.py" \
    "$NRL_ROOT/3rdparty/Automodel-workspace/Automodel/nemo_automodel/components/distributed/parallelizer.py" > "$OUT/code.md5"
echo "[sft-offload] host=$(hostname) ngpu=$NGPU model=$MODEL steps=$STEPS offload=$OFFLOAD mb_tokens=$MB_TOKENS run=$RUN_NAME"
cd "$NRL_ROOT"
# shellcheck disable=SC2086
uv run --locked examples/run_sft.py \
    --config examples/configs/sft.yaml \
    policy.model_name="$MODEL" \
    cluster.gpus_per_node="$NGPU" \
    sft.max_num_steps="$STEPS" \
    sft.val_period=-1 sft.val_at_start=false sft.val_at_end=false \
    checkpointing.enabled=false \
    policy.train_global_batch_size=$((NGPU * 16)) \
    policy.max_total_sequence_length=1024 \
    policy.sequence_packing.enabled=true \
    policy.sequence_packing.train_mb_tokens="$MB_TOKENS" \
    policy.automodel_cfg.activation_checkpointing=true \
    policy.automodel_cfg.activation_offloading="$OFFLOAD" \
    logger.log_dir="$OUT/train" logger.tensorboard_enabled=false logger.wandb_enabled=false \
    ${EXTRA_OVERRIDES:-}
echo "[sft-offload] done"
