#!/usr/bin/env bash
# gc probe for NVIDIA-NeMo/RL#3976: colocated GRPO (Megatron policy + vLLM) with the
# temporary gc.collect() probe (nemo_rl/models/megatron/_gc_probe_3976.py) enabled.
# Runs on the dev machine or a CloudML node that mounts /planning-users.
#
# Env (all optional): RUN_NAME, MODEL (default Qwen/Qwen2.5-1.5B), STEPS (6), NGPU (all
# visible GPUs), PROMPTS (32), GENS (8), GBS (256), EXTRA_OVERRIDES (extra Hydra overrides),
# RAY_TAG (short per-job RAY_TMPDIR suffix), NRL_GC_PROBE_ANALYZE (alternate|first),
# NRL_GC_FREEZE (init: gc.collect() then gc.freeze() at the end of worker __init__).
# Results: /planning-users/yangyeyuan/nemo-rl-3976/logs/gc_probe/$RUN_NAME/
set -euo pipefail

WS=/planning-users/yangyeyuan/nemo-rl-3976
RUN_NAME=${RUN_NAME:-local-$(date +%m%d-%H%M%S)}
OUT=$WS/logs/gc_probe/$RUN_NAME
mkdir -p "$OUT"
exec > >(tee -a "$OUT/entry.log") 2>&1

# shellcheck disable=SC1091
source "$WS/env/env.sh"
# Temporary files go to the personal folder instead of /dev/shm. RAY_TMPDIR is per job:
# init_ray first tries ray.init(address="auto"), which reads ray_current_cluster under
# RAY_TMPDIR, so concurrent jobs sharing it could attach to each other's cluster. Keep the
# tag short: Ray puts unix sockets under it and socket paths are limited to 107 bytes.
export RAY_TMPDIR=/planning-users/yangyeyuan/tmp/ray${RAY_TAG:+/$RAY_TAG}
export TMPDIR=/planning-users/yangyeyuan/tmp/nrl-3976/$RUN_NAME
export PYTHONPYCACHEPREFIX=/planning-users/yangyeyuan/tmp/pycache-nrl
mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$PYTHONPYCACHEPREFIX"
# Model, tokenizer and GSM8K are already in $HF_HOME.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NRL_GC_PROBE_DIR=$OUT/gc

MODEL=${MODEL:-Qwen/Qwen2.5-1.5B}
STEPS=${STEPS:-6}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
PROMPTS=${PROMPTS:-32}
GENS=${GENS:-8}
GBS=${GBS:-256}

{
    hostname
    nvidia-smi
    df -h /dev/shm /planning-users
    free -g
    nproc
} > "$OUT/sysinfo.txt" 2>&1 || true
env | grep -viE 'key|secret|token|passw|cred' | sort > "$OUT/env.txt"
md5sum "$NRL_ROOT/nemo_rl/models/megatron/_gc_probe_3976.py" \
    "$NRL_ROOT/nemo_rl/models/policy/workers/megatron_policy_worker.py" > "$OUT/code.md5"

echo "[gc-probe-entry] host=$(hostname) ngpu=$NGPU model=$MODEL steps=$STEPS run=$RUN_NAME out=$OUT"
echo "[gc-probe-entry] extra overrides: ${EXTRA_OVERRIDES:-<none>}"
echo "[gc-probe-entry] NRL_GC_PROBE_ANALYZE=${NRL_GC_PROBE_ANALYZE:-alternate} NRL_GC_FREEZE=${NRL_GC_FREEZE:-<none>} RAY_TMPDIR=$RAY_TMPDIR"
cd "$NRL_ROOT"
# shellcheck disable=SC2086
uv run --locked examples/run_grpo.py \
    --config examples/configs/grpo_math_1B_megatron.yaml \
    policy.model_name="$MODEL" \
    data.train.dataset_name=gsm8k \
    cluster.gpus_per_node="$NGPU" \
    grpo.max_num_steps="$STEPS" \
    grpo.num_prompts_per_step="$PROMPTS" \
    grpo.num_generations_per_prompt="$GENS" \
    grpo.val_period=-1 \
    policy.train_global_batch_size="$GBS" \
    policy.max_total_sequence_length=512 \
    checkpointing.enabled=false \
    logger.log_dir="$OUT/train" \
    ${EXTRA_OVERRIDES:-}
echo "[gc-probe-entry] done"
