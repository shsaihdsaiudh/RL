#!/usr/bin/env bash
# NeMo-RL #4166 cml job: activation offload on an 8-GPU FSDP2 Nemotron-H (mamba/moe/attention).
#   1. gradient parity on one model instance (plain vs offload, alternating), EP=1 then EP=8 (DeepEP)
#   2. 3 AdamW steps per run (ac / ac_again / ac_offload): memory, fwd/bwd time, host pinned bytes
# Each part runs on its own so a failure in one still leaves the others' results.
# Env: RUN_NAME, SEQ_PARITY (8192), SEQ_MEM (16384). Results: $WS/logs/mgc/$RUN_NAME/
set -uo pipefail
WS=/planning-users/yangyeyuan/nemo-rl-4166
RUN_NAME=${RUN_NAME:-local-$(date +%m%d-%H%M%S)}
OUT=$WS/logs/mgc/$RUN_NAME
mkdir -p "$OUT"
exec > >(tee -a "$OUT/entry.log") 2>&1
{ hostname; nvidia-smi; free -g; nproc; } > "$OUT/sysinfo.txt" 2>&1 || true
md5sum "$WS/RL/nemo_rl/models/automodel/activation_offload.py" "$WS/scratch/multi_gpu_offload_check.py" \
    "$WS/RL/3rdparty/Automodel-workspace/Automodel/nemo_automodel/components/distributed/parallelizer.py" > "$OUT/code.md5"
RUN="bash $WS/scratch/cml/run_multi_gpu_check.sh"
# Same setting as the 1-GPU checks.
export TORCHDYNAMO_DISABLE=1
part() {
    local name=$1
    shift
    echo "[mgc] part=$name env=$* start=$(date +%T)"
    env "$@" $RUN > "$OUT/$name.log" 2>&1
    echo "[mgc] part=$name exit=$? end=$(date +%T)"
    grep -E "^(SAME|COPIES|RESULT|AC_WRAPPED)" "$OUT/$name.log" | cut -c1-600
    grep -E "Traceback|Error" "$OUT/$name.log" | head -5
}
part parity_ep1 SAME_MODEL=1 RECORD=1 PRINT_AC=1 EP_SIZES=1 SEQ="${SEQ_PARITY:-8192}"
part parity_ep8 SAME_MODEL=1 RECORD=1 PRINT_AC=1 EP_SIZES=8 SEQ="${SEQ_PARITY:-8192}"
part memory_ep1 EP_SIZES=1 STEPS=3 SEQ="${SEQ_MEM:-16384}"
part memory_ep8 EP_SIZES=8 STEPS=3 SEQ="${SEQ_MEM:-16384}"
echo "[mgc] done"
