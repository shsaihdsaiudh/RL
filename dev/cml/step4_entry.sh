#!/usr/bin/env bash
# Multi-GPU checks for NVIDIA-NeMo/RL#4169 (fused selected-token logprobs) on bj:
# 1. the LM-head unit tests, whose DP=2 FSDP2 cases need two GPUs;
# 2. the Nemotron 3.5 Super-dimension proxy (real hidden size and vocabulary, 32K tokens)
#    under AutoModel's NemotronH FSDP2 strategy across all GPUs, logits path vs fused path.
# The code is whatever `cmdev push` last synced to $NRL_ROOT.
# Results: /planning-users/yangyeyuan/nemo-rl-4169/logs/cml/$RUN_NAME/
set -euo pipefail

WS=/planning-users/yangyeyuan/nemo-rl-4169
RUN_NAME=${RUN_NAME:-local-$(date +%m%d-%H%M%S)}
OUT=$WS/logs/cml/$RUN_NAME
mkdir -p "$OUT"
exec > >(tee -a "$OUT/entry.log") 2>&1

export NRL_WS=$WS
# shellcheck disable=SC1091
source /planning-users/yangyeyuan/nemo-rl-3976/env/env.sh
export TMPDIR=/planning-users/yangyeyuan/tmp/nrl-4169/$RUN_NAME
export PYTHONPYCACHEPREFIX=/planning-users/yangyeyuan/tmp/pycache-nrl
mkdir -p "$TMPDIR" "$PYTHONPYCACHEPREFIX"

NGPU=$(nvidia-smi -L | wc -l)
hostname
nvidia-smi -L
cd "$NRL_ROOT"
md5sum nemo_rl/distributed/fused_linear_logprobs.py nemo_rl/models/automodel/fused_logprobs.py \
    nemo_rl/models/automodel/train.py "$WS/logs/scratch/super_proxy_fused.py"

PY=$NRL_ROOT/venvs/automodel-test/bin/python
status=0
"$PY" -m pytest -q -p no:cacheprovider -rA --maxfail=100 \
    tests/unit/models/automodel/test_fused_logprobs.py || status=$?
for layers in "${PROXY_LAYERS:-*E}"; do
    LAYERS="$layers" PYTHONPATH=. "$PY" -m torch.distributed.run --standalone \
        --nproc_per_node "$NGPU" "$WS/logs/scratch/super_proxy_fused.py" || status=$?
done
echo "[step4-entry] done status=$status"
exit "$status"
