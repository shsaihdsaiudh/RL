#!/usr/bin/env bash
# Multi-GPU checks for NVIDIA-NeMo/RL#4169 (fused selected-token logprobs) on wlcb:
# 1. the LM-head unit tests, whose DP=2 FSDP2 cases need two GPUs;
# 2. the Nemotron 3.5 Super-dimension proxy (real hidden size and vocabulary, 32K tokens)
#    under AutoModel's NemotronH FSDP2 strategy across all GPUs, logits path vs fused path.
# wlcb has no AutoModel venv, so this borrows the nemo-rl-3976 Megatron worker venv
# (torch 2.13.0+cu130, Transformer Engine) and puts the 4169 source tree and its AutoModel
# submodule source first on PYTHONPATH. The source tree is a tar copy of the local branch.
# Results: /planning-users/yangyeyuan/nemo-rl-4169/logs/cml/$RUN_NAME/
set -euo pipefail

WS=/planning-users/yangyeyuan/nemo-rl-4169
RUN_NAME=${RUN_NAME:-local-$(date +%m%d-%H%M%S)}
OUT=$WS/logs/cml/$RUN_NAME
mkdir -p "$OUT"
exec > >(tee -a "$OUT/entry.log") 2>&1

# shellcheck disable=SC1091
source /planning-users/yangyeyuan/nemo-rl-3976/env/env.sh
export TMPDIR=/planning-users/yangyeyuan/tmp/nrl-4169/$RUN_NAME
export PYTHONPYCACHEPREFIX=/planning-users/yangyeyuan/tmp/pycache-nrl-4169
mkdir -p "$TMPDIR" "$PYTHONPYCACHEPREFIX"

ROOT=$WS/RL
PY=/planning-users/yangyeyuan/nemo-rl-3976/RL/venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
export PYTHONPATH=$ROOT:$ROOT/3rdparty/Automodel-workspace/Automodel:$WS/pytest-overlay
# The borrowed venv has no grouped_gemm; the LM-head comparison does not depend on the
# expert or dispatcher kernels.
export EXPERTS=${EXPERTS:-torch} DISPATCHER=${DISPATCHER:-torch}

NGPU=$(nvidia-smi -L | wc -l)
hostname
nvidia-smi -L
cd "$ROOT"
md5sum nemo_rl/distributed/fused_linear_logprobs.py nemo_rl/models/automodel/fused_logprobs.py \
    nemo_rl/models/automodel/train.py "$WS/logs/scratch/super_proxy_fused.py"
"$PY" -c "import nemo_rl, nemo_automodel, torch, transformer_engine; print(nemo_rl.__file__, nemo_automodel.__file__, torch.__version__, transformer_engine.__version__)"

status=0
"$PY" -m pytest -q -p no:cacheprovider -rA --maxfail=100 \
    tests/unit/models/automodel/test_fused_logprobs.py || status=$?
LAYERS="${PROXY_LAYERS:-*E}" "$PY" -m torch.distributed.run --standalone \
    --nproc_per_node "$NGPU" "$WS/logs/scratch/super_proxy_fused.py" || status=$?
echo "[step4-entry] done status=$status"
exit "$status"
