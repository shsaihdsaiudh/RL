#!/usr/bin/env bash
# 2-GPU checks for NVIDIA-NeMo/RL#4169 (fused selected-token logprobs) on wlcb.
# wlcb has no nemo-rl-4169 venv, so this borrows the nemo-rl-3976 Megatron worker venv
# (torch 2.13.0+cu130, transformers 5.16.1, ray 2.58.0) and puts the 4169 source tree and a
# pytest overlay first on PYTHONPATH. The source tree is a tar copy of the local branch.
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
export PYTHONPATH=$ROOT:$WS/pytest-overlay

hostname
nvidia-smi -L
cd "$ROOT"
md5sum nemo_rl/distributed/fused_linear_logprobs.py nemo_rl/models/automodel/fused_logprobs.py \
    tests/unit/models/automodel/test_fused_logprobs.py tests/unit/distributed/test_fused_linear_logprobs.py
"$PY" -c "import nemo_rl, torch, transformers; print(nemo_rl.__file__, torch.__version__, transformers.__version__)"

status=0
# pyproject addopts has -x; --maxfail overrides it so every test reports.
"$PY" -m pytest -q -p no:cacheprovider -rA --maxfail=100 \
    tests/unit/models/automodel/test_fused_logprobs.py \
    tests/unit/distributed/test_fused_linear_logprobs.py || status=$?
"$PY" -m torch.distributed.run --standalone --nproc_per_node 2 \
    "$WS/logs/scratch/explore_fsdp.py" || status=$?
echo "[dp2-entry] done status=$status"
exit "$status"
