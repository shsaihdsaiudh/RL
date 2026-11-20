#!/usr/bin/env bash
# 2-GPU checks for NVIDIA-NeMo/RL#4169 (fused selected-token logprobs): the unit tests
# that need two ranks, plus a scratch script that prints the FSDP2 gradient differences.
# Runs on a CloudML node that mounts /planning-users. The code is whatever `cmdev push`
# last synced to $NRL_ROOT.
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

hostname
nvidia-smi -L
cd "$NRL_ROOT"
md5sum nemo_rl/distributed/fused_linear_logprobs.py nemo_rl/models/automodel/fused_logprobs.py \
    tests/unit/models/automodel/test_fused_logprobs.py tests/unit/distributed/test_fused_linear_logprobs.py

PY=$NRL_ROOT/venvs/automodel-test/bin/python
status=0
"$PY" -m pytest -q -p no:cacheprovider -rA \
    tests/unit/models/automodel/test_fused_logprobs.py \
    tests/unit/distributed/test_fused_linear_logprobs.py || status=$?
PYTHONPATH=. "$PY" -m torch.distributed.run --standalone --nproc_per_node 2 \
    "$WS/logs/scratch/explore_fsdp.py" || status=$?
echo "[dp2-entry] done status=$status"
exit "$status"
