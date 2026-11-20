#!/usr/bin/env bash
# Usage: [SEQ=.. EP_SIZES=.. NPROC=..] run_multi_gpu_check.sh  -> runs multi_gpu_offload_check.py with torchrun
set -euo pipefail
RL=/planning-users/yangyeyuan/nemo-rl-4166/RL
V=$RL/venvs/nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker
SP=$V/lib/python3.13/site-packages/nvidia
export LD_LIBRARY_PATH=$SP/cu13/lib:$SP/cudnn/lib:$SP/nccl/lib:$SP/nvshmem/lib:${LD_LIBRARY_PATH:-}
export NVTE_FUSED_ATTN=0 PYTHONPATH=$RL TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPYCACHEPREFIX=/planning-users/yangyeyuan/tmp/pycache-nrl4166-mgc
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
cd /tmp
exec $V/bin/python -m torch.distributed.run --nproc-per-node "$NPROC" \
  /planning-users/yangyeyuan/nemo-rl-4166/scratch/multi_gpu_offload_check.py
