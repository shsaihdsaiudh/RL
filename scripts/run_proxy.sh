#!/bin/bash
# Usage: [ENV=..] run_proxy.sh  (runs super_proxy_memory.py against the PR Automodel checkout)
V=/planning-users/yangyeyuan/nemo-rl-4166/RL/venvs/automodel-test
SP=$V/lib/python3.13/site-packages/nvidia
export LD_LIBRARY_PATH=$SP/cu13/lib:$SP/cudnn/lib:$SP/nccl/lib:$SP/nvshmem/lib:$LD_LIBRARY_PATH
cd /planning-users/yangyeyuan/automodel-pr-4166
PYTHONPATH=$PWD:${EXTRA_PYTHONPATH:-} exec $V/bin/python -m torch.distributed.run --nproc-per-node 1 \
  /planning-users/yangyeyuan/nemo-rl-4166/scratch/super_proxy_memory.py
