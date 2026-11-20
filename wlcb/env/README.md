# NeMo RL environment for NVIDIA-NeMo/RL#3976 on cloudml

Bare-metal NeMo RL (no NGC container) on the cloudml dev machine (1x H20, driver 580 / CUDA 13.0).
Built 2026-11-17 from RL main `bfe65cdf6`.

```
/planning-users/yangyeyuan/nemo-rl-3976/
  RL/        NeMo RL checkout with all submodules (copied from a GitHub clone made in WSL)
  env/       toolchain, caches and scripts (this directory)
  hf/        HF_HOME: Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, gsm8k
  logs/      build logs, smoke-run logs, nsys reports (logs/nsys/)
```

## Use

```
source /planning-users/yangyeyuan/nemo-rl-3976/env/env.sh
cd $NRL_ROOT
uv run --locked examples/run_grpo.py --config examples/configs/grpo_math_1B_megatron.yaml ...
```

`env/smoke_grpo_megatron.sh [hydra overrides]` runs 3 steps of colocated GRPO
(Megatron policy + vLLM, Qwen2.5-0.5B, GSM8K). Profile the policy worker with:

```
NRL_NSYS_PROFILE_STEP_RANGE=2:3 NRL_NSYS_WORKER_PATTERNS=megatron_policy_worker \
  env/smoke_grpo_megatron.sh logger.log_dir=$NRL_WS/logs/<name>
```

Reports land in `/dev/shm/nrl-ray/ray/session_*/logs/nsight/`. Copy them to `logs/nsys/`
because /dev/shm does not survive a container restart.

## What is where

| Item | Location | Notes |
|---|---|---|
| uv 0.13.6 | `env/tools/uv` | Same version as `RL/docker/Dockerfile` |
| Python 3.13.14 | `env/python` | Through the Artifactory GitHub mirror |
| CUDA 13.0 toolkit | `env/cuda-13.0` (conda-forge), `env/cuda-home` (CUDA_HOME) | The image's /usr/local/cuda is 12.6 |
| libibverbs-dev headers | `env/sysroot` | DeepEP needs `infiniband/mlx5dv.h` |
| micromamba | `env/tools/mm` | Only used to install the CUDA toolkit |
| uv cache | `env/uv-cache` | Holds every wheel and the source builds |
| driver venv | `RL/.venv` | torch 2.14.0+cu130 |
| worker venvs | `RL/venvs/<actor FQN>` | mcore: torch 2.13.0, TE 2.18, flash-attn 2.8.1, DeepEP, ...; vllm: vLLM 0.33.0 |

Only the mcore and vllm worker venvs (and the ones that add nemo_gym to them) are built.
automodel, sglang, trtllm and modelopt workers are not.

## Network workarounds

The dev machine reaches only the internal network. Downloads go through Artifactory
(`<redacted-host>`): PyPI files, PyTorch wheels, GitHub release assets, conda-forge,
python-build-standalone, Hugging Face. Git sources (github.com) go through the Kiro split
proxy (`~/.config/kiro-tunnel/proxy.env`, ~3 MB/s).

`uv sync --frozen` ignores index settings and downloads from the URLs in `uv.lock`.
`env/nrl-uv-sync.sh` rewrites those URLs to the mirrors for the duration of one sync and
restores `uv.lock` on exit. Versions and sha256 hashes are unchanged. **Any dependency
change needs this script.** Plain `uv run --locked` works offline once the venvs are synced.

## AutoModel workspaces (nemo-rl-4166, nemo-rl-4169)

Added 2026-11-19 from RL main `e6449ae6a` (Automodel submodule r0.6.0 `b916107a`).
These workspaces only have AutoModel venvs; there is no mcore or vLLM worker venv, so
GRPO does not run there yet.

| Item | Location |
|---|---|
| driver venv | `RL/.venv` (torch 2.14.0) |
| worker venvs | `RL/venvs/nemo_rl.models.{policy,value}.workers.automodel_*_worker.*` (`--extra automodel`, torch 2.13.0, TE 2.14.1, flash-attn 2.8.1, mamba-ssm 2.3.2.post1, causal-conv1d 1.6.2.post1, DeepEP, grouped_gemm 1.1.4.post7) |
| pytest venv | `RL/venvs/automodel-test` (`--extra automodel --group test`, editable nemo_rl) |

Build: `NRL_WS=<ws> env/bootstrap_automodel.sh` (driver venv, both worker venvs, phase 2,
`check_venv.py automodel`). The pytest venv:

```
UV_LINK_MODE=hardlink UV_PROJECT_ENVIRONMENT=$NRL_ROOT/venvs/automodel-test \
  env/nrl-uv-sync.sh --extra automodel --group test
```

Run the unit tests from `RL/tests` (CI uses `uv run --extra automodel`, which would
rewrite the driver venv, so use the separate venv instead):

```
cmdev run 'cd tests && ../venvs/automodel-test/bin/python -m pytest unit/models/automodel -q --automodel-only'
```

Verified on 2026-11-19 in both workspaces: `check_venv.py automodel` OK in both worker
venvs, `tests/unit/models/automodel` 417 passed, `examples/run_sft.py` 3 steps on
Qwen2.5-0.5B (squad) through the Ray AutoModel worker. In nemo-rl-4166,
`scratch/smoke_nemotron_h.py` builds a random-init 4-layer Nemotron-H
(mamba/moe/attention/mlp), runs fwd/bwd with the mamba_ssm kernels, and applies the
Nemotron-H FSDP2 strategy with activation checkpointing on a 1-GPU mesh.

Git sources need the Kiro split proxy on 127.0.0.1:18889. If the dev container lost it
(uv fails with `Failed to connect to 127.0.0.1 port 18889`), restart it with
`source ~/.config/kiro-tunnel/dev-shell.sh; kiro-tunnel-start-server; kiro-split-proxy-start`.

## Rebuild

```
nohup env/bootstrap.sh        > logs/bootstrap.log       2>&1 &   # third-party deps, ~1 h cold
nohup env/prefetch_project.sh > logs/prefetch_venvs.log  2>&1 &   # editable nemo_rl into worker venvs
```

`bootstrap.sh` mirrors phase 1 of the Dockerfile's worker-venv prefetch (deps only,
`--no-install-project`). `prefetch_project.sh` is phase 2 (`nemo_rl/utils/prefetch_venvs.py`).
Without phase 2, Ray workers fail with `No module named 'nemo_rl'`. After changing
`pyproject.toml`/`uv.lock`, rerun both, or set `NRL_FORCE_REBUILD_VENVS=true`. Note that a
force rebuild uses the unmirrored lock and will fail here; use the scripts instead.

Cold build time: driver venv ~12 min, mcore venv ~60 min (flash-attn is capped at
MAX_JOBS=4 by `RL/pyproject.toml`), vllm venvs ~5 min, phase 2 ~10 min.

## Build problems hit and fixed

1. `nvidia-resiliency-ext`: `required files libcupti.so and cupti.h not found`. Its build
   uses `os.walk(CUDA_HOME)`, which does not follow directory symlinks. `env/cuda-home` was
   rebuilt with real directories and per-file symlinks (`cp -rs`).
2. DeepEP: `infiniband/mlx5dv.h: No such file`. The `libibverbs-dev` deb was extracted
   into `env/sysroot`; `env.sh` adds it to `CPATH`/`LIBRARY_PATH`.
3. Ray workers: `No module named 'nemo_rl'`. Fixed by phase 2 above.

## Verified

- `env/check_venv.py mcore|vllm`: imports plus GPU ops. TE `LayerNormLinear` fwd/bwd,
  flash-attn, and TE using cuDNN 9.26 from the venv (not the system 9.5).
- `env/smoke_grpo_megatron.sh`: 3 steps finish, about 13 s/step.
- nsys on step 2 (`logs/nsys/megatron_policy_worker_2:3_112478.nsys-rep`):
  `offload_before_refit` 2 calls, 5.15 s total (0.91 s min). D2H 6.92 GB in 955 copies,
  2.91 s total. This matches the triage in the issue thread.
- Known noise: at shutdown vLLM prints `CUDA Error: invalid argument at
  csrc/cumem_allocator.cpp:249`. This happens after training has finished.
