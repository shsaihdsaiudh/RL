# NeMo RL dev environment (bare metal, no NGC container) on cloudml.
# Usage: source /planning-users/yangyeyuan/nemo-rl-3976/env/env.sh
#
# The toolchain in this directory (uv, Python, CUDA 13.0, uv cache with all compiled
# wheels, HF cache) is shared by every NeMo RL workspace. Each workspace has its own
# RL checkout, venvs and logs. Select one by exporting NRL_WS before sourcing, e.g.
#   export NRL_WS=/planning-users/yangyeyuan/nemo-rl-4104
# Default: nemo-rl-3976. Everything heavy lives on /planning-users because the root
# overlay has only a few GB free. See env/README.md for how it was built.

NRL_ENV=/planning-users/yangyeyuan/nemo-rl-3976/env
export NRL_WS=${NRL_WS:-/planning-users/yangyeyuan/nemo-rl-3976}
export NRL_ROOT=$NRL_WS/RL
mkdir -p "$NRL_WS/logs"

# uv + managed Python 3.13.14 (the version pinned in RL/.python-version).
export PATH="$NRL_ENV/tools/uv/bin:$NRL_ENV/cuda-home/bin:$PATH"
export UV_CACHE_DIR=$NRL_ENV/uv-cache
export UV_PYTHON_INSTALL_DIR=$NRL_ENV/python
export UV_PYTHON_INSTALL_MIRROR=<redacted-url>
export UV_HTTP_TIMEOUT=600

# CUDA 13.0 toolkit (conda-forge, matches torch cu130 and the 580 driver). The image's
# /usr/local/cuda is 12.6 and must not be used for builds.
export CUDA_HOME=$NRL_ENV/cuda-home
export CUDA_PATH=$CUDA_HOME
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include/cccl${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
# libibverbs-dev headers (DeepEP needs infiniband/mlx5dv.h), extracted from the Ubuntu
# .deb into env/sysroot because the image only has the runtime libraries.
export CPATH="$NRL_ENV/sysroot/usr/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$NRL_ENV/sysroot/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
# Only build kernels for the H20 (sm_90).
export TORCH_CUDA_ARCH_LIST="9.0"
export FLASH_ATTN_CUDA_ARCHS=90
export NVTE_CUDA_ARCHS=90
export NVTE_FRAMEWORK=pytorch
export NVTE_BUILD_MAX_JOBS=8
export NVTE_BUILD_THREADS_PER_JOB=2
export NVTE_WITH_NCCL_EP=0
export MAX_JOBS=16

# Same as docker/Dockerfile: point TE at the pip-installed cuDNN, not the system 9.5.
NRL_SITE=$NRL_ROOT/.venv/lib/python3.13/site-packages
export CUDNN_HOME=$NRL_SITE/nvidia/cudnn
export CUDNN_PATH=$CUDNN_HOME
export LD_LIBRARY_PATH="$NRL_SITE/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Same as docker/Dockerfile.
export RAY_USAGE_STATS_ENABLED=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

# Keep Ray sessions, temp files and .pyc off the small root overlay and off JuiceFS
# (writing .pyc to JuiceFS makes first imports take minutes).
export RAY_TMPDIR=/dev/shm/nrl-ray
export TMPDIR=/dev/shm/nrl-tmp
export PYTHONPYCACHEPREFIX=/dev/shm/nrl-pycache
mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$PYTHONPYCACHEPREFIX"

# Hugging Face models/datasets through the Artifactory HF remote (cache shared by all
# workspaces).
export HF_HOME=/planning-users/yangyeyuan/nemo-rl-3976/hf
export HF_ENDPOINT=<redacted-url>
export HF_HUB_DOWNLOAD_TIMEOUT=600
# Artifactory's HF remote has no Xet endpoint (xet-read-token -> 404); use plain HTTP.
export HF_HUB_DISABLE_XET=1

# Artifactory redirects downloads to this storage host.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}.mi-fds.net" no_proxy="${no_proxy:+$no_proxy,}.mi-fds.net"

unset NRL_SITE
