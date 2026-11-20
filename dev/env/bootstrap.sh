#!/usr/bin/env bash
# Build the driver venv and the Megatron / vLLM worker venvs, like the hermetic stage of
# RL/docker/Dockerfile. Safe to re-run: finished steps are no-ops for uv.
# Run on the dev machine (NRL_WS selects the workspace, see env.sh):
#   NRL_WS=/planning-users/yangyeyuan/nemo-rl-3976 nohup \
#     /planning-users/yangyeyuan/nemo-rl-3976/env/bootstrap.sh > $NRL_WS/logs/bootstrap.log 2>&1 &
# Then run env/prefetch_project.sh (editable nemo_rl into the worker venvs).
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$NRL_ROOT"
step() { echo "=== [$(date '+%F %T')] $*"; }
# Venvs and the uv cache are on the same JuiceFS volume, so hardlinks work and avoid
# copying several GB per venv (the Dockerfile also hardlinks).
export UV_LINK_MODE=hardlink

step "driver venv (.venv) in $NRL_ROOT"
uv venv --allow-existing .venv
"$HERE/nrl-uv-sync.sh"

step "list worker venvs"
VENV_DIR="${NEMO_RL_VENV_DIR:-$NRL_ROOT/venvs}"
mkdir -p "$VENV_DIR"
.venv/bin/python nemo_rl/distributed/actor_environments.py all > "$NRL_WS/logs/actor_venvs.tsv"
cat "$NRL_WS/logs/actor_venvs.tsv"

# Only the backends needed for GRPO with Megatron + vLLM (MegatronPolicyWorker and
# VllmGenerationWorker also carry nemo_gym). mcore first: it has the source builds
# (TE, flash-attn, mamba, DeepEP, ...).
for want in "--extra mcore" "--extra mcore --extra nemo_gym" "--extra vllm" "--extra vllm --extra nemo_gym"; do
    while IFS=$'\t' read -r venv_name _stage extras; do
        [[ "$extras" == "$want" ]] || continue
        step "worker venv $venv_name ($extras)"
        uv venv --allow-existing "$VENV_DIR/$venv_name"
        # shellcheck disable=SC2086
        UV_PROJECT_ENVIRONMENT="$VENV_DIR/$venv_name" \
            "$HERE/nrl-uv-sync.sh" $extras --no-install-project
    done < "$NRL_WS/logs/actor_venvs.tsv"
done
step "BOOTSTRAP DONE (next: env/prefetch_project.sh)"
