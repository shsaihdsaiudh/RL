#!/usr/bin/env bash
# Build the driver venv and the AutoModel worker venvs (AutomodelPolicyWorker,
# AutomodelValueWorker; both `--extra automodel`) for one NeMo RL workspace, then add the
# editable nemo_rl install to them (phase 2). Counterpart of bootstrap.sh +
# prefetch_project.sh, which only build the Megatron and vLLM venvs.
# Safe to re-run: finished steps are no-ops for uv.
#   NRL_WS=/planning-users/yangyeyuan/nemo-rl-4166 nohup \
#     /planning-users/yangyeyuan/nemo-rl-3976/env/bootstrap_automodel.sh \
#     > /planning-users/yangyeyuan/nemo-rl-4166/logs/bootstrap_automodel.log 2>&1 &
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$NRL_ROOT"
step() { echo "=== [$(date '+%F %T')] $*"; }
export UV_LINK_MODE=hardlink

step "driver venv (.venv) in $NRL_ROOT"
uv venv --allow-existing .venv
"$HERE/nrl-uv-sync.sh"

step "list worker venvs"
VENV_DIR="${NEMO_RL_VENV_DIR:-$NRL_ROOT/venvs}"
mkdir -p "$VENV_DIR"
.venv/bin/python nemo_rl/distributed/actor_environments.py all > "$NRL_WS/logs/actor_venvs.tsv"

names=()
while IFS=$'\t' read -r venv_name _stage extras; do
    [[ "$extras" == "--extra automodel" ]] || continue
    names+=("$venv_name")
    step "worker venv $venv_name ($extras)"
    uv venv --allow-existing "$VENV_DIR/$venv_name"
    # shellcheck disable=SC2086
    UV_PROJECT_ENVIRONMENT="$VENV_DIR/$venv_name" \
        "$HERE/nrl-uv-sync.sh" $extras --no-install-project
done < "$NRL_WS/logs/actor_venvs.tsv"
[[ ${#names[@]} -gt 0 ]] || { echo "no '--extra automodel' actors found" >&2; exit 1; }

step "phase 2: editable nemo_rl into ${names[*]}"
uv run --locked nemo_rl/utils/prefetch_venvs.py "${names[@]}"

step "check"
for n in "${names[@]}"; do
    "$VENV_DIR/$n/bin/python" "$HERE/check_venv.py" automodel
done
step "BOOTSTRAP_AUTOMODEL DONE"
