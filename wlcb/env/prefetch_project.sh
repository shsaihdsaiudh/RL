#!/usr/bin/env bash
# Phase 2 of the worker-venv prefetch (same as RL/docker/Dockerfile release stage): add the
# editable nemo_rl install to each worker venv that bootstrap.sh built with
# --no-install-project. Without it, Ray workers fail with "No module named 'nemo_rl'".
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$NRL_ROOT"
mapfile -t names < <(cut -f1 "$NRL_WS/logs/actor_venvs.tsv" | while read -r n; do
    [[ -x "${NEMO_RL_VENV_DIR:-$NRL_ROOT/venvs}/$n/bin/python" ]] && echo "$n"; done)
printf '%s\n' "${names[@]}"
UV_LINK_MODE=hardlink uv run --locked nemo_rl/utils/prefetch_venvs.py "${names[@]}"
echo "PREFETCH DONE"
