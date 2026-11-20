#!/usr/bin/env bash
# Run `uv sync --frozen <args>` in $NRL_ROOT with the lockfile's download URLs pointed at
# the internal Artifactory mirrors. The dev machine cannot reach files.pythonhosted.org,
# download-r2.pytorch.org or GitHub release assets, and `uv sync --frozen` ignores index
# settings and downloads from the URLs written in uv.lock. Package versions and sha256
# hashes are unchanged, so uv still verifies every file against upstream.
#
# The original uv.lock is restored on exit (a backup is kept in $NRL_WS/uv.lock.upstream
# while the sync runs). Git sources (github.com) are cloned through the Kiro split proxy.
# Set NRL_WS to pick the workspace (see env.sh).
#
# Examples:
#   nrl-uv-sync.sh                                   # base driver venv (.venv)
#   UV_PROJECT_ENVIRONMENT=$NRL_ROOT/venvs/X nrl-uv-sync.sh --extra mcore --no-install-project
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$(readlink -f "$0")")/env.sh"
# Proxy for github.com git clones. The split proxy connects non-allowlisted hosts directly.
if [[ -f "$HOME/.config/kiro-tunnel/proxy.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$HOME/.config/kiro-tunnel/proxy.env"
    set +a
    export NO_PROXY="$NO_PROXY,.mi-fds.net" no_proxy="$NO_PROXY,.mi-fds.net"
fi

cd "$NRL_ROOT"
backup="$NRL_WS/uv.lock.upstream"
if [[ -e "$backup" ]]; then
    echo "error: $backup exists; another sync is running or a previous one was killed." >&2
    echo "Restore it with: cp $backup $NRL_ROOT/uv.lock && rm $backup" >&2
    exit 1
fi
cp uv.lock "$backup"
restore() { cp "$backup" "$NRL_ROOT/uv.lock" && rm -f "$backup"; }
trap restore EXIT

A=<redacted-url>
sed -i \
    -e "s#url = \"https://files.pythonhosted.org/#url = \"$A/pypi-remote-org/#g" \
    -e "s#url = \"https://download-r2.pytorch.org/#url = \"$A/pytorch-remote/#g" \
    -e "s#url = \"https://download.pytorch.org/#url = \"$A/pytorch-generic-remote/#g" \
    -e "s#url = \"https://github.com/\([^\"]*/releases/download/\)#url = \"$A/github/\1#g" \
    uv.lock

# Build-isolation requirements (setuptools, etc.) are resolved outside uv.lock and need
# a reachable default index.
export UV_DEFAULT_INDEX=$A/api/pypi/pypi-virtual/simple

uv sync --frozen "$@"
