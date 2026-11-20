#!/usr/bin/env bash
# Copy the #4166 workspace pieces a cml job needs from bj (ssh cloudml) to wlcb storage (ssh cloudml-cpu),
# through this machine, in N parallel tar streams. Paths stay identical (/planning-users/...).
# Usage: copy_to_wlcb.sh [N]   Logs: /tmp/wlcb-copy/
set -uo pipefail
N=${1:-8}
L=/tmp/wlcb-copy
mkdir -p $L
rm -f $L/*
P=planning-users/yangyeyuan
TARGETS="$P/nemo-rl-4166/RL $P/nemo-rl-4166/scratch $P/nemo-rl-3976/env/uv-cache/environments-v2/nemo-rl-cp3.13.14-4e4bb225d23dacc6"
EXCLUDE="-path $P/nemo-rl-4166/RL/venvs/automodel-test -prune -o -path $P/nemo-rl-4166/RL/venvs/nemo_rl.models.value.workers.automodel_value_worker.AutomodelValueWorker -prune -o -path $P/nemo-rl-4166/RL/tests/unit/unit_results -prune -o -name __pycache__ -prune -o"
# size<TAB>path for every non-directory, plus the directories (size 0) so empty ones survive.
ssh cloudml "cd / && find $TARGETS $EXCLUDE \( -type d -printf '0\t%p\n' -o -printf '%s\t%p\n' \)" > $L/all.tsv
echo "entries=$(wc -l < $L/all.tsv) bytes=$(awk -F'\t' '{s+=$1} END {print s}' $L/all.tsv)"
# Greedy split by size: biggest files first, each to the lightest list. Directories go to list 0.
sort -t$'\t' -k1,1nr $L/all.tsv | awk -F'\t' -v n=$N -v dir=$L '
    $1 == 0 { print $2 > (dir "/list0"); next }
    { best = 0; for (i = 1; i < n; i++) if (load[i] < load[best]) best = i; load[best] += $1; print $2 > (dir "/list" best) }
    END { for (i = 0; i < n; i++) printf "list%d %.2f GiB\n", i, load[i] / 2^30 }'
for i in $(seq 0 $((N - 1))); do
    scp -q $L/list$i cloudml:/tmp/wlcb-copy-list$i
done
start=$(date +%s)
for i in $(seq 0 $((N - 1))); do
    ( ssh cloudml "tar -C / --no-recursion -cf - -T /tmp/wlcb-copy-list$i" | ssh cloudml-cpu "tar -C / -xf -"
      echo "list$i rc=${PIPESTATUS[*]} secs=$(( $(date +%s) - start ))" ) > $L/stream$i.log 2>&1 &
done
wait
cat $L/stream*.log
echo "COPY_DONE secs=$(( $(date +%s) - start ))"
