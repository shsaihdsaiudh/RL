#!/usr/bin/env bash
# Why is pinned H2D ~27-35 GB/s per rank on a wlcb 8-GPU node vs 50 GB/s on 1 GPU
# (NVIDIA-NeMo/RL#3976)? Separates PCIe-switch sharing from NUMA placement.
# Env: RUN_NAME, GIB (per GPU, 4), ITERS (6).
# Results: /planning-users/yangyeyuan/nemo-rl-3976/logs/numa_bench/$RUN_NAME/
set -uo pipefail

WS=/planning-users/yangyeyuan/nemo-rl-3976
RUN_NAME=${RUN_NAME:-local-$(date +%m%d-%H%M%S)}
OUT=$WS/logs/numa_bench/$RUN_NAME
mkdir -p "$OUT"
exec > >(tee -a "$OUT/entry.log") 2>&1

# shellcheck disable=SC1091
source "$WS/env/env.sh"
PY="$NRL_ROOT/venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python"
BENCH=$WS/env/cml/numa_h2d_bench.py
GIB=${GIB:-4}
ITERS=${ITERS:-6}
# torch device i == nvidia-smi GPU i.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
BAR=/dev/shm/numa-bench-bar

{
    hostname
    nvidia-smi
    nvidia-smi topo -m
    lscpu
    for n in /sys/devices/system/node/node*; do echo "$n cpus=$(cat "$n/cpulist")"; grep MemTotal "$n/meminfo"; done
    command -v numactl && numactl -H
    free -g
    cat /proc/self/status | grep -E 'Cpus_allowed_list|Mems_allowed_list'
} > "$OUT/sysinfo.txt" 2>&1
cat "$OUT/sysinfo.txt"

# Same GPU -> NUMA node lookup as ray.sub's topology_probe.sh.
declare -A LOCAL
while read -r g n; do LOCAL[$g]=$n; done < <(
    nvidia-smi topo -m | awk '/^GPU[0-9]/ {gpu=$1; sub(/GPU/,"",gpu); numa=$(NF-1); sub(/[,-].*/,"",numa); if (numa ~ /^[0-9]+$/) print gpu, numa}'
)
NGPU=$(nvidia-smi -L | wc -l)
# Fallback: the GPU's PCI device numa_node in sysfs.
for ((g = 0; g < NGPU; g++)); do
    if [[ -z ${LOCAL[$g]:-} ]]; then
        bus=$(nvidia-smi -i "$g" --query-gpu=pci.bus_id --format=csv,noheader | tr 'A-F' 'a-f')
        n=$(cat "/sys/bus/pci/devices/${bus:4}/numa_node" 2>/dev/null || echo 0)
        LOCAL[$g]=$(( n < 0 ? 0 : n ))
    fi
done
CPU_NODES=()
for n in /sys/devices/system/node/node*; do
    [[ -n $(cat "$n/cpulist") ]] && CPU_NODES+=("${n##*node}")
done
remote_of() {
    local n
    for n in "${CPU_NODES[@]}"; do [[ $n != "$1" ]] && { echo "$n"; return; }; done
    echo "$1"
}
echo "[numa-bench] ngpu=$NGPU cpu_nodes=${CPU_NODES[*]} local_map=$(for g in "${!LOCAL[@]}"; do printf '%s:%s ' "$g" "${LOCAL[$g]}"; done)"

# Two GPUs behind the same PCIe switch (PIX/PXB from GPU0's row), and one on the same
# NUMA node but a different switch (NODE).
row0=$(nvidia-smi topo -m | awk '/^GPU0/')
SW_PEER="" NODE_PEER=""
for ((j = 1; j < NGPU; j++)); do
    v=$(echo "$row0" | awk -v c=$((j + 2)) '{print $c}')
    [[ -z $SW_PEER && ($v == PIX || $v == PXB) ]] && SW_PEER=$j
    [[ -z $NODE_PEER && $v == NODE ]] && NODE_PEER=$j
done
echo "[numa-bench] gpu0 switch peer=${SW_PEER:-none} same-node peer=${NODE_PEER:-none}"

# run_case NAME "gpu:node gpu:node ..."   (node -1 = no binding)
run_case() {
    local name=$1 specs=($2) pids=() s
    rm -rf "$BAR" && mkdir -p "$BAR"
    for s in "${specs[@]}"; do
        "$PY" "$BENCH" --gpu "${s%%:*}" --node "${s##*:}" --gib "$GIB" --iters "$ITERS" \
            --barrier "$BAR" --nproc "${#specs[@]}" --case "$name" >> "$OUT/results.jsonl" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || echo "[numa-bench] $name: a process failed"; done
    echo "[numa-bench] done $name"
}
all() { local g out=""; for ((g = 0; g < NGPU; g++)); do out+="$g:$($1 "$g") "; done; echo "$out"; }
loc() { echo "${LOCAL[$1]}"; }
rem() { remote_of "${LOCAL[$1]}"; }
unb() { echo -1; }

L0=${LOCAL[0]}
LN=${LOCAL[$((NGPU - 1))]}
run_case solo0-local "0:$L0"
run_case solo0-remote "0:$(remote_of "$L0")"
run_case solo0-unbound "0:-1"
run_case solo$((NGPU - 1))-local "$((NGPU - 1)):$LN"
[[ -n $SW_PEER ]] && run_case pair-switch-local "0:$L0 $SW_PEER:${LOCAL[$SW_PEER]}"
[[ -n $NODE_PEER ]] && run_case pair-node-local "0:$L0 $NODE_PEER:${LOCAL[$NODE_PEER]}"
run_case all-local "$(all loc)"
run_case all-remote "$(all rem)"
run_case all-unbound "$(all unb)"
echo "[numa-bench] all done"
