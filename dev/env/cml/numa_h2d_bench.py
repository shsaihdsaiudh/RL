"""Pinned host <-> GPU copy bandwidth under NUMA binding choices (NVIDIA-NeMo/RL#3976).

Host buffers are built like OptimizerStateOffloader's: 1 GiB torch.empty chunks pinned
with cudaHostRegister. Binding mirrors nemo_rl.distributed.numa_utils.bind_to_gpu_numa:
sched_setaffinity to the node's CPUs plus an MPOL_PREFERRED memory policy, both set
before the buffers are allocated and pinned (cudaHostRegister faults the pages in).

Several copies of this script run at once (one per GPU) and start copying together
through a file barrier. Prints one JSON line.

Usage: python numa_h2d_bench.py --gpu G --node N|-1 --gib 4 --iters 6 \
           --barrier DIR --nproc K --case NAME
"""

import argparse
import bisect
import ctypes
import json
import os
import statistics
import time

CHUNK = 1 << 30
MPOL_PREFERRED = 1
SYS_SET_MEMPOLICY = 238  # x86_64


def node_cpus(node: int) -> set[int]:
    text = open(f"/sys/devices/system/node/node{node}/cpulist").read().strip()
    cpus: set[int] = set()
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def bind(node: int) -> bool:
    """Prefer memory on ``node``; also pin to its CPUs where the cpuset allows any.

    Returns whether the CPU pinning happened. The memory policy is what decides
    where the pinned pages land.
    """
    cpus = node_cpus(node) & os.sched_getaffinity(0)
    if cpus:
        os.sched_setaffinity(0, cpus)
    mask = ctypes.c_ulong(1 << node)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(SYS_SET_MEMPOLICY, MPOL_PREFERRED, ctypes.byref(mask), 64) != 0:
        raise OSError(ctypes.get_errno(), "set_mempolicy failed")
    return bool(cpus)


def pages_per_node(ptrs: list[int]) -> dict[str, int]:
    """Sum the N<k>= page counts of the mappings that hold the given addresses."""
    maps = []
    for line in open("/proc/self/numa_maps"):
        fields = line.split()
        maps.append((int(fields[0], 16), fields[1:]))
    maps.sort()
    starts = [m[0] for m in maps]
    total: dict[str, int] = {}
    for idx in {bisect.bisect_right(starts, ptr) - 1 for ptr in ptrs}:
        for f in maps[idx][1]:
            if f.startswith("N") and "=" in f:
                k, v = f.split("=")
                total[k] = total.get(k, 0) + int(v)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--node", type=int, default=-1)
    ap.add_argument("--gib", type=int, default=4)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--barrier", required=True)
    ap.add_argument("--nproc", type=int, required=True)
    ap.add_argument("--case", required=True)
    args = ap.parse_args()

    cpu_bound = bind(args.node) if args.node >= 0 else False
    import torch

    torch.cuda.set_device(args.gpu)
    cudart = torch.cuda.cudart()
    t0 = time.perf_counter()
    host = [torch.empty(CHUNK, dtype=torch.uint8) for _ in range(args.gib)]
    for h in host:
        torch.cuda.check_error(cudart.cudaHostRegister(h.data_ptr(), CHUNK, 0))
    pin_s = time.perf_counter() - t0
    dev = [torch.empty(CHUNK, dtype=torch.uint8, device="cuda") for _ in host]
    placement = pages_per_node([h.data_ptr() for h in host])
    # Warm-up, outside the timed window.
    for h, d in zip(host, dev):
        d.copy_(h, non_blocking=True)
        h.copy_(d, non_blocking=True)
    torch.cuda.synchronize()

    os.makedirs(args.barrier, exist_ok=True)

    def barrier(tag: str) -> None:
        # Every process copies at the same time, in the same direction.
        open(os.path.join(args.barrier, f"{tag}.{args.gpu}"), "w").close()
        prefix = tag + "."
        while sum(f.startswith(prefix) for f in os.listdir(args.barrier)) < args.nproc:
            time.sleep(0.002)

    h2d, d2h = [], []
    for i in range(args.iters):
        barrier(f"h{i}")
        t = time.perf_counter()
        for h, d in zip(host, dev):
            d.copy_(h, non_blocking=True)
        torch.cuda.synchronize()
        h2d.append(time.perf_counter() - t)
        barrier(f"d{i}")
        t = time.perf_counter()
        for h, d in zip(host, dev):
            h.copy_(d, non_blocking=True)
        torch.cuda.synchronize()
        d2h.append(time.perf_counter() - t)

    nbytes = args.gib * CHUNK
    print(
        json.dumps(
            {
                "case": args.case,
                "gpu": args.gpu,
                "node": args.node,
                "cpu_bound": cpu_bound,
                "cpus": f"{min(os.sched_getaffinity(0))}-{max(os.sched_getaffinity(0))}",
                "pages": placement,
                "pin_s": round(pin_s, 2),
                "h2d_gbs": round(nbytes / statistics.median(h2d) / 1e9, 1),
                "d2h_gbs": round(nbytes / statistics.median(d2h) / 1e9, 1),
                "h2d_all": [round(nbytes / x / 1e9, 1) for x in h2d],
            }
        ),
        flush=True,
    )
    for h in host:
        cudart.cudaHostUnregister(h.data_ptr())


if __name__ == "__main__":
    main()
