"""Cost of scanning parameter storages once per offloaded tensor (600 DTensor params, 1 rank)."""

import os
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29533")
dist.init_process_group("nccl", rank=0, world_size=1)
torch.cuda.set_device(0)
mesh = init_device_mesh("cuda", (1,))
params = [torch.nn.Parameter(distribute_tensor(torch.randn(256, 64, device="cuda"), mesh, [Shard(0)])) for _ in range(600)]
plain = [torch.nn.Parameter(torch.randn(256, 64, device="cuda")) for _ in range(600)]


def scan_to_local(ps):
    with torch.no_grad():
        return {(p.to_local() if isinstance(p, DTensor) else p).untyped_storage()._cdata for p in ps}


def scan_ptr(ps):
    out = set()
    for p in ps:
        t = p._local_tensor if isinstance(p, DTensor) else p
        out.add(t.untyped_storage().data_ptr())
    return out


for name, fn, ps in [("to_local+_cdata dtensor", scan_to_local, params), ("_local_tensor+storage ptr dtensor", scan_ptr, params), ("plain storage ptr", scan_ptr, plain)]:
    fn(ps)
    t0 = time.perf_counter()
    for _ in range(100):
        fn(ps)
    print(f"{name}: {(time.perf_counter() - t0) / 100 * 1e3:.3f} ms per scan of 600 params")
dist.destroy_process_group()
