"""Which part of the FSDP2+AC smoke run produces NaN? Runs four variants of the tiny model."""

import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy

sys.path.insert(0, os.path.dirname(__file__))
from smoke_nemotron_h import build  # noqa: E402


def run(name, model, ids):
    out = model(input_ids=ids)
    logits = out.logits if hasattr(out, "logits") else out
    loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), ids.flatten())
    nan_fwd = not torch.isfinite(logits).all().item()
    loss.backward()
    bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(
        p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad).all()]
    print(f"{name:22s} loss={loss.item():.4f} logits_nonfinite={nan_fwd} nonfinite_grads={len(bad)} {bad[:3]}")


def main():
    from nemo_automodel.components.distributed.parallelizer import (
        NemotronHParallelizationStrategy,
        _nemotronh_decoder_blocks,
    )

    dist.init_process_group("nccl")
    torch.cuda.set_device(0)
    dev = torch.device("cuda")
    ids = torch.randint(0, 1024, (2, 512), device=dev)
    mesh = init_device_mesh("cuda", (1, 1, 1), mesh_dim_names=("dp_replicate", "dp_shard_cp", "tp"))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    run("plain", build(dev), ids)

    m = build(dev)
    container, blocks = _nemotronh_decoder_blocks(m)
    items = container.items() if isinstance(container, torch.nn.ModuleDict) else enumerate(container)
    for key, b in list(items):
        if b.block_type in ("mlp", "mamba"):
            container[key] = checkpoint_wrapper(b)
    run("AC only (mlp,mamba)", m, ids)

    run("FSDP2 only", NemotronHParallelizationStrategy().parallelize(build(dev), mesh, mp_policy=mp), ids)
    run(
        "FSDP2 + AC",
        NemotronHParallelizationStrategy().parallelize(build(dev), mesh, mp_policy=mp, activation_checkpointing=True),
        ids,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
