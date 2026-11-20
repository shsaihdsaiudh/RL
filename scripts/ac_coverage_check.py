"""#4166: compare activation checkpointing coverage on the EP=1 Nemotron-H path.

Tiny random-init native Nemotron-V3 (mamba/moe/attention/mlp), 1-GPU FSDP2 mesh, three variants:

  no_ac   - FSDP2 without activation checkpointing (numerical reference)
  old_ac  - only mlp/mamba blocks checkpoint-wrapped (the pre-fix NemotronHParallelizationStrategy)
  new_ac  - NemotronHParallelizationStrategy(activation_checkpointing=True) from the working tree

For each variant: which blocks are wrapped, peak allocated memory of the first train step,
loss over 3 AdamW steps, and max |grad - grad_no_ac| on step 1.

Run: <automodel venv>/bin/python -m torch.distributed.run --nproc-per-node 1 ac_coverage_check.py
"""

import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper, checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy

sys.path.insert(0, os.path.dirname(__file__))
import smoke_nemotron_h  # noqa: E402

LAYERS = ["mamba", "moe", "attention", "mlp", "mamba", "moe", "mamba", "moe"]
SEQ = int(os.environ.get("SEQ", "4096"))
STEPS = 3


def build(device):
    orig = smoke_nemotron_h.TinyNemotronHConfig.__init__

    def patched(self, **overrides):
        orig(self, **overrides)
        self.layers_block_type = list(LAYERS)
        self.num_hidden_layers = len(LAYERS)

    smoke_nemotron_h.TinyNemotronHConfig.__init__ = patched
    try:
        return smoke_nemotron_h.build(device)
    finally:
        smoke_nemotron_h.TinyNemotronHConfig.__init__ = orig


def full(t):
    return t.full_tensor() if hasattr(t, "full_tensor") else t


def strategy_api():
    """Return ``(parallelize, decoder_blocks)`` for either Automodel layout.

    r0.6-era trees expose ``NemotronHParallelizationStrategy`` in ``distributed/parallelizer.py``;
    main moved it to ``models/nemotron_v3/parallelization.py`` as ``NemotronHModelParallelizer``.
    """
    try:
        from nemo_automodel.components.models.nemotron_v3.parallelization import (
            NemotronHModelParallelizer,
            _decoder_blocks,
        )

        return NemotronHModelParallelizer()._apply, _decoder_blocks
    except ImportError:
        from nemo_automodel.components.distributed.parallelizer import (
            NemotronHParallelizationStrategy,
            _nemotronh_decoder_blocks,
        )

        return NemotronHParallelizationStrategy().parallelize, _nemotronh_decoder_blocks


def run(name, variant, device, ids, mesh, ref_grads):
    parallelize, _nemotronh_decoder_blocks = strategy_api()

    model = build(device)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    if variant == "old_ac":
        container, _ = _nemotronh_decoder_blocks(model)
        for key, block in list(container.items()):
            if block.block_type in ("mlp", "mamba"):
                container[key] = checkpoint_wrapper(block)
    model = parallelize(
        model=model, device_mesh=mesh, mp_policy=mp, activation_checkpointing=(variant == "new_ac")
    )
    _, blocks = _nemotronh_decoder_blocks(model)
    wrapped = [
        f"{(b._checkpoint_wrapped_module if isinstance(b, CheckpointWrapper) else b).block_type}"
        f"{'*' if isinstance(b, CheckpointWrapper) else ''}"
        for b in blocks
    ]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses, peak, grads = [], None, None
    for step in range(STEPS):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = model(input_ids=ids)
        logits = out.logits if hasattr(out, "logits") else out
        loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), ids.flatten())
        loss.backward()
        torch.cuda.synchronize()
        if step == 0:
            peak = (torch.cuda.max_memory_allocated() - base) / 2**20
            grads = {
                n.replace("_checkpoint_wrapped_module.", ""): full(p.grad).float().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
        losses.append(loss.item())
        opt.step()
        opt.zero_grad(set_to_none=True)

    finite = all(torch.isfinite(g).all().item() for g in grads.values())
    if ref_grads is None:
        diff = "ref"
    else:
        diff = max((grads[n] - ref_grads[n]).abs().max().item() for n in ref_grads)
        diff = f"{diff:.3e}"
    print(f"[{name}] blocks(*=checkpointed)={wrapped}")
    print(
        f"[{name}] step1 peak_above_base={peak:.0f} MiB  losses={[round(x, 5) for x in losses]}  "
        f"grads_finite={finite}  max|dgrad| vs no_ac={diff}"
    )
    del model, opt
    torch.cuda.empty_cache()
    return grads


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    # NeMo RL disables cuDNN SDPA whenever activation checkpointing is on (automodel/setup.py).
    torch.backends.cuda.enable_cudnn_sdp(False)
    device = torch.device("cuda")
    ids = torch.randint(0, 1024, (1, SEQ), device=device, generator=torch.Generator(device).manual_seed(1))
    mesh = init_device_mesh("cuda", (1, 1, 1), mesh_dim_names=("dp_replicate", "dp_shard_cp", "tp"))
    print(f"torch={torch.__version__} seq={SEQ} layers={LAYERS}")
    # The first variant in a process also pays one-time allocations (kernel workspaces,
    # autotuning, lazy init); run a throwaway pass so every measured variant starts warm.
    run("warmup", "no_ac", device, ids, mesh, None)
    ref = run("no_ac", "no_ac", device, ids, mesh, None)
    run("no_ac_again", "no_ac", device, ids, mesh, ref)
    run("old_ac", "old_ac", device, ids, mesh, ref)
    run("new_ac", "new_ac", device, ids, mesh, ref)
    print("CHECK DONE")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
