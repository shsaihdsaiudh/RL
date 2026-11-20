"""Automodel PR #4834: full / selective activation checkpointing on a dense native Nemotron-H.

Dense Nemotron-H (Nano 4B/9B/12B) has moe_config=None and a config without any expert-count
field. Builds a small random-init model of that shape, parallelizes it on a 1-GPU FSDP2 mesh
through NemotronHModelParallelizer, and runs 3 AdamW steps with no AC, full AC and selective
AC. Reports which blocks are wrapped, step-1 peak memory, per-step losses and the max step-1
grad difference against the no-AC run (a second no-AC run gives the noise floor).

Run from an Automodel checkout:
  PYTHONPATH=$PWD python -m torch.distributed.run --nproc-per-node 1 dense_ac_check.py
Env: SEQ (default 4096), ATTN (te|sdpa, default te).
"""

import json
import os

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy

SEQ = int(os.environ.get("SEQ", "4096"))
ATTN = os.environ.get("ATTN", "te")
LAYERS = ["mamba", "attention", "mlp", "mamba", "attention", "mlp", "mamba", "mlp"]
VOCAB = 1024
MiB = 2**20
EXPERT_FIELDS = ("num_experts", "moe_num_experts", "n_routed_experts", "num_local_experts")


class DenseConfig:
    """Dense Nemotron-H text config: the Nano-9B-v2 field set, scaled down, no MoE fields."""

    def __init__(self):
        self.hidden_size = 256
        self.num_attention_heads = 8
        self.num_key_value_heads = 2
        self.head_dim = 32
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.intermediate_size = 1024
        self.mlp_bias = False
        self.mlp_hidden_act = "relu2"
        self.mamba_num_heads = 8
        self.mamba_head_dim = 64
        self.ssm_state_size = 64
        self.n_groups = 2
        self.chunk_size = 128
        self.conv_kernel = 4
        self.use_conv_bias = True
        self.mamba_hidden_act = "silu"
        self.time_step_limit = (0.0, float("inf"))
        self.time_step_min = 0.001
        self.time_step_max = 0.1
        self.time_step_floor = 1e-4
        self.use_bias = False
        self.layer_norm_epsilon = 1e-5
        self.layers_block_type = list(LAYERS)
        self.num_hidden_layers = len(LAYERS)
        self.vocab_size = VOCAB
        self.torch_dtype = "bfloat16"
        self.initializer_range = 0.02
        self.rescale_prenorm_residual = True
        self.residual_in_fp32 = False
        self.num_nextn_predict_layers = 0
        self.tie_word_embeddings = False

    def to_dict(self):
        return vars(self)


def build(device):
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

    backend = BackendConfig(
        attn=ATTN, linear="te" if ATTN == "te" else "torch", rms_norm="torch_fp32", rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )
    cfg = DenseConfig()
    assert not any(hasattr(cfg, f) for f in EXPERT_FIELDS)
    torch.manual_seed(0)
    with torch.device(device):
        model = NemotronHForCausalLM(cfg, backend=backend)
    model.initialize_weights(buffer_device=device, dtype=torch.bfloat16)
    assert model.model.moe_config is None
    return model.to(torch.float32)


def run(tag, activation_checkpointing, device, mesh, batches):
    from nemo_automodel.components.models.nemotron_v3.parallelization import (
        NemotronHModelParallelizer,
        _decoder_blocks,
    )

    model = build(device)
    if activation_checkpointing == "old":
        # Behavior before the PR: only mamba and mlp blocks checkpoint-wrapped.
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        container, _ = _decoder_blocks(model)
        for key, block in list(container.items()):
            if block.block_type in ("mlp", "mamba"):
                container[key] = checkpoint_wrapper(block)
        activation_checkpointing = False
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    model = NemotronHModelParallelizer()._apply(
        model=model, device_mesh=mesh, mp_policy=mp, activation_checkpointing=activation_checkpointing
    )
    _, blocks = _decoder_blocks(model)
    wrapped = " ".join(
        ("*" + b._checkpoint_wrapped_module.block_type if isinstance(b, CheckpointWrapper) else b.block_type)
        for b in blocks
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses, grads, peak = [], None, None
    for step, ids in enumerate(batches):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = model(input_ids=ids)
        logits = out.logits if hasattr(out, "logits") else out
        loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), ids.flatten())
        del out, logits
        loss.backward()
        torch.cuda.synchronize()
        if step == 0:
            peak = (torch.cuda.max_memory_allocated() - base) / MiB
            grads = {
                n.replace("_checkpoint_wrapped_module.", ""): (
                    p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad
                ).float().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
        losses.append(loss.item())
        opt.step()
        opt.zero_grad(set_to_none=True)
    del model, opt
    torch.cuda.empty_cache()
    return dict(tag=tag, wrapped=wrapped, step1_peak_mib=round(peak, 1), losses=losses), grads


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device("cuda")
    mesh = init_device_mesh("cuda", (1, 1, 1), mesh_dim_names=("dp_replicate", "dp_shard_cp", "tp"))
    print(f"torch={torch.__version__} seq={SEQ} attn={ATTN} gpu={torch.cuda.get_device_name()}", flush=True)
    gens = [torch.Generator(device).manual_seed(10 + i) for i in range(3)]
    batches = [torch.randint(0, VOCAB, (1, SEQ), device=device, generator=g) for g in gens]
    run("warmup", False, device, mesh, batches[:1])
    ref = None
    variants = (
        ("no_ac", False),
        ("no_ac_again", False),
        ("old_ac_mamba_mlp", "old"),
        ("full_ac", True),
        ("selective_ac", "selective"),
    )
    for tag, ac in variants:
        rec, grads = run(tag, ac, device, mesh, batches)
        if ref is None:
            ref = grads
        assert grads.keys() == ref.keys(), (tag, set(grads) ^ set(ref))
        rec["n_grads"] = len(grads)
        rec["grads_finite"] = all(torch.isfinite(g).all().item() for g in grads.values())
        diffs = {n: (grads[n] - ref[n]).abs().max().item() for n in ref}
        worst = max(diffs, key=diffs.get)
        rec["max_dgrad_vs_no_ac"] = diffs[worst]
        rec["worst_param"] = worst
        rec["worst_param_ref_absmax"] = ref[worst].abs().max().item()
        rec["max_rel_dgrad"] = max(diffs[n] / max(ref[n].abs().max().item(), 1e-30) for n in ref)
        print("RESULT " + json.dumps(rec), flush=True)
    print("CHECK DONE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
