"""#4166: activation memory of Nemotron 3.5 Super blocks on one GPU, to project the 16-rank run.

The real run (120B-A12B, FSDP2 over 16 GB200, EP=1, 32K-token microbatches) does not fit on
one H20. Activation memory per rank does not depend on the FSDP degree, so this script builds
small native Nemotron-V3 stacks with the real Super layer dimensions and the recipe backend
(TE attention/linear, fp32 RMSNorm, gmm experts, DeepEP dispatcher) and measures, per block
type, how much memory a block keeps alive from forward until backward:

  no_ac   - no activation checkpointing
  old_ac  - mamba/mlp blocks checkpoint-wrapped only (the pre-fix EP=1 behavior)
  new_ac  - NemotronHModelParallelizer(activation_checkpointing=True) from the working tree

The only reduced dimension is n_routed_experts (512 -> N_EXPERTS, top-22 kept): routed rows per
layer are tokens x 22 either way; the router logits shrink from T x 512 to T x N_EXPERTS fp32.

Per-layer cost = retained([t, t]) - retained([t]), which cancels embedding, final norm and
lm_head. MODE=head measures the real-vocabulary output layer separately.

Run (from an Automodel checkout):
  PYTHONPATH=$PWD python -m torch.distributed.run --nproc-per-node 1 super_proxy_memory.py
Env: SEQ (default 32768), N_EXPERTS (32), MODE (layers|stack|head), DISPATCHER (deepep).
"""

import json
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper, checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy

SEQ = int(os.environ.get("SEQ", "32768"))
N_EXPERTS = int(os.environ.get("N_EXPERTS", "32"))
MODE = os.environ.get("MODE", "layers")
DISPATCHER = os.environ.get("DISPATCHER", "deepep")
EXPERTS = os.environ.get("EXPERTS", "gmm")
ITERS = 3
GiB = 2**30
# hybrid_override_pattern of NVIDIA-Nemotron-3-Super-120B-A12B: M=mamba, E=moe, *=attention.
SUPER_PATTERN = "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
CODE = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}


class SuperConfig:
    """Nemotron-3-Super-120B-A12B text config fields used by the native Nemotron-V3 model."""

    def __init__(self, layers, vocab_size):
        self.hidden_size = 4096
        self.num_attention_heads = 32
        self.num_key_value_heads = 2
        self.head_dim = 128
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.intermediate_size = 2688
        self.mlp_bias = False
        self.mlp_hidden_act = "relu2"
        self.mamba_num_heads = 128
        self.mamba_head_dim = 64
        self.ssm_state_size = 128
        self.n_groups = 8
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
        self.layers_block_type = list(layers)
        self.num_hidden_layers = len(layers)
        self.vocab_size = vocab_size
        self.torch_dtype = "bfloat16"
        self.initializer_range = 0.02
        self.rescale_prenorm_residual = True
        self.residual_in_fp32 = False
        self.n_routed_experts = N_EXPERTS
        self.num_experts_per_tok = 22
        self.n_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 5.0
        self.moe_intermediate_size = 2688
        self.moe_latent_size = 1024
        self.norm_topk_prob = True
        self.moe_shared_expert_intermediate_size = 5376
        self.n_shared_experts = 1
        self.num_nextn_predict_layers = 0
        self.tie_word_embeddings = False

    def to_dict(self):
        return vars(self)


def build(layers, vocab_size, device):
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

    # Backend of examples/configs/recipes/llm/dapo-nemotron3.5-super-vl-120BA12B-16n8g-automodel.yaml
    backend = BackendConfig(
        attn="te",
        linear="te",
        rms_norm="torch_fp32",
        experts=EXPERTS,
        dispatcher=DISPATCHER,
        fake_balanced_gate=False,
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )
    torch.manual_seed(0)
    with torch.device(device):
        model = NemotronHForCausalLM(SuperConfig(layers, vocab_size), backend=backend)
    model.initialize_weights(buffer_device=device, dtype=torch.bfloat16)
    # FP32 stored parameters, BF16 compute (FSDP MixedPrecisionPolicy below), as in the run.
    return model.to(torch.float32)


def parallelize(model, mesh, variant):
    from nemo_automodel.components.models.nemotron_v3.parallelization import (
        NemotronHModelParallelizer,
        _decoder_blocks,
    )

    if variant == "old_ac":
        container, _ = _decoder_blocks(model)
        for key, block in list(container.items()):
            if block.block_type in ("mlp", "mamba"):
                container[key] = checkpoint_wrapper(block)
    out_dtype = os.environ.get("OUTPUT_DTYPE", "none")
    mp = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        # NeMo RL passes automodel_cfg.fsdp_output_dtype (default float32) here.
        output_dtype=None if out_dtype == "none" else getattr(torch, out_dtype),
    )
    model = NemotronHModelParallelizer()._apply(
        model=model, device_mesh=mesh, mp_policy=mp, activation_checkpointing=variant.startswith("new_ac")
    )
    _, blocks = _decoder_blocks(model)
    wrapped = "".join(
        ("*" if isinstance(b, CheckpointWrapper) else "") + (b._checkpoint_wrapped_module if isinstance(b, CheckpointWrapper) else b).block_type[:2]
        + " "
        for b in blocks
    )
    return model, wrapped.strip()


def train_iters(model, ids):
    """Run ITERS forward/backward passes; return memory and time of the last one."""
    rec = None
    for _ in range(ITERS):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        out = model(input_ids=ids)
        logits = out.logits if hasattr(out, "logits") else out
        loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), ids.flatten())
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        retained = torch.cuda.memory_allocated() - base
        fwd_peak = torch.cuda.max_memory_allocated() - base
        torch.cuda.reset_peak_memory_stats()
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        bwd_peak = torch.cuda.max_memory_allocated() - base
        finite = bool(torch.isfinite(loss).item())
        del out, logits, loss
        rec = dict(
            retained_gib=retained / GiB,
            fwd_peak_gib=fwd_peak / GiB,
            bwd_peak_gib=bwd_peak / GiB,
            fwd_s=t1 - t0,
            bwd_s=t2 - t1,
            loss_finite=finite,
        )
    model.zero_grad(set_to_none=True)
    return rec


def measure(tag, layers, variant, device, mesh, vocab_size=1024):
    model = build([CODE[c] for c in layers], vocab_size, device)
    model, wrapped = parallelize(model, mesh, variant)
    ids = torch.randint(0, vocab_size, (1, SEQ), device=device, generator=torch.Generator(device).manual_seed(1))
    rec = train_iters(model, ids)
    experts = sorted({type(m.experts).__name__ for m in model.modules() if hasattr(m, "experts")})
    rec.update(tag=tag, layers=layers, variant=variant, wrapped=wrapped, experts_cls=experts)
    print("RESULT " + json.dumps(rec), flush=True)
    del model
    torch.cuda.empty_cache()
    return rec


def mode_layers(device, mesh):
    measure("warmup", "ME*", "no_ac", device, mesh)
    res = {}
    for t in ("M", "E", "*"):
        for variant in ("no_ac", "new_ac"):
            one = measure(f"{t}x1", t, variant, device, mesh)
            two = measure(f"{t}x2", t + t, variant, device, mesh)
            per = {k: two[k] - one[k] for k in ("retained_gib", "fwd_peak_gib", "bwd_peak_gib", "fwd_s", "bwd_s")}
            res[f"{t}/{variant}"] = per
            print(f"PERLAYER {CODE[t]:9s} {variant:6s} " + json.dumps({k: round(v, 4) for k, v in per.items()}), flush=True)
    print("PERLAYER_JSON " + json.dumps(res), flush=True)


def mode_stack(device, mesh):
    prefix = os.environ.get("PREFIX", SUPER_PATTERN[:9])
    measure("warmup", "ME*", "no_ac", device, mesh)
    for variant in ("no_ac", "old_ac", "new_ac"):
        measure(f"stack:{prefix}", prefix, variant, device, mesh)


def mode_head(device):
    """Real-vocabulary output layer as NeMo RL computes the training loss without chunking.

    BF16 lm_head GEMM, then full FP32 logits, log_softmax and a token gather
    (``_compute_local_logprobs`` with ``logprob_chunk_size=None``).
    """
    vocab, hidden = 131072, 4096
    w = torch.nn.Parameter(torch.randn(vocab, hidden, device=device, dtype=torch.bfloat16) * 0.02)
    ids = torch.randint(0, vocab, (1, SEQ), device=device)
    chunk_size = int(os.environ.get("CHUNK", "2048"))
    for chunk in (None, chunk_size, f"nemo_rl:{chunk_size}"):
        for _ in range(2):
            h = torch.randn(1, SEQ, hidden, device=device, dtype=torch.bfloat16, requires_grad=True)
            w.grad = None
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            logits = h @ w.t()
            if chunk is None:
                lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, ids.unsqueeze(-1))
            elif isinstance(chunk, str):
                # NeMo RL's training loss with logprob_chunk_size set (#4114): a singleton
                # TP group and ChunkedDistributedLogprob, which recomputes each chunk's
                # log_softmax in backward instead of keeping it.
                from nemo_rl.distributed.model_utils import ChunkedDistributedLogprob

                lp = ChunkedDistributedLogprob.apply(logits, ids, 0, vocab, chunk_size, dist.group.WORLD, False)
            else:
                # Plain autograd per chunk (what ``_tp_target_logprobs`` does): bounds the
                # forward peak but keeps every chunk's FP32 log_softmax for backward.
                parts = [
                    torch.log_softmax(logits[:, s : s + chunk].float(), dim=-1).gather(-1, ids[:, s : s + chunk].unsqueeze(-1))
                    for s in range(0, SEQ, chunk)
                ]
                lp = torch.cat(parts, dim=1)
            loss = -lp.mean()
            torch.cuda.synchronize()
            retained = torch.cuda.memory_allocated() - base
            fwd_peak = torch.cuda.max_memory_allocated() - base
            torch.cuda.reset_peak_memory_stats()
            loss.backward()
            torch.cuda.synchronize()
            bwd_peak = torch.cuda.max_memory_allocated() - base
            del logits, lp, loss, h
        rec = dict(
            tag="head", chunk=chunk, retained_gib=retained / GiB, fwd_peak_gib=fwd_peak / GiB, bwd_peak_gib=bwd_peak / GiB
        )
        print("RESULT " + json.dumps(rec), flush=True)
        torch.cuda.empty_cache()


def mode_offload(device, mesh):
    """new_ac with and without NeMo RL's ActivationCpuOffload, 3 AdamW steps each.

    Reports step-3 memory/time (Adam state initialized), per-step losses, and the max
    step-1 grad difference against the first run (a second plain run gives the noise floor).
    """
    from nemo_rl.models.automodel.activation_offload import ActivationCpuOffload

    from contextlib import nullcontext

    prefix = os.environ.get("PREFIX", SUPER_PATTERN[:9])
    vocab_size = 1024
    gens = [torch.Generator(device).manual_seed(10 + i) for i in range(3)]
    batches = [torch.randint(0, vocab_size, (1, SEQ), device=device, generator=g) for g in gens]
    measure("warmup", "ME*", "no_ac", device, mesh)
    ref = None
    for tag, offload in (("new_ac", None), ("new_ac_again", None), ("new_ac_offload", ActivationCpuOffload())):
        model = build([CODE[c] for c in prefix], vocab_size, device)
        model, wrapped = parallelize(model, mesh, "new_ac")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        losses, grads, rec = [], None, None
        for step, ids in enumerate(batches):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            t0 = time.perf_counter()
            with offload if offload is not None else nullcontext():
                out = model(input_ids=ids)
            logits = out.logits if hasattr(out, "logits") else out
            loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), ids.flatten())
            del out, logits
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            retained = torch.cuda.memory_allocated() - base
            fwd_peak = torch.cuda.max_memory_allocated() - base
            torch.cuda.reset_peak_memory_stats()
            loss.backward()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            bwd_peak = torch.cuda.max_memory_allocated() - base
            if step == 0:
                grads = {
                    n.replace("_checkpoint_wrapped_module.", ""): (p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad).float().clone()
                    for n, p in model.named_parameters()
                    if p.grad is not None
                }
            losses.append(loss.item())
            opt.step()
            opt.zero_grad(set_to_none=True)
            rec = dict(
                retained_gib=retained / GiB,
                fwd_peak_gib=fwd_peak / GiB,
                bwd_peak_gib=bwd_peak / GiB,
                fwd_s=t1 - t0,
                bwd_s=t2 - t1,
            )
        host = torch.cuda.host_memory_stats() if hasattr(torch.cuda, "host_memory_stats") else {}
        rec.update(
            tag=tag,
            layers=prefix,
            wrapped=wrapped,
            losses=losses,
            host_pinned_peak_gib=host.get("allocated_bytes.peak", host.get("allocated_bytes.all.peak", 0)) / GiB,
            host_pinned_reserved_gib=host.get("reserved_bytes.current", host.get("reserved_bytes.all.current", 0)) / GiB,
        )
        if ref is None:
            ref = grads
            rec["max_dgrad_vs_new_ac"] = 0.0
        else:
            # NaN-aware: same NaN positions, max |diff| over the finite entries.
            rec["same_nan_mask"] = all(torch.equal(grads[n].isnan(), ref[n].isnan()) for n in ref)
            rec["max_dgrad_vs_new_ac"] = max(
                (grads[n] - ref[n]).nan_to_num(0.0, 0.0, 0.0).abs().max().item() for n in ref
            )
        rec["grads_finite"] = all(torch.isfinite(g).all().item() for g in grads.values())
        print("RESULT " + json.dumps(rec), flush=True)
        del model, opt, grads
        torch.cuda.empty_cache()


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device("cuda")
    mesh = init_device_mesh("cuda", (1, 1, 1), mesh_dim_names=("dp_replicate", "dp_shard_cp", "tp"))
    print(
        f"torch={torch.__version__} seq={SEQ} n_experts={N_EXPERTS} mode={MODE} "
        f"dispatcher={DISPATCHER} experts={EXPERTS} gpu={torch.cuda.get_device_name()}",
        flush=True,
    )
    if MODE == "layers":
        mode_layers(device, mesh)
    elif MODE == "stack":
        mode_stack(device, mesh)
    elif MODE == "head":
        mode_head(device)
    elif MODE == "offload":
        mode_offload(device, mesh)
    print("CHECK DONE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
