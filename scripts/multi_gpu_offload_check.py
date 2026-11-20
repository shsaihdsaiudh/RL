"""#4166: activation offload on a multi-GPU FSDP2 Nemotron-H (mamba/moe/attention), EP=1 and EP>1.

Builds a random-init native Nemotron-V3 through NeMoAutoModelForCausalLM.from_config with the same
DistributedSetup NeMo RL builds (FSDP2Config with bf16 params / fp32 reduce+output, meshes from
MeshContext.build, MoE parallelizer config only when EP>1), so EP=1 goes through the Nemotron-H
strategy (the patched full-AC path) and EP>1 through the MoE parallelizer with DeepEP.

For each EP size, three runs of STEPS AdamW steps on per-rank data:
  ac           activation checkpointing
  ac_again     the same again (nondeterminism floor)
  ac_offload   activation checkpointing + nemo_rl ActivationCpuOffload around the forward
Reports loss per step (DP mean), max |step-1 grad - ac| over local shards (max over ranks),
step-STEPS retained-after-forward / peak memory and fwd/bwd time (max over ranks), host pinned peak.

Run: torchrun --nproc-per-node 8 multi_gpu_offload_check.py   (env: SEQ, EP_SIZES, STEPS)
"""

import json
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config, MoEParallelizerConfig
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.components.models.common import BackendConfig
from torch.distributed.fsdp import MixedPrecisionPolicy
from transformers import NemotronHConfig

from nemo_rl.models.automodel.activation_offload import ActivationCpuOffload

SEQ = int(os.environ.get("SEQ", "8192"))
STEPS = int(os.environ.get("STEPS", "3"))
EP_SIZES = [int(x) for x in os.environ.get("EP_SIZES", "1,8").split(",")]
PATTERN = os.environ.get("PATTERN", "MEM*EMEM*E")
CODE = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}
GiB = 2**30


def hf_config():
    return NemotronHConfig(
        architectures=["NemotronHForCausalLM"],
        vocab_size=32000,
        hidden_size=1024,
        layers_block_type=[CODE[c] for c in PATTERN],
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=2048,
        mlp_hidden_act="relu2",
        ssm_state_size=64,
        mamba_num_heads=16,
        mamba_head_dim=64,
        n_groups=2,
        chunk_size=128,
        n_routed_experts=64,
        n_shared_experts=1,
        moe_intermediate_size=512,
        moe_shared_expert_intermediate_size=1024,
        num_experts_per_tok=6,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        num_nextn_predict_layers=0,
        tie_word_embeddings=False,
    )


def build(ep_size):
    world = dist.get_world_size()
    fsdp2 = FSDP2Config(
        sequence_parallel=False,
        tp_plan=None,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
            output_dtype=getattr(torch, os.environ.get("OUTPUT_DTYPE", "float32")),
        ),
        offload_policy=None,
        activation_checkpointing=True,
        defer_fsdp_grad_sync=True,
    )
    mesh = MeshContext.build(
        fsdp2,
        ParallelismSizes(dp_replicate_size=1, tp_size=1, pp_size=1, cp_size=1, ep_size=ep_size),
        world_size=world,
    )
    setup = DistributedSetup(
        mesh_context=mesh,
        strategy_config=fsdp2,
        pipeline_config=None,
        moe_parallel_config=MoEParallelizerConfig() if ep_size > 1 else None,
        activation_checkpointing=True,
    )
    backend = BackendConfig(
        attn=os.environ.get("ATTN", "te"),
        linear=os.environ.get("LINEAR", "te"),
        rms_norm="torch_fp32",
        experts="gmm" if ep_size > 1 else "torch",
        dispatcher="deepep" if ep_size > 1 else "torch",
        fake_balanced_gate=False,
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )
    torch.manual_seed(0)
    model = NeMoAutoModelForCausalLM.from_config(
        hf_config(), distributed_setup=setup, torch_dtype=torch.float32, backend=backend
    )
    if os.environ.get("PRINT_AC") and dist.get_rank() == 0:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

        wrapped = [n for n, m in model.named_modules() if isinstance(m, CheckpointWrapper)]
        fsdp = [f"{n}:{type(m).__name__}" for n, m in model.named_modules() if "FSDP" in type(m).__name__]
        print("AC_WRAPPED", wrapped, flush=True)
        print("FSDP_MODULES", fsdp[:20], flush=True)
    return model


def local(t):
    return t.to_local() if hasattr(t, "to_local") else t


def run(ep_size, tag, use_offload, ref):
    model = build(ep_size)
    offload = ActivationCpuOffload(model) if use_offload else None
    experts = sorted({type(m.experts).__name__ for m in model.modules() if hasattr(m, "experts")})
    opt = (
        torch.optim.SGD(model.parameters(), lr=1e-2)
        if os.environ.get("OPTIM") == "sgd"
        else torch.optim.AdamW(model.parameters(), lr=1e-4)
    )
    rank = dist.get_rank()
    losses, grads, rec = [], None, {}
    for step in range(STEPS):
        g = torch.Generator("cuda").manual_seed(1000 * step + rank)
        ids = torch.randint(0, 32000, (1, SEQ), device="cuda", generator=g)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        with offload if offload is not None else nullcontext():
            out = model(input_ids=ids)
        logits = out.logits if hasattr(out, "logits") else out
        loss = torch.nn.functional.cross_entropy(local(logits).float().flatten(0, 1), ids.flatten())
        del out, logits
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        retained = torch.cuda.memory_allocated() - base
        fwd_peak = torch.cuda.max_memory_allocated() - base
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        peak = torch.cuda.max_memory_allocated() - base
        if step == 0:
            grads = {n: local(p.grad).float().clone() for n, p in model.named_parameters() if p.grad is not None}
        l = loss.detach().clone()
        dist.all_reduce(l)
        losses.append(round(l.item() / dist.get_world_size(), 6))
        opt.step()
        opt.zero_grad(set_to_none=True)
        stats = torch.tensor([retained / GiB, fwd_peak / GiB, peak / GiB, t1 - t0, t2 - t1], device="cuda")
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        rec = dict(zip(["retained_gib", "fwd_peak_gib", "peak_gib", "fwd_s", "bwd_s"], [round(x, 4) for x in stats.tolist()]))
    finite = torch.tensor(float(all(torch.isfinite(v).all().item() for v in grads.values())), device="cuda")
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if ref is None:
        dgrad = 0.0
    else:
        diffs = sorted(((grads[n] - ref[n]).abs().max().item(), n) for n in ref)
        if dist.get_rank() == 0 and os.environ.get("DEBUG"):
            print("TOPDIFF", tag, [(f"{d:.2e}", n) for d, n in diffs[-8:]], flush=True)
        d = torch.tensor(diffs[-1][0], device="cuda")
        dist.all_reduce(d, op=dist.ReduceOp.MAX)
        dgrad = d.item()
        num = torch.tensor(sum(((grads[n] - ref[n]) ** 2).sum().item() for n in ref), device="cuda")
        den = torch.tensor(sum((ref[n] ** 2).sum().item() for n in ref), device="cuda")
        dist.all_reduce(num)
        dist.all_reduce(den)
        rec["rel_l2_dgrad"] = (num / den).sqrt().item()
    host = torch.cuda.host_memory_stats()
    hp = torch.tensor(host.get("allocated_bytes.peak", host.get("allocated_bytes.all.peak", 0)) / GiB, device="cuda")
    dist.all_reduce(hp, op=dist.ReduceOp.MAX)
    rec.update(
        ep=ep_size, tag=tag, experts=experts, losses=losses, grads_finite=bool(finite.item()),
        max_dgrad_vs_ac=dgrad, host_pinned_peak_gib=round(hp.item(), 3),
    )
    if rank == 0:
        print("RESULT " + json.dumps(rec), flush=True)
    del model, opt
    torch.cuda.empty_cache()
    return grads


def build_check(ep_size):
    """Build the model twice with the same seed and list the tensors that differ."""
    states = []
    for _ in range(2):
        model = build(ep_size)
        state = {f"param:{n}": local(p.detach()).float().clone() for n, p in model.named_parameters()}
        state.update({f"buffer:{n}": local(b).float().clone() for n, b in model.named_buffers() if b is not None})
        states.append(state)
        del model
        torch.cuda.empty_cache()
    a, b = states
    differ = [n for n in a if not torch.equal(a[n].nan_to_num(), b[n].nan_to_num())]
    if dist.get_rank() == 0:
        print(f"BUILD ep={ep_size} tensors={len(a)} differ={len(differ)} {differ[:30]}", flush=True)


def same_model_check(ep_size):
    """Grads of one model and one batch, alternating plain and offloaded backward, no optimizer step."""
    model = build(ep_size)
    if os.environ.get("NAN_TRACE"):
        # Report, in execution order, the first modules whose forward output has a NaN.
        hits = []

        def hook(module, inputs, output, name=None):
            outs = output if isinstance(output, (tuple, list)) else (output,)
            ins = [t for t in inputs if isinstance(t, torch.Tensor)]
            bad_out = any(isinstance(t, torch.Tensor) and t.is_floating_point() and torch.isnan(t).any().item() for t in outs)
            bad_in = any(t.is_floating_point() and torch.isnan(t).any().item() for t in ins)
            if bad_out and len(hits) < 12:
                hits.append(f"{name}({type(module).__name__}) nan_in={bad_in}")

        for n, m in model.named_modules():
            m.register_forward_hook(lambda mod, i, o, n=n: hook(mod, i, o, n))
        with torch.no_grad():
            model(input_ids=torch.randint(0, 32000, (1, SEQ), device="cuda"))
        if dist.get_rank() == 0:
            print("NAN_TRACE", hits, flush=True)
        return
    g = torch.Generator("cuda").manual_seed(dist.get_rank())
    ids = torch.randint(0, 32000, (1, SEQ), device="cuda", generator=g)
    ref = None
    for tag in os.environ.get("SAME_ORDER", "plain,plain_again,offload,plain_again2,offload_again").split(","):
        recorded = globals().get("RECORDED")
        if recorded is not None:
            recorded.clear()
        with ActivationCpuOffload(model) if tag.startswith("offload") else nullcontext():
            out = model(input_ids=ids)
        if recorded and dist.get_rank() == 0 and tag == "offload":
            from collections import Counter

            nbytes = sum(torch.Size(s).numel() * getattr(torch, d).itemsize for s, d, _ in recorded)
            print(f"COPIES n={len(recorded)} host_gib={nbytes / GiB:.4f} kinds={dict(Counter(recorded))}", flush=True)
        logits = out.logits if hasattr(out, "logits") else out
        loss = torch.nn.functional.cross_entropy(local(logits).float().flatten(0, 1), ids.flatten())
        del out, logits
        loss.backward()
        grads = {n: local(p.grad).float().clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        if ref is None:
            ref = grads
        stats = torch.tensor(
            [
                float(sum(((grads[n] - ref[n]) ** 2).sum() for n in ref)),
                float(sum((ref[n] ** 2).sum() for n in ref)),
                float(not all(torch.equal(grads[n], ref[n]) for n in ref)),
            ],
            device="cuda",
            dtype=torch.float64,
        )
        dist.all_reduce(stats)
        rel = (stats[0] / stats[1]).sqrt()
        bitwise = stats[2].item() == 0
        if dist.get_rank() == 0:
            print(f"SAME ep={ep_size} {tag:<14} loss={loss.item():.6f} rel_l2_vs_plain={rel.item():.2e} bitwise_all_ranks={bitwise}", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    mode = os.environ.get("OFFLOAD_MODE", "async")
    if os.environ.get("FORCE_FSDP"):
        # 1-GPU runs: FSDP2Manager.parallelize skips the strategy when world/mesh size is 1.
        # Run the multi-GPU strategy anyway so 1 GPU exercises the same wrapping and FSDP mp policy.
        import nemo_automodel.components.distributed.fsdp2 as F

        def forced(self, model):
            F.fsdp2_strategy_parallelize(
                model,
                device_mesh=self.device_mesh,
                mp_policy=self.mp_policy,
                tp_shard_plan=self.tp_plan,
                offload_policy=self.offload_policy,
                sequence_parallel=bool(self.sequence_parallel),
                activation_checkpointing=self.activation_checkpointing,
                enable_async_tensor_parallel=self.enable_async_tensor_parallel,
                enable_compile=self.enable_compile,
                enable_fsdp2_prefetch=self.enable_fsdp2_prefetch,
                fsdp2_backward_prefetch_depth=self.fsdp2_backward_prefetch_depth,
                fsdp2_forward_prefetch_depth=self.fsdp2_forward_prefetch_depth,
                reshard_after_forward=self.reshard_after_forward,
                activation_checkpointing_scope=self.activation_checkpointing_scope,
                frozen_multimodal_sharding=self.frozen_multimodal_sharding,
            )
            return model

        F.FSDP2Manager.parallelize = forced
    if os.environ.get("RECORD"):
        # Real ActivationCpuOffload (with its dedup); record every host copy it makes.
        import nemo_rl.models.automodel.activation_offload as ao

        real_pack = ao._copy_to_host
        RECORDED = []

        def recording_pack(tensor):
            packed = real_pack(tensor)
            if isinstance(packed, tuple):
                RECORDED.append((tuple(tensor.shape), str(tensor.dtype).replace("torch.", ""), type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else None))
            return packed

        ao._copy_to_host = recording_pack
        globals()["RECORDED"] = RECORDED
    if mode == "clone":
        import nemo_rl.models.automodel.activation_offload as ao

        clo, chi = (int(x) for x in os.environ.get("OFFLOAD_IDX", "0-100000").split("-"))
        ccount = [0]
        CLONED = []

        def cpack(tensor):
            if not ao._is_offloadable(tensor):
                return tensor
            i = ccount[0]
            ccount[0] += 1
            if not (clo <= i < chi):
                return tensor
            CLONED.append((i, tuple(tensor.shape), str(tensor.dtype), type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else None, tensor.data_ptr()))
            return ("c", tensor.clone())

        def cunpack(packed):
            return packed if isinstance(packed, torch.Tensor) else packed[1]

        class CloneHooks(torch.autograd.graph.saved_tensors_hooks):
            def __init__(self):
                super().__init__(cpack, cunpack)

            def __enter__(self):
                ccount[0] = 0
                CLONED.clear()
                return super().__enter__()

        globals()["ActivationCpuOffload"] = CloneHooks
        globals()["G_PACKED"] = CLONED
    if mode == "verify":
        import nemo_rl.models.automodel.activation_offload as ao

        seen = []

        def vpack(tensor):
            if not ao._is_offloadable(tensor):
                return tensor
            host = tensor.detach().cpu().clone()
            tag = (len(seen), tuple(tensor.shape), str(tensor.dtype), type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else None, tensor._version)
            seen.append(tag)
            return ("v", tensor.device, host, tensor, tag)

        def vunpack(packed):
            if isinstance(packed, torch.Tensor):
                return packed
            _, device, host, ref, tag = packed
            same = torch.equal(host, ref.detach().cpu())
            if not same and dist.get_rank() == 0:
                print("MUTATED_AFTER_SAVE", tag, "version_now", ref._version, flush=True)
            return host.to(device)

        class VerifyOffload(torch.autograd.graph.saved_tensors_hooks):
            def __init__(self):
                super().__init__(vpack, vunpack)

        globals()["ActivationCpuOffload"] = VerifyOffload
    if os.environ.get("DEBUG"):
        import nemo_rl.models.automodel.activation_offload as ao

        orig_pack = ao._pack
        G_PACKED = []
        lo, hi = (int(x) for x in os.environ.get("OFFLOAD_IDX", "0-100000").split("-"))
        counter = [0]

        def logging_pack(tensor):
            if not ao._is_offloadable(tensor):
                return tensor
            i = counter[0]
            counter[0] += 1
            if not (lo <= i < hi):
                return tensor
            out = orig_pack(tensor)
            if isinstance(out, tuple) and len(G_PACKED) < 200:
                G_PACKED.append((i, tuple(tensor.shape), str(tensor.dtype), tensor.requires_grad, type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else None))
            return out

        class DebugOffload(torch.autograd.graph.saved_tensors_hooks):
            def __init__(self):
                super().__init__(logging_pack, ao._unpack)

            def __enter__(self):
                counter[0] = 0
                return super().__enter__()

        globals()["ActivationCpuOffload"] = DebugOffload
        globals()["G_PACKED"] = G_PACKED
    if mode in ("sync", "keepref"):
        import nemo_rl.models.automodel.activation_offload as ao

        def pack(tensor):
            if not ao._is_offloadable(tensor):
                return tensor
            host = torch.empty_like(tensor, device="cpu", pin_memory=True)
            host.copy_(tensor, non_blocking=(mode != "sync"))
            if mode == "sync":
                torch.cuda.synchronize()
            return (tensor.device, host, tensor) if mode == "keepref" else (tensor.device, host)

        def unpack(packed):
            if isinstance(packed, torch.Tensor):
                return packed
            return packed[1].to(packed[0], non_blocking=True)

        ao._pack, ao._unpack = pack, unpack
        ActivationCpuOffload.__init__ = lambda self: torch.autograd.graph.saved_tensors_hooks.__init__(self, pack, unpack)
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.enable_cudnn_sdp(False)
    if os.environ.get("DETERMINISTIC"):
        # Needs CUBLAS_WORKSPACE_CONFIG=:4096:8. warn_only: ops without a deterministic kernel still run.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        # Deterministic mode fills torch.empty with NaN; FILL_UNINIT=0 turns that off.
        torch.utils.deterministic.fill_uninitialized_memory = os.environ.get("FILL_UNINIT", "1") == "1"
    if dist.get_rank() == 0:
        print(f"torch={torch.__version__} world={dist.get_world_size()} seq/rank={SEQ} pattern={PATTERN} eps={EP_SIZES}", flush=True)
    for ep in EP_SIZES:
        if os.environ.get("BUILD_CHECK"):
            build_check(ep)
            continue
        if os.environ.get("SAME_MODEL"):
            same_model_check(ep)
            continue
        run(ep, "warmup", False, None)
        ref = run(ep, "ac", False, None)
        run(ep, "ac_again", False, ref)
        for i in range(int(os.environ.get("EXTRA_AGAIN", "0"))):
            run(ep, f"ac_again{i + 2}", False, ref)
        run(ep, "ac_offload", True, ref)
        if (os.environ.get("DEBUG") or mode == "clone") and dist.get_rank() == 0:
            packed = globals()["G_PACKED"]
            print("PACKED", len(packed), packed[:40], flush=True)
    if dist.get_rank() == 0:
        print("CHECK DONE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
