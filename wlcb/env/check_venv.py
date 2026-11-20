"""Smoke-check a NeMo RL worker venv: imports and one GPU op per key package.

Usage: <venv>/bin/python env/check_venv.py mcore|vllm|automodel
"""

import importlib
import sys

import torch

kind = sys.argv[1]
print("torch", torch.__version__, "cuda", torch.version.cuda, "nccl", torch.cuda.nccl.version())
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
print("matmul ok", (x @ x).float().abs().mean().item() > 0)

mods = {
    "mcore": [
        "megatron.core",
        "megatron.bridge",
        "transformer_engine.pytorch",
        "flash_attn",
        "mamba_ssm",
        "causal_conv1d",
        "deep_ep",
        "flash_mla",
        "fast_hadamard_transform",
        "nvidia_resiliency_ext",
        "torch_memory_saver",
        "nemo_gym",
    ],
    "vllm": ["vllm", "flashinfer", "nemo_gym"],
    "automodel": [
        "nemo_rl",
        "nemo_automodel",
        "transformers",
        "transformer_engine.pytorch",
        "flash_attn",
        "mamba_ssm",
        "causal_conv1d",
        "deep_ep",
        "grouped_gemm",
    ],
}[kind]
failed = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print("import ok", m, getattr(mod, "__version__", ""))
    except Exception as e:  # report every failure, not just the first
        failed.append(m)
        print("IMPORT FAILED", m, type(e).__name__, e)

if kind in ("mcore", "automodel") and "transformer_engine.pytorch" not in failed:
    import transformer_engine.pytorch as te

    print("TE cuDNN", te.get_cudnn_version())
    layer = te.LayerNormLinear(1024, 1024, params_dtype=torch.bfloat16).cuda()
    inp = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    layer(inp).sum().backward()
    print("TE LayerNormLinear fwd/bwd ok", torch.isfinite(inp.grad).all().item())
if kind in ("mcore", "automodel") and "flash_attn" not in failed:
    from flash_attn import flash_attn_func

    q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
    print("flash_attn ok", flash_attn_func(q, q, q, causal=True).shape)

print("CHECK", "FAILED " + ",".join(failed) if failed else "OK")
