"""Smoke test for #4166 on one GPU: tiny random-init Nemotron-H (mamba/moe/attention/mlp).

1. Forward + backward without parallelization (mamba_ssm / causal_conv1d kernels run).
2. Apply AutoModel's Nemotron-H FSDP2 strategy (the EP=1 path) with activation
   checkpointing on a 1-GPU mesh and report which decoder blocks got checkpoint-wrapped.

Run: <automodel venv>/bin/python -m torch.distributed.run --nproc-per-node 1 smoke_nemotron_h.py
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.distributed.device_mesh import init_device_mesh


class TinyNemotronHConfig:
    """Same fields as Automodel's MockNemotronV3Config, with all four block types."""

    def __init__(self, **overrides):
        self.num_attention_heads = 8
        self.num_key_value_heads = 4
        self.head_dim = 64
        self.hidden_size = 256
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.intermediate_size = 512
        self.mlp_bias = False
        self.mlp_hidden_act = "relu2"
        self.mamba_num_heads = 8
        self.mamba_head_dim = 32
        self.ssm_state_size = 16
        self.n_groups = 1
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
        self.layers_block_type = ["mamba", "moe", "attention", "mlp"]
        self.num_hidden_layers = len(self.layers_block_type)
        self.vocab_size = 1024
        self.torch_dtype = "bfloat16"
        self.initializer_range = 0.02
        self.rescale_prenorm_residual = True
        self.residual_in_fp32 = False
        self.n_routed_experts = 4
        self.num_experts_per_tok = 2
        self.n_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 1.0
        self.moe_intermediate_size = 128
        self.norm_topk_prob = False
        self.moe_shared_expert_intermediate_size = 128
        self.num_nextn_predict_layers = 0
        self.tie_word_embeddings = False
        for k, v in overrides.items():
            setattr(self, k, v)

    def to_dict(self):
        return vars(self)


def build(device):
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )
    torch.manual_seed(0)
    with torch.device(device):
        model = NemotronHForCausalLM(TinyNemotronHConfig(), backend=backend)
    # The constructor leaves some parameters uninitialized (AutoModel builds on meta and
    # then initializes or loads weights), so initialize explicitly.
    model.initialize_weights(buffer_device=device, dtype=torch.bfloat16)
    return model.to(torch.bfloat16)


def step(model, ids):
    torch.cuda.reset_peak_memory_stats()
    out = model(input_ids=ids)
    logits = out.logits if hasattr(out, "logits") else out
    loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), ids.flatten())
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    finite = all(g is not None and torch.isfinite(g.to_local() if hasattr(g, "to_local") else g).all() for g in grads)
    return loss.item(), finite, torch.cuda.max_memory_allocated() / 2**20


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device("cuda")
    ids = torch.randint(0, 1024, (2, 512), device=device, generator=torch.Generator(device).manual_seed(1))

    model = build(device)
    from nemo_automodel.components.distributed.parallelizer import _nemotronh_decoder_blocks

    _, blocks = _nemotronh_decoder_blocks(model)
    print("block types:", [b.block_type for b in blocks])
    loss, finite, peak = step(model, ids)
    print(f"[plain] loss={loss:.4f} grads_finite={finite} peak={peak:.0f} MiB")

    from torch.distributed.fsdp import MixedPrecisionPolicy

    from nemo_automodel.components.distributed.parallelizer import NemotronHParallelizationStrategy

    model = build(device)
    mesh = init_device_mesh("cuda", (1, 1, 1), mesh_dim_names=("dp_replicate", "dp_shard_cp", "tp"))
    # Same policy NeMo RL builds in nemo_rl/models/automodel/setup.py.
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    model = NemotronHParallelizationStrategy().parallelize(
        model, mesh, mp_policy=mp_policy, activation_checkpointing=True
    )
    _, blocks = _nemotronh_decoder_blocks(model)
    for i, b in enumerate(blocks):
        wrapped = isinstance(b, CheckpointWrapper)
        inner = b._checkpoint_wrapped_module if wrapped else b
        print(f"  layer {i}: {inner.block_type:9s} checkpointed={wrapped}")
    loss, finite, peak = step(model, ids)
    print(f"[fsdp2+AC] loss={loss:.4f} grads_finite={finite} peak={peak:.0f} MiB")
    print("SMOKE OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
