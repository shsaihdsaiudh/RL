# What does this PR do ?

Adds `policy.automodel_cfg.activation_offloading`, which keeps the activations saved during automodel (DTensor/FSDP2) training forward passes in pinned host memory instead of on the GPU.

With `activation_checkpointing: true`, training still keeps the input of every checkpointed block on the GPU from forward until backward. For deep models with long microbatches that is a large share of the memory: the Nemotron 3.5 Super run in #4166 (32K-token microbatches) would keep an estimated 22.5 GiB of bf16 block inputs per rank. The automodel path had no way to move them, while the Megatron path already has `fine_grained_activation_offloading`.

# Issues

Part of #4166. The out-of-memory error in #4166 mostly comes from activation checkpointing covering only the mamba and mlp blocks when `expert_parallel_size == 1`; that part is fixed in Automodel by NVIDIA-NeMo/Automodel#4834. This PR adds the offload on top of it.

# Changelog

- `nemo_rl/models/automodel/activation_offload.py` (new): `ActivationCpuOffload(model)`, a `torch.autograd.graph.saved_tensors_hooks` context. Pack copies plain CUDA tensors of at least 1 MiB to pinned host memory with `non_blocking=True` on the current stream and records that stream on the source, so a source allocated on another stream is not reused before its copy runs; unpack copies them back. Tensor subclasses such as DTensor, and any tensor that shares storage with a parameter of `model` (FSDP's unsharded parameters, `detach()`/`.data` aliases, local tensors of DTensor parameters), stay on the GPU. A tensor that several operations save in one forward is copied once. `empty_host_cache()` returns the cached pinned memory to the system.
- `nemo_rl/models/automodel/train.py`: `forward_with_post_processing_fn` wraps only `model_forward` in the context, so logits and the loss stay on the GPU. `automodel_forward_backward` passes the context only when `forward_only` is false, so logprob, reference and evaluation passes are unaffected.
- `nemo_rl/models/automodel/setup.py`: `validate_and_prepare_config` raises `ValueError` when `activation_offloading` is on and `activation_checkpointing` is off.
- `nemo_rl/models/policy/__init__.py`: `activation_offloading: NotRequired[bool]` next to `activation_checkpointing`.
- Policy and value automodel workers create one `ActivationCpuOffload(self.model)` and reuse it for every microbatch. At the end of each training step they call `empty_host_cache()`: the pinned-memory allocator rounds every block up to a power of two and keeps freed blocks, which would otherwise stay pinned through generation. The next step pins the memory again; on the H20 dev host pinning 4 GiB in 256 MiB blocks took 1.35 s.
- Base example configs (`distillation_math`, `dpo`, `grpo_math_1B`, `ppo_math_1B`, `rm`, `sft`, `vlm_mpo`) and the matching reference configs get `activation_offloading: false`.
- `docs/design-docs/training-backends.md`: new "Activation CPU offload" section under the Automodel backend (usage, key table, host memory sizing, transfer cost).
- `pyrefly.toml`: type-check the new module.

# Usage

```yaml
policy:
  automodel_cfg:
    activation_checkpointing: true
    activation_offloading: true
```

# Data

All runs use torch 2.13 and the automodel worker venv. The Nemotron-H runs use a local Automodel submodule patch equivalent to NVIDIA-NeMo/Automodel#4834, so every block is checkpointed at EP=1.

**Gradient parity, 8×H20e, FSDP8.** Random-init native Nemotron-H with 10 layers `MEM*EMEM*E` (hidden 1024, 64 experts top-6), 8K tokens per rank. One model instance and one batch, running forward and backward five times, alternating plain and offloaded, with no optimizer step:

| Config | Result |
|---|---|
| EP=1 (torch experts) | step-1 gradients bitwise identical on all 8 ranks in all 5 runs |
| EP=8 (DeepEP, `gmm` experts) | step-1 gradients bitwise identical on all 8 ranks in all 5 runs |

Comparing separately built model instances shows run-to-run gradient differences up to about 0.3 relative L2 even without offload, so parity is checked on one instance. On one GPU with `torch.use_deterministic_algorithms`, 10-layer all-MoE, all-Mamba and all-attention models trained for 3 AdamW steps give bitwise identical losses and gradients with and without offload.

**Memory, same model, EP=1, 16K tokens per rank, 8×H20e.** With offload, memory retained after forward goes from 2.56 to 2.10 GiB and the step peak from 6.47 to 6.00 GiB; host pinned memory peaks at 0.47 GiB. The host holds 10 bf16 block inputs and 3 tensors around the head. The block inputs are bf16 for both `fsdp_output_dtype: float32` and `bfloat16`, because FSDP casts each block's input to the parameter dtype before the checkpointed block saves it.

**SFT, Qwen2.5-1.5B, 8×H20e.** Run with `examples/run_sft.py`, activation checkpointing on, sequence packing with 32K tokens per microbatch, 20 steps. Offload off and then on, on the same node:

| | offload off | offload on |
|---|---|---|
| Peak GPU memory, ranks 1–7 (`nvidia-smi`) | 38,707 MiB | 36,115 MiB |
| Median step time (steps 3–20) | 11.04 s | 10.98 s |
| Loss | step 1: 0.7572 | step 1: 0.7572; largest per-step difference over 20 steps: 0.0024 |

The 2.5 GiB reduction matches the size of the block inputs, 28 × 32768 × 1536 × 2 bytes ≈ 2.6 GiB.

The 8-GPU runs above used the first commit. After the second commit (parameter-storage check, `record_stream`, `empty_host_cache`), the 1-GPU same-instance check on the same Nemotron-H model is still bitwise identical with the same 13 host copies, and a 1-GPU Qwen2.5-0.5B SFT smoke run with offload on trains normally (losses 1.6316 / 0.8432 / 0.4961).

# Before your PR is "Ready for review"
**Pre checks**:
- [x] Make sure you read and followed [Contributor guidelines](/NVIDIA-NeMo/RL/blob/main/CONTRIBUTING.md)
- [x] Did you write any new necessary tests?
- [ ] Did you run the unit tests and functional tests locally? Visit our [Testing Guide](/NVIDIA-NeMo/RL/blob/main/docs/testing.md) for how to run tests
- [x] Did you add or update any necessary documentation? Visit our [Document Development Guide](/NVIDIA-NeMo/RL/blob/main/docs/documentation.md) for how to write, build and test the docs.

New tests:
- `tests/unit/models/automodel/test_activation_offload.py` (GPU): training parity and freed memory on a checkpointed model; parameters, their views, `detach()`/`.data` aliases and small tensors stay on the GPU; a parameter registered after construction (as FSDP does) and the local tensor of a DTensor parameter stay on the GPU; the pinned host round trip; a source allocated on another stream is not reused while its copy is queued (fails without `record_stream`); a tensor saved twice is copied once; an in-place write after a save gets a new copy; `empty_host_cache()` releases the pinned copies.
- `test_automodel_setup.py::test_activation_offloading_requires_activation_checkpointing`
- `test_automodel_train.py::test_activation_offload_wraps_only_the_model_forward`
- `test_automodel_train.py::test_forward_backward_offloads_activations_only_when_training`

Local results: the 9 GPU tests above pass, 187 tests pass in `test_automodel_setup.py` and `test_automodel_train.py`, and 458 pass in `test_config_v2.py` and `test_config_validation.py`. `ruff format`, `ruff check` and `pyrefly` report nothing on the changed files. I did not run the functional tests; the SFT runs above are the end-to-end check.

# Additional Information

- Not validated on the full Nemotron 3.5 Super configuration from #4166.
- The copies run on the compute stream, so each microbatch pays one device-to-host and one host-to-device transfer. In the runs above the step time did not change measurably, but it depends on the model, microbatch size and host bandwidth.
