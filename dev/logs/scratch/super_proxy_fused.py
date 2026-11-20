"""#4169 step 4: fused linear logprobs on a Nemotron 3.5 Super-dimension model.

Builds a short native Nemotron-V3 stack with the real Super layer dimensions and the real
131072-entry vocabulary (hidden 4096; the only reduced dimension is n_routed_experts), FP32
stored parameters and BF16 compute, parallelized with AutoModel's NemotronHModelParallelizer
(FSDP2 over all ranks, EP=1, activation checkpointing). Then it runs one NeMo RL
training microbatch through automodel_forward_backward + LossPostProcessor twice on the same
model: the logits path (logprob_chunk_size=CHUNK, the #4114 chunked loss) and the fused path
(FusedLogprobLMHead), and reports peak memory, time and the gradient differences.

Run: python -m torch.distributed.run --nproc-per-node N super_proxy_fused.py
Env: SEQ (32768), LAYERS (ME*), N_EXPERTS (32), CHUNK (2048), FUSED_CHUNK (1024), TEMP (0.7)
"""

import json
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

sys.path.insert(0, "/planning-users/yangyeyuan/nemo-rl-4166/scratch")
import super_proxy_memory as sp  # noqa: E402  (SuperConfig, build: Super dimensions)

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams  # noqa: E402
from nemo_rl.algorithms.loss.interfaces import LossInputType  # noqa: E402
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from nemo_rl.models.automodel.data import ProcessedInputs, ProcessedMicrobatch  # noqa: E402
from nemo_rl.models.automodel.fused_logprobs import FusedLogprobLMHead  # noqa: E402
from nemo_rl.models.automodel.train import LossPostProcessor, automodel_forward_backward  # noqa: E402

SEQ = int(os.environ.get("SEQ", "32768"))
LAYERS = os.environ.get("LAYERS", "ME*")
CHUNK = int(os.environ.get("CHUNK", "2048"))
FUSED_CHUNK = int(os.environ.get("FUSED_CHUNK", "1024"))
TEMP = float(os.environ.get("TEMP", "0.7"))
VOCAB = 131072
GiB = 2**30


class WeightedTokenLoss:
    input_type = LossInputType.LOGPROB

    def __init__(self, fused):
        self.use_fused_linear_logprobs = fused

    def __call__(self, *, data, next_token_logprobs, global_valid_toks, **kwargs):
        loss = (next_token_logprobs * data["weights"] * data["token_mask"][:, 1:]).sum() / global_valid_toks
        return loss, {"loss": loss.detach()}


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")

    model = sp.build([sp.CODE[c] for c in LAYERS], VOCAB, device)
    mesh = init_device_mesh("cuda", (1, world, 1), mesh_dim_names=("dp_replicate", "dp_shard_cp", "tp"))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.float32)
    from nemo_automodel.components.distributed.parallelizer import NemotronHParallelizationStrategy

    model = NemotronHParallelizationStrategy().parallelize(model, mesh, mp_policy=mp, activation_checkpointing=True)
    head = FusedLogprobLMHead(model, chunk_size=FUSED_CHUNK)
    lm_head = model.get_output_embeddings()
    if rank == 0:
        print(f"[setup] world={world} layers={LAYERS} seq={SEQ} vocab={VOCAB} lm_head own FSDP unit={isinstance(lm_head, FSDPModule)} "
              f"lm_head weight dtype={lm_head.weight.dtype} params={sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)

    gen = torch.Generator(device).manual_seed(100 + rank)
    input_ids = torch.randint(0, VOCAB, (1, SEQ), device=device, generator=gen)
    token_mask = torch.ones(1, SEQ, device=device)
    token_mask[:, : SEQ // 8] = 0  # prompt tokens
    data = BatchedDataDict({
        "input_ids": input_ids,
        "input_lengths": torch.tensor([SEQ], device=device),
        "token_mask": token_mask,
        "sample_mask": torch.ones(1, device=device),
        "weights": torch.randn(1, SEQ - 1, device=device, generator=gen),
    })
    mb = ProcessedMicrobatch(data_dict=data, processed_inputs=ProcessedInputs(input_ids=input_ids, seq_len=SEQ),
                             original_batch_size=1, original_seq_len=SEQ)
    sampling = TrainingSamplingParams(top_k=None, top_p=1.0, temperature=TEMP)
    valid_toks = token_mask[:, 1:].sum()
    dist.all_reduce(valid_toks)

    def run(fused):
        processor = LossPostProcessor(
            loss_fn=WeightedTokenLoss(fused), cfg={"logprob_chunk_size": CHUNK}, cp_mesh=None, cp_size=1,
            dp_size=world, sampling_params=sampling, tp_mesh=mesh["tp"],
        )
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        [(loss, _)] = automodel_forward_backward(
            model=model, data_iterator=iter([mb]), post_processing_fn=processor, device_mesh=None,
            padding_token_id=0, autocast_context_factory=nullcontext, allow_flash_attn_args=False,
            global_valid_seqs=torch.tensor(world, device=device), global_valid_toks=valid_toks,
            sampling_params=sampling, dp_size=world, fused_logprob_head=head if fused else None,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak = (torch.cuda.max_memory_allocated() - base) / GiB
        grads = {n: (p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad).float().cpu()
                 for n, p in model.named_parameters() if p.grad is not None}
        dist.all_reduce(loss)
        return loss.item(), peak, elapsed, grads

    results = {}
    for fused in (False, True) * 3:  # keep the fastest of three runs; grads from the last
        loss, peak, elapsed, grads = run(fused)
        best = results.get(fused, (None, None, float("inf"), None))[2]
        results[fused] = (loss, peak, min(best, elapsed), grads)
    (lu, pu, tu, gu), (lf, pf, tf, gf) = results[False], results[True]
    names = sorted(gu)
    # Local shards: sum the squared norms across ranks to get global norms.
    stats = torch.tensor(
        [[(gf[n] - gu[n]).pow(2).sum().item(), gu[n].pow(2).sum().item(), gf[n].pow(2).sum().item(),
          float(not torch.equal(gf[n], gu[n])), float((~torch.isfinite(gu[n])).sum()),
          float((~torch.isfinite(gf[n])).sum())] for n in names],
        device=device, dtype=torch.float64,
    )
    dist.all_reduce(stats)
    nonfinite = {n: (int(stats[i, 4]), int(stats[i, 5])) for i, n in enumerate(names) if stats[i, 4] or stats[i, 5]}
    finite_rows = (stats[:, 4] == 0) & (stats[:, 5] == 0)
    rel = (stats[:, 0] / stats[:, 1].clamp_min(1e-60)).sqrt()
    rel[~finite_rows] = 0
    worst_idx = int(rel.argmax())
    exact = int((stats[:, 3] == 0).sum())
    lm_idx = [i for i, n in enumerate(names) if "lm_head" in n][0]
    peaks = torch.tensor([pu, pf], device=device)
    dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
    if rank == 0:
        print("RESULT " + json.dumps(dict(
            world=world, seq=SEQ, layers=LAYERS, chunk=CHUNK, fused_chunk=FUSED_CHUNK, temperature=TEMP,
            loss_unfused=lu, loss_fused=lf, peak_gib_unfused=peaks[0].item(), peak_gib_fused=peaks[1].item(),
            time_s_unfused=tu, time_s_fused=tf, grads_bitwise_equal=f"{exact}/{len(names)}",
            worst_grad_rel_norm_diff=rel[worst_idx].item(), worst_grad=names[worst_idx],
            lm_head_grad_rel_norm_diff=rel[lm_idx].item(),
            lm_head_grad_norm_unfused=stats[lm_idx, 1].sqrt().item(), lm_head_grad_norm_fused=stats[lm_idx, 2].sqrt().item(),
            nonfinite_grad_elements_unfused_fused=nonfinite,
        )), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
