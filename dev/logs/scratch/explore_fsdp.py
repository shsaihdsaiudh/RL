import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from nemo_rl.models.automodel.fused_logprobs import FusedLogprobLMHead
from tests.unit.models.automodel import test_fused_logprobs as t

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

for layout in ("lm_head_own_unit", "lm_head_in_root", "tied"):
    model = t._tiny_llama(tie_word_embeddings=layout == "tied")
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.float32)
    for layer in model.model.layers:
        fully_shard(layer, mp_policy=mp)
    if layout == "lm_head_own_unit":
        fully_shard(model.lm_head, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    head = FusedLogprobLMHead(model, chunk_size=5)
    ids, w = t._batch(seed=rank + 1)

    def ac():
        return nullcontext() if layout == "lm_head_own_unit" else torch.autocast("cuda", dtype=torch.bfloat16)

    with ac():
        f = t._fused_logprobs(model, head, ids, 0.7)
    (f * w).sum().backward()
    fg = t._grads(model)
    with ac():
        u = t._unfused_logprobs(model, ids, 0.7)
    (u * w).sum().backward()
    ug = t._grads(model)
    worst = max(((fg[n] - ug[n]).abs().max() / ug[n].abs().max()).item() for n in fg)
    exact = sum(int(torch.equal(fg[n], ug[n])) for n in fg)
    lm = "lm_head.weight" if "lm_head.weight" in fg else "model.embed_tokens.weight"
    if rank == 0:
        print(f"[fsdp world={dist.get_world_size()}] {layout}: lp max|d|={(f - u).abs().max().item():.2e} grads bitwise-equal {exact}/{len(fg)} worst max|d|/max|g|={worst:.2e} ({lm} norm {fg[lm].norm().item():.4f} vs {ug[lm].norm().item():.4f})", flush=True)
dist.destroy_process_group()
