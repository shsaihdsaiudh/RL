import torch

from tests.unit.models.automodel import test_fused_logprobs as t
from nemo_rl.models.automodel.fused_logprobs import FusedLogprobLMHead

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
for tie in (False, True):
    for precision in ("bf16_params", "fp32_params_bf16_autocast"):
        model = t._tiny_llama(tie_word_embeddings=tie)
        ac = None
        if precision == "bf16_params":
            model = model.bfloat16()
        else:
            ac = torch.autocast("cuda", dtype=torch.bfloat16)
        head = FusedLogprobLMHead(model, chunk_size=5)
        ids, w = t._batch(1)
        with ac or torch.autocast("cuda", enabled=False):
            f = t._fused_logprobs(model, head, ids, 0.7)
        (f * w).sum().backward()
        fg = t._grads(model)
        with ac or torch.autocast("cuda", enabled=False):
            u = t._unfused_logprobs(model, ids, 0.7)
        (u * w).sum().backward()
        ug = t._grads(model)
        worst = max(((fg[n].float() - ug[n].float()).abs().max() / ug[n].float().abs().max()).item() for n in fg)
        exact = sum(int(torch.equal(fg[n], ug[n])) for n in fg)
        print(f"tie={tie} {precision}: lp max|d|={(f-u).abs().max().item():.2e}  grads bitwise-equal {exact}/{len(fg)}  worst max|d|/max|g|={worst:.2e}")
