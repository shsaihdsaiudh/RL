import torch

import nemo_rl.distributed.model_utils as mu
from nemo_rl.distributed.fused_linear_logprobs import linear_selected_logprobs

torch.distributed.all_reduce = lambda *a, **k: None


def unfused(h, w, t, chunk, temp):
    logits = torch.nn.functional.linear(h, w).to(torch.float32)
    if temp != 1.0:
        logits = logits.div_(temp)
    return mu.ChunkedDistributedLogprob.apply(logits, t, 0, w.shape[0], chunk, None, False)


def run(fn, h0, w0, t, g):
    h = h0.clone().requires_grad_()
    w = w0.clone().requires_grad_()
    out = fn(h, w, t)
    out.backward(g)
    return out.detach(), h.grad, w.grad


for dtype in (torch.float32, torch.bfloat16):
    for temp in (1.0, 0.7):
        for chunk in (1, 7, 64, 1000):
            torch.manual_seed(0)
            B, S, H, V = 2, 37, 48, 257
            h0 = torch.randn(B, S, H, device="cuda").to(dtype)
            w0 = (torch.randn(V, H, device="cuda") * 0.3).to(dtype)
            t = torch.randint(V, (B, S), device="cuda")
            g = torch.randn(B, S, device="cuda")
            ref = run(lambda h, w, t: unfused(h, w, t, chunk, temp), h0, w0, t, g)
            fus = run(lambda h, w, t: linear_selected_logprobs(h, w, t, chunk_size=chunk, temperature=temp), h0, w0, t, g)
            exact = run(lambda h, w, t: (torch.nn.functional.linear(h, w) / temp).log_softmax(-1).gather(-1, t.unsqueeze(-1)).squeeze(-1), h0.double(), w0.double(), t, g.double())
            msg = [f"{str(dtype):15s} T={temp} c={chunk:4d}"]
            for name, a, b, e in zip(("lp", "dh", "dw"), fus, ref, exact):
                msg.append(f"{name}: fus-ref={(a.double()-b.double()).abs().max().item():.2e} fus-ex={(a.double()-e).abs().max().item():.2e} ref-ex={(b.double()-e).abs().max().item():.2e}")
            print(" | ".join(msg))

# weight grad accumulation stress
for dtype in (torch.bfloat16,):
    torch.manual_seed(0)
    N, H, V = 4096, 64, 512
    h0 = torch.randn(1, N, H, device="cuda").to(dtype)
    w0 = (torch.randn(V, H, device="cuda") * 0.3).to(dtype)
    t = torch.randint(V, (1, N), device="cuda")
    g = torch.randn(1, N, device="cuda")
    exact = run(lambda h, w, t: torch.nn.functional.linear(h, w).log_softmax(-1).gather(-1, t.unsqueeze(-1)).squeeze(-1), h0.double(), w0.double(), t, g.double())
    ref = run(lambda h, w, t: unfused(h, w, t, 256, 1.0), h0, w0, t, g)
    for chunk in (1, 16, 256):
        fus = run(lambda h, w, t: linear_selected_logprobs(h, w, t, chunk_size=chunk), h0, w0, t, g)
        def rel(a):
            return ((a.double() - exact[2]).norm() / exact[2].norm()).item()
        print(f"stress c={chunk}: dw rel err fused={rel(fus[2]):.3e} unfused={rel(ref[2]):.3e}")
    # bf16 accumulation (verl-style) for contrast
    hh = h0.reshape(N, H)
    lg = (hh @ w0.t()).float()
    p = lg.softmax(-1)
    oh = torch.zeros_like(p).scatter_(-1, t.reshape(N, 1), 1)
    gl = ((oh - p) * g.reshape(N, 1)).to(dtype)
    acc = torch.zeros_like(w0)
    for s in range(0, N, 16):
        acc += gl[s:s+16].t() @ hh[s:s+16]
    print(f"stress bf16-accum c=16: dw rel err={((acc.double()-exact[2]).norm()/exact[2].norm()).item():.3e}")
