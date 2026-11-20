import argparse
import time

import torch

import nemo_rl.distributed.model_utils as mu
from nemo_rl.distributed.fused_linear_logprobs import linear_selected_logprobs

torch.distributed.all_reduce = lambda *a, **k: None
GiB = 1024**3

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, default=32768)
p.add_argument("--hidden", type=int, default=4096)
p.add_argument("--vocab", type=int, default=131072)
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--unfused-chunk", type=int, default=2048)
p.add_argument("--fused-chunks", type=int, nargs="+", default=[512, 1024, 2048])
args = p.parse_args()

torch.manual_seed(0)
dev = "cuda"
hidden0 = (torch.randn(1, args.tokens, args.hidden, device=dev) * 1.0).bfloat16()
weight0 = (torch.randn(args.vocab, args.hidden, device=dev) * (1.0 / args.hidden**0.5) * 3).bfloat16()
target = torch.randint(args.vocab, (1, args.tokens), device=dev)
grad_out = torch.randn(1, args.tokens, device=dev)


def unfused(h, w):
    logits = torch.nn.functional.linear(h, w).to(torch.float32)
    if args.temperature != 1.0:
        logits = logits.div_(args.temperature)
    return mu.ChunkedDistributedLogprob.apply(logits, target, 0, args.vocab, args.unfused_chunk, None, False)


def measure(name, fn, train=True, reps=3):
    results = None
    times = []
    peaks = []
    for _ in range(reps):
        h = hidden0.clone().requires_grad_(train)
        w = weight0.clone().requires_grad_(train)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        if train:
            out = fn(h, w)
            out.backward(grad_out)
        else:
            with torch.no_grad():
                out = fn(h, w)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        peaks.append((torch.cuda.max_memory_allocated() - base) / GiB)
        results = (out.detach(), h.grad, w.grad) if train else (out,)
        del h, w, out
    print(f"{name:28s} peak_over_inputs={min(peaks):6.2f} GiB  time={min(times[1:] or times)*1000:8.1f} ms", flush=True)
    return results


print(f"tokens={args.tokens} hidden={args.hidden} vocab={args.vocab} T={args.temperature} "
      f"inputs: hidden {hidden0.numel()*2/GiB:.2f} GiB, weight {weight0.numel()*2/GiB:.2f} GiB, "
      f"one fp32 logits table {args.tokens*args.vocab*4/GiB:.1f} GiB", flush=True)
ref = measure(f"unfused(chunk={args.unfused_chunk}) train", unfused)
ref_inf = measure(f"unfused(chunk={args.unfused_chunk}) no_grad", unfused, train=False)
for c in args.fused_chunks:
    fn = lambda h, w, c=c: linear_selected_logprobs(h, w, target, chunk_size=c, temperature=args.temperature)
    fus = measure(f"fused(chunk={c}) train", fn)
    measure(f"fused(chunk={c}) no_grad", fn, train=False)
    lp = (fus[0] - ref[0]).abs()
    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()
    print(f"    vs unfused: logprob max|d|={lp.max().item():.3e} mean|d|={lp.mean().item():.3e}  "
          f"dh rel={rel(fus[1], ref[1]):.3e}  dw rel={rel(fus[2], ref[2]):.3e}", flush=True)
    del fus
