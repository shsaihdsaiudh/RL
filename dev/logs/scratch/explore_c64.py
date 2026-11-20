import sys

import torch

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = sys.argv[1] == "1"
torch.manual_seed(0)
B, S, H, V = 2, 37, 48, 257
h = torch.randn(B * S, H, device="cuda").bfloat16()
w = (torch.randn(V, H, device="cuda") * 0.3).bfloat16()
exact = h.double() @ w.double().t()
full = (h @ w.t()).double()
c64 = torch.cat([(h[:64] @ w.t()), (h[64:] @ w.t())]).double()
first64 = (h[:64] @ w.t()).double()
print("reduced_precision", sys.argv[1])
print("full   vs exact: max", (full - exact).abs().max().item(), "rel", ((full - exact).norm() / exact.norm()).item())
print("c64    vs exact: max", (c64 - exact).abs().max().item(), "rel", ((c64 - exact).norm() / exact.norm()).item())
print("first64 differs from full rows:", (first64 != full[:64]).sum().item(), "of", first64.numel())
print("tail10 differs from full rows:", ((h[64:] @ w.t()).double() != full[64:]).sum().item())
