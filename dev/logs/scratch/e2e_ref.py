"""Compare the e2e fused/unfused logprobs against a float32 HF forward of the same tokens."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name, seq, batch = "Qwen/Qwen2.5-0.5B", 4096, 2
tokenizer = AutoTokenizer.from_pretrained(model_name)
torch.manual_seed(0)
input_ids = torch.randint(0, min(len(tokenizer), 150000), (batch, seq))
lengths = torch.tensor([seq, seq - 517])

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
with torch.no_grad():
    logits = model(input_ids=input_ids.cuda(), use_cache=False).logits.float() / 0.7
    ref = logits[:, :-1].log_softmax(-1).gather(-1, input_ids[:, 1:].cuda().unsqueeze(-1)).squeeze(-1)
ref = torch.cat([torch.zeros(batch, 1, device=ref.device), ref], dim=1).cpu()
mask = torch.arange(seq).unsqueeze(0) < lengths.unsqueeze(1)
mask[:, 0] = False

for name in ("unfused", "fused"):
    lp = torch.load(f"/planning-users/yangyeyuan/nemo-rl-4169/logs/e2e/{name}.pt")["logprobs"]
    d = (lp - ref).abs()[mask]
    print(f"{name:8s} vs fp32: max|d|={d.max().item():.4f} mean|d|={d.mean().item():.5f}")
a = torch.load("/planning-users/yangyeyuan/nemo-rl-4169/logs/e2e/unfused.pt")["logprobs"]
b = torch.load("/planning-users/yangyeyuan/nemo-rl-4169/logs/e2e/fused.pt")["logprobs"]
d = (a - b).abs()[mask]
print(f"fused vs unfused: max|d|={d.max().item():.4f} mean|d|={d.mean().item():.5f}")
print(f"padding stays zero: unfused={bool((a[~mask] == 0).all())} fused={bool((b[~mask] == 0).all())}")
