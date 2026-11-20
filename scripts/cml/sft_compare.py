"""Compare SFT offload off/on runs: per-step loss and step time from entry.log, per-GPU peak memory from nvsmi.csv."""

import re
import statistics
import sys
from collections import defaultdict

base = "/planning-users/yangyeyuan/nemo-rl-4166/logs/sft_offload"
pair = sys.argv[1]
runs = {}
for tag in ("off", "on"):
    log = open(f"{base}/{pair}-{tag}/entry.log", errors="replace").read()
    losses = [float(x) for x in re.findall(r"• Loss: ([0-9.]+)", log)]
    times = [float(x) for x in re.findall(r"• Total step time: ([0-9.]+)s", log)]
    peak = defaultdict(int)
    for line in open(f"{base}/{pair}-{tag}/nvsmi.csv"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[2].isdigit():
            peak[int(parts[1])] = max(peak[int(parts[1])], int(parts[2]))
    runs[tag] = (losses, times, peak)
    print(f"{tag}: steps={len(losses)} step_time median(steps 3+)={statistics.median(times[2:]):.2f}s "
          f"peak_mem_mib per GPU={[peak[i] for i in sorted(peak)]} max={max(peak.values())}")
off, on = runs["off"][0], runs["on"][0]
print("step  loss_off  loss_on   diff")
for i, (a, b) in enumerate(zip(off, on), 1):
    print(f"{i:>4}  {a:.4f}    {b:.4f}   {b - a:+.4f}")
