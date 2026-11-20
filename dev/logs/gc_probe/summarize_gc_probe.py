"""Summarize gc probe JSONL files (one per rank) from _gc_probe_3976.py (issue #3976)."""

import collections
import glob
import json
import os
import re
import statistics
import sys

GIB = 2**30
recs = collections.defaultdict(list)
others = []
autos = []
freezes = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        if r["event"] == "explicit":
            recs[r["rank"]].append(r)
        elif r["event"] == "explicit_other":
            others.append(r)
        elif r["event"] == "freeze":
            freezes.append(r)
        else:
            autos.append(r)

ranks = sorted(recs)
print(f"ranks: {ranks}  explicit calls per rank: {[len(recs[r]) for r in ranks]}")

print("\n## first call per rank (after setup)")
for r in ranks:
    c = recs[r][0]
    print(
        f"rank {r}: {c['caller'].split(':')[1]} {c['seconds']:.3f}s collected={c['collected']} "
        f"alloc_freed={c['allocated_freed'] / GIB:.3f}GiB reserved_freed={c['reserved_freed'] / GIB:.3f}GiB "
        f"cuda_tensors={c.get('cuda_tensor_storages')} cuda={c.get('cuda_tensor_bytes', 0) / GIB:.3f}GiB"
    )
    if r == ranks[0]:
        print("  top types:", c.get("top_types", [])[:8])
        print("  largest CUDA tensors:", c.get("cuda_largest", [])[:6])

print("\n## steady state (calls 2..): per call site")
by_site = collections.defaultdict(list)
for r in ranks:
    for c in recs[r][1:]:
        by_site[c["caller"].split(":")[1] + ":" + c["caller"].split(":")[2]].append(c)
for site, cs in by_site.items():
    secs = [c["seconds"] for c in cs]
    freed = max(c["allocated_freed"] for c in cs)
    cuda = max(c.get("cuda_tensor_storages") or 0 for c in cs)
    print(
        f"{site:32s} n={len(cs):3d} sec median={statistics.median(secs):.3f} "
        f"min={min(secs):.3f} max={max(secs):.3f} max_alloc_freed={freed / GIB:.4f}GiB "
        f"max_cuda_tensors_in_garbage={cuda}"
    )

print("\n## same call index across ranks: spread of seconds (rank skew)")
n = min(len(recs[r]) for r in ranks)
skews = []
for i in range(1, n):
    s = [recs[r][i]["seconds"] for r in ranks]
    skews.append(max(s) - min(s))
print(f"calls={n - 1} max-min per call: median={statistics.median(skews):.3f}s max={max(skews):.3f}s")

per_step = collections.Counter()
tot = sum(c["seconds"] for c in recs[ranks[0]][1:])
print(f"\nrank {ranks[0]} total explicit gc seconds after the first call: {tot:.2f}s over {len(recs[ranks[0]]) - 1} calls")
print(f"tracked objects (rank {ranks[0]}): {recs[ranks[0]][-1]['tracked_objects']}")
print(f"auto gen2 collections: {len(autos)}; explicit gc from torch/megatron: {len(others)}")

if freezes:
    fc = [a["frozen_objects"] for a in freezes]
    print(f"gc.freeze(): ranks={len(freezes)} frozen objects min={min(fc)} max={max(fc)} "
          f"seconds max={max(a['seconds'] for a in freezes):.3f}")

r0 = recs[ranks[0]]
if "rss" in r0[0]:
    print(f"\n## rank {ranks[0]} memory over calls (site, cuda allocated before, host RSS, frozen)")
    for c in r0:
        print(f"  call {c['call']:3d} {c['caller'].split(':')[1]:28s} "
              f"alloc={c['allocated_before'] / GIB:7.3f}GiB rss={c['rss'] / GIB:7.3f}GiB "
              f"tracked={c['tracked_objects']} frozen={c.get('frozen_objects')}")

log = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1].rstrip("/"))), "entry.log")
if os.path.exists(log):
    print("\n## per step (entry.log)")
    step, row = None, {}
    for line in open(log, errors="replace"):
        m = re.search(r"=+ Step (\d+)/\d+ =+", line)
        if m:
            step = int(m.group(1))
        for key, pat in (("loss", r"• Loss: ([-\d.e]+)"), ("gen_kl", r"• Generation KL Error: ([-\d.e]+)"),
                         ("step_s", r"• Total step time: ([\d.]+)s")):
            m = re.search(pat, line)
            if m and step is not None:
                row.setdefault(step, {})[key] = float(m.group(1))
    for s in sorted(row):
        print(f"  step {s:2d} " + " ".join(f"{k}={v}" for k, v in row[s].items()))
    later = [row[s]["step_s"] for s in row if s >= 3 and "step_s" in row[s]]
    if later:
        print(f"  steps>=3 step time: mean={statistics.mean(later):.2f}s median={statistics.median(later):.2f}s n={len(later)}")
