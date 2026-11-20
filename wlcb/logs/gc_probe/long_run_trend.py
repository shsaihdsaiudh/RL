"""Memory and step-time trends of a long gc probe run (issue #3976).

Usage: python3 long_run_trend.py <run_dir> [<run_dir> ...]
Reads <run>/gc/*.jsonl (probe) and <run>/entry.log (step time, loss, generation KL).
"""

import collections
import glob
import json
import os
import re
import statistics
import sys

GIB = 2**30
CHECK = (2, 3, 10, 25, 50, 75, 100)

for run in sys.argv[1:]:
    per_rank = collections.defaultdict(list)
    for f in glob.glob(os.path.join(run, "gc", "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r["event"] == "explicit" and "offload_before_refit" in r["caller"]:
                per_rank[r["rank"]].append(r)
    print(f"#### {os.path.basename(run.rstrip('/'))}")
    # offload_before_refit runs twice per step (outer, then nested); the outer one has
    # the model on GPU, the nested one runs after everything is offloaded.
    rows = {}
    for rank, cs in sorted(per_rank.items()):
        outer, nested = cs[0::2], cs[1::2]
        rows[rank] = (outer, nested)
    print("step | max RSS over ranks (GiB) | rank0 RSS | max alloc outer (GiB) | max alloc nested (GiB)")
    for s in CHECK:
        i = s - 1
        if not all(i < len(o) for o, _ in rows.values()):
            continue
        rss = [o[i]["rss"] / GIB for o, _ in rows.values()]
        ao = [o[i]["allocated_before"] / GIB for o, _ in rows.values()]
        an = [n[i]["allocated_before"] / GIB for _, n in rows.values() if i < len(n)]
        print(f"{s:4d} | {max(rss):7.3f} | {rss[0]:7.3f} | {max(ao):7.3f} | {max(an):7.3f}")
    r0 = rows[min(rows)][0]
    print(f"rank0 frozen objects: first={r0[0].get('frozen_objects')} last={r0[-1].get('frozen_objects')}; "
          f"tracked: first={r0[0]['tracked_objects']} last={r0[-1]['tracked_objects']}")
    step, d = None, collections.defaultdict(dict)
    for line in open(os.path.join(run, "entry.log"), errors="replace"):
        m = re.search(r"=+ Step (\d+)/\d+ =+", line)
        if m:
            step = int(m.group(1))
        for key, pat in (("loss", r"• Loss: ([-\d.e]+)"), ("kl", r"• Generation KL Error: ([-\d.e]+)"),
                         ("t", r"• Total step time: ([\d.]+)s"), ("reward", r"• Avg Reward: ([-\d.e]+)")):
            m = re.search(pat, line)
            if m and step is not None:
                d[step][key] = float(m.group(1))
    for lo, hi in ((3, 10), (11, 50), (51, 100)):
        xs = [d[s] for s in range(lo, hi + 1) if s in d]
        if xs:
            def mean(k):
                v = [x[k] for x in xs if k in x]
                return f"{statistics.mean(v):.4f}" if v else "-"
            print(f"steps {lo:3d}-{hi:3d}: step_s={mean('t')} loss={mean('loss')} gen_kl={mean('kl')} reward={mean('reward')}")
    print()
