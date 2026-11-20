"""Print one line per RESULT record of a multi_gpu_offload_check.py log read from stdin."""

import json
import sys

for line in sys.stdin:
    if not line.startswith("RESULT ") or '"warmup"' in line:
        continue
    r = json.loads(line[7:])
    print(
        f"  {r['tag']:<11} ep={r['ep']} rel_l2={r.get('rel_l2_dgrad', 0):.2e} max={r['max_dgrad_vs_ac']:.2e}"
        f" losses={r['losses']} retained={r['retained_gib']} peak={r['peak_gib']}"
        f" fwd={r['fwd_s']} bwd={r['bwd_s']} host={r['host_pinned_peak_gib']}"
    )
