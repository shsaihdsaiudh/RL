"""Break down optimizer offload/onload cost from an nsys SQLite export (issue #3976)."""

import sqlite3
import sys
from collections import defaultdict

db = sqlite3.connect(sys.argv[1])
q = db.execute
S = dict(q("select id, value from StringIds"))
KIND = {k: v for k, _, v in q("select * from ENUM_CUDA_MEM_KIND")}
COPY = {1: "HtoD", 2: "DtoH", 8: "DtoD"}

nvtx = []
for text, tid, s, e in q(
    "select text, textId, start, end from NVTX_EVENTS where end is not null"
):
    name = text or S.get(tid, "")
    if name.startswith("megatron_policy_worker/"):
        nvtx.append((s, e, name))
nvtx.sort()

copies = list(
    q(
        "select start, end, bytes, copyKind, srcKind, dstKind from "
        "CUPTI_ACTIVITY_KIND_MEMCPY order by start"
    )
)
api = [
    (s, e, S.get(n, str(n)))
    for s, e, n in q(
        "select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start"
    )
]


def summarize(lo, hi, label):
    dur = (hi - lo) / 1e9
    print(f"\n### {label}: {dur:.3f}s")
    agg = defaultdict(lambda: [0, 0, 0])
    for s, e, b, ck, sk, dk in copies:
        if lo <= s < hi:
            k = f"{COPY.get(ck, ck)} {KIND.get(sk, sk)}->{KIND.get(dk, dk)}"
            a = agg[k]
            a[0] += 1
            a[1] += b
            a[2] += e - s
    for k, (n, b, t) in sorted(agg.items(), key=lambda x: -x[1][1]):
        gbps = b / t if t else 0
        print(
            f"  copy {k:28s} n={n:4d} {b / 1e9:7.3f}GB gpu={t / 1e9:6.3f}s {gbps:6.1f}GB/s"
        )
    aagg = defaultdict(lambda: [0, 0])
    busy = 0
    last = lo
    for s, e, n in api:
        if lo <= s < hi:
            aagg[n][0] += 1
            aagg[n][1] += e - s
            s2 = max(s, last)
            if e > s2:
                busy += e - s2
                last = e
    for n, (c, t) in sorted(aagg.items(), key=lambda x: -x[1][1])[:5]:
        print(f"  api  {n:40s} calls={c:4d} {t / 1e9:6.3f}s")
    print(f"  time not inside any CUDA API call (CPU-side): {dur - busy / 1e9:.3f}s")
    inside = [a for a in api if lo <= a[0] < hi]
    gaps = []
    prev = lo
    for s, e, n in inside:
        gaps.append((s - prev, prev, n))
        prev = max(prev, e)
    gaps.append((hi - prev, prev, "<range end>"))
    g, at, nxt = max(gaps)
    print(f"  largest CPU-only gap: {g / 1e9:.3f}s at +{(at - lo) / 1e9:.3f}s, followed by {nxt}")


for s, e, name in nvtx:
    if "offload" in name:
        summarize(s, e, f"{name} @ {s / 1e9:.2f}s")

# prepare_for_training has no NVTX range: find H2D bursts outside named ranges.
print("\n### H2D bursts outside megatron_policy_worker/* ranges (prepare_for_* onload)")
outside = [
    c
    for c in copies
    if c[3] == 1
    and c[2] > 1 << 20
    and not any(s <= c[0] < e for s, e, _ in nvtx)
]
bursts = []
for c in outside:
    if bursts and c[0] - bursts[-1][-1][1] < 200_000_000:
        bursts[-1].append(c)
    else:
        bursts.append([c])
for b in bursts:
    summarize(b[0][0], b[-1][1] + 1, f"H2D burst @ {b[0][0] / 1e9:.2f}s")
