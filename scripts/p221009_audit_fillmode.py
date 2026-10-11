#!/usr/bin/env python3
"""Which fill branch each GRB 221009A FIFO-reset gap took.

Parses the stderr of
    blink sat reconstruct 2022-10-09T13:17:00 --before 50 --after 700
saved to data/p221009/audit/reconstruct.err. Each gap prints one line
"gap[i]: <dur>s  n_lost=<n>  cross-ref|FALLBACK|NO-REF ..." in box order
A, B, C (gap index restarts at 0 per box). Gap start times are matched by
index to the `blink sat detect` lists in detect_{a,b,c}.txt.

Run from the blink repo root:
    ./.venv/bin/python scripts/p221009_audit_fillmode.py
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

D = Path("data/p221009/audit")
T0 = 339945423.0
pat = re.compile(r"^gap\[(\d+)\]: ([\d.]+)s\s+n_lost=(\d+)\s+(cross-ref|FALLBACK|NO-REF)")
recs, box_i, last = [], -1, None
for ln in (D / "reconstruct.err").read_text().splitlines():
    m = pat.match(ln)
    if not m:
        continue
    i = int(m.group(1))
    if i == 0 or last is None or i < last:
        box_i += 1
    last = i
    recs.append(("ABC"[box_i], i, float(m.group(2)), int(m.group(3)), m.group(4)))
df = pd.DataFrame(recs, columns=["box", "idx", "dur", "n_lost", "mode"])
lines = []
def say(s=""):
    print(s); lines.append(s)

for b in "ABC":
    det = pd.read_csv(D / f"detect_{b.lower()}.txt").sort_values("start_met").reset_index(drop=True)
    s = df[df.box == b].copy()
    if len(det) == len(s):
        df.loc[s.index, "t"] = det.start_met.values[s.idx.values] - T0
    c = s["mode"].value_counts().to_dict()
    nl = s.groupby("mode").n_lost.sum().to_dict()
    tot = s.n_lost.sum()
    say(f"Box {b}: {len(s)} gaps (detect list {len(det)}); modes {c}; filled events by mode "
        + ", ".join(f"{k} {v} ({100*v/tot:.1f}%)" for k, v in nl.items()))
say(f"ALL: {len(df)} gaps; mode fraction (by gap) "
    + ", ".join(f"{k} {100*v/len(df):.1f}%" for k, v in df['mode'].value_counts().items())
    + "; by filled events "
    + ", ".join(f"{k} {100*v/df.n_lost.sum():.1f}%" for k, v in df.groupby('mode').n_lost.sum().items()))
if "t" in df:
    bins = [170, 200, 213, 230, 248, 268, 276, 300, 440, 600]
    df["phase"] = pd.cut(df.t, bins)
    tab = df.pivot_table(index="phase", columns="mode", values="n_lost", aggfunc="sum",
                         fill_value=0, observed=False)
    say("\nfilled events by phase (T0-relative s) and mode:\n" + tab.to_string())
(D / "audit_fillmode.txt").write_text("\n".join(lines) + "\n")
