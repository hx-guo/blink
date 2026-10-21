#!/usr/bin/env python3
"""Audit of the event content behind the GRB 221009A HE light curve.

The light curve in make_f10 / numbers_221009a counts every cache event
(observed + filler), with no channel or pulse-width selection. This script
measures what that event population is: NaI vs CsI (pulse_width in [54,70]
is NaI, as in the 1K Pulse_Width convention) and the channel range, per
burst phase, plus the SAA-segment event content.

Run from the blink repo root:
    ./.venv/bin/python scripts/grb221009a/p221009_audit_events.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = "data/cache_221009a_reconstruct.csv"
T0 = 339945423.0
OUT = Path("data/p221009/audit")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CACHE, usecols=["box", "type", "met", "channel", "pulse_width"],
                 dtype={"box": "category", "type": "category", "met": "float64",
                        "channel": "int16", "pulse_width": "int16"})
df["t"] = df["met"] - T0
lines = []
def say(s=""):
    print(s); lines.append(s)

phases = {"pre -50..-10": (-50, -10), "precursor -2..10": (-2, 10),
          "early 175..212": (175, 212), "main 180..300": (180, 300),
          "flare 442..593": (442, 593), "tail 300..440": (300, 440),
          "late tail 600..700": (600, 700)}
say("phase                type      N         NaI%   ch<20%  ch>=190%  ch==255%  ch==0%")
for name, (lo, hi) in phases.items():
    sel = (df.t >= lo) & (df.t < hi)
    for ty in ("EVT", "FILL_GAP"):
        s = df[sel & (df.type == ty)]
        if not len(s):
            continue
        nai = s.pulse_width.between(54, 70).mean() * 100
        say(f"{name:20s} {ty:8s} {len(s):9d}  {nai:6.1f}  {(s.channel < 20).mean()*100:6.1f}  "
            f"{(s.channel >= 190).mean()*100:7.1f}  {(s.channel == 255).mean()*100:7.2f}  "
            f"{(s.channel == 0).mean()*100:6.2f}")

say("\nSAA-segment events (t in [230,248) or [268,276)):")
saa = ((df.t >= 230) & (df.t < 248)) | ((df.t >= 268) & (df.t < 276))
s = df[saa]
say(f"  N={len(s)}  by box/type: {s.groupby(['box','type'], observed=True).size().to_dict()}")
say(f"  per-second counts: {np.bincount(np.floor(s.t - 230).astype(int).clip(0)).tolist()[:50]}")
say(f"  first rows:\n{s.head(8).to_string()}")

(OUT / "audit_events.txt").write_text("\n".join(lines) + "\n")
print(f"wrote {OUT/'audit_events.txt'}")
