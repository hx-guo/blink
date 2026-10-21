#!/usr/bin/env python3
"""Analyze aminfo bit count distribution.
   - 0 bits: no ACD veto (clean photons)
   - 1 bit:  random ACD coincidence (photon + ACD background)
   - 2+ bits: charged particle traversing multiple ACD panels
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Loading 3 boxes...")
df_list = []
for box, csv_path in [("A", "/tmp/260226_boxA_acd.csv"),
                      ("B", "/tmp/260226_boxB_acd.csv"),
                      ("C", "/tmp/260226_boxC_acd.csv")]:
    df = pd.read_csv(csv_path, usecols=["type", "met", "channel", "det_id", "aminfo"],
                     dtype={"type": "category", "met": "float64", "channel": "uint16",
                            "det_id": "int8", "aminfo": "uint32"})
    df = df[df["type"] == "EVT"].copy()
    df["box"] = box
    df["bitcount"] = df["aminfo"].apply(lambda x: bin(int(x)).count("1"))
    df_list.append(df)
    print(f"  Box {box}: {len(df):,} events", flush=True)
big = pd.concat(df_list, ignore_index=True)

# Distribution of bit counts
print(f"\n=== aminfo bit count distribution (3 boxes pooled) ===")
print(big["bitcount"].value_counts().sort_index().to_string())
print(f"\nFraction with 0 bits: {(big['bitcount'] == 0).mean() * 100:.2f}%")
print(f"Fraction with 1 bit:  {(big['bitcount'] == 1).mean() * 100:.2f}%")
print(f"Fraction with 2+ bits: {(big['bitcount'] >= 2).mean() * 100:.2f}%")
print(f"Fraction with 3+ bits: {(big['bitcount'] >= 3).mean() * 100:.2f}%")
print(f"Fraction with 5+ bits: {(big['bitcount'] >= 5).mean() * 100:.2f}%")

# Channel (energy) distribution split by ACD bitcount
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
for bc, label, color in [(0, "0 bits (clean)", "C0"),
                          (1, "1 bit (random coinc)", "C1"),
                          (2, "2 bits", "C2"),
                          (3, "≥3 bits (likely particle)", "C3")]:
    if bc < 3:
        sel = big[big["bitcount"] == bc]
    else:
        sel = big[big["bitcount"] >= bc]
    if len(sel) > 1000:
        ax.hist(sel["channel"], bins=np.arange(0, 256, 4), alpha=0.5,
                 label=f"{label}: {len(sel)/1000:.0f}k", color=color, density=True)
ax.set_xlabel("channel (energy)")
ax.set_ylabel("density")
ax.set_title("Channel distribution by ACD bit count")
ax.legend(fontsize=9)
ax.set_yscale("log")
ax.grid(alpha=0.3)

# bitcount histogram
ax = axes[1]
counts = big["bitcount"].value_counts().sort_index()
ax.bar(counts.index, counts.values, color="C0", edgecolor="k")
ax.set_xlabel("ACD bit count (popcount of aminfo)")
ax.set_ylabel("number of events")
ax.set_title("ACD bit count distribution")
ax.set_yscale("log")
ax.grid(alpha=0.3)

# Per-second rate of high-bitcount events vs total Sci rate
ax = axes[2]
big_a = big[big["box"] == "A"].copy()
big_a["sec"] = big_a["met"].astype(int)
per_sec = big_a.groupby("sec").agg(
    sci_total=("met", "size"),
    sci_clean=("bitcount", lambda x: (x == 0).sum()),
    sci_1bit=("bitcount", lambda x: (x == 1).sum()),
    sci_2plus=("bitcount", lambda x: (x >= 2).sum()),
    sci_3plus=("bitcount", lambda x: (x >= 3).sum()),
).reset_index()
ax.scatter(per_sec["sci_total"], per_sec["sci_2plus"]/per_sec["sci_total"], s=3, alpha=0.4,
            color="C3", label="≥2 bits / total")
ax.scatter(per_sec["sci_total"], per_sec["sci_1bit"]/per_sec["sci_total"], s=3, alpha=0.4,
            color="C1", label="1 bit / total")
ax.scatter(per_sec["sci_total"], (per_sec["sci_total"] - per_sec["sci_clean"])/per_sec["sci_total"], s=3, alpha=0.4,
            color="C0", label="any ACD / total")
ax.set_xlabel("Total Sci rate per sec [cnt/s/box]")
ax.set_ylabel("Fraction with ACD bits")
ax.set_title("Box A: ACD fraction vs rate per second")
ax.legend(fontsize=9)
ax.set_xscale("log")
ax.grid(alpha=0.3, which="both")

fig.tight_layout()
out = "plots/acd_bitcount.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")
