#!/usr/bin/env python3
"""Test the hypothesis that the two branches at high Sci are normal vs low-gain mode,
by examining the Large/PHO ratio. In low gain, ch=275 corresponds to a much higher
real energy, so very few events should exceed it → Large/PHO should drop sharply.

Plot 1: histogram of Large/PHO ratio (look for bimodal distribution)
Plot 2: scatter Sci_pred vs Sci, colored by Large/PHO
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CSV_DIR = Path("n_below_study/per_sec_csvs")
OUT_DIR = Path("plots")
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BETA = 2.0
GAMMA = 1.2

dtype = {"date": "string", "box": "category", "met_sec": "int64",
         "det": "int8", "L_cycles": "int32",
         "PHO": "int32", "Wide": "int32", "Large": "int32", "Sci": "int32"}
files = sorted(CSV_DIR.glob("*.csv"))
print(f"Loading {len(files)} CSVs...")
parts = []
for i, f in enumerate(files):
    try:
        parts.append(pd.read_csv(f, usecols=list(dtype), dtype=dtype))
    except Exception:
        pass
    if (i+1) % 300 == 0:
        print(f"  {i+1}/{len(files)}")
df = pd.concat(parts, ignore_index=True)
df["length"] = df["L_cycles"].astype("float32") * 16e-6
df = df[df["L_cycles"] > L_THRESH].copy()
g = df.groupby(["date", "box", "met_sec"], observed=True)["Sci"].sum()
g.name = "sci_sec_total"
df = df.merge(g, on=["date", "box", "met_sec"])
df = df[df["sci_sec_total"] > SCI_SEC_TOTAL_MIN]

df["sci_rate"] = df["Sci"] / df["length"]
df["pho_corr_rate"] = (df["PHO"] - BETA*df["Wide"] - GAMMA*df["Large"]) / df["length"]
df["nb_rate"] = df["pho_corr_rate"] - df["sci_rate"]
df["large_frac"] = df["Large"] / df["PHO"].clip(lower=1)
df["wide_frac"] = df["Wide"] / df["PHO"].clip(lower=1)
print(f"After filter: {len(df):,} per-det-sec rows")

# === Plot 1: distributions of Large/PHO and Wide/PHO ===
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for col, box in enumerate("ABC"):
    sub = df[df["box"] == box]
    axes[0, col].hist(sub["large_frac"], bins=np.linspace(0, 0.5, 200), log=True,
                      color="steelblue", alpha=0.8)
    axes[0, col].set_xlabel("Large / PHO")
    axes[0, col].set_ylabel("count (log)")
    axes[0, col].set_title(f"Box {box}  Large/PHO distribution")
    axes[0, col].grid(alpha=0.3)
    axes[0, col].axvline(0.05, color="red", ls="--", lw=1, label="0.05 split")
    axes[0, col].legend()

    axes[1, col].hist(sub["wide_frac"], bins=np.linspace(0, 0.5, 200), log=True,
                      color="darkorange", alpha=0.8)
    axes[1, col].set_xlabel("Wide / PHO")
    axes[1, col].set_ylabel("count (log)")
    axes[1, col].set_title(f"Box {box}  Wide/PHO distribution")
    axes[1, col].grid(alpha=0.3)

fig.tight_layout()
out1 = OUT_DIR / "large_frac_distribution.png"
fig.savefig(out1, dpi=130, bbox_inches="tight")
print(f"Saved: {out1}")

# Print stats: percentiles of large_frac per box
print("\nLarge/PHO percentiles per box:")
for box in "ABC":
    sub = df[df["box"] == box]["large_frac"]
    pct = np.percentile(sub, [1, 5, 25, 50, 75, 95, 99])
    print(f"  Box {box}: 1%={pct[0]:.4f}  5%={pct[1]:.4f}  25%={pct[2]:.4f}  "
          f"50%={pct[3]:.4f}  75%={pct[4]:.4f}  95%={pct[5]:.4f}  99%={pct[6]:.4f}")
    # bimodality check: fraction below 0.05 vs above
    n_low = (sub < 0.05).sum()
    n_hi = (sub >= 0.05).sum()
    print(f"          fraction below 0.05: {n_low/len(sub)*100:.2f}% ({n_low:,})  "
          f"above: {n_hi/len(sub)*100:.2f}% ({n_hi:,})")

# === Plot 2: Sci_pred vs Sci colored by Large/PHO ===
fig, axes = plt.subplots(3, 1, figsize=(8.5, 17), sharex=True, sharey=True)
SCI_MIN, SCI_MAX = 40.0, 5000.0
Y_MIN, Y_MAX = 1.0, 5000.0

for ax, box in zip(axes, "ABC"):
    sub = df[df["box"] == box]
    sci = sub["sci_rate"].values
    nb = sub["nb_rate"].values
    pho_corr = sub["pho_corr_rate"].values
    lf = sub["large_frac"].values
    n = len(sub)

    X = np.column_stack([np.ones(n), sci])
    coef, *_ = np.linalg.lstsq(X, nb, rcond=None)
    b, alpha = coef
    sci_pred = (pho_corr - b) / (1 + alpha)
    rms = float(np.sqrt(np.mean((sci - sci_pred) ** 2)))

    sp_pos = np.maximum(sci_pred, Y_MIN * 0.5)
    keep = (sci >= SCI_MIN) & (sci <= SCI_MAX) & (sp_pos <= Y_MAX)
    # color by Large/PHO (clipped to a meaningful range)
    sc = ax.scatter(sci[keep], sp_pos[keep], c=lf[keep],
                    s=1.5, cmap="plasma",
                    norm=plt.Normalize(vmin=0, vmax=0.20),
                    rasterized=True, linewidths=0)

    ax.plot([SCI_MIN, SCI_MAX], [SCI_MIN, SCI_MAX], "k--", lw=1.5, alpha=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(SCI_MIN, SCI_MAX); ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("Sci observed [cnt/s/det]")
    ax.set_ylabel("Sci predicted [cnt/s/det]")
    ax.set_title(f"Box {box}  N={n:,}  b={b:.0f}, α={alpha:.3f}, RMS={rms:.0f}")
    ax.grid(alpha=0.3, which="both")

fig.subplots_adjust(right=0.88)
cax = fig.add_axes([0.90, 0.08, 0.02, 0.84])
cb = fig.colorbar(sc, cax=cax)
cb.set_label("Large / PHO  (proxy for gain mode)")
out2 = OUT_DIR / "n_below_by_large_frac.png"
fig.savefig(out2, dpi=130, bbox_inches="tight")
print(f"Saved: {out2}")
