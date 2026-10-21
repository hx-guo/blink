#!/usr/bin/env python3
"""Simplest linear model: N_below = (PHO - Wide - Large) - Sci, single linear fit
across the full pooled per-second dataset (272+ dates × 3 boxes × 6 det).

This is the OLDEST/SIMPLEST form from plot_n_below.py — no β/γ weights,
no per-slot fit, no saturation — just one global linear N_below = b + α·Sci.

Useful as a baseline for showing the bend / day-to-day spread.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = Path("n_below_study/per_sec_csvs")
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100


def load_all():
    files = sorted(CSV_DIR.glob("*.csv"))
    print(f"Loading {len(files)} CSVs ...")
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32", "Sci": "int32"}
    use = list(dtype)
    parts = []
    for i, f in enumerate(files):
        try:
            parts.append(pd.read_csv(f, usecols=use, dtype=dtype))
        except Exception as e:
            print(f"  ERR {f.name}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}")
    df = pd.concat(parts, ignore_index=True)
    print(f"  total per-det rows = {len(df):,}")
    return df


def main():
    df = load_all()
    df["length"] = df["L_cycles"].astype("float32") * 16e-6
    df = df[df["L_cycles"] > L_THRESH].copy()

    # Per-second box-sum of Sci (used as filter)
    g = df.groupby(["date", "box", "met_sec"], observed=True)["Sci"].sum()
    g.name = "sci_sec_total"
    df = df.merge(g, on=["date", "box", "met_sec"])
    df = df[df["sci_sec_total"] > SCI_SEC_TOTAL_MIN]

    df["sci_rate"] = df["Sci"] / df["length"]
    df["nb_rate"] = (df["PHO"] - df["Wide"] - df["Large"] - df["Sci"]) / df["length"]
    df["year"] = df["date"].str.slice(0, 4).astype(int)
    print(f"After filter: {len(df):,} per-det-sec rows, dates {df['date'].nunique()}, boxes {df['box'].nunique()}")

    # === Single global linear fit, per box ===
    fig, axes = plt.subplots(3, 1, figsize=(8, 16), sharex=True, sharey=True)
    year_uniq = sorted(df["year"].unique())
    cmap = plt.cm.viridis
    yr2col = {y: cmap((i + 0.5) / len(year_uniq)) for i, y in enumerate(year_uniq)}

    results = []
    for ax, box in zip(axes, "ABC"):
        sub = df[df["box"] == box]
        sci = sub["sci_rate"].values
        nb = sub["nb_rate"].values
        pho_minus_wl = sub["nb_rate"].values + sub["sci_rate"].values  # (PHO-Wide-Large)/length
        years = sub["year"].values
        n = len(sub)

        # Linear fit N_below = b + α·Sci
        X = np.column_stack([np.ones(n), sci])
        coef, *_ = np.linalg.lstsq(X, nb, rcond=None)
        b, alpha = coef
        # Invert to predicted Sci rate
        sci_pred = (pho_minus_wl - b) / (1 + alpha)
        rms = float(np.sqrt(np.mean((sci - sci_pred) ** 2)))
        results.append((box, b, alpha, rms, n))

        # Plot scatter colored by year
        for y in year_uniq:
            m = years == y
            ax.scatter(sci[m], sci_pred[m], s=0.5, alpha=0.06, color=yr2col[y],
                       label=str(y), rasterized=True)

        # Binned median of Sci_pred
        bins = np.logspace(np.log10(max(sci.min(), 1)),
                           np.log10(np.percentile(sci, 99.5) + 1), 40)
        bc = 0.5 * (bins[:-1] + bins[1:])
        med = []
        for i in range(len(bins) - 1):
            m = (sci >= bins[i]) & (sci < bins[i + 1])
            med.append(np.median(sci_pred[m]) if m.sum() > 50 else np.nan)
        med = np.array(med)
        ax.plot(bc, med, "k-", lw=2, label="binned median", zorder=5)

        # Reference y = x
        lo, hi = max(sci.min(), 1), np.percentile(sci, 99.5)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5,
                label=f"y = x  (b={b:.0f}, α={alpha:.3f})", zorder=6)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Sci observed [cnt/s/det]")
        if box == "A":
            ax.set_ylabel(r"Sci predicted = $(\mathrm{PHO}-\mathrm{Wide}-\mathrm{Large}-b)/(1+\alpha)$  [cnt/s/det]")
            ax.legend(fontsize=7, markerscale=4, loc="upper left", ncol=2)
        else:
            ax.set_ylabel("Sci predicted [cnt/s/det]")
        ax.set_title(f"Box {box}  (N={n:,})\nb = {b:.0f} cnt/s,  α = {alpha:.3f},  RMS = {rms:.0f}")
        ax.grid(alpha=0.3, which="both")

    fig.suptitle(r"Simplest linear model: $N_{\rm below} = b + \alpha\cdot\mathrm{Sci}$"
                 f"   |   {df['date'].nunique()} dates × 3 boxes × 6 det, "
                 f"{len(df):,} per-det-sec bins", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "n_below_linear_simple_full.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")

    print(f"\n{'Box':>4s} {'b[cnt/s]':>10s} {'α':>8s} {'RMS':>7s} {'N':>10s}")
    for r in results:
        print(f"  {r[0]:>2s}  {r[1]:>10.1f} {r[2]:>8.4f} {r[3]:>7.0f} {r[4]:>10,d}")


if __name__ == "__main__":
    main()
