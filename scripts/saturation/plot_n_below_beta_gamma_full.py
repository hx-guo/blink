#!/usr/bin/env python3
"""Verified conservation model with β=2, γ=1.2:

    PHO = N_n + 2·Wide + 1.2·Large + N_below

So:
    N_below := (PHO - 2·Wide - 1.2·Large) - Sci
    N_below ≈ b + α·Sci       (linear fit on quiet data)
=>  Sci_pred = (PHO - 2·Wide - 1.2·Large - b) / (1 + α)

X-axis: observed Sci rate [cnt/s/det].  Note L_cycles*16μs ≈ 0.94s per engineering
record, so we always divide by length to convert counts → rate.

Y-axis: predicted Sci rate from the inverted conservation equation.

Reference y=x is ideal.  Compare against plot_n_below_linear_simple.py
(β=γ=1 baseline) to see how much of the bend is removed by the corrections.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CSV_DIR = Path("n_below_study/per_sec_csvs")
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

L_THRESH = 50_000          # ≥ 0.8s engineering window
SCI_SEC_TOTAL_MIN = 100    # box-sum Sci > 100 cnt/s to keep meaningful bins

BETA = 2.0
GAMMA = 1.2


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
    df["length"] = df["L_cycles"].astype("float32") * 16e-6  # seconds, ~0.94s typ.
    df = df[df["L_cycles"] > L_THRESH].copy()

    # Per-second box-sum of Sci (used as quality filter, drop SAA/idle)
    g = df.groupby(["date", "box", "met_sec"], observed=True)["Sci"].sum()
    g.name = "sci_sec_total"
    df = df.merge(g, on=["date", "box", "met_sec"])
    df = df[df["sci_sec_total"] > SCI_SEC_TOTAL_MIN]

    df["sci_rate"] = df["Sci"] / df["length"]
    # Corrected conservation: subtract 2*Wide + 1.2*Large from PHO
    df["pho_corr_rate"] = (df["PHO"] - BETA * df["Wide"] - GAMMA * df["Large"]) / df["length"]
    df["nb_rate"] = df["pho_corr_rate"] - df["sci_rate"]
    df["year"] = df["date"].str.slice(0, 4).astype(int)
    print(f"After filter: {len(df):,} per-det-sec rows, "
          f"{df['date'].nunique()} dates, {df['box'].nunique()} boxes")

    # === Hexbin density plot with global linear fit per box ===
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 17), sharex=True, sharey=True)

    SCI_MIN = 40.0      # log-axis lower bound; cut empty bottom-left region
    SCI_MAX = 5000.0    # log-axis upper bound; well above 99.99 percentile (~2500)
    Y_MIN = 1.0         # Sci_pred can go very low for outlier bins
    Y_MAX = 5000.0

    results = []
    last_hb = None
    for ax, box in zip(axes, "ABC"):
        sub = df[df["box"] == box]
        sci = sub["sci_rate"].values
        nb = sub["nb_rate"].values
        pho_corr = sub["pho_corr_rate"].values
        n = len(sub)

        # Linear fit: N_below = b + α·Sci
        X = np.column_stack([np.ones(n), sci])
        coef, *_ = np.linalg.lstsq(X, nb, rcond=None)
        b, alpha = coef

        # Invert: Sci_pred = (PHO - 2W - 1.2L - b) / (1+α)
        sci_pred = (pho_corr - b) / (1 + alpha)
        rms = float(np.sqrt(np.mean((sci - sci_pred) ** 2)))
        results.append((box, b, alpha, rms, n))

        # Scatter colored by local 2D density (computed from a log-log histogram).
        # Each point gets the count of its bin → equivalent to hexbin coloring
        # but preserves individual points.
        sp_pos = np.maximum(sci_pred, Y_MIN * 0.5)
        keep = (sci >= SCI_MIN) & (sci <= SCI_MAX) & (sp_pos <= Y_MAX)
        x = sci[keep]
        y = sp_pos[keep]

        nbins = 200
        x_edges = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), nbins + 1)
        y_edges = np.logspace(np.log10(Y_MIN), np.log10(Y_MAX), nbins + 1)
        H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        # Look up each point's bin count
        ix = np.clip(np.searchsorted(x_edges, x) - 1, 0, nbins - 1)
        iy = np.clip(np.searchsorted(y_edges, y) - 1, 0, nbins - 1)
        density = H[ix, iy]
        # Sort ascending so high-density points sit on top
        order = np.argsort(density)
        sc = ax.scatter(x[order], y[order], c=density[order],
                        s=1.5, cmap="viridis",
                        norm=LogNorm(vmin=1, vmax=density.max()),
                        rasterized=True, linewidths=0)
        last_hb = sc
        ax.set_xscale("log"); ax.set_yscale("log")

        # Binned median of Sci_pred (only bins with ≥500 points)
        bins = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 50)
        bc = 0.5 * (bins[:-1] + bins[1:])
        med = []
        for i in range(len(bins) - 1):
            m = (sci >= bins[i]) & (sci < bins[i + 1])
            med.append(np.median(sci_pred[m]) if m.sum() > 500 else np.nan)
        med = np.array(med)
        med_line, = ax.plot(bc, med, "-", color="orange", lw=2.5, zorder=5,
                            label="binned median (≥500 pts)")

        # Reference y = x
        ref_line, = ax.plot([SCI_MIN, SCI_MAX], [SCI_MIN, SCI_MAX], "r--", lw=1.8,
                            zorder=6, label=f"y = x  (b={b:.0f}, α={alpha:.3f})")

        ax.set_xlim(SCI_MIN, SCI_MAX); ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xlabel("Sci observed [cnt/s/det]")
        if box == "A":
            ax.set_ylabel(r"Sci predicted = $(\mathrm{PHO}-2\mathrm{W}-1.2\mathrm{L}-b)/(1+\alpha)$"
                          "  [cnt/s/det]")
        else:
            ax.set_ylabel("Sci predicted [cnt/s/det]")
        ax.legend(handles=[med_line, ref_line], fontsize=9, loc="upper left",
                  framealpha=0.95)
        ax.set_title(f"Box {box}  (N={n:,})  "
                     f"b = {b:.0f} cnt/s,  α = {alpha:.3f},  RMS = {rms:.0f}")
        ax.grid(alpha=0.3, which="both")

    # Single shared colorbar for density
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.08, 0.02, 0.84])
    cb = fig.colorbar(last_hb, cax=cax)
    cb.set_label("per-det-sec bin count (log scale)")

    fig.suptitle(r"Verified model: $\mathrm{PHO} = N_n + 2\mathrm{W} + 1.2\mathrm{L} + N_b$,"
                 r"  $N_b \approx b + \alpha\,\mathrm{Sci}$"
                 f"   |   {df['date'].nunique()} dates × 3 boxes × 6 det, "
                 f"{len(df):,} per-det-sec bins", fontsize=12, y=0.995)
    # don't call fig.tight_layout — it overrides the manual colorbar axes
    out = OUT_DIR / "n_below_beta_gamma_full.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")

    print(f"\n{'Box':>4s} {'b[cnt/s]':>10s} {'α':>8s} {'RMS':>7s} {'N':>10s}")
    for r in results:
        print(f"  {r[0]:>2s}  {r[1]:>10.1f} {r[2]:>8.4f} {r[3]:>7.0f} {r[4]:>10,d}")


if __name__ == "__main__":
    main()
