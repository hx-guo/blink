#!/usr/bin/env python3
"""Apply M1 model (free β, γ from multi-model comparison) globally and plot.
Same layout as plot_n_below_beta_gamma_full.py for direct comparison.

M1 parameters (from multi-model fit on 4.1M normal-mode points per Box):
  Box A: β=2.576, γ=0.074, α=1.265, b=-374.5
  Box B: β=2.544, γ=0.074, α=1.308, b=-395.5
  Box C: β=2.627, γ=0.094, α=1.319, b=-413.3
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CSV_DIR = Path("n_below_study/per_sec_csvs")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100

# Per-Box M1 fit parameters
FITS = {
    "A": dict(beta=2.576, gamma=0.074, alpha=1.265, b=-374.5),
    "B": dict(beta=2.544, gamma=0.074, alpha=1.308, b=-395.5),
    "C": dict(beta=2.627, gamma=0.094, alpha=1.319, b=-413.3),
}


def load():
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
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(files)}")
    df = pd.concat(parts, ignore_index=True)
    df["length"] = df["L_cycles"].astype("float32") * 16e-6
    df = df[df["L_cycles"] > L_THRESH].copy()
    g = df.groupby(["date","box","met_sec"], observed=True)["Sci"].sum()
    g.name = "sci_sec_total"
    df = df.merge(g, on=["date","box","met_sec"])
    df = df[df["sci_sec_total"] > SCI_SEC_TOTAL_MIN]
    df["sci_rate"] = df["Sci"] / df["length"]
    df["wide_rate"] = df["Wide"] / df["length"]
    df["large_rate"] = df["Large"] / df["length"]
    df["pho_rate"] = df["PHO"] / df["length"]
    print(f"  filtered rows: {len(df):,}")
    return df


def main():
    df = load()
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 17), sharex=True, sharey=True)
    last_hb = None
    SCI_MIN, SCI_MAX = 40.0, 5000.0
    Y_MIN, Y_MAX = 1.0, 5000.0

    for ax, box in zip(axes, "ABC"):
        sub = df[df["box"] == box]
        p = FITS[box]
        # Sci_pred = (PHO - β·W - γ·L - b) / (1+α)
        pho_corr = sub["pho_rate"].values - p["beta"]*sub["wide_rate"].values \
                    - p["gamma"]*sub["large_rate"].values
        sci_pred = (pho_corr - p["b"]) / (1 + p["alpha"])
        sci = sub["sci_rate"].values
        rms = float(np.sqrt(np.mean((sci - sci_pred) ** 2)))

        sp_pos = np.maximum(sci_pred, Y_MIN * 0.5)
        keep = (sci >= SCI_MIN) & (sci <= SCI_MAX) & (sp_pos <= Y_MAX)
        x = sci[keep]; y = sp_pos[keep]
        # Density-colored scatter: 200×200 log-log histogram → per-point density
        nbins = 200
        x_edges = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), nbins + 1)
        y_edges = np.logspace(np.log10(Y_MIN), np.log10(Y_MAX), nbins + 1)
        H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        ix = np.clip(np.searchsorted(x_edges, x) - 1, 0, nbins - 1)
        iy = np.clip(np.searchsorted(y_edges, y) - 1, 0, nbins - 1)
        density = H[ix, iy]
        order = np.argsort(density)
        hb = ax.scatter(x[order], y[order], c=density[order], s=1.5,
                        cmap="viridis", norm=LogNorm(vmin=1, vmax=density.max()),
                        rasterized=True, linewidths=0)
        last_hb = hb
        ax.set_xscale("log"); ax.set_yscale("log")

        # Binned median (≥500 pts)
        bins = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 50)
        bc = 0.5 * (bins[:-1] + bins[1:])
        med = []
        for i in range(len(bins) - 1):
            m = (sci >= bins[i]) & (sci < bins[i+1])
            med.append(np.median(sci_pred[m]) if m.sum() > 500 else np.nan)
        ax.plot(bc, np.array(med), "-", color="orange", lw=2.2, zorder=5,
                label="binned median (≥500 pts)")
        ax.plot([SCI_MIN, SCI_MAX], [SCI_MIN, SCI_MAX], "r--", lw=1.8, zorder=6,
                label=f"y = x  (β={p['beta']:.2f}, γ={p['gamma']:.3f}, "
                      f"α={p['alpha']:.3f}, b={p['b']:.0f})")

        ax.set_xlim(SCI_MIN, SCI_MAX); ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xlabel("Sci observed [cnt/s/det]")
        if box == "A":
            ax.set_ylabel(r"Sci predicted = $(\mathrm{PHO}-\beta\mathrm{W}-\gamma\mathrm{L}-b)/(1+\alpha)$"
                          "  [cnt/s/det]")
        else:
            ax.set_ylabel("Sci predicted [cnt/s/det]")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
        ax.set_title(f"Box {box}  (N={len(sub):,})  RMS = {rms:.0f}")
        ax.grid(alpha=0.3, which="both")

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.08, 0.02, 0.84])
    cb = fig.colorbar(last_hb, cax=cax)
    cb.set_label("per-det-sec bin count (log scale)")

    fig.suptitle(r"M1 model: $\mathrm{PHO} = (1+\alpha)\,N_n + \beta\,\mathrm{Wide} + \gamma\,\mathrm{Large} + b$,"
                 " free fit per Box  "
                 f"|  {df['date'].nunique()} dates × 3 boxes × 6 det, {len(df):,} bins",
                 fontsize=12, y=0.995)
    out = OUT_DIR / "n_below_m1_full.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
