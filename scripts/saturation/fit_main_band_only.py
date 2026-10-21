#!/usr/bin/env python3
"""Re-fit M1 model excluding the low-Sci left clump.

The left clump (Sci < ~300 cnt/s/det) appears to be a different physics regime
(SAA transients, HV switch, idle obs). Including it in the fit may bias the
main-band (β, γ, α, b) estimates.

Fit only Sci > 300 cnt/s/det. Compare parameters and RMS against the full M1 fit.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table_partial.csv.gz")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}
SCI_MAIN_BAND_LO = 300.0  # main band starts here


def load():
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32", "Sci": "int32"}
    files = sorted(CSV_DIR.glob("*.csv"))
    print(f"Loading {len(files)} CSVs...")
    parts = []
    for f in files:
        try:
            parts.append(pd.read_csv(f, usecols=list(dtype), dtype=dtype))
        except Exception:
            pass
    df = pd.concat(parts, ignore_index=True)
    df["length"] = df["L_cycles"].astype("float32") * 16e-6
    df = df[df["L_cycles"] > L_THRESH]
    g = df.groupby(["date","box","met_sec"], observed=True)["Sci"].sum()
    g.name = "sci_sec_total"
    df = df.merge(g, on=["date","box","met_sec"])
    df = df[df["sci_sec_total"] > SCI_SEC_TOTAL_MIN].copy()
    df["sci_rate"] = df["Sci"] / df["length"]
    df["wide_rate"] = df["Wide"] / df["length"]
    df["large_rate"] = df["Large"] / df["length"]
    df["pho_rate"] = df["PHO"] / df["length"]
    df["det_global"] = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")

    hv = pd.read_csv(HV_TABLE,
                     dtype={"date":"string","met_sec":"int64",
                            **{f"hv{i}":"float32" for i in range(18)}})
    hv = hv.set_index(["date","met_sec"]).sort_index()
    keys = pd.MultiIndex.from_arrays(
        [df["date"].astype(str).str.replace("-","",regex=False).values,
         df["met_sec"].values], names=["date","met_sec"])
    hv_arr = hv.reindex(keys).values
    rows = np.arange(len(df))
    df["hv"] = hv_arr[rows, df["det_global"].values.astype(int)]
    df = df[(df["hv"] < -900) & (df["hv"] > -1100)].copy()
    print(f"normal-mode rows: {len(df):,}")
    return df


def free_fit(sub):
    """Fit PHO = (1+α)·Sci + β·Wide + γ·Large + b"""
    X = np.column_stack([np.ones(len(sub)), sub["sci_rate"].values,
                         sub["wide_rate"].values, sub["large_rate"].values])
    y = sub["pho_rate"].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    b, one_plus_a, beta, gamma = coef
    alpha = one_plus_a - 1
    # Predicted Sci
    pho_corr = y - beta*sub["wide_rate"].values - gamma*sub["large_rate"].values
    sci_pred = (pho_corr - b) / (1 + alpha)
    rms = float(np.sqrt(np.mean((sub["sci_rate"].values - sci_pred) ** 2)))
    return dict(b=b, alpha=alpha, beta=beta, gamma=gamma, rms=rms, n=len(sub))


def main():
    df = load()

    print(f"\n=== Fit comparison: full data vs Sci > {SCI_MAIN_BAND_LO} ===")
    fits_full = {}
    fits_main = {}
    print(f"{'Box':>4s} {'cut':>6s} {'N':>10s} {'b':>9s} {'α':>7s} {'β':>7s} {'γ':>7s} {'RMS':>7s}")
    for box in "ABC":
        sub = df[df["box"] == box]
        ff = free_fit(sub)
        fits_full[box] = ff
        print(f"  {box:>2s}    full {ff['n']:>10,d} {ff['b']:>9.1f} {ff['alpha']:>7.3f} "
              f"{ff['beta']:>7.3f} {ff['gamma']:>7.3f} {ff['rms']:>7.1f}")
        sub_main = sub[sub["sci_rate"] > SCI_MAIN_BAND_LO]
        fm = free_fit(sub_main)
        fits_main[box] = fm
        print(f"  {box:>2s}   >{SCI_MAIN_BAND_LO:.0f} {fm['n']:>10,d} {fm['b']:>9.1f} {fm['alpha']:>7.3f} "
              f"{fm['beta']:>7.3f} {fm['gamma']:>7.3f} {fm['rms']:>7.1f}")

    # Plot using MAIN-BAND-ONLY fit on FULL data — see where left clump sits
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 17), sharex=True, sharey=True)
    SCI_MIN, SCI_MAX = 40.0, 5000.0
    Y_MIN, Y_MAX = 1.0, 5000.0
    last_hb = None
    for ax, box in zip(axes, "ABC"):
        sub = df[df["box"] == box]
        p = fits_main[box]
        pho_corr = sub["pho_rate"].values - p["beta"]*sub["wide_rate"].values \
                    - p["gamma"]*sub["large_rate"].values
        sci_pred = (pho_corr - p["b"]) / (1 + p["alpha"])
        sci = sub["sci_rate"].values

        sp_pos = np.maximum(sci_pred, Y_MIN * 0.5)
        keep = (sci >= SCI_MIN) & (sci <= SCI_MAX) & (sp_pos <= Y_MAX)
        x = sci[keep]; y = sp_pos[keep]
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

        # Binned median, computed on subset above SCI_MAIN_BAND_LO
        bins_e = np.logspace(np.log10(SCI_MAIN_BAND_LO), np.log10(SCI_MAX), 30)
        bc = 0.5*(bins_e[:-1] + bins_e[1:])
        med = []
        for i in range(len(bins_e)-1):
            m = (sci >= bins_e[i]) & (sci < bins_e[i+1])
            med.append(np.median(sci_pred[m]) if m.sum() > 500 else np.nan)
        ax.plot(bc, np.array(med), "-", color="orange", lw=2.2, zorder=5,
                label=f"binned median  (Sci > {SCI_MAIN_BAND_LO:.0f})")
        ax.plot([SCI_MIN, SCI_MAX], [SCI_MIN, SCI_MAX], "r--", lw=1.8, zorder=6,
                label=f"y=x  fit_main: β={p['beta']:.2f}, γ={p['gamma']:.3f}, "
                      f"α={p['alpha']:.3f}, b={p['b']:.0f}  RMS={p['rms']:.0f}")
        ax.axvline(SCI_MAIN_BAND_LO, color="gray", ls=":", lw=1, alpha=0.7,
                   label=f"fit cut: Sci={SCI_MAIN_BAND_LO:.0f}")

        ax.set_xlim(SCI_MIN, SCI_MAX); ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xlabel("Sci observed [cnt/s/det]")
        ax.set_ylabel("Sci predicted [cnt/s/det]" if box != "A" else
                      r"Sci predicted = $(\mathrm{PHO}-\beta\mathrm{W}-\gamma\mathrm{L}-b)/(1+\alpha)$  [cnt/s/det]")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
        ax.set_title(f"Box {box}  (N={len(sub):,})  RMS_main = {p['rms']:.0f}")
        ax.grid(alpha=0.3, which="both")

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.08, 0.02, 0.84])
    cb = fig.colorbar(last_hb, cax=cax)
    cb.set_label("per-det-sec bin count (log scale)")

    fig.suptitle(r"Main-band M1 fit (excluding Sci $\leq 300$ left clump). "
                 r"Coefficients from regression on Sci $> 300$ only, applied to full data.",
                 fontsize=11, y=0.995)
    out = OUT_DIR / "n_below_m1_mainband.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
