#!/usr/bin/env python3
"""Three modes side by side, each as a hexbin density panel.
Apples-to-apples with plot_n_below_beta_gamma_full.py but split by PMT HV mode.

Layout: 3 rows (Box A/B/C) x 3 cols (normal / low_gain / off).
Each panel is a 200x200 log-log hexbin; orange binned-median line; red y=x ref.
Fit (b, α) on each panel's data; print all to terminal.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table.csv.gz")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BETA, GAMMA = 2.0, 1.2
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}
SCI_MIN, SCI_MAX = 40.0, 5000.0
Y_MIN, Y_MAX = 1.0, 5000.0
MODES = ["normal", "low_gain", "off"]


def load_per_sec():
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32", "Sci": "int32"}
    files = sorted(CSV_DIR.glob("*.csv"))
    print(f"Loading {len(files)} per-sec CSVs...")
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
    g = df.groupby(["date", "box", "met_sec"], observed=True)["Sci"].sum()
    g.name = "sci_sec_total"
    df = df.merge(g, on=["date", "box", "met_sec"])
    df = df[df["sci_sec_total"] > SCI_SEC_TOTAL_MIN]
    df["sci_rate"] = df["Sci"] / df["length"]
    df["pho_corr_rate"] = (df["PHO"] - BETA*df["Wide"] - GAMMA*df["Large"]) / df["length"]
    df["nb_rate"] = df["pho_corr_rate"] - df["sci_rate"]
    df["box_offset"] = df["box"].map(BOX_OFFSET).astype("int8")
    df["det_global"] = df["box_offset"] + df["det"]
    print(f"  filtered rows: {len(df):,}")
    return df


def load_hv_wide():
    print(f"Loading HV table {HV_TABLE}...")
    hv = pd.read_csv(HV_TABLE,
                     dtype={"date": "string", "met_sec": "int64",
                            **{f"hv{i}": "float32" for i in range(18)}})
    print(f"  wide rows: {len(hv):,}")
    hv = hv.set_index(["date", "met_sec"]).sort_index()
    return hv


def classify(hv):
    out = np.full(len(hv), "missing", dtype=object)
    out[(hv <= -200) & (hv > -1100)] = "normal"
    out[(hv > -200)] = "off"
    out[(hv > -900) & (hv <= -200)] = "low_gain"  # overrides "normal" if in this range
    # explicit narrower band for normal: HV between -1010 and -900 V
    mask_norm = (hv <= -900) & (hv > -1100)
    out[mask_norm] = "normal"
    return out


def main():
    df = load_per_sec()
    hv_wide = load_hv_wide()
    df_date_compact = df["date"].astype(str).str.replace("-", "", regex=False).values
    keys = pd.MultiIndex.from_arrays([df_date_compact, df["met_sec"].values],
                                    names=["date", "met_sec"])
    hv_arr = hv_wide.reindex(keys).values
    rows = np.arange(len(df))
    df["hv"] = hv_arr[rows, df["det_global"].values.astype(int)]
    df = df[df["hv"].notna()].copy()
    df["mode"] = classify(df["hv"].values)
    print(df["mode"].value_counts())

    fig, axes = plt.subplots(3, 3, figsize=(15, 14), sharex=True, sharey=True)
    last_hb = None
    print(f"\n{'Box':>4s} {'Mode':>10s} {'N':>10s} {'b[cnt/s]':>10s} {'α':>8s} {'RMS':>7s}")
    for row_i, box in enumerate("ABC"):
        for col_i, mode in enumerate(MODES):
            ax = axes[row_i, col_i]
            m = df[(df["box"] == box) & (df["mode"] == mode)]
            n = len(m)
            if n < 100:
                ax.text(0.5, 0.5, f"N = {n}\n(too few)", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="gray")
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_xlim(SCI_MIN, SCI_MAX); ax.set_ylim(Y_MIN, Y_MAX)
                if row_i == 0:
                    ax.set_title(f"{mode}  (N={n})", fontsize=10)
                if col_i == 0:
                    ax.set_ylabel(f"Box {box}\nSci predicted [cnt/s/det]")
                continue

            sci = m["sci_rate"].values
            nb = m["nb_rate"].values
            pho_corr = m["pho_corr_rate"].values
            X = np.column_stack([np.ones(n), sci])
            coef, *_ = np.linalg.lstsq(X, nb, rcond=None)
            b, alpha = coef
            sci_pred = (pho_corr - b) / (1 + alpha)
            rms = float(np.sqrt(np.mean((sci - sci_pred) ** 2)))
            print(f"  {box:>2s}  {mode:>10s} {n:>10,d} {b:>10.1f} {alpha:>8.4f} {rms:>7.1f}")

            sp_pos = np.maximum(sci_pred, Y_MIN * 0.5)
            keep = (sci >= SCI_MIN) & (sci <= SCI_MAX) & (sp_pos <= Y_MAX)
            hb = ax.hexbin(sci[keep], sp_pos[keep],
                           gridsize=120, xscale="log", yscale="log",
                           extent=(np.log10(SCI_MIN), np.log10(SCI_MAX),
                                   np.log10(Y_MIN), np.log10(Y_MAX)),
                           cmap="viridis", norm=LogNorm(vmin=1), mincnt=1,
                           rasterized=True)
            last_hb = hb

            # binned median (≥100 pts since per-panel data is smaller)
            bins = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
            bc = 0.5 * (bins[:-1] + bins[1:])
            med = []
            for i in range(len(bins) - 1):
                mb = (sci >= bins[i]) & (sci < bins[i + 1])
                med.append(np.median(sci_pred[mb]) if mb.sum() > 100 else np.nan)
            med = np.array(med)
            ax.plot(bc, med, "-", color="orange", lw=1.8, zorder=5)
            ax.plot([SCI_MIN, SCI_MAX], [SCI_MIN, SCI_MAX], "r--", lw=1.2, zorder=6)

            ax.set_xlim(SCI_MIN, SCI_MAX); ax.set_ylim(Y_MIN, Y_MAX)
            if row_i == 0:
                ax.set_title(f"{mode}  (N={n:,})\nb={b:.0f}, α={alpha:.3f}, RMS={rms:.0f}",
                             fontsize=10)
            else:
                ax.set_title(f"N={n:,}  b={b:.0f}, α={alpha:.3f}, RMS={rms:.0f}",
                             fontsize=9)
            if col_i == 0:
                ax.set_ylabel(f"Box {box}\nSci predicted [cnt/s/det]")
            if row_i == 2:
                ax.set_xlabel("Sci observed [cnt/s/det]")
            ax.grid(alpha=0.3, which="both")

    if last_hb is not None:
        fig.subplots_adjust(right=0.92)
        cax = fig.add_axes([0.93, 0.10, 0.015, 0.78])
        cb = fig.colorbar(last_hb, cax=cax)
        cb.set_label("per-det-sec bin count (log)")

    fig.suptitle(r"Sci$_\mathrm{pred}$ vs Sci$_\mathrm{obs}$, per HV mode  "
                 r"(normal: $-1000$ V, low_gain: $-845$ V, off: $\sim$0 V)",
                 fontsize=12, y=0.995)
    out = OUT_DIR / "n_below_by_hv_hexbin.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
