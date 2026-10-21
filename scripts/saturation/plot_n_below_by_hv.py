#!/usr/bin/env python3
"""Join HV table (from server) with per_sec CSVs, then plot
Sci_pred vs Sci colored by HV mode.

HV modes discovered from server-side histogram (2.3M seconds sampled):
  -1000 V (±10)  ~80%   Normal gain
   -845 V (±5)   ~8%    Low gain ⭐
   -5 V          ~12%   HV OFF (SAA protection)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table.csv.gz")  # downloaded from server
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)

L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BETA, GAMMA = 2.0, 1.2

# Box → PMT offset (det column 0-5 in CSV maps to hv_offset + det_local)
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}


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
        if (i+1) % 300 == 0:
            print(f"  {i+1}/{len(files)}")
    df = pd.concat(parts, ignore_index=True)
    df["length"] = df["L_cycles"].astype("float32") * 16e-6
    df = df[df["L_cycles"] > L_THRESH].copy()
    g = df.groupby(["date","box","met_sec"], observed=True)["Sci"].sum()
    g.name = "sci_sec_total"
    df = df.merge(g, on=["date","box","met_sec"])
    df = df[df["sci_sec_total"] > SCI_SEC_TOTAL_MIN]
    df["sci_rate"] = df["Sci"] / df["length"]
    df["pho_corr_rate"] = (df["PHO"] - BETA*df["Wide"] - GAMMA*df["Large"]) / df["length"]
    df["nb_rate"] = df["pho_corr_rate"] - df["sci_rate"]
    df["box_offset"] = df["box"].map(BOX_OFFSET).astype("int8")
    df["det_global"] = df["box_offset"] + df["det"]
    print(f"  filtered rows: {len(df):,}")
    return df


def load_hv_wide():
    """HV wide table — keep as wide; we'll index-lookup, no melt."""
    print(f"Loading HV table {HV_TABLE}...")
    hv = pd.read_csv(HV_TABLE,
                     dtype={"date": "string", "met_sec": "int64",
                            **{f"hv{i}": "float32" for i in range(18)}})
    print(f"  wide rows: {len(hv):,}")
    # Set index for fast lookup
    hv = hv.set_index(["date", "met_sec"]).sort_index()
    return hv


def main():
    df = load_per_sec()
    hv_wide = load_hv_wide()

    # Reindex HV to df's (date, met_sec) → 14M rows × 18 hv cols (NaN where missing)
    # per_sec CSV uses "2017-06-22" format; HV table uses "20170622" — strip dashes.
    print("Aligning HV to df by (date, met_sec)...")
    df_date_compact = df["date"].astype(str).str.replace("-", "", regex=False).values
    keys = pd.MultiIndex.from_arrays(
        [df_date_compact, df["met_sec"].values],
        names=["date", "met_sec"])
    hv_arr = hv_wide.reindex(keys).values  # shape (14M, 18), float32 with NaN
    # Pick the right column per row using det_global
    print("Looking up per-detector HV...")
    rows = np.arange(len(df))
    df["hv"] = hv_arr[rows, df["det_global"].values.astype(int)]

    n_with_hv = df["hv"].notna().sum()
    print(f"  rows with HV: {n_with_hv:,}/{len(df):,} ({n_with_hv/len(df)*100:.1f}%)")
    df = df[df["hv"].notna()].copy()

    # Classify mode
    def mode_of(hv):
        if hv > -200:
            return "off"
        if hv > -900:
            return "low_gain"
        return "normal"
    df["mode"] = df["hv"].apply(mode_of)
    print(df["mode"].value_counts())

    # Per-mode fits + scatter
    SCI_MIN, SCI_MAX = 40.0, 5000.0
    Y_MIN, Y_MAX = 1.0, 5000.0

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 17), sharex=True, sharey=True)
    mode_colors = {"normal": "#1f77b4", "low_gain": "#d62728", "off": "#7f7f7f"}

    for ax, box in zip(axes, "ABC"):
        sub = df[df["box"] == box]
        # Plot each mode separately
        for mode in ["normal", "low_gain", "off"]:
            m = sub[sub["mode"] == mode]
            if len(m) == 0:
                continue
            sci = m["sci_rate"].values
            nb = m["nb_rate"].values
            pho_corr = m["pho_corr_rate"].values
            n = len(m)

            # Fit (b, α) for this mode separately
            X = np.column_stack([np.ones(n), sci])
            coef, *_ = np.linalg.lstsq(X, nb, rcond=None)
            b, alpha = coef
            sci_pred = (pho_corr - b) / (1 + alpha)
            rms = float(np.sqrt(np.mean((sci - sci_pred) ** 2)))

            sp_pos = np.maximum(sci_pred, Y_MIN * 0.5)
            keep = (sci >= SCI_MIN) & (sci <= SCI_MAX) & (sp_pos <= Y_MAX)
            ax.scatter(sci[keep], sp_pos[keep], s=1.5, alpha=0.15,
                       color=mode_colors[mode], rasterized=True, linewidths=0,
                       label=f"{mode}  N={n:,}  b={b:.0f}, α={alpha:.3f}, RMS={rms:.0f}")
            print(f"  Box {box} / {mode:>8s}:  N={n:>9,d}  b={b:>9.1f}  "
                  f"α={alpha:>7.4f}  RMS={rms:>6.1f}")

        ax.plot([SCI_MIN, SCI_MAX], [SCI_MIN, SCI_MAX], "k--", lw=1.5, alpha=0.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(SCI_MIN, SCI_MAX); ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xlabel("Sci observed [cnt/s/det]")
        ax.set_ylabel("Sci predicted [cnt/s/det]")
        ax.set_title(f"Box {box}  (separated by PMT HV)")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.95, markerscale=4)
        ax.grid(alpha=0.3, which="both")

    fig.suptitle("Sci_pred vs Sci, split by HV mode "
                 r"(normal ≈ −1000 V, low_gain ≈ −845 V, off ≈ 0 V)", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "n_below_by_hv_mode.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
