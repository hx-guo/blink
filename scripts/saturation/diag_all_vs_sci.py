#!/usr/bin/env python3
"""All channels vs Sci_obs — visualizing the saturation everywhere.

Insight: at high group_rate, the PDAU FIFO saturates. Sci_obs itself is no
longer a clean independent variable — it's already affected by losses. All
OBSERVED rates (Sci, PHO, Wide, Large) hit a ceiling together. This is why
no PHO = f(Sci, Wide, Large) regression can work at high Sci — it's fitting
saturated-vs-saturated data.

Plot: median values of every channel vs Sci_obs.

Panel 1 (rates):
  Left axis:  median PHO, Wide, Large, group_rate (all cnt/s/det units)
  Right axis: median Sci_others = group_rate - sci_rate

Panel 2 (fractions on twinx):
  Left axis:  PHO/Sci, Wide/Sci, Large/Sci, Dt/length (dead-time fraction)
  Right axis: ACD1/Sci, ACDN/Sci

If Sci is saturating, we expect at high Sci:
  - PHO/Sci ratio drops (more Wide, more dead time)
  - Wide/Sci ratio drops too eventually
  - Dt/length grows toward 1
  - group_rate / (6 × Sci) plateaus (per-det Sci flattens)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table_partial.csv.gz")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}


def load():
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32",
             "Dt": "int32", "Sci": "int32",
             "Sci_ACD1": "int32", "Sci_ACDN": "int32"}
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

    df["sci_rate"]    = df["Sci"]      / df["length"]
    df["wide_rate"]   = df["Wide"]     / df["length"]
    df["large_rate"]  = df["Large"]    / df["length"]
    df["pho_rate"]    = df["PHO"]      / df["length"]
    df["acd1_rate"]   = df["Sci_ACD1"] / df["length"]
    df["acdn_rate"]   = df["Sci_ACDN"] / df["length"]
    df["group_rate"]  = df["sci_sec_total"] / df["length"]
    df["other_rate"]  = df["group_rate"] - df["sci_rate"]
    df["dt_frac"]     = df["Dt"] / df["L_cycles"]
    df["det_global"]  = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")

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


def median_per_bin(x, y, bins, min_count=50):
    med = np.full(len(bins) - 1, np.nan)
    cnt = np.zeros(len(bins) - 1, dtype=int)
    for i in range(len(bins) - 1):
        m = (x >= bins[i]) & (x < bins[i+1])
        cnt[i] = m.sum()
        if m.sum() > min_count:
            med[i] = np.median(y[m])
    return med, cnt


def main():
    df = load()

    # Bin by per-det Sci (log)
    SCI_MIN, SCI_MAX = 100, 4500
    bins = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
    bc = 0.5 * (bins[:-1] + bins[1:])

    # FIFO saturation rate from M11d fit: G_crit*G_NORM = 15100 cnt/s/box → per-det = 2517
    SCI_FIFO_LIMIT = 15100 / 6.0  # ~2517 cnt/s/det (if dets balanced)

    # Diagnose count distribution
    cnt_sci, _ = median_per_bin(df["sci_rate"].values, df["sci_rate"].values, bins, min_count=0)
    cnt_sci = np.array([((df["sci_rate"] >= bins[i]) & (df["sci_rate"] < bins[i+1])).sum()
                        for i in range(len(bins)-1)])
    print(f"  per-Sci-bin row counts (log-spaced 40 bins):")
    print(f"    Sci 200:  {cnt_sci[np.argmin(np.abs(bc-200))]} rows")
    print(f"    Sci 500:  {cnt_sci[np.argmin(np.abs(bc-500))]} rows")
    print(f"    Sci 1000: {cnt_sci[np.argmin(np.abs(bc-1000))]} rows")
    print(f"    Sci 1500: {cnt_sci[np.argmin(np.abs(bc-1500))]} rows")
    print(f"    Sci 2000: {cnt_sci[np.argmin(np.abs(bc-2000))]} rows")
    print(f"    Sci 2500: {cnt_sci[np.argmin(np.abs(bc-2500))]} rows")
    print(f"    Sci 3000: {cnt_sci[np.argmin(np.abs(bc-3000))]} rows")
    print(f"    Sci 3500: {cnt_sci[np.argmin(np.abs(bc-3500))]} rows")

    # ============ Pool all boxes for cleaner curve ============
    print("\n=== Computing median curves per channel ===")
    metrics_rate = {
        "Sci (= x-axis identity)": ("sci_rate", "k"),
        "PHO": ("pho_rate", "C0"),
        "Wide": ("wide_rate", "C1"),
        "Large × 10": ("large_rate", "C2"),  # scaled up for visibility
        "group_rate / 6": ("group_rate", "C3"),
        "other_rate / 5": ("other_rate", "C4"),
        "Sci_ACD1": ("acd1_rate", "C5"),
        "Sci_ACDN": ("acdn_rate", "C6"),
    }

    medians_rate = {}
    for name, (col, color) in metrics_rate.items():
        med, _ = median_per_bin(df["sci_rate"].values, df[col].values, bins, min_count=50)
        # Apply scaling for visualization
        if "Large × 10" in name:
            med = med * 10
        elif "/ 6" in name:
            med = med / 6.0
        elif "/ 5" in name:
            med = med / 5.0
        medians_rate[name] = (med, color)

    metrics_frac = {
        "PHO / Sci": ("pho_rate", "sci_rate", "C0"),
        "Wide / Sci": ("wide_rate", "sci_rate", "C1"),
        "Large / Sci": ("large_rate", "sci_rate", "C2"),
        "Dt / length": ("dt_frac", None, "C5"),
    }
    medians_frac = {}
    for name, (col_y, col_x, color) in metrics_frac.items():
        if col_x is None:
            med, _ = median_per_bin(df["sci_rate"].values, df[col_y].values, bins, min_count=50)
        else:
            ratio = df[col_y].values / np.maximum(df[col_x].values, 1)
            med, _ = median_per_bin(df["sci_rate"].values, ratio, bins, min_count=50)
        medians_frac[name] = (med, color)

    metrics_acd = {
        "ACD1 / Sci": ("acd1_rate", "sci_rate", "C7"),
        "ACDN / Sci": ("acdn_rate", "sci_rate", "C8"),
    }
    medians_acd = {}
    for name, (col_y, col_x, color) in metrics_acd.items():
        ratio = df[col_y].values / np.maximum(df[col_x].values, 1)
        med, _ = median_per_bin(df["sci_rate"].values, ratio, bins, min_count=50)
        medians_acd[name] = (med, color)

    # ============ Mega plot: 2 rows × 2 cols (or 4×1) ============
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 1: All rates (no scaling messy)
    ax = axes[0, 0]
    for name, (med, color) in medians_rate.items():
        ls = "--" if "Sci (=" in name else "-"
        ax.plot(bc, med, ls, color=color, lw=2, label=name, alpha=0.85)
    # 1:1 line for reference
    ax.plot(bc, bc, ":", color="gray", lw=1, alpha=0.5, label="y=x reference")
    ax.axvline(SCI_FIFO_LIMIT, color="red", ls="--", lw=1.5, alpha=0.7,
               label=f"FIFO limit (Sci~{SCI_FIFO_LIMIT:.0f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(SCI_MIN, SCI_MAX)
    ax.set_ylim(1, 30000)
    ax.set_xlabel("Sci observed [cnt/s/det]")
    ax.set_ylabel("median rate [cnt/s/det]")
    ax.set_title("Panel 1: All rate channels vs Sci_obs\n"
                 "(if FIFO saturates, all rates plateau together)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    # Panel 2: Fractions (twinx for ACD)
    ax = axes[0, 1]
    for name, (med, color) in medians_frac.items():
        ls = "-"
        ax.plot(bc, med, ls, color=color, lw=2, label=name)
    ax.axvline(SCI_FIFO_LIMIT, color="red", ls="--", lw=1.5, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlim(SCI_MIN, SCI_MAX)
    ax.set_ylim(0, 3.5)
    ax.set_xlabel("Sci observed [cnt/s/det]")
    ax.set_ylabel("ratio (left axis)", color="black")
    ax.set_title("Panel 2: Band fractions + dead-time, ACD fractions on twinx", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    # twinx for ACD
    ax_acd = ax.twinx()
    for name, (med, color) in medians_acd.items():
        ax_acd.plot(bc, med, "--", color=color, lw=2, label=name, alpha=0.85)
    ax_acd.set_ylabel("ACD fraction (right axis)", color="C7")
    ax_acd.set_ylim(0, 0.3)
    ax_acd.tick_params(axis="y", labelcolor="C7")
    ax_acd.legend(fontsize=9, loc="upper right")

    # Panel 3: row counts per Sci bin (shows where data exists)
    ax = axes[1, 0]
    cnt_per_bin = np.array([((df["sci_rate"] >= bins[i]) & (df["sci_rate"] < bins[i+1])).sum()
                            for i in range(len(bins)-1)])
    ax.bar(bc, cnt_per_bin, width=np.diff(bins), color="C0", alpha=0.6,
           edgecolor="navy", lw=0.5)
    ax.axvline(SCI_FIFO_LIMIT, color="red", ls="--", lw=1.5, alpha=0.7,
               label=f"FIFO limit Sci~{SCI_FIFO_LIMIT:.0f}")
    ax.axhline(50, color="orange", ls=":", lw=1.5, alpha=0.7,
               label="min_count threshold (50)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(SCI_MIN, SCI_MAX)
    ax.set_ylim(1, 1e6)
    ax.set_xlabel("Sci observed [cnt/s/det]")
    ax.set_ylabel("# rows per Sci bin")
    ax.set_title("Panel 3: Data density per Sci bin\n"
                 "(Sci > 2500: only ~470 rows total → curves break here)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    # Panel 4: dt_frac as direct saturation indicator
    ax = axes[1, 1]
    dt_med, _ = median_per_bin(df["sci_rate"].values, df["dt_frac"].values, bins, min_count=50)
    ax.plot(bc, dt_med, "-", color="C5", lw=2.5, label="Dt / L  (dead-time fraction)")
    # group_rate / something
    gr_med, _ = median_per_bin(df["sci_rate"].values, df["group_rate"].values, bins, min_count=50)
    ax.plot(bc, gr_med / 15100, "--", color="C3", lw=2, label="group_rate / G_crit")
    ax.axhline(1.0, color="red", ls=":", lw=1, alpha=0.5)
    ax.axvline(SCI_FIFO_LIMIT, color="red", ls="--", lw=1.5, alpha=0.7,
               label=f"FIFO limit Sci~{SCI_FIFO_LIMIT:.0f}")
    ax.set_xscale("log")
    ax.set_xlim(SCI_MIN, SCI_MAX)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("Sci observed [cnt/s/det]")
    ax.set_ylabel("fraction / normalized rate")
    ax.set_title("Panel 4: Saturation indicators\n"
                 "(Dt/L → 1 means PDAU spending all time dead)", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Saturation diagnosis: all channels vs Sci_observed\n"
                 "FIFO saturation around group_rate=15000 (Sci~2500/det) — "
                 "above this, Sci itself is no longer linear input",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "diag_all_vs_sci.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")

    # ============ Print numerical values at key Sci points ============
    print(f"\n=== Median values at key Sci points ===")
    sample_sci = [500, 1000, 1500, 2000, 2500, 3000, 4000]
    print(f"{'Sci':>6s}  " +
          "  ".join(f"{c:>8s}" for c in ["PHO", "Wide", "Large", "ACD1/S",
                                         "ACDN/S", "Dt/L", "grp/6",
                                         "PHO/S"]))
    for s in sample_sci:
        idx = np.argmin(np.abs(bc - s))
        if idx < len(bc) - 1:
            pho_v = medians_rate["PHO"][0][idx]
            wide_v = medians_rate["Wide"][0][idx]
            large_v = medians_rate["Large × 10"][0][idx] / 10  # unscale
            acd1_v = medians_acd["ACD1 / Sci"][0][idx]
            acdn_v = medians_acd["ACDN / Sci"][0][idx]
            dt_v = medians_frac["Dt / length"][0][idx]
            gr_v = medians_rate["group_rate / 6"][0][idx]
            pho_sci_v = medians_frac["PHO / Sci"][0][idx]
            print(f"  {s:>5d}  {pho_v:>8.0f}  {wide_v:>8.0f}  {large_v:>8.1f}  "
                  f"{acd1_v:>8.3f}  {acdn_v:>8.3f}  {dt_v:>8.3f}  "
                  f"{gr_v:>8.0f}  {pho_sci_v:>8.3f}")


if __name__ == "__main__":
    main()
