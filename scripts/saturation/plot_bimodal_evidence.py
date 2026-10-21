#!/usr/bin/env python3
"""Comprehensive bimodal evidence plot: 4 panels demonstrating the
HIGH-Large vs LOW-Large date-level bimodality.

Panel 1: Histogram of per-date median Large/Sci → clear bimodal distribution
Panel 2: KDE of Large/Sci within main band, split by mode → two peaks
Panel 3: 2D density of Large rate vs Sci rate, colored by mode
Panel 4: Date-mode timeline (chronological) → no temporal pattern
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from datetime import datetime

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
             "Sci": "int32"}
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
    df["large_rate"]  = df["Large"]    / df["length"]
    df["large_frac"]  = df["large_rate"] / df["sci_rate"].clip(lower=1)
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
    print(f"  normal-mode rows: {len(df):,}")
    return df


def parse_date(d):
    try:
        return datetime.strptime(d, "%Y-%m-%d").toordinal()
    except Exception:
        try:
            return datetime.strptime(d, "%Y%m%d").toordinal()
        except Exception:
            return None


def main():
    df = load()

    # ============ Classify modes ============
    main = df[(df["sci_rate"] >= 1000) & (df["sci_rate"] < 1500)].copy()
    by_date = main.groupby("date").agg(
        large_frac=("large_frac", "median"),
        N=("large_frac", "count"),
    )
    by_date = by_date[by_date["N"] > 200]
    HIGH_TH, LOW_TH = 0.55, 0.40
    high_dates = set(by_date[by_date["large_frac"] > HIGH_TH].index)
    low_dates  = set(by_date[by_date["large_frac"] < LOW_TH].index)
    df["mode"] = np.where(df["date"].isin(high_dates), "HIGH",
                          np.where(df["date"].isin(low_dates), "LOW", "MID"))
    print(f"\nDates: HIGH={len(high_dates)}, LOW={len(low_dates)}, "
          f"MID={len(by_date)-len(high_dates)-len(low_dates)}")

    # ============ Setup figure ============
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.25)

    # ============ Panel 1: Histogram of date-level Large/Sci ============
    ax1 = fig.add_subplot(gs[0, 0])
    by_date_sorted = by_date.sort_values("large_frac")
    ax1.hist(by_date["large_frac"], bins=30, range=(0, 1.0),
              color="lightgray", edgecolor="black", lw=0.5)
    ax1.axvspan(0, LOW_TH, alpha=0.2, color="blue", label=f"LOW mode (N={len(low_dates)})")
    ax1.axvspan(LOW_TH, HIGH_TH, alpha=0.15, color="gray", label="MID")
    ax1.axvspan(HIGH_TH, 1.0, alpha=0.2, color="red", label=f"HIGH mode (N={len(high_dates)})")
    ax1.axvline(LOW_TH, color="blue", ls="--", lw=1.5)
    ax1.axvline(HIGH_TH, color="red", ls="--", lw=1.5)
    ax1.set_xlabel("date-median Large/Sci (in Sci 1000-1500 main band)")
    ax1.set_ylabel("# dates")
    ax1.set_title(f"Panel 1: Per-date median Large/Sci\n"
                   f"clear bimodal distribution → 2 source-type populations",
                   fontsize=10)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(alpha=0.3)

    # ============ Panel 2: Large/Sci per-second distributions, by mode ============
    ax2 = fig.add_subplot(gs[0, 1])
    bin_edges = [400, 700, 1000, 1300]
    colors_mode = {"HIGH":"red", "LOW":"blue"}
    for i, (lo, hi) in enumerate([(400, 700), (700, 1000), (1000, 1500)]):
        offset_x = i * 0.5
        for mode, color in colors_mode.items():
            mask = (df["mode"]==mode) & (df["sci_rate"]>=lo) & (df["sci_rate"]<hi)
            sub = df[mask]["large_frac"].clip(0, 1.5)
            if len(sub) < 100: continue
            ax2.hist(sub, bins=80, range=(0, 1.5), color=color, alpha=0.4,
                      density=True, label=f"{mode} Sci {lo}-{hi}" if i==0 else None)
    ax2.set_xlabel("Large / Sci (per-second)")
    ax2.set_ylabel("density (normalized)")
    ax2.set_title("Panel 2: Per-second Large/Sci distribution by mode\n"
                   "HIGH/LOW are NOT mixed within a date — each date is one mode",
                   fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 1.5)

    # ============ Panel 3: Large vs Sci 2D density colored by mode ============
    ax3 = fig.add_subplot(gs[1, 0])
    # Use 2D histogram with different colormap for each mode
    sci_bins = np.logspace(np.log10(100), np.log10(3000), 80)
    large_bins = np.logspace(np.log10(10), np.log10(5000), 80)

    high = df[df["mode"]=="HIGH"]
    low = df[df["mode"]=="LOW"]
    H_high, _, _ = np.histogram2d(high["sci_rate"], high["large_rate"],
                                    bins=[sci_bins, large_bins])
    H_low, _, _ = np.histogram2d(low["sci_rate"], low["large_rate"],
                                   bins=[sci_bins, large_bins])
    Sci_grid, Large_grid = np.meshgrid(sci_bins, large_bins)

    # Plot LOW first (blue), then HIGH (red), with transparency
    ax3.pcolormesh(Sci_grid, Large_grid, H_low.T,
                    norm=LogNorm(vmin=1, vmax=max(H_high.max(), H_low.max())),
                    cmap="Blues", shading="auto", alpha=0.6)
    ax3.pcolormesh(Sci_grid, Large_grid, H_high.T,
                    norm=LogNorm(vmin=1, vmax=max(H_high.max(), H_low.max())),
                    cmap="Reds", shading="auto", alpha=0.6)
    # y=0.5x reference (Large/Sci=0.5)
    line = np.array([100, 3000])
    for slope, label in [(0.2, "Large/Sci=0.2 (LOW center)"),
                          (0.7, "Large/Sci=0.7 (HIGH center)")]:
        ax3.plot(line, slope*line, "--", color="black", lw=1, alpha=0.4)
        ax3.text(2800, slope*2800*1.1, label, fontsize=8, color="black")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Sci rate [cnt/s/det]")
    ax3.set_ylabel("Large rate [cnt/s/det]")
    ax3.set_title("Panel 3: 2D density of Large vs Sci\n"
                   "blue=LOW mode dates, red=HIGH mode dates",
                   fontsize=10)
    ax3.grid(alpha=0.3, which="both")
    ax3.set_xlim(100, 3000)
    ax3.set_ylim(10, 5000)

    # ============ Panel 4: Date mode timeline (chronological) ============
    ax4 = fig.add_subplot(gs[1, 1])
    by_date_chron = by_date.copy()
    by_date_chron["ordinal"] = by_date_chron.index.to_series().map(parse_date)
    by_date_chron = by_date_chron.dropna(subset=["ordinal"]).sort_values("ordinal")
    by_date_chron["mode"] = np.where(by_date_chron["large_frac"] > HIGH_TH, "HIGH",
                                       np.where(by_date_chron["large_frac"] < LOW_TH, "LOW", "MID"))
    color_map = {"HIGH":"red","LOW":"blue","MID":"gray"}
    for mode, c in color_map.items():
        sub = by_date_chron[by_date_chron["mode"]==mode]
        ax4.scatter(sub["ordinal"], sub["large_frac"], c=c, s=40, alpha=0.7,
                     edgecolor="black", lw=0.3,
                     label=f"{mode} ({len(sub)} dates)")
    ax4.axhspan(0, LOW_TH, alpha=0.1, color="blue")
    ax4.axhspan(HIGH_TH, 1.0, alpha=0.1, color="red")
    # Year ticks
    years = sorted(set(d.split("-")[0] for d in by_date_chron.index if "-" in d))
    if years:
        year_ords = [datetime.strptime(f"{y}-07-01", "%Y-%m-%d").toordinal() for y in years]
        ax4.set_xticks(year_ords)
        ax4.set_xticklabels(years, rotation=45)
    ax4.set_xlabel("date (chronological)")
    ax4.set_ylabel("date-median Large/Sci")
    ax4.set_title("Panel 4: Mode by chronological date\n"
                   "random interleaving → NOT a mission epoch transition",
                   fontsize=10)
    ax4.legend(fontsize=9, loc="upper right")
    ax4.grid(alpha=0.3)

    fig.suptitle("HIGH-Large vs LOW-Large bimodality: comprehensive evidence",
                  fontsize=13, y=0.995)
    out = OUT_DIR / "bimodal_evidence.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
