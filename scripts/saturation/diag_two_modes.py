#!/usr/bin/env python3
"""Compare 'high Large/Sci' vs 'low Large/Sci' date modes in detail.

Goal: find what physically distinguishes them.

Check:
  1. Time distribution: when do dates switch? Alternating? Epoch-based?
  2. Rate ratios: PHO/Sci, Wide/Sci, ACD1/Sci, ACDN/Sci, Dt/L
       — if LOW GAIN: expect lower PHO, higher Wide, lower Large
       — if SOURCE-driven: expect different spectral shape
  3. HV fine structure: any HV difference even within [-1100, -900]?
  4. Per-det rates: which dets respond differently?
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    df_pre_hv = df.copy()  # Before HV filter
    df = df[(df["hv"] < -900) & (df["hv"] > -1100)].copy()
    print(f"normal-mode rows (after HV filter): {len(df):,}")
    print(f"pre-HV-filter rows: {len(df_pre_hv):,}")
    return df, df_pre_hv


def parse_date(d):
    """Parse 'YYYY-MM-DD' to ordinal day."""
    try:
        return datetime.strptime(d, "%Y-%m-%d").toordinal()
    except Exception:
        try:
            return datetime.strptime(d, "%Y%m%d").toordinal()
        except Exception:
            return None


def main():
    df, df_pre = load()

    # ========== 1. Classify dates by Large/Sci in main band ==========
    main = df[(df["sci_rate"] >= 1000) & (df["sci_rate"] < 1500)].copy()
    by_date = main.groupby("date").agg(
        large_frac=("large_rate", lambda x: (x / main.loc[x.index, "sci_rate"]).median()),
        N=("large_rate", "count"),
    )
    by_date = by_date[by_date["N"] > 200]
    print(f"\nDates with > 200 rows in Sci 1000-1500: {len(by_date)}")

    HIGH_TH, LOW_TH = 0.55, 0.40
    high_dates = set(by_date[by_date["large_frac"] > HIGH_TH].index)
    low_dates  = set(by_date[by_date["large_frac"] < LOW_TH].index)
    mid_dates  = set(by_date[(by_date["large_frac"] >= LOW_TH)
                              & (by_date["large_frac"] <= HIGH_TH)].index)
    print(f"  HIGH mode dates (Large/Sci > {HIGH_TH}): {len(high_dates)}")
    print(f"  LOW  mode dates (Large/Sci < {LOW_TH}): {len(low_dates)}")
    print(f"  MID  mode dates                        : {len(mid_dates)}")

    df["mode"] = "MID"
    df.loc[df["date"].isin(high_dates), "mode"] = "HIGH"
    df.loc[df["date"].isin(low_dates), "mode"] = "LOW"

    # ========== 2. Time distribution ==========
    print(f"\n=== Time distribution ===")
    by_date["ordinal"] = by_date.index.to_series().map(parse_date)
    by_date["year"] = by_date.index.to_series().apply(lambda d: d.split("-")[0] if "-" in d else d[:4])
    print(f"  Date range: {by_date['ordinal'].min()} to {by_date['ordinal'].max()}")
    print(f"  Years covered: {sorted(by_date['year'].unique())}")
    print(f"\n  Year × Mode breakdown:")
    by_date["mode"] = "MID"
    by_date.loc[by_date["large_frac"] > HIGH_TH, "mode"] = "HIGH"
    by_date.loc[by_date["large_frac"] < LOW_TH, "mode"] = "LOW"
    pivot = pd.crosstab(by_date["year"], by_date["mode"])
    print(pivot)

    # ========== 3. Rate ratios: HIGH vs LOW ==========
    print(f"\n=== Rate ratio comparison (Sci 1000-1500 main band) ===")
    print(f"{'Stat':>20s}  {'HIGH mode':>15s}  {'LOW mode':>15s}  {'ratio H/L':>12s}")
    main_h = main[main["date"].isin(high_dates)]
    main_l = main[main["date"].isin(low_dates)]
    stats = [
        ("N rows",            len(main_h),                          len(main_l)),
        ("median Sci",        main_h["sci_rate"].median(),          main_l["sci_rate"].median()),
        ("median PHO",        main_h["pho_rate"].median(),          main_l["pho_rate"].median()),
        ("median Wide",       main_h["wide_rate"].median(),         main_l["wide_rate"].median()),
        ("median Large",      main_h["large_rate"].median(),        main_l["large_rate"].median()),
        ("median Sci_ACD1",   main_h["acd1_rate"].median(),         main_l["acd1_rate"].median()),
        ("median Sci_ACDN",   main_h["acdn_rate"].median(),         main_l["acdn_rate"].median()),
        ("median HV",         main_h["hv"].median(),                main_l["hv"].median()),
        ("PHO/Sci",           (main_h["pho_rate"]/main_h["sci_rate"]).median(),
                              (main_l["pho_rate"]/main_l["sci_rate"]).median()),
        ("Wide/Sci",          (main_h["wide_rate"]/main_h["sci_rate"]).median(),
                              (main_l["wide_rate"]/main_l["sci_rate"]).median()),
        ("Large/Sci",         (main_h["large_rate"]/main_h["sci_rate"]).median(),
                              (main_l["large_rate"]/main_l["sci_rate"]).median()),
        ("ACD1/Sci",          (main_h["acd1_rate"]/main_h["sci_rate"]).median(),
                              (main_l["acd1_rate"]/main_l["sci_rate"]).median()),
        ("ACDN/Sci",          (main_h["acdn_rate"]/main_h["sci_rate"]).median(),
                              (main_l["acdn_rate"]/main_l["sci_rate"]).median()),
    ]
    for stat_name, h, l in stats:
        if isinstance(h, int):
            print(f"  {stat_name:>20s}  {h:>15,d}  {l:>15,d}  {'':>12s}")
        else:
            ratio = h / l if l != 0 else float("inf")
            print(f"  {stat_name:>20s}  {h:>15.3f}  {l:>15.3f}  {ratio:>12.3f}")

    # ========== 4. HV distributions in two modes ==========
    print(f"\n=== HV distribution per mode ===")
    print(f"  HIGH mode HV:  min={main_h['hv'].min():.2f}, "
          f"5%={main_h['hv'].quantile(0.05):.2f}, "
          f"50%={main_h['hv'].quantile(0.50):.2f}, "
          f"95%={main_h['hv'].quantile(0.95):.2f}, "
          f"max={main_h['hv'].max():.2f}")
    print(f"  LOW  mode HV:  min={main_l['hv'].min():.2f}, "
          f"5%={main_l['hv'].quantile(0.05):.2f}, "
          f"50%={main_l['hv'].quantile(0.50):.2f}, "
          f"95%={main_l['hv'].quantile(0.95):.2f}, "
          f"max={main_l['hv'].max():.2f}")

    # ========== 5. Plot: comprehensive view ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel A: Large/Sci by date (chronological with real time axis)
    ax = axes[0, 0]
    by_date_sorted = by_date.dropna(subset=["ordinal"]).sort_values("ordinal")
    colors = {"HIGH": "red", "LOW": "blue", "MID": "gray"}
    for mode, color in colors.items():
        sub = by_date_sorted[by_date_sorted["mode"] == mode]
        ax.scatter(sub["ordinal"], sub["large_frac"], c=color, s=40,
                   alpha=0.7, label=f"{mode} mode ({len(sub)} dates)",
                   edgecolor="black", lw=0.3)
    ax.set_xlabel("date (ordinal day)")
    ax.set_ylabel("median Large/Sci (Sci 1000-1500)")
    ax.set_title("Mode by real-time chronology\n"
                 "(if mode flips by epoch: clusters; if random: scattered)")
    ax.axhline(HIGH_TH, color="red", ls=":", lw=1, alpha=0.5)
    ax.axhline(LOW_TH, color="blue", ls=":", lw=1, alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Annotate year labels on x-axis
    years = sorted(set(d.split("-")[0] for d in by_date.index if "-" in d))
    year_ords = [datetime.strptime(f"{y}-07-01", "%Y-%m-%d").toordinal() for y in years]
    ax.set_xticks(year_ords)
    ax.set_xticklabels(years, rotation=45)

    # Panel B: PHO/Sci distribution in each mode
    ax = axes[0, 1]
    for mode, color in [("HIGH", "red"), ("LOW", "blue")]:
        dates = high_dates if mode == "HIGH" else low_dates
        sub = main[main["date"].isin(dates)]
        pho_sci = (sub["pho_rate"] / sub["sci_rate"]).clip(0, 5)
        ax.hist(pho_sci, bins=80, range=(0, 5), color=color, alpha=0.5,
                label=f"{mode} mode (N={len(sub):,})", density=True)
    ax.set_xlabel("PHO / Sci")
    ax.set_ylabel("density")
    ax.set_title("PHO/Sci distribution by mode")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel C: Wide/Sci distribution by mode
    ax = axes[0, 2]
    for mode, color in [("HIGH", "red"), ("LOW", "blue")]:
        dates = high_dates if mode == "HIGH" else low_dates
        sub = main[main["date"].isin(dates)]
        wide_sci = (sub["wide_rate"] / sub["sci_rate"]).clip(0, 0.5)
        ax.hist(wide_sci, bins=80, range=(0, 0.5), color=color, alpha=0.5,
                label=f"{mode} mode", density=True)
    ax.set_xlabel("Wide / Sci")
    ax.set_ylabel("density")
    ax.set_title("Wide/Sci distribution by mode")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel D: Large/Sci vs Sci, separated by mode
    ax = axes[1, 0]
    sci_bins = np.logspace(np.log10(200), np.log10(3000), 35)
    bc = 0.5 * (sci_bins[:-1] + sci_bins[1:])
    for mode, color in [("HIGH", "red"), ("LOW", "blue")]:
        dates = high_dates if mode == "HIGH" else low_dates
        sub = df[df["date"].isin(dates)]
        med = np.array([
            (sub.loc[(sub["sci_rate"] >= sci_bins[i]) & (sub["sci_rate"] < sci_bins[i+1]),
                     "large_rate"] /
             sub.loc[(sub["sci_rate"] >= sci_bins[i]) & (sub["sci_rate"] < sci_bins[i+1]),
                     "sci_rate"]).median()
            for i in range(len(sci_bins) - 1)
        ])
        ax.plot(bc, med, "-", color=color, lw=2,
                label=f"{mode} mode ({len(dates)} dates)")
    ax.set_xscale("log")
    ax.set_xlabel("Sci [cnt/s/det]")
    ax.set_ylabel("Large/Sci median")
    ax.set_title("Large/Sci vs Sci, split by mode")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    # Panel E: PHO/Sci vs Sci, by mode
    ax = axes[1, 1]
    for mode, color in [("HIGH", "red"), ("LOW", "blue")]:
        dates = high_dates if mode == "HIGH" else low_dates
        sub = df[df["date"].isin(dates)]
        med = np.array([
            (sub.loc[(sub["sci_rate"] >= sci_bins[i]) & (sub["sci_rate"] < sci_bins[i+1]),
                     "pho_rate"] /
             sub.loc[(sub["sci_rate"] >= sci_bins[i]) & (sub["sci_rate"] < sci_bins[i+1]),
                     "sci_rate"]).median()
            for i in range(len(sci_bins) - 1)
        ])
        ax.plot(bc, med, "-", color=color, lw=2,
                label=f"{mode} mode")
    ax.set_xscale("log")
    ax.set_xlabel("Sci [cnt/s/det]")
    ax.set_ylabel("PHO/Sci median")
    ax.set_title("PHO/Sci vs Sci, split by mode")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    # Panel F: HV distribution by mode
    ax = axes[1, 2]
    for mode, color in [("HIGH", "red"), ("LOW", "blue")]:
        dates = high_dates if mode == "HIGH" else low_dates
        sub = main[main["date"].isin(dates)]
        ax.hist(sub["hv"], bins=40, range=(-1100, -900), color=color, alpha=0.5,
                label=f"{mode} mode median={sub['hv'].median():.1f}V", density=True)
    ax.set_xlabel("HV [V]")
    ax.set_ylabel("density")
    ax.set_title("HV distribution by mode\n"
                 "(if low-gain mode: should show HV difference)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "diag_two_modes.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
