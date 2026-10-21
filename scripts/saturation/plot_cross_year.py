#!/usr/bin/env python3
"""Plot cross-year validation from year_summary_*.json files.

Three panels:
  1. per-det C_det vs year (18 lines)
  2. B(|mlat|=30°) and B(|mlat|=40°) vs year — solar cycle effect?
  3. Sample count + n_wraps fraction vs year

Output: plots/cross_year_validation.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summaries", nargs="+", required=True, help="year_summary_*.json files")
    p.add_argument("--out", default="plots/cross_year_validation.png")
    args = p.parse_args()

    # Load all summaries
    data = []
    for f in args.summaries:
        with open(f) as fh:
            s = json.load(fh)
            data.append(s)
    data.sort(key=lambda x: int(x["year"]))
    years = [int(d["year"]) for d in data]
    print(f"Loaded {len(data)} years: {years}")

    # Panel 1: C_det per (box, det) vs year
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax = axes[0, 0]
    colors = plt.cm.tab20(np.linspace(0, 1, 18))
    for idx, (box, det) in enumerate([(b, d) for b in "ABC" for d in range(6)]):
        key = f"{box}{det}"
        c_vals = []
        c_errs = []
        ys = []
        for d in data:
            v = d["C_det_per_det"].get(key)
            if v is None:
                continue
            c_vals.append(v["C"])
            c_errs.append(v["C_std"])
            ys.append(int(d["year"]))
        ax.errorbar(ys, c_vals, yerr=c_errs, fmt="o-", color=colors[idx],
                     markersize=4, lw=1, label=f"{box}-{det}")
    ax.set_xlabel("Year")
    ax.set_ylabel("C_det (cnt/s, at |mlat|<5°)")
    ax.set_title("per-(box, det) C_det vs year — electronics drift?")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    # Panel 2: B(|mlat|) at selected bins vs year
    ax = axes[0, 1]
    target_bins = ["20-25", "25-30", "30-35", "35-40", "40-45"]
    colors2 = plt.cm.plasma(np.linspace(0.1, 0.9, len(target_bins)))
    for ci, bin_key in enumerate(target_bins):
        b_vals = []; b_errs = []; ys = []
        for d in data:
            v = d["B_per_mlat_bin"].get(bin_key)
            if v is None:
                continue
            b_vals.append(v["B"])
            b_errs.append(v["B_std"])
            ys.append(int(d["year"]))
        ax.errorbar(ys, b_vals, yerr=b_errs, fmt="o-", color=colors2[ci],
                     markersize=5, lw=1.5, label=f"|mlat|={bin_key}°")
    ax.set_xlabel("Year")
    ax.set_ylabel("B(|mlat|) (cnt/s)")
    ax.set_title("B(|mlat|) vs year — solar cycle modulation?")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: per-year stats
    ax = axes[1, 0]
    total_rows = [d["total_rows"] for d in data]
    clean_rows = [d["clean_rows"] for d in data]
    ax.plot(years, np.array(total_rows) / 1e6, "ko-", label="total rows (M)")
    ax.plot(years, np.array(clean_rows) / 1e6, "bs-", label="clean rows (M)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rows (millions)")
    ax.set_title("Sample size per year")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: n_wraps fraction
    ax = axes[1, 1]
    n_wraps_frac = []
    for d in data:
        nw = d["n_wraps_distribution"]
        total = sum(nw.values())
        if total == 0:
            n_wraps_frac.append(0)
        else:
            wrapped = sum(v for k, v in nw.items() if k != "n=0")
            n_wraps_frac.append(wrapped / total * 100)
    ax.plot(years, n_wraps_frac, "ro-", markersize=6)
    ax.set_xlabel("Year")
    ax.set_ylabel("Fraction of rows with wrap (%)")
    ax.set_title("Wrap fraction per year")
    ax.grid(True, alpha=0.3)

    fig.suptitle("PHO model cross-year validation (per-det C_det stability + B(|mlat|) over solar cycle)",
                  fontsize=13, fontweight="bold", y=1.0)
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # Print summary tables
    print("\n=== Per-det C_det per year ===")
    print(f"  {'(box,det)':<10}" + "".join(f"{y:>10}" for y in years))
    for box in "ABC":
        for det in range(6):
            key = f"{box}{det}"
            row = [f"{box}-{det}".ljust(10)]
            for d in data:
                v = d["C_det_per_det"].get(key)
                row.append(f"{v['C']:>+8.1f}" if v else "      -  ")
            print("  " + "  ".join(row))

    print("\n=== B(|mlat|) by year ===")
    print(f"  {'mlat bin':<10}" + "  ".join(f"{y:>8}" for y in years))
    for bk in target_bins:
        row = [bk.ljust(10)]
        for d in data:
            v = d["B_per_mlat_bin"].get(bk)
            row.append(f"{v['B']:>+6.1f}" if v else "    -  ")
        print("  " + "  ".join(row))


if __name__ == "__main__":
    main()
