#!/usr/bin/env python3
"""Cross-validate the PHO-conservation-derived k(year) against the OULU neutron
monitor (independent cosmic-ray intensity measurement).

If k(year) tracks OULU count rate, it proves k is genuine solar-modulated CR,
and OULU can drive k directly (k = a + b*OULU) without assuming an 11yr sinusoid.

Inputs:
  /tmp/oulu_nm.txt  — OULU daily, NMDB ASCII (date;count_rate), '#'-commented header
Output:
  plots/diag_neutron_vs_k.png  + correlation
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# k(year) reconstructed from PHO conservation (high-|mlat| fit, per v5t_fixed_formula_verify)
K_YEAR = {2017:0.00202, 2018:0.00205, 2019:0.00210, 2020:0.00218, 2021:0.00215,
          2022:0.00199, 2023:0.00179, 2024:0.00166, 2025:0.00174, 2026:0.00188}


def load_oulu(path):
    dates, cr = [], []
    for line in open(path):
        s = line.strip()
        if not s or s.startswith("#") or ";" not in s:
            continue
        d, v = s.split(";")
        try:
            cr.append(float(v)); dates.append(np.datetime64(d.strip().replace(" ", "T")))
        except ValueError:
            continue
    return np.array(dates), np.array(cr)


def main():
    dates, cr = load_oulu("/tmp/oulu_nm.txt")
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    # annual mean OULU
    oulu_yr = {}
    for y in np.unique(years):
        oulu_yr[int(y)] = float(np.nanmean(cr[years == y]))

    common = sorted(set(K_YEAR) & set(oulu_yr))
    k_arr = np.array([K_YEAR[y] for y in common])
    o_arr = np.array([oulu_yr[y] for y in common])

    r = np.corrcoef(k_arr, o_arr)[0, 1]
    b, a = np.polyfit(o_arr, k_arr, 1)  # k = a + b*OULU
    print(f"Pearson r(k, OULU annual) = {r:.3f}")
    print(f"linear: k = {a:.5e} + {b:.5e} * OULU")
    print("\n year   OULU    k_recon  k_from_OULU")
    for y in common:
        kf = a + b * oulu_yr[y]
        print(f" {y}  {oulu_yr[y]:7.2f}  {K_YEAR[y]:.5f}  {kf:.5f}")

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))

    # Panel 1: time series, dual axis
    ax = axes[0]
    ax.plot(dates, cr, lw=0.3, color="steelblue", alpha=0.4, label="OULU daily")
    oyr_x = [np.datetime64(f"{y}-07-01") for y in sorted(oulu_yr)]
    oyr_y = [oulu_yr[y] for y in sorted(oulu_yr)]
    ax.plot(oyr_x, oyr_y, "o-", color="navy", lw=2, label="OULU annual mean")
    ax.set_xlabel("date"); ax.set_ylabel("OULU count rate (corr_for_efficiency)", color="navy")
    ax.tick_params(axis="y", labelcolor="navy")
    ax2 = ax.twinx()
    kx = [np.datetime64(f"{y}-07-01") for y in common]
    ax2.plot(kx, k_arr, "s-", color="crimson", lw=2, markersize=8, label="k(year) from PHO conservation")
    ax2.set_ylabel("k(year)  reconstructed", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax.set_title(f"OULU neutron monitor vs reconstructed k(year)\nboth track solar cycle (2020 max, 2024 min)", fontsize=12)
    ax.grid(True, alpha=0.3)
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=9)

    # Panel 2: scatter k vs OULU + fit
    ax = axes[1]
    sc = ax.scatter(o_arr, k_arr, c=common, cmap="turbo", s=120, zorder=3, edgecolor="k")
    xx = np.linspace(o_arr.min(), o_arr.max(), 50)
    ax.plot(xx, a + b * xx, "k--", lw=1.5, label=f"k = {a:.2e} + {b:.2e}·OULU")
    for y in common:
        ax.annotate(str(y), (oulu_yr[y], K_YEAR[y]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("OULU annual mean count rate"); ax.set_ylabel("k(year) reconstructed")
    ax.set_title(f"k vs OULU  —  Pearson r = {r:.3f}", fontsize=12)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("year")
    ax.legend(loc="upper left", fontsize=10); ax.grid(True, alpha=0.3)

    fig.suptitle("Independent cross-validation: PHO-conservation k(t) vs OULU neutron monitor (CR proxy)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    plt.savefig("plots/diag_neutron_vs_k.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved plots/diag_neutron_vs_k.png")


if __name__ == "__main__":
    main()
