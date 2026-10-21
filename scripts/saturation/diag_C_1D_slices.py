#!/usr/bin/env python3
"""1D slices of the C(t, |mlat|) 2D heatmap.

Reads the saved npz from diag_C_t_mlat_2D.py and plots:
  - Horizontal slices: C vs |mlat| at several fixed times
  - Vertical slices:  C vs time at several fixed |mlat| values
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

NPZ = "n_below_study/v5_npz/C_2D_heatmap.npz"


def main():
    z = np.load(NPZ)
    C       = z["C_med"]           # shape (60, 108)  — actually mean, not median
    n       = z["C_n"]
    months  = z["months"]          # array of "YYYY-MM"
    edges   = z["mlat_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])

    month_dt = np.array([np.datetime64(m + "-15") for m in months])
    n_mlat, n_month = C.shape
    print(f"loaded: {n_mlat} mlat bins × {n_month} months, "
          f"{(n > 200).sum()} valid bins")

    # ─── Horizontal slices (fixed time → mlat axis) ───
    pick_months = ["2017-09", "2019-06", "2021-06", "2023-06", "2025-06", "2026-05"]
    pick_idx_t = [list(months).index(m) for m in pick_months if m in months]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    cmap = plt.cm.viridis
    for k, ti in enumerate(pick_idx_t):
        ax.plot(centers, C[:, ti], '-', lw=1.5,
                color=cmap(k / max(len(pick_idx_t)-1, 1)),
                label=months[ti])
    ax.axvline(20, color='red', ls='--', alpha=0.5, label='|mlat|=20°')
    ax.set_xlabel("|mlat| (deg)", fontsize=12)
    ax.set_ylabel("mean C (cnt/s)", fontsize=12)
    ax.set_title("Horizontal slices — C vs |mlat| at fixed dates", fontsize=12)
    ax.legend(fontsize=10, title='date', loc='upper left')
    ax.grid(alpha=0.3)

    # ─── Vertical slices (fixed |mlat| → time axis) ───
    pick_mlat = [3, 10, 20, 30, 40, 50, 57]  # |mlat| in degrees (bin centers)
    pick_idx_m = [int(np.argmin(np.abs(centers - m))) for m in pick_mlat]

    ax = axes[1]
    cmap = plt.cm.plasma
    for k, mi in enumerate(pick_idx_m):
        y = C[mi, :].copy()
        # mask bins with too few rows
        y[n[mi, :] < 200] = np.nan
        ax.plot(month_dt, y, '-', lw=1.2,
                color=cmap(k / max(len(pick_idx_m)-1, 1)),
                label=f"|mlat|={int(round(centers[mi]))}°")
    ax.set_xlabel("date", fontsize=12)
    ax.set_ylabel("mean C (cnt/s)", fontsize=12)
    ax.set_title("Vertical slices — C vs time at fixed |mlat|", fontsize=12)
    ax.legend(fontsize=10, title='|mlat|', loc='upper right')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("1D slices of C(t, |mlat|)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/diag_C_1D_slices.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
