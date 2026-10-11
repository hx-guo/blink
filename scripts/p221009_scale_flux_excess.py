#!/usr/bin/env python3
"""GRB 221009A: (1) per-bin decomposition of the raw-scale excess over the
single-tau non-paralyzable curve of f4 (plot_scale_flux_dt.py), and
(2) HE/(200 LG) versus GECAM LG hardness on bins free of dead time.

Reads data/p221009/scale/gecam_1s.csv (run p221009_scale_gecam_hardness.py
first). Outputs: flux_excess.csv, ratio_vs_hardness.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from p221009_scale_common import OUT, SCALE, saa_mask, robust

TAU = 2.088e-4      # (GECAM cts/s)^-1, as fitted by plot_scale_flux_dt.py
B_HE = 9331.5
R_B = B_HE / SCALE
LIVE_BKG = 0.993


def main():
    d = pd.read_csv(OUT / "gecam_1s.csv")
    x = d.t.to_numpy()
    g = d.lg_raw.to_numpy()
    he = d.he_net.to_numpy()
    sig = (he > 5e3) & (g > 30) & np.isfinite(g)
    raw = np.where(sig, g / he, np.nan)
    model = (1 + TAU * (g + R_B)) / SCALE
    lf_model = 1 / (1 + TAU * (g + R_B))
    fl = d.he_flive.to_numpy()

    edges = [30, 100, 300, 1000, 3000, 10000, 30000]
    print(f"{'decade':<14}{'n':>4}{'raw/model med':>15}{'model@ctr/med':>15}"
          f"{'200*scale_dt':>14}{'lf_meas/lf_mod':>16}{'LG live':>9}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        w = sig & (g >= lo) & (g < hi)
        ctr = np.sqrt(lo * hi)
        mod_ctr = (1 + TAU * (ctr + R_B)) / SCALE
        sdt = g[w] / d.he_net_dt.to_numpy()[w] * SCALE
        print(f"{lo:>6}-{hi:<7}{w.sum():>4}{np.nanmedian(raw[w] / model[w]):15.3f}"
              f"{np.nanmedian(raw[w]) / mod_ctr:15.3f}{np.nanmedian(sdt):14.3f}"
              f"{np.nanmedian(fl[w] / lf_model[w]):16.3f}{np.nanmedian(d.live_lg[w]):9.3f}")

    w = sig & (g >= 10000)
    tab = pd.DataFrame(dict(t=x[w], lg=g[w], lg_live=d.live_lg[w], he_flive=fl[w],
                            lf_model=lf_model[w], raw_scale=raw[w], model=model[w],
                            raw_over_model=raw[w] / model[w],
                            he_dt_over_200lg=d.he_net_dt[w] / (SCALE * g[w]),
                            he_dt_over_200lg_dc=d.he_net_dt[w] / (SCALE * d.lg_dc[w])))
    tab.to_csv(OUT / "flux_excess.csv", index=False, float_format="%.4g")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # ---- (2) ratio vs LG hardness, 3-s bins, no dead time on either side ----
    e3 = np.arange(-50, 701, 3.0)
    idx = np.digitize(x, e3) - 1
    agg = {}
    for k in (9, 10, 11, 12):
        for s in ("gross", "bkg"):
            agg[f"ch{k}_{s}"] = np.bincount(idx, d[f"ch{k}_{s}"], len(e3) - 1)
    he3 = np.bincount(idx, np.nan_to_num(d.he_net_dt), len(e3) - 1)
    fl3 = np.array([np.nanmin(fl[idx == i]) if (idx == i).any() else np.nan
                    for i in range(len(e3) - 1)])
    lv3 = np.array([np.nanmin(d.live_lg[idx == i]) if (idx == i).any() else np.nan
                    for i in range(len(e3) - 1)])
    n = {k: agg[f"ch{k}_gross"] - agg[f"ch{k}_bkg"] for k in (9, 10, 11, 12)}
    lg3 = sum(n.values())
    hr = n[10] / n[9]
    ehr = hr * np.sqrt((agg["ch10_gross"] + agg["ch10_bkg"]) / n[10] ** 2
                       + (agg["ch9_gross"] + agg["ch9_bkg"]) / n[9] ** 2)
    rat = he3 / (SCALE * lg3)
    x3 = e3[:-1]
    bad = saa_mask(x3, 3.0)
    ok = (~bad & (x3 >= 179) & (lg3 > 3 * 150) & (fl3 > 0.9) & (lv3 > 0.97)
          & (ehr < 0.12) & np.isfinite(rat))
    ph = np.select([x3 < 215, x3 < 300, x3 < 440, x3 < 600], ["calib", "valley", "300-440", "flare"], "tail")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, col in [("calib", "C0"), ("valley", "C2"), ("300-440", "C4"), ("flare", "C3"), ("tail", "C5")]:
        m = ok & (ph == name)
        ax.errorbar(hr[m], rat[m], xerr=ehr[m], fmt="o", ms=4, color=col, label=f"{name} ({m.sum()})")
    rho = pd.Series(hr[ok]).corr(pd.Series(rat[ok]), method="spearman")
    A = np.vstack([hr[ok], np.ones(ok.sum())]).T
    sl, ic = np.linalg.lstsq(A, rat[ok], rcond=None)[0]
    hh = np.linspace(0.3, 0.9, 10)
    ax.plot(hh, sl * hh + ic, "k--", lw=1, label=f"LSQ slope {sl:.2f}, Spearman {rho:.2f}")
    ax.axhline(1, color="0.6", lw=0.6)
    ax.set_xlabel("GECAM LG hardness ch10/ch9 = (0.78-1.5)/(0.38-0.78 MeV)")
    ax.set_ylabel("HE (DT-corr) / (200 x GECAM LG)")
    ax.set_title("3-s bins, HE live > 0.9, LG live > 0.97")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "ratio_vs_hardness.png", dpi=150, bbox_inches="tight")
    print(f"ratio vs HR: n={ok.sum()} spearman={rho:.3f} slope={sl:.3f} icpt={ic:.3f}")
    for name in ("calib", "valley", "300-440", "flare", "tail"):
        m = ok & (ph == name)
        if m.any():
            print(f"  {name:8s} n={m.sum():3d} HR med {np.median(hr[m]):.3f}  ratio med {np.median(rat[m]):.3f}")


if __name__ == "__main__":
    main()
