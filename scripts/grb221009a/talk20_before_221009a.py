#!/usr/bin/env python3
"""GRB 221009A main pulse, raw only: what HXMT/HE actually recorded next to
GECAM-C. No dead-time correction, no reconstruction. Talk-20 slide colours.

HE detected this burst with its CsI anti-coincidence crystal (far off-axis,
hard), so the light curve uses CsI events only — same selection as
talk20_fig_221009a.py, and the GECAM-C effective-area scale is fitted on the
same clean seconds so the two slides are one consistent picture.

Usage: python3 scripts/grb221009a/talk20_before_221009a.py -o <dir>   (run from blink root)
"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../gecam", "../lib", "../talks",)]  # sibling and shared script modules
import talk20_style as ts
from talk20_fig_221009a import load_csi, CACHE_PW, BEFORE, AFTER, BIN, SAA_END
from make_f10 import load_eng_dtau, to_grid
from plot_hxmt_vs_gecam import compute_time_offset, load_gecam_btime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out-dir", default="/tmp")
    a = ap.parse_args()
    ts.apply(13)

    x, obs, allr, r_fill, n_fill, B_evt = load_csi(CACHE_PW, BEFORE, AFTER, BIN)
    et, dtau, flive, tau_us, B_eng = load_eng_dtau(BEFORE, AFTER)
    flive_g = to_grid(x, et, flive)
    g_met, _, _ = compute_time_offset()
    _, g_rate, _ = load_gecam_btime(g_met, BEFORE, AFTER, BIN, "lg", True)
    g_net = np.nan_to_num(g_rate, nan=0.0)
    # Detector-off seconds (HE in SAA working mode) have zero events, zero
    # fillers and zero recorded dead time, so they pass every naive "clean"
    # test while GECAM sits on the peak — they must be cut before the fit.
    off = (allr + B_evt) < 10.0
    off = off | np.r_[off[1:], False] | np.r_[False, off[:-1]]
    clean = ((x > SAA_END) & (x < AFTER) & ~off & (r_fill == 0)
             & (flive_g > 0.95) & (g_net > 50.0) & np.isfinite(obs)
             & (obs > 0))
    scale = float(np.nanmedian(obs[clean] / g_net[clean]))
    gecam = g_net * scale
    print(f"  GECAM-C scale x{scale:.0f} ({int(clean.sum())} clean seconds)",
          file=sys.stderr)

    fig, ax = plt.subplots(figsize=(10.4, 5.4), dpi=300)
    ax.step(x, gecam, where="post", color=ts.RED, lw=2.0, zorder=2)
    ax.step(x, obs, where="post", color=ts.NAVY, lw=2.4, zorder=4)
    ax.set_xlim(172, 300)
    w = (x >= 215) & (x < 240)
    pk_o = np.nanmax(obs[w]); pk_g = np.nanmax(gecam[w])
    xi = x[w][int(np.nanargmax(gecam[w]))]
    ax.set_ylim(-0.06 * pk_g, 1.13 * pk_g)
    ax.set_xlabel("2022-10-09 触发后的秒数")
    ax.set_ylabel("净计数率（事件 / 秒）")
    ax.text(0.014, 0.975, "慧眼 HE 的 CsI，1 秒并道", transform=ax.transAxes,
            va="top", fontsize=13, color=ts.TICK)
    ax.text(233, 0.94 * pk_g, "GECAM-C 独立看到的", color=ts.RED, fontsize=16,
            fontweight="medium")
    ax.text(233, 0.855 * pk_g, "（按有效面积归一）", color=ts.RED, fontsize=13)
    ax.annotate("慧眼 HE 记到的", xy=(196, 0.012 * pk_g),
                xytext=(186, 0.24 * pk_g), color=ts.NAVY, fontsize=16,
                fontweight="medium",
                arrowprops=dict(arrowstyle="-", color=ts.NAVY, lw=1.2))
    ax.annotate("", xy=(xi, pk_g), xytext=(xi, pk_o),
                arrowprops=dict(arrowstyle="<->", color="#282828", lw=1.6))
    frac = 100 * pk_o / pk_g
    txt = f"只记到 {frac:.1f}%" if frac >= 1 else f"只记到 {frac:.1f}%"
    ax.text(xi - 2.5, pk_g * 0.52, txt, color="#282828", fontsize=17,
            fontweight="medium", ha="right")
    ts.clean(ax)
    fig.tight_layout(pad=0.5)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"before_221009a.{ext}",
                    dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"saved {out}/before_221009a.{ext}")
    print(f"peak obs {pk_o:.0f}  peak gecam {pk_g:.0f}  ratio {frac:.2f}%")


if __name__ == "__main__":
    main()
