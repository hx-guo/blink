#!/usr/bin/env python3
"""Interactive scale tuner: HXMT/HE recovery vs SVOM/GRM for GRB 250919A.

Opens a native window (macosx backend) with a slider for the SVOM/GRM scale
factor. The three saturated bins (excluded from the auto fit) are shaded; their
HXMT/SVOM ratios update live as you drag.

Run:
    cd blink && .venv/bin/python scripts/bursts/svom_scale_slider.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
# NOTE: these helper modules call matplotlib.use("Agg") at import time, which
# would silently switch us to the non-interactive backend — so we re-assert an
# interactive backend AFTER importing them, right before building the figure.
from plot_hxmt_vs_ibis_bands import channel_to_kev_lut, fit_background
from plot_hxmt_vs_svom import load_hxmt, load_svom
plt.switch_backend("macosx")      # native window on macOS; use "TkAgg" elsewhere

RECON = "data/recon_cache/250919A_recon.csv"
SVOM = "data/svom_grm/svom_grm_evt_250919_00_v01.fits"
T0 = "2025-09-19T00:29:15"
BW, ELO, EHI = 0.5, 20, 200

print("loading...", file=sys.stderr)
E = channel_to_kev_lut()
ot, oe, ft, fe = load_hxmt(RECON, T0, E)
at, ae = np.concatenate([ot, ft]), np.concatenate([oe, fe])
st, se = load_svom(SVOM, T0, 30, 130)
edges = np.arange(-30, 130 + BW, BW); x = edges[:-1] + BW / 2
bkgm = ((x >= -30) & (x < -24)) | ((x >= 112) & (x < 130))

def net(t, e):
    m = (e >= ELO) & (e < EHI)
    r = np.histogram(t[m], bins=edges)[0] / BW
    return r - fit_background(x, r, bkgm, 1)

nHo, nHa, nS = net(ot, oe), net(at, ae), net(st, se)
satbin = np.histogram(ft, bins=edges)[0] > 0
sm = (x >= -2) & (x < 15) & (~satbin)
auto = nHa[sm].sum() / nS[sm].sum()
satix = np.where(satbin & (x >= -12) & (x < 28))[0]

fig, ax = plt.subplots(figsize=(11, 5.6))
plt.subplots_adjust(bottom=0.26, top=0.93)
for i in satix:
    ax.axvspan(x[i]-BW/2, x[i]+BW/2, color="red", alpha=0.12, zorder=0)
ax.fill_between(x, nHo, nHa, step="mid", color="C1", alpha=0.28, zorder=2)
ax.step(x, nHo, where="mid", color="#20347e", lw=1.2, label="HXMT/HE NaI observed", zorder=4)
ax.step(x, nHa, where="mid", color="#e07a12", lw=1.2, label="HXMT/HE observed + reconstructed", zorder=5)
(svline,) = ax.step(x, nS*auto, where="mid", color="#7d4fd0", lw=1.3,
                    label="SVOM/GRM × scale", zorder=3)
ax.axhline(0, color="grey", lw=0.5)
ax.set_xlim(-12, 28); ax.set_ylim(-180, 1850)
ax.set_xlabel("time since HXMT T0 (s)   [T0 = 2025-09-19T00:29:15 UTC]")
ax.set_ylabel("net rate (evt/s)"); ax.legend(loc="upper right", fontsize=8)
ax.set_title("GRB 250919A — drag the slider to set SVOM/GRM scale  (20–200 keV, 0.5 s)")
txt = ax.text(0.012, 0.94, "", transform=ax.transAxes, va="top", fontsize=9,
              family="monospace", color="dimgray")

sax = plt.axes([0.12, 0.11, 0.62, 0.04])
slider = Slider(sax, "SVOM ×", 0.03, 0.20, valinit=round(auto, 3), valstep=0.001)
bax = plt.axes([0.80, 0.105, 0.12, 0.05])
btn = Button(bax, "auto (%.3f)" % auto)

def update(_=None):
    s = slider.val
    svline.set_ydata(nS * s)
    lines = [f"scale ×{s:.3f}   (auto {auto:.3f})", "saturated-bin  HXMT / (SVOM×scale):"]
    for i in satix:
        r = nHa[i] / (nS[i]*s) if nS[i] > 0 else float("nan")
        lines.append(f"   t={x[i]:+.1f}s :  {r:.2f}")
    txt.set_text("\n".join(lines))
    fig.canvas.draw_idle()

slider.on_changed(update)
btn.on_clicked(lambda _: slider.set_val(round(auto, 3)))
update()
print("window open — drag the slider. close it to exit.", file=sys.stderr)
plt.show()
