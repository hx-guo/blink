"""The 19 marginal holes, stacked: what the hole takes and what comes back.

Left: the affected box against the mean of the two boxes that kept reading,
summed over all 19 holes at 1 ms resolution.  Right: counts recovered after the
hole, as a fraction of what the hole should have held, with the two healthy
boxes carrying the same measurement as the null.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import talk20_style as st
from plot_heb_marginal_holes import HOLES, load, HOLE_FILL, LC, HW, RAW

z = int(HW / RAW)
PRE, POST = 120, 300          # ms drawn before the hole starts / after it ends

prof_hit = np.zeros(PRE + POST); prof_ref = np.zeros(PRE + POST)
post_hit = np.zeros(POST); post_ref = np.zeros(POST)
deficit = 0.0
holes_ms = []

for name, box, t0, ms, t_rel in HOLES:
    h = load(name)
    bi = "ABC".index(box)
    hit = h[bi]
    ref = h[[j for j in range(3) if j != bi]].sum(axis=0) / 2
    W = int(np.ceil(ms))
    e = z + W                                   # bin just past the end of the hole
    bg = hit[z-1000:z].sum(); bgr = ref[z-1000:z].sum()      # counts in 1 s
    # aligned on the END of the hole, so the flush stays sharp; the hole itself
    # then sits in [-W, 0] and its 11.9-17.5 ms spread smears only its left edge
    prof_hit += hit[e-PRE:e+POST]; prof_ref += ref[e-PRE:e+POST]
    post_hit += hit[e:e+POST] - bg / 1000.0
    post_ref += ref[e:e+POST] - bgr / 1000.0
    deficit += bg * ms / 1e3
    holes_ms.append(ms)

mean_hole = float(np.mean(holes_ms))
st.apply(12)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.5))

n = len(HOLES)
t = np.arange(-PRE, POST) + 0.5
ax1.axvspan(-mean_hole, 0, color=HOLE_FILL, lw=0, zorder=0)
ax1.step(t, prof_ref / n * 1e3, where="mid", color=st.NAVY, lw=1.3, zorder=2,
         label="mean of the two boxes that kept reading")
ax1.step(t, prof_hit / n * 1e3, where="mid", color=st.RED, lw=1.5, zorder=3,
         label="the box with the hole")
ax1.set_xlim(-60, 80)
ax1.set_xlabel("time from the end of the hole (ms)")
ax1.set_ylabel("counts s$^{-1}$ per box")
ax1.legend(loc="upper left", fontsize=10.5)
ax1.set_title("19 holes stacked, 1 ms bins", loc="left", fontsize=12, color=st.NAVY)
st.clean(ax1)

# right: cumulative recovery
cum_hit = np.cumsum(post_hit) / deficit
cum_ref = np.cumsum(post_ref) / deficit
t = np.arange(300) + 1
ax2.axhline(1.0, color=st.GREY, lw=1.0, ls=(0, (4, 3)))
ax2.text(290, 1.02, "everything back", ha="right", fontsize=10, color=st.GREY)
ax2.plot(t, cum_hit, color=st.RED, lw=1.8, label="the box with the hole")
ax2.plot(t, cum_ref, color=st.NAVY, lw=1.4, label="the other two boxes (null)")
ax2.set_xlim(0, 300); ax2.set_ylim(-0.1, 1.15)
ax2.set_xlabel("ms after the end of the hole")
ax2.set_ylabel("counts recovered / counts the hole should hold")
ax2.legend(loc="lower right", fontsize=10.5)
ax2.set_title("%.0f of the %.0f missing counts come back within 20 ms"
              % (cum_hit[19] * deficit, deficit), loc="left", fontsize=12, color=st.NAVY)
st.clean(ax2)

fig.tight_layout()
fig.savefig("heb_marginal_holes_stack.png", dpi=150)
print("wrote heb_marginal_holes_stack.png")
print("deficit %.0f, recovered@20ms %.0f (%.2f), @300ms %.2f, null@300ms %.3f"
      % (deficit, cum_hit[19]*deficit, cum_hit[19], cum_hit[299], cum_ref[299]))
