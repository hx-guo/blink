"""Do the isolated FIFO resets have a counterpart in the boxes that kept reading?

A photon burst is seen by all three boxes; a particle track or a local glitch is
not.  For each of the 46 windows whose only reset is a single isolated one, the
resetting box is stacked against the mean of the other two, split by whether
those two see a >5 sigma excess across the gap.
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

SP = ("/private/tmp/claude-501/-Users-skyair-Developer-ihep-blink/"
      "3133f7dc-f0f0-40ac-9dc6-d39abf216e7e/scratchpad")
HW, RAW = 5.0, 0.001
z = int(HW / RAW)
PRE, POST = 120, 200
HOLE_FILL = "#F2C9A0"


def load(name):
    nb = int(round(2 * HW / RAW))
    h = np.zeros((3, nb))
    for r in csv.reader(open(os.path.join(SP, "lc_sat", name + ".csv"))):
        if r[0] == "bin_ms":
            continue
        k = int(r[0])
        h[0, k], h[1, k], h[2, k] = int(r[1]), int(r[2]), int(r[3])
    return h


rows = [l.split() for l in open(os.path.join(SP, "single_resets.txt"))]
gaps = {g["name"]: g for g in csv.DictReader(open("HEB_saturation_resets.csv"))}

groups = {True: [], False: []}
for name, box, t0, _ in rows:
    ms = float(gaps[name]["gap_s"]) * 1e3
    W = int(np.ceil(ms))
    h = load(name)
    bi = "ABC".index(box)
    hit, ref = h[bi], h[[j for j in range(3) if j != bi]].sum(axis=0) / 2
    bgr = ref[z-1000:z-100].sum() / 0.9
    sig = (ref[z-40:z+W+60].sum() - bgr * 0.100) / np.sqrt(max(bgr * 0.1, 1.0))
    e = z + W
    groups[sig > 5].append((hit[e-PRE:e+POST], ref[e-PRE:e+POST], ms))

st.apply(12)
fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.5), sharex=True)
titles = {True: "the other two boxes see it — a real burst (20 windows)",
          False: "the other two boxes see nothing — local to one box (26 windows)"}
t = np.arange(-PRE, POST) + 0.5
for ax, key in zip(axes, (True, False)):
    g = groups[key]
    n = len(g)
    hit = np.sum([a for a, b, m in g], axis=0) / n * 1e3
    ref = np.sum([b for a, b, m in g], axis=0) / n * 1e3
    ax.axvspan(-float(np.mean([m for a, b, m in g])), 0, color=HOLE_FILL, lw=0, zorder=0)
    ax.step(t, ref, where="mid", color=st.NAVY, lw=1.3, zorder=2,
            label="mean of the two boxes that kept reading")
    ax.step(t, hit, where="mid", color=st.RED, lw=1.5, zorder=3,
            label="the box that reset")
    ax.set_xlim(-80, 120)
    ax.set_xlabel("time from the end of the reset gap (ms)")
    ax.set_title(titles[key], loc="left", fontsize=12, color=st.NAVY)
    st.clean(ax)
axes[0].set_ylabel("counts s$^{-1}$ per box")
axes[0].legend(loc="upper right", fontsize=10.5)
fig.tight_layout()
fig.savefig("heb_reset_skytest.png", dpi=150)
print("wrote heb_reset_skytest.png")
