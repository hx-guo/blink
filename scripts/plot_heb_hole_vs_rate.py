"""Hole length against the event rate that produced it.

If a hole is a dumped FIFO, its length is the time span of the events that were
in the buffer: 455 events (M67204H, 4096 x 9 bit at 9 bytes each) divided by the
input rate.  That locus is the curve.  Points should sit on or below it, because
the recorded rate is a lower bound once the buffer starts refusing writes.
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
from plot_heb_marginal_holes import HOLES, load as load_m

SP = ("/private/tmp/claude-501/-Users-skyair-Developer-ihep-blink/"
      "3133f7dc-f0f0-40ac-9dc6-d39abf216e7e/scratchpad")
HW, RAW = 5.0, 0.001
z = int(HW / RAW)
DEPTH = 4096 / 9.0
DRAIN = 109 / 0.007


def load_s(name):
    nb = int(round(2 * HW / RAW))
    h = np.zeros((3, nb))
    for r in csv.reader(open(os.path.join(SP, "lc_sat", name + ".csv"))):
        if r[0] == "bin_ms":
            continue
        k = int(r[0])
        h[0, k], h[1, k], h[2, k] = int(r[1]), int(r[2]), int(r[3])
    return h


gaps = {g["name"]: g for g in csv.DictReader(open("HEB_saturation_resets.csv"))}
pts = {"marginal": [], "reset": []}
for name, box, t0, ms, t_rel in HOLES:
    hit = load_m(name)["ABC".index(box)]
    pts["marginal"].append((ms, hit[z-20:z].max() * 1e3 / 1e3))
for line in open(os.path.join(SP, "single_resets.txt")):
    name, box, t0, _ = line.split()
    hit = load_s(name)["ABC".index(box)]
    pts["reset"].append((float(gaps[name]["gap_s"]) * 1e3,
                         hit[z-20:z].max() * 1e3 / 1e3))

st.apply(12)
fig, ax = plt.subplots(figsize=(8.2, 5.4))
x = np.linspace(8, 55, 300)
ax.plot(x, DEPTH / (x / 1e3) / 1e3, color=st.GREY, lw=1.6,
        label="455 events / hole length — a dumped full FIFO")
ax.axhline(DRAIN / 1e3, color=st.GREY, lw=1.2, ls=(0, (4, 3)))
ax.text(54, DRAIN / 1e3 + 0.7, "MCU read-out, 15.6 kHz", ha="right", fontsize=10,
        color=st.GREY)
for key, color, marker, label in (
        ("reset", st.NAVY, "o", "called saturated by the FIFO-reset criterion (46)"),
        ("marginal", st.RED, "D", "called marginal_hole — rate gate rejected them (19)")):
    a = np.array(pts[key])
    ax.scatter(a[:, 0], a[:, 1], s=46, facecolor=color, edgecolor="white",
               linewidth=0.8, marker=marker, zorder=3, label=label)
ax.set_xlim(8, 55); ax.set_ylim(0, 40)
ax.set_xlabel("length of the hole (ms)")
ax.set_ylabel("peak 1 ms event rate in the 20 ms before it (kHz)")
ax.legend(loc="upper right", fontsize=10.5)
ax.set_title("the two classes are one population", loc="left", fontsize=13,
             color=st.NAVY)
st.clean(ax)
fig.tight_layout()
fig.savefig("heb_hole_vs_rate.png", dpi=150)
print("wrote heb_hole_vs_rate.png")
