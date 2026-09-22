"""Light curves around the 19 marginal holes of the HEB_list saturation audit.

Each hole is a 11-18 ms break in ONE box.  The question the figure answers is
whether the source was bright while the box was blind: if it was, the two boxes
that kept reading would show a spike across the hole, and the break would be a
FIFO reset after all.  So each panel draws the box that has the hole against the
mean of the other two, from the 1K event stream.
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

LC = "/private/tmp/claude-501/-Users-skyair-Developer-ihep-blink/3133f7dc-f0f0-40ac-9dc6-d39abf216e7e/scratchpad/lc"
HW = 5.0          # half width of the extracted stream, s
RAW = 0.001       # bin width it was written with, s

# name, box, hole_start_met, hole_ms, t_rel_trigger
HOLES = [
    ("HEB180405168", "B", 197524981.713222, 16.74, +5.6532),
    ("HEB180427442", "B", 199449426.334142, 16.95, +0.3341),
    ("HEB180718762", "A", 206561919.864658, 17.49, +12.8647),
    ("HEB191108003", "B", 247795550.812100, 16.70, +70.2121),
    ("HEB191202867", "C", 249943743.848460, 17.09, +1.2985),
    ("HEB200326421", "B", 259841333.607586, 16.15, +73.7476),
    ("HEB201221978", "B", 283217331.311378, 16.01, +0.6814),
    ("HEB210129908", "C", 286580940.089990, 16.03, +0.1900),
    ("HEB210406716", "C", 292353091.271456, 15.32, -0.6485),
    ("HEB210807955", "B", 303000928.204000, 15.23, -7.2960),
    ("HEB210818043", "B", 303872552.184170, 15.62, +21.0342),
    ("HEB211105190", "B", 310710909.761458, 11.87, +34.9615),
    ("HEB211207416", "B", 313495152.294902, 17.48, -2.6551),
    ("HEB221226697", "A", 346697080.461520, 15.22, +10.2115),
    ("HEB230525448", "A", 359635571.325408, 16.20, +0.0054),
    ("HEB250403635", "B", 418317326.311138, 16.08, -2.8389),
    ("HEB260305112", "A", 447302533.280942, 15.53, +32.0959),
    ("HEB260412259", "C", 450598449.685510, 15.40, +20.5155),
    ("HEB260611897", "B", 455837575.254856, 15.38, +18.6349),
]

HOLE_FILL = "#F2C9A0"
MCU_LIMIT = 15571.0   # evt/s, 109 events per 7 ms


def load(name):
    n = int(round(2 * HW / RAW))
    h = np.zeros((3, n))
    with open(os.path.join(LC, name + ".csv")) as f:
        for r in csv.reader(f):
            if r[0] == "bin_ms":
                continue
            k = int(r[0])
            h[0, k], h[1, k], h[2, k] = int(r[1]), int(r[2]), int(r[3])
    return h


def rebin(y, k):
    m = (len(y) // k) * k
    return y[:m].reshape(-1, k).sum(axis=1)


def draw(fig_name, half, bin_s, title):
    ncol, nrow = 5, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(17.5, 10.2))
    k = int(round(bin_s / RAW))
    for ax in axes.ravel():
        ax.set_visible(False)

    for i, (name, box, t0, hole_ms, t_rel) in enumerate(HOLES):
        ax = axes.ravel()[i]
        ax.set_visible(True)
        st.clean(ax)
        h = load(name)
        bi = "ABC".index(box)
        t = (np.arange(len(rebin(h[0], k))) + 0.5) * bin_s - HW
        hit = rebin(h[bi], k) / bin_s
        ref = rebin(h[[j for j in range(3) if j != bi]].sum(axis=0), k) / (2 * bin_s)

        m = np.abs(t) <= half
        ax.axvspan(0, hole_ms / 1e3, color=HOLE_FILL, lw=0, zorder=0)
        ax.step(t[m], ref[m], where="mid", color=st.NAVY, lw=1.0, zorder=2)
        ax.step(t[m], hit[m], where="mid", color=st.RED, lw=1.2, zorder=3)
        ax.set_xlim(-half, half)
        top = max(hit[m].max(), ref[m].max())
        ax.set_ylim(0, top * 1.25)
        ax.text(0.03, 0.95, name, transform=ax.transAxes, va="top", fontsize=10.5,
                color=st.NAVY)
        ax.text(0.03, 0.83, "box %s · %.1f ms · T%+.1f s" % (box, hole_ms, t_rel),
                transform=ax.transAxes, va="top", fontsize=9, color=st.GREY)
        ax.tick_params(labelsize=8.5)
        if i % ncol == 0:
            ax.set_ylabel("counts s$^{-1}$", fontsize=9.5)
        if i >= len(HOLES) - ncol:
            ax.set_xlabel("time from hole (s)", fontsize=9.5)

    handles = [plt.Line2D([], [], color=st.RED, lw=1.6),
               plt.Line2D([], [], color=st.NAVY, lw=1.4),
               plt.Rectangle((0, 0), 1, 1, fc=HOLE_FILL, ec="none")]
    fig.legend(handles, ["the box with the hole", "mean of the other two boxes",
                         "the hole"],
               loc="lower right", bbox_to_anchor=(0.985, 0.045), ncol=3, fontsize=10.5)
    fig.suptitle(title, x=0.012, ha="left", fontsize=13.5, color=st.NAVY)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.savefig(fig_name, dpi=140)
    print("wrote", fig_name)


if __name__ == "__main__":
    st.apply(11)
    draw("heb_marginal_holes_zoom.png", 0.25, 0.004,
         "HEB marginal holes — 4 ms bins, ±0.25 s around the hole")
    draw("heb_marginal_holes_wide.png", 5.0, 0.100,
         "HEB marginal holes — 100 ms bins, ±5 s around the hole")
