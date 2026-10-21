#!/usr/bin/env python3
"""Interactive manual annotation tool for Ge 2023 Fig 1.

Run from terminal:
    python3 scripts/grb221009a/ge2023_annotate.py

Controls:
    R              Switch to RED-marker add mode
    B              Switch to BLACK-marker add mode
    D              Switch to DELETE mode
    Z              Zoom in (matplotlib default)
    Left-click     In ADD mode: add a marker at click position.
                   In DELETE mode: remove the nearest marker to click.
    S              Save current CSVs and exit.
    Q / window close: save and quit.

Current detections are loaded from the CSVs at startup and shown as crosses
(green = red marker, orange = black marker).  Each edit updates the in-memory
state; pressing S writes both CSVs back to disk.

Status bar at bottom: current mode + cursor (T_ms, rate) live readout.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

# Disable matplotlib default keybindings so they don't shadow ours.
# Defaults that bite: r=reset-view, b=back-view, s=save-figure, q=quit,
# d/p=pan, h=home, etc.
for k in list(matplotlib.rcParams.keys()):
    if k.startswith("keymap."):
        matplotlib.rcParams[k] = []

PAGE_IMG = "/tmp/ge2023_fig1_hires.png"
RED_CSV = Path("/Users/skyair/Developer/ihep/blink/data/ge2023/ge2023_fig1_red.csv")
BLK_CSV = Path("/Users/skyair/Developer/ihep/blink/data/ge2023/ge2023_fig1_black.csv")

# axis calibration (from earlier digitization)
P1_X, P2_X = 1483, 1720
SLOPE_Y, INTERCEPT_Y = -0.007371, 11.798
PX_PER_MS = (P2_X - P1_X) / 28.97
GE_TO_OURS_MS = 415.5

# fig bbox for cropping the view
FIG_X0, FIG_Y0 = 900, 250
FIG_X1, FIG_Y1 = 2700, 1650


def x_to_t(x):
    return (x - P1_X) / PX_PER_MS


def y_to_rate(y):
    return SLOPE_Y * y + INTERCEPT_Y


def t_to_x(T):
    return P1_X + T * PX_PER_MS


def rate_to_y(r):
    return (r - INTERCEPT_Y) / SLOPE_Y


class Annotator:
    def __init__(self):
        if not Path(PAGE_IMG).exists():
            sys.exit(f"Page image missing: {PAGE_IMG}\nRun the digitizer first.")
        self.img = np.array(Image.open(PAGE_IMG))
        self.red = (np.loadtxt(RED_CSV, delimiter=",", skiprows=1)
                    if RED_CSV.exists() else np.empty((0, 3)))
        self.blk = (np.loadtxt(BLK_CSV, delimiter=",", skiprows=1)
                    if BLK_CSV.exists() else np.empty((0, 3)))
        if self.red.ndim == 1:
            self.red = self.red.reshape(1, 3)
        if self.blk.ndim == 1:
            self.blk = self.blk.reshape(1, 3)
        self.mode = "red"   # 'red' / 'black' / 'delete'

        self.fig, self.ax = plt.subplots(figsize=(18, 12))
        self.ax.imshow(self.img)
        self.ax.set_xlim(FIG_X0, FIG_X1)
        self.ax.set_ylim(FIG_Y1, FIG_Y0)   # y inverted (image)
        self.red_scatter = self.ax.scatter([], [], marker="+", s=120,
                                            c="lime", linewidths=2,
                                            label="red")
        self.blk_scatter = self.ax.scatter([], [], marker="+", s=120,
                                            c="orange", linewidths=2,
                                            label="black")
        self.draw_dashed()
        self.refresh()

        self.title = self.fig.suptitle("", fontsize=14)
        self.update_title()
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def draw_dashed(self):
        # P1, P2 dashed verticals for visual reference
        for x in (P1_X, P2_X):
            self.ax.axvline(x, color="magenta", linestyle="--", linewidth=1,
                            alpha=0.6)

    def refresh(self):
        if len(self.red):
            self.red_scatter.set_offsets(
                np.column_stack([t_to_x(self.red[:, 0]),
                                 rate_to_y(self.red[:, 2])]))
        else:
            self.red_scatter.set_offsets(np.empty((0, 2)))
        if len(self.blk):
            self.blk_scatter.set_offsets(
                np.column_stack([t_to_x(self.blk[:, 0]),
                                 rate_to_y(self.blk[:, 2])]))
        else:
            self.blk_scatter.set_offsets(np.empty((0, 2)))
        self.fig.canvas.draw_idle()

    def update_title(self, T=None, rate=None):
        n_r, n_b = len(self.red), len(self.blk)
        cursor = f"   cursor: T={T:+.2f} ms, rate={rate:.3f}" if T is not None else ""
        self.title.set_text(
            f"[mode: {self.mode.upper()}]   red={n_r}, black={n_b}"
            f"   R=red B=black D=delete S=save Q=quit{cursor}")
        self.fig.canvas.draw_idle()

    def on_motion(self, event):
        if event.inaxes != self.ax: return
        T = x_to_t(event.xdata)
        rate = y_to_rate(event.ydata)
        self.update_title(T, rate)

    def on_click(self, event):
        if event.inaxes != self.ax or event.button != 1: return
        # If the matplotlib toolbar is in pan/zoom mode, ignore the click —
        # let the toolbar handle it for navigation.
        tb = self.fig.canvas.toolbar
        if tb is not None and getattr(tb, "mode", "") != "":
            return
        T = x_to_t(event.xdata)
        rate = y_to_rate(event.ydata)
        if self.mode in ("red", "black"):
            row = np.array([[T, T + GE_TO_OURS_MS, rate]])
            if self.mode == "red":
                self.red = np.vstack([self.red, row])
                self.red = self.red[np.argsort(self.red[:, 0])]
            else:
                self.blk = np.vstack([self.blk, row])
                self.blk = self.blk[np.argsort(self.blk[:, 0])]
            print(f"+ {self.mode}: T={T:+.2f}, rate={rate:.3f}")
        elif self.mode == "delete":
            # find nearest marker (any color) and remove
            all_T = []
            all_rate = []
            tag = []
            for i, r in enumerate(self.red):
                all_T.append(r[0]); all_rate.append(r[2]); tag.append(("red", i))
            for i, b in enumerate(self.blk):
                all_T.append(b[0]); all_rate.append(b[2]); tag.append(("black", i))
            if not all_T:
                return
            dt_px = (np.array(all_T) - T) * PX_PER_MS
            dr_px = (np.array(all_rate) - rate) / -SLOPE_Y
            dist = np.hypot(dt_px, dr_px)
            j = int(np.argmin(dist))
            if dist[j] > 40:
                print(f"no marker within 40 px of click")
                return
            color, idx = tag[j]
            if color == "red":
                removed = self.red[idx]
                self.red = np.delete(self.red, idx, axis=0)
            else:
                removed = self.blk[idx]
                self.blk = np.delete(self.blk, idx, axis=0)
            print(f"- {color}: T={removed[0]:+.2f}, rate={removed[2]:.3f}")
        self.refresh()
        self.update_title(T, rate)

    def on_key(self, event):
        if event.key in ("r", "R"):
            self.mode = "red"
            print("mode -> RED")
        elif event.key in ("b", "B"):
            self.mode = "black"
            print("mode -> BLACK")
        elif event.key in ("d", "D"):
            self.mode = "delete"
            print("mode -> DELETE")
        elif event.key in ("s", "S"):
            self.save()
        elif event.key in ("q", "Q"):
            self.save()
            plt.close(self.fig)
        self.update_title()

    def save(self):
        header = "T_ms_GeFrame,T_ms_OursFrame,Rate_1e4_cnts_s"
        np.savetxt(RED_CSV, self.red, delimiter=",", fmt="%.3f",
                   header=header, comments="")
        np.savetxt(BLK_CSV, self.blk, delimiter=",", fmt="%.3f",
                   header=header, comments="")
        print(f"saved: red={len(self.red)}, black={len(self.blk)}")
        print(f"  {RED_CSV}")
        print(f"  {BLK_CSV}")


def main():
    matplotlib.use("MacOSX")   # native interactive on macOS
    a = Annotator()
    plt.show()
    a.save()


if __name__ == "__main__":
    main()
