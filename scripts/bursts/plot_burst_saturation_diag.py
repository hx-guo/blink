#!/usr/bin/env python3
"""Diagnostic plot for the 2026-06-01T19:12:49.900 burst.

Three panels (Box A/B/C), each showing
  - 1B observed events binned to user-chosen bin width
  - 1B + gap-fill (where FIFO resets caused saturation)
  - FIFO reset intervals (orange vertical bands)
  - 1K reference (grey shaded)
  - C25 model prediction from per-second engineering counters
    (PHO/Wide/Large/Dt/L_cycles + |mlat|), summed over 6 dets per box

Run:
    python3 scripts/plot_burst_20260601.py [--bin 0.1] [--before 50] [--after 350]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)
TRIGGER_UTC = "2026-06-01T19:12:49.900"
EPOCH_ARG = "2026-06-01T19"
BOXES = ["A", "B", "C"]
BOX_DET_LO = {"A": 0, "B": 6, "C": 12}
BOX_DET_HI = {"A": 5, "B": 11, "C": 17}
BOX_ID = {"A": 0, "B": 1, "C": 2}
L_CYCLES_TO_SEC = 16e-6
MET_CORRECTION = 4.0   # 1B Time → 1K MET (matches extract_per_sec_day.py)

BOX_PORTS = {"A": "0766", "B": "1009", "C": "1781"}
DATA_DIR = Path("/Users/skyair/Developer/ihep/blink/data")


def parse_met(s: str) -> float:
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H"):
        try:
            return (datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                    - MET_EPOCH).total_seconds()
        except ValueError:
            continue
    raise ValueError(f"bad time {s}")


def met_to_utc(met: float) -> str:
    return (MET_EPOCH + timedelta(seconds=met)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


# ─────────────────────────────────────────────────────────────────────────────
# CLI calls
# ─────────────────────────────────────────────────────────────────────────────
def run_cli(args: list[str]) -> str:
    env = os.environ.copy()
    env["HXMT_1B_DIR"] = "data/1B"
    env["HXMT_1K_DIR"] = "data/1K"
    p = subprocess.run(["./target/release/blink", *args],
                       capture_output=True, text=True, env=env, cwd=os.getcwd())
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise RuntimeError(f"blink_cli {args} failed")
    return p.stdout


def detect_resets(epoch: str, trigger_utc: str, before: float, after: float):
    """Return {box: [(start_met, stop_met, n_lost), ...]}."""
    out = run_cli(["sat", epoch, "detect", trigger_utc,
                   "--before", str(before), "--after", str(after)])
    resets = {b: [] for b in BOXES}
    for line in out.strip().splitlines():
        p = line.split(",")
        if len(p) < 8 or p[0] == "box" or p[1] != "FifoReset":
            continue
        resets[p[0]].append((float(p[2]), float(p[3]), int(p[7])))
    return resets


def reconstruct_box(epoch: str, box: str, trigger_utc: str,
                    before: float, after: float):
    out = run_cli(["sat", epoch, "--box", box.lower(), "reconstruct",
                   trigger_utc,
                   "--before", str(before), "--after", str(after)])
    obs, fill = [], []
    for line in out.strip().splitlines():
        p = line.split(",")
        if len(p) < 3 or p[0] == "box":
            continue
        if p[1] == "EVT":
            obs.append(float(p[2]))
        elif p[1] == "FILL_GAP":
            fill.append(float(p[2]))
    return np.array(obs), np.array(fill)


def solve_box_per_det(epoch: str, box: str, trigger_met: float,
                      before: float, after: float):
    """Return list[6] of MET arrays, one per detector 0-5."""
    out = run_cli(["sat", epoch, "--box", box.lower(), "solve",
                   f"{trigger_met:.3f}",
                   "--before", str(before), "--after", str(after)])
    per_det = [[] for _ in range(6)]
    for line in out.strip().splitlines():
        p = line.split(",")
        if len(p) < 5 or p[0] == "box" or p[1] != "EVT":
            continue
        d = int(p[4])
        if 0 <= d < 6:
            per_det[d].append(float(p[2]))
    return [np.array(lst) for lst in per_det]


# ─────────────────────────────────────────────────────────────────────────────
# 1K reference + C25 model from engineering data
# ─────────────────────────────────────────────────────────────────────────────
def load_1k_evt(t_ref: float, before: float, after: float):
    """Return dict {box: events MET[]} from 1K HE-Evt."""
    f = ("data/1K/Y202606/20260601-3274/"
         "HXMT_20260601T19_HE-Evt_FFFFFF_V1_1K.FITS")
    with fits.open(f, memmap=True) as hdul:
        d = hdul["Events"].data
        t = d["Time"].astype(np.float64)
        det = d["Det_ID"].astype(np.int8)
    mask_t = (t >= t_ref - before) & (t <= t_ref + after)
    t = t[mask_t]; det = det[mask_t]
    out = {}
    for b in BOXES:
        m = (det >= BOX_DET_LO[b]) & (det <= BOX_DET_HI[b])
        out[b] = t[m]
    return out


def find_1b_eng(box: str, hour_path: str = "data/1B/2026/20260601") -> Path:
    port = BOX_PORTS[box]
    cand = sorted(Path(f"{hour_path}/{port}").glob(
        f"HXMT_1B_{port}_20260601T190000_*.fits"))
    return cand[-1]


def read_he_eng(path: Path) -> dict:
    with fits.open(path, memmap=False) as f:
        d = f["HE_Eng"].data
        col_names = set(f["HE_Eng"].columns.names)
        has_per_box = all(f"Cnt_PHODet_{i}" in col_names for i in range(6))
        path_str = str(path)
        if "0766" in path_str:   start_idx = 0
        elif "1009" in path_str: start_idx = 6
        elif "1781" in path_str: start_idx = 12
        else: raise ValueError(path_str)

        def per_det(base):
            if has_per_box:
                cols = [d[f"{base}_{i}"].astype(np.int32) for i in range(6)]
            else:
                cols = [d[f"{base}_{start_idx + i}"].astype(np.int32)
                        for i in range(6)]
            return np.stack(cols, axis=1)

        offset = int(d["UTC_Last_Bdc"][0]) - int(d["sTime_Last_Bdc"][0])
        return {
            "Time":      d["Time"].astype(np.int64),
            "L_cycles":  d["Length_Time_Cycle"].astype(np.int32),
            "PHO":       per_det("Cnt_PHODet"),
            "Wide":      per_det("Cnt_CsI_PHODet"),
            "Large":     per_det("Cnt_LargeEvt"),
            "Dt":        per_det("DeadTime_PHODet"),
            "offset":    offset,
        }


def read_orbit_1k(path: str) -> dict:
    with fits.open(path, memmap=False) as f:
        d = f["Orbit"].data
        t = np.round(d["Time"].astype(np.float64)).astype(np.int64)
        return {"Time": t, "Lat": d["Lat"].astype(float),
                "Lon": d["Lon"].astype(float)}


def sigm(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def unwrap_v2(pho, large, wide, sci, lc, dt, C):
    L  = lc * L_CYCLES_TO_SEC
    lf = 1.0 - dt / lc
    predicted = pho - (wide + (sci + C) * L) / lf
    n = np.maximum(np.round((predicted - large) / 1024.0).astype(int), 0)
    maxa = pho - wide
    lcr  = large + n * 1024.0
    over = lcr > maxa
    if over.any():
        nmax = np.maximum(np.floor((maxa - large) / 1024.0).astype(int), 0)
        lcr = large + np.where(over, nmax, n) * 1024.0
    return lcr


def C_model_c25(mlat, t_yrs, box_str, det, P):
    A_i = np.array(P["a_det"])
    A = A_i[BOX_ID[box_str] * 6 + det]
    sm = sigm((np.abs(mlat) - P["mu_m"]) / P["k_m"])
    st = sigm((t_yrs - P["mu_t"]) / P["k_t"])
    g  = 1.0 + P["alpha"] * sm
    return A * g * (1.0 - P["amp0"] * g * st) + P["C0"]


def c25_predict_per_box(eng: dict, sci_count_per_det: np.ndarray,
                        lat: np.ndarray, lon: np.ndarray,
                        box: str, t_ref_yrs: float, params: dict,
                        interp: RegularGridInterpolator) -> np.ndarray:
    """Return per-box Sci_rec rate (cnt/s) at each engineering second.

    Inputs (all shape (n_sec,) or (n_sec, 6)):
        eng: from read_he_eng (PHO/Wide/Large/Dt all (n_sec,6))
        sci_count_per_det: (n_sec, 6) — observed Sci_1s per det
        lat, lon: (n_sec,) — geodetic position
    """
    n_sec = len(eng["Time"])
    lc = eng["L_cycles"].astype(float)             # (n_sec,)
    L  = lc * L_CYCLES_TO_SEC
    # geomagnetic |mlat| from AACGM grid
    pts = np.column_stack([lat, lon])
    abs_mlat = np.abs(interp(pts))
    abs_mlat = np.where(np.isnan(abs_mlat), 0.0, abs_mlat)

    rec_total = np.zeros(n_sec, dtype=float)
    for det in range(6):
        pho   = eng["PHO"][:, det].astype(float)
        wide  = eng["Wide"][:, det].astype(float)
        large = eng["Large"][:, det].astype(float)
        dtv   = eng["Dt"][:, det].astype(float)
        sci   = sci_count_per_det[:, det].astype(float)
        lf = 1.0 - dtv / lc

        C_per = C_model_c25(abs_mlat, t_ref_yrs, box, det, params)
        large_v3 = unwrap_v2(pho, large, wide, sci, lc, dtv, C_per)
        max_le = pho - (sci * L + wide) / lf
        n3 = np.round((large_v3 - large) / 1024).astype(int)
        nmax = np.maximum(np.floor((max_le - large) / 1024.0).astype(int), 0)
        large_final = large + np.where(n3 > nmax, nmax, n3) * 1024.0
        sci_rec = (pho - large_final) * lf / L - wide / L - C_per
        rec_total += sci_rec     # cnt/s per det → sum to box
    return rec_total


def aggregate_evt_to_sec(evt_t: np.ndarray, evt_det: np.ndarray,
                        met_floats: np.ndarray, box_idx: int) -> np.ndarray:
    """Per-second per-det Sci count, shape (n_sec, 6)."""
    out = np.zeros((len(met_floats), 6), dtype=np.int32)
    for det in range(6):
        det_global = box_idx * 6 + det
        m = evt_det == det_global
        t = evt_t[m]
        for i, t0 in enumerate(met_floats):
            i0 = int(np.searchsorted(t, t0, side="left"))
            i1 = int(np.searchsorted(t, t0 + 1.0, side="left"))
            out[i, det] = i1 - i0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=float, default=50.0)
    ap.add_argument("--after",  type=float, default=350.0)
    ap.add_argument("--bin",    type=float, default=0.2)
    ap.add_argument("--c25-json", default="/tmp/per_det_25param.json")
    ap.add_argument("--aacgm-grid", default="n_below_study/aacgm_grid_2020.npz")
    ap.add_argument("-o", "--output",
                    default="plots/burst_20260601_T191249_diagnostic.png")
    args = ap.parse_args()

    Path("plots").mkdir(exist_ok=True)

    t_ref = parse_met(TRIGGER_UTC)
    print(f"trigger {TRIGGER_UTC}  →  MET {t_ref:.3f}", file=sys.stderr)

    # 1) Detect FIFO resets
    print("Running detect ...", file=sys.stderr)
    resets = detect_resets(EPOCH_ARG, TRIGGER_UTC, args.before, args.after)
    for b in BOXES:
        print(f"  Box {b}: {len(resets[b])} resets, "
              f"n_lost={sum(r[2] for r in resets[b])}", file=sys.stderr)

    # 2) Reconstruct per box
    print("Running reconstruct ...", file=sys.stderr)
    obs = {}; fill = {}
    for b in BOXES:
        obs[b], fill[b] = reconstruct_box(EPOCH_ARG, b, TRIGGER_UTC,
                                          args.before, args.after)
        print(f"  Box {b}: obs={len(obs[b]):,}  fill={len(fill[b]):,}",
              file=sys.stderr)

    # 3) Load 1K reference
    print("Loading 1K HE-Evt ...", file=sys.stderr)
    m1k = load_1k_evt(t_ref, args.before, args.after)

    # 4) C25 model from engineering data
    print("C25 model from engineering data ...", file=sys.stderr)
    params = json.load(open(args.c25_json))
    grid = np.load(args.aacgm_grid)
    # AACGM grid keys: 'lat'/'lon' (1D edges) + 'aacgm_lat' (2D)
    print(f"  aacgm grid keys: {list(grid.keys())}", file=sys.stderr)
    interp = RegularGridInterpolator(
        (grid["lat_grid"], grid["lon_grid"]), grid["mlat"],
        bounds_error=False, fill_value=np.nan)

    orbit = read_orbit_1k(
        "data/1K/Y202606/20260601-3274/HXMT_20260601T19_Orbit_FFFFFF_V1_1K.FITS")
    orbit_idx = {int(t): i for i, t in enumerate(orbit["Time"])}

    # Days since 2017-06-22 (training start) → fractional years
    t_train_start = (datetime(2017, 6, 22, tzinfo=timezone.utc) -
                     MET_EPOCH).total_seconds()
    t_yrs = (t_ref - t_train_start) / (365.25 * 86400)
    print(f"  t (yrs since 2017-06-22): {t_yrs:.3f}", file=sys.stderr)

    full_evt = fits.open(
        "data/1K/Y202606/20260601-3274/"
        "HXMT_20260601T19_HE-Evt_FFFFFF_V1_1K.FITS", memmap=True)
    full_t = full_evt["Events"].data["Time"].astype(np.float64)
    full_det = full_evt["Events"].data["Det_ID"].astype(np.int8)

    c25_per_box = {}
    c25_t = {}
    for b in BOXES:
        eng_path = find_1b_eng(b)
        eng = read_he_eng(eng_path)
        met_floats = eng["Time"] + eng["offset"] + MET_CORRECTION
        # restrict to plot window for speed
        win_mask = (met_floats >= t_ref - args.before - 5) & \
                   (met_floats <= t_ref + args.after + 5)
        eng_w = {k: (v[win_mask] if k != "offset" else v)
                 for k, v in eng.items()}
        met_w = met_floats[win_mask]
        met_w_int = np.floor(met_w).astype(np.int64)

        # Lat/Lon per sec from orbit
        lat = np.full(len(met_w), np.nan)
        lon = np.full(len(met_w), np.nan)
        for i, s in enumerate(met_w_int):
            j = orbit_idx.get(int(s))
            if j is not None:
                lat[i] = orbit["Lat"][j]; lon[i] = orbit["Lon"][j]

        # Sci_1s per (sec, det)
        sci_per = aggregate_evt_to_sec(full_t, full_det, met_w, BOX_ID[b])
        rec = c25_predict_per_box(eng_w, sci_per, lat, lon, b,
                                  t_yrs, params, interp)
        c25_per_box[b] = rec
        c25_t[b] = met_w
        print(f"  Box {b}: C25 mean rate = {rec.mean():.0f} cnt/s",
              file=sys.stderr)

    full_evt.close()

    # ─────────────────────────────────────────────────────────────────────
    # Plot
    # ─────────────────────────────────────────────────────────────────────
    bin_w = args.bin
    t_min = t_ref - args.before; t_max = t_ref + args.after
    edges = np.arange(t_min, t_max + bin_w, bin_w)
    x = edges[:-1] - t_ref

    # Layout: 3 full-width box panels + 2 sub-second zoom rows below
    # (row 3 = per-box overlay, row 4 = Box-B per-detector breakdown).
    fig = plt.figure(figsize=(22, 19))
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1.05, 1.05],
                          hspace=0.32, wspace=0.15,
                          top=0.965, bottom=0.04, left=0.05, right=0.99)
    axes = [fig.add_subplot(gs[i, :]) for i in range(3)]
    axes[0].sharex(axes[2]); axes[1].sharex(axes[2])

    for ax, b in zip(axes, BOXES):
        r_obs  = np.histogram(obs[b], bins=edges)[0] / bin_w if len(obs[b]) else np.zeros(len(x))
        r_fill = np.histogram(fill[b], bins=edges)[0] / bin_w if len(fill[b]) else np.zeros(len(x))
        r_total = r_obs + r_fill
        r_1k = np.histogram(m1k[b], bins=edges)[0] / bin_w if len(m1k[b]) else np.zeros(len(x))

        # FIFO reset bands. Each reset is ~15ms wide so invisible on a 400s
        # axis — dilate to a minimum visual width of 1.5 s so the user can see
        # WHERE the reset cluster sits. The zoom-in panel below uses true width.
        MIN_VIS = 1.5
        if resets[b]:
            # merge resets within 5s of each other so clusters render as one band
            merged = []
            cur_s, cur_e = resets[b][0][0], resets[b][0][1]
            for rs, re_, _ in resets[b][1:]:
                if rs - cur_e < 5.0:
                    cur_e = max(cur_e, re_)
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = rs, re_
            merged.append((cur_s, cur_e))
            for cs, ce in merged:
                cw = ce - cs
                if cw < MIN_VIS:
                    pad = (MIN_VIS - cw) / 2
                    cs, ce = cs - pad, ce + pad
                ax.axvspan(cs - t_ref, ce - t_ref, color="#F4A460",
                           alpha=0.45, zorder=0, linewidth=0)
                # vertical sentinel line at exact center for sub-bin accuracy
                ax.axvline((cs + ce) / 2 - t_ref, color="#B2182B",
                           lw=0.6, alpha=0.7, zorder=0.5)

        ax.fill_between(x, r_1k, step="post", color="#D9D9D9", alpha=0.9, zorder=1)
        ax.step(x, r_1k, where="post", color="#888", lw=0.6,
                label=f"1K reference ({len(m1k[b]):,})", zorder=2)
        ax.step(x, r_obs, where="post", color="#2166AC", lw=0.7,
                label=f"1B observed ({len(obs[b]):,})", zorder=3)
        if len(fill[b]) > 0:
            # Only paint bins that actually received fills; elsewhere the edge
            # would shadow the 1B-observed line for no information.
            mask = r_fill > 0
            ax.fill_between(x, r_obs, r_total, where=mask, step="post",
                            color="#F4A582", alpha=0.7, edgecolor="#B2182B",
                            linewidth=0.8, zorder=4,
                            label=f"1B + gap-fill ({len(fill[b]):,})")
        ax.step(c25_t[b] - t_ref, c25_per_box[b], where="post",
                color="#762A83", lw=1.6, alpha=0.95, zorder=5,
                label="C25 model from engineering data (1 s steps)")

        ax.set_ylabel(f"Box {b}  rate (cnt/s)", fontsize=13)
        ax.grid(alpha=0.15)
        ax.set_ylim(bottom=0)

        # Custom legend: stack the dilated-band patch and the exact-center line
        # into one composite entry so the meaning of both visual elements is
        # discoverable from the legend alone.
        handles, labels = ax.get_legend_handles_labels()
        if resets[b]:
            n_lost = sum(r[2] for r in resets[b])
            fifo_handle = (
                Patch(facecolor="#F4A460", alpha=0.45),
                Line2D([0], [0], color="#B2182B", lw=0.8, alpha=0.9),
            )
            handles.append(fifo_handle)
            labels.append(
                f"FIFO reset ({len(resets[b])}×, ~{n_lost} lost): "
                f"band = ±0.75 s dilated, line = true centre"
            )
        ax.legend(handles, labels, loc="lower left", fontsize=10,
                  framealpha=0.9,
                  handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})

    axes[-1].set_xlabel(f"Time − T₀ (s)    "
                        f"[T₀ = {TRIGGER_UTC} UTC,  MET {t_ref:.3f}]",
                        fontsize=13)

    # ── Sub-second zooms: one panel per FIFO-reset cluster, showing what
    #    happened at the time the FIFO actually overflowed.
    box_colors = {"A": "#1B9E77", "B": "#D95F02", "C": "#7570B3"}

    # cluster Box-B resets that fall within 1 s of each other
    clusters = []
    if resets["B"]:
        cur = [resets["B"][0]]
        for r in resets["B"][1:]:
            if r[0] - cur[-1][1] < 1.0:
                cur.append(r)
            else:
                clusters.append(cur); cur = [r]
        clusters.append(cur)

    burst_windows = []
    for cl in clusters:
        c_mid = (cl[0][0] + cl[-1][1]) / 2
        c_half = max(0.15, (cl[-1][1] - cl[0][0]) / 2 + 0.10)
        n_lost = sum(r[2] for r in cl)
        burst_windows.append((c_mid - c_half, c_mid + c_half,
                              f"T+{c_mid-t_ref:.2f} s  "
                              f"({len(cl)}× FIFO reset, ~{n_lost} lost)", cl))

    bin_sub = 0.002      # 2 ms
    # Pre-compute Box-B per-det events for each cluster window. Run solve
    # once per cluster with a tight window matching the zoom extent.
    per_det_by_cluster: list[list[np.ndarray]] = []
    for w_start, w_end, _title, _cl in burst_windows[:2]:
        centre = (w_start + w_end) / 2
        half = (w_end - w_start) / 2
        per_det_by_cluster.append(
            solve_box_per_det(EPOCH_ARG, "B", centre, half, half)
        )

    for col, (w_start, w_end, title, cluster) in enumerate(burst_windows[:2]):
        ax_z = fig.add_subplot(gs[3, col])
        edges_z = np.arange(w_start, w_end, bin_sub)
        x_z = edges_z[:-1] - t_ref
        # Compute Box B observed and gap-fill in this window up-front so we
        # can replace the observed step with the reconstructed (obs+fill) step
        # inside FIFO-reset bins — otherwise the gap-fill draws from 0 and
        # appears visually disconnected from the Box-B trace.
        r_obs_B  = np.histogram(obs["B"],  bins=edges_z)[0] / bin_sub
        r_fill_B = np.histogram(fill["B"], bins=edges_z)[0] / bin_sub
        r_recon_B = r_obs_B + r_fill_B

        for b in BOXES:
            if b == "B":
                r_b = r_recon_B if cluster is not None else r_obs_B
            else:
                r_b = np.histogram(obs[b], bins=edges_z)[0] / bin_sub
            ax_z.step(x_z, r_b, where="post", color=box_colors[b], lw=0.85,
                      alpha=0.95, label=f"Box {b}", zorder=3)

        if cluster is not None:
            if (r_fill_B > 0).any():
                # Shade gap-fill contribution as the slice between observed
                # (often 0 inside the gap) and reconstructed Box-B step.
                ax_z.fill_between(x_z, r_obs_B, r_recon_B,
                                  where=r_fill_B > 0, step="post",
                                  color="#F4A582", alpha=0.75,
                                  edgecolor="#B2182B", linewidth=0.6,
                                  zorder=3.5, label="Box B gap-fill")
            for rs, re_, _ in cluster:
                ax_z.axvspan(rs - t_ref, re_ - t_ref, color="#F4A460",
                             alpha=0.45, zorder=0, linewidth=0)
        ax_z.set_title(f"{title}  ({int(bin_sub*1000)} ms bin)", fontsize=10)
        ax_z.set_xlabel("T − T₀ (s)", fontsize=10)
        ax_z.set_ylabel("rate (cnt/s/box)", fontsize=10)
        ax_z.grid(alpha=0.2); ax_z.set_ylim(bottom=0)
        ax_z.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # ── Row 4: Box-B 6-detector breakdown, same time windows. FIFO-reset
    # gaps stay empty (per-det reconstruction is not produced — fill-gap
    # interpolation works at box level only).
    det_palette = ["#1F77B4", "#FF7F0E", "#2CA02C",
                   "#D62728", "#9467BD", "#8C564B"]
    for col, (w_start, w_end, title, cluster) in enumerate(burst_windows[:2]):
        ax_d = fig.add_subplot(gs[4, col])
        edges_z = np.arange(w_start, w_end, bin_sub)
        x_z = edges_z[:-1] - t_ref
        per_det = per_det_by_cluster[col]
        for d in range(6):
            r_d = np.histogram(per_det[d], bins=edges_z)[0] / bin_sub
            ax_d.step(x_z, r_d, where="post", color=det_palette[d],
                      lw=0.85, alpha=0.95, label=f"Det {d}", zorder=3)
        if cluster is not None:
            for rs, re_, _ in cluster:
                ax_d.axvspan(rs - t_ref, re_ - t_ref, color="#F4A460",
                             alpha=0.45, zorder=0, linewidth=0)
        ax_d.set_title(f"Box B per-det  ·  {title}  "
                       f"({int(bin_sub*1000)} ms bin)", fontsize=10)
        ax_d.set_xlabel("T − T₀ (s)", fontsize=10)
        ax_d.set_ylabel("rate (cnt/s/det)", fontsize=10)
        ax_d.grid(alpha=0.2); ax_d.set_ylim(bottom=0)
        ax_d.legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=2)

    fig.suptitle("HXMT/HE  burst at 2026-06-01T19:12:49.900  —  "
                 "saturation diagnostic",
                 fontsize=15, fontweight="bold")
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
