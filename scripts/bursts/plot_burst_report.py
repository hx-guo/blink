#!/usr/bin/env python3
"""Plot saturation diagnostic from a `blink sat report` pack.

Three full-width panels (Box A/B/C) show 1K reference, 1B observed, 1B+gap-fill,
C25 engineering model and FIFO reset bands. Optional zoom rows show per-reset-
cluster activity and per-detector breakdown for the saturated box.

Usage:
    ./target/release/blink sat report <TRIGGER> --before <s> --after <s> -o <pack>
    python3 scripts/bursts/plot_burst_report.py --pack <pack> -o <PNG>
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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
BOXES = ["A", "B", "C"]
BOX_ID = {"A": 0, "B": 1, "C": 2}
L_CYCLES_TO_SEC = 16e-6
MET_CORRECTION = 4.0


def load_pack_csv(path: Path, **kw):
    """Read CSV with header, return 2D array (N, ncols) or None if empty."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2, **kw)
        except (StopIteration, ValueError):
            return None
    return a if a.size else None


def load_box_events(pack: Path, box: str):
    """Return (obs_met, obs_det_local, fill_met, k1_met, resets_array)."""
    d = pack / f"box_{box.lower()}"

    # events_obs.csv columns: met,channel,det_id,pkt_idx,evt_idx,is_second
    obs = load_pack_csv(d / "events_obs.csv")
    if obs is None:
        obs_met = np.array([])
        obs_det = np.array([], dtype=int)
    else:
        mask = obs[:, 5] == 0
        obs_met = obs[mask, 0]
        obs_det = obs[mask, 2].astype(int)

    rec = load_pack_csv(d / "events_rec.csv")
    fill_met = rec[:, 0] if rec is not None else np.array([])

    k1 = load_pack_csv(d / "events_1k.csv")
    k1_met = k1[:, 0] if k1 is not None else np.array([])

    # resets.csv: start_met,stop_met,gap_s,prev_pkt_idx,next_pkt_idx,n_lost,cluster_id
    resets = load_pack_csv(d / "resets.csv")
    if resets is None:
        resets = np.zeros((0, 7))
    return obs_met, obs_det, fill_met, k1_met, resets


def read_he_eng(path: str, box: str) -> dict:
    with fits.open(path, memmap=False) as f:
        d = f["HE_Eng"].data
        col_names = set(f["HE_Eng"].columns.names)
        has_per_box = all(f"Cnt_PHODet_{i}" in col_names for i in range(6))
        start_idx = 0 if has_per_box else {"A": 0, "B": 6, "C": 12}[box]

        def per_det(base):
            if has_per_box:
                cols = [d[f"{base}_{i}"].astype(np.int32) for i in range(6)]
            else:
                cols = [d[f"{base}_{start_idx + i}"].astype(np.int32)
                        for i in range(6)]
            return np.stack(cols, axis=1)

        offset = int(d["UTC_Last_Bdc"][0]) - int(d["sTime_Last_Bdc"][0])
        return {
            "Time":     d["Time"].astype(np.int64),
            "L_cycles": d["Length_Time_Cycle"].astype(np.int32),
            "PHO":      per_det("Cnt_PHODet"),
            "Wide":     per_det("Cnt_CsI_PHODet"),
            "Large":    per_det("Cnt_LargeEvt"),
            "Dt":       per_det("DeadTime_PHODet"),
            "offset":   offset,
        }


def read_orbit(path: str) -> dict:
    with fits.open(path, memmap=False) as f:
        d = f["Orbit"].data
        t = np.round(d["Time"].astype(np.float64)).astype(np.int64)
        return {"Time": t, "Lat": d["Lat"].astype(float),
                "Lon": d["Lon"].astype(float)}


def sigm(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def unwrap_v2(pho, large, wide, sci, lc, dt, C):
    L = lc * L_CYCLES_TO_SEC
    lf = 1.0 - dt / lc
    predicted = pho - (wide + (sci + C) * L) / lf
    n = np.maximum(np.round((predicted - large) / 1024.0).astype(int), 0)
    maxa = pho - wide
    lcr = large + n * 1024.0
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
    g = 1.0 + P["alpha"] * sm
    return A * g * (1.0 - P["amp0"] * g * st) + P["C0"]


def c25_predict_per_box(eng, sci_per_det, lat, lon, box, t_yrs, params, interp):
    n_sec = len(eng["Time"])
    lc = eng["L_cycles"].astype(float)
    L = lc * L_CYCLES_TO_SEC
    pts = np.column_stack([lat, lon])
    abs_mlat = np.abs(interp(pts))
    abs_mlat = np.where(np.isnan(abs_mlat), 0.0, abs_mlat)

    rec_total = np.zeros(n_sec, dtype=float)
    for det in range(6):
        pho   = eng["PHO"][:, det].astype(float)
        wide  = eng["Wide"][:, det].astype(float)
        large = eng["Large"][:, det].astype(float)
        dtv   = eng["Dt"][:, det].astype(float)
        sci   = sci_per_det[:, det].astype(float)
        lf = 1.0 - dtv / lc

        C_per = C_model_c25(abs_mlat, t_yrs, box, det, params)
        large_v3 = unwrap_v2(pho, large, wide, sci, lc, dtv, C_per)
        max_le = pho - (sci * L + wide) / lf
        n3 = np.round((large_v3 - large) / 1024).astype(int)
        nmax = np.maximum(np.floor((max_le - large) / 1024.0).astype(int), 0)
        large_final = large + np.where(n3 > nmax, nmax, n3) * 1024.0
        sci_rec = (pho - large_final) * lf / L - wide / L - C_per
        rec_total += sci_rec
    return rec_total


def aggregate_evt_to_sec(evt_t, evt_det, met_floats):
    """Per-second per-det count from event list (det indexed 0..5 within box)."""
    out = np.zeros((len(met_floats), 6), dtype=np.int32)
    for det in range(6):
        m = evt_det == det
        t = np.sort(evt_t[m])
        for i, t0 in enumerate(met_floats):
            i0 = int(np.searchsorted(t, t0, side="left"))
            i1 = int(np.searchsorted(t, t0 + 1.0, side="left"))
            out[i, det] = i1 - i0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True,
                    help="Pack directory from `blink sat report`")
    ap.add_argument("--bin", type=float, default=0.2,
                    help="Box-panel histogram bin width in seconds")
    ap.add_argument("--bin-zoom", type=float, default=0.002,
                    help="Zoom panel bin width in seconds")
    ap.add_argument("--c25-json", default="/tmp/per_det_25param.json",
                    help="C25 model parameters JSON")
    ap.add_argument("--aacgm-grid", default="n_below_study/aacgm_grid_2020.npz",
                    help="AACGM geomagnetic-lat grid NPZ")
    ap.add_argument("--zoom-box", default="auto",
                    help="Per-det zoom box (A/B/C/auto/none). "
                         "auto = pick box with most resets.")
    ap.add_argument("--no-c25", action="store_true",
                    help="Skip C25 engineering-model overlay")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    with open(args.pack / "manifest.json") as f:
        manifest = json.load(f)
    t_ref = manifest["trigger_met"]
    trigger_label = manifest["trigger_utc"]
    before = manifest["before_s"]
    after = manifest["after_s"]

    print(f"Loading pack {args.pack}", file=sys.stderr)
    obs = {}; obs_det = {}; fill = {}; m1k = {}; resets = {}
    for b in BOXES:
        obs[b], obs_det[b], fill[b], m1k[b], resets[b] = load_box_events(args.pack, b)
        print(f"  Box {b}: obs={len(obs[b]):,} fill={len(fill[b]):,} "
              f"1k={len(m1k[b]):,} resets={len(resets[b])}", file=sys.stderr)

    zb = args.zoom_box.lower()
    if zb == "auto":
        cand = max(BOXES, key=lambda b: len(resets[b]))
        zoom_box = cand if len(resets[cand]) else None
    elif zb == "none":
        zoom_box = None
    else:
        zoom_box = zb.upper()
        if zoom_box not in BOXES:
            sys.exit(f"--zoom-box must be A/B/C/auto/none, got {args.zoom_box}")

    c25_t: dict[str, np.ndarray] = {}
    c25_rate: dict[str, np.ndarray] = {}
    if not args.no_c25:
        try:
            print("Computing C25 model...", file=sys.stderr)
            params = json.load(open(args.c25_json))
            grid = np.load(args.aacgm_grid)
            interp = RegularGridInterpolator(
                (grid["lat_grid"], grid["lon_grid"]), grid["mlat"],
                bounds_error=False, fill_value=np.nan)

            orbit_path = manifest.get("level_1k_orbit")
            if not orbit_path:
                raise RuntimeError("manifest has no level_1k_orbit path")
            orbit = read_orbit(orbit_path)
            orbit_idx = {int(t): i for i, t in enumerate(orbit["Time"])}

            t_train_start = (datetime(2017, 6, 22, tzinfo=timezone.utc)
                             - MET_EPOCH).total_seconds()
            t_yrs = (t_ref - t_train_start) / (365.25 * 86400)
            print(f"  t (yrs since 2017-06-22): {t_yrs:.3f}", file=sys.stderr)

            for b in BOXES:
                eng_path = manifest["level_1b_eng"][b]
                eng = read_he_eng(eng_path, b)
                met_floats = eng["Time"] + eng["offset"] + MET_CORRECTION
                win_mask = ((met_floats >= t_ref - before - 5)
                            & (met_floats <= t_ref + after + 5))
                eng_w = {k: (v[win_mask] if k != "offset" else v)
                         for k, v in eng.items()}
                met_w = met_floats[win_mask]
                met_w_int = np.floor(met_w).astype(np.int64)

                lat = np.full(len(met_w), np.nan)
                lon = np.full(len(met_w), np.nan)
                for i, s in enumerate(met_w_int):
                    j = orbit_idx.get(int(s))
                    if j is not None:
                        lat[i] = orbit["Lat"][j]
                        lon[i] = orbit["Lon"][j]

                sci_per = aggregate_evt_to_sec(obs[b], obs_det[b], met_w)
                rec = c25_predict_per_box(eng_w, sci_per, lat, lon, b,
                                          t_yrs, params, interp)
                c25_t[b] = met_w
                c25_rate[b] = rec
                print(f"  Box {b}: C25 mean rate = {rec.mean():.0f} cnt/s",
                      file=sys.stderr)
        except (FileNotFoundError, KeyError, RuntimeError) as e:
            print(f"  skipping C25 ({e})", file=sys.stderr)

    # ── Plot ──
    bin_w = args.bin
    t_min = t_ref - before; t_max = t_ref + after
    edges = np.arange(t_min, t_max + bin_w, bin_w)
    x = edges[:-1] - t_ref

    has_zooms = zoom_box is not None and len(resets[zoom_box]) > 0
    n_rows = 5 if has_zooms else 3
    height_ratios = [1, 1, 1, 1.05, 1.05] if has_zooms else [1, 1, 1]
    fig_h = 19 if has_zooms else 11

    fig = plt.figure(figsize=(22, fig_h))
    gs = fig.add_gridspec(n_rows, 2, height_ratios=height_ratios,
                          hspace=0.32, wspace=0.15,
                          top=0.965, bottom=0.04, left=0.05, right=0.99)
    axes = [fig.add_subplot(gs[i, :]) for i in range(3)]
    axes[0].sharex(axes[2]); axes[1].sharex(axes[2])

    for ax, b in zip(axes, BOXES):
        r_obs = (np.histogram(obs[b], bins=edges)[0] / bin_w
                 if len(obs[b]) else np.zeros(len(x)))
        r_fill = (np.histogram(fill[b], bins=edges)[0] / bin_w
                  if len(fill[b]) else np.zeros(len(x)))
        r_total = r_obs + r_fill
        r_1k = (np.histogram(m1k[b], bins=edges)[0] / bin_w
                if len(m1k[b]) else np.zeros(len(x)))

        MIN_VIS = 1.5
        rs_b = resets[b]
        if len(rs_b):
            rs_sorted = rs_b[np.argsort(rs_b[:, 0])]
            merged = []
            cur_s, cur_e = rs_sorted[0, 0], rs_sorted[0, 1]
            for r in rs_sorted[1:]:
                if r[0] - cur_e < 5.0:
                    cur_e = max(cur_e, r[1])
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = r[0], r[1]
            merged.append((cur_s, cur_e))
            for cs, ce in merged:
                cw = ce - cs
                if cw < MIN_VIS:
                    pad = (MIN_VIS - cw) / 2
                    cs, ce = cs - pad, ce + pad
                ax.axvspan(cs - t_ref, ce - t_ref, color="#F4A460",
                           alpha=0.45, zorder=0, linewidth=0)
                ax.axvline((cs + ce) / 2 - t_ref, color="#B2182B",
                           lw=0.6, alpha=0.7, zorder=0.5)

        ax.fill_between(x, r_1k, step="post", color="#D9D9D9",
                        alpha=0.9, zorder=1)
        ax.step(x, r_1k, where="post", color="#888", lw=0.6,
                label=f"1K reference ({len(m1k[b]):,})", zorder=2)
        ax.step(x, r_obs, where="post", color="#2166AC", lw=0.7,
                label=f"1B observed ({len(obs[b]):,})", zorder=3)
        if len(fill[b]) > 0:
            mask = r_fill > 0
            ax.fill_between(x, r_obs, r_total, where=mask, step="post",
                            color="#F4A582", alpha=0.7, edgecolor="#B2182B",
                            linewidth=0.8, zorder=4,
                            label=f"1B + gap-fill ({len(fill[b]):,})")
        if b in c25_rate:
            ax.step(c25_t[b] - t_ref, c25_rate[b], where="post",
                    color="#762A83", lw=1.6, alpha=0.95, zorder=5,
                    label="C25 model from engineering data (1 s steps)")

        ax.set_ylabel(f"Box {b}  rate (cnt/s)", fontsize=13)
        ax.grid(alpha=0.15)
        ax.set_ylim(bottom=0)

        handles, labels = ax.get_legend_handles_labels()
        if len(rs_b):
            n_lost = int(rs_b[:, 5].sum())
            fifo_handle = (
                Patch(facecolor="#F4A460", alpha=0.45),
                Line2D([0], [0], color="#B2182B", lw=0.8, alpha=0.9),
            )
            handles.append(fifo_handle)
            labels.append(f"FIFO reset ({len(rs_b)}×, ~{n_lost} lost): "
                          f"band = ±0.75 s dilated, line = true centre")
        ax.legend(handles, labels, loc="lower left", fontsize=10,
                  framealpha=0.9,
                  handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})

    axes[-1].set_xlabel(
        f"Time − T₀ (s)    [T₀ = {trigger_label} UTC,  MET {t_ref:.3f}]",
        fontsize=13)

    if has_zooms:
        box_colors = {"A": "#1B9E77", "B": "#D95F02", "C": "#7570B3"}
        det_palette = ["#1F77B4", "#FF7F0E", "#2CA02C",
                       "#D62728", "#9467BD", "#8C564B"]

        rs_z = resets[zoom_box]
        clusters: dict[int, list] = {}
        for r in rs_z:
            cid = int(r[6])
            clusters.setdefault(cid, []).append(r)

        burst_windows = []
        for cid in sorted(clusters.keys()):
            cl = np.array(clusters[cid])
            c_mid = (cl[0, 0] + cl[-1, 1]) / 2
            c_half = max(0.15, (cl[-1, 1] - cl[0, 0]) / 2 + 0.10)
            n_lost = int(cl[:, 5].sum())
            burst_windows.append((c_mid - c_half, c_mid + c_half,
                                  f"T+{c_mid - t_ref:.2f} s  "
                                  f"({len(cl)}× FIFO reset, ~{n_lost} lost)",
                                  cl))

        bin_sub = args.bin_zoom
        for col, (w_start, w_end, title, cluster) in enumerate(burst_windows[:2]):
            edges_z = np.arange(w_start, w_end, bin_sub)
            x_z = edges_z[:-1] - t_ref

            ax_z = fig.add_subplot(gs[3, col])
            r_obs_Z  = np.histogram(obs[zoom_box],  bins=edges_z)[0] / bin_sub
            r_fill_Z = np.histogram(fill[zoom_box], bins=edges_z)[0] / bin_sub
            r_recon_Z = r_obs_Z + r_fill_Z

            for b in BOXES:
                r_b = r_recon_Z if b == zoom_box else \
                      np.histogram(obs[b], bins=edges_z)[0] / bin_sub
                ax_z.step(x_z, r_b, where="post", color=box_colors[b],
                          lw=0.85, alpha=0.95, label=f"Box {b}", zorder=3)

            if (r_fill_Z > 0).any():
                ax_z.fill_between(x_z, r_obs_Z, r_recon_Z,
                                  where=r_fill_Z > 0, step="post",
                                  color="#F4A582", alpha=0.75,
                                  edgecolor="#B2182B", linewidth=0.6,
                                  zorder=3.5, label=f"Box {zoom_box} gap-fill")
            for r in cluster:
                ax_z.axvspan(r[0] - t_ref, r[1] - t_ref, color="#F4A460",
                             alpha=0.45, zorder=0, linewidth=0)
            ax_z.set_title(f"{title}  ({int(bin_sub*1000)} ms bin)", fontsize=10)
            ax_z.set_xlabel("T − T₀ (s)", fontsize=10)
            ax_z.set_ylabel("rate (cnt/s/box)", fontsize=10)
            ax_z.grid(alpha=0.2); ax_z.set_ylim(bottom=0)
            ax_z.legend(fontsize=8, loc="upper right", framealpha=0.9)

            ax_d = fig.add_subplot(gs[4, col])
            for d in range(6):
                m_d = obs_det[zoom_box] == d
                t_d = obs[zoom_box][m_d]
                r_d = np.histogram(t_d, bins=edges_z)[0] / bin_sub
                ax_d.step(x_z, r_d, where="post", color=det_palette[d],
                          lw=0.85, alpha=0.95, label=f"Det {d}", zorder=3)
            for r in cluster:
                ax_d.axvspan(r[0] - t_ref, r[1] - t_ref, color="#F4A460",
                             alpha=0.45, zorder=0, linewidth=0)
            ax_d.set_title(f"Box {zoom_box} per-det  ·  {title}  "
                           f"({int(bin_sub*1000)} ms bin)", fontsize=10)
            ax_d.set_xlabel("T − T₀ (s)", fontsize=10)
            ax_d.set_ylabel("rate (cnt/s/det)", fontsize=10)
            ax_d.grid(alpha=0.2); ax_d.set_ylim(bottom=0)
            ax_d.legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=2)

    fig.suptitle(f"HXMT/HE  burst at {trigger_label}  —  saturation diagnostic",
                 fontsize=15, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Saved: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
