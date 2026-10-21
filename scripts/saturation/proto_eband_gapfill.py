#!/usr/bin/env python3
"""Energy-band gap-fill prototype: synthetic-gap validation on GRB 260226A.

Question: can the cross-box shape-function gap-fill (paper Algorithm 3)
recover *band-resolved* light curves, and how large is the bias of the
naive alternative (sampling filler channels from the reference boxes'
in-gap histogram)?

Method: in reset-free intervals of the 260226A pack, delete the target
box's events over [t0, t0+D) and recover the per-band lost counts from
the other two boxes. Ground truth = the deleted events.

Three estimators per injected gap, all in expectation form (no sampling
noise):
  M1 per-band  : R_j = mean_r [ k_r(j) * N_r(j, gap) ],
                 k_r(j) from +-0.5 s calibration windows, per band
  M2 ref-hist  : R_j = N_tot * p_j,  p_j = pooled reference in-gap band
                 fractions (the naive histogram-sampling idea)
  M3 calib-hist: R_j = N_tot * q_j,  q_j = target band fractions in the
                 calibration windows (time-independent-spectrum control)
where N_tot = mean_r [ k_r * N_r(gap) ] is the existing total-rate fill.

Usage:
  python3 scripts/saturation/proto_eband_gapfill.py [--pack data/pack_260226a]
      [--n-inj 150] [--seed 1] [-o out.csv]

Prints a bias/scatter table per (D, band, method) and writes per-injection
results to CSV for plotting.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

T0 = 446726270.0
BOXES = ("a", "b", "c")
# common raw-channel band edges = quartiles of the pre-burst plateau
# (box-level distributions agree across boxes to ~1 channel)
BAND_EDGES = np.array([20, 45, 77, 146, 276])
N_BANDS = len(BAND_EDGES) - 1
CALIB_HALF = 0.5           # calibration window on each side of the gap, s
CLEAN_MARGIN = 0.1         # extra clearance to the nearest real reset, s
GAP_DURATIONS = (0.030, 0.100, 0.300)


def load_box(pack: Path, box: str):
    met, ch = [], []
    with open(pack / f"box_{box}" / "events_obs.csv") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row[5] == "0":  # skip SEC rows
                met.append(float(row[0]))
                ch.append(int(row[1]))
    met = np.asarray(met)
    ch = np.asarray(ch, dtype=np.int64)
    ch = np.where(ch < 20, ch + 256, ch)  # pulse-height wrap semantics
    order = np.argsort(met)
    return met[order], ch[order]


def load_resets(pack: Path, box: str):
    out = []
    with open(pack / f"box_{box}" / "resets.csv") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            out.append((float(row[0]), float(row[1])))
    return out


def band_counts(met, ch, lo, hi):
    """Counts per band for events in [lo, hi)."""
    i, j = np.searchsorted(met, (lo, hi))
    return np.histogram(ch[i:j], bins=BAND_EDGES)[0].astype(float)


def recover(target, refs, t0, dur):
    """Run M1/M2/M3 for one injected gap. Returns dict of per-band arrays."""
    t_met, t_ch = target
    g_lo, g_hi = t0, t0 + dur
    windows = ((g_lo - CALIB_HALF, g_lo), (g_hi, g_hi + CALIB_HALF))

    # target calibration counts (gap events excluded by construction:
    # windows are disjoint from the gap)
    t_cal = sum(band_counts(t_met, t_ch, lo, hi) for lo, hi in windows)

    truth = band_counts(t_met, t_ch, g_lo, g_hi)

    m1 = np.zeros(N_BANDS)
    n_tot_parts = []
    ref_gap_pool = np.zeros(N_BANDS)
    n_valid = 0
    for r_met, r_ch in refs:
        r_cal = sum(band_counts(r_met, r_ch, lo, hi) for lo, hi in windows)
        r_gap = band_counts(r_met, r_ch, g_lo, g_hi)
        if r_cal.sum() == 0:
            continue
        # per-band calibration; guard empty bands (not expected here)
        with np.errstate(divide="ignore", invalid="ignore"):
            k_band = np.where(r_cal > 0, t_cal / np.maximum(r_cal, 1e-12), 0.0)
        m1 += k_band * r_gap
        k_tot = t_cal.sum() / r_cal.sum()
        n_tot_parts.append(k_tot * r_gap.sum())
        ref_gap_pool += r_gap
        n_valid += 1
    if n_valid == 0:
        return None
    m1 /= n_valid
    n_tot = float(np.mean(n_tot_parts))

    p_ref = ref_gap_pool / max(ref_gap_pool.sum(), 1.0)
    q_cal = t_cal / max(t_cal.sum(), 1.0)
    return {
        "truth": truth,
        "M1": m1,
        "M2": n_tot * p_ref,
        "M3": n_tot * q_cal,
        "t_cal": t_cal,
        "p_ref": p_ref,
        "q_cal": q_cal,
    }


def real_reset_spectra(pack: Path, boxes):
    """At every real reset gap: reference in-gap band fractions (what
    M1/M2 read) vs the target's calibration-window fractions (what M3
    assumes). Their difference is the error a time-independent-spectrum
    fill would make at real gaps; no ground truth needed."""
    rows = []
    for tgt in BOXES:
        refs = [boxes[b] for b in BOXES if b != tgt]
        t_met, t_ch = boxes[tgt]
        for g_lo, g_hi in load_resets(pack, tgt):
            windows = ((g_lo - CALIB_HALF, g_lo), (g_hi, g_hi + CALIB_HALF))
            t_cal = sum(band_counts(t_met, t_ch, lo, hi) for lo, hi in windows)
            pool = np.zeros(N_BANDS)
            for r_met, r_ch in refs:
                pool += band_counts(r_met, r_ch, g_lo, g_hi)
            if pool.sum() < 50 or t_cal.sum() < 50:
                continue
            p = pool / pool.sum()
            q = t_cal / t_cal.sum()
            for j in range(N_BANDS):
                rows.append({
                    "target": tgt, "t0_rel": g_lo - T0,
                    "dur_ms": (g_hi - g_lo) * 1e3, "band": j,
                    "p_ref": p[j], "q_cal": q[j],
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=Path("data/pack_260226a"))
    ap.add_argument("--n-inj", type=int, default=150,
                    help="injections per (duration, target box)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-o", "--out", type=Path,
                    default=Path("data/proto_eband_results.csv"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    boxes = {b: load_box(args.pack, b) for b in BOXES}
    all_resets = [iv for b in BOXES for iv in load_resets(args.pack, b)]
    for b in BOXES:
        print(f"box {b}: {len(boxes[b][0])} events")

    t_min = max(boxes[b][0][0] for b in BOXES)
    t_max = min(boxes[b][0][-1] for b in BOXES)

    def clean(lo, hi):
        return all(not (s < hi and e > lo) for s, e in all_resets)

    rows = []
    for dur in GAP_DURATIONS:
        for tgt in BOXES:
            refs = [boxes[b] for b in BOXES if b != tgt]
            n_done = n_try = 0
            while n_done < args.n_inj and n_try < args.n_inj * 50:
                n_try += 1
                t0 = rng.uniform(t_min + CALIB_HALF + CLEAN_MARGIN,
                                 t_max - dur - CALIB_HALF - CLEAN_MARGIN)
                if not clean(t0 - CALIB_HALF - CLEAN_MARGIN,
                             t0 + dur + CALIB_HALF + CLEAN_MARGIN):
                    continue
                res = recover(boxes[tgt], refs, t0, dur)
                if res is None or res["truth"].sum() < 10:
                    continue
                if (res["t_cal"] < 100).any():
                    print(f"warn: thin calib band, t0={t0 - T0:+.3f} "
                          f"counts={res['t_cal']}")
                for j in range(N_BANDS):
                    rows.append({
                        "dur_ms": dur * 1e3, "target": tgt,
                        "t0_rel": t0 - T0, "band": j,
                        "truth": res["truth"][j],
                        "M1": res["M1"][j], "M2": res["M2"][j],
                        "M3": res["M3"][j],
                        "p_ref": res["p_ref"][j], "q_cal": res["q_cal"][j],
                    })
                n_done += 1
            print(f"D={dur*1e3:.0f}ms target={tgt}: "
                  f"{n_done} injections ({n_try} tried)")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")

    rr = real_reset_spectra(args.pack, boxes)
    rr_out = args.out.with_name(args.out.stem + "_real_resets.csv")
    with open(rr_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0].keys()))
        w.writeheader()
        w.writerows(rr)
    print(f"wrote {len(rr)} rows to {rr_out}")

    # spectral mismatch at real resets: what M3 would get wrong
    print("\nreal resets: |p_ref - q_cal|/p_ref per band "
          "(M3's in-gap spectrum error; M1/M2 read p_ref directly)")
    by_band = {}
    for r in rr:
        by_band.setdefault(r["band"], []).append(
            abs(r["p_ref"] - r["q_cal"]) / max(r["p_ref"], 1e-9))
    for j in range(N_BANDS):
        e = np.array(by_band.get(j, [0]))
        lo, hi = BAND_EDGES[j], BAND_EDGES[j + 1]
        print(f"  ch{lo:>3}-{hi:<3}: median {np.median(e)*100:5.2f}%  "
              f"p90 {np.percentile(e, 90)*100:5.2f}%  n={len(e)}")

    # ── summary table ──
    import collections
    acc = collections.defaultdict(list)
    for r in rows:
        if r["truth"] > 0:
            for m in ("M1", "M2", "M3"):
                acc[(r["dur_ms"], r["band"], m)].append(
                    (r[m] - r["truth"]) / r["truth"])
            acc[(r["dur_ms"], r["band"], "poisson")].append(
                1.0 / np.sqrt(r["truth"]))

    print("\nrelative error of recovered per-band counts "
          "(mean ± std over injections; poisson = ground-truth floor)")
    hdr = f"{'D':>6} {'band':>4} " + "".join(
        f"{m:>16}" for m in ("M1", "M2", "M3")) + f"{'poisson':>10}"
    print(hdr)
    for dur in GAP_DURATIONS:
        for j in range(N_BANDS):
            cells = []
            for m in ("M1", "M2", "M3"):
                e = np.array(acc[(dur * 1e3, j, m)])
                cells.append(f"{e.mean()*100:+6.2f}±{e.std()*100:5.2f}%")
            floor = np.mean(acc[(dur * 1e3, j, "poisson")]) * 100
            lo, hi = BAND_EDGES[j], BAND_EDGES[j + 1]
            print(f"{dur*1e3:5.0f}m  ch{lo:>3}-{hi:<3} "
                  + " ".join(f"{c:>15}" for c in cells)
                  + f"{floor:9.2f}%")


if __name__ == "__main__":
    main()
