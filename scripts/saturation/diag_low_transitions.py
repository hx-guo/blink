#!/usr/bin/env python3
"""Per-file: detect HIGH↔LOW transitions within each 1-hour window.

For each file (one date, one box, 6 dets):
  - Per (box, det) sort by met_sec
  - Classify each second as HIGH/LOW/AMBIG (Large/Sci ratio)
  - Run-length encode → find segments
  - Count transitions, report durations

Cross-file summary:
  - How many transitions per file?
  - Transition time distribution (where in the hour do switches happen?)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter

CSV_DIR = Path("n_below_study/per_sec_csvs")
L_THRESH = 50_000
SCI_LO, SCI_HI = 400.0, 1000.0
HEPOCH = datetime(2012, 1, 1, 0, 0, 0)


def classify(d):
    r = d["Large"].astype("float32") / d["Sci"].clip(lower=1)
    return np.where(r > 0.5, 1, np.where(r < 0.4, -1, 0))  # 1=HIGH, -1=LOW, 0=AMBIG


def main():
    files = sorted(CSV_DIR.glob("*.csv"))
    files = [f for f in files if f.stat().st_size > 1000]
    print(f"Scanning {len(files):,} files for HIGH↔LOW transitions...")

    transition_counts = Counter()   # # of (box, det) segments by transition count
    transition_times = []           # sec-into-hour of each transition
    transition_examples = []        # (file, box, det, n_transitions) for interesting cases
    all_segments = []               # (box-det, mode, duration)

    for i, f in enumerate(files):
        try:
            d = pd.read_csv(f, usecols=["box","det","met_sec","L_cycles",
                                          "Sci","Large"])
        except Exception:
            continue
        if len(d) == 0: continue
        d = d[d["L_cycles"] > L_THRESH]
        d = d[(d["Sci"] >= SCI_LO) & (d["Sci"] < SCI_HI)]
        if len(d) < 100: continue
        d["cls"] = classify(d)
        d = d.sort_values(["box","det","met_sec"]).reset_index(drop=True)

        # Per (box, det) inside this file
        for (box, det), g in d.groupby(["box","det"]):
            if len(g) < 60: continue
            cls = g["cls"].values
            met = g["met_sec"].values
            # Find transitions: where cls changes (skip AMBIG)
            # Strict: only HIGH↔LOW direct transitions
            transitions = []
            prev_mode = None
            for k in range(len(cls)):
                if cls[k] == 0:  # AMBIG - skip
                    continue
                cur = "HIGH" if cls[k] == 1 else "LOW"
                if prev_mode is not None and cur != prev_mode:
                    transitions.append((int(met[k]), prev_mode + "→" + cur))
                prev_mode = cur

            ntr = len(transitions)
            transition_counts[ntr] += 1
            if ntr >= 1:
                # Record positions (sec into file)
                file_start_met = met[0]
                for t_met, _ in transitions:
                    transition_times.append(int(t_met) - int(file_start_met))
                if ntr >= 3:
                    transition_examples.append((f.name, box, det, ntr,
                                                  transitions[:5]))

            # Also extract run-length segments
            i_start = 0
            for k in range(1, len(cls) + 1):
                if k == len(cls) or cls[k] != cls[i_start]:
                    mode_name = {1: "HIGH", -1: "LOW", 0: "AMBIG"}[cls[i_start]]
                    duration = int(met[k-1] - met[i_start] + 1)
                    if duration >= 5 and mode_name != "AMBIG":
                        all_segments.append((box, det, mode_name, duration))
                    i_start = k

        if (i+1) % 1000 == 0:
            print(f"  ...processed {i+1}/{len(files)}")

    # ========== Summary ==========
    print(f"\n{'='*70}")
    print("Transition count distribution (per (box, det) per file)")
    print(f"{'='*70}")
    total_seq = sum(transition_counts.values())
    print(f"  {'n_transitions':>14s}  {'n_sequences':>12s}  {'pct':>7s}")
    for n in sorted(transition_counts):
        cnt = transition_counts[n]
        marker = ""
        if n == 0: marker = "  ← no switch (all-HIGH or all-LOW)"
        elif n == 1: marker = "  ← single switch (clean transition)"
        elif n >= 5: marker = "  ← multi-switch (intermittent)"
        print(f"  {n:>14d}  {cnt:>12d}  {100*cnt/total_seq:>6.2f}%{marker}")

    print(f"\n{'='*70}")
    print("Where in the hour do transitions happen? (sec into file)")
    print(f"{'='*70}")
    tt = np.array(transition_times)
    print(f"  Total transitions: {len(tt):,}")
    if len(tt) > 0:
        print(f"  Min: {tt.min()} s  Max: {tt.max()} s")
        print(f"  Distribution (15 bins covering 0-3600 s):")
        bins = np.linspace(0, 3600, 16)
        hist, _ = np.histogram(tt, bins=bins)
        for j in range(15):
            bar = "█" * int(50 * hist[j] / hist.max()) if hist.max() > 0 else ""
            print(f"    {int(bins[j]):4d}-{int(bins[j+1]):4d} s  "
                  f"({hist[j]:>6,d})  {bar}")

    print(f"\n{'='*70}")
    print("Segment duration distribution (run-length)")
    print(f"{'='*70}")
    seg_df = pd.DataFrame(all_segments, columns=["box","det","mode","duration"])
    for mode in ["HIGH","LOW"]:
        s = seg_df[seg_df["mode"]==mode]
        print(f"\n  {mode} segments: {len(s):,}")
        if len(s) > 0:
            d = s["duration"]
            print(f"    median={int(d.median()):>5d}s, mean={d.mean():>7.1f}s, "
                  f"p90={int(d.quantile(0.9)):>5d}s, p99={int(d.quantile(0.99)):>5d}s, "
                  f"max={int(d.max()):>5d}s")

    print(f"\n{'='*70}")
    print("Examples: files with multi-switch (≥3 transitions per det-hour)")
    print(f"{'='*70}")
    transition_examples.sort(key=lambda x: -x[3])
    for fname, box, det, ntr, sample_tr in transition_examples[:10]:
        dt = HEPOCH + timedelta(seconds=int(sample_tr[0][0]))
        print(f"\n  {fname} {box}-{det}: {ntr} transitions")
        print(f"    First 5: {sample_tr}")
        print(f"    UTC: {dt.strftime('%Y-%m-%d %H:%M:%S')} (first)")


if __name__ == "__main__":
    main()
