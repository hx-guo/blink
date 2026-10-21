#!/usr/bin/env python3
"""For MIXED (HIGH+LOW) days: check whether PHO/Wide also abnormal during LOW segments.

If only Large is anomalous → LOW is electronics-specific to Large counter
If PHO/Wide also shift → LOW is system-wide event
"""
from pathlib import Path
import numpy as np
import pandas as pd

CSV_DIR = Path("n_below_study/per_sec_csvs")
L_THRESH = 50_000
SCI_LO, SCI_HI = 400.0, 1000.0


def analyze_file(fpath):
    try:
        d = pd.read_csv(fpath, usecols=["box","det","met_sec","L_cycles",
                                          "Sci","Large","PHO","Wide"])
    except Exception:
        return None
    d = d[d["L_cycles"] > L_THRESH]
    d = d[(d["Sci"] >= SCI_LO) & (d["Sci"] < SCI_HI)]
    if len(d) == 0: return None
    d["ratio"] = d["Large"] / d["Sci"].clip(lower=1)
    d["mode"] = np.where(d["ratio"] > 0.5, "HIGH",
                  np.where(d["ratio"] < 0.4, "LOW", "AMBIG"))
    return d


def main():
    # Find MIXED files with moderate LOW% (10-90%) — both modes present in same file
    print("Scanning for MIXED-mode files (LOW% between 10-90%)...")
    mixed_files = []
    for f in sorted(CSV_DIR.glob("*.csv")):
        if f.stat().st_size < 1000: continue
        try:
            d = pd.read_csv(f, usecols=["L_cycles","Sci","Large"])
        except Exception:
            continue
        d = d[d["L_cycles"] > L_THRESH]
        d = d[(d["Sci"] >= SCI_LO) & (d["Sci"] < SCI_HI)]
        if len(d) < 100: continue
        ratio = d["Large"] / d["Sci"].clip(lower=1)
        n_low = (ratio < 0.4).sum()
        n_high = (ratio > 0.5).sum()
        low_pct = 100*n_low/len(d)
        if 10 <= low_pct <= 90 and n_low >= 100 and n_high >= 100:
            mixed_files.append((f, len(d), low_pct))
        if len(mixed_files) >= 30:
            break
    print(f"Found {len(mixed_files)} mixed files")

    # Analyze top 5 mixed
    mixed_files.sort(key=lambda x: -x[2])
    print(f"\n{'='*82}")
    print(f"For each MIXED file: compare HIGH vs LOW median of all engineering counters")
    print(f"{'='*82}")
    for f, n_total, low_pct in mixed_files[:5]:
        d = analyze_file(f)
        if d is None: continue
        h = d[d["mode"]=="HIGH"]
        l = d[d["mode"]=="LOW"]
        print(f"\n  {f.name}  (CLEAN={n_total}, LOW%={low_pct:.1f}%, "
              f"N_HIGH={len(h)}, N_LOW={len(l)})")
        print(f"    {'metric':>10s}  {'HIGH med':>10s}  {'LOW med':>10s}  "
              f"{'LOW/HIGH':>10s}  {'note':>30s}")
        for metric in ["Sci","PHO","Wide","Large","L_cycles"]:
            hm = float(h[metric].median())
            lm = float(l[metric].median())
            ratio = lm/hm if hm > 0 else float('nan')
            if metric == "Large":
                note = "← LOW 定义"
            elif metric == "L_cycles":
                note = "(基线，应近 1.0)"
            elif 0.95 < ratio < 1.05:
                note = "≈ 不变"
            elif ratio < 0.8:
                note = f"显著下降 ({100*(1-ratio):.0f}%)"
            elif ratio > 1.2:
                note = f"显著上升 ({100*(ratio-1):.0f}%)"
            else:
                note = ""
            print(f"    {metric:>10s}  {hm:>10.1f}  {lm:>10.1f}  {ratio:>10.3f}  {note:>30s}")

    # Also check if Large is just shifted, or if its STD is also different
    print(f"\n{'='*82}")
    print(f"Large distribution shape: HIGH vs LOW for top-mixed file")
    print(f"{'='*82}")
    if mixed_files:
        f = mixed_files[0][0]
        d = analyze_file(f)
        h = d[d["mode"]=="HIGH"]
        l = d[d["mode"]=="LOW"]
        print(f"  File: {f.name}")
        print(f"    {'mode':>6s}  {'N':>5s}  {'mean':>8s}  {'std':>8s}  {'cv':>6s}  "
              f"{'10%ile':>8s}  {'90%ile':>8s}")
        for mode_name, mode_data in [("HIGH", h), ("LOW", l)]:
            for col in ["Large","ratio"]:
                v = mode_data[col]
                cv = float(v.std() / v.mean()) if v.mean() != 0 else float('nan')
                print(f"    {mode_name+' '+col:>10s}  {len(v):>5d}  {v.mean():>8.2f}  "
                      f"{v.std():>8.2f}  {cv:>6.3f}  {v.quantile(0.1):>8.2f}  "
                      f"{v.quantile(0.9):>8.2f}")


if __name__ == "__main__":
    main()
