#!/usr/bin/env python3
"""Visualize LOW-mode behavior:
  Top row (4 panels): example Large/Sci time series within 1 hour
    - all-HIGH (stuck)
    - all-LOW (stuck)
    - clean HIGH-LOW-HIGH (2 transitions, single LOW pulse)
    - multi-switch (rapid threshold flickering)

  Bottom row (3 panels):
    - transition count distribution per (det, hour)
    - transition position within the hour
    - transition density per date (timeline)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta

CSV_DIR = Path("n_below_study/per_sec_csvs")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_LO, SCI_HI = 400.0, 1000.0
HEPOCH = datetime(2012, 1, 1, 0, 0, 0)


def classify_ratio(r):
    if r > 0.5: return 1     # HIGH
    if r < 0.4: return -1    # LOW
    return 0                 # AMBIG


def scan_all_files():
    """Single pass: collect (file, box, det) → (n_trans, first_met,
    file_date, transition_times_sec_into_hour)."""
    files = sorted(CSV_DIR.glob("*.csv"))
    files = [f for f in files if f.stat().st_size > 1000]
    print(f"Scanning {len(files):,} files...")

    rows = []
    all_trans_pos = []                # seconds into file for each transition
    date_trans_count = {}             # date → total transitions in that date

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
        d["r"] = d["Large"] / d["Sci"].clip(lower=1)
        d = d.sort_values(["box","det","met_sec"]).reset_index(drop=True)

        for (box, det), g in d.groupby(["box","det"]):
            if len(g) < 60: continue
            r = g["r"].values
            met = g["met_sec"].values
            cls = np.where(r > 0.5, 1, np.where(r < 0.4, -1, 0))
            file_start = met[0]
            n_low = (cls == -1).sum()
            n_high = (cls == 1).sum()
            # Transitions: HIGH ↔ LOW only
            prev = None
            tcount = 0
            tpositions = []
            for k in range(len(cls)):
                if cls[k] == 0: continue
                cur = cls[k]
                if prev is not None and cur != prev:
                    tcount += 1
                    tpositions.append(int(met[k] - file_start))
                prev = cur
            rows.append({
                "file": f.name, "box": str(box), "det": int(det),
                "n_clean": len(g), "n_high": int(n_high), "n_low": int(n_low),
                "n_trans": int(tcount), "first_met": int(met[0]),
            })
            all_trans_pos.extend(tpositions)
            if tcount > 0:
                date = (HEPOCH + timedelta(seconds=int(met[0]))).strftime("%Y-%m-%d")
                date_trans_count[date] = date_trans_count.get(date, 0) + tcount

        if (i+1) % 1000 == 0:
            print(f"  ...{i+1}/{len(files)}")
    return pd.DataFrame(rows), np.array(all_trans_pos), date_trans_count


def load_file_series(fname, box_sel, det_sel):
    """Load Large/Sci time series for one (file, box, det)."""
    d = pd.read_csv(CSV_DIR / fname, usecols=["box","det","met_sec","L_cycles",
                                                "Sci","Large"])
    d = d[d["L_cycles"] > L_THRESH]
    d = d[(d["Sci"] >= SCI_LO) & (d["Sci"] < SCI_HI)]
    d = d[(d["box"]==box_sel) & (d["det"]==det_sel)]
    d = d.sort_values("met_sec")
    d["r"] = d["Large"] / d["Sci"].clip(lower=1)
    file_start = d["met_sec"].min() if len(d) > 0 else 0
    d["sec_into_hour"] = d["met_sec"] - file_start
    return d, int(file_start)


def pick_examples(rows):
    """Pick 4 example (file, box, det): all-HIGH, all-LOW, clean 2-trans, multi."""
    examples = []

    # all-HIGH (n_low==0, n_high large)
    cand = rows[(rows["n_trans"]==0) & (rows["n_low"]==0) & (rows["n_high"]>3000)]
    examples.append(("all-HIGH (stuck)", cand.iloc[0]))

    # all-LOW (n_high==0, n_low large)
    cand = rows[(rows["n_trans"]==0) & (rows["n_high"]==0) & (rows["n_low"]>3000)]
    examples.append(("all-LOW (stuck)", cand.iloc[0]))

    # 2-transition clean (n_trans==2, n_low > 50)
    cand = rows[(rows["n_trans"]==2) & (rows["n_low"]>50) & (rows["n_high"]>500)]
    examples.append(("HIGH-LOW-HIGH (clean pulse)", cand.iloc[0]))

    # multi-switch (200+ transitions)
    cand = rows[rows["n_trans"]>200].sort_values("n_trans", ascending=False)
    examples.append(("rapid flickering (threshold noise)", cand.iloc[0]))

    return examples


def main():
    rows, all_trans_pos, date_trans = scan_all_files()
    print(f"\nTotal (det,file) sequences: {len(rows):,}")
    print(f"Total transitions:         {len(all_trans_pos):,}")
    print(f"Total dates with trans:    {len(date_trans):,}")

    examples = pick_examples(rows)

    # ----- Build figure -----
    fig = plt.figure(figsize=(20, 11))
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1],
                               hspace=0.32, top=0.93, bottom=0.06,
                               left=0.06, right=0.97)

    # Top: 4 time-series example panels
    gs_top = outer[0].subgridspec(1, 4, wspace=0.18)
    for col, (label, ex) in enumerate(examples):
        ax = fig.add_subplot(gs_top[0, col])
        d, file_start = load_file_series(ex["file"], ex["box"], ex["det"])
        utc = HEPOCH + timedelta(seconds=int(file_start))
        # Color points by mode
        for mode_name, color, mask_fn in [
            ("HIGH", "#1f77b4", lambda s: s["r"] > 0.5),
            ("LOW",  "#d62728", lambda s: s["r"] < 0.4),
            ("AMBIG","#999999", lambda s: (s["r"] >= 0.4) & (s["r"] <= 0.5)),
        ]:
            sub = d[mask_fn(d)]
            ax.scatter(sub["sec_into_hour"], sub["r"], s=4, color=color,
                        alpha=0.6, label=f"{mode_name} (N={len(sub)})", rasterized=True)
        ax.axhline(0.5, color="gray", ls="--", lw=0.7, alpha=0.7,
                    label="HIGH thresh (0.5)")
        ax.axhline(0.4, color="gray", ls=":",  lw=0.7, alpha=0.7,
                    label="LOW thresh (0.4)")
        ax.set_xlim(0, 3600); ax.set_ylim(0, 1.2)
        ax.set_xlabel("sec into hour")
        if col == 0:
            ax.set_ylabel("Large / Sci")
        ax.set_title(f"{label}\n{ex['file']} {ex['box']}-{ex['det']}  "
                      f"({utc.strftime('%Y-%m-%d %H:%M UTC')})\n"
                      f"n_trans={ex['n_trans']}, n_HIGH={ex['n_high']}, n_LOW={ex['n_low']}",
                      fontsize=9)
        ax.legend(loc="upper right", fontsize=6, ncol=1, framealpha=0.85)
        ax.grid(alpha=0.3)

    # Bottom: 3 histograms (transition counts, positions, dates)
    gs_bot = outer[1].subgridspec(1, 3, wspace=0.22)

    # B1: transition count distribution (log y)
    ax = fig.add_subplot(gs_bot[0, 0])
    bins = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    counts, edges = np.histogram(rows["n_trans"].values, bins=bins)
    centers = 0.5*(edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    ax.bar(centers, counts, width=widths*0.85, color="#9467bd", edgecolor="black",
            linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("n_transitions per (det, hour)")
    ax.set_ylabel("count of sequences")
    ax.set_title(f"Transition count distribution\n"
                  f"({len(rows):,} (det,hour) sequences; "
                  f"{int((rows['n_trans']==0).sum()):,} 'stuck')", fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    # Annotate
    for i, (cnt, lo, hi) in enumerate(zip(counts, edges[:-1], edges[1:])):
        if cnt > 0:
            ax.text(centers[i], cnt*1.15, f"{cnt}", ha='center',
                     fontsize=7, color="black")

    # B2: transition position within hour
    ax = fig.add_subplot(gs_bot[0, 1])
    pos_bins = np.linspace(0, 3600, 31)
    ax.hist(all_trans_pos, bins=pos_bins, color="#2ca02c", edgecolor="black",
             linewidth=0.5)
    ax.set_xlabel("sec into 1-hour file")
    ax.set_ylabel("# transitions")
    ax.set_title(f"Transition position within hour\n"
                  f"({len(all_trans_pos):,} total transitions across all files)",
                  fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    mean_per_bin = len(all_trans_pos) / 30
    ax.axhline(mean_per_bin, color="red", ls="--", lw=1.2,
                label=f"uniform expectation = {mean_per_bin:.0f}/bin")
    ax.legend(loc="upper right", fontsize=8)

    # B3: transition count per date (timeline)
    ax = fig.add_subplot(gs_bot[0, 2])
    dates_sorted = sorted(date_trans.keys())
    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates_sorted]
    counts_per_date = [date_trans[d] for d in dates_sorted]
    ax.plot(date_objs, counts_per_date, '.-', color="#ff7f0e", alpha=0.7,
             markersize=2, lw=0.4)
    ax.set_yscale("log")
    ax.set_xlabel("date")
    ax.set_ylabel("# transitions per date (log)")
    ax.set_title(f"Transitions per date (timeline)\n"
                  f"({len(dates_sorted):,} distinct dates)", fontsize=10)
    ax.grid(alpha=0.3)
    # Mark top-10 most-transition dates
    top_dates = sorted(date_trans.items(), key=lambda x: -x[1])[:10]
    for d, c in top_dates:
        d_obj = datetime.strptime(d, "%Y-%m-%d")
        ax.annotate(d, (d_obj, c), fontsize=6, alpha=0.7,
                     xytext=(2, 2), textcoords='offset points')

    fig.suptitle("HXMT HE LOW-Large mode behavior: examples + transition statistics  "
                 "(threshold: Large/Sci > 0.5 = HIGH, < 0.4 = LOW)",
                 fontsize=13, y=0.985)

    out = OUT_DIR / "low_mode_visualizations.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {out}")
    desktop = Path.home() / "Desktop" / "low_mode_visualizations.png"
    fig.savefig(desktop, dpi=180, bbox_inches="tight")
    print(f"Saved: {desktop}")


if __name__ == "__main__":
    main()
