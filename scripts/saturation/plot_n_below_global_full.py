#!/usr/bin/env python3
"""Refit N_below = b + α·Sci globally on the full per-second CSV dataset.

CSV columns: date, box, met_sec, det, L_cycles, PHO, Wide, Large, Dt, Sci, Sci_ACD1, Sci_ACDN, CRC_box
- Each row = one second × one detector × one box.
- L_cycles × 16 µs = live time in that second.

Model (same as plot_n_below_global.py):
    N_below := PHO − β·Wide − γ·Large − Sci    (β=2, γ=1)
    N_below  = b + α·Sci    per (date, box, det) slot
    Sci_pred = (PHO − β·Wide − γ·Large − b) / (1 + α)

Outputs (vs the previous 5-date × 90-slot fit):
1. Histograms of (b, α) on ~6800 slots
2. (b, α) per-year trend  → PMT outgassing diagnostic
3. Per-box (A/B/C) box-plot   → Box C blind-det check
4. Per-detector identity consistency across all dates
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BETA = 2.0
GAMMA = 1.0
CSV_DIR = Path("n_below_study/per_sec_csvs")
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

L_THRESH = 50_000          # live-time fraction > 0.8 sec
SCI_SEC_TOTAL_MIN = 100    # total Sci across 6 det in that sec
MIN_POINTS_PER_SLOT = 50


def fit_lin(y, x):
    X = np.column_stack([np.ones_like(x), x])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(c[0]), float(c[1]), y - X @ c


def load_all():
    files = sorted(CSV_DIR.glob("*.csv"))
    print(f"Loading {len(files)} CSVs ...")
    dtype = {
        "date": "string", "box": "category",
        "met_sec": "int64", "det": "int8",
        "L_cycles": "int32", "PHO": "int32", "Wide": "int32",
        "Large": "int32", "Dt": "int32",
        "Sci": "int32", "Sci_ACD1": "int32", "Sci_ACDN": "int32",
        "CRC_box": "int32",
    }
    parts = []
    for i, f in enumerate(files):
        try:
            parts.append(pd.read_csv(f, dtype=dtype))
        except Exception as e:
            print(f"  ERR {f.name}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}")
    df = pd.concat(parts, ignore_index=True)
    print(f"  total rows = {len(df):,}")
    return df


def fit_all(df):
    df["length_s"] = df["L_cycles"].astype("float32") * 16e-6
    df = df[df["L_cycles"] > L_THRESH].copy()

    # Per-second total Sci across detectors of that box (broadcast back)
    g = df.groupby(["date", "box", "met_sec"], observed=True)["Sci"].sum()
    g.name = "sci_sec_total"
    df = df.merge(g, on=["date", "box", "met_sec"])
    df = df[df["sci_sec_total"] > SCI_SEC_TOTAL_MIN]

    slots = []
    for (date, box, det), sub in df.groupby(["date", "box", "det"], observed=True):
        L = sub["length_s"].values
        Sci = sub["Sci"].values / L
        if len(Sci) < MIN_POINTS_PER_SLOT:
            continue
        # Trim 5–95 percentile of Sci rate within this slot
        p5, p95 = np.percentile(Sci, [5, 95])
        sel = (Sci >= p5) & (Sci <= p95)
        if sel.sum() < MIN_POINTS_PER_SLOT:
            continue
        L = L[sel]
        sci = Sci[sel]
        pho = sub["PHO"].values[sel] / L
        wide = sub["Wide"].values[sel] / L
        large = sub["Large"].values[sel] / L
        nb = pho - BETA * wide - GAMMA * large - sci
        b, alpha, resid = fit_lin(nb, sci)
        slots.append({
            "date": date, "box": str(box), "det": int(det),
            "year": int(date[:4]),
            "b": b, "alpha": alpha,
            "rms": float(np.sqrt(np.mean(resid ** 2))),
            "n": int(sel.sum()),
            "sci_med": float(np.median(sci)),
            "nb_med": float(np.median(nb)),
        })
    return pd.DataFrame(slots)


def main():
    df = load_all()
    print("Fitting per-slot ...")
    s = fit_all(df)
    print(f"Total slots fit: {len(s)}")
    s.to_csv(OUT_DIR / "n_below_slots_full.csv", index=False)

    bs = s["b"].values
    al = s["alpha"].values
    rms = s["rms"].values
    b_med, a_med = np.median(bs), np.median(al)

    # === Fig 1: histograms + (b, α) scatter ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    ax.hist(bs, bins=60, color="C0", edgecolor="k", alpha=0.7)
    ax.axvline(b_med, color="r", lw=2, label=f"median = {b_med:.2f}")
    ax.axvline(np.mean(bs), color="g", lw=2, ls="--",
               label=f"mean = {np.mean(bs):.2f} ± {np.std(bs):.2f}")
    ax.set_xlabel("b  [cnt/s/det]")
    ax.set_ylabel("count")
    ax.set_title(f"b distribution  ({len(bs)} slots, range {bs.min():.1f}–{bs.max():.1f})")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(al, bins=60, color="C1", edgecolor="k", alpha=0.7)
    ax.axvline(a_med, color="r", lw=2, label=f"median = {a_med:.4f}")
    ax.axvline(np.mean(al), color="g", lw=2, ls="--",
               label=f"mean = {np.mean(al):.4f} ± {np.std(al):.4f}")
    ax.set_xlabel("α  (Sci-correlated slope)")
    ax.set_ylabel("count")
    ax.set_title(f"α distribution  (range {al.min():.3f}–{al.max():.3f})")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    box_color = {"A": "C0", "B": "C1", "C": "C2"}
    for box in "ABC":
        m = s["box"] == box
        ax.scatter(s.loc[m, "b"], s.loc[m, "alpha"], s=4, alpha=0.3,
                   color=box_color[box], label=f"Box {box} (N={m.sum()})", rasterized=True)
    ax.scatter([b_med], [a_med], marker="*", s=300, color="red",
               edgecolor="k", lw=1, label=f"global median ({b_med:.1f}, {a_med:.3f})", zorder=10)
    ax.set_xlabel("b [cnt/s]"); ax.set_ylabel("α")
    ax.set_title("(b, α) per slot, colored by box")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.hist(rms, bins=60, color="C3", edgecolor="k", alpha=0.7)
    ax.axvline(np.median(rms), color="r", lw=2, label=f"median = {np.median(rms):.1f}")
    ax.set_xlabel("residual RMS [cnt/s]")
    ax.set_ylabel("count")
    ax.set_title("residual RMS of linear fit")
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(f"N_below = (PHO − {BETA}·Wide − {GAMMA}·Large) − Sci  =  b + α·Sci\n"
                 f"Full per-second dataset: {s['date'].nunique()} dates × 3 boxes × 6 det = {len(s)} slots",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "n_below_full_hist.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")

    # === Fig 2: year-bucketed trend ===
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    years = sorted(s["year"].unique())
    ax = axes2[0]
    for box in "ABC":
        m = s["box"] == box
        med = s[m].groupby("year")["b"].median()
        q1 = s[m].groupby("year")["b"].quantile(0.25)
        q3 = s[m].groupby("year")["b"].quantile(0.75)
        ax.fill_between(med.index, q1.values, q3.values, alpha=0.15, color=box_color[box])
        ax.plot(med.index, med.values, marker="o", color=box_color[box],
                label=f"Box {box} (med ± IQR)")
    ax.set_xlabel("year"); ax.set_ylabel("b [cnt/s]")
    ax.set_title("b vs year  — PMT outgassing diagnostic")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes2[1]
    for box in "ABC":
        m = s["box"] == box
        med = s[m].groupby("year")["alpha"].median()
        q1 = s[m].groupby("year")["alpha"].quantile(0.25)
        q3 = s[m].groupby("year")["alpha"].quantile(0.75)
        ax.fill_between(med.index, q1.values, q3.values, alpha=0.15, color=box_color[box])
        ax.plot(med.index, med.values, marker="o", color=box_color[box],
                label=f"Box {box}")
    ax.set_xlabel("year"); ax.set_ylabel("α")
    ax.set_title("α vs year")
    ax.legend(); ax.grid(alpha=0.3)

    fig2.tight_layout()
    out = OUT_DIR / "n_below_full_year_trend.png"
    fig2.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")

    # === Fig 3: per-detector identity (mean ± std across all dates) ===
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    ids, b_mu, b_sd, a_mu, a_sd = [], [], [], [], []
    for box in "ABC":
        for det in range(6):
            sub = s[(s["box"] == box) & (s["det"] == det)]
            if len(sub) < 5:
                continue
            ids.append(f"{box}{det}")
            b_mu.append(sub["b"].mean()); b_sd.append(sub["b"].std())
            a_mu.append(sub["alpha"].mean()); a_sd.append(sub["alpha"].std())
    x = np.arange(len(ids))
    ax = axes3[0]
    ax.errorbar(x, b_mu, yerr=b_sd, fmt="o", capsize=3, color="C0")
    ax.set_xticks(x); ax.set_xticklabels(ids, rotation=45, fontsize=9)
    ax.set_ylabel("b [cnt/s]")
    ax.set_title(f"per-detector b across {s['date'].nunique()} dates (err = std)")
    ax.grid(alpha=0.3)

    ax = axes3[1]
    ax.errorbar(x, a_mu, yerr=a_sd, fmt="s", capsize=3, color="C1")
    ax.set_xticks(x); ax.set_xticklabels(ids, rotation=45, fontsize=9)
    ax.set_ylabel("α")
    ax.set_title("per-detector α across all dates")
    ax.grid(alpha=0.3)
    fig3.tight_layout()
    out = OUT_DIR / "n_below_full_per_det.png"
    fig3.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")

    # === Numerical summary ===
    print(f"\n{'='*64}")
    print(f"N_below = b + α·Sci  fit on {len(s)} (date × box × det) slots")
    print(f"  N_below ≡ PHO − {BETA}·Wide − {GAMMA}·Large − Sci   (rates: cnt/s/det)")
    print(f"  Dates: {s['date'].nunique()}  |  Years: {min(years)}–{max(years)}")
    print(f"{'='*64}")
    print(f"  b   mean = {np.mean(bs):>6.2f}  median = {b_med:>6.2f}  "
          f"std = {np.std(bs):.2f}   range {bs.min():.1f} – {bs.max():.1f}")
    print(f"  α   mean = {np.mean(al):.4f}  median = {a_med:.4f}  "
          f"std = {np.std(al):.4f}   range {al.min():.4f} – {al.max():.4f}")
    print(f"  RMS mean = {np.mean(rms):.1f}  median = {np.median(rms):.1f}")
    print()
    for box in "ABC":
        m = s["box"] == box
        print(f"  Box {box}: b = {s.loc[m,'b'].mean():.2f} ± {s.loc[m,'b'].std():.2f}   "
              f"α = {s.loc[m,'alpha'].mean():.4f} ± {s.loc[m,'alpha'].std():.4f}   (N={m.sum()})")
    print()
    print(f"Recommended global constants:")
    print(f"  β = {BETA}    γ = {GAMMA}    b = {b_med:.2f}    α = {a_med:.4f}")
    print(f"  ⇒  Sci_pred = (PHO − {BETA}·Wide − {GAMMA}·Large − {b_med:.2f}) / {1+a_med:.4f}")


if __name__ == "__main__":
    main()
