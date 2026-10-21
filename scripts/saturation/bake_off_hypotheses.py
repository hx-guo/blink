#!/usr/bin/env python3
"""Bake-off: fit multiple hypothesized constraints on the same training data,
compare RMS and key statistics. Outputs:
  - printed summary table
  - bar chart of total RMS per box × hypothesis  (plots/bake_off.png)

Uses the parquet cache for fast load (~5 sec).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CACHE = Path("n_below_study/train_cache.parquet")
SCI_LO_CLEAN, SCI_HI_CLEAN, BOX_RATE_CAP = 400.0, 1000.0, 6000.0
N_MIN_PERDET = 100
OUT = Path("plots/bake_off.png")


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE)
    print(f"  rows: {len(df):,}")

    clean = ((df["sci_rate"] >= SCI_LO_CLEAN) & (df["sci_rate"] < SCI_HI_CLEAN)
              & (df["group_rate"] < BOX_RATE_CAP))
    sub_all = df[clean].copy()
    print(f"  CLEAN rows: {len(sub_all):,}")

    # ===== Define hypotheses =====
    # Each hypothesis: dict with 'name', 'fit_one' which takes a df and returns
    # (coef[5], use_dt: bool). coef = [b, c_pure, c_ACD, β, γ].
    def fit_full(s, dt_correct=False):
        """V8 5-param fit (or dt-corrected if dt_correct=True)."""
        X = np.column_stack([np.ones(len(s)), s["scipure_rate"], s["acd_rate"],
                              s["wide_rate"], s["large_rate"]])
        y = (s["pho_rate"].values * (1 - s["dt_frac"].values)
             if dt_correct else s["pho_rate"].values)
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        return c, dt_correct

    def fit_cpure1_gamma1(s, dt_correct=False):
        """Fix c_pure=γ=1, fit (b, c_ACD, β)."""
        X = np.column_stack([np.ones(len(s)), s["acd_rate"], s["wide_rate"]])
        rhs = s["pho_rate"].values * (1 - s["dt_frac"].values) if dt_correct else s["pho_rate"].values
        y = rhs - s["scipure_rate"].values - s["large_rate"].values
        c3, *_ = np.linalg.lstsq(X, y, rcond=None)
        return np.array([c3[0], 1.0, c3[1], c3[2], 1.0]), dt_correct

    def fit_cACD2(s, dt_correct=False):
        """Fix c_ACD=2, fit (b, c_pure, β, γ)."""
        X = np.column_stack([np.ones(len(s)), s["scipure_rate"],
                              s["wide_rate"], s["large_rate"]])
        rhs = s["pho_rate"].values * (1 - s["dt_frac"].values) if dt_correct else s["pho_rate"].values
        y = rhs - 2.0 * s["acd_rate"].values
        c4, *_ = np.linalg.lstsq(X, y, rcond=None)
        return np.array([c4[0], c4[1], 2.0, c4[2], c4[3]]), dt_correct

    def fit_all1(s, dt_correct=False):
        """Fix c_pure=γ=c_ACD=1, fit (b, β)."""
        X = np.column_stack([np.ones(len(s)), s["wide_rate"]])
        rhs = s["pho_rate"].values * (1 - s["dt_frac"].values) if dt_correct else s["pho_rate"].values
        y = rhs - s["scipure_rate"].values - s["acd_rate"].values - s["large_rate"].values
        c2, *_ = np.linalg.lstsq(X, y, rcond=None)
        return np.array([c2[0], 1.0, 1.0, c2[1], 1.0]), dt_correct

    def fit_cpure1_cACD2(s, dt_correct=False):
        """Fix c_pure=γ=1, c_ACD=2, fit (b, β)."""
        X = np.column_stack([np.ones(len(s)), s["wide_rate"]])
        rhs = s["pho_rate"].values * (1 - s["dt_frac"].values) if dt_correct else s["pho_rate"].values
        y = rhs - s["scipure_rate"].values - 2.0*s["acd_rate"].values - s["large_rate"].values
        c2, *_ = np.linalg.lstsq(X, y, rcond=None)
        return np.array([c2[0], 1.0, 2.0, c2[1], 1.0]), dt_correct

    def fit_no_wide(s, dt_correct=False):
        """β=0, fit (b, c_pure, c_ACD, γ)."""
        X = np.column_stack([np.ones(len(s)), s["scipure_rate"], s["acd_rate"],
                              s["large_rate"]])
        rhs = s["pho_rate"].values * (1 - s["dt_frac"].values) if dt_correct else s["pho_rate"].values
        c4, *_ = np.linalg.lstsq(X, rhs, rcond=None)
        return np.array([c4[0], c4[1], c4[2], 0.0, c4[3]]), dt_correct

    def fit_no_large(s, dt_correct=False):
        """γ=0, fit (b, c_pure, c_ACD, β)."""
        X = np.column_stack([np.ones(len(s)), s["scipure_rate"], s["acd_rate"],
                              s["wide_rate"]])
        rhs = s["pho_rate"].values * (1 - s["dt_frac"].values) if dt_correct else s["pho_rate"].values
        c4, *_ = np.linalg.lstsq(X, rhs, rcond=None)
        return np.array([c4[0], c4[1], c4[2], c4[3], 0.0]), dt_correct

    # β-global hypothesis: 2-stage (1: pool to get β_global; 2: per-det fit
    # with β fixed). Handled specially below — not in this dict.

    HYPOTHESES = [
        ("V8 baseline",          5, lambda s: fit_full(s)),
        ("c_pure=γ=1",           3, lambda s: fit_cpure1_gamma1(s)),
        ("dt k=1",               5, lambda s: fit_full(s, dt_correct=True)),
        ("dt + c_pure=γ=1",      3, lambda s: fit_cpure1_gamma1(s, dt_correct=True)),
        ("c_ACD=2",              4, lambda s: fit_cACD2(s)),
        ("c_pure=γ=1, c_ACD=2",  2, lambda s: fit_cpure1_cACD2(s)),
        ("all 4 coefs = 1 or 2", 2, lambda s: fit_all1(s)),
        ("no Wide (β=0)",        4, lambda s: fit_no_wide(s)),
        ("no Large (γ=0)",       4, lambda s: fit_no_large(s)),
    ]

    def predict_pho(s, coef, dt_correct):
        """Given fit coef and dt flag, return implied PHO."""
        b, c0, cA, bet, gam = coef
        rhs = (b + c0*s["scipure_rate"].values + cA*s["acd_rate"].values
                + bet*s["wide_rate"].values + gam*s["large_rate"].values)
        if dt_correct:
            return rhs / (1.0 - s["dt_frac"].values)
        return rhs

    # ===== Pre-extract per-(box, det) numpy arrays — one-time work =====
    print("  pre-extracting per-(box, det) numpy arrays (one-time)...", flush=True)
    import time as _time
    import gc
    _t0 = _time.time()
    data = {}              # (box, det) → dict of np.float32 arrays
    box_actuals = {}       # box → concat pho_rate over 6 dets
    for box in "ABC":
        acts = []
        for det in range(6):
            mask = (sub_all["box"]==box) & (sub_all["det"]==det)
            s = sub_all[mask]
            d = {
                "scipure": s["scipure_rate"].values.astype(np.float32),
                "acd":     s["acd_rate"].values.astype(np.float32),
                "wide":    s["wide_rate"].values.astype(np.float32),
                "large":   s["large_rate"].values.astype(np.float32),
                "pho":     s["pho_rate"].values.astype(np.float32),
                "dtfrac":  s["dt_frac"].values.astype(np.float32),
            }
            d["ones"] = np.ones(len(s), dtype=np.float32)
            d["pho_lf"] = d["pho"] * (1.0 - d["dtfrac"])
            data[(box, det)] = d
            acts.append(d["pho"])
        box_actuals[box] = np.concatenate(acts)
    # Drop original heavy DataFrames to free RAM
    del sub_all, df
    gc.collect()
    print(f"  ready in {_time.time()-_t0:.0f}s. "
          f"Pre-extracted arrays for 18 dets. Starting hypothesis loop...", flush=True)

    # ===== Pure-numpy fit & predict helpers =====
    def _fit_full(d, dt_correct):
        X = np.column_stack([d["ones"], d["scipure"], d["acd"], d["wide"], d["large"]])
        y = d["pho_lf"] if dt_correct else d["pho"]
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        return c, dt_correct

    def _fit_cpure1_gamma1(d, dt_correct):
        X = np.column_stack([d["ones"], d["acd"], d["wide"]])
        rhs = d["pho_lf"] if dt_correct else d["pho"]
        y = rhs - d["scipure"] - d["large"]
        c3, *_ = np.linalg.lstsq(X, y, rcond=None)
        return np.array([c3[0], 1.0, c3[1], c3[2], 1.0]), dt_correct

    def _fit_cACD2(d, dt_correct):
        X = np.column_stack([d["ones"], d["scipure"], d["wide"], d["large"]])
        rhs = d["pho_lf"] if dt_correct else d["pho"]
        y = rhs - 2.0*d["acd"]
        c4, *_ = np.linalg.lstsq(X, y, rcond=None)
        return np.array([c4[0], c4[1], 2.0, c4[2], c4[3]]), dt_correct

    def _fit_all1(d, dt_correct):
        X = np.column_stack([d["ones"], d["wide"]])
        rhs = d["pho_lf"] if dt_correct else d["pho"]
        y = rhs - d["scipure"] - d["acd"] - d["large"]
        c2, *_ = np.linalg.lstsq(X, y, rcond=None)
        return np.array([c2[0], 1.0, 1.0, c2[1], 1.0]), dt_correct

    def _fit_cpure1_cACD2(d, dt_correct):
        X = np.column_stack([d["ones"], d["wide"]])
        rhs = d["pho_lf"] if dt_correct else d["pho"]
        y = rhs - d["scipure"] - 2.0*d["acd"] - d["large"]
        c2, *_ = np.linalg.lstsq(X, y, rcond=None)
        return np.array([c2[0], 1.0, 2.0, c2[1], 1.0]), dt_correct

    def _fit_no_wide(d, dt_correct):
        X = np.column_stack([d["ones"], d["scipure"], d["acd"], d["large"]])
        rhs = d["pho_lf"] if dt_correct else d["pho"]
        c4, *_ = np.linalg.lstsq(X, rhs, rcond=None)
        return np.array([c4[0], c4[1], c4[2], 0.0, c4[3]]), dt_correct

    def _fit_no_large(d, dt_correct):
        X = np.column_stack([d["ones"], d["scipure"], d["acd"], d["wide"]])
        rhs = d["pho_lf"] if dt_correct else d["pho"]
        c4, *_ = np.linalg.lstsq(X, rhs, rcond=None)
        return np.array([c4[0], c4[1], c4[2], c4[3], 0.0]), dt_correct

    HYPOTHESES_FAST = [
        ("V8 baseline",          5, _fit_full,            False),
        ("c_pure=γ=1",           3, _fit_cpure1_gamma1,   False),
        ("dt k=1",               5, _fit_full,            True),
        ("dt + c_pure=γ=1",      3, _fit_cpure1_gamma1,   True),
        ("c_ACD=2",              4, _fit_cACD2,           False),
        ("c_pure=γ=1, c_ACD=2",  2, _fit_cpure1_cACD2,    False),
        ("all 4 coefs = 1 or 2", 2, _fit_all1,            False),
        ("no Wide (β=0)",        4, _fit_no_wide,         False),
        ("no Large (γ=0)",       4, _fit_no_large,        False),
    ]

    def _predict(d, coef, dt_correct):
        b, c0, cA, bet, gam = coef
        rhs = b + c0*d["scipure"] + cA*d["acd"] + bet*d["wide"] + gam*d["large"]
        if dt_correct:
            return rhs / (1.0 - d["dtfrac"])
        return rhs

    # ===== Run hypotheses on pre-extracted arrays =====
    results = []
    for hi, (name, n_free, fit_one, dt_correct) in enumerate(HYPOTHESES_FAST):
        t0 = _time.time()
        coefs_per_det = {}
        for box in "ABC":
            for det in range(6):
                d = data[(box, det)]
                if len(d["pho"]) < N_MIN_PERDET:
                    coefs_per_det[(box, det)] = (np.array([0.,0.,0.,0.,0.]), False)
                    continue
                coefs_per_det[(box, det)] = fit_one(d, dt_correct)
        t_fit = _time.time() - t0
        rms_box, b_box = {}, {}
        for box in "ABC":
            preds, bvals = [], []
            for det in range(6):
                coef, dt_corr = coefs_per_det[(box, det)]
                preds.append(_predict(data[(box, det)], coef, dt_corr))
                bvals.append(coef[0])
            pred = np.concatenate(preds)
            rms_box[box] = float(np.sqrt(np.mean((pred - box_actuals[box])**2)))
            b_box[box] = float(np.mean(bvals))
        t_total = _time.time() - t0
        print(f"  [{hi+1}/{len(HYPOTHESES_FAST)}] {name:<25s}  "
              f"fit={t_fit:.1f}s  total={t_total:.1f}s  "
              f"RMS={rms_box['A']:.2f}/{rms_box['B']:.2f}/{rms_box['C']:.2f}",
              flush=True)
        results.append((name, n_free, rms_box, b_box))

    # ===== β-global hypothesis (special, 2-stage, pure-numpy) =====
    print(f"  [β global] pooling 18 dets for global β fit...", flush=True)
    t0 = _time.time()
    # Stage 1: pool all 18-det data
    all_ones    = np.concatenate([data[(b,d)]["ones"]    for b in "ABC" for d in range(6)])
    all_scipure = np.concatenate([data[(b,d)]["scipure"] for b in "ABC" for d in range(6)])
    all_acd     = np.concatenate([data[(b,d)]["acd"]     for b in "ABC" for d in range(6)])
    all_wide    = np.concatenate([data[(b,d)]["wide"]    for b in "ABC" for d in range(6)])
    all_large   = np.concatenate([data[(b,d)]["large"]   for b in "ABC" for d in range(6)])
    all_pho     = np.concatenate([data[(b,d)]["pho"]     for b in "ABC" for d in range(6)])
    X_pool = np.column_stack([all_ones, all_scipure, all_acd, all_wide, all_large])
    c_pool, *_ = np.linalg.lstsq(X_pool, all_pho, rcond=None)
    beta_g = float(c_pool[3])
    del all_ones, all_scipure, all_acd, all_wide, all_large, all_pho, X_pool
    gc.collect()
    # Stage 2: per-det 4-param fit with β fixed
    coefs_per_det = {}
    for box in "ABC":
        for det in range(6):
            d = data[(box, det)]
            if len(d["pho"]) < N_MIN_PERDET:
                coefs_per_det[(box, det)] = (np.array([0.,0.,0.,0.,0.]), False)
                continue
            X = np.column_stack([d["ones"], d["scipure"], d["acd"], d["large"]])
            y = d["pho"] - beta_g * d["wide"]
            c4, *_ = np.linalg.lstsq(X, y, rcond=None)
            coefs_per_det[(box, det)] = (np.array([c4[0], c4[1], c4[2], beta_g, c4[3]]), False)
    rms_box, b_box = {}, {}
    for box in "ABC":
        preds, bvals = [], []
        for det in range(6):
            coef, dt_corr = coefs_per_det[(box, det)]
            preds.append(_predict(data[(box, det)], coef, dt_corr))
            bvals.append(coef[0])
        pred = np.concatenate(preds)
        rms_box[box] = float(np.sqrt(np.mean((pred - box_actuals[box])**2)))
        b_box[box] = float(np.mean(bvals))
    t_total = _time.time() - t0
    print(f"  [β global] total={t_total:.1f}s  β_global={beta_g:.4f}  "
          f"RMS={rms_box['A']:.2f}/{rms_box['B']:.2f}/{rms_box['C']:.2f}",
          flush=True)
    # Insert after H7 in results (β global)
    results.insert(7, ("β global (shared)", "73 total", rms_box, b_box))

    # ===== Print summary table =====
    print(f"\n{'='*100}")
    print(f"{'Hypothesis':<28s}  {'params/det':>10s}  "
          f"{'A RMS':>7s}  {'B RMS':>7s}  {'C RMS':>7s}  "
          f"{'A <b>':>7s}  {'B <b>':>7s}  {'C <b>':>7s}")
    print(f"{'='*100}")
    v8_rms = results[0][2]
    for name, n_free, rms_box, b_box in results:
        nf = str(n_free)
        delta_str = ""
        if name != "V8 baseline":
            # Compute total RMS delta vs V8
            mean_now = np.mean(list(rms_box.values()))
            mean_v8 = np.mean(list(v8_rms.values()))
            d = 100.0 * (mean_now - mean_v8) / mean_v8
            delta_str = f"  Δ={d:+5.1f}%"
        print(f"{name:<28s}  {nf:>10s}  "
              f"{rms_box['A']:>7.2f}  {rms_box['B']:>7.2f}  {rms_box['C']:>7.2f}  "
              f"{b_box['A']:>+7.1f}  {b_box['B']:>+7.1f}  {b_box['C']:>+7.1f}{delta_str}")
    print(f"{'='*100}")

    # ===== Bar chart =====
    fig, (ax_rms, ax_b) = plt.subplots(2, 1, figsize=(13, 9),
                                         gridspec_kw={"height_ratios":[3,2]})
    names = [r[0] for r in results]
    boxes = ["A", "B", "C"]
    colors = {"A":"#d62728", "B":"#2ca02c", "C":"#1f77b4"}
    width = 0.27
    x = np.arange(len(results))
    for i, box in enumerate(boxes):
        rms_vals = [r[2][box] for r in results]
        ax_rms.bar(x + (i-1)*width, rms_vals, width=width,
                    label=f"Box {box}", color=colors[box], alpha=0.85)
    ax_rms.axhline(np.mean([results[0][2][b] for b in boxes]),
                    color='gray', ls='--', lw=1, label="V8 mean")
    ax_rms.set_xticks(x)
    ax_rms.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax_rms.set_ylabel("RMS_PHO [cnt/s/det]")
    ax_rms.set_title("Hypothesis bake-off: training-band RMS by box")
    ax_rms.legend(loc="upper left", fontsize=9)
    ax_rms.grid(axis="y", alpha=0.3)

    # Bottom: mean b per hypothesis
    for i, box in enumerate(boxes):
        b_vals = [r[3][box] for r in results]
        ax_b.bar(x + (i-1)*width, b_vals, width=width,
                  label=f"Box {box}", color=colors[box], alpha=0.85)
    ax_b.axhline(0, color='black', lw=0.8)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax_b.set_ylabel("⟨b⟩ over 6 dets [cnt/s/det]")
    ax_b.set_title("Mean intercept b — closer to 0 = more 'clean' physics")
    ax_b.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    desktop = Path.home() / "Desktop" / OUT.name
    fig.savefig(desktop, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {OUT}\n       {desktop}")


if __name__ == "__main__":
    main()
