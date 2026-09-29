#!/usr/bin/env python3
"""Band-free energy recovery validation on GRB 260226A.

Digs a clean gap in one box, recovers filler channels from the other two
boxes (deterministic quantile of the reference in-gap distribution), and
checks two claims:
  Fig 1: the recovered spectrum reproduces the true deleted spectrum,
         tail included (vs random sampling as a noisier control).
  Fig 2: within-window channel<->time assignment. Sorted assignment makes
         a fake soft-early/hard-late ramp; bit-reversal (low-discrepancy)
         removes it. Compared against the true deleted (time, channel).
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = 446726273.0  # leap-second-aware MET of 2026-02-26T10:37:50 UTC (1B/1K MET includes 3 leap s since 2012)
PACK = Path("data/pack_260226a")
BOXES = ("a", "b", "c")
D = 0.100          # gap duration, s
WIN = 0.020        # spectral sub-window (resolution unit), s
LC_BIN = 0.004     # light-curve bin for Fig 2 (finer than WIN), s
PAD = 0.5 + 0.1    # calib window + clearance to nearest reset
HARD = 146         # hard-tail threshold (top old band edge), channel


def load_box(box):
    met, ch = [], []
    with open(PACK / f"box_{box}" / "events_obs.csv") as f:
        r = csv.reader(f); next(r)
        for row in r:
            if row[5] == "0":                 # skip SEC rows
                met.append(float(row[0])); ch.append(int(row[1]))
    met = np.asarray(met); ch = np.asarray(ch, dtype=np.int64)
    ch = np.where(ch < 20, ch + 256, ch)      # pulse-height wrap
    o = np.argsort(met)
    return met[o], ch[o]


def load_resets(box):
    out = []
    with open(PACK / f"box_{box}" / "resets.csv") as f:
        r = csv.reader(f); next(r)
        for row in r:
            out.append((float(row[0]), float(row[1])))
    return out


def cnt(met, lo, hi):
    i, j = np.searchsorted(met, (lo, hi)); return j - i


def chan_in(met, ch, lo, hi):
    i, j = np.searchsorted(met, (lo, hi)); return ch[i:j]


def met_in(met, lo, hi):
    i, j = np.searchsorted(met, (lo, hi)); return met[i:j]


def even_quantiles(vals, n):
    """n channel values reproducing the distribution of vals (ascending)."""
    s = np.sort(vals); m = len(s)
    pos = np.clip(((np.arange(n) + 0.5) / n * m).astype(int), 0, m - 1)
    return s[pos]


def phi2(i):
    f, b = 0.0, 0.5
    while i > 0:
        f += (i & 1) * b; i >>= 1; b /= 2
    return f


def ldranks(n):
    """slot k (time order) -> channel rank, low-discrepancy (bit-reversal)."""
    ph = np.array([phi2(i) for i in range(n)])
    return np.argsort(np.argsort(ph, kind="stable"), kind="stable")


# ---- load ----
box = {b: load_box(b) for b in BOXES}
resets = [iv for b in BOXES for iv in load_resets(b)]
def clean(lo, hi):
    return all(not (s < hi and e > lo) for s, e in resets)

# ---- pick brightest clean gap for target 'a' ----
tgt = "a"; refs = ("b", "c")
tmet = box[tgt][0]
tlo = max(box[b][0][0] for b in BOXES) + PAD
thi = min(box[b][0][-1] for b in BOXES) - D - PAD
best = None
for t0 in np.arange(tlo, thi, 0.005):
    if not clean(t0 - PAD, t0 + D + PAD):
        continue
    c = cnt(tmet, t0, t0 + D)
    if best is None or c > best[1]:
        best = (t0, c)
t0, _ = best
g_lo, g_hi = t0, t0 + D
print(f"gap: t0={t0 - T0:+.4f}s (rel T0), D={D*1e3:.0f}ms")
for b in BOXES:
    print(f"  box {b}: {cnt(box[b][0], g_lo, g_hi)} events in gap")

# ---- k_tot from +-0.5s adjacent windows ----
wins = ((g_lo - 0.5, g_lo), (g_hi, g_hi + 0.5))
a_cal = sum(cnt(box[tgt][0], lo, hi) for lo, hi in wins)
bc_cal = sum(cnt(box[r][0], lo, hi) for r in refs for lo, hi in wins)
k_tot = a_cal / bc_cal
print(f"k_tot (a / (b+c), adjacent) = {k_tot:.4f}")

# ---- truth (deleted target events) ----
truth_t = met_in(box[tgt][0], g_lo, g_hi)
truth_c = chan_in(*box[tgt], g_lo, g_hi)
N_true = len(truth_c)

# ---- Fig 1: recovered spectrum (whole-gap quantile) vs truth ----
ref_c_gap = np.concatenate([chan_in(*box[r], g_lo, g_hi) for r in refs])
N_lost = int(round(len(ref_c_gap) * k_tot))
rec_c = even_quantiles(ref_c_gap, N_lost)
rng = np.random.default_rng(0)
rand_c = rng.choice(ref_c_gap, size=N_lost, replace=True)   # control
print(f"N_true={N_true}  N_lost(recovered)={N_lost}")
for name, arr in (("truth", truth_c), ("quantile", rec_c), ("random", rand_c)):
    frac = np.mean(arr >= HARD) * 100
    print(f"  {name:9s}: n={len(arr)}  hard-tail(>= {HARD}) = {frac:5.2f}%  "
          f"mean ch={arr.mean():.1f}")

# ---- Fig 2: within-window channel<->time assignment ----
edges = np.arange(g_lo, g_hi + 1e-9, WIN)
all_t, c_true, c_bitrev, c_sorted = [], [], [], []
corr_true = corr_bitrev = corr_sorted = 0.0; nwin = 0
for wlo, whi in zip(edges[:-1], edges[1:]):
    tt = met_in(box[tgt][0], wlo, whi); tc = chan_in(*box[tgt], wlo, whi)
    nw = len(tt)
    if nw < 2:
        continue
    ref_c_w = np.concatenate([chan_in(*box[r], wlo, whi) for r in refs])
    if len(ref_c_w) == 0:
        continue
    qs = even_quantiles(ref_c_w, nw)          # ascending channel values
    ranks = ldranks(nw)
    order_t = np.argsort(tt)                   # slots in time order
    cs = np.empty(nw); cb = np.empty(nw)
    for k in range(nw):
        cs[order_t[k]] = qs[k]                 # sorted: monotone in time (WRONG)
        cb[order_t[k]] = qs[ranks[k]]          # bit-reversal (RIGHT)
    all_t.append(tt); c_true.append(tc); c_bitrev.append(cb); c_sorted.append(cs)
    # within-window time<->channel correlation
    tn = (tt - tt.mean())
    for acc, cc in (("t", tc), ("b", cb), ("s", cs)):
        cn = cc - cc.mean()
        r = (tn @ cn) / (np.sqrt(tn @ tn) * np.sqrt(cn @ cn) + 1e-12)
        if acc == "t": corr_true += r
        elif acc == "b": corr_bitrev += r
        else: corr_sorted += r
    nwin += 1
all_t = np.concatenate(all_t)
c_true = np.concatenate(c_true); c_bitrev = np.concatenate(c_bitrev); c_sorted = np.concatenate(c_sorted)
print(f"\nwithin-window time<->channel Pearson corr (mean over {nwin} windows):")
print(f"  truth      : {corr_true/nwin:+.3f}")
print(f"  bit-reversal: {corr_bitrev/nwin:+.3f}")
print(f"  sorted     : {corr_sorted/nwin:+.3f}   <- fake drift")


def binmean(t, c, edges):
    idx = np.digitize(t, edges) - 1
    m = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        sel = idx == i
        if sel.any():
            m[i] = c[sel].mean()
    return m

lc_edges = np.arange(g_lo, g_hi + 1e-9, LC_BIN)
ctr = (lc_edges[:-1] + lc_edges[1:]) / 2 - T0

# ---- plot ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# panel 1: spectrum
bins = np.arange(20, 285, 6)
ax[0].hist(truth_c, bins=bins, histtype="step", lw=2.2, color="k", label=f"truth (deleted, n={N_true})")
ax[0].hist(rec_c, bins=bins, histtype="step", lw=2.0, color="tab:blue", label=f"recovered quantile (n={N_lost})")
ax[0].hist(rand_c, bins=bins, histtype="step", lw=1.2, color="tab:orange", ls="--", label="random sampling (control)")
ax[0].axvline(HARD, color="grey", ls=":", lw=1)
ax[0].set_yscale("log")
ax[0].set_xlabel("channel"); ax[0].set_ylabel("counts / bin")
ax[0].set_title("Fig 1  recovered spectrum vs truth (log-y, tail visible)")
ax[0].legend(fontsize=9)

# panel 2: mean channel vs time (drift)
ax[1].plot(ctr * 1e3, binmean(all_t, c_true, lc_edges), "-", color="k", lw=2.2, label="truth")
ax[1].plot(ctr * 1e3, binmean(all_t, c_bitrev, lc_edges), "-", color="tab:blue", lw=2.0, label="bit-reversal (right)")
ax[1].plot(ctr * 1e3, binmean(all_t, c_sorted, lc_edges), "--", color="tab:red", lw=1.8, label="sorted (fake drift)")
for e in edges:
    ax[1].axvline((e - T0) * 1e3, color="grey", ls=":", lw=0.6, alpha=0.5)
ax[1].set_xlabel(f"time - T0 [ms]   (dotted = {WIN*1e3:.0f}ms sub-window edges)")
ax[1].set_ylabel(f"mean channel / {LC_BIN*1e3:.0f}ms bin")
ax[1].set_title("Fig 2  time<->channel: sorted ramps, bit-reversal doesn't")
ax[1].legend(fontsize=9)

fig.tight_layout()
out = Path("data/bandfree_validation.png")
fig.savefig(out, dpi=130)
print(f"\nwrote {out}")
