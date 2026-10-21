#!/usr/bin/env python3
"""Test CRC failure hypothesis:
   Higher count rate → more CRC failures → Sci undercounts → predictor over-estimates Sci

Specifically: does CRC_rate scale with PHO_rate? Does correcting Sci by 1/(1-CRC_rate) help?
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load coef table (per-(box, det, date))
coef = pd.read_csv("coef_table_big.csv")

# Load CRC table (per-(box, date) — total events + total CRC)
crc = pd.read_csv("crc_table.csv", header=None,
                   names=["date", "box", "events", "seconds", "crc"])
crc["event_rate"] = crc["events"] / crc["seconds"]  # cnt/s/box (6 detectors summed)
crc["crc_rate"] = crc["crc"] / (crc["events"] + crc["crc"])  # fraction lost
crc["crc_per_sec"] = crc["crc"] / crc["seconds"]

print(f"CRC table: {len(crc)} rows")
print(f"  event_rate range: {crc['event_rate'].min():.0f} – {crc['event_rate'].max():.0f}")
print(f"  CRC rate range: {crc['crc_rate'].min():.5f} – {crc['crc_rate'].max():.5f}")
print(f"  CRC rate median: {crc['crc_rate'].median():.5f}")
print(f"  CRC rate mean: {crc['crc_rate'].mean():.5f}")

# === Plot 1: CRC rate vs event rate (test hypothesis directly) ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
box_colors = {"A": "C0", "B": "C1", "C": "C2"}

# Panel 1: CRC rate vs event rate
ax = axes[0, 0]
for box in "ABC":
    sub = crc[crc["box"] == box]
    ax.scatter(sub["event_rate"], sub["crc_rate"], s=8, alpha=0.4,
                color=box_colors[box], label=f"Box {box}")
ax.set_xlabel("Event rate per box [cnt/s] (sum of 6 dets)")
ax.set_ylabel("CRC failure rate (fraction lost)")
ax.set_title("CRC rate vs event rate — does it scale?")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=10)
# correlation
m_AB = crc[crc["box"].isin(["A","B"])]
rho = np.corrcoef(np.log10(m_AB["event_rate"]), np.log10(m_AB["crc_rate"].clip(1e-7)))[0,1]
ax.text(0.05, 0.95, f"ρ(log) = {rho:+.3f}", transform=ax.transAxes,
         fontsize=10, va="top", bbox=dict(facecolor="white", alpha=0.7))

# Panel 2: CRC count vs event count (raw, expect linear relationship if rate-prop)
ax = axes[0, 1]
for box in "ABC":
    sub = crc[crc["box"] == box]
    ax.scatter(sub["events"], sub["crc"], s=8, alpha=0.4,
                color=box_colors[box], label=f"Box {box}")
# trend line: if CRC ∝ events², then it's quadratic; if CRC ∝ events, linear
events_lin = np.linspace(crc["events"].quantile(0.05), crc["events"].quantile(0.95), 100)
# Linear fit
fit_lin = np.polyfit(crc["events"], crc["crc"], 1)
ax.plot(events_lin, np.polyval(fit_lin, events_lin), "k--",
        label=f"Linear: {fit_lin[1]:+.0f} + {fit_lin[0]:.5f}·events")
# Quadratic fit
fit_quad = np.polyfit(crc["events"], crc["crc"], 2)
ax.plot(events_lin, np.polyval(fit_quad, events_lin), "r:",
        label=f"Quadratic")
ax.set_xlabel("Total events")
ax.set_ylabel("Total CRC errors")
ax.set_title("CRC count vs events (linear trend = constant rate)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)

# Panel 3: CRC rate vs date (does it drift?)
ax = axes[1, 0]
crc["year"] = crc["date"].apply(lambda s: datetime.strptime(s, "%Y-%m-%d").year +
                                  (datetime.strptime(s, "%Y-%m-%d") - datetime(int(s[:4]), 1, 1)).total_seconds()/(365.25*86400))
for box in "ABC":
    sub = crc[crc["box"] == box]
    ax.scatter(sub["year"], sub["crc_rate"], s=8, alpha=0.4,
                color=box_colors[box], label=f"Box {box}")
ax.set_xlabel("Year")
ax.set_ylabel("CRC failure rate")
ax.set_title("CRC rate vs date")
ax.set_yscale("log")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=10)

# Panel 4: residual of composite predictor vs CRC rate
# Apply composite predictor and see if residual correlates with crc_rate
# Build composite formulas
formulas = {}
coef["year"] = coef["date"].apply(lambda s: datetime.strptime(s, "%Y-%m-%d").year +
                                  (datetime.strptime(s, "%Y-%m-%d") - datetime(int(s[:4]), 1, 1)).total_seconds()/(365.25*86400))
for box in "ABC":
    for param in ["a0", "a1", "a2", "a3"]:
        sub = coef[coef["box"] == box]
        agg = sub.groupby("year").agg({param: "mean", "PHO_med": "mean",
                                        "Large_med": "mean", "Wide_med": "mean"}).reset_index()
        y = agg[param].values
        X = np.column_stack([np.ones(len(agg)), agg["year"].values, agg["PHO_med"].values,
                              agg["Large_med"].values, agg["Wide_med"].values])
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        formulas[(box, param)] = c

# For each (box, det, date) compute composite-predicted Sci, and residual
merged = coef.merge(crc[["date", "box", "crc_rate", "event_rate"]].copy(), on=["date", "box"], how="inner")
print(f"\nMerged table: {len(merged)} rows (after CRC join)")

residuals = []
for _, row in merged.iterrows():
    box = row["box"]
    PHO = row["PHO_med"]; W = row["Wide_med"]; L = row["Large_med"]
    yr = row["year"]
    Sci_actual = row["Sci_med"]
    c0 = formulas[(box,"a0")]; c1 = formulas[(box,"a1")]
    c2 = formulas[(box,"a2")]; c3 = formulas[(box,"a3")]
    a0 = c0[0]+c0[1]*yr+c0[2]*PHO+c0[3]*L+c0[4]*W
    a1 = c1[0]+c1[1]*yr+c1[2]*PHO+c1[3]*L+c1[4]*W
    a2 = c2[0]+c2[1]*yr+c2[2]*PHO+c2[3]*L+c2[4]*W
    a3 = c3[0]+c3[1]*yr+c3[2]*PHO+c3[3]*L+c3[4]*W
    Sci_pred = a0 + a1*PHO + a2*W + a3*L
    residuals.append({
        "Sci_actual": Sci_actual, "Sci_pred": Sci_pred,
        "residual": Sci_actual - Sci_pred,
        "ratio": Sci_actual / Sci_pred if Sci_pred > 0 else np.nan,
        "crc_rate": row["crc_rate"], "PHO_med": PHO,
        "box": box,
    })
res = pd.DataFrame(residuals)

ax = axes[1, 1]
for box in "ABC":
    sub = res[res["box"] == box]
    ax.scatter(sub["crc_rate"], sub["residual"]/sub["Sci_pred"], s=4, alpha=0.3,
                color=box_colors[box], label=f"Box {box}")
ax.set_xlabel("CRC failure rate")
ax.set_ylabel("(Sci_obs − Sci_pred)/Sci_pred")
ax.set_title("Composite predictor residual vs CRC rate\nCRC hypothesis: residual should be NEGATIVE at high CRC")
ax.set_xscale("log")
ax.axhline(0, color="r", ls="--", lw=1)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=10)

# Compute correlation
mask = (res["crc_rate"] > 0) & np.isfinite(res["residual"]) & np.isfinite(res["Sci_pred"])
r = res[mask]
rho = np.corrcoef(np.log10(r["crc_rate"]), r["residual"]/r["Sci_pred"])[0, 1]
ax.text(0.05, 0.95, f"ρ(log_CRC, frac_resid) = {rho:+.3f}", transform=ax.transAxes,
         fontsize=10, va="top", bbox=dict(facecolor="white", alpha=0.7))

fig.tight_layout()
out = "plots/crc_hypothesis.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved: {out}")

# === Test if correcting Sci by 1/(1-CRC_rate) reduces residual ===
print(f"\n=== Test: does Sci_corrected = Sci_obs / (1 - CRC_rate) match prediction better? ===")
print(f"{'Method':>20s}  RMS [cnt/s]")
res["Sci_corrected"] = res["Sci_actual"] / (1 - res["crc_rate"])
res["resid_uncorr"] = res["Sci_actual"] - res["Sci_pred"]
res["resid_corr"] = res["Sci_corrected"] - res["Sci_pred"]
print(f"  {'Uncorrected':>20s}  {np.sqrt(np.mean(res['resid_uncorr']**2)):>6.2f}")
print(f"  {'CRC-corrected':>20s}  {np.sqrt(np.mean(res['resid_corr']**2)):>6.2f}")

# Average CRC rate
print(f"\nMean CRC rate: {res['crc_rate'].mean():.5f}")
print(f"Median CRC rate: {res['crc_rate'].median():.5f}")
print(f"95th pct CRC rate: {res['crc_rate'].quantile(0.95):.5f}")
print(f"Max CRC rate: {res['crc_rate'].max():.5f}")
