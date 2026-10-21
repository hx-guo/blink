#!/usr/bin/env python3
"""Test CRC hypothesis at per-second resolution:
   Does CRC count per second scale with EVT count per second?
   If yes: CRC scales with rate → loses Sci → explains high-rate undercount
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("per_sec_crc.csv")
print(f"Loaded {len(df)} per-second rows across {df['date'].nunique()} dates × {df['box'].nunique()} boxes")
print(f"Total EVT: {df['evt'].sum():,}")
print(f"Total CRC: {df['crc'].sum():,}")
print(f"Mean CRC rate: {df['crc'].sum() / df['evt'].sum() * 100:.4f}%")
print(f"Seconds with CRC>0: {(df['crc'] > 0).sum()}/{len(df)} = {(df['crc'] > 0).mean()*100:.2f}%")

# === Plot 1: CRC count vs EVT count per second (the key test) ===
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

date_styles = {
    "20171001": ("C0", "2017-10-01"),
    "20180315": ("C1", "2018-03-15"),
    "20200415": ("C2", "2020-04-15"),
    "20221009": ("C3", "2022-10-09"),
}

# Top-left: CRC vs EVT scatter (all dates, log-log)
ax = axes[0, 0]
for date, (color, label) in date_styles.items():
    sub = df[df["date"] == int(date)]
    crc_pos = sub[sub["crc"] > 0]
    ax.scatter(crc_pos["evt"], crc_pos["crc"], s=5, alpha=0.4, color=color, label=label)
ax.set_xlabel("EVT/sec")
ax.set_ylabel("CRC/sec")
ax.set_title("CRC vs EVT per second (CRC>0 only)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9)

# Top-middle: binned CRC rate vs EVT rate
ax = axes[0, 1]
for date, (color, label) in date_styles.items():
    sub = df[df["date"] == int(date)].copy()
    bins = np.logspace(np.log10(max(sub["evt"].min(), 100)), np.log10(sub["evt"].quantile(0.99) + 1), 25)
    sub["bin"] = pd.cut(sub["evt"], bins)
    grouped = sub.groupby("bin", observed=True).agg(evt_mean=("evt", "mean"),
                                                      crc_mean=("crc", "mean"),
                                                      crc_rate=("crc", lambda x: x.sum() / max(sub["evt"].mean()*len(x), 1)),
                                                      n=("evt", "size")).reset_index()
    grouped = grouped[grouped["n"] >= 10]
    ax.plot(grouped["evt_mean"], grouped["crc_mean"], "-", color=color, lw=1.5, marker="o",
             markersize=4, label=label)
ax.set_xlabel("EVT/sec (binned mean)")
ax.set_ylabel("Mean CRC/sec")
ax.set_title("Binned mean CRC vs EVT per second")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9)

# Top-right: distribution of CRC counts (log)
ax = axes[0, 2]
all_crc = df["crc"].values
ax.hist(all_crc[all_crc > 0], bins=np.logspace(0, 2.5, 30), color="C3", edgecolor="k")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("CRC count per second")
ax.set_ylabel("count of seconds")
ax.set_title(f"Distribution of CRC/sec (when >0)\nseconds with CRC: {(df['crc']>0).sum()}/{len(df)}")
ax.grid(alpha=0.3, which="both")

# Bottom-left: Pearson correlation per date
ax = axes[1, 0]
xs = []
labels = []
rhos_lin = []
rhos_loglog = []
rhos_pos = []
for date, (color, label) in date_styles.items():
    sub = df[df["date"] == int(date)]
    rho_lin = np.corrcoef(sub["evt"], sub["crc"])[0, 1]
    sub_pos = sub[sub["crc"] > 0]
    rho_pos = np.corrcoef(sub_pos["evt"], sub_pos["crc"])[0, 1] if len(sub_pos) > 5 else np.nan
    rho_log = np.corrcoef(np.log10(sub_pos["evt"].clip(1)), np.log10(sub_pos["crc"].clip(1)))[0, 1] if len(sub_pos) > 5 else np.nan
    xs.append(label)
    rhos_lin.append(rho_lin)
    rhos_pos.append(rho_pos)
    rhos_loglog.append(rho_log)
x = np.arange(len(xs))
ax.bar(x - 0.27, rhos_lin, 0.27, color="C0", label="ρ(evt, crc) all secs")
ax.bar(x, rhos_pos, 0.27, color="C1", label="ρ(evt, crc) crc>0 only")
ax.bar(x + 0.27, rhos_loglog, 0.27, color="C3", label="ρ(log evt, log crc) crc>0")
ax.set_xticks(x)
ax.set_xticklabels(xs, rotation=15)
ax.set_ylabel("Pearson ρ")
ax.set_title("CRC-EVT correlation per date")
ax.legend(fontsize=8)
ax.axhline(0, color="k", lw=0.5)
ax.grid(alpha=0.3, axis="y")

# Bottom-middle: time series of CRC rate per date
ax = axes[1, 1]
for date, (color, label) in date_styles.items():
    sub = df[df["date"] == int(date)].sort_values("met_sec")
    sub = sub[sub["box"] == "A"]  # Box A only for cleanness
    t = (sub["met_sec"] - sub["met_sec"].min()) / 60.0  # minutes
    ax.plot(t, sub["crc"], "-", color=color, lw=0.5, alpha=0.7, label=f"{label} A")
ax.set_xlabel("Minutes since file start")
ax.set_ylabel("CRC/sec (Box A)")
ax.set_title("CRC time series — bursty? Or rate-correlated?")
ax.set_yscale("symlog", linthresh=1)
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

# Bottom-right: predicted Sci loss IF CRC failures = lost events
# At what rate does CRC loss become significant?
ax = axes[1, 2]
for date, (color, label) in date_styles.items():
    sub = df[df["date"] == int(date)].copy()
    sub["loss_rate"] = sub["crc"] / (sub["evt"] + sub["crc"]).clip(1)
    bins = np.logspace(np.log10(max(sub["evt"].min(), 100)), np.log10(sub["evt"].quantile(0.99) + 1), 25)
    sub["bin"] = pd.cut(sub["evt"], bins)
    grouped = sub.groupby("bin", observed=True).agg(evt_mean=("evt", "mean"),
                                                      loss=("loss_rate", "mean"),
                                                      n=("evt", "size")).reset_index()
    grouped = grouped[grouped["n"] >= 10]
    ax.plot(grouped["evt_mean"], grouped["loss"]*100, "-", color=color, lw=1.5, marker="o",
             markersize=4, label=label)
ax.set_xlabel("EVT/sec (binned)")
ax.set_ylabel("Mean CRC loss rate [%]")
ax.set_title("CRC loss rate vs event rate\n(Hypothesis: should grow with EVT)")
ax.set_xscale("log")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

fig.tight_layout()
out = "plots/crc_per_sec.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")

# === Numeric summary ===
print(f"\n=== Per-date Pearson correlation (CRC count vs EVT count per second) ===")
print(f"{'Date':>11s} {'ρ_all_sec':>10s} {'ρ_crc>0':>10s} {'ρ_log_crc>0':>13s}")
for date, (color, label) in date_styles.items():
    sub = df[df["date"] == int(date)]
    rho_lin = np.corrcoef(sub["evt"], sub["crc"])[0, 1]
    sub_pos = sub[sub["crc"] > 0]
    rho_pos = np.corrcoef(sub_pos["evt"], sub_pos["crc"])[0, 1] if len(sub_pos) > 5 else np.nan
    rho_log = np.corrcoef(np.log10(sub_pos["evt"].clip(1)), np.log10(sub_pos["crc"].clip(1)))[0, 1] if len(sub_pos) > 5 else np.nan
    print(f"  {label} {rho_lin:>+10.3f} {rho_pos:>+10.3f} {rho_log:>+13.3f}")
