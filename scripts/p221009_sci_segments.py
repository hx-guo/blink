"""Step 1 of the HE variability scouting: list the 'real-event' intervals.

Real-event second: fill fraction < 5 % (n_fill / (n_obs + n_fill)) and
f_live > 0.8, valid in the v3 1 s table.  Also a 0.1 s fill map from the
reconstruction cache, so that sub-second clean stretches inside partly
filled seconds can be used.
writes out/segments_1s.csv, out/segments_runs.csv, out/fill_0p1s.npz
"""
import numpy as np
import pandas as pd
from p221009_sci_common import *

lc = pd.read_csv(LC_V3)
lc["fill_frac"] = lc.n_fill / np.maximum(lc.n_obs + lc.n_fill, 1)
lc["real"] = lc.valid & (lc.fill_frac < 0.05) & (lc.f_live > 0.8) & (lc.n_obs > 0)
lc["he_gross"] = lc.n_obs                       # recorded events in the second
lc["he_net_obs"] = lc.n_obs - lc.B_obs          # net recorded (no dt correction)
lc["he_net_dt"] = lc.dt                          # dead-time corrected net
lc["ratio_gross"] = lc.he_gross / lc.gecam_net.where(lc.gecam_net > 0)
lc["ratio_net"] = lc.he_net_dt / lc.gecam_net_dc.where(lc.gecam_net_dc > 0)
cols = ["t", "valid", "n_obs", "n_fill", "fill_frac", "f_live", "B_obs", "he_net_obs",
        "he_net_dt", "gecam_net", "gecam_net_dc", "gecam_live", "ratio_gross", "ratio_net", "real"]
lc[cols].to_csv(f"{OUT}/segments_1s.csv", index=False, float_format="%.5g")

# contiguous runs of real seconds
r = lc.real.to_numpy().astype(int)
edges = np.flatnonzero(np.diff(np.r_[0, r, 0]))
runs = []
for a, b in zip(edges[::2], edges[1::2]):
    s = lc.iloc[a:b]
    runs.append(dict(t0=int(s.t.iloc[0]), t1=int(s.t.iloc[-1]) + 1, n_s=b - a,
                     he_gross=s.he_gross.mean(), he_net_obs=s.he_net_obs.mean(),
                     he_net_dt=s.he_net_dt.mean(), B_obs=s.B_obs.mean(),
                     gecam_net=s.gecam_net.mean(), gecam_net_dc=s.gecam_net_dc.mean(),
                     fill_max=s.fill_frac.max(), flive_min=s.f_live.min(),
                     ratio_net_med=s.ratio_net.median()))
runs = pd.DataFrame(runs)
runs.to_csv(f"{OUT}/segments_runs.csv", index=False, float_format="%.5g")
pd.set_option("display.width", 200, "display.max_rows", 500)
print(runs.to_string(index=False))

# seconds with notable source and their status (for the windows of interest)
src = lc[(lc.he_net_dt > 2000) & lc.valid]
print("\nsource seconds (net_dt > 2000):", len(src), " of which real:", int(src.real.sum()))
print(lc[(lc.t >= 170) & (lc.t < 230) | (lc.t >= 430) & (lc.t < 610) & (lc.t % 5 == 0)]
      [["t", "n_obs", "fill_frac", "f_live", "he_net_dt", "gecam_net_dc", "ratio_net", "real"]].to_string(index=False))

# 0.1 s fill map
d = np.load(EVT_REC)
t, fill = d["t"], d["fill"]
e = np.arange(-51.0, 760.0 + 1e-9, 0.1)
nf = np.histogram(t[fill], e)[0]
no = np.histogram(t[~fill], e)[0]
np.savez(f"{OUT}/fill_0p1s.npz", edges=e, n_fill=nf, n_obs=no)
ff = nf / np.maximum(nf + no, 1)
clean = (ff < 0.01) & (no > 0)
ed = np.flatnonzero(np.diff(np.r_[0, clean.astype(int), 0]))
print("\n0.1 s clean (fill<1%) runs >= 2 s inside 170-620 s:")
for a, b in zip(ed[::2], ed[1::2]):
    if e[a] > 170 and e[b] < 620 and (b - a) >= 20:
        print(f"  {e[a]:7.1f} - {e[b]:7.1f}  ({(b-a)/10:.1f} s, mean obs {no[a:b].mean()*10:.0f} /s)")
