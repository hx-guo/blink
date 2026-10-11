"""Step 1: GBM detector geometry and data-quality checks for GRB 221009A.

- detector-source angles vs time from the daily poshist (Swift-XRT position);
- per-detector TTE gaps (packet loss) in T0-400...+800 s from the continuous TTE;
- coverage of the trigger CTIME/CSPEC/TTE files;
- overview light curve.
Outputs in $GBM_OUT/geom/.
"""
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import fits
from gdt.missions.fermi.gbm.poshist import GbmPosHist
from gdt.missions.fermi.gbm.detectors import GbmDetectors
import p221009_gbm_common as C

od = C.outdir("geom")

# ---------------------------------------------------------------- angles
ph = GbmPosHist.open(glob.glob(os.path.join(C.TTE_DIR, "glg_poshist_all_221009_v*.fit"))[0])
frame = ph.get_spacecraft_frame()
src = SkyCoord(C.RA, C.DEC, unit="deg")
tgrid = np.arange(-20.0, 600.1, 2.0)
rows = []
for t in tgrid:
    f = frame.at(Time(C.T0 + t, format="fermi"))
    s = src.transform_to(f)
    row = dict(t=t, sc_az=float(np.atleast_1d(s.az.deg)[0]), sc_zen=90.0 - float(np.atleast_1d(s.el.deg)[0]))
    for d in GbmDetectors:
        row[d.name] = float(np.atleast_1d(d.skycoord(f).separation(src).deg)[0])
    rows.append(row)
import pandas as pd
A = pd.DataFrame(rows)
C.write_csv(A, "geom", "angles.csv")
with open(os.path.join(od, "angles_summary.txt"), "w") as fo:
    for t in [-1.0, 4.0, 60.0, 140.0, 176.0, 200.0, 300.0, 500.0]:
        r = A.iloc[np.argmin(abs(A.t - t))]
        ang = " ".join("%s=%5.1f" % (d, r[d]) for d in C.DETS)
        fo.write("t=%6.1f  az=%6.1f zen=%6.1f  %s\n" % (r.t, r.sc_az, r.sc_zen, ang))
    for lo, hi in [(-2, 10), (-2, 44), (120, 166), (170, 216)]:
        m = (A.t >= lo) & (A.t <= hi)
        fo.write("[%g,%g] max angle: %s\n" % (lo, hi, " ".join("%s=%.1f" % (d, A[d][m].max()) for d in C.DETS)))

# ---------------------------------------------------------------- TTE gaps + LC
edges = np.arange(-400.0, 800.001, 1.0)
lc = dict(t=edges[:-1] + 0.5)
gaps = []
for d in C.DETS:
    tte = C.open_tte(d)
    t = C.rel(tte.data.times)
    lc[d] = np.histogram(t, edges)[0]
    dt = np.diff(t)
    g = np.where(dt > 0.05)[0]          # >50 ms with no events (normal spacing ~1 ms)
    for i in g:
        gaps.append(dict(det=d, start=t[i], stop=t[i + 1], dur=dt[i]))
    print(d, len(t), "events; gaps>50ms:", len(g), flush=True)
LC = pd.DataFrame(lc)
C.write_csv(LC, "geom", "tte_rate_1s_alldet.csv")
G = pd.DataFrame(gaps)
C.write_csv(G, "geom", "tte_gaps.csv")
with open(os.path.join(od, "tte_gaps_summary.txt"), "w") as fo:
    if len(G):
        for d, g in G.groupby("det"):
            fo.write("%s: %d gaps, total %.2f s; first %.3f last %.3f; gaps outside BTIs: %s\n" % (
                d, len(g), g.dur.sum(), g.start.min(), g.stop.max(),
                ", ".join("%.2f-%.2f" % (a, b) for a, b in zip(g.start, g.stop)
                          if not any(lo - 1 < a < hi + 1 for lo, hi in C.BTI))[:600]))

# ---------------------------------------------------------------- trigger file coverage
with open(os.path.join(od, "trigger_files.txt"), "w") as fo:
    for f in sorted(glob.glob(os.path.join(C.TRIG_DIR, "glg_*_n4_*.pha")) +
                    glob.glob(os.path.join(C.TRIG_DIR, "glg_*_b1_*.pha")) +
                    glob.glob(os.path.join(C.TRIG_DIR, "glg_tte_n4_*.fit")) +
                    glob.glob(os.path.join(C.TRIG_DIR, "glg_trigdat*.fit"))):
        h = fits.open(f)
        fo.write("%s TSTART-T0=%.3f TSTOP-T0=%.3f TRIGTIME=%s\n" % (
            os.path.basename(f), h[0].header["TSTART"] - C.T0, h[0].header["TSTOP"] - C.T0, h[0].header.get("TRIGTIME")))

# ---------------------------------------------------------------- overview figure
fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
for d in ["n3", "n4", "n6", "n7", "n8"]:
    ax[0].plot(LC.t, LC[d], lw=0.6, label=d)
for d in C.BGO:
    ax[1].plot(LC.t, LC[d], lw=0.6, label=d)
for d in C.DETS:
    ax[2].plot(A.t, A[d], lw=0.8, label=d)
for a in ax[:2]:
    a.set_yscale("log"); a.legend(fontsize=6, ncol=5)
    for lo, hi in C.BTI:
        a.axvspan(lo, hi, color="0.85")
ax[0].set_ylabel("NaI counts/s (all ch)"); ax[1].set_ylabel("BGO counts/s")
ax[2].set_ylabel("angle to GRB (deg)"); ax[2].axhline(60, color="k", ls=":")
ax[2].legend(fontsize=5, ncol=7); ax[2].set_xlabel("t - T0(GBM) (s)")
ax[2].set_xlim(-100, 700)
fig.tight_layout(); fig.savefig(os.path.join(od, "overview.png"), dpi=130)
print("done")
