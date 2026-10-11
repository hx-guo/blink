"""TTE data-loss structure around the BTIs and trigger-vs-continuous TTE
consistency.  Outputs $GBM_OUT/geom/bti_gaps.csv, tte_consistency.txt."""
import glob
import os
import numpy as np
import pandas as pd
from astropy.io import fits
import p221009_gbm_common as C

rows = []
for det in ["n4", "n8", "b1"]:
    t = np.asarray(C.open_tte(det, (150.0, 600.0)).data.times)
    dt = np.diff(t)
    for a, b in [(150, 218.5), (218.5, 278), (278, 507), (507, 515), (515, 600)]:
        m = (t[:-1] >= a) & (t[:-1] < b)
        row = dict(det=det, t1=a, t2=b, events=int(m.sum()), mean_rate=m.sum() / (b - a))
        for g in [0.002, 0.005, 0.01, 0.05]:
            sel = m & (dt > g)
            row["n_gaps_gt%gms" % (g * 1e3)] = int(sel.sum())
            row["time_in_gaps_gt%gms" % (g * 1e3)] = float(dt[sel].sum())
        rows.append(row)
pd.DataFrame(rows).to_csv(os.path.join(C.outdir("geom"), "bti_gaps.csv"), index=False, float_format="%.6g")

with open(os.path.join(C.outdir("geom"), "tte_consistency.txt"), "w") as fo:
    for det in ["n4", "n8", "b1"]:
        h = fits.open(glob.glob(os.path.join(C.TRIG_DIR, "glg_tte_%s_bn221009553_v*.fit" % det))[0])
        tt = h["EVENTS"].data["TIME"] - C.T0; pt = h["EVENTS"].data["PHA"]
        c = C.open_tte(det, (-130.0, 480.0))
        tc = np.asarray(c.data.times); pc = np.asarray(c.data.channels)
        mt = (tt >= -130) & (tt < 480)
        same = (mt.sum() == len(tc)) and np.allclose(np.sort(tt[mt]), np.sort(tc), atol=1e-6) \
            and (np.sort(pt[mt]) == np.sort(pc)).all()
        fo.write("%s trigger TTE %d events, continuous %d events in -130..480 s; identical: %s\n"
                 % (det, mt.sum(), len(tc), same))
print(open(os.path.join(C.outdir("geom"), "tte_consistency.txt")).read())
print(pd.DataFrame(rows).to_string())
