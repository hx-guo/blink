"""Referee 2 M3 update: re-evaluate the quiescent-interval HE exclusion
significances of p221009_gbm_quietpred.py (quietpred.csv) with the resp
group's revised HE measurement (resp.md section 8): II-window net rate
+23 +- 15 (stat) +- 24 (sys) evt/s, and the Cnt_LargeEvt response to the
source (epsilon = 0.05-0.074), which suppresses the measured net rate of a
hard source by (1 - 0.98 epsilon): 0.93 (nominal) and 0.87 (95% bound).
The suppression is applied to the predicted rate.  Output
$GBM_OUT/quiet/quietpred_v2.csv."""
import os
import sys
import numpy as np
import pandas as pd
import p221009_gbm_common as C

src = sys.argv[1] if len(sys.argv) > 1 else "/scratchfs2/gecam/guohx/p221009/gbm/runs/qpred1/quiet/quietpred.csv"
Q = pd.read_csv(src)
MEAS, SST, SSY = 23.0, 15.0, 24.0
rows = []
for _, r in Q[Q.component.isin(["L_BBCOMP", "L_COMPonly", "PL"])].iterrows():
    for f in (1.0, 0.93, 0.87):
        pred = r.he_pred * f
        s_he = np.hypot(SST, SSY)
        s_pred = r.sigma_pred * f
        rows.append(dict(bkg=r.bkg, model=r.model, C=r.C, suppression=f, he_pred=pred, he_meas=MEAS, sigma_he=s_he,
                         excl_sigma=(pred - MEAS) / s_he, excl_sigma_incl_gbm=(pred - MEAS) / np.hypot(s_he, s_pred)))
O = pd.DataFrame(rows)
C.write_csv(O, "quiet", "quietpred_v2.csv")
print(O.to_string(float_format=lambda x: "%.2f" % x))
