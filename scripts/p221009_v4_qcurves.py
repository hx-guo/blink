"""nuFnu curves for Fig Q(c) evaluated with the gdt spectral functions used in
the GBM fits (GRB 221009A, quiescent interval II, GBM T0 121.219-164.228 s).

    run_gbm.sh -argu RUN p221009_v4_qcurves      (gdt venv)

  GBM n4+n8+b1, orb30: PL, COMP, Band from gbm/runs/qspec1/quietspec_fits.csv (gbm.md 3b);
  Lesage+2023 interval II BB (kT 19.27 keV) + COMP (alpha -0.4, Ep 5722 keV) with both
  amplitudes fitted to the same GBM data, orb30 and orb15 (gbm.md 8.1,
  quiet/quietpred.csv 'params' = ln A_BB, ln A_COMP).
Output $GBM_OUT/v4/qcurves_II.csv: E (keV) and E^2 N(E) (erg cm-2 s-1) per model.
"""
import numpy as np
import pandas as pd
from gdt.core.spectra.functions import Band, BlackBody, Comptonized, PowerLaw
import p221009_gbm_common as C

od = C.outdir("v4")
QS = "/scratchfs2/gecam/guohx/p221009/gbm/runs/qspec1/quiet/quietspec_fits.csv"
QP = "/scratchfs2/gecam/guohx/p221009/gbm/runs/qpred1/quiet/quietpred.csv"
e = np.geomspace(10.0, 2.0e4, 400)
out = {"E": e}
S = pd.read_csv(QS)
S = S[(S.window == "II") & (S.bkg == "orb30") & (S.dets == "n4+n8+b1")].set_index("model")
s = S.loc["PL"]
out["gbm_PL"] = PowerLaw().eval([s.A, s["index"], 100.0], e)
s = S.loc["COMP"]
out["gbm_COMP"] = Comptonized().eval([s.A, s.Epeak, s["index"], 100.0], e)
s = S.loc["Band"]
out["gbm_Band"] = Band().eval([s.A, s.Epeak, s.alpha, s.beta, 100.0], e)
Q = pd.read_csv(QP)
for bkg in ("orb30", "orb15"):
    q = Q[(Q.bkg == bkg) & (Q.model == "L_BBCOMP") & (Q.component == "L_BBCOMP")].iloc[0]
    lb, lc = (float(v) for v in q.params.split())
    out[f"lesage_BBCOMP_{bkg}"] = (BlackBody().eval([np.exp(lb), 19.27], e)
                                   + Comptonized().eval([np.exp(lc), 5722.0, -0.4, 100.0], e))
    print(bkg, "ln A_BB, ln A_COMP =", lb, lc, "HE pred (C=1)", q.he_pred_rate_C1)
for k in list(out):
    if k != "E":
        out[k] = e ** 2 * out[k] * C.KEV2ERG
pd.DataFrame(out).to_csv(f"{od}/qcurves_II.csv", index=False, float_format="%.6g")
print("done")
