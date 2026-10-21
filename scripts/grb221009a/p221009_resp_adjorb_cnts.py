"""Adjacent-orbit background, step 2b: per-second particle counters for the
burst day and reference windows (1K HE-Cnts): sum over 18 detectors of
Cnt_LarEvt (events above the upper discriminator) and Cnt_VetoDet (the
anticoincidence detectors), on the same second grid as step 2.

    python p221009_resp_adjorb_cnts.py OUTDIR LABEL DT [LABEL DT ...]"""
import os
import sys

import numpy as np
from astropy.io import fits

from p221009_resp_adjorb_extract import T0, W, hour_files


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    args = sys.argv[2:]
    for label, dt in zip(args[::2], args[1::2]):
        dt = float(dt)
        t1, t2 = T0 + dt + W[0], T0 + dt + W[1]
        nsec = W[1] - W[0]
        le = np.full(nsec, np.nan)
        veto = np.full(nsec, np.nan)
        for f in hour_files(t1, t2, "HE-Cnts"):
            with fits.open(f) as h:
                for ext, pre, arr in (("HE_Cnt_LargeEvent", "Cnt_LarEvt_", le), ("HE_Cnt_VetoDet", "Cnt_VetoDet_", veto)):
                    d = h[ext].data
                    tt = np.asarray(d["Time"], float)
                    v = np.sum([np.asarray(d[f"{pre}{i}"], float) for i in range(18)], axis=0)
                    s = (tt >= t1) & (tt < t2)
                    arr[np.floor(tt[s] - t1).astype(int)] = v[s]
        np.savez_compressed(f"{out}/cnts_{label}.npz", le=le, veto=veto, t_rel=np.arange(W[0], W[1]))
        print(label, "LE mean", np.nanmean(le), "veto mean", np.nanmean(veto), "missing", int(np.isnan(le).sum()), flush=True)


if __name__ == "__main__":
    main()
