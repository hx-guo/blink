"""Adjacent-orbit background, step 2: per-second CsI channel histograms for
the burst day and the selected reference windows.

    python p221009_resp_adjorb_extract.py OUTDIR LABEL DT [LABEL DT ...]

For each LABEL the window is MET T0+DT-420 ... T0+DT+2620 s; events from the
1K HE-Evt hour files covering it are histogrammed per second (relative to
T0+DT) and per unwrapped channel 0..299 for CsI (76 <= pw <= 255).  The HV
mean of every second is stored as well (mode check)."""
import glob
import os
import sys
from datetime import timedelta

import numpy as np
from astropy.io import fits
from astropy.time import Time

D1K = "/hxmt/work/HXMT-DATA/1K"
T0 = 339945423.0
W = (-420, 2620)
NCH = 300
MET0 = Time("2012-01-01T00:00:00", scale="utc")


def hour_files(t1, t2, kind):
    out = []
    # one hour of margin on each side (MET/UTC leap-second offsets at hour edges)
    start = (MET0 + timedelta(seconds=t1 - 3600)).datetime.replace(minute=0, second=0, microsecond=0)
    stop = (MET0 + timedelta(seconds=t2 + 3600)).datetime
    h = start
    while h <= stop:
        ymd = h.strftime("%Y%m%d")
        fs = sorted(glob.glob(f"{D1K}/Y{ymd[:6]}/{ymd}-*/HXMT_{ymd}T{h.hour:02d}_{kind}_FFFFFF_V*_1K.FITS"))
        if fs:
            out.append(max(fs, key=lambda f: int(f.split("_V")[-1].split("_")[0])))
        h += timedelta(hours=1)
    return out


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    args = sys.argv[2:]
    for label, dt in zip(args[::2], args[1::2]):
        dt = float(dt)
        t1, t2 = T0 + dt + W[0], T0 + dt + W[1]
        nsec = W[1] - W[0]
        h = np.zeros((nsec, NCH), np.int32)
        for f in hour_files(t1, t2, "HE-Evt"):
            with fits.open(f, memmap=True) as hd:
                d = hd[1].data
                t = np.asarray(d["Time"], float)
                s = (t >= t1) & (t < t2)
                if not s.any():
                    continue
                t = t[s]
                ch = np.asarray(d["Channel"], np.int16)[s]
                pw = np.asarray(d["Pulse_Width"], np.int16)[s]
                et = np.asarray(d["Event_Type"], np.int8)[s]
            ch = np.where(ch < 20, ch + 256, ch)
            k = (et == 0) & (pw >= 76) & (pw <= 255)
            isec = np.floor(t[k] - t1).astype(np.int64)
            np.add.at(h, (isec, np.clip(ch[k], 0, NCH - 1)), 1)
            print(label, os.path.basename(f), int(k.sum()), flush=True)
        hv = np.full(nsec, np.nan)
        for f in hour_files(t1, t2, "HE-HV"):
            with fits.open(f) as hd:
                d = hd["HE_HV_PHODet"].data
                tt = np.asarray(d["Time"], float)
                m = np.mean([np.asarray(d[f"HV_PHODet_{i}"], float) for i in range(18)], axis=0)
            s = (tt >= t1) & (tt < t2)
            hv[np.floor(tt[s] - t1).astype(int)] = m[s]
        np.savez_compressed(f"{out}/hist_{label}.npz", h=h, hv=hv, dt=dt, t_rel=np.arange(W[0], W[1]))
        print("saved", label, "dt", dt, "events", int(h.sum()), flush=True)


if __name__ == "__main__":
    main()
