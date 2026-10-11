"""Accumulate HE 1K background pulse-height spectra per gain mode (GRB 221009A
response study).  Runs on the IHEP farm, one job per day.

    python p221009_resp_bkgspec.py 2022-10-07 [OUTDIR]

For each hour of the day the HE-HV file decides the mode of every second from
the mean of the 18 PHODet high voltages: low gain (LG) for -880 < HV < -760 V,
normal (NG) for HV < -930 V; seconds within 120 s of a mode change, or with HV
outside both bands (off / ramping), are dropped.  Events are histogrammed by
mode, ACD index (1 = no veto hit), pulse-width class, detector and unwrapped channel
(raw < 20 -> +256, as eband.rs).  The 241Am calibration events (Event_Type 1)
are kept in a separate histogram.  Per-second total rates are kept so that
quiet (low geomagnetic background) seconds can be selected afterwards."""
import glob
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.time import Time

D1K = "/hxmt/work/HXMT-DATA/1K"
PW_EDGES = [(54, 70), (71, 75), (76, 120), (121, 255)]   # NaI / gap / CsI / wide CsI
NCH = 300
MET0 = Time("2012-01-01T00:00:00", scale="utc")


def latest(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda f: int(f.split("_V")[-1].split("_")[0]))


def second_modes(hv_file):
    with fits.open(hv_file) as h:
        d = h["HE_HV_PHODet"].data
        t = np.asarray(d["Time"], float)
        hv = np.mean([np.asarray(d[f"HV_PHODet_{i}"], float) for i in range(18)], axis=0)
    mode = np.full(t.size, -1, int)
    mode[(hv > -880) & (hv < -760)] = 1          # LG
    mode[hv < -930] = 0                           # NG
    # drop +-120 s around any change of state (incl. off)
    change = np.flatnonzero(np.diff(mode) != 0)
    bad = np.zeros(t.size, bool)
    for c in change:
        bad |= np.abs(t - t[c]) <= 120
    mode[bad] = -1
    return np.floor(t).astype(np.int64), mode, hv


def main():
    day = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "/scratchfs2/gecam/guohx/p221009/resp/out"
    os.makedirs(out, exist_ok=True)
    ymd = day.replace("-", "")
    hist = np.zeros((2, 2, len(PW_EDGES), 18, NCH), np.int64)
    am = np.zeros((2, 18, NCH), np.int64)
    sec_rows = []
    for hh in range(24):
        stem = f"{D1K}/Y{ymd[:6]}/{ymd}-*/HXMT_{ymd}T{hh:02d}"
        fe = latest(stem + "_HE-Evt_FFFFFF_V*_1K.FITS")
        fh = latest(stem + "_HE-HV_FFFFFF_V*_1K.FITS")
        if fe is None or fh is None:
            print(f"{day}T{hh:02d}: missing", fe, fh, flush=True)
            continue
        tsec, smode, shv = second_modes(fh)
        with fits.open(fe, memmap=True) as h:
            d = h[1].data
            t = np.asarray(d["Time"], float)
            det = np.asarray(d["Det_ID"], np.int16)
            ch = np.asarray(d["Channel"], np.int16)
            pw = np.asarray(d["Pulse_Width"], np.int16)
            acd = np.asarray(d["ACD"])
            et = np.asarray(d["Event_Type"], np.int8)
        ch = np.where(ch < 20, ch + 256, ch)
        isec = np.floor(t).astype(np.int64)
        s0 = tsec.min()
        mode_arr = np.full(tsec.max() - s0 + 1, -1, int)
        mode_arr[tsec - s0] = smode
        k = isec - s0
        ok = (k >= 0) & (k < mode_arr.size) & (det >= 0) & (det < 18)
        m = np.full(t.size, -1, int)
        m[ok] = mode_arr[k[ok]]
        acd0 = ~acd.reshape(acd.shape[0], -1).any(axis=1) if acd.ndim > 1 else (acd == 0)
        del acd
        phys = et == 0
        for imode in (0, 1):
            sel_m = m == imode
            if not sel_m.any():
                continue
            # 241Am events
            sa = sel_m & (et == 1)
            np.add.at(am[imode], (det[sa].astype(np.int64), np.clip(ch[sa], 0, NCH - 1).astype(np.int64)), 1)
            for ip, (lo, hi) in enumerate(PW_EDGES):
                sp = sel_m & phys & (pw >= lo) & (pw <= hi)
                for ia in (0, 1):
                    s = sp & (acd0 == bool(ia))
                    idx = det[s].astype(np.int64) * NCH + np.clip(ch[s], 0, NCH - 1)
                    hist[imode, ia, ip] += np.bincount(idx, minlength=18 * NCH).reshape(18, NCH)
        # per-second totals (physical events) for quiet-time selection
        ns = np.bincount(k[ok & phys], minlength=mode_arr.size)
        secs = np.arange(mode_arr.size) + s0
        sec_rows.append(np.column_stack([secs, mode_arr, ns]))
        print(f"{day}T{hh:02d}: {os.path.basename(fe)} n={t.size} "
              f"NG s={np.sum(smode == 0)} LG s={np.sum(smode == 1)}", flush=True)
    secs = np.vstack(sec_rows) if sec_rows else np.zeros((0, 3))
    np.savez_compressed(f"{out}/bkgspec_{ymd}.npz", hist=hist, am=am, secs=secs,
                        pw_edges=np.array(PW_EDGES))
    print("saved", f"{out}/bkgspec_{ymd}.npz", "exposure NG/LG s:",
          np.sum(secs[:, 1] == 0), np.sum(secs[:, 1] == 1))


if __name__ == "__main__":
    main()
