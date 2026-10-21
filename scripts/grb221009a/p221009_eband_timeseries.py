#!/usr/bin/env python3
"""GRB 221009A: per-second, per-box energy-resolved series from HE 1B.

Spectral quantities use OBSERVED events only (events_obs.npz). A FIFO reset
discards a contiguous chunk of the event stream regardless of energy, so the
observed events of a second are an unbiased sample of what the front end
digitised; the fillers copy the reference boxes' spectrum by construction and
are therefore never used as spectral evidence here. Band light curves that need
the full count budget additionally use the reconstruction cache (obs + fill).

Output: data/p221009/eband/ts_1s.csv  (one row per second x box, plus box 'all')
"""
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits

OUT = Path("data/p221009/eband")
T0 = 339945423.0
MET_CORRECTION = 4.0          # 1B HE_Eng -> MET (engineering_prediction.py)
L_CYC_TO_SEC = 16e-6
ENG_APID = {"A": "0766", "B": "1009", "C": "1781"}
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}
EDGES = np.arange(-60.0, 761.0, 1.0)
NAI_PW, CSI_PW = (54, 70), (90, 255)
CSI_BANDS = [(70, 150), (150, 300), (300, 700)]     # keV deposited in CsI
NAI_BANDS = [(20, 50), (50, 100), (100, 250)]       # keV deposited in NaI
CACHE = "data/cache_221009a_reconstruct.csv"


def nai_lut():
    import sys
    import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../bursts",)]  # sibling and shared script modules
    from plot_hxmt_vs_ibis_bands import channel_to_kev_lut
    return channel_to_kev_lut()


def csi_lut():
    return np.load("data/hxmt_aux/csi_ch2e_211211.npy")


def classify(ch_raw, pw, lut_n, lut_c):
    """Column name -> boolean mask, for every quantity binned below."""
    chw = np.where(ch_raw < 20, ch_raw + 256, ch_raw)
    nai = (pw >= NAI_PW[0]) & (pw <= NAI_PW[1])
    csi = (pw >= CSI_PW[0]) & (pw <= CSI_PW[1]) & (ch_raw >= 30)
    e_n = lut_n[np.clip(ch_raw, 0, 255)]
    e_c = lut_c[np.clip(ch_raw, 0, 255)]
    m = {"all": np.ones(len(pw), bool), "nai": nai, "csi": csi,
         "pw71_89": (pw > 70) & (pw < 90), "pw_ge110": pw >= 110,
         "wrap": chw >= 256, "ch_ge240": chw >= 240}
    for lo, hi in CSI_BANDS:
        m[f"c{lo}_{hi}"] = csi & (chw < 256) & (e_c >= lo) & (e_c < hi)
    for lo, hi in NAI_BANDS:
        m[f"n{lo}_{hi}"] = nai & (chw < 256) & (e_n >= lo) & (e_n < hi)
    return m


def eng_series():
    rows = []
    for box, apid in ENG_APID.items():
        f = sorted(Path(f"data/1B/2022/20221009/{apid}").glob("*.fits"))[0]
        d = fits.open(f, memmap=True)["HE_Eng"].data
        met = d["Time"].astype(float) + d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0] \
            + MET_CORRECTION
        lc = d["Length_Time_Cycle"].astype(float)
        g = [BOX_OFFSET[box] + k for k in range(6)]
        D = np.vstack([d[f"DeadTime_PHODet_{i}"] for i in g]).astype(float)
        W = np.vstack([d[f"Cnt_CsI_PHODet_{i}"] for i in g]).astype(float)
        rows.append(pd.DataFrame(dict(
            box=box, t=np.round(met - T0), lcyc=lc * L_CYC_TO_SEC,
            flive=1 - D.mean(0) / np.where(lc > 0, lc, np.nan),
            d_spread=(D.max(0) - D.min(0)) / np.maximum(D.mean(0), 1),
            W=W.sum(0))))
    e = pd.concat(rows)
    return e[(e.t >= EDGES[0]) & (e.t < EDGES[-1])]


def main():
    lut_n, lut_c = nai_lut(), csi_lut()
    ev = np.load(OUT / "events_obs.npz")
    t = ev["met"] - T0
    m = classify(ev["raw_ch"].astype(int), ev["pw"].astype(int), lut_n, lut_c)
    frames = []
    for b, box in enumerate("ABC"):
        sel = ev["box"] == b
        df = pd.DataFrame({"t": EDGES[:-1], "box": box})
        for k, mk in m.items():
            df["o_" + k] = np.histogram(t[sel & mk], EDGES)[0]
        frames.append(df)

    # obs + fill counts per band (reconstruction cache; fillers carry channel+pw)
    rec = {box: {} for box in "ABC"}
    for chunk in pd.read_csv(CACHE, usecols=["box", "type", "met", "channel", "pulse_width"],
                             chunksize=4_000_000):
        chunk = chunk[chunk.type != "SEC"]
        mm = classify(chunk.channel.to_numpy(), chunk.pulse_width.to_numpy(), lut_n, lut_c)
        tt = chunk.met.to_numpy() - T0
        isf = (chunk.type == "FILL_GAP").to_numpy()
        for box in "ABC":
            sb = (chunk.box == box).to_numpy()
            for k in ["all", "csi", "nai"] + [k for k in mm if k[0] in "cn" and "_" in k]:
                for tag, s in (("r_", sb), ("f_", sb & isf)):
                    h = np.histogram(tt[s & mm[k]], EDGES)[0]
                    rec[box][tag + k] = rec[box].get(tag + k, 0) + h
    for df in frames:
        for k, v in rec[df.box.iloc[0]].items():
            df[k] = v
    ts = pd.concat(frames)
    ts = ts.merge(eng_series(), on=["box", "t"], how="left")
    ts.to_csv(OUT / "ts_1s.csv", index=False)
    print(f"wrote {OUT/'ts_1s.csv'} ({len(ts)} rows)")


if __name__ == "__main__":
    main()
