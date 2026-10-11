#!/usr/bin/env python3
"""GRB 221009A v3: per-second HE background inputs over three hours
(2022-10-09 T12, T13, T14 UTC = T0-4620 .. T0+6180 s).

For every second: 1K event count, ACD-tagged count, CsI channel-band counts,
3-box mean live fraction and summed Cnt_LargeEvt from the 1B HE_Eng frames,
and from the orbit files the dipole L, sub-satellite lat/lon and the GRB
elevation above the geometric Earth limb. These extend the background
calibration to the previous orbit (An+2023 trained on T0-4220..0 and
2000..5780 s) and to the whole occultation.

Output: data/p221009/v3/hours_1s.npz
"""
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).parent))
from engineering_prediction import BOX_CODE, BOX_OFFSET, MET_CORRECTION  # noqa: E402

OUT = Path("data/p221009/v3")
OUT.mkdir(parents=True, exist_ok=True)
T0 = 339945423.0
K1 = "data/1K/Y202210/20221009-1943"
HOURS = ("12", "13", "14")
BANDS = {"csi_0p4_2p95": (43, 255), "csi_0p6_3": (61, 255)}   # see p221009_v3_bkg.BANDS
CSI_PW = 75
XH = np.arange(-4620.0, 6180.0, 1.0)
EDGES = np.append(XH, XH[-1] + 1)


def evt_hour(h):
    fn = f"{K1}/HXMT_20221009T{h}_HE-Evt_FFFFFF_V1_1K.FITS"
    with fits.open(fn, memmap=True) as f:
        off, nrow = f[1]._data_offset, f[1].header["NAXIS2"]
    dt = np.dtype([("t", ">f8"), ("det", "u1"), ("ch", "u1"), ("pw", "u1"),
                   ("acd", "u1", 3), ("et", "u1"), ("fl", "u1")])
    raw = np.memmap(fn, dtype=dt, mode="r", offset=off, shape=(nrow,))
    out = {}
    step = 5_000_000
    for i0 in range(0, nrow, step):
        r = np.array(raw[i0:i0 + step])
        t = r["t"] - T0
        acd = (r["acd"] != 0).any(1)
        csi = r["pw"] > CSI_PW
        parts = {"n": np.ones(len(t), bool), "acd": acd, "csi_all": csi}
        for k, (lo, hi) in BANDS.items():
            parts[k] = csi & (r["ch"] >= lo) & (r["ch"] <= hi)
        for k, m in parts.items():
            out[k] = out.get(k, 0) + np.histogram(t[m], EDGES)[0]
    return out


def eng_hour(h):
    acc_f, acc_le, acc_n = np.zeros(len(XH)), np.zeros(len(XH)), np.zeros(len(XH))
    nbox = np.zeros(len(XH))
    for box, code in BOX_CODE.items():
        fn = sorted(Path(f"data/1B/2022/20221009/{code}").glob(f"HXMT_1B_{code}_20221009T{h}0000*.fits"))
        for path in fn:
            with fits.open(path, memmap=True) as fe:
                if "HE_Eng" not in fe:
                    continue
                d = fe["HE_Eng"].data
                off = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
                k = np.round(d["Time"].astype(float) + off + MET_CORRECTION - T0).astype(int)
                L = d["Length_Time_Cycle"].astype(float)
                g = [BOX_OFFSET[box] + j for j in range(6)]
                D = np.mean([d[f"DeadTime_PHODet_{i}"].astype(float) for i in g], 0)
                LE = np.sum([d[f"Cnt_LargeEvt_{i}"].astype(float) for i in g], 0)
                j = k - int(XH[0])
                ok = (j >= 0) & (j < len(XH)) & (L > 0)
                np.add.at(acc_f, j[ok], 1 - D[ok] / L[ok])
                np.add.at(acc_le, j[ok], LE[ok])
                np.add.at(nbox, j[ok], 1)
    return acc_f, acc_le, nbox


def orbit_hour(h):
    from astropy.coordinates import SkyCoord, ITRS
    from astropy.time import Time, TimeDelta
    import astropy.units as u
    o = fits.open(f"{K1}/HXMT_20221009T{h}_Orbit_FFFFFF_V1_1K.FITS")[1].data
    t = o["Time"] - T0
    pos = np.vstack([o["X"], o["Y"], o["Z"]]).astype(float)
    r = np.linalg.norm(pos, axis=0)
    pl, pn = np.radians(80.65), np.radians(-72.68)
    m = np.array([np.cos(pl) * np.cos(pn), np.cos(pl) * np.sin(pn), np.sin(pl)])
    maglat = np.degrees(np.arcsin((m[:, None] * pos).sum(0) / r))
    L = (r / 6371.2e3) / np.cos(np.radians(maglat)) ** 2
    ts = Time("2022-10-09T13:17:00", scale="utc") + TimeDelta(t[::10], format="sec")
    c = SkyCoord(ra=288.265 * u.deg, dec=19.773 * u.deg, distance=1e9 * u.pc)
    v = c.transform_to(ITRS(obstime=ts)).cartesian.xyz.value
    v = v / np.linalg.norm(v, axis=0)
    p10, r10 = pos[:, ::10], r[::10]
    nad = np.degrees(np.arccos(np.clip(-(p10 * v).sum(0) / r10, -1, 1)))
    elev = nad - np.degrees(np.arcsin(6378.137e3 / r10))
    return t, o["Lon"], o["Lat"], L, t[::10], elev


def main():
    res = {k: np.zeros(len(XH)) for k in ("n", "acd", "csi_all", *BANDS)}
    fsum, lesum, nb = np.zeros(len(XH)), np.zeros(len(XH)), np.zeros(len(XH))
    ot, olon, olat, oL, et, eel = [], [], [], [], [], []
    for h in HOURS:
        e = evt_hour(h)
        for k in res:
            res[k] += e[k]
        a, b, c = eng_hour(h)
        fsum += a; lesum += b; nb += c
        t, lon, lat, L, t10, el = orbit_hour(h)
        ot.append(t); olon.append(lon); olat.append(lat); oL.append(L); et.append(t10); eel.append(el)
        print(f"hour {h}: {int(e['n'].sum())} events, eng frames {int(c.sum())}", flush=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(nb == 3, fsum / nb, np.nan)
        LE = np.where(nb == 3, lesum, np.nan)
    ot, et = np.concatenate(ot), np.concatenate(et)
    g = lambda y: np.interp(XH + 0.5, ot, np.concatenate(y))
    np.savez(OUT / "hours_1s.npz", t=XH, f=f, LE=LE, nbox=nb,
             lon=g(olon), lat=g(olat), L=g(oL),
             elev=np.interp(XH + 0.5, et, np.concatenate(eel)), **res)


if __name__ == "__main__":
    main()
