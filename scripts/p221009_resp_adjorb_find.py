"""Adjacent-orbit background for HE/CsI (GRB 221009A response study), step 1:
find reference time windows with the same ground track, in low-gain mode,
and record their pointing geometry.

Burst-day track: T0-420 ... T0+2620 s (T0 = 2022-10-09T13:17:00 UTC).  For
every other day 2022-09-20 ... 10-19, times where the sub-satellite point is
within 6 deg of the T0 point and moving in the same latitude direction are
candidates; the offset is refined (+-40 s, 1 s steps) to minimise the mean
great-circle distance of the tracks over T0-420 ... +1000 s.  For each
candidate we record: track distance (mean/max over -420..1000 and
-420..2620 s), the low-gain fraction of seconds (HE-HV mean in -880..-760 V)
and any HV-off seconds, the mean pointing (Att RA/Dec) and the angle between
the pointing axis and the Earth centre (the main geometric driver of the
albedo/CXB part of the CsI background), compared with the burst day.

    python p221009_resp_adjorb_find.py [OUTDIR]"""
import glob
import os
import sys
from datetime import date, timedelta

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.time import Time

D1K = "/hxmt/work/HXMT-DATA/1K"
T0 = 339945423.0
W = (-420, 2620)
WFIT = (-420, 1000)
MET0 = Time("2012-01-01T00:00:00", scale="utc")


def files(day, kind):
    ymd = day.strftime("%Y%m%d")
    out = []
    for hh in range(24):
        fs = sorted(glob.glob(f"{D1K}/Y{ymd[:6]}/{ymd}-*/HXMT_{ymd}T{hh:02d}_{kind}_FFFFFF_V*_1K.FITS"))
        if fs:
            out.append(max(fs, key=lambda f: int(f.split("_V")[-1].split("_")[0])))
    return out


def load(day, kind, ext, cols):
    acc = {c: [] for c in cols}
    for f in files(day, kind):
        try:
            with fits.open(f) as h:
                d = h[ext].data
                for c in cols:
                    acc[c].append(np.asarray(d[c], float))
        except Exception as exc:  # noqa: BLE001
            print("skip", f, exc, flush=True)
    if not acc[cols[0]]:
        return None
    a = {c: np.concatenate(v) for c, v in acc.items()}
    o = np.argsort(a[cols[0]])
    return {c: v[o] for c, v in a.items()}


def gc(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dl = np.deg2rad(lon2 - lon1)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return np.rad2deg(np.arccos(np.clip(c, -1, 1)))


def earth_angle(t, orb, att):
    """Angle (deg) between pointing axis and Earth centre at times t (MET)."""
    x = np.interp(t, orb["Time"], orb["X"]); y = np.interp(t, orb["Time"], orb["Y"])
    z = np.interp(t, orb["Time"], orb["Z"])
    loc = EarthLocation.from_geocentric(x * u.m, y * u.m, z * u.m)
    g = loc.get_gcrs(MET0 + t * u.s).cartesian.xyz.value.T     # satellite position, GCRS (m)
    ra = np.deg2rad(np.interp(t, att["Time"], att["Ra"])); de = np.deg2rad(np.interp(t, att["Time"], att["Dec"]))
    pv = np.column_stack([np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)])
    e = -g / np.linalg.norm(g, axis=1)[:, None]
    return np.rad2deg(np.arccos(np.clip(np.sum(pv * e, 1), -1, 1)))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/scratchfs2/gecam/guohx/p221009/resp/out/adjorb"
    os.makedirs(out, exist_ok=True)
    bday = date(2022, 10, 9)
    ocols = ["Time", "X", "Y", "Z", "Lon", "Lat"]
    ob = load(bday, "Orbit", "Orbit", ocols)
    ab = load(bday, "Att", "ATT_Pointing", ["Time", "Ra", "Dec"])
    tb = T0 + np.arange(W[0], W[1], 10.0)
    latb = np.interp(tb, ob["Time"], ob["Lat"])
    lonb = np.interp(tb, ob["Time"], np.rad2deg(np.unwrap(np.deg2rad(ob["Lon"]))))
    fitm = (tb - T0 >= WFIT[0]) & (tb - T0 < WFIT[1])
    eab = earth_angle(tb, ob, ab)
    lat0, lon0 = np.interp(T0, ob["Time"], ob["Lat"]), np.interp(T0, ob["Time"], ob["Lon"])
    dlat0 = np.sign(np.interp(T0 + 10, ob["Time"], ob["Lat"]) - lat0)
    rows = []
    for k in range(-19, 11):
        day = bday + timedelta(days=k)
        if k == 0:
            continue
        o = load(day, "Orbit", "Orbit", ocols)
        if o is None:
            continue
        lonu = np.rad2deg(np.unwrap(np.deg2rad(o["Lon"])))
        d0 = gc(lat0, lon0, o["Lat"], o["Lon"])
        cand = np.flatnonzero(d0 < 6.0)
        # one candidate per pass: local minima of d0
        picks = []
        for i in cand:
            if i == 0 or i == d0.size - 1 or not (d0[i] <= d0[i - 1] and d0[i] <= d0[i + 1]):
                continue
            if np.sign(o["Lat"][min(i + 10, d0.size - 1)] - o["Lat"][i]) != dlat0:
                continue
            picks.append(o["Time"][i])
        hv = None
        for tc in picks:
            best = None
            for dd in range(-40, 41):
                dt = tc - T0 + dd
                tt = tb + dt
                if tt[0] < o["Time"][0] or tt[-1] > o["Time"][-1]:
                    continue
                la = np.interp(tt, o["Time"], o["Lat"]); lo = np.interp(tt, o["Time"], lonu)
                dist = gc(latb, lonb, la, lo)
                m = dist[fitm].mean()
                if best is None or m < best[0]:
                    best = (m, dt, dist)
            if best is None:
                continue
            m, dt, dist = best
            if hv is None:
                hv = load(day, "HE-HV", "HE_HV_PHODet", ["Time"] + [f"HV_PHODet_{i}" for i in range(18)])
                at = load(day, "Att", "ATT_Pointing", ["Time", "Ra", "Dec"])
            hm = np.mean([hv[f"HV_PHODet_{i}"] for i in range(18)], axis=0)
            sel = (hv["Time"] >= T0 + dt + W[0]) & (hv["Time"] < T0 + dt + W[1])
            lg = np.mean((hm[sel] > -880) & (hm[sel] < -760)) if sel.any() else 0.0
            off = int(np.sum(hm[sel] > -700))
            ng = np.mean(hm[sel] < -930) if sel.any() else 0.0
            ea = earth_angle(tb + dt, o, at) if at is not None else np.full(tb.size, np.nan)
            ra = np.interp(T0 + dt, at["Time"], at["Ra"]); de = np.interp(T0 + dt, at["Time"], at["Dec"])
            rows.append(dict(day=str(day), dt=dt, norb=dt / 5730.0, track_mean=m, track_max=dist[fitm].max(),
                             track_mean_all=dist.mean(), lg_frac=lg, ng_frac=ng, hv_off_s=off,
                             ra=ra, dec=de, dEarth_mean=np.nanmean(ea - eab), dEarth_rms=np.sqrt(np.nanmean((ea - eab) ** 2))))
            r = rows[-1]
            print(f"{r['day']} dt={dt:.0f} ({r['norb']:+.2f} orb) track {m:.2f}/{r['track_max']:.2f} deg, "
                  f"all {r['track_mean_all']:.2f}; LG {lg:.2f} NG {ng:.2f} off {off}; pointing {ra:.1f},{de:.1f}; "
                  f"dEarth {r['dEarth_mean']:+.1f} (rms {r['dEarth_rms']:.1f})", flush=True)
    rb = np.interp(T0, ab["Time"], ab["Ra"]); db = np.interp(T0, ab["Time"], ab["Dec"])
    print(f"burst day pointing {rb:.1f},{db:.1f}; Earth angle {eab.min():.1f}-{eab.max():.1f}")
    keys = list(rows[0])
    with open(f"{out}/candidates.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    np.savetxt(f"{out}/burst_earth_angle.txt", np.column_stack([tb - T0, eab]), fmt="%.1f %.2f")


if __name__ == "__main__":
    main()
