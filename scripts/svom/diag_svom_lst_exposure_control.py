"""对照：把本底按被检验样本的经度分区（和陆海）重新加权，再验平坦性与 f_TGF。

LST = UT + 经度/15，**LST 与经度直接耦合**。留出的 fa > 1 那批在「它自己的
经度分布下」是平的，不等于在「TGF 的经度分布下」也是平的。这里按
经度三分区 × 陆海 共 6 格重新加权再验。
"""
import csv
import datetime as dt
import sys

import numpy as np
from global_land_mask import globe
from scipy.optimize import minimize_scalar

NB = 8
SECTORS = [("美洲", -150.0, -30.0), ("非洲欧洲", -30.0, 60.0), ("亚洲海洋大陆", 60.0, 210.0)]


def eot_hours(doy):
    b = 2 * np.pi * (doy - 81) / 364.0
    return (9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)) / 60.0


def sector(lon):
    x = (lon + 360.0) % 360.0
    for k, (_, lo, hi) in enumerate(SECTORS):
        a, b = (lo + 360.0) % 360.0, (hi + 360.0) % 360.0
        if (a < b and a <= x < b) or (a > b and (x >= a or x < b)):
            return k
    return 2


def prep(path):
    rows = [r for r in csv.DictReader(open(path)) if abs(float(r["lon"])) <= 180
            and r["is_train"] == "0"]
    for r in rows:
        r["fa"] = float(r["fa"])
        lon = float(r["lon"])
        t = dt.datetime.strptime(r["start"][:19], "%Y-%m-%dT%H:%M:%S")
        r["lst"] = ((t.hour + t.minute / 60.0 + t.second / 3600.0 + lon / 15.0
                     + eot_hours(t.timetuple().tm_yday)) % 24.0)
        r["cell"] = sector(lon) * 2 + int(bool(globe.is_land(float(r["lat"]), lon)))
    return rows


def hist(rows):
    h, _ = np.histogram([r["lst"] for r in rows], bins=NB, range=(0, 24))
    return h.astype(float)


def matched_bkg(bkg, sub):
    """按 sub 的 6 格构成把本底重新加权。"""
    w = np.zeros(6)
    for r in sub:
        w[r["cell"]] += 1
    out = np.zeros(NB)
    for c in range(6):
        b = [r for r in bkg if r["cell"] == c]
        if not b or w[c] == 0:
            continue
        h = hist(b)
        out += h / h.sum() * w[c]
    return out


def fit(obs, tmpl, bkg):
    n = obs.sum()
    t, b = tmpl / tmpl.sum(), bkg / bkg.sum()

    def nll(f):
        m = np.maximum(n * (f * t + (1 - f) * b), 1e-9)
        return float((m - obs * np.log(m)).sum())

    r = minimize_scalar(nll, bounds=(-0.5, 2.0), method="bounded", options=dict(xatol=1e-4))
    f0, c = float(r.x), float(r.fun)
    lo, hi = f0, f0
    while lo > -0.5 and nll(lo) - c <= 0.5:
        lo -= 0.005
    while hi < 2.0 and nll(hi) - c <= 0.5:
        hi += 0.005
    return f0, lo, hi, nll(0.0) - c


def main(pool, tmpl_npz):
    rows = prep(pool)
    z = np.load(tmpl_npz)
    t_land, t_ocn = z["land"].astype(float), z["ocean"].astype(float)
    bkg = [r for r in rows if r["fa"] > 1]
    print("本底 fa>1 共 %d 个。6 格（经度区 × 陆海）构成：" % len(bkg)
          + " ".join("%s-%s %d" % (SECTORS[c // 2][0], "陆" if c % 2 else "海",
                                   sum(1 for r in bkg if r["cell"] == c)) for c in range(6)))
    tests = [("显著 fa ≤ 1e-5", lambda r: r["fa"] <= 1e-5),
             ("目录里非证实的那部分", lambda r: r["fa"] <= 1e-5
              and not (r["in_cov"] == "1" and r["assoc"] == "1")),
             ("显著、覆盖外", lambda r: r["fa"] <= 1e-5 and r["in_cov"] == "0"),
             ("fa 1e-3 .. 0.1（对照）", lambda r: 1e-3 < r["fa"] <= 0.1),
             ("fa 0.1 .. 1（对照）", lambda r: 0.1 < r["fa"] <= 1.0)]
    print("\n%-24s %6s %10s %10s %8s %16s %9s"
          % ("被检验的样本", "N", "本底 χ²(原)", "本底 χ²(配)", "f_TGF", "68% 区间", "ΔlnL"))
    for tag, sel in tests:
        sub = [r for r in rows if sel(r)]
        if len(sub) < 20:
            continue
        b_raw = hist(bkg)
        b_mat = matched_bkg(bkg, sub)
        e1, e2 = b_raw.sum() / NB, b_mat.sum() / NB
        chi1 = float(((b_raw - e1) ** 2 / e1).sum())
        chi2 = float(((b_mat - e2) ** 2 / e2).sum())
        # 陆海分开、本底也按 6 格配平
        parts = []
        for isl, tm in ((1, t_land), (0, t_ocn)):
            k = [r for r in sub if r["cell"] % 2 == isl]
            bb = matched_bkg([r for r in bkg if r["cell"] % 2 == isl], k)
            if len(k) >= 10 and bb.sum() > 0:
                parts.append((hist(k), tm, bb))

        def nll(f):
            s = 0.0
            for obs, tt, bb in parts:
                n = obs.sum()
                m = np.maximum(n * (f * tt / tt.sum() + (1 - f) * bb / bb.sum()), 1e-9)
                s += float((m - obs * np.log(m)).sum())
            return s

        r0 = minimize_scalar(nll, bounds=(-0.5, 2.0), method="bounded",
                             options=dict(xatol=1e-4))
        f0, c = float(r0.x), float(r0.fun)
        lo, hi = f0, f0
        while lo > -0.5 and nll(lo) - c <= 0.5:
            lo -= 0.005
        while hi < 2.0 and nll(hi) - c <= 0.5:
            hi += 0.005
        print("%-24s %6d %10.1f %10.1f %8.2f   [%.2f, %.2f] %9.1f"
              % (tag, len(sub), chi1, chi2, f0, lo, hi, nll(0.0) - c))
    print("\n本底 χ² 都是 %d 个自由度。配平后仍平 → LST 结构不是经度–曝光耦合造的。" % (NB - 1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
