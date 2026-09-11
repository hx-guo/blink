"""GRB 250919A 定向检验的加强版：光变确认 + 死时间修正 + 谱指数扫描。

在前一版里三路份额已经把八种假设分开了（χ² 190 对 3391）。这里把三条粗糙之处
补上，看结论是否稳：
  1. 暴发窗按光变自己定，而不是拍脑袋的 ±几秒；
  2. 逐探头死时间修正 n/(1−f)，f = Σ DEAD_TIME / 窗长（亮暴发里 G02 最忙）；
  3. 幂律谱指数从 −2.5 扫到 −0.5，取各假设自己最好的 χ²（不让假设吃亏）。
"""
import glob
import sys
from datetime import datetime, timezone

import numpy as np
from astropy.io import fits
from scipy.spatial.transform import Rotation

D = "/gecamfs/SVOM/Archived-DATA/GRM-DATA/L1B/daily"
CAL = "/gecamfs/SVOM/soft/CALDB/data/svom/grm/bcf/mc_rsp"
REF = datetime(2017, 1, 1, tzinfo=timezone.utc)
M = Rotation.from_matrix([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
DET_PHI = {1: 0.0, 2: 240.0, 3: 120.0}


def latest(pat):
    g = sorted(glob.glob(pat))
    if not g:
        return None
    seen = {}
    for f in g:
        seen[f.rsplit("_v", 1)[0]] = f
    return sorted(seen.values())[-1]


def sph(theta, phi):
    t, p = np.radians(theta), np.radians(phi)
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def radec_vec(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


GRID = []
for d in sorted(glob.glob(CAL + "/gamma_T*_P*")):
    nm = d.rsplit("/", 1)[1]
    GRID.append((int(nm.split("_T")[1].split("_")[0]) / 1000.0,
                 int(nm.split("_P")[1]) / 1000.0, nm))
GV = np.stack([sph(t, p) for t, p, _ in GRID])

_cache = {}


def resp_curve(dirname, det):
    """返回 (ENERG 中心, ENERG 宽, 逐入射能对各沉积道的响应矩阵 cm^2, 道边界)。"""
    key = (dirname, det)
    if key in _cache:
        return _cache[key]
    f = latest("%s/%s/g%02d_x_%s_v*.rsp" % (CAL, dirname, det, dirname.replace("gamma_", "")))
    with fits.open(f) as h:
        d = h["SPECRESP MATRIX"].data
        elo = np.asarray(d["ENERG_LO"], float)
        ehi = np.asarray(d["ENERG_HI"], float)
        mat = np.array([np.asarray(r, float) for r in d["MATRIX"]])
        eb = h["EBOUNDS"].data
        cmin = np.asarray(eb["E_MIN"], float)
        cmax = np.asarray(eb["E_MAX"], float)
    _cache[key] = (elo, ehi, mat, cmin, cmax)
    return _cache[key]


def band_area(dirname, det, e_lo, e_hi, index):
    elo, ehi, mat, cmin, cmax = resp_curve(dirname, det)
    chan = (cmax > e_lo) & (cmin < e_hi)
    resp = mat[:, chan].sum(1)
    ec = np.sqrt(elo * ehi)
    w = ec ** index * (ehi - elo)
    return float((resp * w).sum() / w.sum())


def main(day, hh, t0_iso, ra, dec):
    y, m, dd = day.split("-")
    t0 = datetime.strptime(t0_iso, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc)
    met0 = (t0 - REF).total_seconds()
    fe = latest("%s/%s/%s/%s/grm_evt/svom_grm_evt_%s%s%s_%s_v*.fits" % (D, y, m, dd, y[2:], m, dd, hh))
    fa = latest("%s/%s/%s/%s/att/svom_att_%s%s%s_%s_v*.fits" % (D, y, m, dd, y[2:], m, dd, hh))
    with fits.open(fa) as h:
        d = h["Quaternion"].data
        at = np.asarray(d["TIME"], float)
        aq = np.column_stack([np.asarray(d[c], float) for c in ("Q0", "Q1", "Q2", "Q3")])
    with fits.open(fe) as h:
        per = {}
        for i, det in zip((3, 4, 5), (1, 2, 3)):
            x = h[i].data
            t = np.asarray(x["TIME"], float)
            pi = np.asarray(x["PI"])
            et = np.asarray(x["EVT_TYPE"])
            ac = np.asarray(x["ANTI_COIN"])
            dt = np.asarray(x["DEAD_TIME"], float)
            k = (et == 0) & (ac == 0) & (pi >= 25) & (pi < 256)
            per[det] = (t[k], pi[k], dt[k])
        eb = h["EBOUNDS"].data
        ch_lo = np.asarray(eb["E_MIN"], float)
        ch_hi = np.asarray(eb["E_MAX"], float)

    allt = np.concatenate([per[d][0] for d in (1, 2, 3)])
    edges = np.arange(met0 - 60, met0 + 120, 1.0)
    hist, _ = np.histogram(allt, bins=edges)
    base = np.median(hist)
    print("1 s 光变（相对 T0，只列超出本底 5σ 的秒）：本底 %.0f c/s" % base)
    sig = (hist - base) / np.sqrt(base)
    for i in np.where(sig > 5)[0]:
        print("   T0%+6.0f s   %6d c/s   %+.0fσ" % (edges[i] - met0, hist[i], sig[i]))
    lo_i, hi_i = np.where(sig > 5)[0][[0, -1]]
    b0, b1 = edges[lo_i], edges[hi_i + 1]
    print("暴发窗取 [T0%+.0f, T0%+.0f] s，共 %.0f s" % (b0 - met0, b1 - met0, b1 - b0))

    bl0, bl1 = met0 - 120, met0 - 60
    obs, err, shares = {}, {}, {}
    for det in (1, 2, 3):
        t, pi, dtus = per[det]
        bm = (t >= b0) & (t <= b1)
        nm = (t >= bl0) & (t <= bl1)
        rate = nm.sum() / (bl1 - bl0)
        f = dtus[bm].sum() * 1e-6 / (b1 - b0)
        raw = bm.sum()
        corr = raw / (1 - f)
        bkg = rate * (b1 - b0) / (1 - dtus[nm].sum() * 1e-6 / (bl1 - bl0))
        obs[det] = corr - bkg
        err[det] = np.sqrt(raw + bkg) / (1 - f)
        print("  G%02d 原始 %6d 死时间占比 %.3f 修正后 %8.1f 本底 %8.1f 净 %8.1f"
              % (det, raw, f, corr, bkg, obs[det]))
    tot = sum(obs.values())
    o = np.array([obs[d] / tot for d in (1, 2, 3)])
    oe = np.array([err[d] / tot for d in (1, 2, 3)])
    print("死时间修正后实测份额 %s ± %s" % (np.round(o, 4), np.round(oe, 4)))

    pi_all = np.concatenate([per[d][1][(per[d][0] >= b0) & (per[d][0] <= b1)] for d in (1, 2, 3)])
    e_lo, e_hi = ch_lo[25], ch_hi[255]
    print("折叠能段取 ch25–255 = %.1f–%.1f keV；峰值 PI 中位 %d" % (e_lo, e_hi, int(np.median(pi_all))))

    q = np.array([np.interp(0.5 * (b0 + b1), at, aq[:, k]) for k in range(4)])
    q = q / np.linalg.norm(q)
    src = radec_vec(ra, dec)
    axes = {d: sph(30.0, DET_PHI[d]) for d in (1, 2, 3)}
    idxs = np.arange(-2.5, -0.4, 0.25)
    rows = []
    for oname, ii in (("Q0=标量", [1, 2, 3, 0]), ("Q3=标量", [0, 1, 2, 3])):
        R = Rotation.from_quat(q[ii])
        for dname, base_r in (("q:body→J2000", R.inv()), ("q:J2000→body", R)):
            for mname, mm in (("载荷=M·卫星", M), ("载荷=卫星", None)):
                v = base_r.apply(src)
                if mm is not None:
                    v = mm.apply(v)
                v = v / np.linalg.norm(v)
                th = np.degrees(np.arccos(np.clip(v[2], -1, 1)))
                ph = np.degrees(np.arctan2(v[1], v[0])) % 360
                gi = int(np.argmax(GV @ v))
                gname = GRID[gi][2]
                sep = np.degrees(np.arccos(np.clip(GV[gi] @ v, -1, 1)))
                best = None
                for idx in idxs:
                    ea = np.array([band_area(gname, d, e_lo, e_hi, idx) for d in (1, 2, 3)])
                    p = ea / ea.sum()
                    c2 = float(np.sum((o - p) ** 2 / oe ** 2))
                    if best is None or c2 < best[0]:
                        best = (c2, idx, p)
                ang = [np.degrees(np.arccos(np.clip(axes[d] @ v, -1, 1))) for d in (1, 2, 3)]
                rows.append((best[0], "%s | %s | %s" % (oname, dname, mname),
                             th, ph, sep, ang, best[2], best[1]))
    rows.sort()
    print()
    print("%-40s %7s %7s %6s %26s %24s %6s %10s" %
          ("假设", "θ", "φ", "格点差", "到三路 GRD 夹角", "预测份额", "指数", "χ²(2 dof)"))
    for c2, name, th, ph, sep, ang, p, idx in rows:
        print("%-40s %7.2f %7.2f %6.2f %s %s %6.2f %10.1f"
              % (name, th, ph, sep, np.round(ang, 1), np.round(p, 4), idx, c2))
    print()
    print("最佳: %s  χ² = %.1f，次优 %.1f（相差 %.0f 倍）"
          % (rows[0][1], rows[0][0], rows[1][0], rows[1][0] / max(rows[0][0], 1e-9)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]))
