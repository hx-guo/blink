"""SVOM att 四元数约定的判定：拿表里自带的角速度 wx/wy/wz 当真值。

只用机动（slew）时段的样本——惯性指向时四元数每秒只变 1e-6 rad，和 float32
量化底同量级，没有信号。机动时转速约 1 deg/s，信噪足够。

八种假设 = {标量在头 Q0 / 标量在尾 Q3} × {R 视作 body→inertial / inertial→body}
           × {ω 取 body 系 / inertial 系}
另外检验 w 是在卫星系还是 GRM 载荷系（差一个 M = [[0,0,1],[0,-1,0],[1,0,0]]）。
"""
import glob
import sys

import numpy as np
from astropy.io import fits
from scipy.spatial.transform import Rotation

M = Rotation.from_matrix([[0, 0, 1], [0, -1, 0], [1, 0, 0]])   # 卫星系 -> 载荷系
SLEW_MIN = 1e-3          # rad/s，机动样本的门槛（≈0.057 deg/s）


def load(paths):
    T, Q, W = [], [], []
    for p in paths:
        with fits.open(p) as h:
            d = h["Quaternion"].data
            T.append(np.asarray(d["TIME"], float))
            Q.append(np.column_stack([np.asarray(d[c], float) for c in ("Q0", "Q1", "Q2", "Q3")]))
            W.append(np.column_stack([np.asarray(d[c], float) for c in ("wx", "wy", "wz")]))
    T = np.concatenate(T)
    Q = np.concatenate(Q)
    W = np.concatenate(W)
    o = np.argsort(T, kind="stable")
    T, Q, W = T[o], Q[o], W[o]
    keep = np.concatenate([[True], np.diff(T) > 0])
    return T[keep], Q[keep] / np.linalg.norm(Q[keep], axis=1)[:, None], W[keep]


def main(paths):
    T, Q, W = load(paths)
    dt = np.diff(T)
    ok = np.isclose(dt, 1.0, atol=1e-3)
    wn = np.linalg.norm(W[:-1], axis=1)
    slew = ok & (wn > SLEW_MIN)
    print("样本 %d，连续 1 s 采样对 %d，其中机动样本（|w| > %.0e rad/s）%d"
          % (len(T), ok.sum(), SLEW_MIN, slew.sum()))
    print("机动样本 |w|：中位 %.4f deg/s，最大 %.4f deg/s"
          % (np.degrees(np.median(wn[slew])), np.degrees(wn[slew].max())))
    if slew.sum() < 50:
        print("机动样本太少，判不了")
        return

    meas = W[:-1][slew]
    results = []
    for oname, idx in (("Q0=标量", [1, 2, 3, 0]), ("Q3=标量", [0, 1, 2, 3])):
        R = Rotation.from_quat(Q[:, idx])
        for dname, Rd in (("R:body→inertial", R), ("R:inertial→body", R.inv())):
            rel_b = (Rd[:-1].inv() * Rd[1:]).as_rotvec()
            rel_i = (Rd[1:] * Rd[:-1].inv()).as_rotvec()
            for fname, rel in (("ω@body", rel_b), ("ω@inertial", rel_i)):
                pred = rel[slew] / dt[slew, None]
                for mname, p in (("w 在卫星系", pred), ("w 在载荷系", M.apply(pred))):
                    resid = np.linalg.norm(p - meas, axis=1)
                    rel_err = np.median(resid / np.linalg.norm(meas, axis=1))
                    cc = np.corrcoef(p.ravel(), meas.ravel())[0, 1]
                    results.append((rel_err, cc, "%s | %s | %s | %s"
                                    % (oname, dname, fname, mname)))
    results.sort()
    print()
    print("%-58s %12s %10s" % ("假设", "相对残差中位", "相关"))
    for rel_err, cc, tag in results:
        print("%-58s %12.5f %+10.6f" % (tag, rel_err, cc))
    print()
    best = results[0]
    print("判定：%s" % best[2])
    print("      相对残差中位 %.5f，逐分量相关 %+.6f" % (best[0], best[1]))
    print("      次优假设的相对残差 %.5f（差 %.0f 倍）" % (results[1][0], results[1][0] / best[0]))


if __name__ == "__main__":
    files = []
    for pat in sys.argv[1:]:
        files.extend(sorted(glob.glob(pat)))
    # 同一小时多版本取版本号最大者
    seen = {}
    for f in files:
        seen[f.rsplit("_v", 1)[0]] = f
    main([seen[k] for k in sorted(seen)])
