#!/usr/bin/env python3
"""定四元数约定：用 ATT_Pointing 的 Ra/Dec 当真值，试四种可能，看哪一种对得上。

Q1,Q2,Q3 存在文件里，q0 = sqrt(1 - q1^2 - q2^2 - q3^2)（R(q) = R(-q)，符号无所谓）。
候选约定：
  scalar-last  q = (q1,q2,q3 | q0)     vs  scalar-first 的 (q0 | q1,q2,q3) 解读
  R 把 ECI->body   vs   R^T 把 body->ECI
本体轴取 +X/+Y/+Z 三种，一共 2x3 = 6 组合，各算 body 轴在 ECI 的方向，与 Ra/Dec 比。
"""
import sys
import numpy as np
from astropy.io import fits

path = sys.argv[1]

with fits.open(path) as h:
    pnt = h["ATT_Pointing"].data
    qua = h["ATT_Quater"].data
    t_p = np.array(pnt["Time"], float)
    ra = np.array(pnt["Ra"], float)
    dec = np.array(pnt["Dec"], float)
    t_q = np.array(qua["Time"], float)
    q1 = np.array(qua["Q1"], float)
    q2 = np.array(qua["Q2"], float)
    q3 = np.array(qua["Q3"], float)

print("ATT_Pointing: %d 行, Ra %.4f..%.4f, Dec %.4f..%.4f" % (len(ra), ra.min(), ra.max(), dec.min(), dec.max()))
print("  同一小时内 Ra 变化 %.4f deg, Dec 变化 %.4f deg" % (ra.max() - ra.min(), dec.max() - dec.min()))
print("Q 范数^2 (q1^2+q2^2+q3^2): min %.6f max %.6f" % ((q1**2 + q2**2 + q3**2).min(), (q1**2 + q2**2 + q3**2).max()))
assert np.allclose(t_p, t_q), "两表时标不同"

s = np.clip(1.0 - (q1**2 + q2**2 + q3**2), 0.0, None)
q0 = np.sqrt(s)

# 真值方向
cd = np.cos(np.radians(dec))
truth = np.stack([cd * np.cos(np.radians(ra)), cd * np.sin(np.radians(ra)), np.sin(np.radians(dec))], 1)


def rotmat(a, b, c, d):
    """标量在前的 (a; b,c,d) 四元数的旋转矩阵。"""
    n = len(a)
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = a*a + b*b - c*c - d*d
    R[:, 0, 1] = 2*(b*c - a*d)
    R[:, 0, 2] = 2*(b*d + a*c)
    R[:, 1, 0] = 2*(b*c + a*d)
    R[:, 1, 1] = a*a - b*b + c*c - d*d
    R[:, 1, 2] = 2*(c*d - a*b)
    R[:, 2, 0] = 2*(b*d - a*c)
    R[:, 2, 1] = 2*(c*d + a*b)
    R[:, 2, 2] = a*a - b*b - c*c + d*d
    return R


cands = {
    "scalar-last (q1,q2,q3|q0)": rotmat(q0, q1, q2, q3),
    "scalar-first (q0=Q1 ...)": rotmat(q1, q2, q3, q0),
}
best = None
for cname, R in cands.items():
    for tname, M in (("R (列=body轴在ECI)", R), ("R^T", np.transpose(R, (0, 2, 1)))):
        for ax, an in ((0, "+X"), (1, "+Y"), (2, "+Z")):
            v = M[:, :, ax]
            ang = np.degrees(np.arccos(np.clip((v * truth).sum(1), -1, 1)))
            m = np.median(ang)
            print("  %-26s %-18s %s: 与 Ra/Dec 夹角 中位 %7.3f deg  p90 %7.3f" % (cname, tname, an, m, np.percentile(ang, 90)))
            if best is None or m < best[0]:
                best = (m, cname, tname, an)
print("\n最佳: %s / %s / %s  中位夹角 %.4f deg" % (best[1], best[2], best[3], best[0]))
