"""在若干方位偏移（与 θ 翻转）上重算三路 GRD 的有效面积，供「份额对不上」定位。

83 个闪电证实 TGF 的逐 GRD 计数份额与 CALDB 在真实入射方向上的预测对不上
（ε ≈ 0 而不是 1，见 OPEN-QUESTIONS 第 29 条）。若是方向链里有一个固定的方位
误差，那么在 φ + Δφ 上重算响应、扫 Δφ，似然应当在某个 Δφ 上冒出来；若是响应
本身在探头间的相对归一不准（或者源本来就不是点源），则整条曲线都平。

只要逐探头的能谱加权有效面积，不要整个矩阵：
    A_d(θ, φ) = Σ_E Σ_ch R[d, E, ch] · (E/100)^-1 · ΔE   （ch 限在 25–229）

用法: python3 svom_resp_scan.py <incidence.csv> <out.npz> [WORKERS] [IDX]
"""
import csv
import sys

import numpy as np

sys.path.insert(0, "/gecamfs/SVOM/soft/CALDB/software")
from RSP_Generator import gen_rsp                                   # noqa: E402

import datetime as dt                                               # noqa: E402

REF = dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc)
DPHI = list(range(0, 360, 30))
CH_LO, CH_HI = 25, 230


def met(iso):
    b = iso.rstrip("Z")
    h, _, f = b.partition(".")
    s = dt.datetime.strptime(h, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return (s - REF).total_seconds() + (float("0." + f) if f else 0.0)


def main(inc, out, workers=1, idx=0):
    rows = list(csv.DictReader(open(inc)))
    nv = len(DPHI) + 1                       # 12 个方位偏移 + θ 翻转
    area = np.zeros((len(rows), nv, 3))
    done = np.zeros((len(rows), nv), bool)
    for i, r in enumerate(rows):
        if i % workers != idx:
            continue
        m0 = met(r["start"])
        th, ph = float(r["theta"]), float(r["phi"])
        for v in range(nv):
            if v < len(DPHI):
                t, p = th, (ph + DPHI[v]) % 360
            else:
                t, p = 180.0 - th, ph
            ok = True
            for d in (1, 2, 3):
                g = gen_rsp(detname="g%02d" % d, theta=t, phi=p, MET=m0)
                if g is None:
                    ok = False
                    break
                mtx, el, eh, cl, ch, _, _ = g
                el = np.asarray(el, float)
                eh = np.asarray(eh, float)
                w = np.sqrt(el * eh)
                area[i, v, d - 1] = float(
                    (np.asarray(mtx, float)[:, CH_LO:CH_HI].sum(1)
                     * (w / 100.0) ** (-1.0) * (eh - el)).sum())
            done[i, v] = ok
        print("%3d %s θ=%.1f φ=%.1f 完成 %d/%d" % (i, r["start"][:23], th, ph,
                                                 done[i].sum(), nv), flush=True)
    np.savez_compressed(out, area=area, done=done, dphi=np.array(DPHI),
                        starts=np.array([r["start"] for r in rows]))
    print("→", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 1,
         int(sys.argv[4]) if len(sys.argv) > 4 else 0)
