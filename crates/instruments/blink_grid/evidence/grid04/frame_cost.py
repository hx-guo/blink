"""一帧占多久：τ（每帧）还是 m × τ（每事例）？这条定共帧星的计数上限。

## 为什么是关键

若读出时间是**每帧** τ，一帧最多 4 个事例 ⇒ 输出上限 4/τ = 140 kc/s，一个 TGF 在
79 µs 里能记下 10 个量级的计数。若读出时间是**每事例** τ，输出上限就是 1/τ =
34.95 kc/s，同一个 TGF 在 79 µs 里最多记下 79/28.61 ≈ 2.8 个。**差 4 倍，而这
4 倍直接乘在折扣因子上。**

实测线索：10 ms 计数的硬顶是 **352**（1/τ × 10 ms = 349.5），而那些格子的
E[m] ≈ 1.5、帧周期 43 µs ≈ 1.5 τ——看着就是"每事例 τ"。

## 判据（无模型，直接看数据）

对每一帧 i，量它到下一帧的间隔 Δᵢ，**按本帧事例数 mᵢ 分组**：

- 每帧 τ  ⇒ min(Δ | m) ≡ τ，与 m 无关；
- 每事例 τ ⇒ min(Δ | m) = m·τ，随 m 台阶式上升。

台阶在 tick 上是 120 / 240 / 360 / 480，一眼可辨，不需要拟合。

用法: python3 frame_cost.py <SAT> <YYYY/MM/DD> [最多几次过境]
"""

import glob
import sys

import numpy as np
from astropy.io import fits

BASE = "/gecamfs/Exchange/GSDC/missions/GRID"
ETH = 30.0
TICK = 2.0**22


def load_pass(path):
    with fits.open(path, memmap=False) as hd:
        g = hd["GTI"].data
        gs, ge = float(np.asarray(g["START"])[0]), float(np.asarray(g["STOP"])[0])
        emin = np.asarray(hd["EBOUNDS"].data["E_MIN"], dtype=np.float64)
        nch = emin.size
        T = []
        for i in range(4):
            d = hd["EVENTS%d" % i].data
            t = np.asarray(d["TIME"], dtype=np.float64)
            pi = np.asarray(d["PI"], dtype=np.int32)
            ty = np.asarray(d["EVT_TYPE"], dtype=np.int32)
            ok = (pi >= 1) & (pi < nch)
            e = np.zeros(t.size)
            e[ok] = emin[pi[ok] - 1]
            T.append(t[(ty == 1) & ok & (e >= ETH)])
    return gs, ge, np.sort(np.concatenate(T))


def main():
    sat, day = sys.argv[1], sys.argv[2]
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    vers = sorted(glob.glob("%s/%s/fits7/%s/evt_v*" % (BASE, sat, day)))
    files = sorted(glob.glob(vers[-1] + "/*.fits"))[:lim]
    agg = {}
    for path in files:
        gs, ge, t = load_pass(path)
        if t.size < 10000:
            continue
        tk = np.rint(t * TICK).astype(np.int64)
        ed = np.flatnonzero(np.diff(tk) != 0)
        st = np.concatenate(([0], ed + 1))
        sz = np.concatenate((ed + 1, [tk.size])) - st
        ft = tk[st]
        d = np.diff(ft)
        m = sz[:-1]
        print("\n=== %s  %.0f c/s  帧数 %d  E[m] %.4f"
              % (path.split("/")[-1], t.size / (ge - gs), st.size, sz.mean()))
        for k in (1, 2, 3, 4):
            s = d[m == k]
            if s.size < 20:
                print("  m=%d  n=%d (太少)" % (k, s.size))
                continue
            print("  m=%d  n=%-8d min %4d  p1 %4d  p5 %4d  中位 %6d tick   (m·τ = %d)"
                  % (k, s.size, s.min(), int(np.percentile(s, 1)),
                     int(np.percentile(s, 5)), int(np.median(s)), 120 * k))
            agg.setdefault(k, []).append(s)
            h = np.bincount(s[s <= 600], minlength=601)
            nz = [(i, int(h[i])) for i in range(1, 601) if h[i] > 0][:8]
            print("       最小的 8 个非零 tick 档: %s" % nz)
    print("\n=== 合并 ===")
    for k in sorted(agg):
        s = np.concatenate(agg[k])
        print("  m=%d  n=%-9d min %4d  p0.1 %4d  p1 %4d   m·τ = %d"
              % (k, s.size, s.min(), int(np.percentile(s, 0.1)),
                 int(np.percentile(s, 1)), 120 * k))


if __name__ == "__main__":
    main()
