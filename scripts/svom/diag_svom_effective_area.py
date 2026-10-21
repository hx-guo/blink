"""核对一条说法：大 θ 处低能有效面积被压得更多吗？

第 27 条用它论证「探测选择效应的符号与观测相反」。要核实。
有效面积按入射能 E 给：A(E) = Σ_ch R[d, E, ch]（只数 ch 25–229，即被搜索的道）。
"""
import glob

import numpy as np

files = sorted(glob.glob("respw_*.npz"))
z0 = np.load(files[0])
R = np.zeros(z0["resp"].shape, np.float32)
meta = np.zeros_like(z0["meta"])
done = np.zeros(R.shape[0], bool)
for f in files:
    z = np.load(f)
    d = z["done"]
    R[d] = z["resp"][d]
    meta[d] = z["meta"][d]
    done |= d
el, eh = z0["elo"], z0["ehi"]
ec = np.sqrt(el * eh)
th = meta[done, 8]
A = R[done][:, :, :, 25:230].astype(np.float64).sum(3).sum(1)   # (n_tgf, n_E)

lo = A[th < 90].mean(0)
hi = A[th >= 90].mean(0)
print("入射能  θ<90° 面积  θ≥90° 面积   比值(背/正)")
for e0, e1 in ((42, 80), (80, 150), (150, 300), (300, 600), (600, 1200),
               (1200, 2500), (2500, 5000), (5000, 8000)):
    k = (ec >= e0) & (ec < e1)
    print("%5d-%-5d keV  %8.1f  %8.1f   %.3f"
          % (e0, e1, lo[k].mean(), hi[k].mean(), hi[k].mean() / lo[k].mean()))
print()
print("若背/正的比值随能量上升，说明大 θ 处低能被压得更多 → 大 θ 偏向探到硬谱的暴发，")
print("即选择效应让大 θ 一端看起来更硬；实测大 θ 一端更软，符号相反。")
