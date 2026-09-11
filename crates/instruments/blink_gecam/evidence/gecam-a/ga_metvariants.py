"""MET 还原的三种写法各把窗口挪多少格——纯算术，不碰事例流。

窗边界只要挪一个 ulp（A 星 29.8 ns），中位只有 5 个 ulp 宽的窗就整体错位。
这里直接数：同一批候选，三种写法算出来的 k0（第几个 ulp）与"只舍一次"的
基准差几格。
"""
import datetime as dt
import glob
import json

import numpy as np

EPOCH = dt.datetime(2019, 1, 1, tzinfo=dt.timezone.utc)


def parts(iso):
    head, _, frac = iso.rstrip("Z").partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return int((stamp - EPOCH).total_seconds()), frac


for p in sorted(glob.glob("/scratchfs2/gecam/guohx/gecam_a/data/GECAM-A/*/*/*_signals.json")):
    sig = json.load(open(p))
    if not sig:
        continue
    ref = []      # 只舍一次：整秒与小数拼成十进制串
    two = []      # 两步相加：total_seconds() + float("0."+frac)
    ins = []      # 先折成整数纳秒再乘 1e-9
    for s in sig:
        whole, frac = parts(s["start"])
        t_ref = float(f"{whole}.{frac}" if frac else str(whole))
        ref.append(t_ref)
        two.append(float(whole) + (float("0." + frac) if frac else 0.0))
        ins.append(round(t_ref * 1e9) * 1e-9)
    ref = np.array(ref)
    q = np.spacing(ref)
    k_ref = np.rint(ref / q)
    d_two = np.rint(np.array(two) / q) - k_ref
    d_ins = np.rint(np.array(ins) / q) - k_ref
    n = len(sig)
    print(f"{p.split('/')[-1]:26} n={n:7d} q={q[0]*1e9:6.3f}ns  "
          f"两步相加差 0 格 {100*(d_two==0).mean():6.2f}%  "
          f"整数纳秒差 0 格 {100*(d_ins==0).mean():6.2f}%  "
          f"（整数纳秒的差值分布 {dict(zip(*[x.tolist() for x in np.unique(d_ins, return_counts=True)]))}）")
