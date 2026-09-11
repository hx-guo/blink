#!/usr/bin/env python3
"""口径检验：亮暗 TGF 的 LST 形状一样吗（只有 HXMT 样本够大做得动）。

模板失配是这个方法最大的系统。真值样本（闪电关联）偏亮、被检验的偏暗，
若亮暗形状不同，模板就不能通用。两套样本各做一遍：
  (a) 闪电关联样本内部按计数分三档
  (b) 显著未关联样本（不经闪电台网）内部按计数分三档
"""
import csv, sys
import numpy as np

NBIN = 8
CENTRES = np.arange(NBIN) * 3.0 + 1.5


def lst_hours(iso, lon):
    h = int(iso[11:13]) + int(iso[14:16]) / 60.0 + float(iso[17:19]) / 3600.0
    return (h + lon / 15.0) % 24.0


def hist(x):
    return np.histogram(x, bins=NBIN, range=(0, 24))[0].astype(float)


def compare(name, lst, cnt):
    lo, hi = np.percentile(cnt, [33.3, 66.7])
    d, b = lst[cnt <= lo], lst[cnt > hi]
    hd, hb = hist(d), hist(b)
    pd_, pb = hd / hd.sum(), hb / hb.sum()
    var = pd_ * (1 - pd_) / hd.sum() + pb * (1 - pb) / hb.sum()
    chi2 = ((pd_ - pb) ** 2 / np.maximum(var, 1e-12)).sum()
    print("  %s  暗档 count<=%.0f N=%d，亮档 count>%.0f N=%d" % (name, lo, len(d), hi, len(b)))
    print("    暗 %s  峰 %.1f h" % (" ".join("%5.1f" % x for x in 100 * pd_), CENTRES[pd_.argmax()]))
    print("    亮 %s  峰 %.1f h" % (" ".join("%5.1f" % x for x in 100 * pb), CENTRES[pb.argmax()]))
    print("    两形状 chi2 = %.1f (dof=7)，p99 = 18.5 -> %s"
          % (chi2, "形状不同" if chi2 > 18.5 else "形状一致，模板可通用"))
    return hd, hb


cat = list(csv.DictReader(open(sys.argv[1])))
tgf = [c for c in cat if c['associated'] == '1']
print("=== (a) 闪电关联样本内部 ===")
compare("闪电关联", np.array([lst_hours(c['start'], float(c['longitude'])) for c in tgf]),
        np.array([int(c['count']) for c in tgf], float))

sig = [c for c in cat if c['associated'] != '1']
print("\n=== (b) 目录里未关联的（不经闪电台网）===")
hd, hb = compare("未关联", np.array([lst_hours(c['start'], float(c['longitude'])) for c in sig]),
                 np.array([int(c['count']) for c in sig], float))

print("\n=== 供给全队：HXMT 的 LST 模板（3 h 一格，格心 LST 1.5, 4.5 … 22.5 h）===")
for nm, rowsrc in (("A 闪电关联 2547（带 WWLLN 夜间偏差，慎用）",
                    np.array([lst_hours(c['start'], float(c['longitude'])) for c in tgf])),
                   ("B 显著未关联 4677（不经台网，推荐）",
                    np.array([lst_hours(c['start'], float(c['longitude'])) for c in sig]))):
    h = hist(rowsrc)
    print("  %s" % nm)
    print("    计数 %s" % ", ".join("%d" % x for x in h))
    print("    归一 %s" % ", ".join("%.4f" % x for x in h / h.sum()))
print("  纬度带 |lat| <= 43 deg（HXMT 倾角 43）；LST = UTC + lon/15，未加时差方程（±0.27 h）")
