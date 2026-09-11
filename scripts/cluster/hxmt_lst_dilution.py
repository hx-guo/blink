#!/usr/bin/env python3
"""亮暗形状差是相位差还是稀释差：用亮档当模板拟暗档，f 就是暗档的相对纯度。"""
import csv, sys
import numpy as np

NBIN = 8


def lst_hours(iso, lon):
    h = int(iso[11:13]) + int(iso[14:16]) / 60.0 + float(iso[17:19]) / 3600.0
    return (h + lon / 15.0) % 24.0


def hist(x):
    return np.histogram(x, bins=NBIN, range=(0, 24))[0].astype(float)


def fit(obs, T, B):
    N = obs.sum(); T = T / T.sum(); B = B / B.sum()
    fs = np.linspace(-0.5, 1.5, 2001)
    nll = np.array([-(obs * np.log(np.maximum(N * (f * T + (1 - f) * B), 1e-12))
                      - N * (f * T + (1 - f) * B)).sum() for f in fs])
    i = nll.argmin(); ok = fs[nll <= nll[i] + 1.92]
    mu = N * (fs[i] * T + (1 - fs[i]) * B)
    return fs[i], ok.min(), ok.max(), ((obs - mu) ** 2 / np.maximum(mu, 1e-9)).sum()


bgl = []
with open(sys.argv[1]) as f:
    r = csv.reader(f); next(r)
    for start, fpy, lon, lat, assoc, nb, train in r:
        if float(fpy) > 1 and train == '0':
            bgl.append(lst_hours(start, float(lon)))
B = hist(np.array(bgl))

cat = list(csv.DictReader(open(sys.argv[2])))
sig = [c for c in cat if c['associated'] != '1']
lst = np.array([lst_hours(c['start'], float(c['longitude'])) for c in sig])
cnt = np.array([int(c['count']) for c in sig], float)
lo, hi = np.percentile(cnt, [33.3, 66.7])
hd, hb = hist(lst[cnt <= lo]), hist(lst[cnt > hi])
b, l, h, c2 = fit(hd, hb, B)
print("用亮档模板拟暗档: f = %.3f  95%% [%.3f, %.3f]  chi2 = %.1f / 6" % (b, l, h, c2))
print("  -> 若 chi2 好，亮暗的差就是稀释不是相位：暗档相对亮档的纯度 %.0f%%，"
      "暗档 %d 个里约 %.0f 个是本底" % (100 * b, int(hd.sum()), (1 - b) * hd.sum()))
bb, lb, hb2, c2b = fit(hb, hd, B)
print("反过来用暗档模板拟亮档: f = %.3f  95%% [%.3f, %.3f]  chi2 = %.1f / 6" % (bb, lb, hb2, c2b))
