"""真 TGF 窗内的 PI 谱 vs 同处本底的 PI 谱，逐探头找 ADC 满量程台阶。

起因：140 个真 TGF 的窗内事例有 **15.5%（中位）落在 ch >= 368**，而 ch368 按
EBOUNDS 折约 3.4 MeV——TGF 谱到这里早该掉光了。OPEN-QUESTIONS 第 19 条在
GECAM-A 上实测到 ch368–383 有一个超量程堆积包（ADC 满量程经能量刻度映射过来
的道号），并**推断** B/C 因为一直有在轨 EC、满量程道该在 ch447 附近、被现有
上界挡住大半。这段就是去测 B 星到底在哪，以及那 15.5% 是不是本底本来就有的。

只有配上**本底谱**这 15.5% 才有意义：本底若也是 15%，它什么都不说明。

用法: python3 spec_probe.py <shard> <nshard> <out_prefix>
"""
import csv, json, sys
import numpy as np

sys.path.insert(0, "/scratchfs2/gecam/guohx")
from gecam_features import read_events, hour_file, met, span

RUN = "/scratchfs2/gecam/guohx/gecambrun"
SAT = "GECAM-B"
shard, nshard, prefix = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
NCH, NDET = 448, 25

sigs = json.load(open(RUN + "/tgf_signals.json"))
hours = sorted({x["start"][:13] for x in sigs})
mine = {h for i, h in enumerate(hours) if i % nshard == shard}
sigs = [x for x in sigs if x["start"][:13] in mine]
print("这一片 %d 小时, %d 个真 TGF" % (len(mine), len(sigs)))

win_spec = np.zeros(NCH + 8, dtype=np.int64)     # TGF 窗内
bg_spec = np.zeros(NCH + 8, dtype=np.int64)      # 同处本底（±1 s 挖掉 ±10 ms）
bg_seconds = 0.0
win_seconds = 0.0
det_bg = np.zeros((NDET, NCH + 8), dtype=np.int64)   # 逐探头本底谱，用来找台阶
per_tgf = []

key = None
grd = None
for s in sorted(sigs, key=lambda x: x["start"]):
    iso = s["start"]
    k = iso[:13]
    if k != key:
        key = k
        p = hour_file(SAT, iso, "grd")
        grd = read_events(p, want_pi=True) if p else None
    if grd is None:
        continue
    t, ch, det = grd
    t0 = met(iso, SAT) + s["delay"]
    t1 = t0 + s["bin_size_best"]
    lo, hi = span(t, t0, t1)
    if hi <= lo:
        continue
    w = np.bincount(np.clip(ch[lo:hi], 0, NCH + 7), minlength=NCH + 8)
    win_spec += w
    win_seconds += t1 - t0
    blo, bhi = span(t, t0 - 1.0, t1 + 1.0)
    hlo, hhi = span(t, t0 - 0.01, t1 + 0.01)
    seg_ch = np.concatenate([ch[blo:hlo], ch[hhi:bhi]])
    seg_dt = np.concatenate([det[blo:hlo], det[hhi:bhi]])
    b = np.bincount(np.clip(seg_ch, 0, NCH + 7), minlength=NCH + 8)
    bg_spec += b
    bg_seconds += (2.0 + (t1 - t0)) - (0.02 + (t1 - t0))
    for d in range(1, NDET + 1):
        m = seg_dt == d
        if m.any():
            det_bg[d - 1] += np.bincount(np.clip(seg_ch[m], 0, NCH + 7),
                                         minlength=NCH + 8)
    nw = hi - lo
    per_tgf.append(dict(
        start=iso, n=int(nw),
        ge368=float((ch[lo:hi] >= 368).mean()),
        ge400=float((ch[lo:hi] >= 400).mean()),
        ge440=float((ch[lo:hi] >= 440).mean()),
        bg_ge368=float((seg_ch >= 368).mean()) if len(seg_ch) else 0.0,
        bg_ge400=float((seg_ch >= 400).mean()) if len(seg_ch) else 0.0,
        pi_med=int(np.median(ch[lo:hi]))))

np.savez(prefix + "_%d.npz" % shard, win=win_spec, bg=bg_spec,
         det_bg=det_bg, bg_seconds=bg_seconds, win_seconds=win_seconds,
         n_tgf=len(per_tgf))
json.dump(per_tgf, open(prefix + "_per_%d.json" % shard, "w"))
print("落盘 %d 个" % len(per_tgf))
