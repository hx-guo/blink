"""扫 GECAM-B 那 138 小时全部 300.8 万候选的 signals.json，出三样东西：

1. CPD 定标的**对照组**：逐候选按自己的 `n_acd_bg`（±1 s 基线、标称 2 s）算
   当地期望，统计候选窗内 CPD 命中的 obs/exp。真 TGF 是正样本，这批是对照。
   窗口用候选自己的 `[start, stop]`（`cpd_counts` 就是按这个窗数的），不是最佳格。
2. **逐路基线零格**：25 路 `detectors.baseline` 里有没有零格，对应 OPEN-QUESTIONS
   第 4c 条在 B 星这一半。
3. 分位数与分档统计，外加把 `fa <= 1e-5` 的子集原样落盘（真 TGF 全在这一档，
   要拿它跟正样本并排比）。

用法: python3 pool_scan.py <shard> <nshard> <out_prefix>
"""
import csv, glob, json, sys, math, datetime as dt
import numpy as np


def _abs(iso):
    body = iso.rstrip("Z")
    head, _, frac = body.partition(".")
    t = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=dt.timezone.utc).timestamp()
    return t + (float("0." + frac) if frac else 0.0)


def win_seconds(st, sp):
    """候选窗宽（秒）。整秒字段相同时只减小数秒，跨秒时才走完整解析。"""
    hs, _, fs = st.rstrip("Z").partition(".")
    hp, _, fp = sp.rstrip("Z").partition(".")
    if hs == hp:
        return float("0." + fp) - float("0." + fs)
    return _abs(sp) - _abs(st)

RUN = "/scratchfs2/gecam/guohx/gecambrun"
shard, nshard, prefix = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

files = sorted(glob.glob(RUN + "/data/GECAM-B/*/*/*_signals.json"))
mine = [f for i, f in enumerate(files) if i % nshard == shard]
print("这一片 %d / %d 个日文件" % (len(mine), len(files)))

NDET = 25
# 累加器
n_tot = 0
n_no_cpd = 0
obs_hit = 0          # 窗内 n_acd > 0 的候选数
exp_hit = 0.0        # 偶然期望（Σ 1-e^-μ）
sum_nacd = 0         # 窗内 CPD 计数合计
sum_mu = 0.0         # 期望 CPD 计数合计
sum_ncore = 0
zero_any = 0
zero_hist = np.zeros(NDET + 1, dtype=np.int64)
det_zero = np.zeros(NDET, dtype=np.int64)
base_min_all = []
fa_bins = [1e-10, 1e-7, 1e-5, 1e-3, 1e-1, 1.0, 20.0, 1e9]
fa_cnt = np.zeros(len(fa_bins), dtype=np.int64)
# 分 fa 档的 CPD
fa_obs = np.zeros(len(fa_bins))
fa_exp = np.zeros(len(fa_bins))
fa_n = np.zeros(len(fa_bins))
# 分窗长档
bin_edges = [1e-6, 1e-5, 5e-5, 2e-4, 1e-3, 1e9]
b_obs = np.zeros(len(bin_edges))
b_exp = np.zeros(len(bin_edges))
b_n = np.zeros(len(bin_edges))
# 环境率档
env_edges = [200, 400, 600, 1000, 2000, 1e9]
e_n = np.zeros(len(env_edges))
# 采样用
samp_bin, samp_cnt, samp_env, samp_ratio = [], [], [], []
bright = []

for f in mine:
    for s in json.load(open(f)):
        n_tot += 1
        fa = s["false_positive_per_year"]
        cnt = s["count"]
        bb = s["bin_size_best"]
        a = s.get("acd")
        det = s.get("detectors") or {}
        base = det.get("baseline") or []
        if base:
            z = sum(1 for x in base if x == 0)
            zero_hist[min(z, NDET)] += 1
            if z:
                zero_any += 1
                for i, x in enumerate(base):
                    if x == 0 and i < NDET:
                        det_zero[i] += 1
            if n_tot % 97 == 0:
                base_min_all.append(min(base))
        fi = next(i for i, e in enumerate(fa_bins) if fa <= e)
        fa_cnt[fi] += 1
        bi = next(i for i, e in enumerate(bin_edges) if bb <= e)
        if a is None:
            n_no_cpd += 1
            continue
        # 候选窗宽：cpd_counts 数的就是 [start, stop] 这个窗，不是最佳格
        st, sp = s["start"], s["stop"]
        ws = win_seconds(st, sp)
        rate = a["n_acd_bg"] / 2.0
        mu = rate * ws
        p = 1 - math.exp(-mu) if mu < 700 else 1.0
        hit = 1 if a["n_acd"] > 0 else 0
        obs_hit += hit
        exp_hit += p
        sum_nacd += a["n_acd"]
        sum_mu += mu
        sum_ncore += a["n"]
        fa_obs[fi] += hit
        fa_exp[fi] += p
        fa_n[fi] += 1
        b_obs[bi] += hit
        b_exp[bi] += p
        b_n[bi] += 1
        ei = next(i for i, e in enumerate(env_edges) if rate <= e)
        e_n[ei] += 1
        if n_tot % 97 == 0:
            samp_bin.append(bb)
            samp_cnt.append(cnt)
            samp_env.append(rate)
            samp_ratio.append(a["n_acd"] / max(a["n"], 1))
        if fa <= 1e-5:
            bright.append(dict(
                start=st, fa="%.3e" % fa, count=cnt, bin_us="%.3f" % (bb * 1e6),
                win_us="%.3f" % (ws * 1e6), n=a["n"], n_acd=a["n_acd"],
                n_acd_multi=a["n_acd_multi"], n_bg=a["n_bg"], n_acd_bg=a["n_acd_bg"],
                base_zero=sum(1 for x in base if x == 0) if base else "",
                det_max="%.3f" % (max(det.get("window") or [0]) / max(cnt, 1)),
                lat="%.3f" % s["position"]["latitude"],
                lon="%.3f" % s["position"]["longitude"]))

np.savez(prefix + "_%d.npz" % shard,
         n_tot=n_tot, n_no_cpd=n_no_cpd, obs_hit=obs_hit, exp_hit=exp_hit,
         sum_nacd=sum_nacd, sum_mu=sum_mu, sum_ncore=sum_ncore,
         zero_any=zero_any, zero_hist=zero_hist, det_zero=det_zero,
         base_min=np.array(base_min_all), fa_cnt=fa_cnt,
         fa_obs=fa_obs, fa_exp=fa_exp, fa_n=fa_n,
         b_obs=b_obs, b_exp=b_exp, b_n=b_n, e_n=e_n,
         samp_bin=np.array(samp_bin), samp_cnt=np.array(samp_cnt),
         samp_env=np.array(samp_env), samp_ratio=np.array(samp_ratio))
if bright:
    w = csv.DictWriter(open(prefix + "_bright_%d.csv" % shard, "w", newline=""),
                       fieldnames=list(bright[0].keys()))
    w.writeheader()
    w.writerows(bright)
print("候选 %d 个, 无 CPD 的 %d 个, fa<=1e-5 的 %d 个" % (n_tot, n_no_cpd, len(bright)))
