"""合并 8 片 pool_*.npz。"""
import glob, math
import numpy as np

fs = sorted(glob.glob("/scratchfs2/gecam/guohx/gecambrun/pool_*.npz"))
acc = {}
for f in fs:
    z = np.load(f)
    for k in z.files:
        v = z[k]
        if k in ("base_min", "samp_bin", "samp_cnt", "samp_env", "samp_ratio"):
            acc.setdefault(k, []).append(v)
        else:
            acc[k] = acc.get(k, 0) + v
for k in ("base_min", "samp_bin", "samp_cnt", "samp_env", "samp_ratio"):
    acc[k] = np.concatenate(acc[k])

n = int(acc["n_tot"])
print("候选合计 %d（tgf_compare 报 3008278）" % n)
print("无 CPD 文件的候选 %d" % int(acc["n_no_cpd"]))

print("\n=== 一、CPD 符合（候选窗 [start,stop]，期望按各自 n_acd_bg/2s 算）===")
print("  命中 %d/%d = %.2f%%   偶然期望 %.1f (%.2f%%)   obs/exp = %.3f"
      % (acc["obs_hit"], n, 100 * acc["obs_hit"] / n, acc["exp_hit"],
         100 * acc["exp_hit"] / n, acc["obs_hit"] / acc["exp_hit"]))
print("  窗内 CPD 计数合计 %d, 期望 %.1f, 比 %.3f"
      % (acc["sum_nacd"], acc["sum_mu"], acc["sum_nacd"] / acc["sum_mu"]))
print("  净 CPD/GRD = (Σn_acd − Σμ)/Σn = %.5f"
      % ((acc["sum_nacd"] - acc["sum_mu"]) / acc["sum_ncore"]))

fa_bins = [1e-10, 1e-7, 1e-5, 1e-3, 1e-1, 1.0, 20.0, 1e9]
lbl = ["<=1e-10", "1e-10..1e-7", "1e-7..1e-5", "1e-5..1e-3", "1e-3..0.1",
       "0.1..1", "1..20", ">20"]
print("\n  按 fa 分档:")
print("  %-14s %10s %8s %8s %8s" % ("fa 档", "候选数", "命中%", "期望%", "obs/exp"))
for i, l in enumerate(lbl):
    if acc["fa_n"][i] == 0:
        continue
    print("  %-14s %10d %7.2f%% %7.2f%% %8.3f"
          % (l, acc["fa_n"][i], 100 * acc["fa_obs"][i] / acc["fa_n"][i],
             100 * acc["fa_exp"][i] / acc["fa_n"][i],
             acc["fa_obs"][i] / max(acc["fa_exp"][i], 1e-9)))

be = ["<1 µs", "1-10 µs", "10-50 µs", "50-200 µs", "200-1000 µs", ">1 ms"]
print("\n  按最佳格宽分档:")
print("  %-14s %10s %8s %8s %8s" % ("bin 档", "候选数", "命中%", "期望%", "obs/exp"))
for i, l in enumerate(be):
    if acc["b_n"][i] == 0:
        continue
    print("  %-14s %10d %7.2f%% %7.2f%% %8.3f"
          % (l, acc["b_n"][i], 100 * acc["b_obs"][i] / acc["b_n"][i],
             100 * acc["b_exp"][i] / acc["b_n"][i],
             acc["b_obs"][i] / max(acc["b_exp"][i], 1e-9)))

print("\n=== 二、逐路基线零格（探头级停机）===")
print("  有零格的候选 %d/%d = %.4f%%" % (acc["zero_any"], n, 100 * acc["zero_any"] / n))
zh = acc["zero_hist"]
print("  零格数直方:", {i: int(v) for i, v in enumerate(zh) if v})
dz = acc["det_zero"]
print("  逐路零格次数:", {i + 1: int(v) for i, v in enumerate(dz) if v})
bm = acc["base_min"]
print("  基线向量最小值（1/97 抽样 %d 个）: 中位 %.0f  1%% %.0f  5%% %.0f  最小 %.0f"
      % (len(bm), np.median(bm), *np.percentile(bm, [1, 5]), bm.min()))

print("\n=== 三、候选池分布（1/97 抽样 %d 个）===" % len(acc["samp_bin"]))
b = acc["samp_bin"] * 1e6
c = acc["samp_cnt"]
e = acc["samp_env"]
print("  bin_size_best µs: 中位 %.3f  5%% %.3f  25%% %.3f  75%% %.3f  95%% %.3f"
      % (np.median(b), *np.percentile(b, [5, 25, 75, 95])))
for lim in (1, 10, 50, 100, 200, 1000):
    print("    bin < %5d µs 的 %5.1f%%" % (lim, 100 * (b < lim).mean()))
print("  count: 中位 %.0f  5%% %.0f  95%% %.0f" % (np.median(c), *np.percentile(c, [5, 95])))
print("  CPD 环境率 c/s: 中位 %.0f  5%% %.0f  95%% %.0f" % (np.median(e), *np.percentile(e, [5, 95])))
print("    > 500 c/s 的 %.1f%%" % (100 * (e > 500).mean()))

env_edges = ["<=200", "200-400", "400-600", "600-1000", "1000-2000", ">2000"]
print("  CPD 环境率分档候选数:", {l: int(v) for l, v in zip(env_edges, acc["e_n"])})
