"""GECAM-B：147 个已发表 TGF（Zhao et al. 2023 GRL, Zenodo 10.5281/zenodo.8028217）
的完备性，对当前搜索产物重做。

匹配口径写死：候选窗一律取**最佳格** `[start + delay, start + delay + bin_size_best]`，
与目录 `UT` 比。`UT` 给的是暴发起点，最佳格有时落在后半段，所以 `dt` 的正长尾是
正常的，判定窗取 ±`MATCH_MS` 毫秒而不是要求重叠。

输出三份：
  * `recall.csv` —— 147 行逐条：匹配到没有、`dt`、`fa`、`count`、`bin_us`
  * `pool.csv`   —— 138 小时的账本（searched / excluded、曝光、候选数）
  * 标准输出     —— 完备性、`dt` 分位、以及没找回的那几个的清单

用法: python3 gb_recall.py <data 根目录> <输出目录>
"""

import csv
import datetime as dt
import json
import os
import sys

import numpy as np

EPOCH = (2019, 1, 1)
CATALOG = os.environ.get("GB_CATALOG", "/scratchfs2/gecam/guohx/gecambrun/gecam_tgf_catalog.csv")
MATCH_MS = 10.0   # 判定窗半宽（ms）：目录 UT 与最佳格起点之差的上限
SIGNIFICANT = 1e-5


def met(iso):
    head, _, frac = iso.rstrip("Z").partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    ref = dt.datetime(*EPOCH, tzinfo=dt.timezone.utc)
    return (stamp - ref).total_seconds() + (float("0." + frac) if frac else 0.0)


def main():
    root, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    catalog = list(csv.DictReader(open(CATALOG)))
    print(f"目录 {len(catalog)} 行，落在 {len({c['UT'][:13] for c in catalog})} 个小时、"
          f"{len({c['UT'][:10] for c in catalog})} 天")

    days = sorted({c["UT"][:10] for c in catalog})
    hours_wanted = sorted({c["UT"][:13] for c in catalog})

    # 逐天读候选与账本
    pool = {}      # hour -> list of (t0, t1, fa, count, bin_s, mean, det_frac)
    ledger = {}    # hour -> (status, reason, exposure)
    for day in days:
        base = f"{root}/{day[:4]}/{day[5:7]}/{day.replace('-', '')}"
        sig, hrs = base + "_signals.json", base + "_hours.json"
        if os.path.exists(hrs):
            for rec in json.load(open(hrs))["hours"]:
                ledger["%sT%02d" % (day, rec["hour"])] = rec
        if not os.path.exists(sig):
            continue
        for s in json.load(open(sig)):
            key = s["start"][:13]
            if key not in hours_wanted:
                continue
            t0 = met(s["start"]) + s["delay"]
            pool.setdefault(key, []).append(
                (t0, t0 + s["bin_size_best"], s["false_positive_per_year"], s["count"],
                 s["bin_size_best"], s["mean"]))
    for key in pool:
        pool[key].sort()

    # 【硬断言】产出天数 / 小时数对不上就停。队列空不等于跑完：worker 被挤掉或被误删
    # 之后产物照样齐全、格式正确、数字看着合理，只是建立在残缺输入上。
    n_day_files = sum(1 for d in days
                      if os.path.exists(f"{root}/{d[:4]}/{d[5:7]}/{d.replace('-', '')}_signals.json"))
    n_hour_searched = sum(1 for k in hours_wanted if ledger.get(k, {}).get("status") == "searched")
    print(f"【完整性断言】TGF 所在的天 {n_day_files}/{len(days)} 有候选文件；"
          f"138 小时里 searched {n_hour_searched}/{len(hours_wanted)}")
    if n_day_files != len(days) or n_hour_searched != len(hours_wanted):
        print("！输入残缺，停。补齐之后再算，别在残缺对照组上做判据定标。")
        sys.exit(2)

    rows = []
    for c in catalog:
        key = c["UT"][:13]
        ut = met(c["UT"])
        cands = pool.get(key, [])
        best = None
        for t0, t1, fa, n, binsize, mean in cands:
            d = (t0 - ut) * 1e3
            if abs(d) <= MATCH_MS:
                # 同一暴发可能出好几个候选，取最显著的那个
                if best is None or fa < best[2]:
                    best = (d, t0, fa, n, binsize, mean)
        rows.append({
            "UT": c["UT"], "hour": key,
            "Duration_us": c["Duration_us"], "NetCounts": c["NetCounts"],
            "CPDtoGRD": c["CPDtoGRDcountsRatio"], "lat": c["Latitude_deg"], "lon": c["Longitude_deg"],
            "n_cand_hour": len(cands),
            "matched": int(best is not None),
            "dt_ms": "" if best is None else f"{best[0]:.4f}",
            # 全精度的最佳格起点，供 gb_criteria.py 逐条对回特征表（4 位小数的 dt 对不回去）
            "t0_met": "" if best is None else repr(best[1]),
            "fa": "" if best is None else f"{best[2]:.4e}",
            "count": "" if best is None else best[3],
            "bin_us": "" if best is None else f"{best[4] * 1e6:.3f}",
            "mean": "" if best is None else f"{best[5]:.4f}",
            "significant": "" if best is None else int(best[2] <= SIGNIFICANT),
        })

    with open(os.path.join(outdir, "recall.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    matched = [r for r in rows if r["matched"]]
    missed = [r for r in rows if not r["matched"]]
    sig = [r for r in matched if r["significant"] == 1]
    print(f"\n找回 {len(matched)}/{len(rows)} = {len(matched) / len(rows) * 100:.1f}%；"
          f"其中 fa <= {SIGNIFICANT:g} 的 {len(sig)}/{len(matched)} = "
          f"{len(sig) / max(len(matched), 1) * 100:.1f}%")
    d = np.array([float(r["dt_ms"]) for r in matched])
    print(f"dt（最佳格起点 − 目录 UT，ms）：中位 {np.median(d):+.4f}，"
          f"5–95% {np.percentile(d, 5):+.4f} .. {np.percentile(d, 95):+.4f}，"
          f"min {d.min():+.4f} max {d.max():+.4f}")
    # dt 的中位（+0.096 ms 量级）比暴发时长中位（179.5 µs）还大，所以"没有系统性偏移"
    # 这句话在这个粒度上没有分辨力。分两种可能，靠相关性分开：
    #   * dt 与 Duration_us 正相关 → 目录 UT 给起点、我们的最佳格落在中后段，口径差、无害；
    #   * dt 与两者都不相关而有固定正偏置 → 时基有系统偏移，要回去查 MET↔UTC。
    cat_dur = np.array([float(r["Duration_us"]) for r in matched])
    cat_net = np.array([float(r["NetCounts"]) for r in matched])
    our_bin = np.array([float(r["bin_us"]) for r in matched])

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    print("\ndt 的来源诊断（Spearman 秩相关，n=%d）：" % len(matched))
    print(f"  dt vs 目录 Duration_us : {spearman(d, cat_dur):+.3f}")
    print(f"  dt vs 目录 NetCounts   : {spearman(d, cat_net):+.3f}")
    print(f"  dt vs 我们的 bin_size  : {spearman(d, our_bin):+.3f}")
    print(f"  dt / 目录 Duration_us 之比：中位 {np.median(d * 1e3 / cat_dur):+.3f}，"
          f"|dt| < Duration 的占 {(np.abs(d) * 1e3 < cat_dur).mean() * 100:.1f}%")
    for lo, hi in ((0, 100), (100, 200), (200, 400), (400, 1000)):
        m = (cat_dur >= lo) & (cat_dur < hi)
        if m.sum():
            print(f"  Duration {lo:4d}–{hi:4d} µs（n={m.sum():3d}）：dt 中位 {np.median(d[m]):+.4f} ms")

    nn = np.array([float(r["count"]) for r in matched])
    print(f"找回者的 count：中位 {np.median(nn):.0f}，5–95% {np.percentile(nn, 5):.0f} .. "
          f"{np.percentile(nn, 95):.0f}，min {nn.min():.0f}")

    # 匹配容差要并排给出偶然期望：候选密度这么高，宽容差本身就会"匹配"上东西。
    print("\n匹配容差扫描（偶然期望 = 2·容差 × 该小时候选率，逐 TGF 求和）：")
    for tol in (0.5, 1.0, 3.0, 10.0):
        hit, chance = 0, 0.0
        for c, r in zip(catalog, rows):
            key = c["UT"][:13]
            rate = len(pool.get(key, [])) / max(ledger.get(key, {}).get("searched_seconds", 0.0) or 1.0, 1e-9)
            chance += 1 - np.exp(-2 * tol * 1e-3 * rate)
            if r["dt_ms"] and abs(float(r["dt_ms"])) <= tol:
                hit += 1
        print(f"  ±{tol:5.1f} ms: 找回 {hit}/{len(rows)} = {hit / len(rows) * 100:5.1f}%，"
              f"偶然期望 {chance:.2f} 个（占找回数的 {chance / max(hit, 1) * 100:.2f}%）")

    print(f"\n没找回 {len(missed)} 个：")
    for r in missed:
        print(f"  {r['UT']}  Duration {float(r['Duration_us']):7.1f} µs  "
              f"NetCounts {float(r['NetCounts']):6.1f}  该小时候选 {r['n_cand_hour']}")

    # 账本
    with open(os.path.join(outdir, "pool.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["hour", "status", "reason", "searched_s", "n_signals", "n_cand"])
        total_exp, total_cand, searched = 0.0, 0, 0
        for key in hours_wanted:
            rec = ledger.get(key, {})
            status = rec.get("status", "MISSING")
            exposure = rec.get("searched_seconds", 0.0) or 0.0
            n = len(pool.get(key, []))
            writer.writerow([key, status, rec.get("reason", ""), f"{exposure:.3f}",
                             rec.get("n_signals", ""), n])
            if status == "searched":
                searched += 1
                total_exp += exposure
            total_cand += n
    print(f"\n138 小时账本：searched {searched}，曝光 {total_exp:.0f} s = {total_exp / 86400:.3f} 天，"
          f"候选 {total_cand}（{total_cand / max(total_exp, 1) * 86400:.0f} 个/天）")


if __name__ == "__main__":
    main()
