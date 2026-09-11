"""雷暴邻近度这条判据到底独不独立于 ±5 ms 关联：按当地昼夜拆开看超出。

WWLLN 夜里探测效率高（我们自己量到关联者峰 22.99 LT、未关联者 19.30 LT，
差 +3.69 h）。如果雷暴邻近度的超出主要来自夜间，那它和 ±5 ms 关联共用同一个
效率偏差、不能当独立证据；如果昼夜相当，说明在 300 km/±600 s 这个尺度上
效率偏差被平均掉了，两条可以当准独立。
"""
import csv
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
FAR = ["storm_d+3", "storm_d-3", "storm_d+7", "storm_d-7"]


def lst_of(row):
    body = row["start"].rstrip("Z")
    head, _, _f = body.partition(".")
    t = datetime.strptime(head, "%Y-%m-%dT%H:%M:%S")
    lon = float(row["lon"])
    lon = lon - 360 if lon > 180 else lon
    return (t.hour + t.minute / 60 + t.second / 3600 + lon / 15) % 24


def load(name):
    return list(csv.DictReader(open(os.path.join(HERE, name))))


def excess(rows, thresh):
    if not rows:
        return float("nan"), float("nan"), 0
    real = sum(1 for r in rows if int(r["storm"]) >= thresh) / len(rows)
    far = sum(sum(1 for r in rows if int(r[k]) >= thresh) / len(rows) for k in FAR) / len(FAR)
    return real, far, len(rows)


def main():
    pops = [("目录匹配真 TGF（正对照）", "storm_matched_300km.csv"),
            ("多出来的显著", "storm_extras_300km.csv"),
            ("亚阈随机样（负对照）", "storm_weak_300km.csv")]
    # 当地白天 = 06–18 LT，夜间 = 18–06 LT
    for thresh in (10, 50):
        print(f"=== 至少 {thresh} 次闪电（300 km / ±600 s），按候选自己的当地昼夜拆开 ===")
        print(f"{'人群':<24s} {'段':<4s} {'n':>5s}  {'真时刻':>7s} {'±3/±7天':>8s} {'超出':>8s} {'倍数':>6s}")
        for label, fn in pops:
            rows = load(fn)
            day = [r for r in rows if 6 <= lst_of(r) < 18]
            night = [r for r in rows if not (6 <= lst_of(r) < 18)]
            for seg, rs in (("白天", day), ("夜间", night)):
                real, far, n = excess(rs, thresh)
                print(f"{label:<24s} {seg:<4s} {n:5d}  {100 * real:6.1f}% {100 * far:7.1f}% "
                      f"{100 * (real - far):+7.1f} {real / far if far else float('inf'):6.2f}×")
            print()
        print()

    print("=== 同样的拆分，只看没有 ±5 ms 关联的子集（至少 50 次）===")
    print(f"{'人群':<24s} {'段':<4s} {'n':>5s}  {'真时刻':>7s} {'±3/±7天':>8s} {'超出':>8s} {'倍数':>6s}")
    for label, fn in pops:
        rows = [r for r in load(fn) if r["assoc"] == "0"]
        day = [r for r in rows if 6 <= lst_of(r) < 18]
        night = [r for r in rows if not (6 <= lst_of(r) < 18)]
        for seg, rs in (("白天", day), ("夜间", night)):
            real, far, n = excess(rs, 50)
            print(f"{label:<24s} {seg:<4s} {n:5d}  {100 * real:6.1f}% {100 * far:7.1f}% "
                  f"{100 * (real - far):+7.1f} {real / far if far else float('inf'):6.2f}×")
        print()

    print("=== 旁证：±5 ms 关联率本身的昼夜比（VLF 效率偏差的直接读数）===")
    for label, fn in pops:
        rows = load(fn)
        day = [r for r in rows if 6 <= lst_of(r) < 18]
        night = [r for r in rows if not (6 <= lst_of(r) < 18)]
        kd = sum(1 for r in day if r["assoc"] == "1")
        kn = sum(1 for r in night if r["assoc"] == "1")
        rd = kd / len(day) if day else 0
        rn = kn / len(night) if night else 0
        print(f"  {label:<24s} 白天 {kd:3d}/{len(day):3d} = {100 * rd:5.2f}%   "
              f"夜间 {kn:3d}/{len(night):3d} = {100 * rn:5.2f}%   夜/昼 = "
              f"{rn / rd if rd else float('inf'):.2f}")


if __name__ == "__main__":
    main()
