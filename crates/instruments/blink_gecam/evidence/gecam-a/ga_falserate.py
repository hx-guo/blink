"""`f₃ > 0` 这一刀对**真 TGF** 的误否决率——解析式，不靠候选样本。

判据的零假设标定（「纯本底下 f₃ > 0 的偶然期望」）只说了它不会误杀**本底**，
**没有说它不会误杀信号**。真 TGF 是 n 个光子挤在 W 里，密度比本底高三个数量级，
容差 τ 一放开，几个真光子自己就会连成 ≥3 的簇。这一条必须在用判据之前算，
不能等闪电关联回来才发现真样本被切光了。

单链接分簇：在窗内把 n 个光子当均匀随机点（TGF 的时间结构比这更集中，所以这是
下界），相邻间隔 ≤ τ 就连起来。记 r = n/W、p = 1 − e^(−rτ)：
  * 某个光子与右邻连上的概率 = p；
  * 「≥3 个**不同探头**落进一簇」需要连续两次连上，期望簇数 ≈ (n−2)·p²
    （这里假定 25 路探头下同簇的两三个光子多半来自不同探头，是保守的一侧）；
  * P(整窗至少一个三重) ≈ 1 − exp(−(n−2)·p²)。

τ = 0 时 p = 1 − e^(−n·q/W)，q 是量化格——这就是 C 星那条「窗 ≥ 10 µs 时
f₃ > 0 就是粒子，零参数零阈值」的来处。τ 一旦放到 150 ns，p 涨 5 倍、p² 涨 25 倍，
**零参数的性质就没了**。

用法: python3 ga_falserate.py
"""

import numpy as np

Q_A = 29.8023e-9      # GECAM-A 2023-04-01 之后的时戳格
TAUS = (0.0, 60e-9, 150e-9, 300e-9)
# 已发表 TGF 的典型形态。W 用 gecamB 拿同一套搜索量到的 bin_size_best 中位
# 111.4 µs（140 个，**旧二进制 e34f560，正在重算**）；n 取几档覆盖暗到亮。
CASES = [("暗 TGF", 10, 111.4e-6), ("典型 TGF", 30, 111.4e-6),
         ("亮 TGF", 100, 111.4e-6), ("亮而短", 100, 30e-6),
         ("A 星幸存者中位", 8, 46.6e-6)]


def main():
    print(f"{'样本':<16}{'n':>5}{'W µs':>9}", end="")
    for tau in TAUS:
        print(f"{'τ=' + str(int(tau * 1e9)) + 'ns':>12}", end="")
    print()
    for name, n, w in CASES:
        print(f"{name:<16}{n:>5}{w * 1e6:>9.1f}", end="")
        for tau in TAUS:
            reach = tau if tau > 0 else Q_A
            p = 1.0 - np.exp(-n * reach / w)
            prob = 1.0 - np.exp(-max(n - 2, 0) * p * p)
            print(f"{prob * 100:>11.2f}%", end="")
        print()
    print()
    print("同一张表的另一读法：单链接簇的期望大小 (1+p)/(1−p)，看链会不会跑掉")
    print(f"{'样本':<16}{'n':>5}{'W µs':>9}", end="")
    for tau in TAUS:
        print(f"{'τ=' + str(int(tau * 1e9)) + 'ns':>12}", end="")
    print()
    for name, n, w in CASES:
        print(f"{name:<16}{n:>5}{w * 1e6:>9.1f}", end="")
        for tau in TAUS:
            reach = tau if tau > 0 else Q_A
            p = 1.0 - np.exp(-n * reach / w)
            print(f"{(1 + p) / (1 - p):>12.3f}", end="")
        print()


if __name__ == "__main__":
    main()
