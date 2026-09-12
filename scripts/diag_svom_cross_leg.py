"""两条腿互相检验：它们对「显著池里的本底成分是什么构成」给出可比的说法。

* **LST 腿**（第 32c 条）逐类给 f：陆地 0.84、近岸 0.33、远洋 0.29。
  由它可以反推显著池里本底成分的下垫面构成：N_c × (1 − f_c)。
* **构成腿**（第 34 条）**假定**显著池里的本底成分与低显著档（fa 5–20）的本底同构成，
  由此给出 f ≥ 0.545。

**这两个说法必须自洽。** 若 LST 反推出来的本底构成与低显著档的本底构成差得远，
那就是构成腿那条假定不成立——而那正是统筹点名要查的入口，也是
「下界 0.545 比幂律外推的 0.46 高」这处张力最可能的去处。

用法: python3 crossleg.py <pool.csv>
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import numpy as np

from diag_svom_lst_template_crosscheck import load, PT_NAMES

rows = load(sys.argv[1])


def counts(sel):
    sub = [r for r in rows if sel(r)]
    return np.array([sum(1 for r in sub if r["pt"] == s) for s in range(3)], float)


sig = counts(lambda r: r["fa"] <= 1e-5)
bkg = counts(lambda r: 5.0 < r["fa"] <= 20.0)
p_bkg = bkg / bkg.sum()
print("显著池构成      ：" + "  ".join("%s %5.1f%%" % (PT_NAMES[s], 100 * sig[s] / sig.sum())
                                  for s in range(3)) + "   N=%d" % sig.sum())
print("低显著档本底构成：" + "  ".join("%s %5.1f%%" % (PT_NAMES[s], 100 * p_bkg[s])
                                  for s in range(3)) + "   N=%d" % bkg.sum())

# LST 腿给的逐类 f（GBM 模板 / HXMT 单点模板），以及各自的零分布 sigma
for tag, f, e in (("GBM 模板", np.array([0.84, 0.33, 0.29]), np.array([0.094, 0.246, 0.244])),
                  ("HXMT 单点模板", np.array([0.94, 0.28, 0.47]), np.array([0.113, 0.209, 0.310]))):
    b = sig * (1.0 - f)
    eb = sig * e
    pb = b / b.sum()
    # 误差按独立传播（保守：忽略 f 之间的相关）
    epb = np.sqrt((eb / b.sum()) ** 2 + (b / b.sum() ** 2 * np.sqrt((eb ** 2).sum())) ** 2)
    print("\n== 用 %s 的逐类 f 反推显著池里的本底成分 ==" % tag)
    print("  逐类本底数： " + "  ".join("%s %5.1f ± %.1f" % (PT_NAMES[s], b[s], eb[s])
                                   for s in range(3)) + "   合计 %.0f" % b.sum())
    print("  反推的本底构成：" + "  ".join("%s %5.1f ± %4.1f%%" % (PT_NAMES[s], 100 * pb[s],
                                                          100 * epb[s]) for s in range(3)))
    print("  低显著档本底  ：" + "  ".join("%s %5.1f%%" % (PT_NAMES[s], 100 * p_bkg[s])
                                    for s in range(3)))
    z = (pb - p_bkg) / np.maximum(epb, 1e-9)
    print("  差（σ）       ：" + "  ".join("%s %+5.1f" % (PT_NAMES[s], z[s]) for s in range(3)))
    chi = float((z ** 2)[:2].sum())     # 三个占比只有两个自由度
    print("  ⇒ 两条腿%s（χ² = %.1f/2）" % ("自洽" if chi < 6 else "**不自洽**", chi))

print("""
读法：反推的本底构成若在近岸上明显高于低显著档本底，只有两种可能——
(a) LST 腿的近岸 f 偏低（它只有 1.2σ，本来就定不住），
(b) 构成腿那条「显著池本底与低显著档本底同构成」的假定不成立。
两者都会让 f >= 0.545 这个下界站不住，所以这处要留着不要抹平。""")
