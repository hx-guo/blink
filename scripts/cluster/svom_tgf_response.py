"""为 83 个闪电证实 TGF 导出「沉积谱 + 该入射方向的响应矩阵」，供正向折叠拟合光子谱。

逐个 TGF：
  1. 载入所在小时的事例（EVT_TYPE==0、ANTI_COIN==0，能道不切，留给离线）；
  2. 在 ±5 ms 内做非分箱极大似然拟合（已知本底率 + 高斯脉冲）定出 t0、σ，
     核心窗取 t0 ± 2σ（与 evidence/sample 的口径一致）；
  3. 核心窗与本底环（10–50 ms，AC==0）的逐探头逐道计数；
  4. 逐探头死时间占比 f = Σ DEAD_TIME / 窗长（亮暴发里可到 0.5）；
  5. 入射方向 (θ, φ)（载荷系，由姿态 + 天底给出，见 tgf_incidence.csv），
     调官方 RSP_Generator 生成该方向、该 MET 下三路 GRD 的响应
     （300 入射能 × 259 飞行道，cm²，已含能量分辨与增益）。

用法: python3 svom_tgf_response.py <incidence.csv> <out.npz> [WORKERS] [IDX]
"""
import csv
import datetime as dt
import glob
import sys

sys.path.insert(0, "/gecamfs/SVOM/soft/CALDB/software")
import numpy as np
from astropy.io import fits
from scipy.optimize import minimize
from RSP_Generator import gen_rsp

D = "/gecamfs/SVOM/Archived-DATA/GRM-DATA/L1B/daily"
REF = dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc)
NCH = 259
FIT_HALF = 0.005          # 时间拟合半窗
BKG_IN, BKG_OUT = 0.010, 0.050


def met(iso):
    b = iso.rstrip("Z")
    h, _, f = b.partition(".")
    s = dt.datetime.strptime(h, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return (s - REF).total_seconds() + (float("0." + f) if f else 0.0)


def latest(pat):
    g = sorted(glob.glob(pat))
    if not g:
        return None
    seen = {}
    for f in g:
        seen[f.rsplit("_v", 1)[0]] = f
    return sorted(seen.values())[-1]


def load_hour(m0):
    utc = REF + dt.timedelta(seconds=m0)
    f = latest("%s/%s/grm_evt/svom_grm_evt_%s_%s_v*.fits"
               % (D, utc.strftime("%Y/%m/%d"), utc.strftime("%y%m%d"), utc.strftime("%H")))
    if f is None:
        return None
    cols = {k: [] for k in ("t", "pi", "dead", "det")}
    with fits.open(f) as h:
        gti = [(float(a), float(b)) for a, b in zip(h["GTI"].data["START"], h["GTI"].data["STOP"])]
        for i, det in zip((3, 4, 5), (1, 2, 3)):
            d = h[i].data
            t = np.asarray(d["TIME"], float)
            pi = np.asarray(d["PI"])
            et = np.asarray(d["EVT_TYPE"])
            ac = np.asarray(d["ANTI_COIN"])
            de = np.asarray(d["DEAD_TIME"], float)
            k = (et == 0) & (ac == 0) & (pi < 256)
            cols["t"].append(t[k])
            cols["pi"].append(pi[k].astype(np.int16))
            cols["dead"].append(de[k])
            cols["det"].append(np.full(int(k.sum()), det, np.int8))
    a = {k: np.concatenate(v) for k, v in cols.items()}
    o = np.argsort(a["t"], kind="stable")
    a = {k: v[o] for k, v in a.items()}
    return a, gti


SIG_MIN, SIG_MAX = 2e-6, 2e-3      # σ 的下界取时间戳量化步的两倍；无界会塌到 δ 函数


def fit_pulse(rel, rate):
    """非分箱扩展极大似然：已知本底率 rate（每秒），加一个高斯脉冲。

    σ 必须夹住：似然对 σ→0 是发散的（脉冲塌到单个事例上），无界优化必然跑到那里。
    下界 2 µs 是时间戳量化步（0.954 µs）的两倍，物理上再窄也分辨不出来。"""
    n = len(rel)
    if n < 8:
        return None

    def nll(p):
        t0, lns, lnN = p
        s = float(np.exp(lns))
        if not (SIG_MIN <= s <= SIG_MAX) or not np.isfinite(lnN) or lnN > 12:
            return 1e12
        N = float(np.exp(lnN))
        dens = rate + N * np.exp(-0.5 * ((rel - t0) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        if not np.all(dens > 0):
            return 1e12
        return -(np.log(dens).sum()) + (rate * 2 * FIT_HALF + N)

    best = None
    n_net = max(n - rate * 2 * FIT_HALF, 1.0)
    for s0 in (5e-6, 2e-5, 6e-5, 2e-4, 6e-4):
        for t00 in (0.0, np.median(rel)):
            r = minimize(nll, [t00, np.log(s0), np.log(n_net)], method="Nelder-Mead",
                         options=dict(maxiter=6000, maxfev=6000, xatol=1e-9, fatol=1e-6))
            if best is None or r.fun < best.fun:
                best = r
    s = float(np.clip(np.exp(best.x[1]), SIG_MIN, SIG_MAX))
    return best.x[0], s, float(np.exp(best.x[2]))


def main(inc_csv, out, workers=1, idx=0):
    rows = list(csv.DictReader(open(inc_csv)))
    core = np.zeros((len(rows), 3, NCH))
    bkgs = np.zeros((len(rows), 3, NCH))
    resp = np.zeros((len(rows), 3, 300, NCH), np.float32)
    meta = np.zeros((len(rows), 12))
    done = np.zeros(len(rows), bool)
    elo = ehi = clo = chi = None

    for i, r in enumerate(rows):
        if i % workers != idx:
            continue
        m0 = met(r["start"])
        got = load_hour(m0)
        if got is None:
            print("缺文件", r["start"], flush=True)
            continue
        a, gti = got
        sel = np.abs(a["t"] - m0) <= BKG_OUT
        t = a["t"][sel] - m0
        pi = a["pi"][sel]
        de = a["dead"][sel]
        det = a["det"][sel]
        ring = (np.abs(t) >= BKG_IN) & (np.abs(t) <= BKG_OUT)
        live_ring = 2 * (BKG_OUT - BKG_IN)
        rate = ring.sum() / live_ring
        inner = np.abs(t) <= FIT_HALF
        fit = fit_pulse(t[inner], rate)
        if fit is None:
            print("拟合失败", r["start"], flush=True)
            continue
        t0, sig, nsig = fit
        lo, hi = t0 - 2 * sig, t0 + 2 * sig
        m_core = (t >= lo) & (t <= hi)
        for d in (1, 2, 3):
            kc = m_core & (det == d)
            kb = ring & (det == d)
            core[i, d - 1] = np.bincount(pi[kc], minlength=NCH)[:NCH]
            bkgs[i, d - 1] = np.bincount(pi[kb], minlength=NCH)[:NCH]
            meta[i, 2 + d] = de[kc].sum() * 1e-6 / max(hi - lo, 1e-9)      # 死时间占比
        meta[i, 0] = hi - lo
        meta[i, 1] = live_ring
        meta[i, 2] = m0 + t0
        meta[i, 6] = sig
        meta[i, 7] = nsig
        meta[i, 8] = float(r["theta"])
        meta[i, 9] = float(r["phi"])

        for d in (1, 2, 3):
            g = gen_rsp(detname="g%02d" % d, theta=float(r["theta"]), phi=float(r["phi"]),
                        MET=m0)
            if g is None:
                print("gen_rsp 失败", r["start"], d, flush=True)
                break
            mtx, el, eh, cl, ch, _, _ = g
            resp[i, d - 1] = np.asarray(mtx, np.float32)
            if elo is None:
                elo, ehi, clo, chi = (np.asarray(el, float), np.asarray(eh, float),
                                      np.asarray(cl, float), np.asarray(ch, float))
        else:
            done[i] = True
        print("done %3d %s θ=%.1f φ=%.1f σ=%.1fµs 核心 %.0f 本底率 %.0f c/s"
              % (i, r["start"][:23], float(r["theta"]), float(r["phi"]), sig * 1e6,
                 core[i].sum(), rate), flush=True)

    np.savez_compressed(out, core=core, bkg=bkgs, resp=resp, meta=meta, done=done,
                        elo=elo, ehi=ehi, clo=clo, chi=chi,
                        starts=np.array([r["start"] for r in rows]))
    print("→", out, "完成", int(done.sum()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 1,
         int(sys.argv[4]) if len(sys.argv) > 4 else 0)
