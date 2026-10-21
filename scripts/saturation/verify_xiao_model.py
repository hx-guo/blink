#!/usr/bin/env python3
"""验证 Xiao 2020 的 HE 死时间模型 (per ADC, 6 detectors coupled).

核心论点（论文 Eq. 1）:
  Dt_i = a·N_l_i + b·N_w_i + c·N_n_i
       + a'·N_l_5 + b'·N_w_5 + c'·N_n_5

变量映射:
  N_n  (Normal)  : 完整记录的事例 = Sci EVT count
  N_l  (Large)   : Cnt_LargeEvt   (channel > 275, 仅 1s 计数率)
  N_w  (Wide)    : Cnt_CsI_PHODet (pulse width > 120, 仅 1s 计数率, 与 CsI 无关)
  PHO            : Cnt_PHODet = N_n + N_w + N_l (总触发计数)

方法:
  1. 用 Rust 检测的饱和区间排除受影响的 1s bins
  2. 在剩余非饱和 bins 上线性拟合 (a, b, c, a', b', c')
  3. 残差 → 模型质量
  4. 同时检验 Sci ≈ PHO - CsI - Large
"""
import numpy as np
from astropy.io import fits
import csv
from unwrap_large import unwrap_large

TRIGGER_MET = 446726278.0
MET_CORRECTION = 4.0

BOXES = [
    ("A", "0766", "/tmp/260226_boxA_full.csv", 0),   # det 0..5
    ("B", "1009", "/tmp/260226_boxB_full.csv", 6),   # det 6..11
    ("C", "1781", "/tmp/260226_boxC_full.csv", 12),  # det 12..17
]

# --- 读饱和区间 ---
sat_intervals = {"A": [], "B": [], "C": []}
with open("/tmp/detect_260226a.csv") as f:
    for r in csv.DictReader(f):
        sat_intervals[r["box"]].append((float(r["start_met"]), float(r["stop_met"])))
for k in sat_intervals:
    sat_intervals[k].sort()
print(f"Saturation intervals: A={len(sat_intervals['A'])}, B={len(sat_intervals['B'])}, C={len(sat_intervals['C'])}")
print()

def overlaps_saturation(t0, t1, intervals):
    """1s bin [t0,t1] 是否与任何饱和区间相交"""
    for s, e in intervals:
        if s < t1 and e > t0:
            return True
    return False

for box_name, eng_code, sci_csv, det_off in BOXES:
    eng_file = f"data/1B/2026/20260226/{eng_code}/HXMT_1B_{eng_code}_20260226T100000_G076262_000_004.fits"
    fe = fits.open(eng_file, memmap=True)
    d = fe["HE_Eng"].data
    offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
    met_eng = d["Time"].astype(float) + offset + MET_CORRECTION  # bin 起点
    L_cycles = d["Length_Time_Cycle"].astype(float)              # 1 cycle = 16us
    # N=6 探头：N_n_i (Sci), N_w_i (CsI_PHO), N_l_i (Large) per detector
    det_ids = [det_off + i for i in range(6)]
    PHO  = np.column_stack([d[f"Cnt_PHODet_{i}"].astype(float)     for i in det_ids])
    Wide = np.column_stack([d[f"Cnt_CsI_PHODet_{i}"].astype(float) for i in det_ids])
    Large_raw = np.column_stack([d[f"Cnt_LargeEvt_{i}"].astype(float) for i in det_ids])
    Large = np.column_stack([unwrap_large(PHO[:,i], Large_raw[:,i]) for i in range(6)])
    OOC  = np.column_stack([d[f"Cnt_OOCDet_{i}"].astype(float)     for i in det_ids])
    Dt   = np.column_stack([d[f"DeadTime_PHODet_{i}"].astype(float) for i in det_ids])

    # --- 读 Sci, per detector ---
    det_evts = {i: [] for i in range(6)}
    with open(sci_csv) as f:
        for r in csv.DictReader(f):
            if r["type"] == "EVT":
                det_evts[int(r["det_id"])].append(float(r["met"]))
    for k in det_evts:
        det_evts[k] = np.sort(np.array(det_evts[k]))

    # 在 Length 窗内 bin Sci
    Sci = np.zeros((len(met_eng), 6))
    length_s = L_cycles * 16e-6
    for i in range(len(met_eng)):
        t0 = met_eng[i]; t1 = t0 + length_s[i]
        for det in range(6):
            Sci[i, det] = np.searchsorted(det_evts[det], t1) - np.searchsorted(det_evts[det], t0)

    # 排除饱和 bin (任何 bin 与饱和区间相交则跳过)
    valid = np.ones(len(met_eng), dtype=bool)
    for i in range(len(met_eng)):
        t0 = met_eng[i]; t1 = t0 + length_s[i]
        if overlaps_saturation(t0, t1, sat_intervals[box_name]):
            valid[i] = False
    # 也排除 Sci 太低/Length 异常的 bin（SAA 等）
    valid &= (L_cycles > 50000) & (Sci.sum(axis=1) > 100)

    n_total = len(met_eng); n_valid = valid.sum()
    print(f"=== Box {box_name} ===")
    print(f"  bins: {n_total} total, {n_valid} non-saturated")
    print()

    # --- 测试 1: PHO 分解 ---
    # 假设 1: PHO = Sci + Wide + Large
    # 假设 2: PHO = Sci + Wide + Large + OOC (Am-241 标定源)
    print(f"  {'Det':>3s} {'PHO':>7s} {'Sci':>6s} {'Wide':>5s} {'Large':>6s} {'OOC':>5s} "
          f"{'PHO-W-L':>8s} {'r1=/Sci':>8s} {'r2=(...-OOC)/Sci':>16s}")
    for det in range(6):
        m = valid
        pho_med  = np.median(PHO[m, det])
        sci_med  = np.median(Sci[m, det])
        w_med    = np.median(Wide[m, det])
        l_med    = np.median(Large[m, det])
        o_med    = np.median(OOC[m, det])
        r1 = (PHO[m, det] - Wide[m, det] - Large[m, det]) / np.maximum(Sci[m, det], 1)
        r2 = (PHO[m, det] - Wide[m, det] - Large[m, det] - OOC[m, det]) / np.maximum(Sci[m, det], 1)
        print(f"  {det:>3d} {pho_med:>7.0f} {sci_med:>6.0f} {w_med:>5.0f} {l_med:>6.0f} "
              f"{o_med:>5.0f} {pho_med-w_med-l_med:>8.0f} {np.median(r1):>8.4f} {np.median(r2):>16.4f}")
    print()

    # --- 测试 2: Xiao 2020 拟合 ---
    # 对每个 det，用同 ADC 其他 5 个的总和作为耦合项
    print(f"  Xiao 2020 fit (Dt = a·N_l + b·N_w + c·N_n + a'·N_l_5 + b'·N_w_5 + c'·N_n_5):")
    print(f"  {'Det':>3s}{'a(N_l)':>8s}{'b(N_w)':>8s}{'c(N_n)':>8s}{'a_5':>8s}{'b_5':>8s}{'c_5':>8s}  RMS(us)")
    Dt_us = Dt * 16.0  # 16 us cycles → us
    # 用 Sci 作 N_n（实际记录）
    Nn = Sci
    Nl = Large
    Nw = Wide
    for det in range(6):
        others = [j for j in range(6) if j != det]
        Nl_5 = Nl[:, others].sum(axis=1)
        Nw_5 = Nw[:, others].sum(axis=1)
        Nn_5 = Nn[:, others].sum(axis=1)
        m = valid
        X = np.column_stack([Nl[m, det], Nw[m, det], Nn[m, det],
                             Nl_5[m], Nw_5[m], Nn_5[m]])
        y = Dt_us[m, det]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        rms = np.sqrt(np.mean((y - pred) ** 2))
        rel = rms / np.mean(y)
        print(f"  {det:>3d} " + "".join(f"{c:>8.2f}" for c in coef) + f"  {rms:>7.1f}  ({rel*100:.1f}%)")
    print()

    fe.close()

print("Done.")
