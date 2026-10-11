"""numbers_v4.csv: numbers_v3.csv (unchanged) + precursor / quiet-period /
onset / power-spectrum numbers of the v4 science paper (GRB 221009A).

    python p221009_v4_numbers.py IN OUT

IN holds frozen copies of the group products (all retrieved from the farm runs
named in the 'note' column):
  numbers_v3.csv                         freeze_v3.md
  joint_fits.csv                         gbm.md 5, 5.1, 6.1 (gbm/runs/joint2)
  quietspec_fits.csv, quiet_group_signif.csv, quietjoint_*_total.csv   gbm.md 3b, 6.3
  onset_common.csv                       gbm.md 6.2 (gbm/runs/onsetc2)
  gbm_qpo_sens.csv, gbm_qpo12.csv, qpo12_results.csv   scout.md 7.1, 8.1, 8.2
  quiet_he_windows.csv, he_prec_index.csv              p221009_v4_he.py
Deterministic; no random numbers.
"""
import os
import sys

import numpy as np
import pandas as pd

IN, OUT = sys.argv[1], sys.argv[2]
v3 = pd.read_csv(f"{IN}/numbers_v3.csv", dtype=str, keep_default_na=False)
rows = []


def add(metric, value, unit, note, fmt="%.4g"):
    if isinstance(value, (float, np.floating, int, np.integer)):
        value = "nan" if not np.isfinite(value) else fmt % value
    rows.append(dict(metric=metric, value=value, unit=unit, note=note))


def ep_block(J, sfx, SRC):
    """Ep / alpha / C / narrowing rows for one joint-fit table; metric names get suffix sfx."""
    widths = {}
    for tag, t1, t2 in SEGS:
        sub = J[(J.tag == tag) & np.isclose(J.t1, t1) & np.isclose(J.t2, t2)]
        key = f"{t1:g}_{t2:g}" if tag == "tr" else tag
        for data, lab in (("GBM", "gbm"), ("GBM+HE", "joint")):
            nom = sub[(sub.data == data) & (sub["var"] == "nominal")]
            if nom.empty:
                continue
            n = nom.iloc[0]
            ep, lo, hi = n.Ep, n.Ep_lo1, n.Ep_hi1
            open_hi = not np.isfinite(hi) or ep >= 1.9e5
            if open_hi:
                add(f"prec_ep_{lab}_{key}{sfx}", "lower limit", "keV",
                    f"{data} {t1}..{t2} s: Ep unbounded above; 1sigma lower limit {lo:.0f}, 2sigma {n.Ep_lo2:.0f} keV. {SRC}")
                add(f"prec_ep_{lab}_{key}_lo1{sfx}", lo, "keV", f"1sigma (Delta S=1) lower limit. {SRC}")
                add(f"prec_ep_{lab}_{key}_lo2{sfx}", n.Ep_lo2, "keV", f"2sigma (Delta S=4) lower limit. {SRC}")
            else:
                var = sub[(sub.data == data) & sub["var"].isin(SYSV)]
                dev = (var.Ep - ep).to_numpy()
                sp = max(dev.max(), 0) if dev.size else 0.0
                sm = max(-dev.min(), 0) if dev.size else 0.0
                add(f"prec_ep_{lab}_{key}{sfx}", ep, "keV",
                    f"{data} {t1}..{t2} s best Ep; stat 1sigma [{lo:.0f}, {hi:.0f}]; "
                    + (f"sys +{sp:.0f}/-{abs(sm):.0f} = max shift over {'/'.join(v for v in SYSV if v in set(var['var']))} variants"
                       if dev.size else "sys: no variants run for this interval")
                    + (" (none: GBM-only fit unaffected by HE variants)" if data == "GBM" else "") + f". {SRC}", "%.0f")
                add(f"prec_ep_{lab}_{key}_stat_lo{sfx}", lo, "keV", "1sigma profile-likelihood interval, lower", "%.0f")
                add(f"prec_ep_{lab}_{key}_stat_hi{sfx}", hi, "keV", "1sigma profile-likelihood interval, upper", "%.0f")
                na = data == "GBM+HE" and not dev.size
                add(f"prec_ep_{lab}_{key}_sys_plus{sfx}", "n/a" if na else sp, "keV",
                    "max upward shift over the systematic variants" + (" (none run)" if na else ""), "%.0f")
                add(f"prec_ep_{lab}_{key}_sys_minus{sfx}", "n/a" if na else sm, "keV",
                    "max downward shift over the systematic variants" + (" (none run)" if na else ""), "%.0f")
                if np.isfinite(n.Ep_hi2):
                    add(f"prec_ep_{lab}_{key}_2sig{sfx}", f"{n.Ep_lo2:.0f}-{n.Ep_hi2:.0f}", "keV", "2sigma (Delta S=4) interval")
                widths[(key, lab)] = np.log10(hi / lo)
            add(f"prec_alpha_{lab}_{key}{sfx}", n.alpha, "", f"COMP low-energy index, {data} {t1}..{t2} s. {SRC}", "%.3f")
            if data == "GBM+HE":
                add(f"prec_C_{key}{sfx}", n.C_he, "", f"HE/GBM normalisation (free, 0.3-3) {t1}..{t2} s; chi2_HE {n.chi2_he:.1f}/"
                    f"{int(n.n_he_groups) - 1}" + ("; UNRELIABLE (>9 s tloc extrapolation, gbm.md 5 item 5)"
                                                     if tag == "axel5" else "") + f". {SRC}", "%.2f")
            n7 = sub[(sub.data == data) & (sub["var"] == "n7")]
            if not n7.empty and np.isfinite(n7.iloc[0].Ep_hi1) and n7.iloc[0].Ep < 1.9e5:
                add(f"prec_ep_{lab}_{key}_n7check{sfx}", n7.iloc[0].Ep, "keV",
                    "detector check n7+n8+b1 (not in sys; gbm.md 6.1: C drifts to 1.4 at >4.5 s)", "%.0f")
        if (key, "gbm") in widths and (key, "joint") in widths:
            r = 1 - widths[(key, "joint")] / widths[(key, "gbm")]
            add(f"prec_ep_narrowing_{key}{sfx}", 100 * r + 0.0, "%",
                f"1 - log10-width(1sigma, GBM+HE)/log10-width(GBM) for {t1}..{t2} s. {SRC}", "%.0f")
    mid = [k for k in ("1_2", "2_3", "3_4.5", "4.5_6.5", "6.5_8.576", "Ia", "axel3", "axel4")
           if (k, "gbm") in widths and (k, "joint") in widths]
    rr = np.array([100 * (1 - widths[(k, "joint")] / widths[(k, "gbm")]) for k in mid])
    add(f"prec_ep_narrowing_range{sfx}", f"{rr.min():.0f}-{rr.max():.0f}", "%",
        f"range of the 1sigma log-width reduction over intervals with Ep in 0.3-3 MeV ({', '.join(mid)}); "
        f"median {np.median(rr):.0f}%. {SRC}")
    ok = [k for k in ("1_2", "2_3", "3_4.5", "4.5_6.5", "6.5_8.576")]
    cs = J[(J.tag == "tr") & (J.data == "GBM+HE") & (J["var"] == "nominal") & (J.t1 >= 1) & (J.t2 <= 8.6)].C_he
    add(f"prec_C_range_1_8.6{sfx}", f"{cs.min():.2f}-{cs.max():.2f}", "", f"HE/GBM normalisation over the 1-8.6 s segments. {SRC}")


# ================================================================ time zero
T0S = "p221009_v4_ltt.py (gbm/runs/v4ltt1/v4/t0_offsets.txt)"
add("t0_paper_utc", "2022-10-09T13:17:00.000", "UTC", "paper T0 = HXMT MET 339945423.0; zero of the v2/v3 1-s bins and of the "
    "HE 1K event cache (p221009_timing_bkg.T0_MET); all v3 times and the v4 figures")
add("t0_gbm_trigger_utc", "2022-10-09T13:16:59.9875", "UTC", f"GBM trigger MET 687014224.98753 (gdt Fermi time). {T0S}")
add("t0_hxmt_trigger_utc", "2022-10-09T13:17:00.050", "UTC", "HXMT/HE trigger (HXMT MET 339945423.05); NOT used as a time zero "
    "in v3/v4 numbers, but gbm.md/resp.md 'HXMT T0 = GBM + 0.06 s' refers to it")
add("t0_offset_gbm_trigger", -0.0125, "s", f"GBM trigger UTC - paper T0. {T0S}", "%+.4f")
add("t0_ltt_gbm_hxmt", 0.0017, "s", f"(r_GBM - r_HXMT).n/c at T0 (separation 3147 km); GBM sees the wavefront first. {T0S}", "%+.4f")
add("t0_offset_gbm_to_paper", -0.0108, "s", f"t_paper = t_GBM - 0.0108 s (trigger -0.0125 + light travel +0.0017); used for "
    f"all GBM times in the v4 figures. {T0S}", "%+.4f")

# ================================================================ precursor Ep(t)
J = pd.read_csv(f"{IN}/joint_fits.csv")
J = J[J.model == "COMP"]
SEGS = [("tr", -1, 0), ("tr", 0, 1), ("tr", 1, 2), ("tr", 2, 3), ("tr", 3, 4.5), ("tr", 4.5, 6.5),
        ("tr", 6.5, 8.576), ("Ia", -0.003, 8.576), ("HEwin", -0.84, 10.06), ("axel1", -0.69, 0.23),
        ("axel2", 0.23, 1.0), ("axel3", 1.0, 2.72), ("axel4", 2.72, 7.0), ("axel5", 7.0, 19.95)]
SYSV = ["prior", "prior11", "g0.95", "g1.05", "b_lo", "b_hi"]
SRC = ("p221009_gbm_joint.py, gbm/runs/joint2/joint_fits.csv; gbm.md 5/5.1/6.1; COMP, GBM n4+n8 (20-900 keV)"
       " + b1 (0.4-38 MeV), HE CsI ch20-255 tloc bkg, C free. TIME ZERO: interval bounds are GBM-T0 seconds (GBM trigger "
       "13:16:59.9875 UTC); paper T0 (13:17:00.000) bounds = GBM - 0.0108 s (prec_seg_* rows). The HE events were "
       "windowed at GBM - 0.06 s (gbm.md 'HXMT T0 = 13:17:00.050' convention applied to an HE time axis that is "
       "already paper-T0), i.e. 0.049 s earlier than simultaneous; values kept as computed")
SRC_NEW = ("p221009_gbm_joint.py with DT_HE = 0.0108 s, gbm/runs/joint3/joint/joint_fits.csv (= blink/data/p221009/gbm/"
           "joint_dt0108/joint_fits.csv, md5 d3531878); gbm.md 7.1/7.2; COMP, n4+n8+b1, HE CsI ch20-255 tloc, C free; "
           "sys = max shift over the gbm.md 7.2 variants present for the interval (n7 excluded). "
           "Interval bounds GBM-T0 seconds; paper-T0 bounds in prec_seg_*_T0paper")
for tag, t1, t2 in SEGS:
    key = f"{t1:g}_{t2:g}" if tag == "tr" else tag
    add(f"prec_seg_{key}_T0paper", f"{t1 - 0.0108:.3f}..{t2 - 0.0108:.3f}", "s",
        f"interval {t1:g}..{t2:g} s (GBM T0) in paper T0 (t_GBM - 0.0108 s); HE window used in the joint fit "
        f"{t1 - 0.06:.3f}..{t2 - 0.06:.3f} s (paper T0; joint2, DT_HE=0.06, superseded); the joint3 rerun (*_T0paper Ep rows) "
        f"uses {t1 - 0.0108:.3f}..{t2 - 0.0108:.3f} s")
JNEW = f"{IN}/joint_fits_T0paper.csv"
NEW = os.path.exists(JNEW)
ep_block(J, "", SRC + ("; DT_HE=0.06, SUPERSEDED (作废): use the *_T0paper rows (gbm.md 7, joint fit rerun with "
                       "DT_HE = 0.0108 s, HE window simultaneous with GBM); kept for comparison only" if NEW else ""))
if NEW:
    JN = pd.read_csv(JNEW)
    ep_block(JN[JN.model == "COMP"], "_T0paper", SRC_NEW)
add("prec_axelsson_ep", "15+-3 / 7+-2 / 6+-2 / 1.3(+0.6/-1.2) / 0.33(+0.02/-0.04)", "MeV",
    "Axelsson+2025 ApJS 277, 24 table cparam, COMP, GBM n4/n7/b1+LLE+LAT, segments -0.69/0.23/1.00/2.72/7.00/19.95 s "
    "(GBM T0); precursor_lit.md 1.1")

# ---------------------------------------------------------------- HE-only precursor index
P = pd.read_csv(f"{IN}/he_prec_index.csv")
HSRC = "p221009_v4_he.py (resp DRM precursor_per, tloc bkg, ch20-255, PL 1sigma Delta chi2=1); cf. resp.md 4.1-4.2"
for _, r in P.iterrows():
    sfx = "_T0paper" if (r.wset == "T0paper" and r.tag != "resp_check") else ""
    if r.tag == "resp_check":
        key = f"hxmt_{r.t1_hxmt:g}_{r.t2_hxmt:g}"
    elif r.tag == "tr":
        key = f"{r.t1_gbm:g}_{r.t2_gbm:g}"
    else:
        key = r.tag
    gi = r["index"]
    clip = " (interval clipped at the +-1.2 profile grid)" if (np.isclose(r.index_lo, gi - 1.2, atol=0.006)
                                                              or np.isclose(r.index_hi, gi + 1.2, atol=0.006)) else ""
    wnote = {"jointfit": "HE window = GBM interval - 0.06 s (as in the joint fit)",
             "T0paper": "HE window = GBM interval - 0.0108 s (simultaneous)"}[r.wset] if r.tag != "resp_check" else "resp.md segment"
    add(f"prec_he_index_{key}{sfx}", gi, "",
        f"HE-only 0.17-3.1 MeV PL photon index, paper-T0 window {r.t1_hxmt:g}..{r.t2_hxmt:g} s ({wnote}); 1sigma [{r.index_lo:.3f}, "
        f"{r.index_hi:.3f}]{clip}; net {r.net:.0f} counts; chi2 {r.chi2:.1f}/{int(r.dof)}. {HSRC}", "%.3f")
add("prec_he_index_0_9_sys", 0.10, "", "systematic on the HE index (EC offset +0.08/-0.03, gain +-5% +-0.05, "
    "bkg tlin2 +0.04; resp.md 4.1)", "%.2f")
add("prec_he_lesage_disfavour", "dchi2 +9.7 (0-9 s) / +7.5 (-1-9 s); ~3sigma stat, ~2sigma incl. sys", "",
    "Lesage+2023 Ia COMP (alpha -1.69, Ep 3.98 MeV) shape in HE vs free PL; resp.md 4.1, resp/out/prec_summary.txt")
add("prec_he_gbm_norm_xcal", 1.12, "", "HE/GBM normalisation with GBM Ia shape fixed (resp.md 3.2); "
    "absolute-area systematic +-20% (resp.md 3.3)", "%.2f")

# ================================================================ quiet period
Q = pd.read_csv(f"{IN}/quiet_he_windows.csv")
CONV = 7.6e8
QSRC = ("p221009_v4_he.py: CsI ch20-255, B = a + s*LargeEvt, s from source-free low-gain pass d1008, a from "
        "T0-420..-10 s; sys 15 counts/s (resp.md 7.3-7.4)")
nets = {}
for wn, lab in (("II", "II"), ("20-175", "20_175"), ("115-172", "115_172"), ("II_T0paper", "II_T0paper")):
    q = Q[(Q.ref == "d1008") & (Q.window == wn)].iloc[0]
    alt = Q[(Q.window == wn) & (Q.ref != "d1008")]
    tot = np.hypot(q.stat, q.sys)
    nets[wn] = (q.net_rate, tot)
    add(f"quiet_he_net_{lab}", q.net_rate, "counts/s",
        f"HE net {q.t1:g}..{q.t2:g} s (paper T0 13:17:00.000 = HE cache zero" + ("; II = GBM II - 0.06 s, 0.049 s earlier "
        "than simultaneous, see quiet_he_net_II_T0paper" if wn == "II" else "") + f") = {q.net_rate:+.1f} +- {q.stat:.1f} (stat) +- {q.sys:.0f} (sys); slope "
        f"{q.slope:.3f}; other references " + ", ".join(f"{a.ref} {a.net_rate:+.1f}" for _, a in alt.iterrows())
        + f". {QSRC}", "%+.1f")
    add(f"quiet_he_net_{lab}_stat", q.stat, "counts/s", "statistical (counts + pre-burst offset)", "%.1f")
    ul = (q.net_rate + 3 * tot) / CONV
    add(f"quiet_he_ul3_flux_{lab}", ul, "erg cm-2 s-1",
        f"3sigma UL on time-averaged 0.3-3 MeV flux = (net + 3 sqrt(stat^2+sys^2))/7.6e8 (resp.md 4.4, 7.4; HE area "
        f"scale, x1.12 for the GBM scale). Supersedes resp.md 4.4 2.9e-8 (no sys). {QSRC}", "%.2g")
q = Q[(Q.ref == "d1008") & (Q.window == "-100..-10")].iloc[0]
add("quiet_he_null_pre", q.net_rate, "counts/s", f"same model on pre-burst T0-100..-10 s (should be 0). {QSRC}", "%+.1f")

S = pd.read_csv(f"{IN}/quietspec_fits.csv")
GSRC = "p221009_gbm_quietspec.py, gbm/runs/qspec1/quietspec_fits.csv; gbm.md 3b; n4+n8 (20-500 keV)+b1 (0.4-38 MeV), PG-stat"
for (w, bkg, lab) in (("II", "orb30", "II"), ("HEquiet", "orb30", "20_170")):
    for model in ("PL", "COMP", "Band"):
        s = S[(S.window == w) & (S.bkg == bkg) & (S.model == model) & (S.dets == "n4+n8+b1")]
        if s.empty:
            continue
        s = s.iloc[0]
        if model == "PL":
            par = f"Gamma {s['index']:.3f} (-{s.index_elo:.3f}/+{s.index_ehi:.3f})"
        elif model == "COMP":
            par = f"Ep {s.Epeak:.0f} keV, alpha {s['index']:.2f}"
        else:
            par = f"Ep {s.Epeak:.0f} keV, alpha {s.alpha:.2f}, beta {s.beta:.2f}"
        add(f"quiet_gbm_{lab}_{model}", par, "", f"{w} {s.t1}..{s.t2} s (GBM T0), {bkg} bkg; PG {s.stat:.1f}/{int(s.dof)}; "
            f"A(100 keV) {s.A:.4g} ph cm-2 s-1 keV-1. {GSRC}")
        add(f"quiet_gbm_{lab}_{model}_eflux_10_1000", s.eflux_10_1000, "erg cm-2 s-1", f"{GSRC}", "%.3g")
        add(f"quiet_gbm_{lab}_{model}_eflux_200_3000", s.eflux_200_3000, "erg cm-2 s-1",
            "model value (extrapolation for PL). " + GSRC, "%.3g")
s15 = S[(S.window == "II") & (S.bkg == "orb15") & (S.model == "PL") & (S.dets == "n4+n8+b1")]
if not s15.empty:
    add("quiet_gbm_II_PL_orb15", f"Gamma {s15.iloc[0]['index']:.3f}", "", f"orb15 background. {GSRC}")
G = pd.read_csv(f"{IN}/quiet_group_signif.csv")
g = G[(G.window.str.startswith("121")) & (G.group == "NaI n4+n8 27-300 keV")]
if not g.empty:
    add("quiet_gbm_II_nai_signif", f"{g.signif.min():.1f}-{g.signif.max():.1f}", "sigma",
        f"NaI n4+n8 27-300 keV significance in II over backgrounds {'/'.join(g.bkg)}. "
        "p221009_gbm_quiet.py, quiet_group_signif.csv; gbm.md 3b")
# PL extension prediction (fold of the GBM II fits through the theta=12.7 deg HE DRM: resp adjorb_ana,
# resp/out/adjorb_summary.txt 'predicted HE ... counts (scaled to II)': PL 2233 (orb30) / 2841 (orb15), COMP 687, Band 802)
DUR = 164.168 - 121.159
net, sig = nets["II"]
for model, cnt in (("PL_orb30", 2233), ("PL_orb15", 2841), ("COMP", 687), ("Band", 802)):
    pred = cnt / DUR * 1.12
    z0 = (pred - net) / sig
    za = (pred - net) / np.hypot(sig, 0.2 * pred)
    add(f"quiet_pred_he_{model}", pred, "counts/s",
        f"HE rate predicted by GBM II {model} x HE/GBM area 1.12 (resp.md 7.4; counts from resp/out/adjorb_summary.txt); "
        f"deviation from measured {net:+.1f} +- {sig:.1f}: {z0:.1f} sigma, {za:.1f} sigma incl. +-20% area", "%.0f")
    add(f"quiet_pred_he_{model}_sigma", z0, "sigma", "(pred - net)/sqrt(stat^2+sys^2)", "%.1f")
    add(f"quiet_pred_he_{model}_sigma_area", za, "sigma", "same, +-20% area added in quadrature", "%.1f")
net2, sig2 = nets["II_T0paper"]
for model, cnt in (("PL_orb30", 2233), ("PL_orb15", 2841)):
    pred = cnt / DUR * 1.12
    add(f"quiet_pred_he_{model}_sigma_T0paper", (pred - net2) / sig2, "sigma",
        f"as quiet_pred_he_{model}_sigma, HE window simultaneous with GBM II (paper T0 121.208-164.217 s); "
        f"net {net2:+.1f} +- {sig2:.1f}; incl. +-20% area {(pred - net2) / np.hypot(sig2, 0.2 * pred):.1f} sigma", "%.1f")
add("quiet_old_joint_preference", "Delta S(PL-CPL) 1.5-4.4 (<=2.1 sigma)", "",
    "superseded: gbm.md 6.3 joint fit with the HE proxy background (166 counts/10 s sys); resp.md 7.4 tightens it")

# ================================================================ onset
O = pd.read_csv(f"{IN}/onset_common.csv").set_index("band")
OT = pd.read_csv(f"{IN}/onset_common_T0paper.csv").set_index("band")
OTS = ("p221009_v4_onset.py (= onset2 with the GBM shift as argument; seed 1; the -0.062 run reproduces "
       "onset_common.csv byte for byte), gbm/runs/v4onset/v4/onset_common_T0paper.csv")
OSRC = ("p221009_gbm_onset2.py, gbm/runs/onsetc2/onset_common.csv; gbm.md 6.2; 145-172 s linear bkg, 64 ms. TIME ZERO: "
        "HE values are on the paper axis (HE event cache t = Time - HXMT MET 339945423.0 = 13:17:00.000 UTC, i.e. the v3 "
        "pipeline time, NOT the HXMT trigger); GBM values used t_GBM - 0.062 s (gbm 'HXMT T0' convention) and are "
        "therefore early on the paper axis; paper-axis values recomputed from the events in *_T0paper rows")
for b, lab in (("HE 0.17-3.1 MeV", "he"), ("GBM NaI 100-200 keV", "gbm_nai100_200"), ("GBM BGO 300-700 keV", "gbm_bgo300_700")):
    r = O.loc[b]
    add(f"onset_sig3_{lab}", r.sig3, "s", f"{b}: start of first 3 consecutive 64-ms bins >3 sigma, +- {r.sig3_err:.2f} s "
        f"(100 Poisson resamplings). {OSRC}", "%.2f")
    add(f"onset_r25_{lab}", r.r25, "s", f"{b}: 25% of the 182-183 s reference rate, +- {r.r25_err:.2f}. {OSRC}", "%.2f")
    add(f"onset_r50_{lab}", r.r50, "s", f"{b}: 50% of reference, +- {r.r50_err:.2f}. {OSRC}", "%.2f")
    rt = OT.loc[b]
    for crit in ("sig3", "r25", "r50", "bb"):
        add(f"onset_{crit}_{lab}_T0paper", rt[crit], "s",
            f"{b} {crit} on the paper axis (T0 = 13:17:00.000 UTC), recomputed from the events: "
            f"+- {rt[crit + '_err']:.2f} s. HE events relative to HXMT MET 339945423.0 (resp cache, = v3 axis); GBM "
            f"events t_GBM - 0.0108 s (trigger 13:16:59.9875 + light travel; p221009_v4_ltt.py). {OTS}", "%.2f")
    add(f"onset_bb_{lab}", r.bb, "s", f"{b}: first Bayesian-block change point, +- {r.bb_err:.1f} (unstable). {OSRC}", "%.2f")

# ================================================================ power spectra
K = pd.read_csv(f"{IN}/gbm_qpo_sens.csv")
KS = "p221009_sci_gbm.py, scout/gbm_qpo_sens.csv; scout.md 8.1; 3sigma global detection threshold, Q~4.5, fractional rms"
for stream, lab in (("HE (this run)", "he"), ("GBM NaI n8+n4+n7+n3 8-900 keV", "gbm_nai"),
                    ("GBM BGO b0+b1 0.2-3 MeV", "gbm_bgo")):
    r = K[(K.seg == "stack_bright") & (K.stream == stream)].iloc[0]
    for f in (3, 10, 30, 100, 300, 1000):
        add(f"pds_rms3_stack_{lab}_{f}Hz", 100 * r[f"rms3_{f}Hz"], "%",
            f"{stream}, 191 stacked bright real seconds; min global p {r.min_pglob:.2f}. {KS}", "%.1f")
Q12 = pd.read_csv(f"{IN}/gbm_qpo12.csv")
QS = "p221009_sci_gbm.py, scout/gbm_qpo12.csv; scout.md 8.2; Vaughan 2010, 2000 sims, seed 20221009"
for curve, lo, lab in (("HE v3", 277, "he_v3_277_508"), ("GBM NaI n8+n4+n7+n3 8-900 keV", 280, "gbm_nai_280_508"),
                       ("GBM BGO b0+b1 0.2-3 MeV", 280, "gbm_bgo_280_508"), ("HE obs", 330, "he_obs_330_440"),
                       ("GECAM-C HG", 330, "gecam_hg_330_440")):
    r = Q12[(Q12.curve == curve) & (Q12.lo == lo)]
    if curve in ("HE obs", "GECAM-C HG"):
        r = r[r.hi == 440]
    r = r.iloc[0]
    add(f"qpo12_p_{lab}", r.p_max, "", f"{curve} {r.lo}-{r.hi} s: p (max in 12.3 s band) {r.p_max:.3f}, p(band sum) "
        f"{r.p_band:.2f}, max 2I/S {r.Tc:.2f} at {1 / r.f_best:.1f} s; 3sigma rms (Q=4) {100 * r.rms3_Q4:.0f}%. {QS}", "%.3f")
    add(f"qpo12_rms3_{lab}", 100 * r.rms3_Q4, "%", f"3sigma detectable fractional rms, Q=4. {QS}", "%.0f")

# ================================================================ round 4 (referee 2): shared C, quiet v2
SCE = pd.read_csv(f"{IN}/sharedC_ep.csv")
SCT = pd.read_csv(f"{IN}/sharedC_stot.csv")
SCS = ("p221009_v4_sharedC.py from gbm/runs/joint4 (var C0.80..C1.50, DT_HE 0.0108; gbm.md 8.2): S_tot(C) = sum_i "
       "min_Ep S_i, parabola on dS<=15; P_i(Ep) = min_C [S_i(Ep,C) + sum_j!=i min S_j(C)], 1sigma at dP=1 "
       "(log-interpolated on the 22% Ep grid). No systematic variants were run with a shared C")
c0 = SCE.C_shared.iloc[0]
add("prec_C_shared", c0, "", f"HE/GBM normalisation shared by the 7 intervals -1..8.576 s, +- {SCE.C_sig.iloc[0]:.2f}; "
    "S_tot - min at C=0.8..1.5: " + ", ".join(f"{c:.1f}:{v:.1f}" for c, v in zip(SCT.C, SCT.dS)) + f". {SCS}", "%.2f")
add("prec_C_shared_err", SCE.C_sig.iloc[0], "", f"1sigma of prec_C_shared (parabola curvature). {SCS}", "%.2f")
JN2 = pd.read_csv(f"{IN}/joint_fits_T0paper.csv")
for _, r in SCE.iterrows():
    key = f"{r.t1:g}_{r.t2:g}"
    if pd.isna(r.Ep_hi1):
        add(f"prec_ep_sharedC_{key}", "lower limit", "keV", f"GBM+HE, shared C={c0:.2f}, {r.t1:g}..{r.t2:g} s (GBM T0): "
            f"Ep unbounded above on the 60 MeV grid; 1sigma lower limit {r.Ep_lo1:.0f}, 2sigma {r.Ep_lo2:.0f} keV. {SCS}")
        add(f"prec_ep_sharedC_{key}_lo1", r.Ep_lo1, "keV", f"1sigma lower limit. {SCS}", "%.0f")
        continue
    add(f"prec_ep_sharedC_{key}", r.Ep, "keV", f"MAIN RESULT: GBM+HE COMP Ep with a shared C={c0:.2f}, {r.t1:g}..{r.t2:g} s "
        f"(GBM T0; paper T0 = -0.0108 s); 1sigma [{r.Ep_lo1:.0f}, {r.Ep_hi1:.0f}], 2sigma [{r.Ep_lo2:.0f}, "
        + (f"{r.Ep_hi2:.0f}" if pd.notna(r.Ep_hi2) else "open") + f"]; interval's own best C {r.C_own:.1f}. {SCS}", "%.0f")
    add(f"prec_ep_sharedC_{key}_stat_lo", r.Ep_lo1, "keV", "1sigma lower", "%.0f")
    add(f"prec_ep_sharedC_{key}_stat_hi", r.Ep_hi1, "keV", "1sigma upper", "%.0f")
    g = JN2[(JN2.tag == "tr") & np.isclose(JN2.t1, r.t1) & np.isclose(JN2.t2, r.t2) & (JN2.model == "COMP")
            & (JN2.data == "GBM") & (JN2["var"] == "nominal")].iloc[0]
    if np.isfinite(g.Ep_hi1):
        red = 1 - np.log10(r.Ep_hi1 / r.Ep_lo1) / np.log10(g.Ep_hi1 / g.Ep_lo1)
        add(f"prec_ep_sharedC_narrowing_{key}", 100 * red, "%", "1 - log10-width(1sigma, shared C)/log10-width(GBM alone). "
            + SCS, "%.0f")

# quiet period v2 (resp.md 8.2): bias correction +18.5 (regression of the blind-extrapolation residual on dLE at the
# quiet-period dLE = +640; resp also gives the quiet-like stratum mean -12.6), systematic 24 (stratum rms)
QB, QSY = 18.5, 24.0
QV2 = ("resp.md 8.2 / resp/out/m3_summary.txt: blind LargeEvt extrapolations (156 tests, 5 source-free stretches); "
       "bias correction +18.5 counts/s (residual-vs-dLE regression at the quiet-period dLE=+640; stratum mean would "
       "give +12.6), sys 24 counts/s (quiet-like stratum rms); replaces the +-15 of resp.md 7.3. LargeEvt source "
       "response could lower a hard-spectrum net rate by <=8% (95%: 13%), not applied")
for wn, lab in (("II", "II"), ("II_T0paper", "II_T0paper"), ("20-175", "20_175"), ("115-172", "115_172")):
    q = Q[(Q.ref == "d1008") & (Q.window == wn)].iloc[0]
    nv = q.net_rate + QB
    tot = np.hypot(q.stat, QSY)
    add(f"quiet_he_net_{lab}_v2", nv, "counts/s", f"HE net {q.t1:g}..{q.t2:g} s (paper T0) = {q.net_rate:+.1f} {QB:+.1f} = "
        f"{nv:+.1f} +- {q.stat:.1f} (stat) +- {QSY:.0f} (sys). {QV2}", "%+.1f")
    add(f"quiet_he_ul3_flux_{lab}_v2", (nv + 3 * tot) / CONV, "erg cm-2 s-1",
        f"3sigma UL, 0.3-3 MeV, HE area scale = (net + 3 sqrt(stat^2+24^2))/7.6e8; /1.12 or /0.81 for the GBM scale; "
        f"x1.08 if the LargeEvt response is applied. {QV2}", "%.2g")
    add(f"quiet_he_ul2_flux_{lab}_v2", (nv + 2 * tot) / CONV, "erg cm-2 s-1", f"2sigma UL, same definition. {QV2}", "%.2g")
QPR = pd.read_csv(f"{IN}/quietpred.csv")
q = Q[(Q.ref == "d1008") & (Q.window == "II_T0paper")].iloc[0]
nv, tot = q.net_rate + QB, np.hypot(q.stat, QSY)
for (bkg, model, comp), lab in ((("orb30", "PL", "PL"), "PL_orb30"), (("orb15", "PL", "PL"), "PL_orb15"),
                                (("orb30", "L_BBCOMP", "L_BBCOMP"), "LesageII_orb30"),
                                (("orb15", "L_BBCOMP", "L_BBCOMP"), "LesageII_orb15")):
    r = QPR[(QPR.bkg == bkg) & (QPR.model == model) & (QPR.component == comp)].iloc[0]
    p1, plo, phi = r.he_pred_rate_C1, r.he_pred_lo1, r.he_pred_hi1
    for Cr in (0.81, 1.0, 1.12):
        z = (Cr * p1 - nv) / tot
        zg = (Cr * p1 - nv) / np.hypot(tot, Cr * 0.5 * (phi - plo))
        add(f"quiet_pred_v2_{lab}_C{Cr:g}", z, "sigma",
            f"GBM II {model} ({bkg}) folded into HE (gbm.md 8.1, quiet/quietpred.csv: {p1:.0f} [{plo:.0f}, {phi:.0f}] "
            f"counts/s at C=1) x C={Cr:g} vs measured {nv:+.1f} +- {tot:.1f} (II, paper-T0 window): {z:.1f} sigma; "
            f"{zg:.1f} sigma incl. the GBM-fit error of the prediction. {QV2}", "%.1f")

# ================================================================ round 5 (referee 2 round 2): Fig Q R4 band, R3 panel (d)
import re  # noqa: E402
R4S = ("resp.md 9.2 / resp/out/r4_summary.txt (p221009_resp_r4.py): blind-extrapolation bias regression "
       "resid = a + b dLE + c dLE/dt (scatter 21.4) and dLE-only form (scatter 23.6), evaluated per segment; GBM tail = "
       "gbm prec4 time-resolved COMP fits (n4+n8+b1) folded into HE x1.12. Fig Q(a) band: mean of the two forms +- 21.4")
seg_ = {}
for line in open(f"{IN}/r4_summary.txt"):
    m = re.search(r"\(scatter ([\d.]+)\); dLE only", line)
    if m:
        r4scat = float(m.group(1))
    m = re.match(r"\s+([\d.]+)-([\d.]+)\s*: net\s+([+-][\d.]+) \+-\s+([\d.]+);.*bias\s+([+-][\d.]+) "
                 r"\(dLE-only\s+([+-][\d.]+)\).*GBM tail\s+([\d.]+)", line)
    if m:
        seg_[(m.group(1), m.group(2))] = [float(x) for x in m.groups()[2:]]
add("quiet_he_r4_bias_scatter", r4scat, "counts/s", f"scatter of the blind-test residuals about the dLE+dLE/dt regression; "
    f"half-width of the Fig Q(a) band. {R4S}", "%.1f")
for (s0, s1), (nt, st, b2, b1, tl) in seg_.items():
    lab = f"{float(s0):g}_{float(s1):g}".replace(".", "p")
    mid = 0.5 * (b2 + b1)
    add(f"quiet_he_r4_net_{lab}", nt, "counts/s", f"HE net {s0}..{s1} s, no bias correction, +- {st:.1f} (stat). {R4S}",
        "%+.1f")
    add(f"quiet_he_r4_bias_{lab}", mid, "counts/s", f"expected bias {s0}..{s1} s: dLE+dLE/dt {b2:+.1f}, dLE-only "
        f"{b1:+.1f}, mean {mid:+.1f} (Fig Q(a) band centre); net - bias = {nt - b2:+.1f} / {nt - b1:+.1f}. {R4S}", "%+.1f")
    add(f"quiet_he_r4_gbmtail_{lab}", tl, "counts/s", f"GBM precursor-tail prediction in HE, {s0}..{s1} s. {R4S}", "%.1f")
QF = pd.read_csv(f"{IN}/qfold_II.csv").set_index("model")
QV3 = pd.read_csv(f"{IN}/quietpred_v3.csv")
QV3 = QV3[(QV3.C == 1.0) & (QV3.suppression == 1.0)]
RS = ("Fig Q(d): HE prediction in II at C=1.12 (C range 0.96-1.25, resp.md 9.1), source suppression 0.93 for PL/Lesage "
      "(gbm.md 9.1), 1 for COMP/Band; DRM >3 MeV rows x s3. PL/Lesage from gbm/runs/qpred3 quietpred_v3.csv, COMP/Band "
      "from p221009_v4_qfold.py (gbm/runs/v4qfold qfold_II.csv; reproduces quietpred Lesage 108.4/172.0)")
for lab, src, f in (("COMP_orb30", ("qf", "gbm_COMP_orb30"), 1.0), ("Band_orb30", ("qf", "gbm_Band_orb30"), 1.0),
                    ("PL_orb30", ("orb30", "PL"), 0.93), ("PL_orb15", ("orb15", "PL"), 0.93),
                    ("LesageII_orb30", ("orb30", "L_BBCOMP"), 0.93), ("LesageII_orb15", ("orb15", "L_BBCOMP"), 0.93)):
    if src[0] == "qf":
        p = {k: QF.loc[src[1], f"he_pred_C1_s3_{k:g}"] for k in (1.0, 0.7, 1.3, 0.0)}
    else:
        s_ = QV3[(QV3.bkg == src[0]) & (QV3.model == src[1])].set_index("drm_gt3MeV_scale").he_pred_C1_noSupp
        p = {k: s_.loc[k] for k in (1.0, 0.7, 1.3, 0.0)}
    lo_ = min(p[s] * c * f for s in (0.7, 1.3) for c in (0.96, 1.25))
    hi_ = max(p[s] * c * f for s in (0.7, 1.3) for c in (0.96, 1.25))
    add(f"quiet_pred_r5_{lab}", p[1.0] * 1.12 * f, "counts/s",
        f"C=0.96..1.25: {p[1.0] * 0.96 * f:.0f}..{p[1.0] * 1.25 * f:.0f}; with s3=0.7..1.3: {lo_:.0f}..{hi_:.0f}; "
        f"s3=0 (no >3 MeV deposits): {p[0.0] * 1.12 * f:.0f}; measured {nv:+.1f} +- {tot:.1f}. {RS}", "%.1f")

# ================================================================ round 6 (referee 2 round 2, R1): shared C from joint5+joint6
# (gbm.md 9.4; gbm/runs/r1 from p221009_gbm_r1.py, frozen as j6_*): finer C grid, systematics refitted with C shared
J6 = ("gbm.md 9.4: gbm/runs/r1 (p221009_gbm_sharedC.py, p221009_gbm_r1.py; joint5+joint6, DT_HE 0.0108, C grid "
      "0.85..1.45 + joint4 0.80..1.50); Ep 1sigma from P_i(Ep) = min_C [S_i + sum_j!=i min S_j], i.e. C profiled")
J6C = pd.read_csv(f"{IN}/j6_sharedC_C.csv").set_index("sys")
J6E = pd.read_csv(f"{IN}/j6_sharedC_ep.csv")
J6F = pd.read_csv(f"{IN}/j6_fixedC_ep.csv")
cj6 = J6C.loc["nominal"]
add("prec_C_shared_j6", cj6.C_shared, "", f"HE/GBM normalisation shared by the 7 intervals -1..8.576 s, +- {cj6.C_err:.3f}; "
    f"S_tot(shared) - S_tot(C free per interval) = {cj6.dS_shared_vs_free:.2f} for 6 fewer parameters. "
    "Variants: " + ", ".join(f"{k} {v:.3f}" for k, v in J6C.C_shared.items() if k != "nominal") + f". {J6}", "%.3f")
add("prec_C_shared_j6_err", cj6.C_err, "", f"1sigma of prec_C_shared_j6. {J6}", "%.3f")
FAM = (("tilt-0.06", "tilt-0.03", "tilt+0.03", "tilt+0.06"), ("g0.95", "g1.05"), ("b_lo", "b_hi"))


def sys_comb(T, lab, ref):
    """Instrumental systematic of Ep (gbm.md 9.4 (1)): per family (|delta|<=0.06, gain +-5%, EC b_lo/b_hi) the largest
    upward and downward offset from the nominal, families added in quadrature."""
    up, dn = [], []
    for fam in FAM:
        d = [T[(T.sys == s) & (T.lab == lab)].Ep.iloc[0] / ref - 1 for s in fam]
        up.append(max(max(d), 0.0)); dn.append(min(min(d), 0.0))
    return np.sqrt(np.sum(np.square(up))), -np.sqrt(np.sum(np.square(dn)))


def max_off(T, lab, ref):
    d = [T[(T.sys == s) & (T.lab == lab)].Ep.iloc[0] / ref - 1 for fam in FAM for s in fam]
    return max(max(d), 0.0), min(min(d), 0.0)


F1 = J6F[np.isclose(J6F.C, 1.0)]
for _, r in J6E[J6E.sys == "nominal"].sort_values("t1").iterrows():
    key = f"{r.t1:g}_{r.t2:g}"
    f1 = F1[(F1.sys == "nominal") & (F1.lab == r.lab)].iloc[0]
    if pd.isna(r.Ep_hi1):
        add(f"prec_ep_sharedC_j6_{key}", "lower limit", "keV", f"GBM+HE, shared C={cj6.C_shared:.3f}, {r.t1:g}..{r.t2:g} s "
            f"(GBM T0): Ep unbounded above (best ~{r.Ep / 1e3:.0f} MeV); 1sigma lower limit {r.Ep_lo1:.0f} keV; "
            f"variants: " + ", ".join(f"{s} {v:.0f}" for s, v in J6E[J6E.lab == r.lab].set_index("sys").Ep_lo1.items()
                                     if s != "nominal") + f". {J6}")
        add(f"prec_ep_sharedC_j6_{key}_lo1", r.Ep_lo1, "keV", f"1sigma lower limit. {J6}", "%.0f")
    else:
        su, sd = sys_comb(J6E, r.lab, r.Ep)
        add(f"prec_ep_sharedC_j6_{key}", r.Ep, "keV", f"MAIN RESULT (supersedes prec_ep_sharedC_{key}): GBM+HE COMP Ep, shared "
            f"C={cj6.C_shared:.3f}, {r.t1:g}..{r.t2:g} s (GBM T0; paper T0 = -0.0108 s); 1sigma stat (C profiled) "
            f"[{r.Ep_lo1:.0f}, {r.Ep_hi1:.0f}]; instrumental sys +{100 * su:.0f}/{100 * sd:.0f}%. {J6}", "%.0f")
        add(f"prec_ep_sharedC_j6_{key}_stat_lo", r.Ep_lo1, "keV", "1sigma lower (stat, C profiled)", "%.0f")
        add(f"prec_ep_sharedC_j6_{key}_stat_hi", r.Ep_hi1, "keV", "1sigma upper (stat, C profiled)", "%.0f")
        add(f"prec_ep_sharedC_j6_{key}_sys_up", 100 * su, "%", "instrumental sys, upward: per family (|delta|<=0.06, "
            "gain +-5%, EC b_lo/b_hi) the largest offset, families in quadrature; delta=+-0.1 not included. " + J6, "%.0f")
        add(f"prec_ep_sharedC_j6_{key}_sys_dn", 100 * sd, "%", f"instrumental sys, downward (same definition). {J6}", "%.0f")
    if pd.isna(f1.Ep_hi1) or f1.Ep >= 1.9e5:
        add(f"prec_ep_C1_{key}", "lower limit", "keV", f"GBM+HE, C fixed at 1.00 (independent calibration), {r.t1:g}..{r.t2:g} s: "
            f"Ep unbounded above; 1sigma lower limit {f1.Ep_lo1:.0f} keV; variants "
            f"{F1[F1.lab == r.lab].Ep_lo1.min():.0f}..{F1[F1.lab == r.lab].Ep_lo1.max():.0f}. {J6}")
        add(f"prec_ep_C1_{key}_lo1", f1.Ep_lo1, "keV", f"1sigma lower limit, C = 1.00. {J6}", "%.0f")
    else:
        mu, md = max_off(F1, r.lab, f1.Ep)
        add(f"prec_ep_C1_{key}", f1.Ep, "keV", f"GBM+HE, C fixed at 1.00 (independent calibration; upper end of the "
            f"normalisation systematic), {r.t1:g}..{r.t2:g} s: 1sigma [{f1.Ep_lo1:.0f}, {f1.Ep_hi1:.0f}]; instrumental "
            f"variants at C=1 (largest offset, gbm.md 9.4 (2)) +{100 * mu:.0f}/{100 * md:.0f}%; "
            + (f"Ep(C=1)/Ep(shared) = {f1.Ep / r.Ep:.2f}. " if np.isfinite(r.Ep_hi1) else "") + J6, "%.0f")
        add(f"prec_ep_C1_{key}_stat_lo", f1.Ep_lo1, "keV", "1sigma lower, C = 1.00", "%.0f")
        add(f"prec_ep_C1_{key}_stat_hi", f1.Ep_hi1, "keV", "1sigma upper, C = 1.00", "%.0f")
CDS = open(f"{IN}/j6_cdelta_summary.txt").read()
d0 = re.search(r"\[delta = 0\].*?dS\(C=1\.00\) = ([\d.]+)\s+dS\(C=1\.12\) = ([\d.]+)", CDS)
d6 = re.search(r"\[\|delta\| <= 0\.06\].*?dS\(C=1\.00\) = ([\d.]+)", CDS)
dg = re.search(r"delta profile.*best ([+-]?[\d.]+), 1-sigma \[([+-]?[\d.]+), ([+-]?[\d.]+)\]", CDS)
ds1, ds112, ds1d = float(d0.group(1)), float(d0.group(2)), float(d6.group(1))
add("prec_C_tension_dS_C1", ds1, "", f"S_tot(C=1.00) - min, delta = 0 ({ds1d:.1f} with |delta|<=0.06 free); "
    f"S_tot(C=1.12) - min = {ds112:.1f}. j6_cdelta_summary.txt. {J6}", "%.1f")
add("prec_C_tension_sigma_C1_stat", np.sqrt(ds1), "sigma", f"C = 1.00 vs the precursor, statistics only: sqrt({ds1:.1f}); "
    f"{np.sqrt(ds1d):.1f} sigma with |delta|<=0.06 free; C = 1.12: {np.sqrt(ds112):.1f} sigma. {J6}", "%.1f")
for lab_, ccal in (("cal05", 0.05), ("cal10", 0.10)):
    z = (cj6.C_shared - 1.0) / np.hypot(cj6.C_err, ccal)
    add(f"prec_C_tension_sigma_{lab_}", z, "sigma", f"(C_shared - 1.00)/sqrt({cj6.C_err:.3f}^2 + {ccal:.2f}^2): independent "
        f"calibration C_cal = 1.00 +- {ccal:.2f} (resp.md 9.1: 0.96-1.02 at 180-503 s"
        + ("" if ccal < 0.1 else "; conservative, whole 0.96-1.12 range") + f"). {J6}", "%.1f")
add("prec_delta_global", float(dg.group(1)), "", f"response tilt as one global parameter (C profiled): best "
    f"{dg.group(1)}, 1sigma [{dg.group(2)}, {dg.group(3)}]. {J6}", "%.3f")

# narrowing of the 1sigma Ep interval by HE (same definition as prec_ep_sharedC_narrowing_*): 1 - log10 width (joint6
# shared C) / log10 width (GBM alone, joint_fits_T0paper GBM nominal COMP rows)
nar = {}
for _, r in J6E[(J6E.sys == "nominal") & J6E.Ep_hi1.notna()].sort_values("t1").iterrows():
    g = JN2[(JN2.tag == "tr") & np.isclose(JN2.t1, r.t1) & np.isclose(JN2.t2, r.t2) & (JN2.model == "COMP")
            & (JN2.data == "GBM") & (JN2["var"] == "nominal")].iloc[0]
    f1 = F1[(F1.sys == "nominal") & (F1.lab == r.lab)].iloc[0]
    if not np.isfinite(g.Ep_hi1):
        continue
    wg = np.log10(g.Ep_hi1 / g.Ep_lo1)
    key = f"{r.t1:g}_{r.t2:g}"
    nar[key] = 1 - np.log10(r.Ep_hi1 / r.Ep_lo1) / wg
    n1 = 1 - np.log10(f1.Ep_hi1 / f1.Ep_lo1) / wg
    add(f"prec_narrow_sharedC_j6_{key}", 100 * nar[key], "%", f"1 - log10 width of the 1sigma Ep interval, GBM+HE shared "
        f"C={cj6.C_shared:.3f} [{r.Ep_lo1:.0f}, {r.Ep_hi1:.0f}] / GBM alone [{g.Ep_lo1:.0f}, {g.Ep_hi1:.0f}] keV "
        f"({np.log10(r.Ep_hi1 / r.Ep_lo1):.3f} / {wg:.3f} dex); at C=1.00: {100 * n1:.0f}%. Supersedes "
        f"prec_ep_sharedC_narrowing_{key} (joint4). {J6}", "%.0f")
v_ = np.array([v for k, v in nar.items() if not k.startswith("-1")])
add("prec_narrow_sharedC_j6_range", f"{100 * v_.min():.0f}-{100 * v_.max():.0f}", "%",
    f"range over the {len(v_)} intervals 1..8.576 s (" + ", ".join(f"{k}: {100 * v:.0f}" for k, v in nar.items())
    + f"); -1..0 s listed separately, 0..1 s has only a lower limit. Old joint4 value 37-74%. {J6}")
add("prec_narrow_sharedC_j6_median", 100 * np.median(v_), "%", f"median over 1..8.576 s. {J6}", "%.0f")

# notes: superseded / comparison rows
for rr_ in rows:
    if rr_["metric"] in ("prec_C_shared", "prec_C_shared_err") or (rr_["metric"].startswith("prec_ep_sharedC_")
                                                                     and "_j6_" not in rr_["metric"]):
        rr_["note"] += "; SUPERSEDED by joint6 (gbm.md 9.4): see the *_j6 rows (coarse joint4 grid, C=1.25)"
    m = rr_["metric"]
    if m.startswith("quiet_pred_v2_") and m.endswith("_C0.81"):
        rr_["note"] += "; C=0.81 WITHDRAWN (作废): FIFO loss at 180-216 s, area ratio now 0.96-1.25 (resp.md 9.1)"
    if (m.startswith(("quiet_he_net_", "quiet_he_ul3_flux_", "quiet_pred_he_")) and not m.endswith("_v2")
            and "pred_v2" not in m):
        rr_["note"] += "; SUPERSEDED (作废) by the *_v2 rows (resp.md 8.2: bias +18.5, sys 24)"
    if m.startswith("prec_ep_joint_") and m.endswith("_T0paper"):
        rr_["note"] += "; per-interval free C (comparison); main result now prec_ep_sharedC_* (gbm.md 8.2)"

new = pd.DataFrame(rows)
assert not set(new.metric) & set(v3.metric), "v4 metric name collides with v3"
pd.concat([v3, new], ignore_index=True).to_csv(f"{OUT}/numbers_v4.csv", index=False)
print(f"numbers_v4.csv: {len(v3)} v3 rows + {len(new)} v4 rows")
