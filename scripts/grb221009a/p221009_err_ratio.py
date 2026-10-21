#!/usr/bin/env python3
"""GRB 221009A: HXMT/GECAM ratio statistics with per-bin errors, split into
statistical and systematic parts. Reads data/p221009/errors/lc_221009a_1s_errors.csv
(from p221009_err_budget.py). Deterministic.

Bin 229 is the SAA-mode shut-off second: the HE event stream stops ~0.1-0.4 s
into it while engineering frame 229 integrates mostly detector-off time, so
f_live(229) = 0.86 is not the live fraction of the counted events. It is
listed but excluded from the weighted statistics.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from p221009_err_budget import wstats, WINDOWS, G_FLOOR, GECAM_SCALE  # noqa: E402

OUT = Path("data/p221009/errors")
EDGE_BINS = {229, 248, 267, 276}  # partial-on seconds (HE off 229.2-249.0, 267.0-277.0)
BRIGHT = 1000.0  # GECAM net cts/s


def main():
    lc = pd.read_csv(OUT / "lc_221009a_1s_errors.csv")
    rows = []
    for wname, (lo, hi) in WINDOWS.items():
        d = lc[(lc.t >= lo) & (lc.t < hi) & (lc.gecam_net > G_FLOOR)
               & np.isfinite(lc.rec) & (lc.rec > 0) & np.isfinite(lc.dt_corr)]
        d = d[~d.t.isin(EDGE_BINS)].copy()
        gs = d.gecam_net * GECAM_SCALE
        d["r"] = d.dt_corr / gs
        d["sG"] = d.gecam_sig / d.gecam_net
        d["s_stat"] = d.r * np.sqrt((d.sig_stat / d.dt_corr) ** 2 + d.sG ** 2)
        d["s_hsys"] = d.r * d.sig_sys / d.dt_corr
        d["s_gsys"] = d.r * d.gecam_bsys / d.gecam_net
        d["s_all"] = np.sqrt(d.s_stat ** 2 + d.s_hsys ** 2 + d.s_gsys ** 2)
        d["r_ec"] = d.dt_corr / (d.gecam_net_ec * GECAM_SCALE)
        d["r_borb"] = d.dt_corr_Borb / gs
        for sub, m in (("all", np.ones(len(d), bool)),
                       ("faint", (d.gecam_net <= BRIGHT).to_numpy()),
                       ("bright", (d.gecam_net > BRIGHT).to_numpy())):
            dd = d[m]
            if len(dd) < 3:
                continue
            for tag, col, rcol in (("stat", "s_stat", "r"), ("stat+sys", "s_all", "r"),
                                   ("stat,GECAM-exp-corr", "s_stat", "r_ec"),
                                   ("stat,B_orb(t)", "s_stat", "r_borb")):
                st = wstats(dd[rcol].to_numpy(), dd[col].to_numpy())
                st.update(window=wname, subset=sub, errors=tag)
                rows.append(st)
        d.to_csv(OUT / f"ratio_bins_{wname}_clean.csv", index=False, float_format="%.5g")
    rt = pd.DataFrame(rows)[["window", "subset", "errors", "n", "med", "rsig", "mu",
                             "emu", "chi2", "dof", "s_int", "mu_int", "emu_int", "med_err"]]
    rt.to_csv(OUT / "ratio_stats_clean.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    print(rt.to_string())


if __name__ == "__main__":
    main()
