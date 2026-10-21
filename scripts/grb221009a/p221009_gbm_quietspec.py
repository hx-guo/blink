"""Step 3b (spectra): PG-stat fits of the GBM quiescent-interval emission
(Lesage region II, 121.219-164.228 s, and the HE quiescent window 20-170 s)
with an orbital background spectrum.

Source: TTE (n4, n8, b1) integrated over the window.  Background: daily CSPEC
at +/-30 orbits (and +/-15 as a check), same ground track (shift from
C.orbital_shifts), averaged over the shifted window, then scaled per detector
and CTIME-like energy group to the burst-day TTE in T0-400...-10 s.  Background
variance = orbital counting noise + scale error.
Also: direct BGO/NaI count upper limits in 0.3-3 MeV converted to an energy
flux in 200-3000 keV for two spectral shapes, and the ratio to the precursor
1-s peak (same band, same detectors) for comparison with the HE limit.
Outputs $GBM_OUT/quiet/quietspec_*.csv.
"""
import glob
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from gdt.core.background.primitives import BackgroundSpectrum
import p221009_gbm_common as C
import p221009_gbm_specfit as S

od = C.outdir("quiet")
DETS = ["n4", "n8", "b1"]
PRE = (-400.0, -10.0)
GROUP_EDGES = {"n": [8, 27, 50, 100, 300, 540, 1000, 2000], "b": [200, 930, 2120, 4780, 10000, 22000, 50000]}
P, SH = C.orbital_shifts()


def cspec(det, day):
    f = glob.glob(os.path.join(C.DAILY, "glg_cspec_%s_%s_v*.pha" % (det, day)))[0]
    with fits.open(f) as h:
        d = h["SPECTRUM"].data
        return (np.array(d["TIME"], float) - C.T0, np.array(d["ENDTIME"], float) - C.T0,
                np.array(d["COUNTS"], float), np.array(d["EXPOSURE"], float))


def orb_counts(det, ks, t1, t2):
    """Summed orbital counts and exposure over [t1, t2] (fraction-weighted
    CSPEC bins) for the shifted days."""
    cnt = 0.0; expo = 0.0
    for k in ks:
        ts, te, c, x = cspec(det, C.DAY[k])
        ts = ts - SH[k][0]; te = te - SH[k][0]
        w = np.clip((np.minimum(te, t2) - np.maximum(ts, t1)) / (te - ts), 0, 1)
        cnt = cnt + (w[:, None] * c).sum(0); expo = expo + (w * x).sum()
    return cnt, expo


class OrbData(S.DetData):
    """DetData whose background is the scaled orbital CSPEC spectrum."""

    def __init__(self, det, ks):
        self.det = det
        self.ks = ks
        self.tte = C.open_tte(det, (-400.0, 800.0))
        pre = self.tte.to_pha(time_ranges=[PRE])
        oc, ox = orb_counts(det, ks, *PRE)
        lo = np.asarray(pre.ebounds.low_edges())
        edges = GROUP_EDGES[det[0]]
        self.grp = np.clip(np.searchsorted(edges, lo, side="right") - 1, 0, len(edges) - 2)
        self.scale = np.ones(len(lo)); self.scale_err = np.zeros(len(lo))
        for g in np.unique(self.grp):
            m = self.grp == g
            n = pre.data.counts[m].sum(); b = oc[m].sum() * pre.exposure / ox
            self.scale[m] = n / b
            self.scale_err[m] = np.sqrt(n / b ** 2 + n ** 2 / b ** 3)
        print(det, ks, "group scales", np.round([self.scale[self.grp == g][0] for g in np.unique(self.grp)], 4), flush=True)

    def pieces(self, t1, t2, erange=None):
        erange = erange or C.erange_for(self.det)
        pha = self.tte.to_pha(time_ranges=[(t1, t2)])
        lo, hi = C.chan_range(pha.ebounds, *erange)
        mask = np.zeros(pha.num_chans, bool); mask[lo:hi + 1] = True
        oc, ox = orb_counts(self.det, self.ks, t1, t2)
        rate = self.scale * oc / ox
        err = np.sqrt((self.scale * np.sqrt(np.maximum(oc, 1)) / ox) ** 2 + (self.scale_err * oc / ox) ** 2)
        e = pha.ebounds
        bak = BackgroundSpectrum(rate, err, np.asarray(e.low_edges()), np.asarray(e.high_edges()), np.full(len(rate), pha.exposure))
        return pha, bak, C.response_for(self.det, t1, t2), mask


def main():
    rows = []
    NAI_FIT = (20.0, 500.0)
    for ks_tag, ks in [("orb30", (-30, 30)), ("orb15", (-15, 15))]:
        D = {d: OrbData(d, ks) for d in DETS}
        er = {"n4": NAI_FIT, "n8": NAI_FIT}
        for wname, t1, t2 in [("II", 121.219, 164.228), ("HEquiet", 20.0, 170.0), ("bump", 125.0, 160.0),
                              ("tail", 10.0, 90.0), ("peak0-1", 0.0, 1.024)]:
            for dets in (["n4", "n8"], ["n4", "n8", "b1"]):
                for model in ["PL", "COMP", "BB+COMP", "Band"]:
                    if wname == "peak0-1" and (model in ("BB+COMP",) or len(dets) == 2):
                        continue
                    try:
                        r = S.fit([D[d] for d in dets], t1, t2, model, er if wname != "peak0-1" else None)
                    except Exception as exc:  # noqa: BLE001
                        print("fail", ks_tag, wname, dets, model, exc, flush=True)
                        continue
                    r.pop("_fitter"); r.update(bkg=ks_tag, window=wname)
                    rows.append(r)
                    print(ks_tag, wname, r["dets"], model, "PG %.1f/%d" % (r["stat"], r["dof"]),
                          "Ep", r.get("Epeak"), "idx", r.get("index", r.get("alpha")),
                          "F10-1000 %.3g" % r["eflux_10_1000"], flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(od, "quietspec_fits.csv"), index=False, float_format="%.6g")

        # direct count upper limits in the HE band -> energy flux 200-3000 keV
        from gdt.core.spectra.functions import PowerLaw, Comptonized
        ul = []
        for wname, t1, t2 in [("II", 121.219, 164.228), ("HEquiet", 20.0, 170.0)]:
            for det, (e1, e2) in [("b1", (300.0, 3000.0)), ("n4", (300.0, 900.0)), ("n8", (300.0, 900.0))]:
                pha, bak, rsp, _ = D[det].pieces(t1, t2)
                lo, hi = C.chan_range(pha.ebounds, e1, e2)
                m = np.zeros(pha.num_chans, bool); m[lo:hi + 1] = True
                n = pha.data.counts[m].sum(); b = (bak.rates[m] * pha.exposure).sum()
                sb = np.sqrt(b + ((bak.rate_uncertainty[m] * pha.exposure) ** 2).sum())
                net = n - b; ul2 = max(net, 0) + 2 * sb
                for shape, f, par in [("PL-1.7", PowerLaw(), [1.0, -1.7, 100.0]),
                                      ("PL-2.2", PowerLaw(), [1.0, -2.2, 100.0]),
                                      ("COMP_prec", Comptonized(), [1.0, 2300.0, -1.67, 100.0])]:
                    cr = rsp.fold_spectrum(f.eval, par).counts[m].sum()      # counts/s per unit A
                    fl = f.integrate(par, (200.0, 3000.0), energy=True)  # erg/cm2/s per unit A
                    ul.append(dict(bkg=ks_tag, window=wname, det=det, e1=e1, e2=e2, counts=n, bkg_counts=b,
                                   net=net, sigma=sb, signif=net / sb, shape=shape,
                                   eflux_200_3000=net / pha.exposure / cr * fl,
                                   eflux_200_3000_ul2=ul2 / pha.exposure / cr * fl))
        pd.DataFrame(ul).to_csv(os.path.join(od, "quietspec_ul_%s.csv" % ks_tag), index=False, float_format="%.6g")
    print("done")


if __name__ == "__main__":
    main()
