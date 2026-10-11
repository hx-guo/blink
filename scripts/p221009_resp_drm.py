"""HE/CsI low-gain DRM for GRB 221009A in raw-channel space (response study).

Re-implements the HXMT GRB group's generator (RSPgenerator_v1/HXMTCsIDRM.py,
v240930: Luo+2020 Geant4 library, 4-direction inverse-distance interpolation,
empirical theta factor, FWHM broadening) but folds each detector's deposited
energy onto its own raw channels with the low-gain EC (E = k*ch + b), instead
of the uniform 150-3750 keV PI grid with random PI assignment.  This lets us
fit the summed raw-channel spectrum directly.

    python p221009_resp_drm.py [OUTDIR]

Outputs drm_221009a.npz (per-detector and summed matrices for the precursor,
main-emission and flare directions; effective-area curves), an OGIP .rsp of
the summed precursor matrix, and a check against the group's own DRM."""
import os
import sys

import numpy as np
from astropy.io import fits
from scipy.stats import norm

LIB = "/hxmtfs2/work/GRB/Software/RSPgenerator_v1"
sys.path.append(LIB)
import coord_trans  # noqa: E402  (HXMT group's attitude transform)

RA, DEC = 288.2646, 19.7733                     # Swift/XRT position
T0_UTC = "2022-10-09T13:17:00"
ATT = "/hxmt/work/HXMT-DATA/1K/Y202210/20221009-1943/HXMT_20221009T13_Att_FFFFFF_V1_1K.FITS"
EC_FILE = f"{LIB}/EC_FWHM/2022-10-04_EC_LHV.txt"
FWHM_FILES = {"lab": f"{LIB}/EC_FWHM/CsI_LG_RasE.txt",
              "orbit": f"{LIB}/EC_FWHM/CsI_LHV_Orbit_v1_RasE.txt"}
NCH = 300                                       # unwrapped channel 0..299
DEP_BIN = 4.0
DEP_EDGES = np.arange(0, 4000 + DEP_BIN, DEP_BIN)
DEP_MID = 0.5 * (DEP_EDGES[1:] + DEP_EDGES[:-1])

sim_ang = np.loadtxt(f"{LIB}/SimInfo/Incident_Angle.txt")
SIM_T, SIM_P, SIM_V = sim_ang[:, 0].astype(int), sim_ang[:, 1].astype(int), sim_ang[:, 2:]
sim_e = np.loadtxt(f"{LIB}/SimInfo/Incident_Energy.txt")
EIN_LO, EIN_HI = sim_e[:, 0], sim_e[:, 1]
EC = np.loadtxt(EC_FILE, usecols=[1, 2, 3, 4])   # k, b, k_err, b_err


def ang_vec(theta, phi):
    t, p = np.deg2rad(theta), np.deg2rad(phi)
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def interp_weights(theta, phi):
    v = ang_vec(theta, phi)
    c = SIM_V @ v / np.linalg.norm(SIM_V, axis=1)
    ang = np.arccos(np.clip(c, -1, 1))
    idx = np.argsort(ang)[:4]
    if ang[idx[0]] < 1e-8:
        w = np.array([1.0, 0, 0, 0])
    else:
        w = (1 / ang[idx]) / np.sum(1 / ang[idx])
    return idx, w


def mc_det(det, theta, phi):
    """Angle-interpolated MC matrix (128 incident x 1000 deposited, cm^2)."""
    idx, w = interp_weights(theta, phi)
    m = np.zeros((EIN_LO.size, DEP_MID.size))
    for i, wi in zip(idx, w):
        th, ph = SIM_T[i], SIM_P[i]
        with fits.open(f"{LIB}/RSP_CsI/LG/D{det}/D{det}_T{th}_P{ph}_LG.fits") as h:
            r = np.array(h[1].data["rsp"], float) * (-3.59e-3 * th + 0.983)
        r[15] *= 50 / 49
        m += wi * r
    return m


def broaden(m, a, b):
    sig = a * DEP_MID ** (b + 1) / 2.35
    res = norm.pdf(DEP_MID[None, :], DEP_MID[:, None], sig[:, None]) * DEP_BIN
    return m @ res


def to_channels(m, k, b):
    """Fold deposited-energy matrix onto unwrapped raw channels 0..NCH-1."""
    edges = k * (np.arange(NCH + 1) - 0.5) + b
    cum = np.concatenate([np.zeros((m.shape[0], 1)), np.cumsum(m, axis=1)], axis=1)
    out = np.empty((m.shape[0], NCH))
    for i in range(m.shape[0]):
        ci = np.interp(edges, DEP_EDGES, cum[i], left=0.0, right=cum[i, -1])
        out[i] = np.diff(ci)
    return out


def build(theta, phi, fwhm="lab", ec=EC, dets=range(18)):
    fw = np.loadtxt(FWHM_FILES[fwhm], usecols=[1, 2])
    per = np.zeros((18, EIN_LO.size, NCH))
    dep = np.zeros((EIN_LO.size, DEP_MID.size))
    for d in dets:
        mb = broaden(mc_det(d, theta, phi), *fw[d])
        dep += mb
        per[d] = to_channels(mb, ec[d, 0], ec[d, 1])
    return per, dep


def write_rsp(path, mat, label):
    k, b = np.median(EC[:, 0]), np.median(EC[:, 1])
    ch = np.arange(NCH)
    eb = fits.BinTableHDU.from_columns([
        fits.Column(name="CHANNEL", format="I", array=ch),
        fits.Column(name="E_MIN", format="E", unit="keV", array=k * (ch - 0.5) + b),
        fits.Column(name="E_MAX", format="E", unit="keV", array=k * (ch + 0.5) + b)])
    eb.header["EXTNAME"] = "EBOUNDS"
    mx = fits.BinTableHDU.from_columns([
        fits.Column(name="ENERG_LO", format="E", unit="keV", array=EIN_LO),
        fits.Column(name="ENERG_HI", format="E", unit="keV", array=EIN_HI),
        fits.Column(name="N_GRP", format="I", array=np.ones(EIN_LO.size, int)),
        fits.Column(name="F_CHAN", format="I", array=np.zeros(EIN_LO.size, int)),
        fits.Column(name="N_CHAN", format="I", array=np.full(EIN_LO.size, NCH)),
        fits.Column(name="MATRIX", format=f"{NCH}E", array=mat.astype(np.float32))])
    mx.header["EXTNAME"] = "SPECRESP MATRIX"
    for h in (eb, mx):
        h.header.update(TELESCOP="HXMT", INSTRUME="HE", DETNAM="CsI-sum18", HDUCLASS="OGIP",
                        HDUCLAS1="RESPONSE", CHANTYPE="PHA", DETCHANS=NCH, TLMIN1=0)
        h.header["COMMENT"] = label + "; channel = unwrapped raw channel (raw<20 -> +256)"
    mx.header.update(HDUCLAS2="RSP_MATRIX", HDUVERS="1.3.0")
    eb.header.update(HDUCLAS2="EBOUNDS", HDUVERS="1.2.0")
    fits.HDUList([fits.PrimaryHDU(), eb, mx]).writeto(path, overwrite=True)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/scratchfs2/gecam/guohx/p221009/resp/out"
    os.makedirs(out, exist_ok=True)
    # incident angle along the burst (attitude from 1K Att file)
    th, ph, tl = coord_trans.J2000_Sat(ra=[RA], dec=[DEC], start_time="2022-10-09T13:16:50",
                                       end_time="2022-10-09T13:29:00", step=1, filename=[[ATT]])
    trel = np.arange(th.size) - 10.0
    np.savetxt(f"{out}/angle_1s.txt", np.column_stack([trel, th, ph]), fmt="%.1f %.3f %.3f",
               header="t_rel_T0  theta  phi  (HXMT body frame, HXMT group convention)")
    wins = {"precursor": (-1, 10), "rise": (176, 215), "mid": (287, 440), "flare": (440, 600),
            "group_ref": None}
    res = {}
    for name, w in wins.items():
        if w is None:
            t_, p_ = 11.90, 181.00
        else:
            s = (trel >= w[0]) & (trel < w[1])
            v = np.mean([ang_vec(a, b) for a, b in zip(th[s], ph[s])], axis=0)
            v /= np.linalg.norm(v)
            t_, p_ = np.rad2deg(np.arccos(v[2])), np.rad2deg(np.arctan2(v[1], v[0])) % 360
        per, dep = build(t_, p_)
        res[name] = dict(theta=t_, phi=p_, per=per, dep=dep)
        print(f"{name}: theta={t_:.2f} phi={p_:.2f}  A(ch>=20) at 300/500/1000/3000 keV:",
              [round(float(np.interp(e, np.sqrt(EIN_LO * EIN_HI), per.sum(0)[:, 20:276].sum(1))), 1)
               for e in (300, 500, 1000, 3000)], flush=True)
    # systematics variants for the precursor direction
    tp, pp = res["precursor"]["theta"], res["precursor"]["phi"]
    per_orb, _ = build(tp, pp, fwhm="orbit")
    ec_lo = EC.copy(); ec_lo[:, 1] -= EC[:, 3]
    ec_hi = EC.copy(); ec_hi[:, 1] += EC[:, 3]
    per_blo, _ = build(tp, pp, ec=ec_lo)
    per_bhi, _ = build(tp, pp, ec=ec_hi)
    per_t5, _ = build(tp + 5, pp)
    # compare with the group's merged DRM at (11.90, 181.00): channel-summed area over 150-3750 keV
    with fits.open(f"{LIB}/HXMTCsIDRM_2022-10-09/HED-Merged_LG_theta_11.90_phi_181.00_v240930.rsp") as h:
        grp = np.array(h["SPECRESP MATRIX"].data["MATRIX"], float)
    mine = res["group_ref"]["dep"][:, (DEP_MID > 150) & (DEP_MID < 3750)].sum(1)
    emid = np.sqrt(EIN_LO * EIN_HI)
    sel = (emid > 200) & (emid < 5000)
    print("check vs group DRM (150-3750 keV dep): max |rel diff| for 0.2-5 MeV =",
          float(np.max(np.abs(mine[sel] / grp.sum(1)[sel] - 1))))
    np.savez_compressed(
        f"{out}/drm_221009a.npz", ein_lo=EIN_LO, ein_hi=EIN_HI, ec=EC,
        **{f"{n}_per": r["per"].astype(np.float32) for n, r in res.items()},
        **{f"{n}_dep": r["dep"].astype(np.float32) for n, r in res.items()},
        angles=np.array([[r["theta"], r["phi"]] for r in res.values()]),
        names=np.array(list(res)), grp_area=grp.sum(1),
        prec_fwhm_orbit=per_orb.astype(np.float32), prec_b_lo=per_blo.astype(np.float32),
        prec_b_hi=per_bhi.astype(np.float32), prec_theta5=per_t5.astype(np.float32))
    write_rsp(f"{out}/he_csi_lg_221009a_precursor_sum18.rsp", res["precursor"]["per"].sum(0),
              f"precursor theta={tp:.2f} phi={pp:.2f}, EC 2022-10-04 LHV, FWHM CsI_LG_RasE")
    print("saved")


if __name__ == "__main__":
    main()
