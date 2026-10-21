"""Time-zero bookkeeping for the v4 figures (GRB 221009A).

    run_gbm.sh -argu RUN p221009_v4_ltt      (gdt venv; astropy + gdt Fermi time)

Paper T0 = 2022-10-09T13:17:00.000 UTC = HXMT MET 339945423.0 (v2/v3 pipeline,
1-s bins, HE event cache t = Time - 339945423.0).  Prints:
  - UTC of the GBM trigger (MET 687014224.98753) and of HXMT MET 339945423.0/.05;
  - the light-travel-time difference between Fermi and Insight-HXMT towards
    the burst at T0: dt_ltt = (r_GBM - r_HXMT) . n / c, so that a wavefront
    seen by GBM at UTC u reaches HXMT at u + dt_ltt;
  - the resulting shift t_paper = t_GBM + OFFSET (OFFSET = GBM-trigger UTC
    - paper T0 + dt_ltt), written to $GBM_OUT/v4/t0_offsets.txt.
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import CartesianRepresentation, GCRS, ITRS, SkyCoord
from astropy.io import fits
from astropy.time import Time, TimeDelta
from gdt.missions.fermi.time import Time as FTime
import p221009_gbm_common as C

T0_PAPER = Time("2022-10-09T13:17:00.000", scale="utc")
HXMT_T0_MET = 339945423.0
HXMT_EPOCH = Time("2012-01-01T00:00:00", scale="utc")
ORBIT = "/scratchfs2/gecam/guohx/p221009/blink/data/hxmt_aux/HXMT_20221009T13_Orbit_FFFFFF_V1_1K.FITS"
POSHIST = "/scratchfs2/gecam/guohx/p221009/gbm/daily/glg_poshist_all_221009_v00.fit"

od = C.outdir("v4")
lines = []


def say(s):
    print(s, flush=True)
    lines.append(s)


t_gbm_trig = FTime(C.T0, format="fermi").utc
say("GBM trigger MET %.5f = %s UTC" % (C.T0, t_gbm_trig.isot))
for met in (HXMT_T0_MET, HXMT_T0_MET + 0.05):
    say("HXMT MET %.3f = %s UTC" % (met, (HXMT_EPOCH + TimeDelta(met, format="sec")).utc.isot))
d_trig = (t_gbm_trig - T0_PAPER).to_value(u.s)
say("GBM trigger - paper T0 = %+.4f s" % d_trig)

n = SkyCoord(C.RA * u.deg, C.DEC * u.deg, frame="icrs").cartesian.xyz.value
# HXMT: Orbit X/Y/Z are Earth-fixed (freeze_v3.md 2.1), metres
o = fits.open(ORBIT)[1].data
k = np.argmin(np.abs(o["Time"] - HXMT_T0_MET))
th = HXMT_EPOCH + TimeDelta(float(o["Time"][k]), format="sec")
itrs = ITRS(CartesianRepresentation(o["X"][k] * u.m, o["Y"][k] * u.m, o["Z"][k] * u.m), obstime=th)
r_h = itrs.transform_to(GCRS(obstime=th)).cartesian.xyz.to_value(u.m)
say("HXMT orbit row t-T0 = %+.1f s, |r| = %.0f km" % (o["Time"][k] - HXMT_T0_MET, np.linalg.norm(r_h) / 1e3))
# Fermi: poshist POS_X/Y/Z, J2000 (~GCRS), metres
p = fits.open(POSHIST)[1].data
met_t0 = C.T0 - d_trig                       # GBM MET at paper T0
j = np.argmin(np.abs(p["SCLK_UTC"] - met_t0))
r_g = np.array([p["POS_X"][j], p["POS_Y"][j], p["POS_Z"][j]], float)
say("Fermi poshist row t-T0 = %+.1f s, |r| = %.0f km" % (p["SCLK_UTC"][j] - met_t0, np.linalg.norm(r_g) / 1e3))
say("separation |r_GBM - r_HXMT| = %.0f km" % (np.linalg.norm(r_g - r_h) / 1e3))
ltt = float(np.dot(r_g - r_h, n) / 299792458.0)
say("light-travel dt_ltt = (r_GBM - r_HXMT).n/c = %+.4f s (positive: GBM sees the wavefront first)" % ltt)
off = d_trig + ltt
say("OFFSET: t_paper = t_GBM %+.4f s  (trigger-UTC part %+.4f, light-travel part %+.4f)" % (off, d_trig, ltt))
with open(f"{od}/t0_offsets.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
    f.write("offset_gbm_to_paper %.4f\noffset_trig %.4f\nltt %.4f\n" % (off, d_trig, ltt))
