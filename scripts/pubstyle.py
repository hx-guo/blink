"""Publication figure style for the HXMT/HE saturation paper (SCPMA).

Figures are laid out 1:1 with the printed page: single-column figures
use COL_W and full-width (figure*) figures use FULL_W as their figsize
width, so \\includegraphics applies no scaling and fonts render at
their nominal size in print.

Nominal sizes (identical across all figures): 9 pt axis labels and
panel tags, 8 pt tick labels, legends and annotations. The paper body
is 10 pt and captions 8 pt, so no figure text is smaller than the
caption and the figures stay legible without zooming.

Measured from the SCPMA template (2026-07): \\columnwidth inside
multicols = 243.27 pt = 3.366 in; \\textwidth = 506.46 pt = 7.007 in.

Usage:
    import pubstyle
    pubstyle.apply()
    fig, ax = plt.subplots(figsize=(pubstyle.COL_W, 2.2))
"""
import matplotlib

COL_W = 3.37    # single-column width in inches
FULL_W = 7.01   # full text width in inches (figure*)

# Paper-wide colour roles
C_OBS = "#20347e"     # dark blue : HXMT observed
C_RECON = "#5b9bd5"   # light blue: HXMT observed + reconstructed
C_GBM = "#ff7f0e"     # orange    : Fermi/GBM reference
C_EXT2 = "#7d4fd0"    # purple    : SVOM/GRM, ASIM/MXGS
C_ENG = "#2e8b57"     # green     : engineering channel
C_SAT = "#D62728"     # red       : saturation shading (use low alpha)


def apply():
    matplotlib.rcParams.update({
        # match the SCPMA body font (times/txfonts): STIX is the
        # Times-compatible family shipped with matplotlib
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.framealpha": 0.9,
        "legend.handlelength": 1.5,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.3,
        "legend.borderaxespad": 0.4,
        "pdf.fonttype": 42,
        # no tight bbox: the PDF must stay exactly figsize (COL_W/FULL_W)
        # so \includegraphics scales by 1.00 and fonts render at true size
    })
