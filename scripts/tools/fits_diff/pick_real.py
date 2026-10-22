"""Pick real archive files for every reader: a fixed-seed sample of days, one hour (or pass) per day.

    python3 pick_real.py OUT.tsv [--days N] [--seed S]

Archive roots are the readers' defaults, overridden by the same environment variables blink reads
(HXMT_1K_DIR, HXMT_1B_DIR, FERMI_GBM_DIR, GECAM_ARCHIVE, GECAM_C_ARCHIVE, SVOM_ARCHIVE, GRID_ARCHIVE).
Days are sampled from the day directories that exist, so missing roots just contribute nothing.
"""
import argparse, glob, os, random, re, sys

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--days', type=int, default=6, help='days per product')
ap.add_argument('--seed', type=int, default=20261021)
args = ap.parse_args()
rng = random.Random(args.seed)
env = lambda key, default: os.environ.get(key, default)
R1K = env('HXMT_1K_DIR', '/hxmt/work/HXMT-DATA/1K')
R1B = env('HXMT_1B_DIR', '/hxmtfs/data/Archive_tmp/1B')
GBM = env('FERMI_GBM_DIR', '/hxmtfs/data/Fermi_GBM')
GAB = env('GECAM_ARCHIVE', '/gecamfs/Archived-DATA/GSDC/LEVEL1/daily')
GC = env('GECAM_C_ARCHIVE', '/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily')
SV = env('SVOM_ARCHIVE', '/gecamfs/SVOM/Archived-DATA/GRM-DATA/L1B/daily')
GR = env('GRID_ARCHIVE', '/gecamfs/Exchange/GSDC/missions/GRID')

out = []
def days(pattern):
    found = sorted(glob.glob(pattern))
    print(f'{len(found):6d} day directories  {pattern}', file=sys.stderr)
    return rng.sample(found, min(args.days, len(found)))

def fits_in(pattern):
    return sorted(f for f in glob.glob(pattern) if re.search(r'\.(fits?|FITS)(\.gz)?$', f))

def add(kind, path):
    out.append((kind, path))

# HXMT 1K: Evt with the hour's Orbit and Att
for day in days(f'{R1K}/Y??????/????????-????'):
    evts = fits_in(f'{day}/HXMT_*_HE-Evt_FFFFFF_V*')
    if evts:
        evt = rng.choice(evts)
        add('hxmt_evt', evt)
        hour = re.search(r'HXMT_(\d{8}T\d{2})_', evt).group(1)
        for kind, name in [('hxmt_orbit', 'Orbit'), ('hxmt_att', 'Att')]:
            for f in fits_in(f'{day}/HXMT_{hour}_{name}_FFFFFF_V*')[-1:]:
                add(kind, f)

# HXMT 1B: science packets and engineering data of the three boxes
for day in days(f'{R1B}/????/????????'):
    for kind, codes in [('hxmt_sci', ('0642', '0922', '1686')), ('hxmt_eng', ('0766', '1009', '1781'))]:
        files = [f for code in codes for f in fits_in(f'{day}/{code}/HXMT_1B_*')]
        if files:
            add(kind, rng.choice(files))

# Fermi/GBM: one NaI and one BGO TTE file of an hour, and the day's position history
for pattern in [f'{GBM}/????/??/??/current', f'{GBM}/????/??????/tte', f'{GBM}/????/??/??????/tte']:
    for day in days(pattern):
        for det in ('n', 'b'):
            files = fits_in(f'{day}/glg_tte_{det}?_*')
            if files:
                add('gbm_tte', rng.choice(files))
        for f in fits_in(f'{day}/glg_poshist_all_*')[-1:]:
            add('gbm_poshist', f)
poshist = fits_in(f'{GBM}/poshist/glg_poshist_all_*')
for f in rng.sample(poshist, min(args.days, len(poshist))):
    add('gbm_poshist', f)

# GECAM: GRD events, CPD events and position/attitude of one hour
for root, kind, subdirs in [(f'{GAB}/????/??/??/GECAM_A', 'gecam_evt', ('GRD_evt', 'CPD_evt', 'posatt')),
                            (f'{GAB}/????/??/??/GECAM_B', 'gecam_evt', ('GRD_evt', 'CPD_evt', 'posatt')),
                            (f'{GC}/????/??/??', 'gecam_evt_c', ('GRD_EVT', 'CPD_EVT', 'POSATT'))]:
    for day in days(root):
        for k, sub in zip((kind, 'gecam_cpd', 'gecam_posatt'), subdirs):
            files = fits_in(f'{day}/{sub}/*')
            if files:
                add(k, rng.choice(files))

# SVOM/GRM
for day in days(f'{SV}/????/??/??'):
    for kind, sub in [('svom_evt', 'grm_evt'), ('svom_att', 'att'), ('svom_orb', 'orb')]:
        files = fits_in(f'{day}/{sub}/*')
        if files:
            add(kind, rng.choice(files))

# GRID: one pass of each product directory's latest version
for day in days(f'{GR}/*/*/????/??/??'):
    versions = sorted(glob.glob(f'{day}/*/'))
    files = fits_in(f'{versions[-1]}*') if versions else []
    if files:
        f = rng.choice(files)
        if '_posatt_' in os.path.basename(f):
            add('grid_posatt', f)
        elif '_evt_' in os.path.basename(f):
            add('grid_evt', f)
            add('grid_gti', f)

with open(args.out, 'w') as f:
    for kind, path in out:
        f.write(f'{kind}\t{path}\n')
counts = {}
for kind, _ in out:
    counts[kind] = counts.get(kind, 0) + 1
print(len(out), 'files:', ', '.join(f'{k} {n}' for k, n in sorted(counts.items())), file=sys.stderr)
