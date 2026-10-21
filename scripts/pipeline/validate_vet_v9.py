"""Check `blink vet` against the GRB removal and electron beams of the v9 catalog (HXMT paper, Secs 4.6, 6.1).

On a machine with the 1K archive:
    blink vet --list $V9/catalog_v9.csv.with_grb -o vet_v9.csv
    python scripts/pipeline/validate_vet_v9.py vet_v9.csv $V9

V9 holds catalog_v9.csv.with_grb (3870 entries before GRB removal), catalog_v9.csv (3853, column
electron_beam), grb_check/grb_removed.csv (17 removed) and grb_check/rule.npz (Z per entry of
catalog_v9.csv.with_grb, from 10 ms bins).

The GRB rule of the catalog: Z > 5 with the largest box share < 0.6, plus every entry within 1 s of a
flagged one. vet gives the first part per entry; the 1 s extension is applied here as the catalog step
will. Differences to expect: vet counts events, not 10 ms bins, and subtracts each box's own events of
the candidate instead of a third of all of them, so Z and the box share move slightly.
"""
import csv, sys
from datetime import datetime

import numpy as np

vet_csv, v9 = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(vet_csv)))
t = np.array([datetime.fromisoformat(r['start'][:26]).timestamp() for r in rows])
z = np.array([float(r['grb_z']) if r['grb_z'] else np.nan for r in rows])
flag = np.array([r['grb_flag'] == '1' for r in rows])
beam = np.array([r['electron_beam'] == '1' for r in rows])
print(f'{len(rows)} entries vetted, {np.isnan(z).sum()} without Z (background gap or missing data)')

# GRB removal: per-entry flag, then the same burst within 1 s
near = np.abs(t[:, None] - t[flag][None, :]).min(1) < 1.0 if flag.any() else np.zeros(len(t), bool)
removed = flag | near
ref = {r['start'] for r in csv.DictReader(open(f'{v9}/grb_check/grb_removed.csv'))}
got = {r['start'] for r, m in zip(rows, removed) if m}
print(f'\nGRB removal: vet {len(got)}, v9 {len(ref)}, common {len(got & ref)}')
for s in sorted(got - ref):
    print('  only vet:', s)
for s in sorted(ref - got):
    print('  only v9: ', s)

# Z against the binned Z of rule.py (same row order as catalog_v9.csv.with_grb)
rule = np.load(f'{v9}/grb_check/rule.npz')
if len(rule['Z']) == len(rows):
    ok = ~np.isnan(z) & ~np.isnan(rule['Z'])
    d = z[ok] - rule['Z'][ok]
    print(f'\nZ vet - rule.py over {ok.sum()} entries: median {np.median(d):+.2f}, '
          f'5-95% [{np.percentile(d, 5):+.2f}, {np.percentile(d, 95):+.2f}]')
    cross = ok & ((z > 5) != (rule['Z'] > 5))
    for i in np.where(cross)[0]:
        print(f'  Z crosses 5: {rows[i]["start"]}  vet {z[i]:.2f}  rule.py {rule["Z"][i]:.2f}  '
              f'box share {rows[i]["grb_max_box_share"]}')
else:
    print(f'\nrule.npz has {len(rule["Z"])} entries, the vet list {len(rows)}: Z not compared row by row')

# electron beams, on the entries kept in the catalog
cat = {r['start']: r['electron_beam'] == '1' for r in csv.DictReader(open(f'{v9}/catalog_v9.csv'))}
got = {r['start'] for r, b in zip(rows, beam) if b and r['start'] in cat}
ref = {s for s, b in cat.items() if b}
print(f'\nelectron beams: vet {len(got)}, v9 {len(ref)}, common {len(got & ref)}')
for s in sorted(got ^ ref):
    r = next(r for r in rows if r['start'] == s)
    print(f'  {"only vet" if s in got else "only v9 "}: {s}  P {r["beam_p"]}  '
          f'top {r["acd_top_only"]} side {r["acd_side_only"]}')
