"""Purity and completeness of the real-time catalog, checked with lightning data it does not use.

    python scripts/pipeline/purity.py CATALOG_DIR TGFS_JSON [--offline catalog_v9.csv] [-o OUT_DIR]

CATALOG_DIR  a catalog of `blink catalog --out-dir` (catalog.csv, rejected.csv)
TGFS_JSON    `blink wwlln` on the same candidate pool: lightning.associated, coincidence_probability and
             in_coverage per candidate, joined to the catalog on the start time (same string)
--offline    an offline catalog with columns start, false_positive_per_year, tier, associated
             (catalog_v6.csv / catalog_v9.csv): its significance-selected entries (tier direct) are compared
             with the real-time catalog over the same days

WWLLN association as in the HXMT paper: a stroke within 800 km of the sub-satellite point and 5 ms of the
peak after the light-travel correction. Only final events inside the WWLLN coverage count. Chance
associations expected = sum of coincidence_probability. Fractions are given with binomial standard errors.

Writes summary.txt and the tables as CSV to OUT_DIR (default: CATALOG_DIR/purity).
duration_us is the duration of the best interval, as in the HXMT paper.
"""
import argparse, csv, json, math, os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('catalog')
ap.add_argument('tgfs')
ap.add_argument('--offline')
ap.add_argument('-o', '--out')
args = ap.parse_args()
out_dir = args.out or os.path.join(args.catalog, 'purity')
os.makedirs(out_dir, exist_ok=True)
lines = []


def say(s=''):
    print(s)
    lines.append(s)


def frac(k, n):
    if n == 0:
        return '–'
    p = k / n
    return f'{100 * p:.1f} ± {100 * math.sqrt(p * (1 - p) / n):.1f} %  ({k}/{n})'


# lightning of every candidate of the pool, by start
lightning = {}
for t in json.load(open(args.tgfs)):
    l = t['lightning']
    lightning[t['signal']['start']] = (bool(l['associated']), l.get('coincidence_probability'), l.get('in_coverage', True))

cat = list(csv.DictReader(open(os.path.join(args.catalog, 'catalog.csv'))))
final = [r for r in cat if r['status'] == 'final']
covered = [r for r in final if r['start'] in lightning and lightning[r['start']][2]]
missing = [r for r in final if r['start'] not in lightning]
say(f'catalog {args.catalog}: {len(cat)} events, {len(final)} final, {len(covered)} final inside the WWLLN coverage')
if missing:
    say(f'  WARNING: {len(missing)} final events have no entry in {args.tgfs} (different candidate pool?)')

assoc = [r for r in covered if lightning[r['start']][0]]
chance = sum(lightning[r['start']][1] or 0.0 for r in covered)
say(f'associated with WWLLN: {frac(len(assoc), len(covered))}; chance associations expected {chance:.1f}')


def table(name, key, bins):
    """Association fraction in bins of key(row)."""
    rows = []
    say(f'\nby {name}:')
    for lo, hi, label in bins:
        sel = [r for r in covered if lo <= key(r) < hi]
        k = sum(lightning[r['start']][0] for r in sel)
        c = sum(lightning[r['start']][1] or 0.0 for r in sel)
        say(f'  {label:>16}  {frac(k, len(sel))}  chance {c:.1f}')
        rows.append({name: label, 'n': len(sel), 'associated': k, 'chance': round(c, 2)})
    with open(os.path.join(out_dir, f'by_{name}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, [name, 'n', 'associated', 'chance'])
        w.writeheader()
        w.writerows(rows)


table('duration_us', lambda r: float(r['duration_us']),
      [(0, 100, '< 100'), (100, 200, '100-200'), (200, 400, '200-400'), (400, 1000, '400-1000'), (1000, 1e9, '>= 1000')])
table('abs_latitude', lambda r: abs(float(r['latitude'])),
      [(0, 10, '0-10'), (10, 20, '10-20'), (20, 30, '20-30'), (30, 90, '>= 30')])
table('fa', lambda r: math.log10(float(r['false_positive_per_year'])),
      [(-1e9, -12, '< 1e-12'), (-12, -9, '1e-12 - 1e-9'), (-9, -7, '1e-9 - 1e-7'), (-7, -5.0000001, '1e-7 - 1e-5')])

if args.offline:
    off = [r for r in csv.DictReader(open(args.offline)) if r.get('tier', 'direct') == 'direct']
    days = {r['start'][:10] for r in cat}
    off = [r for r in off if r['start'][:10] in days]
    ours = {r['start']: r for r in final}
    theirs = {r['start']: r for r in off}
    both = ours.keys() & theirs.keys()
    only_ours = [ours[s] for s in ours.keys() - theirs.keys()]
    only_theirs = [theirs[s] for s in theirs.keys() - ours.keys()]
    say(f'\noffline catalog {args.offline}: {len(off)} significance-selected entries on the days of this catalog')
    say(f'  in both {len(both)}, only real-time {len(only_ours)}, only offline {len(only_theirs)}')
    off_cov = [r for r in off if r['start'] in lightning and lightning[r['start']][2]]
    say(f'  offline associated (its own column): {frac(sum(r["associated"] == "1" for r in off_cov), len(off_cov))}')
    for label, group in (('only real-time', only_ours), ('only offline', only_theirs)):
        g = [r for r in group if r['start'] in lightning and lightning[r['start']][2]]
        say(f'  {label}: associated {frac(sum(lightning[r["start"]][0] for r in g), len(g))}')
    # why entries are only offline: rejected here, not final, or not a candidate at all
    rejected = {r['start']: r['reason'] for r in csv.DictReader(open(os.path.join(args.catalog, 'rejected.csv')))}
    provisional = {r['start'] for r in cat if r['status'] != 'final'}
    reasons = defaultdict(int)
    with open(os.path.join(out_dir, 'only_offline.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['start', 'false_positive_per_year', 'associated', 'why'])
        for r in sorted(only_theirs, key=lambda r: r['start']):
            why = rejected.get(r['start']) or ('provisional' if r['start'] in provisional else 'not selected')
            reasons[why] += 1
            w.writerow([r['start'], r['false_positive_per_year'], r.get('associated', ''), why])
    say('  only offline, why: ' + ', '.join(f'{k} {v}' for k, v in sorted(reasons.items())))
    with open(os.path.join(out_dir, 'only_realtime.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['start', 'false_positive_per_year', 'associated'])
        for r in sorted(only_ours, key=lambda r: r['start']):
            l = lightning.get(r['start'])
            w.writerow([r['start'], r['false_positive_per_year'], '' if l is None else int(l[0])])

open(os.path.join(out_dir, 'summary.txt'), 'w').write('\n'.join(lines) + '\n')
print(f'\n-> {out_dir}')
