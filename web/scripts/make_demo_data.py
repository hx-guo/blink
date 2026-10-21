"""Demo data for developing the website without the cluster.

    python web/scripts/make_demo_data.py [--blink target/debug/blink]   (run from the repo root)

Takes the real HXMT catalog data/catalog_v6.csv (direct tier only, fa <= 1e-5) and lays it out as the
real-time pipeline would: a catalog directory (catalog.csv, rejected.csv, manifest.json) in the format
of `blink catalog --out-dir`, daily *_vet.json files, and a name registry. The light curves in the vet
files are SIMULATED (Poisson background plus a burst with the catalog's counts and duration); the
manifest carries "demo": true and the website shows a banner. Then runs the real `blink name --apply`
and `blink export` into web/public/data.

The exposure is MODELED too: a circular orbit at 43 degrees inclination, uniform in longitude, local
time and month, with no exposure inside an elliptical stand-in for the SAA (HE is off there);
exposure.json carries "demo": true.

Everything except web/public/data goes to web/.demo (both are git-ignored).
"""
import argparse, csv, json, math, os, random, shutil, subprocess
from datetime import datetime, timezone

ap = argparse.ArgumentParser()
ap.add_argument('--blink', default='target/debug/blink')
ap.add_argument('--catalog', default='data/catalog_v6.csv')
ap.add_argument('--provisional-days', type=int, default=3, help='events of the last N days stay provisional')
args = ap.parse_args()

DEMO, OUT = 'web/.demo', 'web/public/data'
SLUG, INSTRUMENT = 'Insight-HXMT_HE', 'Insight-HXMT/HE'
random.seed(20261021)
shutil.rmtree(DEMO, ignore_errors=True)
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(f'{DEMO}/catalog')
os.makedirs(f'{DEMO}/registry')


def ns(t):
    """UTC string with up to ns digits -> epoch ns."""
    base, frac = t.rstrip('Z').split('.')
    s = datetime.fromisoformat(base).replace(tzinfo=timezone.utc).timestamp()
    return int(s) * 10**9 + int(frac.ljust(9, '0'))


def utc(n):
    s, f = divmod(n, 10**9)
    return datetime.fromtimestamp(s, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S') + f'.{f:09d}Z'


def solar_time(peak_ns, lon):
    """Apparent solar time HH:MM:SS (mean solar time + NOAA equation of time; the pipeline uses blink_solar)."""
    t = datetime.fromtimestamp(peak_ns // 10**9, timezone.utc)
    b = 2 * math.pi * (t.timetuple().tm_yday - 81) / 364
    eot_min = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)
    s = (t.hour * 3600 + t.minute * 60 + t.second + lon / 15 * 3600 + eot_min * 60) % 86400
    return f'{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}'


def poisson(lam):
    if lam > 50:
        return max(0, round(random.gauss(lam, math.sqrt(lam))))
    k, p, L = 0, 1.0, math.exp(-lam)
    while True:
        p *= random.random()
        if p <= L:
            return k
        k += 1


rows = [r for r in csv.DictReader(open(args.catalog)) if r['tier'] == 'direct']
last = max(r['date'] for r in rows)
last_ns = ns(max(r['start'] for r in rows))
cols = ('cand_id,instrument,status,pending,start,stop,peak,duration_us,count,false_positive_per_year,'
        'longitude,latitude,altitude_km,solar_time,neighbors_10min,grb_z,grb_max_box_share,electron_beam,beam_p,'
        'det_share_max,n,n_acd,n_acd_multi,n_bg,n_acd_bg').split(',')
out_rows, vets = [], {}
for r in rows:
    t0, t1 = ns(r['start']), ns(r['stop'])
    peak = (t0 + t1) // 2
    dur_us = (t1 - t0) / 1e3
    provisional = (last_ns - t0) < args.provisional_days * 86400 * 10**9
    lon = (float(r['longitude']) + 180) % 360 - 180
    cand = f"{SLUG}-{datetime.fromtimestamp(t0 // 10**9, timezone.utc):%Y%m%dT%H%M%S}.{(t0 % 10**9) // 1000:06d}"
    z = round(random.gauss(0, 1), 2)
    out_rows.append({
        'cand_id': cand, 'instrument': INSTRUMENT,
        'status': 'provisional' if provisional else 'final',
        'pending': 'neighbors_unsearched' if provisional else '',
        'start': r['start'], 'stop': r['stop'], 'peak': utc(peak),
        'duration_us': f'{dur_us:.3f}', 'count': r['count'],
        'false_positive_per_year': r['false_positive_per_year'],
        'longitude': f'{lon:.4f}', 'latitude': r['latitude'],
        'altitude_km': f"{float(r['altitude']) / 1e3:.1f}", 'solar_time': solar_time(peak, lon),
        'neighbors_10min': r['neighbors_10min'],
        'grb_z': f'{z:.2f}', 'grb_max_box_share': '0.400', 'electron_beam': '0', 'beam_p': '5.000e-1',
        'det_share_max': '0.150', 'n': r['n'], 'n_acd': r['n_acd'], 'n_acd_multi': r['n_acd_multi'],
        'n_bg': r['n_bg'], 'n_acd_bg': r['n_acd_bg'],
    })
    # simulated light curve: background 1e4/s over [start-5ms, stop+5ms], burst of `count` events
    span = (t1 - t0) / 1e9 + 0.01
    lc = []
    for _ in range(poisson(1e4 * span)):
        t = (random.uniform(-0.005, (t1 - t0) / 1e9 + 0.005) * 1e9 + t0 - peak) / 1e3
        lc.append((t, False))
    sigma_us = max(dur_us / 4, 20)
    for _ in range(int(r['count'])):
        lc.append((random.gauss(0, sigma_us), True))
    lc.sort()
    events = [{'t_us': round(t, 2), 'det': random.randrange(18),
               'channel': min(255, 38 + int(random.expovariate(1 / 40))),
               'acd': (1 << random.randrange(18)) if random.random() < (0.3 if burst else 0.15) else 0}
              for t, burst in lc]
    context = [poisson(100) for _ in range(400)]
    context[200] += int(r['count'])
    vets.setdefault(r['date'], []).append({
        'start': r['start'], 'stop': r['stop'], 'instrument': INSTRUMENT,
        'false_positive_per_year': float(r['false_positive_per_year']),
        'grb': {'observed': 4000, 'expected': 4000.0, 'z': z, 'group_excess': [0, 0, 0],
                'max_group_share': 0.4, 'flagged': False},
        'det_share_max': 0.15,
        'light_curve': {'peak': utc(peak), 'events': events, 'context_bin_s': 0.01,
                        'context_start_s': -2.0, 'context': context},
    })

with open(f'{DEMO}/catalog/catalog.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, cols)
    w.writeheader()
    w.writerows(out_rows)
open(f'{DEMO}/catalog/rejected.csv', 'w').write('cand_id,start,false_positive_per_year,reason,neighbors_10min,grb_z,grb_max_box_share\n')
n_final = sum(r['status'] == 'final' for r in out_rows)
searched = 6.31 * 365.25 * 86400 * (len(rows) / 3247)   # paper live time, scaled to the v6 range
json.dump({
    'instrument': INSTRUMENT, 'slug': SLUG, 'demo': True,
    'generated_at': utc(ns(datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z'))),
    'blink_git': 'demo',
    'data': {'first_day': '2017-06-15', 'last_day': f'{last[:4]}-{last[4:6]}-{last[6:]}',
             'days': 0, 'days_without_hours': 0, 'searched_seconds': searched},
    'selection': {'max_false_positive_per_year': 1e-5, 'train_half_window_s': 600, 'train_threshold': 34,
                  'train_threshold_fixed': True, 'grb_same_burst_s': 1.0, 'electron_beam_p': 1e-9,
                  'electron_beam_top_share_reference': 0.56},
    'counts': {'pool': 1709490, 'significant': len(rows), 'train_removed': 0, 'grb_removed': 0,
               'grb_same_burst_removed': 0, 'catalog': len(rows), 'final': n_final,
               'provisional': len(rows) - n_final, 'pending': {'neighbors_unsearched': len(rows) - n_final},
               'electron_beams': 0},
}, open(f'{DEMO}/catalog/manifest.json', 'w'), indent=1)
# modeled exposure: time fraction below latitude phi on a circular orbit, F = 1/2 + asin(sin phi / sin i) / pi
inc = math.radians(43.0)
F = lambda deg: 0.5 + math.asin(max(-1.0, min(1.0, math.sin(math.radians(deg)) / math.sin(inc)))) / math.pi
lat_frac = [F(i - 90 + 1) - F(i - 90) for i in range(180)]
cells = lambda m: [[i, j, round(s, 1)] for (i, j), s in sorted(m.items()) if s > 0]
# stand-in SAA: ellipse centred at 35W 35S, 45 deg x 13 deg semi-axes (south of the TGFs seen over Brazil)
saa = lambda i, j: ((((j + 0.5 + 180) % 360 - 180) + 35) / 45) ** 2 + ((i + 0.5 - 90 + 35) / 13) ** 2 < 1
emap = {(i, j): 0.0 if saa(i, j) else searched * lat_frac[i] / 360 for i in range(180) for j in range(360) if lat_frac[i] > 0}
elst = {(i, j): searched * lat_frac[i] / 48 for i in range(180) for j in range(48) if lat_frac[i] > 0}
emon = {(i, j): searched * lat_frac[i] / 12 for i in range(180) for j in range(1, 13) if lat_frac[i] > 0}
json.dump({'instrument': INSTRUMENT, 'demo': True, 'live_seconds': searched, 'no_ephemeris_seconds': 0.0,
           'days_with_exposure': 0, 'days': 0,
           'map': {'cells': cells(emap)}, 'lst': {'cells': cells(elst)}, 'month': {'cells': cells(emon)}},
          open(f'{DEMO}/catalog/exposure.json', 'w'))

for day, v in vets.items():
    d = f'{DEMO}/data/{SLUG}/{day[:4]}/{day[4:6]}'
    os.makedirs(d, exist_ok=True)
    json.dump(v, open(f'{d}/{day}_vet.json', 'w'))
open(f'{DEMO}/registry/public_instruments.txt', 'w').write(INSTRUMENT + '\n')
print(f'demo: {len(out_rows)} events ({n_final} final), {len(vets)} vet days')

subprocess.run([args.blink, 'name', f'{DEMO}/catalog', '--registry', f'{DEMO}/registry',
                '--version', 'demo', '--apply', '--allow-dirty'], check=True, stdout=subprocess.DEVNULL)
subprocess.run([args.blink, 'export', f'{DEMO}/catalog', '--registry', f'{DEMO}/registry',
                '--data', f'{DEMO}/data', '-o', OUT, '--public-only'], check=True)
