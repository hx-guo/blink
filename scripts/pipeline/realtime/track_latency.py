"""Record when each event of the real-time catalog was first seen as provisional and as final.

    track_latency.py CATALOG_DIR LATENCY_CSV --now YYYYMMDDTHHMMSSZ

LATENCY_CSV keeps one row per candidate (key: instrument and start): peak, first_provisional,
first_final, last_status, last_seen. An event that is final at its first appearance has
first_provisional empty. Events that disappear keep their row (last_seen tells when). The latency of
the pipeline is first_final - peak (Sect. 5.4 of the paper); use only events whose peak is after the
start of real-time operation, since the first run finds the whole history at once.
"""
import argparse, csv, os
from datetime import datetime, timezone

ap = argparse.ArgumentParser()
ap.add_argument('catalog')
ap.add_argument('latency_csv')
ap.add_argument('--now', required=True, help='run time, UTC, e.g. 20261021T110000Z')
args = ap.parse_args()
now = datetime.strptime(args.now, '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

COLS = ['instrument', 'start', 'peak', 'first_provisional', 'first_final', 'last_status', 'last_seen']
rows = {}
if os.path.exists(args.latency_csv):
    for r in csv.DictReader(open(args.latency_csv)):
        rows[(r['instrument'], r['start'])] = r

for c in csv.DictReader(open(os.path.join(args.catalog, 'catalog.csv'))):
    key = (c['instrument'], c['start'])
    r = rows.setdefault(key, {'instrument': c['instrument'], 'start': c['start'], 'peak': c['peak'],
                              'first_provisional': '', 'first_final': '', 'last_status': '', 'last_seen': ''})
    if c['status'] == 'final' and not r['first_final']:
        r['first_final'] = now
    if c['status'] == 'provisional' and not r['first_provisional'] and not r['first_final']:
        r['first_provisional'] = now
    r['last_status'] = c['status']
    r['last_seen'] = now

tmp = args.latency_csv + '.tmp'
with open(tmp, 'w', newline='') as f:
    w = csv.DictWriter(f, COLS)
    w.writeheader()
    w.writerows(sorted(rows.values(), key=lambda r: (r['instrument'], r['start'])))
os.replace(tmp, args.latency_csv)
