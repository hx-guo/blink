"""CI check of the TGF name registry: published names are never changed or removed.

    python check_registry.py REGISTRY_DIR [BASE_REF]

Compares names.csv in the working tree with names.csv at BASE_REF (default HEAD~1) of the registry's
git repository. Fails if a row was removed or reordered, if name, peak_utc, assigned_at or assigned_by
of an existing row changed, if a name appears twice, if a status is not active/retracted/merged, or if
merged_into points to a name that does not exist. history.csv must only grow by appended lines.
"""
import csv, io, subprocess, sys

reg = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else 'HEAD~1'
IMMUTABLE = ('name', 'peak_utc', 'assigned_at', 'assigned_by')


def at(ref, path):
    r = subprocess.run(['git', '-C', reg, 'show', f'{ref}:{path}'], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ''


errors = []
old = list(csv.DictReader(io.StringIO(at(base, 'names.csv'))))
new = list(csv.DictReader(open(f'{reg}/names.csv')))
if len(new) < len(old):
    errors.append(f'names.csv lost rows: {len(old)} -> {len(new)}')
for a, b in zip(old, new):
    for k in IMMUTABLE:
        if a[k] != b[k]:
            errors.append(f'{a["name"]}: {k} changed {a[k]!r} -> {b[k]!r}')
names = [r['name'] for r in new]
if len(set(names)) != len(names):
    errors.append('duplicated names')
for r in new:
    if r['status'] not in ('active', 'retracted', 'merged'):
        errors.append(f'{r["name"]}: bad status {r["status"]!r}')
    if r['status'] == 'merged' and r['merged_into'] not in names:
        errors.append(f'{r["name"]}: merged into unknown {r["merged_into"]!r}')
old_history = at(base, 'history.csv')
if not open(f'{reg}/history.csv').read().startswith(old_history):
    errors.append('history.csv was edited, not appended to')

print('\n'.join(errors) or f'registry ok: {len(new)} names ({len(new) - len(old)} new since {base})')
sys.exit(1 if errors else 0)
