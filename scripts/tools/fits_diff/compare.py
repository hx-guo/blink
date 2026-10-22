"""Compare the digests of two builds (old.tsv, new.tsv) over the file list: compare.py OLD NEW LIST"""
import collections, sys

def load(p):
    out = {}
    for line in open(p):
        parts = line.rstrip('\n').split('\t')
        out[(parts[0], parts[1])] = parts[2:]
    return out

old, new = load(sys.argv[1]), load(sys.argv[2])
listed = [tuple(l.rstrip('\n').split('\t')) for l in open(sys.argv[3]) if l.strip()]
cat = collections.defaultdict(list)
for key in listed:
    o, n = old.get(key), new.get(key)
    if o is None or n is None:
        cat['missing result'].append((key, o, n))
    elif o[0] == 'OK' and n[0] == 'OK':
        cat['identical' if o[1:] == n[1:] else 'DIFFERENT DATA'].append((key, o, n))
    elif o[0] == 'ERR' and n[0] == 'ERR':
        cat['both error'].append((key, o, n))
    else:
        cat['ONE SIDE ERRORS'].append((key, o, n))

by_kind = collections.defaultdict(collections.Counter)
for c, items in cat.items():
    for (kind, _), _, _ in items:
        by_kind[kind][c] += 1
print(f'{len(listed)} files')
for c in ['identical', 'DIFFERENT DATA', 'ONE SIDE ERRORS', 'both error', 'missing result']:
    print(f'  {c:16s} {len(cat[c])}')
print()
for kind in sorted(by_kind):
    print(f'  {kind:14s} ' + ', '.join(f'{c} {n}' for c, n in sorted(by_kind[kind].items())))
for c in ['DIFFERENT DATA', 'ONE SIDE ERRORS', 'both error', 'missing result']:
    if cat[c]:
        print(f'\n== {c}')
        for (kind, path), o, n in cat[c]:
            print(f'  {kind}  {path.split("/")[-1]}\n      old: {o}\n      new: {n}')
sys.exit(1 if cat['DIFFERENT DATA'] or cat['ONE SIDE ERRORS'] or cat['missing result'] else 0)
