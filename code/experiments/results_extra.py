"""Descriptive numbers for the manuscript that the registered analysis does not print directly (not a registered
analysis; every number is read from the per-run audits and is traceable to them).

Usage: python results_extra.py <runs_root> <out.json>"""
import json, sys, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROBUST = ('A-H', 'A-S', 'F', 'DDRPC')


def load(root):
    cases = []
    for f in ('cases_v1.json', 'cases_addA_v1.json'):
        cases += json.loads((HERE / f).read_text(encoding='utf-8'))['cases']
    aud = {}
    for c in cases:
        k = c['run_key'][:20]
        if k not in aud:
            p = Path(root) / k / 'audit.json'
            aud[k] = json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
    return cases, aud


def main():
    root, outp = sys.argv[1], Path(sys.argv[2])
    cases, aud = load(root)
    out = {}
    # label counts per family and method over unique run keys (first registration wins)
    first = {}
    for c in cases:
        first.setdefault(c['run_key'][:20], c)
    lab = collections.defaultdict(collections.Counter)
    for k, c in first.items():
        a = aud.get(k)
        lab[f"{c['family']}|{c['method']}"]['missing' if a is None else '+'.join(sorted(a['labels']))] += 1
    out['labels_by_family_method'] = {k: dict(v) for k, v in sorted(lab.items())}
    # theory-covered robust runs: tube ratio, lateral fallback, empirical items, activation margins
    tc = collections.defaultdict(lambda: {'n': 0, 'tube_ratio_max': 0.0, 'tube_ratio_argmax': None, 'L': 0, 'E': 0,
                                          'act_radius_margin_min': None, 'act_box_margin_min': None, 'activations': 0,
                                          'body_margin_min': None, 'P': 0, 'M': 0, 'K': 0})
    seen = set()
    for c in cases:
        k = c['run_key'][:20]
        if c['method'] not in ROBUST or not c.get('theory_covered') or (k, c['method']) in seen:
            continue
        seen.add((k, c['method'])); a = aud.get(k)
        if a is None or 'C0' in a['labels'] or 'X' in a['labels'] or 'I' in a['labels']:
            continue
        r = tc[c['method']]; r['n'] += 1
        if a.get('tube_ratio_max') is not None and a['tube_ratio_max'] > r['tube_ratio_max']:
            r['tube_ratio_max'] = a['tube_ratio_max']; r['tube_ratio_argmax'] = c['id']
        for L in ('L', 'E', 'P', 'M', 'K'):
            r[L] += int(L in a['labels'])
        r['activations'] += a.get('activations') or 0
        for key, src in (('act_radius_margin_min', 'activation_radius_margin_min'), ('act_box_margin_min', 'activation_box_margin_min')):
            v = a.get(src)
            if v is not None and (r[key] is None or v < r[key]):
                r[key] = v
        bm = (a.get('empirical') or {}).get('body_separating_margin')
        if bm is not None and (r['body_margin_min'] is None or bm < r['body_margin_min']):
            r['body_margin_min'] = bm
    out['theory_covered_robust'] = dict(tc)
    # N and P: violations anywhere (all families), with first-violation items
    np_rows = []
    for k, c in first.items():
        a = aud.get(k)
        if c['method'] in ('N', 'P') and a is not None and 'P' in a['labels']:
            fv = {x: t for x, t in (a.get('first_violation') or {}).items() if t is not None}
            np_rows.append({'id': c['id'], 'labels': a['labels'], 'first_violation': fv,
                            'margins': {x: v for x, v in (a.get('margins') or {}).items() if v is not None and v < 0}})
    out['N_P_violations'] = np_rows
    # all runs with P, M, K, E, I, X labels (any method), for the text
    bad = []
    for k, c in first.items():
        a = aud.get(k)
        if a is not None and set(a['labels']) & {'P', 'M', 'K', 'E', 'I', 'X'}:
            bad.append({'id': c['id'], 'method': c['method'], 'family': c['family'], 'labels': a['labels'],
                        'theory_covered': c.get('theory_covered')})
    out['flagged_runs'] = bad
    out['counts'] = {'unique_runs': len(first), 'audited': sum(a is not None for a in aud.values()),
                     'flagged': len(bad)}
    outp.write_text(json.dumps(out, indent=1, default=str), encoding='utf-8', newline='\n')
    print(json.dumps(out['counts']), json.dumps({m: {k: v for k, v in r.items() if k in ('n', 'tube_ratio_max', 'L', 'E', 'P', 'M', 'K')}
                                                 for m, r in out['theory_covered_robust'].items()}))


if __name__ == '__main__':
    main()
