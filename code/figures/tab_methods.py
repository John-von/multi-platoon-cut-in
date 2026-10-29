"""Numbers of Table III (internal comparison methods and their main results).
EX1a: registered analysis (hypotheses.json H3_H4 EX1a_table). EX1b, EX1c: recomputed archive rows
(data/extracted/all_runs.json). EX4 outcome counts: fig_need_data.json (independent verification rows).
Usage: python tab_methods.py <extracted_data_dir> <fig_need_data.json> <out_dir>"""
import json, sys
from pathlib import Path
from data_inputs import Inputs
I = Inputs(with_d=True, d_dir=sys.argv[1])
need = json.loads(Path(sys.argv[2]).read_text())['panel_b']['counts']
M = ['A-H', 'A-S', 'F', 'DDRPC', 'N', 'P']
tab = I.hyp['H3_H4']['EX1a_table']
assert len(tab) == 11
out = {}
for m in M:
    t = [r[m]['last_done'] for r in tab]; j = [r[m]['Jv'] for r in tab]
    assert all(r[m]['labels'] == ['C3'] for r in tab)
    out[m] = {'EX1a_t': (min(t), max(t)), 'EX1a_Jv': (min(j), max(j))}
def drows(fam, m, **cell):
    rs = []
    for c in I.family(fam, **cell):
        if c['method'] != m: continue
        d = I.d(c); assert d is not None, c['id']
        rs.append(d)
    return rs
for m in M:
    rs = drows('EX1b', m)
    if rs:
        labs = sorted({tuple(r['labels']) for r in rs}); t = [r['last_done'] for r in rs]
        def first_merge(r):
            ev = [e for e in r['events'] if e.get('kind', 'merge') == 'merge' and e['reason'] == 'accepted']
            return ev[0]['T'] if ev else None
        firstT = sorted({first_merge(r) for r in rs if r['events']} - {None})
        out[m]['EX1b'] = {'n': len(rs), 'labels': labs, 't': (min(t), max(t)), 'first_accepted_merge_T': firstT}
    rs = drows('EX1c', m)
    if rs:
        labs = sorted({tuple(r['labels']) for r in rs}); t = [r['last_done'] for r in rs]
        out[m]['EX1c'] = {'n': len(rs), 'labels': labs, 't': (min(t), max(t))}
    out[m]['EX4'] = need[m]
# A-S vs F in EX1b per condition
bF = {json.dumps({k: v for k, v in c['cell'].items()}, sort_keys=True): I.d(c)['last_done'] for c in I.family('EX1b') if c['method'] == 'F'}
bS = {json.dumps({k: v for k, v in c['cell'].items()}, sort_keys=True): I.d(c)['last_done'] for c in I.family('EX1b') if c['method'] == 'A-S'}
bH = {json.dumps({k: v for k, v in c['cell'].items()}, sort_keys=True): I.d(c)['last_done'] for c in I.family('EX1b') if c['method'] == 'A-H'}
dSF = sorted(round(bF[k] - bS[k], 3) for k in bS); dHF = sorted(round(bH[k] - bF[k], 3) for k in bH)
print(json.dumps(out, indent=1)); print('F - A-S (EX1b):', dSF); print('A-H - F (EX1b):', dHF)
OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'tab_methods_data.json').write_text(json.dumps({'methods': out, 'EX1b_F_minus_AS': dSF, 'EX1b_AH_minus_F': dHF}, indent=1))
