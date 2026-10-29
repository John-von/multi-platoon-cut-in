"""WP5-v20 registered analysis (design sections 6-8).  Reads the registered case list and the per-run audits; writes
analysis/hypotheses.json and analysis/tables.json.  No decision rule may be changed after results are seen; a
changed rule is a new registered version.

Usage: python analyze.py <runs_root> [--cases cases_v1.json] [--out analysis]
runs_root/<run_key[:20]>/{case.json,result.json,arrays.npz,audit.json} (one directory per unique run)."""
import argparse, collections, json, math
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
TIMING_ARR = {'timing', 'planning_time', 'wall_time', 'loop_cost'}
RNG_SEED = 20260927
ALPHA = 0.05


def cp_interval(k, n, alpha=ALPHA):
    if n == 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


def hodges_lehmann(d):
    d = np.asarray(d, float)
    w = (d[:, None] + d[None, :]) / 2
    return float(np.median(w[np.triu_indices(len(d))]))


def hl_bootstrap(d, B=10000):
    d = np.asarray(d, float); rng = np.random.default_rng(RNG_SEED)
    est = [hodges_lehmann(d[rng.integers(0, len(d), len(d))]) for _ in range(B)]
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i]); adj = [None] * len(pvals); run = 0.
    for r, i in enumerate(order):
        run = max(run, min(1., (len(pvals) - r) * pvals[i])); adj[i] = run
    return adj


class Data:
    def __init__(self, root, cases_file):
        self.root = Path(root)
        self.cases = json.loads(Path(cases_file).read_text(encoding='utf-8'))['cases']
        self.by_id = {c['id']: c for c in self.cases}
        self._aud = {}

    def dir(self, c):
        return self.root / c['run_key'][:20]

    def audit(self, c):
        k = c['run_key']
        if k not in self._aud:
            p = self.dir(c) / 'audit.json'
            self._aud[k] = json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
        return self._aud[k]

    def select(self, **kw):
        out = []
        for c in self.cases:
            ok = True
            for k, v in kw.items():
                val = c['cell'].get(k) if k in c['cell'] else c.get(k)
                if callable(v):
                    ok = ok and v(val)
                else:
                    ok = ok and val == v
            if ok:
                out.append(c)
        return out

    def identical(self, c1, c2, exclude=()):
        """All non-timing arrays (minus `exclude`) bitwise equal."""
        A_, B_ = np.load(self.dir(c1) / 'arrays.npz'), np.load(self.dir(c2) / 'arrays.npz')
        skip = TIMING_ARR | set(exclude)
        if set(A_.files) - skip != set(B_.files) - skip:
            return False
        return all(A_[k].shape == B_[k].shape and np.array_equal(A_[k], B_[k], equal_nan=A_[k].dtype.kind in 'fc')
                   for k in A_.files if k not in skip)


def labels(a):
    return set() if a is None else set(a.get('labels', []))


def complete(a):
    return a is not None and 'C3' in labels(a)


def first_T(a):
    for j in (a or {}).get('planning_jobs', []):
        if j.get('accepted') and j.get('kind') == 'merge':
            return j.get('T_found')
    for t, T in (a or {}).get('plans', []):
        return T
    return None


def cellkey(c, drop=('variant',)):
    return tuple(sorted((k, v) for k, v in c['cell'].items() if k not in drop))


def coverage(D):
    """v1.1: registered unique runs per family and how many have an audit; decisions over families with missing
    audits are flagged 'incomplete' instead of being evaluated over a partial (possibly empty) set."""
    fam = collections.defaultdict(set); have = collections.defaultdict(set)
    for c in D.cases:
        fam[c['family']].add(c['run_key'])
        if D.audit(c) is not None:
            have[c['family']].add(c['run_key'])
    return {f: {'registered_unique_runs': len(k), 'audited': len(have[f]), 'complete': len(have[f]) == len(k)}
            for f, k in sorted(fam.items())}


def h1(D, methods=('A-H',)):
    seen = {}
    for c in D.cases:
        if c['method'] in methods and c['theory_covered']:
            seen.setdefault(c['run_key'], c)
    rows = collections.Counter(); viol = []
    for c in seen.values():
        a = D.audit(c); L = labels(a)
        if a is None:
            rows['missing'] += 1; continue
        if 'C0' in L:
            rows['initial_uncertified'] += 1; continue
        if 'X' in L or 'I' in L:
            rows['exception'] += 1; viol.append((c['id'], sorted(L))); continue
        rows['evaluable'] += 1
        for lab in ('P', 'M', 'K', 'E', 'L', 'D'):
            if lab in L:
                rows[lab] += 1
        if L & {'P', 'M', 'K'}:
            viol.append((c['id'], sorted(L)))
    n = rows['evaluable']; k = sum(1 for _, L in viol if set(L) & {'P', 'M', 'K'})
    decision = 'supported' if (k == 0 and rows['exception'] == 0 and rows['missing'] == 0) else \
        ('not_supported' if k else 'incomplete_or_exception')
    return {'unique_runs': len(seen), 'counts': dict(rows), 'theory_violations': k,
            'rate_95ci': cp_interval(k, n), 'violating_or_exception': viol[:50], 'decision': decision}


def h2(D, ref='A-H'):
    cells = collections.defaultdict(dict)
    for c in D.cases:
        if c['family'] in ('EX4', 'EX2c', 'EX2a') and c['cell']['ell'] == 0 and c['cell']['seed'] is None:
            cells[cellkey(c)][c['method']] = c
    out = []
    for key, m in cells.items():
        bad = [x for x in ('N', 'P') if x in m and 'P' in labels(D.audit(m[x]))]
        if not bad or ref not in m:
            continue
        a = D.audit(m[ref]); L = labels(a)
        if 'C0' in L or 'C1' in L:
            lab = 'screening'
        elif 'P' in L or 'X' in L or 'I' in L:
            lab = 'reference_violates'
        else:
            lab = 'execution'
        out.append({'cell': dict(key), 'violating': {x: D.audit(m[x]).get('first_violation') for x in bad},
                    'reference': m[ref]['id'], 'reference_labels': sorted(L), 'label': lab})
    if not out:
        dec = 'not_supported'
    elif any(r['label'] == 'execution' for r in out):
        dec = 'supported_execution'
    else:
        dec = 'screening_only'
    return {'reference': ref, 'cells_with_N_or_P_violation': len(out), 'rows': out, 'decision': dec}


def paired(D, fam, a_m, b_m, corner, metric):
    """Differences b - a over seeds where both complete (completion) or both evaluable (Jv needs both complete)."""
    A_ = {c['cell']['seed']: c for c in D.select(family=fam, method=a_m, corner=corner)}
    B_ = {c['cell']['seed']: c for c in D.select(family=fam, method=b_m, corner=corner)}
    d = []; n_pairs = 0
    for s in sorted(set(A_) & set(B_)):
        n_pairs += 1; aa, bb = D.audit(A_[s]), D.audit(B_[s])
        if complete(aa) and complete(bb):
            d.append(bb[metric] - aa[metric])
    return np.array(d, float), n_pairs


def h3h4(D):
    tests = []
    for base in ('F', 'DDRPC'):
        for ours in ('A-S', 'A-H'):
            for corner in ('nominal', 'low', 'high'):
                for metric in ('last_done', 'Jv'):
                    d, n = paired(D, 'EX1c', ours, base, corner, metric)
                    row = {'comparison': f'{base} - {ours}', 'corner': corner, 'metric': metric, 'pairs': n,
                           'both_complete': int(len(d))}
                    if len(d) == 0:
                        row.update(p=None, note='no both-complete pairs')
                    elif np.all(d == 0):
                        row.update(p=1.0, hl=0.0, ci=[0.0, 0.0], note='all differences zero')
                    else:
                        row.update(p=float(stats.wilcoxon(d, zero_method='pratt', alternative='two-sided').pvalue),
                                   hl=hodges_lehmann(d), ci=hl_bootstrap(d), median=float(np.median(d)),
                                   positive=int((d > 0).sum()), negative=int((d < 0).sum()), zero=int((d == 0).sum()))
                    tests.append(row)
    # v1.1: a test without both-complete pairs is 'not_evaluable'; it keeps p = 1 inside the
    # registered 24-test Holm family (conservative, family size unchanged) but can never support H3 or H4.
    ps = [t['p'] if t.get('p') is not None else 1.0 for t in tests]
    for t, q in zip(tests, holm(ps)):
        t['p_holm'] = q
        if t['both_complete'] == 0:
            t['verdict'] = 'not_evaluable'
        else:
            t['verdict'] = ('ours_better' if (q < ALPHA and t.get('hl', 0) > 0) else
                            'ours_worse' if (q < ALPHA and t.get('hl', 0) < 0) else 'no_significant_difference')
    ex1a = []
    for c in D.select(family='EX1a', method='F'):
        row = {'condition': c['cell']['corner']}
        for m in ('A-H', 'A-S', 'F', 'DDRPC', 'N', 'P'):
            cc = [x for x in D.select(family='EX1a', method=m) if x['cell']['corner'] == c['cell']['corner']]
            a = D.audit(cc[0]) if cc else None
            row[m] = None if a is None else {'labels': sorted(labels(a)), 'last_done': a.get('last_done'), 'Jv': a.get('Jv')}
        ex1a.append(row)
    both = [r for r in ex1a if r['F'] and r['A-S'] and 'C3' in r['F']['labels'] and 'C3' in r['A-S']['labels']]
    det_ok = None if not both else all(r['F']['last_done'] - r['A-S']['last_done'] > 0 for r in both)
    comp = [t for t in tests if t['comparison'].startswith('F') and t['metric'] == 'last_done']
    sig = [t['verdict'] == 'ours_better' for t in comp]
    evaluable = [t for t in comp if t['verdict'] != 'not_evaluable']
    if det_ok is None and not evaluable:
        h3 = 'not_evaluable'
    elif det_ok is True and len(evaluable) == len(comp) and all(sig):
        h3 = 'supported'
    elif det_ok is True or any(sig):
        h3 = 'partially_supported'
    else:
        h3 = 'not_supported'
    return {'EX1a_table': ex1a, 'EX1a_both_complete_conditions': len(both),
            'EX1a_A-S_earlier_than_F_in_all_both_complete': det_ok, 'EX1c_tests': tests,
            'holm_family_size': len(tests), 'evaluable_tests': sum(t['verdict'] != 'not_evaluable' for t in tests),
            'H3_decision': h3,
            'H4_summary': [{k: t.get(k) for k in ('comparison', 'corner', 'metric', 'both_complete', 'hl', 'p_holm', 'verdict')}
                           for t in tests if t['comparison'].startswith('DDRPC')]}


def h5(D):
    d = []
    for fam in ('EX1a', 'EX1c'):
        H = {cellkey(c): c for c in D.select(family=fam, method='A-H')}
        S_ = {cellkey(c): c for c in D.select(family=fam, method='A-S')}
        for k in set(H) & set(S_):
            a, b = D.audit(H[k]), D.audit(S_[k])
            if complete(a) and complete(b):
                d.append(a['last_done'] - b['last_done'])
    d = np.array(d)
    if not len(d):
        return {'n': 0, 'decision': 'not_evaluable'}
    med = float(np.median(d))
    return {'n': int(len(d)), 'median': med, 'min': float(d.min()), 'max': float(d.max()),
            'prediction': 'median in [2.5, 4.0] s', 'decision': 'as_predicted' if 2.5 <= med <= 4.0 else 'not_as_predicted'}


def h6(D, prescreen):
    pred = {(r['s'], r['ell']): r['initial_T'] is not None for r in prescreen}
    rows = []; agree = 0; cert_complete = 0; cert = 0
    for c in D.select(family='EX2a'):
        a = D.audit(c); L = labels(a)
        obs = a is not None and 'C0' not in L and 'X' not in L
        p = pred.get((c['cell']['s'], c['cell']['ell']))
        agree += int(obs == p); cert += int(obs); cert_complete += int(obs and complete(a))
        rows.append({'s': c['cell']['s'], 'ell': c['cell']['ell'], 'corner': c['cell']['corner'], 'predicted_certified': p,
                     'observed_certified': obs, 'labels': sorted(L), 'first_T': first_T(a),
                     'last_done': None if a is None else a.get('last_done')})
    mono = []
    for corner in ('nominal', 'high'):
        for axis, fixed in (('s', 'ell'), ('ell', 's')):
            groups = collections.defaultdict(list)
            for r in rows:
                if r['corner'] == corner and r['first_T'] is not None:
                    groups[r[fixed]].append((r[axis], r['first_T']))
            for g, pts in groups.items():
                pts.sort()
                mono += [{'corner': corner, 'vary': axis, fixed: g, 'from': p1, 'to': p2}
                         for p1, p2 in zip(pts, pts[1:]) if p2[1] < p1[1]]
    return {'n': len(rows), 'prediction_agreement': agree, 'certified': cert, 'certified_and_complete': cert_complete,
            'first_T_decreases': mono, 'rows': rows}


def h7(D):
    rows = []
    for c in D.select(family='EX3'):
        a = D.audit(c) or {}; ev = a.get('recorder_event') or {}
        fv = a.get('first_violation') or a.get('prefix_first_violation') or {}
        tp = [t for t in fv.values() if t is not None]
        rows.append({'id': c['id'], 'f': c['cell']['f'], 'corner': c['cell']['corner'], 'seed': c['cell']['seed'],
                     'labels': sorted(labels(a)), 't_empty_set': None if not ev else round(ev['step'] * 0.1, 3),
                     't_escape': a.get('parameter_escape_time'), 't_ratio_over_1': a.get('tube_ratio_first_over_1'),
                     't_containment_fail': a.get('activation_containment_first_fail'),
                     't_physical': min(tp) if tp else None})
    f1 = [r for r in rows if r['f'] <= 1.0]
    p1 = bool(f1) and all(r['labels'] and not (set(r['labels']) & {'P', 'M', 'K', 'I', 'X'}) for r in f1)
    p2 = all(r['t_physical'] is None for r in rows if r['f'] < 3)
    p3 = all(r['t_physical'] is None or any(t is not None and t <= r['t_physical'] for t in
                                            (r['t_escape'], r['t_ratio_over_1'], r['t_empty_set'])) for r in rows)
    return {'rows': rows, 'pred_f_le_1_clean': p1, 'pred_no_physical_below_f3': p2,
            'pred_physical_preceded_by_certificate_event': p3}


def h8(D):
    out = []
    for c in D.select(family='EX5'):
        cell, var = c['cell']['variant'].split('_', 1)
        if var == 'V0':
            continue
        base = [x for x in D.select(family='EX5') if x['cell']['variant'] == f'{cell}_V0' and x['cell']['f'] == c['cell']['f']]
        if not base:
            continue
        a, b = D.audit(c), D.audit(base[0])
        row = {'cell': cell, 'f': c['cell']['f'], 'variant': var, 'labels': sorted(labels(a)), 'base_labels': sorted(labels(b)),
               'last_done': None if a is None else a.get('last_done'), 'base_last_done': None if b is None else b.get('last_done'),
               'first_T': first_T(a), 'base_first_T': first_T(b)}
        both = bool(a and b and 'C0' not in labels(a) and 'C0' not in labels(b))
        try:
            row['identical_to_base'] = both and D.identical(c, base[0])
            row['identical_except_lateral_levels'] = both and D.identical(c, base[0], ('lateral_levels', 'lateral_slack'))
        except Exception:
            row['identical_to_base'] = row['identical_except_lateral_levels'] = None
        row['ell'] = c['cell']['ell']
        row['base_monitor_all_true'] = None if b is None else b.get('activation_observed_monitor_all')
        out.append(row)
    preds = []
    for r in out:
        p = None
        if r['variant'] == 'V1' and r['cell'] in ('C5', 'C6', 'C7', 'C8'):
            p = 'C0' in r['labels']
        elif r['variant'] == 'V1':
            p = (r['first_T'] or 99) > (r['base_first_T'] or 0) or ((r['last_done'] or 99) > (r['base_last_done'] or 0))
        elif r['variant'] == 'V3':
            p = r['identical_to_base']
        elif r['variant'] == 'V4' and r['base_monitor_all_true']:
            p = r['identical_to_base']
        elif r['variant'] == 'V5':
            p = r['identical_except_lateral_levels'] if r['ell'] == 0 else (r['labels'] == r['base_labels'])
        elif r['variant'] in ('V7', 'V8', 'V9') and r['cell'] not in ('C9', 'C10'):
            p = r['labels'] == r['base_labels']
        if p is not None:
            preds.append({'cell': r['cell'], 'f': r['f'], 'variant': r['variant'], 'as_predicted': bool(p)})
    lat = collections.defaultdict(dict)
    for r in out:
        if r['variant'].startswith('V6'):
            lat[(r['cell'], r['f'])][r['variant']] = r['last_done']
    for r in out:
        if r['variant'].startswith('V6'):
            lat[(r['cell'], r['f'])]['V6L30_20'] = r['base_last_done']
    order = ['V6L20_20', 'V6L30_20', 'V6L40_30', 'V6L60_40']
    lat_rows = []
    for (cell, f), d in lat.items():
        vals = [d.get(k) for k in order]
        ok = all(x is None or y is None or y >= x - 1e-9 for x, y in zip(vals, vals[1:]))
        lat_rows.append({'cell': cell, 'f': f, 'last_done_by_latency': dict(zip(order, vals)), 'non_decreasing': ok})
    return {'rows': out, 'predictions': preds, 'latency': lat_rows}


def h9(D):
    rows = []
    for c in D.select(family='EX6'):
        scen, var = c['cell']['variant'].split('_', 1)
        if var == 'base':
            continue
        base = [x for x in D.select(family='EX6') if x['cell']['variant'] == f'{scen}_base'][0]
        a, b = D.audit(c), D.audit(base)
        cat = lambda z: sorted(labels(z) & {'C0', 'C1', 'C2', 'C3', 'X', 'I'})
        row = {'scenario': scen, 'variant': var, 'category': cat(a), 'base_category': cat(b),
               'same_category': cat(a) == cat(b), 'last_done': None if a is None else a.get('last_done'),
               'base_last_done': None if b is None else b.get('last_done')}
        if complete(a) and complete(b):
            row['d_last_done'] = a['last_done'] - b['last_done']
            row['rel_d_Jv'] = (a['Jv'] - b['Jv']) / b['Jv'] if b['Jv'] else None
        if var in ('b8', 'b18', 'dmax60'):
            row['as_predicted'] = row['same_category'] and (row.get('d_last_done') is None or
                                  (abs(row['d_last_done']) <= 0.2 + 1e-9 and abs(row.get('rel_d_Jv') or 0) <= 0.01))
        elif var in ('merge1s', 'merge4s'):
            row['as_predicted'] = row['same_category'] and (row.get('d_last_done') is None or abs(row['d_last_done']) <= 2 + 1e-9)
        rows.append(row)
    return {'rows': rows}


def h10(D):
    rows = []
    for c in D.select(family='EX7'):
        a = D.audit(c)
        rows.append({'id': c['id'], 'variant': c['cell']['variant'], 'labels': sorted(labels(a)),
                     'last_done': None if a is None else a.get('last_done')})
    safe = bool(rows) and all(r['labels'] and not (set(r['labels']) & {'P', 'M', 'K', 'X', 'I'}) for r in rows)
    age = collections.defaultdict(dict)
    for r in rows:
        k, v = r['variant'].split('_', 1)
        if v == 'base' or v.startswith('age'):
            age[k][0 if v == 'base' else int(v[3:])] = r['last_done']
    mono = {k: all(d[x] is None or d[y] is None or d[y] >= d[x] - 1e-9 for x, y in zip(sorted(d), sorted(d)[1:]))
            for k, d in age.items()}
    noadm = {r['variant']: ('C1' in r['labels']) for r in rows if r['variant'].split('_', 1)[1] in ('drop012', 'lateAll')}
    return {'rows': rows, 'all_safe': safe, 'completion_non_decreasing_with_age': mono, 'no_admission_predicted_cases': noadm}


def representatives(D):
    reps = {}
    c = D.select(family='EX1a', method='A-H', corner='nominal'); reps['1_main_A-H'] = c[0]['id'] if c else None
    c = D.select(family='EX1b', method='A-H', corner='nominal'); reps['2_noise_A-H_ell5'] = c[0]['id'] if c else None
    worst = None
    for c in D.select(family='EX4', method=lambda m: m in ('N', 'P')):
        a = D.audit(c)
        if a is None or 'margins' not in a:
            continue
        mar = [v / lim for k, v in a['margins'].items() if v is not None for lim in [{'ordinary_gap': 10, 'merge_gap': 10,
               'source_gap': 10, 'speed': 23, 'acceleration': 2, 'window_jerk': 1, 'input': 5.49}[k]]]
        m = min(mar) if mar else None
        if m is not None and (worst is None or m < worst[0]):
            worst = (m, c)
    reps['3_EX4_N_or_P_worst'] = None if worst is None else worst[1]['id']
    if worst is not None:
        twin = [x for x in D.select(family='EX4', method='A-H') if cellkey(x) == cellkey(worst[1])]
        reps['4_same_cell_A-H'] = twin[0]['id'] if twin else None
    ex3 = [(c['cell']['f'], c) for c in D.select(family='EX3', corner='nominal', seed=None)]
    viol = [(f, c) for f, c in ex3 if 'P' in labels(D.audit(c))]
    pick = max(viol or ex3, key=lambda t: t[0]) if (viol or ex3) else None
    reps['5_EX3_beyond_bound'] = None if pick is None else pick[1]['id']
    c = D.select(family='EX2a', corner='nominal', s=10, ell=0); reps['6_envelope_s10'] = c[0]['id'] if c else None
    return reps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('runs_root')
    ap.add_argument('--cases', default=str(HERE / 'cases_v1.json')); ap.add_argument('--out', default=str(HERE / 'analysis'))
    a = ap.parse_args()
    D = Data(a.runs_root, a.cases); out = Path(a.out); out.mkdir(exist_ok=True)
    pre = json.loads((HERE / 'prescreen_initial.json').read_text(encoding='utf-8'))
    cov = coverage(D)
    H = {'coverage': cov, 'analysis_version': '1.1',
         'H1_A-H': h1(D), 'H1prime_sync_robust': h1(D, ('A-S', 'F', 'TUBE', 'DDRPC')), 'H2_A-H': h2(D, 'A-H'),
         'H2_A-S': h2(D, 'A-S'), 'H3_H4': h3h4(D), 'H5': h5(D), 'H6': h6(D, pre), 'H7': h7(D), 'H8': h8(D), 'H9': h9(D),
         'H10': h10(D), 'representatives': representatives(D)}
    (out / 'hypotheses.json').write_text(json.dumps(H, indent=1, default=str), encoding='utf-8', newline='\n')
    brief = {'all_families_complete': all(v['complete'] for v in cov.values()),
             'incomplete_families': [f for f, v in cov.items() if not v['complete']],
             'H1_A-H': H['H1_A-H']['decision'], 'H1prime': H['H1prime_sync_robust']['decision'],
             'H2_A-H': H['H2_A-H']['decision'], 'H2_A-S': H['H2_A-S']['decision'], 'H3': H['H3_H4']['H3_decision'],
             'H5': H['H5'].get('decision'), 'H6_agreement': f"{H['H6']['prediction_agreement']}/{H['H6']['n']}",
             'H7': {k: v for k, v in H['H7'].items() if k.startswith('pred')},
             'H8_as_predicted': f"{sum(r['as_predicted'] for r in H['H8']['predictions'])}/{len(H['H8']['predictions'])}",
             'H9_as_predicted': f"{sum(bool(r.get('as_predicted')) for r in H['H9']['rows'] if 'as_predicted' in r)}/"
                                f"{sum(1 for r in H['H9']['rows'] if 'as_predicted' in r)}",
             'H10_all_safe': H['H10']['all_safe'], 'representatives': H['representatives']}
    (out / 'brief.json').write_text(json.dumps(brief, indent=1, default=str), encoding='utf-8', newline='\n')
    print(json.dumps(brief, indent=1, default=str))


if __name__ == '__main__':
    main()
