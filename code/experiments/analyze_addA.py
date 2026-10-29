"""WP5-v20 addendum-A registered analysis (design addendum A, section 4).  Reads cases_v1.json + cases_addA_v1.json
and the per-run audits (EX9 audits come from audit_lateral.py); writes analysis_addA/hypotheses_addA.json.
Usage: python analyze_addA.py <runs_root> [--out analysis_addA]"""
import argparse, collections, json
from pathlib import Path
import numpy as np
from scipy import stats
import analyze as AN

HERE = Path(__file__).resolve().parent


class Data2(AN.Data):
    def __init__(self, root):
        self.root = Path(root)
        self.cases = []
        for f in ('cases_v1.json', 'cases_addA_v1.json'):
            self.cases += json.loads((HERE / f).read_text(encoding='utf-8'))['cases']
        self.by_id = {c['id']: c for c in self.cases}
        self._aud = {}


def L(a):
    return AN.labels(a)


def ha9(D):
    rows = []
    for c in D.select(family='EX9'):
        a = D.audit(c) or {}
        ex = a.get('lateral_extra', {})
        rows.append({'id': c['id'], 'variant': c['cell']['variant'], 'mild': c.get('note') == 'mild', 'labels': sorted(L(a)),
                     'last_done': a.get('last_done'), 'corridor_excess_above_m': ex.get('corridor_excess_above_m'),
                     'corridor_excess_below_m': ex.get('corridor_excess_below_m'),
                     'body_margin': (a.get('empirical') or {}).get('body_separating_margin'),
                     'levels': (a.get('empirical') or {}).get('lateral_levels_count'),
                     'lateral_solver_failures': (a.get('empirical') or {}).get('lateral_solver_failures'),
                     'lateral_fallback_samples': (a.get('empirical') or {}).get('lateral_fallback_samples'),
                     'corridor_above_duration_s': ex.get('corridor_above_duration_s_per_vehicle'),
                     'corridor_below_duration_s': ex.get('corridor_below_duration_s_per_vehicle'),
                     'lateral_force_sha256': a.get('lateral_force_sha256')})
    mild = [r for r in rows if r['mild']]
    p92 = bool(mild) and all('C3' in r['labels'] and not ({'P', 'M', 'K', 'E', 'X', 'I'} & set(r['labels'])) for r in mild)
    # v1.1: the runner catches lateral QP failures (counts lateral_solver_failures and uses the
    # shifted fallback), so they rarely surface as label X; HA9.3 counts the runner's failure counter, X separately.
    fails = [r['lateral_solver_failures'] for r in rows]
    p93 = None if (not rows or any(f is None for f in fails)) else int(sum(fails))
    x93 = sum('X' in r['labels'] for r in rows)
    p94 = bool(rows) and all(r['labels'] and not ({'P', 'M', 'K'} & set(r['labels'])) for r in rows)
    # paired random P0 vs C3
    P0 = {r['variant'].split('_seed')[1]: r for r in rows if r['variant'].startswith('L_P0_seed')}
    C3 = {r['variant'].split('_seed')[1]: r for r in rows if r['variant'].startswith('L_C3_seed')}
    pairs = sorted(set(P0) & set(C3))
    comp = lambda r: 'C3' in r['labels'] and 'X' not in r['labels']
    b = sum(1 for s in pairs if comp(P0[s]) and not comp(C3[s])); cc = sum(1 for s in pairs if comp(C3[s]) and not comp(P0[s]))
    mcnemar = float(stats.binomtest(b, b + cc, 0.5).pvalue) if b + cc else 1.0
    forces_equal = all(P0[s]['lateral_force_sha256'] == C3[s]['lateral_force_sha256'] for s in pairs)
    d = np.array([C3[s]['body_margin'] - P0[s]['body_margin'] for s in pairs
                  if P0[s]['body_margin'] is not None and C3[s]['body_margin'] is not None], float)
    body = {'n': int(len(d))}
    if len(d) and not np.all(d == 0):
        body.update(median=float(np.median(d)), hl=AN.hodges_lehmann(d), ci=AN.hl_bootstrap(d),
                    p=float(stats.wilcoxon(d, zero_method='pratt').pvalue))
    axes = collections.OrderedDict()
    for r in rows:
        v = r['variant'][2:]
        for ax in ('Pfr', 'Pf', 'Pr', 'Pm', 'Pi'):
            if v.startswith(ax + '_'):
                axes.setdefault(ax, []).append((v, r['labels'])); break
    return {'rows': rows, 'HA9_2_mild_all_complete_and_safe': p92, 'HA9_3_lateral_solver_failures_total': p93,
            'HA9_3_no_solver_abort': (None if p93 is None else (p93 == 0 and x93 == 0)), 'HA9_3_software_exceptions_X': x93,
            'HA9_4_longitudinal_no_P_M_K': p94,
            'Lc_max_excess_m': max([max(r['corridor_excess_above_m'] or 0, r['corridor_excess_below_m'] or 0) for r in rows] or [None]),
            'HA9_5_random': {'pairs': len(pairs), 'completed_P0': sum(comp(P0[s]) for s in pairs),
                             'completed_C3': sum(comp(C3[s]) for s in pairs), 'discordant_P0_only': b,
                             'discordant_C3_only': cc, 'exact_mcnemar_p': mcnemar,
                             'forces_identical_within_pairs': forces_equal, 'body_margin_C3_minus_P0': body},
            'single_factor_axes': axes}


def ha10(D):
    rows = []
    for c in D.select(family='EX10'):
        a = D.audit(c) or {}
        jobs = a.get('planning_jobs') or []
        pred = c.get('note', '').endswith('True')
        rej = [r for j in jobs for r in [j.get('rejection_reason')] if r]
        # v1.1: every merge candidate's rejection reason, from the audit's search record
        reasons = [rs for j in jobs if j.get('kind') == 'merge' for (_, rs) in (j.get('search') or [])]
        rows.append({'id': c['id'], 'family_scene': c['cell'].get('family_scene'), 'geometry': c['cell'].get('geometry'),
                     'boundary': c['cell']['boundary'], 'predicted_lp_feasible': pred, 'labels': sorted(L(a)),
                     'admitted': a.get('admitted'), 'completed': a.get('completed'), 'last_done': a.get('last_done'),
                     'first_T': AN.first_T(a), 'n_jobs': len(jobs), 'job_rejection_reasons': rej,
                     'merge_candidate_reasons': reasons})
    infeas = [r for r in rows if not r['predicted_lp_feasible']]
    p101 = bool(infeas) and all('C1' in r['labels'] for r in infeas)
    p101r = bool(infeas) and all(r['merge_candidate_reasons'] and set(r['merge_candidate_reasons']) == {'joint_allocation'}
                                 for r in infeas)
    p103 = bool(rows) and all(r['labels'] and not ({'P', 'M', 'K'} & set(r['labels'])) for r in rows)
    return {'rows': rows, 'HA10_1_infeasible_never_admitted': p101,
            'HA10_1_all_merge_candidates_rejected_by_joint_allocation': p101r,
            'HA10_1_decision': (p101 and p101r), 'HA10_1_n': len(infeas), 'HA10_3_no_P_M_K': p103}


def ha11(D):
    det = collections.defaultdict(dict)
    for c in D.select(family='EX11', seed=None):
        key = (c['cell']['boundary'], c['cell']['corner'], c['cell']['variant'].split('_')[1])
        det[key][c['cell']['mode']] = D.audit(c)
    dom, tight = [], []
    for key, m in det.items():
        cj = AN.complete(m.get('joint'))
        for other in ('sequential', 'independent'):
            if AN.complete(m.get(other)) and not cj:
                dom.append({'cell': key, 'mode': other})
        if key[2] == 'Q3' and key[1] == 'nominal' and key[0] in (26, 28):
            tight.append({'cell': key, 'joint': sorted(L(m.get('joint'))), 'sequential': sorted(L(m.get('sequential'))),
                          'independent': sorted(L(m.get('independent'))),
                          'as_predicted': AN.complete(m.get('joint')) and 'C1' in L(m.get('sequential')) and 'C1' in L(m.get('independent'))})
    seeds = collections.defaultdict(dict)
    for c in D.select(family='EX11', seed=lambda s: s is not None):
        seeds[c['cell']['seed']][c['cell']['mode']] = D.audit(c)
    ind_adm = sum(1 for s in seeds.values() if (s.get('independent') or {}).get('admitted'))
    tests = []
    for metric in ('last_done', 'Jv'):
        d = np.array([s['sequential'][metric] - s['joint'][metric] for s in seeds.values()
                      if AN.complete(s.get('joint')) and AN.complete(s.get('sequential'))], float)
        row = {'metric': metric, 'comparison': 'sequential - joint', 'both_complete': int(len(d))}
        if len(d) == 0:
            row['p'] = None
        elif np.all(d == 0):
            row.update(p=1.0, hl=0.0, ci=[0., 0.])
        else:
            row.update(p=float(stats.wilcoxon(d, zero_method='pratt').pvalue), hl=AN.hodges_lehmann(d), ci=AN.hl_bootstrap(d),
                       median=float(np.median(d)))
        tests.append(row)
    for t, q in zip(tests, AN.holm([t['p'] if t.get('p') is not None else 1. for t in tests])):
        t['p_holm'] = q
    allruns = [D.audit(c) for c in D.select(family='EX11')]
    p114 = bool(allruns) and all(a is not None and not ({'P', 'M', 'K'} & L(a)) for a in allruns)
    x114 = sum(1 for a in allruns if a is not None and ({'X', 'I'} & L(a)))
    return {'HA11_1_joint_weakly_dominates': not dom, 'counterexamples': dom, 'HA11_2_tight': tight,
            'HA11_3_independent_admitted_of_50': ind_adm, 'HA11_3_tests': tests, 'HA11_4_no_P_M_K': p114,
            'HA11_software_exceptions_X_or_I': x114}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('runs_root'); ap.add_argument('--out', default=str(HERE / 'analysis_addA'))
    a = ap.parse_args(); D = Data2(a.runs_root); out = Path(a.out); out.mkdir(exist_ok=True)
    cov = AN.coverage(D)
    H = {'coverage': cov, 'analysis_version': '1.1',
         'H1_pooled_v1_plus_addA': AN.h1(D), 'EX9': ha9(D), 'EX10': ha10(D), 'EX11': ha11(D)}
    (out / 'hypotheses_addA.json').write_text(json.dumps(H, indent=1, default=str), encoding='utf-8', newline='\n')
    brief = {'all_families_complete': all(v['complete'] for v in cov.values()),
             'incomplete_families': [f for f, v in cov.items() if not v['complete']],
             'H1_pooled': H['H1_pooled_v1_plus_addA']['decision'],
             'HA9': {k: v for k, v in H['EX9'].items() if k.startswith('HA9') and k != 'HA9_5_random'},
             'HA10': {k: v for k, v in H['EX10'].items() if k.startswith('HA10')},
             'HA11': {k: H['EX11'][k] for k in ('HA11_1_joint_weakly_dominates', 'HA11_3_independent_admitted_of_50', 'HA11_4_no_P_M_K')}}
    (out / 'brief_addA.json').write_text(json.dumps(brief, indent=1, default=str), encoding='utf-8', newline='\n'); print(json.dumps(brief, indent=1, default=str))


if __name__ == '__main__':
    main()
