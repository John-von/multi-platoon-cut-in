"""Independent re-computation of the registered H3/H4/H5 statistics (verification aid, not a registered analysis).

Usage: python verify_stats.py <runs_root> <analysis/hypotheses.json> <out.json>

Re-implements, without scipy.stats and without importing analyze.py:
  * pairing of EX1c runs by seed (both complete = label C3 in both audits) for F/DDRPC vs A-S/A-H, 3 corners, 2 metrics;
  * Wilcoxon signed-rank p-values: exact null distribution by dynamic programming when there are no zero differences
    and no ties (and n <= 50), otherwise the normal approximation with Pratt's zero handling and tie correction;
  * Hodges-Lehmann estimate (median of Walsh averages including i = j), Holm adjustment over the 24 tests;
  * H5 (A-H minus A-S completion time over EX1a and EX1c cells where both complete).
It then compares pair counts, HL estimates, Holm-adjusted p-values and verdicts with the registered analysis output."""
import json, math, sys, collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def audits(root):
    cases = json.loads((HERE / 'cases_v1.json').read_text(encoding='utf-8'))['cases']
    A = {}
    for c in cases:
        p = Path(root) / c['run_key'][:20] / 'audit.json'
        A[c['id']] = json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
    return cases, A


def is_complete(a):
    return a is not None and 'C3' in a.get('labels', [])


def rank_abs(x):
    """Average ranks of |x| (1-based)."""
    order = np.argsort(np.abs(x), kind='mergesort'); ax = np.abs(x)[order]
    r = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and ax[j + 1] == ax[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r


def wilcoxon_p(d):
    d = np.asarray(d, float); n = len(d)
    r = rank_abs(d)                      # Pratt: rank including zeros, then drop the zero ranks
    nz = d != 0
    Wp = float(r[(d > 0)].sum()); Wm = float(r[(d < 0)].sum())
    ties = len(np.unique(np.abs(d[nz]))) < nz.sum()
    if nz.all() and not ties and n <= 50:
        # exact two-sided p-value of T+ under H0 (ranks 1..n, each sign equally likely)
        maxs = n * (n + 1) // 2; cnt = np.zeros(maxs + 1); cnt[0] = 1
        for k in range(1, n + 1):
            cnt[k:] = cnt[k:] + cnt[:-k].copy() if k <= maxs else cnt[k:]
        cnt = cnt / cnt.sum(); T = int(round(min(Wp, Wm)))
        p = min(1.0, 2 * cnt[:T + 1].sum())
        return p, 'exact', Wp
    # normal approximation (Pratt 1959; Cureton 1967 variance with zeros ranked then dropped)
    rz = r[~nz]; rn = r[nz]
    mean = (r.sum() - rz.sum()) / 2
    # variance: sum of squared ranks of non-zero differences / 4 (tie-corrected automatically by using mid-ranks)
    var = (rn ** 2).sum() / 4
    if var == 0:
        return 1.0, 'degenerate', Wp
    z = (Wp - mean) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))
    return p, 'normal_pratt', Wp


def hl(d):
    d = np.sort(np.asarray(d, float)); i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


def holm(p):
    idx = sorted(range(len(p)), key=lambda i: p[i]); out = [0.] * len(p); run = 0.
    for r, i in enumerate(idx):
        run = max(run, min(1., (len(p) - r) * p[i])); out[i] = run
    return out


def main():
    root, reg, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    cases, A = audits(root)
    H = json.loads(Path(reg).read_text(encoding='utf-8'))
    by = collections.defaultdict(dict)
    for c in cases:
        if c['family'] == 'EX1c':
            by[(c['cell']['corner'], c['cell']['seed'])][c['method']] = A[c['id']]
    tests = []
    for base in ('F', 'DDRPC'):
        for ours in ('A-S', 'A-H'):
            for corner in ('nominal', 'low', 'high'):
                for metric in ('last_done', 'Jv'):
                    d = []; pairs = 0
                    for (cn, seed), m in sorted(by.items(), key=lambda t: (t[0][0], t[0][1])):
                        if cn != corner or ours not in m or base not in m:
                            continue
                        pairs += 1
                        if is_complete(m[ours]) and is_complete(m[base]):
                            d.append(m[base][metric] - m[ours][metric])
                    row = {'comparison': f'{base} - {ours}', 'corner': corner, 'metric': metric, 'pairs': pairs, 'both_complete': len(d)}
                    if d and np.any(np.asarray(d) != 0):
                        p, how, Wp = wilcoxon_p(d); row.update(p=p, p_method=how, hl=hl(d), median=float(np.median(d)))
                    elif d:
                        row.update(p=1.0, p_method='all_zero', hl=0.0)
                    else:
                        row.update(p=None)
                    tests.append(row)
    adj = holm([t['p'] if t['p'] is not None else 1.0 for t in tests])
    for t, q in zip(tests, adj):
        t['p_holm'] = q
        t['verdict'] = ('not_evaluable' if t['both_complete'] == 0 else
                        'ours_better' if q < 0.05 and t.get('hl', 0) > 0 else
                        'ours_worse' if q < 0.05 and t.get('hl', 0) < 0 else 'no_significant_difference')
    regt = H['H3_H4']['EX1c_tests']
    cmp = []
    for mine, theirs in zip(tests, regt):
        key = (mine['comparison'], mine['corner'], mine['metric'])
        assert key == (theirs['comparison'], theirs['corner'], theirs['metric']), key
        cmp.append({'test': ' / '.join(key), 'pairs': [mine['pairs'], theirs['pairs']],
                    'both_complete': [mine['both_complete'], theirs['both_complete']],
                    'hl': [mine.get('hl'), theirs.get('hl')], 'p_holm': [mine['p_holm'], theirs.get('p_holm')],
                    'p_method_mine': mine.get('p_method'), 'verdict': [mine['verdict'], theirs['verdict']],
                    'agree': mine['verdict'] == theirs['verdict'] and mine['both_complete'] == theirs['both_complete']
                             and (mine.get('hl') is None or abs(mine['hl'] - theirs['hl']) < 1e-12)})
    # H5
    d5 = []
    for fam in ('EX1a', 'EX1c'):
        cell = collections.defaultdict(dict)
        for c in cases:
            if c['family'] == fam and c['method'] in ('A-H', 'A-S'):
                k = tuple(sorted((a, b) for a, b in c['cell'].items() if a != 'variant'))
                cell[k][c['method']] = A[c['id']]
        for m in cell.values():
            if is_complete(m.get('A-H')) and is_complete(m.get('A-S')):
                d5.append(m['A-H']['last_done'] - m['A-S']['last_done'])
    h5 = {'n': len(d5), 'median': float(np.median(d5)) if d5 else None}
    h5['agree'] = h5['n'] == H['H5'].get('n') and (h5['median'] is None or abs(h5['median'] - H['H5']['median']) < 1e-12)
    res = {'tests_agree': all(c['agree'] for c in cmp), 'tests': cmp, 'H5': h5}
    Path(outp).write_text(json.dumps(res, indent=1), encoding='utf-8', newline='\n')
    print(json.dumps({'tests_agree': res['tests_agree'], 'n_tests': len(cmp),
                      'disagreements': [c for c in cmp if not c['agree']][:5], 'H5': h5}))


if __name__ == '__main__':
    main()
