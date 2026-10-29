"""Generate manuscript/results_macros.tex from the registered audits.

Usage: python make_results_macros.py <runs_root> <out.tex>
Defines \\res@<id> for the table cells of the manuscript:
  EX1a.<condition>.<method>   't_done/J_v' for completed runs (1 and 3 decimals), otherwise the outcome label(s)
  H1.<AH|AS|F|DD>.<n|P|M|K|E|cp|C0|X>   H1/H1' counts per method (theory-covered, unique run keys, both registries;
                                         only A-H has runs in addendum A, so the other rows equal the main registry)"""
import json, sys
from pathlib import Path
import analyze as AN

COND = ['nominal', 'm0.9', 'm1.1', 't0.7', 't1.3', 'low', 'high', 'H1801', 'H1802', 'H1803', 'H1804']
METH = ['A-H', 'A-S', 'F', 'DDRPC', 'N', 'P']


def cell_text(a):
    if a is None:
        return 'n/a'
    L = AN.labels(a)
    tag = ''.join(x for x in ('P', 'M', 'K') if x in L)
    if 'C3' in L:
        txt = f"{a['last_done']:.1f}/{a['Jv']:.3f}"
    else:
        txt = next((x for x in ('C0', 'C1', 'C2', 'I', 'X') if x in L), '?')
    return txt + (f'$^{{\\mathrm{{{tag}}}}}$' if tag else '')


def main():
    root, out = sys.argv[1], Path(sys.argv[2])
    D = AN.Data(root, str(AN.HERE / 'cases_v1.json'))
    import analyze_addA as AA
    DP = AA.Data2(root)   # v1 + addendum A: H1 counts over all registered theory-covered runs (pooled, as in analyze_addA)
    vals = {}
    for c in D.select(family='EX1a'):
        cond = c['cell']['corner']; m = c['method']
        if m in METH and cond in COND:
            vals[f'EX1a.{cond}.{m}'] = cell_text(D.audit(c))
    for key, methods in (('AH', ('A-H',)), ('AS', ('A-S',)), ('F', ('F', 'TUBE')), ('DD', ('DDRPC',))):
        h = AN.h1(DP, methods); cnt = h['counts']
        lo, hi = h['rate_95ci']
        vals.update({f'H1.{key}.n': '{:,}'.format(cnt.get('evaluable', 0)).replace(',', '{,}'), f'H1.{key}.P': str(cnt.get('P', 0)),
                     f'H1.{key}.M': str(cnt.get('M', 0)), f'H1.{key}.K': str(cnt.get('K', 0)),
                     f'H1.{key}.E': str(cnt.get('E', 0)), f'H1.{key}.C0': str(cnt.get('initial_uncertified', 0)),
                     f'H1.{key}.X': str(cnt.get('exception', 0)),
                     f'H1.{key}.cp': '--' if hi is None else f'[{lo:.3f}, {hi:.3f}]'})
    lines = ['% Statistics computed from registered run records.', '\\makeatletter']
    lines += [f'\\@namedef{{res@{k}}}{{{v}}}' for k, v in sorted(vals.items())]
    lines += ['\\makeatother', '']
    out.write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print(f'{len(vals)} macros -> {out}')


if __name__ == '__main__':
    main()
