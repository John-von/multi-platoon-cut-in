"""Generate supplement tables from the registered analysis outputs.

Usage: python make_supp_tables.py <analysis/hypotheses.json> <analysis_addA/hypotheses_addA.json> <out_dir>
Writes supp_ex1c_tests.tex (the 24 registered H3/H4 tests), supp_ex4_cells.tex (EX4 outcome per cell and method),
supp_ha11.tex (EX11 deterministic cells)."""
import json, sys
from pathlib import Path

VER = {'ours_better': 'ours better', 'ours_worse': 'ours worse', 'no_significant_difference': 'n.s.',
       'not_evaluable': 'not evaluable'}


def f(x, fmt='{:.2f}'):
    return '--' if x is None else fmt.format(x)


def ex1c(H, out):
    L = ['% Statistics computed from registered run records.']
    TD = '$t_{\\rm done}$'; JV = '$J_v$'; EOL = '\\\\'
    for t in H['H3_H4']['EX1c_tests']:
        ci = t.get('ci')
        metric = TD if t['metric'] == 'last_done' else JV
        cis = '--' if not ci else '$[{:+.3f}, {:+.3f}]$'.format(ci[0], ci[1])
        ph = '--' if t.get('p_holm') is None else ('$<$0.001' if t['p_holm'] < 1e-3 else '{:.3f}'.format(t['p_holm']))
        comp = t['comparison'].replace('DDRPC', 'D-DRPC')
        L.append(f"{comp} & {t['corner']} & {metric} & {t['both_complete']}/{t['pairs']} & {f(t.get('hl'), '${:+.3f}$')} & "
                 f"{cis} & {ph} & {VER[t['verdict']]}" + EOL)
    head = ['\\begin{tabular}{@{}lllrrrrl@{}}', '\\toprule',
            'Comparison & Corner & Metric & Pairs & HL & 95\\% CI & $p_{\\rm Holm}$ & Verdict\\\\', '\\midrule']
    L = L[:1] + head + L[1:] + ['\\bottomrule', '\\end{tabular}']
    (out / 'supp_ex1c_tests.tex').write_text('\n'.join(L) + '\n', encoding='utf-8', newline='\n')


def ha11(HA, out):
    L = ['% Statistics computed from registered run records.']
    EOL = '\\\\'
    for t in HA['EX11']['HA11_2_tight']:
        j, s, i = '+'.join(t['joint']), '+'.join(t['sequential']), '+'.join(t['independent'])
        ok = 'yes' if t['as_predicted'] else 'no'
        L.append(f"{float(t['cell'][0]):.0f} & {t['cell'][1]} & {t['cell'][2]} & {j} & {s} & {i} & {ok}" + EOL)
    (out / 'supp_ha11_tight.tex').write_text('\n'.join(L) + '\n', encoding='utf-8', newline='\n')


def main():
    H = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    HA = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
    out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    ex1c(H, out); ha11(HA, out)
    print('written to', out)


if __name__ == '__main__':
    main()
