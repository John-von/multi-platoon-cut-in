"""Generate the EX10 table of the English manuscript (tab_ex10.tex) from the registered outputs.
The Chinese manuscript uses the same rows with a Chinese header (tab_ex10_zh.tex).

Usage: python make_ex10_table.py <analysis_addA/hypotheses_addA.json> <out.tex>
Rows: the eight geometries; columns: platoon sizes, N, chain margin at 30 m (static, prescreen_qn.json; range over the
three scene families), closed-loop outcome of A-H at 30 m, the smallest LP-restoring spacing B_min (static, 2 m grid),
and the outcome at B_min and B_min+4.  Outcome cells count the scene families whose run completed all cut-ins (C3), and
name any other label."""
import json, sys, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ['Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'N_x1p5', 'N_x2', 'N_x3']
NAME = {'Q2': '$Q=2$', 'Q3': '$Q=3$', 'Q4': '$Q=4$', 'Q5': '$Q=5$', 'Q6': '$Q=6$',
        'N_x1p5': '$\\times1.5$', 'N_x2': '$\\times2$', 'N_x3': '$\\times3$'}


def cell(rows):
    """Number of scene families whose run completed all cut-ins (C3) without P/M/K, out of the runs in the cell;
    any other label is appended (e.g. '0/3 C1')."""
    if not rows:
        return '--'
    ok = sum(1 for r in rows if 'C3' in r['labels'] and not ({'P', 'M', 'K', 'X', 'I'} & set(r['labels'])))
    other = sorted({x for r in rows for x in r['labels'] if x in ('C0', 'C2', 'P', 'M', 'K', 'X', 'I')})
    return f"{ok}/{len(rows)}" + (f" ({'+'.join(other)})" if other else '')


def rng(vals, fmt):
    lo, hi = min(vals), max(vals)
    return fmt.format(lo) if lo == hi else (fmt.format(lo) + '/' + fmt.format(hi))


def main():
    H = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))['EX10']['rows']
    pre = json.loads((HERE / 'prescreen_qn.json').read_text(encoding='utf-8'))['rows']
    P = collections.defaultdict(dict)
    for r in pre:
        P[r['geometry']][r['family']] = r
    lines = ['% Statistics computed from registered run records.']
    for g in ORDER:
        fams = P[g]; any_r = next(iter(fams.values()))
        margins = [r['chain_margin_b30'] for r in fams.values()]
        bstar = [r['min_boundary_lp_feasible'] for r in fams.values()]
        at30 = [r for r in H if r['geometry'] == g and abs(r['boundary'] - 30.0) < 1e-9]
        if min(bstar) > 30:
            at_b = [r for r in H if r['geometry'] == g and abs(r['boundary'] - P[g][r['family_scene']]['min_boundary_lp_feasible']) < 1e-9]
            at_b4 = [r for r in H if r['geometry'] == g and abs(r['boundary'] - P[g][r['family_scene']]['min_boundary_lp_feasible'] - 4) < 1e-9]
            bs = rng(bstar, '{:.0f}')
        else:
            at_b = at_b4 = []; bs = '$\\le$30'
        sizes = '/'.join(str(x) for x in any_r['queue_sizes']) if len(any_r['queue_sizes']) <= 3 else f"8/7$\\times${len(any_r['queue_sizes']) - 1}"
        lines.append(f"{NAME[g]} & {any_r['N']} & ${min(margins):.2f}$ & {cell(at30)} & {bs.replace('/', '--')} & {cell(at_b)} & {cell(at_b4)}\\\\")
    head = ['\\begin{tabular}{@{}lrrcccc@{}}', '\\toprule',
            'Configuration & Vehicles & Margin (m) & 30~m & $B_{\\min}$ (m) & $B_{\\min}$ & $B_{\\min}{+}4$~m\\\\', '\\midrule']
    lines = lines[:1] + head + lines[1:] + ['\\bottomrule', '\\end{tabular}']
    Path(sys.argv[2]).write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
