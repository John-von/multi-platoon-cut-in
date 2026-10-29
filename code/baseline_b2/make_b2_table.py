"""Table of the B2 planning-level acceptance (nine vehicles) from the B2 run records
(data/baseline_b2/runs/single/G*/...); the values are generated, not typed.
Usage: python make_b2_table.py <baseline_b2 dir holding runs/> <out.tex> <out.json>"""
import glob, json, sys
from pathlib import Path

B = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUTJ = Path(sys.argv[3])
GS = (16, 22, 28); VS = ('main', 'sensitivity', 'original')
NAME = {'main': 'main (1~s window jerk)', 'sensitivity': 'sensitivity (node jerk bound)', 'original': 'original (no jerk limit)'}


def calls_of(folder):
    out = []
    for f in sorted(glob.glob(str(folder / '**' / 'result.json'), recursive=True)):
        r = json.loads(Path(f).read_text(encoding='utf-8'))
        if 'solve_wall_s' in r:
            out.append(r)
    return out


rows = {}
for v in VS:
    for g in GS:
        d = B / 'runs' / 'single' / f'G{g}'
        res = json.loads((d / v / 'result.json').read_text(encoding='utf-8'))
        st = json.loads((d / v / 'fixed' / 'fixed_stage.json').read_text(encoding='utf-8'))
        assert res['status'] == 'accepted_plan'
        cand = st['candidates']
        acc = sorted(float(k) for k, c in cand.items() if c['accepted']); rej = sorted(float(k) for k, c in cand.items() if not c['accepted'])
        calls = calls_of(d / v)                       # free call (main/original) and every fixed-t_f call of this version
        if v in ('main', 'sensitivity'):
            calls += calls_of(d / 'sensitivity' / 'call_1') if v == 'main' else []   # S0 is shared by main and sensitivity
        statuses = sorted({c.get('return_status') for c in calls})
        rows[(v, g)] = {'tf': res['tf'], 'objective': res['objective_raw_lambda100_gamma1'], 'start_tf': st['start_tf'],
                        'accepted': acc, 'rejected': rej, 'n_calls': len(calls),
                        'solver_h': sum(c['solve_wall_s'] for c in calls) / 3600, 'build_h': sum(c.get('build_wall_s') or 0 for c in calls) / 3600,
                        'statuses': statuses}
best = {v: min(GS, key=lambda g: rows[(v, g)]['objective']) for v in VS}
lines = ['% Statistics computed from registered run records.', '\\begin{tabular}{@{}llrrrrrr@{}}', '\\toprule',
         'Version & Guess & Start $t_f$ (s) & Plan $t_f$ (s) & Objective & Candidates & Calls & Solver (h)\\\\', '\\midrule']
for v in VS:
    for i, g in enumerate(GS):
        r = rows[(v, g)]; mark = '$^{*}$' if best[v] == g else ''
        lines.append(f"{NAME[v] if i == 0 else ''} & G{g} & {r['start_tf']:.3f} & {r['tf']:.3f}{mark} & {r['objective']:.2f} & "
                     f"{len(r['accepted'])}/{len(r['accepted']) + len(r['rejected'])} & {r['n_calls']} & {r['solver_h']:.2f}\\\\")
    if v != VS[-1]:
        lines.append('\\addlinespace')
lines += ['\\bottomrule', '\\end{tabular}']
OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
OUTJ.write_text(json.dumps({f'{v}/G{g}': r for (v, g), r in rows.items()} | {'best': best}, indent=1), encoding='utf-8')
print('\n'.join(lines))
