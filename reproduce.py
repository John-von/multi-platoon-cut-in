"""Run the published simulation configurations and reconstruct analysis inputs."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXP = ROOT / 'code/experiments'
RUNS = ROOT / 'data/recomputed/runs'
MAIN_KEY = 'cb13911c1dadd2de7413'
FAMILIES = ('EX1a', 'EX2a', 'EX4', 'EX1b', 'EX5', 'EX6', 'EX7', 'EX8a',
            'EX8b', 'EX3', 'EX2b', 'EX2c', 'EX2d', 'EX1c', 'EX1d')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                   cwd=ROOT, check=True)


def cases():
    return [case for name in ('cases_v1.json', 'cases_addA_v1.json')
            for case in read(EXP / name)['cases']]


def check():
    for path in (ROOT / 'code').rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for name in ('cases_v1', 'cases_addA_v1'):
        actual = hashlib.sha256((EXP / (name + '.json')).read_bytes()).hexdigest()
        expected = (EXP / (name + '.sha256')).read_text().split()[0]
        if actual != expected:
            raise RuntimeError(f'Configuration checksum mismatch: {name}')
    tree = ast.parse((EXP / 'run_case.py').read_text(encoding='utf-8'))
    hashes = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'E1A' for t in node.targets))
    for name, expected in hashes.items():
        path = ROOT / 'code/sim/sources' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Simulation input checksum mismatch: {name}')
    print(f'Checked source syntax, frozen inputs and {len({c["run_key"] for c in cases()})} unique runs.')


def require_runs():
    keys = sorted({c['run_key'][:20] for c in cases()})
    missing = [key for key in keys if any(not (RUNS / key / name).is_file()
               for name in ('case.json', 'result.json', 'arrays.npz', 'audit.json'))]
    if missing:
        raise RuntimeError(f'{len(missing)} runs are incomplete. Run the simulate stage first.')


def demo():
    case = next(c for c in cases() if c['run_key'].startswith(MAIN_KEY))
    dest = RUNS / MAIN_KEY
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / 'result.json').exists():
        (dest / 'case.json').write_text(json.dumps(case, indent=2), encoding='utf-8')
        run('code/experiments/run_case.py', dest / 'case.json', dest)
    result = read(dest / 'result.json')
    if result.get('_meta', {}).get('exception'):
        raise RuntimeError(result['_meta']['exception'])
    run('code/experiments/audit_run.py', dest)
    audit = read(dest / 'audit.json')
    if 'C3' not in audit['labels'] or set(audit['labels']) & {'P', 'M', 'K', 'X', 'I'}:
        raise RuntimeError(f'Main-scenario audit failed: {audit["labels"]}')
    print(f'Main-scenario outputs: {dest.relative_to(ROOT)}')


def simulate(jobs):
    if jobs < 1:
        raise ValueError('--jobs must be positive')
    for family in FAMILIES:
        run('code/experiments/run_family.py', RUNS, '--family', family, '--jobs', jobs)
    for family in ('EX11', 'EX10', 'EX9'):
        run('code/experiments/run_family_addA.py', RUNS, '--family', family, '--jobs', jobs)
    require_runs()


def analyze():
    require_runs()
    out = ROOT / 'data/recomputed'
    for folder in ('analysis', 'analysis_addA', 'verify'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    run('code/experiments/analyze.py', RUNS, '--out', out / 'analysis')
    run('code/experiments/analyze_addA.py', RUNS, '--out', out / 'analysis_addA')
    run('code/experiments/verify_raw.py', RUNS, out / 'verify/verify_raw_all.json')
    run('code/experiments/verify_stats.py', RUNS, out / 'analysis/hypotheses.json',
        out / 'verify/verify_stats.json')
    run('code/experiments/results_extra.py', RUNS, out / 'analysis/results_extra.json')
    run('code/postprocess/extract_figure_data.py')
    diag = ROOT / 'data/lateral_diag'
    diag.mkdir(parents=True, exist_ok=True)
    keys = list({(c['run_key'][:20], c['cell']['variant'])
                 for c in cases() if c['family'] == 'EX9'})
    keys.sort()
    (diag / 'ex9_keys.json').write_text(json.dumps(keys), encoding='utf-8')
    temporary = diag / 'ex9_post_next.jsonl'
    if temporary.exists():
        raise FileExistsError(temporary)
    run('code/postprocess/ex9_post.py', diag / 'ex9_keys.json', 0, len(keys), temporary, RUNS)
    os.replace(temporary, diag / 'ex9_post_final.jsonl')
    run('code/experiments/make_ex10_table.py', out / 'analysis_addA/hypotheses_addA.json',
        out / 'table_iv.tex')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('check', 'demo', 'simulate', 'analyze', 'figures'))
    parser.add_argument('--jobs', type=int, default=2, help='Concurrent simulation cases')
    parser.add_argument('--figures', type=int, nargs='+', default=[4, 5, 6, 7, 8])
    parser.add_argument('--b2-solution', type=Path, help='Selected B2 solution.npz for Figure 9')
    args = parser.parse_args()
    check()
    if args.stage == 'demo':
        demo()
    elif args.stage == 'simulate':
        simulate(args.jobs)
    elif args.stage == 'analyze':
        analyze()
    elif args.stage == 'figures':
        extra = ['--figures', *map(str, args.figures)]
        if args.b2_solution:
            extra += ['--b2-solution', str(args.b2_solution.resolve())]
        run('code/figures/make_figures.py', *extra)


if __name__ == '__main__':
    main()
