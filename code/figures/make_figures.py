"""Draw selected data figures from locally generated simulation results."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D = ROOT / 'data'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out_dir', nargs='?', type=Path, default=ROOT / 'figures')
    parser.add_argument('--figures', type=int, nargs='+', choices=range(4, 11), default=[4, 5, 6, 7, 8])
    parser.add_argument('--b2-solution', type=Path)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    run = D / 'recomputed/runs/cb13911c1dadd2de7413'
    jobs = {
        4: ['fig_main.py', run, out],
        5: ['fig_need.py', out, D / 'extracted'],
        6: ['fig_range.py', out],
        7: ['fig_alloc.py', out],
        8: ['fig_lateral.py', out, D / 'extracted', D / 'lateral_diag/ex9_post_final.jsonl'],
        9: ['fig_b2.py', args.b2_solution, run, out],
        10: ['fig_timing.py', ROOT / 'code/timing/out/planning_jobs.csv',
             ROOT / 'code/timing/out/run_summary.csv', out],
    }
    if 9 in args.figures and (args.b2_solution is None or not args.b2_solution.is_file()):
        parser.error('Figure 9 requires --b2-solution pointing to a completed B2 solution.npz.')
    if not (HERE / 'fonts/texgyretermes-regular.ttf').exists():
        subprocess.run([sys.executable, str(HERE / 'otf2ttf.py'), str(HERE / 'fonts')], check=True)
    out.mkdir(parents=True, exist_ok=True)
    for number in sorted(set(args.figures)):
        script, *arguments = jobs[number]
        subprocess.run([sys.executable, str(HERE / script), *map(str, arguments)], cwd=HERE, check=True)
    if 5 in args.figures:
        subprocess.run([sys.executable, str(HERE / 'tab_methods.py'), str(D / 'extracted'),
                        str(out / 'fig_need_data.json'), str(out)], cwd=HERE, check=True)


if __name__ == '__main__':
    main()
