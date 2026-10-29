"""Generate supp_timing_table.tex (supplementary table of the host timing study WP7b-v20) from the raw result.json
files of the 52 host runs (data/timing/host_runs). Pooled control-step percentiles use the per-step control_steps.csv
written by recompute_timing.py; the script asserts that its per-run 99th percentile, maximum, and sample counts equal the raw per-run summaries.

Usage: python make_timing_table.py <host_runs_dir> <control_steps.csv> <out.tex>"""
import json, glob, os, csv, collections, sys
import numpy as np
W, CSV, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
runs = {os.path.basename(d): json.load(open(os.path.join(d, 'result.json'), encoding='utf-8')) for d in sorted(glob.glob(os.path.join(W, 'T*_*')))
        if 'replay' not in d and os.path.isfile(os.path.join(d, 'result.json'))}
steps = collections.defaultdict(list)
for r in csv.DictReader(open(CSV, encoding='utf-8-sig')):
    steps[r['run']].append(float(r['control_ms']))
for n, r in runs.items():   # consistency with raw per-run summaries
    c = np.array(steps[n]); assert len(c) == r['control_samples']
    assert abs(np.percentile(c, 99) - r['control_p99_s'] * 1e3) < 1e-9 and abs(c.max() - r['control_max_s'] * 1e3) < 1e-9
ne = lambda j: isinstance(j.get('search'), list) and len(j['search']) > 0
f3 = lambda x: '%.3f' % x
lines = [r'\begin{tabular}{@{}llrrcccrc@{}}', r'\toprule',
         r' & & & Tasks & \multicolumn{2}{c}{First task (s)} & Later & Missed & Vehicle-layer step (ms)\\',
         r'\cmidrule(lr){5-6}',
         r'Config. & Proc. & Runs & (nonempty) & comp.\ median/max & done after dispatch & max (s) & deadlines & pooled p99 / run p99 / max\\',
         r'\midrule']
for T in ('T1', 'T2', 'T3', 'T4'):
    for w, lab in (('16w', '16'), ('1w', '1')):
        names = [n for n in runs if n.startswith(T + '_' + w + '_')]
        J = [j for n in names for j in runs[n]['planning_jobs']]
        first = [runs[n]['planning_jobs'][0] for n in names]
        later = [j['compute_seconds'] for n in names for j in runs[n]['planning_jobs'][1:] if ne(j)]
        pooled = np.percentile(np.concatenate([steps[n] for n in names]), 99)
        lines.append(' & '.join([T, lab, str(len(names)), '%d (%d)' % (len(J), sum(ne(j) for j in J)),
            f3(np.median([j['compute_seconds'] for j in first])) + '/' + f3(max(j['compute_seconds'] for j in first)),
            f3(max(j['worker_finished_wall_s'] - j['dispatch_s'] for j in first)),
            f3(max(later)) if later else '--', str(sum(bool(j.get('deadline_missed')) for j in J)),
            '%.2f / %.2f / %.2f' % (pooled, max(runs[n]['control_p99_s'] * 1e3 for n in names),
                                    max(runs[n]['control_max_s'] * 1e3 for n in names))]) + r'\\')
lines += [r'\bottomrule', r'\end{tabular}']
open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines) + '\n')
print(open(OUT, encoding='utf-8').read())
