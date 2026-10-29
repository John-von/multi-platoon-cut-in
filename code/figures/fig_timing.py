"""Fig. 10: longest computing time as a share of the time budget.
Each bar is the longest time over the runs of one configuration and process count, as a percentage of its budget.
Blocks: first planning task (budget 3 s), later nonempty planning tasks (budget 2 s; T1 and T4 have none), longest
vehicle-layer step of each run (sampling period 0.1 s). Light bars: 1 process (3 runs per configuration); dark bars:
16 processes (10 runs). Missed deadlines are labelled inside the bars.
Data: data/timing/planning_jobs.csv and run_summary.csv (written by code/timing/recompute_timing.py).
Usage: python fig_timing.py <planning_jobs.csv> <run_summary.csv> <out_dir>"""
import csv, json, sys, hashlib
from pathlib import Path
import numpy as np
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
import figstyle as S

JOBS, RUNS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
read = lambda p: list(csv.DictReader(p.open(encoding='utf-8-sig', newline='')))
jobs, runs = read(JOBS), read(RUNS)
COL = {1: '#A6CDE8', 16: S.METHOD['A-H']}
MISS = '#B2182B'
S.apply()
fig, ax = S.plt.subplots(figsize=(S.SINGLE_IN, 3.65))
fig.subplots_adjust(left=.085, right=.975, bottom=.105, top=.985)
S.axes(ax, None, grid=False)
groups = [('First planning task (budget 3 s)', 'first', ['T1', 'T2', 'T3', 'T4']),
          ('Later planning tasks (budget 2 s)', 'subsequent', ['T2', 'T3']),
          ('Longest vehicle step (sampling period 0.1 s)', 'vehicle', ['T1', 'T2', 'T3', 'T4'])]
H, DY, PITCH = .5, .3, 1.32
y = 0.0; yt, yl, stats = [], [], {}
for title, stage, configs in groups:
    ax.text(1.0, y + .05, title, ha='left', va='center', color='#333333')
    y += 1.0
    for c in configs:
        for w, dy in [(1, -DY), (16, DY)]:
            if stage == 'vehicle':
                vals = np.array([float(r['control_max_ms']) / 100 * 100 for r in runs if r['config'] == c and int(r['workers']) == w])
                miss = np.zeros(len(vals), bool)
            else:
                jj = [j for j in jobs if j['config'] == c and int(j['workers']) == w and j['stage'] == stage and j['nonempty'] == 'True']
                vals = np.array([float(j['finish_from_scheduled_dispatch_s']) / float(j['nominal_budget_s']) * 100 for j in jj])
                miss = np.array([j['deadline_missed'] == 'True' for j in jj])
            assert len(vals) == (3 if w == 1 else 10), (stage, c, w, len(vals))
            v = float(vals.max()); yy = y + dy
            ax.barh(yy, v, height=H, color=COL[w], edgecolor='none', zorder=2)
            if miss.any():                              # inside the bar, left of the deadline line
                ax.text(v - 2.0, yy, f'{v:.2f}%, {int(miss.sum())} missed', ha='right', va='center', color=MISS, zorder=4)
            elif v > 80:                                # long bar: value inside the bar
                ax.text(v - 2.0, yy, f'{v:.2f}%', ha='right', va='center', color='#333333', zorder=4)
            else:
                ax.text(v + 1.2, yy, f'{v:.2f}%', ha='left', va='center', color='#333333', zorder=4)
            stats[f'{stage}/{c}/{w}'] = {'n': int(len(vals)), 'min_pct': float(vals.min()), 'max_pct': v,
                                         'missed': int(miss.sum())}
        yt.append(y); yl.append(c); y += PITCH
    y += .25
ax.axvline(100, color=S.REF, lw=.7, ls=(0, (4, 3)), zorder=1.5)
ax.text(98.5, -.05, 'deadline', ha='right', va='center')
ax.set_yticks(yt); ax.set_yticklabels(yl); ax.set_ylim(y - .45, -.6)
ax.tick_params(axis='y', length=0)
ax.set_xlim(0, 106); ax.set_xticks([0, 25, 50, 75, 100])
ax.grid(axis='x', color=S.GRID, linewidth=.35)
ax.set_xlabel('Longest time as a share of the budget (%)', labelpad=2)
ax.legend(handles=[Patch(facecolor=COL[1], edgecolor='none', label='1 process'),
                   Patch(facecolor=COL[16], edgecolor='none', label='16 processes')],
          loc='lower right', bbox_to_anchor=(98.5, 0.0), bbox_transform=blended_transform_factory(ax.transData, ax.transAxes),
          ncol=1, handlelength=1.3, handleheight=.8, labelspacing=.3)   # left of the deadline line
# numbers quoted in the text (Section V): 16 processes at most 53.94 % (first) and 11.82 % (later), vehicle step at
# most 35.99 %, and five missed deadlines with one process (T2 and T3 first tasks)
assert f"{max(stats[f'first/{c}/16']['max_pct'] for c in ['T1', 'T2', 'T3', 'T4']):.2f}" == '53.94'
assert f"{max(stats[f'subsequent/{c}/16']['max_pct'] for c in ['T2', 'T3']):.2f}" == '11.82'
assert f"{max(s['max_pct'] for k, s in stats.items() if k.startswith('vehicle/')):.2f}" == '35.99'
assert sum(s['missed'] for s in stats.values()) == 5 and stats['first/T2/1']['missed'] == 3 and stats['first/T3/1']['missed'] == 2
checks = S.export(fig, OUT / 'fig_timing')
out = {'sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (JOBS, RUNS)}, 'stats': stats,
       'figure_checks': checks}
(OUT / 'fig_timing_data.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps({k: round(v['max_pct'], 2) for k, v in stats.items()})); print(checks)
