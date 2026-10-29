"""Fig. 5: necessity of certification (EX4, tight initial spacing).
(a) Largest vehicle speed in the 10.25 m, high-corner cell (kappa = 1) for N and A-H: recomputed archive
    (data/extracted/trajectories.json).
(b) Outcomes over the 42 EX4 cells: independent verification rows, cross-checked against the registered H2 rows;
    outcome categories are named in a legend.
Usage: python fig_need.py <out_dir> <extracted_data_dir>"""
import json, sys
from pathlib import Path
import numpy as np
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
import figstyle as S
from data_inputs import Inputs

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
I = Inputs(with_d=True, d_dir=sys.argv[2])
F_CELL = 1.0
VMAX = 23.0
CERT = ('A-H', 'A-S', 'F', 'DDRPC')
counts = {}
for m in CERT + ('N', 'P'):
    cnt = {'C3': 0, 'P': 0, 'C1': 0, 'C0': 0}
    for c in I.family('EX4'):
        if c['method'] != m:
            continue
        lab = set(I.labels(c))
        k = 'P' if 'P' in lab else [x for x in ('C3', 'C2', 'C1', 'C0') if x in lab][0]
        cnt[k] += 1
    assert sum(cnt.values()) == 42, (m, cnt)
    counts[m] = cnt
assert all(counts[m] == counts['A-H'] for m in CERT)
h2 = I.hyp['H2_A-H']['rows']
viol_cells = {(r['cell']['gap'], r['cell']['corner'], r['cell']['f']): sorted(r['violating']) for r in h2}
assert sum('N' in v for v in viol_cells.values()) == counts['N']['P'] and sum('P' in v for v in viol_cells.values()) == counts['P']['P']
traj = {}
for c in I.family('EX4', gap=10.25, corner='high', boundary=30, f=F_CELL):
    if c['method'] in ('N', 'A-H'):
        e = I.d_traj.get(c['id'])
        assert e is not None, f'missing recomputed trajectory {c["id"]}'
        traj[c['method']] = (np.array(e['time']), np.array(e['max_speed']), c)

S.apply()
fig = S.plt.figure(figsize=(S.SINGLE_IN, 2.9))
ax1 = fig.add_axes([.175, .6, .8, .375]); ax2 = fig.add_axes([.175, .125, .8, .35])
S.axes(ax1, 'a'); S.axes(ax2, 'b', grid=False)
t, vN, cN = traj['N']; tA, vA, cA = traj['A-H']
ax1.plot(tA, vA, color=S.METHOD['A-H'], lw=1.1, label='A-H (certified)', zorder=3)
ax1.plot(t, vN, color=S.METHOD['N'], lw=1.1, ls=(0, (4, 1.5)), label='N (nominal, no envelope)', zorder=3)
over = vN > VMAX + 1e-9
ax1.fill_between(t, VMAX, vN, where=over, color=S.METHOD['N'], alpha=.35, lw=0, zorder=2)
S.bound(ax1, VMAX, 'speed limit 23 m/s', x=.985)
k = int(np.flatnonzero(over)[0]); kmax = int(np.argmax(vN))
h2row = [r for r in h2 if r['cell']['gap'] == 10.25 and r['cell']['corner'] == 'high' and r['cell']['f'] == F_CELL][0]
assert abs(h2row['violating']['N']['speed'] - float(t[k])) < 1e-9
ax1.annotate(f'+{vN[kmax] - VMAX:.2f} m/s', xy=(t[kmax], vN[kmax]), xytext=(t[kmax] + 3.2, vN[kmax] + .07),
             color=S.METHOD['N'], arrowprops={'arrowstyle': '-', 'color': S.METHOD['N'], 'lw': .6})
ax1.set_xlim(5, 30); ax1.set_xticks([5, 10, 15, 20, 25, 30]); ax1.set_ylim(22.55, 23.24); ax1.set_yticks([22.6, 22.8, 23.0, 23.2])
ax1.set_xlabel('Time (s)', labelpad=1.5); ax1.set_ylabel('Max. speed (m/s)', labelpad=2)
# legend right of both curves, centered between the 22.6 and 22.8 m/s grid lines
ax1.legend(loc='center right', bbox_to_anchor=(1.0, 22.7), bbox_transform=blended_transform_factory(ax1.transAxes, ax1.transData),
           handlelength=2.2)
ORDER = [('A-H, A-S,\nF, D-DRPC', counts['A-H']), ('N', counts['N']), ('P', counts['P'])]
SEG = [('C3', '#7FA7CF', None, 'no violation'), ('P', S.METHOD['N'], None, 'speed limit exceeded'),
       ('C1', '0.80', None, 'no cut-in admitted'), ('C0', '0.96', '////', 'not certified')]
YPOS = [-.08, 1.3, 2.3]                               # extra room for the two-line first label
for y, (name, cnt) in zip(YPOS, ORDER):
    left = 0
    for key, col, hatch, _ in SEG:
        n = cnt[key]
        if n:
            ax2.barh(y, n, left=left, height=.62, color=col, edgecolor='0.55' if hatch else 'white', hatch=hatch, lw=.5, zorder=2)
            ax2.text(left + n / 2, y, f'{n}', ha='center', va='center', color='white' if key == 'P' else 'black',
                     bbox={'facecolor': col, 'edgecolor': 'none', 'pad': .3} if hatch else None, zorder=3)
        left += n
ax2.set_yticks(YPOS); ax2.set_yticklabels([n for n, _ in ORDER], linespacing=.95); ax2.set_ylim(2.8, -2.12)
ax2.set_xlim(0, 42); ax2.set_xticks([0, 6, 12, 18, 24, 30, 36, 42])
ax2.vlines([6, 12, 18, 24, 30, 36], 2.8, YPOS[0] - .31, color=S.GRID, lw=.35, zorder=0)   # grid over the bars only, not under the legend
ax2.tick_params(axis='y', length=0, right=False)
ax2.set_xlabel('EX4 cells (of 42)', labelpad=1.5)
handles = [Patch(facecolor=col, edgecolor='0.55' if hatch else col, hatch=hatch, lw=.5, label=lab) for _, col, hatch, lab in SEG]
ax2.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=2, handlelength=1.3, handleheight=.8,
           columnspacing=.7, labelspacing=.2)
checks = S.export(fig, OUT / 'fig_need')
out = {'panel_a': {'source': 'recomputed archive, data/extracted/trajectories.json',
                   'cell': {'gap': 10.25, 'corner': 'high', 'f': F_CELL, 'boundary': 30}, 'ids': {'N': cN['id'], 'A-H': cA['id']},
                   'N_first_over_limit_s': float(t[k]), 'N_max_speed': float(vN[kmax]), 'AH_max_speed': float(vA.max())},
       'panel_b': {'source': 'independent verification rows; H2 rows cross-checked', 'counts': counts},
       'inputs_sha256': I.files, 'figure_checks': checks}
(OUT / 'fig_need_data.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps(out['panel_a'])); print(json.dumps(out['panel_b']['counts'])); print(json.dumps(checks))
