"""Fig. 6: certified range of EX2a (A-H) over the residual level s and the noise level l.
Data: registered analysis outputs only (data/recomputed/analysis/hypotheses.json, H6 rows: observed initial
certification, first merge horizon, labels) and the registry. Cell value = first merge horizon (s); C0 cells hatched.
Usage: python fig_range.py <out_dir>"""
import json, sys
from pathlib import Path
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import figstyle as S
from data_inputs import Inputs

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
I = Inputs()
rows = I.hyp['H6']['rows']
assert len(rows) == 50 and I.hyp['H6']['prediction_agreement'] == 50
SV = sorted({r['s'] for r in rows}); LV = sorted({r['ell'] for r in rows})
grid = {}
for r in rows:
    assert r['predicted_certified'] == r['observed_certified']
    lab = r['labels']
    if r['observed_certified']:
        assert lab == ['C3'] and r['first_T'] is not None
    else:
        assert lab == ['C0'] and r['first_T'] is None
    grid[(r['corner'], r['s'], r['ell'])] = r
# cross-check against the independent verification rows (labels per run key)
for c in I.family('EX2a'):
    key = (c['cell']['corner'], c['cell']['s'], c['cell']['ell'])
    assert I.labels(c) == grid[key]['labels'], key
TS = sorted({r['first_T'] for r in rows if r['first_T']})
lo, hi = min(TS), max(TS)
cmap = LinearSegmentedColormap.from_list('seq', ['#E8F0F8', '#9CC0DD', '#4A86B8', '#1F4E79'])

S.apply()
fig, axs = S.plt.subplots(1, 2, figsize=(S.SINGLE_IN, 1.72))
fig.subplots_adjust(left=.12, right=.985, bottom=.2, top=.975, wspace=.10)
for ax, corner, letter in zip(axs, ('nominal', 'high'), 'ab'):
    S.axes(ax, None, grid=False, ticks=False); ax.grid(False)
    for j, s in enumerate(SV):
        for i, l in enumerate(LV):
            r = grid[(corner, s, l)]
            if r['first_T'] is None:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor='0.95', edgecolor='0.72', hatch='////', lw=0))
            else:
                u = (r['first_T'] - lo) / (hi - lo)
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=cmap(u), edgecolor='none'))
                corner_cell = (j == 0 and i == len(LV) - 1)     # holds the panel letter: number moved to the lower right
                ax.text(j + (.40 if corner_cell else 0), i - (.36 if corner_cell else 0), f"{r['first_T']}",
                        ha='right' if corner_cell else 'center', va='bottom' if corner_cell else 'center',
                        color='white' if u > .55 else 'black')
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor='none', edgecolor='white', lw=.8))
    ax.set_xlim(-.5, len(SV) - .5); ax.set_ylim(-.5, len(LV) - .5); ax.set_aspect('auto')
    ax.set_xticks(range(len(SV))); ax.set_xticklabels([str(s) for s in SV])
    ax.set_yticks(range(len(LV))); ax.set_yticklabels([str(l) for l in LV] if letter == 'a' else [])
    ax.tick_params(length=0, pad=2, top=False, right=False)
    ax.set_xlabel('Residual level $s$', labelpad=1.5)
    ax.text(-.5 + .05, len(LV) - .5 - .05, f'({letter})', ha='left', va='top', fontweight='bold', zorder=10,
            color='black')
axs[0].set_ylabel('Noise level $\\ell$', labelpad=2)
# the non-monotone horizon of Section VII-D: high corner, s = 1, l = 5 (14 s) versus l = 7 (12 s)
ax = axs[1]
dec = I.hyp['H6']['first_T_decreases']
assert dec == [{'corner': 'high', 'vary': 'ell', 's': 1, 'from': [5, 14], 'to': [7, 12]}]
j = SV.index(1)
for l in (5, 7):
    i = LV.index(l)
    ax.add_patch(Rectangle((j - .5 + .06, i - .5 + .06), .88, .88, facecolor='none', edgecolor='black', lw=.9, ls=(0, (2, 1.2))))
checks = S.export(fig, OUT / 'fig_range')
data = {'source': 'registered analysis (data/recomputed/analysis/hypotheses.json H6), labels cross-checked against '
                  'data/recomputed/verify/verify_raw_all.json', 'inputs_sha256': I.files,
        'grid': [{'corner': c, 's': s, 'ell': l, 'first_T': grid[(c, s, l)]['first_T'], 'labels': grid[(c, s, l)]['labels'],
                  'last_done': grid[(c, s, l)]['last_done']} for c in ('nominal', 'high') for s in SV for l in LV],
        'horizons': TS, 'figure_checks': checks}
(OUT / 'fig_range_data.json').write_text(json.dumps(data, indent=1), encoding='utf-8')
print(json.dumps(checks))
