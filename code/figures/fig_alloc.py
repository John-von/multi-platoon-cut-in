"""Fig. 7: allocation ablation EX11 (A-H, Q = 3, deterministic cells).
J_v of completed runs over the initial boundary spacing B_0 for joint, sequential and independent allocation, at the
nominal parameters and with mass +10%; runs that admitted no cut-in (C1) are marked in a band below the data.
Data: independent verification rows (data/recomputed/verify/verify_raw_all.json: labels, J_v) through the registry;
the registered EX11 summary (hypotheses_addA.json) is cross-checked.
The paired random-residual differences at 30 m are not drawn: over the 50 pairs the registered estimates are 1.4 s
(95% CI 1.4 to 1.4 s) and 1.444 m (1.443 to 1.445 m), so the points would collapse onto one location; the values are
given in the text.
Usage: python fig_alloc.py <out_dir>"""
import json, sys
from pathlib import Path
import numpy as np
import figstyle as S
from data_inputs import Inputs

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
I = Inputs()
MODES = ('joint', 'sequential', 'independent')
COL = {'joint': S.METHOD['A-H'], 'sequential': '#882255', 'independent': '#999933'}
MK = {'joint': 'o', 'sequential': 's', 'independent': '^'}
LS = {'joint': '-', 'sequential': (0, (4, 2)), 'independent': (0, (1, 1.3))}
OFF = {'joint': -.25, 'sequential': 0.0, 'independent': .25}
cases = [c for c in I.family('EX11') if c['cell'].get('seed') is None and c['cell']['variant'].endswith('Q3')]
data = {}
for corner in ('nominal', 'm1.1'):
    for mode in MODES:
        xs = sorted([c for c in cases if c['cell']['corner'] == corner and c['cell']['mode'] == mode], key=lambda c: c['cell']['boundary'])
        assert len(xs) == 6
        rows = []
        for c in xs:
            lab = I.labels(c); assert lab in (['C3'], ['C1']), (c['id'], lab)
            v = I.verify[c['run_key']]; assert abs(v['Jv_recomputed'] - v['Jv_reported']) < 1e-9
            rows.append({'B0': c['cell']['boundary'], 'labels': lab, 'Jv': v['Jv_reported'] if lab == ['C3'] else None, 'id': c['id']})
        data[(corner, mode)] = rows
# registered EX11 checks
A = I.hypA['EX11']
assert A['HA11_1_joint_weakly_dominates'] and not A['counterexamples']
for t in A['HA11_2_tight']:
    b, corner, q = t['cell']
    for mode in MODES:
        r = [x for x in data[(corner, mode)] if x['B0'] == b][0]
        assert r['labels'] == t[mode], (t, mode)

S.apply()
fig, axs = S.plt.subplots(1, 2, figsize=(S.SINGLE_IN, 2.1), sharey=True)
fig.subplots_adjust(left=.13, right=.985, bottom=.2, top=.975, wspace=.08)
YB = 6.5                                               # band for runs without admission
for ax, corner, letter in zip(axs, ('nominal', 'm1.1'), 'ab'):
    S.axes(ax, letter)
    ax.tick_params(labelleft=(letter == 'a'))
    for mode in MODES:
        rows = data[(corner, mode)]
        b = np.array([r['B0'] for r in rows], float) + OFF[mode]
        j = np.array([np.nan if r['Jv'] is None else r['Jv'] for r in rows])
        ax.plot(b, j, color=COL[mode], ls=LS[mode], lw=1.0, marker=MK[mode], ms=3.2, mfc='white' if mode != 'joint' else COL[mode],
                mew=.8, label=mode, zorder=3)
        nb = [x for x, r in zip(b, rows) if r['Jv'] is None]
        ax.plot(nb, [YB] * len(nb), ls='none', marker='x', ms=3.6, mew=.9, color=COL[mode], zorder=3)
    ax.axhline(6.86, color='0.75', lw=.5)
    ax.set_xlim(25, 37); ax.set_xticks([26, 28, 30, 32, 34, 36])
    ax.set_xlabel('Initial boundary $B_\\mathrm{0}$ (m)', labelpad=1.5)
axs[0].set_ylim(6.15, 11.9); axs[0].set_yticks([7, 8, 9, 10, 11])
axs[0].set_ylabel('$J_v$ of completed runs (m)', labelpad=2)
axs[1].text(36.85, YB, 'not admitted', color='0.25', va='center', ha='right')
axs[1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1, handlelength=2.0)
checks = S.export(fig, OUT / 'fig_alloc')
out = {'source': 'independent verification rows (data/recomputed/verify/verify_raw_all.json) via the registries; '
                 'registered EX11 summary cross-checked (hypotheses_addA.json HA11_1/HA11_2)', 'inputs_sha256': I.files,
       'rows': {f'{k[0]}/{k[1]}': v for k, v in data.items()},
       'random_30m_registered_tests': A['HA11_3_tests'], 'random_30m_independent_admitted_of_50': A['HA11_3_independent_admitted_of_50'],
       'figure_checks': checks}
(OUT / 'fig_alloc_data.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps(checks))
