"""Fig. 8: lateral plant mismatch EX9 (A-H).
(a) reduced-stiffness corner (C_f, C_r x0.7, m_y x1.2, I_z x1.3) against the nominal plant;
(b) oversteering corner: each cut-in is drawn up to its first crossing of a road edge, which is marked with a cross;
    the axes show the whole road (-1.75 to 5.25 m);
(c) share of runs with each outcome per lateral-plant group (numbers in the segments are run counts):
      settled        = all cut-ins completed and each stays within 0.3 m of the target-lane centre afterwards;
      oscillating    = all cut-ins completed, on the road, but a deviation above 0.3 m after completion;
      road departure = label E (a cut-in leaves the road), whether or not the lane change was completed.
    Groups (EX9 has 145 runs): both cornering stiffnesses x0.7 (L_C3* and L_Pfr_0p7), the oversteering corner
    (L_C2, L_C2_pulse_pos, L_C2_pulse_neg) and all other plants.
The three cut-ins differ in colour and line style; the comparison runs are gray.
Data: recomputed archive (data/extracted/trajectories.json) and the diagnostic data/lateral_diag/ex9_post_final.jsonl,
cross-checked against the original EX9 rows.
Usage: python fig_lateral.py <out_dir> <extracted_data_dir> <ex9_post.jsonl>"""
import json, sys
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import figstyle as S
from data_inputs import Inputs, sha

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
I = Inputs(with_d=True, d_dir=sys.argv[2])
DIAG = Path(sys.argv[3]); I.files['recomputed/diag/' + DIAG.name] = sha(DIAG)
ORIG = {r['variant']: r for r in I.hypA['EX9']['rows']}
PREFIX = 'EX9__A-H__nominal__s1_l0_f0p5__g10p5_b30__'
EDGE_LO, EDGE_HI, LANE_LINE, SETTLED = -1.75, 5.25, 1.75, 0.3
need = ('L_C3', 'L_C2', 'L_C2_pulse_neg', 'L_P0')
tr = {}
for v in need:
    e = I.d_traj.get(PREFIX + v)
    assert e is not None and 'body_y' in e, v
    by = np.array(e['body_y']); o = ORIG[v]
    assert sorted(e['labels']) == sorted(o['labels']), (v, e['labels'], o['labels'])
    exc = max(0.0, float(by.max()) - 3.5)
    if 'E' not in o['labels']:
        assert abs(exc - o['corridor_excess_above_m']) <= 2e-6, (v, exc, o['corridor_excess_above_m'])
    tr[v] = {'t': np.arange(by.shape[1]) * e['body_y_dt'], 'y': by, 'labels': e['labels'], 'excess': exc}
diag = [json.loads(x) for x in DIAG.read_text(encoding='utf-8').splitlines() if x.strip()]
assert len(diag) == 145 and not any(r.get('missing') for r in diag)
for r in diag:
    assert sorted(r['labels']) == sorted(ORIG[r['variant']]['labels']), r['variant']


def first_exit(t, y):
    """First crossing of a road edge (linear interpolation); None if the trace stays on the road."""
    out = (y > EDGE_HI) | (y < EDGE_LO)
    if not out.any():
        return None
    k = int(np.flatnonzero(out)[0]); edge = EDGE_HI if y[k] > EDGE_HI else EDGE_LO
    a = (edge - y[k - 1]) / (y[k] - y[k - 1])
    return k, float(t[k - 1] + a * (t[k] - t[k - 1])), edge


GROUPS = ['other\nplants', 'reduced\nstiffness', 'oversteer']


def group(v):
    if v.startswith('L_C3') or v == 'L_Pfr_0p7':
        return 1
    if v.startswith('L_C2'):
        return 2
    return 0


def outcome(r):
    if 'E' in r['labels']:
        return 'departure'
    pcs = [p for p in r['per_cut'] if p]
    assert len(pcs) == 3, r['variant']                  # every run without E completed all three cut-ins
    return 'settled' if max(p['dev_max_after'] for p in pcs) <= SETTLED else 'oscillating'


OUTC = [('settled', '#99DDFF', 'settled'), ('oscillating', '#DDAA33', 'oscillating'), ('departure', '#882255', 'road departure')]
cnt = [{k: 0 for k, _, _ in OUTC} for _ in GROUPS]
for r in diag:
    cnt[group(r['variant'])][outcome(r)] += 1
assert sum(sum(c.values()) for c in cnt) == 145
assert cnt[0] == {'settled': 88, 'oscillating': 0, 'departure': 0}, cnt[0]     # numbers quoted in the text
assert cnt[1] == {'settled': 0, 'oscillating': 54, 'departure': 0}, cnt[1]
assert cnt[2] == {'settled': 1, 'oscillating': 0, 'departure': 2}, cnt[2]
dev_osc = [max(p['dev_max_after'] for p in r['per_cut'] if p) for r in diag if group(r['variant']) == 1]
dev_corner = [max(p['dev_max_after'] for p in r['per_cut'] if p) for r in diag if r['variant'].startswith('L_C3')]

S.apply()
fig = S.plt.figure(figsize=(S.DOUBLE_IN, 2.1))
axa = fig.add_axes([.057, .185, .355, .785]); axb = fig.add_axes([.47, .185, .245, .785]); axc = fig.add_axes([.775, .185, .215, .785])
for ax, letter in zip((axa, axb, axc), 'abc'):
    S.axes(ax, letter, grid=ax is not axc)


GRAY3 = tuple(Line2D([], [], color='0.62', ls=S.CUT_LS[c], lw=.9) for c in range(3))   # legend sample of a gray run: three cut-ins


def cuts(ax, t, Y, color=None, lw=1.1):
    """Three cut-ins: own colour and line style each; color='0.62' draws a gray comparison run."""
    for c in range(Y.shape[0]):
        ax.plot(t, Y[c], color=S.CUT[c] if color is None else color, lw=lw, ls=S.CUT_LS[c])


ax = axa
ax.axhline(LANE_LINE, color=S.REF, lw=.6, ls=(0, (4, 3))); ax.text(20.4, 1.66, 'lane line', ha='left', va='top')   # below the line, clear of the traces
cuts(ax, tr['L_P0']['t'], tr['L_P0']['y'], '0.62', lw=.9)
cuts(ax, tr['L_C3']['t'], tr['L_C3']['y'])
ax.set_xlim(14, 50); ax.set_xticks([15, 20, 25, 30, 35, 40, 45, 50]); ax.set_xlabel('Time (s)', labelpad=1.5)
ax.set_ylim(-.2, 4.6); ax.set_yticks([0, 1.75, 3.5]); ax.set_yticklabels(['0', '1.75', '3.5'])
ax.set_ylabel('Cut-in lateral position (m)', labelpad=2)
h1 = [Line2D([], [], color=S.CUT[c], ls=S.CUT_LS[c], lw=1.1) for c in range(3)]
leg1 = ax.legend(h1, ['$c_\\mathrm{1}$', '$c_\\mathrm{2}$', '$c_\\mathrm{3}$'], loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=3, handlelength=2.2,
                 columnspacing=.8)
ax.add_artist(leg1)
ax.legend([GRAY3, tuple(Line2D([], [], color=S.CUT[c], ls=S.CUT_LS[c], lw=1.1) for c in range(3))],
          ['nominal plant', 'reduced stiffness'], loc='lower right', bbox_to_anchor=(1.0, .02), ncol=1, handlelength=3.0,
          columnspacing=.9, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})

ax = axb
for yb in (EDGE_LO, EDGE_HI):
    ax.axhline(yb, color=S.REF, lw=.6, ls=(0, (4, 3)))
ax.axhline(LANE_LINE, color='0.75', lw=.5, ls=(0, (2, 2)))
ax.text(14.4, EDGE_HI - .14, 'road edge', ha='left', va='top')
ax.text(14.4, EDGE_LO + .14, 'road edge', ha='left', va='bottom')
cuts(ax, tr['L_C2_pulse_neg']['t'], tr['L_C2_pulse_neg']['y'], '0.62', lw=.9)
exits = []
t, Y = tr['L_C2']['t'], tr['L_C2']['y']
for c in range(Y.shape[0]):
    ex = first_exit(t, Y[c])
    assert ex is not None, c                          # every cut-in of L_C2 leaves the road
    k, tx, edge = ex; exits.append({'cut_in': c + 1, 't_exit_s': tx, 'edge_m': edge})
    ax.plot(np.r_[t[:k], tx], np.r_[Y[c, :k], edge], color=S.CUT[c], lw=1.1, ls=S.CUT_LS[c])
    ax.plot(tx, edge, marker='x', color='black', ms=3.4, mew=.9, zorder=5)
for v in ('L_C2_pulse_neg',):
    assert all(first_exit(tr[v]['t'], tr[v]['y'][c]) is None for c in range(3))
tmax = max(e['t_exit_s'] for e in exits)
ax.annotate('leaves the road', xy=(tmax + .12, EDGE_HI - .08), xytext=(tmax + .35, 4.42), ha='left',
            va='center', arrowprops={'arrowstyle': '-', 'color': 'black', 'lw': .6, 'shrinkA': 1, 'shrinkB': 1.5})
ax.set_xlim(14, 35); ax.set_xticks([15, 20, 25, 30, 35]); ax.set_xlabel('Time (s)', labelpad=1.5)
ax.set_ylim(-2.6, 7.4); ax.set_yticks([-1.75, 0, 1.75, 3.5, 5.25]); ax.set_yticklabels(['$-$1.75', '0', '1.75', '3.5', '5.25'])
ax.legend([GRAY3, tuple(Line2D([], [], color=S.CUT[c], ls=S.CUT_LS[c], lw=1.1) for c in range(3))],
          ['negative pulse', 'no pulse'], loc='upper left', bbox_to_anchor=(.13, 1.0), ncol=1, handlelength=3.0,
          handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})

ax = axc
x = np.arange(len(GROUPS)); bottom = np.zeros(len(GROUPS))
n = np.array([sum(c.values()) for c in cnt])
for key, col, lab in OUTC:
    share = np.array([c[key] for c in cnt]) / n * 100
    ax.bar(x, share, bottom=bottom, width=.6, color=col, edgecolor='white', lw=.6, zorder=2, label=lab)
    for i in range(len(GROUPS)):
        if cnt[i][key]:
            ax.text(x[i], bottom[i] + share[i] / 2, str(cnt[i][key]), ha='center', va='center',
                    color='white' if key == 'departure' else 'black', zorder=3)
    bottom += share
ax.set_xticks(x); ax.set_xticklabels(GROUPS, linespacing=.95)
ax.tick_params(axis='x', length=0, top=False, pad=2.5)
ax.set_xlim(-.55, 2.55); ax.set_ylim(0, 140); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel('Share of runs (%)', labelpad=2)
ax.legend(handles=[Patch(facecolor=col, edgecolor='none', label=lab) for _, col, lab in OUTC], loc='upper right',
          bbox_to_anchor=(1.0, 1.0), ncol=1, handlelength=1.1, handleheight=.8, labelspacing=.15)
checks = S.export(fig, OUT / 'fig_lateral')
out = {'source': {'(a)(b)': 'recomputed archive, data/extracted/trajectories.json',
                  '(c)': 'diagnostic ' + DIAG.name + ' on the recomputed archive'},
       'inputs_sha256': I.files,
       'variants': {v: {'labels': tr[v]['labels'], 'corridor_excess_above_m': tr[v]['excess']} for v in need},
       'panel_b_road_exits_L_C2': exits,
       'panel_c_counts': {g.replace('\n', ' '): c for g, c in zip(GROUPS, cnt)},
       'panel_c_rule': f'settled: max deviation after completion <= {SETTLED} m; departure: label E',
       'reduced_stiffness_dev_max_m': {'all_54': [min(dev_osc), max(dev_osc)], 'corner_53': [min(dev_corner), max(dev_corner)]},
       'figure_checks': checks}
(OUT / 'fig_lateral_data.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps(out['panel_c_counts'])); print(json.dumps(exits)); print(out['reduced_stiffness_dev_max_m']); print(json.dumps(checks))
