"""Fig. 4: main run of A-H at nominal parameters (EX1a, s = 1, l = 0).
(a) longitudinal positions relative to the common-speed reference, (b) lateral positions of the cut-ins, (c) speeds,
(d) accelerations, (e) largest envelope use in each phase of the run (initial plan before the first activation, gap
creation up to the first lane-change start, lane changes, after the cut-ins), (f) width of the parameter intervals. The three cut-ins differ in
colour and line style; panels (b)-(d) carry legends only.
Data: original run cb13911c (data/original/runs_selected/cb13911c1dadd2de7413: arrays.npz, result.json, audit.json).
Usage: python fig_main.py <run_dir> <out_dir>"""
import json, hashlib, sys
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator
import figstyle as S

RUN = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
S.apply()
with np.load(RUN / 'arrays.npz', allow_pickle=False) as z:
    A = {k: z[k] for k in z.files}
result = json.loads((RUN / 'result.json').read_text('utf-8')); audit = json.loads((RUN / 'audit.json').read_text('utf-8'))
assert audit['family'] == 'EX1a' and audit['method'] == 'A-H'
t, x, b = A['time'], A['state'], A['bounds']; sizes = [8, 7, 7]
reference = np.r_[0., np.cumsum(A['common_speed']) * .1]; relative_p = x[:, :, 0] - reference[:, None]
width_percent = 100 * (b[..., 1] - b[..., 0]) / b.mean(axis=-1); median_width = np.median(width_percent, axis=1)
contraction_k = int(np.flatnonzero(median_width[:, 1] < median_width[0, 1] - 1e-6)[0])
jerk = float(np.max(np.abs(A['acceleration'][10:] - A['acceleration'][:-10])))
assert abs(jerk - audit['metrics']['window_jerk_max']) < 1e-10
use_pct = 100 * A['tube_ratio']


def lines(ax, tt, values, cuts=True):
    first = 0
    for q, n in enumerate(sizes):
        for i in range(first, first + n):
            ax.plot(tt, values[:, i], color=S.QUEUE[q], alpha=.55, lw=.9, ls=S.QUEUE_LS[q])
        first += n
    if cuts:
        for q in range(values.shape[1] - first):
            ax.plot(tt, values[:, first + q], color=S.CUT[q], lw=1.3, ls=S.CUT_LS[q], zorder=4)


fig, axs = S.plt.subplots(2, 3, figsize=(S.DOUBLE_IN, 3.0))
fig.subplots_adjust(left=.07, right=.985, bottom=.13, top=.94, hspace=.5, wspace=.40)
for ax, letter in zip(axs.flat, 'abcdef'):
    S.axes(ax, letter); ax.set_xlabel('Time (s)', labelpad=2); ax.set_xlim(0, 30); ax.set_xticks([0, 10, 20, 30])
ax = axs[0, 0]; lines(ax, t, relative_p)
ax.set_ylabel('Relative position (m)', labelpad=2); ax.set_ylim(relative_p.min() - 8, relative_p.max() + 40)
for event in (5, 8):                      # request and first activation; named in the text
    ax.axvline(event, color=S.REF, lw=.6, ls=(0, (4, 3)), zorder=2)
first = 0
for q, n in enumerate(sizes):                 # platoon labels right of the frame, clear of the lines
    gy = relative_p[np.argmin(abs(t - 30)), first:first + n].mean()
    ax.annotate(f'$q\\;\\mathrm{{=}}\\;\\mathrm{{{q+1}}}$', xy=(1.0, gy), xycoords=('axes fraction', 'data'), xytext=(2.5, 0),
                textcoords='offset points', ha='left', va='center', annotation_clip=False)
    first += n
ax = axs[0, 1]
for q in range(3):
    y = A['lateral'][:, q, 0]
    ax.plot(t, y, color=S.CUT[q], lw=1.3, ls=S.CUT_LS[q], label=f'$c_\\mathrm{{{q+1}}}$')
    for ev, mk in [(result['merge_start'][q], 'o'), (result['merge_done'][q], 's')]:
        ax.plot(ev, np.interp(ev, t, y), marker=mk, color=S.CUT[q], ms=3, markerfacecolor='white', markeredgewidth=.7, zorder=5)
ax.set_ylim(-.25, 4.35); ax.set_yticks([0, 1, 2, 3.5]); ax.set_yticklabels(['0', '1', '2', '3.5'])
ax.set_ylabel('Lateral position (m)', labelpad=2)
leg_c = ax.legend(loc='center left', bbox_to_anchor=(.0, .55), ncol=1, handlelength=1.8, labelspacing=.1)
ax.add_artist(leg_c)
ax.legend([Line2D([], [], color='0.3', marker=m, ms=3, markerfacecolor='white', markeredgewidth=.7, ls='none') for m in ('o', 's')],
          ['start', 'end'], loc='lower right', bbox_to_anchor=(1.0, .12), ncol=1, handlelength=1.0, labelspacing=.1)
ax = axs[0, 2]; lines(ax, t, x[:, :, 1]); ax.set_ylabel('Speed (m/s)', labelpad=2); ax.set_ylim(17.8, 24.4)
h_lim = ax.axhline(23, color=S.REF, lw=.6, ls=(0, (4, 3)), zorder=1.5)
ax.legend([h_lim], ['speed limit'], loc='upper right', bbox_to_anchor=(1.0, 1.0), handlelength=2.2)
ax = axs[1, 0]; ta = t[:len(A['acceleration'])]; lines(ax, ta, A['acceleration'])
ax.set_ylabel('Acceleration (m/s$^\\mathrm{2}$)', labelpad=2); ax.set_ylim(-1.15, 1.3); ax.set_yticks([-1, 0, 1])
ax = axs[1, 1]; tu = t[:len(use_pct)]
k_act = int(np.flatnonzero(A['plan_age'][1:] == 0)[0]) + 1        # activation of the first merge candidate
t_act, t_ms, t_md = float(t[k_act]), min(result['merge_start']), max(result['merge_done'])
PHASES = [('initial\nplan', tu <= t_act - 1e-9), ('gap\ncreation', (tu >= t_act - 1e-9) & (tu < t_ms - 1e-9)),
          ('lane\nchange', (tu >= t_ms - 1e-9) & (tu <= t_md + 1e-9)), ('after\ncut-in', tu > t_md + 1e-9)]
assert sum(int(m.sum()) for _, m in PHASES) == len(tu)
phase_max = [float(use_pct[m].max()) for _, m in PHASES]
xp = np.arange(len(PHASES))
ax.bar(xp, phase_max, width=.6, color=S.METHOD['A-H'], zorder=2)
for xi, v in zip(xp, phase_max):
    ax.text(xi, v + 3, f'{v:.2f}', ha='center', va='bottom')
S.bound(ax, 100, 'envelope boundary')
ax.set_xlim(-.6, len(PHASES) - .4); ax.set_xticks(xp); ax.set_xticklabels([n for n, _ in PHASES], linespacing=.95)
ax.tick_params(axis='x', length=0, top=False, pad=2.5); ax.set_xlabel('')
ax.set_ylabel('Envelope use (%)', labelpad=2); ax.set_ylim(0, 125); ax.set_yticks([0, 25, 50, 75, 100])
ax = axs[1, 2]
ax.plot(t, median_width[:, 0], color=S.METHOD['A-H'], lw=1.2, label='mass $\\mu$')
ax.plot(t, median_width[:, 1], color=S.METHOD['D-DRPC'], lw=1.2, ls=(0, (4, 2)), label='lag $\\lambda$')
ax.set_xlim(0, 20); ax.set_xticks([0, 5, 10, 15, 20]); ax.set_yscale('log'); ax.set_ylim(.25, 260)
ax.set_yticks([.3, 1, 10, 60]); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_ylabel('Interval width (%)', labelpad=2)
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=2, handlelength=1.9, columnspacing=.8)
ax.annotate('contraction\nafter excitation', xy=(float(t[contraction_k]), median_width[contraction_k, 1]), xytext=(10.4, 9.5),
            va='center', ha='left', arrowprops={'arrowstyle': '->', 'lw': .6, 'color': 'black'}, annotation_clip=False)
pairs = []
for a in axs.flat:
    letters = [tx for tx in a.texts if tx.get_text().startswith('(')]
    others = [tx for tx in a.texts if not tx.get_text().startswith('(')] + [a.yaxis.label]
    for lg in [a.get_legend()] + [r for r in a.artists if hasattr(r, 'get_texts')]:
        if lg: others += list(lg.get_texts())
    pairs += [(l, o) for l in letters for o in others]
clashes = S.overlaps(fig, pairs)
layout = S.export(fig, OUT / 'fig_main')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
facts = {'source_sha256': sha(RUN / 'arrays.npz'), 'letter_text_overlaps': clashes, 'layout': layout,
         'median_width_pct': {'mu_t0': float(median_width[0, 0]), 'mu_1s': float(median_width[10, 0]),
                              'lam_t0': float(median_width[0, 1]), 'lam_8.0s': float(median_width[80, 1]),
                              'lam_8.1s': float(median_width[81, 1])},
         'jerk': jerk, 'contraction_time_s': float(t[contraction_k]),
         'envelope_use_pct_max': float(use_pct.max()), 'envelope_use_pct_after_1s_max': float(use_pct[10:].max()),
         'envelope_use_phase_max_pct': {n.replace('\n', ' '): v for (n, _), v in zip(PHASES, phase_max)},
         'envelope_use_phase_argmax_s': {n.replace('\n', ' '): float(tu[m][np.argmax(use_pct[m])]) for n, m in PHASES},
         'phase_bounds_s': {'activation': t_act, 'first_lane_change_start': t_ms, 'last_lane_change_end': t_md},
         'merge_start_s': result['merge_start'], 'merge_done_s': result['merge_done']}
(OUT / 'fig_main_checks.json').write_text(json.dumps(facts, indent=1), encoding='utf-8'); print(json.dumps({k: v for k, v in facts.items() if k != 'layout'}, indent=0)[:900]); print(layout)
