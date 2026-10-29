"""Fig. 9: external baseline B2 against the A-H run.
B2 = centralized nominal planner adapted from Duan et al. (2023), main version with the one-second jerk windows,
selected plan (initial guess G22, fixed terminal time 6.282 s, accepted call 3). It plans platoon 1 (8 vehicles) and
cut-in c1 from the measured state of the A-H main run at the request (t = 5 s), with nominal model and exact states;
it was not executed in closed loop. A-H: original EX1a nominal run cb13911c (all three platoons).
(a) Longitudinal position of platoon 1 and c1 relative to the common-speed reference.
(b) Lateral position of c1; the two vertical lines mark the request and the end of the B2 plan.
Usage: python fig_b2.py <b2_solution.npz> <ah_run_dir> <out_dir>"""
import json, sys, hashlib
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import figstyle as S

B2 = Path(sys.argv[1]); RUN = Path(sys.argv[2]); OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
z = np.load(B2, allow_pickle=False)
gvi = z['global_vehicle_indices']; assert list(gvi) == [0, 1, 2, 3, 4, 5, 6, 7, 22]
tf = float(z['tf']); tb = z['time']; pb = z['p_ref']; yb = z['y_plan']; vb = z['v_ref']
with np.load(RUN / 'arrays.npz', allow_pickle=False) as r:
    t = r['time']; x = r['state']; lat = r['lateral']; cs = r['common_speed']
res = json.loads((RUN / 'result.json').read_text('utf-8'))
ref = np.r_[0., np.cumsum(cs) * .1]
k5 = int(np.argmin(abs(t - 5.0))); assert abs(t[k5] - 5.0) < 1e-9
assert np.allclose(pb[:, 0], x[k5, gvi, 0], atol=1e-9), 'B2 start differs from the A-H state at the request'
assert np.allclose(cs, 20.0)
keep = tb <= tf + 1e-9                     # the plan ends at t_f
tB = 5.0 + tb[keep]; relB = pb[:, keep] - 20.0 * tB[None, :]
relA = x[:, :, 0] - ref[:, None]
S.apply()
fig = S.plt.figure(figsize=(S.SINGLE_IN, 2.7))
ax1 = fig.add_axes([.155, .555, .825, .425]); ax2 = fig.add_axes([.155, .125, .825, .30])
S.axes(ax1, 'a'); S.axes(ax2, 'b')
for i in range(8):
    ax1.plot(t, relA[:, i], color=S.QUEUE[0], lw=.9, alpha=.75, zorder=2)
    ax1.plot(tB, relB[i], color=S.METHOD['B2'], lw=1.0, ls=(0, (3, 1.6)), zorder=3)
ax1.plot(t, relA[:, 22], color=S.CUT_COLOR, lw=1.3, zorder=4)
ax1.plot(tB, relB[8], color=S.METHOD['B2'], lw=1.3, ls=(0, (3, 1.6)), zorder=4)
ax1.set_ylim(relA[:, :8].min() - 10, relA[:, :8].max() + 42)
top_a = (relA[:, :8].max() + 12 - ax1.get_ylim()[0]) / np.ptp(ax1.get_ylim())   # (a): event lines end below the legend row
for ax, top in ((ax1, top_a), (ax2, 1.0)):
    ax.axvline(5.0, ymax=top, color=S.REF, lw=.6, ls=(0, (4, 3)), zorder=1)
    ax.axvline(5.0 + tf, ymax=top, color=S.METHOD['B2'], lw=.6, ls=(0, (1, 1.5)), zorder=1)
    ax.set_xlim(0, 25); ax.set_xticks([0, 5, 10, 15, 20, 25])
ax1.set_ylabel('Relative position (m)', labelpad=2); ax1.set_xlabel('Time (s)', labelpad=1.5)
ax1.legend([Line2D([], [], color=S.QUEUE[0], lw=.9), Line2D([], [], color=S.CUT_COLOR, lw=1.3),
            Line2D([], [], color=S.METHOD['B2'], lw=1.1, ls=(0, (3, 1.6)))],
           ['A-H, platoon 1', 'A-H, $c_\\mathrm{1}$', 'B2 plan'], loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=3,
           handlelength=1.9, columnspacing=.7)                 # one row, above all trajectories
ax2.plot(t, lat[:, 0, 0], color=S.CUT_COLOR, lw=1.3, label='A-H')
ax2.plot(tB, yb[8, keep], color=S.METHOD['B2'], lw=1.3, ls=(0, (3, 1.6)), label='B2 plan')
ax2.set_ylim(-.3, 4.85); ax2.set_yticks([0, 1.75, 3.5]); ax2.set_yticklabels(['0', '1.75', '3.5']); ax2.set_ylabel('$c_\\mathrm{1}$ lateral pos. (m)', labelpad=2)
ax2.set_xlabel('Time (s)', labelpad=1.5)
ax2.legend(loc='center left', bbox_to_anchor=(.465, 2.625), bbox_transform=blended_transform_factory(ax2.transAxes, ax2.transData),
           ncol=1)                                             # between the 1.75 and 3.5 m grid lines
ax2.text(5.25, 4.2, 'request', ha='left', va='center')
ax2.text(5.0 + tf + .25, 4.2, 'B2 plan ends', ha='left', va='center')
checks = S.export(fig, OUT / 'fig_b2')
k_start = res['merge_start'][0]; k_done = res['merge_done'][0]
facts = {'b2_solution_sha256': sha(B2), 'ah_arrays_sha256': sha(RUN / 'arrays.npz'), 'tf': tf,
         'b2_insertion_done_at_s': 5.0 + tf, 'ah_c1_lane_change_start_s': k_start, 'ah_c1_done_s': k_done,
         'ah_last_done_s': res.get('last_done'), 'b2_max_speed': float(vb[:, :].max()), 'b2_min_speed': float(vb.min()),
         'figure_checks': checks}
(OUT / 'fig_b2_data.json').write_text(json.dumps(facts, indent=1), encoding='utf-8'); print(json.dumps(facts, indent=1))
