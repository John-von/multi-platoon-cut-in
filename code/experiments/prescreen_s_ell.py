"""WP5-v20 design prescreen (static, t = 0, no closed loop): for a grid of residual scale s and measurement-noise level ell,
which initial (cruise) and merge candidates does the frozen-candidate v20 code certify?  Used only to place grid points;
registered in the design as predictions.  Real CVXOPT is not needed (no lateral solve)."""
import os, sys, json, time, itertools
os.environ['V20_PLAN_WORKERS'] = '2'
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get('V20_SRC', HERE.parent / 'sim' / 'sources'))
OSQP = Path(os.environ.get('V20_OSQP_STANDIN', HERE.parent / 'sim' / 'osqp_only'))
sys.path[:0] = ([str(OSQP)] if OSQP.exists() else []) + [str(SRC)]
import numpy as np, robust_core as R, simulation as S
MAIN = dict(enabled=True, lateral_soft=True, condition_s=True, jerk_mode='window', jerk_window=10, replan_mode='handover',
            activation_latency=[30, 20], containment_gate=False, init_refine=True)
EPS = np.array([1e-3, 5e-5, .00015 / 1.89010989])
def setup(s, ell, gap=10.5):
    R.configure_v20(dict(MAIN, meas_eps=(ell * EPS).tolist()))
    sc = S.scenario(gap); x = np.c_[sc['p0'], sc['v0'], R.DRAG * sc['v0'] ** 2 + R.GF]
    return sc, x, np.repeat(R.PRIOR[None], len(x), 0), (.001 * s, .00015 * s)
rows = []
for s, ell in itertools.product([1, 2, 3, 5, 7, 10, 12, 15], [0, 1, 2, 5, 7, 10, 15, 20]):
    sc, x, th, noise = setup(s, ell)
    init = None; why0 = None
    for T in [12, 20, 26, 30, 36]:
        p, why0 = S.make_plan(x, th, sc, T, 'cruise', noise)
        if p: init = T; break
    rows.append(dict(s=s, ell=ell, initial_T=init, initial_last_reason=None if init else why0)); print(json.dumps(rows[-1]), flush=True)
(HERE / 'prescreen_initial.json').write_text(json.dumps(rows, indent=1), encoding='utf-8', newline='\n')
