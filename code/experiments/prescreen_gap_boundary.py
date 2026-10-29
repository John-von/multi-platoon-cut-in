"""WP5-v20 design prescreen 2 (static, t=0, no closed loop): initial (cruise) certification and the t=0 merge plan
for tight initial gaps and queue boundaries, s=1, ell in {0,5}.  Used only to place grid points (registered as
predictions).  Frozen-candidate v20 code, read-only import."""
import os, sys, json, itertools
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
rows = []
if __name__ == '__main__':
    for (gap, boundary), ell in itertools.product([(10.2, 30), (10.25, 30), (10.3, 30), (10.5, 30), (11, 30), (12, 30),
                                                   (10.5, 18), (10.5, 24), (10.5, 26)], [0, 5]):
        R.configure_v20(dict(MAIN, meas_eps=(ell * EPS).tolist()))
        sc = S.scenario(gap, boundary); x = np.c_[sc['p0'], sc['v0'], R.DRAG * sc['v0'] ** 2 + R.GF]
        th = np.repeat(R.PRIOR[None], len(x), 0); noise = (.001, .00015)
        init = None; why = []
        for T in [12, 20, 26, 30, 36]:
            p, w = S.make_plan(x, th, sc, T, 'cruise', noise); why.append(w)
            if p: init = T; break
        merge = None; mwhy = []
        if init:
            for T in [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 34]:
                p, w = S.make_plan(x, th, sc, T, 'merge', noise); mwhy.append(w)
                if p: merge = T; break
        rows.append(dict(gap=gap, boundary=boundary, ell=ell, initial_T=init, initial_reasons=why, t0_merge_T=merge, merge_reasons=mwhy))
        print(json.dumps(rows[-1]), flush=True)
    (HERE / 'prescreen_gap_boundary.json').write_text(json.dumps(rows, indent=1), encoding='utf-8', newline='\n')
