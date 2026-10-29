"""EX10 (Q/N-v20) design prescreen (static, t = 0, no closed loop).  For every family x geometry: (a) the closed-form
necessary condition of the joint allocation LP (v19 structural_check.chain_condition, copied) and LP feasibility with the
widest capacity the search can return (+-40 m) using the frozen robust_core.allocation (unchanged v19 -> v20); (b) the
smallest initial queue spacing on a 2 m grid (30..60 m) that restores LP feasibility; (c) the first certified initial
(cruise) horizon of the v20 main configuration (s = 1, ell = 0) over the probe list [12,20,26,30,36,40,50,60,80,100]
(the runner itself only tries [12,20,26,30,36]).  Registered as predictions only."""
import os, sys, json
os.environ['V20_PLAN_WORKERS'] = '1'
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get('V20_SRC', HERE.parent / 'sim' / 'sources'))
OSQP = Path(os.environ.get('V20_OSQP_STANDIN', HERE.parent / 'sim' / 'osqp_only'))
sys.path[:0] = ([str(OSQP)] if OSQP.exists() else []) + [str(SRC)]
import numpy as np, robust_core as R, simulation as S
import adapters_addA as ADA, casegen as CG
S0, SSTAR, BMIN, EPS = 10.5, 12.0, 12.0, 0.01
C = (SSTAR + SSTAR - S0) / 2; D = SSTAR - S0
FAMILIES = ("supplied_synthetic", "bounded_speed_synthetic", "stage1_template")
GEOMS = [(f"Q{q}", [8] + [7] * (q - 1)) for q in (2, 3, 4, 5, 6)] + \
        [(f"N_{tag}", [round(v * m) for v in (8, 7, 7)]) for m, tag in ((1.5, "x1p5"), (2., "x2"), (3., "x3"))]
PROBE = [12, 20, 26, 30, 36, 40, 50, 60, 80, 100]
def counts(scene):
    N, Q = scene['N'], scene['Q']; starts = list(scene['starts']) + [N]; a, r = [], []
    for q in range(Q):
        s, e = starts[q], starts[q + 1]; f = int(scene['front'][q]); a.append(int(f - s + 1)); r.append(int(e - (f + 1)))
    return a, r
def chain_condition(a, r, b0):
    Q = len(a); d = [(b0 - BMIN) - 2 * C - D * (r[q] - 1 + a[q + 1] - 1) for q in range(Q - 1)]; worst = np.inf
    for i in range(Q - 1):
        acc = 0.0
        for j in range(i, Q - 1):
            acc += d[j]; worst = min(worst, acc + 2 * (C - EPS))
    return worst
def lp_feasible(scene, cap=40.0):
    N, Q = scene['N'], scene['Q']; M = N + Q; p = np.asarray(scene['p0'], float)
    caps = np.c_[np.full(M, -cap), np.full(M, cap), np.zeros(M)]; caps[N:, 2] = p[N:]
    return R.allocation(p[:N], np.asarray(scene['starts']), np.asarray(scene['front']), np.asarray(scene['rear']), caps) is not None
def initial_T(scene):
    R.configure_v20(dict(CG.V20_MAIN, meas_eps=[0., 0., 0.]))
    x = np.c_[scene['p0'], scene['v0'], R.DRAG * scene['v0'] ** 2 + R.GF]; th = np.repeat(R.PRIOR[None], len(x), 0)
    for T in PROBE:
        p, w = S.make_plan(x, th, scene, T, 'cruise', (.001, .00015))
        if p: return T, w
    return None, w
rows = []
if __name__ == '__main__':
    for fam in FAMILIES:
        for label, sizes in GEOMS:
            row = {'family': fam, 'geometry': label, 'queue_sizes': sizes}
            b_restore = None
            for b in [30.] + [float(x) for x in range(32, 62, 2)]:
                sc = ADA.construct_scene(S, fam, sizes, b)
                a, r = counts(sc); m = chain_condition(a, r, b); feas = lp_feasible(sc)
                if b == 30.:
                    row.update(N=int(sc['N']), Q=int(sc['Q']), M=int(sc['N'] + sc['Q']), chain_margin_b30=round(float(m), 4),
                               necessary_condition_b30=bool(m >= 0), lp_feasible_b30=bool(feas))
                if feas and b_restore is None:
                    b_restore = b; row['chain_margin_at_restore'] = round(float(m), 4)
                    break
            row['min_boundary_lp_feasible'] = b_restore
            T, why = initial_T(ADA.construct_scene(S, fam, sizes, 30.))
            row['initial_T_b30_probe'] = T; row['initial_last_reason_b30'] = None if T else why
            if b_restore and b_restore > 30:
                T2, why2 = initial_T(ADA.construct_scene(S, fam, sizes, b_restore))
                row['initial_T_restore_probe'] = T2
            rows.append(row); print(json.dumps(row), flush=True)
    agree = all(r['necessary_condition_b30'] == r['lp_feasible_b30'] for r in rows)
    (HERE / 'prescreen_qn.json').write_text(json.dumps({'rows': rows, 'closed_form_agrees_with_lp_b30': agree}, indent=1), encoding='utf-8', newline='\n')
    print('agree', agree)
