"""WP5-v20 design prescreen 3 (static, t = 0, no closed loop): for the eight EX5 cells, the first certified initial
(cruise) candidate and the first certified t = 0 merge candidate under the base configuration and under the
mechanism variants V1 (sampled jerk), V2 (init_refine off), V7 (ellipsoid-only), V8 (current-gap admission) and
V9 (no terminal checks).  Registered as predictions only; the closed loop plans later (request 5 s, activation 8 s)."""
import os, sys, json, copy
os.environ['V20_PLAN_WORKERS'] = '1'
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get('V20_SRC', HERE.parent / 'sim' / 'sources'))
OSQP = Path(os.environ.get('V20_OSQP_STANDIN', HERE.parent / 'sim' / 'osqp_only'))
sys.path[:0] = ([str(OSQP)] if OSQP.exists() else []) + [str(SRC)]
import numpy as np, robust_core as R, simulation as S, async_execution as A
import adapters as AD, casegen as CG
ORIG = {'cert19': R._certificate_v19, 'cert': R.certificate, 'mp': S.make_plan}
def reset():
    R._certificate_v19 = ORIG['cert19']; R.certificate = ORIG['cert']; S.certificate = ORIG['cert']; A.certificate = ORIG['cert']
    S.make_plan = ORIG['mp']; A.make_plan = ORIG['mp']
def screen(cell, variant):
    reset()
    kw = dict(CG.EX5_CELLS[cell]); corner = kw.pop('corner', 'nominal')
    s = kw.get('s', 1); ell = kw.get('ell', 0); gap = kw.get('gap', 10.5); b = kw.get('boundary', 30)
    v20 = dict(CG.V20_MAIN, meas_eps=[ell * e for e in CG.EPS])
    if variant == 'V1': v20.update(jerk_mode='sampled', jerk_window=None)
    if variant == 'V2': v20.update(init_refine=False)
    if variant in ('V7', 'V8', 'V9'): AD.install_mechanism(R, S, A, variant)
    R.configure_v20(v20)
    mass, tau = CG.params(corner); mu = 1 / np.broadcast_to(mass, (25,))
    sc = S.scenario(gap, b); x = np.c_[sc['p0'], sc['v0'], R.DRAG * sc['v0'] ** 2 + R.GF / mu]
    th = np.repeat(R.PRIOR[None], len(x), 0); noise = (.001 * s, .00015 * s)
    mp = S.make_plan
    init = None; ir = []
    for T in [12, 20, 26, 30, 36]:
        p, w = mp(x, th, sc, T, 'cruise', noise); ir.append(w)
        if p: init = T; break
    merge = None; mr = []; gates = None
    if init:
        for T in CG.MERGE_BASE:
            p, w = mp(x, th, sc, T, 'merge', noise); mr.append(w)
            if p: merge = T; gates = [int(g) for g in p['gates']]; break
    return dict(cell=cell, variant=variant, initial_T=init, initial_reasons=ir, merge_T=merge, gates=gates, merge_reasons=mr)
rows = []
if __name__ == '__main__':
    for cell in CG.EX5_CELLS:
        for v in ['V0', 'V1', 'V2', 'V7', 'V8', 'V9']:
            rows.append(screen(cell, v)); r = rows[-1]
            print(cell, v, r['initial_T'], r['merge_T'], r['gates'], r['merge_reasons'][-3:], flush=True)
    (HERE / 'prescreen_ablation.json').write_text(json.dumps(rows, indent=1), encoding='utf-8', newline='\n')
