"""Unit tests of b2_nlp_v6 (including the fixed-t_f stage of rule E-10). Part A needs NumPy only; part B needs
CasADi. No optimizer call. Writes test_b2_nlp_v6.json (test_b2_nlp_v6_numpy_only.json without CasADi)."""
import json, os, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import b2_nlp_v6 as N
import b2_components as B

checks = []


def check(cid, ok, **info):
    checks.append({'id': cid, 'pass': bool(ok), **info})
    print(('PASS ' if ok else 'FAIL ') + cid, {k: v for k, v in info.items() if not isinstance(v, (list, dict))}, flush=True)


# ---------------------------------------------------------------- A1 E-3 zone function
D = 900.0
dx, dy = B.EPS_A / N.ZONE_H, B.EPS_A / (D + 10.0)
g = lambda xc, yc, xo, role: float(N.zone_g(xc, yc, xo, role, D, np.maximum))
yin = 1.0
x0f, x1f, y0, y1 = B.zone_rect('front', 0.0, D)
cases = {
    'counterexample_inside': g(12.0, 1.0, 0.0, 'front') < 0,
    'x_left_out_1.8e-4_violates': g(x0f - 1.8e-4, yin, 0.0, 'front') < -B.EPS_A * 0 and g(x0f - 1.8e-4, yin, 0.0, 'front') < 0,
    'x_left_out_2.0e-4_satisfies': g(x0f - 2.0e-4, yin, 0.0, 'front') > N.TOL,
    'x_right_far_end_out_1.8e-4_violates': g(x1f + 1.8e-4, yin, 0.0, 'front') < 0,
    'x_right_far_end_out_2.0e-4_satisfies': g(x1f + 2.0e-4, yin, 0.0, 'front') > N.TOL,
    'below_lower_edge_1mm_satisfies': g(5.0, y0 - 1e-3, 0.0, 'front') > N.TOL,
    'y_equals_ys_inside': g(5.0, B.Y_S, 0.0, 'front') < 0,
    'x_equals_xf_minus_10_inside': g(-10.0, yin, 0.0, 'front') < 0,
    'rear_x_out_2.0e-4_satisfies': g(10.0 + 2.0e-4, yin, 0.0, 'rear') > N.TOL,
    'rear_x_out_1.8e-4_violates': g(10.0 + 1.8e-4, yin, 0.0, 'rear') < 0,
    'rear_far_end_inside': g(-D, yin, 0.0, 'rear') < 0,
    'corners_violate': all(g(xc, yc, 0.0, 'front') < 0 for xc in (x0f, x1f) for yc in (y0, y1)),
    'above_top_edge_2e-6_satisfies': g(5.0, y1 + 2e-6, 0.0, 'front') > 0,
}
check('E3_boundary_list', all(cases.values()), cases=cases, dx=dx, dy=dy)
rng = np.random.default_rng(20261014)
grads = []
for _ in range(400):
    xc = rng.uniform(x0f + 1e-3, x1f - 1e-3); yc = rng.uniform(y0 + 1e-4, y1 - 1e-4); e = 1e-7
    gx = (g(xc + e, yc, 0.0, 'front') - g(xc - e, yc, 0.0, 'front')) / (2 * e)
    gy = (g(xc, yc + e, 0.0, 'front') - g(xc, yc - e, 0.0, 'front')) / (2 * e)
    grads.append(np.hypot(gx, gy))
check('E3_interior_gradient_nonzero', min(grads) > 0.5, min_grad_norm=float(min(grads)))
dis = []; n_out = 0
for _ in range(20000):
    xc = rng.uniform(x0f - 3e-4, x0f + 3e-4) if rng.random() < .5 else rng.uniform(x1f - 3e-4, x1f + 3e-4)
    yc = rng.uniform(y0 - 3e-6, y0 + 3e-6) if rng.random() < .5 else rng.uniform(y1 - 3e-6, y1 + 3e-6)
    new = g(xc, yc, 0.0, 'front') >= 0
    old = float(B.zone_margin(xc, yc, (x0f, x1f, y0, y1))) >= 0
    n_out += old
    if new != old:
        ddx = max(x0f - xc, xc - x1f); ddy = max(y0 - yc, yc - y1)
        dis.append({'new': new, 'old': old, 'dx_over': ddx / dx, 'dy_over': ddy / dy})
ok = all((not d['new']) and d['old'] and 0 < d['dx_over'] < 1 and 0 < d['dy_over'] < 1 and d['dx_over'] + d['dy_over'] >= 1 - 1e-9 for d in dis)
check('E3_vs_area_form_disagree_only_in_corner_triangles', ok, n_disagree=len(dis), n_samples=20000)

# ---------------------------------------------------------------- scene / initial guesses
sc = N.scene(True); scf = N.scene(False)
inits = {}
for scope, s in (('single', sc), ('full', scf)):
    for Tg in (16, 22, 28):
        tf, S, U, meta = N.initial(s, Tg); gate = N.initial_gate(s, tf, S, U)
        inits[(scope, Tg)] = (tf, S, U)
        zone_nodes = {k: v for k, v in gate['node_min_margins'].items() if k.startswith('zone_')}
        check(f'A6_initial_gate_{scope}_G{Tg}', gate['pass'], tf=tf, max_violation=gate['max_required_violation'],
              zone_min=min(zone_nodes.values()), zone_interp_min=min(gate['interpolated_zone_min_margins'].values()))

# ---------------------------------------------------------------- A2 E-1 physical span
errs = []; frac_ok = True
for tf_r in (5.3, 9.87, 16.0, 21.234, 28.0, 33.9):
    for tf in (tf_r * .9, tf_r, tf_r * 1.1):
        for j in N.pdd_window_starts(tf_r):
            reps = []
            for jj in (j, j + 10):
                br, k = N.freeze_grid(jj, tf_r)
                if br == 'plan':
                    f = N.NT * jj * N.DT / tf - k
                    reps.append((k + f) * tf / N.NT)
                    if tf == tf_r and not (-1e-9 <= f <= 1 + 1e-9):
                        frac_ok = False
                else:
                    reps.append(jj * N.DT)
            errs.append(abs(reps[1] - reps[0] - 1.0))
check('E1_window_physical_span_exactly_1s_any_tf', max(errs) <= 1e-12 and frac_ok, max_err=float(max(errs)), n=len(errs))

# ---------------------------------------------------------------- A3 NLP-form vs audit-form parity (numeric)
def consistent(tf, S, U):
    """Node states satisfying the trapezoidal collocation equations exactly (fixed point per
    interval) for the given controls. The issued reference extrapolates from the last plan
    sample, the NLP extension starts from node N_T: the two coincide when the collocation
    equations hold, so parity is tested on a dynamically consistent node set."""
    S2 = S.copy(); h = tf / N.NT
    for i in range(S.shape[0]):
        for k in range(N.NT):
            fk = B.rhs(S2[i, :, k], U[i, :, k]); nxt = S2[i, :, k] + h * fk
            for _ in range(60):
                nxt = S2[i, :, k] + h / 2 * (fk + B.rhs(nxt, U[i, :, k + 1]))
            S2[i, :, k + 1] = nxt
    return S2


tf, S0g, U = inits[('single', 22)]
S = consistent(tf, S0g, U)
colres = max(float(abs(B.collocation_residual(np.linspace(0, tf, N.NT + 1), S[i], U[i])).max()) for i in range(sc['m']))
check('A3_consistent_test_trajectory', colres <= 1e-12, collocation_residual=colres, tf=tf, tf_on_grid=bool(abs(tf / .1 - round(tf / .1)) < 1e-12))
nd = N.Nodes.numeric(tf, S, U)
fz = lambda j: N.freeze_grid(j, tf)
tj, sj, uj, dj, cp, refs, rv, ra, jw = N.executed_reference(sc, tf, S, U)
e_p = e_pdd = e_v = e_a = e_w = 0.
for i in range(sc['m']):
    for j in range(len(tj)):
        e_p = max(e_p, abs(N.phys_p(nd, i, j * N.DT, fz(j), N.NumOps) - cp[0, i, j]))
        if j * N.DT <= tf:
            e_pdd = max(e_pdd, abs(N.phys_pdd(nd, i, j, fz(j), sc['history'], N.NumOps) - cp[3, i, j]))
    for j in range(ra.shape[1]):
        v, a = N.secant_va(nd, i, j, fz, N.NumOps)
        e_v = max(e_v, abs(v - rv[i, j])); e_a = max(e_a, abs(a - ra[i, j]))
    for j in range(-10, jw.shape[0] - 10):
        e_w = max(e_w, abs(N.secant_window(nd, i, j, fz, sc['history'], N.NumOps) - jw[j + 10, i]))
check('A3_parity_position', e_p <= 1e-9, max_abs=e_p)
check('A3_parity_pdd_complete_formula', e_pdd <= 1e-9, max_abs=e_pdd)
check('A3_parity_secant_v_a', e_v <= 1e-9 and e_a <= 1e-8, max_v=e_v, max_a=e_a)
check('A3_parity_secant_window_incl_history_and_tail', e_w <= 1e-8, max_abs=e_w)
# documented junction property on the (inconsistent) registered initial guess: the only NLP/audit
# difference is at the t_f junction and equals the final-interval collocation residual / DT
ndg = N.Nodes.numeric(tf, S0g, U); fzg = lambda j: N.freeze_grid(j, tf)
_, _, _, _, cpg, _, rvg, rag, _ = N.executed_reference(sc, tf, S0g, U)
M = int(np.ceil(tf / .1)); away = 0.
for i in range(sc['m']):
    for j in range(M - 2):
        v, a = N.secant_va(ndg, i, j, fzg, N.NumOps); away = max(away, abs(a - rag[i, j]), abs(v - rvg[i, j]))
check('A3_initial_guess_parity_away_from_tf_junction', away <= 1e-8, max_abs=away)
# perturbed t_f: endpoints whose interval is unchanged must reproduce the audit interpolation at the new t_f
tf2 = tf * (1 + 3e-4); nd2 = N.Nodes.numeric(tf2, S, U)
tj2, sj2, uj2, dj2, cp2, *_ = N.executed_reference(sc, tf2, S, U)
e2 = 0.; used = 0
for j in range(len(tj2)):
    if j * N.DT <= tf and N.freeze_grid(j, tf) == N.freeze_grid(j, tf2):
        used += 1
        for i in range(sc['m']):
            e2 = max(e2, abs(N.phys_p(nd2, i, j * N.DT, fz(j), N.NumOps) - cp2[0, i, j]))
check('A3_perturbed_tf_unchanged_intervals_match_audit', e2 <= 1e-9 and used > 0.8 * len(tj2), max_abs=e2, endpoints=used, of=len(tj2))

# ---------------------------------------------------------------- A4 E-5 pruning counts
counts = {}
for Tg in (16, 22, 28):
    t_, S_, U_ = inits[('single', Tg)]
    counts[Tg] = len(N.pair_set(sc, t_, S_, U_)[0])
check('A4_pruning_centre_distance_9veh', all(c == 19 for c in counts.values()), counts=counts,
      note='19 pairs under the pruning rule')
countsf = {Tg: len(N.pair_set(scf, *inits[('full', Tg)])[0]) for Tg in (16, 22, 28)}
check('A4_pruning_full_recorded', True, counts=countsf)

# ---------------------------------------------------------------- A5 audit plumbing
rep, arr = N.audit(sc, 'main', tf, S, U)
hist_ok = all(N.freeze_grid(j, tf) == ('history', j + 10) for j in range(-10, 0))
check('A5_history_rows_and_audit_runs', hist_ok and isinstance(rep['pass'], bool), initial_guess_pass=rep['pass'],
      n_bad_windows=len(rep['bad_window']), max_violation=rep['max_violation'])

# ---------------------------------------------------------------- A7 E-10 direction search and bisection (fake evaluator)
import tempfile
real_eval = N.fixed_eval
for thr, ts in ((6.43, 6.73), (7.4, 7.1065), (5.0, 6.73), (9.9, 7.1)):
    calls_seen = []
    def fake(sc_, variant, tf_fix, S_, U_, sets_, pairs_, folder, budget_s, t0):
        calls_seen.append(tf_fix); ok = tf_fix >= thr
        return {'tf': tf_fix, 'accepted': ok, 'status': 'accepted' if ok else 'not_accepted_within_call_cap', 'calls': [],
                'objective': 100 * tf_fix + 1.0, 'solver_ok_last': True, 'folder': str(folder), 'final_call_folder': str(folder),
                'S': S_, 'U': U_, 'sets': sets_, 'pairs': pairs_}
    N.fixed_eval = fake
    with tempfile.TemporaryDirectory() as tmp:
        res = N.fixed_stage(None, 'main', ts, None, None, {'points': set(), 'secs': set(), 'wins': set()}, [], Path(tmp), 1e9, time.perf_counter())
    N.fixed_eval = real_eval
    if thr == 5.0:          # all 4 downward steps accepted -> lower bound not found, lowest accepted reported
        ok = res['status'] == 'accepted_plan' and res['reason'] == 'no_infeasible_below_within_steps' and len(calls_seen) == 5
    elif thr == 9.9:        # never accepted within 6 upward steps
        ok = res['status'] == 'no_accepted_plan' and res['reason'] == 'no_accepted_tf_within_steps' and len(calls_seen) == 7
    else:
        ok = (res['status'] == 'accepted_plan' and res['bisection_complete'] and res['lowest_accepted_tf'] >= thr
              and res['lowest_rejected_tf'] < thr and res['resolution_s'] / res['lowest_accepted_tf'] <= N.REL_RES + 1e-12
              and abs(res['best_tf'] - res['lowest_accepted_tf']) < 1e-12)
    check(f'A7_E10_search_threshold_{thr}', ok, best_tf=res['best_tf'], n_eval=len(calls_seen), reason=res['reason'],
          resolution=res.get('resolution_s'))

# ---------------------------------------------------------------- B CasADi
ca = N.ca
casadi_ok = ca is not None
if casadi_ok:
    F, C = N.symbolic_functions(); mr = mm = 0.
    for _ in range(50):
        s = np.r_[rng.uniform(-100, 100, 2), rng.uniform(5, 23), rng.uniform(-.3, .3), rng.uniform(-.2, .2)]
        u = rng.uniform([-.5, -.1], [.5, .1])
        mr = max(mr, float(abs(np.array(F(s, u)).ravel() - B.rhs(s, u)).max()))
        mm = max(mm, float(abs(np.array(C(s, u)).ravel() - np.array(B.centre_from_rear(*s, *u))).max()))
    check('B1_rhs_centre_symbolic_numpy', mr <= 1e-12 and mm <= 1e-12, rhs=mr, centre=mm)
    # B2 group-wise parity at a point whose t_f differs from the freezing t_f (exercises E-1 smooth f)
    t_r = tf; tfw = tf * (1 + 2e-3)
    pts = [0.37, 0.5]; secs = [3, 40, int(np.ceil(tf / .1)) - 1]; wins = [(0, -10), (0, -3), (sc['cut'][0], 25), (2, int(np.ceil(tf / .1)) - 4)]
    pairs = N.pair_set(sc, tf, S, U)[0]
    w = N.pack(tfw, S, U); ndw = N.Nodes.numeric(tfw, S, U); fzr = lambda j: N.freeze_grid(j, t_r)
    res = {}
    for variant in N.VARIANTS:
        model = N.build(sc, variant, t_r, pts, secs, wins, pairs)
        gv = np.array(model['eval'](w)[0]).ravel(); at = 0; chunks = {}
        for name, n in model['groups']:
            chunks.setdefault(name, []).append(gv[at:at + n]); at += n
        chunks = {k: np.concatenate(v) for k, v in chunks.items()}
        errs = {}
        if 'window_secant' in chunks:
            ref = [N.secant_window(ndw, i, j, fzr, sc['history'], N.NumOps) for j in N.window_starts(t_r) for i in range(sc['m'])]
            errs['window_secant'] = float(abs(chunks['window_secant'] - np.array(ref)).max())
            errs['window_secant_count'] = int(len(ref) == model['n_main_windows'] == len(N.window_starts(t_r)) * sc['m'])
        if 'secant_v' in chunks:
            ref = [N.secant_va(ndw, i, j, fzr, N.NumOps) for j in sorted(secs) for i in range(sc['m'])]
            errs['secant_v'] = float(abs(chunks['secant_v'] - np.array([r[0] for r in ref])).max())
            errs['secant_a'] = float(abs(chunks['secant_a'] - np.array([r[1] for r in ref])).max())
        if 'secant_window' in chunks:
            ref = [N.secant_window(ndw, i, j, fzr, sc['history'], N.NumOps) for i, j in sorted(wins)]
            errs['secant_window'] = float(abs(chunks['secant_window'] - np.array(ref)).max())
        if 'node_jerk' in chunks:
            ncp = N.centres(S, U)
            ref = (np.diff(ncp[3], axis=1) * N.NT / tfw).ravel()
            errs['node_jerk'] = float(abs(chunks['node_jerk'] - ref).max())
            errs['crossing_jerk'] = float(abs(chunks['crossing_jerk'] - (ncp[3][:, 0] - sc['history'][-1])).max())
        ncp = N.centres(S, U); zref = []
        for k in range(N.NT + 1):
            for c, f, r in zip(sc['cut'], sc['fronts'], sc['rears']):
                for role, o in [('front', f), ('rear', r)]:
                    zref.append(N.zone_g(ncp[0, c, k], ncp[1, c, k], ncp[0, o, k], role, B.zone_D(sc['p0'][c], sc['p0'][o]), np.maximum))
        errs['node_zone'] = float(abs(chunks['node_zone'] - np.array(zref)).max())
        res[variant] = errs
        expect = {'main': {'window_secant', 'secant_v'}, 'sensitivity': {'node_jerk', 'crossing_jerk', 'secant_v', 'secant_window'},
                  'original': {'secant_v'}}[variant]
        present = set(k for k in ('window_pdd', 'window_secant', 'secant_v', 'secant_window', 'node_jerk', 'crossing_jerk') if k in chunks)
        check(f'B2_group_parity_{variant}', present == expect and all(v <= 1e-9 for k, v in errs.items() if not k.endswith('_count'))
              and errs.get('window_secant_count', 1) == 1, errors=errs, groups=sorted(present))
    mfix = N.build(sc, 'main', tf, pts, secs, wins, pairs, fixed_tf=True)
    check('B5_fixed_tf_bounds', mfix['lbx'][0] == mfix['ubx'][0] == tf and N.build(sc, 'main', tf, pts, secs, wins, pairs)['ubx'][0] == 34,
          lbx0=float(mfix['lbx'][0]), ubx0=float(mfix['ubx'][0]))
    # B3 zone gradient inside (CasADi derivative)
    px, py = ca.SX.sym('px'), ca.SX.sym('py')
    J = ca.Function('J', [px, py], [ca.jacobian(N.zone_g(px, py, 0.0, 'front', D, ca.fmax), ca.vertcat(px, py))])
    gmin = min(float(np.linalg.norm(np.array(J(rng.uniform(x0f + 1e-3, x1f - 1e-3), rng.uniform(y0 + 1e-4, y1 - 1e-4))))) for _ in range(200))
    check('B3_E3_interior_gradient_casadi', gmin > 0.5, min_norm=gmin)

out = {'scope': 'unit tests of b2_nlp_v6; no optimizer call', 'casadi_available': casadi_ok,
       'casadi_version': getattr(ca, '__version__', None), 'numpy': np.__version__, 'python': sys.version,
       'b2_nlp_v6_sha256': N.sha(HERE / 'b2_nlp_v6.py'), 'test_sha256': N.sha(__file__),
       'tests': checks, 'pass': all(c['pass'] for c in checks), 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
(HERE / ('test_b2_nlp_v6.json' if casadi_ok else 'test_b2_nlp_v6_numpy_only.json')).write_text(
    json.dumps(N.clean(out), ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
print('passed', sum(c['pass'] for c in checks), 'of', len(checks), '; casadi', casadi_ok)
sys.exit(0 if out['pass'] else 1)
