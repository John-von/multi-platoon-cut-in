"""B2 baseline planning problem (centralized planner adapted from Duan et al., 2023, cited as N01), implementation v6.

Unchanged N01 core: kinematic bicycle (eq. 1), trapezoidal collocation N_T=50, double circles, lane bounds (19)(20),
terminal conditions, cost (26) with lambda/gamma=100 (objective divided by 100), IPOPT+MUMPS settings.

Adaptation rules for the cut-in task:
E-1  Constraints whose definition contains a physical time difference (A-B2-3' 1-s windows, secant speed/acceleration
     stencils, secant windows) use physical endpoints on the 0.1-s grid. Per call the branch (history t<0 /
     plan 0<=t<=tf_r / constant-speed extension t>tf_r) and the collocation interval k (same rule as the audit
     interpolation) are frozen; inside the NLP the fraction f = N_T t / t_f - k is a smooth function of the decision
     variable t_f (polynomial extension when f leaves [0,1]). Single-time point constraints keep the normalised tau
     of the original rule.
E-2  Stage S0 = first solve of the sensitivity version (node bound |pdd_{j+1}-pdd_j| <= (t_f/N_T) * 1 m/s^3). The
     main version is warm-started from S0. Caps: main S0+5, sensitivity 5 incl. S0, original 5.
E-3  Directed forbidden zone g = max{(x0-dx)-xc, xc-(x1+dx), (y0-dy)-yc, yc-(y1+dy)} >= 0, dx = eps_a/H,
     dy = eps_a/L; also used for acceptance.
E-4  Jerk acceptance of main and sensitivity on 1-s windows of the issued secant reference (history and post-t_f
     zeros included); the sensitivity node bound is re-checked at nodes; the original version neither applies nor
     accepts a jerk limit.
E-5  Pair pruning by body-centre distance over nodes U 0.1-s grid (>30 m omitted); after every solve all pairs are
     audited (double-circle gap) on nodes U grid and violating pairs are re-added.
E-7  Main version: the 1-s windows act directly on the issued secant acceleration
     a_bar(j) = (p(t_{j+2}) - 2p(t_{j+1}) + p(t_j))/0.01 of the centre position (E-1 endpoints, history rows for j<0,
     zero on the extension), i.e. exactly the acceptance quantity; all windows are rebuilt every call.
E-8  Sensitivity version: node bound plus |pdd_0 - a_hist| <= 0.05 m/s^2 (continuity with the last measured 0.1-s
     interval); acceptance as in E-4.
E-10 Calls that contain physical-time constraints are solved with t_f FIXED (bounds lbx=ubx), so the E-1 freezing is
     exact; the minimal acceptable t_f is found by a direction search (3% steps) and bisection to 1% relative
     resolution; each candidate t_f is solved with constraint generation (<=3 calls, 900 s CPU per call). Free-t_f
     stage: S0 (+1 free main call) or the first original call. Identical free-stage solutions of different initial
     values share the fixed stage.
I-1  Sensitivity version only: a secant 1-s window that fails acceptance is added, for that vehicle, as a constraint
     on the secant stencils themselves (E-1 endpoints) and kept for the remaining calls of the chain.

The numerical helpers are backend-generic, so the same code path is evaluated with NumPy in the tests and with
CasADi SX in the NLP."""
from pathlib import Path
import argparse, copy, hashlib, json, os, sys, time

for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('B2_ROOT', str(HERE.parents[1])))          # package root
os.environ.setdefault('V20_SRC', str(ROOT / 'code' / 'sim' / 'sources'))
os.environ.setdefault('V20_ROOT', str(ROOT / 'code' / 'sim'))
sys.path.insert(0, str(HERE))
import numpy as np
try:
    import casadi as ca
except ImportError:          # numeric parts stay testable without CasADi
    ca = None
import b2_components as B
import frozen_scene

CROSSING_BOUND = 0.05          # E-8: |pdd_0 - a_hist| <= 0.05 m/s^2
REQUEST = ROOT / 'data' / 'baseline_b2' / 'request_state_EX1a_nominal.json'
COMPONENTS_SHA = '8f2cbcc2e11a95a1804a1d940bd973c04ab884098e28e34dd5743b46a7ddcb93'

NT = 50
TOL = 1e-6
DT = 0.1
PAIR_RADIUS = 30.0
VARIANTS = ('main', 'sensitivity', 'original')
MAX_CALLS = {'main': 5, 'sensitivity': 5, 'original': 5}   # main additionally has S0
CHAIN_BUDGET_S = 3 * 3600.0          # E-10: per job
FIXED_CPU_S = 900.0
FIXED_MAX_CALLS = 3
STEP = 0.03
REL_RES = 0.01
MAX_DOWN = 4
MAX_UP = 6
OPTIONS = {'ipopt.linear_solver': 'mumps', 'ipopt.tol': 1e-8, 'ipopt.constr_viol_tol': 1e-8,
           'ipopt.acceptable_tol': 1e-7, 'ipopt.max_iter': 3000, 'ipopt.max_cpu_time': 1800.,
           'ipopt.mu_strategy': 'adaptive', 'ipopt.bound_relax_factor': 0., 'ipopt.print_level': 5,
           'print_time': True}
OK_STATUS = ('Solve_Succeeded', 'Solved_To_Acceptable_Level')
ZONE_H = B.Y_TOP - (B.Y_S - 1e-3)          # rectangle height H of A-B2-2''


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clean(x):
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return str(x)
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in (sorted(x) if isinstance(x, set) else x)]
    return x


def dump(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    x = {'b2_nlp_v6_sha256': sha(__file__), **x}
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(x), ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    os.replace(tmp, p)


# ------------------------------------------------------------------ backends
class NumOps:
    cos = staticmethod(np.cos)
    sin = staticmethod(np.sin)
    fmax = staticmethod(np.maximum)

    @staticmethod
    def vec(xs):
        return np.array(xs, dtype=float)

    @staticmethod
    def centre(s, u):
        return B.centre_from_rear(*[s[j] for j in range(5)], *[u[j] for j in range(2)])


class SymOps:
    cos = staticmethod(lambda x: ca.cos(x))
    sin = staticmethod(lambda x: ca.sin(x))
    fmax = staticmethod(lambda a, b: ca.fmax(a, b))

    @staticmethod
    def vec(xs):
        return ca.vertcat(*xs)

    centre = None      # set per build to the CasADi centre Function


class Nodes:
    """Node data of one candidate: per-vehicle S (5,NT+1), U (2,NT+1), F (5,NT+1) and t_f.
    Works with NumPy arrays or CasADi SX matrices."""

    def __init__(self, S, U, F, tf):
        self.S, self.U, self.F, self.tf = S, U, F, tf

    @classmethod
    def numeric(cls, tf, S, U):
        F = [np.stack([np.asarray(B.rhs(S[i][:, k], U[i][:, k]), float) for k in range(S.shape[2])], axis=1)
             for i in range(S.shape[0])]
        return cls([S[i] for i in range(S.shape[0])], [U[i] for i in range(U.shape[0])], F, float(tf))


# ------------------------------------------------------------------ E-1 physical endpoints
def freeze(t, tf_r):
    """Branch and interval frozen for physical time t (s after t0) in a call built at tf_r.
    history -> (branch, history row); plan -> (branch, k) with the audit interpolation rule;
    extension -> (branch, -1)."""
    if t < -1e-9:
        return ('history', int(round((t + 1.0) / DT)))
    if t > tf_r + 1e-9:
        return ('extension', -1)
    tn = np.linspace(0.0, tf_r, NT + 1)
    k = int(np.clip(np.searchsorted(tn, t, side='right') - 1, 0, NT - 1))
    return ('plan', k)


def freeze_grid(j, tf_r):
    return freeze(j * DT, tf_r)


def phys_state(nd, i, t, frz, ops):
    """State and control of vehicle i at physical time t with frozen (branch, k)."""
    br, k = frz
    if br == 'extension':
        return nd.S[i][:, NT] + ops.vec([20.0 * (t - nd.tf), 0., 0., 0., 0.]), ops.vec([0., 0.])
    if br != 'plan':
        raise ValueError('state requested on history branch')
    h = nd.tf / NT
    f = NT * t / nd.tf - k
    Sk, Sk1, Fk, Fk1 = nd.S[i][:, k], nd.S[i][:, k + 1], nd.F[i][:, k], nd.F[i][:, k + 1]
    s = Sk + h * (Fk * f + 0.5 * f * f * (Fk1 - Fk))
    u = nd.U[i][:, k] + f * (nd.U[i][:, k + 1] - nd.U[i][:, k])
    return s, u


def phys_p(nd, i, t, frz, ops):
    s, _ = phys_state(nd, i, t, frz, ops)
    return s[0] + B.L_R * ops.cos(s[3])


def phys_pdd(nd, i, j, frz, hist, ops):
    """Complete-formula centre acceleration at grid index j (t = j*DT); history rows for t<0,
    0 on the registered constant-speed extension."""
    br, k = frz
    if br == 'history':
        return float(hist[k, i])
    if br == 'extension':
        return 0.0
    s, u = phys_state(nd, i, j * DT, frz, ops)
    return ops.centre(s, u)[3]


def secant_va(nd, i, j, fz, ops):
    """Issued secant speed and acceleration at grid index j >= 0 (positions at j, j+1, j+2)."""
    ps = [phys_p(nd, i, (j + m) * DT, fz(j + m), ops) for m in range(3)]
    return (ps[1] - ps[0]) / DT, (ps[2] - 2 * ps[1] + ps[0]) / DT ** 2


def secant_a(nd, i, j, fz, ops):
    if all(fz(j + m)[0] == 'extension' for m in range(3)):
        return 0.0                       # exactly zero on the linear extension
    return secant_va(nd, i, j, fz, ops)[1]


def secant_window(nd, i, j, fz, hist, ops):
    """Window starting at grid index j (j>=-10): a_bar(j+10) - a_bar(j); history rows for j<0."""
    start = float(hist[j + 10, i]) if j < 0 else secant_a(nd, i, j, fz, ops)
    return secant_a(nd, i, j + 10, fz, ops) - start


def tau_state(nd, i, tau):
    """Single-time point constraint at normalised tau (original rule: k=floor(tau*NT), f const)."""
    if tau > 1:
        raise ValueError('point constraints are only added inside the plan')
    k = min(NT - 1, max(0, int(np.floor(tau * NT))))
    f = tau * NT - k
    h = nd.tf / NT
    Sk, Sk1, Fk, Fk1 = nd.S[i][:, k], nd.S[i][:, k + 1], nd.F[i][:, k], nd.F[i][:, k + 1]
    return Sk + h * (Fk * f + 0.5 * f * f * (Fk1 - Fk)), nd.U[i][:, k] + f * (nd.U[i][:, k + 1] - nd.U[i][:, k])


# ------------------------------------------------------------------ E-3 zone
def zone_g(px, py, po, role, D, fmax):
    x0, x1, y0, y1 = B.zone_rect(role, po, D)
    dx = B.EPS_A / ZONE_H
    dy = B.EPS_A / (D + 10.0)
    return fmax(fmax((x0 - dx) - px, px - (x1 + dx)), fmax((y0 - dy) - py, py - (y1 + dy)))


# ------------------------------------------------------------------ scene and initial guess
def scene(single=False):
    assert sha(HERE / 'b2_components.py') == COMPONENTS_SHA, 'b2_components hash mismatch'
    raw = json.loads(REQUEST.read_text('utf-8'))
    sc = frozen_scene.load_scenario()(); N, Q = sc['N'], sc['Q']
    assert raw['lateral_columns'] == ['y', 'heading', 'v_y', 'yaw_rate']
    queues = [list(range(int(sc['starts'][q]), int(sc['starts'][q + 1]) if q + 1 < Q else N)) for q in range(Q)]
    if single:
        queues = queues[:1]; Q = 1
    ids = [i for q in queues for i in q] + list(range(N, N + Q)); lookup = {g: i for i, g in enumerate(ids)}
    m = len(ids); s0 = np.zeros((m, 5)); x = np.array(raw['state']); lat = np.array(raw['lateral'])
    for k, g in enumerate(ids):
        y, th = (3.5, 0.) if g < N else (lat[g - N, 0], lat[g - N, 1])
        s0[k] = [x[g, 0] - B.L_R * np.cos(th), y - B.L_R * np.sin(th), x[g, 1], th, 0.]
    qs = [[lookup[i] for i in q] for q in queues]; cut = [lookup[N + q] for q in range(Q)]
    fronts = [lookup[int(sc['front'][q])] for q in range(Q)]; rears = [lookup[int(sc['rear'][q])] for q in range(Q)]
    orders = []
    for q, c, f in zip(qs, cut, fronts):
        j = q.index(f); orders.append(q[:j + 1] + [c] + q[j + 1:])
    history = np.asarray(raw['acceleration_history'], dtype=float)
    assert history.shape == (10, N + sc['Q']) and np.all(np.isfinite(history))
    hist = history[:, ids].copy()
    return {'m': m, 'ids': ids, 's0': s0, 'queues': qs, 'ordinary': [i for q in qs for i in q],
            'cut': cut, 'fronts': fronts, 'rears': rears, 'orders': orders, 'p0': x[ids, 0],
            'history': hist, 'source_sha256': raw['source_sha256'], 'single': single}


def initial(sc, Tg):
    m = sc['m']; p0 = sc['p0']; q = sc['queues']
    targets = B.target_configuration(dict(enumerate(p0)), q, dict(enumerate(sc['cut'])), dict(enumerate(sc['fronts'])))
    pairs = {c: (f, r) for c, f, r in zip(sc['cut'], sc['fronts'], sc['rears'])}
    g = B.initial_guess(p0, targets, pairs, Tg)
    tf = float(g['tf_guess']); tn = np.linspace(0, tf, NT + 1); t = g['t'] - 5
    S = np.zeros((m, 5, NT + 1)); U = np.zeros((m, 2, NT + 1))
    for i in range(m):
        p = g['p'][i]; y = np.full_like(p, 3.5) if i in sc['ordinary'] else g['y'][i]
        pd = np.gradient(p, t, edge_order=2); yd = np.gradient(y, t, edge_order=2)
        th = np.arctan2(yd, pd); v = np.hypot(pd, yd)
        thd = np.gradient(th, t, edge_order=2); ph = np.arctan(B.LW * thd / v)
        a = np.gradient(v, t, edge_order=2); om = np.gradient(ph, t, edge_order=2)
        z = [p - B.L_R * np.cos(th), y - B.L_R * np.sin(th), v, th, ph]
        for j in range(5):
            S[i, j] = np.interp(tn, t, z[j])
        for j, ar in enumerate([a, om]):
            U[i, j] = np.interp(tn, t, ar)
        S[i, :, 0] = sc['s0'][i]; S[i, 1:, -1] = [3.5, 20., 0., 0.]
        S[i, 0, -1] = targets[i] + 20 * tf - B.L_R
        U[i, :, -1] = 0.
        U[i, :, 0] = 0.
    return tf, S, U, {'Tg': Tg, 't_open_absolute_s': g['t_open'], 'targets_rho': targets,
                      'construction': 'registered initial guess (A-B2-6\' steps 1-4, item 7 true initial state)'}


def centres(S, U):
    return np.asarray(B.centre_from_rear(*[S[:, i] for i in range(5)], *[U[:, i] for i in range(2)]))


def all_pairs(sc):
    return [(i, j) for i in range(sc['m']) for j in range(i + 1, sc['m'])]


def geometric_margins(sc, S, U, detail=False):
    """Natural-unit margins (>=0 satisfied) of sampled states (vehicle, state, time)."""
    cp = centres(S, U); p, y, pd, pdd = cp
    rows = {}

    def add(name, value):
        rows[name] = np.asarray(value, float)
    add('rear_speed_lower', S[:, 2] - 5); add('rear_speed_upper', 23 - S[:, 2]); add('a', 2 - abs(U[:, 0]))
    add('theta', np.pi / 3 - abs(S[:, 3])); add('phi', .25 - abs(S[:, 4])); add('omega', .1 - abs(U[:, 1]))
    add('centre_speed_lower', pd - 5); add('centre_speed_upper', 23 - pd); add('centre_acc', 2 - abs(pdd))
    circles = B.circle_centres(S[:, 0], S[:, 1], S[:, 3])
    low = np.array([1.75 if i in sc['ordinary'] else -1.75 for i in range(sc['m'])])[:, None]
    for j, (cx, cy) in enumerate(circles):
        add(f'lane_lower_{j}', cy - low - B.CIRCLE_R); add(f'lane_upper_{j}', 5.25 - B.CIRCLE_R - cy)
    for a, b in zip(sc['ordinary'][:-1], sc['ordinary'][1:]):
        add(f'ordinary_gap_{a}_{b}', p[a] - p[b] - 10)
    for a, b in zip(sc['cut'][:-1], sc['cut'][1:]):
        add(f'source_gap_{a}_{b}', p[a] - p[b] - 10)
    area = {}
    for c, f, r in zip(sc['cut'], sc['fronts'], sc['rears']):
        for role, o in [('front', f), ('rear', r)]:
            D = B.zone_D(sc['p0'][c], sc['p0'][o])
            add(f'zone_{c}_{role}', zone_g(p[c], y[c], p[o], role, D, np.maximum))
            area[f'zone_area_form_{c}_{role}'] = B.zone_margin(p[c], y[c], B.zone_rect(role, p[o], D))
    pair_gap = {}
    mins = np.full(p.shape[1], np.inf)
    for i, j in all_pairs(sc):
        gij = np.full(p.shape[1], np.inf)
        for xa, ya in circles:
            for xb, yb in circles:
                gij = np.minimum(gij, np.hypot(xa[i] - xb[j], ya[i] - yb[j]) - 2 * B.CIRCLE_R)
        pair_gap[(i, j)] = float(gij.min()); mins = np.minimum(mins, gij)
    add('circle_gap_all_pairs', mins)
    if detail:
        return rows, pair_gap, area
    return rows


def endpoint_residual(sc, S, U):
    p = centres(S, U)[0]
    parts = {'initial': float(abs(S[:, :, 0] - sc['s0']).max()),
             'terminal_y_v_theta_phi': float(abs(S[:, 1:, -1] - np.array([3.5, 20, 0, 0])).max()),
             'terminal_controls': float(abs(U[:, :, -1]).max())}
    gaps = []
    for order in sc['orders']:
        gaps.extend(abs(p[a, -1] - p[b, -1] - 12) for a, b in zip(order[:-1], order[1:]))
    parts['terminal_T2_in_queue'] = max(gaps, default=0.)
    parts['terminal_between_queue'] = max([max(0., 12 - p[a[-1], -1] + p[b[0], -1])
                                          for a, b in zip(sc['orders'][:-1], sc['orders'][1:])], default=0.)
    return parts


def interpolate(sc, tf, S, U, times):
    times = np.asarray(times); ss = []; uu = []; dd = []; tn = np.linspace(0, tf, NT + 1)
    for i in range(sc['m']):
        s, u, sd = B.interpolate(tn, S[i], U[i], np.clip(times, 0, tf))
        after = times > tf
        if after.any():
            s[:, after] = S[i, :, -1, None]; s[0, after] += 20 * (times[after] - tf); u[:, after] = 0.
            sd[:, after] = np.array([20, 0, 0, 0, 0])[:, None]
        ss.append(s); uu.append(u); dd.append(sd)
    return np.array(ss), np.array(uu), np.array(dd)


def audit_times(tf):
    tn = np.linspace(0, tf, NT + 1)
    return np.unique(np.r_[tn, np.arange(0, tf + 1e-9, DT), tf]), tn


def initial_gate(sc, tf, S, U):
    """A-B2-6' item 7 node-level gate (zones with the E-3 function used for acceptance)."""
    rows, _, area = geometric_margins(sc, S, U, detail=True)
    times = np.unique(np.r_[np.arange(0, tf + 1e-9, DT), tf]); si, ui, _ = interpolate(sc, tf, S, U, times)
    interp, _, area_i = geometric_margins(sc, si, ui, detail=True)
    nodes = {k: float(np.min(v)) for k, v in rows.items()}
    dense = {k: float(np.min(v)) for k, v in interp.items() if k.startswith('zone_')}
    required = [v for k, v in nodes.items() if k.startswith(('zone_', 'rear_speed', 'centre_')) or k in ('a', 'theta', 'phi', 'omega')] + list(dense.values())
    ep = endpoint_residual(sc, S, U); bad = max(max(ep.values()), max(0., -min(required)), max(0., 1 - tf, tf - 34))
    col = max(float(abs(B.collocation_residual(np.linspace(0, tf, NT + 1), S[i], U[i])).max()) for i in range(sc['m']))
    return {'pass': bad <= TOL, 'max_required_violation': bad, 'node_min_margins': nodes,
            'interpolated_zone_min_margins': dense,
            'area_form_zone_min_margins_diagnostic': {k: float(np.min(v)) for k, v in {**area, **{k + '_interp': v for k, v in area_i.items()}}.items()},
            'endpoint_residuals': ep, 'collocation_residual_diagnostic_not_initial_gate': col}


# ------------------------------------------------------------------ E-5 pruning
def centre_distances(sc, tf, S, U):
    times, _ = audit_times(tf)
    ss, uu, _ = interpolate(sc, tf, S, U, times); cp = centres(ss, uu)
    return {(i, j): float(np.min(np.hypot(cp[0][i] - cp[0][j], cp[1][i] - cp[1][j]))) for i, j in all_pairs(sc)}


def pair_set(sc, tf, S, U, violators=()):
    d = centre_distances(sc, tf, S, U)
    keep = set(pr for pr, v in d.items() if v <= PAIR_RADIUS) | set(tuple(v) for v in violators)
    return sorted(keep), d


# ------------------------------------------------------------------ NLP
def symbolic_functions():
    s = ca.SX.sym('s', 5); u = ca.SX.sym('u', 2)
    sl = [s[i] for i in range(5)]; ul = [u[i] for i in range(2)]
    return (ca.Function('rhs', [s, u], [ca.vertcat(*B.rhs(sl, ul))]),
            ca.Function('centre', [s, u], [ca.vertcat(*B.centre_from_rear(*sl, *ul))]))


def window_starts(tf_r):
    return list(range(-10, int(np.floor(tf_r / DT + 1e-8)) + 1))


def pdd_window_starts(tf_r):
    return list(range(-10, int(np.floor(tf_r / DT + 1e-8)) + 1))


def build(sc, variant, tf_r, points=(), secs=(), wins=(), pair_subset=None, fixed_tf=False):
    """NLP of one call. tf_r: t_f of the warm start (freezing of E-1 endpoints).
    points: normalised tau (single-time constraints); secs: grid indices of secant stencils;
    wins: (vehicle, start index) of I-1 secant windows."""
    assert variant in VARIANTS
    m = sc['m']; F, CENT = symbolic_functions()
    tf = ca.SX.sym('tf'); S = [ca.SX.sym(f'S{i}', 5, NT + 1) for i in range(m)]; U = [ca.SX.sym(f'U{i}', 2, NT + 1) for i in range(m)]
    w = ca.vertcat(tf, *[ca.vec(a) for pair in zip(S, U) for a in pair]); g = []; lo = []; hi = []; groups = []

    def constrain(v, l, h, label):
        flat = ca.vec(v); n = int(flat.numel()); g.append(flat)
        lo.extend(np.broadcast_to(l, (n,))); hi.extend(np.broadcast_to(h, (n,))); groups.append((label, n))
    FF = [F.map(NT + 1)(s, u) for s, u in zip(S, U)]
    CP = [CENT.map(NT + 1)(s, u) for s, u in zip(S, U)]
    nd = Nodes(S, U, FF, tf)
    ops = SymOps; ops.centre = staticmethod(lambda s, u: CENT(s, u))
    frozen = {}

    def fz(j):
        if j not in frozen:
            frozen[j] = freeze_grid(j, tf_r)
        return frozen[j]
    pairs = all_pairs(sc) if pair_subset is None else [tuple(p) for p in pair_subset]

    def point(s, u, cp, label):
        for i in range(m):
            constrain(s[i][2], 5, 23, label + '_speed'); constrain(s[i][3], -np.pi / 3, np.pi / 3, label + '_theta')
            constrain(s[i][4], -.25, .25, label + '_phi')
            constrain(u[i][0], -2, 2, label + '_a'); constrain(u[i][1], -.1, .1, label + '_omega')
            constrain(cp[i][2], 5, 23, label + '_pd'); constrain(cp[i][3], -2, 2, label + '_pdd')
            for off in B.CIRCLE_OFFSETS:
                cy = s[i][1] + off * ca.sin(s[i][3]); lower = 1.75 if i in sc['ordinary'] else -1.75
                constrain(cy, lower + B.CIRCLE_R, 5.25 - B.CIRCLE_R, label + '_lane')
        for i, j in zip(sc['ordinary'][:-1], sc['ordinary'][1:]):
            constrain(cp[i][0] - cp[j][0], 10, np.inf, label + '_ordinary')
        for i, j in zip(sc['cut'][:-1], sc['cut'][1:]):
            constrain(cp[i][0] - cp[j][0], 10, np.inf, label + '_source')
        for c, f, r in zip(sc['cut'], sc['fronts'], sc['rears']):
            for role, o in [('front', f), ('rear', r)]:
                constrain(zone_g(cp[c][0], cp[c][1], cp[o][0], role, B.zone_D(sc['p0'][c], sc['p0'][o]), ca.fmax), 0, np.inf, label + '_zone')
        for i, j in pairs:
            for oi in B.CIRCLE_OFFSETS:
                for oj in B.CIRCLE_OFFSETS:
                    dx = s[i][0] + oi * ca.cos(s[i][3]) - s[j][0] - oj * ca.cos(s[j][3])
                    dy = s[i][1] + oi * ca.sin(s[i][3]) - s[j][1] - oj * ca.sin(s[j][3])
                    constrain(dx * dx + dy * dy, (2 * B.CIRCLE_R) ** 2, np.inf, label + '_circle_squared')
    energy = 0
    for i in range(m):
        constrain(S[i][:, 0] - sc['s0'][i], 0, 0, 'initial')
        constrain(S[i][1:, -1] - ca.DM([3.5, 20, 0, 0]), 0, 0, 'terminal'); constrain(U[i][:, -1], 0, 0, 'terminal_u')
        constrain(S[i][:, 1:] - S[i][:, :-1] - tf / (2 * NT) * (FF[i][:, 1:] + FF[i][:, :-1]), 0, 0, 'dynamics')
        ee = U[i][0, :] ** 2 + U[i][1, :] ** 2 + S[i][3, :] ** 2 + S[i][4, :] ** 2
        energy += tf / NT * (ca.sum2(ee) - .5 * (ee[0] + ee[-1]))
    for order in sc['orders']:
        for i, j in zip(order[:-1], order[1:]):
            constrain(CP[i][0, -1] - CP[j][0, -1], 12, 12, 'terminal_gap')
    for a, b in zip(sc['orders'][:-1], sc['orders'][1:]):
        constrain(CP[a[-1]][0, -1] - CP[b[0]][0, -1], 12, np.inf, 'terminal_boundary')
    for k in range(NT + 1):
        point([s[:, k] for s in S], [u[:, k] for u in U], [c[:, k] for c in CP], 'node')
    for tau in sorted(set(points)):
        pr = [tau_state(nd, i, tau) for i in range(m)]
        point([p[0] for p in pr], [p[1] for p in pr], [CENT(*p) for p in pr], 'dense')
    for j in sorted(set(secs)):
        for i in range(m):
            v, a = secant_va(nd, i, j, fz, ops)
            constrain(v, 5, 23, 'secant_v'); constrain(a, -2, 2, 'secant_a')
    n_main_windows = 0
    if variant == 'main':                                   # A-B2-3' main, E-7: issued secant windows, E-1 endpoints
        for j in window_starts(tf_r):
            for i in range(m):
                constrain(secant_window(nd, i, j, fz, sc['history'], ops), -1, 1, 'window_secant')
                n_main_windows += 1
    if variant == 'sensitivity':                            # A-B2-3' sensitivity: node bound + E-8 crossing
        for i in range(m):
            constrain((CP[i][3, 1:] - CP[i][3, :-1]) * (NT / tf), -1, 1, 'node_jerk')
            constrain(CP[i][3, 0] - float(sc['history'][-1, i]), -CROSSING_BOUND, CROSSING_BOUND, 'crossing_jerk')
    win_list = sorted(set(tuple(x) for x in wins)) if variant == 'sensitivity' else []   # I-1 (sensitivity only)
    for i, j in win_list:                                   # I-1
        constrain(secant_window(nd, i, j, fz, sc['history'], ops), -1, 1, 'secant_window')
    expr = ca.vertcat(*g); obj = tf + energy / 100.
    lb = np.full(int(w.numel()), -np.inf); ub = np.full(int(w.numel()), np.inf); lb[0] = 1; ub[0] = 34
    if fixed_tf:                                            # E-10: t_f fixed, E-1 freezing exact
        lb[0] = ub[0] = float(tf_r)
    return {'nlp': {'x': w, 'f': obj, 'g': expr}, 'lbg': np.array(lo, float), 'ubg': np.array(hi, float), 'lbx': lb, 'ubx': ub,
            'groups': groups, 'frozen': dict(frozen), 'pair_subset': pairs, 'n_main_windows': n_main_windows,
            'n_secant_windows': len(win_list), 'eval': ca.Function('evaluate', [w], [expr, obj, energy])}


def pack(tf, S, U):
    return np.r_[tf, *[v.ravel(order='F') for pair in zip(S, U) for v in pair]]


def unpack(w, m):
    w = np.asarray(w).ravel(); s = []; u = []; at = 1
    for i in range(m):
        s.append(w[at:at + 5 * (NT + 1)].reshape(5, NT + 1, order='F')); at += 5 * (NT + 1)
        u.append(w[at:at + 2 * (NT + 1)].reshape(2, NT + 1, order='F')); at += 2 * (NT + 1)
    return float(w[0]), np.array(s), np.array(u)


# ------------------------------------------------------------------ acceptance (E-3/E-4/E-5)
def executed_reference(sc, tf, S, U):
    tj = np.arange(int(np.ceil(tf / .1)) + 1) * .1
    sj, uj, dj = interpolate(sc, tf, S, U, tj); cp = centres(sj, uj)
    refs = [B.execution_reference(cp[0, i], 20) for i in range(sc['m'])]
    rv = np.array([r['v'] for r in refs]); ra = np.array([r['a'] for r in refs])
    hist = np.vstack([sc['history'], ra.T, np.zeros((10, sc['m']))])
    jw = hist[10:] - hist[:-10]                       # row w <-> window start index j = w - 10
    return tj, sj, uj, dj, cp, refs, rv, ra, jw


def audit(sc, variant, tf, S, U):
    times, tn = audit_times(tf)
    ss, uu, _ = interpolate(sc, tf, S, U, times)
    rows, pair_gap, area = geometric_margins(sc, ss, uu, detail=True)
    mins = {k: float(v.min()) for k, v in rows.items()}
    argmin = {}
    for k, v in rows.items():
        ix = np.unravel_index(int(np.argmin(v)), v.shape)
        argmin[k] = {'time_s': float(times[ix[-1]]), 'vehicle': int(ix[0]) if v.ndim == 2 else None}
    is_node = np.array([np.min(np.abs(tn - t)) <= 1e-9 for t in times])
    badpoint = set()
    for k, v in rows.items():
        if k.startswith(('centre_speed', 'centre_acc')):
            continue                                   # diagnostic; accepted on the secant reference
        viol = np.any(v < -TOL, axis=0) if v.ndim == 2 else (v < -TOL)
        badpoint.update(float(times[jx] / tf) for jx in np.flatnonzero(viol & ~is_node))
    violating_pairs = sorted(pr for pr, gmin in pair_gap.items() if gmin < -TOL)
    tj, sj, uj, dj, cp, refs, rv, ra, jw = executed_reference(sc, tf, S, U)
    bad_sec = set(int(j) for j in np.flatnonzero(np.any((rv < 5 - TOL) | (rv > 23 + TOL), axis=0)))
    bad_sec |= set(int(j) for j in np.flatnonzero(np.any(np.abs(ra) > 2 + TOL, axis=0)))
    win_margin = 1 - np.abs(jw)
    jerk = variant in ('main', 'sensitivity')
    bad_win = sorted((int(i), int(wr - 10)) for wr, i in zip(*np.nonzero(win_margin < -TOL))) if jerk else []
    execmins = {'secant_speed': float(min((rv - 5).min(), (23 - rv).min())), 'secant_acc': float((2 - abs(ra)).min())}
    wix = np.unravel_index(int(np.argmin(win_margin)), win_margin.shape)
    window_info = {'min_margin': float(win_margin.min()), 'start_time_s': float((wix[0] - 10) * DT), 'vehicle': int(wix[1]),
                   'n_windows_checked': int(win_margin.size), 'n_violating': int((win_margin < -TOL).sum())}
    ncp = centres(S, U)
    node_jerk = np.diff(ncp[3], axis=1) * NT / tf
    node_jerk_margin = float((1 - abs(node_jerk)).min())
    crossing_margin = float((CROSSING_BOUND - abs(ncp[3][:, 0] - sc['history'][-1])).min())
    ep = endpoint_residual(sc, S, U)
    col = max(float(abs(B.collocation_residual(tn, s, u)).max()) for s, u in zip(S, U))
    accepted = {k: v for k, v in mins.items() if not k.startswith(('centre_speed', 'centre_acc'))}
    accepted.update(execmins)
    if jerk:
        accepted['secant_window'] = window_info['min_margin']
    if variant == 'sensitivity':
        accepted['node_jerk'] = node_jerk_margin
        accepted['crossing_jerk'] = crossing_margin
    maxvio = max(0., -min(accepted.values()), max(ep.values()), col)
    disc = 0.; zq = 0.
    for r in refs:
        for j in range(len(r['q'])):
            nxt = B.nominal_step(r['p'][j], r['v'][j], r['z'][j], r['q'][j])
            disc = max(disc, max(abs(nxt[k] - r[key][j + 1]) for k, key in enumerate(['p', 'v', 'z'])))
        z = B.DRAG * r['v'][:-1] ** 2 + (r['a'] + B.GF) / B.MU_HAT; q = z[:-1] + np.diff(z) / B.LAM_HAT
        zq = max(zq, float(abs(z - r['z']).max()), float(abs(q - r['q']).max()))
    clipped = np.minimum(cp[1], 3.495); clipdiff = cp[1] - clipped
    rhs = np.array([[B.rhs(sj[i, :, j], uj[i, :, j]) for j in range(len(tj))] for i in range(sc['m'])]).transpose(0, 2, 1)
    interp_pd = dj[:, 0] - B.L_R * np.sin(sj[:, 3]) * dj[:, 3]
    report = {'variant': variant, 'pass': maxvio <= TOL, 'max_violation': maxvio, 'dynamics_residual': col,
              'endpoint_residuals': ep, 'accepted_min_margins': accepted, 'all_geometry_min_margins': mins,
              'all_geometry_argmin': argmin, 'execution_min_margins': execmins, 'secant_window': window_info,
              'jerk_acceptance_applies': jerk, 'node_jerk_margin': node_jerk_margin, 'crossing_margin_m_s2': crossing_margin,
              'node_jerk_acceptance_applies': variant == 'sensitivity',
              'zone_area_form_min_margins_diagnostic': {k: float(np.min(v)) for k, v in area.items()},
              'violating_pairs': [list(p) for p in violating_pairs],
              'bad_point_tau': sorted(badpoint), 'bad_secant_start': sorted(bad_sec), 'bad_window': [list(x) for x in bad_win],
              'regressions': {'1_discrete_one_step_max': disc,
                              '2_secant_minus_rhs_v': float(abs(rv[:, :len(tj)] - cp[2]).max()),
                              '2_secant_minus_rhs_a': float(abs(ra[:, :len(tj)] - cp[3]).max()),
                              '3_zq_formula_max': zq,
                              '4_and_5': 'synthetic reduction/nonzero-heading tests',
                              '6_clipping_max_m': float(abs(clipdiff).max()),
                              '6_clipping_duration_s_per_vehicle': np.count_nonzero(abs(clipdiff[:, :-1]) > 1e-12, axis=1) * .1,
                              '7_secant_bounds_and_windows': {**execmins, 'secant_window': window_info['min_margin']}},
              'derivative_diagnostics': {'rhs_minus_quadratic_derivative_max': float(abs(rhs - dj).max()),
                                         'rhs_pdot_minus_interpolant_pdot_max': float(abs(cp[2] - interp_pd).max())}}
    arrays = {'time': tj, 'p_ref': cp[0], 'y_plan': cp[1], 'y_ref': clipped, 'v_ref': rv, 'a_ref': ra,
              'z_ref': np.array([r['z'] for r in refs]), 'q_ref': np.array([r['q'] for r in refs]), 'window_jw': jw}
    return report, arrays


def frozen_mismatch(frozen, tf_new):
    """Endpoints whose branch/interval at the new t_f differs from the frozen one (E-1 record)."""
    out = []
    for j, f in frozen.items():
        now = freeze_grid(j, tf_new)
        if now != f:
            out.append([j, list(f), list(now)])
    return out


# ------------------------------------------------------------------ one IPOPT call
def solve_call(sc, variant, tf, S, U, sets, pairs, folder, label, fixed_tf=False):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    t_build = time.perf_counter()
    model = build(sc, variant, tf, sets['points'], sets['secs'], sets['wins'], pair_subset=pairs, fixed_tf=fixed_tf)
    opts = dict(OPTIONS)
    if fixed_tf:
        opts['ipopt.max_cpu_time'] = FIXED_CPU_S
    solver = ca.nlpsol('B2', 'ipopt', model['nlp'], opts); build_s = time.perf_counter() - t_build
    w0 = pack(tf, S, U)
    t0 = time.perf_counter()
    sol = solver(x0=w0, lbg=model['lbg'], ubg=model['ubg'], lbx=model['lbx'], ubx=model['ubx'])
    solve_s = time.perf_counter() - t0
    stats = solver.stats(); w = np.array(sol['x']).ravel(); tf1, S1, U1 = unpack(w, sc['m'])
    g = np.array(sol['g']).ravel()
    primal = float(np.maximum.reduce([model['lbg'] - g, g - model['ubg'], np.zeros_like(g)]).max()) if g.size else 0.
    lamg = np.array(sol['lam_g']).ravel(); lamx = np.array(sol['lam_x']).ravel()
    t_kkt = time.perf_counter()
    x = model['nlp']['x']
    grad = ca.Function('grad_f', [x], [ca.gradient(model['nlp']['f'], x)])
    jac = ca.Function('jac_g', [x], [ca.jacobian(model['nlp']['g'], x)])
    dual = float(abs(np.array(grad(w)).ravel() + np.array(ca.mtimes(jac(w).T, ca.DM(lamg))).ravel() + lamx).max())
    kkt_s = time.perf_counter() - t_kkt
    rep, arrays = audit(sc, variant, tf1, S1, U1)
    vals = model['eval'](w); energy = float(vals[2])
    ok = stats['return_status'] in OK_STATUS
    accepted = bool(ok and rep['pass'] and primal <= TOL)
    np.savez_compressed(folder / 'solution.npz', S=S1, U=U1, tf=tf1, lam_g=lamg, lam_x=lamx,
                        global_vehicle_indices=np.array(sc['ids']), **arrays)
    groups = {}
    for name, n in model['groups']:
        groups[name] = groups.get(name, 0) + n
    rec = {'label': label, 'variant': variant, 'return_status': stats['return_status'], 'solver_ok': ok, 'accepted': accepted,
           'iter_count': stats.get('iter_count'), 'tf_start': tf, 'tf': tf1, 'energy': energy,
           'objective_raw_lambda100_gamma1': 100 * tf1 + energy, 'objective_scaled': float(sol['f']),
           'solver_primal_natural_max': primal, 'stationarity_scaled_objective_inf': dual,
           'ipopt_final_inf_pr': (stats.get('iterations', {}).get('inf_pr') or [None])[-1],
           'ipopt_final_inf_du': (stats.get('iterations', {}).get('inf_du') or [None])[-1],
           'build_wall_s': build_s, 'solve_wall_s': solve_s, 'kkt_wall_s': kkt_s, 't_proc_total': stats.get('t_proc_total'),
           'options': opts, 'fixed_tf': bool(fixed_tf), 'ipopt_nlp_scaling': 'IPOPT default gradient-based; objective divided by 100',
           'n_variables': int(w.size), 'n_constraints': int(g.size), 'constraint_groups': groups,
           'n_point_times': len(sets['points']), 'n_secant_stencils': len(sets['secs']),
           'n_main_windows': model['n_main_windows'], 'n_secant_windows': model['n_secant_windows'],
           'pair_constraints': len(model['pair_subset']), 'pair_subset': [list(p) for p in model['pair_subset']],
           'frozen_endpoints': len(model['frozen']),
           'frozen_branch_counts': {b: sum(1 for f in model['frozen'].values() if f[0] == b) for b in ('history', 'plan', 'extension')},
           'frozen_vs_final_mismatch': frozen_mismatch(model['frozen'], tf1),
           'final_audit': rep, 'npz_sha256': sha(folder / 'solution.npz')}
    dump(folder / 'result.json', rec)
    return rec, (tf1, S1, U1), rep


def next_sets(sets, rep, variant):
    out = {'points': set(sets['points']) | set(rep['bad_point_tau']),
           'secs': set(sets['secs']) | set(rep['bad_secant_start']),
           'wins': set(tuple(x) for x in sets['wins'])}
    if variant == 'sensitivity':                     # I-1 only for the sensitivity version (E-7/E-8)
        out['wins'] |= set(tuple(x) for x in rep['bad_window'])
    return out


def sets_json(sets):
    return {'points': sorted(sets['points']), 'secs': sorted(sets['secs']), 'wins': sorted(list(x) for x in sets['wins'])}


def sets_load(d):
    return {'points': set(d['points']), 'secs': set(d['secs']), 'wins': set(tuple(x) for x in d['wins'])}


def run_chain(sc, variant, tf, S, U, root, first_call, max_calls, budget_s, t_start, sets=None, pairs=None):
    """Calls first_call..first_call+max_calls-1 of one version; stops on solver failure,
    acceptance or budget. Returns status, records and the last iterate."""
    root = Path(root)
    sets = sets or {'points': set(), 'secs': set(), 'wins': set()}
    if pairs is None:
        pairs, _ = pair_set(sc, tf, S, U)
    calls = []; status = 'planning_acceptance_failed'
    for n in range(first_call, first_call + max_calls):
        if time.perf_counter() - t_start > budget_s:
            status = 'budget_exhausted'; break
        rec, (tf, S, U), rep = solve_call(sc, variant, tf, S, U, sets, pairs, root / f'call_{n}', f'{variant} call {n}')
        calls.append(rec)
        sets = next_sets(sets, rep, variant)
        pairs, _ = pair_set(sc, tf, S, U, violators=rep['violating_pairs'])
        dump(root / f'call_{n}' / 'next.json', {'sets': sets_json(sets), 'pairs': [list(p) for p in pairs]})
        if not rec['solver_ok']:
            status = 'solver_failed'; break
        if rec['accepted']:
            status = 'accepted_plan'; break
    return status, calls, (tf, S, U), sets, pairs


def summary(rec_list):
    return [{k: r.get(k) for k in ('label', 'return_status', 'accepted', 'iter_count', 'tf', 'objective_raw_lambda100_gamma1',
                                   'solver_primal_natural_max', 'solve_wall_s', 'build_wall_s', 'n_constraints', 'pair_constraints',
                                   'n_point_times', 'n_secant_stencils', 'n_main_windows', 'n_secant_windows')}
            | {'max_violation': r['final_audit']['max_violation'], 'secant_window_min': r['final_audit']['secant_window']['min_margin']}
            for r in rec_list]


# ------------------------------------------------------------------ E-10 fixed-t_f stage
class BudgetExhausted(Exception):
    pass


def fixed_eval(sc, variant, tf_fix, S, U, sets, pairs, folder, budget_s, t0):
    """Constraint generation at a fixed t_f (<= FIXED_MAX_CALLS calls)."""
    folder = Path(folder); calls = []; status = 'not_accepted_within_call_cap'
    sets = copy.deepcopy(sets); last = None
    for n in range(1, FIXED_MAX_CALLS + 1):
        if time.perf_counter() - t0 > budget_s:
            status = 'budget_exhausted'; break
        rec, (tf1, S, U), rep = solve_call(sc, variant, tf_fix, S, U, sets, pairs, folder / f'call_{n}',
                                           f'{variant} fixed tf={tf_fix:.6f} call {n}', fixed_tf=True)
        calls.append(rec); last = rec
        sets = next_sets(sets, rep, variant)
        pairs, _ = pair_set(sc, tf1, S, U, violators=rep['violating_pairs'])
        if not rec['solver_ok']:
            status = 'solver_failed'; break
        if rec['accepted']:
            status = 'accepted'; break
    return {'tf': tf_fix, 'accepted': status == 'accepted', 'status': status, 'calls': summary(calls),
            'objective': None if last is None else last['objective_raw_lambda100_gamma1'],
            'solver_ok_last': None if last is None else last['solver_ok'],
            'folder': str(folder), 'final_call_folder': str(folder / f'call_{len(calls)}') if calls else None,
            'S': S, 'U': U, 'sets': sets, 'pairs': pairs}


def fixed_stage(sc, variant, t_s, S, U, sets, pairs, root, budget_s, t0):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    cands = {}; order = []
    key = lambda tf: round(float(tf), 6)

    def warm(tf):
        acc = [k for k in cands if cands[k]['accepted']]
        pool = acc or [k for k in cands if cands[k]['solver_ok_last']]
        if not pool:
            return S, U, sets, pairs
        k = min(pool, key=lambda c: abs(c - tf)); c = cands[k]
        return c['S'], c['U'], c['sets'], c['pairs']

    def record(state):
        dump(root / 'fixed_stage.json', {'variant': variant, 'start_tf': t_s, 'state': state, 'order': order,
                                         'candidates': {f'{k:.6f}': {kk: v for kk, v in cands[k].items() if kk not in ('S', 'U', 'sets', 'pairs')}
                                                        for k in order}})

    def ev(tf):
        k = key(tf)
        if k in cands:
            return cands[k]['accepted']
        if time.perf_counter() - t0 > budget_s:
            raise BudgetExhausted()
        Sw, Uw, setsw, pairsw = warm(k)
        cands[k] = fixed_eval(sc, variant, k, Sw, Uw, setsw, pairsw, root / f'tf_{k:.6f}', budget_s, t0)
        order.append(k); record('running')
        if cands[k]['status'] == 'budget_exhausted':
            raise BudgetExhausted()
        return cands[k]['accepted']

    lo = hi = None; complete = False; reason = None
    try:
        tf = key(t_s)
        if ev(tf):
            hi = tf
            for _ in range(MAX_DOWN):
                tf = key(tf * (1 - STEP))
                if ev(tf):
                    hi = tf
                else:
                    lo = tf; break
        else:
            lo = tf
            for _ in range(MAX_UP):
                tf = key(tf * (1 + STEP))
                if ev(tf):
                    hi = tf; break
                lo = tf
        if hi is not None and lo is not None:
            while (hi - lo) / hi > REL_RES:
                mid = key(0.5 * (lo + hi))
                if mid in (lo, hi):
                    break
                if ev(mid):
                    hi = mid
                else:
                    lo = mid
            complete = True
        reason = 'bracket_closed' if complete else ('no_infeasible_below_within_steps' if hi is not None else 'no_accepted_tf_within_steps')
    except BudgetExhausted:
        reason = 'budget_exhausted'
    acc = [k for k in order if cands[k]['accepted']]
    best = min(acc, key=lambda k: cands[k]['objective']) if acc else None
    out = {'variant': variant, 'status': 'accepted_plan' if best is not None else 'no_accepted_plan', 'reason': reason,
           'bisection_complete': complete, 'lowest_rejected_tf': lo, 'lowest_accepted_tf': hi,
           'resolution_s': (hi - lo) if (hi is not None and lo is not None) else None,
           'best_tf': best, 'best_objective': None if best is None else cands[best]['objective'],
           'best_folder': None if best is None else cands[best]['final_call_folder'],
           'n_candidates': len(order), 'n_calls': sum(len(cands[k]['calls']) for k in order)}
    record('finished')
    dump(root / 'fixed_stage_result.json', out)
    return out


# ------------------------------------------------------------------ jobs
def prepare_initial(scope, Tg, out):
    sc = scene(scope == 'single'); tf, S, U, meta = initial(sc, Tg); gate = initial_gate(sc, tf, S, U)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'initial.npz').exists():
        np.savez_compressed(out / 'initial.npz', S=S, U=U, tf=tf, global_vehicle_indices=np.array(sc['ids']))
        pairs, dist = pair_set(sc, tf, S, U)
        dump(out / 'initial_gate.json', {'scope': scope, 'm': sc['m'], 'Tg': Tg, 'metadata': meta, 'gate': gate,
                                         'initial_pairs_E5': [list(p) for p in pairs], 'n_initial_pairs_E5': len(pairs),
                                         'initial_centre_distances': {f'{i}-{j}': v for (i, j), v in dist.items()}})
    return sc, tf, S, U, gate


def free_record(folder):
    r = json.loads((Path(folder) / 'result.json').read_text('utf-8'))
    return {k: r.get(k) for k in ('return_status', 'solver_ok', 'accepted', 'tf', 'objective_raw_lambda100_gamma1', 'iter_count', 'solve_wall_s')} | \
        {'max_violation': r['final_audit']['max_violation'], 'folder': str(folder)}


def job_free_main(scope, Tg, out):
    """Free-t_f stage of the main and sensitivity versions: S0, then one free main call."""
    out = Path(out); sc, tf, S, U, gate = prepare_initial(scope, Tg, out)
    if not gate['pass']:
        dump(out / 'free_main.json', {'status': 'initial_construction_failed', 'gate': gate}); return
    t0 = time.perf_counter()
    st, calls, (tf1, S1, U1), sets, pairs = run_chain(sc, 'sensitivity', tf, S, U, out / 'sensitivity', 1, 1, CHAIN_BUDGET_S, t0)
    s0 = free_record(out / 'sensitivity' / 'call_1') if calls else None
    rec = {'scope': scope, 'Tg': Tg, 'm': sc['m'], 'S0_status': st, 'S0': s0, 'S0_wall_s': time.perf_counter() - t0}
    if not calls or not calls[-1]['solver_ok']:
        rec['main_free'] = None; rec['note'] = 'S0 solver failure: main version not started (E-2)'
        dump(out / 'free_main.json', rec); return
    st_m, calls_m, _, _, _ = run_chain(sc, 'main', tf1, S1, U1, out / 'main', 1, 1, CHAIN_BUDGET_S, t0,
                                       sets=copy.deepcopy(sets), pairs=pairs)
    rec['main_free_status'] = st_m
    rec['main_free'] = free_record(out / 'main' / 'call_1') if calls_m else None
    rec['wall_s'] = time.perf_counter() - t0
    dump(out / 'free_main.json', rec)


def job_original_free(scope, Tg, out):
    out = Path(out); sc, tf, S, U, gate = prepare_initial(scope, Tg, out)
    if not gate['pass']:
        dump(out / 'free_original.json', {'status': 'initial_construction_failed', 'gate': gate}); return
    t0 = time.perf_counter()
    st, calls, _, _, _ = run_chain(sc, 'original', tf, S, U, out / 'original', 1, 1, CHAIN_BUDGET_S, t0)
    dump(out / 'free_original.json', {'scope': scope, 'Tg': Tg, 'm': sc['m'], 'status': st,
                                      'original_free': free_record(out / 'original' / 'call_1') if calls else None,
                                      'wall_s': time.perf_counter() - t0})


SOURCE = {'main': ('main', 'call_1'), 'sensitivity': ('sensitivity', 'call_1'), 'original': ('original', 'call_1')}


def job_fixed(variant, scope, Tg, out, share=(), budget_s=CHAIN_BUDGET_S):
    """E-10 fixed-t_f stage for one version, starting from the free-stage solution of initial Tg.
    `share` lists other initial values whose free-stage solution is identical (rule 3)."""
    out = Path(out); sc = scene(scope == 'single')
    src = out / SOURCE[variant][0] / SOURCE[variant][1]
    z = np.load(src / 'solution.npz', allow_pickle=False)
    nx = json.loads((src / 'next.json').read_text('utf-8'))
    free = free_record(src)
    t0 = time.perf_counter()
    if free['accepted']:
        res = {'variant': variant, 'status': 'accepted_plan', 'reason': 'free stage accepted', 'best_tf': free['tf'],
               'best_objective': free['objective_raw_lambda100_gamma1'], 'best_folder': str(src)}
    elif not free['solver_ok']:
        res = {'variant': variant, 'status': 'solver_failed', 'reason': 'free stage solver failure', 'best_tf': None}
    else:
        res = fixed_stage(sc, variant, float(z['tf']), z['S'], z['U'], sets_load(nx['sets']), [tuple(p) for p in nx['pairs']],
                          out / variant / 'fixed', budget_s, t0)
    final = {'variant': variant, 'scope': scope, 'Tg': Tg, 'm': sc['m'], 'free_stage': free, 'fixed_stage': res,
             'status': res['status'], 'tf': res.get('best_tf'), 'objective_raw_lambda100_gamma1': res.get('best_objective'),
             'plan_folder': res.get('best_folder'), 'wall_s': time.perf_counter() - t0, 'shared_with': list(share)}
    dump(out / variant / 'result.json', final)
    for other in share:
        dump(out.parent / f'G{other}' / variant / 'result.json',
             {**final, 'Tg': other, 'reused_from': f'G{Tg}',
              'reuse_rule': 'E-10 item 3: identical free-stage solution, fixed stage computed once'})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('job', choices=['free_main', 'original_free', 'fixed'])
    ap.add_argument('--scope', choices=['single', 'full'], required=True)
    ap.add_argument('--Tg', type=int, choices=[16, 22, 28], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--variant', choices=list(VARIANTS))
    ap.add_argument('--share', default='')
    ap.add_argument('--budget', type=float, default=CHAIN_BUDGET_S)
    a = ap.parse_args()
    if ca is None:
        raise SystemExit('CasADi is required for solving')
    print('casadi', ca.__version__, ca.__file__, flush=True)
    if a.job == 'free_main':
        job_free_main(a.scope, a.Tg, a.out)
    elif a.job == 'original_free':
        job_original_free(a.scope, a.Tg, a.out)
    else:
        share = [int(x) for x in a.share.split(',') if x.strip()]
        job_fixed(a.variant, a.scope, a.Tg, a.out, share, a.budget)
