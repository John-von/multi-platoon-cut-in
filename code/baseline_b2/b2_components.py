"""B2 baseline: numerical components that need no NLP solver.

Centre-point mapping and execution-layer secant reference, trapezoid-consistent quadratic interpolation,
A-B2-2'' (directed forbidden zone, triangle-area form with strictness eps_a), A-B2-5 (double-circle geometry)
and A-B2-6' (registered initial guesses G16/G22/G28). Pure NumPy; the IPOPT problem is built in b2_nlp_v6.py."""
import numpy as np

# --- constants (execution-layer reference, A-B2-4', A-B2-5) --------------------
LW, LF_OVH, LR_OVH, LB = 2.8, 1.2, 0.8, 1.8      # wheelbase, front/rear overhang, width
L_R = 1.6                                         # rear axle -> body centre
HALF_L, HALF_W = 2.4, 0.9
CIRCLE_R = 0.5 * np.sqrt((0.5 * (LW + LF_OVH + LR_OVH)) ** 2 + LB ** 2)   # = 1.5
CIRCLE_OFFSETS = ((3 * LW + 3 * LF_OVH - LR_OVH) / 4, (LW + LF_OVH - 3 * LR_OVH) / 4)  # = 2.8, 0.4
DT = 0.1
DRAG, GF = 1 / 1500, 0.098
PRIOR = np.array([[0.9090909090909091, 1.1111111111111112],
                  [0.10989010989010989, 0.20408163265306123]])   # metric.json parameter_box
MU_HAT, LAM_HAT = PRIOR.mean(axis=1)
VMIN, VMAX, AMAX, TMAX = 5.0, 23.0, 2.0, 34.0
Y_S, Y_TOP, EPS_A = 0.01, 5.25, 1e-3

# --- mapping rear-axle state -> body-centre reference ------------------------
def centre_from_rear(x, y, v, th, ph, a, om):
    """Return p, y_ref, pdot, pddot (complete expressions)."""
    thd = v * np.tan(ph) / LW
    thdd = (a * np.tan(ph) + v / np.cos(ph) ** 2 * om) / LW
    p = x + L_R * np.cos(th)
    yr = y + L_R * np.sin(th)
    pd = v * np.cos(th) - L_R * np.sin(th) * thd
    pdd = a * np.cos(th) - v * np.sin(th) * thd - L_R * np.cos(th) * thd ** 2 - L_R * np.sin(th) * thdd
    return p, yr, pd, pdd

def rhs(s, u):
    """Kinematic bicycle (N01 eq. 1): s=(x,y,v,theta,phi), u=(a,omega)."""
    x, y, v, th, ph = s
    a, om = u
    return np.array([v * np.cos(th), v * np.sin(th), a, v * np.tan(ph) / LW, om])

# --- trapezoid-consistent quadratic interpolation ----------------------------
def interpolate(t_nodes, S, U, t_query):
    """S: (5,K+1) node states, U: (2,K+1) node controls. Controls linear, states
    s(t_k+tau)=s_k+f_k tau + tau^2/(2h)(f_{k+1}-f_k). Returns states, controls and
    the interpolant derivative (linear interpolation of node right-hand sides)."""
    t_nodes = np.asarray(t_nodes, float); t_query = np.atleast_1d(np.asarray(t_query, float))
    F = np.stack([rhs(S[:, k], U[:, k]) for k in range(S.shape[1])], axis=1)
    k = np.clip(np.searchsorted(t_nodes, t_query, side='right') - 1, 0, len(t_nodes) - 2)
    h = t_nodes[k + 1] - t_nodes[k]; tau = t_query - t_nodes[k]
    s = S[:, k] + F[:, k] * tau + tau ** 2 / (2 * h) * (F[:, k + 1] - F[:, k])
    sd = F[:, k] + tau / h * (F[:, k + 1] - F[:, k])
    u = U[:, k] + tau / h * (U[:, k + 1] - U[:, k])
    return s, u, sd

def collocation_residual(t_nodes, S, U):
    h = np.diff(t_nodes)
    F = np.stack([rhs(S[:, k], U[:, k]) for k in range(S.shape[1])], axis=1)
    return S[:, 1:] - S[:, :-1] - h * (F[:, 1:] + F[:, :-1]) / 2

# --- execution-layer reference (secant) -------------------------------------
def execution_reference(p_samples, v_terminal, mu_hat=MU_HAT, lam_hat=LAM_HAT, extra=3):
    """p_samples: plan centre position at t0+jh, j=0..M. Extrapolate position (not
    hold) with v_T for i=1..extra, then secant v, a and feed-forward z, q."""
    p = np.asarray(p_samples, float)
    p = np.r_[p, p[-1] + DT * v_terminal * np.arange(1, extra + 1)]
    v = np.diff(p) / DT
    a = np.diff(v) / DT
    z = DRAG * v[:-1] ** 2 + (a + GF) / mu_hat
    q = z[:-1] + np.diff(z) / lam_hat
    return {'p': p, 'v': v, 'a': a, 'z': z, 'q': q}

def nominal_step(p, v, z, q, mu=MU_HAT, lam=LAM_HAT):
    """robust_core.plant_step without disturbance (for regression item 1)."""
    return p + DT * v, v + DT * (mu * (z - DRAG * v * v) - GF), z + lam * (q - z)

# --- A-B2-2'' directed forbidden zone ----------------------------------------
def zone_D(p_c0, p_o0):
    return abs(p_c0 - p_o0) + VMAX * TMAX + 4 * L_R

def zone_rect(role, x_other, D):
    """front f: [x_f-10, x_f+D]; rear r: [x_r-D, x_r+10]; y in [y_s-1e-3, y_top]."""
    if role == 'front':
        return (x_other - 10.0, x_other + D, Y_S - 1e-3, Y_TOP)
    return (x_other - D, x_other + 10.0, Y_S - 1e-3, Y_TOP)

def triangle_area_sum(px, py, rect):
    """Sum of the four triangles (point, consecutive CCW vertices); equals the
    rectangle area iff the point is inside or on the boundary (N01 eq. 16/17)."""
    x0, x1, y0, y1 = rect
    V = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    tot = 0.0
    for (ax, ay), (bx, by) in zip(V, V[1:] + V[:1]):
        tot += 0.5 * abs((ax - px) * (by - py) - (bx - px) * (ay - py))
    return tot

def zone_margin(px, py, rect, eps_a=EPS_A):
    """>=0 means the strictified constraint 'area sum >= rect area + eps_a' holds."""
    x0, x1, y0, y1 = rect
    return triangle_area_sum(px, py, rect) - (x1 - x0) * (y1 - y0) - eps_a

# --- A-B2-5 double circles ----------------------------------------------------
def circle_centres(x, y, th):
    return [(x + o * np.cos(th), y + o * np.sin(th)) for o in CIRCLE_OFFSETS]

# --- A-B2-6' registered initial guesses ---------------------------------------
def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5

def target_configuration(rho0, queues, cut_of_queue, front_of_queue, spacing=12.0):
    """queues: list (front->back) of lists of ordinary indices (front->back).
    cut_of_queue[q], front_of_queue[q]: cut-in index and its designated front
    vehicle. Returns rho* per A-B2-6' step 1."""
    rho_star = {}
    prev_tail = None
    for q, members in enumerate(queues):
        order = []
        for i in members:
            order.append(i)
            if i == front_of_queue[q]:
                order.append(cut_of_queue[q])
        head = order[0]
        start = rho0[head] if prev_tail is None else min(rho0[head], prev_tail - spacing)
        for j, i in enumerate(order):
            rho_star[i] = start - spacing * j
        prev_tail = rho_star[order[-1]]
    return rho_star

def initial_guess(p0, rho_star, cut_pairs, Tg, t0=5.0, v_ref=20.0, delta_init=0.5, grid=None):
    """Return dict of centre trajectories p_k(t), y_k(t) on 'grid' (absolute times)
    and t_f guess. cut_pairs: {c: (f, r)}."""
    if grid is None:
        grid = np.arange(t0, t0 + 60.0 + 1e-9, 0.01)
    rho0 = {k: p0[k] for k in rho_star}
    s = smoothstep((grid - t0) / Tg)
    rho = {k: rho0[k] + s * (rho_star[k] - rho0[k]) for k in rho_star}
    P = {k: rho[k] + v_ref * (grid - t0) for k in rho}
    Y = {k: np.zeros_like(grid) for k in rho}
    t_open = {}
    for c, (f, r) in cut_pairs.items():
        ok = ((P[f] - P[c]) >= 10 + delta_init) & ((P[c] - P[r]) >= 10 + delta_init)
        idx = np.argmax(ok) if ok.any() else None
        if idx is None:
            raise ValueError('initial guess never opens a gap for cut-in %r' % c)
        t_open[c] = grid[idx]
        Y[c] = 3.5 * smoothstep((grid - t_open[c]) / 4.0)
    tf = max(Tg, max(t_open.values()) + 5.0 - t0)
    return {'t': grid, 'p': P, 'y': Y, 't_open': t_open, 'tf_guess': tf}
