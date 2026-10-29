"""Per-run audit for WP5-v20 (registered definitions; design section 4).

Usage: python audit_run.py <run_dir> [<frozen_analysis_dir>]   (run_dir holds case.json, result.json, arrays.npz)
Writes <run_dir>/audit.json.  Reads only the saved evidence; the body (SAT) check calls the frozen
frozen_analysis/trajectory_audit.py (SHA-256 4cc095ff...) unchanged.

Theory-covered physical constraints (sampling instants, tolerance 1e-6): ordinary gaps >= 10 m; each gated pair
(front-cut, cut-rear) >= 10 m from the gate step on; adjacent cut-ins in the source lane >= 10 m while neither has
started lateral motion; speed in [5, 23] m/s; |a| <= 2 m/s^2; 1 s window jerk |j^W| <= 1 m/s^3; input in [UMIN, UMAX].
Certificate consistency (robust methods): tube ratio <= 1 (+1e-6); true parameters inside the identification set
(+1e-9); for the handover runner, true error inside the handover initial set at every activation (runner flag, 1e-8).
Empirical (not covered by the theory): lateral y in [0, 3.5] m, |steering| <= 0.25 rad, |steering step| <= 0.01 rad,
no body overlap (SAT).  Sampled jerk is reported, not checked."""
import json, sys
from pathlib import Path
import numpy as np

DT = 0.1; TS = .96 / (1500 * .35); UMIN, UMAX = -2000 * TS, 3000 * TS; TOL = 1e-6
import os, hashlib
# v1.1: the frozen SAT audit is read from ./frozen_analysis unless V20_FROZEN_ANALYSIS names another folder; it is
# hash-checked before use and its path and hash are recorded.
FROZEN_ANALYSIS = Path(os.environ.get('V20_FROZEN_ANALYSIS', Path(__file__).resolve().parent / 'frozen_analysis'))
TRAJECTORY_AUDIT_SHA256 = '4cc095ff04d2ce82aee136b49395945cba4f9733d9f61186528f88eb99bd2a81'


def frozen_audit_record(frozen_analysis):
    p = Path(frozen_analysis) / 'trajectory_audit.py'
    if not p.exists():
        raise SystemExit(f'frozen trajectory_audit.py not found under {frozen_analysis}; set V20_FROZEN_ANALYSIS')
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != TRAJECTORY_AUDIT_SHA256:
        raise SystemExit(f'trajectory_audit.py hash mismatch: {h}')
    return {'path': str(p.resolve()), 'sha256': h}
ROBUST = {'A-H', 'A-S', 'F', 'TUBE', 'DDRPC'}


def first_time(mask):
    idx = np.flatnonzero(mask)
    return None if not len(idx) else round(float(idx[0] * DT), 3)


def physical(X, U, ALOG, mode=None, front=None, rear=None, Q=3):
    """Theory-covered physical metrics; returns (metrics, margins, violations{name: first_time or None})."""
    M = X.shape[1]; N = M - Q
    m, mar, vio = {}, {}, {}
    gaps = X[:, :N - 1, 0] - X[:, 1:N, 0]
    m['ordinary_gap_min'] = float(gaps.min()); mar['ordinary_gap'] = m['ordinary_gap_min'] - 10
    vio['ordinary_gap'] = first_time((gaps < 10 - TOL).any(axis=1))
    v = X[:, :, 1]
    m['speed_range'] = [float(v.min()), float(v.max())]
    mar['speed'] = min(m['speed_range'][0] - 5, 23 - m['speed_range'][1])
    vio['speed'] = first_time(((v < 5 - TOL) | (v > 23 + TOL)).any(axis=1))
    m['acc_abs_max'] = float(np.abs(ALOG).max()) if ALOG.size else 0.
    mar['acceleration'] = 2 - m['acc_abs_max']
    vio['acceleration'] = first_time((np.abs(ALOG) > 2 + TOL).any(axis=1)) if ALOG.size else None
    jw = (ALOG[10:] - ALOG[:-10]) / (10 * DT) if len(ALOG) > 10 else np.zeros((0, M))
    js = np.diff(ALOG, axis=0) / DT if len(ALOG) > 1 else np.zeros((0, M))
    m['window_jerk_max'] = float(np.abs(jw).max()) if jw.size else 0.
    m['sampled_jerk_max'] = float(np.abs(js).max()) if js.size else 0.
    m['sampled_jerk_over_1_fraction'] = float(np.mean(np.abs(js) > 1 + TOL)) if js.size else 0.
    mar['window_jerk'] = 1 - m['window_jerk_max']
    vio['window_jerk'] = (None if not jw.size else
                          (lambda t: None if t is None else round(t + 10 * DT, 3))(first_time((np.abs(jw) > 1 + TOL).any(axis=1))))
    m['input_range'] = [float(U.min()), float(U.max())] if U.size else [0., 0.]
    mar['input'] = min(m['input_range'][0] - UMIN, UMAX - m['input_range'][1])
    vio['input'] = first_time(((U < UMIN - TOL) | (U > UMAX + TOL)).any(axis=1)) if U.size else None
    if mode is not None:
        starts = []
        for c in range(Q):
            s = np.flatnonzero(mode[:, c] >= 1); starts.append(int(s[0]) if len(s) else None)
        mg_min, mg_first = None, None
        for c in range(Q):
            if starts[c] is None:
                continue
            at = starts[c] - 1   # mode is stored at k+1 after the step in which it switched
            for g in (X[at:, front[c], 0] - X[at:, N + c, 0], X[at:, N + c, 0] - X[at:, rear[c], 0]):
                mg_min = float(g.min()) if mg_min is None else min(mg_min, float(g.min()))
                t = first_time(g < 10 - TOL)
                if t is not None:
                    t = round(t + at * DT, 3); mg_first = t if mg_first is None else min(mg_first, t)
        m['merge_gap_min_from_gate'] = mg_min; mar['merge_gap'] = None if mg_min is None else mg_min - 10
        vio['merge_gap'] = mg_first
        sg_min, sg_first = None, None
        for c in range(Q - 1):
            end = min([s for s in (starts[c], starts[c + 1]) if s is not None], default=X.shape[0])
            g = X[:end, N + c, 0] - X[:end, N + c + 1, 0]
            if g.size:
                sg_min = float(g.min()) if sg_min is None else min(sg_min, float(g.min()))
                t = first_time(g < 10 - TOL)
                if t is not None:
                    sg_first = t if sg_first is None else min(sg_first, t)
        m['source_gap_min_before_lateral'] = sg_min; mar['source_gap'] = None if sg_min is None else sg_min - 10
        vio['source_gap'] = sg_first
        m['gate_steps'] = starts
    return m, mar, vio


def audit(run_dir, frozen_analysis=FROZEN_ANALYSIS):
    run = Path(run_dir)
    case = json.loads((run / 'case.json').read_text(encoding='utf-8'))
    res = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    a = dict(np.load(run / 'arrays.npz')) if (run / 'arrays.npz').exists() else {}
    meta = res.get('_meta', {}); exc = meta.get('exception')
    out = {'id': case['id'], 'run_key': case.get('run_key'), 'family': case['family'], 'method': case['method'],
           'cell': case['cell'], 'theory_covered': case.get('theory_covered'), 'status': res.get('status'),
           'exception': None if exc is None else {'type': exc['type'], 'message': exc['message']},
           'audit_version': '1.1', 'frozen_audit': frozen_audit_record(frozen_analysis)}
    labels = []
    empty_set = exc is not None and exc['type'] == 'ValueError' and 'Empty parameter set' in exc['message']
    if exc is not None:
        labels.append('I' if empty_set else 'X')
        out['recorder_event'] = meta.get('recorder_event')
        if 'state' in a and a.get('complete_steps', 0):
            K = int(a['complete_steps'])
            m, mar, vio = physical(a['state'][:K + 1], a['input'][:K], a['acceleration'][:K])
            out.update(prefix_steps=K, prefix_metrics=m, prefix_margins=mar, prefix_first_violation=vio)
            if any(t is not None for t in vio.values()):
                labels.append('P')
        out['labels'] = labels
        return out
    if res.get('status') == 'initial_not_certified':
        out['labels'] = ['C0']
        out['initial_events'] = res.get('events')
        return out
    X, U, ALOG, mode = a['state'], a['input'], a['acceleration'], a['mode']
    Q = mode.shape[1]; front, rear = a['front'].astype(int), a['rear'].astype(int)
    m, mar, vio = physical(X, U, ALOG, mode, front, rear, Q)
    out.update(metrics=m, margins=mar, first_violation=vio)
    done = np.array([np.nan if x is None else x for x in res.get('merge_done', [])], float)
    start = np.array([np.nan if x is None else x for x in res.get('merge_start', [])], float)
    out['completed'] = int(np.isfinite(done).sum()); out['admitted'] = int(np.isfinite(start).sum())
    out['last_done'] = res.get('last_done'); out['Jv'] = res.get('Jv')
    out['merge_start'] = res.get('merge_start'); out['merge_done'] = res.get('merge_done')
    if out['completed'] == Q:
        labels.append('C3')
    elif out['admitted'] == 0:
        labels.append('C1')
    else:
        labels.append('C2')
    if any(t is not None for t in vio.values()):
        labels.append('P')
    # certificate consistency
    bh, truth = a['bounds'], a['truth']
    inside = np.all((bh[..., 0] <= truth[None] + 1e-9) & (truth[None] <= bh[..., 1] + 1e-9), axis=(1, 2))
    out['parameters_contained'] = bool(inside.all()); out['parameter_escape_time'] = first_time(~inside)
    ratio = a['tube_ratio']; out['tube_ratio_max'] = float(ratio.max())
    out['tube_ratio_first_over_1'] = first_time(ratio > 1 + TOL)
    if case['method'] in ROBUST and (not out['parameters_contained'] or out['tube_ratio_max'] > 1 + TOL):
        labels.append('M')
    checks = res.get('activation_checks') or []
    out['activations'] = len(checks)
    out['activation_containment_all'] = all(c.get('true_containment', True) for c in checks)
    out['activation_observed_monitor_all'] = all(c.get('observed_monitor', True) for c in checks)
    out['activation_radius_margin_min'] = min((c['radius_margin_min'] for c in checks), default=None)
    out['activation_box_margin_min'] = min((c['box_margin_min'] for c in checks), default=None)
    fails = [c['step'] for c in checks if not c.get('true_containment', True)]
    out['activation_containment_first_fail'] = None if not fails else round(fails[0] * DT, 3)
    if case['runner'] == 'handover' and fails:
        labels.append('K')
    jobs = res.get('planning_jobs') or []
    out['planning_jobs'] = [{k: j.get(k) for k in ('id', 'kind', 'dispatch_s', 'deadline_s', 'accepted', 'rejection_reason',
                                                     'deadline_missed', 'injected_drop')} |
                            {'T_found': ([e.get('T') for e in (j.get('search') or []) if e.get('reason') == 'accepted'] or [None])[0],
                             'n_candidates': len(j.get('search') or []),
                             'search': [[e.get('T'), e.get('reason')] for e in (j.get('search') or [])]} for j in jobs]
    out['deadline_misses'] = int(sum(bool(j.get('deadline_missed')) for j in jobs))
    if out['deadline_misses']:
        labels.append('D')
    if case['runner'] == 'sync':
        out['plans'] = [(p['time'], p['T']) for p in res.get('plans', [])]
    # empirical
    emp = {}
    lat = a.get('lateral'); st = a.get('steering')
    if lat is not None and lat.size:
        emp['lateral_y_range'] = [float(lat[:, :, 0].min()), float(lat[:, :, 0].max())]
    if st is not None and st.size:
        emp['steering_abs_max'] = float(np.abs(st).max())
        emp['steering_step_max'] = float(np.abs(np.diff(np.vstack([np.zeros((1, st.shape[1])), st]), axis=0)).max())
    if 'lateral_levels' in a:
        lv = a['lateral_levels']; emp['lateral_levels_count'] = {str(j): int(np.sum(lv == j)) for j in (1, 2, 3)}
    emp['lateral_fallback_samples'] = res.get('lateral_fallback_samples')
    emp['lateral_solver_failures'] = res.get('lateral_solver_failures')
    sys.path.insert(0, str(frozen_analysis))
    from trajectory_audit import sat_audit
    phys, _ = sat_audit(res, a)
    emp['body_overlap_samples'] = phys['body_overlap_samples']; emp['body_separating_margin'] = phys['body_separating_margin']
    e_vio = {'lateral_y': 'lateral_y_range' in emp and not (emp['lateral_y_range'][0] >= -TOL and emp['lateral_y_range'][1] <= 3.5 + TOL),
             'steering': 'steering_abs_max' in emp and not (emp['steering_abs_max'] <= .25 + TOL and emp['steering_step_max'] <= .01 + TOL),
             'body_overlap': emp['body_overlap_samples'] != 0}
    emp['violations'] = e_vio
    out['empirical'] = emp
    if any(e_vio.values()):
        labels.append('E')
    lv = emp.get('lateral_levels_count') or {}
    if lv.get('2', 0) or lv.get('3', 0) or (emp.get('lateral_fallback_samples') or 0):
        labels.append('L')
    out['labels'] = labels
    return out


if __name__ == '__main__':
    r = audit(sys.argv[1], Path(sys.argv[2]) if len(sys.argv) > 2 else FROZEN_ANALYSIS)
    (Path(sys.argv[1]) / 'audit.json').write_text(json.dumps(r, indent=1, default=str), encoding='utf-8', newline='\n')
    print(json.dumps({k: r.get(k) for k in ('id', 'status', 'labels')}))
