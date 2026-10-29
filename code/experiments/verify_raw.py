"""Independent re-check of the registered audits from the raw run arrays (verification aid, not a registered analysis).

Usage: python verify_raw.py <runs_root> <out.json>

For every run directory with audit.json, recompute from arrays.npz / result.json with code written separately from
audit_run.py (different derivations where possible):
  * acceleration from the logged speeds, a_k = (v_{k+1} - v_k)/h, compared with the logged acceleration array;
  * 1 s window jerk from that acceleration; speed and input ranges;
  * ordinary spacing of index-consecutive ordinary vehicles; cut-in/neighbour spacing from the lateral-start time in
    result.json (merge_start), not from the mode array; source-lane spacing of consecutive cut-ins before either starts;
  * error inside the logged coordinate box (|e| <= tube_box, a check the ratio test does not perform);
  * parameters inside the identification boxes;
  * J_v recomputed from the speeds and the common speed, t_done from merge_done;
and compare the violation flags (P, M) and the reported t_done/J_v with audit.json.  Any disagreement is listed."""
import json, sys
from pathlib import Path
import numpy as np

H = 0.1; W = 10; TOL = 1e-6
QS = 0.96 / (1500 * 0.35); QMIN, QMAX = -2000 * QS, 3000 * QS
ROBUST = {'A-H', 'A-S', 'F', 'TUBE', 'DDRPC'}


def check(run):
    case = json.loads((run / 'case.json').read_text(encoding='utf-8'))
    res = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    aud = json.loads((run / 'audit.json').read_text(encoding='utf-8'))
    out = {'id': case['id'], 'labels': aud.get('labels'), 'issues': []}
    if res.get('_meta', {}).get('exception') is not None or res.get('status') == 'initial_not_certified':
        out['skipped'] = 'exception_or_C0'
        return out
    Z = np.load(run / 'arrays.npz')
    X = Z['state']; U = Z['input']; A_log = Z['acceleration']
    Q = int(len(Z['front'])); M = X.shape[1]; N = M - Q
    v = X[:, :, 1]
    a = np.diff(v, axis=0) / H                                   # (K, M)
    n = min(len(a), len(A_log))
    out['acc_log_vs_speed_diff_max'] = float(np.abs(a[:n] - A_log[:n]).max()) if n else 0.0
    if out['acc_log_vs_speed_diff_max'] > 1e-9:
        out['issues'].append('logged acceleration differs from speed differences')
    jw = (a[W:] - a[:-W]) / (W * H)
    viol = {
        'speed': bool(((v < 5 - TOL) | (v > 23 + TOL)).any()),
        'acceleration': bool((np.abs(a) > 2 + TOL).any()),
        'window_jerk': bool((np.abs(jw) > 1 + TOL).any()) if jw.size else False,
        'input': bool(((U < QMIN - TOL) | (U > QMAX + TOL)).any()),
        'ordinary_gap': bool(((X[:, :N - 1, 0] - X[:, 1:N, 0]) < 10 - TOL).any()),
    }
    front = Z['front'].astype(int); rear = Z['rear'].astype(int)
    ms = [None if t is None else int(round(t / H)) for t in (res.get('merge_start') or [None] * Q)]
    mg = []
    for c in range(Q):
        if ms[c] is None:
            continue
        k0 = ms[c]
        mg.append(float(min((X[k0:, front[c], 0] - X[k0:, N + c, 0]).min(), (X[k0:, N + c, 0] - X[k0:, rear[c], 0]).min())))
    viol['merge_gap'] = bool(mg and min(mg) < 10 - TOL)
    sg = []
    for c in range(Q - 1):
        ends = [s for s in (ms[c], ms[c + 1]) if s is not None]
        e = min(ends) if ends else X.shape[0]
        if e > 0:
            sg.append(float((X[:e, N + c, 0] - X[:e, N + c + 1, 0]).min()))
    viol['source_gap'] = bool(sg and min(sg) < 10 - TOL)
    out['recomputed_violations'] = {k: x for k, x in viol.items() if x}
    out['merge_gap_min'] = min(mg) if mg else None
    P_mine = any(viol.values())
    P_audit = 'P' in (aud.get('labels') or [])
    if P_mine != P_audit:
        out['issues'].append(f'P flag differs: recomputed {P_mine}, audit {P_audit}')
    # the audit takes the gate from the mode array; compare the two merge-gap minima
    amg = (aud.get('metrics') or {}).get('merge_gap_min_from_gate')
    if amg is not None and mg and abs(amg - min(mg)) > 1e-9:
        out['merge_gap_min_audit'] = amg
        if (amg < 10 - TOL) != (min(mg) < 10 - TOL):
            out['issues'].append('merge-gap verdict depends on the gate definition')
    # certificate consistency
    e = Z['error']; tb = Z['tube_box']
    box_out = np.abs(e) > tb + 1e-9
    out['box_containment_all'] = not bool(box_out.any())
    B = Z['bounds']; th = Z['truth']
    par_in = bool(np.all((B[..., 0] <= th[None] + 1e-9) & (th[None] <= B[..., 1] + 1e-9)))
    M_mine = case['method'] in ROBUST and (not par_in or float(Z['tube_ratio'].max()) > 1 + TOL)
    if M_mine != ('M' in (aud.get('labels') or [])):
        out['issues'].append('M flag differs')
    if case['method'] in ROBUST and not out['box_containment_all']:
        # consistent with the audit only if the audit also flagged a certificate mismatch (M)
        if 'M' in (aud.get('labels') or []):
            out['box_failure_with_M_label'] = True
        else:
            out['issues'].append('error outside the logged coordinate box without label M')
    # performance
    cs = Z['common_speed']; k_req = int(round(5.0 / H))
    ordv = v[:len(cs), :N]
    Jv = H / N * float(np.abs(ordv[k_req:] - cs[k_req:, None]).sum())
    out['Jv_recomputed'] = Jv; out['Jv_reported'] = res.get('Jv')
    if res.get('Jv') is not None and abs(Jv - res['Jv']) > 1e-6 * max(1, abs(Jv)):
        out['issues'].append(f"J_v differs: {Jv} vs {res['Jv']}")
    md = res.get('merge_done') or []
    td = max(md) if md and all(x is not None for x in md) else None
    if (td is None) != (res.get('last_done') is None) or (td is not None and abs(td - res['last_done']) > 1e-9):
        out['issues'].append('t_done differs from max(merge_done)')
    return out


def main():
    root = Path(sys.argv[1]); outp = Path(sys.argv[2])
    rows = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / 'audit.json').exists() and (d / 'arrays.npz').exists():
            try:
                rows.append(check(d))
            except Exception as ex:  # a failed check is itself a finding
                rows.append({'dir': d.name, 'issues': [f'verify error: {type(ex).__name__}: {ex}']})
    bad = [r for r in rows if r.get('issues')]
    summ = {'runs_checked': len(rows), 'runs_with_issues': len(bad),
            'skipped': sum(1 for r in rows if r.get('skipped')),
            'P_recomputed': sum(1 for r in rows if r.get('recomputed_violations')),
            'box_containment_failures': sum(1 for r in rows if r.get('box_containment_all') is False),
            'box_failures_robust_with_M_label': sum(1 for r in rows if r.get('box_failure_with_M_label')),
            'acc_log_diff_max': max((r.get('acc_log_vs_speed_diff_max', 0) for r in rows), default=0)}
    outp.write_text(json.dumps({'summary': summ, 'issues': bad, 'rows': rows}, indent=1), encoding='utf-8', newline='\n')
    print(json.dumps(summ))
    for r in bad[:20]:
        print(r.get('id', r.get('dir')), r['issues'])


if __name__ == '__main__':
    main()
