"""EX9 (WP8-v20) audit = audit_run.audit (registered v1 definitions) + the lateral diagnostics of the v19 WP8
audit ('extra' block copied): sub-step lateral velocity and yaw rate (|vy| <= 5 m/s,
|r| <= 1 rad/s), planning corridor [0, 3.5] m (prediction uses 3.499 m), vehicle body inside the two-lane road
[-1.75, 5.25] m, slip-angle magnitudes, terminal lateral error.  Corridor excess is reported separately from body
road-edge and body-overlap safety.  Usage: python audit_lateral.py <run_dir>"""
import json, sys
from pathlib import Path
import numpy as np
import audit_run


def lateral_extra(a):
    Q = a['mode'].shape[1]; N = a['state'].shape[1] - Q
    main = a['lateral']; sub = a.get('lateral_substate', main)
    pose = a['body_poses']; yaw = pose[:, :, 2]
    extent = 2.4 * abs(np.sin(yaw)) + .9 * abs(np.cos(yaw))
    lower = pose[:, :, 1] - extent + 1.75; upper = 5.25 - pose[:, :, 1] - extent
    by = pose[:, N:, 1]
    lower_corridor = np.maximum(0., -by); upper_corridor = np.maximum(0., by - 3.5)
    extra = dict(main_vy_max=float(abs(main[:, :, 2]).max()), substep_vy_max=float(abs(sub[:, :, 2]).max()),
                 main_yawrate_max=float(abs(main[:, :, 3]).max()), substep_yawrate_max=float(abs(sub[:, :, 3]).max()),
                 substep_y_min=float(sub[:, :, 0].min()), substep_y_max=float(sub[:, :, 0].max()),
                 body_road_lower_margin=float(lower.min()), body_road_upper_margin=float(upper.min()),
                 road_boundary_pass=bool(min(lower.min(), upper.min()) >= -1e-8),
                 corridor_excess_above_m=float(upper_corridor.max()), corridor_excess_below_m=float(lower_corridor.max()),
                 corridor_above_duration_s_per_vehicle=(np.count_nonzero(upper_corridor[1:] > 1e-6, axis=0) * .02).tolist(),
                 corridor_below_duration_s_per_vehicle=(np.count_nonzero(lower_corridor[1:] > 1e-6, axis=0) * .02).tolist(),
                 substep_phi_max=float(abs(sub[:, :, 1]).max()),
                 terminal_lateral_error_RMSE=float(np.sqrt(np.mean((main[-1, :, 0] - 3.5) ** 2))))
    if 'slip_angles' in a:
        extra['max_abs_slip_angle_rad'] = np.abs(a['slip_angles']).max(axis=(0, 1)).tolist()
    extra['state_pass'] = bool(max(extra['main_vy_max'], extra['substep_vy_max']) <= 5 + 1e-6 and
                               max(extra['main_yawrate_max'], extra['substep_yawrate_max']) <= 1 + 1e-6)
    extra['corridor_pass'] = bool(extra['corridor_excess_above_m'] <= 1e-6 and extra['corridor_excess_below_m'] <= 1e-6)
    return extra


def audit(run_dir):
    out = audit_run.audit(run_dir)
    run = Path(run_dir)
    a = dict(np.load(run / 'arrays.npz')) if (run / 'arrays.npz').exists() else {}
    if 'mode' in a and 'lateral' in a and len(a.get('input', [])) >= 2:
        out['lateral_extra'] = ex = lateral_extra(a)
        labs = set(out.get('labels', []))
        # With the softened corridor (M1) a corridor excess is an expected, separately reported outcome: E is
        # re-derived here from the safety-relevant empirical items only (steering, body overlap, body inside the
        # road, sub-step lateral velocity / yaw rate); the corridor excess gets its own label Lc.
        ev = out.get('empirical', {}).get('violations', {})
        e_safety = bool(ev.get('steering') or ev.get('body_overlap') or not ex['state_pass'] or not ex['road_boundary_pass'])
        labs.discard('E')
        if e_safety:
            labs.add('E')
        if not ex['corridor_pass']:
            labs.add('Lc')
        out['labels'] = sorted(labs)
        out['empirical_safety_violation'] = e_safety
    res = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    out['lateral_force_sha256'] = res.get('_meta', {}).get('lateral_force_sha256')
    return out


if __name__ == '__main__':
    r = audit(sys.argv[1])
    (Path(sys.argv[1]) / 'audit.json').write_text(json.dumps(r, indent=1, default=str), encoding='utf-8', newline='\n')
    print(json.dumps({k: r.get(k) for k in ('id', 'status', 'labels')}))
