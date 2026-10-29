"""Compact per-run data for the figures and tables, extracted from the recomputed archive.
Reads data/recomputed/runs/<key20>/{audit,result}.json (and arrays.npz for the few trajectory panels) for every
registered identity and writes all_runs.json, trajectories.json and extract_meta.json to data/extracted
(or --out DIR). Read-only on the archive. EX9 panels also carry the cut-in body lateral position at the 0.02 s
sub-steps (body_poses) and the steering angles of the corner and pulse variants; EX4 panels also carry the speed of
every vehicle at the instant of the largest speed."""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                 # package root
WP = ROOT / 'code' / 'experiments'
RUNS = ROOT / 'data' / 'recomputed' / 'runs'
DATA = ROOT / 'data' / 'extracted'
if '--out' in sys.argv:
    DATA = Path(sys.argv[sys.argv.index('--out') + 1])
DATA.mkdir(parents=True, exist_ok=True)
EX9_DETAIL = {'L_C1', 'L_C1_pulse_neg', 'L_C1_pulse_pos', 'L_C2', 'L_C2_pulse_neg', 'L_C2_pulse_pos',
              'L_C3', 'L_C3_pulse_neg', 'L_C3_pulse_pos', 'L_C4', 'L_C4_pulse_neg', 'L_C4_pulse_pos', 'L_P0'}


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def registry():
    by_key = {}
    for f in ('cases_v1.json', 'cases_addA_v1.json'):
        for c in load(WP / f)['cases']:
            by_key.setdefault(c['run_key'], []).append(c)
    return by_key


def first_merge_T(a, r):
    for p in r.get('accepted_plans') or []:
        if p.get('kind') == 'merge':
            return p.get('T')
    return None


def summary(c, a, r):
    m = a.get('metrics') or {}
    return {'id': c['id'], 'family': c['family'], 'method': c['method'], 'cell': c['cell'], 'labels': a.get('labels'),
            'status': a.get('status'), 'last_done': a.get('last_done'), 'Jv': a.get('Jv'), 'admitted': a.get('admitted'),
            'completed': a.get('completed'), 'first_merge_T': first_merge_T(a, r), 'events': r.get('events'),
            'margins': a.get('margins'), 'speed_max': (m.get('speed_range') or [None, None])[1],
            'first_violation': a.get('first_violation'), 'tube_ratio_max': a.get('tube_ratio_max'),
            'planning_jobs': [{k: j.get(k) for k in ('kind', 'dispatch_s', 'accepted', 'rejection_reason', 'T_found', 'injected_drop', 'deadline_missed')}
                              for j in (a.get('planning_jobs') or [])]}


def main():
    t0 = time.time(); reg = registry(); rows = []; missing = []
    for key, cases in sorted(reg.items()):
        d = RUNS / key[:20]
        if not (d / 'audit.json').exists():
            missing.append(key); continue
        a = load(d / 'audit.json'); r = load(d / 'result.json')
        for c in cases:                                   # every registered alias gets a row
            rows.append(summary(c, a, r) | {'run_key': key})
    (DATA / 'all_runs.json').write_text(json.dumps({'rows': rows, 'missing_keys': missing}, indent=0), encoding='utf-8')
    # trajectory panels
    traj = {}
    for c_key, cases in reg.items():
        for c in cases:
            cell = c['cell']
            want = None
            if c['family'] == 'EX4' and cell['gap'] == 10.25 and cell['corner'] == 'high' and cell['boundary'] == 30 and c['method'] in ('N', 'A-H'):
                want = 'ex4'
            if c['family'] == 'EX9' and cell.get('seed') is None:
                want = 'ex9'
            if want is None:
                continue
            d = RUNS / c_key[:20]
            if not (d / 'arrays.npz').exists() or not (d / 'audit.json').exists():
                continue
            with np.load(d / 'arrays.npz', allow_pickle=False) as z:
                t = z['time']
                if want == 'ex4':
                    v = z['state'][:, :, 1]; k = int(np.argmax(v.max(axis=1)))
                    traj[c['id']] = {'time': t.tolist(), 'max_speed': v.max(axis=1).tolist(), 'f': cell['f'],
                                     'argmax_vehicle': int(np.argmax(v[k])), 'speeds_at_peak': v[k].tolist(), 'peak_t': float(t[k])}
                else:
                    e = {'time': t.tolist(), 'y': z['lateral'][:, :, 0].T.tolist(), 'variant': cell['variant'],
                         'steering_abs_max': float(np.abs(z['steering']).max()) if 'steering' in z.files else None}
                    if cell['variant'] in EX9_DETAIL:
                        Q = z['mode'].shape[1]; N = z['state'].shape[1] - Q
                        e['body_y_dt'] = 0.02
                        e['body_y'] = np.round(z['body_poses'][:, N:, 1].T, 6).tolist()
                        e['steering'] = np.round(z['steering'].T, 6).tolist() if 'steering' in z.files else None
                        e['labels'] = load(d / 'audit.json').get('labels')
                    traj[c['id']] = e
    (DATA / 'trajectories.json').write_text(json.dumps(traj), encoding='utf-8')
    meta = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'n_rows': len(rows), 'n_missing_keys': len(missing),
            'n_traj': len(traj), 'seconds': round(time.time() - t0, 1), 'out': str(DATA),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (DATA / 'extract_meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(meta)


if __name__ == '__main__':
    main()
