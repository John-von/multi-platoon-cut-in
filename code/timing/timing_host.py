"""WP7b-v20 host timing: real-time runs of the handover runner on the timing host (Windows PC).

Usage (Windows or Linux, from this folder):
  python timing_host.py --src ../sim/sources --out <dir> --workers N [--repeats 10] [--configs T1 T2 T3 T4]
  python timing_host.py --src ... --out <dir> --workers 1 --repeats 3            (serial reference)
For every config: R host-mode runs (real time, 50 s each) with V20_PLAN_WORKERS=N, then one replay run of the same
config with every job ready at dispatch; non-timing arrays of each host run are compared with the replay run.
Configs use disturbance_fraction 0.75, which no registered WP5-v20 case uses (no registered outcome is produced here).
Writes <out>/<config>_<workers>w_r<k>/{cfg.json,result.json,arrays.npz}, <out>/<config>_replay/..., <out>/timing_summary_<N>w.json."""
import argparse, copy, hashlib, json, os, platform, sys, time
from pathlib import Path
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = Path(__file__).resolve().parent
TIMING_ARR = {'timing', 'planning_time', 'wall_time', 'loop_cost', 'tick_lateness', 'control_deadline_miss', 'lateral_fallback'}
EPS = [1e-3, 5e-5, .00015 / 1.89010989]
V20 = dict(enabled=True, lateral_soft=True, condition_s=True, jerk_mode='window', jerk_window=10, replan_mode='handover',
           activation_latency=[30, 20], containment_gate=False, init_refine=True)
E1A = {'async_execution.py': 'b6d15027e6d13cadc269c4f6226286eb0ea51a1c1ac93182970bbd303953faba',
       'lateral_fallback.py': '275119c8211847ae3928485b651cab3eb6c9ae538180fbf6fa34a3e8d0dec437',
       'lateral_fast.py': 'b3765e62fc3c8af447bc08aae79038935734f3e45324084fab7cb7f47abb47a1',
       'robust_core.py': '5279520a2b4228ffae69db664abfa9696f54268cf3dd674841cb0eb099fda67a',
       'simulation.py': '97e0b11140e42bc65c787806046fcaab4adce565392beecc579029d4b688f69e',
       'verify_core.py': 'dfa89634eedac9687962925eb4b543da5a7533f28afe8029920607e7ebb262e4'}


def cfg_for(name):
    s, ell, mass, tau, extra = {'T1': (1, 0, 1.0, 1.0, {}), 'T2': (5, 5, 1.1, 1.3, {}), 'T3': (10, 0, 1.0, 1.0, {}),
                                'T4': (1, 5, 1.0, 1.0, {'uplink_age_steps': 10})}[name]
    cfg = {'execution': 'host', 'time_sim': 50, 'request_time': 5.0, 'max_attempts': 3, 'acc_bound': .001 * s,
           'torque_bound': .00015 * s, 'disturbance_fraction': 0.75, 'initial_gap': 10.5, 'boundary': 30,
           'mass_scale': mass, 'tau_scale': tau, 'durations': [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 34],
           'v20': dict(V20, meas_eps=[ell * e for e in EPS])}
    cfg.update(extra)
    return cfg


def clean(v):
    import numpy as np
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, np.generic): return clean(v.item())
    if isinstance(v, float) and not np.isfinite(v): return None
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    return v


def single(a):
    """One run in this (fresh) process: --single <config> --mode host|replay --dir <run_dir>."""
    src = Path(a.src).resolve(); d = Path(a.dir); d.mkdir(parents=True, exist_ok=True)
    os.environ['V20_PLAN_WORKERS'] = str(a.workers)
    standin = os.environ.get('V20_OSQP_STANDIN')   # optional import-only OSQP stand-in (../sim/osqp_only)
    sys.path[:0] = [str(HERE)] + ([standin] if standin else []) + [str(src)]
    import numpy as np, simulation as S, async_execution as A
    cfg = cfg_for(a.single)
    if a.mode == 'replay':
        cfg['execution'] = 'replay'; cfg['delay_schedule_s'] = [0.0] * 10
    else:
        A.ping = S.warm_plan_pool   # harness hook of the E1a patch: runs in the planning worker before the clock origin
    t = time.time(); res, arr = A.run_async(cfg); res['harness_wall_s'] = time.time() - t
    (d / 'cfg.json').write_text(json.dumps(cfg, indent=1))
    np.savez_compressed(d / 'arrays.npz', **{k: np.asarray(v) for k, v in arr.items()})
    (d / 'result.json').write_text(json.dumps(clean(res), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True); ap.add_argument('--out', default=None)
    ap.add_argument('--workers', type=int, required=True); ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--configs', nargs='*', default=['T1', 'T2', 'T3', 'T4'])
    ap.add_argument('--single'); ap.add_argument('--mode', choices=['host', 'replay']); ap.add_argument('--dir')
    a = ap.parse_args()
    src = Path(a.src).resolve()
    bad = {k: v for k, v in E1A.items() if hashlib.sha256((src / k).read_bytes()).hexdigest() != v}
    if bad:
        raise SystemExit(f'not the E1a sources: {bad}')
    if a.single:
        return single(a)
    import subprocess, numpy as np
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    def sub(name, mode, d):
        if (d / 'result.json').exists():
            return 0
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--src', str(src), '--workers', str(a.workers),
                            '--single', name, '--mode', mode, '--dir', str(d)], capture_output=True, text=True)
        d.mkdir(parents=True, exist_ok=True); (d / 'run.log').write_text(p.stdout + p.stderr)
        return p.returncode
    probe = subprocess.run([sys.executable, '-c', 'import platform,os,numpy,cvxopt;print(platform.python_version(),platform.platform(),'
                            'platform.processor(),os.cpu_count(),numpy.__version__,cvxopt.__version__)'], capture_output=True, text=True)
    summary = {'env': {'probe': probe.stdout.strip(), 'workers': a.workers}, 'configs': {}}
    for name in a.configs:
        rows = []
        rdir = out / f'{name}_replay'
        if sub(name, 'replay', rdir):
            raise SystemExit(f'replay run failed: {rdir}/run.log')
        ref = np.load(rdir / 'arrays.npz')
        for r in range(a.repeats):
            d = out / f'{name}_{a.workers}w_r{r:02d}'
            if sub(name, 'host', d):
                rows.append({'repeat': r, 'error': (d / 'run.log').read_text()[-2000:]}); continue
            res = json.loads((d / 'result.json').read_text()); arr = np.load(d / 'arrays.npz')
            jobs = res.get('planning_jobs') or []
            same = [k for k in ref.files if k not in TIMING_ARR and not (
                k in arr.files and arr[k].shape == ref[k].shape and np.array_equal(arr[k], ref[k], equal_nan=ref[k].dtype.kind in 'fc'))]
            rows.append({'repeat': r, 'status': res.get('status'), 'last_done': res.get('last_done'),
                         'jobs': [{'id': j['id'], 'kind': j['kind'], 'n_candidates': len(j.get('search') or []),
                                   'compute_s': j.get('compute_seconds'), 'response_s': j.get('response_seconds'),
                                   'budget_s': round(j['deadline_s'] - j['dispatch_s'], 3), 'deadline_missed': j.get('deadline_missed'),
                                   'accepted': j.get('accepted')} for j in jobs],
                         'planning_deadline_misses': res.get('planning_deadline_misses'),
                         'control_deadline_misses': res.get('control_deadline_misses'),
                         'control_p95_s': res.get('control_p95_s'), 'control_p99_s': res.get('control_p99_s'),
                         'control_max_s': res.get('control_max_s'), 'tick_lateness_max_s': res.get('tick_lateness_max_s'),
                         'lateral_fallback_samples': res.get('lateral_fallback_samples'),
                         'nontiming_arrays_differing_from_replay': same})
            print(json.dumps({'config': name, 'repeat': r, 'misses': rows[-1]['planning_deadline_misses'],
                              'max_compute': max((j['compute_s'] or 0) for j in rows[-1]['jobs']) if rows[-1]['jobs'] else None,
                              'equal_to_replay': not same}), flush=True)
        ok = [rr for rr in rows if 'error' not in rr]
        comp = [j['compute_s'] for rr in ok for j in rr['jobs'] if j['compute_s'] is not None]
        ratio = [j['compute_s'] / j['budget_s'] for rr in ok for j in rr['jobs'] if j['compute_s'] is not None and j['budget_s']]
        summary['configs'][name] = {
            'repeats': len(rows), 'failed_runs': len(rows) - len(ok), 'jobs': len(comp),
            'compute_s': {'max': max(comp, default=None), 'p99': float(np.quantile(comp, .99)) if comp else None,
                          'median': float(np.median(comp)) if comp else None},
            'compute_over_budget_max_ratio': max(ratio, default=None),
            'planning_deadline_misses_total': sum(rr['planning_deadline_misses'] or 0 for rr in ok),
            'control_deadline_misses_total': sum(rr['control_deadline_misses'] or 0 for rr in ok),
            'control_p99_s_max': max(((rr['control_p99_s'] or 0) for rr in ok), default=None),
            'control_max_s_max': max(((rr['control_max_s'] or 0) for rr in ok), default=None),
            'runs_equal_to_replay': sum(1 for rr in ok if not rr['nontiming_arrays_differing_from_replay']),
            'rows': rows}
    (out / f'timing_summary_{a.workers}w.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'rows'} for k, v in summary['configs'].items()}, indent=1))


if __name__ == '__main__':
    main()
