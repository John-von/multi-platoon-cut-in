"""Addendum-A runner: identical to run_case.py (v1) plus the addendum adapters AD5 (lateral plant), AD6 (scene) and
AD7 (allocation variant).  Cases of cases_v1.json run identically through either script.

Usage: python run_case_addA.py <case.json> <out_dir> [--src <sources_dir>] [--allow-unfrozen]
The case file holds one entry of cases_v1.json: {id, cfg_sha256, runner, workers, adapters, cfg, ...}.
Writes <out_dir>/result.json (runner result + _meta) and <out_dir>/arrays.npz (all runner arrays; for a run stopped
by an exception with AD1 installed, the recorded prefix).  Frozen files are imported read-only; before running, the
six sources, metric.json and stage1_plan.mat are hashed and compared with the E1a freeze manifest values below.
Lateral QP: real CVXOPT 1.3.2.  lateral_fast imports osqp at import time; only that import is served by the
stand-in module in ../sim/osqp_only (no registered configuration selects the OSQP path; asserted)."""
import argparse, hashlib, json, os, platform, sys, time, traceback
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

HERE = Path(__file__).resolve().parent
E1A = {'async_execution.py': 'b6d15027e6d13cadc269c4f6226286eb0ea51a1c1ac93182970bbd303953faba',
       'lateral_fallback.py': '275119c8211847ae3928485b651cab3eb6c9ae538180fbf6fa34a3e8d0dec437',
       'lateral_fast.py': 'b3765e62fc3c8af447bc08aae79038935734f3e45324084fab7cb7f47abb47a1',
       'robust_core.py': '5279520a2b4228ffae69db664abfa9696f54268cf3dd674841cb0eb099fda67a',
       'simulation.py': '97e0b11140e42bc65c787806046fcaab4adce565392beecc579029d4b688f69e',
       'verify_core.py': 'dfa89634eedac9687962925eb4b543da5a7533f28afe8029920607e7ebb262e4',
       '../metric.json': '8c06df3039aa077a91bd972fb6cb5b8c88ac75c3db8a9509b45176646c24bcdc',
       '../baseline/stage1_plan.mat': '3d66d27ec0939a44c9371267b0a51f9bf7524f44db3cb164c8938be680690fa7'}
# Default layout of this package: ../sim/sources and ../sim/osqp_only (V20_SRC and V20_OSQP_STANDIN override them).
DEFAULT_SRC = Path(os.environ.get('V20_SRC', HERE.parent / 'sim' / 'sources'))
OSQP_STANDIN = Path(os.environ.get('V20_OSQP_STANDIN', HERE.parent / 'sim' / 'osqp_only'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clean(v):
    import numpy as np
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, np.generic): return clean(v.item())
    if isinstance(v, float) and not np.isfinite(v): return None
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('case'); ap.add_argument('out')
    ap.add_argument('--src', default=str(DEFAULT_SRC)); ap.add_argument('--allow-unfrozen', action='store_true')
    a = ap.parse_args()
    case = json.loads(Path(a.case).read_text(encoding='utf-8'))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        raise FileExistsError(f'evidence exists: {out}')
    src = Path(a.src).resolve()
    hashes = {k: sha(src / k) for k in E1A}
    mismatch = {k: v for k, v in hashes.items() if v != E1A[k]}
    if mismatch and not a.allow_unfrozen:
        raise SystemExit(f'source hash mismatch vs E1a manifest: {mismatch}')
    cfg_canon = json.dumps(case['cfg'], sort_keys=True, separators=(',', ':'))
    if hashlib.sha256(cfg_canon.encode()).hexdigest() != case['cfg_sha256']:
        raise SystemExit('cfg hash mismatch')
    adapters = case.get('adapters') or {}
    workers = int(case.get('workers', 2))
    # AD6 only replaces scenario() in the parent process; the capacity workers never call it, so an AD6-only case
    # may use several capacity processes (EX10-T timing).  Every other adapter must run with one process, because
    # spawn-started workers do not see in-process replacements.
    if adapters and workers != 1 and set(adapters) - {'AD6'}:
        raise SystemExit('adapter runs (other than AD6-only) must use V20_PLAN_WORKERS=1')
    os.environ['V20_PLAN_WORKERS'] = str(workers)
    sys.path[:0] = ([str(OSQP_STANDIN)] if OSQP_STANDIN.exists() else []) + [str(src)]
    import numpy as np, scipy, cvxopt, osqp
    assert not hasattr(cvxopt, 'STANDIN') and cvxopt.__version__ == '1.3.2', 'real CVXOPT 1.3.2 required'
    cfg = case['cfg']
    assert cfg.get('lateral_solver') in (None, 'cvxopt')
    import robust_core as R, simulation as S, async_execution as A
    import adapters as AD, adapters_addA as ADA
    rec = None; run_async = A.run_async; plant = None
    if 'AD6' in adapters:
        ADA.install_scene(S, A, adapters['AD6'])
    sc = S.scenario(cfg.get('initial_gap', 10.5), cfg.get('boundary', 30.))
    if 'AD5' in adapters:
        plant = ADA.install_lateral(S, A, adapters['AD5'], steps=round(cfg.get('time_sim', 50) / 0.1), q=sc['Q'])
    if 'AD7' in adapters:
        ADA.install_allocation(R, S, adapters['AD7'])
    if 'AD1' in adapters:
        sc = S.scenario(cfg.get('initial_gap', 10.5), cfg.get('boundary', 30.))
        rec = AD.Recorder(sc['N'] + sc['Q']).install(A)
    if 'AD2' in adapters:
        AD.install_capacity(R, S, **adapters['AD2'])
    if 'AD3' in adapters:
        run_async = AD.load_run_async_initial_list(A, identity=(adapters['AD3'] == 'identity'))
    if 'AD4' in adapters:
        variant, _, mode = str(adapters['AD4']).partition('@')
        AD.install_mechanism(R, S, A, variant, identity=(mode == 'identity'))
    meta = {'case_id': case['id'], 'cfg_sha256': case['cfg_sha256'], 'runner': case['runner'], 'workers': workers,
            'adapters': adapters, 'adapter_diffs': dict(AD.REGISTRY), 'source_dir': str(src), 'source_hashes': hashes,
            'frozen_E1a_match': not mismatch, 'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'cvxopt': cvxopt.__version__, 'cvxopt_file': cvxopt.__file__,
            'osqp': 'import-only stand-in' if getattr(osqp, 'STANDIN', False) else getattr(osqp, '__version__', 'real'),
            'platform': platform.platform(), 'host_cpus': os.cpu_count()}
    t0 = time.time(); arrays = {}
    try:
        if case['runner'] == 'handover':
            assert cfg['v20']['replan_mode'] == 'handover' and cfg.get('execution') == 'replay'
            res, arrays = run_async(cfg)
        else:
            assert cfg['v20']['replan_mode'] == 'sync' and 'execution' not in cfg
            res, arrays = S.simulate(cfg)
        meta['exception'] = None
    except Exception as e:  # recorded, never swallowed silently
        res = {'status': 'exception', 'cfg': cfg}
        meta['exception'] = {'type': type(e).__name__, 'message': str(e), 'traceback': traceback.format_exc()[-4000:]}
        if rec is not None:
            meta['recorder_event'] = rec.exception
            arrays = rec.prefix()
            meta['prefix_complete_steps'] = arrays.get('complete_steps')
    meta['wall_s'] = time.time() - t0
    if plant is not None:
        plant.add_arrays(arrays); meta['lateral_force_sha256'] = plant.force_hash; meta['lateral_plant_calls'] = plant.calls
    meta['scene'] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in sc.items()}
    res = dict(res); res['_meta'] = meta
    arrays = arrays or {}
    np.savez_compressed(out / 'arrays.npz', **{k: np.asarray(v) for k, v in arrays.items() if np.asarray(v).dtype != object})
    (out / 'result.json').write_text(json.dumps(clean(res), indent=1, ensure_ascii=False), encoding='utf-8', newline='\n')
    print(json.dumps({'id': case['id'], 'status': res.get('status'), 'exception': (meta['exception'] or {}).get('type'),
                      'wall_s': round(meta['wall_s'], 1)}), flush=True)


if __name__ == '__main__':
    main()
