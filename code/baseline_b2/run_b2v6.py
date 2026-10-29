"""Batch driver of the B2 planning study (fixed-t_f stage, rule E-10 of b2_nlp_v6.py).
  test : unit tests with the solver Python -> test_b2_nlp_v6.json
  lock : protocol.json + protocol.sha256 (code and input hashes, budgets)
  run  : P1 nine vehicles
           P1a free stage of main/sensitivity (S0 + one free main call), G16/G22/G28 in parallel
           P1b fixed stage of the main version, one job per group of identical free-stage solutions (parallel)
           P1c fixed stage of the sensitivity version from S0, grouped likewise (parallel)
           P1d original version: free first call (parallel), then fixed stage per group (parallel)
         P2 25 vehicles, main version only, only if a nine-vehicle main plan was accepted: free stage G16, G22, G28
            in sequence, then the fixed stage per group in sequence
  all  : test -> lock -> run
Budgets: 3 h per job, 6 h for P1, 10 h for P2. Writes only below this folder (runs/, progress.json, protocol.json).
No retries; no overwriting."""
import ctypes, datetime as dt, hashlib, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # package root
PY = Path(sys.executable)
RUNS = HERE / 'runs'
JOB_S = 3 * 3600.0
P1_S = 6 * 3600.0
P2_S = 10 * 3600.0
TGS = (16, 22, 28)
CODE = ('b2_nlp_v6.py', 'b2_components.py', 'frozen_scene.py', 'test_b2_nlp_v6.py', 'run_b2v6.py')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=1, default=str) + '\n', encoding='utf-8', newline='\n')
    os.replace(tmp, p)


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def keep_awake(on):
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))
        return True
    except Exception:
        return False


def env():
    e = os.environ.copy()
    e.update({'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
              'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'V20_SRC': str(ROOT / 'code' / 'sim' / 'sources'),
              'V20_ROOT': str(ROOT / 'code' / 'sim'), 'V20_PLAN_WORKERS': '1'})
    return e


def run_cmd(args, log, limit):
    info = {'command': [str(a) for a in args], 'started_utc': now(), 'limit_s': limit}
    t0 = time.monotonic(); Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'wb') as h:
        p = subprocess.Popen([str(a) for a in args], cwd=str(HERE), env=env(), stdout=h, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            info['exit_code'] = p.wait(timeout=max(1.0, limit)); info['killed_at_budget'] = False
        except subprocess.TimeoutExpired:
            p.kill(); p.wait(); info['exit_code'] = p.returncode; info['killed_at_budget'] = True
    info['wall_s'] = round(time.monotonic() - t0, 3); info['finished_utc'] = now()
    return info


def test():
    r = run_cmd([PY, '-B', HERE / 'test_b2_nlp_v6.py'], HERE / 'test_console.txt', 1800)
    save(HERE / 'test_run.json', r)
    t = HERE / 'test_b2_nlp_v6.json'
    return r['exit_code'] == 0 and t.exists() and load(t)['pass'] and load(t)['casadi_available']


def lock():
    if (HERE / 'protocol.json').exists():
        raise SystemExit('protocol already locked; no overwrite')
    t = load(HERE / 'test_b2_nlp_v6.json')
    if not (t['pass'] and t['casadi_available'] and t['b2_nlp_v6_sha256'] == sha(HERE / 'b2_nlp_v6.py')
            and t['test_sha256'] == sha(HERE / 'test_b2_nlp_v6.py')):
        raise SystemExit('tests missing, failed or stale')
    prot = {'name': 'B2 v6 W4-1 (E-10 fixed-t_f stage)', 'locked_utc': now(),
            'code_sha256': {f: sha(HERE / f) for f in CODE}, 'test_result_sha256': sha(HERE / 'test_b2_nlp_v6.json'),
            'request_state_sha256': sha(ROOT / 'data/baseline_b2/request_state_EX1a_nominal.json'),
            'source_npz_sha256': sha(ROOT / 'data/original/runs_selected/cb13911c1dadd2de7413/arrays.npz'),
            'python': str(PY), 'budgets_s': {'per_job': JOB_S, 'P1': P1_S, 'P2': P2_S},
            'no_retry': True, 'closed_loop': False, 'lambda_over_gamma': 100}
    save(HERE / 'protocol.json', prot)
    (HERE / 'protocol.sha256').write_text(sha(HERE / 'protocol.json') + '  protocol.json\n', encoding='utf-8', newline='\n')


def job(args, tag, limit):
    if (RUNS / 'jobs' / f'{tag}.json').exists():
        return load(RUNS / 'jobs' / f'{tag}.json')
    r = run_cmd([PY, '-B', HERE / 'b2_nlp_v6.py', *args], RUNS / 'jobs' / f'{tag}_console.txt', limit)
    r['tag'] = tag
    save(RUNS / 'jobs' / f'{tag}.json', r)
    return r


def groups(scope, tgs, src_rel, accept_ok):
    """Groups initial values whose free-stage solution is identical (E-10 item 3)."""
    sols = {}
    for tg in tgs:
        d = RUNS / scope / f'G{tg}' / src_rel
        if not (d / 'solution.npz').exists() or not (d / 'result.json').exists():
            continue
        r = load(d / 'result.json')
        with np.load(d / 'solution.npz', allow_pickle=False) as z:
            sols[tg] = (float(z['tf']), float(r['objective_raw_lambda100_gamma1']), z['S'].copy(), z['U'].copy())
    out = []; used = set()
    for tg in tgs:
        if tg not in sols or tg in used:
            continue
        g = [tg]; used.add(tg)
        for o in tgs:
            if o in sols and o not in used:
                a, b = sols[tg], sols[o]
                same = (abs(a[0] - b[0]) <= 1e-6 and abs(a[1] - b[1]) <= 1e-8 * max(1.0, abs(a[1]))
                        and float(np.max(np.abs(a[2] - b[2]))) <= 1e-4 and float(np.max(np.abs(a[3] - b[3]))) <= 1e-4)
                if same:
                    g.append(o); used.add(o)
        out.append(g)
    return out


def parallel(calls, status, name):
    status['phases'][name] = {'started_utc': now(), 'jobs': [c[1] for c in calls]}; save(HERE / 'progress.json', status)
    with ThreadPoolExecutor(max(1, len(calls))) as ex:
        rows = list(ex.map(lambda c: job(*c), calls))
    status['phases'][name].update({'finished_utc': now(), 'rows': rows}); save(HERE / 'progress.json', status)
    return rows


def run():
    p = load(HERE / 'protocol.json')
    if sha(HERE / 'protocol.json') != (HERE / 'protocol.sha256').read_text(encoding='utf-8').split()[0]:
        raise SystemExit('protocol hash changed')
    bad = [f for f, h in p['code_sha256'].items() if sha(HERE / f) != h]
    if bad:
        raise SystemExit(f'code changed after lock: {bad}')
    status = {'started_utc': now(), 'protocol_sha256': sha(HERE / 'protocol.json'), 'phases': {}, 'keep_awake': keep_awake(True)}
    t1 = time.monotonic(); rem = lambda: P1_S - (time.monotonic() - t1)
    sc = 'single'; base = RUNS / sc
    parallel([(['free_main', '--scope', sc, '--Tg', str(tg), '--out', base / f'G{tg}'], f'free_main_{sc}_G{tg}', min(JOB_S, rem())) for tg in TGS],
             status, 'P1a_free_main')
    for variant, src, name in (('main', 'main/call_1', 'P1b_fixed_main'), ('sensitivity', 'sensitivity/call_1', 'P1c_fixed_sensitivity')):
        if rem() <= 60:
            status['phases'][name] = {'status': 'not_started_phase_budget'}; continue
        gs = groups(sc, TGS, src, True)
        status[f'groups_{variant}'] = gs
        parallel([(['fixed', '--variant', variant, '--scope', sc, '--Tg', str(g[0]), '--out', base / f'G{g[0]}',
                    '--share', ','.join(str(x) for x in g[1:]), '--budget', str(min(JOB_S, rem()))],
                   f'fixed_{variant}_{sc}_G{g[0]}', min(JOB_S, rem()) + 300) for g in gs], status, name)
    if rem() > 60:
        parallel([(['original_free', '--scope', sc, '--Tg', str(tg), '--out', base / f'G{tg}'], f'original_free_{sc}_G{tg}', min(JOB_S, rem()))
                  for tg in TGS], status, 'P1d_original_free')
        if rem() > 60:
            gs = groups(sc, TGS, 'original/call_1', True); status['groups_original'] = gs
            parallel([(['fixed', '--variant', 'original', '--scope', sc, '--Tg', str(g[0]), '--out', base / f'G{g[0]}',
                        '--share', ','.join(str(x) for x in g[1:]), '--budget', str(min(JOB_S, rem()))],
                       f'fixed_original_{sc}_G{g[0]}', min(JOB_S, rem()) + 300) for g in gs], status, 'P1d_fixed_original')
    main_ok = [tg for tg in TGS if (base / f'G{tg}' / 'main' / 'result.json').exists()
               and load(base / f'G{tg}' / 'main' / 'result.json').get('status') == 'accepted_plan']
    status['nine_vehicle_main_accepted'] = main_ok
    if main_ok:
        t2 = time.monotonic(); rem2 = lambda: P2_S - (time.monotonic() - t2); sc = 'full'; base = RUNS / sc
        for tg in TGS:
            if rem2() <= 60:
                break
            parallel([(['free_main', '--scope', sc, '--Tg', str(tg), '--out', base / f'G{tg}'], f'free_main_{sc}_G{tg}', min(JOB_S, rem2()))],
                     status, f'P2_free_main_G{tg}')
        gs = groups(sc, TGS, 'main/call_1', True); status['groups_full_main'] = gs
        for g in gs:
            if rem2() <= 60:
                break
            parallel([(['fixed', '--variant', 'main', '--scope', sc, '--Tg', str(g[0]), '--out', base / f'G{g[0]}',
                        '--share', ','.join(str(x) for x in g[1:]), '--budget', str(min(JOB_S, rem2()))],
                       f'fixed_main_{sc}_G{g[0]}', min(JOB_S, rem2()) + 300)], status, f'P2_fixed_main_G{g[0]}')
    else:
        status['phases']['P2'] = {'status': 'not_started', 'reason': 'no nine-vehicle main plan accepted (registered stop)'}
    status['finished_utc'] = now(); keep_awake(False); save(HERE / 'progress.json', status)


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if a == 'test':
        print('tests pass' if test() else 'TESTS FAILED')
    elif a == 'lock':
        lock()
    elif a == 'run':
        run()
    elif a == 'all':
        if not test():
            save(HERE / 'progress.json', {'stopped': 'tests failed; nothing locked or solved', 'utc': now()})
            raise SystemExit(1)
        lock(); run()
