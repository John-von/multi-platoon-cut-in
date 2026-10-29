"""Zero-intervention regression for the WP5-v20 adapters (20 s horizon; these configurations are NOT in the registered
list, so no registered outcome is observed here).

(a) identity checks, must be bitwise equal to the unadapted run (all non-timing arrays and non-timing JSON fields):
    AD1 recorder installed; AD2 with b=13, Dmax=40 (recompiled capacity); AD3 substituted run_async without the
    'initial_durations' key; AD3 recompiled without substitution; AD4 V7/V8/V9 recompiled without substitution.
(b) patch-effective checks, must differ from the unadapted run: AD2 b=8; AD3 with initial_durations=[20];
    AD4 V7, V8, V9 active.  (These only show that the patch takes effect; their values are not results.)
Usage: python zero_intervention.py  -> zi/zi_summary.json"""
import json, os, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import casegen

HERE = Path(__file__).resolve().parent
OUT = HERE / 'zi'; OUT.mkdir(exist_ok=True)
TIMING_JSON = {'initial_plan_seconds', 'compute_seconds', 'worker_started_wall_s', 'worker_finished_wall_s', 'worker_pid',
               'result_observed_wall_s', 'queue_wait_s', 'response_seconds', 'result_observed_latency_s',
               'dispatch_wall_s', 'activation_wall_s', 'planning_over_0_1_fraction', 'planning_mean_s', 'planning_p95_s',
               'planning_p99_s', 'planning_max_s', 'control_mean_s', 'control_p95_s', 'control_p99_s', 'control_max_s',
               'control_over_0_1_fraction', 'control_active_p95_s', 'loop_max_s', 'wall_elapsed_s', 'tick_lateness_max_s',
               'host_process', 'python', 'runtime_s', 'step_mean_s', 'step_max_s', 'step_p95_s', 'over_0_1_fraction',
               'tracking_mean_s', 'planning_total_s', '_meta'}
TIMING_ARR = {'timing', 'planning_time', 'wall_time', 'loop_cost'}


def strip(v):
    if isinstance(v, dict):
        return {k: strip(x) for k, x in v.items() if k not in TIMING_JSON}
    if isinstance(v, list):
        return [strip(x) for x in v]
    return v


def zi_case(tag, base_kwargs, adapters=None, extra=None):
    c = casegen.make('ZI', 'A-H', adapters=adapters, extra=dict({'time_sim': 20}, **(extra or {})), **base_kwargs)
    c['workers'] = 1
    c['id'] = tag
    c['cfg_sha256'] = casegen.sha(c['cfg'])
    return c


def run(c):
    d = OUT / c['id']
    if (d / 'result.json').exists():
        return
    d.mkdir(exist_ok=True)
    (d / 'case.json').write_text(json.dumps(c, indent=1), encoding='utf-8', newline='\n')
    p = subprocess.run([sys.executable, str(HERE / 'run_case.py'), str(d / 'case.json'), str(d)],
                       capture_output=True, text=True)
    (d / 'run.log').write_text(p.stdout + p.stderr, encoding='utf-8', newline='\n')


def compare(a, b):
    ja = json.loads((OUT / a / 'result.json').read_text(encoding='utf-8')); jb = json.loads((OUT / b / 'result.json').read_text(encoding='utf-8'))
    A_, B_ = np.load(OUT / a / 'arrays.npz'), np.load(OUT / b / 'arrays.npz')
    diff = [k for k in sorted(set(A_.files) | set(B_.files)) if k not in TIMING_ARR and (
        k not in A_.files or k not in B_.files or A_[k].shape != B_[k].shape or
        not np.array_equal(A_[k], B_[k], equal_nan=A_[k].dtype.kind in 'fc'))]
    return {'arrays_compared': len([k for k in A_.files if k not in TIMING_ARR]), 'arrays_differing': diff,
            'json_nontiming_equal': strip(ja) == strip(jb), 'status': [ja.get('status'), jb.get('status')],
            'exception': [ja['_meta'].get('exception'), jb['_meta'].get('exception')]}


if __name__ == '__main__':
    bases = {'H1': dict(corner='nominal', ell=5), 'H2': dict(corner='high', s=5, ell=5)}
    cases, identity, effective = [], [], []
    for h, kw in bases.items():
        cases.append(zi_case(f'{h}_base', kw))
        for tag, ad, ex in [('AD1', {'AD1': True}, None), ('AD2id', {'AD2': {'b': 13, 'limit': 40.0}}, None),
                            ('AD3sub', {'AD3': True}, None), ('AD3id', {'AD3': 'identity'}, None),
                            ('V7id', {'AD4': 'V7@identity'}, None), ('V8id', {'AD4': 'V8@identity'}, None),
                            ('V9id', {'AD4': 'V9@identity'}, None)]:
            cases.append(zi_case(f'{h}_{tag}', kw, ad, ex)); identity.append((f'{h}_base', f'{h}_{tag}'))
        for tag, ad, ex in [('AD2b8', {'AD2': {'b': 8, 'limit': 40.0}}, None),
                            ('AD3init20', {'AD3': True}, {'initial_durations': [20]}),
                            ('V7', {'AD4': 'V7'}, None), ('V8', {'AD4': 'V8'}, None), ('V9', {'AD4': 'V9'}, None)]:
            cases.append(zi_case(f'{h}_{tag}', kw, ad, ex)); effective.append((f'{h}_base', f'{h}_{tag}'))
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(run, cases))
    rows = {'identity': {f'{a} vs {b}': compare(a, b) for a, b in identity},
            'patch_effective': {f'{a} vs {b}': compare(a, b) for a, b in effective}}
    ok_id = all(not r['arrays_differing'] and r['json_nontiming_equal'] and r['exception'] == [None, None]
                for r in rows['identity'].values())
    ok_eff = all((r['arrays_differing'] or not r['json_nontiming_equal']) for r in rows['patch_effective'].values())
    rows['summary'] = {'identity_all_bitwise': ok_id, 'patch_effective_all_differ': ok_eff,
                       'n_identity': len(identity), 'n_effective': len(effective)}
    (OUT / 'zi_summary.json').write_text(json.dumps(rows, indent=1, default=str), encoding='utf-8', newline='\n')
    print(json.dumps(rows['summary']))
    for k, r in {**rows['identity'], **rows['patch_effective']}.items():
        print(k, r['status'], r['arrays_differing'][:6], r['json_nontiming_equal'],
              [None if e is None else e.get('type') for e in r['exception']])
