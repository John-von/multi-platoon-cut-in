"""Zero-intervention regression for the addendum-A adapters (20 s horizon; not registered configurations).
Identity (must be bitwise equal to the unadapted run): AD5 with nominal plant and no force; AD6 with the stage1_template
[8,7,7] scene at 30 m; AD7 'independent@identity' (frozen allocation recompiled without substitution).  Patch-effective
(must differ): AD5 corner C3; AD5 Fy pulse; AD6 template [8,7,7,7]; AD7 independent; AD7 sequential.
Usage: python zero_intervention_addA.py -> zi_addA/zi_addA_summary.json"""
import json, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import casegen
from zero_intervention import strip, TIMING_ARR
HERE = Path(__file__).resolve().parent; OUT = HERE / 'zi_addA'; OUT.mkdir(exist_ok=True)
def mk(tag, adapters=None, **kw):
    c = casegen.make('ZI', 'A-H', 'nominal', adapters=adapters, extra={'time_sim': 20}, **kw)
    c['workers'] = 1; c['id'] = tag; c['cfg_sha256'] = casegen.sha(c['cfg']); return c
def run(c):
    d = OUT / c['id']
    if (d / 'result.json').exists(): return
    d.mkdir(exist_ok=True); (d / 'case.json').write_text(json.dumps(c, indent=1), encoding='utf-8', newline='\n')
    p = subprocess.run([sys.executable, str(HERE / 'run_case_addA.py'), str(d / 'case.json'), str(d)], capture_output=True, text=True)
    (d / 'run.log').write_text(p.stdout + p.stderr, encoding='utf-8', newline='\n')
EXTRA = {'lateral_substate', 'slip_angles', 'lateral_external_force', 'pregenerated_force'}
def compare(a, b):
    ja = json.loads((OUT / a / 'result.json').read_text(encoding='utf-8')); jb = json.loads((OUT / b / 'result.json').read_text(encoding='utf-8'))
    A_, B_ = np.load(OUT / a / 'arrays.npz'), np.load(OUT / b / 'arrays.npz')
    keys = (set(A_.files) | set(B_.files)) - TIMING_ARR - EXTRA
    diff = sorted(k for k in keys if k not in A_.files or k not in B_.files or A_[k].shape != B_[k].shape or
                  not np.array_equal(A_[k], B_[k], equal_nan=A_[k].dtype.kind in 'fc'))
    return {'arrays_compared': len(keys), 'arrays_differing': diff, 'json_nontiming_equal': strip(ja) == strip(jb),
            'status': [ja.get('status'), jb.get('status')], 'exception': [ja['_meta'].get('exception'), jb['_meta'].get('exception')]}
if __name__ == '__main__':
    base = mk('base')
    ident = [mk('AD5id', {'AD5': {'scales': [1., 1., 1., 1.], 'pulse': [0., 0.], 'lateral_seed': None}}),
             mk('AD6id', {'AD6': {'family': 'stage1_template', 'queue_sizes': [8, 7, 7], 'boundary': 30.}}),
             mk('AD7id', {'AD7': 'independent@identity'})]
    eff = [mk('AD5C3', {'AD5': {'scales': [.7, .7, 1.2, 1.3], 'pulse': [0., 0.], 'lateral_seed': None}}),
           mk('AD5Fy', {'AD5': {'scales': [1., 1., 1., 1.], 'pulse': [.1 * 1500 * 9.81, 0.], 'lateral_seed': None}}),
           mk('AD6Q4', {'AD6': {'family': 'stage1_template', 'queue_sizes': [8, 7, 7, 7], 'boundary': 30.}}),
           mk('AD7ind', {'AD7': 'independent'}), mk('AD7seq', {'AD7': 'sequential'})]
    with ThreadPoolExecutor(2) as ex: list(ex.map(run, [base] + ident + eff))
    rows = {'identity': {c['id']: compare('base', c['id']) for c in ident},
            'patch_effective': {c['id']: compare('base', c['id']) for c in eff}}
    # AD5 identity must also reproduce the lateral trajectory at sub-steps: lateral_substate[::5] == lateral
    a = np.load(OUT / 'AD5id' / 'arrays.npz'); rows['AD5id_substate_vs_main_max_abs'] = float(np.abs(a['lateral_substate'][::5] - a['lateral']).max())
    ok = all(not r['arrays_differing'] and r['json_nontiming_equal'] and r['exception'] == [None, None] for r in rows['identity'].values())
    rows['summary'] = {'identity_all_bitwise': ok, 'substate_consistent': rows['AD5id_substate_vs_main_max_abs'] == 0.0,
                       'patch_effective_differ': {k: bool(r['arrays_differing'] or not r['json_nontiming_equal']) for k, r in rows['patch_effective'].items()}}
    (OUT / 'zi_addA_summary.json').write_text(json.dumps(rows, indent=1, default=str), encoding='utf-8', newline='\n'); print(json.dumps(rows['summary'], indent=1))
    for k, r in {**rows['identity'], **rows['patch_effective']}.items():
        print(k, r['status'], r['arrays_differing'][:5], r['json_nontiming_equal'], [None if e is None else e.get('type') for e in r['exception']])
