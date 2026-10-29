"""Check (not registered): an AD6-only case gives bitwise identical non-timing arrays with 1 and 2 capacity processes
(20 s horizon, stage1_template [8,7,7,7] at 30 m).  Supports the AD6-only exception in run_case_addA.py (EX10-T)."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import casegen
from zero_intervention import strip, TIMING_ARR
HERE = Path(__file__).resolve().parent; OUT = HERE / 'zi_ad6_workers'; OUT.mkdir(exist_ok=True)
def mk(tag, w):
    c = casegen.make('ZI', 'A-H', 'nominal', adapters={'AD6': {'family': 'stage1_template', 'queue_sizes': [8, 7, 7, 7], 'boundary': 30.}},
                     extra={'time_sim': 20})
    c['workers'] = w; c['id'] = tag; c['cfg_sha256'] = casegen.sha(c['cfg']); return c
for c in (mk('w1', 1), mk('w2', 2)):
    d = OUT / c['id']
    if (d / 'result.json').exists(): continue
    d.mkdir(exist_ok=True); (d / 'case.json').write_text(json.dumps(c, indent=1), encoding='utf-8', newline='\n')
    p = subprocess.run([sys.executable, str(HERE / 'run_case_addA.py'), str(d / 'case.json'), str(d)], capture_output=True, text=True)
    (d / 'run.log').write_text(p.stdout + p.stderr, encoding='utf-8', newline='\n'); print(p.stdout.strip()[-300:], p.returncode)
A_, B_ = np.load(OUT / 'w1' / 'arrays.npz'), np.load(OUT / 'w2' / 'arrays.npz')
keys = (set(A_.files) | set(B_.files)) - TIMING_ARR
diff = sorted(k for k in keys if k not in A_.files or k not in B_.files or A_[k].shape != B_[k].shape or not np.array_equal(A_[k], B_[k], equal_nan=A_[k].dtype.kind in 'fc'))
ja = json.loads((OUT / 'w1' / 'result.json').read_text(encoding='utf-8')); jb = json.loads((OUT / 'w2' / 'result.json').read_text(encoding='utf-8'))
ja['_meta'].pop('workers'); jb['_meta'].pop('workers')
res = {'arrays_compared': len(keys), 'arrays_differing': diff, 'json_nontiming_equal': strip(ja) == strip(jb),
       'workers': [1, 2], 'status': [ja.get('status'), jb.get('status')]}
(OUT / 'summary.json').write_text(json.dumps(res, indent=1), encoding='utf-8', newline='\n'); print(json.dumps(res))
