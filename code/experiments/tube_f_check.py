"""Independent check of the TUBE == F finding (design section 3.4): the registered EX1a nominal cases of methods TUBE
and F, shortened to 20 s (so no registered outcome is produced), are run once each with run_case.py and all non-timing
arrays and non-timing JSON fields are compared.  Usage: python tube_f_check.py"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import casegen
from zero_intervention import strip, TIMING_ARR
HERE = Path(__file__).resolve().parent; OUT = HERE / 'tube_f_check'; OUT.mkdir(exist_ok=True)
rows = {}
for m in ('TUBE', 'F'):
    c = casegen.make('CHECK', m, 'nominal', extra={'time_sim': 20})
    c['id'] = f'check_{m}_nominal_20s'; d = OUT / c['id']; d.mkdir(exist_ok=True)
    (d / 'case.json').write_text(json.dumps(c, indent=1), encoding='utf-8', newline='\n')
    if not (d / 'result.json').exists():
        p = subprocess.run([sys.executable, str(HERE / 'run_case.py'), str(d / 'case.json'), str(d)], capture_output=True, text=True)
        (d / 'run.log').write_text(p.stdout + p.stderr, encoding='utf-8', newline='\n')
    rows[m] = d
a, b = (np.load(rows[m] / 'arrays.npz') for m in ('TUBE', 'F'))
diff = [k for k in sorted(set(a.files) | set(b.files)) if k not in TIMING_ARR and (
    k not in a.files or k not in b.files or not np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind in 'fc'))]
ja, jb = (json.loads((rows[m] / 'result.json').read_text(encoding='utf-8')) for m in ('TUBE', 'F'))
ja['cfg'].pop('method'); jb['cfg'].pop('method')
res = {'arrays_compared': len([k for k in a.files if k not in TIMING_ARR]), 'arrays_differing': diff,
       'json_nontiming_equal_except_method': strip(ja) == strip(jb)}
(OUT / 'tube_f_check.json').write_text(json.dumps(res, indent=1), encoding='utf-8', newline='\n'); print(json.dumps(res))
