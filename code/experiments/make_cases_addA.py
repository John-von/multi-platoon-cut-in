"""Write the registered WP5-v20 addendum-A case list (cases_addA_v1.json) and its SHA-256.  Deterministic."""
import collections, hashlib, json
from pathlib import Path
import casegen_addA
HERE = Path(__file__).resolve().parent
cases = sorted(casegen_addA.all_cases(), key=lambda c: c['id'])
v1 = {c['run_key'] for c in json.loads((HERE / 'cases_v1.json').read_text(encoding='utf-8'))['cases']}
doc = {'registry': 'WP5-v20 cases addendum A v1', 'generator': 'casegen_addA.py', 'n_cases': len(cases),
       'n_unique_runs': len({c['run_key'] for c in cases}),
       'n_runs_shared_with_v1': len({c['run_key'] for c in cases} & v1), 'cases': cases}
text = json.dumps(doc, sort_keys=True, indent=0, separators=(',', ':'), ensure_ascii=False)
(HERE / 'cases_addA_v1.json').write_text(text, encoding='utf-8', newline='\n')
h = hashlib.sha256(text.encode('utf-8')).hexdigest()
(HERE / 'cases_addA_v1.sha256').write_text(f'{h}  cases_addA_v1.json\n', encoding='utf-8', newline='\n')
fam = collections.Counter(c['family'] for c in cases)
summary = {'sha256': h, 'n_cases': len(cases), 'n_unique_runs': doc['n_unique_runs'], 'n_runs_shared_with_v1': doc['n_runs_shared_with_v1'],
           'per_family': dict(sorted(fam.items())), 'adapter_cases': sum(1 for c in cases if c['adapters']),
           'new_runs_not_in_v1': len({c['run_key'] for c in cases} - v1)}
(HERE / 'cases_addA_v1_summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8', newline='\n')
print(json.dumps(summary, indent=1))
