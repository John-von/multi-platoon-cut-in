"""Write the registered WP5-v20 case list (cases_v1.json) and its SHA-256.  Deterministic; re-running must reproduce
the same bytes.  Usage: python make_cases.py"""
import collections, hashlib, json
from pathlib import Path
import casegen
HERE = Path(__file__).resolve().parent
cases = sorted(casegen.all_cases(), key=lambda c: c['id'])
doc = {'registry': 'WP5-v20 cases v1', 'generator': 'casegen.py', 'n_cases': len(cases),
       'n_unique_runs': len({c['run_key'] for c in cases}), 'cases': cases}
text = json.dumps(doc, sort_keys=True, indent=0, separators=(',', ':'), ensure_ascii=False)
(HERE / 'cases_v1.json').write_text(text, encoding='utf-8', newline='\n')
h = hashlib.sha256(text.encode('utf-8')).hexdigest()
(HERE / 'cases_v1.sha256').write_text(f'{h}  cases_v1.json\n', encoding='utf-8', newline='\n')
fam = collections.Counter(c['family'] for c in cases)
uniq = collections.defaultdict(set)
for c in cases: uniq[c['family']].add(c['run_key'])
by_method = collections.Counter((c['family'], c['method']) for c in cases)
summary = {'sha256': h, 'n_cases': len(cases), 'n_unique_runs': doc['n_unique_runs'],
           'per_family': {k: {'cases': v, 'unique_run_keys_within_family': len(uniq[k])} for k, v in sorted(fam.items())},
           'per_family_method': {f'{a}/{b}': n for (a, b), n in sorted(by_method.items())},
           'covered_A-H': sum(1 for c in cases if c['method'] == 'A-H' and c['theory_covered']),
           'adapter_cases': sum(1 for c in cases if c['adapters'])}
(HERE / 'cases_v1_summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8', newline='\n')
print(json.dumps(summary, indent=1, ensure_ascii=False))
