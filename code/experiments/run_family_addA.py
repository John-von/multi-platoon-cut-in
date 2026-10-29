"""Execute the registered addendum-A cases (unique run keys) and audit them (same rules as run_family.py).

Usage: python run_family_addA.py <runs_root> [--family EX11 EX10 EX9] [--jobs 2] [--limit N]
Checks cases_addA_v1.json against cases_addA_v1.sha256 first.  One directory per unique run key:
<runs_root>/<run_key[:20]>/ (case.json, result.json, arrays.npz, audit.json, run.log).  Run keys shared with
cases_v1.json (53 EX11 'joint' cases) are run once, by whichever batch reaches them first; existing complete evidence is
kept and never overwritten; every attempt is appended to <runs_root>/attempts.jsonl.  Runner: run_case_addA.py (a
superset of run_case.py; v1 cases run identically through either).  Audit: audit_lateral.py for EX9 (audit_run.audit
plus the lateral diagnostics), audit_run.py otherwise."""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('--family', nargs='*')
    ap.add_argument('--jobs', type=int, default=2); ap.add_argument('--limit', type=int)
    a = ap.parse_args(); root = Path(a.root); root.mkdir(parents=True, exist_ok=True)
    text = (HERE / 'cases_addA_v1.json').read_text(encoding='utf-8')
    reg = (HERE / 'cases_addA_v1.sha256').read_text(encoding='utf-8').split()[0]
    if hashlib.sha256(text.encode('utf-8')).hexdigest() != reg:
        raise SystemExit('cases_addA_v1.json does not match the registered SHA-256')
    cases = json.loads(text)['cases']
    todo = {}
    for c in cases:
        if a.family and c['family'] not in a.family: continue
        todo.setdefault(c['run_key'], c)
    items = list(todo.values())[: a.limit] if a.limit else list(todo.values())
    log = root / 'attempts.jsonl'

    def one(c):
        d = root / c['run_key'][:20]
        if (d / 'audit.json').exists(): return 'kept'
        d.mkdir(exist_ok=True)
        if not (d / 'result.json').exists():
            (d / 'case.json').write_text(json.dumps(c, indent=1), encoding='utf-8', newline='\n')
            t = time.time()
            p = subprocess.run([sys.executable, str(HERE / 'run_case_addA.py'), str(d / 'case.json'), str(d)],
                               capture_output=True, text=True)
            (d / 'run.log').write_text(p.stdout + p.stderr, encoding='utf-8', newline='\n')
            with open(log, 'a', encoding='utf-8', newline='\n') as f:
                f.write(json.dumps({'run_key': c['run_key'], 'id': c['id'], 'rc': p.returncode, 'start': t,
                                    'wall_s': round(time.time() - t, 2), 'runner': 'run_case_addA.py'}) + '\n')
            if p.returncode: return 'run_failed'
        auditor = 'audit_lateral.py' if c['family'] == 'EX9' else 'audit_run.py'
        p = subprocess.run([sys.executable, str(HERE / auditor), str(d)], capture_output=True, text=True)
        if p.returncode:
            (d / 'audit.err').write_text(p.stdout + p.stderr, encoding='utf-8', newline='\n'); return 'audit_failed'
        return 'done'

    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(one, items))
    idx = {c['id']: c['run_key'][:20] for c in cases if (not a.family or c['family'] in a.family)}
    (root / f"index_addA_{'_'.join(a.family) if a.family else 'all'}.json").write_text(json.dumps(idx, indent=0),
                                                                                     encoding='utf-8', newline='\n')
    from collections import Counter
    print(json.dumps(Counter(res)))


if __name__ == '__main__':
    main()
