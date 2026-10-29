"""EX10-T (design addendum A, section 3): timing subset of EX10 on the timing host (Windows PC, 16 logical processors).

Selects the 12 registered EX10 configurations of the stage1_template family (8 geometries at 30 m and the 4
LP-infeasible geometries at their smallest restoring spacing b*), runs each ONCE through run_case_addA.py with
V20_PLAN_WORKERS = N (replay mode, as registered; outcomes are host-independent), sequentially (one run at a time,
no other load), and writes a timing summary.  Only the 'workers' field of each case is changed; 'cfg' and its
SHA-256 are untouched.  Existing evidence is never overwritten.

Usage (from this folder; the sources are read from ../sim/sources unless V20_SRC is set):
  python ex10t.py <out_dir> --workers N"""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def pick():
    text = (HERE / 'cases_addA_v1.json').read_text(encoding='utf-8')
    reg = (HERE / 'cases_addA_v1.sha256').read_text(encoding='utf-8').split()[0]
    if hashlib.sha256(text.encode('utf-8')).hexdigest() != reg:
        raise SystemExit('cases_addA_v1.json does not match the registered SHA-256')
    cases = [c for c in json.loads(text)['cases'] if c['family'] == 'EX10' and c['cell'].get('family_scene') == 'stage1_template']
    by_geom = {}
    for c in cases:
        by_geom.setdefault(c['cell']['geometry'], []).append(c)
    out = []
    for g, cs in by_geom.items():
        cs = sorted(cs, key=lambda c: c['cell']['boundary'])
        out.append(cs[0])                        # 30 m
        if len(cs) > 1:
            out.append(cs[1])                    # b* (smallest restoring spacing)
    assert len(out) == 12, len(out)
    return out


def summarize(d):
    r = json.loads((d / 'result.json').read_text(encoding='utf-8'))
    jobs = r.get('planning_jobs') or []
    meta = r['_meta']; sc = meta.get('scene', {})
    return {'id': meta['case_id'], 'vehicles': int(sc.get('N', 0)) + int(sc.get('Q', 0)), 'status': r.get('status'),
            'exception': (meta.get('exception') or {}).get('type'), 'workers': meta['workers'], 'wall_s': meta['wall_s'],
            'initial_plan_seconds': r.get('initial_plan_seconds'),
            'jobs': [{'kind': j.get('kind'), 'compute_seconds': j.get('compute_seconds'),
                      'horizons_tried': len(j.get('search') or []), 'accepted': j.get('accepted'),
                      'rejection_reason': j.get('rejection_reason')} for j in jobs],
            'control_p95_s': r.get('control_p95_s'), 'control_p99_s': r.get('control_p99_s'), 'control_max_s': r.get('control_max_s'),
            'host_cpus': meta.get('host_cpus'), 'platform': meta.get('platform')}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--workers', type=int, required=True)
    a = ap.parse_args(); root = Path(a.out); root.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in pick():
        d = root / c['run_key'][:20]
        if not (d / 'result.json').exists():
            d.mkdir(exist_ok=True); c = dict(c); c['workers'] = a.workers
            (d / 'case.json').write_text(json.dumps(c, indent=1), encoding='utf-8', newline='\n')
            t = time.time()
            p = subprocess.run([sys.executable, str(HERE / 'run_case_addA.py'), str(d / 'case.json'), str(d)],
                               capture_output=True, text=True)
            (d / 'run.log').write_text(p.stdout + p.stderr, encoding='utf-8', newline='\n')
            print(json.dumps({'id': c['id'], 'rc': p.returncode, 'wall_s': round(time.time() - t, 1)}), flush=True)
            if p.returncode:
                rows.append({'id': c['id'], 'run_failed': True}); continue
        rows.append(summarize(d))
    (root / 'ex10t_summary.json').write_text(json.dumps(rows, indent=1), encoding='utf-8', newline='\n')
    print(json.dumps([{k: r.get(k) for k in ('id', 'vehicles', 'status', 'wall_s')} for r in rows], indent=0))


if __name__ == '__main__':
    main()
