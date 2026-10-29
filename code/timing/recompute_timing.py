"""Independently reconstruct timing from raw files; never import simulation code."""
import ast
import math
import re
import sys
import platform
import numpy as np
from common import HERE, ROOT, sha, read, save, csvsave

RAW = ROOT / 'data/timing/host_runs'
OUT = HERE / 'out'
EXCLUDED = {'timing', 'planning_time', 'wall_time', 'loop_cost', 'tick_lateness',
            'control_deadline_miss', 'lateral_fallback'}
checks = []
inputs = {}


def check(label, actual, expected, atol=1e-12):
    if isinstance(actual, (int, float, np.number)) and not isinstance(actual, (bool, np.bool_)):
        ok = expected is not None and math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=1e-12)
        actual = float(actual)
    else:
        ok = actual == expected
    checks.append({'check': label, 'passed': bool(ok), 'actual': actual, 'expected': expected})


def track(p):
    inputs[p.relative_to(ROOT).as_posix()] = sha(p)
    return p


def stats(x):
    x = np.asarray(x, float)
    assert x.ndim == 1 and np.all(np.isfinite(x))
    return {'n': len(x), 'median': float(np.median(x)) if len(x) else None,
            'p95': float(np.quantile(x, .95, method='linear')) if len(x) else None,
            'p99': float(np.quantile(x, .99, method='linear')) if len(x) else None,
            'max': float(np.max(x)) if len(x) else None}


def main():
    # Literal extraction only: no import or execution of the timing harness.
    tree = ast.parse(track(HERE / 'timing_host.py').read_text(encoding='utf-8'))
    expected_src = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'E1A' for t in n.targets))
    for name, digest in expected_src.items():
        p = track(ROOT / 'code/sim/sources' / name)
        check('source.' + name, sha(p), digest)
    summaries = {w: read(track(RAW / f'timing_summary_{w}w.json')) for w in (1, 16)}
    directories = sorted(p for p in RAW.iterdir() if p.is_dir() and re.fullmatch(r'T[1-4]_(?:1|16)w_r\d{2}', p.name))
    check('host_run_count', len(directories), 52)
    jobs, runs, samples, configurations = [], [], [], []
    replay = {}
    for c in ('T1','T2','T3','T4'):
        p = track(RAW / f'{c}_replay/arrays.npz')
        with np.load(p, allow_pickle=False) as z:
            replay[c] = {k:z[k] for k in z.files}
        track(RAW / f'{c}_replay/result.json')
        track(RAW / f'{c}_replay/cfg.json')
    for directory in directories:
        c, workers, rep = re.fullmatch(r'(T[1-4])_(1|16)w_r(\d{2})',directory.name).groups()
        workers, rep = int(workers), int(rep)
        res = read(track(directory/'result.json')); cfg = read(track(directory/'cfg.json'))
        check(directory.name+'.cfg_saved', cfg, res['cfg'])
        with np.load(track(directory/'arrays.npz'), allow_pickle=False) as z:
            arr = {k:z[k] for k in z.files}
        times = arr['time']; dt = float(np.diff(times)[0]); timing = arr['timing']; active = arr['active_control']
        assert timing.shape == active.shape == arr['control_deadline_miss'].shape == (500,)
        check(directory.name+'.dt', dt, .1)
        check(directory.name+'.disturbance_fraction', cfg['disturbance_fraction'], .75)
        all_s = stats(timing); active_s = stats(timing[active])
        for stat in ('p95','p99','max'):
            check(directory.name+'.control_'+stat, all_s[stat], res['control_'+stat+'_s'])
        check(directory.name+'.active_p95', active_s['p95'], res['control_active_p95_s'])
        check(directory.name+'.active_count', int(active.sum()), res['active_control_samples'])
        check(directory.name+'.miss_flags', int(arr['control_deadline_miss'].sum()),res['control_deadline_misses'])
        check(directory.name+'.loop_max', float(arr['loop_cost'].max()),res['loop_max_s'])
        check(directory.name+'.tick_lateness_max', float(arr['tick_lateness'].max()),res['tick_lateness_max_s'])
        # This reproduces the documented non-timing comparison, including active_control.
        different=[]
        check(directory.name+'.array_key_set',sorted(arr),sorted(replay[c]))
        for k,b in replay[c].items():
            if k not in EXCLUDED and (k not in arr or arr[k].shape!=b.shape or
                not np.array_equal(arr[k],b,equal_nan=b.dtype.kind in 'fc')):
                different.append(k)
        rr={'run':directory.name,'config':c,'workers':workers,'repeat':rep,
            'status':res['status'],'control_samples':len(timing),'active_samples':int(active.sum()),
            'control_p99_ms':1000*all_s['p99'],'active_control_p99_ms':1000*active_s['p99'],
            'control_max_ms':1000*all_s['max'],'control_p95_ms':1000*all_s['p95'],
            'active_control_max_ms':1000*active_s['max'],
            'control_misses':int(arr['control_deadline_miss'].sum()),
            'planning_misses':sum(bool(j['deadline_missed']) for j in res['planning_jobs']),
            'loop_max_ms':float(arr['loop_cost'].max()*1000),
            'tick_lateness_max_ms':float(arr['tick_lateness'].max()*1000),
            'lateral_fallback_samples':int(arr['lateral_fallback'].sum()),
            'replay_equal':not different,'different_arrays':','.join(different),
            'result_file':(directory/'result.json').relative_to(ROOT).as_posix(),
            'npz_sha256':inputs[(directory/'arrays.npz').relative_to(ROOT).as_posix()]}
        runs.append(rr)
        check(directory.name+'.planning_misses',rr['planning_misses'],res['planning_deadline_misses'])
        if rr['planning_misses']==0 and rr['control_misses']==0:
            check(directory.name+'.no_timeout_replay', different, [])
        saved = next(x for x in summaries[workers]['configs'][c]['rows'] if x['repeat']==rep)
        check(directory.name+'.saved_replay_difference', different, saved['nontiming_arrays_differing_from_replay'])
        for k in range(len(timing)):
            samples.append({'run':directory.name,'config':c,'workers':workers,'repeat':rep,'step':k,
                            'sim_time_s':float(times[k]),'control_ms':float(timing[k]*1000),
                            'active':bool(active[k]),'deadline_missed':bool(arr['control_deadline_miss'][k]),
                            'loop_ms':float(arr['loop_cost'][k]*1000),'tick_lateness_ms':float(arr['tick_lateness'][k]*1000)})
        for index,j in enumerate(res['planning_jobs']):
            # Worker completion and main-process observation are different moments.
            response=j['worker_finished_wall_s']-j['dispatch_wall_s']
            observed=j['result_observed_wall_s']-j['dispatch_wall_s']
            queue=j['worker_started_wall_s']-j['dispatch_wall_s']
            compute=j['worker_finished_wall_s']-j['worker_started_wall_s']
            missed=(j['worker_finished_wall_s']>j['deadline_s']+1e-10 or
                    j['result_observed_wall_s']>j['activation_wall_s'] or j['injected_drop'])
            check(directory.name+f'.job{index}.response',response,j['response_seconds'],1e-10)
            check(directory.name+f'.job{index}.compute',compute,j['compute_seconds'],1e-10)
            check(directory.name+f'.job{index}.queue',queue,j['queue_wait_s'],1e-10)
            check(directory.name+f'.job{index}.observed',observed,j['result_observed_latency_s'],1e-10)
            check(directory.name+f'.job{index}.deadline',missed,j['deadline_missed'])
            jobs.append({'run':directory.name,'config':c,'workers':workers,'repeat':rep,'id':j['id'],
                'stage':'first' if index==0 else 'subsequent','kind':j['kind'],'n_candidates':len(j['search']),
                'nonempty':bool(j['search']),'compute_s':j['compute_seconds'],'response_s':response,
                'observed_latency_s':observed,'queue_s':queue,
                'polling_and_delivery_s':j['result_observed_wall_s']-j['worker_finished_wall_s'],
                'dispatch_lateness_s':j['dispatch_wall_s']-j['dispatch_s'],
                'finish_from_scheduled_dispatch_s':j['worker_finished_wall_s']-j['dispatch_s'],
                'observed_from_scheduled_dispatch_s':j['result_observed_wall_s']-j['dispatch_s'],
                'nominal_budget_s':j['deadline_s']-j['dispatch_s'],
                'actual_dispatch_to_deadline_s':j['deadline_s']-j['dispatch_wall_s'],
                'completion_margin_s':j['deadline_s']-j['worker_finished_wall_s'],
                'observation_to_activation_margin_s':j['activation_wall_s']-j['result_observed_wall_s'],
                'deadline_missed':bool(j['deadline_missed']),'accepted':bool(j['accepted']),
                'rejection_reason':j['rejection_reason']})
    for workers in (1,16):
        for c in ('T1','T2','T3','T4'):
            jj=[j for j in jobs if j['workers']==workers and j['config']==c]
            rr=[r for r in runs if r['workers']==workers and r['config']==c]
            ss=[s for s in samples if s['workers']==workers and s['config']==c]
            comp=stats([j['compute_s'] for j in jj]); saved=summaries[workers]['configs'][c]
            row={'config':c,'workers':workers,'runs':len(rr),'jobs':len(jj),
                'nonempty_jobs':sum(j['nonempty'] for j in jj),'empty_jobs':sum(not j['nonempty'] for j in jj),
                'planning_misses':sum(r['planning_misses'] for r in rr),
                'control_misses':sum(r['control_misses'] for r in rr),
                'replay_equal_runs':sum(r['replay_equal'] for r in rr),
                'all_jobs_compute_s':comp,'first_compute_s':stats([j['compute_s'] for j in jj if j['stage']=='first']),
                'first_response_s':stats([j['response_s'] for j in jj if j['stage']=='first']),
                'first_observed_latency_s':stats([j['observed_latency_s'] for j in jj if j['stage']=='first']),
                'first_finish_from_schedule_s':stats([j['finish_from_scheduled_dispatch_s'] for j in jj if j['stage']=='first']),
                'subsequent_nonempty_compute_s':stats([j['compute_s'] for j in jj if j['stage']=='subsequent' and j['nonempty']]),
                'control_all_pooled_ms':stats([s['control_ms'] for s in ss]),
                'control_active_pooled_ms':stats([s['control_ms'] for s in ss if s['active']]),
                'max_per_run_control_p99_ms':max(r['control_p99_ms'] for r in rr),
                'max_per_run_active_control_p99_ms':max(r['active_control_p99_ms'] for r in rr),
                'min_completion_margin_s':min(j['completion_margin_s'] for j in jj),
                'max_loop_ms':max(r['loop_max_ms'] for r in rr)}
            configurations.append(row)
            for stat in ('max','p99','median'):check(f'{c}.{workers}w.saved_compute_{stat}',comp[stat],saved['compute_s'][stat])
            for label,key in [('planning_misses','planning_deadline_misses_total'),('control_misses','control_deadline_misses_total'),
                              ('replay_equal_runs','runs_equal_to_replay'),('jobs','jobs')]:
                check(f'{c}.{workers}w.saved_{label}',row[label],saved[key])
            check(f'{c}.{workers}w.saved_control_p99',row['max_per_run_control_p99_ms']/1000,saved['control_p99_s_max'])
    totals={}
    for workers in (1,16):
        jj=[j for j in jobs if j['workers']==workers];rr=[r for r in runs if r['workers']==workers];ss=[s for s in samples if s['workers']==workers]
        totals[str(workers)]={'runs':len(rr),'jobs':len(jj),'nonempty_jobs':sum(j['nonempty'] for j in jj),
          'control_samples':len(ss),'active_control_samples':sum(s['active'] for s in ss),
          'control_misses':sum(r['control_misses'] for r in rr),'planning_misses':sum(r['planning_misses'] for r in rr),
          'replay_equal_runs':sum(r['replay_equal'] for r in rr),
          'control_all_pooled_ms':stats([s['control_ms'] for s in ss]),'control_active_pooled_ms':stats([s['control_ms'] for s in ss if s['active']]),
          'max_per_run_control_p99_ms':max(r['control_p99_ms'] for r in rr),
          'max_per_run_active_control_p99_ms':max(r['active_control_p99_ms'] for r in rr),
          'first_compute_s':stats([j['compute_s'] for j in jj if j['stage']=='first']),
          'first_response_s':stats([j['response_s'] for j in jj if j['stage']=='first']),
          'first_finish_from_schedule_s':stats([j['finish_from_scheduled_dispatch_s'] for j in jj if j['stage']=='first']),
          'subsequent_nonempty_compute_s':stats([j['compute_s'] for j in jj if j['stage']=='subsequent' and j['nonempty']]),
          'min_completion_margin_s':min(j['completion_margin_s'] for j in jj)}
    save(OUT/'recomputed_summary.json',{'environments_as_recorded':{str(w):summaries[w]['env'] for w in (1,16)},
         'recomputation_environment':{'python':platform.python_version(),'numpy':np.__version__},
         'configuration':configurations,'totals':totals,'excluded_arrays_for_replay':sorted(EXCLUDED),
         'quantile_method':'numpy linear; pooled samples and max of per-run percentiles reported separately',
         'active_definition':'active_control == bool(any(mode)); includes completed cut-ins still under lateral control',
         'control_deadline_definition':'Recorded wall-clock flag checked after control; not inferred solely from timing > 0.1 s',
         'planning_deadline_definition':'Worker ready by nominal activation plus result received at the activation check',
         'new_vehicle_simulations':0})
    csvsave(OUT/'planning_jobs.csv',jobs);csvsave(OUT/'control_steps.csv',samples);csvsave(OUT/'run_summary.csv',runs)
    csvsave(OUT/'configuration_summary.csv',[{**{k:v for k,v in row.items() if not isinstance(v,dict)},
        **{k+'_'+sk:sv for k,v in row.items() if isinstance(v,dict) for sk,sv in v.items()}} for row in configurations])
    csvsave(OUT/'deadline_misses.csv',[j for j in jobs if j['deadline_missed']])
    save(OUT/'checks.json',{'passed':all(x['passed'] for x in checks),'n_checks':len(checks),'checks':checks})
    save(OUT/'input_hashes.json',inputs)
    print('Timing checks:',len(checks),'passed:',all(x['passed'] for x in checks))
    print('TOTALS',totals)
    bad=[x for x in checks if not x['passed']]
    if bad: print('MISMATCHES',bad); sys.exit(1)


if __name__=='__main__':main()
