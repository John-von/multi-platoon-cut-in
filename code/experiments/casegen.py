"""WP5-v20 case generator (registered list).  Pure Python + NumPy; no simulation code is imported.

Every case: {id, family, cell, method, runner, workers, adapters, theory_covered, cfg, cfg_sha256}.
cfg_sha256 = SHA-256 of json.dumps(cfg, sort_keys=True, separators=(',', ':')).  Identical cfg+runner+adapters
are run once (run key = SHA-256 of the canonical {runner, adapters, workers, cfg}); cases keep their own ids."""
import hashlib, json
import numpy as np

EPS = [1e-3, 5e-5, .00015 / 1.89010989]            # ell = 1: 1 mm, 5e-5 m/s, 7.936e-5 (normalized force state)
MERGE_BASE = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 34]
V20_MAIN = dict(enabled=True, lateral_soft=True, condition_s=True, jerk_mode='window', jerk_window=10,
                replan_mode='handover', activation_latency=[30, 20], containment_gate=False, init_refine=True)
CORNERS = {'nominal': (1.0, 1.0), 'm0.9': (0.9, 1.0), 'm1.1': (1.1, 1.0), 't0.7': (1.0, 0.7), 't1.3': (1.0, 1.3),
           'low': (0.9, 0.7), 'high': (1.1, 1.3)}
HETERO = [1801, 1802, 1803, 1804]
M_VEH = 25                                           # 22 ordinary (8/7/7) + 3 cut-ins, simulation index order
SYNC = {'A-S': 'adaptive', 'F': 'fixed', 'TUBE': 'tube', 'DDRPC': 'ddrpc', 'N': 'nominal', 'P': 'point'}


def hetero(seed):
    rng = np.random.default_rng(seed)
    mass = rng.uniform(0.9, 1.1, M_VEH); tau = rng.uniform(0.7, 1.3, M_VEH)
    return [float(x) for x in mass], [float(x) for x in tau]


def params(corner):
    if corner.startswith('H'):
        return hetero(int(corner[1:]))
    return CORNERS[corner]


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'))


def sha(x):
    return hashlib.sha256(canon(x).encode()).hexdigest()


def fmt(x):
    return ('%g' % x).replace('.', 'p')


def make(family, method, corner='nominal', s=1, ell=0, f=0.5, gap=10.5, boundary=30, seed=None, variant='V0',
         v20_over=None, extra=None, adapters=None, covered=None, note=None):
    mass, tau = params(corner)
    v20 = dict(V20_MAIN, meas_eps=[ell * e for e in EPS])
    if method != 'A-H':
        v20['replan_mode'] = 'sync'
    if v20_over:
        v20.update(v20_over)
    cfg = {'time_sim': 50, 'request_time': 5.0, 'max_attempts': 3, 'acc_bound': .001 * s, 'torque_bound': .00015 * s,
           'disturbance_fraction': f, 'initial_gap': gap, 'boundary': boundary, 'mass_scale': mass, 'tau_scale': tau,
           'durations': list(MERGE_BASE), 'v20': v20}
    if method == 'A-H':
        cfg.update(execution='replay', delay_schedule_s=[0.0] * 10)
        runner = 'handover'
    else:
        cfg.update(method=SYNC[method], replan_until_admission=True, plan_acc_bound=.001 * s,
                   plan_torque_bound=.00015 * s)
        runner = 'sync'
    if seed is not None:
        cfg['seed'] = int(seed)
    if extra:
        cfg.update(extra)
    adapters = adapters or {}
    workers = 1   # serial capacity path (bitwise equal to the parallel patch path; 2 cases run concurrently)
    cell = dict(corner=corner, s=s, ell=ell, f=f, gap=gap, boundary=boundary, seed=seed, variant=variant)
    cid = (f'{family}__{method}__{corner}__s{fmt(s)}_l{fmt(ell)}_f{fmt(f)}__g{fmt(gap)}_b{fmt(boundary)}__{variant}'
           + ('' if seed is None else f'__seed{seed:03d}'))
    if covered is None:
        covered = method in ('A-H', 'A-S', 'F', 'TUBE', 'DDRPC') and f <= 1.0
    return {'id': cid, 'family': family, 'cell': cell, 'method': method, 'runner': runner, 'workers': workers,
            'adapters': adapters, 'theory_covered': bool(covered), 'note': note, 'cfg': cfg, 'cfg_sha256': sha(cfg),
            'run_key': sha({'runner': runner, 'adapters': adapters, 'workers': workers, 'cfg': cfg})}


def ex1():
    out = []
    conds = ['nominal', 'm0.9', 'm1.1', 't0.7', 't1.3', 'low', 'high'] + [f'H{h}' for h in HETERO]
    for c in conds:
        for m in ['A-H', 'A-S', 'F', 'DDRPC', 'N', 'P']:
            out.append(make('EX1a', m, c))
    out.append(make('EX1a', 'TUBE', 'nominal', note='equivalence record: TUBE and F share one code path'))
    for c in conds:
        for m in ['A-H', 'A-S', 'F', 'N', 'P']:
            out.append(make('EX1b', m, c, ell=5))
    for c in ['nominal', 'low', 'high']:
        for m in ['A-H', 'A-S', 'F', 'DDRPC']:
            for sd in range(1, 51):
                out.append(make('EX1c', m, c, seed=sd))
        for sd in range(1, 51):
            out.append(make('EX1d', 'A-H', c, ell=5, seed=sd))
    return out


def ex2():
    out = []
    for s in [1, 3, 5, 7, 10]:
        for ell in [0, 2, 5, 7, 10]:
            for c in ['nominal', 'high']:
                out.append(make('EX2a', 'A-H', c, s=s, ell=ell))
    for (s, ell) in [(1, 0), (1, 10), (3, 10), (5, 7), (7, 5), (10, 0)]:
        for c in ['nominal', 'high']:
            for sd in range(1, 11):
                out.append(make('EX2b', 'A-H', c, s=s, ell=ell, seed=sd))
    for s in [1, 3, 5, 7, 10]:
        for c in ['low', 'nominal', 'high']:
            for m in ['A-S', 'F', 'DDRPC', 'N', 'P']:
                out.append(make('EX2c', m, c, s=s))
            if c == 'low':
                out.append(make('EX2c', 'A-H', c, s=s))
    for gap in [11, 12]:
        for s in [15, 20]:
            for c in ['nominal', 'high']:
                out.append(make('EX2d', 'A-H', c, s=s, gap=gap))
    return out


def ex3():
    out = []
    for f in [1.0, 1.5, 2, 3, 5, 10]:
        for c in ['nominal', 'high']:
            out.append(make('EX3', 'A-H', c, f=f, adapters={'AD1': True}))
    for f in [1.5, 3, 10]:
        for c in ['nominal', 'high']:
            for sd in range(1, 6):
                out.append(make('EX3', 'A-H', c, f=f, seed=sd, adapters={'AD1': True}))
    return out


def ex4():
    out = []
    scenes = [(10.2, 30), (10.25, 30), (10.3, 30), (10.5, 30), (10.5, 26), (10.5, 24), (10.5, 18)]
    for gap, b in scenes:
        for c in ['low', 'nominal', 'high']:
            for f in [0.5, 1.0]:
                for m in ['A-H', 'A-S', 'F', 'DDRPC', 'N', 'P']:
                    out.append(make('EX4', m, c, f=f, gap=gap, boundary=b))
    return out


EX5_CELLS = {'C1': dict(), 'C2': dict(boundary=26), 'C3': dict(gap=10.3), 'C4': dict(corner='high'),
             'C5': dict(s=3), 'C6': dict(ell=5), 'C7': dict(s=5, ell=5), 'C8': dict(s=10),
             'C9': dict(extra={'cruise_cycle': True}), 'C10': dict(corner='high', extra={'cruise_cycle': True})}


def ex5():
    out = []
    switch = {'V0': {}, 'V1': {'jerk_mode': 'sampled', 'jerk_window': None}, 'V2': {'init_refine': False},
              'V3': {'condition_s': False}, 'V4': {'containment_gate': True}, 'V5': {'lateral_soft': False}}
    for name, kw in EX5_CELLS.items():
        for f in [0.5, 1.0]:
            base = dict(kw); corner = base.pop('corner', 'nominal')
            for v, over in switch.items():
                cov = None
                if v == 'V2':
                    cov = base.get('ell', 0) == 0
                if v == 'V3':
                    cov = False
                out.append(make('EX5', 'A-H', corner, f=f, variant=f'{name}_{v}', v20_over=over, covered=cov, **base))
            for v in ['V7', 'V8', 'V9']:
                out.append(make('EX5', 'A-H', corner, f=f, variant=f'{name}_{v}', adapters={'AD4': v},
                                covered=(v == 'V7'), **base))
            if f == 0.5:
                for lat in ([20, 20], [40, 30], [60, 40]):
                    out.append(make('EX5', 'A-H', corner, f=f, variant=f'{name}_V6L{lat[0]}_{lat[1]}',
                                    v20_over={'activation_latency': lat}, **base))
    return out


def ex6():
    out = []
    scen = {'S1': dict(), 'S2': dict(boundary=26), 'S3': dict(corner='high'), 'S4': dict(s=10), 'S5': dict(s=5, ell=7),
            'S6': dict(s=10, ell=5), 'S7': dict(gap=10.2)}
    variants = {'base': dict(),
                'merge1s': dict(extra={'durations': list(range(12, 35))}),
                'merge4s': dict(extra={'durations': [12, 16, 20, 24, 28, 32]}),
                'mergeExt': dict(extra={'durations': MERGE_BASE + [40, 44, 48, 52, 56, 60]}),
                'b8': dict(adapters={'AD2': {'b': 8, 'limit': 40.0}}),
                'b18': dict(adapters={'AD2': {'b': 18, 'limit': 40.0}}),
                'dmax60': dict(adapters={'AD2': {'b': 13, 'limit': 60.0}}),
                'initExt': dict(adapters={'AD3': True},
                                extra={'initial_durations': [12, 20, 26, 30, 36, 40, 44, 48, 52, 56, 60]})}
    for name, kw in scen.items():
        for v, vk in variants.items():
            base = dict(kw); corner = base.pop('corner', 'nominal')
            out.append(make('EX6', 'A-H', corner, variant=f'{name}_{v}', **base, **vk))
    return out


def ex7():
    out = []
    conds = {'K1': dict(), 'K2': dict(ell=5), 'K3': dict(s=5, ell=5, corner='high')}
    cut = [False] * 22 + [True] * 3
    variants = {'base': {}, 'age5': {'uplink_age_steps': 5}, 'age10': {'uplink_age_steps': 10},
                'age20': {'uplink_age_steps': 20}, 'age30': {'uplink_age_steps': 30},
                'missAll': {'uplink_missing': True}, 'missCut': {'uplink_missing': cut},
                'drop0': {'drop_jobs': [0]}, 'drop01': {'drop_jobs': [0, 1]}, 'drop012': {'drop_jobs': [0, 1, 2]},
                'lateFirst': {'delay_schedule_s': [3.5] + [0.0] * 9}, 'lateAll': {'delay_schedule_s': [3.5] * 10},
                'ontimeFirst': {'delay_schedule_s': [2.9] + [0.0] * 9}}
    for name, kw in conds.items():
        for v, extra in variants.items():
            base = dict(kw); corner = base.pop('corner', 'nominal')
            out.append(make('EX7', 'A-H', corner, variant=f'{name}_{v}', extra=extra, **base))
    return out


def ex8():
    out = []
    for c in ['nominal', 'high']:
        for m in ['A-H', 'A-S', 'F']:
            out.append(make('EX8a', m, c, extra={'cruise_cycle': True}))
        for m in ['A-S', 'F', 'DDRPC', 'N', 'P']:
            out.append(make('EX8b', m, c, extra={'overlap_cruise': True, 'initial_constant_backup': True},
                            note='overlap_cruise is implemented only in the sync runner'))
    return out


def all_cases():
    cases = ex1() + ex2() + ex3() + ex4() + ex5() + ex6() + ex7() + ex8()
    ids = [c['id'] for c in cases]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise RuntimeError(f'duplicate ids: {sorted(dup)[:5]}')
    return cases
