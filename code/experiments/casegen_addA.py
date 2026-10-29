"""WP5-v20 addendum A case generator: EX9 (WP8-v20 lateral mismatch), EX10 (Q/N-v20 scale and mechanism),
EX11 (WP2-v20 allocation ablation).  Uses casegen.make() for the base configuration so that every case not
touched by an adapter is byte-identical to the corresponding v1 configuration (and shares its run key)."""
import json
from pathlib import Path

import casegen as CG

HERE = Path(__file__).resolve().parent
LAT_CORNERS = {'C1': [.7, 1.3, 1.2, 1.3], 'C2': [1.3, .7, 1.2, 1.3], 'C3': [.7, .7, 1.2, 1.3], 'C4': [1.3, 1.3, .8, .7]}
QN_FAMILIES = ("supplied_synthetic", "bounded_speed_synthetic", "stage1_template")
QN_GEOMS = [(f"Q{q}", [8] + [7] * (q - 1)) for q in (2, 3, 4, 5, 6)] + \
           [(f"N_{tag}", [round(v * m) for v in (8, 7, 7)]) for m, tag in ((1.5, "x1p5"), (2., "x2"), (3., "x3"))]


def fmt(x):
    return ('%g' % x).replace('.', 'p').replace('-', 'm')


def ex9():
    """145 cases, grid identical to the v19 WP8 design."""
    rows = [dict(id='P0', scales=[1., 1., 1., 1.], mild=False)]
    for axis, indices, values in [('Pf', [0], [.7, .85, 1.15, 1.3]), ('Pr', [1], [.7, .85, 1.15, 1.3]),
                                  ('Pfr', [0, 1], [.7, .85, 1.15, 1.3]), ('Pm', [2], [.8, .9, 1.1, 1.2]),
                                  ('Pi', [3], [.7, .85, 1.15, 1.3])]:
        for s in values:
            scales = [1., 1., 1., 1.]
            for i in indices:
                scales[i] = s
            rows.append(dict(id=f'{axis}_{s}'.replace('.', 'p'), scales=scales, mild=s in [.85, 1.15, .9, 1.1]))
    for label, scales in LAT_CORNERS.items():
        rows.append(dict(id=label, scales=scales, mild=False))
    for channel, amplitudes in [('Fy', [.02 * 1500 * 9.81, .05 * 1500 * 9.81, .10 * 1500 * 9.81]), ('Mz', [200., 500., 1000.])]:
        for level, amp in enumerate(amplitudes, 1):
            for sign in [-1, 1]:
                rows.append(dict(id=f'{channel}_level{level}_{"neg" if sign < 0 else "pos"}', scales=[1., 1., 1., 1.],
                                 pulse=[amp * sign, 0.] if channel == 'Fy' else [0., amp * sign], mild=False))
    for label, scales in LAT_CORNERS.items():
        for sign in [-1, 1]:
            rows.append(dict(id=f'{label}_pulse_{"neg" if sign < 0 else "pos"}', scales=scales,
                             pulse=[sign * .05 * 1500 * 9.81, sign * 500.], mild=False))
    for seed in range(1, 51):
        for label, scales in [('P0', [1., 1., 1., 1.]), ('C3', LAT_CORNERS['C3'])]:
            rows.append(dict(id=f'{label}_seed{seed:03d}', scales=scales, lateral_seed=seed, mild=False))
    assert len(rows) == 145 and sum(r['mild'] for r in rows) == 10
    out = []
    for r in rows:
        spec = {'scales': r['scales'], 'pulse': r.get('pulse', [0., 0.]), 'lateral_seed': r.get('lateral_seed')}
        c = CG.make('EX9', 'A-H', 'nominal', variant=f"L_{r['id']}", adapters={'AD5': spec},
                    note='mild' if r['mild'] else None)
        out.append(c)
    return out


def ex10():
    """Q/N-v20: 3 families x 8 geometries at 30 m; LP-infeasible geometries also at the smallest LP-restoring spacing
    b* and at b*+4 (from prescreen_qn.json, 2 m grid)."""
    pre = {(r['family'], r['geometry']): r for r in json.loads((HERE / 'prescreen_qn.json').read_text(encoding='utf-8'))['rows']}
    out = []
    for fam in QN_FAMILIES:
        for label, sizes in QN_GEOMS:
            p = pre[(fam, label)]
            bs = [30.]
            if not p['lp_feasible_b30']:
                bs += [float(p['min_boundary_lp_feasible']), float(p['min_boundary_lp_feasible']) + 4.]
            for b in bs:
                spec = {'family': fam, 'queue_sizes': sizes, 'boundary': b}
                c = CG.make('EX10', 'A-H', 'nominal', boundary=b, variant=f'{fam}_{label}_b{fmt(b)}', adapters={'AD6': spec},
                            note=f"predicted_lp_feasible={p['lp_feasible_b30'] if b == 30. else True}")
                c['cell']['geometry'] = label; c['cell']['family_scene'] = fam; c['cell']['queue_sizes'] = sizes
                out.append(c)
    return out


def ex11():
    """WP2-v20: joint (frozen) vs sequential vs independent allocation; grid as the v19 WP2 study."""
    out = []
    for b in (26, 28, 30, 32, 34, 36):
        for corner in ('nominal', 'm1.1'):
            for mode in ('joint', 'sequential', 'independent'):
                ad = {} if mode == 'joint' else {'AD7': mode}
                c = CG.make('EX11', 'A-H', corner, boundary=b, variant=f'{mode}_Q3', adapters=ad)
                c['cell']['mode'] = mode; out.append(c)
    for b in (30, 34):
        for mode in ('joint', 'sequential', 'independent'):
            ad = {'AD6': {'family': 'stage1_template', 'queue_sizes': [8, 7, 7, 7], 'boundary': float(b)}}
            if mode != 'joint':
                ad['AD7'] = mode
            c = CG.make('EX11', 'A-H', 'nominal', boundary=b, variant=f'{mode}_Q4', adapters=ad)
            c['cell']['mode'] = mode; out.append(c)
    for seed in range(1, 51):
        for mode in ('joint', 'sequential', 'independent'):
            ad = {} if mode == 'joint' else {'AD7': mode}
            c = CG.make('EX11', 'A-H', 'nominal', boundary=30, seed=seed, variant=f'{mode}_Q3', adapters=ad)
            c['cell']['mode'] = mode; out.append(c)
    assert len(out) == 192
    return out


def all_cases():
    cases = ex9() + ex10() + ex11()
    ids = [c['id'] for c in cases]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise RuntimeError(f'duplicate ids: {sorted(dup)[:5]}')
    return cases
