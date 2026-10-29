"""Diagnostic of the EX9 runs (lateral plant mismatch, A-H): lateral behaviour of the cut-ins after completion.
Reads <runs_root>/<key>/{arrays.npz,result.json,audit.json} (default: data/recomputed/runs); read-only.
Usage: python ex9_post.py <keys.json> <start> <stop> <out.jsonl> [<runs_root>]
keys.json holds [run directory, variant] pairs (data/lateral_diag/ex9_keys.json); entries start..stop-1 are processed.
One JSON line per run is appended: per cut-in, from its completion time to 50 s, the extreme lateral positions of the
body centre, the largest deviation from the target-lane centre (3.5 m), the largest deviation over 40-50 s, the time
with the centre below the lane line (1.75 m) and the time with the lower body edge below it."""
import json, sys
from pathlib import Path
import numpy as np
keys = json.load(open(sys.argv[1]))[int(sys.argv[2]):int(sys.argv[3])]
R = Path(sys.argv[5]) if len(sys.argv) > 5 else Path(__file__).resolve().parents[2] / 'data' / 'recomputed' / 'runs'
with open(sys.argv[4], 'a') as h:
    for k, var in keys:
        d = R / k
        if not (d / 'audit.json').exists():
            h.write(json.dumps({'key': k, 'variant': var, 'missing': True}) + '\n'); continue
        r = json.loads((d / 'result.json').read_text()); a = json.loads((d / 'audit.json').read_text())
        with np.load(d / 'arrays.npz') as z:
            bp = z['body_poses']; Q = z['mode'].shape[1]; N = z['state'].shape[1] - Q
            y = bp[:, N:, 1]; psi = bp[:, N:, 2]
        t = np.arange(y.shape[0]) * 0.02
        ext = 2.4 * np.abs(np.sin(psi)) + 0.9 * np.abs(np.cos(psi))
        out = {'key': k, 'variant': var, 'labels': a.get('labels'), 'merge_done': r.get('merge_done'), 'per_cut': []}
        for c in range(Q):
            dn = (r.get('merge_done') or [None] * Q)[c]
            if dn is None:
                out['per_cut'].append(None); continue
            m = t >= dn - 1e-9
            dev = np.abs(y[m, c] - 3.5)
            low_edge = (y[m, c] - ext[m, c])
            last = t >= 40 - 1e-9
            first_big = t[m][np.argmax(dev > 0.5)] if (dev > 0.5).any() else None
            out['per_cut'].append({'done': dn, 'y_min_after': float(y[m, c].min()), 'y_max_after': float(y[m, c].max()),
                                   'dev_max_after': float(dev.max()), 'first_dev_over_0p5': None if first_big is None else float(first_big),
                                   'center_in_source_lane_s': float((y[m, c] < 1.75).sum() * 0.02),
                                   'body_edge_below_1p75_s': float((low_edge < 1.75).sum() * 0.02),
                                   'dev_max_last10s': float(np.abs(y[last, c] - 3.5).max())})
        h.write(json.dumps(out) + '\n')
