"""Load registered analysis outputs and extracted trajectories from local simulation runs.
Verification rows use one registered identity per run key; aliases are resolved through that key.
"""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # package root
REG = ROOT / 'code' / 'experiments'
ORIG = ROOT / 'data' / 'recomputed'
EXTRACTED = ROOT / 'data' / 'extracted'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


class Inputs:
    def __init__(self, with_d=False, d_dir=None):
        self.files = {}
        self.cases = []
        for f in ('cases_v1.json', 'cases_addA_v1.json'):
            p = REG / f
            self.cases += load(p)['cases']; self.files[f'registry/{f}'] = sha(p)
        self.by_id = {}
        self.by_key = {}
        for c in self.cases:
            self.by_id.setdefault(c['id'], c)
            self.by_key.setdefault(c['run_key'], []).append(c)
        p = ORIG / 'verify' / 'verify_raw_all.json'
        rows = load(p)['rows']; self.files['recomputed/verify_raw_all.json'] = sha(p)
        self.verify = {}                                   # run_key -> verification row
        for r in rows:
            self.verify[self.by_id[r['id']]['run_key']] = r
        p = ORIG / 'analysis' / 'hypotheses.json'; self.hyp = load(p); self.files['recomputed/hypotheses.json'] = sha(p)
        p = ORIG / 'analysis_addA' / 'hypotheses_addA.json'; self.hypA = load(p); self.files['recomputed/hypotheses_addA.json'] = sha(p)
        self.d_rows = {}; self.d_traj = {}
        if with_d:
            d = Path(d_dir) if d_dir else EXTRACTED
            p = d / 'all_runs.json'
            if p.exists():
                self.files['recomputed/all_runs.json'] = sha(p)
                for r in load(p)['rows']:
                    self.d_rows[r['run_key']] = r
            p = d / 'trajectories.json'
            if p.exists():
                self.files['recomputed/trajectories.json'] = sha(p); self.d_traj = load(p)

    def family(self, fam, **cell):
        out = []
        for c in self.cases:
            if c['family'] != fam:
                continue
            if all(c['cell'].get(k) == v for k, v in cell.items()):
                out.append(c)
        return out

    def labels(self, c):
        """Audit labels of the run of case c (through its run key)."""
        v = self.verify.get(c['run_key'])
        return None if v is None else v['labels']

    def jv(self, c):
        v = self.verify.get(c['run_key'])
        return None if v is None else v.get('Jv_recomputed')

    def d(self, c):
        return self.d_rows.get(c['run_key'])
