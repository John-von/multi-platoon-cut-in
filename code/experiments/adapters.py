"""WP5-v20 test adapters (process-local; the frozen files on disk are never modified).

AD1 recorder     observation-only wrappers around async_execution.plant_step / update_set_measured / update_set.
                 A run that stops on an empty identification set (frozen update_set raises ValueError) then still
                 leaves the executed state/input prefix.  Nothing returned to the simulation is altered.
AD2 capacity     capacity() with b bisection steps and displacement limit Dmax (frozen: b = 13, Dmax = 40 m).
AD3 initial list run_async() whose initial (cruise) candidate list is read from cfg['initial_durations']
                 (frozen literal [12,20,26,30,36]; the key is absent in every other family).
AD4 mechanisms   V7 ellipsoid-only envelope (box = EXT*radius, initial box ignored);
                 V8 current-gap admission (gate = first step whose current two-sided lower gap >= 10.05 m, no
                    suffix requirement, no gate terminal check);
                 V9 no terminal checks (ordinary terminal gap, condition-S terminal gap, gate terminal gap and the
                    certificate's 'terminal' residual are not required; window-jerk terms that cross into the
                    terminal continuation are kept, they belong to the M4 window definition).

Every source-level change is a verbatim substitution applied to the frozen function source (inspect.getsource);
each substitution must match exactly once (asserted) and the unified diff is returned for the run record.  The
recompiled function executes in the frozen module's own globals.  With an empty substitution list the recompiled
function is the frozen one; zero_intervention.py checks this bitwise.  All adapter runs use V20_PLAN_WORKERS=1
(the per-vehicle capacity pool would not see a patched capacity/certificate under the spawn start method)."""
import difflib
import inspect

import numpy as np

REGISTRY = {}   # name -> unified diff (for the run record)


def recompile(module, func_name, subs):
    src = inspect.getsource(getattr(module, func_name))
    new = src
    for old, rep in subs:
        n = new.count(old)
        if n != 1:
            raise RuntimeError(f'{module.__name__}.{func_name}: substitution matched {n} times: {old!r}')
        new = new.replace(old, rep)
    ns = {}
    exec(compile(new, f'<adapter {module.__name__}.{func_name}>', 'exec'), module.__dict__, ns)
    diff = ''.join(difflib.unified_diff(src.splitlines(True), new.splitlines(True),
                                        f'frozen/{module.__name__}.{func_name}', f'adapter/{module.__name__}.{func_name}'))
    REGISTRY[f'{module.__name__}.{func_name}'] = diff
    return ns[func_name]


# ---------------------------------------------------------------- AD1
class Recorder:
    """Keeps (step, vehicle, x, u, theta, da, dw, x_next) for every plant_step call and the first set-update
    exception.  M plant_step calls make one sampling step (the frozen loops call it for i in range(M))."""

    def __init__(self, M):
        self.M = M; self.calls = 0; self.rows = []; self.set_calls = 0; self.exception = None

    def install(self, A):
        orig_plant, orig_usm, orig_us = A.plant_step, A.update_set_measured, A.update_set
        rec = self

        def plant_step(x, u, theta, da=0., dw=0.):
            xn = orig_plant(x, u, theta, da, dw)
            rec.rows.append((rec.calls // rec.M, rec.calls % rec.M, np.array(x, float), float(u),
                             np.array(theta, float), float(da), float(dw), np.array(xn, float)))
            rec.calls += 1
            return xn

        def wrap(fn, name):
            def inner(*a, **k):
                i = rec.set_calls; rec.set_calls += 1
                try:
                    return fn(*a, **k)
                except ValueError as e:
                    if rec.exception is None:
                        rec.exception = {'function': name, 'set_call': i, 'step': i // rec.M, 'vehicle': i % rec.M,
                                         'message': str(e)}
                    raise
            return inner

        A.plant_step = plant_step
        A.update_set_measured = wrap(orig_usm, 'update_set_measured')
        A.update_set = wrap(orig_us, 'update_set')
        return self

    def prefix(self):
        """Arrays of the executed prefix: state[k] for k = 0..K (K complete steps), input[k], acceleration[k]."""
        if not self.rows:
            return {}
        counts = {}
        for r in self.rows:
            counts[r[0]] = counts.get(r[0], 0) + 1
        K = 0
        while counts.get(K, 0) == self.M:
            K += 1
        X = np.zeros((K + 1, self.M, 3)); U = np.zeros((K, self.M))
        for (k, i, x, u, th, da, dw, xn) in self.rows:
            if k < K:
                X[k, i] = x; U[k, i] = u; X[k + 1, i] = xn
        acc = (X[1:, :, 1] - X[:-1, :, 1]) / 0.1
        return {'state': X, 'input': U, 'acceleration': acc, 'complete_steps': K}


# ---------------------------------------------------------------- AD2
def install_capacity(R, S, b=13, limit=40.):
    subs = []
    if b != 13:
        subs.append(('for it in range(13):', f'for it in range({int(b)}):'))
    if limit != 40.:
        subs.append(('limit=40.,', f'limit={float(limit)!r},'))
    fn = recompile(R, 'capacity', subs)
    R.capacity = fn; S.capacity = fn
    return fn


# ---------------------------------------------------------------- AD3
def load_run_async_initial_list(A, identity=False):
    subs = [] if identity else [('    for T in [12,20,26,30,36]:\n',
                                 "    for T in cfg.get('initial_durations',[12,20,26,30,36]):\n")]
    return recompile(A, 'run_async', subs)


# ---------------------------------------------------------------- AD4
V7_SUBS = [('    box[0]=EXT*initial_radius if initial_box is None else np.minimum(initial_box,EXT*initial_radius)\n',
            '    box[0]=EXT*initial_radius\n'),
           ('        box[k+1]=np.minimum(Aabs@box[k]+[0,wv[k],wt[k]],EXT*rad[k+1])\n',
            '        box[k+1]=EXT*rad[k+1]\n')]
V8_SUBS = [("            if min(tf,tr)<10-1e-7:return None,'terminal_merge_gap'\n", ''),
           ('            suffix=np.minimum.accumulate(worst[::-1])[::-1]\n', '            suffix=worst\n')]
V9_MAKEPLAN_SUBS = [("    if min(gaps.min(),terminal_gaps.min())<10-1e-7:return None,'ordinary_gap_tube'\n",
                     "    if gaps.min()<10-1e-7:return None,'ordinary_gap_tube'\n"),
                    ("        if tsg.size and tsg.min()<10-1e-7:return None,'source_terminal_gap'\n", '        pass\n'),
                    ("            if min(tf,tr)<10-1e-7:return None,'terminal_merge_gap'\n", '')]
V9_CERT_SUBS = [("    out['pass']=bool(max(out['residuals'].values())<=1e-7)\n    return out\n",
                 "    out['pass']=bool(max(v_ for k_,v_ in out['residuals'].items() if k_!='terminal')<=1e-7)\n    return out\n")]


def install_mechanism(R, S, A, variant, identity=False):
    if variant == 'V7':
        R._certificate_v19 = recompile(R, '_certificate_v19', [] if identity else V7_SUBS)
    elif variant == 'V8':
        fn = recompile(S, 'make_plan', [] if identity else V8_SUBS)
        S.make_plan = fn; A.make_plan = fn
    elif variant == 'V9':
        fn = recompile(S, 'make_plan', [] if identity else V9_MAKEPLAN_SUBS)
        S.make_plan = fn; A.make_plan = fn
        cert = recompile(R, 'certificate', [] if identity else V9_CERT_SUBS)
        R.certificate = cert; S.certificate = cert; A.certificate = cert
    else:
        raise ValueError(variant)
