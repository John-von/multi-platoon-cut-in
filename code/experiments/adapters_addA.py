"""WP5-v20 addendum A test adapters (process-local; frozen files on disk are never modified).

AD5 lateral plant   WP8-v20 (EX9): the ACTUAL lateral plant gets scaled front/rear cornering stiffness, lateral mass and
                    yaw inertia plus an external lateral force Fy and yaw moment Mz, applied only after each cut-in's own
                    lateral activation; the nominal lateral MPC is unchanged.  Class body copied verbatim from the
                    v19 WP8 adapter (used for the v19 WP8 145-run study); installed by
                    rebinding async_execution.lateral_step (A-H runner) and simulation.lateral_step.
AD6 scene           Q/N-v20 (EX10) and the 4-queue WP2 cases (EX11): scene construction copied verbatim from the
                    v19 Q/N study (synthetic_scenario from phaseB_qn_mechanism/sources/simulation.py; construct_scene
                    from run_verified_sweep.py, with the initial queue spacing as an argument as in structural_check.py
                    and ablation_admission.template_scene).  Installed by rebinding simulation.scenario and
                    async_execution.scenario to return the constructed scene.  For stage1_template [8,7,7] it reproduces
                    the paper scene exactly (asserted), for every tested initial queue spacing.
AD7 allocation      WP2-v20 (EX11): 'independent' = frozen robust_core.allocation recompiled with the inter-queue boundary
                    rows removed (verbatim deletion, asserted to match once); 'sequential' = greedy front-to-back rule,
                    copied verbatim from the v19 WP2 adapter (ablation_admission.sequential_allocation).
                    Installed by rebinding simulation.allocation (make_plan looks it up there).
All adapter runs use V20_PLAN_WORKERS=1, except AD6-only cases (EX10-T timing; see run_case_addA.py)."""
import hashlib
import sys

import numpy as np

import adapters as AD   # recompile() and REGISTRY from the v1 adapters


# ---------------------------------------------------------------- AD5 (verbatim v19 class body)
class LateralPlant:
    def __init__(self,case,steps=500,q=3):
        self.case=case;self.steps=steps;self.q=q
        self.full=np.zeros((steps*5+1,q,4));self.slip=np.zeros((steps*5+1,q,2));self.force_sub=np.zeros((steps*5+1,q,2));self.forces=np.zeros((steps,q,2))
        if case.get('seed') is not None:
            for vehicle in range(q):
                for channel,bound in enumerate([.02*1500*9.81,200.]):self.forces[:,vehicle,channel]=np.random.default_rng(np.random.SeedSequence([case['seed'],8,vehicle,channel])).uniform(-bound,bound,steps)
        self.force_hash=hashlib.sha256(np.ascontiguousarray(self.forces).tobytes()).hexdigest();self.calls=0
    def force(self,k,c,local_time):
        if self.case.get('seed') is not None:return self.forces[k,c]
        amp=np.array(self.case.get('pulse',[0.,0.]))
        return amp*np.sin(np.pi*local_time/4) if 0<=local_time<=4 else amp*0
    def step(self,x,v,delta):
        context=sys._getframe(1).f_locals;k=context['k'];c=context['c'];start=context['starts'][c]
        cf,cr,m,iz=np.array([50000.,50000.,1500.,2500.])*np.array(self.case['scales'])
        lf=1.2;lr=1.6;dt=.02;states=[]
        def rhs(s,t):
            _,phi,vy,r=s
            ff=cf*(delta-np.arctan((vy+lf*r)/v));fr=-cr*np.arctan((vy-lr*r)/v)
            fy,mz=self.force(k,c,t-start)
            return np.array([v*np.sin(phi)+vy*np.cos(phi),r,(ff+fr)/m-v*r+fy/m,(lf*ff-lr*fr)/iz+mz/iz])
        for ss in range(5):
            t=k*.1+ss*dt
            k1=rhs(x,t);k2=rhs(x+dt*k1/2,t+dt/2);k3=rhs(x+dt*k2/2,t+dt/2);k4=rhs(x+dt*k3,t+dt)
            x=x+dt*(k1+2*k2+2*k3+k4)/6;states.append(x.copy())
            at=k*5+ss+1;self.full[at,c]=x;self.slip[at,c]=[delta-np.arctan((x[2]+lf*x[3])/v),-np.arctan((x[2]-lr*x[3])/v)];self.force_sub[at,c]=self.force(k,c,t+dt-start)
        self.calls+=1
        return x,np.array(states)
    def add_arrays(self,a):
        if not a:return
        k=len(a['input']);a['lateral_substate']=self.full[:k*5+1].copy();a['slip_angles']=self.slip[:k*5+1].copy();a['lateral_external_force']=self.force_sub[:k*5+1].copy()
        a['pregenerated_force']=self.forces.copy() # stimulus array, explicitly full duration, never described as observed trajectory


def install_lateral(S, A, spec, steps=500, q=3):
    plant = LateralPlant({'scales': list(spec['scales']), 'pulse': spec.get('pulse', [0., 0.]),
                          'seed': spec.get('lateral_seed')}, steps=steps, q=q)
    A.lateral_step = plant.step; S.lateral_step = plant.step
    return plant


# ---------------------------------------------------------------- AD6 (verbatim v19 scene construction)
def synthetic_scenario(Q=3,queue_sizes=None,initial_gap=10.5,boundary=30.,
                        cutin_speeds=None,cutin_fraction=.5,common_speed=20.):
    if queue_sizes is None:
        queue_sizes=[8]+[7]*(Q-1)  # extends the paper's own 8/7/7 pattern to any Q
    queue_sizes=list(queue_sizes)
    assert len(queue_sizes)==Q,'queue_sizes must have exactly Q entries'
    assert all(s>=2 for s in queue_sizes),'each queue needs >=2 regular vehicles so a mid-queue cut-in gap exists'
    N=sum(queue_sizes)
    starts=np.cumsum([0]+queue_sizes[:-1])
    pos=np.zeros(N)
    for q,first in enumerate(starts):
        last=starts[q+1] if q+1<Q else N
        pos[first]=-10 if q==0 else pos[first-1]-boundary
        pos[first:last]=pos[first]-initial_gap*np.arange(last-first)
    front=np.array([starts[q]+max(0,min(queue_sizes[q]-2,round(cutin_fraction*(queue_sizes[q]-1))))
                     for q in range(Q)],int)
    rear=front+1
    cut=(pos[front]+pos[rear])/2
    vs=(common_speed+1-np.arange(Q,dtype=float) if cutin_speeds is None
        else np.array(cutin_speeds,float))  # default 21,20,19,... matches the
                                             # paper's stated cut-in speeds at Q=3
    return {'N':N,'Q':Q,'starts':starts,'front':front,'rear':rear,'p0':np.r_[pos,cut],
            'v0':np.r_[np.full(N,common_speed),vs],'initial_gap':initial_gap,'boundary':boundary}


def construct_scene(S, family, queue_sizes, boundary=30.):
    """v19 run_verified_sweep.construct_scene with the initial queue spacing as an argument."""
    sizes, q = list(queue_sizes), len(queue_sizes)
    if family == "supplied_synthetic":
        return synthetic_scenario(Q=q, queue_sizes=sizes, boundary=boundary)
    speeds = np.resize(np.array([21., 20., 19.]), q)
    scene = synthetic_scenario(Q=q, queue_sizes=sizes, cutin_speeds=speeds, boundary=boundary)
    if family == "bounded_speed_synthetic":
        return scene
    assert family == "stage1_template", family
    base = S.scenario()
    base_sizes = np.diff(np.r_[base["starts"], base["N"]])
    offsets = base["p0"][base["N"]:] - (base["p0"][base["front"]] + base["p0"][base["rear"]]) / 2
    fronts, cuts = [], []
    for i in range(q):
        template = i % base["Q"]
        gap_fraction = (base["front"][template] - base["starts"][template]) / (base_sizes[template] - 2)
        front = scene["starts"][i] + round(gap_fraction * (sizes[i] - 2))
        fronts.append(front)
        cuts.append((scene["p0"][front] + scene["p0"][front + 1]) / 2 + offsets[template])
    scene["front"] = np.array(fronts, dtype=int)
    scene["rear"] = scene["front"] + 1
    scene["p0"][scene["N"]:] = cuts
    scene["v0"][scene["N"]:] = np.resize(base["v0"][base["N"]:], q)
    if sizes == [8, 7, 7]:
        ref = S.scenario(10.5, boundary)
        assert all(np.array_equal(np.asarray(scene[k]), np.asarray(ref[k])) for k in ref), "template [8,7,7] != frozen scene"
    return scene


def install_scene(S, A, spec):
    """spec = {'family':..., 'queue_sizes':[...], 'boundary':b}.  Returns the scene; scenario() returns a fresh copy."""
    orig = S.scenario
    scene = construct_scene(type('S', (), {'scenario': staticmethod(orig)}), spec['family'], spec['queue_sizes'],
                            float(spec['boundary']))

    def scenario(*args, **kwargs):
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in scene.items()}
    S.scenario = scenario; A.scenario = scenario
    return scene


# ---------------------------------------------------------------- AD7
INDEPENDENT_SUBS = [("    for q in range(Q-1):\n"
                     "        head=queue_start[q+1];tail=head-1;row=np.zeros(Q+M);row[q]=-1;row[q+1]=1\n"
                     "        rows.append(row);rhs.append(positions[tail]-positions[head]+preferred[tail]-preferred[head]-boundary)\n",
                     "")]


def sequential_allocation(positions, queue_start, front, rear, caps, nominal_gap=12., boundary=12., objective='l1'):
    """Greedy front-to-back coordination: queue q minimises its own total
    displacement subject to its own constraints and to the inter-queue
    boundary row with the ALREADY FIXED upstream queue q-1 (no look-ahead)."""
    N = len(positions); Q = len(queue_start)
    # per-queue quantities computed exactly as allocation() does
    preferred = np.zeros(N); owner = np.zeros(N, dtype=int)
    lb = np.full(Q, -np.inf); ub = np.full(Q, np.inf)
    for q in range(Q):
        ix = np.arange(queue_start[q], queue_start[q + 1] if q + 1 < Q else N)
        tar = -nominal_gap * np.arange(len(ix)); tar[ix >= rear[q]] -= nominal_gap
        loc = front[q] - ix[0]
        tar += (positions[front[q]] + positions[rear[q]]) / 2 - (tar[loc] + tar[loc + 1]) / 2
        preferred[ix] = tar - positions[ix]; owner[ix] = q
        lb[q] = max(.01 - preferred[ix[ix <= front[q]]]); ub[q] = min(-.01 - preferred[ix[ix >= rear[q]]])
        lb[q] = max(lb[q], np.max(caps[ix, 0] - preferred[ix])); ub[q] = min(ub[q], np.min(caps[ix, 1] - preferred[ix]))
    M = N + Q
    cp = (positions[front] + preferred[front] + positions[rear] + preferred[rear]) / 2
    cpref = cp - caps[N:, 2]
    for q in range(Q): lb[q] = max(lb[q], caps[N + q, 0] - cpref[q]); ub[q] = min(ub[q], caps[N + q, 1] - cpref[q])
    pref = np.r_[preferred, cpref]
    if np.any(lb > ub): return None
    beta = np.zeros(Q)
    for q in range(Q):
        lo, hi = lb[q], ub[q]
        if q > 0:
            head = queue_start[q]; tail = head - 1
            hi = min(hi, beta[q - 1] + positions[tail] - positions[head] + preferred[tail] - preferred[head] - boundary)
        if lo > hi: return None
        ix = np.flatnonzero(owner == q)
        # 1-D L1 minimisation of sum |preferred_i + b| over [lo, hi]: clip the median
        b = float(np.median(-preferred[ix]))
        beta[q] = min(max(b, lo), hi)
    Bmat = np.zeros((M, Q)); Bmat[np.arange(N), owner] = 1; Bmat[N:] = np.eye(Q)
    delta = pref + Bmat @ beta
    return {'delta': delta, 'beta': beta, 'preferred': pref, 'owner': owner,
            'bounds': np.c_[lb, ub], 'objective': float(np.abs(delta[:N]).sum())}


def install_allocation(R, S, mode):
    if mode == 'independent':
        fn = AD.recompile(R, 'allocation', INDEPENDENT_SUBS)
    elif mode == 'independent@identity':
        fn = AD.recompile(R, 'allocation', [])
    elif mode == 'sequential':
        fn = sequential_allocation
    else:
        raise ValueError(mode)
    S.allocation = fn
    return fn
