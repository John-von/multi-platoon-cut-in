"""Full sampled vehicle experiment with certified longitudinal backup plans.

Lateral bicycle, MPC weights and RK4 substeps follow frozen simulate_v16.m.
The new longitudinal optimizer is a reference/space-time safety filter with
common feedback, not a relabeling of the old DMPC solver.
"""
import sys,json,time,itertools,hashlib
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.linalg import expm
from cvxopt import matrix,solvers
from robust_core import *
import os as _os
from concurrent.futures import ProcessPoolExecutor as _PPE
import multiprocessing.util as _mpu

# PERF (E1-a prototype, 2026-09-27): optional deterministic parallel evaluation of the per-vehicle
# capacity() calls in make_plan.  Arithmetic is unchanged: every capacity() call runs the same code on the
# same inputs in a worker process whose v20 options are set to the caller's; results are collected in
# vehicle order and the serial early-exit decision ('capacity_zero' at the first empty capacity) is
# reproduced exactly.  V20_PLAN_WORKERS<=1 (default) keeps the original serial path.
_CAP_POOL=None;_CAP_PID=None
def _plan_workers():
    try:return int(_os.environ.get('V20_PLAN_WORKERS','1'))
    except ValueError:return 1
def _cap_pool(workers):
    # (a) a pool object inherited through fork belongs to the parent and cannot run tasks in the child,
    # so the pool is rebuilt per process; (b) the pool is shut down by a multiprocessing finalizer, which runs
    # before a worker process joins its own children -- otherwise a planning worker that owns this pool never
    # exits and the caller's executor.shutdown(wait=True) blocks (async host mode).  Priority 100 makes it run
    # before the call queue's own close finalizer (priority 10), which would stop the feeder thread first.
    global _CAP_POOL,_CAP_PID
    if _CAP_POOL is None or _CAP_PID!=_os.getpid() or _CAP_POOL._max_workers!=workers:
        if _CAP_POOL is not None and _CAP_PID==_os.getpid():_CAP_POOL.shutdown(wait=True)
        _CAP_POOL=_PPE(max_workers=workers);_CAP_PID=_os.getpid()
        _mpu.Finalize(_CAP_POOL,_CAP_POOL.shutdown,kwargs={'wait':True},exitpriority=100)
    return _CAP_POOL
def warm_plan_pool():
    # Harness hook (timing runs only): start the capacity worker pool ahead of the first planning call so that
    # worker start-up is not charged to it.  Never called by the simulation itself; returns the pid like ping().
    workers=_plan_workers()
    if workers>1:list(_cap_pool(workers).map(abs,range(workers)))
    return _os.getpid()
def _cap_task(args):
    options,ref,noise,r0,b0,hb,robust=args
    import robust_core as _rc
    _rc.configure_v20(options)
    return _rc.capacity(ref,*noise,initial_radius=r0,initial_box=b0,history_bounds=hb) if robust else _rc.capacity(ref,0,0)
solvers.options.update({'show_progress':False,'abstol':1e-8,'reltol':1e-7,'feastol':1e-8})

def scenario(initial_gap=10.5,boundary=30.,cutin_speeds=None):
    src=loadmat(ROOT/'baseline/stage1_plan.mat',simplify_cells=True)
    starts=np.array(src['queue_start'],int)-1;N=int(src['num_vehicles'])
    front=np.array([c['frontIndex'] for c in src['cutin']],int)-1;rear=front+1
    Q=len(starts)  # PHASE-B: was hardcoded 3; now the number of target queues
                   # actually present in stage1_plan.mat (front/rear/starts
                   # already vary with it -- only the returned 'Q' and the
                   # loop boundary below assumed exactly 3 queues).
    pos=np.zeros(N)
    for q,first in enumerate(starts):
        last=starts[q+1] if q+1<Q else N  # PHASE-B: was `if q<2 else N`,
                                           # i.e. hardcoded for Q==3; this is
                                           # the literal generalization
                                           # (q<2 <=> q+1<3 when Q==3), so
                                           # behavior at Q==3 is unchanged.
        pos[first]=-10 if q==0 else pos[first-1]-boundary
        pos[first:last]=pos[first]-initial_gap*np.arange(last-first)
    # Preserve the original cut-in offset relative to its target gap midpoint.
    old=np.array(src['positions'],float);oldcut=np.array(src['cfg']['cutin_pos'],float)
    cut=(pos[front]+pos[rear])/2+oldcut-(old[front]+old[rear])/2
    vs=np.array(src['cfg']['cutin_speed'],float) if cutin_speeds is None else np.array(cutin_speeds,float)
    return {'N':N,'Q':Q,'starts':starts,'front':front,'rear':rear,'p0':np.r_[pos,cut],
            'v0':np.r_[np.full(N,20.),vs],'initial_gap':initial_gap,'boundary':boundary}

def make_plan(x,theta,scene,T,kind='merge',noise=(.001,.00015),common=None,
              robust=True,fixed_delta=None,active=None,initial_radius=None,initial_box=None,history_bounds=None):
    N=scene['N'];Mveh=x.shape[0];Q=scene['Q']
    if V20_OPTIONS.get('enabled', False) and V20_OPTIONS.get('init_refine', False):
        eps=np.asarray(V20_OPTIONS.get('meas_eps', np.zeros(3)), dtype=float)
        if initial_radius is None:
            initial_radius=np.full(Mveh, measurement_radius(eps))
        if initial_box is None:
            initial_box=np.repeat(np.abs(eps)[None,:], Mveh, axis=0)
    r0=np.zeros(Mveh) if initial_radius is None else np.asarray(initial_radius)
    b0=np.zeros((Mveh,3)) if initial_box is None else np.asarray(initial_box)
    refs=[ref_coeff(x[i],T,theta[i],common) for i in range(Mveh)]
    if kind=='merge' and fixed_delta is None:
        caps=[];workers=_plan_workers();parallel=None
        if workers>1:
            tasks=[(dict(V20_OPTIONS),ref,noise,r0[i],b0[i],None if history_bounds is None else history_bounds[i],robust)
                   for i,ref in enumerate(refs)]
            parallel=list(_cap_pool(workers).map(_cap_task,tasks))
        for i,ref in enumerate(refs):
            if parallel is not None:cap=parallel[i]
            else:cap=capacity(ref,*noise,initial_radius=r0[i],initial_box=b0[i],history_bounds=None if history_bounds is None else history_bounds[i]) if robust else capacity(ref,0,0)
            if cap is None:return None,'capacity_zero'
            caps.append(cap)
        cap_array=np.c_[np.array(caps),np.zeros(Mveh)]
        cap_array[N:,2]=x[N:,0]
        alloc=allocation(x[:N,0],scene['starts'],scene['front'],scene['rear'],cap_array)
        if alloc is None:return None,'joint_allocation'
        delta=alloc['delta']
    else:
        delta=np.zeros(Mveh) if fixed_delta is None else np.array(fixed_delta)
        alloc={'delta':delta,'beta':np.zeros(Q),'objective':float(sum(abs(delta[:N])))}
    cert=[certificate(r,[d,d],*(noise if robust else (0,0)),initial_radius=r0[i],initial_box=b0[i],history_bounds=None if history_bounds is None else history_bounds[i]) for i,(r,d) in enumerate(zip(refs,delta))]
    if not all(c['pass'] for c in cert):return None,'trajectory_constraints'
    nom=[];feed=[]
    for r,d in zip(refs,delta):
        n,u=evaluate_ref(r,d);nom.append(n);feed.append(u)
    nom=np.stack(nom,axis=1);feed=np.stack(feed,axis=1)
    boxes=np.stack([c['box'] for c in cert],axis=1)
    terminal=np.stack([c['terminal_box'] for c in cert])
    nominal_gaps=nom[:,:N-1,0]-nom[:,1:N,0]
    gaps=nominal_gaps-boxes[:,:N-1,0]-boxes[:,1:N,0]
    terminal_gaps=nom[-1,:N-1,0]-nom[-1,1:N,0]-terminal[:N-1,0]-terminal[1:N,0]
    if min(gaps.min(),terminal_gaps.min())<10-1e-7:return None,'ordinary_gap_tube'
    # Source-lane cut-ins keep their longitudinal ordering before lateral entry.
    sg=nom[:,N:-1,0]-nom[:,N+1:,0]-boxes[:,N:-1,0]-boxes[:,N+1:,0]
    if sg.size and sg.min()<10-1e-7:return None,'source_gap_tube'
    if V20_OPTIONS.get('enabled', False) and V20_OPTIONS.get('condition_s', False):
        tsg=nom[-1,N:-1,0]-nom[-1,N+1:,0]-terminal[N:-1,0]-terminal[N+1:,0]
        if tsg.size and tsg.min()<10-1e-7:return None,'source_terminal_gap'
    gates=[]
    for q,(f,r) in enumerate(zip(scene['front'],scene['rear'])):
        c=N+q
        gf=nom[:,f,0]-nom[:,c,0]-boxes[:,f,0]-boxes[:,c,0]
        gr=nom[:,c,0]-nom[:,r,0]-boxes[:,c,0]-boxes[:,r,0]
        tf=nom[-1,f,0]-nom[-1,c,0]-terminal[f,0]-terminal[c,0]
        tr=nom[-1,c,0]-nom[-1,r,0]-terminal[c,0]-terminal[r,0]
        worst=np.minimum(gf,gr)
        if kind=='merge' or (active is not None and active[q]):
            if min(tf,tr)<10-1e-7:return None,'terminal_merge_gap'
            suffix=np.minimum.accumulate(worst[::-1])[::-1]
            ix=np.flatnonzero(suffix>=10.05)
            if not len(ix):return None,'merge_gap_tube'
            gate=int(ix[0])
            if active is not None and active[q] and gate>0:return None,'active_contract'
            gates.append(gate)
        else:gates.append(None)
    return {'nom':nom,'feed':feed,'ref':refs,'cert':cert,'boxes':boxes,'terminal':terminal,
            'theta':theta.copy(),'delta':delta,'alloc':alloc,'T':T,'gates':gates,
            'kind':kind,'robust':robust,'noise':noise},'accepted'

def sample_plan(plan,index):
    steps=len(plan['feed'])
    if index<steps:return plan['nom'][index],plan['feed'][index],plan['boxes'][index]
    n=plan['nom'][-1].copy();n[:,0]+=(index-steps)*DT*n[:,1]
    return n,n[:,2].copy(),plan['terminal']

def lateral_step(x,v,delta):
    def rhs(s):
        _,phi,vy,r=s;lf=1.2;lr=1.6;m=1500;iz=2500;cf=cr=50000
        ff=cf*(delta-np.arctan((vy+lf*r)/v))
        fr=-cr*np.arctan((vy-lr*r)/v)
        return np.array([v*np.sin(phi)+vy*np.cos(phi),r,(ff+fr)/m-v*r,(lf*ff-lr*fr)/iz])
    states=[];dt=DT/5
    for _ in range(5):
        k1=rhs(x);k2=rhs(x+dt*k1/2);k3=rhs(x+dt*k2/2);k4=rhs(x+dt*k3)
        x=x+dt*(k1+2*k2+2*k3+k4)/6;states.append(x.copy())
    return x,np.array(states)

def lateral_control(x,v,yr,last,H=40,return_plan=False,previous_plan=None,return_info=False):
    lf=1.2;lr=1.6;m=1500;iz=2500;cf=cr=50000
    A=np.array([[0,v,1,0],[0,0,0,1],[0,0,-(cf+cr)/(m*v),-v-(cf*lf-cr*lr)/(m*v)],
                [0,0,-(cf*lf-cr*lr)/(iz*v),-(cf*lf**2+cr*lr**2)/(iz*v)]])
    B=np.array([0,0,cf/m,cf*lf/iz])
    aug=np.zeros((5,5));aug[:4,:4]=A;aug[:4,4]=B
    Z=expm(aug*DT);A=Z[:4,:4];B=Z[:4,4]
    F=np.zeros((4*H,4));G=np.zeros((4*H,H));ak=np.eye(4)
    for k in range(H):
        ak=A@ak;F[4*k:4*k+4]=ak
        if k:G[4*k:4*k+4]=A@G[4*k-4:4*k]
        G[4*k:4*k+4,k]=B
    target=np.zeros((H,4));target[:,0]=yr
    q=np.tile([80.,8.,1.,1.],H);E=np.eye(H)-np.diag(np.ones(H-1),-1);rr=np.zeros(H);rr[0]=last
    HH=2*(G.T@(q[:,None]*G)+2*np.eye(H)+4*E.T@E)
    lin=2*(G.T@(q*(F@x-target.ravel()))-4*E.T@rr)
    iy=np.arange(0,4*H,4);iv=iy+2;ir=iy+3;fx=F@x
    AA=np.vstack([G[iy],-G[iy],G[iv],-G[iv],G[ir],-G[ir],E,-E,np.eye(H),-np.eye(H)])
    bb=np.r_[3.499-fx[iy],fx[iy],5-fx[iv],5+fx[iv],1-fx[ir],1+fx[ir],.01+rr,.01-rr,
             np.full(2*H,.25)]
    enabled=V20_OPTIONS.get('enabled',False) and V20_OPTIONS.get('lateral_soft',False)
    if not enabled:
        sol=solvers.qp(matrix((HH+HH.T)/2),matrix(lin),matrix(AA),matrix(bb))
        if sol['status']!='optimal':raise RuntimeError('Lateral QP failed: '+sol['status'])
        u=np.array(sol['x']).ravel()
        if np.max(AA@u-bb)>2e-6:raise RuntimeError('Lateral constraint residual')
        return u.copy() if return_plan else u[0]
    from lateral_fallback import soften,hard_info
    u=None
    try:
        sol=solvers.qp(matrix((HH+HH.T)/2),matrix(lin),matrix(AA),matrix(bb))
        if sol['status']=='optimal':
            candidate=np.array(sol['x']).ravel()
            if np.isfinite(candidate).all() and np.max(AA@candidate-bb)<=2e-6:u=candidate
    except (ValueError,ArithmeticError,RuntimeError):pass
    if u is None:u,info=soften(HH,lin,AA,bb,last,previous_plan,'cvxopt')
    else:info=hard_info(AA,bb,u)
    answer=u.copy() if return_plan else u[0]
    return (answer,info) if return_info else answer


def smooth(s):
    z=np.clip(s,0,1);return 10*z**3-15*z**4+6*z**5

def simulate(cfg):
    configure_v20(cfg.get('v20', {}))
    if V20_OPTIONS.get('enabled',False) and V20_OPTIONS.get('replan_mode')=='handover':
        from async_execution import run_async
        return run_async(cfg)
    v20=bool(V20_OPTIONS.get('enabled',False))
    scene=scenario(cfg.get('initial_gap',10.5),cfg.get('boundary',30),cfg.get('cutin_speeds'))
    N=scene['N'];Q=scene['Q'];M=N+Q;steps=round(cfg.get('time_sim',55)/DT)  # PHASE-B: Q was hardcoded 3 (as "N+3") throughout this function
    mass=np.broadcast_to(np.array(cfg.get('mass_scale',1.)),(M,))
    tau=np.broadcast_to(np.array(cfg.get('tau_scale',1.)),(M,))
    truth=np.c_[1/mass,DT/(.7*tau)]
    x=np.c_[scene['p0'],scene['v0'],DRAG*scene['v0']**2+GF/truth[:,0]]
    bounds=np.repeat(PRIOR[None,:,:],M,axis=0)
    eps=np.asarray(V20_OPTIONS.get('meas_eps',np.zeros(3))) if v20 else np.zeros(3)
    mrng=np.random.default_rng(int(cfg.get('seed',0))+1000003)
    eta=mrng.uniform(-eps,eps,size=(steps+1,M,3)) if np.any(eps) else np.zeros((steps+1,M,3))
    XH=np.zeros((steps+1,M,3));XH[0]=x+eta[0]
    previous_lateral=np.zeros((Q,40));lat_levels=np.zeros((steps,Q),int);lat_slack=np.zeros((steps,Q,3))
    def history(k):
        if not v20 or V20_OPTIONS.get('jerk_mode')!='window':return None
        W=int(V20_OPTIONS.get('jerk_window') or 10);out=np.full((M,W,2),np.nan)
        for j,at in enumerate(range(k-W,k)):
            if at<0:continue
            a=(XH[at+1,:,1]-XH[at,:,1])/DT
            out[:,j,0]=a-2*eps[1]/DT;out[:,j,1]=a+2*eps[1]/DT
        return out
    method=cfg.get('method','adaptive')
    # tube uses a fixed prior model and a fixed tube; ddrpc uses an online
    # data model with a residual tube. Both are separately reported baselines.
    robust=method in ['adaptive','fixed','tube','ddrpc']
    noise=(cfg.get('acc_bound',.001),cfg.get('torque_bound',.00015))
    plan_noise=(cfg.get('plan_acc_bound',noise[0]),cfg.get('plan_torque_bound',noise[1]))
    actual_noise=cfg.get('disturbance_fraction',.5)
    # PHASE-B: optional seeded randomized disturbance, additive-only. With no
    # seed (the default), behavior is byte-identical to before this patch --
    # every existing cited run stays reproducible. With a seed, replaces the
    # deterministic sinusoid below with i.i.d. draws of the SAME declared
    # amplitude (actual_noise fraction of the certificate's acc/torque
    # bounds), so the robust certificate's per-step bound assumption is still
    # honored -- only the shape/correlation of the disturbance changes.
    rng=np.random.default_rng(cfg['seed']) if cfg.get('seed') is not None else None
    def common_for(t,T):
        if not cfg.get('overlap_cruise',False):return None
        z=t+np.arange(round(T/DT)+2)*DT-6
        return 20+.6*(smooth(z/6)-2*smooth((z-6)/10)+smooth((z-16)/6))
    dtheta=np.tile(np.array([[PRIOR[0].mean(),PRIOR[1].mean()]]),(M,1))
    dwidth=np.tile((PRIOR[:,1]-PRIOR[:,0])/2,(M,1))
    dnoise=np.tile(np.array(plan_noise),(M,1))
    data_prev=[];data_out=[];dtheta_hist=[];dnoise_hist=[]
    def planning_theta():
        if method in ['fixed','tube']:return np.repeat(PRIOR[None,:,:],M,axis=0)
        if method=='nominal':return np.tile(np.array([[1.,1.],[DT/.7,DT/.7]]),(M,1,1))
        if method=='point':return np.repeat(bounds.mean(axis=2)[:,:,None],2,axis=2)
        if method=='ddrpc':
            lo=np.maximum(PRIOR[:,0],dtheta-dwidth);hi=np.minimum(PRIOR[:,1],dtheta+dwidth)
            return np.stack([lo,hi],axis=2)
        return bounds.copy()
    tclock=time.perf_counter();plan=None;events=[]
    for T in [12,20,26,30,36]:
        cs=None if cfg.get('initial_constant_backup',False) else common_for(0,T)
        if cs is not None and not np.allclose(cs[-3:],cs[-1],atol=1e-10,rtol=0):continue
        pn=plan_noise if method!='ddrpc' else tuple(np.max(dnoise,axis=0))
        plan,reason=make_plan(XH[0] if v20 else x,planning_theta(),scene,T,'cruise',pn,cs,robust=robust)
        events.append({'time':0.,'T':T,'kind':'initial','reason':reason})
        if plan:break
    init_seconds=time.perf_counter()-tclock
    if plan is None:return {'status':'initial_not_certified','events':events,'cfg':cfg,'initial_plan_seconds':init_seconds},{}
    plan_at=0;mergeplan=False;committed=False;starts=np.full(Q,np.nan);done=np.full(Q,np.nan)
    lat=np.zeros((Q,4));steer=np.zeros(Q);mode=np.zeros(Q,int)
    X=np.zeros((steps+1,M,3));X[0]=x;U=np.zeros((steps,M));LH=np.zeros((steps+1,Q,4))
    BH=np.zeros((steps+1,M,2,2));BH[0]=bounds;timing=np.zeros(steps);planning=np.zeros(steps);SH=np.zeros((steps,Q))
    errors=np.zeros((steps,M,3));tubes=np.zeros((steps,M,3));ratiomax=np.zeros(steps)
    body=np.zeros((steps*5+1,M,3));body[0,:,0]=x[:,0];body[0,:N,1]=3.5
    ALOG=np.zeros((steps,M));predres=np.zeros(steps);modehist=np.zeros((steps+1,Q),int)  # PHASE-B: all six "3"s above were the hardcoded queue count
    stable_since=None;release=None;release_stable_since=None
    cycle_pending=bool(cfg.get('cruise_cycle',False));cycle_done=False
    plans_saved=[];fullstart=time.perf_counter();request=cfg.get('request_time',5.)
    attempts=0;status='completed';lastk=steps
    for k in range(steps):
        tick=time.perf_counter();t=k*DT
        # Replan only before lateral admission. Existing accepted plan remains
        # the executable backup if a new candidate fails or is deliberately timed out.
        if t>=request-1e-8 and not committed and attempts<cfg.get('max_attempts',3) and (attempts==0 or k%10==0) and (not mergeplan or method=='adaptive' or cfg.get('replan_until_admission',False)):  # PHASE-B bonus: async_execution.py already reads this from cfg (default 3, unchanged); simulate() had it hardcoded -- made consistent, same default
            pclock=time.perf_counter();candidate=None;attempts+=1
            for T in cfg.get('durations',[12,14,16,18,20,22,24,26,28,30,34]):
                if mergeplan and t+T>=plan_at*DT+plan['T']-1.0:break
                cs=common_for(t,T)
                if cs is not None and not np.allclose(cs[-3:],cs[-1],atol=1e-10,rtol=0):continue
                pn=plan_noise if method!='ddrpc' else tuple(np.max(dnoise,axis=0))
                pp,reason=make_plan(XH[k] if v20 else x,planning_theta(),scene,T,'merge',pn,cs,robust=robust,history_bounds=history(k))
                events.append({'time':t,'T':T,'kind':'merge','reason':reason})
                if pp:candidate=pp;break
            if candidate and not (cfg.get('inject_timeout',False) and mergeplan):
                plan=candidate;plan_at=k;mergeplan=True
                plans_saved.append({'time':t,'T':plan['T'],'delta':plan['delta'].tolist(),
                                    'beta':plan['alloc']['beta'].tolist(),'theta':plan['theta'].tolist(),
                                    'gates':plan['gates'],'max_radius':max(float(c['radius'].max()) for c in plan['cert'])})
            elif candidate:
                events.append({'time':t,'kind':'timeout','reason':'retained_certified_plan'})
            planning[k]=time.perf_counter()-pclock
        ix=k-plan_at
        if mergeplan:
            for c in range(Q):  # PHASE-B: was range(3)
                if mode[c]==0 and ix>=plan['gates'][c]:mode[c]=1;starts[c]=t;committed=True
        nom,uff,tube=sample_plan(plan,ix)
        e=x-nom;u=uff+((XH[k]-nom) if v20 else e)@K
        errors[k]=e;tubes[k]=tube
        cert_index=min(ix,len(plan['cert'][0]['radius'])-1)
        norm_e=np.sqrt(np.einsum('bi,ij,bj->b',e,P,e))
        radii=np.array([c['radius'][ix] if ix<len(c['radius']) else c['terminal_radius'] for c in plan['cert']])
        ratiomax[k]=np.max(norm_e/(radii+1e-10))
        # No saturation or artificial projection: any violation remains evidence.
        if rng is None:  # PHASE-B: unchanged deterministic default
            dist_a=actual_noise*noise[0]*np.sin(.83*t+np.arange(M)*.37)
            dist_t=actual_noise*noise[1]*np.sin(1.37*t+np.arange(M)*.51)
        else:  # PHASE-B: seeded randomized mode, same declared amplitude
            dist_a=actual_noise*noise[0]*rng.uniform(-1.,1.,M)
            dist_t=actual_noise*noise[1]*rng.uniform(-1.,1.,M)
        xn=np.stack([plant_step(x[i],u[i],truth[i],dist_a[i],dist_t[i]) for i in range(M)])
        ALOG[k]=(xn[:,1]-x[:,1])/DT
        XH[k+1]=xn+eta[k+1]
        if v20 and np.any(eps):
            nextbounds=np.stack([update_set_measured(bounds[i],XH[k,i],u[i],XH[k+1,i],*noise,meas_eps=eps) for i in range(M)])
        else:nextbounds=np.stack([update_set(bounds[i],x[i],u[i],xn[i],*noise) for i in range(M)])
        if not(np.all(nextbounds[:,:,0]<=truth+1e-9) and np.all(truth<=nextbounds[:,:,1]+1e-9)):
            raise AssertionError('True parameters escaped identification set')
        U[k]=u;bounds=nextbounds
        if method=='ddrpc':
            # Low-order data-driven identification of the two unknown scalar
            # coefficients. The residual envelope is carried into the next
            # predictive certificate and never uses the simulated truth.
            for i in range(M):
                f=x[i,2]-DRAG*x[i,1]**2
                y=(xn[i,1]-x[i,1])/DT+GF
                zf=u[i]-x[i,2]
                zy=xn[i,2]-x[i,2]
                data_prev.append((i,f,zf));data_out.append((i,y,zy))
            for i in range(M):
                rows=[q for q in data_prev if q[0]==i]
                outs=[q for q in data_out if q[0]==i]
                if rows:
                    ff=np.array([q[1] for q in rows]);zz=np.array([q[2] for q in rows])
                    yy=np.array([q[1] for q in outs]);ww=np.array([q[2] for q in outs])
                    dtheta[i,0]=np.clip(float(np.dot(ff,yy)/max(np.dot(ff,ff),1e-12)),PRIOR[0,0],PRIOR[0,1])
                    dtheta[i,1]=np.clip(float(np.dot(zz,ww)/max(np.dot(zz,zz),1e-12)),PRIOR[1,0],PRIOR[1,1])
                    dnoise[i,0]=max(noise[0],float(np.max(np.abs(yy-dtheta[i,0]*ff))))
                    dnoise[i,1]=max(noise[1],float(np.max(np.abs(ww-dtheta[i,1]*zz))))
                    dwidth[i]=np.maximum(np.abs(bounds[i,:,1]-dtheta[i]),np.abs(dtheta[i]-bounds[i,:,0]))
            dtheta_hist.append(dtheta.copy());dnoise_hist.append(dnoise.copy())
        lat_sub=np.zeros((5,Q,4))  # PHASE-B: was (5,3,4)
        for c in range(Q):  # PHASE-B: was range(3)
            if mode[c]:
                yr=3.495*smooth((t+(np.arange(40)+1)*DT-starts[c])/4)
                if v20 and V20_OPTIONS.get('lateral_soft',False):
                    seq,li=lateral_control(lat[c],XH[k,N+c,1],yr,steer[c],return_plan=True,
                        previous_plan=previous_lateral[c],return_info=True)
                    previous_lateral[c]=seq;steer[c]=seq[0]
                    lat_levels[k,c]={'hard':1,'soft':2,'shift':3}[li['level']];lat_slack[k,c]=li['slack_max']
                else:steer[c]=lateral_control(lat[c],x[N+c,1],yr,steer[c])
                lat[c],lat_sub[:,c]=lateral_step(lat[c],x[N+c,1],steer[c])
                if mode[c]==1 and t+DT>=starts[c]+4-1e-8 and abs(lat[c,0]-3.5)<=.1 and abs(lat[c,2])<=.2:
                    mode[c]=2;done[c]=t+DT
        for ss in range(5):
            at=k*5+ss+1;body[at,:,0]=x[:,0]+(ss+1)*DT/5*x[:,1]
            body[at,:N,1]=3.5;body[at,N:,1]=lat_sub[ss,:,0];body[at,N:,2]=lat_sub[ss,:,1]
        x=xn;X[k+1]=x;LH[k+1]=lat;BH[k+1]=bounds;modehist[k+1]=mode;SH[k]=steer
        # Gate uses observed velocity differences; true parameters are not read.
        stable=np.all(mode==2) and max(abs(x[:,1]-20))<=.05 and max(abs(ALOG[k]))<=.05
        if stable:
            if stable_since is None:stable_since=t+DT
        else:stable_since=None
        if cycle_pending and stable_since is not None and t+DT-stable_since>=5-1e-8:
            pclock=time.perf_counter();T=24.;tt=np.arange(round(T/DT)+2)*DT
            vs=20+2.5*(smooth(tt/6)-2*smooth((tt-6)/10)+smooth((tt-16)/6))
            pn=plan_noise if method!='ddrpc' else tuple(np.max(dnoise,axis=0))
            candidate,reason=make_plan(XH[k+1] if v20 else x,planning_theta(),scene,T,'cycle',pn,vs,robust,
                                       fixed_delta=np.zeros(M),active=np.ones(Q,bool),history_bounds=history(k+1))  # PHASE-B: was np.ones(3,bool)
            events.append({'time':t+DT,'T':T,'kind':'cycle','reason':reason})
            if candidate:
                plan=candidate;plan_at=k+1;release=t+DT;release_stable_since=stable_since
                cycle_pending=False;cycle_done=True
                # Do not overwrite previously accepted start/completion records.
            planning[k]+=time.perf_counter()-pclock
        timing[k]=time.perf_counter()-tick
    runtime=time.perf_counter()-fullstart
    tt=np.arange(steps+1)*DT
    j=np.diff(ALOG,axis=0)/DT
    gaps=X[:,:N-1,0]-X[:,1:N,0]
    gate_gaps=[]
    for c in range(Q):  # PHASE-B: was range(3)
        if np.isfinite(starts[c]):
            at=round(starts[c]/DT);gate_gaps.extend([X[at:,scene['front'][c],0]-X[at:,N+c,0],X[at:,N+c,0]-X[at:,scene['rear'][c],0]])
    min_merge=min((z.min() for z in gate_gaps),default=float('nan'))
    # Common-speed metric is corrected for a released cycle when present.
    common=np.full(steps,20.)
    if cfg.get('overlap_cruise',False):
        z=np.arange(steps)*DT-6
        common+=.6*(smooth(z/6)-2*smooth((z-6)/10)+smooth((z-16)/6))
    if release is not None:
        z=np.arange(steps)*DT-release
        common+=2.5*(smooth(z/6)-2*smooth((z-6)/10)+smooth((z-16)/6))
    mask=tt[:-1]>=request-1e-8
    audit_jerk=(ALOG[10:]-ALOG[:-10])/(10*DT) if v20 and V20_OPTIONS.get('jerk_mode')=='window' else j
    safe=(X[:,:,1].max()<=23+1e-6 and X[:,:,1].min()>=5-1e-6 and abs(ALOG).max()<=2+1e-6 and
          abs(audit_jerk).max()<=1+1e-6 and U.min()>=UMIN-1e-6 and U.max()<=UMAX+1e-6 and gaps.min()>=10-1e-6 and
          (not len(gate_gaps) or min_merge>=10-1e-6))
    status='completed' if np.all(np.isfinite(done)) else 'admission_not_completed'
    result={'status':status,'cfg':cfg,'completed':int(np.isfinite(done).sum()),'constraint_pass':bool(safe),
            'merge_start':starts.tolist(),'merge_done':done.tolist(),'last_done':float(np.nanmax(done)) if np.any(np.isfinite(done)) else None,
            'Jv':float(np.abs(X[:-1,:N,1]-common[:,None])[mask].sum()*DT/N),'v_max':float(X[:,:,1].max()),
            'a_max':float(abs(ALOG).max()),'j_max':float(abs(j).max()),'ordinary_gap_min':float(gaps.min()),'merge_gap_min':float(min_merge),
            'tube_ratio_max':float(ratiomax.max()),'initial_plan_seconds':init_seconds,
            'runtime_s':runtime,'step_mean_s':float(timing.mean()),'step_max_s':float(timing.max()),
            'step_p95_s':float(np.quantile(timing,.95)),'over_0_1_fraction':float(np.mean(timing>.1)),
            'tracking_mean_s':float(np.mean(timing-planning)),'planning_total_s':float(sum(planning)),
            'planning_max_s':float(max(planning)),'plans':plans_saved,'events':events,'cycle_release':release,
            'stable_since':stable_since,'release_stable_since':release_stable_since,
            'stable_hold_before_release':None if release is None else release-release_stable_since,'cycle_executed':cycle_done,
            'posterior_mu_width':float(np.mean(bounds[:,0,1]-bounds[:,0,0])),
            'posterior_lambda_width':float(np.mean(bounds[:,1,1]-bounds[:,1,0])),
            'parameters_contained':True}
    if method=='ddrpc' and dtheta_hist:
        hist=np.array(dtheta_hist)
        result.update({'data_model_samples':int(len(data_prev)),
                       'data_model_parameter_width':float(np.mean(np.ptp(hist,axis=0))),
                       'data_residual_acc_bound':float(np.max(dnoise[:,0])),
                       'data_residual_torque_bound':float(np.max(dnoise[:,1]))})
    arrays={'time':tt,'state':X,'input':U,'lateral':LH,'bounds':BH,'truth':truth,'acceleration':ALOG,
            'error':errors,'tube_box':tubes,'tube_ratio':ratiomax,'timing':timing,'planning_time':planning,
            'body_poses':body,'mode':modehist,'steering':SH,'common_speed':common,'front':scene['front'],'rear':scene['rear']}
    if v20:
        arrays.update(measured_state=XH,measurement_noise=eta,lateral_levels=lat_levels,lateral_slack=lat_slack)
        result['window_jerk_max']=float(np.max(np.abs((ALOG[10:]-ALOG[:-10])/(10*DT))))
    return result,arrays

def run(cfg,out):
    out=Path(out);out.parent.mkdir(exist_ok=True,parents=True)
    if out.with_suffix('.json').exists():raise FileExistsError('Evidence exists; choose a new run id')
    result,arrays=simulate(cfg)
    sources=sorted(Path(__file__).parent.glob('*.py'))
    result['source_hashes']={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    out.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
    if arrays:np.savez_compressed(out.with_suffix('.npz'),**arrays)
    print(json.dumps({k:v for k,v in result.items() if k not in ['events','plans','source_hashes']},ensure_ascii=False),flush=True)
    return result

if __name__=='__main__':
    cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    run(cfg,sys.argv[2])
