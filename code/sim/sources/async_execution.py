"""Future-dated certified handoff with a continuously ticking 100 ms plant.

Modes: host = actual separate planning process + wall-clock pacing;
replay = measured computation cost is replayed on virtual event clock.
The candidate starts at an OLD nominal future state, with OLD uncertainty,
not at a stale measured state. Late/cancelled candidates are never applied.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time,hashlib,platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from robust_core import *
from simulation import scenario,make_plan,sample_plan,lateral_step,smooth,lateral_control as cvxopt_lateral
from lateral_fast import lateral_control as osqp_lateral

def ping():return os.getpid()

def plan_job(payload):
    configure_v20(payload.get('v20',{}))
    start=time.perf_counter();events=[];candidate=None
    kind=payload.get('kind','merge')
    for T in payload['durations']:
        if payload.get('end_before') is not None and payload['activate']*DT+T>=payload['end_before']-.1:break
        vs=None
        if kind=='cycle':
            tt=np.arange(round(T/DT)+2)*DT
            vs=20+2.5*(smooth(tt/6)-2*smooth((tt-6)/10)+smooth((tt-16)/6))
        candidate,reason=make_plan(payload['x'],payload['theta'],payload['scene'],T,kind,payload['noise'],vs,
            initial_radius=payload['radius'],initial_box=payload['box'],history_bounds=payload.get('history_bounds'),
            fixed_delta=np.zeros(len(payload['x'])) if kind=='cycle' else None,
            active=np.ones(payload['scene']['Q'],bool) if kind=='cycle' else None)  # PHASE-B: was np.ones(3,bool)
        events.append({'T':T,'reason':reason})
        if candidate:break
    finished=time.perf_counter()
    return {'plan':candidate,'events':events,'compute_seconds':finished-start,
            'worker_started_perf':start,'worker_finished_perf':finished,'worker_pid':os.getpid()}

def run_async(cfg):
    configure_v20(cfg.get('v20', {}))
    v20=bool(V20_OPTIONS.get('enabled',False))
    lateral_control=osqp_lateral if cfg.get('lateral_solver')=='osqp' else cvxopt_lateral
    scene=scenario(cfg.get('initial_gap',10.5),cfg.get('boundary',30.));N=scene['N'];Q=scene['Q'];M=N+Q  # PHASE-B: Q was hardcoded 3 (as "N+3")
    steps=round(cfg.get('time_sim',50)/DT);mode_type=cfg.get('execution','replay')
    mass=np.broadcast_to(cfg.get('mass_scale',1.),(M,));tau=np.broadcast_to(cfg.get('tau_scale',1.),(M,))
    truth=np.c_[1/mass,DT/(.7*tau)];noise=(cfg.get('acc_bound',.001),cfg.get('torque_bound',.00015));fraction=cfg.get('disturbance_fraction',.5)
    meas_eps=np.asarray(cfg.get('v20',{}).get('meas_eps',(0.,0.,0.)),dtype=float)
    # PHASE-B: optional seeded randomized disturbance -- see simulation.py's
    # simulate() for the identical rationale (no seed => byte-identical to
    # before this patch; seed given => i.i.d. draws of the same declared
    # amplitude instead of the deterministic sinusoid used below).
    rng=np.random.default_rng(cfg['seed']) if cfg.get('seed') is not None else None
    meas_rng=np.random.default_rng(int(cfg.get('seed',0))+1000003)
    x=np.c_[scene['p0'],scene['v0'],DRAG*scene['v0']**2+GF/truth[:,0]]
    measurement_noise=meas_rng.uniform(-meas_eps,meas_eps,size=(steps+1,M,3)) if v20 and np.any(meas_eps) else np.zeros((steps+1,M,3))
    XH=np.zeros((steps+1,M,3));XH[0]=x+measurement_noise[0]
    bounds=np.repeat(PRIOR[None,:,:],M,axis=0)
    boot=time.perf_counter();plan=None;initial_events=[]
    for T in [12,20,26,30,36]:
        plan,reason=make_plan(XH[0] if v20 else x,bounds,scene,T,'cruise',noise)
        initial_events.append({'T':T,'reason':reason})
        if plan:break
    init_seconds=time.perf_counter()-boot
    if plan is None:return {'status':'initial_not_certified','cfg':cfg,'initial_plan_seconds':init_seconds,'events':initial_events},{}
    pool=None
    if mode_type=='host':
        pool=ProcessPoolExecutor(max_workers=1);pool.submit(ping).result()
    plan_at=0;mergeplan=False;committed=False;starts=np.full(Q,np.nan);done=starts.copy()
    lat=np.zeros((Q,4));steer=np.zeros(Q);mode=np.zeros(Q,int)
    X=np.zeros((steps+1,M,3));X[0]=x;U=np.zeros((steps,M));LH=np.zeros((steps+1,Q,4));BH=np.zeros((steps+1,M,2,2));BH[0]=bounds
    SH=np.zeros((steps,Q));ALOG=np.zeros((steps,M));MH=np.zeros((steps+1,Q),int)
    body=np.zeros((steps*5+1,M,3));body[0,:,0]=x[:,0];body[0,:N,1]=3.5
    errors=np.zeros((steps,M,3));tubes=errors.copy();ratio=np.zeros(steps);timing=np.zeros(steps)
    plan_age=np.zeros(steps);pending_age=np.zeros(steps);tick_lateness=np.zeros(steps);wall=np.zeros(steps+1)
    records=[];pending=None;attempts=0;release=None;stable_since=None;hold_start=None;last_dispatch=-1000
    request=cfg.get('request_time',5.);lead_steps=round(cfg.get('planning_lead_s',3.)/DT)
    stable_hold_required=cfg.get('cruise_cycle',False);cycle_requested=False;accepted=[];hold_count=0
    origin=time.perf_counter();control_misses=0;lateral_failures=0;futures={}
    control_miss_flags=np.zeros(steps,bool);active_control=np.zeros(steps,bool);loop_cost=np.zeros(steps)
    backup_steering=np.zeros((Q,40));fallback_used=np.zeros(steps,bool);control_injected=np.zeros(steps)  # PHASE-B: all seven "3"s above (Q,4)/(Q)/(steps+1,Q,4)/(steps,Q)/(steps+1,Q)/(Q,40) were the hardcoded queue count
    plan_history=[(0,plan)]
    lat_levels=np.zeros((steps,Q),int);lat_slack=np.zeros((steps,Q,3));last_full=None
    previous_lateral=np.zeros((Q,40));activation_checks=[]
    def collect_job(job_id,answer,observed_s):
        rec=records[job_id]
        rec.update(compute_seconds=answer['compute_seconds'],search=answer['events'],
                   worker_started_wall_s=answer['worker_started_perf']-origin,
                   worker_finished_wall_s=answer['worker_finished_perf']-origin,
                   worker_pid=answer['worker_pid'],result_observed_wall_s=observed_s)
        rec['queue_wait_s']=max(0,rec['worker_started_wall_s']-rec['dispatch_wall_s'])
        rec['response_seconds']=rec['worker_finished_wall_s']-rec['dispatch_wall_s']
        rec['result_observed_latency_s']=observed_s-rec['dispatch_wall_s']
    def activate_pending(k,t):
        nonlocal pending,plan,plan_at,mergeplan,release,hold_start
        if pending is None or k!=pending['activate']:return
        rec=records[pending['id']]
        ready=(pending['result'] is not None and
               (rec['worker_finished_wall_s']<=t+1e-10 if mode_type=='host' else pending['ready']<=t+1e-10))
        pp=None if not ready else pending['result']['plan']
        late=(not ready) or pending['force_drop'];allow_contract=not committed or pending['kind']=='cycle'
        contains=False;observed_contains=False;check=None
        if pp is not None:
            ee=x-pp['nom'][0];observed=XH[k]-pp['nom'][0]
            rr=np.array([c['radius'][0] for c in pp['cert']]);bb=pp['boxes'][0]
            norm=np.sqrt(np.einsum('bi,ij,bj->b',ee,P,ee))
            contains=bool(np.all(norm<=rr+1e-8) and np.all(abs(ee)<=bb+1e-8))
            observed_contains=bool(np.all(np.sqrt(np.einsum('bi,ij,bj->b',observed,P,observed))<=rr+measurement_radius(meas_eps)+1e-8)
                and np.all(abs(observed)<=bb+meas_eps+1e-8))
            check={'step':k,'radius_margin_min':float(np.min(rr-norm)),
                   'box_margin_min':float(np.min(bb-abs(ee))),
                   'true_error':ee.tolist(),'initial_radius':rr.tolist(),'initial_box':bb.tolist(),
                   'observed_monitor':observed_contains,'true_containment':contains}
            activation_checks.append(check)
        gate=V20_OPTIONS.get('containment_gate',False) if v20 else True
        gate_ok=(observed_contains if v20 else contains) or not gate
        accepted_now=bool(pp is not None and not late and allow_contract and gate_ok)
        rec.update(deadline_missed=bool(not ready),injected_drop=pending['force_drop'],accepted=accepted_now,
            contains_actual_state=contains,observed_monitor=observed_contains,containment_gate=gate,
            contract_valid=allow_contract,activation_s=t,
            activation_wall_s=time.perf_counter()-origin if mode_type=='host' else None,
            rejection_reason=None if accepted_now else ('deadline' if not ready else 'injected_drop' if late else
                'no_candidate' if pp is None else 'contract_locked' if not allow_contract else 'state_outside_tube'))
        if accepted_now:
            plan=pp;plan_at=k;mergeplan=True;plan_history.append((k,plan))
            accepted.append({'time':t,'T':plan['T'],'kind':pending['kind'],'delta':plan['delta'].tolist(),'gates':plan['gates'],
                'certificate_residuals':[c['residuals'] for c in plan['cert']],
                'history_residuals':[c.get('window_residuals',{}).get('history') for c in plan['cert']]})
            if pending['kind']=='cycle':release=t;hold_start=stable_since
        pending=None
    try:
        for k in range(steps):
            if mode_type=='host':
                delay=origin+k*DT-time.perf_counter()
                if delay>0:time.sleep(delay)
            tick=time.perf_counter();t=k*DT;wall[k]=tick-origin;tick_lateness[k]=max(0,wall[k]-t) if mode_type=='host' else 0.
            if mode_type=='host':
                for jid,fut in list(futures.items()):
                    if fut.done() and records[jid].get('compute_seconds') is None:
                        answer=fut.result();collect_job(jid,answer,time.perf_counter()-origin)
                        if pending is not None and pending['id']==jid:pending['result']=answer
            if pending is not None:
                pending_age[k]=(k-pending['dispatch'])*DT;hold_count+=1
            activate_pending(k,t)
            # Dispatch at most three pre-admission updates, always predicting from
            # the currently executable certificate. No truth is passed to worker.
            issue_merge=t>=request-1e-8 and not committed and attempts<cfg.get('max_attempts',3) and k-last_dispatch>=10
            issue_cycle=stable_hold_required and not cycle_requested and stable_since is not None and t-stable_since>=5-1e-8
            if pending is None and (issue_merge or issue_cycle):
                kind='cycle' if issue_cycle else 'merge';local_lead=round(cfg.get('cycle_lead_s',.3)/DT) if kind=='cycle' else (round(cfg.get('update_lead_s', 2. if cfg.get('v20', {}).get('enabled', False) else 1.)/DT) if mergeplan else lead_steps);activate=k+local_lead
                if v20:
                    latencies=V20_OPTIONS.get('activation_latency',[30,20])
                    local_lead=int(latencies[0 if attempts==0 and kind=='merge' else 1]) if isinstance(latencies,(list,tuple)) else int(latencies)
                    if local_lead<0:raise ValueError('Negative activation latency')
                    activate=k+local_lead
                if activate<steps:
                    nom,_,bb=sample_plan(plan,activate-plan_at)
                    rr=np.array([c['radius'][activate-plan_at] if activate-plan_at<len(c['radius']) else c['terminal_radius'] for c in plan['cert']])
                    old_rr=rr.copy();old_bb=bb.copy();hb=None;theta_received=bounds.copy()
                    frame_indices=np.full(M,k,int);refined_used=np.zeros(M,bool)
                    if v20:
                        ages=np.broadcast_to(cfg.get('uplink_age_steps',0),(M,)).astype(int)
                        missing=np.broadcast_to(cfg.get('uplink_missing',False),(M,)).astype(bool)
                        frame_indices=np.maximum(-1,k-ages);frame_indices[missing]=-1
                        for i,km in enumerate(frame_indices):
                            theta_received[i]=BH[km,i] if km>=0 else plan['theta'][i]
                            if V20_OPTIONS.get('init_refine',False) and km>=plan_at:
                                at_frame=sample_plan(plan,km-plan_at)[0][i]
                                ri,bi=propagate_handover_set(XH[km,i]-at_frame,meas_eps,plan['ref'][i],
                                    plan['cert'][i],km-plan_at,activate-plan_at)
                                rr[i]=min(rr[i],ri);bb[i]=np.minimum(np.minimum(bb[i],bi),EXT*rr[i]);refined_used[i]=True
                        if V20_OPTIONS.get('jerk_mode')=='window':
                            W=int(V20_OPTIONS.get('jerk_window') or 10);hb=np.full((M,W,2),np.nan)
                            for j,hist_k in enumerate(range(activate-W,activate)):
                                if hist_k<0:continue
                                old_at,old=next((at,p) for at,p in reversed(plan_history) if at<=hist_k)
                                for i,km in enumerate(frame_indices):
                                    if hist_k+1<=km:
                                        ahat=(XH[hist_k+1,i,1]-XH[hist_k,i,1])/DT
                                        hb[i,j]=[ahat-2*meas_eps[1]/DT,ahat+2*meas_eps[1]/DT]
                                    else:
                                        hb[i,j]=acceleration_interval(old['ref'][i],old['cert'][i],old['delta'][i],hist_k-old_at,old['noise'][0])
                    payload={'x':nom,'theta':theta_received,'radius':rr,'box':bb,'scene':scene,'noise':noise,'activate':activate,'v20':cfg.get('v20',{}),'history_bounds':hb,
                             'durations':[24.] if kind=='cycle' else cfg.get('durations',[12,14,16,18,20,22,24,26,28,30,34]),
                             'end_before':None if not mergeplan or kind=='cycle' else plan_at*DT+plan['T'],'kind':kind}
                    rec={'id':len(records),'kind':kind,'dispatch_s':t,'deadline_s':activate*DT,'snapshot_s':t,
                         'initial_radius_max':float(rr.max()),'old_radius_max':float(old_rr.max()),
                         'initial_radius':rr.tolist(),'initial_box':bb.tolist(),'frame_indices':frame_indices.tolist(),
                         'refined_used':refined_used.tolist(),'history_bounds':None if hb is None else hb.tolist(),'compute_seconds':None,
                         'dispatch_wall_s':time.perf_counter()-origin if mode_type=='host' else None}
                    records.append(rec);force_drop=len(records)-1 in cfg.get('drop_jobs',[])
                    pending={'id':len(records)-1,'dispatch':k,'activate':activate,'result':None,'force_drop':force_drop,'kind':kind}
                    if mode_type=='host':
                        pending['future']=pool.submit(plan_job,payload);futures[pending['id']]=pending['future']
                    else:
                        answer=plan_job(payload);delay=cfg.get('delay_schedule_s',[None]*20)[len(records)-1] if len(records)<=len(cfg.get('delay_schedule_s',[])) else None
                        delay=answer['compute_seconds'] if delay is None else delay
                        pending.update(result=answer,ready=t+delay)
                        rec.update(compute_seconds=answer['compute_seconds'],replayed_delay_s=delay,ready_observed_s=t+delay,search=answer['events'])
                    attempts+=int(kind=='merge');cycle_requested|=(kind=='cycle');last_dispatch=k
                    if v20 and activate==k:activate_pending(k,t)
            # Worker may be slow; all of these physical/control steps continue.
            control_start=time.perf_counter();ix=k-plan_at;nom,uff,tube=sample_plan(plan,ix)
            xhat=XH[k] if v20 else x
            if mergeplan:
                for c in range(Q):  # PHASE-B: was range(3)
                    if mode[c]==0 and ix>=plan['gates'][c]:mode[c]=1;starts[c]=t;committed=True
            e=xhat-nom;u=uff+e@K;errors[k]=x-nom;tubes[k]=tube
            radii=np.array([c['radius'][ix] if ix<len(c['radius']) else c['terminal_radius'] for c in plan['cert']])
            ratio[k]=np.max(np.sqrt(np.einsum('bi,ij,bj->b',errors[k],P,errors[k]))/(radii+1e-10));plan_age[k]=ix*DT
            new_sequences=backup_steering.copy();solver_failed=False
            active_control[k]=bool(np.any(mode))
            for c in range(Q):  # PHASE-B: was range(3)
                if mode[c]:
                    yr=3.495*smooth((t+(np.arange(40)+1)*DT-starts[c])/4)
                    try:
                        if v20 and V20_OPTIONS.get('lateral_soft',False):
                            new_sequences[c],li=lateral_control(lat[c],xhat[N+c,1],yr,steer[c],return_plan=True,
                                previous_plan=previous_lateral[c],return_info=True)
                            lat_levels[k,c]={'hard':1,'soft':2,'shift':3}[li['level']];lat_slack[k,c]=li['slack_max']
                        else:new_sequences[c]=lateral_control(lat[c],x[N+c,1],yr,steer[c],return_plan=True)
                    except RuntimeError:lateral_failures+=1;solver_failed=True
            injected_delay=cfg.get('control_delay_by_step_s',{}).get(str(k),0.)
            if injected_delay:
                time.sleep(injected_delay);control_injected[k]=injected_delay
            timing[k]=time.perf_counter()-control_start
            if mode_type=='host' and time.perf_counter()>origin+(k+1)*DT:
                control_misses+=1;control_miss_flags[k]=True;fallback_used[k]=True
            elif solver_failed:fallback_used[k]=True
            else:backup_steering=new_sequences
            # A stale single steering value can diverge during repeated misses.
            # Execute the remaining optimized sequence; zeros initialize cruise.
            previous_lateral=backup_steering.copy()
            steer=backup_steering[:,0].copy()
            backup_steering=np.c_[backup_steering[:,1:],backup_steering[:,-1]]
            if rng is None:  # PHASE-B: unchanged deterministic default
                da=fraction*noise[0]*np.sin(.83*t+np.arange(M)*.37);dw=fraction*noise[1]*np.sin(1.37*t+np.arange(M)*.51)
            else:  # PHASE-B: seeded randomized mode, same declared amplitude
                da=fraction*noise[0]*rng.uniform(-1.,1.,M);dw=fraction*noise[1]*rng.uniform(-1.,1.,M)
            xn=np.stack([plant_step(x[i],u[i],truth[i],da[i],dw[i]) for i in range(M)])
            ALOG[k]=(xn[:,1]-x[:,1])/DT
            XH[k+1]=xn+measurement_noise[k+1]
            if v20 and np.any(meas_eps):
                bounds=np.stack([update_set_measured(bounds[i],xhat[i],u[i],XH[k+1,i],*noise,meas_eps=meas_eps) for i in range(M)])
            else:
                bounds=np.stack([update_set(bounds[i],x[i],u[i],xn[i],*noise) for i in range(M)])
            lsub=np.zeros((5,Q,4))  # PHASE-B: was (5,3,4)
            for c in range(Q):  # PHASE-B: was range(3)
                if mode[c]:
                    lat[c],lsub[:,c]=lateral_step(lat[c],x[N+c,1],steer[c])
                    if mode[c]==1 and t+DT>=starts[c]+4-1e-8 and abs(lat[c,0]-3.5)<=.1 and abs(lat[c,2])<=.2:mode[c]=2;done[c]=t+DT
            for ss in range(5):
                at=k*5+ss+1;body[at,:,0]=x[:,0]+(ss+1)*DT/5*x[:,1];body[at,:N,1]=3.5;body[at,N:,1:]=lsub[ss,:,[0,1]].T
            x=xn;X[k+1]=x;LH[k+1]=lat;BH[k+1]=bounds;MH[k+1]=mode;U[k]=u;SH[k]=steer
            stable=np.all(mode==2) and max(abs(x[:,1]-20))<=.05 and max(abs(ALOG[k]))<=.05
            if stable and stable_since is None:stable_since=t+DT
            if not stable and release is None:stable_since=None
            loop_cost[k]=time.perf_counter()-tick
            if mode_type=='host':
                left=origin+(k+1)*DT-time.perf_counter()
                if left>0:time.sleep(left)
            wall[k+1]=time.perf_counter()-origin
        if pending is not None:records[pending['id']]['unresolved_at_end']=True
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
            # Collect every submitted job, including rejected/late candidates.
            # This is after physical execution and cannot change acceptance.
            for jid,fut in futures.items():
                if records[jid].get('compute_seconds') is None:
                    collect_job(jid,fut.result(),time.perf_counter()-origin)
    common=np.full(steps,20.)
    if release is not None:
        z=np.arange(steps)*DT-release;common+=2.5*(smooth(z/6)-2*smooth((z-6)/10)+smooth((z-16)/6))
    costs=[r['compute_seconds'] for r in records if r.get('compute_seconds') is not None]
    W=int(cfg.get('v20',{}).get('jerk_window',10) or 10)
    sampled_jerk=np.diff(ALOG,axis=0)/DT if len(ALOG)>1 else np.zeros((0,M))
    window_jerk=(ALOG[W:]-ALOG[:-W])/(W*DT) if len(ALOG)>W else np.zeros((0,M))
    contained=bool(np.all(BH[:,:,:,0] <= truth[None,:,:] + 1e-9) and np.all(truth[None,:,:] <= BH[:,:,:,1] + 1e-9))
    result={'cfg':cfg,'status':'completed' if np.all(np.isfinite(done)) else 'admission_not_completed','completed':int(np.isfinite(done).sum()),
            'last_done':float(np.nanmax(done)) if np.any(np.isfinite(done)) else None,'merge_start':starts.tolist(),'merge_done':done.tolist(),
            'Jv':float(abs(X[int(round(request/DT)):-1,:N,1]-common[int(round(request/DT)): ,None]).sum()*DT/N),
            'initial_plan_seconds':init_seconds,'planning_jobs':records,'accepted_plans':accepted,'planning_calls':len(records),
            'activation_checks':activation_checks,'events':initial_events,
            'lateral_levels_count':{str(j):int(np.sum(lat_levels==j)) for j in (1,2,3)},
            'lateral_slack_max':lat_slack.max(axis=(0,1)).tolist(),
            'planning_deadline_misses':sum(r.get('deadline_missed',False) for r in records),
            'planning_over_0_1_fraction':float(np.mean(np.array(costs)>.1)) if costs else None,
            'planning_mean_s':float(np.mean(costs)) if costs else None,'planning_p95_s':float(np.quantile(costs,.95)) if costs else None,
            'planning_p99_s':float(np.quantile(costs,.99)) if costs else None,'planning_max_s':max(costs,default=0.),
            'control_mean_s':float(timing.mean()),'control_p95_s':float(np.quantile(timing,.95)),
            'control_p99_s':float(np.quantile(timing,.99)),'control_max_s':float(timing.max()),
            'control_deadline_misses':control_misses,'control_over_0_1_fraction':float(np.mean(timing>.1)),
            'lateral_fallback':'previous_optimized_sequence_suffix',
            'lateral_fallback_samples':int(fallback_used.sum()),
            'injected_control_delay_samples':int(np.count_nonzero(control_injected)),
            'planning_timing_count':len(costs),'control_samples':steps,
            'active_control_samples':int(active_control.sum()),
            'control_active_p95_s':float(np.quantile(timing[active_control],.95)) if np.any(active_control) else None,
            'loop_max_s':float(loop_cost.max()),
            'old_plan_hold_samples':hold_count,'max_pending_age_s':float(pending_age.max()),'max_plan_age_s':float(plan_age.max()),
            'lateral_solver_failures':lateral_failures,'wall_elapsed_s':wall[-1],'tick_lateness_max_s':float(tick_lateness.max()),
            'tube_ratio_max':float(ratio.max()),'parameters_contained':contained,
            'sampled_jerk_max_s_inv3':float(np.max(np.abs(sampled_jerk))) if sampled_jerk.size else 0.,
            'window_jerk_max_s_inv3':float(np.max(np.abs(window_jerk))) if window_jerk.size else 0.,
            'cycle_release':release,'release_stable_since':hold_start,
            'stable_hold_before_release':None if release is None else release-hold_start,
            'host_process':os.getpid(),'python':platform.python_version()}
    arrays={'time':np.arange(steps+1)*DT,'state':X,'input':U,'lateral':LH,'bounds':BH,'truth':truth,'acceleration':ALOG,
            'error':errors,'tube_box':tubes,'tube_ratio':ratio,'timing':timing,'body_poses':body,'mode':MH,'steering':SH,
            'common_speed':common,'front':scene['front'],'rear':scene['rear'],'plan_age':plan_age,'pending_age':pending_age,
            'tick_lateness':tick_lateness,'wall_time':wall,'loop_cost':loop_cost,
            'control_deadline_miss':control_miss_flags,'active_control':active_control,
            'lateral_fallback':fallback_used,'control_delay_injected':control_injected}
    if v20:arrays.update(measured_state=XH,measurement_noise=measurement_noise,lateral_levels=lat_levels,lateral_slack=lat_slack)
    return result,arrays

def save_run(cfg,out):
    out=Path(out);out.parent.mkdir(exist_ok=True,parents=True)
    if out.with_suffix('.json').exists():raise FileExistsError(out)
    result,arrays=run_async(cfg)
    result['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    out.with_suffix('.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    if arrays:np.savez_compressed(out.with_suffix('.npz'),**arrays)
    print(json.dumps({k:v for k,v in result.items() if k not in ['planning_jobs','source_hashes','accepted_plans']},ensure_ascii=False),flush=True)

if __name__=='__main__':save_run(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig')),sys.argv[2])
