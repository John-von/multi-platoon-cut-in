"""Bounded-parameter reference safety filter for the frozen v16 longitudinal plant.

No true plant parameters enter planning, identification or control. Reference
polynomials, norm bounds and interval supports are sufficient inner conditions.
"""
import json,itertools
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[1]
MET=json.loads((ROOT/'metric.json').read_text())
K=np.array(MET['K']);P=np.array(MET['P']);PINV=np.linalg.inv(P)
L=np.linalg.cholesky(P).T;EXT=np.sqrt(np.diag(PINV))
DT=.1;DRAG=1/1500;GF=.098;TSCALE=.96/(1500*.35)
UMIN=-2000*TSCALE;UMAX=3000*TSCALE;VMIN=5.;VMAX=23.
PRIOR=np.array(MET['parameter_box']);CORNERS=np.array(list(itertools.product([-1.,1.],repeat=3)))

# v20 is opt-in.  With all options disabled the original v19 code path is
# preserved so that the T0 comparison has a well-defined reference.
V20_OPTIONS = {
    'enabled': False,
    'meas_eps': np.zeros(3),
    'init_refine': False,
    'jerk_mode': 'sampled',
    'jerk_window': None,
}

def configure_v20(options=None):
    """Set process-local v20 switches; no v19 file is modified."""
    base = {'enabled': False, 'meas_eps': np.zeros(3), 'init_refine': False,
            'jerk_mode': 'sampled', 'jerk_window': None}
    if options:
        base.update(options)
    base['meas_eps'] = np.asarray(base['meas_eps'], dtype=float)
    V20_OPTIONS.clear()
    V20_OPTIONS.update(base)
    return V20_OPTIONS

def measurement_radius(eps):
    eps = np.asarray(eps, dtype=float)
    pts = np.array(list(itertools.product([-1., 1.], repeat=3)), dtype=float) * eps
    return float(np.sqrt(np.einsum('bi,ij,bj->b', pts, P, pts)).max())

def handover_initial_set(old_radius, old_box, delta, meas_eps):
    """Safe zero-age intersection used by the v20 handover path.

    For a positive activation lead the caller keeps the old certified envelope
    and propagates it with the old plan; this helper is deliberately limited to
    the zero-age case so it cannot silently under-approximate a future set.
    """
    eps=np.asarray(meas_eps,dtype=float)
    d=np.asarray(delta,dtype=float)
    rm=float(np.sqrt(d@P@d)+measurement_radius(eps))
    bm=np.abs(d)+eps
    r=float(min(float(old_radius),rm))
    b=np.minimum(np.asarray(old_box,dtype=float),bm)
    b=np.minimum(b,EXT*r)
    return r,b

def propagate_handover_set(delta, meas_eps, ref, cert, start, stop):
    """Propagate a received frame with the OLD plan's certified forcing.

    start and stop are indices relative to that plan; stop is the activation
    index. No measured or true state after start is used.
    """
    if start < 0 or stop < start:
        raise ValueError('Frame must belong to the executing plan')
    eps=np.asarray(meas_eps,dtype=float);d=np.asarray(delta,dtype=float)
    rho,Aabs,_,_,_,_=bounds_metric(ref['theta'])
    radius=float(np.sqrt(d@P@d)+measurement_radius(eps))
    box=np.minimum(np.abs(d)+eps,EXT*radius)
    for j in range(int(start),int(stop)):
        forcing=cert['forcing_radius'][j] if j < ref['M'] else cert['terminal_forcing_radius']
        force_box=cert['forcing_box'][j] if j < ref['M'] else cert['terminal_forcing_box']
        radius=rho*radius+forcing
        box=np.minimum(Aabs@box+force_box,EXT*radius)
    return radius,box

def update_set_measured(bounds, xhat, u, xhat_next, acc_bound=.001,
                        torque_bound=.00015, meas_eps=None):
    """Set update using measured states with the v20 strip enlargement."""
    eps = np.zeros(3) if meas_eps is None else np.asarray(meas_eps, dtype=float)
    mu_bar = float(np.max(PRIOR[0]))
    da = 2*eps[1]/DT + mu_bar*eps[2] + mu_bar*DRAG*(2*VMAX*eps[1] + eps[1]**2)
    dz = (1.0 + float(np.max(np.abs(1.0-PRIOR[1]))))*eps[2]
    return update_set(bounds, xhat, u, xhat_next, acc_bound + da,
                      torque_bound + dz)

def plant_step(x,u,theta,da=0.,dw=0.):
    mu,lam=theta
    p,v,t=x
    return np.array([p+DT*v,v+DT*(mu*(t-DRAG*v*v)-GF+da),t+lam*(u-t)+dw])

def update_set(bounds,x,u,xnext,acc_bound=.001,torque_bound=.00015):
    """Intersect scalar strips; bounds are physical process/model residuals."""
    out=bounds.copy()
    coeff=[x[2]-DRAG*x[1]**2,u-x[2]]
    obs=[(xnext[1]-x[1])/DT+GF,xnext[2]-x[2]]
    noise=[acc_bound,torque_bound]
    for j,(f,y,d) in enumerate(zip(coeff,obs,noise)):
        if abs(f)>1e-10:
            ends=sorted([(y-d)/f,(y+d)/f])
            out[j]=[max(out[j,0],ends[0]-1e-11),min(out[j,1],ends[1]+1e-11)]
    if np.any(out[:,0]>out[:,1]):raise ValueError('Empty parameter set: declared residual bounds invalid')
    return out

@lru_cache(maxsize=4096)
def metric_bounds(key):
    box=np.array(key).reshape(2,2)
    mats=[]
    for mu,lam,vs in itertools.product(box[0],box[1],[10.,46.]):
        mats.append(np.array([[1,DT,0],[0,1-DT*mu*DRAG*vs,DT*mu],
                              [lam*K[0],lam*K[1],1-lam+lam*K[2]]]))
    rho=max(np.linalg.norm(L@a@np.linalg.inv(L),2) for a in mats)+1e-10
    # Interval arithmetic encloses the whole state segment, not only vertices
    # of a nonlinear gradient. For v in [5,23] and endpoint |a|<=2, the
    # concavity correction along a state segment is <= mu*c*18^2/4 < 0.061.
    # Thus [-3,3] encloses the noise-free acceleration on every segment.
    def mul(a,b):
        z=np.outer(a,b);return np.array([z.min(),z.max()])
    def sc(a,s):return np.sort(np.asarray(a)*s)
    mu=box[0];lam=box[1];v=np.array([5.,23.]);acc=np.array([-3.,3.])
    av=sc(mul(mu,v),-2*DRAG);at=mu
    fac=sc(mul(mu,lam),1/DT)
    vv=2*v+2*DT*acc
    jp=sc(fac,K[0])
    jv=sc(fac,K[1])+sc(mul(mu,2*acc+mul(vv,av)),-DRAG)
    jt=sc(fac,K[2]-1)+sc(mul(mul(mu,mu),vv),-DRAG)
    def support(g):
        g=np.array(g);lo=g[:,0];hi=g[:,1]
        pts=(hi+lo)/2+CORNERS*(hi-lo)/2
        return float(np.sqrt(np.einsum('bi,ij,bj->b',pts,PINV,pts)).max()),np.maximum(abs(lo),abs(hi))
    an,ab=support([[0,0],av,at]);jn,jb=support([jp,jv,jt])
    return rho,np.max(abs(np.array(mats)),axis=0),an,ab,jn,jb

def bounds_metric(theta):return metric_bounds(tuple(theta.ravel()))

@lru_cache(maxsize=128)
def interpolation(M):
    tt=np.arange(M+3)/M
    V=np.vander(tt,6,increasing=True)
    C=V[[0,1,2,M,M+1,M+2]]
    return V@np.linalg.inv(C)

def ref_coeff(x0,T,theta,common_speed=None):
    """Reference matches measured p/v/torque at sample 0 exactly.

    Polynomial coefficients are constant, linear and quadratic in displacement.
    Six discrete endpoint conditions handle the actuator state, including a
    known time-varying common speed sequence. No continuous/discrete mixing.
    """
    M=round(T/DT);mu,lam=theta.mean(axis=1)
    speed=np.full(M+2,20.) if common_speed is None else np.asarray(common_speed[:M+2])
    if len(speed)!=M+2 or not np.allclose(speed[-3:],speed[-1],atol=1e-10,rtol=0):
        raise ValueError('Reference must include M+2 speeds and end at constant cruise')
    common=np.r_[0.,DT*np.cumsum(speed)]
    nominal_a0=mu*(x0[2]-DRAG*x0[1]**2)-GF
    targets=np.array([0.,DT*x0[1]-common[1],2*DT*x0[1]+DT**2*nominal_a0-common[2],0.,0.,0.])
    interp=interpolation(M);s=interp@np.array([0.,0.,0.,1.,1.,1.])
    base=x0[0]+common+interp@targets
    pos=np.c_[base,s,np.zeros(M+3)]
    vel=np.diff(pos,axis=0)/DT;acc=np.diff(vel,axis=0)/DT
    tq=acc/mu;tq[:,0]+=GF/mu
    tq[:,0]+=DRAG*vel[:M+1,0]**2
    tq[:,1]+=2*DRAG*vel[:M+1,0]*vel[:M+1,1]
    tq[:,2]+=DRAG*vel[:M+1,1]**2
    inp=tq[:-1]+np.diff(tq,axis=0)/lam
    return {'p':pos[:M+1],'v':vel[:M+1],'a':acc,'t':tq,'q':inp,
            'theta':theta.copy(),'center':np.array([mu,lam]),'T':T,'x0':np.array(x0),'M':M}

def val(c,d):return c[...,0]+c[...,1]*d+c[...,2]*d*d

def extrema(c,interval):
    lo,hi=interval;v0=val(c,lo);v1=val(c,hi)
    low=np.minimum(v0,v1);high=np.maximum(v0,v1)
    q=c[...,2];root=np.divide(-c[...,1],2*q,out=np.zeros_like(q),where=abs(q)>1e-16)
    use=(abs(q)>1e-16)&(root>lo)&(root<hi)
    v=val(c,root)
    return np.where(use,np.minimum(low,v),low),np.where(use,np.maximum(high,v),high)

def absmax(c,interval):
    lo,hi=extrema(c,interval);return np.maximum(abs(lo),abs(hi))

def quadratic_from_values(f):
    zero=f(0.);plus=f(1.);minus=f(-1.)
    return np.stack([zero,(plus-minus)/2,(plus+minus)/2-zero],axis=-1)

def _certificate_v19(ref,interval,acc_noise=.001,torque_noise=.00015,initial_radius=0.,initial_box=None):
    """Certify every displacement in a closed interval under one feedback K.

    Envelope used for a range may be conservative. Direct candidate checks use
    singleton intervals. Terminal invariant ball is checked separately.
    """
    theta=ref['theta'];mu0,lam0=ref['center'];M=ref['M']
    rho,Aabs,an,ab,jn,jb=bounds_metric(theta)
    force=ref['a'].copy();force[:,0]+=GF;force/=mu0
    qdiff=ref['q']-ref['t'][:-1]
    wv=DT*(np.max(abs(theta[0]-mu0))*absmax(force[:-1],interval)+acc_noise)
    wt=np.max(abs(theta[1]-lam0))*absmax(qdiff,interval)+torque_noise
    if V20_OPTIONS.get('enabled', False) and np.any(V20_OPTIONS.get('meas_eps', np.zeros(3))):
        wt = wt + float(np.max(theta[1])) * float(np.abs(K) @ np.asarray(V20_OPTIONS['meas_eps'], dtype=float))
    d=np.sqrt(P[1,1]*wv**2+2*abs(P[1,2])*wv*wt+P[2,2]*wt**2)
    rad=np.zeros(M+1);box=np.zeros((M+1,3))
    rad[0]=initial_radius
    box[0]=EXT*initial_radius if initial_box is None else np.minimum(initial_box,EXT*initial_radius)
    for k in range(M):
        rad[k+1]=rho*rad[k]+d[k]
        box[k+1]=np.minimum(Aabs@box[k]+[0,wv[k],wt[k]],EXT*rad[k+1])
    speedlo,speedhi=extrema(ref['v'],interval)
    inputlo,inputhi=extrema(ref['q'],interval)
    umargin=np.minimum(box[:-1]@abs(K),np.sqrt(K@PINV@K)*rad[:-1])
    amargin=np.minimum(box@ab,an*rad)
    noise_jerk=2*acc_noise/DT+theta[0,1]*torque_noise/DT+theta[0,1]*DRAG*(2*(23+DT*2)*acc_noise+DT*acc_noise**2)
    jmargin=np.minimum(box[:-1]@jb,jn*rad[:-1])+noise_jerk
    amin=np.full(M+1,np.inf);amax=-amin.copy();jmin=np.full(M,np.inf);jmax=-jmin.copy()
    for mu,lam in itertools.product(theta[0],theta[1]):
        acoef=force*mu;acoef[:,0]-=GF
        low,high=extrema(acoef,interval);amin=np.minimum(amin,low);amax=np.maximum(amax,high)
        def jf(delta):
            v=val(ref['v'][:-1],delta);a=val(acoef[:-1],delta)
            return mu*lam/DT*val(qdiff,delta)-mu*DRAG*(2*v*a+DT*a*a)
        low,high=extrema(quadratic_from_values(jf),interval);jmin=np.minimum(jmin,low);jmax=np.maximum(jmax,high)
    # J is cubic in inverse mass. Endpoint evaluation alone is insufficient.
    # The linear-interpolation remainder is <= width(mu)^2 * sup|J''|/8.
    fm=absmax(force[:-1],interval);vm=absmax(ref['v'][:-1],interval)
    curvature=4*DRAG*vm*fm+4*DT*DRAG*fm*GF+6*DT*DRAG*theta[0,1]*fm**2
    interpolation_error=(theta[0,1]-theta[0,0])**2*curvature/8
    jmin-=interpolation_error;jmax+=interpolation_error
    residuals={'speed':float(max(np.max(5-speedlo+box[:,1]),np.max(speedhi+box[:,1]-23))),
               'acceleration':float(max(np.max(-2-amin+amargin+acc_noise),np.max(amax+amargin+acc_noise-2))),
               'jerk':float(max(np.max(-1-jmin+jmargin),np.max(jmax+jmargin-1))),
               'input':float(max(np.max(UMIN-inputlo+umargin),np.max(inputhi+umargin-UMAX)))}
    # Constant cruise terminal continuation, with the same center and K.
    vend=val(ref['v'][-1],sum(interval)/2);tend=val(ref['t'][-1],sum(interval)/2)
    fv=DT*(np.max(abs(theta[0]-mu0))*abs(tend-DRAG*vend**2)+acc_noise)
    ft=torque_noise
    measurement_support=0.
    if V20_OPTIONS.get('enabled', False):
        measurement_support=float(np.abs(K) @ np.asarray(V20_OPTIONS.get('meas_eps',np.zeros(3))))
        ft+=float(np.max(theta[1]))*measurement_support
    dinf=np.sqrt(P[1,1]*fv**2+2*abs(P[1,2])*fv*ft+P[2,2]*ft**2)
    terminal=max(rad[-1],dinf/(1-rho))
    # q=t for constant cruise, so no parameter-lag forcing remains.
    arange=theta[0]*(tend-DRAG*vend**2)-GF
    jr=[]
    for mu in theta[0]:
        a=mu*(tend-DRAG*vend**2)-GF
        jr.append(-mu*DRAG*(2*vend*a+DT*a*a))
    finf=abs(tend-DRAG*vend**2)
    terminal_curvature=4*DRAG*abs(vend)*finf+4*DT*DRAG*finf*GF+6*DT*DRAG*theta[0,1]*finf**2
    terminal_jerk_error=(theta[0,1]-theta[0,0])**2*terminal_curvature/8
    terminal_res=max(5-vend+EXT[1]*terminal,vend+EXT[1]*terminal-23,
                     max(abs(arange))+an*terminal+acc_noise-2,
                     max(abs(np.array(jr)))+terminal_jerk_error+jn*terminal+noise_jerk-1,
                     UMIN-tend+np.sqrt(K@PINV@K)*terminal,tend+np.sqrt(K@PINV@K)*terminal-UMAX)
    residuals['terminal']=float(terminal_res)
    out={'pass':max(residuals.values())<=1e-7,'residuals':residuals,'radius':rad,'box':box,
         'terminal_radius':terminal,'terminal_box':EXT*terminal,'rho':rho}
    if V20_OPTIONS.get('enabled',False):
        out.update(forcing_radius=d,forcing_box=np.column_stack([np.zeros(M),wv,wt]),
                   terminal_forcing_radius=float(dinf),terminal_forcing_box=np.array([0.,fv,ft]))
    return out

def certificate(ref, interval, acc_noise=.001, torque_noise=.00015,
                initial_radius=0., initial_box=None, history_bounds=None):
    if not V20_OPTIONS.get('enabled', False):
        return _certificate_v19(ref, interval, acc_noise, torque_noise,
                                initial_radius, initial_box)
    eps=np.asarray(V20_OPTIONS.get('meas_eps',np.zeros(3)),dtype=float)
    out=_certificate_v19(ref,interval,acc_noise,torque_noise,initial_radius,initial_box)
    theta=ref['theta'];mu0=ref['center'][0];M=ref['M']
    rho,Aabs,an,ab,jn,jb=bounds_metric(theta)
    support=float(np.abs(K)@eps);kk=float(np.sqrt(K@PINV@K))
    out['residuals']['input']+=support
    vend=val(ref['v'][-1],sum(interval)/2);tend=val(ref['t'][-1],sum(interval)/2)
    term=out['terminal_radius']
    terminal_input=max(UMIN-tend+kk*term+support,tend+kk*term+support-UMAX)
    out['input_measurement_support']=support
    out['terminal_input_residual']=float(terminal_input)
    out['residuals']['terminal']=float(max(out['residuals']['terminal'],terminal_input))
    if V20_OPTIONS.get('jerk_mode')=='window':
        W=int(V20_OPTIONS.get('jerk_window') or 10)
        if W<1:raise ValueError('jerk_window must be positive')
        force=ref['a'].copy();force[:,0]+=GF;force/=mu0
        force_ext=np.vstack([force,np.repeat(force[-1:],W,axis=0)])
        margin=np.minimum(out['box']@ab,an*out['radius'])
        terminal_margin=min(float(out['terminal_box']@ab),an*term)
        margin_ext=np.r_[margin,np.full(W,terminal_margin)]
        internal=-np.inf
        for mu in theta[0]:
            acc=force_ext*mu;acc[:,0]-=GF
            lo,hi=extrema(acc[W:]-acc[:-W],interval)
            internal=max(internal,float(np.max((np.maximum(abs(lo),abs(hi))+
                margin_ext[W:]+margin_ext[:-W]+2*acc_noise)/(W*DT)))-1.)
        cross=-np.inf
        if history_bounds is not None:
            hb=np.asarray(history_bounds,dtype=float)
            if hb.ndim!=2 or hb.shape[1]!=2 or len(hb)>W:
                raise ValueError('history_bounds must have shape (n<=W,2)')
            # The last history entry is k_a-1, paired with new index W-1.
            # A W-entry history also verifies the switch endpoint j=0.
            for j,(low,high) in enumerate(hb,start=W-len(hb)):
                if np.isnan(low) and np.isnan(high):continue
                if not np.isfinite([low,high]).all() or low>high:
                    raise ValueError('Invalid history interval')
                for mu in theta[0]:
                    ac=force_ext[j]*mu;ac=ac.copy();ac[0]-=GF
                    alo,ahi=extrema(ac,interval)
                    bound=max(abs(float(ahi)-low),abs(float(alo)-high))
                    cross=max(cross,(bound+margin_ext[j]+acc_noise)/(W*DT)-1.)
        terminal_window=(2*terminal_margin+2*acc_noise)/(W*DT)-1.
        out['window_residuals']={'within_and_terminal_crossing':float(internal),
            'history':float(cross) if np.isfinite(cross) else None,'terminal':float(terminal_window)}
        out['residuals']['jerk']=float(max(internal,cross,terminal_window))
        arange=theta[0]*(tend-DRAG*vend**2)-GF
        out['residuals']['terminal']=float(max(5-vend+EXT[1]*term,
            vend+EXT[1]*term-23,max(abs(arange))+an*term+acc_noise-2,terminal_input))
    elif support:
        # Sampled-jerk ablation: fresh measurement noise acts through lambda K eta.
        extra=theta[0,1]*theta[1,1]*support/DT
        out['residuals']['jerk']+=extra
        # A conservative extension of the old combined terminal residual.
        out['residuals']['terminal']+=extra
    out['pass']=bool(max(out['residuals'].values())<=1e-7)
    return out


def acceleration_interval(ref,cert,delta,index,acc_noise):
    """Certified true acceleration interval at a past or future old-plan step."""
    _,_,an,ab,_,_=bounds_metric(ref['theta'])
    if index<0:raise ValueError('Negative old-plan index')
    j=min(index,ref['M'])
    force=(float(val(ref['a'][j],delta))+GF)/ref['center'][0]
    nominal=ref['theta'][0]*force-GF
    if index<=ref['M']:
        margin=min(float(cert['box'][j]@ab),an*float(cert['radius'][j]))
    else:margin=min(float(cert['terminal_box']@ab),an*float(cert['terminal_radius']))
    return np.array([nominal.min()-margin-acc_noise,nominal.max()+margin+acc_noise])


def capacity(ref,acc_noise=.001,torque_noise=.00015,limit=40.,initial_radius=0.,initial_box=None,history_bounds=None):
    if not certificate(ref,[0,0],acc_noise,torque_noise,initial_radius,initial_box,history_bounds)['pass']:return None
    ends=[]
    for sign in [-1,1]:
        lo=0.;hi=limit
        for it in range(13):
            mid=(lo+hi)/2;interval=sorted([0.,sign*mid])
            if certificate(ref,interval,acc_noise,torque_noise,initial_radius,initial_box,history_bounds)['pass']:lo=mid
            else:hi=mid
        ends.append(sign*lo)
    return np.array(ends)

def allocation(positions,queue_start,front,rear,caps,nominal_gap=12.,boundary=12.,objective='l1'):
    N=len(positions);Q=len(queue_start);preferred=np.zeros(N);owner=np.zeros(N,dtype=int)
    lb=np.full(Q,-np.inf);ub=np.full(Q,np.inf)
    for q in range(Q):
        ix=np.arange(queue_start[q],queue_start[q+1] if q+1<Q else N)
        tar=-nominal_gap*np.arange(len(ix));tar[ix>=rear[q]]-=nominal_gap
        loc=front[q]-ix[0]
        tar+=(positions[front[q]]+positions[rear[q]])/2-(tar[loc]+tar[loc+1])/2
        preferred[ix]=tar-positions[ix];owner[ix]=q
        lb[q]=max(.01-preferred[ix[ix<=front[q]]]);ub[q]=min(-.01-preferred[ix[ix>=rear[q]]])
        lb[q]=max(lb[q],np.max(caps[ix,0]-preferred[ix]));ub[q]=min(ub[q],np.min(caps[ix,1]-preferred[ix]))
    M=N+Q;Bmat=np.zeros((M,Q));Bmat[np.arange(N),owner]=1
    Bmat[N:]=np.eye(Q)
    cp=(positions[front]+preferred[front]+positions[rear]+preferred[rear])/2
    # Cut-in endpoint constraint is supplied relative to each source-lane anchor.
    cpref=cp-caps[N:,2]
    for q in range(Q):lb[q]=max(lb[q],caps[N+q,0]-cpref[q]);ub[q]=min(ub[q],caps[N+q,1]-cpref[q])
    pref=np.r_[preferred,cpref]
    if np.any(lb>ub):return None
    rows=[];rhs=[]
    for q in range(Q-1):
        head=queue_start[q+1];tail=head-1;row=np.zeros(Q+M);row[q]=-1;row[q+1]=1
        rows.append(row);rhs.append(positions[tail]-positions[head]+preferred[tail]-preferred[head]-boundary)
    rows.extend(np.c_[Bmat,-np.eye(M)]);rhs.extend(-pref)
    rows.extend(np.c_[-Bmat,-np.eye(M)]);rhs.extend(pref)
    weights=np.r_[np.ones(N),np.zeros(Q)]
    ans=linprog(np.r_[np.zeros(Q),weights],A_ub=np.array(rows),b_ub=np.array(rhs),
                bounds=list(zip(lb,ub))+[(0,None)]*M,method='highs')
    if not ans.success:return None
    return {'delta':pref+Bmat@ans.x[:Q],'beta':ans.x[:Q],'preferred':pref,'owner':owner,
            'bounds':np.c_[lb,ub],'objective':float(ans.fun)}

def evaluate_ref(ref,delta):
    return np.column_stack([val(ref[c],delta) for c in ['p','v','t']]),val(ref['q'],delta)

