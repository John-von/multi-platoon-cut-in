"""Independent trajectory-level checks of certificate and physical equations."""
import json,itertools
import numpy as np
from robust_core import *

rng=np.random.default_rng(18092026)
tested=0;max_ratio=0.;max_model_error=0.;max_state_match=0.
for T,interval,theta in [(30,[-12,15],PRIOR),(14,[-12,12],np.array([[.997,1.003],[.139,.147]]))]:
    x0=np.array([-50.,20.,DRAG*400+GF])
    ref=ref_coeff(x0,T,theta);cert=certificate(ref,interval)
    assert cert['pass'],cert['residuals']
    for trial in range(120):
        mu,lam=theta[:,0]+rng.random(2)*(theta[:,1]-theta[:,0])
        if trial<4:mu,lam=list(itertools.product(theta[0],theta[1]))[trial]
        delta=rng.uniform(*interval);nom,feed=evaluate_ref(ref,delta)
        max_state_match=max(max_state_match,float(max(abs(nom[0]-x0))))
        x=x0.copy();aa=[]
        # The independent loop uses physical torque and mass, not plant_step.
        for k in range(len(feed)):
            err=x-nom[k];u=feed[k]+K@err
            da=rng.uniform(-.001,.001);dw=rng.uniform(-.00015,.00015)
            m=1500/mu;tau=DT/lam;torque=x[2]/TSCALE;command=u/TSCALE
            accel=(.96*torque/.35-x[1]**2-m*.098)/m+da
            tqnext=torque+DT/tau*(command-torque)+dw/TSCALE
            xn=np.array([x[0]+DT*x[1],x[1]+DT*accel,tqnext*TSCALE])
            max_model_error=max(max_model_error,float(max(abs(xn-plant_step(x,u,[mu,lam],da,dw)))))
            er=xn-nom[k+1];rr=np.sqrt(er@P@er)
            max_ratio=max(max_ratio,float(rr/(cert['radius'][k+1]+1e-14)))
            assert rr<=cert['radius'][k+1]+1e-7
            assert np.all(abs(er)<=cert['box'][k+1]+1e-7)
            assert 5-1e-7<=xn[1]<=23+1e-7 and abs(accel)<=2+1e-7 and UMIN-1e-7<=u<=UMAX+1e-7
            if aa:assert abs(accel-aa[-1])/DT<=1+1e-7
            aa.append(accel);x=xn;tested+=1
        # Verify invariant terminal continuation against new disturbances.
        terminal=cert['terminal_radius'];n=nom[-1].copy()
        for k in range(100):
            e=x-n;u=n[2]+K@e
            x=plant_step(x,u,[mu,lam],rng.uniform(-.001,.001),rng.uniform(-.00015,.00015))
            n[0]+=DT*n[1]
            assert np.sqrt((x-n)@P@(x-n))<=terminal+1e-7
        tested+=100

# Time-varying reference: discrete identities and initial measured torque.
T=24.;tt=np.arange(round(T/DT)+2)*DT
sm=lambda z:10*np.clip(z,0,1)**3-15*np.clip(z,0,1)**4+6*np.clip(z,0,1)**5
vs=20+2*(sm(tt/6)-2*sm((tt-6)/10)+sm((tt-16)/6))
x0=np.array([17.,20.3,DRAG*20.3**2+GF+.02])
ref=ref_coeff(x0,T,np.array([[.999,1.001],[.142,.144]]),vs)
nom,u=evaluate_ref(ref,4.)
assert max(abs(nom[0]-x0))<1e-8
assert np.max(abs(np.diff(nom[:,0])-DT*nom[:-1,1]))<1e-8
assert abs(nom[-1,1]-20)<1e-7
assert abs(nom[-1,2]-(DRAG*400+GF))<1e-7

report={'independent_transition_checks':tested,'certificate_trajectories':240,
        'maximum_tube_norm_ratio':max_ratio,'physical_model_max_error':max_model_error,
        'initial_state_max_error':max_state_match,'time_varying_discrete_reference_pass':True,
        'terminal_continuation_pass':True,'note':'Random/vertex trajectory tests complement, not replace, the analytical sufficient conditions.'}
(ROOT/'qa/core_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
