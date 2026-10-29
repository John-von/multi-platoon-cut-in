"""Pure trajectory audit extracted from the applied Phase-B patch.
No report generation or filesystem writes occur on import.
"""
import itertools
import numpy as np

def sat_audit(data,a):
    x=a['state'];dt=.1
    mode=a['mode']
    # PHASE-B: N=22 and range(3) below were hardcoded for the one 25-vehicle/
    # 3-queue scenario every existing result file happens to use. Both are
    # now read from the arrays every simulate()/run_async() output already
    # saves, so this audit works unchanged on those same old files (mode.
    # shape[1] is 3 for every one of them, exactly as before) and on any new
    # Q/N scalability-sweep result without further changes.
    Q=mode.shape[1];N=x.shape[1]-Q
    acc=np.diff(x[:,:,1],axis=0)/dt;j=np.diff(acc,axis=0)/dt
    gaps=x[:,:N-1,0]-x[:,1:N,0]
    mg=[]
    for q in range(Q):  # PHASE-B: was range(3)
        ids=np.flatnonzero(mode[:,q]>0)
        if len(ids):
            at=max(0,int(ids[0])-1);f=int(a['front'][q]);r=int(a['rear'][q]);mg += [x[at:,f,0]-x[at:,N+q,0],x[at:,N+q,0]-x[at:,r,0]]
    torque=x[:,:,2]*(1500*.35)/.96;cmd=a['input']*(1500*.35)/.96
    physical={'v_min':float(x[:,:,1].min()),'v_max':float(x[:,:,1].max()),'a_max':float(abs(acc).max()),
      'j_max':float(abs(j).max()),'input_min':float(cmd.min()),'input_max':float(cmd.max()),
      'ordinary_gap_min':float(gaps.min()),'merge_gap_min':float(min((z.min() for z in mg),default=np.nan)),
      'lateral_y_min':float(a['lateral'][:,:,0].min()),'lateral_y_max':float(a['lateral'][:,:,0].max()),
      'steering_max':float(abs(a['steering']).max()),'steering_step_max':float(abs(np.diff(np.vstack([np.zeros((1,Q)),a['steering']]),axis=0)).max())}  # PHASE-B: was np.zeros((1,3))
    overlap=0;pairchecks=0;minsep=float('inf')
    if 'body_poses' in a:
        poses=a['body_poses'];yaw=poses[:,:,2];ex=np.stack([np.cos(yaw),np.sin(yaw)],axis=-1);ey=np.stack([-np.sin(yaw),np.cos(yaw)],axis=-1)
        for i,jj in itertools.combinations(range(poses.shape[1]),2):
            dp=poses[:,i,:2]-poses[:,jj,:2];axes=np.stack([ex[:,i],ey[:,i],ex[:,jj],ey[:,jj]],axis=1)
            projection=abs(np.einsum('ki,kai->ka',dp,axes));ri=2.4*abs(np.einsum('ki,kai->ka',ex[:,i],axes))+.9*abs(np.einsum('ki,kai->ka',ey[:,i],axes));rj=2.4*abs(np.einsum('ki,kai->ka',ex[:,jj],axes))+.9*abs(np.einsum('ki,kai->ka',ey[:,jj],axes))
            sep=(projection-ri-rj).max(axis=1);minsep=min(minsep,float(sep.min()));overlap+=int(np.count_nonzero(sep<-1e-8));pairchecks+=len(sep)
    contained=bool(np.all(a['bounds'][...,0]<=a['truth'][None,...]+1e-9) and np.all(a['truth'][None,...]<=a['bounds'][...,1]+1e-9))
    physical.update({'body_separating_margin':minsep,'body_overlap_samples':overlap,'body_pair_checks':pairchecks,
                     'parameters_contained':contained,'tube_ratio_max':float(a['tube_ratio'].max())})
    passed=physical['v_min']>=5-1e-6 and physical['v_max']<=23+1e-6 and physical['a_max']<=2+1e-6 and physical['j_max']<=1+1e-6 and physical['input_min']>=-2000-1e-5 and physical['input_max']<=3000+1e-5 and physical['ordinary_gap_min']>=10-1e-6 and (np.isnan(physical['merge_gap_min']) or physical['merge_gap_min']>=10-1e-6) and overlap==0 and physical['lateral_y_min']>=-1e-6 and physical['lateral_y_max']<=3.5+1e-6 and physical['steering_max']<=.25+1e-6 and physical['steering_step_max']<=.01+1e-6
    return physical,passed
