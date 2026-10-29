"""Same v18 lateral objective and inequalities, solved by sparse OSQP."""
import time
import numpy as np
from scipy.linalg import expm
from scipy import sparse
import osqp
from robust_core import DT,V20_OPTIONS

def lateral_control(x,v,yr,last,H=40,return_plan=False,previous_plan=None,return_info=False):
    lf=1.2;lr=1.6;m=1500;iz=2500;cf=cr=50000
    A=np.array([[0,v,1,0],[0,0,0,1],[0,0,-(cf+cr)/(m*v),-v-(cf*lf-cr*lr)/(m*v)],
                [0,0,-(cf*lf-cr*lr)/(iz*v),-(cf*lf**2+cr*lr**2)/(iz*v)]])
    B=np.array([0,0,cf/m,cf*lf/iz]);aug=np.zeros((5,5));aug[:4,:4]=A;aug[:4,4]=B
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
    bb=np.r_[3.499-fx[iy],fx[iy],5-fx[iv],5+fx[iv],1-fx[ir],1+fx[ir],.01+rr,.01-rr,np.full(2*H,.25)]
    enabled=V20_OPTIONS.get('enabled',False) and V20_OPTIONS.get('lateral_soft',False)
    u=None;status='exception'
    try:
        qp=osqp.OSQP();qp.setup(P=sparse.csc_matrix((HH+HH.T)/2),q=lin,A=sparse.csc_matrix(AA),
                               l=np.full(len(bb),-np.inf),u=bb,verbose=False,eps_abs=1e-7,eps_rel=1e-7,
                               max_iter=10000,polishing=False)
        sol=qp.solve(raise_error=False);status=sol.info.status
        if sol.x is not None and status.startswith('solved') and np.max(AA@sol.x-bb)<=2e-6:u=sol.x.copy()
    except (ValueError,ArithmeticError,RuntimeError):
        if not enabled:raise
    if not enabled:
        if u is None:raise RuntimeError('Lateral OSQP: '+status)
        return u.copy() if return_plan else u[0]
    from lateral_fallback import soften,hard_info
    if u is None:u,info=soften(HH,lin,AA,bb,last,previous_plan,'osqp')
    else:info=hard_info(AA,bb,u)
    answer=u.copy() if return_plan else u[0]
    return (answer,info) if return_info else answer
