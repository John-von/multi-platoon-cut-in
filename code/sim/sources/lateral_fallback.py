"""M1 fallback shared by the CVXOPT and OSQP execution paths."""
import numpy as np
from scipy import sparse


def soften(HH, lin, AA, bb, last, previous_plan, backend):
    H=len(lin);I=np.eye(H)
    slack=np.zeros((6*H,3*H))
    for block in range(6):
        slack[block*H:(block+1)*H,(block//2)*H:(block//2+1)*H]=-I
    A=np.vstack([np.c_[AA[:6*H],slack],
                 np.c_[AA[6*H:],np.zeros((4*H,3*H))],
                 np.c_[np.zeros((3*H,H)),-np.eye(3*H)]])
    b=np.r_[bb,np.zeros(3*H)]
    P=np.zeros((4*H,4*H));P[:H,:H]=(HH+HH.T)/2
    P[H:,H:]=2e4*np.eye(3*H);q=np.r_[lin,np.full(3*H,1e4)]
    reason=None;z=None
    try:
        if backend=='cvxopt':
            from cvxopt import matrix,solvers
            answer=solvers.qp(matrix(P),matrix(q),matrix(A),matrix(b))
            reason=answer['status']
            if reason=='optimal':z=np.asarray(answer['x']).ravel()
        else:
            import osqp
            qp=osqp.OSQP()
            qp.setup(P=sparse.csc_matrix(P),q=q,A=sparse.csc_matrix(A),
                     l=np.full(len(b),-np.inf),u=b,verbose=False,
                     eps_abs=1e-8,eps_rel=1e-8,max_iter=50000,polishing=True)
            answer=qp.solve(raise_error=False);reason=answer.info.status
            if answer.x is not None and reason.startswith('solved'):z=answer.x
    except (ValueError,ArithmeticError,RuntimeError) as exc:
        reason=type(exc).__name__+': '+str(exc)
    if z is not None and np.isfinite(z).all() and np.max(A@z-b)<=2e-6:
        return z[:H].copy(),{'level':'soft','slack_max':np.maximum(0,z[H:].reshape(3,H).max(axis=1)).tolist(),
            'slack_first':np.maximum(0,z[H:].reshape(3,H)[:,0]).tolist(),
            'constraint_residual':float(np.max(A@z-b)),'soft_status':reason}
    if previous_plan is None:
        u=np.full(H,last)
    else:
        previous=np.asarray(previous_plan,dtype=float)
        if previous.shape!=(H,):raise ValueError('Invalid previous steering sequence')
        u=np.r_[previous[1:],previous[-1]]
    residual=float(np.max(AA[6*H:]@u-bb[6*H:]))
    if residual>2e-6:
        raise RuntimeError('Shifted steering fallback violates hard input constraints')
    excess=np.maximum(0,(AA[:6*H]@u-bb[:6*H]).reshape(3,2,H).max(axis=1))
    return u,{'level':'shift','slack_max':excess.max(axis=1).tolist(),
        'slack_first':excess[:,0].tolist(),'constraint_residual':residual,'soft_status':reason}


def hard_info(AA,bb,u):
    return {'level':'hard','slack_max':[0.,0.,0.],'slack_first':[0.,0.,0.],
            'constraint_residual':float(np.max(AA@u-bb))}
