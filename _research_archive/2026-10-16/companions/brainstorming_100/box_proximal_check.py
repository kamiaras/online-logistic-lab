"""Numerical diagnostics for the root box-RGO proof; not a float certificate."""
import json, math
from pathlib import Path
import numpy as np

rng=np.random.default_rng(7391)
def sigmoid(z):
    return np.exp(-np.logaddexp(0.,-z))
def center(X,y,B,Y):
    n,d=X.shape; lam=B**-2; E=float(np.sum(X*X))/4
    inveta=lam+2*d*E; mu=lam+inveta; smooth=mu+E
    R=max(np.linalg.norm(X,axis=1),default=0.)
    def grad(w):return mu*w-inveta*Y-X.T@(y*sigmoid(-y*(X@w)))
    def value(w):return lam*np.dot(w,w)/2+inveta*np.sum((w-Y)**2)/2+np.logaddexp(0.,-y*(X@w)).sum()
    k=max(0,math.ceil(math.log(max(1.,6*n*R/math.sqrt(mu)),3)))
    w=np.clip(Y/(1+lam/inveta),-2*B,2*B)
    for _ in range(k):w=np.clip(w-grad(w)/smooth,-2*B,2*B)
    g0=grad(w);v=np.clip(w-g0/smooth,-2*B,2*B)
    normal=smooth*(w-v)-g0;g=grad(v);residual=g+normal
    return v,g,normal,residual,mu,E,value,k

maxratio=0.;normal_error=0.;cases=0
for d in [1,2,5,10]:
    for n in [0,1,7,31]:
        X=rng.normal(size=(n,d)); X/=np.maximum(1.,np.linalg.norm(X,axis=1,keepdims=True))
        y=rng.choice([-1.,1.],size=n)
        for scale in [0.,1.,100.,1e5]:
            Y=rng.normal(size=d)*scale
            v,g,nv,r,mu,E,fun,k=center(X,y,1.3,Y)
            maxratio=max(maxratio,float(np.linalg.norm(r)/math.sqrt(mu)))
            z=rng.uniform(-2.6,2.6,size=(100,d))
            normal_error=max(normal_error,float(np.max((z-v)@nv)))
            cases+=1

nodes,weights=np.polynomial.legendre.leggauss(512)
rows=[]
for n in [0,1,7,31]:
    X=np.ones((n,1));y=np.where(np.arange(n)%3==0,-1.,1.)
    for Y0 in [-20.,-3.,0.,3.,20.]:
        v,g,nv,r,mu,E,fun,k=center(X,y,1.,np.array([Y0]))
        z=2*nodes;w=2*weights
        F=np.array([fun(np.array([zz])) for zz in z])
        Q=fun(v)+g[0]*(z-v[0])+mu*(z-v[0])**2/2
        def logint(f):
            shift=float(np.min(f));return -shift+math.log(float(w@np.exp(-(f-shift))))
        ratio=math.exp(logint(Q)-logint(F))
        diff=F-Q
        rows.append(dict(n=n,Y=Y0,expected_proposals=ratio,min_envelope_gap=float(min(diff)),residual_ratio=float(np.linalg.norm(r)/math.sqrt(mu))))

out=dict(status='Numerical diagnostics only; no bit or floating-point certificate',center_cases=cases,max_KKT_residual_ratio=maxratio,max_normal_cone_violation=max(0.,normal_error),max_expected_proposals=max(r['expected_proposals'] for r in rows),rejection_bound=math.e,quadrature_cases=rows)
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in out.items() if k!='quadrature_cases'}))
assert maxratio<=1.000001
assert normal_error<=1e-6
assert out['max_expected_proposals']<=math.e+1e-6
assert min(r['min_envelope_gap'] for r in rows)>=-1e-7
