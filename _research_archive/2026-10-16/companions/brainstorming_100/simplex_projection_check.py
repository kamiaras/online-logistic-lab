"""Exploratory identity checks, not a certified implementation or runtime theorem."""
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss

def expit(x):
    return np.exp(-np.logaddexp(0, -np.asarray(x)))

def basis_element(knots, values):
    a = np.asarray(knots)
    x = np.asarray(values)
    b = np.asarray([(x >= l) & (x < r) for l, r in zip(a[:-1], a[1:])], dtype=float)
    for degree in range(1, len(a)-1):
        nxt=[]
        for i in range(len(a)-degree-1):
            left, right = a[i+degree]-a[i], a[i+degree+1]-a[i+1]
            v = np.zeros_like(x)
            if left > 0:
                v += (x-a[i])/left*b[i]
            if right > 0:
                v += (a[i+degree+1]-x)/right*b[i+1]
            nxt.append(v)
        b=np.asarray(nxt)
    return b[0]

def score_rule(knots, extra_degree=40):
    a = np.sort(np.asarray(knots, dtype=float))
    n = len(a)
    if a[-1] == a[0]:
        return a[:1], np.ones(1)
    width = a[-1] - a[0]
    u = (a - a[0]) / width
    order = int(np.ceil((n - 1 + extra_degree) / 2))
    g, w = leggauss(order)
    pieces = [(l, r) for l, r in zip(u[:-1], u[1:]) if r > l]
    xs, ws = [], []
    for l, r in pieces:
        v = l + (r - l) * (g + 1) / 2
        weights = (r-l) * w/2 * (n-1) * basis_element(u, v)
        xs.extend(a[0] + width*v)
        ws.extend(weights)
    return np.asarray(xs), np.asarray(ws)

def expect(a, f):
    x, w = score_rule(a)
    return float(w @ f(x))

cases = [
    [-2., 3.],
    [-2., 0., 3.],
    [-1., -1., 1., 1.],
    [-2., -1e-12, 0., 1e-12, 2.],
    np.linspace(-5., 5., 21).tolist(),
    (2 + 1e-11*np.arange(8)).tolist(),
]
out=[]
for a in cases:
    a=np.asarray(a)
    x,w=score_rule(a)
    n=len(a)
    mean=float(a.mean())
    variance=float(np.mean((a-mean)**2)/(n+1))
    gradients=np.array([expect(np.append(a, aj), expit)/n for aj in a])
    # Dirichlet size-bias gives the vertex derivatives of expected softplus.
    eps=1e-5
    if np.ptp(a)>1e-5:
        fd=[]
        for j in range(n):
            plus,minus=a.copy(),a.copy()
            plus[j]+=eps;minus[j]-=eps
            fd.append((expect(plus,lambda z:np.logaddexp(0,z))-expect(minus,lambda z:np.logaddexp(0,z)))/(2*eps))
        grad_error=float(np.max(np.abs(gradients-fd)))
    else:
        grad_error=None
    out.append(dict(d=n-1,node_count=len(x),mass_error=abs(float(w.sum())-1),
       min_weight=float(w.min()),mean_error=abs(float(w@x)-mean),
       variance_error=abs(float(w@((x-mean)**2))-variance),
       sigmoid=expect(a,expit),gradient_sum_error=abs(float(gradients.sum())-expect(a,expit)),
       gradient_finite_difference_error=grad_error))
Path(__file__).with_suffix(".json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
