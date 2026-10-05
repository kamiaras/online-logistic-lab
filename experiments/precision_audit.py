"""Offline high-precision check of AIOLI's represented scalar equations.

This checks root residuals for the float64 q,v actually computed, not matrix
rounding error or a full floating-point regret theorem.
"""
from decimal import Decimal, localcontext
import json
from pathlib import Path
from time import perf_counter
from threadpoolctl import threadpool_limits
from .aioli import AIOLI
from .data import stream_case


def residual(z,q,v):
    with localcontext() as ctx:
        ctx.prec=80
        zz,qq,vv=map(Decimal.from_float,[z,q,v])
        tail=(-abs(zz)).exp()
        tanh=(1-2*tail/(1+tail))*(1 if zz>=0 else -1)
        return float(abs(zz+qq*tanh/2-vv))


def main():
    base=Path(__file__).resolve().parent;out=[]
    for seed in [72001,72002,72003]:
        cid=f'signal_d10_B10000_T1000_s{seed}';folder=base/'data'/cid
        a=AIOLI(10,10000.,1000,seed);vals=[];start=perf_counter()
        with threadpool_limits(limits=1):
            for X,Y in stream_case(folder,10,1000):
                for x,y in zip(X,Y):
                    z=a.predict(x);vals.append(residual(z,a._last_q,a._last_v));a.observe(x,int(y))
        out.append(dict(case_id=cid,rounds=len(vals),max_absolute_scalar_residual=max(vals),
                        sum_absolute_scalar_residuals=sum(vals),seconds=perf_counter()-start,
                        root_rounding_stalls=a.metrics.counters.get('root_rounding_stalls',0),
                        curvature_underflows=a.metrics.counters.get('curvature_underflows',0)))
    result=dict(results=out,precision_decimal_digits=80,
                interpretation='For each represented q,v, phi is increasing with derivative at least 1, so |phi(z)| bounds distance to its exact scalar root. The sum bounds direct scalar-prediction loss perturbations with those statistics frozen, not the effect of altered subsequent updates.',
                scope='Offline diagnostic. Decimal rounding is not interval arithmetic; Cholesky/statistic rounding and curvature underflow are not certified by this check.')
    (base/'results'/'precision_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
