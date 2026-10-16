"""Allocation-free prescribed population/cost projections, distinct from timings."""
import json
import math
from pathlib import Path
from .proximal import ProximalLogOdds


def main():
    base=Path(__file__).resolve().parent
    cases=json.loads((base/'results'/'screen'/'cases.json').read_text())
    seen=set();rows=[]
    for c in cases:
        d,B,T=(c[k] for k in ['d','B','T']);key=(d,B,T)
        if key in seen:continue
        seen.add(key)
        p=ProximalLogOdds.preflight(d,B,T,{'mode':'prescribed'})
        par=p['prescribed'];m=par['m'];b2=B*B
        H,J=par['H'],par['J']
        upper=(1000*m*d*(T+1)+100*m*d*(J+10)*((1+H)*T*(T+1)/2+H*d*b2*T*(T+1)*(2*T+1)/24)) if m else None
        expected_sigmoid_m=T
        published_m=math.ceil(32*T**3*math.log(T/.05))
        rows.append(dict(d=d,B=B,T=T,ours=p,
                         ours_theorem_expected_operation_upper_bound=upper,
                         ours_particle_array_bytes=8*m*d if m else None,
                         sigmoid_expected_population=expected_sigmoid_m,
                         sigmoid_expected_particle_bytes=8*expected_sigmoid_m*d,
                         sigmoid_published_population_delta_005=published_m,
                         sigmoid_published_particle_bytes_delta_005=8*published_m*d,
                         sigmoid_mixing_constants_unspecified=True,
                         AIOLI_completed_state_bytes=8*(d*d+d),
                         note='Counts/projections, not measured hardware work or memory; Gaussian priors differ.'))
    (base/'results'/'prescribed_feasibility.json').write_text(json.dumps(rows,indent=2))
    print(f'{len(rows)} parameter combinations projected without allocating clouds')


if __name__=='__main__':main()
