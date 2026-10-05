import json, math
from pathlib import Path

def softplus(z):
    return max(z,0)+math.log1p(math.exp(-abs(z)))
out=[]
for B in [1.,2.,5.,10.]:
  for pattern in [[1,-1],[1,1,1,-1],[1]*9+[-1],[1]*99+[-1]]:
    A=1/B**2;h=0.;loss=0.;np_=0;nm=0;rows=[]
    for t in range(1,20001):
      m=h/A;v=1/A;xi=math.sqrt(m*m+v)
      curvature=math.tanh(xi/2)/(2*xi) if xi else .25
      A+=curvature
      pred=h/A
      y=pattern[(t-1)%len(pattern)]
      loss+=softplus(-y*pred)
      np_+=(y==1);nm+=(y==-1);h+=y/2
      if t in [100,1000,10000,20000]:
        u=max(-B,min(B,math.log(np_/nm))) if np_ and nm else B*(1 if np_ else -1)
        regret=loss-np_*softplus(-u)-nm*softplus(u)
        rows.append(dict(t=t,regret=regret,score=pred))
    out.append(dict(B=B,period=len(pattern),positive_count=pattern.count(1),rows=rows))
Path(__file__).with_suffix(".json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
