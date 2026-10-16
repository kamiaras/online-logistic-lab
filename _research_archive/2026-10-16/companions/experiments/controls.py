"""Small supervision and event-counter-overhead controls, outside main timings."""
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from .aioli import AIOLI
from .sigmoid import SigmoidMALA
from .proximal import ProximalLogOdds
from .memory import ProcessLimit


def main():
    base=Path(__file__).resolve().parent
    result={}
    if os.name=='nt':
        child='import sys;sys.stdin.readline();a=bytearray(256*1024**2);print(len(a))'
        p=subprocess.Popen([sys.executable,'-c',child],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
        job=ProcessLimit(p._handle,64*1024**2)
        stdout,stderr=p.communicate('GO\n',timeout=15);job.close()
        result['hard_memory_limit_control']=dict(limit_bytes=64*1024**2,requested_bytes=256*1024**2,
                                                returncode=p.returncode,allocation_rejected='MemoryError' in stderr,
                                                stdout=stdout,stderr=stderr)
        if not result['hard_memory_limit_control']['allocation_rejected']:
            raise AssertionError('Windows hard memory cap did not reject oversized allocation')
    rng=np.random.default_rng(88001);X=rng.normal(size=(200,10));X/=np.linalg.norm(X,axis=1)[:,None]
    y=rng.choice([-1,1],200);measurements=[]
    with threadpool_limits(limits=1):
        for name,cls in [('aioli',AIOLI),('ours',ProximalLogOdds),('sigmoid',SigmoidMALA)]:
            reference=None
            for rep in range(5):
                for enabled in ([True,False] if rep%2==0 else [False,True]):
                    cfg={} if name=='aioli' else dict(particles=16,transitions=1)
                    a=cls(10,10.,200,98201,cfg)
                    if not enabled:a.metrics.inc=lambda *args,**kwargs:None
                    start=perf_counter();pred=[]
                    for x,yi in zip(X,y):
                        pred.append(a.predict(x));a.observe(x,int(yi))
                    seconds=perf_counter()-start
                    if reference is None:reference=np.array(pred)
                    if not np.array_equal(reference,np.array(pred)):
                        raise AssertionError('Counter disabling changed predictions')
                    measurements.append(dict(algorithm=name,rep=rep,event_counters_enabled=enabled,seconds=seconds))
    result['event_counter_overhead_control']=measurements
    result['overhead_scope']='Only Metrics.inc disabled; phase timers, validation, Python and common harness remain. Alternating paired order; no speed significance claim.'
    result['algorithm_source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in base.glob('*.py')}
    (base/'results'/'controls.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
