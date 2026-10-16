"""Shared-tangent runner: original isolation protocol, separate worker module."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter, sleep
import psutil
from .memory import ProcessLimit

ROOT=Path(__file__).resolve().parents[1]


def environment():
    import importlib.metadata as md
    import numpy as np
    from threadpoolctl import threadpool_info
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                processor=platform.processor(), logical_cpus=os.cpu_count(),
                memory_total=psutil.virtual_memory().total, memory_available=psutil.virtual_memory().available,
                packages={k:md.version(k) for k in ['numpy','scipy','matplotlib','psutil','threadpoolctl','pytest']},
                threadpools=threadpool_info(), worker_blas_threads=1)


def run_one(spec, remaining_wall=None):
    out=Path(spec['output']); out.mkdir(parents=True, exist_ok=True)
    spec_path=out/'spec.json'
    if (out/'supervisor.json').exists():
        if not spec_path.exists() or json.loads(spec_path.read_text()) != spec:
            raise ValueError('Refusing to reuse a completed output with a different specification: '+str(out))
        return json.loads((out/'supervisor.json').read_text())
    spec_path.write_text(json.dumps(spec,indent=2))
    env=os.environ.copy()
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        env[key]='1'
    job=None; proc=None; start=perf_counter(); reason=None; observed_rss=0; observed_private=0
    with (out/'stdout.txt').open('w') as stdout, (out/'stderr.txt').open('w') as stderr:
        try:
            proc=subprocess.Popen([sys.executable,'-m','experiments.shared_worker',str(spec_path),'--gate'],
                                  cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,
                                  text=True,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            cap=int(spec.get('memory_bytes',2*1024**3))
            job=ProcessLimit(proc._handle if os.name=='nt' else 0, cap)
            proc.stdin.write('GO\n'); proc.stdin.flush(); proc.stdin.close()
            monitored=psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    mem=monitored.memory_info()
                    observed_rss=max(observed_rss,mem.rss)
                    private=getattr(mem,'private',mem.vms)
                    observed_private=max(observed_private,private)
                    if private>cap:
                        reason='memory_limit'; proc.kill(); break
                except psutil.NoSuchProcess:
                    break
                wall_limit=spec['seconds']+20 if remaining_wall is None else min(spec['seconds']+20,remaining_wall)
                if perf_counter()-start>wall_limit:
                    reason='external_timeout'; proc.kill(); break
                sleep(.1)
            proc.wait()
        except Exception as exc:
            reason='supervisor_error: '+repr(exc)
            if proc is not None and proc.poll() is None:
                proc.kill(); proc.wait()
        finally:
            if job is not None:
                job.close()
    result=dict(process_wall_seconds=perf_counter()-start, returncode=proc.returncode if proc else None,
                termination=reason, monitored_peak_rss=observed_rss, monitored_peak_private=observed_private,
                hard_process_commit_limit=spec.get('memory_bytes',2*1024**3),
                os_hard_limit_active=os.name=='nt' and job is not None)
    try:
        result['worker']=json.loads((out/'result.json').read_text())
    except (OSError,json.JSONDecodeError):
        progress=out/'progress.jsonl'
        lines=progress.read_text().splitlines() if progress.exists() else []
        for line in reversed(lines):
            try:
                result['worker']=json.loads(line); break
            except json.JSONDecodeError:
                continue
        result.setdefault('worker',dict(round=0,cumulative_loss=0,online_seconds=0))
        result['worker']['status']=reason or 'abnormal_exit'
    (out/'supervisor.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('--wall-budget',type=float,default=3600)
    args=p.parse_args(); manifest=Path(args.manifest)
    specs=json.loads(manifest.read_text()); start=perf_counter()
    (manifest.parent/'environment.json').write_text(json.dumps(environment(),indent=2))
    for i,spec in enumerate(specs):
        if perf_counter()-start>=args.wall_budget:
            print('Campaign wall budget reached',flush=True);break
        result=run_one(spec,args.wall_budget-(perf_counter()-start)); w=result['worker']
        print(json.dumps(dict(index=i+1,total=len(specs),run=Path(spec['output']).name,
                              status=w.get('status'),round=w.get('round'),
                              seconds=round(result['process_wall_seconds'],3))),flush=True)


if __name__=='__main__':
    main()
