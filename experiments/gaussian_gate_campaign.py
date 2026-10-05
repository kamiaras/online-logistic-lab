"""Predeclared fresh gated-Gaussian comparison; no historical results overwritten."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter

from .computation_first_campaign import assert_idle

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'experiments/results/gaussian_feature_gate_comparison_2026-09-11'
CAP = 2**31
METHODS = ('gaussian_gate','author_aioli','author_sigmoid','canonical_aioli')
SOURCE_FILES = ['GAUSSIAN_FEATURE_GATE_COMPLETE.md','GAUSSIAN_FEATURE_GATE_EXPERIMENT_PLAN.md',
                'experiments/gaussian_feature_gate.py','experiments/gaussian_gate_worker.py',
                'experiments/gaussian_gate_supervisor.py','experiments/gaussian_gate_campaign.py',
                'experiments/aioli.py','experiments/author_baselines.py','experiments/common.py',
                'experiments/data.py','experiments/memory.py','experiments/worker_job.py',
                'research/author_code_comparison_2026-09-11/a9a_experiment.py',
                'research/author_code_comparison_2026-09-11/libsvm_datasets.py']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def digest(paths):
    out={}
    for raw in paths:
        p=ROOT/raw
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):
                h.update(chunk)
        out[str(p.relative_to(ROOT))]=h.hexdigest()
    return out


def config(method,B):
    if method=='gaussian_gate':
        return {'R':1.0,'matrix_backend':'inverse'}
    if method=='canonical_aioli':
        return {'R':1.0,'matrix_backend':'inverse'}
    if method=='author_aioli':
        return {'R':1.0,'lam':B**-2,'T_inner':80,'omega_solver':'lbfgs'}
    return {'n_particles':12,'step_size':.01,'mala_steps_per_round':300,
            'grad_clip':50.,'adapt_step':True,'target_accept':.57,
            'rm_gamma0':.5,'rm_gamma_decay':.25,'step_size_min':1e-4,
            'step_size_max':1.,'mala_window':None}


def make_spec(output,stage,d,B,T,data_seed,method,rounds,seconds):
    case=f'signal_d{d}_B{int(B)}_T{T}_s{data_seed}'
    data=ROOT/'experiments/data'/case
    meta=read(data/'data.json')['case']
    assert meta['d']==d and meta['B']==B and meta['T']==T and meta['B_generator']==3.
    name=f'{method}__{case}'
    return dict(stage=stage,case=case,d=d,B=float(B),T=T,rounds=rounds,data_seed=data_seed,
                seed=910001,algorithm=method,config=config(method,float(B)),seconds=seconds,
                memory_bytes=CAP,data_path=str(data),output=str(output/stage/name),
                stop_path=str(output/'STOP_REQUESTED'))


def specifications(output,stage):
    if stage=='smoke':
        specs=[make_spec(output,stage,10,100,1000,72001,m,8,30.) for m in METHODS]
        specs.append(make_spec(output,stage,1000,1000,100000,72001,'gaussian_gate',8,30.))
    elif stage=='primary':
        specs=[make_spec(output,stage,d,B,1000,s,m,500,90.)
               for d,B in [(10,100),(100,1000)] for s in (72001,72002,72003) for m in METHODS]
        random.Random(9112604).shuffle(specs)
    elif stage=='stress':
        specs=[make_spec(output,stage,1000,1000,100000,72001,m,1000,90.)
               for m in ('gaussian_gate','author_aioli')]
    else:
        raise ValueError(stage)
    return specs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['smoke','primary','stress'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args()
    assert_idle()
    from .gaussian_gate_supervisor import run_one
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    folder=output/args.stage
    folder.mkdir(exist_ok=False)
    specs=specifications(output,args.stage)
    dump(folder/'manifest.json',specs)
    source_before=digest(SOURCE_FILES)
    data_files=sorted({str((Path(s['data_path'])/n).relative_to(ROOT)) for s in specs for n in ('X.bin','y.bin','data.json')})
    data_before=digest(data_files)
    dump(folder/'source_hashes_before.json',source_before)
    dump(folder/'data_hashes_before.json',data_before)
    tick=perf_counter()
    attempts=[]
    error=None
    try:
        for spec in specs:
            if (output/'STOP_REQUESTED').exists():
                break
            dump(folder/'progress.json',dict(status='running',current=spec,attempts=attempts))
            result=run_one(spec)
            worker=result.get('worker',{})
            item=dict(algorithm=spec['algorithm'],case=spec['case'],output=spec['output'],
                      status=result.get('status',worker.get('status')),round=worker.get('committed_round',worker.get('round',0)),
                      online_seconds=worker.get('online_seconds'),cumulative_loss=worker.get('cumulative_loss'))
            attempts.append(item)
            print(json.dumps(item),flush=True)
            if item['status'] in ('supervisor_error','error','memory_error','abnormal_exit') or str(item['status']).startswith('supervisor_error'):
                raise RuntimeError('Stop stage after implementation/worker failure: '+str(item))
    except Exception as exc:
        error=repr(exc)
        raise
    finally:
        source_after=digest(SOURCE_FILES)
        data_after=digest(data_files)
        dump(folder/'source_hashes_after.json',source_after)
        dump(folder/'data_hashes_after.json',data_after)
        record=dict(generated_utc=datetime.now(timezone.utc).isoformat(),stage=args.stage,
                    attempts=attempts,planned=len(specs),attempted=len(attempts),
                    terminal_counts=dict(Counter(a['status'] for a in attempts)),error=error,
                    sources_unchanged=source_before==source_after,data_unchanged=data_before==data_after,
                    elapsed_wall_seconds=perf_counter()-tick,status='terminal')
        dump(folder/'result.json',record)
        dump(folder/'progress.json',record)
    if not record['sources_unchanged'] or not record['data_unchanged']:
        raise RuntimeError('Measured sources/data changed')


if __name__=='__main__':
    main()
