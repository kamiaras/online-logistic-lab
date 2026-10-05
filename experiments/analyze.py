"""Fit shared prefix comparators and retain numerical regret intervals."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from .comparator import fit_comparator

BASE=Path(__file__).resolve().parent


def load_runs(folder):
    specs=json.loads((folder/'manifest.json').read_text())
    result=[]
    for spec in specs:
        out=Path(spec['output'])
        if not (out/'supervisor.json').exists():continue
        sup=json.loads((out/'supervisor.json').read_text());worker=sup['worker']
        losses=np.fromfile(out/'losses.bin',dtype='<f8') if (out/'losses.bin').exists() else np.empty(0)
        n=min(len(losses),worker['round'])
        times=np.fromfile(out/'online_times.bin',dtype='<f8') if (out/'online_times.bin').exists() else np.empty(0)
        result.append(dict(spec=spec,worker=worker,supervisor=sup,losses=losses[:n],prefix=np.cumsum(losses[:n]),times=times[:n]))
    return result


def select_calibration():
    folder=BASE/'results'/'calibration';runs=load_runs(folder)
    expected=json.loads((folder/'manifest.json').read_text())
    if len(runs)!=len(expected):
        raise RuntimeError('Calibration is incomplete; finish the full equal-budget grid before selecting')
    if any(r['worker'].get('status') not in ('completed','time_budget') for r in runs):
        raise RuntimeError('Diagnose failed calibration runs before selecting')
    groups=defaultdict(list)
    for r in runs:groups[r['spec']['case_id']].append(r)
    scores=defaultdict(list);common={}
    for cid,rr in groups.items():
        eligible=[r for r in rr if 'bridge' not in r['spec']['label']]
        n=min(len(r['losses']) for r in eligible)
        if n<1:raise RuntimeError('Calibration contains failed run; diagnose before selecting')
        common[cid]=n
        for r in eligible:
            scores[(r['spec']['algorithm'],r['spec']['label'])].append(float(r['prefix'][n-1]/n))
    selected={}
    table=[]
    for (algo,label),vals in scores.items():
        table.append(dict(algorithm=algo,label=label,mean_prefix_average_loss=float(np.mean(vals)),case_values=vals))
    for algo in ['ours','sigmoid']:
        best=min((r for r in table if r['algorithm']==algo),key=lambda r:r['mean_prefix_average_loss'])
        example=next(r for r in runs if r['spec']['algorithm']==algo and r['spec']['label']==best['label'])
        selected[algo]=dict(config=example['spec']['config'],label=best['label'],selection=best)
    selected['selection_protocol']=dict(metric='Mean per-round loss over paired common prefixes of all non-bridge calibration configurations',
                                        common_prefixes=common,held_out_data_seeds=True,
                                        nominal_search_budget_seconds_per_method=6*8*len(groups))
    (folder/'selected_configs.json').write_text(json.dumps(selected,indent=2))
    (folder/'configuration_scores.json').write_text(json.dumps(table,indent=2))
    print(json.dumps(selected,indent=2))


def analyze(stage, comparator_seconds=1.):
    folder=BASE/'results'/stage;runs=load_runs(folder);groups=defaultdict(list)
    expected=json.loads((folder/'manifest.json').read_text())
    if len(runs)!=len(expected):
        raise RuntimeError('Stage has unmeasured runs; finish or explicitly revise its manifest before analysis')
    for r in runs:groups[r['spec']['case_id']].append(r)
    summaries=[];curves=[];start=perf_counter()
    for ci,(cid,rr) in enumerate(groups.items()):
        case=json.loads((Path(rr[0]['spec']['data_path'])/'data.json').read_text())['case']
        nonempty=[len(r['losses']) for r in rr if len(r['losses'])]
        if not nonempty:continue
        common=min(nonempty) if len(nonempty)==len(rr) else 0
        if stage=='confirmation':
            common=500 if common>=500 else 0
        largest=max(nonempty)
        points=set(np.unique(np.geomspace(1,largest,20).astype(int)).tolist()+nonempty)
        if common:points.add(common)
        cachefile=folder/f'{cid}__comparators.json'
        cache=json.loads(cachefile.read_text()) if cachefile.exists() else {}
        X=np.memmap(Path(rr[0]['spec']['data_path'])/'X.bin',dtype='<f8',mode='r',shape=(case['T'],case['d']))
        y=np.memmap(Path(rr[0]['spec']['data_path'])/'y.bin',dtype='i1',mode='r',shape=(case['T'],))
        initial=None
        with threadpool_limits(limits=1):
            for n in sorted(points):
                if str(n) in cache:
                    prior=cache[str(n)];initial=np.asarray(prior['theta'])
                    if prior['gap']<=.01 or prior.get('max_seconds',0)>=comparator_seconds:
                        continue
                fit=fit_comparator(X[:n],y[:n],case['B'],initial=initial,tolerance=.01,max_seconds=comparator_seconds)
                initial=fit['theta']
                cache[str(n)]={k:(v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else v) for k,v in fit.items()}
                cachefile.write_text(json.dumps(cache))
        for r in rr:
            n=len(r['losses']);spec=r['spec'];w=r['worker'];sup=r['supervisor']
            row={k:spec[k] for k in ['algorithm','label','case_id','d','B','T','data_seed','sampler_seed','stage']}
            row.update(family=case['family'],B_generator=case['B_generator'],completed_rounds=n,status=w.get('status'),
                       online_seconds=w.get('online_seconds'),process_wall_seconds=sup['process_wall_seconds'],
                       cpu_seconds=w.get('cpu_seconds'),state_bytes=w.get('state_bytes'),
                       peak_working_set=max(w.get('memory',{}).get('peak_working_set',0),sup['monitored_peak_rss']),
                       peak_private_commit=max(w.get('memory',{}).get('peak_private_commit',0),sup['monitored_peak_private']),
                       interrupted_measurements_censored=bool(sup.get('termination')),
                       baseline_memory=w.get('baseline_memory'),config=spec['config'],diagnostics=w.get('diagnostics'),
                       metrics=w.get('metrics'),common_round=common)
            for name,t in [('final',n),('common',common)]:
                if t:
                    fit=cache[str(t)];loss=float(r['prefix'][t-1]);lo=loss-fit['loss']
                    width=min(fit['gap'],max(0.,fit['loss']))
                    row[name]=dict(round=t,loss=loss,average_loss=loss/t,comparator_loss=fit['loss'],
                                   comparator_gap=width,comparator_first_order_gap=fit['gap'],
                                   regret_lower=lo,regret_upper=lo+width,
                                   online_seconds=float(r['times'][t-1]) if len(r['times'])>=t else None)
            summaries.append(row)
            for t in sorted(points):
                if t>n:continue
                fit=cache[str(t)];loss=float(r['prefix'][t-1]);lo=loss-fit['loss']
                width=min(fit['gap'],max(0.,fit['loss']))
                curves.append(dict(case_id=cid,algorithm=spec['algorithm'],label=spec['label'],
                                   data_seed=spec['data_seed'],sampler_seed=spec['sampler_seed'],
                                   run_id=Path(spec['output']).name,round=t,
                                   loss=loss,regret_lower=lo,regret_upper=lo+width,gap=width,
                                   first_order_gap=fit['gap']))
        del X,y
        print(json.dumps(dict(case=ci+1,total=len(groups),id=cid,common=common,largest=largest,
                              comparator_seconds=round(sum(v['elapsed'] for v in cache.values()),3))),flush=True)
    (folder/'summary.json').write_text(json.dumps(summaries,indent=2))
    (folder/'curves.json').write_text(json.dumps(curves))
    (folder/'analysis_timing.json').write_text(json.dumps(dict(elapsed_seconds=perf_counter()-start)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--select',action='store_true');p.add_argument('--comparator-seconds',type=float,default=1.)
    args=p.parse_args()
    if args.select:select_calibration()
    else:analyze(args.stage,args.comparator_seconds)
