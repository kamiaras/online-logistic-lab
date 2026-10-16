"""Offline matched-prefix analysis of the predeclared gated Gaussian comparison."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .comparator import fit_comparator
from .gaussian_gate_campaign import DEFAULT, METHODS, ROOT, read, dump

LABELS={'gaussian_gate':'Gated Gaussian','author_aioli':'Released AIOLI',
        'author_sigmoid':'Released sigmoid/MALA','canonical_aioli':'Canonical AIOLI control'}


def load_run(path,columns=7):
    p=Path(path)
    spec=read(p/'spec.json')
    sup=read(p/'supervisor.json')
    worker=sup.get('worker',{})
    if (p/'result.json').exists():
        worker=read(p/'result.json')
    n=int(worker.get('committed_round',worker.get('round',0)))
    tr=np.fromfile(p/'trajectory.bin',dtype='<f8',count=n*columns).reshape(n,columns)
    assert len(tr)==n and np.isfinite(tr[:,:5]).all()
    return {'spec':spec,'supervisor':sup,'worker':worker,'trajectory':tr,'columns':columns,'path':str(p)}


def evaluate(run,n,comparator,stage,historical=False):
    tr=run['trajectory'][:n]
    spec=run['spec']
    labels=np.fromfile(Path(spec['data_path'])/'y.bin',dtype='i1',count=n)
    loss=float(math.fsum(tr[:,2]))
    direct=float(np.logaddexp(0,-labels*tr[:,1]).sum())
    assert abs(loss-direct)<=1e-8*max(1,loss)
    low=max(0.,comparator['loss']-comparator['gap'])
    sup=run['supervisor']
    mem=run['worker'].get('memory',{})
    peak=max(float(sup.get('monitored_peak_private',0) or 0),
             float(mem.get('peak_private_commit',mem.get('peak_private',mem.get('private_commit',mem.get('private',0)))) or 0))
    initialization=float(run['worker'].get('initialization_seconds',0.))
    result=dict(stage=stage,case=spec.get('case'),algorithm=spec['algorithm'],n=n,
                d=spec['d'],B=spec['B'],T=spec['T'],data_seed=spec.get('data_seed'),
                run_path=run['path'],status=sup.get('status',run['worker'].get('status')),
                completed_rounds=len(run['trajectory']),historical_timing=historical,
                cumulative_loss=loss,mean_loss=loss/n,online_seconds=float(tr[-1,3]),
                initialization_seconds=initialization,
                post_initialization_seconds=max(0.,float(tr[-1,3])-initialization),
                state_bytes=int(tr[-1,4]),run_peak_private_bytes=peak,
                comparator_loss=comparator['loss'],comparator_gap=comparator['gap'],
                regret_lower=loss-comparator['loss'],regret_upper=loss-low,
                mean_abs_logit=float(np.abs(tr[:,1]).mean()),
                sign_accuracy=float(np.mean(labels*tr[:,1]>0)),
                primary_endpoint_available=n==500 if stage=='primary' else n==1000)
    if run['columns']==7 and np.isfinite(tr[:,5]).all():
        result['author_clipped_loss']=float(math.fsum(tr[:,5]))
    if spec['algorithm']=='gaussian_gate':
        base_losses=np.logaddexp(0,-labels*tr[:,0])
        result.update(base_gate_loss=float(math.fsum(base_losses)),
                      gaussian_minus_base_loss=float(math.fsum(tr[:,2]-base_losses)),
                      max_abs_gaussian_minus_base_logit=float(np.max(np.abs(tr[:,1]-tr[:,0]))))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--comparator-seconds',type=float,default=3.)
    args=parser.parse_args()
    output=args.output.resolve()
    runs={stage:[load_run(s['output']) for s in read(output/stage/'manifest.json')
                 if (Path(s['output'])/'supervisor.json').exists()] for stage in ('primary','stress')}
    fits={}
    records=[]
    paired=[]
    comparators=output/'comparators'
    comparators.mkdir(exist_ok=True)

    def fit(spec,n):
        key=spec['case']+'__n'+str(n)
        if key not in fits:
            stored=comparators/(key+'.json')
            if stored.exists():
                f=read(stored)
            else:
                X=np.memmap(Path(spec['data_path'])/'X.bin',dtype='<f8',mode='r',shape=(spec['T'],spec['d']))[:n]
                y=np.fromfile(Path(spec['data_path'])/'y.bin',dtype='i1',count=n)
                f=fit_comparator(X,y,spec['B'],tolerance=1e-3,max_seconds=args.comparator_seconds)
                theta=f.pop('theta')
                np.save(comparators/(key+'.npy'),theta)
                f.update(case=spec['case'],n=n,B=spec['B'],data_path=spec['data_path'])
                dump(stored,f)
            fits[key]=f
        return fits[key]

    with threadpool_limits(limits=1):
        for case in sorted({r['spec']['case'] for r in runs['primary']}):
            group={r['spec']['algorithm']:r for r in runs['primary'] if r['spec']['case']==case}
            assert set(group)==set(METHODS)
            common=min(len(r['trajectory']) for r in group.values())
            prefixes=sorted({common,*[p for p in (100,500) if p<=common]})
            for n in prefixes:
                if n==0: continue
                c=fit(next(iter(group.values()))['spec'],n)
                for run in group.values(): records.append(evaluate(run,n,c,'primary'))
            if common:
                gate=group['gaussian_gate']['trajectory'][:common]
                control=group['canonical_aioli']['trajectory'][:common]
                paired.append(dict(case=case,n=common,
                    max_abs_base_control_logit_difference=float(np.max(np.abs(gate[:,0]-control[:,1]))),
                    gaussian_minus_control_loss=float(math.fsum(gate[:,2]-control[:,2])),
                    gaussian_to_control_time=float(gate[-1,3]/control[-1,3])))
        archived=ROOT/'experiments/results/author_baselines_d1000_B1000_T100000/author_sigmoid'
        if runs['stress']:
            historical=load_run(archived,5)
            historical['spec']['algorithm']='author_sigmoid'
            stress=runs['stress']+[historical]
            common=min(len(r['trajectory']) for r in stress)
            prefixes=sorted({common,*[p for p in (100,1000) if p<=common]})
            for n in prefixes:
                if n==0: continue
                c=fit(stress[0]['spec'],n)
                for run in stress: records.append(evaluate(run,n,c,'stress',run is historical))
            old_aioli=load_run(ROOT/'experiments/results/author_baselines_d1000_B1000_T100000/author_aioli',5)
            fresh=next(r for r in stress if r['spec']['algorithm']=='author_aioli')
            n=min(len(fresh['trajectory']),len(old_aioli['trajectory']))
            replay=dict(n=n,max_abs_fresh_vs_archived_aioli_logit_difference=float(np.max(np.abs(fresh['trajectory'][:n,1]-old_aioli['trajectory'][:n,1]))))
        else: replay=None

    aggregate=[]
    for d in sorted({r['d'] for r in records if r['stage']=='primary'}):
        for n in (100,500):
            for method in METHODS:
                rows=[r for r in records if r['stage']=='primary' and r['d']==d and r['n']==n and r['algorithm']==method]
                if not rows: continue
                aggregate.append(dict(d=d,B=rows[0]['B'],n=n,algorithm=method,datasets=len(rows),
                    mean_loss=float(np.mean([r['mean_loss'] for r in rows])),
                    mean_online_seconds=float(np.mean([r['online_seconds'] for r in rows])),
                    mean_post_initialization_seconds=float(np.mean([r['post_initialization_seconds'] for r in rows])),
                    mean_initialization_seconds=float(np.mean([r['initialization_seconds'] for r in rows])),
                    mean_state_bytes=float(np.mean([r['state_bytes'] for r in rows])),
                    mean_peak_private_bytes=float(np.mean([r['run_peak_private_bytes'] for r in rows])),
                    mean_regret_lower=float(np.mean([r['regret_lower'] for r in rows])),
                    mean_regret_upper=float(np.mean([r['regret_upper'] for r in rows]))))
    comparisons=[]
    for d in (10,100):
        for method in ('author_aioli','author_sigmoid','canonical_aioli'):
            diffs=[]
            ratios=[]
            for g in [r for r in records if r['stage']=='primary' and r['d']==d and r['n']==500 and r['algorithm']=='gaussian_gate']:
                other=next(r for r in records if r['stage']=='primary' and r['case']==g['case'] and r['n']==500 and r['algorithm']==method)
                diffs.append(g['mean_loss']-other['mean_loss'])
                ratios.append(g['online_seconds']/other['online_seconds'])
            if diffs: comparisons.append(dict(d=d,baseline=method,datasets=len(diffs),
                mean_loss_difference=float(np.mean(diffs)),datasets_gaussian_lower_loss=sum(v<0 for v in diffs),
                per_dataset_loss_differences=diffs,per_dataset_time_ratios=ratios))
    result=dict(records=records,aggregates=aggregate,paired_gate_control=paired,comparisons=comparisons,
                comparator_fits=fits,stress_replay_check=replay,
                primary_target=500,secondary_target=100,stress_target=1000,
                finite_precision_certified=False,author_sigmoid_stress_reused=True,
                report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump(output/'summary.json',result)
    make_plot(output,runs['primary'])
    write_markdown(output,result)
    print(json.dumps({'primary_500': [r for r in aggregate if r['n']==500],
                      'stress_1000':[r for r in records if r['stage']=='stress' and r['n']==1000],
                      'paired':paired},indent=2))


def make_plot(output,runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'gaussian_gate':'#c64b33','author_aioli':'#2e6fbb','author_sigmoid':'#2b8b69','canonical_aioli':'#777777'}
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for col,d in enumerate((10,100)):
        common=min(len(r['trajectory']) for r in runs if r['spec']['d']==d)
        x=np.arange(1,common+1)
        for method in METHODS:
            group=[r['trajectory'][:common] for r in runs if r['spec']['d']==d and r['spec']['algorithm']==method]
            if not group: continue
            losses=np.array([np.cumsum(t[:,2])/x for t in group])
            times=np.array([t[:,3] for t in group])
            kwargs=dict(color=colors[method],label=LABELS[method],linestyle='--' if method=='canonical_aioli' else '-')
            axes[0,col].plot(x,losses.mean(axis=0),**kwargs)
            axes[1,col].plot(x,times.mean(axis=0),**kwargs)
        axes[0,col].axhline(math.log(2),color='#aaa',linewidth=1,linestyle=':',label='Neutral')
        B=100 if d==10 else 1000
        axes[0,col].set_title(f'd={d}, B={B}; mean of 3 fixed datasets')
        axes[0,col].set_ylabel('Cumulative loss / rounds')
        axes[1,col].set_ylabel('Cumulative online seconds')
        axes[1,col].set_yscale('log')
        axes[1,col].set_xlabel('Completed rounds')
        for ax in axes[:,col]: ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8)
    fig.savefig(output/'comparison.png',dpi=170)
    fig.savefig(output/'comparison.svg')
    plt.close(fig)


def write_markdown(output,data):
    lines=['# Gated Gaussian implementation: measured comparison','',
           '2026-09-11. Fresh measurements of the implemented fixed-mean Gaussian feature gate against the pinned released AIOLI and sigmoid/MALA classes. Canonical inverse AIOLI is a diagnostic control. See the predeclared [plan](GAUSSIAN_FEATURE_GATE_EXPERIMENT_PLAN.md).','',
           '## Fresh 500-round comparison','',
           'Entries are means across three ordinary-signal datasets per setting, with identical examples and one fixed sampler seed per dataset. Algorithms are configured for T=1000. Lower loss is better; neutral loss is 0.693147 per round. Online time includes initialization, prediction, and observation.','',
           '| d | B | Method | Datasets reaching shared endpoint | Loss / round | Online seconds | After setup seconds | Regret interval | Retained arrays (KiB) |',
           '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in data['aggregates']:
        if r['n']!=500: continue
        lines.append(f"|{r['d']}|{r['B']:g}|{LABELS[r['algorithm']]}|{r['datasets']} / 3|{r['mean_loss']:.9f}|{r['mean_online_seconds']:.6f}|{r['mean_post_initialization_seconds']:.6f}|[{r['mean_regret_lower']:.6f}, {r['mean_regret_upper']:.6f}]|{r['mean_state_bytes']/1024:.3f}|")
    lines+=['','Setup includes lazy loading of the pinned author module inside author constructors. This is charged to author online initialization; the worker metadata phrase saying module imports are excluded is too broad. The after-setup column subtracts measured construction time and compares prediction plus observation work. Neither column is an implementation-independent arithmetic measurement.','',
            'Regret intervals use one shared numerical Euclidean-ball comparator at each matched data prefix. They are not interval-arithmetic certificates. The Gaussian-minus-baseline regret difference equals its loss difference, independently of comparator optimization accuracy.','',
            '### Paired loss differences','',
            '| d | Baseline | Gaussian minus baseline loss / round | Datasets where Gaussian has lower loss |',
            '|---:|---|---:|---:|']
    for r in data['comparisons']:
        lines.append(f"|{r['d']}|{LABELS[r['baseline']]}|{r['mean_loss_difference']:.9f}|{r['datasets_gaussian_lower_loss']} / {r['datasets']}|")
    lines+=['','## Large-case check','',
            'The following rows use the same first1000 examples of the previous d1000,B1000,T100000 dataset. The gate and released AIOLI were rerun freshly. Released sigmoid/MALA is the saved prefix of the earlier run cancelled at1234; its time is historical and is not a controlled fresh speed ratio.','',
            '| Method | Loss / round | Online seconds | After setup seconds | Regret interval | Time source |',
            '|---|---:|---:|---:|---:|---|']
    for r in data['records']:
        if r['stage']!='stress' or r['n']!=1000: continue
        lines.append(f"|{LABELS[r['algorithm']]}|{r['mean_loss']:.9f}|{r['online_seconds']:.6f}|{r['post_initialization_seconds']:.6f}|[{r['regret_lower']:.6f}, {r['regret_upper']:.6f}]|{'Historical' if r['historical_timing'] else 'Fresh'}|")
    if data['stress_replay_check']:
        r=data['stress_replay_check']
        lines+=['',f"The fresh released-AIOLI scores match its archived first{r['n']} scores with maximum absolute difference {r['max_abs_fresh_vs_archived_aioli_logit_difference']:.3g}."]
    maxbase=max(p['max_abs_base_control_logit_difference'] for p in data['paired_gate_control'])
    maxmix=max(abs(p['gaussian_minus_control_loss']) for p in data['paired_gate_control'])
    lines+=['','## What was actually implemented','',
            'The class implements 2d predetermined coordinate means, current-feature pair selection, stable log-domain gate weights, a shared isotropic covariance, prescribed finite scalar bisection, and the actual prescribed Gaussian forecast quadrature. The gate uses canonical inverse-AIOLI surrogate statistics. The fixed integration grid is cached, so measured quadrature state is O(log^(3/2)(eT)) rather than the proof\'s streamed O(log(eT)). Every cached array is counted. The finite forecast is not replaced by the base gate sigmoid.','',
            f'The canonical control isolates the Gaussian layer: maximum base-score discrepancy across primary paired runs is {maxbase:.3g}; maximum absolute cumulative Gaussian-minus-control loss is {maxmix:.3g}. These are floating-point observations, not an identity between the mathematical Gaussian predictor and the control.','',
            'The requested released AIOLI remains unchanged. Its effective initial precision is B², whereas the new gate and canonical control use B^-2; its linear-statistic increment also differs by a factor of two. Therefore differences from released AIOLI cannot be attributed solely to Gaussian integration or to scalar solver quality.','',
            'Released sigmoid/MALA uses the fixed selected12-particle/300-step setting with adaptation, clipping50, and full history. No universal claim follows about all sampler configurations. The three datasets per setting are exploratory and use only one sampler seed each.','',
            '## Verification and resource scope','',
            'Each measured process has one BLAS thread and an internally verified hard2GiB private-commit limit. Raw trajectories, per-round times, allocated state, OS peaks, statuses, source/data hashes, and numerical comparator fits are retained. Input/recording/offline analysis are outside online timing. A time-censored result is compared only at its committed common prefix. No historical measured source or dataset was overwritten. Float64 tests do not certify the exact-real regret theorem.','',
            'Implementation: [gaussian_feature_gate.py](experiments/gaussian_feature_gate.py). Reproduce with the new gaussian_gate_campaign module in smoke, primary, then stress order, followed by gaussian_gate_report. Use a fresh output directory.','',
            f'Raw summary: [{output.name}/summary.json](experiments/results/{output.name}/summary.json). Plot: [comparison.png](experiments/results/{output.name}/comparison.png). Independent audit: [GAUSSIAN_FEATURE_GATE_EXPERIMENT_AUDIT.json](GAUSSIAN_FEATURE_GATE_EXPERIMENT_AUDIT.json).','']
    (ROOT/'GAUSSIAN_FEATURE_GATE_EXPERIMENT_RESULT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    main()
