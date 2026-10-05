"""Generate static scientific figures and a compact measured-results report."""
from collections import defaultdict
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
NAMES={'ours':'Ours (capped)','sigmoid':'Sigmoid/MALA (capped)','aioli':'AIOLI'}
COLORS={'ours':'#be3a34','sigmoid':'#2868ae','aioli':'#278155'}


def main():
    folder=BASE/'results'/'screen'; rows=json.loads((folder/'summary.json').read_text())
    plots=BASE/'figures';plots.mkdir(exist_ok=True)
    group=defaultdict(list)
    for r in rows:group[(r['family'],r['d'],r['B'],r['T'],r['B_generator'],r['algorithm'])].append(r)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for algo in NAMES:
        all_rows=[r for r in rows if r['algorithm']==algo]
        rr=[r for r in all_rows if r.get('common') and r['common'].get('online_seconds') is not None]
        axes[0].scatter([r['common']['online_seconds'] for r in rr],
                        [r['common']['regret_lower']/r['common_round'] for r in rr],
                        s=22,alpha=.6,c=COLORS[algo],label=NAMES[algo])
        axes[1].scatter([r['completed_rounds'] for r in all_rows],
                        [r['peak_private_commit']/1024**2 for r in all_rows],
                        s=22,alpha=.6,c=COLORS[algo])
        axes[2].scatter([r['d'] for r in all_rows],
                        [r['completed_rounds']/r['T'] for r in all_rows],
                        s=22,alpha=.6,c=COLORS[algo])
    axes[0].set(xscale='log',yscale='symlog',xlabel='Online seconds at common prefix',ylabel='Regret lower bound / common rounds',title='Paired prefixes; different cases pooled')
    axes[1].set(xscale='log',yscale='log',xlabel='Completed rounds',ylabel='Peak private commit (MiB)',title='Memory at each run’s stopping point')
    axes[2].set(xscale='log',xlabel='Dimension d',ylabel='Completed / requested rounds',title='Completion under 15-second online caps')
    axes[2].set_yscale('log')
    axes[0].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(plots/'overview.png',dpi=170);plt.close(fig)
    # Hold d, T, generator and family fixed for the B comparison.
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for algo in NAMES:
        vals=[]
        for B in [1.,10.,100.,1000.]:
            rr=[r for r in group.get(('signal',100,B,1000,3.,algo),[]) if r.get('common')]
            if not rr:continue
            vals.append((B,np.mean([r['common']['regret_lower']/r['common_round'] for r in rr]),
                         np.mean([r['common']['regret_upper']/r['common_round'] for r in rr]),
                         np.mean([r['completed_rounds'] for r in rr]),
                         np.mean([r['peak_private_commit']/1024**2 for r in rr])))
        if vals:
            a=np.array(vals)
            axes[0].plot(a[:,0],a[:,1],'-o',color=COLORS[algo],label=NAMES[algo])
            axes[0].fill_between(a[:,0],a[:,1],a[:,2],color=COLORS[algo],alpha=.15)
            axes[1].plot(a[:,0],a[:,3],'-o',color=COLORS[algo])
            axes[2].plot(a[:,0],a[:,4],'-o',color=COLORS[algo])
    axes[0].set(ylabel='Mean regret interval / common rounds',yscale='symlog')
    axes[1].set(ylabel='Mean completed rounds');axes[2].set(ylabel='Mean peak private commit (MiB)')
    for ax in axes:ax.set(xlabel='Comparator radius B',xscale='log');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Fixed data: signal family, d=100, T=1000; three paired seeds')
    fig.savefig(plots/'radius_sweep.png',dpi=170);plt.close(fig)
    curves=json.loads((folder/'curves.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,(d,B,T) in zip(axes,[(100,1,1000),(100,100,1000),(1000,1000,10000)]):
        cid=f'signal_d{d}_B{B}_T{T}_s72001'
        for algo in NAMES:
            rr=sorted([r for r in curves if r['case_id']==cid and r['algorithm']==algo],key=lambda r:r['round'])
            if not rr:continue
            t=[r['round'] for r in rr];lo=[r['regret_lower'] for r in rr];hi=[r['regret_upper'] for r in rr]
            ax.plot(t,lo,color=COLORS[algo],label=NAMES[algo]);ax.fill_between(t,lo,hi,color=COLORS[algo],alpha=.15)
        common=[r['common_round'] for r in rows if r['case_id']==cid]
        if common and min(common)>0:ax.axvline(min(common),color='#666666',linestyle=':',label='Common prefix')
        ax.set(title=f'B={B}, d={d}, requested T={T:,}',xlabel='Completed round',ylabel='Cumulative regret interval',xscale='log')
        ax.set_yscale('symlog',linthresh=10);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('Example regret curves: fixed data seed 72001; shading is comparator error')
    fig.savefig(plots/'regret_curves.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.3),layout='constrained')
    for algo in NAMES:
        stage='long_horizon' if algo=='aioli' else 'screen'
        specs=json.loads((BASE/'results'/stage/'manifest.json').read_text())
        spec=next(s for s in specs if s['case_id']=='signal_d10_B100_T1000000_s72001' and s['algorithm']==algo)
        out=Path(spec['output']);cost=np.fromfile(out/'online_times.bin',dtype='<f8')
        result=json.loads((out/'result.json').read_text());n=result['round'];cost=cost[:n]
        cost=cost-result['initialization_seconds']
        edges=np.unique(np.r_[0,np.geomspace(1,n,35).astype(int)])
        xx=[];yy=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            xx.append(np.sqrt((lo+1)*hi));yy.append((cost[hi-1]-(cost[lo-1] if lo else 0))/(hi-lo))
        ax.plot(xx,yy,'-o',markersize=3,color=COLORS[algo],label=NAMES[algo])
    ax.set(xscale='log',yscale='log',xlabel='Observed round (logarithmic bins)',ylabel='Mean online seconds per round',
           title='Measured round cost: B=100, d=10, configured T=1,000,000')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(plots/'round_cost.png',dpi=170);plt.close(fig)
    lines=['# Measured Python comparison: proximal log-odds, sigmoid/MALA, and AIOLI','',
           'These are measured float64 reference implementations. The randomized methods use explicitly capped practical schedules; their theoretical regret guarantees do not apply automatically. AIOLI uses the audited finite scalar solve and a Cholesky matrix update.','',
           'Ours implements the original `SMALL_CLOUD_COMPLETE.md` schedule. The later stratified/fresh-restart refinements in `ELEGANT_LOG_ODDS_SINGLE_FLOW.md` are outside this measured study.','',
           '## Execution and interpretation','',
           f'The main screen contains {len(rows)} measured runs across {len(set(r["case_id"] for r in rows))} data cases. Each case has three algorithms; datasets use three paired seeds. Each worker has one BLAS thread, a 2 GiB hard private-commit limit, and a 15-second online budget. Initialization, prediction and update count toward that budget. Import, input, recording and offline comparator work are separately recorded. Measured workers run sequentially.','',
           'Regret is against the best fixed vector with Euclidean norm at most B. Put G=gᵀu+B‖g‖ and Δ=min{L(u),G}; nonnegative logistic loss gives the numerical regret interval [A−L(u), A−L(u)+Δ]. Both the raw first-order gap G and interval width Δ are retained. These are not confidence intervals or rigorous floating-point enclosures. Negative realized regret is retained.','',
           'The table averages three seeds. Common-prefix lengths can differ by seed; the listed range makes this explicit. Timings in this table are for those common prefixes. Memory is the peak for the entire run, which may continue past that prefix. The separate completion column shows mean completed rounds out of the requested horizon.','',
           '![Radius sweep](figures/radius_sweep.png)','',
           '## Main measurements','',
           '| Family | B / generating B | d | Requested T | Method | Common rounds (range) | Mean regret interval | Mean common-prefix seconds | Mean peak private MiB | Mean completed rounds |',
           '|---|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    for key,rr in sorted(group.items(),key=lambda kv:(kv[0][0],kv[0][4],kv[0][1],kv[0][2],kv[0][3],kv[0][5])):
        family,d,B,T,truth,algo=key;valid=[r for r in rr if r.get('common')]
        if not valid:continue
        cr=[r['common_round'] for r in valid]
        lo=np.mean([r['common']['regret_lower'] for r in valid]);hi=np.mean([r['common']['regret_upper'] for r in valid])
        times=[r['common']['online_seconds'] for r in valid if r['common'].get('online_seconds') is not None]
        sec=np.mean(times) if times else float('nan');mem=np.mean([r['peak_private_commit']/1024**2 for r in valid]);n=np.mean([r['completed_rounds'] for r in valid])
        lines.append(f'| {family} | {B:g} / {truth:g} | {d} | {T:,} | {NAMES[algo]} | {min(cr)}–{max(cr)} | [{lo:.3f}, {hi:.3f}] | {sec:.4f} | {mem:.1f} | {n:.0f} |')
    statuses=defaultdict(int)
    for r in rows:statuses[(r['algorithm'],r['status'])]+=1
    lines+=['','## Completion and numerical limits','','| Method | Status | Runs |','|---|---|---:|']
    for (algo,status),n in sorted(statuses.items()):lines.append(f'| {NAMES[algo]} | {status} | {n} |')
    gaps=[r['final']['comparator_gap'] for r in rows if r.get('final')]
    lines+=['',f'Final comparator interval widths: median {np.median(gaps):.6g}, maximum {max(gaps):.6g}. A wide interval is retained as uncertainty, not treated as an exact optimum.','',
            '## Reproduction and raw records','',
            'See `README.md` for commands, tests, conventions, and the exact selected configurations. `results/calibration/` contains the held-out parameter sweep; `results/screen/manifest.json` specifies every requested run. `summary.json` and `curves.json` contain numerical results; each run retains its specification, loss and time arrays, diagnostic checkpoints, errors, OS memory counters, and supervisor result. Dataset manifests retain seeds, norm checks, generating parameters, and hashes.','',
            'Arithmetic counters measure algorithm events (such as proposals and historical observation/particle pairs). Matrix-kernel FLOPs are estimates derived from executed dimensions. They are not measured CPU instructions, and scalar/transcendental costs are recorded separately where implemented. Nested phase timers overlap and must not be summed.','',
            'This screen evaluates the declared implementations and budgets, not the best possible engineering of each algorithm. It does not establish a general ranking across datasets, mixing guarantees, or the published asymptotic rates. Sparse inputs currently use the same dense algorithm kernels.','',
            '![Regret curves](figures/regret_curves.png)','',
            '![Measured cost per round](figures/round_cost.png)','',
            '![Overview](figures/overview.png)','']
    (BASE/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(BASE/'RESULTS.md')


if __name__=='__main__':main()
