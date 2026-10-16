"""Hierarchical paired summaries: sampler repetitions nested within data seeds."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t


def main():
    base=Path(__file__).resolve().parent
    rows=json.loads((base/'results'/'confirmation'/'summary.json').read_text())
    bydata=defaultdict(lambda:defaultdict(list))
    for r in rows:bydata[r['data_seed']][r['algorithm']].append(r)
    estimates=[]
    for first,second in [('ours','sigmoid'),('ours','aioli'),('sigmoid','aioli')]:
        vals=[];detail=[]
        for seed,algos in sorted(bydata.items()):
            if len(algos['ours'])!=3 or len(algos['sigmoid'])!=3 or len(algos['aioli'])!=1:
                raise RuntimeError('Confirmation has incomplete nested repetitions')
            rr=algos[first]+algos[second]
            lengths={r['common_round'] for r in rr}
            if lengths!={500}:
                raise RuntimeError('A confirmation member did not complete the predeclared 500 rounds')
            n=next(iter(lengths))
            # The same comparator cancels exactly in paired regret differences.
            a=np.mean([r['common']['loss'] for r in algos[first]])/n
            b=np.mean([r['common']['loss'] for r in algos[second]])/n
            vals.append(a-b);detail.append(dict(data_seed=seed,common_round=n,mean_loss_difference_per_round=a-b))
        N=len(vals);mean=float(np.mean(vals));se=float(np.std(vals,ddof=1)/np.sqrt(N))
        radius=float(student_t.ppf(.975,N-1)*se)
        estimates.append(dict(first=first,second=second,datasets=N,
                              mean_paired_regret_difference_per_round=mean,
                              standard_error_across_dataset_clusters=se,
                              approximate_95_percent_t_interval=[mean-radius,mean+radius],
                              first_lower_loss_datasets=int(np.sum(np.array(vals)<0)),
                              dataset_details=detail,
                              interval_scope='Approximate Student-t interval across independent data seeds after averaging sampler repetitions; no multiple-comparison correction or time-uniform guarantee.'))
    result=dict(setting=dict(B=100,d=10,T=1000,evaluation_round=500,family='signal',B_generator=3),
                sampler_repetitions_per_dataset=3,aioli_repetitions_per_dataset=1,
                comparisons=estimates,
                target='Loss/regret per round at the predeclared fixed 500-round prefix, with algorithm horizon parameter T=1000.')
    (base/'results'/'confirmation'/'paired_statistics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
