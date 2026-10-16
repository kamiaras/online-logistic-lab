"""Direct paired comparisons for implementation controls; no comparator needed."""
import json
from pathlib import Path
import numpy as np


def main():
    base=Path(__file__).resolve().parent
    def manifest(stage):return json.loads((base/'results'/stage/'manifest.json').read_text())
    def result(spec):
        out=Path(spec['output']);sup=json.loads((out/'supervisor.json').read_text());w=sup['worker']
        return dict(spec=spec,worker=w,loss=np.fromfile(out/'losses.bin',dtype='<f8')[:w['round']],
                    times=np.fromfile(out/'online_times.bin',dtype='<f8')[:w['round']],supervisor=sup)
    def paired(a,b):
        n=min(len(a['loss']),len(b['loss']))
        return dict(common_round=n,first_run=Path(a['spec']['output']).name,second_run=Path(b['spec']['output']).name,
                    first_minus_second_cumulative_loss=float(np.sum(a['loss'][:n]-b['loss'][:n])),
                    maximum_absolute_round_loss_difference=float(np.max(np.abs(a['loss'][:n]-b['loss'][:n]))),
                    first_online_seconds_at_common=float(a['times'][n-1]),
                    second_online_seconds_at_common=float(b['times'][n-1]),
                    first_completed_rounds=len(a['loss']),second_completed_rounds=len(b['loss']),
                    first_peak_private_MiB=max(a['worker']['memory']['peak_private_commit'],a['supervisor']['monitored_peak_private'])/2**20,
                    second_peak_private_MiB=max(b['worker']['memory']['peak_private_commit'],b['supervisor']['monitored_peak_private'])/2**20,
                    note='Same fixed data and shared prefix; regret difference equals loss difference.')
    matrix=[]
    for spec in manifest('matrix_control'):
        reference=next(s for s in manifest('screen') if s['case_id']==spec['case_id'] and s['algorithm']=='aioli')
        row=paired(result(spec),result(reference));row['first_backend']='inverse';row['second_backend']='cholesky';matrix.append(row)
    tempered=[]
    for spec in manifest('tempered'):
        reference=next(s for s in manifest('calibration') if s['case_id']==spec['case_id'] and s['algorithm']=='sigmoid' and s['label']=='m16n16')
        row=paired(result(spec),result(reference));row['first_schedule']='20 bridges / 20 total lazy steps';row['second_schedule']='1 bridge / 16 total lazy steps';tempered.append(row)
    (base/'results'/'auxiliary_comparisons.json').write_text(json.dumps(dict(matrix=matrix,tempered=tempered),indent=2))
    stages=['calibration','prescribed','screen','confirmation','long_horizon','tempered','matrix_control']
    totals=[]
    for stage in stages:
        specs=manifest(stage);rr=[result(s) for s in specs]
        totals.append(dict(stage=stage,runs=len(rr),online_seconds=sum(r['worker']['online_seconds'] for r in rr),
                           process_wall_seconds=sum(r['supervisor']['process_wall_seconds'] for r in rr),
                           statuses={status:sum(r['worker']['status']==status for r in rr) for status in {r['worker']['status'] for r in rr}}))
    (base/'results'/'campaign_totals.json').write_text(json.dumps(totals,indent=2))
    print(json.dumps(dict(matrix=matrix,tempered=tempered,totals=totals),indent=2))


if __name__=='__main__':main()
