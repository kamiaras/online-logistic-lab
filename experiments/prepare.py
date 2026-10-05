"""Create exact manifests and data; no unrequested external datasets."""
import argparse
import json
import random
from pathlib import Path
from .data import generate_case

BASE=Path(__file__).resolve().parent


def specification(case,algorithm,config,label,seconds,stage):
    cid=case['id'];run_id=f'{cid}__{algorithm}__{label}'
    return dict(algorithm=algorithm,config=config,d=case['d'],B=case['B'],T=case['T'],
                sampler_seed=case['data_seed']+89173,data_seed=case['data_seed'],case_id=cid,
                data_path=str(BASE/'data'/cid),output=str(BASE/'results'/stage/run_id),
                seconds=seconds,memory_bytes=2*1024**3,stage=stage,label=label)


def case_record(family,d,B,T,seed,truth=3.,tag=''):
    return dict(id=f'{tag}{family}_d{d}_B{B:g}_T{T}_s{seed}',family=family,d=d,B=B,T=T,
                data_seed=seed,B_generator=truth)


def prepare(stage):
    specs=[];cases=[]
    if stage=='smoke':
        cases=[case_record('signal',2,1.,30,70001),case_record('signal',10,1000.,50,70002)]
        configs=[('aioli',{},'default'),('ours',{'particles':4,'transitions':1},'m4n1'),
                 ('sigmoid',{'particles':4,'transitions':4},'m4n4')]
        seconds=15
    elif stage=='prescribed':
        cases=[case_record('signal',d,B,10,70020+i) for i,(d,B) in enumerate([(1,.1),(1,1.),(2,1.)])]
        configs=[('ours',{'mode':'prescribed'},'prescribed')]
        seconds=15
    elif stage=='calibration':
        cases=[case_record('signal',10,10.,1000,71001),
               case_record('signal',100,100.,1000,71002),
               case_record('isotropic',1000,1000.,1000,71003),
               case_record('rare_flip',100,10.,1000,71004)]
        configs=[('aioli',{},'default')]
        for algo in ['ours','sigmoid']:
            for m,n in [(16,1),(16,4),(16,16),(64,1),(64,4),(256,1)]:
                configs.append((algo,dict(particles=m,transitions=n),f'm{m}n{n}'))
        # Small explicit bridge control: four stages with four total transitions.
        configs.append(('sigmoid',dict(particles=16,transitions=4,bridges=4),'m16n4bridge4'))
        seconds=8
    elif stage=='screen':
        templates=[]
        for B in [1.,10.,100.,1000.]:templates.append(('signal',100,B,1000,3.))
        for d in [10,500,1000]:templates.append(('signal',d,100.,1000,3.))
        for T in [10000,100000]:templates.append(('signal',10,10.,T,3.))
        for f in ['isotropic','separable','rare_flip','correlated','coordinate','shift','random']:
            templates.append((f,100,100.,1000,3.))
        templates += [('signal',1000,1000.,10000,3.),('signal',5000,100.,1000,3.),
                      ('signal',10,10000.,1000,3.),('signal',10,100.,1000000,3.),
                      ('signal',100,1000.,1000,1000.)]
        for seed in [72001,72002,72003]:
            cases.extend(case_record(*r[:4],seed,truth=r[4],tag=('strong_' if r[4]>3 else '')) for r in templates)
        chosen=json.loads((BASE/'results'/'calibration'/'selected_configs.json').read_text())
        configs=[('aioli',{},'default')]+[(a,chosen[a]['config'],chosen[a]['label']) for a in ['ours','sigmoid']]
        seconds=15
    elif stage=='confirmation':
        # Independent seeds; one predeclared moderate-dimensional, large-B case.
        cases=[case_record('signal',10,100.,1000,s) for s in range(73001,73011)]
        chosen=json.loads((BASE/'results'/'calibration'/'selected_configs.json').read_text())
        configs=[('aioli',{},'default')]
        for a in ['ours','sigmoid']:
            configs.extend((a,chosen[a]['config'],chosen[a]['label']+f'_rep{r}') for r in range(3))
        seconds=15
    elif stage=='long_horizon':
        cases=[case_record('signal',10,100.,1000000,72001)]
        configs=[('aioli',{},'default')]
        seconds=240
    elif stage=='tempered':
        cases=[case_record('signal',10,10.,1000,71001),case_record('rare_flip',100,10.,1000,71004)]
        configs=[('sigmoid',dict(particles=16,transitions=20,bridges=20),'m16n20bridge20')]
        seconds=8
    elif stage=='matrix_control':
        cases=[case_record('signal',1000,1000.,10000,72001),case_record('signal',5000,100.,1000,72001)]
        configs=[('aioli',dict(matrix_backend='inverse'),'inverse')]
        seconds=15
    else:
        raise ValueError(stage)
    folder=BASE/'results'/stage;folder.mkdir(parents=True,exist_ok=True)
    for case in cases:
        generate_case(BASE/'data'/case['id'],case)
        for algo,config,label in configs:
            spec=specification(case,algo,config,label,seconds,stage)
            if stage=='confirmation':
                spec['stop_after_round']=500
            if stage=='confirmation' and algo!='aioli':
                rep=int(label[-1]);spec['sampler_seed']+=1000003*rep+(20000003 if algo=='sigmoid' else 0)
            specs.append(spec)
    if stage=='confirmation':
        random.Random(991031).shuffle(specs)
    (folder/'cases.json').write_text(json.dumps(cases,indent=2))
    (folder/'manifest.json').write_text(json.dumps(specs,indent=2))
    print(json.dumps(dict(stage=stage,cases=len(cases),runs=len(specs),
                          maximum_algorithm_seconds=sum(s['seconds'] for s in specs))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','calibration','screen','prescribed','confirmation','long_horizon','tempered','matrix_control']);prepare(p.parse_args().stage)
