"""Frozen comparison grids for the user's shared-tangent manuscript."""
import argparse
import hashlib
import json
from pathlib import Path
import random

BASE = Path(__file__).resolve().parent
GRID = [(16, 1), (16, 4), (16, 16), (64, 1), (64, 4), (256, 1)]
STAGES = ['calibration', 'smoke', 'prescribed', 'screen', 'deep_scale',
          'confirmation', 'deep_budget', 'long_horizon', 'deep_long']


def prepare(source_stage):
    source = BASE / 'results' / source_stage
    old = json.loads((source / 'manifest.json').read_text())
    by_case = {}
    for spec in old:
        by_case.setdefault(spec['case_id'], spec)
    stage = 'shared_' + source_stage
    folder = BASE / 'results' / stage
    folder.mkdir(parents=True, exist_ok=True)
    if source_stage == 'calibration':
        configs = [(dict(mode='practical', tilts=m, burn_in=n, transitions=n), f'm{m}n{n}')
                   for m, n in GRID]
    elif source_stage == 'prescribed':
        configs = [(dict(mode='prescribed'), 'prescribed')]
    elif source_stage == 'smoke':
        configs = [(dict(mode='practical', tilts=4, burn_in=1, transitions=1), 'm4n1')]
    else:
        chosen = json.loads((BASE / 'results/shared_calibration/selected_config.json').read_text())
        configs = [(chosen['config'], chosen['label'])]
    specs = []
    for cid, original in by_case.items():
        for config, label in configs:
            repetitions = range(3) if source_stage == 'confirmation' else range(1)
            for rep in repetitions:
                suffix = label + (f'_rep{rep}' if source_stage == 'confirmation' else '')
                spec = dict(original)
                spec.update(algorithm='shared', config=config, label=suffix, stage=stage,
                            output=str(folder / f'{cid}__shared__{suffix}'),
                            source_stage=source_stage, source_manifest=str(source / 'manifest.json'),
                            sampler_seed=original['data_seed'] + 89173 +
                            (40000003 + 1000003*rep if source_stage == 'confirmation' else 0))
                # Comparison uses the source stage's original configured horizon,
                # exact data, time allowance and 2 GiB process cap.
                specs.append(spec)
    random.Random(20260911 + STAGES.index(source_stage)).shuffle(specs)
    manifest = folder / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != specs:
        raise RuntimeError('Refusing to replace an existing different manifest: ' + str(manifest))
    manifest.write_text(json.dumps(specs, indent=2))
    cases = json.loads((source / 'cases.json').read_text())
    (folder / 'cases.json').write_text(json.dumps(cases, indent=2))
    names = ['SHARED_TANGENT_INPUT.md', 'shared_tangent.py', 'shared_tangent_primitives.py',
             'shared_worker.py', 'shared_runner.py', 'common.py', 'proximal.py', 'sigmoid.py',
             'aioli.py', 'data.py', 'memory.py']
    hashes = {name: hashlib.sha256((BASE/name).read_bytes()).hexdigest()
              for name in names if (BASE/name).exists()}
    snapshot = folder / 'source_hashes.json'
    if snapshot.exists() and json.loads(snapshot.read_text()) != hashes:
        raise RuntimeError('Measured-source snapshot changed; use a separate stage directory.')
    snapshot.write_text(json.dumps(hashes, indent=2))
    print(json.dumps(dict(stage=stage, runs=len(specs), cases=len(by_case),
                          online_allowance_seconds=sum(s['seconds'] for s in specs))))
    return specs


def select():
    import numpy as np
    folder = BASE / 'results/shared_calibration'
    specs = json.loads((folder/'manifest.json').read_text())
    groups = {}
    for spec in specs:
        out = Path(spec['output'])
        sup = json.loads((out/'supervisor.json').read_text())
        if sup['worker']['status'] not in ('completed', 'time_budget'):
            raise RuntimeError('Diagnose failed calibration run before selection: '+str(out))
        n = sup['worker']['round']
        values = np.fromfile(out/'losses.bin', dtype='<f8')[:n]
        if len(values) != n or not n:
            raise RuntimeError('No valid calibration prefix: '+str(out))
        groups.setdefault(spec['case_id'], []).append((spec, values))
    common, scores = {}, {}
    for cid, rows in groups.items():
        n = min(len(v) for _, v in rows)
        common[cid] = n
        for spec, values in rows:
            scores.setdefault(spec['label'], []).append(float(values[:n].sum()/n))
    table = [dict(label=label, mean_loss_per_round=float(np.mean(values)),
                  dataset_values=values) for label, values in scores.items()]
    best = min(table, key=lambda r:(r['mean_loss_per_round'], r['label']))
    config = next(s['config'] for s in specs if s['label']==best['label'])
    result = dict(config=config, label=best['label'], common_prefixes=common,
                  selection=best, scores=table, grid=[list(x) for x in GRID],
                  protocol='Six predeclared configurations; four existing held-out calibration datasets; '
                           'mean per-round loss at within-dataset all-grid common prefix. '
                           'm denotes correlated tilt states, not independent particles; N caps both initial and later mixing.')
    (folder/'selected_config.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage', choices=STAGES+['select'])
    args=parser.parse_args()
    select() if args.stage=='select' else prepare(args.stage)
