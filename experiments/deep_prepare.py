"""Larger follow-up studies; frozen original algorithms and distinct output dirs."""
import argparse
import json
import random
from pathlib import Path
from threadpoolctl import threadpool_limits
from .data import generate_case
from .prepare import case_record, specification

BASE = Path(__file__).resolve().parent


def prepare(stage):
    cases, specs = [], []
    default = [('ours', {'particles': 256, 'transitions': 1}, 'm256n1'),
               ('sigmoid', {'particles': 16, 'transitions': 16}, 'm16n16'),
               ('aioli', {}, 'default')]
    if stage == 'deep_scale':
        templates = [(10, 100., 100000, 120), (1000, 1000., 10000, 120),
                     (10000, 1000., 2000, 60), (50000, 10000., 128, 60),
                     (10, 1e6, 10000, 60), (100, 10000., 10000, 60)]
        for seed in [74001, 74002]:
            for d, B, T, seconds in templates:
                case = case_record('signal', d, B, T, seed, tag='deep_')
                cases.append(case)
                for algorithm, config, label in default:
                    specs.append(specification(case, algorithm, config, label, seconds, stage))
        random.Random(140910).shuffle(specs)
    elif stage == 'deep_budget':
        # Fixed grid: diagnostic budget sensitivity, not selection on test data.
        configurations = [('aioli', {}, 'default')]
        for algorithm in ['ours', 'sigmoid']:
            for m, steps in [(256, 1), (1024, 1), (4096, 1), (256, 16),
                             (64, 64), (16, 256)]:
                configurations.append((algorithm, {'particles': m, 'transitions': steps},
                                       f'm{m}n{steps}'))
        configurations.append(('sigmoid', {'particles': 16, 'transitions': 16}, 'm16n16'))
        for seed in [75001, 75002]:
            case = case_record('signal', 100, 1000., 2000, seed, tag='budget_')
            cases.append(case)
            for algorithm, config, label in configurations:
                specs.append(specification(case, algorithm, config, label, 45, stage))
        random.Random(140911).shuffle(specs)
    elif stage == 'deep_long':
        case = case_record('signal', 10, 1000., 10000000, 76001, tag='long_')
        cases.append(case)
        specs.append(specification(case, 'aioli', {}, 'default', 2100, stage))
    else:
        raise ValueError(stage)
    folder = BASE / 'results' / stage
    folder.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        for case in cases:
            generate_case(BASE / 'data' / case['id'], case)
    (folder / 'cases.json').write_text(json.dumps(cases, indent=2))
    (folder / 'manifest.json').write_text(json.dumps(specs, indent=2))
    print(json.dumps(dict(stage=stage, runs=len(specs), cases=len(cases),
                          maximum_online_seconds=sum(s['seconds'] for s in specs),
                          data_bytes=sum(c['T']*(8*c['d']+1) for c in cases))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['deep_scale', 'deep_budget', 'deep_long'])
    prepare(parser.parse_args().stage)
