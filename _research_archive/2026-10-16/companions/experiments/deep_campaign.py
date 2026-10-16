"""Sequential follow-up orchestration; no simultaneous measured workers."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--await-scale', action='store_true')
    args = parser.parse_args()
    log_path = BASE / 'results' / 'deep_campaign_progress.jsonl'

    def record(phase, status, **more):
        message = dict(phase=phase, status=status,
                       utc=datetime.now(timezone.utc).isoformat(), **more)
        with log_path.open('a') as stream:
            stream.write(json.dumps(message) + '\n')
        print(json.dumps(message), flush=True)

    env = os.environ.copy()
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[name] = '1'

    def run(phase, *arguments):
        record(phase, 'started')
        begin = monotonic()
        result = subprocess.run([sys.executable, *arguments], cwd=ROOT, env=env)
        record(phase, 'completed' if result.returncode == 0 else 'failed',
               seconds=monotonic()-begin, returncode=result.returncode)
        if result.returncode:
            raise SystemExit(result.returncode)

    if args.await_scale:
        record('await_existing_scale', 'waiting')
        manifest = json.loads((BASE / 'results' / 'deep_scale' / 'manifest.json').read_text())
        start = monotonic()
        while not all((Path(s['output']) / 'supervisor.json').exists() for s in manifest):
            if monotonic() - start > 3600:
                raise RuntimeError('Existing scale campaign did not finish within its declared wall allowance')
            sleep(5)
        record('await_existing_scale', 'completed')

    run('frontier_planner_checks', '-m', 'pytest', 'experiments/test_memory_frontier.py', '-q')
    run('history_probe_small_checks', '-m', 'experiments.history_frontier', '--self-test')
    run('actual_memory_frontiers', '-m', 'experiments.memory_frontier', '--tag', 'deep',
        '--seconds', '30', '--wall-budget', '900', '--max-probes', '40', '--refinements', '3', '--resume')
    for mib in [1536, 1856, 1920, 1968, 2000, 2048]:
        for algorithm in ['ours', 'sigmoid']:
            run(f'history_{algorithm}_{mib}', '-m', 'experiments.history_frontier',
                '--algorithm', algorithm, '--dimension', '1000', '--B', '1',
                '--history-mib', str(mib), '--particles', '2', '--seconds', '120',
                '--output', str(BASE / 'results' / 'history_frontier_deep' / f'{algorithm}_{mib}'))
    run('prepare_budget', '-m', 'experiments.deep_prepare', 'deep_budget')
    run('sampling_budget_sensitivity', '-m', 'experiments.runner',
        str(BASE / 'results' / 'deep_budget' / 'manifest.json'), '--wall-budget', '1900')
    run('prepare_long', '-m', 'experiments.deep_prepare', 'deep_long')
    run('ten_million_rounds', '-m', 'experiments.runner',
        str(BASE / 'results' / 'deep_long' / 'manifest.json'), '--wall-budget', '2150')
    run('paired_prefix_comparisons', '-m', 'experiments.deep_pairwise')
    for stage in ['deep_scale', 'deep_budget']:
        run(f'comparators_{stage}', '-m', 'experiments.analyze', stage, '--comparator-seconds', '2')
    run('comparators_deep_long', '-m', 'experiments.analyze', 'deep_long', '--comparator-seconds', '3')
    record('all_measurements_and_basic_analysis', 'completed')


if __name__ == '__main__':
    main()
