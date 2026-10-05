"""Sequential, resumable shared-tangent campaign; no concurrent measured work."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from .shared_prepare import BASE, prepare, select
from .shared_runner import environment, run_one

ORDER = ['smoke', 'prescribed', 'calibration', 'screen', 'deep_scale',
         'confirmation', 'deep_budget', 'long_horizon', 'deep_long']


def campaign(stages):
    log_path = BASE / 'results/shared_campaign_progress.jsonl'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    with log_path.open('a', encoding='utf-8') as log:
        for source in stages:
            if source not in ('smoke', 'prescribed', 'calibration'):
                chosen = BASE / 'results/shared_calibration/selected_config.json'
                if not chosen.exists():
                    select()
            specs = prepare(source)
            folder = BASE / 'results' / ('shared_' + source)
            env_path = folder / 'environment.json'
            if not env_path.exists():
                env_path.write_text(json.dumps(environment(), indent=2))
            for i, spec in enumerate(specs):
                result = run_one(spec)
                w = result['worker']
                item = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                            stage=source, index=i+1, total=len(specs),
                            run=Path(spec['output']).name,
                            status=w.get('status'), round=w.get('round'),
                            loss=w.get('cumulative_loss'),
                            online_seconds=w.get('online_seconds'),
                            wall_seconds=result['process_wall_seconds'],
                            campaign_seconds=perf_counter()-start)
                line = json.dumps(item)
                log.write(line+'\n'); log.flush()
                print(line, flush=True)
                if w.get('status') in ('error', 'abnormal_exit'):
                    raise RuntimeError('Diagnose numerical/implementation failure before continuing: '
                                       + spec['output'])
            if source == 'calibration':
                select()
    print(json.dumps(dict(status='campaign_completed', stages=stages,
                          elapsed_seconds=perf_counter()-start)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stages', nargs='*', choices=ORDER, default=ORDER)
    args = p.parse_args()
    campaign(args.stages or ORDER)
