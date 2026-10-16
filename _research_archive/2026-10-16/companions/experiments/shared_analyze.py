"""Offline common-prefix comparisons; invoke after every measured worker exits."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from .shared_compare_prepare import STAGES, _no_measured_workers, prepare


def main(stages, seconds):
    _no_measured_workers()
    # Also reject an idle campaign between measured subprocesses.
    import psutil
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.info['pid'] != os.getpid() and 'experiments.shared_campaign' in (process.info['cmdline'] or []):
            raise RuntimeError('Finish the shared campaign before offline analysis')
    from .analyze import analyze
    start = perf_counter()
    log_path = Path(__file__).resolve().parent/'results/shared_analysis_progress.jsonl'
    with log_path.open('a', encoding='utf-8') as log:
        for stage in stages:
            prepare(stage)
            analyze('shared_compare_'+stage, comparator_seconds=seconds)
            row = dict(stage=stage, status='analyzed', elapsed_seconds=perf_counter()-start)
            line = json.dumps(row)
            log.write(line+'\n'); log.flush()
            print(line, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stages', nargs='*', choices=STAGES, default=list(STAGES))
    parser.add_argument('--comparator-seconds', type=float, default=1.)
    args = parser.parse_args()
    main(args.stages or STAGES, args.comparator_seconds)
