"""Offline scalar-residual audit of completed extreme-B AIOLI runs."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from .aioli import AIOLI
from .data import stream_case
from .precision_audit import residual


def _no_measured_workers():
    import psutil
    modules = {'experiments.worker', 'experiments.runner',
               'experiments.memory_frontier', 'experiments.history_frontier'}
    active = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            if modules.intersection(process.info['cmdline'] or []):
                active.append(process.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if active:
        raise RuntimeError(f'Measured workers are active; defer offline replay: {active}')


def main():
    _no_measured_workers()
    base = Path(__file__).resolve().parent
    specs = json.loads((base / 'results/deep_scale/manifest.json').read_text())
    records = []
    for spec in specs:
        if spec['algorithm'] != 'aioli' or spec['d'] > 100 or spec['B'] < 10000:
            continue
        output = Path(spec['output'])
        result = json.loads((output / 'result.json').read_text())
        if result['status'] != 'completed':
            continue
        _no_measured_workers()
        recorded = np.fromfile(output / 'losses.bin', dtype='<f8')
        values, losses = [], []
        started = perf_counter()
        with threadpool_limits(limits=1):
            algorithm = AIOLI(spec['d'], spec['B'], spec['T'], spec['sampler_seed'], spec['config'])
            for X, Y in stream_case(spec['data_path'], spec['d'], spec['T']):
                for x, y in zip(X, Y):
                    score = algorithm.predict(x)
                    values.append(residual(score, algorithm._last_q, algorithm._last_v))
                    losses.append(float(np.logaddexp(0., -int(y)*score)))
                    algorithm.observe(x, int(y))
        losses = np.asarray(losses)
        difference = float(np.max(np.abs(losses-recorded)))
        assert np.array_equal(losses, recorded), (spec['case_id'], difference)
        records.append(dict(case_id=spec['case_id'], d=spec['d'], B=spec['B'],
                            rounds=len(values), max_absolute_scalar_residual=max(values),
                            sum_absolute_scalar_residuals=sum(values),
                            exact_replay_of_recorded_losses=True, max_loss_difference=difference,
                            elapsed_seconds=perf_counter()-started,
                            root_rounding_stalls=algorithm.metrics.counters.get('root_rounding_stalls', 0),
                            curvature_underflows=algorithm.metrics.counters.get('curvature_underflows', 0)))
    report = dict(decimal_precision=80, runs=records,
                  scope='Offline diagnostic, excluded from measured online runs. The represented q,v are frozen for each residual; matrix rounding and subsequent update perturbations are not certified. Decimal arithmetic is not an interval proof.',
                  interpretation='The scalar equation has derivative at least one, so its exact residual bounds scalar root error at the represented statistics. Summing these bounds concerns only direct scalar prediction errors with those statistics frozen.')
    (base / 'results/deep_precision_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
