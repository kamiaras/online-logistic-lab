"""Isolated shared-tangent comparison worker; original harness body retained."""
import sys
if __name__ == '__main__' and '--gate' in sys.argv:
    if sys.stdin.readline().strip() != 'GO':
        raise SystemExit('Worker not authorized by supervisor')
import argparse
import json
from pathlib import Path
from time import perf_counter, process_time
import traceback
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from .data import stream_case
from .memory import memory_snapshot


def run(spec):
    from .aioli import AIOLI
    from .sigmoid import SigmoidMALA
    from .proximal import ProximalLogOdds
    from .shared_tangent import SharedTangentLogOdds
    out = Path(spec['output']); out.mkdir(parents=True, exist_ok=True)
    T, d, B = (spec[k] for k in ['T','d','B'])
    stop_after_round=int(spec.get('stop_after_round',T))
    if not 1<=stop_after_round<=T:
        raise ValueError('stop_after_round must be between 1 and the configured horizon')
    points = set(np.unique(np.geomspace(1, T, 22).astype(int)))
    points.add(stop_after_round)
    start = perf_counter(); cpu_start = process_time(); online = 0.; input_time = 0.
    prediction_time = update_time = init_time = 0.
    baseline = memory_snapshot()
    a = None; n = 0; loss = 0.; status = 'running'; error = None
    last_write = start
    def snapshot():
        return dict(round=n, cumulative_loss=loss, online_seconds=online,
                    elapsed_seconds=perf_counter()-start, cpu_seconds=process_time()-cpu_start,
                    prediction_seconds=prediction_time, update_seconds=update_time,
                    initialization_seconds=init_time, input_seconds=input_time,
                    memory=memory_snapshot(), state_bytes=a.state_bytes() if a else 0,
                    metrics=a.metrics.as_dict() if a else {},
                    diagnostics=a.diagnostics() if a else {},
                    rng_state=a.rng.bit_generator.state if a is not None and hasattr(a,'rng') else None)
    with (out/'progress.jsonl').open('w') as log, (out/'losses.bin').open('wb') as losses, (out/'online_times.bin').open('wb') as costs:
        try:
            with threadpool_limits(limits=1):
                s = perf_counter()
                cls = {'aioli': AIOLI, 'sigmoid': SigmoidMALA, 'ours': ProximalLogOdds, 'shared': SharedTangentLogOdds}[spec['algorithm']]
                try:
                    a = cls(d=d, B=B, T=T, seed=spec['sampler_seed'], config=spec.get('config', {}))
                finally:
                    init_time = perf_counter()-s; online += init_time
                iterator = iter(stream_case(spec['data_path'], d, T))
                stopped = False
                while not stopped:
                    s = perf_counter()
                    try:
                        X, Y = next(iterator)
                    except StopIteration:
                        break
                    input_time += perf_counter()-s
                    for x, y in zip(X, Y):
                        if n>=stop_after_round:
                            status='evaluation_completed';stopped=True;break
                        if online >= spec['seconds']:
                            status='time_budget'; stopped=True; break
                        s=perf_counter()
                        try:
                            score=float(a.predict(x))
                        finally:
                            dt=perf_counter()-s; prediction_time+=dt; online+=dt
                        if not np.isfinite(score):
                            raise FloatingPointError('Non-finite prediction')
                        current=float(np.logaddexp(0., -int(y)*score))
                        s=perf_counter()
                        try:
                            a.observe(x, int(y))
                        finally:
                            dt=perf_counter()-s; update_time+=dt; online+=dt
                        n+=1; loss+=current
                        losses.write(np.asarray([current], dtype='<f8').tobytes())
                        costs.write(np.asarray([online], dtype='<f8').tobytes())
                        now=perf_counter()
                        if n in points or now-last_write>=1.:
                            log.write(json.dumps(snapshot())+'\n'); log.flush(); losses.flush(); costs.flush()
                            last_write=now
                if n==T:
                    status='completed'
                elif n==stop_after_round:
                    status='evaluation_completed'
        except Exception as exc:
            status='memory_error' if isinstance(exc, MemoryError) else 'error'
            error=repr(exc)
            (out/'traceback.txt').write_text(traceback.format_exc())
        final = snapshot()
        final.update(status=status, error=error, baseline_memory=baseline,
                     configured_horizon=T, threadpools=threadpool_info())
        log.write(json.dumps(final)+'\n')
    (out/'result.json').write_text(json.dumps(final, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('spec'); p.add_argument('--gate', action='store_true')
    args=p.parse_args(); run(json.loads(Path(args.spec).read_text()))
