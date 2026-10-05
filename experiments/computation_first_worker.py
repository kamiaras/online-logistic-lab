"""Isolated four-method recorder for the named computation-first algorithm.

Actual output losses are primary. Raw selected logits and common output-only c4
clipping are separate diagnostics; neither changes a baseline's state updates.
"""
import sys
if __name__ == "__main__" and "--gate" in sys.argv:
    if sys.stdin.readline().strip() != "GO":
        raise SystemExit("Worker did not receive its supervisor gate")

import json
import math
import os
from pathlib import Path
from time import perf_counter, process_time
import traceback

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .memory import memory_snapshot


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def run(spec):
    from .computation_first_log_odds import ComputationFirstLogOdds
    from .proximal import ProximalLogOdds
    from .sigmoid import SigmoidMALA
    from .aioli import AIOLI
    registry = dict(selected=ComputationFirstLogOdds, ours=ProximalLogOdds,
                    sigmoid=SigmoidMALA, aioli=AIOLI)
    Algorithm = registry[spec["method"]]
    output, source = Path(spec["output"]), Path(spec["data_path"])
    output.mkdir(parents=True, exist_ok=True)
    d, T, endpoint = int(spec["d"]), int(spec["T"]), int(spec["rounds"])
    if not 1 <= endpoint <= T or not math.isfinite(spec["seconds"]) or spec["seconds"] <= 0:
        raise ValueError("Endpoint must lie in the configured horizon; seconds must be positive finite")
    start, cpu_start = perf_counter(), process_time()
    online = initialization = prediction = update = input_time = recording_time = 0.
    status, error, a, phase = "running", None, None, "input"
    baseline_memory = memory_snapshot()
    keys = ("output_logits", "output_losses", "raw_logits", "raw_losses", "guarded_logits", "guarded_losses",
            "online_times", "labels", "state_bytes", "peak_private_commit", "peak_working_set")
    saved = {key: [] for key in keys}
    counter_rows = []

    def snapshot(current_status):
        recorded = len(counter_rows)
        diagnostic_error = None
        try:
            diagnostics = a.diagnostics() if a is not None else {}
            state_bytes = a.state_bytes() if a is not None else 0
        except Exception as exc:
            diagnostics, state_bytes, diagnostic_error = {}, None, repr(exc)
        return dict(schema_version=1, status=current_status, error=error, failure_phase=phase if error else None,
                    round=recorded, method=spec["method"], algorithm_class=Algorithm.__name__,
                    config=spec["config"], configured_horizon=T, fixed_endpoint=endpoint,
                    cumulative_loss=math.fsum(saved["output_losses"][:recorded]),
                    raw_cumulative_loss=math.fsum(saved["raw_losses"][:recorded]),
                    guarded_cumulative_loss=math.fsum(saved["guarded_losses"][:recorded]),
                    online_seconds=online, initialization_seconds=initialization,
                    prediction_seconds=prediction, update_seconds=update, input_seconds=input_time,
                    recording_seconds=recording_time, elapsed_seconds=perf_counter()-start,
                    cpu_seconds=process_time()-cpu_start, memory=memory_snapshot(), baseline_memory=baseline_memory,
                    threadpools=threadpool_info(), state_bytes=state_bytes,
                    actual_population=getattr(a, "m", None),
                    metrics=a.metrics.as_dict() if a is not None else {}, diagnostics=diagnostics,
                    diagnostics_error=diagnostic_error,
                    primary_prediction="Actual return from predict; selected is already c4 guarded, baselines retain native outputs",
                    secondary_guard="Output-only clip of raw logit to +/-log(4*t*(t+1)); does not alter states",
                    practical_caps_have_prescribed_regret_certificate=False,
                    finite_precision_certificate=False,
                    recording_seconds_scope="Trajectory array conversion/compression/write only; JSON snapshots and other recorder overhead remain in elapsed time",
                    timing_scope="Construction plus predict/observe; data loading, telemetry and compression excluded from online time")

    def checkpoint(current_status):
        nonlocal recording_time
        tick = perf_counter()
        recorded = len(counter_rows)
        arrays = {key: np.asarray(value[:recorded], dtype=(np.int64 if key in ("labels", "state_bytes") else np.float64))
                  for key, value in saved.items()}
        counter_names = sorted(set().union(*(row.keys() for row in counter_rows))) if counter_rows else []
        for key in counter_names:
            arrays["counter__"+key] = np.asarray([row.get(key, 0) for row in counter_rows], dtype=np.int64)
        temporary = output/"trajectory.tmp.npz"
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, output/"trajectory.npz")
        recording_time += perf_counter()-tick
        record = snapshot(current_status)
        atomic_json(output/"checkpoint.json", record)
        return record

    with threadpool_limits(limits=1):
        try:
            tick = perf_counter()
            X = np.fromfile(source/"X.bin", dtype="<f8", count=endpoint*d).reshape(endpoint, d)
            Y = np.fromfile(source/"y.bin", dtype="i1", count=endpoint)
            input_time += perf_counter()-tick
            if len(Y) != endpoint:
                raise EOFError("Incomplete label data")
            if not np.all(np.isfinite(X)) or not np.all(np.isin(Y, [-1, 1])):
                raise ValueError("Dataset contains invalid features or labels")
            phase = "initialization"
            tick = perf_counter()
            try:
                a = Algorithm(d, spec["B"], T, spec["seed"], spec["config"])
            finally:
                initialization = perf_counter()-tick
                online += initialization
            checkpoint("running")
            last_checkpoint = perf_counter()
            for t, (x, y) in enumerate(zip(X, Y), 1):
                if online >= spec["seconds"]:
                    status = "time_budget"
                    break
                phase = "predict"
                tick = perf_counter()
                try:
                    actual = float(a.predict(x))
                finally:
                    elapsed = perf_counter()-tick
                    prediction += elapsed
                    online += elapsed
                raw = float(a._last_schedule["raw_logit"]) if spec["method"] == "selected" else actual
                if not math.isfinite(actual) or not math.isfinite(raw):
                    raise FloatingPointError("Nonfinite predictive logit")
                phase = "observe"
                tick = perf_counter()
                try:
                    a.observe(x, int(y))
                finally:
                    elapsed = perf_counter()-tick
                    update += elapsed
                    online += elapsed
                phase = "record"
                clip = math.log(4.)+math.log(t)+math.log(t+1)
                guarded = min(max(raw, -clip), clip)
                memory = memory_snapshot()
                values = dict(output_logits=actual, output_losses=float(np.logaddexp(0., -int(y)*actual)),
                              raw_logits=raw, raw_losses=float(np.logaddexp(0., -int(y)*raw)),
                              guarded_logits=guarded, guarded_losses=float(np.logaddexp(0., -int(y)*guarded)),
                              online_times=online, labels=int(y), state_bytes=a.state_bytes(),
                              peak_private_commit=memory["peak_private_commit"], peak_working_set=memory["peak_working_set"])
                for key in keys:
                    saved[key].append(values[key])
                counter_rows.append(dict(a.metrics.counters))
                if t % 50 == 0 or perf_counter()-last_checkpoint >= 1.:
                    checkpoint("running")
                    last_checkpoint = perf_counter()
            if len(counter_rows) == endpoint:
                status = "completed"
            elif status == "running":
                status = "incomplete_input"
        except Exception as exc:
            status = "memory_error" if isinstance(exc, MemoryError) else "error"
            error = repr(exc)
            (output/"traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        final = checkpoint(status)
        atomic_json(output/"result.json", final)
    return final


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
