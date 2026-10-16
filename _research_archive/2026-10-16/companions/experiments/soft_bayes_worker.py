"""Isolated, fully completed-round recorder for the soft-Bayes / released AIOLI comparison.

The supervisor must attach the actual interpreter to its Windows Job Object
before this module imports any numerical library. Only a flushed checkpoint
prefix is committed; records beyond that prefix after a kill are censored.
"""
import sys

JOB_VERIFICATION = None
if __name__ == "__main__":
    import json
    import os
    from pathlib import Path

    if len(sys.argv) != 3 or sys.argv[2] != "--gate":
        raise SystemExit("Launch through experiments.soft_bayes_supervisor")
    _spec_path = Path(sys.argv[1])
    _early_spec = json.loads(_spec_path.read_text(encoding="utf-8"))
    _cap = int(_early_spec.get("memory_bytes", 2 * 1024**3))
    if _cap != 2 * 1024**3:
        raise SystemExit("This benchmark requires the prescribed hard 2 GiB cap")
    _ready = _spec_path.parent / "worker_ready.json"
    _ready.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    if sys.stdin.readline().strip() != "GO":
        raise SystemExit("Worker did not receive its supervisor gate")
    from .worker_job import verify_current_limit
    JOB_VERIFICATION = verify_current_limit(_cap)
    (_ready.parent / "job_verification.json").write_text(
        json.dumps(JOB_VERIFICATION), encoding="utf-8")

import json
import math
import os
from pathlib import Path
from time import perf_counter, process_time
import traceback

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .data import stream_case
from .memory import memory_snapshot

COLUMNS = ("raw_logit", "output_logit", "loss", "online_seconds",
           "state_bytes", "author_clipped_loss")
RECORD_BYTES = 8 * len(COLUMNS)


def algorithm_class(name):
    if name == "soft_bayes":
        from .soft_bayes import SoftBayes
        return SoftBayes
    if name == "author_aioli":
        from .author_baselines import AuthorAIOLI
        return AuthorAIOLI
    raise ValueError(f"Unknown algorithm: {name!r}")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _safe_diagnostics(value, prefix="diagnostics", replaced=None):
    """Report nonfinite optional diagnostics explicitly, never emit JSON NaN."""
    if replaced is None:
        replaced = []
    if isinstance(value, dict):
        result = {str(k): _safe_diagnostics(v, f"{prefix}.{k}", replaced)[0]
                  for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        result = [_safe_diagnostics(v, f"{prefix}[{i}]", replaced)[0]
                  for i, v in enumerate(value)]
    elif isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            replaced.append(prefix)
            result = None
    elif isinstance(value, np.integer):
        result = int(value)
    elif isinstance(value, np.bool_):
        result = bool(value)
    else:
        result = value
    return result, replaced


def run(spec):
    if JOB_VERIFICATION is None:
        raise RuntimeError("A verified supervisor gate is required")
    method = spec["algorithm"]
    d, T, B = int(spec["d"]), int(spec["T"]), float(spec["B"])
    endpoint, seconds = int(spec.get("rounds", T)), float(spec["seconds"])
    if d < 1 or T < 1 or d != spec["d"] or T != spec["T"]:
        raise ValueError("d and T must be positive integers")
    if not math.isfinite(B) or B <= 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("B and the online-time guard must be positive finite")
    if endpoint != spec.get("rounds", T) or not 1 <= endpoint <= T:
        raise ValueError("rounds must be an integer from 1 through configured T")
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "trajectory.bin"
    stop_path = Path(spec.get("stop_path", output.parent / "STOP_REQUESTED"))
    if any((output / name).exists() for name in ("trajectory.bin", "progress.json", "result.json")):
        raise FileExistsError("Use a fresh output directory")
    config = dict(spec.get("config", {}))
    is_author = method == "author_aioli"
    start, cpu_start = perf_counter(), process_time()
    online = initialization = prediction = update = input_time = recording = 0.
    native_prediction = native_update = 0.
    recorded_rounds = committed_round = 0
    cumulative_loss = loss_compensation = 0.
    committed_loss = committed_online = 0.
    author_cumulative_loss = author_loss_compensation = 0.
    committed_author_loss = 0. if is_author else None
    status, error, error_phase, phase = "running", None, None, "algorithm_import"
    a, Algorithm = None, None
    baseline = memory_snapshot()
    last_checkpoint = perf_counter()

    with trajectory_path.open("xb") as trajectory, threadpool_limits(limits=1):
        def snapshot(current_status):
            diagnostic_error = None
            try:
                diagnostics = a.diagnostics() if a is not None else {}
                state_bytes = a.state_bytes() if a is not None else 0
            except Exception as exc:
                diagnostics, state_bytes, diagnostic_error = {}, None, repr(exc)
            diagnostics, replacements = _safe_diagnostics(diagnostics)
            metrics, metric_replacements = _safe_diagnostics(
                a.metrics.as_dict() if a is not None else {}, "metrics")
            return dict(
                schema_version=1, job_verification=JOB_VERIFICATION,
                status=current_status, error=error, failure_phase=error_phase,
                round=committed_round, committed_round=committed_round,
                recorded_rounds=recorded_rounds, configured_horizon=T,
                target_rounds=endpoint, algorithm=method,
                algorithm_class=Algorithm.__name__ if Algorithm is not None else None,
                native_algorithm_class=type(a.model).__name__ if a is not None and hasattr(a, "model") else None,
                d=d, B=B, seed=spec["seed"], config=config,
                author_commit=spec.get("author_commit"), author_source=spec.get("author_source"),
                actual_population=getattr(a, "m", getattr(a, "n_particles", getattr(getattr(a, "model", None), "S", None))),
                cumulative_loss=committed_loss, online_seconds=online,
                author_reported_cumulative_loss=committed_author_loss,
                completed_prefix_online_seconds=committed_online,
                initialization_seconds=initialization, prediction_seconds=prediction,
                update_seconds=update, native_prediction_seconds=native_prediction,
                native_update_seconds=native_update,
                adapter_overhead_seconds=max(0., prediction + update - native_prediction - native_update) if is_author else None,
                input_seconds=input_time, recording_seconds=recording,
                elapsed_seconds=perf_counter() - start, cpu_seconds=process_time() - cpu_start,
                memory=memory_snapshot(), baseline_memory=baseline, state_bytes=state_bytes,
                metrics=metrics, diagnostics=diagnostics, diagnostics_error=diagnostic_error,
                nonfinite_diagnostics_replaced_with_null=replacements + metric_replacements,
                threadpools=threadpool_info(),
                trajectory=dict(path=str(trajectory_path), columns=list(COLUMNS),
                                dtype="little-endian float64", record_bytes=RECORD_BYTES,
                                committed_bytes=committed_round * RECORD_BYTES,
                                file_bytes=trajectory_path.stat().st_size),
                checkpoint_scope="Only committed_round complete records are published. Later bytes are uncommitted. Error timing/diagnostics may include an unfinished next round.",
                checkpoint_flush_policy="Flush trajectory before atomic progress every completed round and on termination",
                timing_scope="Online time is algorithm construction plus predict/observe. Numerical-library/module imports, input streaming, and recording are excluded. Native author function times are separate subsets.",
                primary_prediction="Actual returned output logit with stable logistic loss; raw_logit equals output_logit",
                author_clipped_loss_scope="Unchanged author's separately probability-clipped loss; NaN binary sentinel for other methods",
                input_block_rows=128, recorder_retains_per_round_history=False,
                prediction_diagnostics_change_native_updates=False,
                practical_caps_have_prescribed_regret_certificate=False,
                finite_precision_certificate=False)

        def checkpoint(current_status):
            nonlocal committed_round, committed_loss, committed_author_loss, committed_online
            nonlocal recording, last_checkpoint, phase
            previous_phase, phase = phase, "checkpoint"
            tick = perf_counter()
            trajectory.flush()
            committed_round, committed_loss = recorded_rounds, cumulative_loss
            committed_author_loss = author_cumulative_loss if is_author else None
            if recorded_rounds == 0:
                committed_online = 0.
            record = snapshot(current_status)
            atomic_json(output / "progress.json", record)
            recording += perf_counter() - tick
            last_checkpoint, phase = perf_counter(), previous_phase
            return record

        try:
            Algorithm = algorithm_class(method)
            if is_author:
                # Native dependencies load lazily in AuthorAIOLI.__init__.
                # Charge imports to neither predictor's online initialization.
                from .author_baselines import author_module
                author_module()
            phase = "initialization"
            tick = perf_counter()
            try:
                a = Algorithm(d, B, T, spec["seed"], config)
            finally:
                initialization = perf_counter() - tick
                online += initialization
            checkpoint("running")
            iterator = iter(stream_case(spec["data_path"], d, endpoint))
            stopped = False
            while not stopped:
                if stop_path.exists():
                    status = "cancelled_by_user"
                    break
                if online >= seconds:
                    status = "time_budget"
                    break
                phase, tick = "input", perf_counter()
                try:
                    xb, yb = next(iterator)
                    if len(xb) > 128 or not np.all(np.isfinite(xb)) or not np.all(np.isin(yb, [-1, 1])):
                        raise ValueError("Invalid streamed feature/label block")
                except StopIteration:
                    break
                finally:
                    input_time += perf_counter() - tick
                for x, y in zip(xb, yb):
                    if stop_path.exists():
                        status, stopped = "cancelled_by_user", True
                        break
                    if online >= seconds:
                        status, stopped = "time_budget", True
                        break
                    phase, tick = "predict", perf_counter()
                    try:
                        actual = float(a.predict(x))
                    finally:
                        dt = perf_counter() - tick
                        prediction += dt
                        online += dt
                        native_prediction += float(getattr(a, "last_native_predict_seconds", 0.))
                    raw = actual
                    if not math.isfinite(actual) or not math.isfinite(raw):
                        raise FloatingPointError("Nonfinite predictive logit")
                    phase, tick = "observe", perf_counter()
                    try:
                        a.observe(x, int(y))
                    finally:
                        dt = perf_counter() - tick
                        update += dt
                        online += dt
                        native_update += float(getattr(a, "last_native_update_seconds", 0.))
                    phase, tick = "record", perf_counter()
                    loss = float(np.logaddexp(0., -int(y) * actual))
                    reported_loss = float(a.last_reported_loss) if is_author else math.nan
                    state = float(a.state_bytes())
                    if not all(math.isfinite(v) for v in (loss, online, state)) or loss < 0 or state < 0:
                        raise FloatingPointError("Invalid loss, cumulative time, or state size")
                    if is_author and (not math.isfinite(reported_loss) or reported_loss < 0):
                        raise FloatingPointError("Invalid author clipped loss")
                    payload = np.asarray((raw, actual, loss, online, state, reported_loss), dtype="<f8").tobytes()
                    if trajectory.write(payload) != RECORD_BYTES:
                        raise OSError("Partial trajectory record write")
                    recorded_rounds += 1
                    adjusted = loss - loss_compensation
                    next_sum = cumulative_loss + adjusted
                    loss_compensation = (next_sum - cumulative_loss) - adjusted
                    cumulative_loss = next_sum
                    if is_author:
                        adjusted = reported_loss - author_loss_compensation
                        next_sum = author_cumulative_loss + adjusted
                        author_loss_compensation = (next_sum - author_cumulative_loss) - adjusted
                        author_cumulative_loss = next_sum
                    committed_online = online
                    recording += perf_counter() - tick
                    checkpoint("running")
            if recorded_rounds == endpoint:
                status = "completed"
            elif status == "running":
                status = "incomplete_input"
        except Exception as exc:
            status = "memory_error" if isinstance(exc, MemoryError) else "error"
            error, error_phase = repr(exc), phase
            (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        final = checkpoint(status)
        final["recording_seconds"] = recording
        atomic_json(output / "progress.json", final)
        atomic_json(output / "result.json", final)
    return final


if __name__ == "__main__":
    _result = run(_early_spec)
    print(json.dumps({key: _result[key] for key in
                      ("status", "committed_round", "cumulative_loss", "online_seconds")}))
