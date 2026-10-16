"""Stream one of four unchanged algorithms without retaining recorder history.

trajectory.bin has five little-endian float64 columns. progress.json publishes
only a flushed completed prefix; an interrupted file may contain additional
uncommitted rows or a partial final record. The recorder never changes a
method's returned prediction or its native sampling and numerical guards.
"""
import sys
JOB_VERIFICATION = None
if __name__ == "__main__" and "--gate" in sys.argv:
    import os, json
    from pathlib import Path
    ready = Path(sys.argv[1]).parent / "worker_ready.json"
    ready.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    if sys.stdin.readline().strip() != "GO":
        raise SystemExit("Worker did not receive its supervisor gate")
    from .worker_job import verify_current_limit
    JOB_VERIFICATION = verify_current_limit(2 * 1024**3)
    (ready.parent / "job_verification.json").write_text(json.dumps(JOB_VERIFICATION), encoding="utf-8")

import json
import math
import os
from pathlib import Path
from time import perf_counter, process_time
import traceback

# Set these before importing NumPy/SciPy, including for direct smoke runs.
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .data import stream_case
from .memory import memory_snapshot


COLUMNS = ("raw_logit", "output_logit", "loss", "online_seconds", "state_bytes")
RECORD_BYTES = 8*len(COLUMNS)


def algorithm_class(name):
    """Import only the requested unchanged implementation."""
    from importlib import import_module
    implementations = {
        "selected": ("computation_first_log_odds", "ComputationFirstLogOdds"),
        "ours": ("proximal", "ProximalLogOdds"),
        "sigmoid": ("sigmoid", "SigmoidMALA"),
        "aioli": ("aioli", "AIOLI"),
    }
    if name not in implementations:
        raise ValueError(f"Unknown algorithm: {name!r}")
    module, class_name = implementations[name]
    return getattr(import_module("."+module, __package__), class_name)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def run(spec):
    method = spec["algorithm"]
    Algorithm = algorithm_class(method)
    d, T, B = int(spec["d"]), int(spec["T"]), float(spec["B"])
    seconds = float(spec["seconds"])
    if d < 1 or T < 1 or d != spec["d"] or T != spec["T"]:
        raise ValueError("d and T must be positive integers")
    if not math.isfinite(B) or B <= 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("B and the online-time guard must be positive finite")
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output/"trajectory.bin"
    # A direct invocation cannot truncate an earlier completed or partial run.
    if any((output/name).exists() for name in ("trajectory.bin", "progress.json", "result.json")):
        raise FileExistsError("Use a fresh output directory for the long trajectory")
    start, cpu_start = perf_counter(), process_time()
    online = initialization = prediction = update = input_time = recording = 0.
    recorded_rounds = committed_round = 0
    cumulative_loss = loss_compensation = 0.
    committed_loss = committed_online = 0.
    status, error, error_phase, phase = "running", None, None, "initialization"
    a = None
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
            return dict(schema_version=1, job_verification=JOB_VERIFICATION, status=current_status, error=error, failure_phase=error_phase,
                        round=committed_round, committed_round=committed_round,
                        recorded_rounds=recorded_rounds, configured_horizon=T, target_rounds=T,
                        algorithm=method, algorithm_class=Algorithm.__name__, d=d, B=B, seed=spec["seed"],
                        config=dict(spec.get("config", {})),
                        actual_population=getattr(a, "m", getattr(a, "n_particles", None)),
                        cumulative_loss=committed_loss, online_seconds=online,
                        completed_prefix_online_seconds=committed_online,
                        initialization_seconds=initialization, prediction_seconds=prediction,
                        update_seconds=update, input_seconds=input_time, recording_seconds=recording,
                        elapsed_seconds=perf_counter()-start, cpu_seconds=process_time()-cpu_start,
                        memory=memory_snapshot(), baseline_memory=baseline, state_bytes=state_bytes,
                        metrics=a.metrics.as_dict() if a is not None else {}, diagnostics=diagnostics,
                        diagnostics_error=diagnostic_error, threadpools=threadpool_info(),
                        trajectory=dict(path=str(trajectory_path), columns=list(COLUMNS),
                                        dtype="little-endian float64", record_bytes=RECORD_BYTES,
                                        committed_bytes=committed_round*RECORD_BYTES,
                                        file_bytes=trajectory_path.stat().st_size),
                        checkpoint_scope="Use only committed_round complete records; subsequent file bytes are uncommitted. Error diagnostics/timing may include an unfinished next round.",
                        checkpoint_flush_policy="Flush before atomic progress update every 5 seconds or 512 rounds and at termination",
                        timing_scope="Construction plus predict/observe; streaming input and recording excluded from online time",
                        primary_prediction=("Actual clipped predict output; raw pre-clipping logit retained separately"
                                            if method == "selected" else
                                            "Actual native predict output; raw_logit equals output_logit; no recorder clipping"),
                        input_block_rows=128, recorder_retains_per_round_history=False,
                        practical_caps_have_prescribed_regret_certificate=False,
                        finite_precision_certificate=False)

        def checkpoint(current_status):
            nonlocal committed_round, committed_loss, committed_online, recording, last_checkpoint, phase
            previous_phase, phase = phase, "checkpoint"
            tick = perf_counter()
            trajectory.flush()
            committed_round = recorded_rounds
            committed_loss = cumulative_loss
            # For an error, online may include time spent on an unrecorded
            # next prediction. The stored completed-prefix timer stays separate.
            if recorded_rounds == 0:
                committed_online = 0.
            record = snapshot(current_status)
            atomic_json(output/"progress.json", record)
            recording += perf_counter()-tick
            last_checkpoint = perf_counter()
            phase = previous_phase
            return record

        try:
            tick = perf_counter()
            try:
                a = Algorithm(d, B, T, spec["seed"], spec.get("config", {}))
            finally:
                initialization = perf_counter()-tick
                online += initialization
            checkpoint("running")
            iterator = iter(stream_case(spec["data_path"], d, T))
            stopped = False
            while not stopped:
                if online >= seconds:
                    status = "time_budget"
                    break
                phase = "input"
                tick = perf_counter()
                try:
                    xb, yb = next(iterator)
                    if len(xb) > 128 or not np.all(np.isfinite(xb)) or not np.all(np.isin(yb, [-1, 1])):
                        raise ValueError("Invalid streamed feature/label block")
                except StopIteration:
                    break
                finally:
                    input_time += perf_counter()-tick
                for x, y in zip(xb, yb):
                    if online >= seconds:
                        status, stopped = "time_budget", True
                        break
                    phase = "predict"
                    tick = perf_counter()
                    try:
                        actual = float(a.predict(x))
                    finally:
                        dt = perf_counter()-tick
                        prediction += dt
                        online += dt
                    raw = float(a._last_schedule["raw_logit"]) if method == "selected" else actual
                    if not math.isfinite(actual) or not math.isfinite(raw):
                        raise FloatingPointError("Nonfinite predictive logit")
                    phase = "observe"
                    tick = perf_counter()
                    try:
                        a.observe(x, int(y))
                    finally:
                        dt = perf_counter()-tick
                        update += dt
                        online += dt
                    phase = "record"
                    tick = perf_counter()
                    loss = float(np.logaddexp(0., -int(y)*actual))
                    state = float(a.state_bytes())
                    if not all(math.isfinite(value) for value in (loss, online, state)) or loss < 0 or state < 0:
                        raise FloatingPointError("Invalid loss, cumulative time, or state size")
                    payload = np.asarray((raw, actual, loss, online, state), dtype="<f8").tobytes()
                    written = trajectory.write(payload)
                    if written != RECORD_BYTES:
                        raise OSError("Partial trajectory record write")
                    recorded_rounds += 1
                    # Constant-space compensated sum; no retained loss list.
                    adjusted = loss-loss_compensation
                    next_sum = cumulative_loss+adjusted
                    loss_compensation = (next_sum-cumulative_loss)-adjusted
                    cumulative_loss = next_sum
                    committed_online = online
                    recording += perf_counter()-tick
                    if recorded_rounds % 512 == 0 or perf_counter()-last_checkpoint >= 5. or recorded_rounds == T:
                        checkpoint("running")
            if recorded_rounds == T:
                status = "completed"
            elif status == "running":
                status = "incomplete_input"
        except Exception as exc:
            status = "memory_error" if isinstance(exc, MemoryError) else "error"
            error, error_phase = repr(exc), phase
            (output/"traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        final = checkpoint(status)
        final["recording_seconds"] = recording
        atomic_json(output/"progress.json", final)
        atomic_json(output/"result.json", final)
    return final


if __name__ == "__main__":
    result = run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    print(json.dumps({key: result[key] for key in ("status", "committed_round", "cumulative_loss", "online_seconds")}))
