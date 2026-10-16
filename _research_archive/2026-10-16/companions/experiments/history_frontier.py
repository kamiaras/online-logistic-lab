"""History kernel probe, not online regret experiment.

Example (run probes sequentially, outside another timing campaign)::

    python -m experiments.history_frontier --algorithm ours --dimension 1000 \
        --history-mib 1792 --particles 2 --seconds 120 --output results/probe

This deliberately preloads synthetic history without executing its online
predictions. The cloud remains at its sampled prior until one real predict call.
There is no regret, posterior-mixing, or complete-trajectory measurement here.
Only this module's subprocess supervisor may start the numerical worker.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from time import perf_counter, process_time, sleep


ROOT = Path(__file__).resolve().parents[1]
LABEL = "history kernel probe, not online regret experiment"
MIB = 1024 ** 2
BLOCK_ROWS = 1024


def _target_rows(dimension, history_mib=None, rows=None):
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if rows is not None:
        if rows < 1:
            raise ValueError("rows must be positive")
        return int(rows)
    if history_mib is None or not math.isfinite(history_mib) or history_mib <= 0:
        raise ValueError("history-mib must be finite and positive")
    blocks = int(history_mib * MIB) // (8 * BLOCK_ROWS * (dimension + 1))
    if blocks < 1:
        raise ValueError("history-mib must accommodate one allocated history block")
    return blocks * BLOCK_ROWS


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _probe_config(algorithm, d, rows, particles):
    """Relax constructor advice only; the subprocess OS limit stays unchanged."""
    config = {"particles": particles, "transitions": 1}
    overrides = {}
    if algorithm == "ours":
        horizon = rows + 1
        # Match the preflight's full-horizon capacity, using requested m as an
        # upper bound on actual m. This may exceed the OS cap intentionally:
        # allocation must reach the actual memory boundary, not a 512 MiB guard.
        cells = ((horizon + 1023) // 1024) * 1024 * (d + 1)
        cells += particles * (d + 1) + 16 * d + 128
        overrides = {"max_state_bytes": max(2112 * MIB, 8 * cells + 64 * MIB),
                     "max_particles": max(100000, particles)}
        config.update(overrides)
    else:
        config.update(bridges=1, lazy=False, history_block_size=BLOCK_ROWS)
    return config, overrides


def _prepare_valid_history(history):
    """Discard only wholly uninitialized trailing capacity after failed append.

    Allocation of the pair in History.append is atomic with respect to its
    block list. A later row-copy failure can nevertheless leave an empty tail
    block. Partial last blocks are retained; batches slices their valid prefix.
    """
    n = history.n
    before_blocks = len(history.blocks)
    before_bytes = history.nbytes
    capacity_before = sum(len(x) for x, _ in history.blocks)
    if n < 0 or n > capacity_before:
        raise RuntimeError("History length exceeds allocated capacity")
    keep_blocks = (n + history.block_size - 1) // history.block_size
    while len(history.blocks) > keep_blocks:
        history.blocks.pop()
    valid_rows = 0
    for x, y in history.batches():
        if len(x) == 0 or len(x) != len(y):
            raise RuntimeError("History iteration exposed an empty or inconsistent batch")
        valid_rows += len(x)
    if valid_rows != n:
        raise RuntimeError("History iteration does not match the valid row count")
    return dict(valid_rows=n, valid_batch_rows=valid_rows,
                capacity_rows_before=capacity_before,
                capacity_rows_after=sum(len(x) for x, _ in history.blocks),
                blocks_before=before_blocks, blocks_after=len(history.blocks),
                removed_wholly_unused_blocks=before_blocks - len(history.blocks),
                removed_wholly_unused_ndarray_bytes=before_bytes - history.nbytes,
                partially_filled_final_block_rows=n % history.block_size,
                only_valid_rows_exposed=True)


def _numerical_worker(spec):
    # Called only after the supervisor's gate, with the OS limit installed.
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits
    from .memory import memory_snapshot
    from .proximal import ProximalLogOdds
    from .sigmoid import SigmoidMALA

    out = Path(spec["output"])
    d, target = spec["d"], spec["history_rows"]
    start, cpu_start = perf_counter(), process_time()
    a = None
    phase, append_status, predict_status = "initialization", "not_started", "not_started"
    append_error = predict_error = None
    generation_seconds = append_seconds = init_seconds = predict_seconds = 0.
    generated_rows = uniform_draws = label_draws = 0
    append_attempts = 0
    score = None
    prediction_calls = 0
    before_predict_metrics = after_predict_metrics = None
    history_validity = None
    baseline = memory_snapshot()
    actual_particles = 0
    max_input_buffer_bytes = 0
    generation_block_rows = max(1, min(spec["stream_rows"],
                                      spec["input_buffer_bytes"] // (8 * d + 16)))
    rng = np.random.default_rng(spec["data_seed"])
    block = labels = norms = None
    deadline = start + spec["seconds"]
    last_checkpoint = start

    def snapshot(status):
        n = a.history.n if a is not None else 0
        capacity = len(a.history.blocks) * BLOCK_ROWS if a is not None else 0
        return dict(
            label=LABEL, status=status, phase=phase,
            algorithm=spec["algorithm"], dimension=d, B=spec["B"],
            configured_horizon=target + 1, target_history_rows=target,
            retained_history_rows=n, allocated_history_capacity_rows=capacity,
            actual_particles=actual_particles, executed_online_rounds=0,
            algorithm_config=spec["algorithm_config"],
            advisory_constructor_overrides=spec["advisory_constructor_overrides"],
            hard_process_commit_limit=spec["memory_bytes"],
            preload_is_executed_online_trajectory=False,
            initial_cloud="independent prior draws, unchanged during preload",
            history_append_calls=append_attempts, history_append_completed_calls=n,
            history_feature_scalar_writes=n * d,
            history_label_scalar_writes=n,
            failed_append_may_have_partial_writes=append_attempts > n,
            history_validity_before_predict=history_validity,
            history_ndarray_bytes=a.history.nbytes if a is not None else 0,
            owned_ndarray_bytes=a.state_bytes() if a is not None else 0,
            memory=memory_snapshot(), baseline_memory=baseline,
            append_status=append_status, append_error=append_error,
            predict_status=predict_status, predict_error=predict_error,
            prediction_logit=score, prediction_calls=prediction_calls,
            generated_rows=generated_rows, synthetic_uniform_draws=uniform_draws,
            synthetic_label_draws=label_draws,
            generation_block_rows=generation_block_rows,
            maximum_input_ndarray_bytes=max_input_buffer_bytes,
            generation_seconds=generation_seconds, append_seconds=append_seconds,
            initialization_seconds=init_seconds, predict_seconds=predict_seconds,
            elapsed_seconds=perf_counter() - start,
            cpu_seconds=process_time() - cpu_start,
            metrics=a.metrics.as_dict() if a is not None else {},
            diagnostics=a.diagnostics() if a is not None else {},
            metrics_before_predict=before_predict_metrics,
            metrics_after_predict=after_predict_metrics,
            operation_scope=("append calls and sampler event counters are exact executed counts; "
                             "scalar-write counts cover completed appended rows only, excluding "
                             "any partial writes in a failed append; instrumented arithmetic "
                             "counters are kernel-work estimates, not hardware instructions"),
            input_distribution="iid dense Rademacher coordinates +/-0.5/sqrt(d); iid +/-1 labels",
            query_definition="first coordinate unit vector; no query label is supplied",
        )

    with (out / "progress.jsonl").open("w", encoding="utf-8") as progress:
        def checkpoint(status="running"):
            record = snapshot(status)
            progress.write(json.dumps(record, allow_nan=False) + "\n")
            progress.flush()
            return record

        checkpoint()
        try:
            with threadpool_limits(limits=1):
                t = perf_counter()
                try:
                    if spec["algorithm"] == "ours":
                        a = ProximalLogOdds(d, spec["B"], target + 1, spec["sampler_seed"],
                                             spec["algorithm_config"])
                        actual_particles = a.m
                    else:
                        a = SigmoidMALA(d, spec["B"], target + 1, spec["sampler_seed"],
                                       spec["algorithm_config"])
                        actual_particles = a.n_particles
                finally:
                    init_seconds += perf_counter() - t
                phase, append_status = "history_append", "running"
                checkpoint()
                try:
                    while a.history.n < target:
                        if perf_counter() >= deadline:
                            append_status = "time_budget"
                            break
                        count = min(generation_block_rows, target - a.history.n)
                        t = perf_counter()
                        try:
                            block = rng.random((count, d))
                            uniform_draws += count * d
                            # In-place conversion bounds temporary storage independently of n.
                            np.greater_equal(block, .5, out=block)
                            block *= 2.
                            block -= 1.
                            block *= .5 / math.sqrt(d)
                            labels = rng.integers(0, 2, size=count, dtype=np.int8)
                            label_draws += count
                            labels *= 2
                            labels -= 1
                            norms = np.einsum("ij,ij->i", block, block)
                            generated_rows += count
                            max_input_buffer_bytes = max(max_input_buffer_bytes,
                                                         block.nbytes + labels.nbytes + norms.nbytes)
                        finally:
                            generation_seconds += perf_counter() - t
                        t = perf_counter()
                        try:
                            for i in range(count):
                                append_attempts += 1
                                a.history.append(block[i], int(labels[i]))
                                if spec["algorithm"] == "ours":
                                    a.sum_norm2 += float(norms[i])
                                else:
                                    a._squared_feature_sum += float(norms[i])
                        finally:
                            append_seconds += perf_counter() - t
                        block = labels = norms = None
                        if perf_counter() - last_checkpoint >= 1.:
                            checkpoint()
                            last_checkpoint = perf_counter()
                    else:
                        append_status = "completed"
                except Exception as exc:
                    append_status = "memory_error" if isinstance(exc, MemoryError) else "error"
                    append_error = repr(exc)
                finally:
                    # Free the bounded input buffers even if appending a history
                    # block failed; then test the retained prefix if time remains.
                    block = labels = norms = None
                    checkpoint()

                phase = "history_validation"
                history_validity = _prepare_valid_history(a.history)
                checkpoint()
                if a.history.n and perf_counter() < deadline:
                    phase, predict_status = "one_predict", "running"
                    before_predict_metrics = a.metrics.as_dict()
                    checkpoint()
                    call_start = None
                    try:
                        query = np.zeros(d)
                        query[0] = 1.
                        prediction_calls += 1
                        checkpoint()
                        call_start = perf_counter()
                        score = float(a.predict(query))
                        if not math.isfinite(score):
                            score = None
                            raise FloatingPointError("non-finite prediction logit")
                        predict_status = "completed"
                    except Exception as exc:
                        predict_status = "memory_error" if isinstance(exc, MemoryError) else "error"
                        predict_error = repr(exc)
                    finally:
                        if call_start is not None:
                            predict_seconds += perf_counter() - call_start
                        after_predict_metrics = a.metrics.as_dict()
                        checkpoint()
                elif append_status == "time_budget":
                    predict_status = "skipped_time_budget"
                phase = "finished"
                status = ("completed" if append_status == predict_status == "completed"
                          else "partial_history_predict_completed" if predict_status == "completed"
                          else "incomplete")
        except Exception as exc:
            status = "memory_error" if isinstance(exc, MemoryError) else "error"
            append_error = repr(exc)
        result = checkpoint(status)
    result["threadpools"] = threadpool_info()
    result["numpy_version"] = np.__version__
    _write_json(out / "result.json", result)
    return result


def run_one(spec):
    """Start a fresh process; the parent never imports numerical libraries."""
    import psutil
    from .memory import ProcessLimit

    if os.name != "nt":
        raise RuntimeError("This probe requires Windows Job Object hard commit isolation")
    if not 0 < spec["memory_bytes"] <= 2048 * MIB:
        raise ValueError("Hard process commit limit must be positive and at most 2 GiB")
    out = Path(spec["output"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config, overrides = _probe_config(spec["algorithm"], spec["d"],
                                      spec["history_rows"], spec["particles"])
    spec = {**spec, "output": str(out), "algorithm_config": config,
            "advisory_constructor_overrides": overrides}
    spec_path = out / "spec.json"
    if (out / "supervisor.json").exists():
        if json.loads(spec_path.read_text(encoding="utf-8")) != spec:
            raise ValueError("Existing output has a different specification")
        return json.loads((out / "supervisor.json").read_text(encoding="utf-8"))
    _write_json(spec_path, spec)
    hashes = {}
    for name in ("history_frontier.py", "common.py", "proximal.py", "sigmoid.py", "memory.py"):
        hashes[name] = hashlib.sha256((ROOT / "experiments" / name).read_bytes()).hexdigest()
    _write_json(out / "environment.json", dict(python=sys.version, executable=sys.executable,
                platform=platform.platform(), source_sha256=hashes, worker_blas_threads=1,
                memory_available=psutil.virtual_memory().available))
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    job = proc = None
    hard_limit_active = False
    reason = None
    peak_private = peak_rss = 0
    start = perf_counter()
    with (out / "stdout.txt").open("w") as stdout, (out / "stderr.txt").open("w") as stderr:
        try:
            proc = subprocess.Popen([sys.executable, "-m", "experiments.history_frontier",
                                     "--worker", str(spec_path)], cwd=ROOT, env=env,
                                    stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                    text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            job = ProcessLimit(proc._handle, spec["memory_bytes"])
            hard_limit_active = job.handle is not None
            if not hard_limit_active:
                raise RuntimeError("OS hard process commit limit was not installed")
            proc.stdin.write("GO\n")
            proc.stdin.flush()
            proc.stdin.close()
            monitored = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    mem = monitored.memory_info()
                    peak_rss = max(peak_rss, mem.rss)
                    private = getattr(mem, "private", mem.vms)
                    peak_private = max(peak_private, private)
                    if private > spec["memory_bytes"]:
                        reason = "memory_limit"
                        proc.kill()
                        break
                except psutil.NoSuchProcess:
                    break
                if perf_counter() - start > spec["seconds"] + 20.:
                    reason = "external_timeout"
                    proc.kill()
                    break
                sleep(.1)
            proc.wait()
        except Exception as exc:
            reason = "supervisor_error: " + repr(exc)
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
        finally:
            if job is not None:
                job.close()
    result = dict(label=LABEL, process_wall_seconds=perf_counter() - start,
                  returncode=proc.returncode if proc else None, termination=reason,
                  monitored_peak_private=peak_private, monitored_peak_rss=peak_rss,
                  hard_process_commit_limit=spec["memory_bytes"],
                  os_hard_limit_active=hard_limit_active, checkpoint_is_censored=False)
    try:
        result["worker"] = json.loads((out / "result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        result["checkpoint_is_censored"] = True
        result["worker"] = dict(label=LABEL, status=reason or "abnormal_exit")
        progress = out / "progress.jsonl"
        if progress.exists():
            for line in reversed(progress.read_text(encoding="utf-8").splitlines()):
                try:
                    result["worker"] = json.loads(line)
                    result["worker"]["status"] = reason or "abnormal_exit"
                    break
                except json.JSONDecodeError:
                    continue
    _write_json(out / "supervisor.json", result)
    return result


def _spec(args):
    if not (math.isfinite(args.B) and args.B > 0 and args.particles > 0
            and math.isfinite(args.seconds) and args.seconds > 0
            and 0 < args.memory_mib <= 2048 and args.stream_rows > 0):
        raise ValueError("Positive finite budgets required; memory-mib cannot exceed 2048")
    rows = _target_rows(args.dimension, args.history_mib, args.rows)
    config, overrides = _probe_config(args.algorithm, args.dimension, rows, args.particles)
    return dict(label=LABEL, algorithm=args.algorithm, d=args.dimension, B=args.B,
                history_rows=rows, algorithm_config=config,
                advisory_constructor_overrides=overrides,
                requested_history_mib=args.history_mib, particles=args.particles,
                seconds=args.seconds, memory_bytes=int(args.memory_mib * MIB),
                data_seed=20260910, sampler_seed=93071, stream_rows=args.stream_rows,
                input_buffer_bytes=2 * MIB, output=str(Path(args.output).resolve()))


def self_test():
    """Tiny isolated integration checks; run only outside a timed campaign."""
    import numpy as np
    from unittest.mock import patch
    from .common import History, potential

    assert _target_rows(3, rows=7) == 7
    assert _target_rows(3, history_mib=1) == 32 * BLOCK_ROWS
    config, overrides = _probe_config("ours", 1000, 400000, 200001)
    cells = ((400001 + 1023) // 1024) * 1024 * 1001 + 200001 * 1001 + 16000 + 128
    assert config["max_state_bytes"] > 8 * cells > 2048 * MIB
    assert overrides["max_particles"] >= 200001
    # Failure in allocating the second array does not insert half of a block.
    history = History(3, block_size=2)
    for _ in range(2):
        history.append(np.zeros(3), 1.)
    empty = np.empty
    allocations = 0

    def fail_second_array(*args, **kwargs):
        nonlocal allocations
        allocations += 1
        if allocations == 2:
            raise MemoryError("injected second-array allocation failure")
        return empty(*args, **kwargs)

    with patch("experiments.common.np.empty", side_effect=fail_second_array):
        try:
            history.append(np.zeros(3), 1.)
        except MemoryError:
            pass
        else:
            raise AssertionError("Injected allocation failure did not occur")
    assert history.n == 2 and len(history.blocks) == 1
    # Model the later failure point: block insertion succeeded, but row copy
    # did not commit n. Only this wholly unused tail capacity may be discarded.
    history.blocks.append((np.full((2, 3), np.nan), np.full(2, np.nan)))
    validity = _prepare_valid_history(history)
    assert validity["removed_wholly_unused_blocks"] == 1
    assert validity["capacity_rows_before"] == 4
    assert validity["valid_batch_rows"] == 2
    history.append(np.ones(3) / 4., -1.)
    history.blocks[-1][0][1] = np.nan
    history.blocks[-1][1][1] = np.nan
    validity = _prepare_valid_history(history)
    assert validity["removed_wholly_unused_blocks"] == 0
    assert validity["partially_filled_final_block_rows"] == 1
    values, gradients = potential(np.zeros((1, 3)), 1., history)
    assert np.isfinite(values).all() and np.isfinite(gradients).all()
    for d, mib in ((0, 1), (1000, .01)):
        try:
            _target_rows(d, history_mib=mib)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid history budget accepted")
    with tempfile.TemporaryDirectory(prefix="history_frontier_test_") as folder:
        for algorithm in ("ours", "sigmoid"):
            args = argparse.Namespace(algorithm=algorithm, dimension=3, B=1., rows=32,
                     history_mib=None, particles=2, seconds=20., memory_mib=256.,
                     stream_rows=7, output=str(Path(folder) / algorithm))
            result = run_one(_spec(args))
            worker = result["worker"]
            assert result["os_hard_limit_active"]
            assert worker["status"] == "completed", result
            assert worker["executed_online_rounds"] == 0
            assert worker["history_append_calls"] == 32
            assert worker["history_append_completed_calls"] == 32
            assert worker["history_feature_scalar_writes"] == 96
            assert worker["history_ndarray_bytes"] == 8 * BLOCK_ROWS * 4
            assert worker["prediction_calls"] == 1
            assert worker["history_validity_before_predict"]["only_valid_rows_exposed"]
            if algorithm == "ours":
                assert worker["advisory_constructor_overrides"]["max_state_bytes"] > 2048 * MIB
            before = worker["metrics_before_predict"]["counters"]
            after = worker["metrics_after_predict"]["counters"]
            assert after["history_particle_pairs"] > before.get("history_particle_pairs", 0)
            assert math.isfinite(worker["prediction_logit"])
    print("History frontier small checks passed; no large probe was run.")


def main():
    parser = argparse.ArgumentParser(description=LABEL)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--algorithm", choices=("ours", "sigmoid"), default="ours")
    parser.add_argument("--dimension", type=int, default=1000)
    parser.add_argument("--B", type=float, default=1.)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--history-mib", type=float)
    target.add_argument("--rows", type=int)
    parser.add_argument("--particles", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--memory-mib", type=float, default=2048.)
    parser.add_argument("--stream-rows", type=int, default=32)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.worker:
        if sys.stdin.readline().strip() != "GO":
            raise SystemExit("Worker not authorized by supervisor")
        _numerical_worker(json.loads(Path(args.worker).read_text(encoding="utf-8")))
    elif args.self_test:
        self_test()
    else:
        if args.output is None:
            parser.error("--output is required")
        if args.history_mib is None and args.rows is None:
            args.history_mib = 1792.
        result = run_one(_spec(args))
        worker = result["worker"]
        print(json.dumps(dict(label=LABEL, status=worker.get("status"),
              retained_history_rows=worker.get("retained_history_rows"),
              predict_status=worker.get("predict_status"), output=args.output)))


if __name__ == "__main__":
    main()
