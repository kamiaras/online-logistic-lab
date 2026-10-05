"""One streamed, hard-memory-limited full-horizon computation-first test."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep

import psutil

from .computation_first_campaign import assert_idle, ROOT, SOURCES
from .memory import ProcessLimit

CAP = 2 * 1024**3


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def digest(paths):
    result = {}
    for path in paths:
        h = hashlib.sha256()
        with (ROOT / path).open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024**2), b""):
                h.update(block)
        result[str(path)] = h.hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--B", type=float, default=1000.)
    parser.add_argument("--T", type=int, default=100000)
    parser.add_argument("--seconds", type=float, default=21600.)
    parser.add_argument("--data-seed", type=int, default=72001)
    parser.add_argument("--seed", type=int, default=910001)
    args = parser.parse_args()
    assert_idle()
    from .computation_first_log_odds import ComputationFirstLogOdds
    from .data import generate_case
    from threadpoolctl import threadpool_limits
    config = dict(mode="practical", lanes=16, transitions=1, particle_batch=64,
                  history_block_size=128, max_state_bytes=1536 * 1024**2)
    preflight = ComputationFirstLogOdds.preflight(args.d, args.B, args.T, config)
    if not preflight["feasible"]:
        raise RuntimeError(preflight["budget_reasons"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = SOURCES + ["experiments/computation_first_long_worker.py",
                         "experiments/computation_first_long_campaign.py",
                         "COMPUTATION_FIRST_LONG_PLAN.md"]
    before = digest(sources)
    dump(output / "source_hashes_before.json", before)
    dump(output / "preflight.json", preflight)
    cid = f"signal_d{args.d}_B{args.B:g}_T{args.T}_s{args.data_seed}"
    data_path = ROOT / "experiments/data" / cid
    case = dict(id=cid, family="signal", d=args.d, B=args.B, T=args.T,
                data_seed=args.data_seed, B_generator=3.)
    preparation_start = perf_counter()
    with threadpool_limits(limits=1):
        meta = generate_case(data_path, case)
    data_files = [str((data_path / name).relative_to(ROOT)) for name in ("X.bin", "y.bin", "data.json")]
    data_before = digest(data_files)
    dump(output / "data_hashes_before.json", data_before)
    spec = dict(case=cid, d=args.d, B=args.B, T=args.T, rounds=args.T,
                seed=args.seed, data_seed=args.data_seed, config=config,
                data_path=str(data_path), output=str(output), seconds=args.seconds,
                memory_bytes=CAP, trajectory_columns=["raw_logit", "output_logit", "loss", "online_seconds", "state_bytes"])
    dump(output / "spec.json", spec)
    dump(output / "manifest.json", [spec])
    dump(output / "preparation.json", dict(seconds=perf_counter()-preparation_start, data=meta,
                                          streaming_input=True, input_block_rows=128,
                                          recorder_bytes_at_full_horizon=40 * args.T,
                                          memory_limit=CAP, worker_threads=1))
    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = "1"
    start, proc, job, reason, peak, peak_rss, hard = perf_counter(), None, None, None, 0, 0, False
    print(json.dumps(dict(status="starting", spec=str(output / "spec.json"),
                          estimated_state_bytes=preflight["estimated_state_bytes"])), flush=True)
    with (output / "stdout.txt").open("w") as stdout, (output / "stderr.txt").open("w") as stderr:
        try:
            proc = subprocess.Popen([sys.executable, "-m", "experiments.computation_first_long_worker",
                                     str(output / "spec.json"), "--gate"], cwd=ROOT, env=env,
                                    stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, text=True,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            job = ProcessLimit(proc._handle if os.name == "nt" else 0, CAP)
            hard = os.name == "nt" and job.handle is not None
            if not hard:
                raise RuntimeError("Windows hard memory limit was not installed")
            proc.stdin.write("GO\n")
            proc.stdin.flush()
            proc.stdin.close()
            monitor = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    info = monitor.memory_info()
                    peak = max(peak, getattr(info, "private", info.vms))
                    peak_rss = max(peak_rss, info.rss)
                except psutil.NoSuchProcess:
                    pass
                if perf_counter() - start > args.seconds + 1200.:
                    reason = "external_wall_budget"
                elif peak > CAP:
                    reason = "memory_limit"
                if reason:
                    proc.kill()
                    break
                sleep(.5)
            proc.wait()
        except Exception as exc:
            reason = "supervisor_error: " + repr(exc)
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
            if job is not None:
                job.close()
    final_path = output / "result.json"
    worker = json.loads(final_path.read_text()) if final_path.exists() else (
        json.loads((output / "progress.json").read_text()) if (output / "progress.json").exists() else {})
    after, data_after = digest(sources), digest(data_files)
    dump(output / "source_hashes_after.json", after)
    dump(output / "data_hashes_after.json", data_after)
    result = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                  status=reason or (worker.get("status") if final_path.exists() else "abnormal_exit"),
                  termination=reason, returncode=proc.returncode if proc else None,
                  process_wall_seconds=perf_counter()-start, monitored_peak_private=peak,
                  monitored_peak_rss=peak_rss, hard_process_commit_limit=CAP,
                  os_hard_limit_active=hard, worker_final_available=final_path.exists(),
                  checkpoint_is_censored=not final_path.exists() or reason is not None,
                  sources_and_data_unchanged=before == after and data_before == data_after,
                  worker=worker)
    dump(output / "supervisor.json", result)
    print(json.dumps({k: result[k] for k in ("status", "process_wall_seconds", "monitored_peak_private", "sources_and_data_unchanged")}
                     | dict(round=worker.get("round", 0))), flush=True)
    if not result["sources_and_data_unchanged"]:
        raise RuntimeError("Source or dataset identity changed")


if __name__ == "__main__":
    main()
