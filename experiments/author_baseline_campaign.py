"""Run the pinned authors baselines; reuse the completed New16/1 trajectory."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep

import psutil

from .computation_first_campaign import assert_idle, ROOT, SOURCES
from .computation_first_overnight_campaign import digest
from .worker_job import attach_worker

CAP = 2**31
ARCHIVE = ROOT / "experiments/results/computation_first_overnight_d1000_B1000_T100000"
DEFAULT_OUT = ROOT / "experiments/results/author_baselines_d1000_B1000_T100000"
AUTHOR_COMMIT = "22c6ec54cd45908c0a39cbfa931a60e55646308f"
AUTHOR_FOLDER = "research/author_code_comparison_2026-09-11"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def specifications(output, seconds):
    previous = read(ARCHIVE / "spec.json")
    selected = {**previous, "algorithm": "selected", "reused_completed_run": True}
    configs = {
        "author_aioli": {"R": 1.0, "lam": 1e-6, "T_inner": 80, "omega_solver": "lbfgs"},
        "author_sigmoid": {
            "n_particles": 12, "step_size": .01, "mala_steps_per_round": 300,
            "grad_clip": 50., "adapt_step": True, "target_accept": .57,
            "rm_gamma0": .5, "rm_gamma_decay": .25,
            "step_size_min": 1e-4, "step_size_max": 1., "mala_window": None,
        },
    }
    specs = [selected]
    for algorithm, config in configs.items():
        spec = {**previous, "algorithm": algorithm, "config": config, "seconds": seconds,
                "output": str(output / algorithm), "reused_completed_run": False,
                "author_commit": AUTHOR_COMMIT,
                "author_source": str(ROOT / AUTHOR_FOLDER / "a9a_experiment.py"),
                "stop_path": str(output / "STOP_REQUESTED"),
                "prediction_loss_scope": "unclipped score-based logistic loss; authors clipped-probability loss recorded separately"}
        specs.append(spec)
    return specs


def run_one(spec):
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "spec.json", spec)
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    start, proc, job, reason, peak, peak_rss = perf_counter(), None, None, None, 0, 0
    hard, actual_pid = False, None
    with (output / "stdout.txt").open("w") as stdout, (output / "stderr.txt").open("w") as stderr:
        try:
            proc = subprocess.Popen([sys.executable, "-m", "experiments.author_baseline_worker",
                                     str(output / "spec.json"), "--gate"], cwd=ROOT, env=env,
                                    stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, text=True,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            ready = output / "worker_ready.json"
            while actual_pid is None:
                try:
                    actual_pid = read(ready)["pid"]
                except (FileNotFoundError, json.JSONDecodeError):
                    if proc.poll() is not None or perf_counter() - start > 20.:
                        raise RuntimeError("Actual worker failed before PID handshake")
                    sleep(.05)
            job = attach_worker(actual_pid, CAP)
            hard = os.name == "nt" and job.handle is not None
            if not hard:
                raise RuntimeError("Hard memory cap was not attached")
            proc.stdin.write("GO\n")
            proc.stdin.flush()
            proc.stdin.close()
            monitor = psutil.Process(actual_pid)
            while proc.poll() is None:
                try:
                    info = monitor.memory_info()
                    peak = max(peak, getattr(info, "private", info.vms))
                    peak_rss = max(peak_rss, info.rss)
                except psutil.NoSuchProcess:
                    pass
                if perf_counter() - start > spec["seconds"] + 1200.:
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
    wall = perf_counter() - start
    final_available = (output / "result.json").exists()
    worker = read(output / "result.json") if final_available else (
        read(output / "progress.json") if (output / "progress.json").exists() else {})
    result = dict(status=reason or (worker.get("status") if final_available else "abnormal_exit"),
                  termination=reason, returncode=proc.returncode if proc else None,
                  actual_worker_pid=actual_pid, process_wall_seconds=wall,
                  monitored_peak_private=peak, monitored_peak_rss=peak_rss,
                  hard_process_commit_limit=CAP, os_hard_limit_active=hard,
                  worker_final_available=final_available, checkpoint_is_censored=not final_available or reason is not None,
                  worker=worker)
    dump(output / "supervisor.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seconds", type=float, default=43200.)
    args = parser.parse_args()
    assert_idle()
    output = args.output.resolve()
    specs = specifications(output, args.seconds)
    assert specs[0]["d"] == 1000 and specs[0]["B"] == 1000 and specs[0]["T"] == 100000
    assert read(ARCHIVE / "supervisor.json")["status"] == "completed"
    sources = SOURCES + ["experiments/author_baselines.py", "experiments/author_baseline_worker.py",
                         "experiments/author_baseline_campaign.py", "experiments/launch_author_baselines.ps1", "experiments/worker_job.py",
                         "experiments/computation_first_overnight_campaign.py", "AUTHOR_BASELINE_RUN_PLAN.md"]
    sources += [AUTHOR_FOLDER + "/" + name for name in
                ("a9a_experiment.py", "b_sweep_aioli_ew.py", "libsvm_datasets.py", "README.md", "requirements.txt",
                 "dependencies_versions.json")]
    before = digest(sources)
    data = Path(specs[0]["data_path"])
    data_files = [str((data / name).relative_to(ROOT)) for name in ("X.bin", "y.bin", "data.json")]
    data_before = digest(data_files)
    assert data_before == read(ARCHIVE / "data_hashes_after.json")
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "manifest.json", specs)
    dump(output / "source_hashes_before.json", before)
    dump(output / "data_hashes_before.json", data_before)
    dump(output / "policy.json", dict(target_rounds=100000, fixed_regret_endpoints=[1000, 10000, 50000, 100000],
                                      online_seconds_per_method=args.seconds, order=[s["algorithm"] for s in specs[1:]],
                                      reused_selected_run=str(ARCHIVE), workers_at_once=1, blas_threads=1,
                                      hard_private_commit_bytes=CAP, primary_metric="unclipped score-based loss and regret at identical prefixes",
                                      secondary_metric="authors reported probability-clipped loss and regret",
                                      author_commit=AUTHOR_COMMIT,
                                      new16_reused_not_rerun=True,
                                      prior_campaign_cancelled="experiments/results/large_regret_d1000_B1000_T100000/cancellation.json",
                                      author_aioli_is_sigmoid_authors_baseline=True,
                                      aioli_source_inconsistencies_preserved=True,
                                      sigmoid_B_cap_bypassed_to_preserve_requested_B=1000,
                                      sigmoid_B_sweep_steps_at_requested_B=300))
    start, attempts, unexpected = perf_counter(), [], None
    try:
        for spec in specs[1:]:
            if (output / "STOP_REQUESTED").exists():
                break
            dump(output / "campaign_progress.json", dict(status="running", current_algorithm=spec["algorithm"],
                                                          completed_attempts=attempts))
            supervisor = run_one(spec)
            record = dict(algorithm=spec["algorithm"], status=supervisor["status"],
                          round=supervisor["worker"].get("committed_round", 0),
                          online_seconds=supervisor["worker"].get("online_seconds"))
            attempts.append(record)
            print(json.dumps(record), flush=True)
    except Exception as exc:
        unexpected = repr(exc)
        raise
    finally:
        after, data_after = digest(sources), digest(data_files)
        dump(output / "source_hashes_after.json", after)
        dump(output / "data_hashes_after.json", data_after)
        result = dict(generated_utc=datetime.now(timezone.utc).isoformat(), attempts=attempts,
                      all_baseline_attempts_terminal=len(attempts) == 2,
                      cancelled_by_user=(output / "STOP_REQUESTED").exists(), unexpected_error=unexpected,
                      sources_and_data_unchanged=before == after and data_before == data_after,
                      total_wall_seconds=perf_counter()-start)
        dump(output / "campaign_result.json", result)
        dump(output / "campaign_progress.json", {**result, "status": "terminal"})
    if not result["sources_and_data_unchanged"]:
        raise RuntimeError("Measured source or dataset identity changed")


if __name__ == "__main__":
    main()
