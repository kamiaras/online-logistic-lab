"""Fresh isolated 24-run comparison of the named selected class and controls.

Example: python -m experiments.computation_first_campaign --output experiments/results/computation_first_comparison
Existing campaign directories are refused; no old source or measurement changes.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter, sleep

import psutil

from .memory import ProcessLimit


ROOT = Path(__file__).resolve().parents[1]
CAP = 2*1024**3
SOURCE_MANIFEST = ROOT/"experiments/results/computation_followup/manifest.json"
HISTORICAL_SCREEN = ROOT/"experiments/results/screen/manifest.json"
ORDER_SEED = 914906
SOURCES = ["COMPUTATION_FIRST_RUN_PLAN.md", "COMPUTATION_FIRST_LOG_ODDS_COMPLETE.md",
           "experiments/computation_first_worker.py", "experiments/computation_first_campaign.py",
           "experiments/computation_first_log_odds.py", "experiments/curvature_persistent_log_odds.py",
           "experiments/batched_persistent_log_odds.py", "experiments/persistent_log_odds.py",
           "experiments/shared_tangent.py", "experiments/shared_tangent_primitives.py",
           "experiments/proximal.py", "experiments/aioli.py", "experiments/sigmoid.py",
           "experiments/common.py", "experiments/data.py", "experiments/memory.py",
           "experiments/results/computation_followup/manifest.json", "experiments/results/screen/manifest.json",
           "experiments/results/calibration/selected_configs.json"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def hashes(paths):
    return {str(path): hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths}


def assert_idle():
    active = []
    # Windows venv python.exe can be a waiting launcher of this interpreter;
    # its identical -m command line is not a second experiment campaign.
    own_tree = {os.getpid(), *(parent.pid for parent in psutil.Process().parents())}
    for process in psutil.process_iter(["pid", "cmdline", "name"]):
        if process.pid in own_tree:
            continue
        try:
            words = process.info["cmdline"] or []
            if any(word.startswith("experiments.") and any(token in word for token in
                    ("worker", "runner", "campaign", "analyze", "report", "audit", "precision")) for word in words):
                active.append(process.pid)
            elif not words and "python" in (process.info["name"] or "").lower():
                active.append(process.pid)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            if "python" in (process.info["name"] or "").lower():
                active.append(process.pid)
    if active:
        raise RuntimeError("Other experiment processes must finish first: "+str(active))


def manifest(output, seconds=90., rounds=500):
    source, old = read(SOURCE_MANIFEST), read(HISTORICAL_SCREEN)
    cases = {}
    for spec in source:
        cases.setdefault(spec["case"], spec)
    if len(cases) != 2:
        raise ValueError("Expected the two frozen exploratory datasets")
    choices = read(ROOT/"experiments/results/calibration/selected_configs.json")
    if choices["sigmoid"]["config"] != dict(particles=16, transitions=16):
        raise ValueError("The historical sigmoid selection changed")
    specs = []
    for cid, case in sorted(cases.items()):
        if case["T"] != 1000 or not 1 <= rounds <= case["T"]:
            raise ValueError("Frozen configured horizon is 1000 and endpoint must lie within it")
        aioli = [spec for spec in old if spec["case_id"] == cid and spec["algorithm"] == "aioli"]
        sigmoid = [spec for spec in old if spec["case_id"] == cid and spec["algorithm"] == "sigmoid"]
        if len(aioli) != 1 or len(sigmoid) != 1 or sigmoid[0]["config"] != choices["sigmoid"]["config"]:
            raise ValueError("Cannot recover unique exact historical baseline settings for "+cid)
        configs = dict(selected=dict(mode="practical", lanes=16, transitions=1, history_block_size=128, particle_batch=64),
                       ours=dict(mode="practical", particles=256, transitions=1, particle_batch=64),
                       sigmoid=dict(sigmoid[0]["config"]), aioli=dict(aioli[0]["config"]))
        meta = read(Path(case["data_path"])/"data.json")["case"]
        for seed in (910001, 910002, 910003):
            for method, config in configs.items():
                spec = {key: case[key] for key in ("case", "d", "B", "T", "data_path")}
                spec.update(method=method, config=config, seed=seed, data_seed=meta["data_seed"],
                            rounds=rounds, seconds=seconds, memory_bytes=CAP,
                            output=str(Path(output).resolve()/f"{method}__{cid}__s{seed}"))
                specs.append(spec)
    random.Random(ORDER_SEED).shuffle(specs)
    return specs


def run_one(spec, deadline):
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=False)
    dump(output/"spec.json", spec)
    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = "1"
    start, proc, job, reason, peak_private, peak_rss, hard_active = perf_counter(), None, None, None, 0, 0, False
    with (output/"stdout.txt").open("w") as stdout, (output/"stderr.txt").open("w") as stderr:
        try:
            proc = subprocess.Popen([sys.executable, "-m", "experiments.computation_first_worker",
                                     str(output/"spec.json"), "--gate"], cwd=ROOT, env=env,
                                    stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, text=True,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            job = ProcessLimit(proc._handle if os.name == "nt" else 0, CAP)
            hard_active = os.name == "nt" and job.handle is not None
            if not hard_active:
                raise RuntimeError("This campaign requires its Windows hard process commit limit")
            proc.stdin.write("GO\n")
            proc.stdin.flush()
            proc.stdin.close()
            monitor = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    info = monitor.memory_info()
                    peak_private = max(peak_private, getattr(info, "private", info.vms))
                    peak_rss = max(peak_rss, info.rss)
                except psutil.NoSuchProcess:
                    pass
                if perf_counter() >= deadline:
                    reason = "campaign_wall_budget"
                elif perf_counter()-start > spec["seconds"]+20.:
                    reason = "external_timeout"
                elif peak_private > CAP:
                    reason = "memory_limit"
                if reason:
                    proc.kill()
                    break
                sleep(.1)
            proc.wait()
        except Exception as exc:
            reason = "supervisor_error: "+repr(exc)
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
            if job is not None:
                job.close()
    has_final = (output/"result.json").exists()
    worker = read(output/"result.json") if has_final else read(output/"checkpoint.json") if (output/"checkpoint.json").exists() else {}
    status = worker.get("status") if has_final else reason or "abnormal_exit"
    if reason:
        status = reason
    result = dict(status=status, returncode=proc.returncode if proc else None, termination=reason,
                  process_wall_seconds=perf_counter()-start, monitored_peak_private=peak_private, monitored_peak_rss=peak_rss,
                  hard_process_commit_limit=CAP, os_hard_limit_active=hard_active,
                  worker_final_available=has_final, checkpoint_is_censored=not has_final or reason is not None,
                  worker=worker, completed_rounds=worker.get("round", 0),
                  checkpoint_scope="If censored, only the saved completed checkpoint is usable; missing checkpoint means zero retained rounds")
    dump(output/"supervisor.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"experiments/results/computation_first_comparison")
    parser.add_argument("--seconds", type=float, default=90.)
    parser.add_argument("--wall-budget", type=float, default=600.)
    parser.add_argument("--rounds", type=int, default=500)
    args = parser.parse_args()
    if not all(math.isfinite(v) and v > 0 for v in (args.seconds, args.wall_budget)):
        raise ValueError("Time budgets must be positive finite")
    assert_idle()
    output = args.output.resolve()
    specs = manifest(output, args.seconds, args.rounds)
    source_before = hashes(SOURCES)
    data_paths = sorted({str((Path(s["data_path"])/name).resolve().relative_to(ROOT))
                         for s in specs for name in ("X.bin", "y.bin", "data.json")})
    data_before = hashes(data_paths)
    output.mkdir(parents=True, exist_ok=False)
    dump(output/"manifest.json", specs)
    dump(output/"source_hashes_before.json", source_before)
    dump(output/"data_hashes_before.json", data_before)
    dump(output/"environment.json", dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                                         logical_cpus=os.cpu_count(), memory_total=psutil.virtual_memory().total,
                                         worker_blas_threads=1, workers_at_once=1, private_commit_cap=CAP))
    dump(output/"policy.json", dict(rounds=args.rounds, online_seconds_per_run=args.seconds,
                                    external_grace_seconds=20., wall_budget_seconds=args.wall_budget,
                                    order_seed=ORDER_SEED, expected_runs=24, sampler_seeds=[910001, 910002, 910003],
                                    paired_fixed_datasets=2, data_selection_scope="Previously explored fixed datasets; descriptive rerun, not held-out confirmation"))
    start, attempted = perf_counter(), []
    deadline = start+args.wall_budget
    unexpected = None
    try:
        for index, spec in enumerate(specs):
            if perf_counter() >= deadline:
                break
            supervisor = run_one(spec, deadline)
            row = dict(index=index+1, method=spec["method"], case=spec["case"], seed=spec["seed"],
                       output=spec["output"], status=supervisor["status"], round=supervisor["completed_rounds"],
                       process_wall_seconds=supervisor["process_wall_seconds"])
            attempted.append(row)
            print(json.dumps(row), flush=True)
    except Exception as exc:
        unexpected = repr(exc)
        raise
    finally:
        source_after, data_after = hashes(SOURCES), hashes(data_paths)
        dump(output/"source_hashes_after.json", source_after)
        dump(output/"data_hashes_after.json", data_after)
        unchanged = source_before == source_after and data_before == data_after
        started = {row["output"] for row in attempted}
        result = dict(generated_utc=datetime.now(timezone.utc).isoformat(), attempted=attempted,
                      expected_runs=len(specs), terminal_attempts=len(attempted),
                      statuses=dict(Counter(row["status"] for row in attempted)),
                      unstarted=[s["output"] for s in specs if s["output"] not in started],
                      wall_seconds=perf_counter()-start, sources_and_data_unchanged=unchanged, unexpected_error=unexpected,
                      all_runs_terminal=len(attempted) == len(specs))
        dump(output/"campaign_result.json", result)
    if not unchanged:
        raise RuntimeError("A measured source or dataset changed during the campaign")
    print(json.dumps(dict(output=str(output), attempted=len(attempted), unstarted=len(specs)-len(attempted),
                          statuses=result["statuses"], sources_and_data_unchanged=unchanged)), flush=True)


if __name__ == "__main__":
    main()
