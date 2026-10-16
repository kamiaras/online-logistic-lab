"""One isolated 18-run exploratory stage; no prior manifests are modified."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter, sleep
import psutil
from .memory import ProcessLimit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"experiments/results/computation_followup"
SOURCES = ["COMPUTATION_FOLLOWUP_PLAN.md", "experiments/computation_followup.py",
           "experiments/computation_followup_worker.py", "experiments/computation_followup_campaign.py",
           "experiments/curvature_persistent_log_odds.py", "experiments/batched_persistent_log_odds.py",
           "experiments/persistent_log_odds.py", "experiments/shared_tangent.py",
           "experiments/shared_tangent_primitives.py", "experiments/proximal.py",
           "experiments/common.py", "experiments/data.py", "experiments/memory.py"]
SOURCES += ["experiments/results/shared_antithetic_ablation/manifest.json"]


def hashes(paths):
    return {str(path):hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths}


def manifest():
    controls = json.loads((ROOT/"experiments/results/shared_antithetic_ablation/manifest.json").read_text())
    assert len(controls) == 6
    specs = []
    for c in controls:
        for method in ["current16", "legacy16", "original256"]:
            config = dict(mode="practical", transitions=1, particle_batch=64)
            if method == "original256":
                config["particles"] = 256
            else:
                config.update(lanes=16, history_block_size=128)
            spec = {key:c[key] for key in ["case", "d", "B", "T", "seed", "data_path"]}
            spec.update(method=method, rounds=500, seconds=90., config=config,
                        output=str(OUT/f'{method}__{c["case"]}__s{c["seed"]}'))
            specs.append(spec)
    random.Random(914405).shuffle(specs)
    return specs


def run_one(spec, campaign_deadline):
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    (out/"spec.json").write_text(json.dumps(spec, indent=2))
    env = os.environ.copy()
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        env[key] = "1"
    start, proc, job, peak, reason = perf_counter(), None, None, 0, None
    with (out/"stdout.txt").open("w") as stdout, (out/"stderr.txt").open("w") as stderr:
        try:
            proc = subprocess.Popen([sys.executable, "-m", "experiments.computation_followup_worker",
                                     str(out/"spec.json"), "--gate"], cwd=ROOT, env=env,
                                    stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, text=True,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            job = ProcessLimit(proc._handle if os.name == "nt" else 0, 2*1024**3)
            proc.stdin.write("GO\n")
            proc.stdin.flush()
            proc.stdin.close()
            monitor = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    info = monitor.memory_info()
                    peak = max(peak, getattr(info, "private", info.vms))
                except psutil.NoSuchProcess:
                    pass
                if perf_counter() >= campaign_deadline:
                    reason = "campaign_wall_budget"
                    proc.kill()
                    break
                if perf_counter()-start > 110.:
                    reason = "external_timeout"
                    proc.kill()
                    break
                sleep(.1)
            proc.wait()
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
            if job is not None:
                job.close()
    result = dict(returncode=proc.returncode, reason=reason, wall_seconds=perf_counter()-start,
                  peak_private_bytes=peak, hard_process_commit_limit=2*1024**3,
                  os_hard_limit_active=os.name == "nt")
    (out/"supervisor.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    # Refuse overlap with any other experiment worker, including older stages.
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            words = process.info["cmdline"] or []
            if process.pid != os.getpid() and any(
                    word.startswith("experiments.") and (word.endswith("_worker") or word.endswith(".worker"))
                    for word in words):
                raise RuntimeError("Another measurement worker is active")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    OUT.mkdir(parents=True, exist_ok=True)
    specs = manifest()
    assert not (OUT/"manifest.json").exists(), "Refusing to overwrite this exploratory stage"
    data_paths = sorted({str((Path(s["data_path"])/name).relative_to(ROOT))
                         for s in specs for name in ["X.bin", "y.bin"]})
    before, data_before = hashes(SOURCES), hashes(data_paths)
    (OUT/"manifest.json").write_text(json.dumps(specs, indent=2))
    (OUT/"source_hashes_before.json").write_text(json.dumps(before, indent=2))
    (OUT/"data_hashes_before.json").write_text(json.dumps(data_before, indent=2))
    start, attempted = perf_counter(), []
    deadline = start+360.
    for i, spec in enumerate(specs):
        if perf_counter() >= deadline:
            break
        supervisor = run_one(spec, deadline)
        result_path = Path(spec["output"])/"result.json"
        worker = json.loads(result_path.read_text()) if result_path.exists() else {}
        row = dict(index=i+1, total=18, method=spec["method"], case=spec["case"], seed=spec["seed"],
                   status=worker.get("status", "abnormal_exit"), round=worker.get("round", 0),
                   wall_seconds=supervisor["wall_seconds"])
        attempted.append(row)
        print(json.dumps(row), flush=True)
    after, data_after = hashes(SOURCES), hashes(data_paths)
    (OUT/"source_hashes_after.json").write_text(json.dumps(after, indent=2))
    (OUT/"data_hashes_after.json").write_text(json.dumps(data_after, indent=2))
    unchanged = before == after and data_before == data_after
    (OUT/"campaign_result.json").write_text(json.dumps(dict(
        attempted=attempted, wall_seconds=perf_counter()-start, sources_and_data_unchanged=unchanged,
        exploratory_after_initial_results=True), indent=2))
    assert unchanged, "A frozen source or dataset changed during the stage"


if __name__ == "__main__":
    main()
