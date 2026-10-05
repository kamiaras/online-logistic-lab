"""Six fixed initialization-only controls, run after the primary continuation."""
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
from .shared_ablation_campaign import ROOT, OUT as PRIMARY, source_hashes

OUT = ROOT / "experiments/results/shared_antithetic_ablation"
SOURCE = ["experiments/shared_ablation.py", "experiments/shared_ablation_worker.py",
          "experiments/shared_ablation_campaign.py", "experiments/shared_antithetic_ablation.py",
          "experiments/shared_antithetic_worker.py", "experiments/shared_antithetic_campaign.py",
          "SHARED_ANTITHETIC_ABLATION_PLAN.md", "experiments/common.py"]


def active_measurements():
    found = []
    names = ("experiments.shared_ablation_campaign", "experiments.shared_ablation_continue",
             "experiments.shared_ablation_worker", "experiments.shared_antithetic_worker")
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            if p.pid != os.getpid() and any(n in " ".join(p.info["cmdline"] or []) for n in names):
                found.append(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return found


def run_one(spec):
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(spec, indent=2))
    env = os.environ.copy()
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        env[key] = "1"
    start, job, proc, peak, reason = perf_counter(), None, None, 0, None
    with (out / "stdout.txt").open("w") as stdout, (out / "stderr.txt").open("w") as stderr:
        try:
            proc = subprocess.Popen([sys.executable, "-m", "experiments.shared_antithetic_worker",
                str(out / "spec.json"), "--gate"], cwd=ROOT, env=env, stdin=subprocess.PIPE,
                stdout=stdout, stderr=stderr, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            job = ProcessLimit(proc._handle if os.name == "nt" else 0, 2 * 1024**3)
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
                if perf_counter() - start > 110:
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
    (out / "supervisor.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    assert not active_measurements(), "Other measurement workers or supervisors are active"
    originals = json.loads((PRIMARY / "manifest.json").read_text())
    selected = [s for s in originals if s["stage"] == "primary" and s["seed"] <= 910003
                and s["config"]["layout"] == "lanes" and s["config"]["persistence"] == "warm"]
    assert len(selected) == 6
    specs = []
    for original in selected:
        spec = json.loads(json.dumps(original))
        spec["control_output"] = original["output"]
        spec["stage"] = "antithetic"
        spec["output"] = str(OUT / Path(original["output"]).name.replace("primary__", "antithetic__", 1))
        spec["config"]["prior_pairing"] = "opposite_midpoints"
        specs.append(spec)
    random.Random(914401).shuffle(specs)
    OUT.mkdir(parents=True, exist_ok=True)
    before = source_hashes() | {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE}
    original_hashes = json.loads((PRIMARY / "source_hashes_before.json").read_text())
    assert all(before[p] == h for p, h in original_hashes.items())
    mp = OUT / "manifest.json"
    assert not mp.exists(), "Do not overwrite this fixed extension"
    mp.write_text(json.dumps(specs, indent=2))
    (OUT / "source_hashes_before.json").write_text(json.dumps(before, indent=2))
    start, attempted = perf_counter(), []
    for i, spec in enumerate(specs):
        if perf_counter() - start >= 480:
            break
        sup = run_one(spec)
        rp = Path(spec["output"]) / "result.json"
        result = json.loads(rp.read_text()) if rp.exists() else {}
        row = dict(index=i+1, name=Path(spec["output"]).name,
                   status=result.get("status", "abnormal_exit"), round=result.get("round", 0),
                   seconds=round(sup["wall_seconds"], 3))
        attempted.append(row)
        print(json.dumps(row), flush=True)
    after = source_hashes() | {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE}
    assert before == after
    (OUT / "source_hashes_after.json").write_text(json.dumps(after, indent=2))
    (OUT / "campaign_result.json").write_text(json.dumps(dict(
        wall_seconds=perf_counter()-start, attempted=attempted), indent=2))


if __name__ == "__main__":
    main()
