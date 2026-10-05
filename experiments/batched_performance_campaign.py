"""Sequential fixed-endpoint batching and algorithmic-squeeze controls."""
import argparse
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
from .shared_ablation_campaign import ROOT, source_hashes


def run_one(spec):
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    (out/"spec.json").write_text(json.dumps(spec, indent=2))
    env = os.environ.copy()
    for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        env[k] = "1"
    start, proc, job, peak, reason = perf_counter(), None, None, 0, None
    with (out/"stdout.txt").open("w") as stdout, (out/"stderr.txt").open("w") as stderr:
        try:
            proc = subprocess.Popen([sys.executable, "-m", "experiments.batched_performance_worker",
                str(out/"spec.json"), "--gate"], cwd=ROOT, env=env, stdin=subprocess.PIPE,
                stdout=stdout, stderr=stderr, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            job = ProcessLimit(proc._handle if os.name == "nt" else 0, 2*1024**3)
            proc.stdin.write("GO\n"); proc.stdin.flush(); proc.stdin.close()
            monitor = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    info = monitor.memory_info(); peak = max(peak, getattr(info, "private", info.vms))
                except psutil.NoSuchProcess:
                    pass
                if perf_counter()-start > 110:
                    reason = "external_timeout"; proc.kill(); break
                sleep(.1)
            proc.wait()
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill(); proc.wait()
            if job is not None:
                job.close()
    result = dict(returncode=proc.returncode, reason=reason, wall_seconds=perf_counter()-start,
                  peak_private_bytes=peak, hard_process_commit_limit=2*1024**3, os_hard_limit_active=os.name == "nt")
    (out/"supervisor.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["full", "curvature"], default="full")
    args = parser.parse_args()
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            words = p.info["cmdline"] or []
            if p.pid != os.getpid() and any(w in words for w in [
                "experiments.shared_ablation_worker", "experiments.shared_ablation_continue",
                "experiments.shared_antithetic_worker", "experiments.shared_antithetic_campaign",
                "experiments.batched_performance_worker"]):
                raise RuntimeError("Another measurement process is active")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    out = ROOT/"experiments/results"/("batched_performance" if args.backend == "full" else "curvature_performance")
    out.mkdir(parents=True, exist_ok=True)
    controls = json.loads((ROOT/"experiments/results/shared_antithetic_ablation/manifest.json").read_text())
    specs = []
    allocations = [(16, 4), (64, 1), (256, 1)] if args.backend == "full" else [(64, 1)]
    for m, steps in allocations:
        group = []
        for c in controls:
            label = f'{args.backend}__{c["case"]}__m{m}n{steps}__s{c["seed"]}'
            spec = {k: c[k] for k in ["case", "d", "B", "T", "seed", "rounds", "seconds", "data_path"]}
            spec.update(backend=args.backend, output=str(out/label), sequential_control=c["output"],
                config=dict(mode="practical", lanes=m, transitions=steps, particle_batch=64, history_block_size=128))
            group.append(spec)
        random.Random(914402+m).shuffle(group)
        specs.extend(group)
    sources = ["experiments/persistent_log_odds.py", "experiments/batched_persistent_log_odds.py",
               "experiments/batched_performance_worker.py", "experiments/batched_performance_campaign.py",
               "experiments/common.py", "BATCHED_PERFORMANCE_PLAN.md"]
    if args.backend == "curvature":
        sources += ["experiments/curvature_persistent_log_odds.py", "CURVATURE_PERFORMANCE_PLAN.md"]
    def hashes():
        return source_hashes() | {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    before = hashes()
    mp = out/"manifest.json"
    assert not mp.exists(), "Refusing to overwrite fixed campaign"
    mp.write_text(json.dumps(specs, indent=2))
    (out/"source_hashes_before.json").write_text(json.dumps(before, indent=2))
    start, attempted = perf_counter(), []
    for i, spec in enumerate(specs):
        if perf_counter()-start >= (720 if args.backend == "full" else 360):
            break
        sup = run_one(spec)
        rp = Path(spec["output"])/"result.json"
        r = json.loads(rp.read_text()) if rp.exists() else {}
        row = dict(index=i+1, total=len(specs), name=Path(spec["output"]).name,
                   status=r.get("status", "abnormal_exit"), round=r.get("round", 0), seconds=round(sup["wall_seconds"], 3))
        attempted.append(row); print(json.dumps(row), flush=True)
    after = hashes()
    assert before == after
    (out/"source_hashes_after.json").write_text(json.dumps(after, indent=2))
    (out/"campaign_result.json").write_text(json.dumps(dict(wall_seconds=perf_counter()-start, attempted=attempted), indent=2))


if __name__ == "__main__":
    main()
