"""Sequential guarded campaign; historical supervisors remain unchanged."""
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter, sleep
import hashlib
import psutil
from .memory import ProcessLimit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"experiments/results/shared_ablation"
CORE = ["experiments/shared_tangent.py", "experiments/shared_tangent_primitives.py",
        "experiments/proximal.py", "experiments/data.py", "experiments/sigmoid.py",
        "experiments/aioli.py", "INTEGRATED_LOG_ODDS_COMPLETE.md"]


def source_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in CORE}


def manifest():
    specs = []
    cases = [("signal_d10_B100_T1000_s72001", 10, 100.),
             ("signal_d100_B1000_T1000_s72001", 100, 1000.)]
    def add(case, d, B, seed, layout, persistence, acceptance="full",
            m=16, steps=4, rounds=500, stage="primary"):
        label = f"{stage}__{case}__B{B:g}__{layout}_{persistence}_{acceptance}_m{m}n{steps}__s{seed}"
        specs.append(dict(case=case, d=d, B=B, T=1000, seed=seed,
                          rounds=rounds, seconds=90., stage=stage,
                          data_path=str(ROOT/"experiments/data"/case),
                          output=str(OUT/label),
                          config=dict(mode="practical", tilts=m, burn_in=steps,
                                      transitions=steps, history_block_size=128,
                                      layout=layout, persistence=persistence,
                                      acceptance=acceptance,
                                      capture_checkpoints=[10, 50, 100, 250, 500],
                                      capture_per_round=2)))
    for case, d, B in cases:
        for seed in range(910001, 910007):
            for layout in ["sweep", "lanes"]:
                for persistence in ["fresh", "warm"]:
                    add(case, d, B, seed, layout, persistence)
        for seed in range(910001, 910004):
            add(case, d, B, seed, "sweep", "fresh", "poisson", rounds=100, stage="acceptance")
            for m in [64, 256]:
                add(case, d, B, seed, "lanes", "warm", m=m, steps=1, stage="budget")
    case, d, _ = cases[0]
    for B in [1., 10., 1000.]:
        for seed in range(910001, 910004):
            add(case, d, B, seed, "sweep", "fresh", rounds=250, stage="radius")
    random.Random(914400).shuffle(specs)
    return specs


def run_one(spec):
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    (out/"spec.json").write_text(json.dumps(spec, indent=2))
    env = os.environ.copy()
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        env[key] = "1"
    start, job, proc, peak = perf_counter(), None, None, 0
    reason = None
    with (out/"stdout.txt").open("w") as stdout, (out/"stderr.txt").open("w") as stderr:
        try:
            proc = subprocess.Popen([sys.executable, "-m", "experiments.shared_ablation_worker",
                                     str(out/"spec.json"), "--gate"], cwd=ROOT, env=env,
                                    stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                    text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
            job = ProcessLimit(proc._handle if os.name=="nt" else 0, 2*1024**3)
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
                if perf_counter()-start > 110:
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
    result = dict(returncode=proc.returncode, reason=reason,
                  wall_seconds=perf_counter()-start, peak_private_bytes=peak,
                  hard_process_commit_limit=2*1024**3, os_hard_limit_active=os.name=="nt")
    (out/"supervisor.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    specs = manifest()
    mp = OUT/"manifest.json"
    if mp.exists():
        assert json.loads(mp.read_text()) == specs
    else:
        mp.write_text(json.dumps(specs, indent=2))
        (OUT/"source_hashes_before.json").write_text(json.dumps(source_hashes(), indent=2))
    start = perf_counter()
    for i, spec in enumerate(specs):
        if perf_counter()-start >= 1200:
            print("Campaign wall budget reached", flush=True)
            break
        if (Path(spec["output"])/"supervisor.json").exists():
            continue
        result = run_one(spec)
        rp = Path(spec["output"])/"result.json"
        worker = json.loads(rp.read_text()) if rp.exists() else {}
        print(json.dumps(dict(index=i+1, total=len(specs), name=Path(spec["output"]).name,
                              status=worker.get("status", "abnormal_exit"),
                              round=worker.get("round", 0),
                              seconds=round(result["wall_seconds"], 3))), flush=True)
    after = source_hashes()
    before = json.loads((OUT/"source_hashes_before.json").read_text())
    assert before == after, "Historical core changed"
    (OUT/"source_hashes_after.json").write_text(json.dumps(after, indent=2))


if __name__ == "__main__":
    main()
