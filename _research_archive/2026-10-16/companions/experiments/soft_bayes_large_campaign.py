"""One three-hour-per-method, fully prescribed Soft-Bayes / AIOLI pair."""
import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
from time import perf_counter
import traceback

from .soft_bayes_large_supervisor import ROOT, CAP, assert_idle, dump, read, run_one, source_hashes

STUDY = ROOT / "experiments/results/soft_bayes_full_d1000_B1000_T100000"
DATA = ROOT / "experiments/data/signal_d1000_B1000_T100000_s72001"
PRIOR = ROOT / "experiments/results/author_baselines_d1000_B1000_T100000"
EXPECTED_SOFT = "6a45ab2ba3d3d3007366a8448072021ef9329b6930b972b627a5b0c4d1c933e4"
EXPECTED_AUTHOR = "2632499867210720d9b0156678994597ce3cd5c09a079b692cebf5de706ab9cc"


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def data_hashes():
    return {name: digest(DATA / name) for name in ("X.bin", "y.bin", "data.json")}


def prepare(seconds):
    assert_idle()
    if seconds != 10800.:
        raise ValueError("The user selected three hours per method")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    from .soft_bayes import SoftBayes
    if STUDY.exists():
        raise FileExistsError("Use the existing prepared manifest; do not prepare twice")
    if digest(ROOT / "experiments/soft_bayes.py") != EXPECTED_SOFT:
        raise RuntimeError("The tested soft-Bayes source changed")
    if digest(ROOT / "research/author_code_comparison_2026-09-11/a9a_experiment.py") != EXPECTED_AUTHOR:
        raise RuntimeError("The requested released AIOLI source changed")
    hashes = data_hashes()
    archived_hashes = read(PRIOR / "data_hashes_before.json")
    archived_by_name = {Path(name).name: value for name, value in archived_hashes.items()}
    if hashes != archived_by_name:
        raise RuntimeError("Existing fixed dataset no longer matches the archived author run")
    metadata = read(DATA / "data.json")
    expected_case = dict(d=1000, B=1000., T=100000, data_seed=72001, B_generator=3.)
    if any(metadata["case"].get(k) != v for k, v in expected_case.items()):
        raise RuntimeError("Dataset parameters differ")
    config = dict(mode="prescribed", R=1., max_total_moves=None,
                  max_state_bytes=1536 * 1024**2)
    preflight = SoftBayes.preflight(1000, 1000., 100000, config)
    if not preflight["feasible"]:
        raise RuntimeError(preflight["budget_reasons"])
    # Only the first feature is needed for the literal second-round count.
    import numpy as np
    x = np.fromfile(DATA / "X.bin", dtype="<f8", count=1000)
    import math
    energy = float(x @ x) / 4.
    H = preflight["prescribed"]["H"]
    n2 = math.ceil(H / math.log1p(1e-6 / (1e-6 + 2 * energy)))
    preflight["round_two"] = dict(previous_energy=energy, transitions_per_sample=n2,
                                  samples=100000, total_transitions=n2 * 100000,
                                  first_feature_nonzero=bool(np.any(x)))
    common = dict(d=1000, B=1000., T=100000, rounds=100000, seconds=seconds,
                  memory_bytes=CAP, data_seed=72001, data_path=str(DATA), data_hashes=hashes,
                  case_id="signal_d1000_B1000_T100000_s72001",
                  stop_path=str(STUDY / "STOP_REQUESTED"))
    runs = [dict(common, algorithm="soft_bayes", label="prescribed", seed=912301, config=config,
                 output=str(STUDY / "soft_bayes")),
            dict(common, algorithm="author_aioli", label="released_aioli", seed=910001,
                 config=dict(R=1., lam=1e-6, T_inner=80, omega_solver="lbfgs"),
                 author_commit="22c6ec54cd45908c0a39cbfa931a60e55646308f",
                 output=str(STUDY / "author_aioli"))]
    STUDY.mkdir(parents=True, exist_ok=False)
    dump(STUDY / "manifest.json", dict(schema_version=1, prepared_utc=utc(),
         requested_parameters=expected_case, online_seconds_per_method=seconds,
         workers_at_once=1, blas_threads=1, hard_private_commit_bytes=CAP,
         order=[r["algorithm"] for r in runs], runs=runs, preflight=preflight,
         data_path=str(DATA), data_hashes=hashes, dataset_metadata=metadata,
         protocol=str(ROOT / "SOFT_BAYES_LARGE_RUN_PLAN.md")))
    dump(STUDY / "preflight.json", preflight)
    dump(STUDY / "status.json", dict(status="prepared", prepared_utc=utc(),
         pending_methods=[r["algorithm"] for r in runs]))
    print("Prepared one fully prescribed SoftBayes run and one fresh released AIOLI run.", flush=True)


def run():
    assert_idle()
    manifest = read(STUDY / "manifest.json")
    if any((Path(s["output"]) / "spec.json").exists() for s in manifest["runs"]):
        raise FileExistsError("This one-shot campaign has already started; inspect its status")
    before = source_hashes()
    if data_hashes() != manifest["data_hashes"]:
        raise RuntimeError("Dataset changed after preparation")
    dump(STUDY / "source_hashes_before.json", before)
    dump(STUDY / "data_hashes_before.json", manifest["data_hashes"])
    start, completed = perf_counter(), []
    try:
        for index, spec in enumerate(manifest["runs"]):
            if (STUDY / "STOP_REQUESTED").exists():
                dump(STUDY / "status.json", dict(status="cancelled_by_user", completed=completed,
                     unstarted_methods=[s["algorithm"] for s in manifest["runs"][index:]], updated_utc=utc()))
                return
            if source_hashes() != before or data_hashes() != manifest["data_hashes"]:
                raise RuntimeError("Frozen input or source changed between runs")
            dump(STUDY / "status.json", dict(status="running", current_method=spec["algorithm"],
                 supervisor_pid=os.getpid(), completed=completed, updated_utc=utc(),
                 pending_methods=[s["algorithm"] for s in manifest["runs"][index+1:]]))
            print("Starting " + spec["algorithm"], flush=True)
            result = run_one(spec)
            completed.append(dict(algorithm=spec["algorithm"], status=result["status"],
                 completed_rounds=result["worker"].get("committed_round", 0),
                 process_wall_seconds=result["process_wall_seconds"]))
            if not result["sources_unchanged"] or data_hashes() != manifest["data_hashes"]:
                raise RuntimeError("Frozen source or data changed during run")
            dump(Path(spec["output"]) / "data_hashes_after.json", manifest["data_hashes"])
            print(str(completed[-1]), flush=True)
        dump(STUDY / "source_hashes_after.json", source_hashes())
        dump(STUDY / "data_hashes_after.json", data_hashes())
        dump(STUDY / "status.json", dict(status="reporting", completed=completed,
             updated_utc=utc(), elapsed_seconds=perf_counter() - start))
        from .soft_bayes_large_report import generate
        generate(STUDY)
        dump(STUDY / "status.json", dict(status="finished", completed=completed,
             updated_utc=utc(), elapsed_seconds=perf_counter() - start,
             report=str(STUDY / "REPORT.md")))
    except Exception as exc:
        (STUDY / "campaign_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        dump(STUDY / "status.json", dict(status="campaign_error", error=repr(exc),
             completed=completed, updated_utc=utc(), elapsed_seconds=perf_counter() - start))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--seconds", type=float, default=10800.)
    args = parser.parse_args()
    prepare(args.seconds) if args.prepare else run()


if __name__ == "__main__":
    main()
