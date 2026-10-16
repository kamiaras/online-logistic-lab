"""Offline report for the completed streamed overnight measurement."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .comparator import fit_comparator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/results/computation_first_overnight_d1000_B1000_T100000"


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
                              default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()), encoding="utf-8")


def main():
    spec, sup = read(OUT / "spec.json"), read(OUT / "supervisor.json")
    worker = read(OUT / "result.json")
    n, d = worker["committed_round"], spec["d"]
    assert sup["status"] == worker["status"] == "completed" and n == spec["T"] == 100000
    raw = np.fromfile(OUT / "trajectory.bin", dtype="<f8").reshape(-1, 5)
    assert raw.shape == (n, 5) and np.isfinite(raw).all()
    assert np.all(np.diff(raw[:, 3]) >= 0) and np.all(np.diff(raw[:, 4]) >= 0)
    data = Path(spec["data_path"])
    X = np.memmap(data / "X.bin", mode="r", dtype="<f8", shape=(n, d))
    y = np.memmap(data / "y.bin", mode="r", dtype="i1", shape=(n,))
    assert np.allclose(raw[:, 2], np.logaddexp(0., -y * raw[:, 1]), rtol=1e-14, atol=1e-14)
    loss = math.fsum(raw[:, 2])
    assert math.isclose(loss, worker["cumulative_loss"], rel_tol=1e-14)
    job = read(OUT / "job_verification.json")
    assert job["verified_inside_actual_interpreter"] and job["process_memory_limit"] == 2**31
    assert all(p["num_threads"] == 1 for p in worker["threadpools"] if p["user_api"] == "blas")
    checked_hashes = {}
    for kind in ("source", "data"):
        before, after = (read(OUT / f"{kind}_hashes_{suffix}.json") for suffix in ("before", "after"))
        assert before == after
        for path, expected in after.items():
            h = hashlib.sha256()
            with (ROOT / path).open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024**2), b""):
                    h.update(block)
            assert h.hexdigest() == expected, path
        checked_hashes[kind] = len(after)
    with threadpool_limits(limits=1):
        fit = fit_comparator(X, y, spec["B"], tolerance=.01, max_seconds=60.)
    save(OUT / "comparator.json", fit)
    width = min(fit["gap"], max(0., fit["loss"]))
    lower, upper = loss - fit["loss"], loss - fit["loss"] + width
    peak = max(worker["memory"]["peak_private_commit"], sup["monitored_peak_private"])
    segments = []
    for lo, hi in ((0, 500), (0, 1000), (0, 10000), (0, 50000), (0, 100000), (50000, 100000), (90000, 100000)):
        segments.append(dict(first_round=lo+1, last_round=hi, mean_loss=float(raw[lo:hi, 2].mean()),
                             online_seconds=float(raw[hi-1, 3] - (raw[lo-1, 3] if lo else 0.))))
    audit = dict(passed=True, rounds=n, trajectory_bytes=raw.nbytes,
                 checked_current_and_before_after_hashes=checked_hashes,
                 actual_interpreter_memory_cap_verified=True, worker_blas_threads=1,
                 cumulative_loss=loss, mean_loss=loss/n, constant_half_loss=n*math.log(2.),
                 regret_lower=lower, regret_upper=upper, comparator_gap=fit["gap"],
                 comparator_status=fit["status"], comparator_loss=fit["loss"],
                 peak_private_bytes=peak, state_bytes=worker["state_bytes"],
                 online_seconds=worker["online_seconds"], process_wall_seconds=sup["process_wall_seconds"],
                 numerical_regret_is_not_a_float64_error_certificate=True, segments=segments)
    save(OUT / "audit.json", audit)
    text = ["# Overnight computation-first test", "",
            "All 100,000 rounds completed for d=1000, B=1000 with practical 16 lanes/one move. The algorithm fits within the verified 2 GiB process-memory limit.", "",
            "| Measurement | Result |", "|---|---:|",
            f'| Online time | {worker["online_seconds"]:.6f} seconds ({worker["online_seconds"]/3600:.4f} hours) |',
            f'| Worker wall time | {sup["process_wall_seconds"]:.6f} seconds |',
            f'| Peak private memory | {peak} bytes ({peak/2**20:.3f} MiB) |',
            f'| Final retained algorithm arrays | {worker["state_bytes"]} bytes ({worker["state_bytes"]/2**20:.3f} MiB) |',
            f'| Cumulative output loss | {loss:.9f} |', f'| Mean output loss | {loss/n:.9f} |',
            f'| Numerical regret interval | [{lower:.6f}, {upper:.6f}] |',
            f'| Offline comparator loss | {fit["loss"]:.9f} |',
            f'| Numerical first-order comparator gap | {fit["gap"]:.9f} |', "",
            f'Predicting one half on every round has cumulative loss {n*math.log(2.):.6f}, or {math.log(2.):.6f} per round. This run therefore fits in memory but has poor predictive loss at the practical cap.', "",
            "| Rounds | Mean output loss | Online seconds |", "|---|---:|---:|"]
    text += [f'| {r["first_round"]}–{r["last_round"]} | {r["mean_loss"]:.6f} | {r["online_seconds"]:.3f} |' for r in segments]
    text += ["", "The input is one fixed ordinary-signal dataset, generator radius 3, data seed 72001 and sampler seed 910001. No large-setting baselines were rerun. The core algorithm sources and dataset hashes stayed unchanged. The recorder streams input and retains no extra full dataset or per-round Python counter history.", "",
             f'The offline comparator minimizes unregularized logistic loss on the Euclidean ball of radius 1000. Fitting used one BLAS thread, a 60-second allowance and status `{fit["status"]}`. A feasible comparator and its first-order gap give the reported numerical interval; no interval-arithmetic or floating-point error certificate is claimed.', "",
             f'Runtime was dominated by full residual tests ({worker["metrics"]["times"]["full_test"]:.3f} s) and gradient/summary rebuilds ({worker["metrics"]["times"]["gradient_refresh"]:.3f} s). These are internal timer scopes, not a separate total-time accounting. The trace records {worker["metrics"]["counters"]["row_curvature_underflows"]} row-curvature underflows and {worker["metrics"]["counters"]["roundoff_gap_clamps"]} small residual clamps, with no terminal numerical error.', "",
             "This single-seed practical 16/1 experiment does not inherit the prescribed exact-real regret theorem. See audit.json, comparator.json, independent_audit.json, supervisor.json and the unchanged binary trajectory for details."]
    (OUT / "REPORT.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "segments"}))


if __name__ == "__main__":
    main()
