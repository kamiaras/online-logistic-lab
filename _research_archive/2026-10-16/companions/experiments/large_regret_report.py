"""Offline four-way regret at identical prefixes of the fixed 100,000-round data.

Never run while experiment workers are active. The default requires terminal
supervisors for all four methods. Censored prefixes remain censored; actual
native outputs, not postprocessed baseline predictions, determine every loss.
No old output or old comparator file is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDER = ROOT / "experiments/results/large_regret_d1000_B1000_T100000"
ORDER = ("selected", "ours", "sigmoid", "aioli")
NAMES = {"selected": "Computation-first 16/1", "ours": "Original 256/1",
         "sigmoid": "Sigmoid/MALA 16/16", "aioli": "AIOLI"}
CLASSES = {"selected": "ComputationFirstLogOdds", "ours": "ProximalLogOdds",
           "sigmoid": "SigmoidMALA", "aioli": "AIOLI"}
COLUMNS = ("raw_logit", "output_logit", "loss", "online_seconds", "state_bytes")
CAP = 2 ** 31
TARGET = 100000
FIXED = (1000, 10000, 50000)


def read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Required completed artifact unavailable: " + str(path)) from exc


def write(path, value):
    def serial(item):
        if hasattr(item, "tolist"):
            return item.tolist()
        if hasattr(item, "item"):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=serial, allow_nan=False), encoding="utf-8")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            result.update(block)
    return result.hexdigest()


def no_active_experiments():
    import psutil
    active = []
    own_tree = {os.getpid(), *(parent.pid for parent in psutil.Process().parents())}
    for process in psutil.process_iter(["pid", "cmdline", "name"]):
        if process.pid in own_tree:
            continue
        try:
            words = process.info["cmdline"] or []
            if any(word.startswith("experiments.") and any(token in word for token in
                   ("worker", "runner", "campaign", "analyze", "report", "audit", "precision"))
                   for word in words):
                active.append(process.pid)
            elif not words and "python" in (process.info["name"] or "").lower():
                active.append(process.pid)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            if "python" in (process.info["name"] or "").lower():
                active.append(process.pid)
    if active:
        raise RuntimeError("Finish experiment processes before offline analysis: " + str(active))


class Audit:
    def __init__(self):
        self.checks, self.warnings, self.hash_cache = Counter(), [], {}

    def require(self, condition, name, detail=""):
        self.checks[name] += 1
        if not condition:
            raise RuntimeError(name + (": " + str(detail) if detail else ""))

    def file_hash(self, path):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.hash_cache:
            self.hash_cache[key] = digest(path)
        return self.hash_cache[key]

    def frozen(self, folder, domain):
        before = read(folder / (domain + "_hashes_before.json"))
        after = read(folder / (domain + "_hashes_after.json"))
        self.require(isinstance(before, dict) and bool(before) and before == after,
                     domain + "_before_after_identical", folder)
        normalized = {}
        for name, expected in after.items():
            path = (ROOT / name).resolve()
            self.require(self.file_hash(path) == expected, domain + "_current_hash", path)
            normalized[str(path)] = expected
        return normalized


def close(a, b, absolute=1e-7):
    return math.isclose(float(a), float(b), rel_tol=2e-12, abs_tol=absolute)


class Run:
    def __init__(self, spec, audit, allow_partial):
        import numpy as np
        self.spec, self.algorithm, self.output = spec, spec["algorithm"], Path(spec["output"])
        self.n, self.status, self.worker, self.supervisor = 0, "not_started", {}, {}
        self.trace, self.prefixes, self.memory = np.empty((0, 5)), {0: 0.}, None
        self.raw_bytes = 0
        terminal = self.output / "supervisor.json"
        if not terminal.exists():
            audit.require(allow_partial, "missing_terminal_supervisor", self.algorithm)
            audit.warnings.append("No terminal supervisor for " + self.algorithm + "; all its prefixes are unavailable")
            return
        raw_spec = read(self.output / "spec.json")
        if self.algorithm == "selected" and spec.get("reused_completed_run"):
            expected = {**raw_spec, "algorithm": "selected", "reused_completed_run": True}
            audit.require(expected == spec, "archived_manifest_spec", self.algorithm)
        else:
            audit.require(raw_spec == spec, "raw_manifest_spec", self.algorithm)
        self.supervisor = read(terminal)
        sup = self.supervisor
        self.status = sup.get("status", "unknown")
        audit.require(self.status not in (None, "running", "pending", "unknown"),
                      "terminal_supervisor_status", self.algorithm)
        audit.require(sup.get("returncode") is not None or sup.get("termination") is not None,
                      "worker_termination_established", self.algorithm)
        final = self.output / "result.json"
        self.worker = read(final) if final.exists() else dict(sup.get("worker") or {})
        w = self.worker
        self.n = int(w.get("committed_round", w.get("round", 0)))
        audit.require(0 <= self.n <= TARGET, "valid_committed_round", self.algorithm)
        audit.require(sup.get("os_hard_limit_active") and sup.get("hard_process_commit_limit") == CAP
                      and spec.get("memory_bytes") == CAP, "verified_supervisor_cap", self.algorithm)
        verification = self.output / "job_verification.json"
        job = read(verification) if verification.exists() else w.get("job_verification")
        audit.require(bool(job) and job.get("verified_inside_actual_interpreter")
                      and job.get("job_member") and job.get("process_memory_limit") == CAP
                      and bool(job.get("limit_flags", 0) & 0x100), "actual_worker_job_cap", self.algorithm)
        actual_pid = sup.get("actual_worker_pid")
        if actual_pid is not None:
            audit.require(job.get("pid") == actual_pid, "actual_worker_pid", self.algorithm)
        if w:
            audit.require(w.get("algorithm_class") == CLASSES[self.algorithm], "algorithm_class", self.algorithm)
            audit.require(w.get("d") == spec["d"] and w.get("B") == spec["B"]
                          and w.get("configured_horizon") == spec["T"] and w.get("seed") == spec["seed"]
                          and w.get("config") == spec["config"], "worker_parameters", self.algorithm)
            pools = [pool for pool in w.get("threadpools", []) if pool.get("user_api") == "blas"]
            audit.require(bool(pools) and all(pool.get("num_threads") == 1 for pool in pools),
                          "one_blas_thread", self.algorithm)
        trace_path = self.output / "trajectory.bin"
        self.raw_bytes = trace_path.stat().st_size if trace_path.exists() else 0
        audit.require(self.raw_bytes >= 40 * self.n, "committed_bytes_available", self.algorithm)
        if self.n:
            meta = w.get("trajectory", {})
            audit.require(meta.get("columns") == list(COLUMNS) and meta.get("record_bytes") == 40
                          and meta.get("committed_bytes") == self.n * 40, "trajectory_schema", self.algorithm)
            self.trace = np.memmap(trace_path, dtype="<f8", mode="r", shape=(self.raw_bytes // 40, 5))[:self.n]
        censored = bool(sup.get("checkpoint_is_censored") or sup.get("termination"))
        if self.status == "completed":
            audit.require(self.n == TARGET and w.get("status") == "completed" and not censored
                          and sup.get("returncode") == 0 and self.raw_bytes == TARGET * 40,
                          "complete_horizon_is_complete", self.algorithm)
        elif self.n < TARGET:
            audit.warnings.append(f"{self.algorithm}: {self.status}; only {self.n} committed rounds, not {TARGET}")
        if self.raw_bytes != self.n * 40:
            audit.warnings.append(f"{self.algorithm}: ignoring {self.raw_bytes-self.n*40} uncommitted trailing bytes")
        values = [sup.get("monitored_peak_private", 0), w.get("memory", {}).get("peak_private_commit", 0)]
        self.memory = max(values) if any(values) else None

    def validate_trace(self, y, audit):
        import numpy as np
        last_time = 0.
        pieces = []
        for lo in range(0, self.n, 8192):
            hi = min(self.n, lo + 8192)
            block = self.trace[lo:hi]
            audit.require(np.all(np.isfinite(block)), "finite_trace", self.algorithm)
            audit.require(np.all(block[:, 2] >= 0) and np.all(block[:, 4] >= 0)
                          and np.all(block[:, 4] == np.floor(block[:, 4])), "nonnegative_loss_integer_state", self.algorithm)
            audit.require(block[0, 3] >= last_time and np.all(np.diff(block[:, 3]) >= 0),
                          "monotone_online_time", self.algorithm)
            last_time = float(block[-1, 3])
            expected_loss = np.logaddexp(0., -y[lo:hi] * block[:, 1])
            audit.require(np.allclose(expected_loss, block[:, 2], rtol=1e-13, atol=1e-13),
                          "actual_output_loss", self.algorithm)
            if self.algorithm == "selected":
                t = np.arange(lo + 1, hi + 1, dtype=np.float64)
                guard = np.log(4.) + np.log(t) + np.log(t + 1)
                audit.require(np.allclose(block[:, 1], np.clip(block[:, 0], -guard, guard), rtol=1e-13, atol=1e-13),
                              "selected_native_guard", self.algorithm)
            else:
                audit.require(np.array_equal(block[:, 0], block[:, 1]), "baseline_native_output_unchanged", self.algorithm)
            pieces.append(float(np.sum(block[:, 2])))
        self.prefixes[self.n] = math.fsum(pieces)
        if self.worker:
            audit.require(close(self.prefixes[self.n], self.worker.get("cumulative_loss", 0.)),
                          "committed_loss_total", self.algorithm)
            audit.require(close(last_time, self.worker.get("completed_prefix_online_seconds", 0.)),
                          "committed_prefix_time", self.algorithm)
        if self.n == TARGET and self.status == "completed":
            audit.require(close(last_time, self.worker["online_seconds"]), "full_online_time", self.algorithm)

    def loss(self, n):
        import numpy as np
        if n not in self.prefixes:
            self.prefixes[n] = math.fsum(float(np.sum(self.trace[lo:min(n, lo+8192), 2]))
                                        for lo in range(0, n, 8192))
        return self.prefixes[n]

    def row(self, n, fit):
        base = dict(algorithm=self.algorithm, name=NAMES[self.algorithm], round=n,
                    attained_round=self.n, status=self.status, available=self.n >= n and n > 0,
                    attempt_peak_private_MiB=None if self.memory is None else self.memory / 2**20)
        if not base["available"]:
            return {**base, "reason": "Requested prefix was not completed; no extrapolation"}
        value = self.loss(n)
        gap = min(fit["gap"], max(0., fit["loss"]))
        return {**base, "cumulative_loss": value, "regret_lower": value-fit["loss"],
                "regret_upper": value-fit["loss"]+gap, "comparator_loss": fit["loss"],
                "comparator_gap": gap, "online_seconds": float(self.trace[n-1, 3]),
                "algorithm_state_bytes": int(self.trace[n-1, 4])}

    def public(self):
        return dict(algorithm=self.algorithm, spec=self.spec, status=self.status, attained_round=self.n,
                    full_horizon_completed=self.n == TARGET, output=str(self.output),
                    committed_bytes=self.n*40, raw_file_bytes=self.raw_bytes,
                    attempt_peak_private_bytes=self.memory,
                    attempt_online_seconds=self.worker.get("online_seconds"),
                    process_wall_seconds=self.supervisor.get("process_wall_seconds"),
                    error=self.worker.get("error"), failure_phase=self.worker.get("failure_phase"),
                    termination=self.supervisor.get("termination"),
                    metrics=self.worker.get("metrics", {}), diagnostics=self.worker.get("diagnostics", {}))


def validate_fit(X, y, B, fit, n, audit):
    import numpy as np
    from scipy.special import expit
    theta = np.asarray(fit["theta"], dtype=np.float64)
    audit.require(theta.shape == (X.shape[1],) and np.all(np.isfinite(theta)), "comparator_vector", n)
    radius = float(np.linalg.norm(theta))
    audit.require(radius <= B, "comparator_feasibility", n)
    value, gradient = 0., np.zeros_like(theta)
    batch_size = max(1, min(8192, 1048576 // X.shape[1]))
    for lo in range(0, n, batch_size):
        hi = min(n, lo+batch_size)
        block, labels = X[lo:hi], y[lo:hi]
        score = block @ theta
        value += float(np.sum(np.logaddexp(0., -labels*score)))
        gradient += block.T @ (-labels * expit(-labels*score))
    gap = max(0., float(theta @ gradient + B*np.linalg.norm(gradient)))
    audit.require(math.isfinite(fit["gap"]) and fit["gap"] >= 0 and math.isfinite(value),
                  "finite_nonnegative_comparator_gap", n)
    audit.require(close(value, fit["loss"], absolute=1e-6), "comparator_loss_recalculated", n)
    audit.require(abs(gap-fit["gap"]) <= max(1e-7, 1e-7*max(gap, fit["gap"])),
                  "comparator_gap_recalculated", n)
    return dict(theta_norm=radius, B=B, independent_loss=value, stored_gap=fit["gap"],
                independent_gap=gap, finite_precision_certificate=False,
                scope="Independent float64 feasibility, loss and first-order-gap check; no rigorous rounding enclosure")


def get_fit(n, folder, archived, fingerprint, X, y, audit):
    from .comparator import fit_comparator
    key = {**fingerprint, "prefix_rounds": n, "objective": "unregularized logistic loss on Euclidean B-ball"}
    target = folder / "fits" / f"prefix_{n}.json"
    if target.exists():
        envelope = read(target)
        audit.require(envelope.get("fingerprint") == key, "cached_comparator_fingerprint", n)
        fit = envelope["fit"]
        validate_fit(X, y, fingerprint["B"], fit, n, audit)
        return envelope
    if n == TARGET:
        source_path = archived / "comparator.json"
        fit = read(source_path)
        source = dict(kind="reused_completed_selected_comparator", path=str(source_path),
                      sha256=audit.file_hash(source_path),
                      reuse_basis="Matching archived/current data SHA256 and d,B,T, followed by independent loss/gap evaluation")
    else:
        fit = fit_comparator(X[:n], y[:n], fingerprint["B"], tolerance=.01, max_seconds=60.)
        source = dict(kind="new_offline_fit", one_blas_thread=True, max_seconds=60.,
                      deadline_is_soft=True, tolerance=.01)
    validation = validate_fit(X, y, fingerprint["B"], fit, n, audit)
    envelope = dict(schema_version=1, fingerprint=key, fit=fit, validation=validation, source=source)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Only the new comparison directory receives a fit. Existing fit files are
    # reused above and archived comparators are never overwritten.
    with target.open("x", encoding="utf-8") as handle:
        json.dump(envelope, handle, indent=2,
                  default=lambda value: value.tolist() if hasattr(value, "tolist") else value.item(),
                  allow_nan=False)
    return envelope


def fmt(value, digits=3):
    return "—" if value is None else f"{value:,.{digits}f}"


def table(rows):
    lines = ["| Algorithm | Regret interval | Online seconds at this prefix | Algorithm state MiB | Whole-attempt peak MiB |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        interval = (f"[{fmt(row['regret_lower'])}, {fmt(row['regret_upper'])}]" if row["available"] else "unavailable")
        state = row.get("algorithm_state_bytes")
        lines.append(f"| {row['name']} | {interval} | {fmt(row.get('online_seconds'))} | {fmt(None if state is None else state/2**20)} | {fmt(row['attempt_peak_private_MiB'])} |")
    return lines


def report_text(analysis, fits):
    primary = analysis["primary"]
    full = [row for row in primary if row["available"]]
    if len(full) == 4:
        best = min(full, key=lambda row: row["cumulative_loss"])
        lead = f"**{best['name']} has the lowest recorded regret at 100,000 rounds** on this fixed dataset."
    else:
        missing = ", ".join(row["name"] for row in primary if not row["available"])
        lead = "**The full 100,000-round comparison is incomplete.** These methods did not complete that prefix: " + missing + "."
    lines = ["# Four-way large-horizon regret comparison", "", lead, "",
             "All methods use the same fixed ordinary-signal data: d=1,000, B=1,000, T=100,000, generator radius 3, data seed 72001 and sampler seed 910001. The computation-first run is the completed archived trajectory; the three baseline attempts were subsequently measured. One dataset and one sampler trajectory do not establish an expected-regret ranking.", "",
             "## Primary endpoint: 100,000 rounds", ""]
    lines += table(primary)
    lines += ["", "Regret is actual native-prediction loss minus the same offline B-ball logistic comparator at exactly this prefix. The selected algorithm's native clipping remains included. Baseline outputs are neither clipped by this report nor replaced with another loss benchmark. Missing endpoints are unavailable and are not extrapolated.", "",
              "## Common completed prefix", ""]
    common = analysis["common"]
    if common is None:
        lines += ["No positive prefix was completed by all four methods."]
    else:
        lines += [f"All four methods completed **{common['round']:,} rounds**; the following regret intervals use their identical data prefix.", ""]
        lines += table(common["rows"])
    for point in analysis["checkpoints"]:
        if point["all_four_available"] and point["round"] != analysis["common_round"]:
            lines += ["", f"## Fixed checkpoint: {point['round']:,} rounds", ""]
            lines += table(point["rows"])
    unavailable = [point["round"] for point in analysis["checkpoints"] if not point["all_four_available"]]
    if unavailable:
        lines += ["", "Predeclared checkpoints unavailable for a complete four-way comparison: " + ", ".join(f"{n:,}" for n in unavailable) + "."]
    lines += ["", "## Completion and numerical scope", "",
              "| Algorithm | Terminal status | Committed rounds | Configured online allowance (seconds) |",
              "|---|---|---:|---:|"]
    for run in analysis["runs"]:
        lines.append(f"| {NAMES[run['algorithm']]} | {run['status']} | {run['attained_round']:,} | {fmt(run['spec'].get('seconds'), 0)} |")
    lines += ["", "The numerical comparator is unregularized logistic regression constrained by ||u||₂≤B. Each interval adds its numerical first-order optimality gap to the feasible-comparator regret lower endpoint; these are optimization intervals, not confidence intervals or rigorous floating-point enclosures. Pairwise regret differences at a common prefix equal loss differences exactly because the comparator cancels.", "",
              "The full-horizon comparator is reused only after matching data/parameter fingerprints and rechecking its loss and gradient gap. Other required prefixes use one shared offline fit, a 60-second soft fitting limit and one BLAS thread. Fitting time is excluded from online prediction time.", "",
              "| Prefix | Comparator loss | Numerical gap | Origin |", "|---:|---:|---:|---|"]
    for n, envelope in sorted(fits.items()):
        fit = envelope["fit"]
        lines.append(f"| {n:,} | {fmt(fit['loss'], 6)} | {fmt(fit['gap'], 9)} | {envelope['source']['kind']} |")
    lines += ["", "Online time counts initialization, prediction and observation. Whole-attempt peak memory includes Python, libraries, data and temporary work; it is not memory measured at a shortened comparison prefix. Algorithm state bytes in each table do refer to its displayed prefix. Interrupted timing/diagnostics can include an unfinished next round, but regret uses only flushed committed records.", "",
              "Each actual worker verifies its membership in a Windows Job Object with the 2 GiB process-commit limit, and all measured BLAS pools use one thread. The selected implementation uses 16 paired lanes/one transition; original uses 256 particles/one transition; sigmoid/MALA uses 16 particles/16 steps. The original implementation's advisory state guard is raised to 1.5 GiB to permit this retained history; its algorithmic budgets and the actual 2 GiB process limit are unchanged.", "",
              "These are practical fixed-budget float64 implementations. The prescribed exact-real regret theorem does not certify the capped sampler runs. The reused selected timing and new baseline timings occurred at different times, so machine variability also affects runtime comparisons.", "",
              "Artifacts: [analysis.json](analysis.json), [audit.json](audit.json), [manifest.json](manifest.json), and the new [fits](fits) directory. Original trajectories, sources and the archived full-horizon comparator were not overwritten."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--allow-partial", action="store_true",
                        help="After all processes stop, retain unstarted methods as unavailable instead of requiring every supervisor")
    args = parser.parse_args()
    start = perf_counter()
    no_active_experiments()
    folder, audit = args.folder.resolve(), Audit()
    # This directory must differ from the archived selected-run directory.
    specs = read(folder / "manifest.json")
    audit.require(len(specs) == 4 and {spec.get("algorithm") for spec in specs} == set(ORDER), "four_unique_methods")
    specs = sorted(specs, key=lambda spec: ORDER.index(spec["algorithm"]))
    audit.require(all(spec["d"] == 1000 and spec["B"] == 1000 and spec["T"] == TARGET for spec in specs),
                  "declared_large_setting")
    audit.require(len({(str(Path(spec["data_path"]).resolve()), spec["seed"]) for spec in specs}) == 1,
                  "same_data_and_sampler_seed")
    archived = Path(specs[0]["output"]).resolve()
    audit.require(folder != archived and specs[0].get("reused_completed_run") is True, "separate_archived_output")
    # Before hashing data or importing numerical code, reject missing terminal
    # supervisors by default. The process guard above also applies in partial mode.
    if not args.allow_partial:
        audit.require(all((Path(spec["output"])/"supervisor.json").exists() for spec in specs),
                      "all_four_terminal_supervisors_required")
    frozen_sources = audit.frozen(folder, "source")
    data_hashes = audit.frozen(folder, "data")
    archived_sources = audit.frozen(archived, "source")
    archived_data = audit.frozen(archived, "data")
    audit.require(data_hashes == archived_data, "archived_and_new_data_identical")
    data = Path(specs[0]["data_path"]).resolve()
    expected_data_paths = {str(data/name) for name in ("X.bin", "y.bin", "data.json")}
    audit.require(set(data_hashes) == expected_data_paths, "complete_data_fingerprint")
    metadata = read(data / "data.json")["case"]
    audit.require(metadata.get("d") == 1000 and metadata.get("B") == 1000
                  and metadata.get("T") == TARGET and metadata.get("data_seed") == 72001
                  and metadata.get("family") == "signal" and metadata.get("B_generator") == 3,
                  "stored_data_parameters")
    audit.require((data/"X.bin").stat().st_size == TARGET*1000*8
                  and (data/"y.bin").stat().st_size == TARGET, "data_file_lengths")
    import numpy as np
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        X = np.memmap(data/"X.bin", dtype="<f8", mode="r", shape=(TARGET, 1000))
        y = np.memmap(data/"y.bin", dtype="i1", mode="r", shape=(TARGET,))
        audit.require(np.all(np.isin(y, (-1, 1))), "valid_labels")
        runs = [Run(spec, audit, args.allow_partial) for spec in specs]
        for run in runs:
            run.validate_trace(y, audit)
        common = min(run.n for run in runs)
        needed = sorted({TARGET} | ({common} if common else set()) | {n for n in FIXED if n <= common})
        fingerprint = dict(d=1000, B=1000., configured_horizon=TARGET, data_path=str(data),
                           data_sha256={Path(path).name: value for path, value in data_hashes.items()},
                           case=metadata.get("id", specs[0].get("case")))
        fits = {n: get_fit(n, folder, archived, fingerprint, X, y, audit) for n in needed}
        points = {n: [run.row(n, fits[n]["fit"]) for run in runs] for n in needed}
        checkpoints = [dict(round=n, all_four_available=n <= common, rows=points.get(n, [])) for n in FIXED]
        pairs = []
        for n, rows in points.items():
            for first, second in combinations(rows, 2):
                if first["available"] and second["available"]:
                    pairs.append(dict(round=n, first=first["algorithm"], second=second["algorithm"],
                                      regret_difference=first["cumulative_loss"]-second["cumulative_loss"],
                                      meaning="First minus second at the identical prefix; comparator cancels"))
        analysis = dict(schema_version=1, data_fingerprint=fingerprint, target_rounds=TARGET,
                        all_methods_reached_horizon=all(run.n == TARGET for run in runs), common_round=common,
                        primary=points[TARGET], common=None if not common else dict(round=common, rows=points[common]),
                        checkpoints=checkpoints, paired_regret_differences=pairs, runs=[run.public() for run in runs],
                        fit_files={str(n): str(folder/"fits"/f"prefix_{n}.json") for n in fits},
                        finite_precision_certificate=False)
    write(folder/"analysis.json", analysis)
    (folder/"REPORT.md").write_text(report_text(analysis, fits), encoding="utf-8")
    result = dict(schema_version=1, passed=True, checks=dict(audit.checks), warnings=audit.warnings,
                  generated_utc=datetime.now(timezone.utc).isoformat(), offline_seconds=perf_counter()-start,
                  all_terminal_supervisors=all(bool(run.supervisor) for run in runs),
                  all_methods_reached_horizon=analysis["all_methods_reached_horizon"],
                  common_round=common, primary_output="Column1 actual output; column2 actual loss (zero-indexed)",
                  source_hashes=frozen_sources, archived_source_hashes=archived_sources, data_hashes=data_hashes,
                  artifact_sha256={name: digest(folder/name) for name in ("manifest.json", "analysis.json", "REPORT.md")},
                  fit_sha256={str(n): digest(folder/"fits"/f"prefix_{n}.json") for n in fits},
                  finite_precision_certificate=False,
                  scope="Artifact consistency and numerical comparator checks; timeout is an explicitly unavailable full-horizon result, not an extrapolation")
    write(folder/"audit.json", result)
    print(json.dumps(dict(report=str(folder/"REPORT.md"), audit_passed=True,
                          all_methods_reached_horizon=analysis["all_methods_reached_horizon"],
                          common_round=common, statuses={run.algorithm: run.status for run in runs},
                          offline_seconds=perf_counter()-start), allow_nan=False))


if __name__ == "__main__":
    main()
