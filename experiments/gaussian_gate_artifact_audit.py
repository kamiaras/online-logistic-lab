"""Independent, read-only validation of the completed Gaussian-gate campaign.

Only audit.json is created. This does not rerun a learner, refit a comparator,
or modify measured files. All numerical checks are float64 diagnostics rather
than interval-arithmetic or finite-precision regret certificates.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "experiments/results/gaussian_feature_gate_comparison_2026-09-11"
CAP = 2**31
EXPECTED = {"smoke": 5, "primary": 24, "stress": 2}
COLUMNS = ("raw_logit", "output_logit", "loss", "online_seconds", "state_bytes",
           "author_clipped_loss", "covariance_variance")
METHODS = {"gaussian_gate", "canonical_aioli", "author_aioli", "author_sigmoid"}
TERMINAL = {"completed", "time_budget", "cancelled_by_user", "external_wall_budget",
            "memory_error", "memory_limit", "error", "abnormal_exit", "incomplete_input"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolved(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(4 * 1024**2), b""):
            value.update(part)
    return value.hexdigest()


def close(a, b, atol=1e-8, rtol=2e-11):
    return (math.isfinite(float(a)) and math.isfinite(float(b))
            and abs(float(a) - float(b)) <= atol + rtol * max(abs(float(a)), abs(float(b))))


class Audit:
    def __init__(self):
        self.checks = Counter()
        self.hashes = {}
        self.warnings = []

    def require(self, condition, name, detail=""):
        if not condition:
            raise AssertionError(f"{name}: {detail}")
        self.checks[name] += 1

    def file_hash(self, path):
        path = resolved(path)
        if str(path) not in self.hashes:
            self.hashes[str(path)] = digest(path)
        return self.hashes[str(path)]

    def frozen(self, folder, kind):
        before = read(folder / f"{kind}_hashes_before.json")
        after = read(folder / f"{kind}_hashes_after.json")
        self.require(before == after and bool(before), f"{kind}_before_after_identical", folder)
        for path, expected in before.items():
            self.require(self.file_hash(path).lower() == expected.lower(),
                         f"{kind}_current_hash_matches_measured", path)
        return {str(resolved(path)): value for path, value in before.items()}


def _first(mapping, *names, default=None):
    return next((mapping[name] for name in names if name in mapping), default)


def validate_protocol(spec, audit):
    """Independently state the selected configuration, not imported dispatch."""
    method, stage = spec["algorithm"], spec["stage"]
    if method in {"gaussian_gate", "canonical_aioli"}:
        expected_config = {"R": 1., "matrix_backend": "inverse"}
    elif method == "author_aioli":
        expected_config = {"R": 1., "lam": float(spec["B"])**-2,
                           "T_inner": 80, "omega_solver": "lbfgs"}
    else:
        expected_config = dict(n_particles=12, step_size=.01, mala_steps_per_round=300,
                               grad_clip=50., adapt_step=True, target_accept=.57,
                               rm_gamma0=.5, rm_gamma_decay=.25, step_size_min=1e-4,
                               step_size_max=1., mala_window=None)
    audit.require(spec["config"] == expected_config, "predeclared_algorithm_configuration", spec["output"])
    audit.require(spec["seed"] == 910001 and spec["memory_bytes"] == CAP,
                  "predeclared_seed_and_memory", spec["output"])
    if stage == "primary":
        audit.require((spec["d"], spec["B"]) in {(10, 100.), (100, 1000.)}
                      and spec["T"] == 1000 and spec["rounds"] == 500
                      and spec["data_seed"] in {72001, 72002, 72003}
                      and spec["seconds"] == 90., "predeclared_primary_setting", spec["output"])
    elif stage == "stress":
        audit.require((spec["d"], spec["B"], spec["T"], spec["rounds"], spec["data_seed"])
                      == (1000, 1000., 100000, 1000, 72001)
                      and method in {"gaussian_gate", "author_aioli"}
                      and spec["seconds"] == 90., "predeclared_stress_setting", spec["output"])
    else:
        audit.require(spec["rounds"] == 8 and spec["seconds"] == 30.,
                      "predeclared_smoke_setting", spec["output"])


def load_run(spec, audit, np):
    output = resolved(spec["output"])
    method, case = spec["algorithm"], spec["case"]
    audit.require(method in METHODS, "recognized_method", method)
    validate_protocol(spec, audit)
    audit.require(read(output / "spec.json") == spec, "run_spec_matches_manifest", output)
    supervisor = read(output / "supervisor.json")
    audit.frozen(output, "source")
    result_path = output / "result.json"
    result = read(result_path) if result_path.exists() else read(output / "progress.json")
    audit.require(supervisor.get("worker") == result, "supervisor_embeds_same_worker_snapshot", output)
    audit.require(supervisor.get("sources_unchanged") is True,
                  "supervisor_reports_unchanged_source", output)
    status = supervisor.get("status", result.get("status"))
    audit.require(status in TERMINAL or str(status).startswith("supervisor_error"),
                  "terminal_supervisor_status", output)
    # A failed numerical attempt is representable, but must not pass as a
    # successful experiment completion. No endpoint is synthesized below.
    if status not in {"completed", "time_budget", "cancelled_by_user", "external_wall_budget"}:
        audit.warnings.append({"output": str(output), "status": status,
                               "meaning": "Failed attempt; no success or full endpoint inferred"})
    n = int(_first(result, "committed_round", "round", default=0))
    target, T, d = int(spec.get("rounds", spec["T"])), int(spec["T"]), int(spec["d"])
    audit.require(0 <= n <= target <= T, "valid_committed_prefix", output)
    audit.require(status != "completed" or n == target, "completed_means_target_reached", output)
    audit.require(supervisor.get("os_hard_limit_active") is True,
                  "supervisor_hard_cap_attached", output)
    audit.require(supervisor.get("hard_process_commit_limit") == CAP,
                  "supervisor_hard_cap_bytes", output)
    job = result.get("job_verification") or {}
    audit.require(job.get("job_member") is True
                  and job.get("process_memory_limit") == CAP
                  and job.get("verified_inside_actual_interpreter") is True,
                  "actual_interpreter_job_verified", output)
    audit.require(job.get("pid") == supervisor.get("actual_worker_pid"),
                  "verified_pid_is_supervised_interpreter", output)
    audit.require(supervisor.get("monitored_peak_private", 0) <= CAP,
                  "monitored_peak_within_private_cap", output)
    memory = result.get("memory", {})
    audit.require(memory.get("peak_private_commit", memory.get("private_commit", 0)) <= CAP,
                  "worker_private_peak_within_cap", output)
    pools = result.get("threadpools", [])
    audit.require(bool(pools) and all(pool.get("num_threads") == 1 for pool in pools),
                  "all_numeric_pools_single_thread", output)
    trajectory = output / "trajectory.bin"
    audit.require(trajectory.stat().st_size >= n * 8 * len(COLUMNS),
                  "committed_trajectory_bytes_present", output)
    if status == "completed":
        audit.require(trajectory.stat().st_size == n * 8 * len(COLUMNS),
                      "completed_trajectory_has_no_extra_records", output)
    metadata = result.get("trajectory", {})
    if "columns" in metadata:
        audit.require(tuple(metadata["columns"]) == COLUMNS, "trajectory_column_names", output)
    values = np.fromfile(trajectory, dtype="<f8", count=n * len(COLUMNS)).reshape(n, len(COLUMNS))
    audit.require(np.isfinite(values[:, :5]).all(), "first_five_columns_finite", output)
    audit.require((values[:, 2:5] >= 0).all(), "loss_time_state_nonnegative", output)
    audit.require((np.diff(values[:, 3]) >= -1e-12).all(), "online_time_nondecreasing", output)
    audit.require((values[:, 4] == np.floor(values[:, 4])).all()
                  and (values[:, 4] < CAP).all(), "state_bytes_integral_below_cap", output)
    for col in (5, 6):
        audit.require(not np.isinf(values[:, col]).any(), "optional_column_no_infinity", output)
    if method.startswith("author_"):
        audit.require(np.isfinite(values[:, 5]).all() and (values[:, 5] >= 0).all()
                      and (values[:, 5] < 28).all(), "author_reported_loss_present", output)
        audit.require(close(math.fsum(values[:, 5].tolist()),
                            result.get("author_reported_cumulative_loss", 0.)),
                      "author_reported_total_matches_trace", output)
    else:
        audit.require(np.isnan(values[:, 5]).all(), "non_author_reported_loss_absent", output)
    if method == "gaussian_gate":
        audit.require(np.isfinite(values[:, 6]).all() and (values[:, 6] > 0).all(),
                      "positive_gaussian_covariance", output)
    else:
        audit.require(np.isnan(values[:, 6]).all(), "non_gaussian_covariance_absent", output)
    data = resolved(spec["data_path"])
    meta = read(data / "data.json")["case"]
    audit.require(meta["d"] == d and meta["T"] == T and meta["B"] == spec["B"]
                  and meta["data_seed"] == spec["data_seed"] and meta["B_generator"] == 3.,
                  "dataset_parameters_match_spec", output)
    labels = np.fromfile(data / "y.bin", dtype="i1", count=n)
    audit.require(len(labels) == n and np.isin(labels, (-1, 1)).all(), "valid_prefix_labels", output)
    actual_loss = np.logaddexp(0., -labels * values[:, 1])
    audit.require(np.allclose(actual_loss, values[:, 2], atol=1e-10, rtol=2e-12),
                  "trace_losses_recomputed_from_output", output)
    total = math.fsum(values[:, 2].tolist())
    audit.require(close(total, result.get("cumulative_loss", 0.)), "committed_loss_total_matches", output)
    if n:
        prefix_online = _first(result, "completed_prefix_online_seconds", "online_seconds")
        audit.require(close(values[-1, 3], prefix_online, atol=1e-6),
                      "prefix_online_time_matches_trace", output)
        if status == "completed":
            audit.require(close(values[-1, 4], result["state_bytes"], atol=0, rtol=0),
                          "final_state_bytes_matches_trace", output)
            audit.require(close(result["online_seconds"],
                                result["initialization_seconds"] + result["prediction_seconds"]
                                + result["update_seconds"], atol=1e-6),
                          "online_time_equals_initialization_prediction_update", output)
    public = dict(stage=spec["stage"], case=case, algorithm=method, status=status,
                  committed_rounds=n, target_rounds=target, cumulative_loss=total,
                  mean_loss=total/n if n else None,
                  prefix_online_seconds=float(values[-1, 3]) if n else None,
                  output=str(output), trajectory_sha256=audit.file_hash(trajectory),
                  result_sha256=audit.file_hash(result_path if result_path.exists() else output / "progress.json"))
    run = dict(spec=spec, result=result, supervisor=supervisor, values=values,
               labels=labels, n=n, public=public)
    if method == "gaussian_gate" and n:
        validate_gaussian(run, audit, np)
    return run


def validate_gaussian(run, audit, np):
    spec, result, values, n = (run[key] for key in ("spec", "result", "values", "n"))
    d, T, R = int(spec["d"]), int(spec["T"]), float(spec.get("config", {}).get("R", 1.))
    data = resolved(spec["data_path"])
    X = np.memmap(data / "X.bin", mode="r", dtype="<f8", shape=(T, d))
    norm_squared = np.empty(n)
    for lo in range(0, n, 128):
        hi = min(n, lo+128)
        block = X[lo:hi]
        audit.require(np.isfinite(block).all(), "gaussian_features_finite", spec["case"])
        norm_squared[lo:hi] = np.einsum("ij,ij->i", block, block)
    precision = T * R * R if R else 1.
    expected_tau = np.empty(n)
    for t in range(n):
        expected_tau[t] = 1./precision
        precision += float(norm_squared[t])/(4.*d)
    audit.require(np.allclose(expected_tau, values[:, 6], atol=1e-15, rtol=2e-12),
                  "gaussian_covariance_recurrence", spec["case"])
    projected = values[:, 6]*norm_squared
    audit.require((projected <= 1./T + 1e-14).all(), "gaussian_projected_variance_budget", spec["case"])
    base_loss = np.logaddexp(0., -run["labels"]*values[:, 0])
    excess = values[:, 2] - base_loss
    eta = 1./(7.*T)
    quadrature_allowance = math.log1p(eta)-math.log1p(-eta)
    bound = projected/8. + quadrature_allowance
    tolerance = 1e-8 + 4e-12*np.maximum(np.abs(base_loss), np.abs(values[:, 2]))
    audit.require((excess <= bound+tolerance).all(), "observed_smoothing_plus_quadrature_loss_budget", spec["case"])
    diagnostic = result.get("diagnostics", {})
    audit.require(diagnostic.get("gaussian_count") == 2*d
                  and diagnostic.get("active_gaussians_per_nonzero_prediction") == 2,
                  "fixed_dictionary_and_active_counts", spec["case"])
    audit.require(diagnostic.get("history_rows_retained") == 0
                  and diagnostic.get("integration_error_feeds_gate") is False,
                  "gate_has_no_history_or_quadrature_feedback", spec["case"])
    audit.require(diagnostic.get("quadrature_prescribed") is True,
                  "prescribed_quadrature_reported", spec["case"])
    if run["public"]["status"] == "completed":
        audit.require(close(diagnostic["covariance_precision"], precision),
                      "terminal_covariance_precision_recomputed", spec["case"])
    run["public"]["gaussian_transfer_check"] = dict(
        max_observed_loss_excess=float(np.max(excess)),
        cumulative_observed_loss_excess=math.fsum(excess.tolist()),
        cumulative_projected_smoothing_bound=math.fsum((projected/8.).tolist()),
        per_round_quadrature_bound=quadrature_allowance,
        cumulative_including_quadrature_bound=float(np.sum(bound)),
        scope="Observed float64 consistency only; no finite-precision regret certificate")


def compare_matched_controls(runs, audit, np):
    keyed = {(r["spec"]["stage"], r["spec"]["case"], r["spec"]["algorithm"]): r for r in runs}
    pairs = []
    for (stage, case, method), gaussian in keyed.items():
        if method != "gaussian_gate":
            continue
        canonical = keyed.get((stage, case, "canonical_aioli"))
        if canonical is None:
            continue
        n = min(gaussian["n"], canonical["n"])
        if not n:
            continue
        gs, cs = gaussian["values"][:n], canonical["values"][:n]
        audit.require(np.allclose(gs[:, 0], cs[:, 1], atol=1e-9, rtol=2e-12),
                      "gaussian_base_matches_canonical_control", (stage, case))
        audit.require(np.array_equal(cs[:, 0], cs[:, 1]),
                      "canonical_control_output_equals_raw", (stage, case))
        pairs.append(dict(stage=stage, case=case, common_rounds=n,
                          maximum_base_control_logit_difference=float(np.max(np.abs(gs[:, 0]-cs[:, 1]))),
                          base_control_float64_identical=bool(np.array_equal(gs[:, 0], cs[:, 1])),
                          gaussian_minus_control_cumulative_loss=math.fsum((gs[:, 2]-cs[:, 2]).tolist())))
    audit.require(len([p for p in pairs if p["stage"] == "primary"]) == 6,
                  "all_six_primary_matched_controls_available")
    return pairs


def walk_records(value):
    """Locate explicit numerical endpoint records without assuming presentation nesting."""
    if isinstance(value, dict):
        if ("case" in value and ("algorithm" in value or "method" in value)
                and any(name in value for name in ("n", "rounds", "round", "prefix", "prefix_rounds"))
                and any(name in value for name in ("cumulative_loss", "loss", "mean_loss", "loss_per_round"))):
            yield value
        for child in value.values():
            yield from walk_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_records(child)


def validate_summary(folder, runs, audit):
    path = folder / "summary.json"
    audit.require(path.exists(), "final_summary_exists", path)
    summary = read(path)
    count = 0
    for record in walk_records(summary):
        if record.get("available") is False or record.get("reused") is True or record.get("historical") is True:
            continue
        method, case = _first(record, "algorithm", "method"), record["case"]
        raw_n = _first(record, "n", "rounds", "round", "prefix", "prefix_rounds")
        if not isinstance(raw_n, (int, float)):
            continue
        n = int(raw_n)
        candidates = [r for r in runs if r["spec"]["case"] == case
                      and r["spec"]["algorithm"] == method and r["n"] >= n
                      and ("stage" not in record or r["spec"]["stage"] == record["stage"])]
        # Smoke records can otherwise share a primary case; choose the unique
        # requested stage, or a unique feasible longest-endpoint run.
        if len(candidates) != 1 and "stage" not in record:
            candidates = [r for r in candidates if r["spec"]["stage"] != "smoke"]
        audit.require(len(candidates) == 1 and n >= 1, "summary_record_has_unique_trace", record)
        values = candidates[0]["values"][:n]
        total = math.fsum(values[:, 2].tolist())
        for key in ("cumulative_loss", "loss"):
            if key in record:
                audit.require(close(record[key], total), "summary_cumulative_loss_recomputed", (case, method, n))
        for key in ("mean_loss", "loss_per_round"):
            if key in record:
                audit.require(close(record[key], total/n), "summary_mean_loss_recomputed", (case, method, n))
        if "online_seconds" in record:
            audit.require(close(record["online_seconds"], values[-1, 3], atol=1e-6),
                          "summary_prefix_time_recomputed", (case, method, n))
        for key in ("post_initialization_seconds", "prediction_update_seconds"):
            if key in record:
                initialization = candidates[0]["result"]["initialization_seconds"]
                audit.require(close(record[key], values[-1, 3]-initialization, atol=1e-6),
                              "summary_post_initialization_time_recomputed", (case, method, n))
        lower, upper = record.get("regret_lower"), record.get("regret_upper")
        if lower is not None or upper is not None:
            audit.require(lower is not None and upper is not None and lower <= upper,
                          "summary_regret_interval_ordered", record)
            comparator = record.get("comparator_loss")
            gap = record.get("comparator_gap")
            if comparator is not None:
                audit.require(close(lower, total-comparator), "summary_regret_lower_recomputed", record)
                if gap is not None:
                    audit.require(gap >= 0 and close(upper, total-comparator+gap),
                                  "summary_regret_upper_recomputed", record)
        count += 1
    audit.require(count >= 24, "all_primary_summary_records_audited", count)
    return dict(records_checked=count, sha256=audit.file_hash(path),
                regret_scope="Shared comparator arithmetic checked when loss/gap fields are present; no comparator refit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=DEFAULT)
    args = parser.parse_args()
    # This guard runs before data hashing or numerical imports.
    from .computation_first_campaign import assert_idle
    assert_idle()
    import numpy as np
    from threadpoolctl import threadpool_limits
    folder = args.folder.resolve()
    start, audit, all_runs, stages = perf_counter(), Audit(), [], {}
    with threadpool_limits(limits=1):
        for stage, expected in EXPECTED.items():
            current = folder / stage
            specs = read(current / "manifest.json")
            stage_result = read(current / "result.json")
            audit.require(len(specs) == expected, "stage_predeclared_count", stage)
            audit.require(stage_result.get("status") == "terminal"
                          and stage_result.get("planned") == expected
                          and stage_result.get("attempted") == expected,
                          "every_planned_attempt_terminal", stage)
            audit.require(stage_result.get("error") is None, "campaign_no_unexpected_error", stage)
            source, data = audit.frozen(current, "source"), audit.frozen(current, "data")
            audit.require(len({spec["output"] for spec in specs}) == expected, "unique_output_per_attempt", stage)
            audit.require(all(spec["stage"] == stage for spec in specs), "manifest_stage_matches", stage)
            runs = [load_run(spec, audit, np) for spec in specs]
            status_counts = dict(Counter(r["public"]["status"] for r in runs))
            audit.require(status_counts == stage_result["terminal_counts"], "stage_terminal_counts_recomputed", stage)
            all_runs.extend(runs)
            stages[stage] = dict(planned=expected, terminal_counts=status_counts,
                                 source_hashes=source, data_hashes=data)
        primary = [r for r in all_runs if r["spec"]["stage"] == "primary"]
        grouped = {}
        for run in primary:
            grouped.setdefault(run["spec"]["case"], []).append(run)
        audit.require(len(grouped) == 6, "six_primary_datasets")
        common = []
        for case, runs in grouped.items():
            audit.require({r["spec"]["algorithm"] for r in runs} == METHODS and len(runs) == 4,
                          "four_methods_on_each_primary_dataset", case)
            audit.require(len({(r["spec"]["data_path"], r["spec"]["seed"], r["spec"]["T"], r["spec"]["B"])
                               for r in runs}) == 1, "shared_data_seed_and_parameters", case)
            n = min(r["n"] for r in runs)
            common.append(dict(case=case, common_rounds=n,
                               primary_500_available=n >= 500,
                               losses={r["spec"]["algorithm"]: math.fsum(r["values"][:n, 2].tolist()) for r in runs}))
        pairs = compare_matched_controls(all_runs, audit, np)
        summary = validate_summary(folder, all_runs, audit)
    record = dict(schema_version=1, passed=True, generated_utc=datetime.now(timezone.utc).isoformat(),
                  checks=dict(audit.checks), warnings=audit.warnings, stages=stages,
                  runs=[r["public"] for r in all_runs], primary_common_prefixes=common,
                  matched_gate_controls=pairs, summary=summary, elapsed_seconds=perf_counter()-start,
                  timing_metadata_erratum=("The frozen worker phrase that all module imports are excluded "
                      "is too broad: released author_module loads lazily inside construction, so that import "
                      "is included in recorded initialization/online time. Numeric timing fields and "
                      "post-initialization comparisons are authoritative; measured source is preserved."),
                  finite_precision_certificate=False,
                  scope="Independent recorded-source/data/worker/trace and endpoint consistency audit; no learner reruns or comparator fits")
    target = folder / "audit.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps(dict(passed=True, audit=str(target), checked_runs=len(all_runs),
                          checked_summary_records=summary["records_checked"])))


if __name__ == "__main__":
    main()
