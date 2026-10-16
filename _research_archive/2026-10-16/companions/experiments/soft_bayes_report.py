"""Audit and report the completed frozen Soft Bayes / released AIOLI study.

Run only after all measured attempts finish. Every comparison uses equal data
and a common completed prefix; numerical comparator fits are shared by prefix.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "experiments/results/soft_bayes_2026-09-11"
COMMIT = "22c6ec54cd45908c0a39cbfa931a60e55646308f"
AUTHOR_SHA = "2632499867210720d9b0156678994597ce3cd5c09a079b692cebf5de706ab9cc"
COLUMNS = ["raw_logit", "output_logit", "loss", "online_seconds",
           "state_bytes", "author_clipped_loss"]
CAP = 2 * 1024**3


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def span(values):
    values = list(values)
    return [min(values), max(values)] if values else None


def number(value, digits=6):
    return "unavailable" if value is None else f"{value:.{digits}f}"


def display_span(values, scale=1., digits=0):
    if values is None:
        return "unavailable"
    low, high = values[0] / scale, values[1] / scale
    return number(low, digits) if low == high else f"{number(low, digits)}–{number(high, digits)}"


def generate(study):
    # A terminal supervisor file is required even for killed/censored attempts.
    # Inspect this prerequisite before importing any numerical libraries.
    study = Path(study).resolve()
    manifest_path = study / "manifest.json"
    manifest = read(manifest_path)
    missing = [s["id"] for s in manifest["runs"]
               if not (Path(s["output"]) / "supervisor.json").exists()]
    if missing:
        raise RuntimeError("Measurements have not finished: " + ", ".join(missing))
    from .soft_bayes_supervisor import assert_idle
    assert_idle()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    import numpy as np
    from scipy.special import expit
    from threadpoolctl import threadpool_limits
    from .comparator import fit_comparator

    started = perf_counter()
    checks, records, dataset_arrays = [], {}, {}
    audit_path = ROOT / "SOFT_BAYES_ARTIFACT_AUDIT.json"

    def check(name, passed, detail=None):
        item = {"check": name, "passed": bool(passed)}
        if detail is not None:
            item["detail"] = detail
        checks.append(item)
        return bool(passed)

    def audit_document():
        return dict(schema_version=1, manifest=str(manifest_path),
                    manifest_sha256=digest(manifest_path),
                    report_source_sha256=digest(Path(__file__)),
                    passed=all(c["passed"] for c in checks),
                    checked_attempts=len(records), checks=checks,
                    statuses=dict(Counter(r["status"] for r in records.values())),
                    numerical_comparator_certificate="float64 first-order gap; not interval arithmetic",
                    scope="Committed data/source/recording/resource audit, not a proof of sampler mixing or finite-bit regret",
                    elapsed_seconds=perf_counter() - started)

    def require_checks():
        dump(audit_path, audit_document())
        failures = [c["check"] for c in checks if not c["passed"]]
        if failures:
            raise RuntimeError("Artifact audit failed: " + "; ".join(failures))

    check("manifest_attempt_count", len(manifest["runs"]) == manifest["expected_attempts"] == 51)
    check("manifest_unique_ids", len({s["id"] for s in manifest["runs"]}) == len(manifest["runs"]))
    author_path = ROOT / "research/author_code_comparison_2026-09-11/a9a_experiment.py"
    check("released_author_source_hash", digest(author_path) == AUTHOR_SHA)
    source_versions = defaultdict(set)
    dataset_checks = {}
    for case_id, dataset in manifest["datasets"].items():
        folder = Path(dataset["path"])
        hashes = {name: digest(folder / name) for name in ("X.bin", "y.bin", "data.json")}
        check(f"dataset_hashes:{case_id}", hashes == dataset["hashes"])
        metadata = read(folder / "data.json")
        check(f"dataset_metadata:{case_id}", metadata == dataset["metadata"])
        case = metadata["case"]
        d, T = int(case["d"]), int(case["T"])
        check(f"dataset_file_sizes:{case_id}", (folder / "X.bin").stat().st_size == 8*d*T
              and (folder / "y.bin").stat().st_size == T)
        X = np.fromfile(folder / "X.bin", dtype="<f8").reshape(T, d)
        y = np.fromfile(folder / "y.bin", dtype="i1")
        check(f"dataset_values:{case_id}", np.isfinite(X).all() and np.isin(y, [-1, 1]).all()
              and np.max(np.linalg.norm(X, axis=1)) <= 1. + 1e-12)
        check(f"dataset_generator_radius:{case_id}", case["B_generator"] == min(3., case["B"]))
        dataset_arrays[case_id] = (X, y)
        dataset_checks[case_id] = {"hashes": hashes, "rows": T, "d": d,
                                  "B": case["B"], "B_generator": case["B_generator"]}

    for planned in manifest["runs"]:
        run_id, folder = planned["id"], Path(planned["output"])
        spec, supervisor = read(folder / "spec.json"), read(folder / "supervisor.json")
        worker = supervisor.get("worker", {})
        n = int(worker.get("committed_round", 0))
        check(f"specification:{run_id}", all(spec.get(k) == v for k, v in planned.items()))
        check(f"input_identity:{run_id}", spec["data_hashes"] == manifest["datasets"][spec["case_id"]]["hashes"])
        before, after = read(folder / "source_hashes_before.json"), read(folder / "source_hashes_after.json")
        check(f"sources_unchanged:{run_id}", before == after and supervisor["sources_unchanged"])
        for path, value in before.items():
            source_versions[path].add(value)
        check(f"current_source_identity:{run_id}", all(digest(ROOT / p) == h for p, h in after.items()))
        verification = supervisor.get("job_verification") or {}
        check(f"hard_cap:{run_id}", supervisor.get("hard_process_commit_limit") == CAP
              and supervisor.get("os_hard_limit_active") is True
              and verification.get("process_memory_limit") == CAP
              and verification.get("verified_inside_actual_interpreter") is True
              and verification.get("pid") == supervisor.get("actual_worker_pid"))
        pools = worker.get("threadpools", [])
        check(f"one_worker_and_thread:{run_id}", supervisor.get("workers_at_once") == 1
              and supervisor.get("blas_threads") == 1 and bool(pools)
              and all(p.get("num_threads") == 1 for p in pools))
        check(f"terminal_status:{run_id}", supervisor.get("status") not in (None, "running"))
        check(f"prefix_bounds:{run_id}", 0 <= n <= spec["rounds"] <= spec["T"])
        check(f"completed_status:{run_id}", supervisor["status"] != "completed"
              or (n == spec["rounds"] and not supervisor["checkpoint_is_censored"]))
        trajectory = worker.get("trajectory", {})
        path = folder / "trajectory.bin"
        size = path.stat().st_size
        check(f"trajectory_schema:{run_id}", trajectory.get("columns") == COLUMNS
              and trajectory.get("record_bytes") == 48
              and trajectory.get("committed_bytes") == n*48 and size >= n*48)
        values = np.fromfile(path, dtype="<f8", count=n*6).reshape(n, 6)
        check(f"trajectory_values:{run_id}", np.isfinite(values[:, :5]).all()
              and np.all(values[:, 2:5] >= 0.)
              and np.array_equal(values[:, 0], values[:, 1])
              and np.all(np.diff(values[:, 3]) >= 0.))
        labels = dataset_arrays[spec["case_id"]][1][:n]
        expected_loss = np.logaddexp(0., -labels*values[:, 1])
        residual = float(np.max(np.abs(expected_loss-values[:, 2]))) if n else 0.
        check(f"stable_loss:{run_id}", residual <= 5e-12, {"max_absolute_residual": residual})
        total = math.fsum(values[:, 2].tolist())
        check(f"committed_total:{run_id}", abs(total-worker.get("cumulative_loss", 0.)) <= 1e-8)
        if spec["algorithm"] == "author_aioli":
            clipped = np.clip(expit(values[:, 1]), 1e-12, 1.-1e-12)
            expected_clipped = np.where(labels == 1, -np.log(clipped), -np.log(1.-clipped))
            clipped_residual = float(np.max(np.abs(expected_clipped-values[:, 5]))) if n else 0.
            native_total = math.fsum(values[:, 5].tolist())
            check(f"native_reported_loss:{run_id}", np.isfinite(values[:, 5]).all()
                  and clipped_residual <= 2e-8
                  and abs(native_total-worker["author_reported_cumulative_loss"]) <= 1e-8,
                  {"max_absolute_residual": clipped_residual})
            expected_state = 8*(spec["d"]**2+2*spec["d"])
            check(f"author_state_and_source:{run_id}", np.all(values[:, 4] == expected_state)
                  and worker["diagnostics"]["pinned_commit"] == COMMIT
                  and worker["diagnostics"]["native_prediction_and_update_unchanged"] is True
                  and worker["diagnostics"]["configuration"]["R"] == 1.
                  and worker["diagnostics"]["configuration"]["lam"] == 1./spec["B"]**2
                  and worker["diagnostics"]["configuration"]["T_inner"] == 80
                  and worker["diagnostics"]["configuration"]["omega_solver"] == "lbfgs")
        else:
            native_total = None
            check(f"non_author_sentinel:{run_id}", np.isnan(values[:, 5]).all())
            diagnostic = worker.get("diagnostics", {})
            expected_samples = spec["T"] if spec["config"]["mode"] == "prescribed" else min(
                spec["T"], spec["config"].get("samples", 16))
            check(f"uncapped_rejection_and_schedule:{run_id}",
                  "conditional_rejection_cap" in diagnostic and "normal_rejection_cap" in diagnostic
                  and diagnostic["conditional_rejection_cap"] is None
                  and diagnostic["normal_rejection_cap"] is None
                  and diagnostic.get("mode") == spec["config"]["mode"]
                  and diagnostic.get("samples") == expected_samples
                  and diagnostic.get("prescribed_samples") == spec["T"])
        check(f"prefix_time:{run_id}", n == 0 or abs(values[-1, 3]-worker["completed_prefix_online_seconds"]) <= 1e-9)
        internal_peak = worker.get("memory", {}).get("peak_private_commit", 0)
        sampled_peak = supervisor.get("monitored_peak_private", 0)
        record = dict(id=run_id, label=spec["label"], stage=spec["stage"],
                      case_id=spec["case_id"], d=spec["d"], B=spec["B"], T=spec["T"],
                      seed=spec["seed"], status=supervisor["status"],
                      worker_status=worker.get("status"), failure_phase=worker.get("failure_phase"),
                      error=worker.get("error"), committed_round=n,
                      full_completion=supervisor["status"] == "completed" and n == spec["T"],
                      cumulative_loss=total, mean_loss=total/n if n else None,
                      author_reported_cumulative_loss=native_total,
                      completed_prefix_online_seconds=float(values[-1, 3]) if n else 0.,
                      recorded_online_seconds=worker.get("online_seconds"),
                      timing_is_censored=not supervisor["worker_final_available"],
                      recorded_online_seconds_scope=(
                          "Last checkpoint only: lower bound on attempted online work; unfinished prediction time is unknown"
                          if not supervisor["worker_final_available"] else
                          "Final worker total, including timed work on any unfinished round that raised a caught error"),
                      process_wall_seconds=supervisor["process_wall_seconds"],
                      final_committed_state_bytes=int(values[-1, 4]) if n else None,
                      peak_private_bytes=max(internal_peak, sampled_peak),
                      peak_measurement_is_censored=supervisor["checkpoint_is_censored"],
                      checkpoint_is_censored=supervisor["checkpoint_is_censored"],
                      uncommitted_bytes=size-n*48,
                      trajectory_sha256=digest(path),
                      numerical_failures=worker.get("diagnostics", {}).get("numerical_failures"),
                      diagnostics=worker.get("diagnostics", {}),
                      metrics=worker.get("metrics", {}), output=str(folder), values=values)
        records[run_id] = record
    check("one_source_version_per_measured_file", all(len(v) == 1 for v in source_versions.values()))
    require_checks()

    comparators, comparator_dir = {}, study / "comparators"
    comparator_dir.mkdir(exist_ok=True)

    def comparator(case_id, prefix):
        if prefix == 0:
            return None
        key = f"{case_id}__n{prefix}"
        if key not in comparators:
            X, y = dataset_arrays[case_id]
            B = manifest["datasets"][case_id]["metadata"]["case"]["B"]
            with threadpool_limits(limits=1):
                result = fit_comparator(X[:prefix], y[:prefix], B,
                                        tolerance=.001, max_seconds=2.)
            theta = result.pop("theta")
            actual_loss = float(np.logaddexp(0., -y[:prefix]*(X[:prefix]@theta)).sum())
            check(f"comparator_valid:{key}", np.isfinite(theta).all()
                  and result["theta_norm"] <= B*(1.+1e-12)
                  and abs(actual_loss-result["loss"]) <= 1e-7
                  and result["gap"] >= 0.)
            theta_path = comparator_dir / (key + ".npy")
            np.save(theta_path, theta, allow_pickle=False)
            result.update(case_id=case_id, prefix=prefix, B=B,
                          nonnegative_loss_lower_bound=max(0., result["loss_lower_bound"]),
                          theta_path=str(theta_path), theta_sha256=digest(theta_path),
                          input_hashes=manifest["datasets"][case_id]["hashes"])
            dump(comparator_dir / (key + ".json"), result)
            comparators[key] = result
        return comparators[key]

    def prefix_stats(record, prefix):
        if prefix == 0:
            return dict(prefix=0, cumulative_loss=0., mean_loss=None,
                        online_seconds=0., state_bytes=None, numerical_regret_interval=None,
                        author_clipped_cumulative_loss=None)
        values = record["values"][:prefix]
        total = math.fsum(values[:, 2].tolist())
        fit = comparator(record["case_id"], prefix)
        return dict(prefix=prefix, cumulative_loss=total, mean_loss=total/prefix,
                    online_seconds=float(values[-1, 3]), state_bytes=int(values[-1, 4]),
                    numerical_regret_interval=[total-fit["loss"], total-fit["nonnegative_loss_lower_bound"]],
                    comparator_key=f"{record['case_id']}__n{prefix}",
                    author_clipped_cumulative_loss=math.fsum(values[:, 5].tolist())
                    if record["label"] == "author_aioli" else None)

    controls = {r["case_id"]: r for r in records.values() if r["label"] == "author_aioli"}
    groups, pairs = defaultdict(list), []
    for record in records.values():
        if record["label"] == "author_aioli":
            continue
        groups[(record["d"], record["B"], record["T"], record["label"])].append(record)
        author = controls[record["case_id"]]
        prefix = min(record["committed_round"], author["committed_round"])
        soft, aioli = prefix_stats(record, prefix), prefix_stats(author, prefix)
        pairs.append(dict(soft_id=record["id"], author_id=author["id"], case_id=record["case_id"],
                          label=record["label"], prefix=prefix, soft=soft, author=aioli,
                          cumulative_loss_difference=soft["cumulative_loss"]-aioli["cumulative_loss"],
                          same_comparator_regret_difference=True))

    aggregates = []
    for (d, B, T, label), runs in sorted(groups.items()):
        case_ids = sorted({r["case_id"] for r in runs})
        author_runs = [controls[c] for c in case_ids]
        prefix = min(r["committed_round"] for r in runs+author_runs)
        datasets = []
        for case_id in case_ids:
            members = [r for r in runs if r["case_id"] == case_id]
            soft_values = [prefix_stats(r, prefix) for r in members]
            author = prefix_stats(controls[case_id], prefix)
            soft_mean = dict(cumulative_loss=mean(v["cumulative_loss"] for v in soft_values),
                             mean_loss=mean(v["mean_loss"] for v in soft_values) if prefix else None,
                             online_seconds=mean(v["online_seconds"] for v in soft_values),
                             numerical_regret_interval=[mean(v["numerical_regret_interval"][j] for v in soft_values)
                                                        for j in (0, 1)] if prefix else None)
            datasets.append(dict(case_id=case_id, sampler_repetitions=len(members), soft=soft_mean,
                                 author=author, soft_run_ids=[r["id"] for r in members],
                                 author_run_id=controls[case_id]["id"]))
        soft_mean = dict(cumulative_loss=mean(v["soft"]["cumulative_loss"] for v in datasets),
                         mean_loss=mean(v["soft"]["mean_loss"] for v in datasets) if prefix else None,
                         online_seconds=mean(v["soft"]["online_seconds"] for v in datasets),
                         numerical_regret_interval=[mean(v["soft"]["numerical_regret_interval"][j] for v in datasets)
                                                    for j in (0, 1)] if prefix else None)
        author_mean = dict(cumulative_loss=mean(v["author"]["cumulative_loss"] for v in datasets),
                           mean_loss=mean(v["author"]["mean_loss"] for v in datasets) if prefix else None,
                           online_seconds=mean(v["author"]["online_seconds"] for v in datasets),
                           numerical_regret_interval=[mean(v["author"]["numerical_regret_interval"][j] for v in datasets)
                                                      for j in (0, 1)] if prefix else None)
        def attained(entries):
            return dict(attempts=len(entries), completions=sum(r["full_completion"] for r in entries),
                        statuses=dict(Counter(r["status"] for r in entries)),
                        completed_round_range=span(r["committed_round"] for r in entries),
                        final_committed_state_bytes_range=span(r["final_committed_state_bytes"] for r in entries
                                                              if r["final_committed_state_bytes"] is not None),
                        peak_private_bytes_range=span(r["peak_private_bytes"] for r in entries),
                        censored_peak_count=sum(r["peak_measurement_is_censored"] for r in entries))
        aggregates.append(dict(d=d, B=B, T=T, label=label, prefix=prefix, datasets=len(datasets),
                               sampler_attempts=len(runs), soft=soft_mean, author=author_mean,
                               neutral_mean_loss=math.log(2),
                               cumulative_loss_difference=soft_mean["cumulative_loss"]-author_mean["cumulative_loss"],
                               online_time_ratio=soft_mean["online_seconds"]/author_mean["online_seconds"]
                               if author_mean["online_seconds"] > 0. else None,
                               soft_attained=attained(runs), author_attained=attained(author_runs),
                               dataset_means=datasets))
        check(f"aggregate_equal_prefix:{d}:{B}:{T}:{label}", all(prefix <= r["committed_round"]
              for r in runs+author_runs))
    # Give every attempt a same-data comparator at its own attained endpoint too.
    for record in records.values():
        record["final_prefix_metrics"] = prefix_stats(record, record["committed_round"])
    require_checks()
    serial_records = [{k: v for k, v in r.items() if k != "values"} for r in records.values()]
    summary = dict(schema_version=1, manifest=str(manifest_path), manifest_sha256=digest(manifest_path),
                   implementation="SOFT_BAYES_COMPLETE.md main schedule, float64; practical counts separately labeled",
                   comparator="Sigmoid/MALA paper authors' unchanged released AIOLI", author_commit=COMMIT,
                   author_source_sha256=AUTHOR_SHA, neutral_mean_loss=math.log(2),
                   attempted_runs=len(records), full_completions=sum(r["full_completion"] for r in records.values()),
                   statuses=dict(Counter(r["status"] for r in records.values())),
                   aggregation="Same prefix across all seeds and controls within row; average sampler seeds within dataset, then data seeds equally",
                   aggregates=aggregates, pairs=pairs, runs=serial_records, datasets=dataset_checks,
                   comparators=comparators, measured_source_hashes={p: sorted(v) for p, v in source_versions.items()},
                   preflight=manifest["preflight"],
                   limits={"workers_at_once": 1, "blas_threads": 1, "hard_private_commit_bytes": CAP,
                           "comparator_max_seconds": 2., "comparator_gap_tolerance": .001},
                   caveats=["Float64/practical caps have no inherited exact-real regret theorem",
                            "Numerical regret intervals use an optimization gap, not rigorous floating-point enclosures",
                            "Memory at each terminal prefix is not an equal-horizon memory comparison",
                            "For killed workers, recorded online time is a last-checkpoint lower bound; actual attempted online time is unknown",
                            "Process-wall time is separate and includes imports, input/recording and supervision",
                            "This exploratory synthetic study does not reproduce tuned real-data paper figures"],
                   audit_path=str(audit_path), report_elapsed_seconds=perf_counter()-started)
    summary["plot_path"] = make_plot(summary, study)
    dump(study / "summary.json", summary)
    markdown = make_markdown(summary, study)
    (ROOT / "SOFT_BAYES_COMPARISON_RESULT.md").write_text(markdown, encoding="utf-8")
    check("summary_written", (study / "summary.json").exists())
    dump(audit_path, audit_document())
    print(json.dumps({"attempts": len(records), "statuses": summary["statuses"],
                      "aggregate_rows": len(aggregates), "comparator_fits": len(comparators),
                      "audit_passed": all(c["passed"] for c in checks),
                      "report": str(ROOT / "SOFT_BAYES_COMPARISON_RESULT.md")}))
    return summary


def make_plot(summary, study):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in summary["aggregates"] if r["label"] != "prescribed_resource" and r["prefix"] > 1]
    if not rows:
        return None
    names = []
    for row in rows:
        schedule = row["label"].replace("practical_m", "m").replace("n", "/") if row["label"].startswith("practical") else row["label"]
        names.append(f"d={row['d']}, B={row['B']:g}, {schedule}; n={row['prefix']}")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(4.5, .60*len(rows)+1.4)), sharey=True)
    positions = list(range(len(rows)))
    colors = {"soft": "#17747D", "author": "#C26D32"}
    for ax, metric in zip(axes, ("mean_loss", "online_seconds")):
        for side, offset, label in (("soft", -.18, "Soft Bayes"), ("author", .18, "Released AIOLI")):
            ax.barh([p+offset for p in positions], [r[side][metric] for r in rows],
                    height=.33, color=colors[side], label=label)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(positions, labels=names)
    axes[0].invert_yaxis()
    axes[0].axvline(math.log(2), color="#5E6570", linestyle="--", linewidth=1, label="Neutral")
    axes[0].set_xlabel("Mean logistic loss per round")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Online seconds at same completed prefix (log scale)")
    fig.suptitle("Soft Bayes and released AIOLI: equal-prefix comparisons", fontsize=14)
    fig.text(.5, .012, "Means over data seeds after averaging sampler repetitions. Fixed synthetic data; practical caps are exploratory.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0., .035, 1., .95))
    path = Path(study) / "equal_prefix_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def make_markdown(summary, study):
    completed = summary["full_completions"]
    status_text = ", ".join(f"{count} {name}" for name, count in sorted(summary["statuses"].items()))
    selected_rows = [r for r in summary["aggregates"] if r["label"] != "prescribed_resource" and r["prefix"] > 1]
    loss_wins = sum(r["cumulative_loss_difference"] < -1e-10 for r in selected_rows)
    faster = sum(r["online_time_ratio"] is not None and r["online_time_ratio"] < 1 for r in selected_rows)
    lines = ["# Soft Bayes implementation and released-AIOLI comparison", "", "2026-09-11.", "",
             f"Implemented the main algorithm in [SOFT_BAYES_COMPLETE.md](SOFT_BAYES_COMPLETE.md) and ran all {summary['attempted_runs']} frozen attempts against the sigmoid/MALA authors' unchanged released AIOLI code. {completed} attempts completed their configured horizon; terminal statuses are {status_text}.", "",
             f"Among the {len(selected_rows)} non-resource aggregate comparisons with at least two common completed rounds, Soft Bayes has lower mean loss in {loss_wins} and lower mean online time in {faster}. These are exploratory comparisons at common attained prefixes; short or one-round prefixes provide little evidence about learning.", "",
             "The implementation is [experiments/soft_bayes.py](experiments/soft_bayes.py). It retains raw feature history and a label vector, starts every sample independently from the unbounded Gaussian prior, uses the specified proximal transition and uncapped conditional rejection, then averages probabilities with add-one smoothing. Prescribed runs use m=T and the manuscript mixing count. Practical labels cap only samples and transitions. Section 10's optional primitive-operation budget is not implemented; external experiment deadlines are not that algorithm.", "",
             "The comparator is [AuthorAIOLI](experiments/author_baselines.py), loading the unchanged release at commit `"+COMMIT+"`. Its named lambda=B^-2 produces actual initial precision B², and its linear-statistic increment differs from the defining AIOLI surrogate. These behaviors are preserved. It is the sigmoid-paper authors' comparison code, not verified original-AIOLI-author code; see [source audit](SOFT_BAYES_AIOLI_SOURCE_AUDIT.md).", "",
             "## Equal-prefix prediction and time", "",
             "Every row uses one common completed prefix across its data seeds, sampler repetitions and matching AIOLI controls. When a run reaches a time limit, this shared endpoint is selected by attained runtime and can depend on sampler randomness; it is not an estimate at a predeclared completed horizon. Means first average sampler seeds within each dataset and then weight datasets equally. Resource rows use only the first dataset and one sampler seed. Online seconds include construction and prediction/update; module imports, input streaming and recording are excluded. Time ratios are ratios of these mean times. Neutral loss is **0.693147181 per round**.", "",
             "| d | B | Configured T | Soft Bayes schedule | Common rounds | Soft loss/round | AIOLI loss/round | Soft seconds | AIOLI seconds | Time ratio |",
             "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["aggregates"]:
        lines.append(f"| {row['d']} | {row['B']:g} | {row['T']} | {row['label']} | {row['prefix']} | "
                     f"{number(row['soft']['mean_loss'])} | {number(row['author']['mean_loss'])} | "
                     f"{number(row['soft']['online_seconds'])} | {number(row['author']['online_seconds'])} | {number(row['online_time_ratio'], 3)} |")
    if summary.get("plot_path"):
        lines += ["", "![Equal-prefix loss and time comparisons](experiments/results/soft_bayes_2026-09-11/equal_prefix_comparison.png)"]
    lines += ["", "One-round comparisons contain the exact initial symmetry prediction of one half and therefore do not test posterior learning. Completed-prefix online time excludes any unfinished next prediction. When the worker returns a final record, its total also includes timed unfinished work that raised a caught error. When the supervisor kills a worker, its recorded online time stops at the last checkpoint and is only a lower bound: time spent inside the interrupted prediction is unknown. Process-wall time is measured separately and includes imports, input/recording and supervision. No configured-horizon loss or runtime is extrapolated.", "",
              "## Numerical comparator regret", "",
              "One Euclidean-ball comparator is fitted per unique dataset/prefix after all measured workers finish, with a two-second cooperative allowance and first-order gap target 0.001. The same fit is used by both methods and all repetitions. Intervals below average the dataset-level cumulative regret endpoints. They account for numerical optimization gaps, not floating-point error; an unfinished fit remains usable with its reported wider gap. Regret differences equal loss differences against the same comparator.", "",
              "| d | B | Schedule | Prefix | Soft cumulative regret interval | AIOLI cumulative regret interval |",
              "|---:|---:|---|---:|---:|---:|"]
    for row in summary["aggregates"]:
        def interval(value):
            return "unavailable" if value is None else f"[{value[0]:.6f}, {value[1]:.6f}]"
        lines.append(f"| {row['d']} | {row['B']:g} | {row['label']} | {row['prefix']} | "
                     f"{interval(row['soft']['numerical_regret_interval'])} | {interval(row['author']['numerical_regret_interval'])} |")
    lines += ["", "## Completed horizons and memory", "",
              "The next table describes each method at its own attained endpoint. Round ranges make these different endpoints explicit; this is not an equal-horizon memory comparison. State counts owned persistent arrays, whereas peak private MiB includes runtime, imports, scratch, recorder and input buffers. For completed workers, peak memory is the maximum of the internal OS peak and supervisor samples. Killed-run peaks are lower bounds; a `*` marks a group containing censored peaks.", "",
              "| d | B | Schedule/method | Full completions | Final rounds | Final state bytes | Peak private MiB |",
              "|---:|---:|---|---:|---:|---:|---:|"]
    shown_author = set()
    for row in summary["aggregates"]:
        att = row["soft_attained"]
        mark = "*" if att["censored_peak_count"] else ""
        lines.append(f"| {row['d']} | {row['B']:g} | {row['label']} | {att['completions']}/{att['attempts']} | "
                     f"{display_span(att['completed_round_range'])} | {display_span(att['final_committed_state_bytes_range'])} | "
                     f"{display_span(att['peak_private_bytes_range'], 1024**2, 3)}{mark} |")
        key = (row["d"], row["B"], row["T"])
        if key not in shown_author and row["label"] != "prescribed_resource":
            shown_author.add(key)
            att = row["author_attained"]
            mark = "*" if att["censored_peak_count"] else ""
            lines.append(f"| {row['d']} | {row['B']:g} | released AIOLI | {att['completions']}/{att['attempts']} | "
                         f"{display_span(att['completed_round_range'])} | {display_span(att['final_committed_state_bytes_range'])} | "
                         f"{display_span(att['peak_private_bytes_range'], 1024**2, 3)}{mark} |")
    max_clip_difference = max(abs(r["cumulative_loss"]-r["author_reported_cumulative_loss"])
                              for r in summary["runs"] if r["label"] == "author_aioli")
    lines += ["", f"The largest per-run difference between stable AIOLI score loss and its native probability-clipped reported total is {max_clip_difference:.12g}. Both totals are retained; the primary table uses stable score loss.", "",
              "## Every measured attempt", "",
              "Losses and prefix times here are individual completed-prefix values and must not be compared across unequal row counts. A ≥ mark on recorded online seconds means the worker was killed: the displayed checkpoint total is a lower bound on attempted online time. The separate process-wall measurement also includes work outside the online timer. Full seed/configuration details and per-pair equal-prefix records are saved in the summary JSON.", "",
              "| Dataset | Schedule | Sampler seed | Status | Completed rounds | Loss/round | Prefix seconds | Recorded online seconds | Process-wall seconds |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for run in sorted(summary["runs"], key=lambda r: (r["d"], r["B"], r["case_id"], r["label"], r["seed"])):
        recorded_online = number(run['recorded_online_seconds'])
        if run["timing_is_censored"]:
            recorded_online = "≥" + recorded_online
        lines.append(f"| {run['case_id']} | {run['label']} | {run['seed']} | {run['status']} | {run['committed_round']}/{run['T']} | "
                     f"{number(run['mean_loss'])} | {number(run['completed_prefix_online_seconds'])} | {recorded_online} | {number(run['process_wall_seconds'])} |")
    lines += ["", "## Reproducibility and limits", "",
              "All attempts use fixed stored synthetic rows, R=1 and B_generator=min(3,B), with algorithm/comparator B kept separate. The study has two data seeds, two sampler repetitions for each non-resource Soft Bayes cell, and one deterministic AIOLI control per dataset. The manifest and [protocol](SOFT_BAYES_EXPERIMENT_PLAN.md) were fixed before outcomes. Each measured worker has one BLAS thread and a verified actual-process Windows Job Object cap of 2 GiB private commit; workers execute sequentially.", "",
              "[The artifact audit](SOFT_BAYES_ARTIFACT_AUDIT.json) checks source/data identity, all terminal statuses, cap and thread evidence, committed record boundaries, recalculated losses, equal prefixes and comparator feasibility. Source hashes are stable across all measured attempts. Tests and this audit validate implementation behavior and recording; they do not establish prescribed-chain mixing in float64 or comparator regret for practical caps.", "",
              "Artifacts: [implementation](experiments/soft_bayes.py), [campaign](experiments/soft_bayes_campaign.py), [report generator](experiments/soft_bayes_report.py), [manifest](experiments/results/soft_bayes_2026-09-11/manifest.json), [complete summary](experiments/results/soft_bayes_2026-09-11/summary.json), [comparator fits](experiments/results/soft_bayes_2026-09-11/comparators), and each run's spec, supervisor record, source hashes and binary trajectory under the study's `runs` directory.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    args = parser.parse_args()
    generate(args.study)


if __name__ == "__main__":
    main()
