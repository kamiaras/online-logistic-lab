"""Fixed-endpoint analysis of the isolated shared-tangent ablation campaign.

Run after the supervised campaign, from the project root:
    python -m experiments.shared_ablation_analyze

Reads every planned manifest cell, including unstarted/failed/censored cells.
Writes analysis.json and analysis.md beside the manifest. No algorithm is run,
no source result is overwritten, and no timeout-selected endpoint is substituted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "experiments/results/shared_ablation"
ENDPOINTS = (100, 250, 500)
PRIMARY_SEEDS = tuple(range(910001, 910007))
SECONDARY_SEEDS = PRIMARY_SEEDS[:3]
CELLS = {
    "SF": ("sweep", "fresh"), "SW": ("sweep", "warm"),
    "IF": ("lanes", "fresh"), "IW": ("lanes", "warm"),
}
CONTRASTS = {
    "warm_minus_fresh_sweep": {"SW": 1, "SF": -1},
    "warm_minus_fresh_lanes": {"IW": 1, "IF": -1},
    "lanes_minus_sweep_fresh": {"IF": 1, "SF": -1},
    "lanes_minus_sweep_warm": {"IW": 1, "SW": -1},
    "persistence_main_effect": {"SW": .5, "SF": -.5, "IW": .5, "IF": -.5},
    "layout_main_effect": {"IF": .5, "SF": -.5, "IW": .5, "SW": -.5},
    "interaction": {"IW": 1, "IF": -1, "SW": -1, "SF": 1},
}
METRICS = ("loss", "guard_c1_loss", "guard_c3_loss", "guard_c4_loss", "online_seconds")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_optional_json(path, errors):
    try:
        return read_json(path)
    except (OSError, ValueError) as exc:
        errors.append(f"{path.name}: {exc}")
        return None


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summary(values, expected):
    """Always expose denominators; incomplete means are descriptive only."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    out = dict(n=n, expected_n=expected, all_planned_available=n == expected,
               available_mean=float(np.mean(values)) if n else None,
               available_sd=float(np.std(values, ddof=1)) if n > 1 else None,
               minimum=float(np.min(values)) if n else None,
               maximum=float(np.max(values)) if n else None)
    out["complete_mean"] = out["available_mean"] if n == expected else None
    # These intervals concern sampler repetitions on this ONE fixed dataset.
    # Incomplete sets can depend on runtime and receive no nominal interval.
    t975 = {2: 12.706204736, 3: 4.302652730, 4: 3.182446305,
            5: 2.776445105, 6: 2.570581836}
    out["conditional_sampler_t95"] = None
    if n == expected and n in t975:
        half = t975[n] * out["available_sd"] / math.sqrt(n)
        out["conditional_sampler_t95"] = [out["available_mean"] - half,
                                             out["available_mean"] + half]
    return out


def guards(logits, labels, factor):
    rounds = np.arange(1, len(logits) + 1, dtype=float)
    caps = np.log(factor * rounds * (rounds + 1))
    clipped = np.clip(logits, -caps, caps)
    return clipped, np.logaddexp(0., -labels * clipped)


def verify_data(folder):
    """Replay the generator's recorded block-interleaved hash, not its RNG."""
    folder = Path(folder)
    errors = []
    metadata = read_optional_json(folder / "data.json", errors)
    out = dict(path=str(folder), metadata=metadata, errors=errors)
    if metadata is None:
        out["errors"].append("missing data.json")
        return out
    d, T = int(metadata["case"]["d"]), int(metadata["case"]["T"])
    block_size = int(metadata["block_size"])
    h = hashlib.sha256()
    try:
        with (folder / "X.bin").open("rb") as fx, (folder / "y.bin").open("rb") as fy:
            for start in range(0, T, block_size):
                n = min(block_size, T - start)
                xb, yb = fx.read(8 * d * n), fy.read(n)
                if len(xb) != 8 * d * n or len(yb) != n:
                    raise ValueError("data file shorter than declared horizon")
                h.update(xb)
                h.update(yb)
            if fx.read(1) or fy.read(1):
                raise ValueError("data file longer than declared horizon")
        out["computed_interleaved_sha256"] = h.hexdigest()
        out["recorded_hash_matches"] = h.hexdigest() == metadata["sha256_interleaved_blocks"]
        if not out["recorded_hash_matches"]:
            out["errors"].append("recorded data hash mismatch")
        out["X_sha256"] = sha256(folder / "X.bin")
        out["y_sha256"] = sha256(folder / "y.bin")
    except (OSError, ValueError) as exc:
        out["errors"].append(str(exc))
    return out


def read_run(spec, campaign, data_checks):
    # Relocatable campaign directories use the original manifest's run basename.
    folder = campaign / Path(spec["output"]).name
    read_errors = []
    result = read_optional_json(folder / "result.json", read_errors)
    supervisor = read_optional_json(folder / "supervisor.json", read_errors)
    progress = read_optional_json(folder / "progress.json", read_errors)
    status = result.get("status", "unknown") if result else (
        "abnormal_exit" if supervisor else "running_or_incomplete" if folder.exists() else "not_started")
    record = dict(id=folder.name, spec=spec, status=status, result=result,
                  supervisor=supervisor, progress_only=progress, errors=[], read_errors=read_errors,
                  trajectory_rounds=0, trajectory_valid=False, endpoints={})
    if data_checks[str(Path(spec["data_path"]))]["errors"]:
        record["errors"].append("input data verification failed")
    arrays = None
    trajectory = folder / "trajectory.npz"
    if trajectory.exists():
        try:
            with np.load(trajectory, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            required = ("logits", "losses", "online_times", "scores", "initial_scores", "labels")
            if any(key not in arrays for key in required):
                raise ValueError("trajectory missing a required array")
            n = len(arrays["logits"])
            record["trajectory_rounds"] = n
            if any(len(arrays[key]) != n for key in required):
                raise ValueError("trajectory lengths disagree")
            if n > spec["rounds"] or n > spec["T"]:
                raise ValueError("trajectory exceeds requested endpoint/horizon")
            for key in ("logits", "losses", "online_times", "labels"):
                if arrays[key].ndim != 1 or not np.all(np.isfinite(arrays[key])):
                    raise ValueError(f"nonfinite or non-vector {key}")
            y = np.fromfile(Path(spec["data_path"]) / "y.bin", dtype="i1", count=n)
            if not np.array_equal(arrays["labels"], y) or not np.all(np.isin(y, [-1, 1])):
                raise ValueError("trajectory labels differ from the fixed input sequence")
            if result and int(result["round"]) != n:
                raise ValueError("result round differs from trajectory length")
            if arrays["scores"].ndim != 2 or arrays["initial_scores"].shape != arrays["scores"].shape:
                raise ValueError("score/start matrix dimensions disagree")
            record["recorded_score_count"] = arrays["scores"].shape[1]
            if arrays["scores"].shape[1] != spec["config"]["tilts"]:
                raise ValueError("actual score count differs from the planned cap")
            skipped = np.all(np.isnan(arrays["scores"]), axis=1)
            if not np.all(np.isfinite(arrays["scores"][~skipped])):
                raise ValueError("partial NaN or nonfinite sampled score row")
            if not np.all(np.isnan(arrays["initial_scores"][skipped])) or not np.all(
                    np.isfinite(arrays["initial_scores"][~skipped])):
                raise ValueError("invalid initial scores or skipped row")
            if np.any(arrays["logits"][skipped] != 0) or not np.allclose(
                    arrays["scores"][~skipped].mean(axis=1), arrays["logits"][~skipped],
                    rtol=1e-12, atol=1e-10):
                raise ValueError("logits disagree with their recorded scores/skip")
            losses = np.logaddexp(0., -y * arrays["logits"])
            if not np.allclose(losses, arrays["losses"], rtol=1e-12, atol=1e-10):
                raise ValueError("stored logistic loss mismatch")
            if np.any(np.diff(arrays["online_times"]) < 0) or np.any(arrays["online_times"] < 0):
                raise ValueError("invalid cumulative online times")
            if result and not math.isclose(float(losses.sum()), result["cumulative_loss"],
                                           rel_tol=1e-11, abs_tol=1e-8):
                raise ValueError("cumulative loss mismatch")
            record["sampled_rounds_in_trajectory"] = int(np.sum(~skipped))
            record["skipped_rounds_in_trajectory"] = int(np.sum(skipped))
            record["move_budget_per_sampled_round"] = (
                spec["config"]["burn_in"] + (record["recorded_score_count"] - 1)
                * spec["config"]["transitions"])
            if result:
                counters = result.get("metrics", {}).get("counters", {})
                expected_moves = int(np.sum(~skipped)) * record["move_budget_per_sampled_round"]
                if counters.get("proximal_steps", 0) != expected_moves:
                    # On a caught error an unfinished prediction can consume work
                    # without producing a trajectory row. Charge it, do not erase it.
                    record["unfinished_prediction_work"] = dict(
                        recorded_moves=counters.get("proximal_steps", 0),
                        completed_prediction_moves=expected_moves)
                    if status == "completed":
                        raise ValueError("completed-run proximal move counter mismatch")
            for factor in (1, 3, 4):
                clipped, guarded_losses = guards(arrays["logits"], y, factor)
                arrays[f"guard_c{factor}_losses"] = guarded_losses
                if factor == 1 and "guarded_logits" in arrays and not np.allclose(
                        clipped, arrays["guarded_logits"], rtol=1e-12, atol=1e-12):
                    raise ValueError("worker's c=1 guard mismatch")
                point_budget = 1 / (factor * np.arange(1, n + 1) * np.arange(2, n + 2))
                if np.any(guarded_losses - losses > point_budget + 1e-9):
                    raise ValueError(f"factor-{factor} pointwise guard inequality failed")
            record["trajectory_valid"] = not record["errors"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            record["errors"].append(str(exc))
            arrays = None
    for endpoint in ENDPOINTS:
        key = str(endpoint)
        if endpoint > spec["rounds"]:
            record["endpoints"][key] = dict(available=False, reason="not_requested")
        elif not record["trajectory_valid"]:
            record["endpoints"][key] = dict(available=False, reason=(
                "invalid_trajectory" if record["errors"] else "no_saved_trajectory"))
        elif record["trajectory_rounds"] < endpoint:
            record["endpoints"][key] = dict(available=False, reason="fixed_endpoint_unattained")
        else:
            z = arrays["logits"][:endpoint]
            loss = float(arrays["losses"][:endpoint].sum())
            e = dict(available=True, n=endpoint, loss=loss, loss_per_round=loss / endpoint,
                     online_seconds=float(arrays["online_times"][endpoint - 1]),
                     mean_absolute_logit=float(np.mean(np.abs(z))),
                     maximum_absolute_logit=float(np.max(np.abs(z))),
                     sampled_rounds=int(np.sum(np.all(np.isfinite(arrays["scores"][:endpoint]), axis=1))))
            for factor in (1, 3, 4):
                value = float(arrays[f"guard_c{factor}_losses"][:endpoint].sum())
                e.update({f"guard_c{factor}_loss": value,
                          f"guard_c{factor}_loss_per_round": value / endpoint,
                          f"guard_c{factor}_loss_change": value - loss,
                          f"guard_c{factor}_increase_bound": endpoint / (factor * (endpoint + 1)),
                          f"guard_c{factor}_changed_rounds": int(np.sum(np.abs(z) > np.log(
                              factor * np.arange(1, endpoint + 1) * np.arange(2, endpoint + 2))))})
            record["endpoints"][key] = e
    return record, arrays


def identify(record):
    s, c = record["spec"], record["spec"]["config"]
    return (s["case"], float(s["B"]), int(s["seed"]), c["layout"],
            c["persistence"], c["acceptance"], int(c["tilts"]), int(c["burn_in"]))


def endpoint_for(record, endpoint):
    return record["endpoints"][str(endpoint)] if record else dict(available=False, reason="not_in_manifest")


def aggregate_cells(primary, cases, seeds_by_case=None):
    rows, contrasts = [], []
    for case in cases:
        seeds = (seeds_by_case or {}).get(case, PRIMARY_SEEDS)
        for endpoint in ENDPOINTS:
            for cell in CELLS:
                points = [endpoint_for(primary.get((case, seed, cell)), endpoint) for seed in seeds]
                rows.append(dict(case=case, endpoint=endpoint, cell=cell,
                                 seeds=[dict(seed=s, **p) for s, p in zip(seeds, points)],
                                 metrics={key: summary([p[key] for p in points if p["available"]], len(seeds))
                                          for key in METRICS}))
            for name, weights in CONTRASTS.items():
                paired = []
                for seed in seeds:
                    points = {cell: endpoint_for(primary.get((case, seed, cell)), endpoint) for cell in weights}
                    missing = {cell: p["reason"] for cell, p in points.items() if not p["available"]}
                    row = dict(seed=seed, available=not missing, missing=missing)
                    if not missing:
                        row["differences"] = {key: sum(weights[cell] * points[cell][key] for cell in weights)
                                              for key in METRICS}
                    paired.append(row)
                contrasts.append(dict(case=case, endpoint=endpoint, contrast=name,
                                      signed_cell_weights=weights, seeds=paired,
                                      metrics={key: summary([p["differences"][key] for p in paired if p["available"]], len(seeds))
                                               for key in METRICS}))
    return rows, contrasts


def covariance_diagnostic(scores, initial_scores):
    """Columns are distinct tilts; rows are independent sampler repetitions."""
    repetitions, m = scores.shape
    centered = scores - scores.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / (repetitions - 1)
    variances = np.maximum(0., np.diag(covariance))
    mean_marginal = float(np.mean(variances))
    diagonal_mean = mean_marginal / m
    variance_mean = float(np.var(scores.mean(axis=1), ddof=1))
    denominator = np.sqrt(variances[:, None] * variances[None, :])
    use = (denominator > 0) & ~np.eye(m, dtype=bool)
    correlations = covariance[use] / denominator[use]
    return dict(repetitions=repetitions, m=m, mean_marginal_variance=mean_marginal,
                variance_of_average=variance_mean, diagonal_only_variance_of_average=diagonal_mean,
                offdiagonal_variance_contribution=variance_mean - diagonal_mean,
                covariance_inflation=variance_mean / diagonal_mean if diagonal_mean > 0 else None,
                equivalent_independent_score_count=mean_marginal / variance_mean if variance_mean > 0 else None,
                mean_offdiagonal_correlation=float(np.mean(correlations)) if len(correlations) else None,
                offdiagonal_pairs_with_positive_variance=int(np.sum(use)),
                sampler_mean_logit=float(np.mean(scores)),
                mean_absolute_midpoint_move=float(np.mean(np.abs(scores - initial_scores))),
                rms_midpoint_move=float(np.sqrt(np.mean((scores - initial_scores) ** 2))))


def covariance_rows(primary, arrays, cases, seeds_by_case=None):
    rows = []
    for case in cases:
        seeds = (seeds_by_case or {}).get(case, PRIMARY_SEEDS)
        for cell in CELLS:
            records = [primary.get((case, seed, cell)) for seed in seeds]
            for t in range(1, max(ENDPOINTS) + 1):
                missing, score_rows, start_rows = {}, [], []
                skipped = []
                for seed, record in zip(seeds, records):
                    a = arrays.get(record["id"]) if record and record["trajectory_valid"] else None
                    if a is None or len(a["logits"]) < t:
                        missing[str(seed)] = "no_valid_saved_round"
                    elif np.all(np.isnan(a["scores"][t - 1])):
                        skipped.append(seed)
                    else:
                        score_rows.append(a["scores"][t - 1])
                        start_rows.append(a["initial_scores"][t - 1])
                row = dict(case=case, cell=cell, round=t, available=False,
                           missing=missing, skipped_seeds=skipped,
                           expected_repetitions=len(seeds),
                           available_sampled_repetitions=len(score_rows))
                if missing:
                    row["reason"] = "all_declared_repetitions_required"
                elif len(skipped) == len(seeds):
                    row["reason"] = "deterministic_skip"
                elif skipped:
                    row["reason"] = "inconsistent_skip_across_sampler_seeds"
                else:
                    row["available"] = True
                    row["diagnostic"] = covariance_diagnostic(np.stack(score_rows), np.stack(start_rows))
                rows.append(row)
    return rows


def paired_secondary(records, lookup):
    rows = []
    for record in records:
        s, c = record["spec"], record["spec"]["config"]
        if s["stage"] not in ("acceptance", "budget", "radius"):
            continue
        base = list(identify(record))
        if s["stage"] == "acceptance":
            base[5] = "full"
        elif s["stage"] == "budget":
            base[6:8] = [16, 4]
        else:
            base[1] = 100.
        reference = lookup.get(tuple(base))
        for endpoint in ENDPOINTS:
            left, right = endpoint_for(record, endpoint), endpoint_for(reference, endpoint)
            available = left["available"] and right["available"]
            row = dict(stage=s["stage"], case=s["case"], seed=s["seed"], endpoint=endpoint,
                       B=s["B"], tilts=c["tilts"], moves_per_midpoint=c["burn_in"],
                       run=record["id"], reference=reference["id"] if reference else None,
                       available=available, missing={})
            if available:
                row["difference"] = {key: left[key] - right[key] for key in METRICS}
                row["run_endpoint"], row["reference_endpoint"] = left, right
                row["online_time_ratio"] = left["online_seconds"] / right["online_seconds"] if right["online_seconds"] > 0 else None
            else:
                row["missing"] = {name: p["reason"] for name, p in [("run", left), ("reference", right)] if not p["available"]}
            rows.append(row)
    grouped = []
    keys = sorted({(r["stage"], r["case"], r["B"], r["tilts"], r["moves_per_midpoint"], r["endpoint"]) for r in rows})
    for key in keys:
        selected = [r for r in rows if (r["stage"], r["case"], r["B"], r["tilts"], r["moves_per_midpoint"], r["endpoint"]) == key]
        grouped.append(dict(stage=key[0], case=key[1], B=key[2], tilts=key[3], moves_per_midpoint=key[4], endpoint=key[5],
                            seeds=[dict(seed=r["seed"], available=r["available"], missing=r["missing"]) for r in selected],
                            differences={metric: summary([r["difference"][metric] for r in selected if r["available"]], 3) for metric in METRICS},
                            mean_time_ratio=summary([r["online_time_ratio"] for r in selected if r["available"] and r["online_time_ratio"] is not None], 3)))
    return rows, grouped


def radius_checks(records, lookup, data_checks):
    rows = []
    for record in records:
        if record["spec"]["stage"] != "radius":
            continue
        base = list(identify(record))
        base[1] = 100.
        reference = lookup.get(tuple(base))
        s = record["spec"]
        reference_spec = reference["spec"] if reference else None
        original = data_checks[str(Path(s["data_path"]))]
        other = data_checks[str(Path(reference_spec["data_path"]))] if reference else None
        rows.append(dict(run=record["id"], reference=reference["id"] if reference else None,
                         B=s["B"], reference_B=100., sampler_seed=s["seed"],
                         same_input_path=bool(reference and Path(s["data_path"]).resolve() == Path(reference_spec["data_path"]).resolve()),
                         same_verified_data_hash=bool(other and not original["errors"] and not other["errors"]
                             and original.get("computed_interleaved_sha256") == other.get("computed_interleaved_sha256")),
                         same_configured_horizon=bool(reference and s["T"] == reference_spec["T"]),
                         same_data_generator=bool(other and original["metadata"]["case"] == other["metadata"]["case"])))
    return rows


def fmt(value):
    return "unavailable" if value is None else f"{value:.6g}"


def write_markdown(report, destination):
    targets = "; ".join(f"{case}: {len(seeds)} seeds" for case, seeds in report["declared_primary_seeds"].items())
    lines = ["# Shared-tangent ablation: fixed-endpoint analysis", "",
             f"Generated {report['generated_utc']}. All {report['planned_runs']} manifest cells are retained.", "",
             "Status counts: " + "; ".join(f"{k}: {v}" for k, v in sorted(report["status_counts"].items())) + ".", "",
             "SF = fresh sweep; SW = persistent sweep; IF = fresh separate midpoint lanes; IW = persistent lanes. "
             "The four primary cells each use 16 midpoints and 64 completed proximal moves per sampled round, with full-residual testing. "
             "These practical schedules have no prescribed-regret guarantee.", "",
             "Missing fixed endpoints stay missing. Tables show means of available repetitions and their denominator; the JSON separately "
             "sets `complete_mean` to null when any prescribed repetition is unavailable. Censored available means are descriptive. "
             "Declared primary targets: " + targets + ". Original six-seed cells remain separately available in the JSON. "
             "The default follows the runtime-only continuation amendment. Sampler repetitions are conditional on each "
             "of two fixed datasets, not independent data replications.", "",
             "## Primary fixed endpoints", "",
             "| Dataset | Round | Cell | n/target | Raw loss/round | Guard c=3 loss/round | Guard c=4 loss/round | Online seconds |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in report["primary_cells"]:
        m = row["metrics"]
        per_round = lambda key: (m[key]["available_mean"] / row["endpoint"] if m[key]["available_mean"] is not None else None)
        lines.append(f"| {row['case']} | {row['endpoint']} | {row['cell']} | {m['loss']['n']}/{m['loss']['expected_n']} | {fmt(per_round('loss'))} | {fmt(per_round('guard_c3_loss'))} | {fmt(per_round('guard_c4_loss'))} | {fmt(m['online_seconds']['available_mean'])} |")
    lines += ["", "## Paired primary effects", "",
              "Signed effects are cumulative loss differences at the fixed endpoint; negative favors the named change. "
              "The interaction is (IW−IF)−(SW−SF). The JSON retains all seven contrasts and every seed, including missing pairs.", "",
              "| Dataset | Round | Contrast | paired n/target | Raw loss difference | Guard c=3 difference | Guard c=4 difference |",
              "|---|---:|---|---:|---:|---:|---:|"]
    for row in report["primary_contrasts"]:
        if row["contrast"] not in ("persistence_main_effect", "layout_main_effect", "interaction"):
            continue
        m = row["metrics"]
        lines.append(f"| {row['case']} | {row['endpoint']} | {row['contrast']} | {m['loss']['n']}/{m['loss']['expected_n']} | {fmt(m['loss']['available_mean'])} | {fmt(m['guard_c3_loss']['available_mean'])} | {fmt(m['guard_c4_loss']['available_mean'])} |")
    lines += ["", "## Secondary matched comparisons", "",
              "Acceptance rows are Poisson minus contemporary full testing. Budget rows compare 64×1 or 256×1 persistent lanes "
              "with 16×4 persistent lanes: 64×1 matches move count, 256×1 uses four times the moves, and both change the midpoint grid. "
              "Radius rows change model B against B=100 on the same input file. All differences use paired sampler-seed labels.", "",
              "| Stage | Dataset | B | m×moves | Round | paired n/3 | Raw loss difference | Guard c=3 difference | Guard c=4 difference | Mean time ratio |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["secondary_groups"]:
        # Keep all requested endpoints in the readable table; not-requested cells
        # remain in JSON rather than filling this table with intentional blanks.
        if (row["stage"] == "acceptance" and row["endpoint"] > 100) or (row["stage"] == "radius" and row["endpoint"] > 250):
            continue
        m = row["differences"]
        lines.append(f"| {row['stage']} | {row['case']} | {row['B']:g} | {row['tilts']}×{row['moves_per_midpoint']} | {row['endpoint']} | {m['loss']['n']}/3 | {fmt(m['loss']['available_mean'])} | {fmt(m['guard_c3_loss']['available_mean'])} | {fmt(m['guard_c4_loss']['available_mean'])} | {fmt(row['mean_time_ratio']['available_mean'])} |")
    lines += ["", "## Centered score dependence", "",
              "Each tilt is centered across the declared independent full-trajectory sampler repetitions at the same round. "
              "Equivalent independent-score count = mean marginal variance / variance of the score average. "
              "It is a noisy descriptive ratio, can exceed 16, and is not stationary ESS or an estimate of bias. "
              "All declared repetitions are required; no runtime-selected subset supplies a covariance estimate. "
              "With R repetitions the empirical covariance has rank at most R−1, making R=3 particularly imprecise. "
              "The JSON contains all rounds, with skipped and missing rows explicit.", "",
              "| Dataset | Round | Cell | Status | Mean off-diagonal correlation | Equivalent count | Variance of average |",
              "|---|---:|---|---|---:|---:|---:|"]
    for row in report["covariance_by_round"]:
        if row["round"] not in ENDPOINTS:
            continue
        d = row.get("diagnostic", {})
        status = f"{row['expected_repetitions']} repetitions" if row['available'] else row['reason']
        lines.append(f"| {row['case']} | {row['round']} | {row['cell']} | {status} | {fmt(d.get('mean_offdiagonal_correlation'))} | {fmt(d.get('equivalent_independent_score_count'))} | {fmt(d.get('variance_of_average'))} |")
    lines += ["", "## Guard and interpretation limits", "",
              "The original c=1 and c=3 stored-logit counterfactuals are retained, and c=4 was added by an algebraic "
              "allocation amendment before effect analysis: Kₜ=log(c t(t+1)). All three are present in the JSON. "
              "Clipping increases loss by at most 1/[c t(t+1)] on each round, hence by at most n/[c(n+1)] through round n; "
              "the maximum guarded round loss is log(1+c t(t+1)). Negative changes can be much larger in magnitude. "
              "Fixed data and prediction-independent updates make this valid postprocessing; it does not prove regret for the capped sampler.", "",
              "Runtime comparisons use cumulative trajectory online times at the same endpoint. Final counters, memory and timers in each run's "
              "JSON refer to the whole attempt, not a shorter endpoint. Nested timers must not be summed. Persistent `state_bytes` omits "
              "diagnostic score and capture arrays; measured process peak commit includes them. A supervisor-killed worker may expose progress "
              "without saved per-round arrays; those endpoints remain unavailable. The 20-minute campaign budget is a dispatch limit checked between runs.", "",
              "B checks use verified X/y content and configured horizon; model B changes the prior and comparator class. The script reports loss, "
              "not an uncomputed cross-B regret. The two signal datasets do not test adaptive data, reversals or all parameter regimes.", "",
              f"Data verification errors: {sum(bool(x['errors']) for x in report['data_checks'].values())}. "
              f"Run validation errors: {sum(bool(x['errors']) for x in report['runs'])}.", "",
              "Full seed-level endpoints, guard effects, provenance, missing-cell reasons and all covariance rounds are in `analysis.json`."]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(campaign, d100_repetitions=3):
    specs = read_json(campaign / "manifest.json")
    if not isinstance(specs, list) or not specs:
        raise ValueError("missing or empty manifest")
    data_checks = {path: verify_data(path) for path in sorted({str(Path(s["data_path"])) for s in specs})}
    records, arrays = [], {}
    for spec in specs:
        record, a = read_run(spec, campaign, data_checks)
        records.append(record)
        if a is not None:
            arrays[record["id"]] = a
    lookup = {}
    for record in records:
        key = identify(record)
        if key in lookup:
            raise ValueError(f"duplicate algorithm/data/seed manifest cell: {key}")
        lookup[key] = record
    primary = {}
    for record in records:
        if record["spec"]["stage"] == "primary":
            c = record["spec"]["config"]
            cell = next(k for k, v in CELLS.items() if v == (c["layout"], c["persistence"]))
            primary[(record["spec"]["case"], record["spec"]["seed"], cell)] = record
    cases = sorted({s["case"] for s in specs if s["stage"] == "primary"})
    dimensions = {s["case"]: s["d"] for s in specs if s["stage"] == "primary"}
    seeds_by_case = {case: PRIMARY_SEEDS[:d100_repetitions] if dimensions[case] == 100 else PRIMARY_SEEDS for case in cases}
    cells, contrasts = aggregate_cells(primary, cases, seeds_by_case)
    original_cells, original_contrasts = aggregate_cells(primary, cases)
    secondary, secondary_groups = paired_secondary(records, lookup)
    before = read_json(campaign / "source_hashes_before.json")
    after = read_json(campaign / "source_hashes_after.json")
    continuation_manifest = read_json(campaign / "continuation_manifest.json")
    continuation_result = read_json(campaign / "continuation_result.json")
    expected_continuation = [s for s in specs if s["stage"] == "primary"
                             and (s["d"] == 10 or s["seed"] <= 910003)]
    continuation_ids = {Path(s["output"]).name for s in expected_continuation}
    names = ["experiments/shared_ablation.py", "experiments/shared_ablation_worker.py",
             "experiments/shared_ablation_campaign.py", "experiments/shared_ablation_analyze.py",
             "experiments/shared_ablation_continue.py", "experiments/common.py",
             "SHARED_ABLATION_PLAN.md", "SHARED_ABLATION_AMENDMENT.md"]
    provenance = dict(recorded_before=before, recorded_after=after,
                      before_after_match=(before == after if before is not None and after is not None else None),
                      current_analysis_time_hashes={name: sha256(ROOT / name) for name in names if (ROOT / name).exists()},
                      current_historical_hashes={name: sha256(ROOT / name) for name in before or {} if (ROOT / name).exists()},
                      manifest_sha256=sha256(campaign / "manifest.json"))
    return dict(generated_utc=datetime.now(timezone.utc).isoformat(), planned_runs=len(specs),
                status_counts=dict(Counter(r["status"] for r in records)),
                stage_counts=dict(Counter(s["stage"] for s in specs)),
                fixed_endpoints=list(ENDPOINTS), provenance=provenance, data_checks=data_checks,
                continuation=dict(recorded_manifest=continuation_manifest, recorded_result=continuation_result,
                    target_specs_match_original=(continuation_manifest.get("targets") == expected_continuation
                                                 if continuation_manifest else None),
                    declared_target_count=len(expected_continuation),
                    target_saved_500_endpoint_count=sum(r["id"] in continuation_ids and r["endpoints"]["500"]["available"] for r in records),
                    terminal_attempts_are_not_completed_endpoints=True),
                runs=records, primary_cells=cells, primary_contrasts=contrasts,
                declared_primary_seeds=seeds_by_case,
                original_six_seed_cells=original_cells, original_six_seed_contrasts=original_contrasts,
                additional_primary_runs=[r["id"] for r in records if r["spec"]["stage"] == "primary"
                                         and r["spec"]["seed"] not in seeds_by_case[r["spec"]["case"]]],
                secondary_pairs=secondary, secondary_groups=secondary_groups,
                radius_same_sequence_checks=radius_checks(records, lookup, data_checks),
                covariance_by_round=covariance_rows(primary, arrays, cases, seeds_by_case),
                original_six_seed_covariance_by_round=covariance_rows(primary, arrays, cases),
                interpretation=[
                    "All planned cells retained; no time-selected common endpoint replaces 100/250/500.",
                    "Available-subset summaries may be runtime selected; complete_mean requires every planned repetition.",
                    "Conditional t intervals describe sampler randomness on each fixed dataset and are exploratory, not population-data intervals.",
                    "Warm 64x1 and 256x1 also change the midpoint grid; only 64x1 matches the original 64 completed moves.",
                    "Declared primary seeds are listed explicitly; the default follows the amended six-seed d10/three-seed d100 targets. Every original manifest cell is retained separately.",
                    "Covariance requires all declared seed rows; tilts are centered separately and no stationary ESS is calculated.",
                    "Initial scores are each midpoint's pre-move scores, not a shared round-start vector.",
                    "Guards c=1 and c=3 retain their original roles; c=4 was added algebraically before effects. All are output-only counterfactuals; practical-cap mixing/regret remains unproved.",
                    "Final counters and process peaks concern whole attempts and are not shorter-endpoint metrics.",
                    "Progress-only records have no recoverable trajectory; missing endpoint values are never inferred.",
                    "Current code hashes supplement but do not retrospectively constitute a pre-run provenance record."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT, help="campaign directory containing manifest.json")
    parser.add_argument("--output", type=Path, help="separate output directory; defaults to campaign directory")
    parser.add_argument("--d100-repetitions", type=int, choices=(3, 6), default=3,
                        help="amended primary target is 3; select 6 to reproduce original primary denominators")
    args = parser.parse_args()
    report = analyze(args.root.resolve(), args.d100_repetitions)
    destination = (args.output or args.root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_markdown(report, destination / "analysis.md")
    print(json.dumps(dict(planned=report["planned_runs"], statuses=report["status_counts"],
                          invalid_runs=sum(bool(r["errors"]) for r in report["runs"]),
                          output=str(destination))))


if __name__ == "__main__":
    main()
