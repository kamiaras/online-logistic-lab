"""Analyze six initialization-only antithetic controls without altering the primary study.

Run only after measured workers finish:
    python -m experiments.shared_antithetic_analyze

Every extension cell is matched through its explicit control_output pointer.
Missing endpoints remain missing. No sampler, benchmark or comparison fit runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .shared_ablation_analyze import (
    ROOT, DEFAULT as PRIMARY, ENDPOINTS, SECONDARY_SEEDS, METRICS,
    read_json, read_run, verify_data, endpoint_for, summary, sha256, fmt,
)

DEFAULT = ROOT / "experiments/results/shared_antithetic_ablation"


def initial_pair_diagnostic(record, arrays):
    """The first sampled initial-score row precedes this cloud's first moves."""
    if arrays is None or not record["trajectory_valid"]:
        return dict(available=False, reason="no_valid_saved_trajectory")
    initial = arrays["initial_scores"]
    rows = np.flatnonzero(np.all(np.isfinite(initial), axis=1))
    if not len(rows):
        return dict(available=False, reason="no_recorded_sampled_round")
    t = int(rows[0])
    values = initial[t]
    m = len(values)
    if m % 2:
        return dict(available=False, reason="odd_score_count")
    pair_sums = values[:m // 2] + values[::-1][:m // 2]
    return dict(available=True, first_sampled_round=t + 1, m=m,
                initial_scores=values.tolist(), opposite_pair_score_sums=pair_sums.tolist(),
                maximum_absolute_pair_score_sum=float(np.max(np.abs(pair_sums))),
                rms_pair_score_sum=float(np.sqrt(np.mean(pair_sums ** 2))),
                mean_initial_score=float(np.mean(values)),
                pairwise_mean_initial_score=float(np.sum(pair_sums) / m),
                maximum_absolute_initial_score=float(np.max(np.abs(values))),
                all_pair_score_sums_exactly_zero=bool(np.all(pair_sums == 0.)),
                note="Projected initial scores only; source audit establishes coefficient-state reflection.")


def initialization_counters(record, paired):
    counters = (record.get("result") or {}).get("metrics", {}).get("counters", {})
    keys = ("prior_initializations", "prior_coordinates", "independent_prior_vectors",
            "antithetic_prior_vectors", "sampled_rounds", "proximal_steps", "recorded_tilts")
    raw = {key: counters.get(key, 0) for key in keys}
    d, m = record["spec"]["d"], record["spec"]["config"]["tilts"]
    sampled = record.get("sampled_rounds_in_trajectory", 0)
    out = dict(raw_counters=raw, count_scope="entire attempt",
               sampled_completed_predictions=sampled,
               independent_draw_vectors_from_coordinate_counter=raw["prior_coordinates"] / d,
               prior_initializations_meaning="initialized lane states, including reflected states in this old-protocol extension")
    if sampled:
        expected = dict(prior_initializations=m, prior_coordinates=d * m // (2 if paired else 1))
        if paired:
            expected.update(independent_prior_vectors=m // 2, antithetic_prior_vectors=m // 2)
        out["expected_after_one_completed_initialization"] = expected
        out["one_initialization_counter_match"] = all(raw[k] == v for k, v in expected.items())
    else:
        out["one_initialization_counter_match"] = None
    return out


def spec_differences(treatment, control):
    if control is None:
        return ["control not found in the original primary manifest"]
    differences = []
    for key in ("case", "d", "B", "T", "seed", "rounds", "seconds", "data_path"):
        if treatment.get(key) != control.get(key):
            differences.append(key)
    config = dict(treatment["config"])
    if config.pop("prior_pairing", None) != "opposite_midpoints":
        differences.append("expected opposite_midpoints prior_pairing missing")
    if config != control["config"]:
        differences.append("configuration differs beyond prior_pairing")
    if treatment["control_output"] != control["output"]:
        differences.append("control_output does not equal original output pointer")
    if (config.get("layout"), config.get("persistence"), config.get("acceptance"),
            config.get("tilts"), config.get("burn_in"), config.get("transitions")) != (
                "lanes", "warm", "full", 16, 4, 4):
        differences.append("unexpected original warm 16x4 control settings")
    return differences


def analyze(campaign, primary):
    specs = read_json(campaign / "manifest.json")
    originals = read_json(primary / "manifest.json")
    if not isinstance(specs, list) or not isinstance(originals, list):
        raise ValueError("Both extension and original primary manifests are required")
    control_by_name = {Path(s["output"]).name: s for s in originals}
    data_checks = {path: verify_data(path) for path in sorted({str(Path(s["data_path"])) for s in specs})}
    runs, comparisons = [], []
    manifest_errors = []
    if len(specs) != 6:
        manifest_errors.append(f"Expected six extension cells; found {len(specs)}")
    seen = set()
    for spec in specs:
        identity = (spec["case"], spec["seed"])
        if identity in seen or spec["seed"] not in SECONDARY_SEEDS:
            manifest_errors.append(f"Duplicate or unexpected extension seed: {identity}")
        seen.add(identity)
        treatment, arrays = read_run(spec, campaign, data_checks)
        control_spec = control_by_name.get(Path(spec["control_output"]).name)
        differences = spec_differences(spec, control_spec)
        control, control_arrays = (read_run(control_spec, primary, data_checks)
                                   if control_spec is not None else (None, None))
        treatment["initial_pair_diagnostic"] = initial_pair_diagnostic(treatment, arrays)
        treatment["initialization_counters"] = initialization_counters(treatment, True)
        if control:
            control["initial_pair_diagnostic"] = initial_pair_diagnostic(control, control_arrays)
            control["initialization_counters"] = initialization_counters(control, False)
        runs.append(dict(treatment=treatment, control=control, spec_differences=differences,
                         matched_except_initial_pairing=not differences))
        for endpoint in ENDPOINTS:
            left, right = endpoint_for(treatment, endpoint), endpoint_for(control, endpoint)
            missing = {name: point["reason"] for name, point in [("treatment", left), ("control", right)]
                       if not point["available"]}
            if differences:
                missing["matching"] = differences
            row = dict(case=spec["case"], B=spec["B"], d=spec["d"], seed=spec["seed"], endpoint=endpoint,
                       treatment=treatment["id"], control=control["id"] if control else None,
                       available=not missing, missing=missing,
                       treatment_endpoint=left, control_endpoint=right)
            if not missing:
                row["difference"] = {key: left[key] - right[key] for key in METRICS}
                row["online_time_ratio"] = left["online_seconds"] / right["online_seconds"] if right["online_seconds"] > 0 else None
            comparisons.append(row)
    cases = sorted({s["case"] for s in specs})
    groups = []
    for case in cases:
        for endpoint in ENDPOINTS:
            points = [next((row for row in comparisons if (row["case"], row["endpoint"], row["seed"]) == (case, endpoint, seed)),
                           dict(seed=seed, available=False, missing={"treatment": "not_in_extension_manifest"}))
                      for seed in SECONDARY_SEEDS]
            groups.append(dict(case=case, endpoint=endpoint,
                seeds=[dict(seed=row["seed"], available=row["available"], missing=row["missing"]) for row in points],
                differences={metric: summary([row["difference"][metric] for row in points if row["available"]], 3)
                             for metric in METRICS},
                method_means={method: {metric: summary([row[method + "_endpoint"][metric] for row in points
                    if row.get(method + "_endpoint", {}).get("available")], 3) for metric in METRICS}
                    for method in ("treatment", "control")},
                time_ratios=summary([row["online_time_ratio"] for row in points if row["available"]
                                    and row["online_time_ratio"] is not None], 3)))
    before, after = read_json(campaign / "source_hashes_before.json"), read_json(campaign / "source_hashes_after.json")
    return dict(generated_utc=datetime.now(timezone.utc).isoformat(), expected_new_runs=6,
                original_primary_manifest_cell_count=len(originals), extension_manifest_cell_count=len(specs),
                manifest_errors=manifest_errors, data_checks=data_checks,
                treatment_status_counts=dict(Counter(row["treatment"]["status"] for row in runs)),
                control_status_counts=dict(Counter(row["control"]["status"] if row["control"] else "not_in_manifest" for row in runs)),
                runs=runs, endpoint_comparisons=comparisons, grouped_comparisons=groups,
                provenance=dict(recorded_before=before, recorded_after=after,
                    before_after_match=before == after if before is not None and after is not None else None,
                    extension_manifest_sha256=sha256(campaign / "manifest.json"),
                    original_manifest_sha256=sha256(primary / "manifest.json"),
                    analyzer_sha256=sha256(Path(__file__)),
                    helper_sha256=sha256(ROOT / "experiments/shared_ablation_analyze.py")),
                interpretation=[
                    "Six new treatment cells are separate from the unchanged original 75-cell study.",
                    "Paired differences are antithetic initialization minus the explicitly linked independent warm control.",
                    "All comparisons use fixed 100/250/500 endpoints; missing cells are retained, not replaced by common prefixes.",
                    "Every dataset-level denominator is three predeclared sampler seeds; incomplete means are descriptive only.",
                    "Initial cancellation uses the first non-NaN pre-move score row, not later warmed states or a selected round.",
                    "The manipulated initialization also reduces independent prior draws from sixteen to eight; full work remains charged.",
                    "Old-protocol extension counters use prior_initializations for sixteen lane states; prior_coordinates counts eight actual draw vectors.",
                    "The new prescribed PersistentLogOdds uses different population, warm mixing and raw history; it is not this measured control.",
                    "Original factor-one and factor-three guards are retained; factor four was added algebraically before effect analysis. All are output-only postprocessing on fixed data.",
                    "Extension runs follow the primary campaign rather than being interleaved with it; timing differences may include temporal system variation.",
                    "No covariance/ESS or posterior bias is inferred from three paired-control repetitions."])


def write_markdown(report, destination):
    lines = ["# Initialization-only antithetic controls", "", f"Generated {report['generated_utc']}.", "",
             f"The extension has {report['extension_manifest_cell_count']} planned cells; the original manifest retains "
             f"{report['original_primary_manifest_cell_count']} cells. Treatment statuses: {report['treatment_status_counts']}. "
             f"Linked control statuses: {report['control_status_counts']}.", "",
             "Each treatment is the old persistent 16-lane/four-move full-test control with eight prior draws and eight opposite partners. "
             "The new prescribed reference is a separate implementation. Differences below are antithetic minus independent initialization; "
             "negative loss differences favor the antithetic treatment. Guard c uses log(c t(t+1)). The original c=1 and c=3 "
             "analyses are retained; c=4 was added by an algebraic allocation amendment before effect analysis.", "",
             "## Fixed-endpoint means and paired effects", "",
             "All denominators are the same three predeclared sampler seeds on each fixed dataset. The table displays available means with "
             "their denominators; incomplete means can be runtime selected and are descriptive. The JSON leaves complete_mean null when a seed is missing.", "",
             "| Dataset | Round | Paired n/3 | Independent raw loss/round (n/3) | Antithetic raw loss/round (n/3) | Raw total difference | c=1 total difference | c=3 total difference | c=4 total difference | Mean online-time ratio |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in report["grouped_comparisons"]:
        changes, methods = group["differences"], group["method_means"]
        method_values = []
        for method in ("control", "treatment"):
            stats = methods[method]["loss"]
            value = stats["available_mean"]
            method_values.append(f"{fmt(value / group['endpoint'] if value is not None else None)} ({stats['n']}/3)")
        lines.append(f"| {group['case']} | {group['endpoint']} | {changes['loss']['n']}/3 | {method_values[0]} | {method_values[1]} | {fmt(changes['loss']['available_mean'])} | {fmt(changes['guard_c1_loss']['available_mean'])} | {fmt(changes['guard_c3_loss']['available_mean'])} | {fmt(changes['guard_c4_loss']['available_mean'])} | {fmt(group['time_ratios']['available_mean'])} |")
    lines += ["", "## Every paired seed", "",
              "| Dataset | Seed | Round | Status | Raw total difference | c=1 difference | c=3 difference | c=4 difference | Online seconds difference |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for row in report["endpoint_comparisons"]:
        values = row.get("difference", {})
        status = "available" if row["available"] else json.dumps(row["missing"], ensure_ascii=False)
        lines.append(f"| {row['case']} | {row['seed']} | {row['endpoint']} | {status} | {fmt(values.get('loss'))} | {fmt(values.get('guard_c1_loss'))} | {fmt(values.get('guard_c3_loss'))} | {fmt(values.get('guard_c4_loss'))} | {fmt(values.get('online_seconds'))} |")
    lines += ["", "## First prior initialization", "",
              "Pair sums are taken from the first sampled pre-move score row. This checks cancellation of the observed feature projection; "
              "source inspection establishes how coefficient vectors were reflected. Sparse absent event counters are recorded as zero.", "",
              "| Dataset | Seed | Method | First sampled round | Max absolute initial pair sum | Pairwise initial mean | Prior draw vectors | Initialized lane states |",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    for entry in report["runs"]:
        for method in ("control", "treatment"):
            run = entry[method]
            if run is None:
                continue
            diagnostic, counter = run["initial_pair_diagnostic"], run["initialization_counters"]
            lines.append(f"| {run['spec']['case']} | {run['spec']['seed']} | {method} | {diagnostic.get('first_sampled_round', 'unavailable')} | {fmt(diagnostic.get('maximum_absolute_pair_score_sum'))} | {fmt(diagnostic.get('pairwise_mean_initial_score'))} | {fmt(counter['independent_draw_vectors_from_coordinate_counter'])} | {counter['raw_counters']['prior_initializations']} |")
    lines += ["", "All guard counterfactuals preserve the recorded sampler states on fixed data. Through n rounds, their one-sided loss "
              "increase bounds are n/[c(n+1)]; they do not establish low regret for arbitrary practical caps. No missing loss or score is inferred "
              "from progress-only output. Final counters and memory describe whole attempts, while endpoint times come from completed trajectory rows.", "",
              "The six new runs are sequential after the primary campaign, not interleaved timing repeats. One worker, one BLAS thread and "
              "a fixed memory cap control major resource differences, but temporal system variation can still affect timing. Three sampler seeds "
              "on a fixed dataset do not justify population-data inference, stationary ESS, or an estimate of posterior bias.", "",
              f"Manifest/matching errors: {len(report['manifest_errors']) + sum(bool(row['spec_differences']) for row in report['runs'])}. "
              "Full raw/guarded endpoint values, validation, counters, seed differences and source hashes are in `analysis.json`."]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--primary-root", type=Path, default=PRIMARY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.root.resolve(), args.primary_root.resolve())
    output = (args.output or args.root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_markdown(report, output / "analysis.md")
    print(json.dumps(dict(planned=report["extension_manifest_cell_count"], statuses=report["treatment_status_counts"],
                          manifest_errors=report["manifest_errors"], output=str(output))))


if __name__ == "__main__":
    main()
