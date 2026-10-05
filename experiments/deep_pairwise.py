"""Descriptive paired-prefix summaries; no workers or comparator fitting.

Run after the measured campaigns with ``python -m experiments.deep_pairwise``.
Missing/zero-round runs remain explicit. All differences mean first minus
second; a negative loss difference favors the first configuration. The same
prefix comparator cancels, so this is also the corresponding regret difference.
"""
import argparse
from collections import defaultdict
from itertools import combinations
import json
import math
from pathlib import Path


BASE = Path(__file__).resolve().parent
BASELINES = {"ours": "m256n1", "sigmoid": "m16n16"}
CHECKPOINTS = (100, 500, 1000)


def _identity(spec):
    return {"algorithm": spec["algorithm"], "label": spec["label"],
            "config": spec.get("config", {})}


def _run_key(spec):
    return str(Path(spec["output"]).resolve())


def _status(spec, run):
    if run is None:
        return {"run_id": Path(spec["output"]).name, "measured": False,
                "completed_rounds": 0, "status": "not_measured",
                "termination": None, "error": None,
                "sampler_seed": spec.get("sampler_seed")}
    worker, supervisor = run["worker"], run["supervisor"]
    return {"run_id": Path(spec["output"]).name, "measured": True,
            "completed_rounds": len(run["losses"]),
            "worker_reported_rounds": worker.get("round"),
            "status": worker.get("status", "unknown"),
            "termination": supervisor.get("termination"),
            "error": worker.get("error"),
            "sampler_seed": spec.get("sampler_seed")}


def _at_prefix(first, second, n):
    point = {"round": n, "available": False,
             "first_cumulative_loss": None, "second_cumulative_loss": None,
             "first_minus_second_cumulative_loss": None,
             "first_minus_second_average_loss": None,
             "first_online_seconds": None, "second_online_seconds": None}
    if n < 1:
        point["reason"] = "no_shared_completed_rounds"
        return point
    if (first is None or second is None or len(first["losses"]) < n
            or len(second["losses"]) < n):
        point["reason"] = "checkpoint_not_completed_by_both"
        return point
    a, b = float(first["prefix"][n - 1]), float(second["prefix"][n - 1])
    difference = a - b
    if not all(math.isfinite(v) for v in (a, b, difference)):
        point["reason"] = "nonfinite_accumulated_loss"
        return point
    point.update(available=True, first_cumulative_loss=a,
                 second_cumulative_loss=b,
                 first_minus_second_cumulative_loss=difference,
                 first_minus_second_average_loss=difference / n)
    for name, run in (("first", first), ("second", second)):
        if len(run["times"]) >= n:
            seconds = float(run["times"][n - 1])
            point[name + "_online_seconds"] = seconds if math.isfinite(seconds) else None
    return point


def _pair(first_spec, second_spec, runs, case, comparison):
    if Path(first_spec["data_path"]).resolve() != Path(second_spec["data_path"]).resolve():
        raise ValueError("Paired runs must use the same stored data")
    for field in ("case_id", "d", "B", "T", "data_seed"):
        if first_spec[field] != second_spec[field]:
            raise ValueError("Paired run metadata differ: " + field)
    first, second = runs.get(_run_key(first_spec)), runs.get(_run_key(second_spec))
    first_status, second_status = _status(first_spec, first), _status(second_spec, second)
    n = min(first_status["completed_rounds"], second_status["completed_rounds"])
    # Case ids and seeds identify replicates, not separate experimental settings.
    setting = {k: v for k, v in case.items() if k not in ("id", "data_seed")}
    return {"stage": first_spec["stage"], "case_id": first_spec["case_id"],
            "data_seed": first_spec["data_seed"], "setting": setting,
            "data_path": first_spec["data_path"], "comparison": comparison,
            "first": _identity(first_spec), "second": _identity(second_spec),
            "first_run": first_status, "second_run": second_status,
            "common": _at_prefix(first, second, n),
            "checkpoints": [_at_prefix(first, second, k) for k in CHECKPOINTS]}


def build_stage(specs, loaded_runs, case_records=None):
    """Build summaries from already-loaded completed runs, retaining missing ones."""
    runs = {_run_key(r["spec"]): r for r in loaded_runs}
    expected_keys = [_run_key(s) for s in specs]
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError("Duplicate manifest output path")
    if len(runs) != len(loaded_runs) or set(runs) - set(expected_keys):
        raise ValueError("Duplicate or unexpected loaded run")
    cases = {c["id"]: c for c in (case_records or [])}
    groups = defaultdict(list)
    for spec in specs:
        groups[spec["case_id"]].append(spec)
    pairwise, baseline_pairs = [], []
    for case_id, entries in sorted(groups.items()):
        entries = sorted(entries, key=lambda s: (s["algorithm"], s["label"]))
        identities = [(s["algorithm"], s["label"]) for s in entries]
        if len(identities) != len(set(identities)):
            raise ValueError("Repeated algorithm/label within case: " + case_id)
        case = cases.get(case_id, {k: entries[0][k] for k in ("d", "B", "T")})
        for first, second in combinations(entries, 2):
            pairwise.append(_pair(first, second, runs, case, "unordered_pair"))
        if entries[0]["stage"] == "deep_budget":
            index = {(s["algorithm"], s["label"]): s for s in entries}
            for spec in entries:
                algorithm = spec["algorithm"]
                if algorithm not in BASELINES:
                    continue
                key = (algorithm, BASELINES[algorithm])
                if key not in index:
                    raise ValueError("Budget manifest lacks original baseline: " + repr(key))
                row = _pair(spec, index[key], runs, case, "configuration_minus_original_baseline")
                row["is_baseline_self_comparison"] = spec is index[key]
                baseline_pairs.append(row)
    return {"expected_runs": len(specs), "measured_runs": len(runs),
            "complete": len(runs) == len(specs),
            "missing_run_ids": [Path(s["output"]).name for s in specs
                                if _run_key(s) not in runs],
            "pairwise": pairwise, "budget_vs_original_baseline": baseline_pairs}


def seed_summaries(rows):
    """Describe paired datasets without confidence intervals or significance claims."""
    groups = defaultdict(list)
    for row in rows:
        key = {k: row[k] for k in ("stage", "setting", "comparison", "first", "second")}
        for checkpoint, point in [("pair_common", row["common"])] + [
                (str(p["round"]), p) for p in row["checkpoints"]]:
            keyed = {**key, "checkpoint": checkpoint}
            groups[json.dumps(keyed, sort_keys=True)].append((row, point))
    summaries = []
    for key, observations in sorted(groups.items()):
        available = [(r, p) for r, p in observations if p["available"]]
        differences = [p["first_minus_second_cumulative_loss"] for _, p in available]
        normalized = [p["first_minus_second_average_loss"] for _, p in available]
        rounds = [p["round"] for _, p in available]
        mean = lambda values: math.fsum(v / len(values) for v in values) if values else None
        summaries.append({**json.loads(key), "paired_datasets_expected": len(observations),
                          "paired_datasets_available": len(available),
                          "data_seeds": [r["data_seed"] for r, _ in observations],
                          "available_data_seeds": [r["data_seed"] for r, _ in available],
                          "common_rounds_by_available_dataset": rounds,
                          "same_round_across_available_datasets": len(set(rounds)) <= 1 if rounds else None,
                          "mean_cumulative_loss_difference": mean(differences),
                          "minimum_cumulative_loss_difference": min(differences) if differences else None,
                          "maximum_cumulative_loss_difference": max(differences) if differences else None,
                          "mean_per_round_loss_difference": mean(normalized),
                          "dataset_values": [{"case_id": r["case_id"], "data_seed": r["data_seed"],
                                              **p} for r, p in observations]})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=("deep_scale", "deep_budget"),
                        default=["deep_scale", "deep_budget"])
    parser.add_argument("--results", type=Path, default=BASE / "results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    # Imported only when loading stored campaign outputs; it does not fit comparators.
    from .analyze import load_runs
    stages = {}
    for stage in args.stages:
        folder = args.results / stage
        if not (folder / "manifest.json").exists():
            stages[stage] = {"complete": False, "status": "manifest_missing",
                             "pairwise": [], "budget_vs_original_baseline": []}
            continue
        specs = json.loads((folder / "manifest.json").read_text())
        cases = json.loads((folder / "cases.json").read_text()) if (folder / "cases.json").exists() else []
        stages[stage] = build_stage(specs, load_runs(folder), cases)
    rows = [r for stage in stages.values() for r in stage["pairwise"]]
    baselines = [r for stage in stages.values() for r in stage["budget_vs_original_baseline"]]
    report = {"schema_version": 1, "complete": all(s["complete"] for s in stages.values()),
              "interpretation": {
                  "direction": "First minus second; negative favors the first configuration.",
                  "regret": "At the identical fixed-data prefix, the common comparator cancels exactly.",
                  "numerics": "Differences use recorded float64 cumulative losses; no optimization or statistical estimate is substituted.",
                  "zeros": "Missing or zero-round runs have unavailable comparisons, never imputed zero loss.",
                  "seeds": "Descriptive paired datasets only; no confidence intervals or significance claims.",
                  "different_prefixes": "Pair-common summaries may average different attained horizons across seeds; use fixed checkpoints for identical horizons.",
                  "time": "Cumulative measured online time at the prefix, including initialization; not full process wall time.",
                  "baseline": BASELINES},
              "stages": stages, "paired_seed_summaries": seed_summaries(rows),
              "budget_baseline_seed_summaries": seed_summaries(baselines)}
    output = args.output or args.results / "deep_pairwise.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(output), "complete": report["complete"],
                      "pairwise_rows": len(rows), "baseline_rows": len(baselines)}))


if __name__ == "__main__":
    main()
