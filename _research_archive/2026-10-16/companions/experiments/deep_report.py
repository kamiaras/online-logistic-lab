"""Report completed deeper experiments without fitting or launching workers.

Run only after all measurements, ordinary analyses, paired summaries, and the
offline long-comparator refinement: python -m experiments.deep_report
Numerical/plotting libraries are imported only inside the figure writer.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
STAGES = ("deep_scale", "deep_budget", "deep_long")
HISTORY_MIB = (1536, 1856, 1920, 1968, 2000, 2048)
MIB = 1024**2
CAP = 2 * 1024**3
NAMES = {"ours": "Proximal log-odds", "sigmoid": "Sigmoid/MALA", "aioli": "AIOLI"}
COLORS = {"ours": "#b53b33", "sigmoid": "#2868ae", "aioli": "#278155"}
MARKERS = {"completed": "o", "memory_failure": "X", "time_censored": "^", "other_failure": "s"}


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Required completed artifact is unavailable: {path}. Finish its producing stage first.") from exc


def _key(row):
    return tuple(row[k] for k in ("case_id", "algorithm", "label", "sampler_seed"))


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value, digits=3):
    return f"{value:,.{digits}f}" if _finite(value) else "—"


def integer(value):
    return f"{int(value):,}" if _finite(value) else "—"


def precise(value):
    return f"{value:.6g}" if _finite(value) else "—"


def mean(values):
    values = [v for v in values if _finite(v)]
    return math.fsum(v / len(values) for v in values) if values else None


def span(values):
    values = [v for v in values if _finite(v)]
    if not values:
        return "—"
    return integer(min(values)) if min(values) == max(values) else f"{integer(min(values))}–{integer(max(values))}"


def interval(points):
    valid = [p for p in points if p and _finite(p.get("regret_lower")) and _finite(p.get("regret_upper"))]
    if not valid:
        return "—"
    return f"[{fmt(mean(p['regret_lower'] for p in valid))}, {fmt(mean(p['regret_upper'] for p in valid))}]"


def link(path, label=None):
    path = Path(path).resolve()
    try:
        target = path.relative_to(ROOT).as_posix()
    except ValueError:
        target = path.as_posix()
    return f"[{label or path.name}](<{target}>)"


def outcome(status, termination=None, error=None):
    if status in ("memory_error", "memory_failure") or termination == "memory_limit" or "MemoryError" in str(error):
        return "memory_failure"
    if status in ("time_budget", "external_timeout", "time_censored") or termination == "external_timeout":
        return "time_censored"
    if status in ("completed", "evaluation_completed"):
        return "completed"
    return "other_failure"


def _phase(path):
    from .memory_frontier import failure_phase
    return failure_phase(path.read_text(encoding="utf-8")) if path.exists() else "unknown"


def _no_measured_workers():
    import psutil
    blocked = {"experiments.worker", "experiments.runner", "experiments.memory_frontier", "experiments.history_frontier"}
    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if blocked.intersection(proc.info["cmdline"] or []):
                found.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if found:
        raise RuntimeError(f"Measured experiment processes are active ({found}); defer report generation until they stop.")


def load_artifacts(results):
    """Validate complete artifacts; preserve failed runs missing from summaries."""
    results = Path(results).resolve()
    bundle = {"results": results, "stages": {}, "sources": []}
    for stage in STAGES:
        folder = results / stage
        paths = [folder / n for n in ("manifest.json", "summary.json", "cases.json", "environment.json")]
        specs, summaries, cases, environment = [read_json(p) for p in paths]
        if not isinstance(specs, list) or not specs:
            raise RuntimeError(f"Empty or malformed manifest: {paths[0]}")
        index = {_key(r): r for r in summaries}
        if len(index) != len(summaries) or len({_key(s) for s in specs}) != len(specs):
            raise RuntimeError(f"Duplicate run identity in {stage}")
        if set(index) - {_key(s) for s in specs}:
            raise RuntimeError(f"Summary has unexpected runs in {stage}")
        case_index = {c["id"]: c for c in cases}
        rows, data_meta = [], {}
        for spec in specs:
            out = Path(spec["output"])
            sup = read_json(out / "supervisor.json")
            worker = sup.get("worker", {})
            if worker.get("status") in (None, "running"):
                raise RuntimeError(f"Run has no terminal status: {out}")
            if spec.get("memory_bytes") != CAP or not sup.get("os_hard_limit_active"):
                raise RuntimeError(f"Expected an active 2 GiB worker limit: {out}")
            if read_json(out / "spec.json") != spec:
                raise RuntimeError(f"Manifest and measured specification disagree: {out}")
            n = int(worker.get("round", 0))
            losses = out / "losses.bin"
            available = losses.stat().st_size // 8 if losses.exists() else 0
            if n < 0 or n > spec["T"] or available < n:
                raise RuntimeError(f"Recorded completion/loss length is inconsistent: {out}")
            row = dict(index.get(_key(spec), {}))
            if not row and n:
                raise RuntimeError(f"Positive-round run missing from {stage}/summary.json; rerun analysis: {out}")
            if row and row.get("completed_rounds") != n:
                raise RuntimeError(f"Stale summary completion count: {out}")
            # analyze skips a case when every run has zero losses. Its manifest
            # and terminal supervisors still contribute to every denominator.
            if not row:
                row.update(spec, completed_rounds=0, common_round=0,
                           final=None, common=None, synthesized_zero_round=True)
            row.update(_spec=spec, _supervisor=sup, _worker=worker, _output=out,
                       status=worker.get("status"), error=worker.get("error"),
                       termination=sup.get("termination"),
                       outcome=outcome(worker.get("status"), sup.get("termination"), worker.get("error")),
                       phase=_phase(out / "traceback.txt"),
                       online_seconds=worker.get("online_seconds"),
                       process_wall_seconds=sup.get("process_wall_seconds"),
                       state_bytes=worker.get("state_bytes"),
                       peak_private_commit=max(worker.get("memory", {}).get("peak_private_commit", 0),
                                               sup.get("monitored_peak_private", 0)))
            row.update(family=case_index[spec["case_id"]]["family"],
                       B_generator=case_index[spec["case_id"]].get("B_generator", 3.))
            rows.append(row)
            if spec["case_id"] not in data_meta:
                data_meta[spec["case_id"]] = read_json(Path(spec["data_path"]) / "data.json")
        bundle["stages"][stage] = dict(rows=rows, specs=specs, cases=cases,
                                       environment=environment, data_meta=data_meta)
        bundle["sources"].extend(paths)
    pair_path = results / "deep_pairwise.json"
    bundle["pairwise"] = read_json(pair_path)
    if not bundle["pairwise"].get("complete"):
        raise RuntimeError("deep_pairwise.json is incomplete; rerun it after every scale/budget worker finishes.")
    for stage in ("deep_scale", "deep_budget"):
        p = bundle["pairwise"].get("stages", {}).get(stage, {})
        if not p.get("complete") or p.get("measured_runs") != len(bundle["stages"][stage]["rows"]):
            raise RuntimeError(f"Paired summary is missing or stale for {stage}")
        by_id = {r['_output'].name: r for r in bundle['stages'][stage]['rows']}
        for pair in p["pairwise"]:
            for name in ("first_run", "second_run"):
                pr = pair[name]
                if pr["completed_rounds"] != by_id[pr["run_id"]]["completed_rounds"]:
                    raise RuntimeError(f"Stale paired completion count: {pr['run_id']}")
    precision_path = results / "deep_precision_audit.json"
    bundle["precision"] = read_json(precision_path)
    precision = bundle["precision"]
    if (precision.get("decimal_precision") != 80 or not isinstance(precision.get("runs"), list)
            or not isinstance(precision.get("scope"), str) or not isinstance(precision.get("interpretation"), str)):
        raise RuntimeError("deep_precision_audit.json lacks the expected 80-digit per-run audit schema.")
    eligible = {r["case_id"]: r for r in bundle["stages"]["deep_scale"]["rows"]
                if r["algorithm"] == "aioli" and r["d"] <= 100 and r["B"] >= 10000 and r["status"] == "completed"}
    audited = {r["case_id"]: r for r in precision["runs"]}
    if len(audited) != len(precision["runs"]) or set(audited) != set(eligible):
        raise RuntimeError("Precision audit is missing, duplicates, or adds eligible completed large-B AIOLI runs; rerun deep_precision.")
    for cid, p in audited.items():
        r = eligible[cid]
        if p["d"] != r["d"] or p["B"] != r["B"] or p["rounds"] != r["completed_rounds"]:
            raise RuntimeError("Precision audit has stale run parameters or completion count: " + cid)
        if not p.get("exact_replay_of_recorded_losses") or p.get("max_loss_difference") != 0:
            raise RuntimeError("Precision audit did not reproduce the measured loss sequence exactly: " + cid)
        for key in ("max_absolute_scalar_residual", "sum_absolute_scalar_residuals", "elapsed_seconds"):
            if not _finite(p.get(key)) or p[key] < 0:
                raise RuntimeError("Invalid precision-audit diagnostic: " + cid + "/" + key)
        counters = r["_worker"].get("metrics", {}).get("counters", {})
        for key in ("root_rounding_stalls", "curvature_underflows"):
            if p.get(key) != counters.get(key, 0):
                raise RuntimeError("Precision replay counters disagree with measured run: " + cid + "/" + key)
    bundle["precision_measured_runs"] = eligible
    bundle["sources"].append(precision_path)
    frontier = results / "memory_frontier_deep"
    bundle["frontier"] = read_json(frontier / "frontier_summary.json")
    bundle["frontier_policy"] = read_json(frontier / "policy.json")
    bundle["frontier_specs"] = read_json(frontier / "manifest.json")
    if bundle["frontier"].get("status") in ("running", "reclassified_pending_resume"):
        raise RuntimeError("Memory frontier campaign is running or has reclassified probes pending resume; finish it before reporting.")
    records_path = frontier / "frontier_records.jsonl"
    if not records_path.exists():
        raise RuntimeError(f"Missing required memory-frontier records: {records_path}")
    bundle["frontier_records"] = [json.loads(s) for s in records_path.read_text().splitlines() if s.strip()]
    classification_audit = frontier / "classification_audit.json"
    if classification_audit.exists():
        bundle["frontier_classification_audit"] = read_json(classification_audit)
        if bundle["frontier_classification_audit"].get("status") != "completed":
            raise RuntimeError("Memory-frontier classification repair is incomplete.")
        bundle["sources"].append(classification_audit)
    if len(bundle["frontier_records"]) != bundle["frontier"].get("attempts"):
        raise RuntimeError("Memory frontier records and summary have inconsistent attempt counts.")
    if any(r.get("memory_limit_bytes") != CAP or not r.get("os_hard_limit_active")
           for r in bundle["frontier_records"]):
        raise RuntimeError("A memory-frontier record lacks the active 2 GiB limit.")
    history = []
    for mib in HISTORY_MIB:
        for algorithm in ("ours", "sigmoid"):
            out = results / "history_frontier_deep" / f"{algorithm}_{mib}"
            sup, spec = read_json(out / "supervisor.json"), read_json(out / "spec.json")
            worker = sup.get("worker", {})
            if not sup.get("os_hard_limit_active") or sup.get("hard_process_commit_limit") != CAP:
                raise RuntimeError(f"History probe lacks active 2 GiB isolation: {out}")
            if worker.get("executed_online_rounds", 0) != 0:
                raise RuntimeError(f"History preload was unexpectedly labeled an online trajectory: {out}")
            if sup.get("checkpoint_is_censored"):
                state = outcome(worker.get("status"), sup.get("termination"))
            elif worker.get("append_status") == "memory_error" or worker.get("predict_status") == "memory_error":
                state = "memory_failure"
            elif worker.get("append_status") == "time_budget" or worker.get("predict_status") == "skipped_time_budget":
                state = "time_censored"
            else:
                state = outcome(worker.get("status"), sup.get("termination"), worker.get("append_error"))
            history.append(dict(algorithm=algorithm, requested_mib=mib, spec=spec,
                                worker=worker, supervisor=sup, output=out, outcome=state,
                                peak_private_commit=max(worker.get("memory", {}).get("peak_private_commit", 0),
                                                        sup.get("monitored_peak_private", 0))))
    bundle["history"] = history
    bundle["long_curves"] = read_json(results / "deep_long" / "curves.json")
    audit_path = results / "deep_long" / "newton_refinement_audit.json"
    bundle["newton"] = read_json(audit_path)
    # A refined cache must have been propagated by the ordinary analyzer.
    for row in bundle["stages"]["deep_long"]["rows"]:
        if row.get("final"):
            cache = read_json(results / "deep_long" / (row["case_id"] + "__comparators.json"))
            fit = cache[str(row["completed_rounds"])]
            if not math.isclose(fit["loss"], row["final"]["comparator_loss"], rel_tol=1e-12, abs_tol=1e-9):
                raise RuntimeError("Long-run comparator cache is newer than summary.json; propagate the refined analysis first.")
            if not math.isclose(fit["gap"], row["final"]["comparator_first_order_gap"], rel_tol=1e-12, abs_tol=1e-9):
                raise RuntimeError("Long-run summary has a stale comparator gap; propagate the refined analysis first.")
    bundle["sources"].extend([pair_path, frontier / "frontier_summary.json", records_path,
                              frontier / "policy.json", frontier / "manifest.json", audit_path,
                              results / "deep_long" / "curves.json"])
    bundle["sources"].extend(r["output"] / "supervisor.json" for r in history)
    bundle["sources"].extend(r["output"] / "spec.json" for r in history)
    return bundle


def _groups(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["d"], r["B"], r["T"], r["algorithm"], r["label"])].append(r)
    return sorted(groups.items())


def _table(lines, headers, rows):
    lines.extend(["", "| " + " | ".join(headers) + " |",
                  "| " + " | ".join("---" for _ in headers) + " |"])
    lines.extend("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in row) + " |" for row in rows)
    lines.append("")


def _paired_table_rows(bundle, budget=False):
    source = bundle["pairwise"]["budget_baseline_seed_summaries" if budget else "paired_seed_summaries"]
    selected = [s for s in source if (budget or s["stage"] == "deep_scale")
                and s["first"]["algorithm"] == (s["second"]["algorithm"] if budget else "ours")
                and (budget or s["second"]["algorithm"] == "sigmoid")]
    if budget:
        selected = [s for s in selected if s["first"]["label"] != s["second"]["label"]]
    groups = defaultdict(dict)
    for row in selected:
        if row["checkpoint"] in ("100", "pair_common"):
            k = json.dumps({n: row[n] for n in ("setting", "first", "second")}, sort_keys=True)
            groups[k][row["checkpoint"]] = row
    result = []
    for _, pair in sorted(groups.items()):
        common, fixed = pair.get("pair_common"), pair.get("100")
        seed = common or fixed
        setting = seed["setting"]
        shared = span(common["common_rounds_by_available_dataset"]) if common else "—"
        avail = lambda r: f"{r['paired_datasets_available']}/{r['paired_datasets_expected']}" if r else "0/0"
        method = f"{NAMES[seed['first']['algorithm']]} {seed['first']['label']}"
        label = (method + " − " + seed["second"]["label"]) if budget else f"{integer(setting['d'])}; {setting['B']:g}; {integer(setting['T'])}"
        result.append([label, avail(fixed), fmt(fixed.get("mean_cumulative_loss_difference") if fixed else None),
                       avail(common), shared,
                       fmt(common.get("mean_per_round_loss_difference") if common else None, 5)])
    return result


def _fixed_pair(bundle, first, second, checkpoint=100):
    """Normalize a named pair's stored checkpoint to first minus second."""
    points, expected = [], 0
    for pair in bundle["pairwise"]["stages"]["deep_budget"]["pairwise"]:
        a = (pair["first"]["algorithm"], pair["first"]["label"])
        b = (pair["second"]["algorithm"], pair["second"]["label"])
        if {a, b} != {first, second}:
            continue
        expected += 1
        p = next((p for p in pair["checkpoints"] if p["round"] == checkpoint), None)
        if not p or not p["available"]:
            continue
        f, s = ("first", "second") if a == first else ("second", "first")
        points.append(dict(data_seed=pair["data_seed"],
                           first_loss=p[f+"_cumulative_loss"]/checkpoint,
                           second_loss=p[s+"_cumulative_loss"]/checkpoint,
                           first_seconds=p[f+"_online_seconds"], second_seconds=p[s+"_online_seconds"]))
    result = {"expected": expected, "available": len(points), "points": points,
              **{key: mean(p[key] for p in points) for key in
                 ("first_loss", "second_loss", "first_seconds", "second_seconds")}}
    if not all(_finite(p[key]) for p in points for key in ("first_seconds", "second_seconds")):
        # Do not silently report a timing mean over fewer datasets than the
        # loss denominator printed with this paired comparison.
        result["first_seconds"] = result["second_seconds"] = None
    return result


def _throughput_finding(bundle):
    """Describe observed matched-prefix timing reversals; do not fit a threshold."""
    groups = defaultdict(list)
    for pair in bundle["pairwise"]["stages"]["deep_scale"]["pairwise"]:
        algorithms = (pair["first"]["algorithm"], pair["second"]["algorithm"])
        if "aioli" not in algorithms:
            continue
        other = algorithms[1] if algorithms[0] == "aioli" else algorithms[0]
        s = pair["setting"]
        key = (other, s["B"], s.get("family"), s.get("B_generator"), s["d"], s["T"])
        groups[key].append(pair)
    settings = defaultdict(list)
    for key, pairs in sorted(groups.items()):
        ratios, rounds = [], []
        for pair in pairs:
            p = pair["common"]
            if not p["available"]:
                continue
            a, b = p["first_online_seconds"], p["second_online_seconds"]
            if not _finite(a) or not _finite(b) or min(a, b) <= 0:
                continue
            ratios.append(a/b if pair["first"]["algorithm"] == "aioli" else b/a)
            rounds.append(p["round"])
        if len(ratios) == len(pairs) and ratios:
            settings[key[:4]].append(dict(d=key[4], T=key[5], ratio=mean(ratios), rounds=rounds, pairs=len(pairs)))
    findings = []
    for key, values in sorted(settings.items()):
        values.sort(key=lambda r: (r["d"], r["T"]))
        for lower, higher in zip(values, values[1:]):
            if lower["d"] < higher["d"] and lower["ratio"] < 1 < higher["ratio"]:
                findings.append(f"Against {NAMES[key[0]]} at B={key[1]:g}, the mean AIOLI/sampler "
                                f"online-time ratio at each dataset's shared prefix changed from {fmt(lower['ratio'], 2)} "
                                f"(d={integer(lower['d'])}, T={integer(lower['T'])}, prefixes {span(lower['rounds'])}) "
                                f"to {fmt(higher['ratio'], 2)} (d={integer(higher['d'])}, T={integer(higher['T'])}, "
                                f"prefixes {span(higher['rounds'])}); {lower['pairs']}/{higher['pairs']} paired datasets respectively.")
                break
    if findings:
        return "Throughput order reversed in the saved grid: " + " ".join(findings[:2]) + " Horizons and attained prefixes differ; this is not an isolated causal dimension effect or a fitted crossover threshold."
    return "The saved fully paired common-prefix timings do not show a lower-to-higher-dimension reversal from AIOLI faster to AIOLI slower. Completion counts and each measured setting remain in the tables below."


def key_findings(bundle):
    findings = []
    for r in bundle["stages"]["deep_long"]["rows"]:
        findings.append(f"AIOLI recorded {integer(r['completed_rounds'])}/{integer(r['T'])} configured rounds "
                        f"at d={integer(r['d'])}, B={r['B']:g}: {fmt(r['online_seconds'])} online seconds, "
                        f"{fmt(r['peak_private_commit']/MIB, 2)} MiB peak private commit, "
                        f"final numerical regret interval {interval([r.get('final')])}; status `{r['status']}`. "
                        "This is one actual trajectory, with no extrapolation to uncompleted rounds.")
    sampler_pairs = [p for p in bundle["pairwise"]["stages"]["deep_scale"]["pairwise"]
                     if {p["first"]["algorithm"], p["second"]["algorithm"]} == {"ours", "sigmoid"}]
    differences, prefix_rounds = [], []
    seeds_by_setting = defaultdict(set)
    for pair in sampler_pairs:
        setting = pair["setting"]
        seeds_by_setting[(setting["d"], setting["B"], setting["T"])].add(pair["data_seed"])
        point = pair["common"]
        if not point["available"]:
            continue
        delta = point["first_minus_second_cumulative_loss"]
        if not _finite(delta):
            raise RuntimeError("Available sampler comparison has a nonfinite loss difference: " + pair["case_id"])
        differences.append(delta if pair["first"]["algorithm"] == "ours" else -delta)
        prefix_rounds.append(point["round"])
    seed_note = ("two data seeds per setting" if seeds_by_setting and all(len(v) == 2 for v in seeds_by_setting.values())
                 else "the per-setting data-seed counts in the settings table")
    findings.append(f"Across {len(differences)}/{len(sampler_pairs)} available/expected deep-scale datasets "
                    f"({len(seeds_by_setting)} settings; {seed_note}), sigmoid/MALA had lower cumulative loss "
                    f"than proximal log-odds in {sum(d > 0 for d in differences)}/{len(differences)} available pairs; "
                    f"proximal had lower loss in {sum(d < 0 for d in differences)}/{len(differences)}, "
                    f"with {sum(d == 0 for d in differences)} exact recorded ties and {len(sampler_pairs)-len(differences)} unavailable pairs. "
                    f"Each comparison uses that pair's common completed prefix (range {span(prefix_rounds)} rounds), "
                    "so its comparator cancels and the regret difference is identical. These are time-selected prefixes, "
                    "not a common fixed horizon across settings or a significance test.")
    findings.append(_throughput_finding(bundle))
    configs = sorted({(s["config"].get("particles"), s["label"]) for s in bundle["stages"]["deep_budget"]["specs"]
                      if s["algorithm"] == "ours" and s["config"].get("transitions") == 1
                      and s["config"].get("particles", 0) > 256}, reverse=True)
    cloud_choice = None
    for particles, label in configs:
        p = _fixed_pair(bundle, ("ours", label), ("ours", "m256n1"))
        if p["available"]:
            cloud_choice = label
            direction = "reduced" if p["first_loss"] < p["second_loss"] else "increased" if p["first_loss"] > p["second_loss"] else "left unchanged"
            text = (f"At fixed round 100 (d=100, B=1000, configured T=2000), increasing proximal particles "
                    f"from 256 to {particles} with one transition {direction} mean loss/round "
                    f"from {fmt(p['second_loss'], 6)} to {fmt(p['first_loss'], 6)}; mean prefix online time "
                    f"changed from {fmt(p['second_seconds'], 4)} to {fmt(p['first_seconds'], 4)} s "
                    f"({p['available']}/{p['expected']} paired datasets).")
            q = _fixed_pair(bundle, ("ours", label), ("sigmoid", "m16n16"))
            if q["available"]:
                relation = "higher" if q["first_loss"] > q["second_loss"] else "lower" if q["first_loss"] < q["second_loss"] else "equal"
                text += (f" Its loss/round was {relation} than selected sigmoid/MALA m16n16: "
                         f"{fmt(q['first_loss'], 6)} versus {fmt(q['second_loss'], 6)} "
                         f"({q['available']}/{q['expected']} paired datasets at round 100).")
            findings.append(text)
            break
    if cloud_choice is None:
        findings.append("No larger proximal cloud has an available fixed-100 paired comparison with m256n1; no particle-count benefit is inferred from different stopping horizons.")
    p = _fixed_pair(bundle, ("ours", "m256n16"), ("ours", "m256n1"))
    if p["available"]:
        findings.append(f"With 256 proximal particles, raising transitions from 1 to 16 changed mean loss/round "
                        f"at 100 from {fmt(p['second_loss'], 6)} to {fmt(p['first_loss'], 6)}, and mean online time "
                        f"from {fmt(p['second_seconds'], 4)} to {fmt(p['first_seconds'], 4)} s "
                        f"({p['available']}/{p['expected']} paired datasets). These two-seed budget observations are descriptive, not a significance test.")
    audited = bundle["precision"]["runs"]
    if audited:
        findings.append(f"The separate 80-digit scalar audit exactly replayed all {len(audited)} eligible completed "
                        f"large-B AIOLI loss sequences. The largest reported scalar residual was "
                        f"{precise(max(p['max_absolute_scalar_residual'] for p in audited))}. "
                        "This diagnostic freezes represented scalar statistics and does not certify matrix rounding or a floating-point regret guarantee.")
    return findings


def _precision_section(lines, bundle):
    lines.extend(["## Large-B numerical diagnostics and scalar precision audit", "",
                  "The following first table uses each original measured AIOLI run's own counters and diagnostics. "
                  "Unavailable failed-constructor diagnostics are not replaced by zero. A rounding stall counts a scalar bisection "
                  "that could no longer change its float64 interval; curvature underflow counts a sigmoid curvature tail rounded to zero. "
                  "Neither count alone measures total prediction error."])
    measured = sorted([r for r in bundle["stages"]["deep_scale"]["rows"]
                       if r["algorithm"] == "aioli" and r["B"] >= 10000],
                      key=lambda r: (r["d"], r["B"], r["data_seed"]))
    rows = []
    for r in measured:
        w = r["_worker"]
        counters = w.get("metrics", {}).get("counters", {})
        diagnostics = w.get("diagnostics") or {}
        value = lambda key: counters.get(key, 0) if w.get("metrics") else None
        rows.append([integer(r["d"]), f"{r['B']:g}", r["data_seed"],
                     f"{integer(r['completed_rounds'])}/{integer(r['T'])}", r["status"],
                     integer(diagnostics.get("bisections")), integer(value("root_rounding_stalls")),
                     integer(value("curvature_underflows")), precise(diagnostics.get("max_abs_logit"))])
    _table(lines, ["d", "B", "Data seed", "Actual/configured rounds", "Status", "Prescribed K", "Root rounding stalls", "Curvature underflows", "Max absolute represented logit"], rows)
    precision = bundle["precision"]
    lines.extend([f"The separate {precision['decimal_precision']}-digit decimal audit covers exactly the completed AIOLI "
                  "deep-scale runs with d≤100 and B≥10000. It is offline evaluation, excluded from measured online times. "
                  "Each included replay must reproduce the recorded loss array exactly before this report is generated.", "",
                  precision["scope"], "", precision["interpretation"], "",
                  "In particular, the reported decimal residuals are not outward-rounded interval enclosures. "
                  "Freezing the represented q and v excludes rounding in the matrix state and propagated update errors, "
                  "so even very small residuals do not certify the entire floating-point algorithm or its regret theorem."])
    rows = []
    for p in sorted(precision["runs"], key=lambda p: (p["d"], p["B"], p["case_id"])):
        r = bundle["precision_measured_runs"][p["case_id"]]
        rows.append([integer(p["d"]), f"{p['B']:g}", r["data_seed"], integer(p["rounds"]),
                     precise(p["max_absolute_scalar_residual"]), precise(p["sum_absolute_scalar_residuals"]),
                     "yes" if p["exact_replay_of_recorded_losses"] else "no", precise(p["max_loss_difference"]),
                     fmt(p["elapsed_seconds"])])
    _table(lines, ["d", "B", "Data seed", "Audited rounds", "Max absolute scalar residual", "Sum absolute scalar residuals", "Exact loss replay", "Max replay loss difference", "Offline audit seconds"], rows)
    lines.extend(["Raw scope, per-run measurements, and replay flags: " + link(bundle["results"] / "deep_precision_audit.json")
                  + "; procedure: " + link(BASE / "deep_precision.py") + ".", ""])


def write_report(bundle, target, figure_folder):
    results = bundle["results"]
    all_rows = [r for s in STAGES for r in bundle["stages"][s]["rows"]]
    lines = ["# Deeper measured Python experiments", "",
             f"The three online stages contain {len(all_rows)} measured runs. The observations below are derived from saved terminal artifacts; "
             "separate memory-frontier and retained-history probes are counted later.", "",
             "These measurements use the original proximal log-odds implementation and schedule in SMALL_CLOUD_COMPLETE.md, "
             "with the declared practical particle/transition caps, the sigmoid/MALA implementation, and the audited AIOLI scalar solve with Cholesky updates. "
             "Later stratified/fresh-restart and fused deterministic-sweep refinements are outside this study. "
             "Practical sampler caps do not inherit the prescribed theoretical regret guarantee. "
             "Float64 predictions and comparator gaps have no rigorous floating-point certificate.", "",
             "Workers run sequentially with one BLAS thread and a 2 GiB hard process private-commit limit. "
             "Online time includes initialization, prediction, and observation updates; process wall time also includes imports, input, and recording. "
             "An online budget is checked between rounds, with a supervisor wall allowance of 20 additional seconds. "
             "Completion and time-censored outcomes therefore use the recorded status, not a time threshold inferred afterward. "
             "Memory peaks cover the whole attempted run, not just a comparison prefix. No projected runtime or extrapolated memory results appear here.", "",
             "## Exact settings and data", "",
             "The signal generator uses R=1, generating-vector norm 3, and feature projection magnitude 1/2 along its signal direction; "
             "the nominal generating logit magnitude is 1.5 in these dimensions. Thus the signal does not shrink as dimension increases. "
             "B is the comparator/prior scale, independently varied from the generating norm. Increasing B therefore tests a broader allowed model and prior, "
             "not a stronger generating signal. Labels are drawn from the stated logistic signal. "
             "These are descriptive synthetic-data probes with two data seeds per scale/budget setting and one long-run seed, not a general ranking or significance test."]
    lines[4:4] = ["- " + finding for finding in key_findings(bundle)] + ["",
                   "The timing and budget comparisons use the saved " + link(results / "deep_pairwise.json", "paired-prefix records")
                   + ". The per-pair comparison horizon and available dataset count are retained throughout.", ""]
    environment = bundle["stages"]["deep_scale"]["environment"]
    lines.extend(["", f"Measurement environment: {environment.get('platform', 'see environment.json')}; "
                  f"Python {str(environment.get('python', 'unknown')).splitlines()[0]}; "
                  f"{environment.get('logical_cpus', 'unknown')} logical CPUs; "
                  f"NumPy {environment.get('packages', {}).get('numpy', 'unknown')}, "
                  f"SciPy {environment.get('packages', {}).get('scipy', 'unknown')}. "
                  "Every numerical worker still uses one BLAS thread; machine details and available memory at each stage are retained in its environment.json."])
    settings = []
    for stage in STAGES:
        grouped = defaultdict(list)
        for spec in bundle["stages"][stage]["specs"]:
            grouped[(spec["d"], spec["B"], spec["T"], spec["seconds"])].append(spec)
        for (d, B, T, seconds), rr in sorted(grouped.items()):
            settings.append([stage, integer(d), f"{B:g}", integer(T), fmt(seconds, 0),
                             ", ".join(str(s) for s in sorted({r['data_seed'] for r in rr})),
                             ", ".join(str(s) for s in sorted({r['sampler_seed'] for r in rr}))])
    _table(lines, ["Stage", "d", "B", "Configured T", "Online cap (s)", "Data seeds", "Sampler seeds"], settings)
    configurations = []
    for stage in STAGES:
        unique = {(s["algorithm"], s["label"], json.dumps(s["config"], sort_keys=True)) for s in bundle["stages"][stage]["specs"]}
        for algorithm, label, config in sorted(unique):
            configurations.append([stage, NAMES[algorithm], label, "`" + config + "`"])
    _table(lines, ["Stage", "Method", "Label", "Exact explicit configuration (remaining defaults in linked source)"], configurations)
    lines.extend(["R=1 throughout. AIOLI uses λ=B⁻² and the horizon-dependent prescribed scalar-bisection count. "
                  "Sigmoid/MALA defaults are one bridge, lazy steps, step multiplier 0.5, and history blocks of 1024 rows. "
                  "Proximal defaults are practical mode, batches of at most 64 particles, and history blocks of 1024 rows; its actual m can be below the cap. "
                  "The full measured configurations, default values, and diagnostic schedules remain in the run specifications and source links below.",
                  "", "## Completion and failures"])
    counts = []
    for stage in STAGES:
        for algorithm in NAMES:
            rr = [r for r in bundle["stages"][stage]["rows"] if r["algorithm"] == algorithm]
            if not rr:
                continue
            cc = Counter(r["outcome"] for r in rr)
            counts.append([stage, NAMES[algorithm], len(rr), cc["completed"], cc["time_censored"],
                           cc["memory_failure"], cc["other_failure"], sum(r["completed_rounds"] == 0 for r in rr),
                           "; ".join(f"{s}: {n}" for s, n in sorted(Counter(r['status'] for r in rr).items()))])
    _table(lines, ["Stage", "Method", "Runs", "Completed", "Time-censored", "Memory failure", "Other failure", "Zero rounds", "Recorded statuses"], counts)
    lines.extend(["A zero-round run is a failed or censored attempt, never zero loss. Missing required artifacts stop this report. "
                  "Terminal zero-round runs remain in the denominators even if the ordinary comparator analyzer omitted their all-zero case.", "",
                  "## Scale: actual horizons and numerical regret intervals", "",
                  "Let A be a method's cumulative loss at the stated prefix, L(u) the feasible offline comparator loss, "
                  "g its gradient, G=gᵀu+B‖g‖, and Δ=min{G,L(u)}. Convexity and nonnegative logistic loss give the real-arithmetic "
                  "regret interval [A−L(u), A−L(u)+Δ]; here every quantity is evaluated in float64. "
                  "These are optimization-error intervals, not statistical confidence intervals. Negative realized regret is retained. "
                  "Rows average the two data seeds only where a shared comparator prefix is available; the available denominator and range are explicit. "
                  "Common means the prefix completed by all three methods on that dataset, so it can be zero when AIOLI fails during initialization."])
    scale_rows = []
    for (d, B, T, algorithm, label), rr in _groups(bundle["stages"]["deep_scale"]["rows"]):
        common = [r["common"] for r in rr if r.get("common")]
        scale_rows.append([integer(d), f"{B:g}", integer(T), NAMES[algorithm],
                           span([r["completed_rounds"] for r in rr]), f"{len(common)}/{len(rr)}",
                           span([p["round"] for p in common]), interval(common),
                           fmt(mean(p.get("online_seconds") for p in common)),
                           fmt(mean(r["online_seconds"] for r in rr)),
                           fmt(mean(r["peak_private_commit"] / MIB for r in rr), 1)])
    _table(lines, ["d", "B", "Configured T", "Method", "Actual completed range", "Common available", "Common rounds", "Mean regret interval", "Mean common online s", "Mean full-attempt online s", "Mean full-attempt peak MiB"], scale_rows)
    lines.extend([f"![Scale completion and measured memory]({link_target(figure_folder / 'deep_scale.png')})", "",
                  "## Sampler pairs when the three-method common prefix is short or unavailable", "",
                  "Every difference below is proximal log-odds minus sigmoid/MALA on exactly the same stored prefix. "
                  "The comparator cancels, so loss and regret differences coincide. Negative favors proximal. "
                  "The fixed-100 column uses an identical horizon when both methods reach it. "
                  "Pair-common endpoints depend on completion and can differ across seeds; their mean per-round differences are descriptive and time-selected."])
    _table(lines, ["d; B; configured T", "100 available", "Mean Δ loss at 100", "Pair-common available", "Pair-common round range", "Mean pair-common Δ loss / round"], _paired_table_rows(bundle))
    lines.extend(["", "## Particle and transition budget sensitivity", "",
                  "All budget runs use d=100, B=1000, configured T=2000 and a 45-second online cap. "
                  "The grid is a diagnostic sensitivity analysis, not a newly selected winner. "
                  "Each change is paired with the same method's original selected baseline (proximal m256n1, sigmoid m16n16) on its data seed. "
                  "First minus baseline is the sign convention; negative means lower loss for the changed configuration. "
                  "Unavailable fixed-100 comparisons are retained. Increasing particles and increasing transitions change different costs; labels give both exact values."])
    _table(lines, ["Configuration − original baseline", "100 available", "Mean Δ loss at 100", "Pair-common available", "Pair-common round range", "Mean pair-common Δ loss / round"], _paired_table_rows(bundle, True))
    budget_rows = []
    for (_, _, T, algorithm, label), rr in _groups(bundle["stages"]["deep_budget"]["rows"]):
        common = [r["common"] for r in rr if r.get("common")]
        budget_rows.append([NAMES[algorithm], label, span([r["completed_rounds"] for r in rr]), integer(T),
                            span([p['round'] for p in common]), interval(common),
                            fmt(mean(r["online_seconds"] for r in rr)), fmt(mean(r["peak_private_commit"]/MIB for r in rr), 1)])
    _table(lines, ["Method", "Configuration", "Actual completed range", "Configured T", "All-grid common rounds", "Mean all-grid common regret interval", "Mean full-attempt online s", "Mean full-attempt peak MiB"], budget_rows)
    lines.extend([f"![Fixed-checkpoint budget sensitivity]({link_target(figure_folder / 'deep_budget.png')})", ""])
    _precision_section(lines, bundle)
    lines.extend(["## Actual short-probe memory frontiers", "",
                  "These are fresh actual constructors followed by up to two predict/observe rounds on streamed data, not dummy allocations or regret comparisons. "
                  "A bracket is an observed completed short probe and observed memory failure; it is not a deterministic universal capacity threshold. "
                  "Each axis holds its other parameters fixed; the separate largest dimension and particle count are not a jointly tested feasible configuration. "
                  "Timeouts supply neither side of the bracket. Failure phase is taken from a traceback; unknown means the evidence does not locate it. "
                  "An explicit fatal OpenBLAS allocation-error message also establishes a memory failure, but its phase remains unknown. "
                  "Without a final worker memory snapshot, the sampled process peak is only a lower bound (marked ≥); a small monitor reading does not establish low peak memory. "
                  "Algorithm timing/state are unavailable without a checkpoint, or explicitly censored at the last checkpoint. "
                  "The OS cap stays at 2 GiB. Proximal advisory allocation guards are explicitly raised for these probes; its numerical algorithms are unchanged."])
    policy = bundle["frontier_policy"]
    lines.extend([f"Policy: B={policy['B']:g}, R=1, configured T={policy['T']}, requested rounds={policy['rounds']}, "
                  f"online cap={policy['seconds_per_worker']:g} s per worker, at most {policy['max_probes']} probes, "
                  f"{policy['refinements_per_axis']} refinements per axis, relative bracket tolerance={policy['relative_tolerance']:g}. "
                  f"Campaign stopped with `{bundle['frontier']['status']}` after {bundle['frontier']['attempts']} attempts. "
                  "Particle axes use one transition; dimension axes use proximal m256n1 and sigmoid m16n16."])
    lines.append("Memory-probe data seeds: " + ", ".join(str(s) for s in sorted({s['data_seed'] for s in bundle['frontier_specs']}))
                 + "; sampler seeds: " + ", ".join(str(s) for s in sorted({s['sampler_seed'] for s in bundle['frontier_specs']})) + ".")
    memory_rows = []
    for axis in bundle["frontier"]["axes"]:
        rr = [r for r in bundle["frontier_records"] if r["axis"] == axis["axis"]]
        failure = next((r for r in rr if r["value"] == axis["smallest_memory_failure"] and r["outcome"] == "memory_failure"), None)
        success = next((r for r in rr if r["value"] == axis["largest_completed_probe"] and r["outcome"] == "completed"), None)
        counts = Counter(r["outcome"] for r in rr)
        cc = [counts["completed"], counts["memory_failure"], counts["time_censored"]]
        cc.append(len(rr)-sum(cc))
        fixed = "d=" + span([r["d"] for r in rr]) if axis["parameter"] == "particles" else (
            "m=" + span([r["actual_particles"] for r in rr]) if axis["algorithm"] != "aioli" else "Cholesky")
        def peak(record):
            return ("≥" if record.get("memory_peak_is_lower_bound") else "") + fmt(record["peak_private_commit"]/MIB, 1) if record else "—"
        phases = Counter(r["failure_phase"] for r in rr if r["outcome"] == "memory_failure")
        memory_rows.append([axis["axis"], fixed, "/".join(str(v) for v in cc), integer(axis["largest_completed_probe"]),
                            peak(success), integer(axis["smallest_memory_failure"]),
                            failure["failure_phase"] if failure else "—", peak(failure),
                            "; ".join(f"{k}: {v}" for k, v in sorted(phases.items())),
                            "yes" if axis["observed_nonmonotonicity"] else "no"])
    _table(lines, ["Axis", "Fixed parameter", "Completed/memory/time/other counts", "Largest completed", "Peak MiB at lower", "Smallest memory failure", "Failure phase at upper", "Peak MiB at upper", "All memory-failure phases", "Nonmonotonic observations"], memory_rows)
    native = [r for r in bundle["frontier_records"] if r.get("native_allocation_failure")]
    if native:
        _table(lines, ["Native allocation failure", "Phase", "Timing/state status", "Memory evidence", "Retained stderr"],
               [[f"{r['axis']}={r['value']}", r["failure_phase"], r.get("timing_measurement_status", "unavailable"),
                 r.get("memory_peak_source", "unknown") + "; lower bound", link(r["stderr_path"])] for r in native])
    if bundle.get("frontier_classification_audit"):
        lines.append("Derived classifications were repaired from retained raw evidence; the original worker files and original source snapshot remain unchanged. "
                     "Only memory_frontier.py changed among the recorded source modules; the current module source and policy were archived, "
                     "and regenerated recorded numerical specifications were verified unchanged. The module changes received code review; "
                     "these automated field/hash checks are not a machine proof of source equivalence. Future dispatched probes are also checked against the policy and recorded templates. "
                     + link(results / "memory_frontier_deep" / "classification_audit.json", "Classification audit and archive references") + ".")
    lines.extend([f"![Actual memory frontier probes]({link_target(figure_folder / 'deep_memory_frontier.png')})", "",
                  "## Retained-history kernel probes: zero online rounds", "",
                  "These probes append synthetic history directly, keeping the sampled prior cloud unchanged, and then attempt one real predict call. "
                  "They execute zero online predict/observe trajectories and provide no regret result or posterior-mixing guarantee. "
                  "Each uses d=1000, B=1, m=2, one transition, data seed 20260910 and sampler seed 93071. "
                  "Inputs have independent coordinates ±0.5/√d and independent ±1 labels; the query is the first coordinate unit vector. "
                  "The configured horizon is requested history rows plus one. Sigmoid uses a nonlazy step with multiplier 0.5 and one bridge. "
                  "History allocation is block-rounded to 1024 rows; requests in MiB therefore need not equal retained allocation. "
                  "After an allocation failure, only valid retained rows are exposed to the attempted prediction. "
                  "The 120-second per-probe deadline includes initialization, generation, appends and prediction; supervisor allowance is 20 seconds. "
                  "Partial prediction times or last checkpoints remain censored when the call does not finish."])
    history_rows = []
    for r in bundle["history"]:
        w, s = r["worker"], r["spec"]
        history_rows.append([NAMES[r["algorithm"]], r["requested_mib"], integer(s["history_rows"]),
                             integer(w.get("retained_history_rows")), integer(w.get("configured_horizon", s["history_rows"]+1)),
                             fmt(w.get("history_ndarray_bytes", 0)/MIB, 1), fmt(r["peak_private_commit"]/MIB, 1),
                             w.get("append_status", "unknown"), w.get("predict_status", "unknown"),
                             fmt(w.get("predict_seconds")), "yes" if r["supervisor"].get("checkpoint_is_censored") else "no"])
    _table(lines, ["Method", "Requested history MiB", "Requested rows", "Retained valid rows", "Configured T", "Allocated history MiB", "Peak process MiB", "Append status", "One-predict status", "Predict seconds (possibly partial)", "Censored checkpoint"], history_rows)
    lines.extend([f"![Retained-history memory and kernel timing]({link_target(figure_folder / 'deep_history.png')})", "",
                  "## AIOLI with configured T=10,000,000", "",
                  "This is a single measured AIOLI trajectory, not an extrapolation or a paired comparison with samplers at ten million rounds. "
                  "The table reports its actual terminal horizon even if the full configured horizon was not completed. "
                  "Offline streamed Newton refinement only improves evaluation of the same norm-ball comparator; it changes neither online predictions nor their measured runtime."])
    long_rows, long_costs = [], []
    for r in bundle["stages"]["deep_long"]["rows"]:
        w, p = r["_worker"], r.get("final")
        long_rows.append([integer(r["d"]), f"{r['B']:g}", integer(r["T"]), integer(r["completed_rounds"]), r["status"],
                          interval([p]), precise(p.get("comparator_first_order_gap") if p else None)])
        long_costs.append([fmt(w.get("initialization_seconds")), fmt(w.get("prediction_seconds")), fmt(w.get("update_seconds")),
                           fmt(r["online_seconds"]), fmt(r["process_wall_seconds"]), fmt(r["peak_private_commit"]/MIB, 1), integer(r["state_bytes"])])
    _table(lines, ["d", "B", "Configured T", "Actual completed", "Status", "Final regret interval", "Raw first-order gap"], long_rows)
    _table(lines, ["Init s", "Prediction s", "Update s", "Online s", "Process wall s", "Peak MiB", "Owned state bytes"], long_costs)
    attempts = bundle["newton"].get("attempts", [])
    limits = sorted({(a['refinement'].get('max_iterations'), a['refinement'].get('max_seconds'),
                      a['refinement'].get('tolerance')) for a in attempts})
    limit_text = "; ".join(f"{integer(it)} iterations, {fmt(sec, 0)} seconds, tolerance {tol:g}"
                           for it, sec, tol in limits) if limits else "no attempted refinement recorded"
    lines.extend([f"The comparator refinement audit retains {len(attempts)} attempted prefix refinements, "
                  f"{sum(bool(a.get('cache_replaced')) for a in attempts)} accepted cache replacements, "
                  f"and {fmt(sum(a['refinement'].get('elapsed', 0) for a in attempts))} s of separately recorded offline refinement. "
                  f"Recorded per-prefix limits: {limit_text}. Boundary-limited or failed refinements retain the previous fit.", "",
                  f"![Measured long-horizon regret and round cost]({link_target(figure_folder / 'deep_long.png')})", "",
                  "## Raw evidence and reproduction", "",
                  "Run these commands from the workspace using experiments/.venv/Scripts/python.exe, sequentially and after other measured jobs are idle. "
                  "The report itself performs no comparator fitting and launches no workers:", "", "```powershell",
                  "& 'experiments/.venv/Scripts/python.exe' -m experiments.deep_pairwise",
                  "& 'experiments/.venv/Scripts/python.exe' -m experiments.analyze deep_scale --comparator-seconds 2",
                  "& 'experiments/.venv/Scripts/python.exe' -m experiments.analyze deep_budget --comparator-seconds 2",
                  "& 'experiments/.venv/Scripts/python.exe' -m experiments.analyze deep_long --comparator-seconds 3",
                  "& 'experiments/.venv/Scripts/python.exe' -m experiments.deep_comparator_refine --stage deep_long",
                  "& 'experiments/.venv/Scripts/python.exe' -m experiments.deep_precision",
                  "& 'experiments/.venv/Scripts/python.exe' -m experiments.deep_report",
                  "& 'experiments/.venv/Scripts/python.exe' -m experiments.deep_audit", "```", "",
                  "The complete measurement orchestration and exact frozen input grids are in " + link(BASE/"deep_campaign.py") + " and " + link(BASE/"deep_prepare.py") + ". "
                  "Individual manifests are authoritative; rerunning a finished worker reuses only an identical stored specification.", ""])
    for stage in STAGES:
        folder = results / stage
        lines.append(f"- {stage}: " + ", ".join(link(folder/name) for name in ("manifest.json", "cases.json", "summary.json", "environment.json")) + ".")
    lines.extend(["- Paired comparisons: " + link(results/"deep_pairwise.json") + ".",
                  "- Actual memory probes: " + ", ".join(link(results/"memory_frontier_deep"/n) for n in ("policy.json", "manifest.json", "frontier_summary.json", "frontier_records.jsonl")) + ".",
                  "- Offline long comparator provenance: " + link(results/"deep_long"/"newton_refinement_audit.json") + ".",
                  "- Large-B scalar precision audit: " + link(results/"deep_precision_audit.json") + ".",
                  "- Final saved-artifact consistency audit: " + link(results/"deep_artifact_audit.json") + ".",
                  "- Algorithms: " + ", ".join(link(BASE/n) for n in ("proximal.py", "sigmoid.py", "aioli.py", "common.py")) + ".",
                  "- Report generator: " + link(BASE/"deep_report.py") + ".", ""])
    _table(lines, ["History probe", "Exact specification", "Supervisor + terminal/checkpoint evidence"],
           [[r["output"].name, link(r["output"]/"spec.json"), link(r["output"]/"supervisor.json")] for r in bundle["history"]])
    failures = [r for r in all_rows if r["outcome"] in ("memory_failure", "other_failure")]
    if failures:
        _table(lines, ["Failed online attempt", "Phase", "Recorded error", "Raw supervisor"],
               [[r["_output"].name, r["phase"], r.get("error") or r.get("termination") or r["status"],
                 link(r["_output"]/"supervisor.json")] for r in failures])
    lines.extend(["Event counters and matrix-kernel FLOPs are implementation instrumentation, not hardware instruction counts. "
                  "Failed kernels can leave partial work uncounted. Nested diagnostic timers overlap and must not be added. "
                  "A high acceptance rate does not establish sampler mixing; a completed probe does not certify statistical accuracy.", ""])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def link_target(path):
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def write_figures(bundle, folder):
    """Static scientific figures; exclusively measured points, no projections."""
    os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".matplotlib"))
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    folder.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        fig.savefig(folder / name, dpi=180, bbox_inches="tight")
        plt.close(fig)
    def legends(ax):
        handles = [Line2D([], [], color=c, marker="o", linestyle="none", label=NAMES[a]) for a, c in COLORS.items()]
        handles += [Line2D([], [], color="#555555", marker=m, linestyle="none", label=s.replace("_", " ")) for s, m in MARKERS.items()]
        ax.legend(handles=handles, fontsize=7, ncol=2)
    scale = bundle["stages"]["deep_scale"]["rows"]
    settings = sorted({(r["d"], r["B"], r["T"]) for r in scale})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for r in scale:
        setting = (r["d"], r["B"], r["T"])
        offset = {"ours": -.23, "sigmoid": 0., "aioli": .23}[r["algorithm"]]
        offset += -.035 if r["data_seed"] == min(s["data_seed"] for s in scale) else .035
        x = settings.index(setting) + offset
        style = dict(c=COLORS[r["algorithm"]], marker=MARKERS[r["outcome"]], s=39, alpha=.85)
        axes[0].scatter(x, r["completed_rounds"]/r["T"], **style)
        axes[1].scatter(x, r["peak_private_commit"]/MIB, **style)
    labels = [f"d={d:,}\nB={B:g}\nT={T:,}" for d, B, T in settings]
    for ax in axes:
        ax.set_xticks(range(len(settings)), labels, fontsize=8)
        ax.grid(alpha=.2, axis="y")
    axes[0].set(yscale="symlog", ylabel="Actually completed / configured rounds", title="Two measured seeds per setting; zero retained")
    axes[0].set_yscale("symlog", linthresh=.001)
    axes[0].set_ylim(-.00015, 1.3)
    axes[0].set_yticks([0., .001, .01, .1, 1.], ["0", "0.001", "0.01", "0.1", "1"])
    axes[1].axhline(2048, color="black", ls="--", lw=1, label="2 GiB process cap")
    axes[1].set(ylabel="Full-attempt peak private commit (MiB)", title="Completed, censored, and failed attempts")
    legends(axes[0])
    save(fig, "deep_scale.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    summaries = bundle["pairwise"]["budget_baseline_seed_summaries"]
    for ax, algorithm in zip(axes, ("ours", "sigmoid")):
        rr = sorted([r for r in summaries if r["checkpoint"] == "100" and r["first"]["algorithm"] == algorithm
                     and r["first"]["label"] != r["second"]["label"]], key=lambda r: (r['first']['config']['particles'], r['first']['config']['transitions']))
        for i, r in enumerate(rr):
            for j, p in enumerate(r["dataset_values"]):
                if p["available"]:
                    ax.scatter(i + (j-.5)*.1, p["first_minus_second_average_loss"], s=34, color=COLORS[algorithm])
            ax.text(i, .98, f"{r['paired_datasets_available']}/{r['paired_datasets_expected']}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
        ax.axhline(0, c="#555555", lw=1)
        ax.set_xticks(range(len(rr)), [r["first"]["label"] for r in rr], rotation=35, ha="right")
        baseline = "m256n1" if algorithm == "ours" else "m16n16"
        ax.set(title=f"{NAMES[algorithm]} minus {baseline}", ylabel="Paired loss difference / round at fixed round 100")
        ax.grid(alpha=.2, axis="y")
    fig.suptitle("B=1000, d=100, configured T=2000; each dot is one paired dataset; fractions show availability")
    save(fig, "deep_budget.png")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), layout="constrained")
    for ax, axis in zip(axes.flat, bundle["frontier"]["axes"]):
        for r in bundle["frontier_records"]:
            if r["axis"] != axis["axis"]:
                continue
            kind = r["outcome"] if r["outcome"] in MARKERS else "other_failure"
            ax.scatter(r["value"], r["peak_private_commit"]/MIB, c=COLORS[r["algorithm"]], marker=MARKERS[kind], s=42)
            if r["outcome"] == "memory_failure" and (r["value"] == axis["smallest_memory_failure"] or r.get("native_allocation_failure")):
                note = r["failure_phase"] + ("\nmonitor lower bound" if r.get("memory_peak_is_lower_bound") else "")
                ax.annotate(note, (r["value"], r["peak_private_commit"]/MIB), xytext=(3, 4), textcoords="offset points", fontsize=6)
        ax.axhline(2048, c="black", ls="--", lw=1)
        ax.set(xscale="log", xlabel=axis["parameter"], ylabel="Peak private commit (MiB)", title=axis["axis"])
        ax.grid(alpha=.2)
    axes.flat[-1].axis("off")
    legends(axes.flat[-1])
    fig.suptitle("Actual constructor + short online probes; dashed line = 2 GiB limit; no extrapolation")
    save(fig, "deep_memory_frontier.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for r in bundle["history"]:
        w = r["worker"]
        x = r["requested_mib"] + (-2 if r["algorithm"] == "ours" else 2)
        style = dict(c=COLORS[r["algorithm"]], marker=MARKERS[r["outcome"]], s=48)
        axes[0].scatter(x, r["peak_private_commit"]/MIB, **style)
        seconds = w.get("predict_seconds")
        if _finite(seconds) and seconds > 0:
            marker = "o" if w.get("predict_status") == "completed" else MARKERS[r["outcome"]]
            axes[1].scatter(x, seconds, c=COLORS[r["algorithm"]], marker=marker, s=48)
    axes[0].axhline(2048, color="black", ls="--", lw=1)
    axes[0].set(ylabel="Whole-probe peak private commit (MiB)", title="History preload and one attempted predict")
    axes[1].set(yscale="log", ylabel="One-predict seconds (unfinished calls are partial)", title="No online trajectory or regret measured")
    for ax in axes:
        ax.set(xlabel="Requested history allocation (MiB)", xticks=HISTORY_MIB)
        ax.tick_params(axis="x", labelrotation=40)
        ax.grid(alpha=.2)
    legends(axes[0])
    save(fig, "deep_history.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for r in bundle["stages"]["deep_long"]["rows"]:
        curves = sorted([p for p in bundle["long_curves"] if p["case_id"] == r["case_id"]
                         and p["algorithm"] == r["algorithm"] and p["label"] == r["label"]
                         and p["sampler_seed"] == r["sampler_seed"]], key=lambda p: p["round"])
        axes[0].plot([p["round"] for p in curves], [p["regret_lower"] for p in curves], color=COLORS['aioli'])
        axes[0].fill_between([p["round"] for p in curves], [p["regret_lower"] for p in curves], [p["regret_upper"] for p in curves], color=COLORS['aioli'], alpha=.25)
        n = r["completed_rounds"]
        if n:
            path = r["_output"] / "online_times.bin"
            if not path.exists() or path.stat().st_size < 8*n:
                raise RuntimeError(f"Missing complete long-run time array: {path}")
            times = np.memmap(path, dtype="<f8", mode="r", shape=(n,))
            edges = sorted(set([0, n] + [int(v) for v in np.geomspace(1, n, 40)]))
            init = r["_worker"]["initialization_seconds"]
            xx, yy = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                start = float(times[lo-1]) if lo else init
                elapsed = float(times[hi-1]) - start
                if elapsed > 0:
                    xx.append(math.sqrt((lo+1)*hi))
                    yy.append(elapsed/(hi-lo))
            axes[1].plot(xx, yy, "-o", ms=3, c=COLORS['aioli'])
            del times
        fig.suptitle(f"AIOLI: B={r['B']:g}, d={r['d']}, actual {n:,} / configured {r['T']:,} rounds; {r['status']}")
    axes[0].set(xscale="log", yscale="symlog", xlabel="Actually completed round", ylabel="Cumulative regret interval", title="Shading = numerical comparator uncertainty")
    if bundle["long_curves"] and all(c["regret_lower"] >= 0 for c in bundle["long_curves"]):
        axes[0].set_ylim(bottom=0.)
    axes[1].set(xscale="log", yscale="log", xlabel="Observed round (geometric bins)", ylabel="Mean predict + observe seconds / round", title="Recorded cumulative online times; initialization removed")
    for ax in axes:
        ax.grid(alpha=.2)
    save(fig, "deep_long.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=BASE / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "DEEP_EXPERIMENT_RESULT.md")
    parser.add_argument("--figures", type=Path, default=BASE / "figures")
    args = parser.parse_args()
    _no_measured_workers()
    bundle = load_artifacts(args.results)
    _no_measured_workers()
    write_figures(bundle, args.figures)
    write_report(bundle, args.output, args.figures)
    provenance = {"deterministic_from_input_artifacts": True, "report": str(args.output.resolve()),
                  "input_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle["sources"]},
                  "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "measurement_or_comparator_fitting_performed": False}
    (args.figures / "deep_report_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.output.resolve()), "figures": str(args.figures.resolve()),
                      "online_runs": sum(len(v['rows']) for v in bundle['stages'].values()),
                      "memory_probes": len(bundle['frontier_records']), "history_probes": len(bundle['history'])}))


if __name__ == "__main__":
    main()
