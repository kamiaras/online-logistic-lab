"""Fixed-endpoint execution/work analysis; never runs a sampler or comparator fit.

Reads 18 full-backend cells, six curvature cells, and linked sequential controls.
Run only after measurement workers finish. Writes analysis.json and analysis.md
under experiments/results/batched_computation_analysis by default.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from .shared_ablation_analyze import (
    ROOT, ENDPOINTS, SECONDARY_SEEDS, read_json, read_optional_json, read_run as read_old_run,
    verify_data, guards, summary, fmt, sha256,
)
from .shared_tangent import _parameters, _mixing_counts

RESULTS = ROOT / "experiments/results"
WORK = ("moves", "proposals", "full_test_rows", "gradient_refresh_rows", "estimated_matmul_flops")
LOSS = ("loss", "guard_c1_loss", "guard_c3_loss", "guard_c4_loss", "online_seconds")
REQUIRED = ("logits", "guarded_logits", "losses", "guarded_losses", "online_times",
            "skipped", "lanes", "transitions", "labels") + WORK


def ratios(counts, denominator):
    return {key: value / denominator if denominator else None for key, value in counts.items()}


def logit_distribution(values):
    return dict(mean=float(np.mean(values)), mean_absolute=float(np.mean(np.abs(values))),
                rms=float(np.sqrt(np.mean(values ** 2))), maximum_absolute=float(np.max(np.abs(values))),
                quantiles={str(q): float(np.quantile(values, q)) for q in (.05, .25, .5, .75, .95)})


def endpoint(a, spec, n, result=None):
    z, labels = a["logits"][:n], a["labels"][:n]
    losses = np.logaddexp(0., -labels * z)
    sampled = int(np.sum(~a["skipped"][:n].astype(bool)))
    counts = {key: int(a[key][n - 1]) for key in WORK}
    out = dict(available=True, endpoint=n, loss=float(losses.sum()), loss_per_round=float(losses.mean()),
               online_seconds=float(a["online_times"][n - 1]), sampled_rounds=sampled,
               raw_logit_distribution=logit_distribution(z), counts=counts,
               counts_per_round=ratios(counts, n), counts_per_sampled_round=ratios(counts, sampled),
               counts_per_move=ratios(counts, counts["moves"]), counts_per_proposal=ratios(counts, counts["proposals"]))
    for c in (1, 3, 4):
        clipped, guarded_loss = guards(z, labels, c)
        out[f"guard_c{c}_loss"] = float(guarded_loss.sum())
        out[f"guard_c{c}_loss_per_round"] = float(guarded_loss.mean())
        out[f"guard_c{c}_loss_change"] = float(guarded_loss.sum() - losses.sum())
        out[f"guard_c{c}_increase_bound"] = n / (c * (n + 1))
        out[f"guard_c{c}_clipped_rounds"] = int(np.sum(clipped != z))
    if spec["backend"] == "curvature":
        d = spec["d"]
        coordinates = dict(summary_query_coordinates=8*d*counts["proposals"],
                           summary_rebuild_coordinates=d*counts["gradient_refresh_rows"],
                           summary_append_coordinates=d*int(np.sum(a["lanes"][:n])))
        # These three are exact structural reconstructions, not saved arrays.
        out["summary_coordinate_events"] = coordinates
        out["summary_coordinate_count_source"] = "reconstructed from source invariants and endpoint arrays"
        out["summary_coordinate_events_per_round"] = ratios(coordinates, n)
        out["summary_coordinate_events_per_move"] = ratios(coordinates, counts["moves"])
        t = np.arange(1, n + 1, dtype=np.int64)
        gradient_increments = np.diff(np.r_[0, a["gradient_refresh_rows"][:n]]).astype(np.int64)
        if np.any(gradient_increments % t):
            raise ValueError("gradient rows are not complete current-history lane passes")
        rejected_lanes = gradient_increments // t
        batch = int(spec["config"].get("particle_batch", 64))
        lower = int(np.sum(t * ((rejected_lanes + batch - 1) // batch)))
        upper = counts["gradient_refresh_rows"]
        counters = (result or {}).get("metrics", {}).get("counters", {})
        terminal = bool(result and n == len(a["logits"]) == result.get("round")
                        and result.get("status") in ("completed", "time_budget"))
        exact = counters.get("summary_rebuild_history_rows", 0) if terminal else None
        out["summary_rebuild_history_rows"] = dict(value=exact, lower_bound=lower, upper_bound=upper,
            source="terminal counter" if terminal else "not recorded per round; bounds only")
        if exact is not None and not lower <= exact <= upper:
            raise ValueError("terminal summary history rows violate structural bounds")
    return out


def read_performance(spec, folder, data_checks):
    out = folder / Path(spec["output"]).name
    read_errors = []
    result = read_optional_json(out / "result.json", read_errors)
    supervisor = read_optional_json(out / "supervisor.json", read_errors)
    progress = read_optional_json(out / "progress.json", read_errors)
    record = dict(id=out.name, spec=spec, result=result, supervisor=supervisor, progress=progress,
        status=result.get("status", "unknown") if result else (
            "abnormal_exit" if supervisor else "running_or_incomplete" if out.exists() else "not_started"),
        errors=[], read_errors=read_errors, trajectory_valid=False, trajectory_rounds=0, endpoints={})
    a = None
    if data_checks[str(Path(spec["data_path"]))]["errors"]:
        record["errors"].append("input data hash/length verification failed")
    if (out / "trajectory.npz").exists():
        try:
            with np.load(out / "trajectory.npz", allow_pickle=False) as archive:
                a = {key: archive[key].copy() for key in archive.files}
            if any(key not in a for key in REQUIRED):
                raise ValueError("missing required trajectory array")
            n = len(a["logits"])
            record["trajectory_rounds"] = n
            if not 0 < n <= spec["rounds"] or any(a[key].shape != (n,) for key in REQUIRED):
                raise ValueError("invalid or inconsistent trajectory dimensions")
            if any(not np.all(np.isfinite(a[key])) for key in REQUIRED):
                raise ValueError("nonfinite trajectory value")
            if not np.all(np.isin(a["skipped"], [0, 1])):
                raise ValueError("invalid skip flags")
            a["skipped"] = a["skipped"].astype(bool)
            labels = np.fromfile(Path(spec["data_path"]) / "y.bin", dtype="i1", count=n)
            if not np.array_equal(a["labels"], labels) or not np.all(np.isin(labels, [-1, 1])):
                raise ValueError("labels differ from fixed data")
            raw_loss = np.logaddexp(0., -labels*a["logits"])
            c4, c4loss = guards(a["logits"], labels, 4)
            for key, expected in (("losses", raw_loss), ("guarded_logits", c4), ("guarded_losses", c4loss)):
                if not np.allclose(a[key], expected, rtol=1e-12, atol=1e-10):
                    raise ValueError(f"{key} failed independent float recomputation")
            t = np.arange(1, n + 1)
            if np.any(c4loss-raw_loss > 1/(4*t*(t+1))+1e-9):
                raise ValueError("c4 pointwise guard inequality failed")
            if np.any(a["logits"][a["skipped"]] != 0):
                raise ValueError("skipped round has a nonzero raw logit")
            for key in ("lanes", "transitions") + WORK:
                if np.any(a[key] < 0) or np.any(a[key] != np.floor(a[key])):
                    raise ValueError(f"noninteger or negative {key}")
            if (np.any(a["lanes"] < 2) or np.any(a["lanes"] % 2) or np.any(a["lanes"] > spec["config"]["lanes"])
                    or len(np.unique(a["lanes"])) != 1 or np.any(a["transitions"] < 1)
                    or np.any(a["transitions"] > spec["config"]["transitions"])):
                raise ValueError("actual M/N violates the declared practical caps")
            expected_moves = np.cumsum(np.where(a["skipped"], 0, a["lanes"]*a["transitions"]))
            if not np.array_equal(expected_moves, a["moves"]):
                raise ValueError("cumulative moves differ from actual per-round M/N/skip")
            for key in WORK + ("online_times",):
                if np.any(np.diff(a[key]) < 0):
                    raise ValueError(f"nonmonotone cumulative {key}")
            if np.any(a["proposals"] < a["moves"]):
                raise ValueError("fewer conditional proposals than completed moves")
            if result:
                if result.get("round") != n:
                    raise ValueError("result/trajectory round disagreement")
                for key, values in (("cumulative_loss", raw_loss), ("guarded_loss", c4loss)):
                    if not math.isclose(result[key], float(values.sum()), rel_tol=1e-11, abs_tol=1e-8):
                        raise ValueError(f"result {key} mismatch")
                counters = result.get("metrics", {}).get("counters", {})
                names = dict(moves="proximal_steps", proposals="rqo_proposals")
                if result["status"] in ("completed", "time_budget"):
                    for key in WORK:
                        if int(a[key][-1]) != counters.get(names.get(key, key), 0):
                            raise ValueError(f"terminal cumulative counter mismatch: {key}")
                m, d = int(a["lanes"][-1]), spec["d"]
                diagnostics = result.get("diagnostics", {})
                expected_extra = 64*m*d if spec["backend"] == "curvature" else 0
                observed_extra = diagnostics.get("extra_curvature_summary_bytes", 0)
                record["memory_accounting"] = dict(state_bytes=result.get("state_bytes"),
                    per_lane_cache_bytes=diagnostics.get("extra_lane_cache_bytes"),
                    extra_curvature_summary_bytes=observed_extra, expected_curvature_summary_bytes=expected_extra,
                    extra_curvature_real_cells=8*m*d if spec["backend"] == "curvature" else 0,
                    process_peak_private_bytes=(supervisor or {}).get("peak_private_bytes"),
                    source="whole-attempt counters/monitor; not an endpoint-specific peak")
                if observed_extra != expected_extra:
                    raise ValueError("curvature summary allocation differs from 8*M*d floats")
            record["trajectory_valid"] = not record["errors"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            record["errors"].append(str(exc))
            a = None
    for n in ENDPOINTS:
        if n > spec["rounds"]:
            point = dict(available=False, reason="not_requested")
        elif not record["trajectory_valid"]:
            point = dict(available=False, reason="invalid_trajectory" if record["errors"] else "no_saved_trajectory")
        elif n > record["trajectory_rounds"]:
            point = dict(available=False, reason="fixed_endpoint_unattained")
        else:
            try:
                point = endpoint(a, spec, n, result)
            except ValueError as exc:
                record["errors"].append(str(exc))
                point = dict(available=False, reason="invalid_work_reconstruction")
        record["endpoints"][str(n)] = point
    return record, a


def reconstruct_old_schedule(spec, a):
    """Read-only deterministic replay of the old schedule, with no chain states."""
    n, d, B = len(a["logits"]), spec["d"], spec["B"]
    X = np.fromfile(Path(spec["data_path"])/"X.bin", dtype="<f8", count=n*d).reshape(n, d)
    p = _parameters(d, B, spec["T"], float(spec["config"].get("R", 1.)))
    m = a["scores"].shape[1]
    E = E1 = low = 0.
    ledger, observed = np.zeros(d), np.zeros(d)
    initial, later, skipped = [], [], []
    for x, y in zip(X, a["labels"]):
        norm2, l1 = float(x@x), float(np.sum(np.abs(x)))
        E += norm2/4
        E1 += l1*l1/4
        row_a = math.exp(-2*B*l1-2*math.log1p(math.exp(-2*B*l1)))
        ledger += row_a*(2*x*x-np.abs(x)*l1)
        low = max(0., low, float(np.min(ledger)))
        gamma, inveta = p["lam"]+low, p["lam"]+max(E1, 2*E)
        N, ell, _, _ = _mixing_counts(p["H"], gamma, inveta)
        N = min(N or spec["config"]["burn_in"], spec["config"]["burn_in"])
        ell = min(ell or spec["config"]["transitions"], spec["config"]["transitions"])
        delta = min(2*B*l1, float(np.linalg.norm(observed))*math.sqrt(norm2)/gamma)
        lower = norm2/(p["lam"]+4*p["a"]*E)
        skip = delta == 0 or (lower > 0 and math.log(delta)-math.log(2)
                <= math.log(3/8)+math.log(lower)-p["log_m"])
        initial.append(N); later.append(ell); skipped.append(skip)
        observed += .5*int(y)*x
    skipped = np.asarray(skipped)
    recorded_skip = np.all(np.isnan(a["scores"]), axis=1)
    moves = np.cumsum(np.where(skipped, 0, np.asarray(initial)+(m-1)*np.asarray(later)))
    return dict(m=m, initial=np.asarray(initial), later=np.asarray(later), skipped=skipped, moves=moves,
                reconstruction_matches_recorded_skips=bool(np.array_equal(skipped, recorded_skip)))


def point(record, n):
    return record["endpoints"][str(n)] if record else dict(available=False, reason="not_in_manifest")


def array_match(left, right):
    return dict(exact=bool(np.array_equal(left, right)),
                within_tolerance=bool(np.allclose(left, right, rtol=1e-12, atol=1e-10)),
                max_absolute_difference=float(np.max(np.abs(np.asarray(left, float)-np.asarray(right, float)))))


def compare_curvature(curve, full, ca, fa, n):
    c, f = point(curve, n), point(full, n)
    row = dict(case=curve["spec"]["case"], seed=curve["spec"]["seed"], endpoint=n,
               curvature=curve["id"], full=full["id"] if full else None, available=False,
               missing={name: p["reason"] for name, p in (("curvature", c), ("full", f)) if not p["available"]})
    if row["missing"]:
        return row
    matches = {key: array_match(ca[key][:n], fa[key][:n]) for key in
               ("logits", "guarded_logits", "skipped", "lanes", "transitions", "moves", "proposals", "gradient_refresh_rows")}
    exact = all(value["exact"] for value in matches.values())
    saved = f["counts"]["full_test_rows"]-c["counts"]["full_test_rows"]
    row.update(available=True, trajectory_matches=matches, all_checked_paths_exact=exact,
               difference={key: c[key]-f[key] for key in LOSS},
               online_time_ratio=c["online_seconds"]/f["online_seconds"],
               curvature_endpoint=c, full_endpoint=f,
               history_lane_rows_saved=saved,
               saved_row_fraction=saved/f["counts"]["full_test_rows"] if f["counts"]["full_test_rows"] else None,
               full_rows_never_larger_at_completed_prefixes=bool(np.all(ca["full_test_rows"][:n] <= fa["full_test_rows"][:n])),
               saved_rows_are_pathwise_on_identical_recorded_path=exact,
               saved_row_dot_proxy=4*curve["spec"]["d"]*saved,
               observed_matmul_proxy_difference=f["counts"]["estimated_matmul_flops"]-c["counts"]["estimated_matmul_flops"])
    row["proxy_difference_equals_saved_row_dot_proxy"] = row["saved_row_dot_proxy"] == row["observed_matmul_proxy_difference"]
    counters = (curve.get("result") or {}).get("metrics", {}).get("counters", {})
    if n == curve["trajectory_rounds"] and curve["status"] == "completed":
        row["terminal_recorded_rows_avoided"] = counters.get("history_rows_avoided_by_squeeze", 0)
        row["recorded_avoided_equals_observed_difference"] = row["terminal_recorded_rows_avoided"] == saved
        if full and full.get("memory_accounting") and curve.get("memory_accounting") and full["trajectory_rounds"] == n:
            row["observed_extra_state_bytes"] = curve["memory_accounting"]["state_bytes"]-full["memory_accounting"]["state_bytes"]
            row["expected_extra_state_bytes"] = 64*int(ca["lanes"][n-1])*curve["spec"]["d"]
            row["extra_state_matches_eight_Md_floats"] = row["observed_extra_state_bytes"] == row["expected_extra_state_bytes"]
    return row


def compare_sequential(batch, old, ba, oa, schedule, n):
    b, s = point(batch, n), point(old, n)
    row = dict(case=batch["spec"]["case"], seed=batch["spec"]["seed"], endpoint=n,
               batched=batch["id"], sequential=old["id"] if old else None, available=False,
               missing={name: p["reason"] for name, p in (("batched", b), ("sequential", s)) if not p["available"]})
    if row["missing"] or schedule is None:
        return row
    sampled = ~ba["skipped"][:n]
    checks = dict(old_reconstruction_matches_recorded_skip=schedule["reconstruction_matches_recorded_skips"],
        equal_skips=bool(np.array_equal(ba["skipped"][:n], schedule["skipped"][:n])),
        equal_midpoint_counts=bool(np.all(ba["lanes"][:n] == schedule["m"])),
        equal_initial_moves_on_sampled_rounds=bool(np.array_equal(ba["transitions"][:n][sampled], schedule["initial"][:n][sampled])),
        equal_other_midpoint_moves_on_sampled_rounds=bool(np.array_equal(ba["transitions"][:n][sampled], schedule["later"][:n][sampled])),
        equal_cumulative_moves=bool(np.array_equal(ba["moves"][:n], schedule["moves"][:n])))
    row.update(available=True, schedule_checks=checks, actual_schedules_match=all(checks.values()),
               difference={key: b[key]-s[key] for key in LOSS},
               online_time_ratio=b["online_seconds"]/s["online_seconds"],
               speed_ratio_sequential_over_batched=s["online_seconds"]/b["online_seconds"],
               batched_endpoint=b, sequential_endpoint=s,
               sequential_raw_logit_distribution=logit_distribution(oa["logits"][:n]))
    return row


def analyze(full_root, curve_root, sequential_root):
    fm, cm, sm = (read_json(p/"manifest.json") or [] for p in (full_root, curve_root, sequential_root))
    if not fm or not sm:
        raise ValueError("Full and sequential manifests are required")
    data_checks = {path: verify_data(path) for path in sorted({str(Path(s["data_path"])) for s in fm+cm+sm})}
    records, arrays = [], {}
    for folder, specs in ((full_root, fm), (curve_root, cm)):
        for spec in specs:
            record, a = read_performance(spec, folder, data_checks)
            records.append(record)
            if a is not None:
                arrays[record["id"]] = a
    old, old_arrays, old_schedules = {}, {}, {}
    for spec in sm:
        record, a = read_old_run(spec, sequential_root, data_checks)
        old[record["id"]] = record
        if a is not None and record["trajectory_valid"]:
            old_arrays[record["id"]] = a
            old_schedules[record["id"]] = reconstruct_old_schedule(spec, a)
            record["schedule_reconstruction"] = dict(
                matches_recorded_skips=old_schedules[record["id"]]["reconstruction_matches_recorded_skips"],
                source="deterministic schedule replay only; no sampling or comparator fitting")
    index = {(r["spec"]["backend"], r["spec"]["case"], r["spec"]["seed"], r["spec"]["config"]["lanes"]): r for r in records}
    curvature, sequential = [], []
    for r in records:
        s = r["spec"]
        if s["backend"] == "curvature":
            f = index.get(("full", s["case"], s["seed"], 64))
            for n in ENDPOINTS:
                curvature.append(compare_curvature(r, f, arrays.get(r["id"]), arrays.get(f["id"]) if f else None, n))
        elif s["config"]["lanes"] == 16:
            name = Path(s["sequential_control"]).name
            for n in ENDPOINTS:
                sequential.append(compare_sequential(r, old.get(name), arrays.get(r["id"]), old_arrays.get(name), old_schedules.get(name), n))
    groups = []
    for backend, specs in (("full", fm), ("curvature", cm)):
        for case in sorted({s["case"] for s in specs}):
            for m in sorted({s["config"]["lanes"] for s in specs if s["case"] == case}):
                for n in ENDPOINTS:
                    cells = [point(index.get((backend, case, seed, m)), n) for seed in SECONDARY_SEEDS]
                    groups.append(dict(backend=backend, case=case, lanes=m, endpoint=n,
                        seeds=[dict(seed=seed, available=p["available"], reason=p.get("reason")) for seed, p in zip(SECONDARY_SEEDS, cells)],
                        metrics={key: summary([p[key] for p in cells if p["available"]], 3) for key in LOSS},
                        work={key: summary([p["counts"][key] for p in cells if p["available"]], 3) for key in WORK}))
    paired_groups = []
    for kind, comparisons in (("curvature_minus_full", curvature), ("batched_minus_sequential", sequential)):
        for case in sorted({r["case"] for r in comparisons}):
            for n in ENDPOINTS:
                cells = [next((r for r in comparisons if (r["case"], r["seed"], r["endpoint"]) == (case, seed, n)),
                              dict(seed=seed, available=False, missing={"run": "not_in_manifest"})) for seed in SECONDARY_SEEDS]
                paired_groups.append(dict(kind=kind, case=case, endpoint=n,
                    seeds=[dict(seed=r["seed"], available=r["available"], missing=r.get("missing", {})) for r in cells],
                    differences={key: summary([r["difference"][key] for r in cells if r["available"]], 3) for key in LOSS},
                    time_ratio=summary([r["online_time_ratio"] for r in cells if r["available"]], 3),
                    exact_path_count=sum(r["available"] and r.get("all_checked_paths_exact", False) for r in cells) if kind.startswith("curvature") else None,
                    matched_schedule_count=sum(r["available"] and r.get("actual_schedules_match", False) for r in cells) if kind.startswith("batched") else None,
                    rows_saved=summary([r["history_lane_rows_saved"] for r in cells if r["available"]], 3) if kind.startswith("curvature") else None))
    provenance = {}
    for name, folder in (("full", full_root), ("curvature", curve_root), ("sequential", sequential_root)):
        before, after = read_json(folder/"source_hashes_before.json"), read_json(folder/"source_hashes_after.json")
        provenance[name] = dict(recorded_before=before, recorded_after=after,
            unchanged=before == after if before is not None and after is not None else None,
            manifest_sha256=sha256(folder/"manifest.json") if (folder/"manifest.json").exists() else None)
    return dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        expected_full_cells=18, expected_curvature_cells=6, expected_sequential_controls=6,
        manifest_counts=dict(full=len(fm), curvature=len(cm), sequential=len(sm)),
        statuses={backend: dict(Counter(r["status"] for r in records if r["spec"]["backend"] == backend)) for backend in ("full", "curvature")},
        records=records, sequential_controls=list(old.values()), grouped_endpoints=groups,
        curvature_pairs=curvature, sequential_pairs=sequential, paired_groups=paired_groups,
        data_checks=data_checks, provenance=provenance,
        analysis_hashes={str(p.relative_to(ROOT)): sha256(p) for p in (Path(__file__), ROOT/"experiments/shared_ablation_analyze.py")},
        limits=["All endpoint means retain the three predeclared sampler seeds; incomplete means are descriptive only.",
            "Matmul proxy excludes summary queries, classification, additions, scalar functions and other runtime work.",
            "Summary append/query/rebuild coordinate-event counts are structurally reconstructed; unweighted rebuild history rows are terminal-only with prefix bounds.",
            "History row counters count lane-row evaluations, not disk reads; rebuilding shared batch summaries has a separate unweighted history-row counter.",
            "Whole-process memory peaks and persistent bytes refer to whole attempts, not shorter endpoints.",
            "Sequential timing comparison changes batching, cache placement, history representation and draw/reduction order; schedule equality is checked first.",
            "Curvature/full same-seed path checks are numerical evidence on these recorded paths, not a finite-bit theorem.",
            "Both exact mechanisms preserve the ideal posterior kernels and prescribed 2+C_T guarantee; practical caps are not certified schedules.",
            "No comparator was fitted; these are loss/work comparisons, not newly computed regret estimates."])


def write_markdown(report, path):
    lines = ["# Batched and curvature acceptance: fixed-endpoint analysis", "",
        f"Generated {report['generated_utc']}. Manifest counts: {report['manifest_counts']}. Statuses: {report['statuses']}.", "",
        "All means display available counts out of three predeclared sampler seeds on each fixed dataset. Missing cells are retained. "
        "The JSON separately requires all three for a complete_mean; incomplete available means can be runtime selected.", "",
        "## Loss and time", "",
        "| Backend | Dataset | Lanes | Round | n/3 | Raw loss/round | c4 loss/round | Online seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for g in report["grouped_endpoints"]:
        metrics = g["metrics"]
        per = lambda key: metrics[key]["available_mean"]/g["endpoint"] if metrics[key]["available_mean"] is not None else None
        lines.append(f"| {g['backend']} | {g['case']} | {g['lanes']} | {g['endpoint']} | {metrics['loss']['n']}/3 | {fmt(per('loss'))} | {fmt(per('guard_c4_loss'))} | {fmt(metrics['online_seconds']['available_mean'])} |")
    lines += ["", "## Matched comparisons", "",
        "Differences are first named method minus second. The sequential comparison is evaluated with the same c4 guard and checks every "
        "recorded skip and reconstructed old midpoint/move schedule. The curvature comparison checks complete prefix arrays both exactly and with tolerance.", "",
        "| Comparison | Dataset | Round | pairs/3 | Raw loss difference | c4 loss difference | Mean time ratio | Exact paths or matched schedules /3 | Mean history lane-rows saved |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in report["paired_groups"]:
        m = g["differences"]
        verified = g["exact_path_count"] if g["exact_path_count"] is not None else g["matched_schedule_count"]
        rows = g["rows_saved"]["available_mean"] if g["rows_saved"] else None
        lines.append(f"| {g['kind']} | {g['case']} | {g['endpoint']} | {m['loss']['n']}/3 | {fmt(m['loss']['available_mean'])} | {fmt(m['guard_c4_loss']['available_mean'])} | {fmt(g['time_ratio']['available_mean'])} | {verified}/3 | {fmt(rows)} |")
    lines += ["", "## Curvature work, with omitted proxy costs exposed", "",
        "Rows and matmul counts are directly recorded at each endpoint. The three summary coordinate counts follow exact code invariants: "
        "query=8d×proposals, rebuild=d×gradient lane-rows, append=d×sum(actual M). They are coordinate-event counts, not total CPU instructions. "
        "The unweighted summary history scans were not saved per round: exact terminal values are shown where available, otherwise honest bounds.", "",
        "| Dataset | Seed | Round | Full-test lane-rows | Gradient lane-rows | Matmul proxy | Query coordinates | Rebuild coordinates | Append coordinates | Rebuild history rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["records"]:
        if r["spec"]["backend"] != "curvature":
            continue
        for n in ENDPOINTS:
            p = point(r, n)
            if not p["available"]:
                lines.append(f"| {r['spec']['case']} | {r['spec']['seed']} | {n} | unavailable: {p['reason']} | — | — | — | — | — | — |")
                continue
            counts, c = p["counts"], p["summary_coordinate_events"]
            history = p["summary_rebuild_history_rows"]
            h = str(history["value"]) if history["value"] is not None else f"[{history['lower_bound']}, {history['upper_bound']}]"
            lines.append(f"| {r['spec']['case']} | {r['spec']['seed']} | {n} | {counts['full_test_rows']} | {counts['gradient_refresh_rows']} | {counts['estimated_matmul_flops']} | {c['summary_query_coordinates']} | {c['summary_rebuild_coordinates']} | {c['summary_append_coordinates']} | {h} |")
    lines += ["", "Each saved lane-row avoids two d-dimensional score products, represented by 4d in the source's matmul proxy. "
        "That saving does not include the new summary maintenance/query costs. The JSON contains normalized counts per round, sampled round, "
        "move and proposal, actual extra 8Md summary storage, same-path checks and saved-row consistency checks.", "",
        "## Scope", "", *["- "+text for text in report["limits"]], "",
        f"Run validation errors: {sum(bool(r['errors']) for r in report['records'])}. "
        "Every seed-level endpoint, counter, missing reason and validation result is retained in analysis.json."]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-root", type=Path, default=RESULTS/"batched_performance")
    p.add_argument("--curvature-root", type=Path, default=RESULTS/"curvature_performance")
    p.add_argument("--sequential-root", type=Path, default=RESULTS/"shared_antithetic_ablation")
    p.add_argument("--output", type=Path, default=RESULTS/"batched_computation_analysis")
    args = p.parse_args()
    report = analyze(args.full_root.resolve(), args.curvature_root.resolve(), args.sequential_root.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_markdown(report, args.output/"analysis.md")
    print(json.dumps(dict(manifests=report["manifest_counts"], statuses=report["statuses"],
                         validation_errors=sum(bool(r["errors"]) for r in report["records"]), output=str(args.output))))


if __name__ == "__main__":
    main()
