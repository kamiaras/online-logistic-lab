"""Read-only artifact verification after ALL computation workers have stopped.

No sampler, timing experiment, comparator fit, or test runner is invoked. The
only write is COMPUTATION_FIRST_ARTIFACT_AUDIT.json. Prior test execution is
reported with its actual conversational provenance, never synthesized as a log.
"""
from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from .shared_ablation_analyze import ROOT, read_json, sha256, verify_data, read_run, guards
from .batched_performance_analyze import read_performance

RESULTS = ROOT / "experiments/results"
STAGES = {"shared_ablation": (75, 50), "shared_antithetic_ablation": (6, 6),
          "batched_performance": (18, 18), "curvature_performance": (6, 6),
          "computation_followup": (18, 18)}
HISTORICAL = ("experiments/shared_tangent.py", "experiments/shared_tangent_primitives.py",
              "experiments/proximal.py", "experiments/data.py", "experiments/sigmoid.py",
              "experiments/aioli.py")
TEST_FILES = ("test_persistent_log_odds.py", "test_shared_antithetic_ablation.py",
              "test_shared_ablation_analysis.py", "test_batched_persistent_log_odds.py",
              "test_curvature_persistent_log_odds.py", "test_computation_followup.py",
              "test_computation_first_log_odds.py")
CURVATURE_MEASURED_HASH = "191bd20ec2298526112fefa8fbc83a694aefcba37c85ee2b9c95acfdb8a08402"
CURVATURE_METADATA_HASH = "79fee1137bc84eb6c92764256b2fcfd538de7351671eb67296f4b0b46b2272bd"


class Audit:
    def __init__(self):
        self.errors = []

    def check(self, condition, message):
        if not condition:
            self.errors.append(message)
        return bool(condition)


def hash_pair(audit, folder, stem):
    before = read_json(folder / (stem + "_before.json"))
    after = read_json(folder / (stem + "_after.json"))
    equal = audit.check(bool(before) and before == after, f"{folder.name}: {stem} before/after mismatch or missing")
    current, documented_exceptions = {}, []
    for name, expected in (before or {}).items():
        # Manuscripts may receive a separately recorded post-campaign update.
        # All source modules and data files must still equal measured versions.
        if name.endswith((".py", ".bin")):
            path = ROOT / name
            actual = sha256(path) if path.exists() else None
            current[name] = actual == expected
            exception = False
            if (folder.name == "curvature_performance" and name == "experiments/curvature_persistent_log_odds.py"
                    and expected == CURVATURE_MEASURED_HASH and actual == CURVATURE_METADATA_HASH):
                snapshot = ROOT/"experiments/source_snapshots/curvature_performance/curvature_persistent_log_odds.py"
                provenance = read_json(ROOT/"CURVATURE_SOURCE_PROVENANCE.json") or {}
                content = path.read_bytes()
                matching_lines = [line for line in content.splitlines(keepends=True) if line.strip() ==
                    b'memory_representation="raw features, paired states, lane tangents and eight diagonal curvature summaries",']
                exception = (snapshot.exists() and len(matching_lines) == 1
                    and sha256(snapshot) == expected
                    and content.replace(matching_lines[0], b"", 1) == snapshot.read_bytes()
                    and provenance.get("measured_sha256") == expected
                    and provenance.get("current_sha256") == actual)
                documented_exceptions.append(dict(file=name, verified=exception, measured_sha256=expected,
                    current_sha256=actual, archived_source=str(snapshot.relative_to(ROOT)),
                    reason="Exactly one added preflight memory_representation metadata field; byte reconstruction verified",
                    provenance_sha256=sha256(ROOT/"CURVATURE_SOURCE_PROVENANCE.json")))
            audit.check(actual == expected or exception, f"{folder.name}: current frozen source/data changed: {name}")
    return dict(before_after_equal=equal, current_source_data_matches=current,
                documented_metadata_exceptions=documented_exceptions,
                before_file_sha256=sha256(folder/(stem+"_before.json")) if before else None,
                after_file_sha256=sha256(folder/(stem+"_after.json")) if after else None)


def read_followup(spec, folder, audit):
    out = folder / Path(spec["output"]).name
    result, supervisor = read_json(out/"result.json"), read_json(out/"supervisor.json")
    a = None
    errors = []
    if (out/"trajectory.npz").exists():
        try:
            with np.load(out/"trajectory.npz", allow_pickle=False) as archive:
                a = {key: archive[key].copy() for key in archive.files}
            n = len(a["logits"])
            if any(value.shape != (n,) for value in a.values()) or not 0 < n <= min(spec["rounds"], spec["T"]):
                raise ValueError("trajectory shape/endpoint")
            if any(not np.all(np.isfinite(value)) for value in a.values()):
                raise ValueError("nonfinite trajectory")
            labels = np.fromfile(Path(spec["data_path"])/"y.bin", dtype="i1", count=n)
            if not np.array_equal(labels, a["labels"]):
                raise ValueError("input labels")
            raw = np.logaddexp(0., -labels*a["logits"])
            c4, loss4 = guards(a["logits"], labels, 4)
            for key, expected in (("losses", raw), ("guarded_logits", c4), ("guarded_losses", loss4)):
                if not np.allclose(a[key], expected, rtol=1e-12, atol=1e-10):
                    raise ValueError(key+" recomputation")
            tguard = np.arange(1, n+1, dtype=float)
            if np.any(loss4-raw > 1/(4*tguard*(tguard+1))+1e-9):
                raise ValueError("c4 pointwise loss inequality")
            cumulative = [key for key in a if key not in (
                "logits", "guarded_logits", "losses", "guarded_losses", "labels", "N_actual", "inveta", "mu", "skipped")]
            for key in cumulative:
                if np.any(a[key] < 0) or np.any(np.diff(a[key]) < 0):
                    raise ValueError("nonmonotone "+key)
            if not np.all(a["N_actual"] == 1) or not np.all(np.isin(a["skipped"], [0, 1])):
                raise ValueError("actual transitions/skip flags")
            m = 256 if spec["method"] == "original256" else 16
            expected_moves = np.cumsum(np.where(a["skipped"], 0, m*a["N_actual"]))
            if not np.array_equal(expected_moves, a["proximal_steps"]):
                raise ValueError("actual population/move/skip ledger")
            X = np.fromfile(Path(spec["data_path"])/"X.bin", dtype="<f8", count=n*spec["d"]).reshape(n, spec["d"])
            energy = np.cumsum(np.sum(X*X, axis=1))/4
            energy1 = np.cumsum(np.sum(np.abs(X), axis=1)**2)/4
            lam = spec["B"]**-2
            P = np.maximum(energy1, 2*energy) if spec["method"] == "current16" else 2*spec["d"]*energy
            if not np.allclose(a["inveta"], lam+P, rtol=1e-12, atol=1e-10):
                raise ValueError("proximal step formula")
            if not np.allclose(a["mu"], 2*lam+P, rtol=1e-12, atol=1e-10):
                raise ValueError("mu formula")
            if result:
                if result["round"] != n or result["actual_population"] != m:
                    raise ValueError("terminal round/population")
                for key, expected in (("cumulative_loss", raw.sum()), ("guarded_loss", loss4.sum())):
                    if not math.isclose(result[key], expected, rel_tol=1e-11, abs_tol=1e-8):
                        raise ValueError("terminal "+key)
                if result["status"] in ("completed", "time_budget"):
                    for key in cumulative:
                        if key in ("online_times", "original_gradient_feature_lane_rows_derived"):
                            continue
                        if a[key][-1] != result["metrics"]["counters"].get(key, 0):
                            raise ValueError("terminal native counter "+key)
                if spec["method"] == "original256":
                    if result["algorithm_class"] != "ProximalLogOdds":
                        raise ValueError("original core class")
                    t = np.arange(1, n+1)
                    k = np.maximum(0, np.ceil(np.log(np.maximum(1., 6*t/np.sqrt(a["mu"])))/np.log(3.)))
                    derived = np.cumsum((k+2)*m*a["N_actual"]*t)
                    if not np.array_equal(a["original_gradient_feature_lane_rows_derived"], derived):
                        raise ValueError("original derived full-gradient row ledger")
        except (KeyError, ValueError, OSError) as exc:
            errors.append(str(exc))
    else:
        n = 0
    audit.errors.extend(f"{out.name}: {e}" for e in errors)
    return dict(id=out.name, spec=spec, result=result, supervisor=supervisor,
                status=(result or {}).get("status", "abnormal_exit" if supervisor else "not_started"),
                trajectory_rounds=len(a["logits"]) if a is not None else 0,
                trajectory_valid=a is not None and not errors, errors=errors), a


def discover_tests():
    rows = []
    for name in TEST_FILES:
        path = ROOT/"experiments"/name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions, cases = 0, 0
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            functions += 1
            multiple = 1
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "parametrize":
                    multiple *= len(ast.literal_eval(dec.args[1]))
            cases += multiple
        rows.append(dict(file=str(path.relative_to(ROOT)), functions=functions,
                         statically_declared_cases=cases, sha256=sha256(path)))
    return dict(files=rows, statically_declared_cases=sum(r["statically_declared_cases"] for r in rows),
        prior_execution_report=dict(prior_root_reported_passes=37, followup_agent_reported_passes=5,
            selected_class_root_reported_passes=6, total_reported_passes=48,
            selected_class_command="python -m pytest experiments/test_computation_first_log_odds.py -q",
            selected_class_reported_exit_code=0, selected_class_reported_seconds=.30,
            source="Parent and implementation-agent messages reporting their tool executions",
            independently_reexecuted_by_this_audit=False, machine_execution_log_created=False),
        limitation="AST enumeration validates present declarations, not whether or when tests executed. Prior pass counts retain conversational provenance.")


def audit_ledger(audit, filename):
    ledger = read_json(ROOT/filename) or {}
    rows = ledger.get("rows", [])
    audit.check(len(rows) == 6, filename+": arithmetic ledger row count")
    for row in rows:
        path = Path(row["source_result"])
        audit.check(sha256(path) == row["source_result_sha256"], row["run"]+": ledger source hash")
        result = read_json(path)
        c, d, m, n = result["metrics"]["counters"], row["d"], row["M"], result["round"]
        q, g, f, app, saved = (c["summary_queries"], c["summary_rebuild_lane_rows"],
            c["summary_rebuilds"], c["summary_append_lane_rows"], c["history_rows_avoided_by_squeeze"])
        audit.check((row["Q"], row["G"], row["F"], row["append_lane_rows"], row["S"]) == (q, g, f, app, saved), row["run"]+": ledger native count")
        audit.check(app == m*n and q == c["rqo_proposals"] and f == c["rqo_rejections"] and g == c["gradient_refresh_rows"], row["run"]+": ledger structural identities")
        coarse = 100*(d+1)*(q+g+app)+16*m*d
        tight = (40*d+96)*q+(10*d+16)*g+(16*d+1)*f+(6*d+16)*app+16*m*d
        lower = (4*d-2)*saved
        for key, expected in dict(extra_upper_coarse=coarse, extra_upper_tight=tight,
            saved_dot_operations_lower=lower, guaranteed_reduction_coarse=lower-coarse,
            guaranteed_reduction_tight=lower-tight).items():
            audit.check(row[key] == expected, row["run"]+": ledger formula "+key)
    return dict(file=filename, rows_checked=len(rows), sha256=sha256(ROOT/filename),
        paired_full_measurement=(filename == "CURVATURE_SUMMARY_ARITHMETIC_LEDGER.json"),
        comparison="six separately measured common-stream full/curvature pairs" if filename.startswith("CURVATURE")
            else "counterfactual common-stream full evaluator; not separately timed in follow-up")


def main():
    audit = Audit()
    manifests = {name: read_json(RESULTS/name/"manifest.json") or [] for name in STAGES}
    # Source/result files are only inspected after all declared measurement
    # stages have their terminal campaign records; running this early is error.
    for name in STAGES:
        terminal = RESULTS/name/("continuation_result.json" if name == "shared_ablation" else "campaign_result.json")
        audit.check(terminal.exists(), name+": missing terminal campaign record")
    data = {p: verify_data(p) for p in sorted({s["data_path"] for specs in manifests.values() for s in specs})}
    for name, check in data.items():
        audit.errors.extend(name+": "+e for e in check["errors"])
    controls = manifests["shared_antithetic_ablation"]
    control_keys = {(s["case"], s["d"], s["B"], s["T"], s["seed"], s["data_path"]) for s in controls}
    audit.check(len(control_keys) == 6 and {s["d"] for s in controls} == {10, 100}
                and {s["seed"] for s in controls} == {910001, 910002, 910003},
                "six fixed comparative controls do not match declared dimensions/seeds")
    for stage, levels in (("batched_performance", {(16, 4), (64, 1), (256, 1)}),
                          ("curvature_performance", {(64, 1)}),
                          ("computation_followup", {"current16", "legacy16", "original256"})):
        actual = set()
        for s in manifests[stage]:
            case_key = (s["case"], s["d"], s["B"], s["T"], s["seed"], s["data_path"])
            level = s["method"] if stage == "computation_followup" else (s["config"]["lanes"], s["config"]["transitions"])
            actual.add((case_key, level))
            audit.check(s["rounds"] == 500 and s["T"] == 1000 and s["seconds"] == 90., stage+": changed fixed endpoint/budget")
        audit.check(actual == {(key, level) for key in control_keys for level in levels}, stage+": comparative design cells changed")
    stages, records, arrays = {}, {}, {}
    for name, (planned, expected_attempts) in STAGES.items():
        folder, specs = RESULTS/name, manifests[name]
        audit.check(len(specs) == planned, name+": planned manifest count")
        ids = [Path(s["output"]).name for s in specs]
        audit.check(len(set(ids)) == len(ids), name+": duplicate output IDs")
        rows = []
        for spec in specs:
            if name in ("shared_ablation", "shared_antithetic_ablation"):
                record, a = read_run(spec, folder, data)
            elif name == "computation_followup":
                record, a = read_followup(spec, folder, audit)
            else:
                record, a = read_performance(spec, folder, data)
            rid, sup, result = record["id"], record.get("supervisor"), record.get("result")
            records[rid], arrays[rid] = record, a
            audit.errors.extend(rid+": "+e for e in record.get("errors", [])+record.get("read_errors", []))
            if sup:
                audit.check(isinstance(sup.get("returncode"), int), rid+": nonterminal supervisor")
                audit.check(sup.get("hard_process_commit_limit") == 2*1024**3 and sup.get("os_hard_limit_active") is True,
                            rid+": hard memory gate metadata")
                audit.check(sup.get("wall_seconds", -1) >= 0, rid+": invalid process wall time")
            else:
                audit.check(not result and a is None, rid+": started artifacts without terminal supervisor")
            if result:
                audit.check(result.get("configured_horizon") == spec["T"], rid+": configured horizon mismatch")
                audit.check(result.get("fixed_endpoint") == spec["rounds"], rid+": fixed endpoint mismatch")
                if result.get("status") == "completed":
                    audit.check(record["trajectory_valid"] and record["trajectory_rounds"] == spec["rounds"], rid+": invalid completion")
                    audit.check(sup is not None and sup.get("returncode") == 0 and sup.get("reason") is None,
                                rid+": completed worker has non-success supervisor")
                if a is not None and len(a["online_times"]):
                    audit.check(result.get("online_seconds", -1)+1e-6 >= float(a["online_times"][-1]), rid+": total online time below trajectory")
                audit.check(all(pool.get("num_threads") == 1 for pool in result.get("threadpools", [])), rid+": multithreaded numerical pool")
            rows.append(dict(id=rid, status=record["status"], terminal_supervisor=bool(sup),
                trajectory_rounds=record["trajectory_rounds"], trajectory_valid=record["trajectory_valid"],
                result_sha256=sha256(folder/rid/"result.json") if result else None,
                trajectory_sha256=sha256(folder/rid/"trajectory.npz") if a is not None else None))
        attempted = sum(r["terminal_supervisor"] for r in rows)
        audit.check(attempted == expected_attempts, name+": terminal attempt count")
        stages[name] = dict(planned=planned, expected_attempts=expected_attempts, actual_attempts=attempted,
            unstarted=planned-attempted, statuses=dict(Counter(r["status"] for r in rows)), runs=rows,
            source_integrity=hash_pair(audit, folder, "source_hashes"))
    stages["computation_followup"]["data_integrity"] = hash_pair(audit, RESULTS/"computation_followup", "data_hashes")
    baseline = read_json(RESULTS/"shared_ablation/source_hashes_before.json") or {}
    historical = {p: sha256(ROOT/p) == baseline.get(p) for p in HISTORICAL}
    audit.check(all(historical.values()), "historical algorithm/data source hash changed")
    targets = [s for s in manifests["shared_ablation"] if s["stage"] == "primary" and (s["d"] == 10 or s["seed"] <= 910003)]
    continuation = read_json(RESULTS/"shared_ablation/continuation_manifest.json") or {}
    audit.check(continuation.get("targets") == targets, "continuation changed original target specs/order")
    reached = sum(records[Path(s["output"]).name]["trajectory_valid"] and
                  records[Path(s["output"]).name]["trajectory_rounds"] >= 500 for s in targets)
    audit.check(len(targets) == reached == 36, "amended primary target is not 36 valid 500-round trajectories")

    matches = []
    for cs in manifests["curvature_performance"]:
        fs = next((s for s in manifests["batched_performance"] if
            (s["case"], s["seed"], s["config"]["lanes"], s["config"]["transitions"]) ==
            (cs["case"], cs["seed"], 64, 1)), None)
        if not audit.check(fs is not None, "missing full curvature control"):
            continue
        cid, fid = Path(cs["output"]).name, Path(fs["output"]).name
        ca, fa = arrays[cid], arrays[fid]
        exact = {}
        if ca is None or fa is None:
            audit.check(False, cid+": missing paired trajectory")
            continue
        for key in ("logits", "guarded_logits", "skipped", "lanes", "transitions", "moves", "proposals", "gradient_refresh_rows"):
            exact[key] = bool(np.array_equal(ca[key], fa[key]))
        audit.check(all(exact.values()), cid+": same-stream state/work path mismatch")
        saved = fa["full_test_rows"]-ca["full_test_rows"]
        c = records[cid]["result"]["metrics"]["counters"]
        audit.check(np.all(saved >= 0) and np.all(np.diff(saved) >= 0), cid+": invalid saved-row prefix")
        audit.check(int(saved[-1]) == c["history_rows_avoided_by_squeeze"], cid+": saved-row terminal identity")
        audit.check(np.array_equal(fa["estimated_matmul_flops"]-ca["estimated_matmul_flops"], 4*cs["d"]*saved), cid+": saved dot proxy identity")
        matches.append(dict(curvature=cid, full=fid, exact=exact, saved_rows=int(saved[-1]),
                            saved_rows_at_endpoints={str(n): int(saved[n-1]) for n in (100, 250, 500)}))

    ledgers = [audit_ledger(audit, name) for name in
               ("CURVATURE_SUMMARY_ARITHMETIC_LEDGER.json", "COMPUTATION_FIRST_ARITHMETIC_LEDGER.json")]
    tests = discover_tests()
    audit.check(tests["statically_declared_cases"] == 48, "current declared test count differs from reported 48")
    total = sum(s["actual_attempts"] for s in stages.values())
    audit.check(total == 98, "combined terminal attempt count differs from 98")
    report = dict(generated_utc=datetime.now(timezone.utc).isoformat(), passed=not audit.errors,
        errors=audit.errors, error_count=len(audit.errors), scope="Read-only validation; no worker, comparator fit or test execution",
        planned_manifest_cells=sum(s["planned"] for s in stages.values()), terminal_attempts=total,
        unstarted_original_cells=stages["shared_ablation"]["unstarted"], stages=stages,
        historical_current_hashes_match=historical, data=data,
        amended_primary=dict(targets=len(targets), valid_500_round_trajectories=int(reached)),
        curvature_full_exact_pairs=matches, arithmetic_ledgers=ledgers, tests=tests,
        selected_implementation=dict(file="experiments/computation_first_log_odds.py",
            sha256=sha256(ROOT/"experiments/computation_first_log_odds.py"),
            tests_sha256=sha256(ROOT/"experiments/test_computation_first_log_odds.py"),
            step="inveta=lambda+2*d*E", practical_defaults=dict(lanes=16, transitions=1),
            static_review="Counts, uniform bound, budgets, memory constraints, zero-R handling and practical equivalence audited independently",
            production_measurements_of_this_new_class=0,
            measured_equivalent_control="LegacyStepCurvatureLogOdds practical16/1; selected class additionally supports matching prescribed counts"),
        interpretation="Practical caps have no prescribed regret certificate. Exact-real operation bounds and observed float paths are distinct from finite-bit certification. Follow-up selection occurred after initial performance effects.")
    destination = ROOT/"COMPUTATION_FIRST_ARTIFACT_AUDIT.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(output=str(destination), passed=report["passed"], errors=audit.errors,
                         terminal_attempts=total, primary500=int(reached))))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
