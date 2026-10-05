"""Pure planning and serialization tests: no arrays, predictors, or workers."""
import json

import pytest

from .memory_frontier import (
    Axis, Frontier, MEMORY_BYTES, build_spec, classify_attempt, default_axes,
    failure_phase, run_campaign, reclassify_campaign, _source_hashes,
    _check_resume_sources, OPENBLAS_ALLOCATION_FAILURE,
    _validate_probe_against_policy, _active_measurement_pids,
)


def _record(value, outcome, role="expand", initialized=True, phase=None):
    return dict(value=value, outcome=outcome, role=role,
                initialization_completed=initialized, failure_phase=phase)


def test_active_guard_excludes_only_self_and_identical_argument_launcher_ancestors():
    args = ["-m", "experiments.memory_frontier", "--tag", "deep", "--reclassify"]
    processes = [
        {"pid": 30, "ppid": 20, "cmdline": ["base/python.exe", *args]},
        {"pid": 20, "ppid": 10, "cmdline": ["venv/python.exe", *args]},
        # An ancestor running a different command is not a launcher exemption.
        {"pid": 10, "ppid": 1, "cmdline": ["python.exe", "-m", "experiments.runner", "other-manifest.json"]},
        {"pid": 40, "ppid": 1, "cmdline": ["python.exe", "-m", "experiments.worker", "spec.json"]},
        # Identical arguments outside the ancestry must still block repair.
        {"pid": 50, "ppid": 1, "cmdline": ["venv/python.exe", *args]},
        {"pid": 60, "ppid": 1, "cmdline": ["python.exe", "-m", "experiments.history_frontier", "--worker", "spec.json"]},
    ]
    assert _active_measurement_pids(processes, 30) == [10, 40, 50, 60]
    assert _active_measurement_pids(processes[:2], 30) == []
    # If the ancestry/command evidence is absent, do not waive another process.
    assert _active_measurement_pids(processes[1:], 30) == [10, 20, 40, 50, 60]


def test_frontier_refines_only_observed_success_and_memory_failure():
    frontier = Frontier(Axis("x", "aioli", "dimension", 10, 80), refinements=2)
    assert frontier.next_probe() == (10, "expand")
    frontier.add(_record(10, "completed"))
    assert frontier.next_probe() == (20, "expand")
    frontier.add(_record(20, "time_censored"))
    assert frontier.completed_lower == 10
    assert frontier.next_probe() == (40, "expand")
    frontier.add(_record(40, "memory_failure"))
    assert frontier.next_probe() == (25, "refine")
    frontier.add(_record(25, "completed", role="refine"))
    assert frontier.next_probe() == (32, "refine")
    frontier.add(_record(32, "memory_failure", role="refine"))
    assert frontier.next_probe() is None
    assert frontier.summary()["observed_bracket"] == [25, 32]


def test_first_memory_failure_seeks_completed_lower_bound():
    frontier = Frontier(Axis("x", "ours", "particles", 16, 64))
    frontier.add(_record(16, "memory_failure", initialized=False, phase="initialization"))
    assert frontier.next_probe() == (8, "backoff")
    frontier.add(_record(8, "completed", role="backoff"))
    assert frontier.next_probe() == (12, "refine")
    assert frontier.summary()["smallest_initialization_memory_failure"] == 16


def test_numerical_failures_are_not_memory_boundaries():
    frontier = Frontier(Axis("x", "ours", "particles", 16, 64))
    frontier.add(_record(16, "numerical_failure"))
    assert frontier.next_probe() is None
    assert frontier.memory_upper is None


def test_probe_configs_preserve_os_limit_and_selected_dimension_settings(tmp_path):
    axes = {a.name: a for a in default_axes()}
    ours = build_spec(axes["ours_dimension"], 200000, tmp_path)
    sig = build_spec(axes["sigmoid_dimension"], 200000, tmp_path)
    pop = build_spec(axes["ours_particles"], 1048576, tmp_path)
    assert ours["config"]["particles"] == 256
    assert ours["config"]["transitions"] == 1
    assert sig["config"]["particles"] == 16
    assert sig["config"]["transitions"] == 16
    assert pop["config"]["particles"] == 1048576
    assert pop["config"]["max_particles"] == 1048576
    assert pop["config"]["max_state_bytes"] > MEMORY_BYTES
    assert all(s["memory_bytes"] == MEMORY_BYTES for s in (ours, sig, pop))
    assert all(s["T"] == 4 and s["stop_after_round"] == 2 for s in (ours, sig, pop))
    assert ours["data_path"] == sig["data_path"]
    assert ours["output"] != sig["output"]


@pytest.mark.parametrize("call,phase", [
    ("a = cls(d=d, B=B)", "initialization"),
    ("score=float(a.predict(x))", "predict"),
    ("a.observe(x, int(y))", "observe"),
    ("worker killed without traceback", "unknown"),
])
def test_traceback_phase_identification(call, phase):
    assert failure_phase(call) == phase


def test_failed_constructor_record_survives_without_algorithm_state(tmp_path):
    axis = default_axes()[0]
    spec = build_spec(axis, 16000, tmp_path)
    supervisor = {"worker": {"round": 0, "status": "memory_error",
                             "state_bytes": 0, "diagnostics": {},
                             "initialization_seconds": .2, "error": "MemoryError()"},
                  "monitored_peak_private": 123, "monitored_peak_rss": 80,
                  "hard_process_commit_limit": MEMORY_BYTES,
                  "os_hard_limit_active": True, "returncode": 0}
    row = classify_attempt(spec, supervisor, "a = cls(d=d)\nMemoryError")
    assert row["outcome"] == "memory_failure"
    assert row["failure_phase"] == "initialization"
    assert not row["initialization_completed"]
    assert row["peak_private_commit"] == 123
    json.dumps(row, allow_nan=False)


def test_supervisor_kill_does_not_invent_failure_phase(tmp_path):
    spec = build_spec(default_axes()[0], 16000, tmp_path)
    row = classify_attempt(spec, {"worker": {"round": 0},
                                  "termination": "memory_limit"})
    assert row["outcome"] == "memory_failure"
    assert row["failure_phase"] == "unknown"


def test_campaign_fake_runner_retains_every_attempt_and_resumes(tmp_path):
    folder = tmp_path / "memory_frontier_pure_test"
    axis = Axis("aioli_dimension", "aioli", "dimension", 2, 8)
    calls = []
    def generate(path, case):
        # No data generation in this pure test.
        calls.append(("data", case["d"]))
    def runner(spec, remaining):
        value = spec["frontier"]["value"]
        calls.append(("worker", value))
        if value >= 4:
            return {"worker": {"status": "memory_error", "round": 0,
                                "error": "MemoryError()", "state_bytes": 0},
                    "termination": None, "process_wall_seconds": .1}
        return {"worker": {"status": "evaluation_completed", "round": 2,
                            "state_bytes": 128, "diagnostics": {"matrix_backend": "cholesky"}},
                "termination": None, "process_wall_seconds": .1}
    kwargs = dict(runner=runner, generate=generate, environment_reader=lambda: {"fake": True})
    result = run_campaign(folder, [axis], **kwargs)
    assert result["attempts"] == 3
    records = [json.loads(line) for line in (folder / "frontier_records.jsonl").read_text().splitlines()]
    assert [r["value"] for r in records] == [2, 4, 3]
    assert [r["outcome"] for r in records] == ["completed", "memory_failure", "completed"]
    assert len(json.loads((folder / "manifest.json").read_text())) == 3
    before = len(calls)
    result2 = run_campaign(folder, [axis], resume=True, **kwargs)
    assert result2["attempts"] == 3 and len(calls) == before
    with pytest.raises(ValueError):
        run_campaign(folder, [axis], **kwargs)
    with pytest.raises(ValueError):
        run_campaign(folder, [axis], resume=True, rounds=1, **kwargs)


def test_protected_stage_names_rejected(tmp_path):
    with pytest.raises(ValueError):
        run_campaign(tmp_path / "screen", [default_axes()[0]],
                     runner=lambda *_: None, generate=lambda *_: None,
                     environment_reader=lambda: {})


def test_explicit_native_allocation_failure_preserves_unknown_phase_and_censoring(tmp_path):
    spec = build_spec(default_axes()[0], 16000, tmp_path)
    supervisor = {"worker": {"round": 0, "status": "abnormal_exit", "online_seconds": 0},
                  "returncode": 1, "termination": None,
                  "monitored_peak_private": 806912, "monitored_peak_rss": 4366336,
                  "process_wall_seconds": .61}
    row = classify_attempt(spec, supervisor, stderr_text=OPENBLAS_ALLOCATION_FAILURE + "\n")
    assert row["outcome"] == "memory_failure"
    assert row["failure_phase"] == "unknown"
    assert row["native_allocation_failure"]
    assert row["native_allocation_evidence"] == [OPENBLAS_ALLOCATION_FAILURE]
    assert row["native_failure_phase_not_inferred"]
    assert row["worker_status"] == "abnormal_exit" and row["returncode"] == 1
    assert row["stderr_path"].endswith("stderr.txt") and len(row["stderr_text_sha256"]) == 64
    assert row["peak_private_commit"] == 806912
    assert row["memory_peak_is_lower_bound"] and row["memory_peak_source"] == "supervisor_monitor_only"
    assert row["timing_measurement_status"] == row["state_measurement_status"] == "unavailable"
    assert all(row[k] is None for k in ("initialization_seconds", "prediction_seconds", "update_seconds", "online_seconds", "state_bytes"))
    assert row["process_wall_seconds"] == .61
    assert not row["initialization_completed"]
    # Neither a generic BLAS warning nor a successful process supplies this evidence.
    other = classify_attempt(spec, supervisor, stderr_text="OpenBLAS warning: thread allocation failed")
    assert other["outcome"] == "other_failure" and not other["native_allocation_failure"]
    zero = classify_attempt(spec, {**supervisor, "returncode": 0}, stderr_text=OPENBLAS_ALLOCATION_FAILURE)
    assert not zero["native_allocation_failure"]


def test_censored_checkpoint_retains_values_with_explicit_flags(tmp_path):
    spec = build_spec(default_axes()[0], 16000, tmp_path)
    sup = {"worker": {"round": 1, "status": "external_timeout", "online_seconds": 3.,
                       "state_bytes": 128, "memory": {"peak_private_commit": 1000}},
           "termination": "external_timeout", "monitored_peak_private": 1100}
    row = classify_attempt(spec, sup)
    assert row["outcome"] == "time_censored" and row["online_seconds"] == 3.
    assert row["state_bytes"] == 128 and row["timing_measurement_status"] == "checkpoint_censored"
    assert row["worker_checkpoint_available"] and row["completed_rounds_censored"]
    assert row["memory_peak_is_lower_bound"]


def test_reclassification_archives_derivatives_preserves_raw_and_authorizes_only_classifier(tmp_path, monkeypatch):
    folder = tmp_path / "memory_frontier_repair_test"
    folder.mkdir()
    axis = default_axes()[0]
    policy = {"axes": [vars(axis)], "refinements_per_axis": 3,
              "relative_tolerance": .1, "max_probes": 40, "stage": folder.name,
              "B": 1000., "T": 4, "rounds": 2, "seconds_per_worker": 20.,
              "memory_bytes": MEMORY_BYTES}
    original_sources = {**_source_hashes(), "memory_frontier.py": "historical-classifier"}
    write = lambda path, value: path.write_text(json.dumps(value), encoding="utf-8")
    write(folder / "policy.json", policy)
    write(folder / "source_hashes.json", original_sources)
    specs, records, raw_paths = [], [], [folder / "policy.json", folder / "source_hashes.json"]
    for sequence, value in enumerate((8000, 16000), 1):
        spec = build_spec(axis, value, folder)
        out = tmp_path / "memory_frontier_repair_test" / "aioli_dimension" / f"dimension_{value}"
        out.mkdir(parents=True)
        if value == 8000:
            worker = {"round": 2, "status": "evaluation_completed", "online_seconds": 1.,
                      "memory": {"peak_private_commit": 1000}, "state_bytes": 128}
            sup = {"worker": worker, "returncode": 0, "termination": None}
            stderr = ""
        else:
            sup = {"worker": {"round": 0, "status": "abnormal_exit", "online_seconds": 0},
                   "returncode": 1, "termination": None, "monitored_peak_private": 806912}
            stderr = OPENBLAS_ALLOCATION_FAILURE
        write(out / "spec.json", spec)
        write(out / "supervisor.json", sup)
        (out / "stderr.txt").write_text(stderr, encoding="utf-8")
        raw_paths.extend([out / "spec.json", out / "supervisor.json", out / "stderr.txt"])
        # Model the previous classifier, which did not consume stderr.
        records.append({**classify_attempt(spec, sup), "sequence": sequence,
                        "role": "expand", "recorded_utc": "original-time"})
        specs.append(spec)
    write(folder / "manifest.json", specs)
    raw_paths.append(folder / "manifest.json")
    records_path = folder / "frontier_records.jsonl"
    records_path.write_text("".join(json.dumps(r)+"\n" for r in records))
    write(folder / "frontier_summary.json", {"attempts": 2, "status": "frontiers_finished", "axes": []})
    old_bytes = records_path.read_bytes()
    immutable = {p: p.read_bytes() for p in raw_paths}
    with pytest.raises(ValueError):
        _check_resume_sources(folder)
    original_builder = build_spec
    with monkeypatch.context() as patch:
        def changed_builder(*args, **kwargs):
            return {**original_builder(*args, **kwargs), "seconds": 21.}
        # Source identity still reports only memory_frontier.py changed. That
        # fact must not authorize changed numerical specifications.
        patch.setattr("experiments.memory_frontier.build_spec", changed_builder)
        with pytest.raises(ValueError, match="build_spec changes"):
            reclassify_campaign(folder, process_guard=lambda: None)
    assert records_path.read_bytes() == old_bytes
    assert not (folder / "classification_audit.json").exists()
    result = reclassify_campaign(folder, process_guard=lambda: None)
    assert result["classification_changes"] == 1
    assert result["next_probes"]["aioli_dimension"] == (12000, "refine")
    repaired = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert repaired[-1]["outcome"] == "memory_failure" and repaired[-1]["failure_phase"] == "unknown"
    assert repaired[-1]["online_seconds"] is None
    assert all(r["recorded_utc"] == "original-time" for r in repaired)
    from pathlib import Path
    assert (Path(result["archive"]) / "frontier_records.jsonl").read_bytes() == old_bytes
    assert all(p.read_bytes() == value for p, value in immutable.items())
    audit = json.loads((folder / "classification_audit.json").read_text())
    assert audit["status"] == "completed" and not audit["raw_worker_artifacts_modified"]
    assert audit["repairs"][-1]["classification_changes"] == 1
    event = audit["repairs"][-1]
    assert event["recorded_specs_verified_unchanged"] == 2
    assert not event["machine_proof_of_source_equivalence"]
    assert Path(event["module_source_path"]).exists()
    assert json.loads(Path(event["archived_policy_path"]).read_text()) == policy
    _check_resume_sources(folder)
    # A builder branch changing only a future value is checked immediately
    # before dispatch against an independently derived policy/template spec.
    future = original_builder(axis, 12000, folder)
    _validate_probe_against_policy(future, folder, policy, specs[0])
    with pytest.raises(ValueError, match="Probe settings differ"):
        _validate_probe_against_policy({**future, "seconds": 21.}, folder, policy, specs[0])
    changed_algorithm = {**_source_hashes(), "aioli.py": "changed-algorithm"}
    monkeypatch.setattr("experiments.memory_frontier._source_hashes", lambda: changed_algorithm)
    with pytest.raises(ValueError):
        _check_resume_sources(folder)
    with pytest.raises(ValueError):
        reclassify_campaign(folder, process_guard=lambda: None)
