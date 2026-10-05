"""Adaptive memory probes using the unchanged, isolated experiment worker.

Example (only run after other measured campaigns are idle):
    python -m experiments.memory_frontier --tag depth1 --wall-budget 540

Five independent axes vary actual algorithm parameters. No dummy allocation is
used. Success means completion of the requested short online probe; timeouts are
censored, not memory failures. Refinement gives an observed bracket, not a proof
of a deterministic capacity threshold. This is not a regret comparison.

Importing this module and its pure planner tests never imports a predictor,
allocates datasets, or launches workers.
"""
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from time import perf_counter


BASE = Path(__file__).resolve().parent
MEMORY_BYTES = 2 * 1024**3
AXIS_NAMES = ("aioli_dimension", "ours_particles", "sigmoid_particles",
              "ours_dimension", "sigmoid_dimension")
CLASSIFICATION_VERSION = 2
OPENBLAS_ALLOCATION_FAILURE = "OpenBLAS error: Memory allocation still failed after 10 retries, giving up."


@dataclass(frozen=True)
class Axis:
    name: str
    algorithm: str
    parameter: str
    start: int
    maximum: int
    fixed_dimension: int = 128
    minimum: int = 1
    growth: int = 2


def default_axes(particle_dimension=128):
    return [
        Axis("aioli_dimension", "aioli", "dimension", 8000, 32768, minimum=128),
        Axis("ours_particles", "ours", "particles", 131072, 8388608,
             fixed_dimension=particle_dimension),
        Axis("sigmoid_particles", "sigmoid", "particles", 131072, 8388608,
             fixed_dimension=particle_dimension),
        Axis("ours_dimension", "ours", "dimension", 50000, 800000, minimum=1000),
        Axis("sigmoid_dimension", "sigmoid", "dimension", 50000, 800000, minimum=1000),
    ]


@dataclass
class Frontier:
    axis: Axis
    refinements: int = 1
    relative_tolerance: float = 0.1
    history: list = field(default_factory=list)

    @property
    def completed_lower(self):
        values = [r["value"] for r in self.history if r["outcome"] == "completed"]
        return max(values) if values else None

    @property
    def memory_upper(self):
        values = [r["value"] for r in self.history if r["outcome"] == "memory_failure"]
        return min(values) if values else None

    def next_probe(self):
        """Return (integer value, role), or None; never promote a timeout."""
        seen = {r["value"] for r in self.history}
        if not seen:
            return self.axis.start, "expand"
        if any(r["outcome"] not in ("completed", "memory_failure", "time_censored")
               for r in self.history):
            return None
        lower, upper = self.completed_lower, self.memory_upper
        if upper is None:
            last = max(seen)
            if last >= self.axis.maximum:
                return None
            return min(self.axis.maximum, max(last + 1, last * self.axis.growth)), "expand"
        if lower is None:
            # A first failure is not a bracket: seek a genuinely completed probe.
            candidate = min(seen) // self.axis.growth
            candidate = max(self.axis.minimum, candidate)
            return None if candidate in seen else (candidate, "backoff")
        if lower >= upper:  # Observations contradict monotonicity.
            return None
        if upper - lower <= max(1, self.relative_tolerance * lower):
            return None
        used = sum(r["role"] == "refine" for r in self.history)
        if used >= self.refinements:
            return None
        candidate = (lower + upper) // 2
        if candidate in seen:
            return None
        return candidate, "refine"

    def add(self, record):
        if any(r["value"] == record["value"] for r in self.history):
            raise ValueError("A frontier parameter was recorded twice")
        self.history.append(record)

    def summary(self):
        lower, upper = self.completed_lower, self.memory_upper
        initialized = [r["value"] for r in self.history if r["initialization_completed"]]
        init_failures = [r["value"] for r in self.history
                         if r["outcome"] == "memory_failure"
                         and r["failure_phase"] == "initialization"]
        return {
            "axis": self.axis.name, "algorithm": self.axis.algorithm,
            "parameter": self.axis.parameter, "attempts": len(self.history),
            "largest_completed_probe": lower, "smallest_memory_failure": upper,
            "largest_reported_initialized": max(initialized) if initialized else None,
            "smallest_initialization_memory_failure": min(init_failures) if init_failures else None,
            "observed_bracket": [lower, upper] if lower is not None and upper is not None
                                and lower < upper else None,
            "observed_nonmonotonicity": lower is not None and upper is not None and lower >= upper,
            "time_censored_probes": sum(r["outcome"] == "time_censored" for r in self.history),
            "next_probe": self.next_probe(),
            "refinement_assumes_approximate_memory_monotonicity": True,
        }


def build_spec(axis, value, folder, *, T=4, rounds=2, B=1000.0,
               seconds=20.0, data_seed=78001, sampler_seed=878001):
    """A normal runner/worker specification; the process cap is immutable."""
    folder = Path(folder).resolve()
    d = value if axis.parameter == "dimension" else axis.fixed_dimension
    if axis.algorithm == "aioli":
        config, particles = {"matrix_backend": "cholesky"}, None
    else:
        particles = value if axis.parameter == "particles" else (
            256 if axis.algorithm == "ours" else 16)
        config = {"particles": particles, "transitions": 1 if axis.algorithm == "ours" else 16}
        if axis.algorithm == "ours":
            config.update(mode="practical", particle_batch=64,
                          max_particles=particles, max_state_bytes=2**63 - 1)
        elif axis.parameter == "particles":
            # Two rounds include a real MALA refresh; one transition suffices
            # to expose its temporary arrays without multiplying probe time.
            config["transitions"] = 1
    cid = f"signal_d{d}_B{B:g}_T{T}_s{data_seed}"
    label = f"{axis.parameter}_{value}"
    return {
        "algorithm": axis.algorithm, "config": config, "d": d, "B": B, "T": T,
        "stop_after_round": rounds, "sampler_seed": sampler_seed,
        "data_seed": data_seed, "case_id": cid,
        "data_path": str(folder / "data" / cid),
        "output": str(folder / axis.name / label),
        "seconds": seconds, "memory_bytes": MEMORY_BYTES,
        "stage": folder.name, "label": label,
        "frontier": {
            "axis": axis.name, "parameter": axis.parameter, "value": value,
            "purpose": "actual_constructor_and_short_online_memory_probe",
            "not_a_regret_comparison": True,
            "advisory_preflight_overridden": axis.algorithm == "ours",
            "process_memory_limit_overridden": False,
            "requested_particles": particles,
        },
    }


def allocation_estimates(spec):
    """Explanatory array-size arithmetic, never used to allocate memory."""
    d = spec["d"]
    m = spec["config"].get("particles", 0)
    if spec["algorithm"] == "aioli":
        return {"post_init_persistent_array_bytes": 8 * (d*d + d),
                "dominant_initializer_arrays_bytes": 8 * d*d,
                "first_history_block_bytes": 0}
    return {
        "post_init_persistent_array_bytes": 8*m*d + (8*m if spec["algorithm"] == "ours" else 0),
        "dominant_initializer_arrays_bytes": (24 if spec["algorithm"] == "ours" else 16)*m*d,
        "first_history_block_bytes": 8*1024*(d+1),
        "refresh_temporaries_not_fully_estimated": True,
    }


def failure_phase(traceback_text):
    """Use the worker callsite, not round zero, to identify an actual failure."""
    if re.search(r"a\s*=\s*cls\s*\(", traceback_text):
        return "initialization"
    if re.search(r"a\.predict\s*\(", traceback_text):
        return "predict"
    if re.search(r"a\.observe\s*\(", traceback_text):
        return "observe"
    return "unknown"


def classify_attempt(spec, supervisor, traceback_text="", stderr_text=""):
    worker = supervisor.get("worker", {})
    status, termination = worker.get("status"), supervisor.get("termination")
    error = worker.get("error") or ""
    text = traceback_text + "\n" + str(error)
    native_lines = [line.strip() for line in stderr_text.splitlines()
                    if line.strip() == OPENBLAS_ALLOCATION_FAILURE]
    native_failure = bool(native_lines and supervisor.get("returncode") not in (None, 0))
    if native_failure or status == "memory_error" or termination == "memory_limit" or (
            status == "error" and ("MemoryError" in text or "_ArrayMemoryError" in text)):
        outcome = "memory_failure"
    elif status in ("completed", "evaluation_completed") and worker.get("round", 0) >= spec["stop_after_round"]:
        outcome = "completed"
    elif status in ("time_budget", "external_timeout") or termination == "external_timeout":
        outcome = "time_censored"
    elif "PrescribedBudgetExceeded" in text:
        outcome = "preflight_refusal"
    elif termination and str(termination).startswith("supervisor_error"):
        outcome = "supervisor_failure"
    elif "FloatingPointError" in text:
        outcome = "numerical_failure"
    else:
        outcome = "other_failure"
    diagnostics = worker.get("diagnostics") or {}
    initialized = bool(diagnostics or worker.get("state_bytes", 0) or worker.get("round", 0))
    actual_m = diagnostics.get("particles")
    if actual_m is None:
        actual_m = (diagnostics.get("configuration") or {}).get("particles")
    requested_m = spec["frontier"]["requested_particles"]
    if outcome == "completed" and requested_m is not None and actual_m != requested_m:
        outcome = "configuration_mismatch"
    memory = worker.get("memory") or {}
    final_snapshot = (status in ("completed", "evaluation_completed", "time_budget", "memory_error", "error")
                      and termination is None and bool(memory) and not native_failure)
    checkpoint_snapshot = bool(memory or diagnostics or worker.get("state_bytes", 0) or worker.get("round", 0))
    measurement_status = "final_snapshot" if final_snapshot else "checkpoint_censored" if checkpoint_snapshot else "unavailable"
    def measured(name):
        # The runner's no-checkpoint fallback contains dummy online_seconds=0.
        # It is not evidence that a native-failed process consumed zero work.
        return worker.get(name) if final_snapshot or checkpoint_snapshot else None
    record = {
        "axis": spec["frontier"]["axis"], "value": spec["frontier"]["value"],
        "algorithm": spec["algorithm"], "parameter": spec["frontier"]["parameter"],
        "d": spec["d"], "requested_particles": requested_m, "actual_particles": actual_m,
        "configured_horizon": spec["T"], "requested_rounds": spec["stop_after_round"],
        "completed_rounds": int(worker.get("round", 0)), "outcome": outcome,
        "worker_status": status, "termination": termination,
        "failure_phase": ("unknown" if native_failure else
                          failure_phase(traceback_text) if outcome != "completed" else None),
        "classification_version": CLASSIFICATION_VERSION,
        "native_allocation_failure": native_failure,
        "native_allocation_evidence": native_lines if native_failure else [],
        "native_failure_phase_not_inferred": native_failure,
        "stderr_path": str(Path(spec["output"]) / "stderr.txt") if stderr_text else None,
        "stderr_text_sha256": hashlib.sha256(stderr_text.encode("utf-8")).hexdigest() if stderr_text else None,
        "initialization_completed": initialized,
        "initialization_seconds": measured("initialization_seconds"),
        "prediction_seconds": measured("prediction_seconds"),
        "update_seconds": measured("update_seconds"),
        "online_seconds": measured("online_seconds"),
        "process_wall_seconds": supervisor.get("process_wall_seconds"),
        "state_bytes": measured("state_bytes"),
        "worker_final_snapshot_available": final_snapshot,
        "worker_checkpoint_available": checkpoint_snapshot and not final_snapshot,
        "timing_measurement_status": measurement_status,
        "state_measurement_status": measurement_status,
        "completed_rounds_censored": not final_snapshot,
        "memory_peak_is_lower_bound": not final_snapshot,
        "memory_peak_source": ("worker_final_and_supervisor_monitor" if final_snapshot else
                               "worker_checkpoint_and_supervisor_monitor" if checkpoint_snapshot and memory else
                               "supervisor_monitor_only"),
        "peak_private_commit": max(memory.get("peak_private_commit", 0),
                                   supervisor.get("monitored_peak_private", 0)),
        "peak_working_set": max(memory.get("peak_working_set", 0),
                                supervisor.get("monitored_peak_rss", 0)),
        "memory_limit_bytes": supervisor.get("hard_process_commit_limit", MEMORY_BYTES),
        "os_hard_limit_active": supervisor.get("os_hard_limit_active", False),
        "error": error or None, "returncode": supervisor.get("returncode"),
        "output": spec["output"], "traceback_path": (
            str(Path(spec["output"]) / "traceback.txt") if traceback_text else None),
        "allocation_estimates": allocation_estimates(spec),
    }
    return record


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _source_hashes():
    names = ("aioli.py", "proximal.py", "sigmoid.py", "common.py",
             "runner.py", "worker.py", "memory.py", "data.py", "memory_frontier.py")
    return {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest() for name in names}


def _policy_spec(axis, value, folder, policy, reference=None):
    """Independent concrete dispatch contract, using an archived run as template.

    This deliberately does not call build_spec. It checks numerical inputs and
    metadata for the next value, including branches not exercised previously;
    it is not a proof of equivalence of the surrounding Python source.
    """
    folder = Path(folder).resolve()
    if policy["memory_bytes"] != MEMORY_BYTES or policy["stage"] != folder.name:
        raise ValueError("Recorded policy has a different stage or memory cap")
    d = value if axis.parameter == "dimension" else axis.fixed_dimension
    B, T = policy["B"], policy["T"]
    data_seed = reference["data_seed"] if reference else 78001
    sampler_seed = reference["sampler_seed"] if reference else 878001
    config = dict(reference["config"]) if reference else {}
    if axis.algorithm == "aioli":
        if not reference:
            config = {"matrix_backend": "cholesky"}
        particles = None
    else:
        particles = value if axis.parameter == "particles" else policy["sampler_dimension_axes"][axis.algorithm]["particles"]
        transitions = (policy["particle_axis_transition_budget"] if axis.parameter == "particles"
                       else policy["sampler_dimension_axes"][axis.algorithm]["transitions"])
        config.update(particles=particles, transitions=transitions)
        if axis.algorithm == "ours":
            if not reference:
                config.update(mode="practical", particle_batch=64, max_state_bytes=2**63-1)
            config["max_particles"] = particles
    cid = f"signal_d{d}_B{B:g}_T{T}_s{data_seed}"
    label = f"{axis.parameter}_{value}"
    metadata = dict(reference["frontier"]) if reference else {
        "purpose": "actual_constructor_and_short_online_memory_probe",
        "not_a_regret_comparison": True,
        "advisory_preflight_overridden": axis.algorithm == "ours",
        "process_memory_limit_overridden": False,
    }
    metadata.update(axis=axis.name, parameter=axis.parameter, value=value, requested_particles=particles)
    result = dict(reference) if reference else {}
    result.update(algorithm=axis.algorithm, config=config, d=d, B=B, T=T,
                  stop_after_round=policy["rounds"], sampler_seed=sampler_seed,
                  data_seed=data_seed, case_id=cid, data_path=str(folder / "data" / cid),
                  output=str(folder / axis.name / label), seconds=policy["seconds_per_worker"],
                  memory_bytes=policy["memory_bytes"], stage=folder.name, label=label, frontier=metadata)
    return result


def _validate_probe_against_policy(spec, folder, policy, reference=None):
    axes = {a["name"]: Axis(**a) for a in policy["axes"]}
    name = spec.get("frontier", {}).get("axis")
    if name not in axes:
        raise ValueError("Probe axis is not in the recorded policy")
    value = spec["frontier"]["value"]
    axis = axes[name]
    if not isinstance(value, int) or not axis.minimum <= value <= axis.maximum:
        raise ValueError("Probe value is outside its recorded policy bounds")
    expected = _policy_spec(axis, value, folder, policy, reference)
    if spec != expected:
        changed = sorted(k for k in set(spec) | set(expected) if spec.get(k) != expected.get(k))
        raise ValueError("Probe settings differ from recorded policy/template: " + ", ".join(changed))


def _verify_recorded_specs(folder, policy, specs, *, require_raw=False):
    """Regenerate every existing spec from policy axes and compare all fields."""
    axes = {a["name"]: Axis(**a) for a in policy["axes"]}
    references, verified, seen = {}, [], set()
    for spec in specs:
        out = Path(spec["output"]).resolve()
        if not out.is_relative_to(folder) or str(out) in seen:
            raise ValueError("Manifest output is outside the campaign or duplicated: " + str(out))
        seen.add(str(out))
        raw_path = out / "spec.json"
        if raw_path.exists():
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            if raw != spec:
                raise ValueError("Raw specification differs from manifest: " + str(raw_path))
        elif require_raw:
            raise ValueError("Cannot verify an existing probe without its raw specification: " + str(raw_path))
        else:
            raw = spec
        name = spec["frontier"]["axis"]
        if name not in axes:
            raise ValueError("Recorded probe axis is absent from policy: " + name)
        axis = axes[name]
        regenerated = build_spec(axis, spec["frontier"]["value"], folder,
                                 T=policy["T"], rounds=policy["rounds"], B=policy["B"],
                                 seconds=policy["seconds_per_worker"])
        if regenerated != raw:
            changed = sorted(k for k in set(regenerated) | set(raw) if regenerated.get(k) != raw.get(k))
            raise ValueError("Current build_spec changes an existing recorded specification: "
                             + str(raw_path) + "; fields: " + ", ".join(changed))
        reference = references.setdefault(name, raw)
        _validate_probe_against_policy(raw, folder, policy, reference)
        verified.append({"output": str(out), "specification": raw,
                         "raw_spec_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest() if raw_path.exists() else None})
    return references, verified


def _check_resume_sources(folder):
    """Check source identity plus concrete specifications; not source equivalence."""
    original = json.loads((folder / "source_hashes.json").read_text())
    current = _source_hashes()
    audit_path = folder / "classification_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    if audit is not None and audit.get("status") != "completed":
        raise ValueError("Classification repair was interrupted; rerun --reclassify before resuming")
    if original == current:
        return
    changed = {name for name in set(original) | set(current) if original.get(name) != current.get(name)}
    if changed == {"memory_frontier.py"} and audit is not None:
        if (audit.get("status") == "completed" and audit.get("original_measurement_source_hashes") == original
                and audit.get("authorized_resume_source_hashes") == current):
            event = audit["repairs"][-1]
            source_copy = Path(event["module_source_path"])
            if hashlib.sha256(source_copy.read_bytes()).hexdigest() != current["memory_frontier.py"]:
                raise ValueError("Archived module source does not match the reviewed resume source")
            archived_policy = json.loads(Path(event["archived_policy_path"]).read_text())
            policy = json.loads((folder / "policy.json").read_text())
            if policy != archived_policy:
                raise ValueError("Current policy differs from the archived repair policy")
            _verify_recorded_specs(folder, policy, json.loads((folder / "manifest.json").read_text()))
            return
    raise ValueError("Source files changed; a reviewed memory_frontier.py update needs an offline --reclassify repair, "
                     "while algorithm/worker/data changes require a new frontier tag")


def _active_measurement_pids(processes, current_pid):
    """Ignore self and demonstrable identical-argument launcher ancestors only."""
    blocked = {"experiments.worker", "experiments.runner", "experiments.memory_frontier",
               "experiments.history_frontier"}
    by_pid = {p["pid"]: p for p in processes}
    own = by_pid.get(current_pid, {})
    own_arguments = tuple((own.get("cmdline") or [])[1:])
    ignored, seen = {current_pid}, {current_pid}
    parent_pid = own.get("ppid")
    while parent_pid in by_pid and parent_pid not in seen:
        seen.add(parent_pid)
        parent = by_pid[parent_pid]
        if own_arguments and tuple((parent.get("cmdline") or [])[1:]) == own_arguments:
            ignored.add(parent_pid)
        parent_pid = parent.get("ppid")
    return sorted(p["pid"] for p in processes
                  if p["pid"] not in ignored and blocked.intersection(p.get("cmdline") or []))


def _no_active_measurements():
    import psutil
    current_pid = psutil.Process().pid
    processes = []
    for proc in psutil.process_iter(["pid", "ppid", "cmdline"]):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    found = _active_measurement_pids(processes, current_pid)
    if found:
        raise RuntimeError(f"Measured processes are active ({found}); reclassify only after all measured jobs finish")


def reclassify_campaign(folder, *, process_guard=None):
    """Rebuild derived records from immutable raw evidence, without running workers.

    The original source_hashes.json, policy, manifests and all per-worker files
    remain byte-for-byte unchanged. Archive derived files before atomic replacement.
    The audit records that only this module changed and concrete specifications
    were verified. It does not certify semantic equivalence of the source.
    The optional guard injection is for filesystem-only tests.
    """
    (process_guard or _no_active_measurements)()
    folder = Path(folder).resolve()
    if not folder.name.startswith("memory_frontier_"):
        raise ValueError("Reclassification requires a dedicated memory_frontier_<tag> directory")
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))
    policy_bytes = (folder / "policy.json").read_bytes()
    policy = json.loads(policy_bytes)
    original_sources, current_sources = read(folder / "source_hashes.json"), _source_hashes()
    changed_sources = {k for k in set(original_sources) | set(current_sources)
                       if original_sources.get(k) != current_sources.get(k)}
    if changed_sources - {"memory_frontier.py"}:
        raise ValueError("Refusing module-update repair across changed algorithm/worker/data sources: "
                         + ", ".join(sorted(changed_sources - {"memory_frontier.py"})))
    records_path, summary_path = folder / "frontier_records.jsonl", folder / "frontier_summary.json"
    old_records_bytes, old_summary_bytes = records_path.read_bytes(), summary_path.read_bytes()
    old_records = [json.loads(line) for line in old_records_bytes.decode("utf-8").splitlines() if line.strip()]
    old_summary = json.loads(old_summary_bytes)
    if len(old_records) != old_summary.get("attempts"):
        raise ValueError("Derived record count disagrees with frontier summary; resolve the interrupted campaign first")
    manifest_bytes = (folder / "manifest.json").read_bytes()
    specs = json.loads(manifest_bytes)
    references, verified_specs = _verify_recorded_specs(folder, policy, specs, require_raw=True)
    if set(references) != {a["name"] for a in policy["axes"]}:
        raise ValueError("Module-update resume needs a verified existing raw specification for every policy axis")
    by_output = {str(Path(s["output"]).resolve()): s for s in specs}
    axes = [Axis(**a) for a in policy["axes"]]
    states = {a.name: Frontier(a, policy["refinements_per_axis"], policy["relative_tolerance"]) for a in axes}
    new_records, changes, raw_hashes = [], [], {}
    for old in old_records:
        out = Path(old["output"]).resolve()
        if not out.is_relative_to(folder):
            raise ValueError("A raw output directory escapes this campaign: " + str(out))
        spec = read(out / "spec.json")
        if by_output.get(str(out)) != spec:
            raise ValueError("Raw specification disagrees with manifest: " + str(out))
        supervisor = read(out / "supervisor.json")
        if supervisor.get("worker", {}).get("status") in (None, "running"):
            raise ValueError("Raw supervisor has no terminal outcome: " + str(out))
        tb, stderr = out / "traceback.txt", out / "stderr.txt"
        new = {**old, **classify_attempt(spec, supervisor,
                    tb.read_text(encoding="utf-8") if tb.exists() else "",
                    stderr.read_text(encoding="utf-8") if stderr.exists() else "")}
        for name in ("axis", "value", "algorithm", "parameter"):
            if new[name] != old[name]:
                raise ValueError("Derived attempt identity disagrees with its raw specification: " + str(out))
        states[new["axis"]].add(new)
        new_records.append(new)
        if old != new:
            changes.append({"sequence": old["sequence"], "output": str(out),
                            "changed_fields": {k: {"old": old.get(k), "new": new.get(k)}
                                               for k in set(old) | set(new) if old.get(k) != new.get(k)},
                            "classification_changed": old["outcome"] != new["outcome"]})
        for p in (out / "spec.json", out / "supervisor.json", tb, stderr):
            if p.exists():
                raw_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    archive = folder / "classification_archives" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    (archive / "frontier_records.jsonl").write_bytes(old_records_bytes)
    (archive / "frontier_summary.json").write_bytes(old_summary_bytes)
    (archive / "policy.json").write_bytes(policy_bytes)
    (archive / "manifest.json").write_bytes(manifest_bytes)
    module_bytes = (BASE / "memory_frontier.py").read_bytes()
    if hashlib.sha256(module_bytes).hexdigest() != current_sources["memory_frontier.py"]:
        raise RuntimeError("Module source changed during repair")
    (archive / "memory_frontier.py").write_bytes(module_bytes)
    _write_json(archive / "verified_specifications.json", verified_specs)
    audit_path = folder / "classification_audit.json"
    previous = read(audit_path) if audit_path.exists() else None
    if previous is not None:
        (archive / "classification_audit.json").write_bytes(audit_path.read_bytes())
    new_bytes = ("".join(json.dumps(r, allow_nan=False) + "\n" for r in new_records)).encode("utf-8")
    event = {"utc": stamp, "archive": str(archive), "records": len(new_records),
             "classification_changes": sum(c["classification_changed"] for c in changes), "changes": changes,
             "old_records_sha256": hashlib.sha256(old_records_bytes).hexdigest(),
             "new_records_sha256": hashlib.sha256(new_bytes).hexdigest(), "raw_evidence_sha256": raw_hashes,
             "module_source_sha256": current_sources["memory_frontier.py"],
             "module_source_path": str(archive / "memory_frontier.py"),
             "archived_policy_path": str(archive / "policy.json"),
             "verified_specifications_path": str(archive / "verified_specifications.json"),
             "recorded_specs_verified_unchanged": len(verified_specs),
             "changed_source_modules": sorted(changed_sources),
             "source_review_note": "Module changes reviewed for native-failure classification, censoring, and audited resume; "
                                   "automated validation checks concrete recorded/future probe specifications, not source equivalence.",
             "machine_proof_of_source_equivalence": False}
    audit = {"schema_version": 1, "status": "prepared", "raw_worker_artifacts_modified": False,
             "policy_modified": False, "classification_version": CLASSIFICATION_VERSION,
             "original_measurement_source_hashes": original_sources,
             "authorized_resume_source_hashes": current_sources,
             "repairs": ([] if previous is None else previous.get("repairs", [])) + [event]}
    # Preserve the audit before replacing any derived file. A crash leaves a
    # prepared audit that cannot authorize resume and can be repaired again.
    _write_json(audit_path, audit)
    if records_path.read_bytes() != old_records_bytes or summary_path.read_bytes() != old_summary_bytes:
        raise RuntimeError("Campaign outputs changed during repair; stop concurrent work and retry")
    if (folder / "policy.json").read_bytes() != policy_bytes or (folder / "manifest.json").read_bytes() != manifest_bytes:
        raise RuntimeError("Policy or manifest changed during repair; stop concurrent work and retry")
    for path, digest in raw_hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Raw evidence changed during repair: " + path)
    temporary = records_path.with_name("frontier_records.jsonl.reclassify.tmp")
    temporary.write_bytes(new_bytes)
    temporary.replace(records_path)
    pending = any(state.next_probe() is not None for state in states.values()) and len(new_records) < policy["max_probes"]
    summary = {**old_summary, "axes": [states[a.name].summary() for a in axes],
               "classification_version": CLASSIFICATION_VERSION, "classification_audit": str(audit_path),
               "status_before_reclassification": old_summary.get("status"),
               "status": "reclassified_pending_resume" if pending else old_summary.get("status"),
               "reclassified_utc": stamp}
    if "last_record" in summary and new_records:
        summary["last_record"] = new_records[-1]
    _write_json(summary_path, summary)
    audit["status"] = "completed"
    _write_json(audit_path, audit)
    return {"stage": folder.name, "records": len(new_records),
            "classification_changes": event["classification_changes"],
            "archive": str(archive), "audit": str(audit_path), "pending_resume": pending,
            "next_probes": {a.name: states[a.name].next_probe() for a in axes}}


def run_campaign(folder, axes, *, T=4, rounds=2, B=1000., seconds=20.,
                 wall_budget=540., max_probes=25, refinements=1,
                 relative_tolerance=.1, resume=False, runner=None, generate=None,
                 environment_reader=None):
    """Run one worker at a time. Optional injected functions enable pure tests."""
    if T < 1 or not 1 <= rounds <= T:
        raise ValueError("Require 1 <= rounds <= T")
    if not math.isfinite(B) or B <= 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("B and seconds must be positive and finite")
    if wall_budget <= 0 or max_probes < 1 or refinements < 0:
        raise ValueError("Invalid campaign budget")
    if not 0 < relative_tolerance < 1:
        raise ValueError("relative_tolerance must be in (0,1)")
    folder = Path(folder).resolve()
    if not folder.name.startswith("memory_frontier_"):
        raise ValueError("Use a dedicated memory_frontier_<tag> stage directory")
    policy = {
        "schema_version": 1, "stage": folder.name,
        "axes": [vars(a) for a in axes], "B": B, "T": T, "rounds": rounds,
        "seconds_per_worker": seconds, "max_probes": max_probes,
        "refinements_per_axis": refinements, "relative_tolerance": relative_tolerance,
        "memory_bytes": MEMORY_BYTES,
        "measured_workers_are_sequential": True,
        "sampler_dimension_axes": {"ours": {"particles": 256, "transitions": 1},
                                   "sigmoid": {"particles": 16, "transitions": 16}},
        "particle_axis_transition_budget": 1,
        "no_artificial_allocation_probe": True,
        "time_censored_probes_are_not_memory_failures": True,
        "advisory_proximal_preflight_overridden_but_os_limit_unchanged": True,
        "wall_budget_scope": "this invocation, including data preparation",
        "not_a_regret_or_statistical_comparison": True,
    }
    policy_path = folder / "policy.json"
    if folder.exists() and any(folder.iterdir()):
        if not resume or not policy_path.exists():
            raise ValueError("Nonempty stage requires --resume and a matching policy")
        if json.loads(policy_path.read_text()) != policy:
            raise ValueError("Refusing to resume with a changed frontier policy")
        _check_resume_sources(folder)
    else:
        folder.mkdir(parents=True, exist_ok=True)
        _write_json(policy_path, policy)
        _write_json(folder / "source_hashes.json", _source_hashes())
    if runner is None:
        from .runner import run_one
        runner = run_one
    if generate is None:
        from .data import generate_case
        generate = generate_case
    if environment_reader is None:
        from .runner import environment
        environment_reader = environment
    if not (folder / "environment.json").exists():
        _write_json(folder / "environment.json", environment_reader())
    records_path = folder / "frontier_records.jsonl"
    records = []
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    states = {axis.name: Frontier(axis, refinements, relative_tolerance) for axis in axes}
    for record in records:
        states[record["axis"]].add(record)
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    references, _ = _verify_recorded_specs(folder, policy, manifest)
    cases_path = folder / "cases.json"
    cases = json.loads(cases_path.read_text()) if cases_path.exists() else []
    started = perf_counter()
    stop_reason = "frontiers_finished"
    while True:
        made_progress = False
        for axis in axes:
            if len(records) >= max_probes:
                stop_reason = "probe_budget"
                break
            if perf_counter() - started >= wall_budget:
                stop_reason = "wall_budget"
                break
            proposal = states[axis.name].next_probe()
            if proposal is None:
                continue
            value, role = proposal
            spec = build_spec(axis, value, folder, T=T, rounds=rounds, B=B, seconds=seconds)
            _validate_probe_against_policy(spec, folder, policy, references.get(axis.name))
            case = {"id": spec["case_id"], "family": "signal", "d": spec["d"],
                    "B": B, "B_generator": 3., "T": T, "data_seed": spec["data_seed"]}
            generate(spec["data_path"], case)
            if case not in cases:
                cases.append(case)
                _write_json(cases_path, cases)
            if spec not in manifest:
                manifest.append(spec)
                _write_json(manifest_path, manifest)
                axis_folder = folder / axis.name
                axis_folder.mkdir(exist_ok=True)
                _write_json(axis_folder / "manifest.json",
                            [s for s in manifest if s["frontier"]["axis"] == axis.name])
            remaining = wall_budget - (perf_counter() - started)
            if remaining <= 0:
                stop_reason = "wall_budget"
                break
            if json.loads(policy_path.read_text()) != policy:
                raise ValueError("Recorded policy changed before dispatch")
            _validate_probe_against_policy(spec, folder, policy, references.get(axis.name))
            supervisor = runner(spec, remaining)
            tb = Path(spec["output"]) / "traceback.txt"
            stderr = Path(spec["output"]) / "stderr.txt"
            record = classify_attempt(spec, supervisor, tb.read_text() if tb.exists() else "",
                                      stderr.read_text() if stderr.exists() else "")
            record.update(role=role, sequence=len(records) + 1,
                          recorded_utc=datetime.now(timezone.utc).isoformat())
            with records_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
            records.append(record)
            states[axis.name].add(record)
            made_progress = True
            summary = {
                "schema_version": 1, "stage": folder.name, "attempts": len(records),
                "memory_limit_bytes": MEMORY_BYTES,
                "axes": [states[a.name].summary() for a in axes],
                "last_record": record,
                "elapsed_this_invocation": perf_counter() - started,
                "status": "running",
                "interpretation": "Observed short-probe brackets; timeouts and unknown failure phases remain explicit.",
            }
            _write_json(folder / "frontier_summary.json", summary)
            print(json.dumps({k: record[k] for k in (
                "axis", "value", "outcome", "failure_phase", "completed_rounds",
                "peak_private_commit", "process_wall_seconds")}), flush=True)
        if stop_reason != "frontiers_finished" or not made_progress:
            break
    summary = {
        "schema_version": 1, "stage": folder.name, "attempts": len(records),
        "memory_limit_bytes": MEMORY_BYTES,
        "axes": [states[a.name].summary() for a in axes],
        "elapsed_this_invocation": perf_counter() - started,
        "status": stop_reason,
        "records_path": str(records_path), "manifest_path": str(manifest_path),
        "interpretation": "Observed short-probe brackets, not universal capacity thresholds or regret results.",
    }
    _write_json(folder / "frontier_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="New dedicated stage suffix")
    parser.add_argument("--axes", nargs="+", choices=AXIS_NAMES, default=list(AXIS_NAMES))
    parser.add_argument("--particle-dimension", type=int, default=128)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--B", type=float, default=1000.)
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--wall-budget", type=float, default=540.)
    parser.add_argument("--max-probes", type=int, default=25)
    parser.add_argument("--refinements", type=int, default=1)
    parser.add_argument("--relative-tolerance", type=float, default=.1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reclassify", action="store_true",
                        help="Offline repair of derived classifications from retained raw evidence; launches no workers")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.tag):
        parser.error("tag must contain only letters, digits, underscores or hyphens")
    if args.particle_dimension < 1:
        parser.error("particle-dimension must be positive")
    if args.reclassify:
        print(json.dumps(reclassify_campaign(BASE / "results" / ("memory_frontier_" + args.tag)), indent=2))
        return
    axes = [a for a in default_axes(args.particle_dimension) if a.name in args.axes]
    result = run_campaign(BASE / "results" / ("memory_frontier_" + args.tag), axes,
                          T=args.T, rounds=args.rounds, B=args.B, seconds=args.seconds,
                          wall_budget=args.wall_budget, max_probes=args.max_probes,
                          refinements=args.refinements,
                          relative_tolerance=args.relative_tolerance, resume=args.resume)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
