"""Prepare comparisons against archived runs, without rerunning algorithms.

Run only after measured workers finish, for example::

    python -m experiments.shared_compare_prepare all
    python -m experiments.analyze shared_compare_screen --comparator-seconds 1

Original manifests/specifications and raw output paths are retained verbatim as
JSON values. Comparator files are copied only on first preparation: an existing
comparison cache may have been refined and is never overwritten here. This
module reads metadata and copies files; it performs no numerical optimization.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


BASE = Path(__file__).resolve().parent
STAGES = ("prescribed", "screen", "deep_scale", "confirmation", "deep_budget",
          "long_horizon", "deep_long")
ACTIVE_MODULES = {
    "experiments.worker", "experiments.runner", "experiments.shared_worker",
    "experiments.shared_runner", "experiments.memory_frontier",
    "experiments.history_frontier",
}


def _no_measured_workers():
    import psutil
    active = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            if process.info["pid"] != os.getpid() and ACTIVE_MODULES.intersection(
                    process.info["cmdline"] or []):
                active.append(process.info["pid"])
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as exc:
            raise RuntimeError("Cannot establish that measured workers have stopped") from exc
    if active:
        raise RuntimeError(f"Finish measured workers before preparing comparisons: {active}")


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Required completed artifact is unavailable: {path}") from exc


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file_record(path, required=False):
    path = Path(path).resolve()
    if required and not path.is_file():
        raise RuntimeError(f"Required source artifact is missing: {path}")
    return dict(path=str(path), exists=path.is_file(),
                sha256=_digest(path) if path.is_file() else None)


def _load_complete(folder):
    manifest = folder/"manifest.json"
    specs = _read(manifest)
    if not isinstance(specs, list) or not specs:
        raise RuntimeError(f"Manifest must contain a nonempty list: {manifest}")
    evidence, outputs = [], set()
    for spec in specs:
        output = Path(spec["output"]).resolve()
        if str(output) in outputs:
            raise RuntimeError(f"Duplicate raw output path: {output}")
        outputs.add(str(output))
        raw_spec = output/"spec.json"
        if _read(raw_spec) != spec:
            raise RuntimeError(f"Raw specification differs from its source manifest: {raw_spec}")
        supervisor_path = output/"supervisor.json"
        supervisor = _read(supervisor_path)
        worker = supervisor.get("worker")
        if not isinstance(worker, dict):
            raise RuntimeError(f"No final worker snapshot: {supervisor_path}")
        status = worker.get("status")
        if not status or status in ("running", "pending", "starting", "not_measured"):
            raise RuntimeError(f"Nonterminal worker status in {supervisor_path}: {status}")
        if supervisor.get("returncode") is None and not supervisor.get("termination"):
            raise RuntimeError(f"Supervisor does not establish termination: {supervisor_path}")
        rounds = worker.get("round")
        if (isinstance(rounds, bool) or not isinstance(rounds, int)
                or not 0 <= rounds <= int(spec["T"])):
            raise RuntimeError(f"Invalid completed-round count: {supervisor_path}")
        # Zero-round guard/allocation failures are deliberately retained. For
        # nonempty interrupted runs the supervisor may have only the last
        # flushed snapshot; analyze.py consistently uses its completed prefix.
        evidence.append(dict(output=str(output), algorithm=spec["algorithm"],
                             case_id=spec["case_id"], label=spec["label"],
                             status=status, completed_rounds=rounds,
                             termination=supervisor.get("termination"),
                             returncode=supervisor.get("returncode"),
                             spec=_file_record(raw_spec, required=True),
                             supervisor=_file_record(supervisor_path, required=True),
                             result_path=str(output/"result.json"),
                             losses_path=str(output/"losses.bin"),
                             times_path=str(output/"online_times.bin")))
    return specs, evidence


def _source_evidence(folder):
    return dict(stage=folder.name, manifest=_file_record(folder/"manifest.json", True),
                cases=_file_record(folder/"cases.json", True),
                environment=_file_record(folder/"environment.json", True),
                provenance=[_file_record(folder/name) for name in
                            ("source_hashes.json", "provenance.json", "protocol.json",
                             "selected_config.json", "selected_configs.json")])


def prepare(source_stage):
    """Return the combined manifest path; refuse unfinished or changed inputs."""
    if source_stage not in STAGES:
        raise ValueError(f"Unknown comparison source: {source_stage}")
    _no_measured_workers()
    old_folder = BASE/"results"/source_stage
    new_folder = BASE/"results"/("shared_"+source_stage)
    target = BASE/"results"/("shared_compare_"+source_stage)
    old_specs, old_runs = _load_complete(old_folder)
    new_specs, new_runs = _load_complete(new_folder)
    old_cases, new_cases = _read(old_folder/"cases.json"), _read(new_folder/"cases.json")
    if old_cases != new_cases:
        raise RuntimeError("Archived and new cases differ; paired data must be identical")
    old_by_case = {}
    for spec in old_specs:
        old_by_case.setdefault(spec["case_id"], spec)
    if {spec["case_id"] for spec in new_specs} != set(old_by_case):
        raise RuntimeError("New and archived manifests must cover the same cases")
    for spec in new_specs:
        source = old_by_case[spec["case_id"]]
        for key in ("case_id", "d", "B", "T", "data_seed"):
            if spec[key] != source[key]:
                raise RuntimeError(f"Paired data metadata differ in {key}: {spec['case_id']}")
        if Path(spec["data_path"]).resolve() != Path(source["data_path"]).resolve():
            raise RuntimeError(f"Paired runs do not use the same stored data: {spec['case_id']}")
    combined = old_specs+new_specs
    outputs = [str(Path(spec["output"]).resolve()) for spec in combined]
    if len(outputs) != len(set(outputs)):
        raise RuntimeError("Combined manifest would duplicate raw output paths")
    sources = dict(archived=_source_evidence(old_folder), shared=_source_evidence(new_folder))
    # Validate every source first, before creating any derived artifacts.
    manifest, cases_path = target/"manifest.json", target/"cases.json"
    if manifest.exists() and _read(manifest) != combined:
        raise RuntimeError(f"Refusing to replace a different comparison manifest: {manifest}")
    if cases_path.exists() and _read(cases_path) != old_cases:
        raise RuntimeError(f"Refusing to replace different comparison cases: {cases_path}")
    target.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        with manifest.open("x", encoding="utf-8") as handle:
            json.dump(combined, handle, indent=2)
    if not cases_path.exists():
        # Preserve the source file exactly, including its formatting.
        with cases_path.open("xb") as handle:
            handle.write((old_folder/"cases.json").read_bytes())
    caches = []
    for case_id in old_by_case:
        source_cache = old_folder/f"{case_id}__comparators.json"
        target_cache = target/source_cache.name
        action = "source_unavailable"
        if target_cache.exists():
            action = "retained_existing"
        elif source_cache.is_file():
            with target_cache.open("xb") as handle:
                handle.write(source_cache.read_bytes())
            action = "copied_archived"
        caches.append(dict(case_id=case_id, action=action,
                           archived=_file_record(source_cache),
                           combined=_file_record(target_cache)))
    provenance = dict(
        prepared_utc=datetime.now(timezone.utc).isoformat(), source_stage=source_stage,
        combined_stage=target.name, manifest=_file_record(manifest, True),
        cases=_file_record(cases_path, True), sources=sources,
        run_counts=dict(archived=len(old_specs), shared=len(new_specs), total=len(combined)),
        terminal_status_counts=dict(Counter(row["status"] for row in old_runs+new_runs)),
        zero_round_runs=[row["output"] for row in old_runs+new_runs if row["completed_rounds"] == 0],
        runs=dict(archived=old_runs, shared=new_runs), comparator_caches=caches,
        global_provenance=[_file_record(BASE/"results"/name) for name in
                           ("controls.json", "deep_artifact_audit.json", "shared_controls.json")],
        source_spec_values_and_raw_paths_unchanged=True,
        comparator_cache_policy="Copy archived cache only if absent; preserve every existing combined cache",
        comparison_scope="Archived timings were measured earlier; combined files do not constitute simultaneous reruns",
        numerical_jobs_executed=False)
    (target/"comparison_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(dict(stage=target.name, **provenance["run_counts"],
                          zero_round_runs=len(provenance["zero_round_runs"]), manifest=str(manifest))), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, "all"))
    args = parser.parse_args()
    for stage in STAGES if args.stage == "all" else (args.stage,):
        prepare(stage)


if __name__ == "__main__":
    main()
