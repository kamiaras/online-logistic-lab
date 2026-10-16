"""Report the supplied shared-tangent algorithm against preserved old runs.

This module launches no workers and fits no comparators. Run after every
measurement process has exited and shared_compare_* analyses are complete.
Recorded loss files are memory-mapped; prefix sums use bounded chunks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import median


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
SOURCES = ("prescribed", "screen", "deep_scale", "confirmation", "deep_budget",
           "long_horizon", "deep_long")
CAP = 2 * 1024 ** 3
MIB = 1024 ** 2
NAMES = {"shared": "Shared tangent", "ours": "Original ours",
         "sigmoid": "Sigmoid/MALA", "aioli": "AIOLI"}
COLORS = {"shared": "#7735a4", "ours": "#be4938",
          "sigmoid": "#2570b7", "aioli": "#2a8059"}
CORE = ("aioli.py", "common.py", "proximal.py", "sigmoid.py", "data.py")
INPUT_HASH = "31a7036351d27bf00c335432102bca00d808408be8e8330e3ae13d228c8b860a"


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Required completed artifact unavailable: " + str(path)) from exc


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _key(row):
    return tuple(row[k] for k in ("case_id", "algorithm", "label", "sampler_seed"))


def _base_label(label):
    return re.sub(r"_rep\d+$", "", label)


def _rep(label):
    match = re.search(r"_rep(\d+)$", label)
    return int(match[1]) if match else None


def _mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _fmt(value, digits=3):
    return f"{value:,.{digits}f}" if _finite(value) else "—"


def _span(values):
    values = list(values)
    return "—" if not values else (f"{min(values):,}" if min(values) == max(values)
                                   else f"{min(values):,}–{max(values):,}")


def _link(path, label=None):
    path = Path(path).resolve()
    return f"[{label or path.name}](<{path.as_posix()}>)"


def _guard():
    import psutil
    # Explicit tokens avoid mistaking this report's own name for a benchmark.
    blocked = {"experiments." + name for name in (
        "worker", "runner", "shared_worker", "shared_runner", "shared_campaign",
        "deep_campaign", "memory_frontier", "history_frontier", "shared_memory_frontier",
        "shared_history_frontier", "controls", "precision_audit", "deep_precision",
        "deep_comparator_refine", "analyze", "shared_refine")}
    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if proc.info["pid"] != os.getpid() and blocked.intersection(proc.info["cmdline"] or []):
                found.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if found:
        raise RuntimeError(f"Numerical experiment processes are active ({found}); finish them before reporting")


class RecordedRun:
    def __init__(self, spec, summary, case):
        import numpy as np
        self.spec, self.summary, self.case = spec, summary, case
        self.output = Path(spec["output"])
        self.supervisor = read_json(self.output / "supervisor.json")
        self.worker = self.supervisor.get("worker", {})
        self.n = int(self.worker.get("round", 0))
        if self.worker.get("status") in (None, "running"):
            raise RuntimeError("Missing terminal status: " + str(self.output))
        if read_json(self.output / "spec.json") != spec:
            raise RuntimeError("Manifest differs from executed specification: " + str(self.output))
        if spec.get("memory_bytes") != CAP or not self.supervisor.get("os_hard_limit_active"):
            raise RuntimeError("Missing hard 2 GiB limit: " + str(self.output))
        if self.n < 0 or self.n > spec["T"]:
            raise RuntimeError("Invalid completed-round count: " + str(self.output))
        if self.n and summary is None:
            raise RuntimeError("Positive-round run is missing from analysis: " + str(self.output))
        if summary is not None and summary.get("completed_rounds") != self.n:
            raise RuntimeError("Stale analyzed completion count: " + str(self.output))
        self._arrays = {}
        self.raw_lengths = {}
        for name in ("losses", "online_times"):
            path = self.output / (name + ".bin")
            size = path.stat().st_size if path.exists() else 0
            if size % 8 or size // 8 < self.n:
                raise RuntimeError("Incomplete binary measurement: " + str(path))
            self.raw_lengths[name] = size // 8
            self._arrays[name] = (np.memmap(path, dtype="<f8", mode="r", shape=(size // 8,))
                                  if size else np.empty(0))
        self._prefix = {0: 0.}
        self._memory = {}
        progress = self.output / "progress.jsonl"
        if progress.exists():
            with progress.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        point = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # An interrupted final JSONL write is not a measured point.
                    n = point.get("round")
                    memory = point.get("memory", {}).get("peak_private_commit")
                    # Error snapshots can include an unfinished next prediction;
                    # they are whole-attempt memory, not prefix-matched memory.
                    prefix_valid = point.get("status") in (None, "completed", "evaluation_completed", "time_budget")
                    if isinstance(n, int) and _finite(memory) and prefix_valid:
                        self._memory[n] = memory
        memory = self.worker.get("memory", {}).get("peak_private_commit")
        if (_finite(memory) and self.worker.get("status") in ("completed", "evaluation_completed", "time_budget")
                and self.supervisor.get("termination") is None):
            self._memory[self.n] = memory
        self.peak_private = max(self.supervisor.get("monitored_peak_private", 0),
                                self.worker.get("memory", {}).get("peak_private_commit", 0))
        if summary:
            for name in ("common", "final"):
                point = summary.get(name)
                if not point:
                    continue
                n = point["round"]
                loss = self.loss(n)
                if abs(loss - point["loss"]) > 2e-10 * max(1, abs(loss)):
                    raise RuntimeError("Analyzed loss differs from raw prefix: " + str(self.output))
                if point["regret_lower"] > point["regret_upper"]:
                    raise RuntimeError("Reversed numerical regret interval: " + str(self.output))

    def loss(self, n):
        import numpy as np
        if not 0 <= n <= self.n:
            raise ValueError("Unavailable recorded prefix")
        if n not in self._prefix:
            partials = []
            for start in range(0, n, 32768):
                values = self._arrays["losses"][start:min(n, start + 32768)]
                if np.any(~np.isfinite(values)) or np.any(values < 0):
                    raise RuntimeError("Nonfinite/negative raw logistic loss: " + str(self.output))
                partials.append(float(np.sum(values, dtype=np.float64)))
            self._prefix[n] = math.fsum(partials)
        return self._prefix[n]

    def time(self, n):
        if not 1 <= n <= self.n:
            return None
        value = float(self._arrays["online_times"][n - 1])
        if not math.isfinite(value) or value < 0:
            raise RuntimeError("Invalid measured prefix time: " + str(self.output))
        return value

    def identity(self):
        return {"run_id": self.output.name, "output": str(self.output),
                **{k: self.spec[k] for k in ("algorithm", "label", "config", "sampler_seed")}}

    def public(self):
        return {**self.spec, "run_id": self.output.name, "completed_rounds": self.n,
                "status": self.worker.get("status"), "termination": self.supervisor.get("termination"),
                "error": self.worker.get("error"), "family": self.case["family"],
                "B_generator": self.case.get("B_generator"),
                "process_wall_seconds": self.supervisor.get("process_wall_seconds"),
                "online_seconds": self.worker.get("online_seconds"),
                "state_bytes": self.worker.get("state_bytes"),
                "peak_private_commit": self.peak_private,
                "memory_scope": "Whole measured attempt, not matched to a comparison prefix",
                "raw_array_lengths": self.raw_lengths,
                "common": None if self.summary is None else self.summary.get("common"),
                "final": None if self.summary is None else self.summary.get("final"),
                "metrics": self.worker.get("metrics") or {},
                "diagnostics": self.worker.get("diagnostics") or {}}


def at_prefix(new, old, n):
    point = {"round": n, "available": False}
    if n < 1 or new.n < n or old.n < n:
        point["reason"] = "no_shared_completed_rounds" if n < 1 else "checkpoint_not_reached_by_both"
        return point
    a, b = new.loss(n), old.loss(n)
    ta, tb = new.time(n), old.time(n)
    point.update(available=True, new_loss=a, old_loss=b, loss_difference=a-b,
                 average_loss_difference=(a-b)/n, new_online_seconds=ta,
                 old_online_seconds=tb, time_ratio=ta/tb if tb > 0 else None,
                 new_prefix_peak_private=new._memory.get(n),
                 old_prefix_peak_private=old._memory.get(n),
                 prefix_memory_scope="Exact recorded round snapshot if available; otherwise null",
                 new_attempt_peak_private=new.peak_private,
                 old_attempt_peak_private=old.peak_private)
    return point


def load_stages(results, sources):
    stages = {}
    for source in sources:
        folder = results / ("shared_compare_" + source)
        specs, summaries, cases = [read_json(folder / name) for name in
                                  ("manifest.json", "summary.json", "cases.json")]
        read_json(folder / "curves.json")
        if not specs or len({_key(s) for s in specs}) != len(specs):
            raise RuntimeError("Empty or duplicate combined manifest: " + str(folder))
        index = {_key(row): row for row in summaries}
        if len(index) != len(summaries) or set(index) - {_key(s) for s in specs}:
            raise RuntimeError("Unexpected/duplicate analyzed runs: " + str(folder))
        case_index = {row["id"]: row for row in cases}
        runs = [RecordedRun(spec, index.get(_key(spec)), case_index[spec["case_id"]]) for spec in specs]
        expected_new = read_json(results / ("shared_" + source) / "manifest.json")
        if {str(Path(s["output"]).resolve()) for s in expected_new} != {
                str(r.output.resolve()) for r in runs if r.spec["algorithm"] == "shared"}:
            raise RuntimeError("Combined manifest omits/adds a shared-tangent run: " + source)
        # The original environment remains separate, because timing is reused.
        for stage in (source, "shared_" + source):
            environment = read_json(results / stage / "environment.json")
            if environment.get("worker_blas_threads") != 1:
                raise RuntimeError("Expected one BLAS thread in " + stage)
        groups = defaultdict(list)
        for run in runs:
            groups[run.spec["case_id"]].append(run)
        pairs = []
        for cid, entries in sorted(groups.items()):
            new_runs = [r for r in entries if r.spec["algorithm"] == "shared"]
            old_runs = [r for r in entries if r.spec["algorithm"] != "shared"]
            if not new_runs:
                raise RuntimeError("Combined case lacks shared-tangent run: " + cid)
            setting = {k: v for k, v in new_runs[0].case.items() if k not in ("id", "data_seed")}
            for new in new_runs:
                for old in old_runs:
                    a, b = _rep(new.spec["label"]), _rep(old.spec["label"])
                    if a is not None and b is not None and a != b:
                        continue
                    if Path(new.spec["data_path"]).resolve() != Path(old.spec["data_path"]).resolve():
                        raise RuntimeError("Pair does not use identical stored data")
                    if any(new.spec[k] != old.spec[k] for k in ("d", "B", "T", "data_seed")):
                        raise RuntimeError("Paired parameter mismatch")
                    pairs.append({"source": source, "case_id": cid, "setting": setting,
                                  "data_seed": new.spec["data_seed"], "data_path": new.spec["data_path"],
                                  "new": new.identity(), "old": old.identity(),
                                  "common": at_prefix(new, old, min(new.n, old.n)),
                                  "checkpoints": [at_prefix(new, old, n) for n in (100, 500, 1000)]})
        counts = lambda rr: dict(Counter(r.worker.get("status", "unknown") for r in rr))
        new_runs = [r for r in runs if r.spec["algorithm"] == "shared"]
        stages[source] = {"folder": folder, "runs": runs, "groups": groups,
                          "public": {"expected_runs": len(specs), "new_runs": len(new_runs),
                                     "old_runs": len(runs)-len(new_runs),
                                     "new_status_counts": counts(new_runs),
                                     "all_status_counts": counts(runs),
                                     "rows": [r.public() for r in runs], "pairwise": pairs}}
    return stages


def load_auxiliary_counts(results):
    output = {}
    for source in ("calibration", "smoke"):
        folder = results / ("shared_" + source)
        specs = read_json(folder / "manifest.json")
        rows = []
        if read_json(folder / "environment.json").get("worker_blas_threads") != 1:
            raise RuntimeError("Expected one BLAS thread in " + str(folder))
        for spec in specs:
            raw = Path(spec["output"])
            sup = read_json(raw / "supervisor.json")
            worker = sup.get("worker", {})
            if read_json(raw / "spec.json") != spec or worker.get("status") in (None, "running"):
                raise RuntimeError("Auxiliary shared run is incomplete or changed: " + str(raw))
            if spec.get("memory_bytes") != CAP or not sup.get("os_hard_limit_active"):
                raise RuntimeError("Auxiliary run lacks the 2 GiB cap: " + str(raw))
            rows.append({"output": str(raw), "case_id": spec["case_id"], "label": spec["label"],
                         "status": worker["status"], "completed_rounds": worker.get("round", 0)})
        output[source] = {"new_runs": len(rows), "new_status_counts": dict(Counter(r["status"] for r in rows)),
                          "rows": rows}
    return output


def seed_summaries(stages):
    grouped = defaultdict(list)
    for stage in stages.values():
        for pair in stage["public"]["pairwise"]:
            identity = {"source": pair["source"], "setting": pair["setting"],
                        "new_label": _base_label(pair["new"]["label"]),
                        "new_config": pair["new"]["config"],
                        "old_algorithm": pair["old"]["algorithm"],
                        "old_label": _base_label(pair["old"]["label"]),
                        "old_config": pair["old"]["config"]}
            for checkpoint, point in [("pair_common", pair["common"])] + [
                    (str(p["round"]), p) for p in pair["checkpoints"]]:
                key = json.dumps({**identity, "checkpoint": checkpoint}, sort_keys=True)
                grouped[key].append((pair, point))
    output = []
    for key, rows in sorted(grouped.items()):
        by_seed = defaultdict(list)
        for pair, point in rows:
            by_seed[pair["data_seed"]].append((pair, point))
        values = []
        for seed, repeats in sorted(by_seed.items()):
            good = [p for _, p in repeats if p["available"]]
            if good:
                new_time = _mean(p["new_online_seconds"] for p in good)
                old_time = _mean(p["old_online_seconds"] for p in good)
                values.append({"data_seed": seed, "available": True,
                               "repetitions_expected": len(repeats), "repetitions_available": len(good),
                               "rounds": [p["round"] for p in good],
                               "loss_difference": _mean(p["loss_difference"] for p in good),
                               "average_loss_difference": _mean(p["average_loss_difference"] for p in good),
                               "new_online_seconds": new_time, "old_online_seconds": old_time,
                               "time_ratio": new_time/old_time if old_time > 0 else None})
            else:
                values.append({"data_seed": seed, "available": False,
                               "repetitions_expected": len(repeats), "repetitions_available": 0})
        good = [v for v in values if v["available"]]
        differences = [v["average_loss_difference"] for v in good]
        output.append({**json.loads(key), "datasets_expected": len(values),
                       "datasets_available": len(good), "dataset_values": values,
                       "mean_loss_difference": _mean(v["loss_difference"] for v in good),
                       "mean_average_loss_difference": _mean(differences),
                       "minimum_average_loss_difference": min(differences) if differences else None,
                       "maximum_average_loss_difference": max(differences) if differences else None,
                       "mean_time_ratio": _mean(v["time_ratio"] for v in good if _finite(v["time_ratio"]))})
    return output


def confirmation(stages):
    if "confirmation" not in stages:
        return []
    from scipy.stats import t as student_t
    results = []
    groups = stages["confirmation"]["groups"]
    for n in (500, 100):
        for algorithm in ("ours", "sigmoid", "aioli"):
            datasets = []
            for cid, runs in sorted(groups.items()):
                new = [r for r in runs if r.spec["algorithm"] == "shared"]
                old = [r for r in runs if r.spec["algorithm"] == algorithm]
                expected_old = 1 if algorithm == "aioli" else 3
                if len(new) != 3 or len(old) != expected_old:
                    raise RuntimeError("Confirmation replicate manifest is incomplete: " + cid)
                ready = all(r.n >= n for r in new + old)
                row = {"case_id": cid, "data_seed": new[0].spec["data_seed"], "available": ready,
                       "new_repetitions_reached": sum(r.n >= n for r in new),
                       "old_repetitions_reached": sum(r.n >= n for r in old),
                       "new_repetitions_expected": 3, "old_repetitions_expected": expected_old}
                if ready:
                    a, b = _mean(r.loss(n) for r in new), _mean(r.loss(n) for r in old)
                    row.update(new_mean_loss=a, old_mean_loss=b, loss_difference=a-b,
                               average_loss_difference=(a-b)/n,
                               new_mean_seconds=_mean(r.time(n) for r in new),
                               old_mean_seconds=_mean(r.time(n) for r in old))
                datasets.append(row)
            complete = all(r["available"] for r in datasets)
            row = {"round": n, "old_algorithm": algorithm, "complete": complete,
                   "datasets_expected": len(datasets),
                   "datasets_available": sum(r["available"] for r in datasets),
                   "dataset_values": datasets, "mean_loss_difference": None,
                   "cluster_t_interval_95": None,
                   "inference_scope": "Approximate Student-t interval across dataset means; sampler repeats averaged within each dataset. Computed only if every prescribed dataset/repetition reached the checkpoint."}
            if complete:
                differences = [r["loss_difference"] for r in datasets]
                average = _mean(differences)
                row["mean_loss_difference"] = average
                if len(differences) > 1:
                    variance = math.fsum((v-average)**2 for v in differences)/(len(differences)-1)
                    half = float(student_t.ppf(.975, len(differences)-1))*math.sqrt(variance/len(differences))
                    row["cluster_t_interval_95"] = [average-half, average+half]
            results.append(row)
    return results


def provenance(results, sources):
    controls = read_json(results / "controls.json")["algorithm_source_sha256"]
    hashes = {name: _hash(BASE / name) for name in CORE}
    if any(hashes[name] != controls[name] for name in CORE):
        raise RuntimeError("An original algorithm/data core file differs from the preserved controls")
    manuscript = BASE / "SHARED_TANGENT_INPUT.md"
    if _hash(manuscript) != INPUT_HASH:
        raise RuntimeError("Supplied manuscript copy changed")
    stage_hashes = {}
    for source in sources:
        path = results / ("shared_" + source) / "source_hashes.json"
        snapshot = read_json(path)
        for name, digest in snapshot.items():
            if _hash(BASE / name) != digest:
                raise RuntimeError("Measured shared source changed: " + name)
        stage_hashes[source] = snapshot
    return {"original_core_source_sha256": hashes, "original_core_controls_match": True,
            "manuscript": str(manuscript), "manuscript_sha256": INPUT_HASH,
            "shared_stage_source_hashes": stage_hashes,
            "report_source_sha256": _hash(__file__),
            "baseline_timing": "Preserved historical measured timings; baseline workers were not rerun contemporaneously.",
            "limits": {"worker_private_commit_bytes": CAP, "worker_blas_threads": 1,
                       "measured_workers_at_once": 1}}


def prescribed_preflight(stages, results):
    """Parameter-only metadata: no sampler/RNG/history allocation or trajectory."""
    from .shared_tangent import SharedTangentLogOdds
    settings = defaultdict(set)
    for source, stage in stages.items():
        for run in stage["runs"]:
            if run.spec["algorithm"] == "shared":
                s = run.spec
                settings[(s["d"], s["B"], s["T"], s["config"].get("R", 1.))].add(source)
    rows = []
    for (d, B, T, R), sources in sorted(settings.items()):
        result = SharedTangentLogOdds.preflight(d, B, T, {"mode": "prescribed", "R": R})
        p = result["prescribed"]
        rows.append({"d": d, "B": B, "T": T, "R": R, "sources": sorted(sources),
                     "m": p["m"], "N_max": p["N_max"], "ell_max": p["ell_max"],
                     "worst_case_total_transitions": result["estimated_total_moves"],
                     "estimated_state_bytes": result["estimated_state_bytes"],
                     "state_estimate_exceeds_worker_cap": result["estimated_state_bytes"] > CAP,
                     "default_guard_feasible": result["feasible"],
                     "guard_reasons": result["budget_reasons"], "full_preflight": result})
    artifact = {"schema_version": 1, "allocation_free": True,
                "interpretation": "Parameter-only worst-case upper schedule estimates using gamma>=lambda, no skips and the maximum allowed feature energies. A default guard rejection is not a lower bound, proof of computational impossibility, or statement of actual data-dependent work. State estimates count the raw-plus-prefix block representation and spare arrays, excluding Python/transient allocations.",
                "rows": rows}
    path = results / "shared_prescribed_preflight.json"
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    return artifact


def _regime(setting):
    truth = setting.get("B_generator")
    family = setting["family"] + (f" [Bgen={truth:g}]" if _finite(truth) else "")
    return f"{family}  d={setting['d']:,}, B={setting['B']:g}, T={setting['T']:,}"


def figures(stages, summaries, directory):
    os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    directory.mkdir(parents=True, exist_ok=True)
    selected = [r for r in summaries if r["source"] in ("screen", "deep_scale")
                and r["checkpoint"] == "pair_common"]
    regimes = sorted({json.dumps({"source": r["source"], "setting": r["setting"]}, sort_keys=True)
                      for r in selected}, key=lambda value: (json.loads(value)["source"],
                                                              json.loads(value)["setting"]["family"],
                                                              json.loads(value)["setting"].get("B_generator", 0),
                                                              json.loads(value)["setting"]["d"],
                                                              json.loads(value)["setting"]["B"],
                                                              json.loads(value)["setting"]["T"]))
    positions = {key: i for i, key in enumerate(regimes)}
    labels = [json.loads(key)["source"] + " · " + _regime(json.loads(key)["setting"]) for key in regimes]
    height = max(5., .40*len(regimes)+1.8)
    outputs = []
    for kind, filename, xlabel in (("loss", "shared_loss_difference.png", "New minus baseline loss per round (negative favors new)"),
                                   ("time", "shared_time_ratio.png", "New / baseline online time at their exact paired prefix")):
        fig, ax = plt.subplots(figsize=(13.5, height))
        displayed_values = []
        for algorithm, offset in (("ours", -.20), ("sigmoid", 0.), ("aioli", .20)):
            rows = [r for r in selected if r["old_algorithm"] == algorithm and r["datasets_available"]]
            keyname = "mean_average_loss_difference" if kind == "loss" else "mean_time_ratio"
            rows = [r for r in rows if _finite(r[keyname]) and (kind == "loss" or r[keyname] > 0)]
            xs = [r[keyname] for r in rows]
            displayed_values.extend(xs)
            ys = [positions[json.dumps({"source": r["source"], "setting": r["setting"]}, sort_keys=True)] + offset
                  for r in rows]
            ax.scatter(xs, ys, label=NAMES[algorithm], color=COLORS[algorithm], s=30, zorder=3)
        ax.axvline(0 if kind == "loss" else 1, color="#777777", lw=1)
        if kind == "loss":
            ax.set_xscale("symlog", linthresh=.01)
            if displayed_values:
                # Linear autoscaling around a zero reference line can allocate
                # half a symlog chart to an empty negative range. Preserve zero
                # and a small linear margin, then pad actual signed extremes.
                lower = min(-.003, 1.7 * min(displayed_values))
                upper = max(.003, 1.7 * max(displayed_values))
                ax.set_xlim(lower, upper)
        else:
            ax.set_xscale("log")
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=.18)
        ax.legend(loc="lower left", bbox_to_anchor=(.02, 1.005), ncol=3, frameon=False)
        ax.set_title("Recorded paired prefixes; descriptive means across datasets\nUnavailable comparisons omitted; attained horizons can differ", pad=60, fontsize=12)
        fig.tight_layout()
        path = directory / filename
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
    fig, ax = plt.subplots(figsize=(13.5, height))
    for algorithm, offset in (("shared", -.27), ("ours", -.09), ("sigmoid", .09), ("aioli", .27)):
        xs, ys, failed_x, failed_y = [], [], [], []
        for key, pos in positions.items():
            identity = json.loads(key)
            runs = [r for r in stages[identity["source"]]["runs"] if r.spec["algorithm"] == algorithm
                    and {k: v for k, v in r.case.items() if k not in ("id", "data_seed")} == identity["setting"]]
            peaks = [r.peak_private/MIB for r in runs if r.peak_private > 0]
            if peaks:
                # A refused large allocation can leave a deceptively small
                # observed process peak. Mark zero-round failures explicitly.
                failure = bool(runs) and all(r.n == 0 for r in runs)
                (failed_x if failure else xs).append(max(peaks))
                (failed_y if failure else ys).append(pos+offset)
        ax.scatter(xs, ys, label=NAMES[algorithm], color=COLORS[algorithm], s=28)
        ax.scatter(failed_x, failed_y, color=COLORS[algorithm], marker="X", s=68)
    ax.scatter([], [], marker="X", s=68, color="#333333", label="Zero-round failure")
    ax.axvline(CAP/MIB, color="#777777", ls="--", lw=1, label="2 GiB cap")
    ax.set_xscale("log"); ax.set_yticks(range(len(labels)), labels, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("Maximum recorded whole-attempt private commit across seeds (MiB)")
    ax.set_title("Memory over each full attempt; completed horizons differ\nX: failed before completing a round; the observed peak is not the memory required", pad=75, fontsize=11)
    ax.legend(loc="lower left", bbox_to_anchor=(.02, 1.005), ncol=3, frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=.18); fig.tight_layout()
    path = directory / "shared_memory.png"; fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig); outputs.append(path)
    diagnostic_stages = [s for s in ("screen", "deep_scale", "confirmation", "deep_budget", "long_horizon", "deep_long") if s in stages]
    totals = {}
    for source in diagnostic_stages:
        counter = Counter()
        for run in stages[source]["runs"]:
            if run.spec["algorithm"] == "shared":
                counter.update((run.worker.get("metrics") or {}).get("counters", {}))
        totals[source] = counter
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    x = np.arange(len(diagnostic_stages))
    for metric, denominator, offset, color, label in (
            ("cache_misses", "proximal_steps", -.18, "#7735a4", "Cache misses / completed move"),
            ("poisson_cap_continuations", "poisson_tests", .18, "#2570b7", "Full continuation / residual test")):
        values = [totals[s][metric]/totals[s][denominator] if totals[s][denominator] else np.nan for s in diagnostic_stages]
        axes[0].bar(x+offset, values, width=.34, color=color, label=label)
    axes[0].set_ylabel("Recorded count ratio"); axes[0].legend(fontsize=8)
    for metric, offset, color, label in (("gradient_refresh_rows", -.24, "#7735a4", "Gradient refresh rows"),
                                        ("poisson_rows", 0., "#2570b7", "Poisson sampled rows"),
                                        ("fallback_rows", .24, "#be4938", "Full-continuation rows")):
        values = [totals[s][metric]/totals[s]["predictions"] if totals[s]["predictions"] else np.nan for s in diagnostic_stages]
        axes[1].bar(x+offset, values, width=.23, color=color, label=label)
    axes[1].set_yscale("symlog", linthresh=1); axes[1].set_ylabel("Recorded rows / completed prediction")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x, diagnostic_stages, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Shared-tangent diagnostics over whole attempts\nCounters may include work in a timed-out unfinished prediction; these are not instruction counts", fontsize=11)
    fig.tight_layout()
    path = directory / "shared_diagnostics.png"; fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig); outputs.append(path)
    return outputs


def write_markdown(bundle, stages, selected, outputs, output):
    total_new = (sum(stage["public"]["new_runs"] for stage in stages.values())
                 + sum(stage["new_runs"] for stage in bundle["auxiliary_stages"].values()))
    lines = ["# Shared-tangent manuscript: measured comparison", "",
             "The supplied shared-tangent algorithm was implemented with its retained tangent, clipped-center rejection repair, capped Poisson-factor test, one tilt sweep, curvature ledger, and prescribed skip threshold. Practical caps limit tilts and mixing steps; these measured capped runs do not inherit the manuscript's regret guarantee.", "",
             f"Held-out calibration selected `{selected['label']}`: `{json.dumps(selected['config'], sort_keys=True)}`. Here tilts are correlated chain states, not independent particles. The six-configuration calibration is recorded in " + _link(BASE / "results/shared_calibration/selected_config.json") + ".", "",
             "Its within-dataset all-configuration common prefixes were **" + ", ".join(str(n) for n in sorted(selected["common_prefixes"].values()))
             + " rounds**. This is a short, noisy calibration, so the selected cap should not be interpreted as an established optimum.", "",
             "Every new run reused the exact previously stored features, labels, configured horizon, time allowance and 2 GiB private-commit cap, with one worker and one BLAS thread. Baseline measurements are preserved historical runs, so timing ratios also reflect possible between-session machine variation. Loss/regret differences always use the same completed data prefix.", ""]
    outcomes = []
    for old in ("ours", "sigmoid", "aioli"):
        pairs = [p for source, stage in stages.items() if source in ("screen", "deep_scale")
                 for p in stage["public"]["pairwise"]
                 if p["old"]["algorithm"] == old and p["common"]["available"]]
        ratios = [p["common"]["time_ratio"] for p in pairs if _finite(p["common"].get("time_ratio"))]
        outcomes.append({"old": old, "available": len(pairs),
                         "lower_regret": sum(p["common"]["loss_difference"] < 0 for p in pairs),
                         "faster": sum(value < 1 for value in ratios), "timed": len(ratios),
                         "median_time_factor": median(ratios) if ratios else None})
    lead = ("Under the tested practical caps, the shared-tangent version achieved neither lower recorded loss nor faster time at any available paired prefix in the screen and scaling sets."
            if all(row["available"] and not row["lower_regret"] and not row["faster"] for row in outcomes)
            else "The table pools the screen and deeper scaling sets at each pair's exact common completed prefix.")
    outcome_lines = [lead, "", "| Baseline | Available pairs | New lower regret | New faster | Median new / baseline time |",
                     "|---|---:|---:|---:|---:|"]
    for row in outcomes:
        outcome_lines.append(f"| {NAMES[row['old']]} | {row['available']} | {row['lower_regret']}/{row['available']} | {row['faster']}/{row['timed']} | {_fmt(row['median_time_factor'], 2)}× |")
    outcome_lines += ["", "These are practical-budget results with reused historical baseline timings. Prefixes are time-selected; this table does not compare the fully prescribed theoretical schedules.", ""]
    lines[2:2] = outcome_lines
    lines += ["", "## Completed attempts", "", f"There are **{total_new} new attempts** in total, including calibration and smoke checks. Baseline attempts below are reused records and are not counted as newly executed work.", "",
              "| Stage | New attempts | New terminal statuses | Reused baseline attempts |", "|---|---:|---|---:|"]
    for source, p in bundle["auxiliary_stages"].items():
        counts = "; ".join(f"{k}: {v}" for k, v in sorted(p["new_status_counts"].items()))
        lines.append(f"| {source} | {p['new_runs']} | {counts} | 0 |")
    for source, stage in stages.items():
        p = stage["public"]
        counts = "; ".join(f"{k}: {v}" for k, v in sorted(p["new_status_counts"].items()))
        lines.append(f"| {source} | {p['new_runs']} | {counts} | {p['old_runs']} |")
    lines += ["", "Zero-round errors remain in these counts and never become zero-loss comparisons. Full-horizon, time-budget and preflight-failure outcomes are distinct in the machine-readable results.", "",
              "## Screen and scaling: four-method common prefixes", "",
              "Regret intervals below are means of numerical optimization intervals at each dataset's four-method common completed prefix. They are optimization uncertainty intervals, not confidence intervals or floating-point certificates. If any method reaches zero rounds, that dataset's four-method comparison is unavailable. Different seeds can have different common prefixes.", "",
              "Bgen is the stored generator parameter, distinct from the prior/comparator radius B. Ordinary-signal and strong-signal settings retain separate rows even when their family, d, B and T match.", "",
              "| Stage / data family | Truth Bgen | d | B | Configured T | Prefix range; available datasets | New regret | Original regret | Sigmoid regret | AIOLI regret |", "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for source in ("screen", "deep_scale"):
        if source not in stages:
            continue
        regimes = defaultdict(list)
        for cid, runs in stages[source]["groups"].items():
            case = runs[0].case
            key = (case["family"], case.get("B_generator"), case["d"], case["B"], case["T"])
            regimes[key].append(runs)
        for (family, Bgen, d, B, T), datasets in sorted(regimes.items()):
            good = [runs for runs in datasets if all(r.summary and r.summary.get("common") for r in runs)]
            prefixes = [runs[0].summary["common"]["round"] for runs in good]
            cells = []
            for algorithm in ("shared", "ours", "sigmoid", "aioli"):
                points = [r.summary["common"] for runs in good for r in runs if r.spec["algorithm"] == algorithm]
                cells.append("—" if not points else f"[{_fmt(_mean(p['regret_lower'] for p in points))}, {_fmt(_mean(p['regret_upper'] for p in points))}]")
            lines.append(f"| {source} / {family} | {_fmt(Bgen, 0)} | {d:,} | {B:g} | {T:,} | {_span(prefixes)}; {len(good)}/{len(datasets)} | " + " | ".join(cells) + " |")
    lines += ["", "## Pairwise loss, computation and memory", "",
              "Negative loss differences favor the new algorithm. The comparator cancels exactly at a shared prefix, so each recorded loss difference is also the regret difference there. The time ratio is new / baseline cumulative online time at that same prefix; values below one mean the new run used less recorded time. Whole-attempt memory is shown separately and is not matched to these prefixes. Exact checkpoint data at 100, 500 and 1,000 are retained in JSON, including unavailable checkpoints.", "",
              "In the memory figure, **X marks a zero-round failure**. In particular, the old dense AIOLI allocation at d=50,000 was refused: its small observed process peak does not mean that the algorithm successfully operates there with little memory. The cap prevents the attempted array allocation from becoming resident.", "",
              "| Stage / regime | Baseline configuration | Paired prefix range | Datasets | Mean loss difference / round | Mean time ratio |", "|---|---|---|---:|---:|---:|"]
    for row in bundle["paired_seed_summaries"]:
        if row["checkpoint"] != "pair_common":
            continue
        rounds = [n for v in row["dataset_values"] if v["available"] for n in v["rounds"]]
        lines.append(f"| {row['source']} / {_regime(row['setting'])} | {NAMES.get(row['old_algorithm'], row['old_algorithm'])} `{row['old_label']}` | {_span(rounds)} | {row['datasets_available']}/{row['datasets_expected']} | {_fmt(row['mean_average_loss_difference'], 6)} | {_fmt(row['mean_time_ratio'])} |")
    lines += ["", "## Confirmation with independent sampler repetitions", "",
              "The fixed 500-round comparison is primary; 100 rounds is a prespecified secondary checkpoint. Each dataset first averages three new runs and three original/sigmoid runs (AIOLI has one deterministic run). The interval uses dataset means as the sampling units. It is reported only if all prescribed repetitions of every dataset reached the checkpoint; no successful subset replaces a missing primary result.", "",
              "| Round | Baseline | Complete datasets | Mean cumulative loss difference | Approximate dataset-cluster 95% interval |", "|---:|---|---:|---:|---|"]
    for row in bundle["confirmation"]:
        ci = row["cluster_t_interval_95"]
        interval = "unavailable" if ci is None else f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"
        lines.append(f"| {row['round']} | {NAMES[row['old_algorithm']]} | {row['datasets_available']}/{row['datasets_expected']} | {_fmt(row['mean_loss_difference'])} | {interval} |")
    lines += ["", "## Long horizons and prescribed runs", "",
              "The configured horizon does not imply that a capped sampler completed it. These rows show actual attained rounds. Regret intervals use each run's own final prefix and must not be compared across different attained horizons.", "",
              "| Stage | Algorithm | d / B / configured T | Completed rounds | Status | Online seconds | Whole-attempt peak MiB | Final numerical regret interval |", "|---|---|---|---:|---|---:|---:|---|"]
    for source in ("prescribed", "long_horizon", "deep_long"):
        if source not in stages:
            continue
        for run in sorted(stages[source]["runs"], key=lambda r: (r.spec["case_id"], r.spec["algorithm"], r.spec["label"])):
            p = run.summary.get("final") if run.summary else None
            interval = "—" if not p else f"[{_fmt(p['regret_lower'])}, {_fmt(p['regret_upper'])}]"
            s = run.spec
            lines.append(f"| {source} | {NAMES.get(s['algorithm'], s['algorithm'])} `{s['label']}` | {s['d']:,} / {s['B']:g} / {s['T']:,} | {run.n:,} | {run.worker.get('status')} | {_fmt(run.worker.get('online_seconds'))} | {_fmt(run.peak_private/MIB)} | {interval} |")
    pf = bundle["prescribed_preflight"]
    lines += ["", f"An allocation-free prescribed preflight was also evaluated on all {len(pf['rows'])} unique parameter settings. The default worst-case transition guard rejects {sum(not r['default_guard_feasible'] for r in pf['rows'])} of these settings. "
              + _link(BASE / "results/shared_prescribed_preflight.json", "Explicit prescribed counts and memory estimates")
              + " retain m, maximum initial/inter-tilt transitions, total worst-case transitions, state estimates and every guard reason. These are sufficient schedule upper estimates, not minimum necessary work or impossibility results; real data curvature and skipped rounds can lower the actual work."]
    lines += ["", "## Internal work and numerical scope", "",
              "The new implementation stores raw features and separate weighted prefixes, using the manuscript's explicitly permitted extra history array for floating-point stability. The Gaussian sampler uses only uniforms and the manuscript's single two-sided exponential envelope. The rejection continuation retains the essential factor `1-u/C`. Numerical guards raise failures rather than silently changing a target distribution.", "",
              "The diagnostic figure counts cache misses, full gradient refresh rows, sampled Poisson rows and full-residual continuation rows. Counters describe recorded kernels, not measured arithmetic instructions; unfinished timed-out kernels can have incomplete counters. A high first-proposal acceptance rate does not alone demonstrate fast mixing. Reported sampling budgets are practical caps and the exact-real regret theorem does not certify their measured losses.", ""]
    lines += ["| New-run stage | Poisson-test time / predict time | Gradient-refresh time / predict time | Conditional-proposal time / predict time |", "|---|---:|---:|---:|"]
    for source, stage in stages.items():
        totals = Counter()
        for run in stage["runs"]:
            if run.spec["algorithm"] == "shared":
                totals.update((run.worker.get("metrics") or {}).get("times", {}))
        denominator = totals["predict"]
        values = [totals[key]/denominator if denominator else None for key in
                  ("poisson_test", "gradient_refresh", "conditional_proposal")]
        lines.append(f"| {source} | " + " | ".join(_fmt(v) for v in values) + " |")
    lines += ["", "Timer scopes are hierarchical: Poisson testing includes its full-residual continuation, conditional draws include proposal/testing/repair work, and prediction includes the whole procedure. The fractions above are separate diagnostics and are not stacked or summed as disjoint runtime components.", ""]
    for path in outputs:
        lines += [f"![{path.stem.replace('_', ' ')}]({path.resolve().as_posix()})", ""]
    lines += ["## Reproducibility and raw results", "",
              "- " + _link(BASE / "SHARED_TANGENT_INPUT.md", "Exact supplied manuscript") + ".",
              "- " + _link(BASE / "shared_tangent.py", "Shared-tangent implementation") + " and " + _link(BASE / "shared_tangent_primitives.py", "uniform-only primitives") + ".",
              "- " + _link(BASE / "results/shared_comparison.json", "All run metadata, exact paired prefixes, checkpoints and confirmation clusters") + ".",
              "- " + _link(BASE / "results/shared_comparison_provenance.json", "Source hashes, hard limits and provenance") + ".",
              "- " + _link(BASE / "README.md", "Experiment instructions") + ".", "",
              "The original AIOLI, proximal, sigmoid, common and data source hashes match the preserved earlier controls. All results and figures here are produced from the terminal manifests and raw recorded arrays; no baseline measurement is overwritten."]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=BASE / "results")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--output", type=Path, default=ROOT / "SHARED_TANGENT_EXPERIMENT_RESULT.md")
    args = parser.parse_args()
    _guard()
    results = args.results.resolve()
    selected = read_json(results / "shared_calibration/selected_config.json")
    source_provenance = provenance(results, args.sources)
    auxiliary = load_auxiliary_counts(results)
    stages = load_stages(results, args.sources)
    summaries = seed_summaries(stages)
    preflight = prescribed_preflight(stages, results)
    bundle = {"schema_version": 1, "complete": True, "selected_config": selected,
              "auxiliary_stages": auxiliary,
              "stages": {name: stage["public"] for name, stage in stages.items()},
              "paired_seed_summaries": summaries, "confirmation": confirmation(stages),
              "prescribed_preflight": preflight,
              "interpretation": {
                  "difference": "New minus preserved baseline; negative favors the new algorithm.",
                  "regret_difference": "The identical-prefix comparator cancels exactly; values remain float64 numerical recordings.",
                  "prefixes": "Pair-common horizons are selected by attained completion/time budgets; fixed100/500/1000 availability is explicit.",
                  "memory": "Whole-attempt peak is separate from exact-round snapshot memory; missing prefix snapshots are null.",
                  "baseline_times": "Historical measured timings are reused, not measured contemporaneously.",
                  "inference": "Screen and scale are descriptive. Only complete fixed-checkpoint confirmation receives approximate dataset-cluster intervals.",
                  "guarantee": "Practical tilt/mixing caps do not inherit the manuscript's regret theorem."}}
    result_path = results / "shared_comparison.json"
    result_path.write_text(json.dumps(bundle, indent=2, allow_nan=False), encoding="utf-8")
    outputs = figures(stages, summaries, BASE / "figures")
    write_markdown(bundle, stages, selected, outputs, args.output)
    source_provenance.update(report=str(args.output.resolve()), report_sha256=_hash(args.output),
                             comparison_json=str(result_path), comparison_json_sha256=_hash(result_path),
                             prescribed_preflight_sha256=_hash(results / "shared_prescribed_preflight.json"),
                             figures={str(path): _hash(path) for path in outputs},
                             input_artifacts={str(results / ("shared_compare_"+source) / filename):
                                              _hash(results / ("shared_compare_"+source) / filename)
                                              for source in args.sources
                                              for filename in ("manifest.json", "summary.json", "curves.json", "cases.json")})
    provenance_path = results / "shared_comparison_provenance.json"
    provenance_path.write_text(json.dumps(source_provenance, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"report": str(args.output), "json": str(result_path),
                      "new_runs": sum(s["public"]["new_runs"] for s in stages.values()),
                      "paired_rows": sum(len(s["public"]["pairwise"]) for s in stages.values()),
                      "figures": [str(path) for path in outputs]}))


if __name__ == "__main__":
    main()
