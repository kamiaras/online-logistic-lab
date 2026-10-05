"""Isolated gated recorder for the fixed exploratory computation follow-up."""
import sys
if __name__ == "__main__" and "--gate" in sys.argv:
    if sys.stdin.readline().strip() != "GO":
        raise SystemExit("Missing supervisor gate")
import json
import math
from pathlib import Path
from time import perf_counter
import traceback
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from .memory import memory_snapshot


def run(spec):
    if spec["method"] == "current16":
        from .curvature_persistent_log_odds import CurvaturePersistentLogOdds as Algorithm
    elif spec["method"] == "legacy16":
        from .computation_followup import LegacyStepCurvatureLogOdds as Algorithm
    else:
        from .proximal import ProximalLogOdds as Algorithm
    out, source = Path(spec["output"]), Path(spec["data_path"])
    d, n = spec["d"], spec["rounds"]
    X = np.fromfile(source/"X.bin", dtype="<f8", count=n*d).reshape(n, d)
    Y = np.fromfile(source/"y.bin", dtype="i1", count=n)
    native = ["proximal_steps", "rqo_proposals", "full_test_rows", "gradient_refresh_rows",
              "cache_append_rows", "history_particle_pairs", "logistic_pairs", "potential_calls",
              "projected_center_steps", "estimated_matmul_flops", "summary_queries",
              "summary_certified_accepts", "summary_full_continuations",
              "history_rows_avoided_by_squeeze", "summary_query_coordinates",
              "summary_append_coordinates", "summary_rebuild_coordinates"]
    saved = {key: [] for key in ["logits", "guarded_logits", "losses", "guarded_losses",
                               "online_times", "N_actual", "inveta", "mu", "skipped",
                               "original_gradient_feature_lane_rows_derived"]+native}
    a, online, status, error, original_gradient_pairs = None, 0., "running", None, 0
    with threadpool_limits(limits=1):
        try:
            start = perf_counter()
            a = Algorithm(d, spec["B"], spec["T"], spec["seed"], spec["config"])
            online += perf_counter()-start
            for t, (x, y) in enumerate(zip(X, Y), 1):
                if online >= spec["seconds"]:
                    status = "time_budget"
                    break
                start = perf_counter()
                output = a.predict(x)
                schedule = a._last_schedule
                raw = output if spec["method"] == "original256" else schedule["raw_logit"]
                a.observe(x, int(y))
                online += perf_counter()-start
                cap = math.log(4.)+math.log(t)+math.log(t+1)
                guarded = float(np.clip(raw, -cap, cap))
                if spec["method"] == "original256":
                    original_gradient_pairs += (schedule["k"]+2)*a.m*schedule["N_actual"]*t
                values = dict(logits=raw, guarded_logits=guarded,
                              losses=float(np.logaddexp(0., -int(y)*raw)),
                              guarded_losses=float(np.logaddexp(0., -int(y)*guarded)),
                              online_times=online, N_actual=schedule["N_actual"],
                              inveta=schedule["inveta"], mu=schedule["mu"],
                              skipped=schedule.get("skipped", False),
                              original_gradient_feature_lane_rows_derived=original_gradient_pairs)
                values.update({key:a.metrics.counters[key] for key in native})
                for key in saved:
                    saved[key].append(values[key])
                if t % 50 == 0:
                    (out/"progress.json").write_text(json.dumps(dict(round=t, online_seconds=online)))
            if len(saved["logits"]) == n:
                status = "completed"
        except Exception as exc:
            status, error = "error", repr(exc)
            (out/"traceback.txt").write_text(traceback.format_exc())
        result = dict(status=status, error=error, round=len(saved["logits"]),
                      method=spec["method"], algorithm_class=Algorithm.__name__,
                      configured_horizon=spec["T"], fixed_endpoint=n,
                      cumulative_loss=sum(saved["losses"]), guarded_loss=sum(saved["guarded_losses"]),
                      online_seconds=online, memory=memory_snapshot(), threadpools=threadpool_info(),
                      metrics=a.metrics.as_dict() if a else {}, state_bytes=a.state_bytes() if a else 0,
                      actual_population=a.m if a else None, diagnostics=a.diagnostics() if a else {},
                      original_gradient_feature_lane_rows_derived=original_gradient_pairs,
                      exploratory_after_initial_results=True, cap_regret_certificate=False)
        if saved["logits"]:
            np.savez_compressed(out/"trajectory.npz", **saved, labels=Y[:len(saved["logits"])])
    (out/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()))
