"""Gated recorder for paired batched computation controls."""
import sys
if __name__ == "__main__" and "--gate" in sys.argv:
    if sys.stdin.readline().strip() != "GO":
        raise SystemExit("Missing supervisor gate")
import json
from pathlib import Path
from time import perf_counter
import traceback
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from .memory import memory_snapshot


def run(spec):
    if spec["backend"] == "full":
        from .batched_persistent_log_odds import BatchedPersistentLogOdds as Algorithm
    else:
        from .curvature_persistent_log_odds import CurvaturePersistentLogOdds as Algorithm
    source, out = Path(spec["data_path"]), Path(spec["output"])
    d, n = spec["d"], spec["rounds"]
    X = np.fromfile(source/"X.bin", dtype="<f8", count=d*n).reshape(n, d)
    Y = np.fromfile(source/"y.bin", dtype="i1", count=n)
    keys = ["logits", "guarded_logits", "losses", "guarded_losses", "online_times",
            "skipped", "lanes", "transitions", "moves", "full_test_rows",
            "gradient_refresh_rows", "estimated_matmul_flops", "proposals"]
    saved = {k: [] for k in keys}
    a, online, status, error = None, 0., "running", None
    with threadpool_limits(limits=1):
        try:
            start = perf_counter()
            a = Algorithm(d, spec["B"], spec["T"], spec["seed"], spec["config"])
            online += perf_counter()-start
            for x, y in zip(X, Y):
                if online >= spec["seconds"]:
                    status = "time_budget"
                    break
                start = perf_counter()
                guarded = a.predict(x)
                raw = a._last_schedule["raw_logit"]
                a.observe(x, int(y))
                online += perf_counter()-start
                schedule, counts = a._last_schedule, a.metrics.counters
                values = dict(logits=raw, guarded_logits=guarded,
                    losses=float(np.logaddexp(0., -int(y)*raw)),
                    guarded_losses=float(np.logaddexp(0., -int(y)*guarded)), online_times=online,
                    skipped=schedule["skipped"], lanes=a.m, transitions=schedule["N_actual"],
                    moves=counts["proximal_steps"], full_test_rows=counts["full_test_rows"],
                    gradient_refresh_rows=counts["gradient_refresh_rows"],
                    estimated_matmul_flops=counts["estimated_matmul_flops"], proposals=counts["rqo_proposals"])
                for k in keys:
                    saved[k].append(values[k])
                if len(saved["logits"]) % 50 == 0:
                    (out/"progress.json").write_text(json.dumps(dict(round=len(saved["logits"]),
                        online_seconds=online)))
            if len(saved["logits"]) == n:
                status = "completed"
        except Exception as exc:
            status, error = "error", repr(exc)
            (out/"traceback.txt").write_text(traceback.format_exc())
        result = dict(status=status, error=error, round=len(saved["logits"]),
            configured_horizon=spec["T"], fixed_endpoint=n, online_seconds=online,
            cumulative_loss=sum(saved["losses"]), guarded_loss=sum(saved["guarded_losses"]),
            memory=memory_snapshot(), threadpools=threadpool_info(),
            metrics=a.metrics.as_dict() if a else {}, state_bytes=a.state_bytes() if a else 0,
            diagnostics=a.diagnostics() if a else {})
        if saved["logits"]:
            np.savez_compressed(out/"trajectory.npz", **saved, labels=Y[:len(saved["logits"])])
    (out/"result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()))
