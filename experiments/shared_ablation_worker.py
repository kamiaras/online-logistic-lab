"""Isolated bounded worker for the fixed-prefix ablation plan."""
import sys
if __name__ == "__main__" and "--gate" in sys.argv:
    if sys.stdin.readline().strip() != "GO":
        raise SystemExit("Missing supervisor gate")
import json
from pathlib import Path
from time import perf_counter
import traceback
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from .shared_ablation import SharedAblation, guarded_logits
from .memory import memory_snapshot


def run(spec):
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    d, T, nmax = spec["d"], spec["T"], spec["rounds"]
    source = Path(spec["data_path"])
    X = np.fromfile(source/"X.bin", dtype="<f8", count=d*nmax).reshape(nmax, d)
    Y = np.fromfile(source/"y.bin", dtype="i1", count=nmax)
    scores, starts, logits, losses, times = [], [], [], [], []
    a = None
    status, error, online = "running", None, 0.
    with threadpool_limits(limits=1):
        try:
            start = perf_counter()
            a = SharedAblation(d, spec["B"], T, spec["seed"], spec["config"])
            online += perf_counter()-start
            for x, y in zip(X, Y):
                if online >= spec["seconds"]:
                    status = "time_budget"
                    break
                start = perf_counter()
                z = a.predict(x)
                a.observe(x, int(y))
                online += perf_counter()-start
                logits.append(z)
                losses.append(float(np.logaddexp(0., -int(y)*z)))
                times.append(online)
                scores.append(a.last_scores.copy())
                starts.append(a.last_initial_scores.copy())
                if len(logits) % 50 == 0:
                    (out/"progress.json").write_text(json.dumps(dict(
                        round=len(logits), loss=sum(losses), online_seconds=online)))
            if len(logits) == nmax:
                status = "completed"
        except Exception as exc:
            status, error = "error", repr(exc)
            (out/"traceback.txt").write_text(traceback.format_exc())
        result = dict(status=status, error=error, round=len(logits),
                      configured_horizon=T, fixed_endpoint=nmax,
                      cumulative_loss=sum(losses), online_seconds=online,
                      memory=memory_snapshot(), threadpools=threadpool_info(),
                      metrics=a.metrics.as_dict() if a else {},
                      state_bytes=a.state_bytes() if a else 0)
        if logits:
            gl = guarded_logits(logits)
            result["guarded_loss"] = float(np.logaddexp(0., -Y[:len(gl)]*gl).sum())
            np.savez_compressed(out/"trajectory.npz", logits=logits, losses=losses,
                                online_times=times, scores=scores,
                                initial_scores=starts, labels=Y[:len(logits)],
                                guarded_logits=gl)
        if a is not None and a.capture:
            np.savez_compressed(out/"captures.npz",
                                **{key:np.asarray([row[key] for row in a.capture])
                                   for key in ["t", "v", "z", "g", "r"]})
    (out/"result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()))
