"""Lightweight scalar replay of completed outputs; never imports NumPy/BLAS."""
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

from .online_confidence import OnlineConfidence, PRIORS, SCALES, logsumexp, softplus


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "ours": ROOT / "experiments/results/computation_first_overnight_d1000_B1000_T100000/trajectory.bin",
    "released_aioli": ROOT / "experiments/results/author_baselines_d1000_B1000_T100000/author_aioli/trajectory.bin",
    "labels": ROOT / "experiments/data/signal_d1000_B1000_T100000_s72001/y.bin",
    "confidence_source": Path(__file__).with_name("online_confidence.py"),
    "replay_source": Path(__file__),
    "protocol": ROOT / "AIOLI_INSPIRED_REPLAY_PLAN.md",
    "selected_source": ROOT / "experiments/computation_first_log_odds.py",
    "author_source": ROOT / "research/author_code_comparison_2026-09-11/a9a_experiment.py",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(out):
    out.mkdir(parents=True, exist_ok=False)
    hashes = {k: digest(v) for k, v in FILES.items()}
    labels = array.array("b", FILES["labels"].read_bytes())
    n = len(labels)
    if n != 100000 or any(y not in (-1, 1) for y in labels):
        raise ValueError("Unexpected label count or encoding")
    ours = FILES["ours"].read_bytes()
    author = FILES["released_aioli"].read_bytes()
    if len(ours) != 40*n or len(author) != 40*n:
        raise ValueError("Expected completed 100000-round trajectories")
    model = OnlineConfidence()
    loss_totals = {k: 0. for k in ("ours", "released_aioli", "confidence")}
    expert_totals = [0.] * len(SCALES)
    abs_totals = {"ours": 0., "released_aioli": 0.}
    max_abs = {"ours": 0., "released_aioli": 0.}
    sign_correct = {"ours": 0, "released_aioli": 0}
    max_saved_loss_error = max_identity_error = 0.
    checkpoints = []
    start = time.perf_counter()
    with (out / "confidence_trajectory.bin").open("wb") as trace:
        for t, (y, old, aioli) in enumerate(zip(labels, struct.iter_unpack("<5d", ours),
                                              struct.iter_unpack("<5d", author)), 1):
            for name, row in (("ours", old), ("released_aioli", aioli)):
                z = row[1]
                recalculated = softplus(-y*z)
                max_saved_loss_error = max(max_saved_loss_error, abs(recalculated-row[2]))
                loss_totals[name] += row[2]
                abs_totals[name] += abs(z)
                max_abs[name] = max(max_abs[name], abs(z))
                sign_correct[name] += int(y*z > 0.)
            z = model.predict(old[1])
            loss = model.observe(y)
            loss_totals["confidence"] += loss
            trace.write(struct.pack("<2d", z, loss))
            expert_totals = [v+softplus(-y*a*old[1]) for v, a in zip(expert_totals, SCALES)]
            if t in (1000, 10000, 50000, 100000):
                identity = -logsumexp(math.log(w)-v for w, v in zip(PRIORS, expert_totals))
                error = abs(identity-loss_totals["confidence"])
                max_identity_error = max(max_identity_error, error)
                checkpoints.append(dict(rounds=t, cumulative_loss=dict(loss_totals),
                    mean_loss={k: v/t for k, v in loss_totals.items()},
                    neutral_mean_loss=math.log(2.),
                    fixed_scale_mean_losses=[v/t for v in expert_totals],
                    confidence_weights=model.diagnostics()["weights"],
                    mixture_identity_error=error,
                    base_bound_slack=loss_totals["ours"]+math.log(2)-loss_totals["confidence"],
                    neutral_bound_slack=t*math.log(2)+math.log(4)-loss_totals["confidence"]))
    elapsed = time.perf_counter()-start
    after = {k: digest(v) for k, v in FILES.items()}
    checks = dict(input_source_hashes_unchanged=hashes == after,
                  saved_losses_match=max_saved_loss_error < 1e-10,
                  telescoping_identity_matches=max_identity_error < 1e-6,
                  base_bound_at_checkpoints=all(c["base_bound_slack"] >= -1e-6 for c in checkpoints),
                  neutral_bound_at_checkpoints=all(c["neutral_bound_slack"] >= -1e-6 for c in checkpoints),
                  no_sampler_or_feature_data_loaded=True)
    result = dict(status="passed" if all(checks.values()) else "failed", rounds=n,
                  protocol="AIOLI_INSPIRED_REPLAY_PLAN.md", exploratory=True,
                  held_out_confirmation=False, full_online_rerun=False,
                  scopes="Replay completed guarded logits, chronological labels, no sampler feedback",
                  scales=SCALES, initial_weights=PRIORS, checkpoints=checkpoints,
                  baseline_mean_abs_logits={k: v/n for k, v in abs_totals.items()},
                  baseline_max_abs_logits=max_abs,
                  strict_sign_accuracy={k: v/n for k, v in sign_correct.items()},
                  max_saved_loss_error=max_saved_loss_error,
                  max_mixture_identity_error=max_identity_error,
                  replay_wall_seconds=elapsed, checks=checks,
                  inputs={k: {"path": str(v.relative_to(ROOT)), "sha256": hashes[k]}
                          for k, v in FILES.items()},
                  trace=dict(file="confidence_trajectory.bin", columns=["confidence_logit", "loss"],
                             dtype="little-endian float64", sha256=digest(out / "confidence_trajectory.bin")),
                  final_confidence=model.diagnostics())
    (out / "replay.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "checkpoints", "baseline_mean_abs_logits",
                                            "strict_sign_accuracy", "replay_wall_seconds", "checks")}, indent=2))
    if not all(checks.values()):
        raise RuntimeError("Replay checks failed; retain output for diagnosis")


if __name__ == "__main__":
    replay(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "experiments/results/aioli_confidence_replay_2026-09-11")
