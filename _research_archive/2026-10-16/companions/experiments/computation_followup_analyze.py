"""Descriptive fixed-endpoint analysis of the isolated exploratory follow-up."""
import json
import hashlib
from pathlib import Path
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"experiments/results/computation_followup"


def analyze():
    specs = json.loads((OUT/"manifest.json").read_text())
    campaign = json.loads((OUT/"campaign_result.json").read_text())
    assert campaign["sources_and_data_unchanged"]
    assert json.loads((OUT/"source_hashes_before.json").read_text()) == json.loads((OUT/"source_hashes_after.json").read_text())
    assert json.loads((OUT/"data_hashes_before.json").read_text()) == json.loads((OUT/"data_hashes_after.json").read_text())
    rows = []
    for spec in specs:
        path = Path(spec["output"])
        result = json.loads((path/"result.json").read_text())
        supervisor = json.loads((path/"supervisor.json").read_text())
        arrays = np.load(path/"trajectory.npz")
        n = result["round"]
        assert n == 500 and result["status"] == "completed"
        assert supervisor["returncode"] == 0 and supervisor["reason"] is None
        assert supervisor["os_hard_limit_active"] and supervisor["hard_process_commit_limit"] == 2*1024**3
        assert all(p["num_threads"] == 1 for p in result["threadpools"])
        assert result["actual_population"] == (256 if spec["method"] == "original256" else 16)
        assert np.all(arrays["N_actual"] == 1)
        assert abs(arrays["losses"].sum()-result["cumulative_loss"]) < 1e-8
        assert abs(arrays["guarded_losses"].sum()-result["guarded_loss"]) < 1e-8
        caps = np.log(4*np.arange(1, n+1)*np.arange(2, n+2))
        np.testing.assert_allclose(arrays["guarded_logits"], np.clip(arrays["logits"], -caps, caps), atol=1e-12)
        for name in ["online_times", "proximal_steps", "full_test_rows", "gradient_refresh_rows"]:
            assert np.all(np.diff(arrays[name]) >= 0)
        counts = result["metrics"]["counters"]
        original = spec["method"] == "original256"
        gradient_pairs = (result["original_gradient_feature_lane_rows_derived"] if original
                          else counts.get("gradient_refresh_rows", 0))
        row = dict(case=spec["case"], d=spec["d"], B=spec["B"], method=spec["method"], seed=spec["seed"],
                   round=n, raw_loss_per_round=result["cumulative_loss"]/n,
                   c4_loss_per_round=result["guarded_loss"]/n, online_seconds=result["online_seconds"],
                   wall_seconds=supervisor["wall_seconds"], moves=counts["proximal_steps"],
                   private_peak_MiB=max(supervisor["peak_private_bytes"],result["memory"]["peak_private_commit"])/2**20,
                   state_bytes=result["state_bytes"], full_gradient_feature_lane_rows=gradient_pairs,
                   gradient_count_is_derived=original, full_test_rows=counts.get("full_test_rows", 0),
                   native_history_particle_pairs=counts.get("history_particle_pairs", 0),
                   incremental_cache_lane_rows=counts.get("cache_append_rows", 0),
                   matrix_flop_proxy=counts.get("estimated_matmul_flops", 0),
                   summary_query_coordinates=counts.get("summary_query_coordinates", 0),
                   summary_append_coordinates=counts.get("summary_append_coordinates", 0),
                   summary_rebuild_coordinates=counts.get("summary_rebuild_coordinates", 0),
                   history_rows_avoided_by_squeeze=counts.get("history_rows_avoided_by_squeeze", 0),
                   result_path=str(path/"result.json"))
        rows.append(row)
    groups = []
    numeric = [k for k in rows[0] if k.endswith("seconds") or k.endswith("per_round") or k in
               ["moves", "private_peak_MiB", "state_bytes", "full_gradient_feature_lane_rows", "full_test_rows",
                "native_history_particle_pairs", "incremental_cache_lane_rows", "matrix_flop_proxy",
                "summary_query_coordinates", "summary_append_coordinates", "summary_rebuild_coordinates",
                "history_rows_avoided_by_squeeze"]]
    for d in [10, 100]:
        for method in ["current16", "legacy16", "original256"]:
            selected = [r for r in rows if r["d"] == d and r["method"] == method]
            assert len(selected) == 3
            group = dict(d=d, B=selected[0]["B"], method=method, runs=3)
            group.update({key:statistics.mean(r[key] for r in selected) for key in numeric})
            group["raw_loss_range"] = [min(r["raw_loss_per_round"] for r in selected), max(r["raw_loss_per_round"] for r in selected)]
            group["c4_loss_range"] = [min(r["c4_loss_per_round"] for r in selected), max(r["c4_loss_per_round"] for r in selected)]
            groups.append(group)
    prior_rows, prior_source_mismatches = [], {}
    prior_dir = ROOT/"experiments/results/curvature_performance"
    if (prior_dir/"campaign_result.json").exists():
        old_hashes = json.loads((prior_dir/"source_hashes_before.json").read_text())
        new_hashes = json.loads((OUT/"source_hashes_before.json").read_text())
        shared = ["experiments/curvature_persistent_log_odds.py", "experiments/batched_persistent_log_odds.py",
                  "experiments/persistent_log_odds.py", "experiments/shared_tangent.py",
                  "experiments/shared_tangent_primitives.py", "experiments/proximal.py"]
        prior_source_mismatches = {k:dict(prior=old_hashes[k], followup=new_hashes[k])
                                  for k in shared if old_hashes[k] != new_hashes[k]}
        for spec in json.loads((prior_dir/"manifest.json").read_text()):
            if spec["config"]["lanes"] != 64:
                continue
            result = json.loads((Path(spec["output"])/"result.json").read_text())
            assert result["status"] == "completed" and result["round"] == 500
            prior_rows.append(dict(case=spec["case"], seed=spec["seed"], d=spec["d"],
                                   raw_loss_per_round=result["cumulative_loss"]/500,
                                   c4_loss_per_round=result["guarded_loss"]/500,
                                   online_seconds=result["online_seconds"],
                                   moves=result["metrics"]["counters"]["proximal_steps"]))
    provenance_path = ROOT/"CURVATURE_SOURCE_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["byte_verified"]
    assert hashlib.sha256((ROOT/provenance["archived_measured_source"]).read_bytes()).hexdigest() == provenance["measured_sha256"]
    assert hashlib.sha256((ROOT/provenance["measured_file"]).read_bytes()).hexdigest() == provenance["current_sha256"]
    assert prior_source_mismatches == {"experiments/curvature_persistent_log_odds.py":
                                     dict(prior=provenance["measured_sha256"], followup=provenance["current_sha256"])}
    analysis = dict(exploratory_after_initial_results=True, rows=rows, groups=groups,
                    prior64_context=prior_rows, campaign_wall_seconds=campaign["wall_seconds"],
                    prior64_source_mismatches=prior_source_mismatches,
                    source_provenance="CURVATURE_SOURCE_PROVENANCE.json",
                    between_stage_change="byte-verified preflight memory_representation metadata only",
                    audit="18 completed endpoints, raw/c4 arrays, counters, hashes, hard cap and BLAS1 verified")
    (OUT/"analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    labels = dict(current16="Current curvature 16/1", legacy16="Original step + curvature 16/1", original256="Unchanged original 256/1")
    text = ["# Exploratory computation follow-up result", "",
            "All 18 declared attempts completed 500 rounds. This stage was selected after viewing earlier performance results; it is exploratory, not confirmation. The two datasets and three sampler seeds were fixed before this stage. No core source or data hash changed during this 18-run stage.", "",
            "Restoring the original step helped the 10-dimensional case: mean raw loss/round fell from 1.931778 to 0.799005 at the same 16-lane/one-move allocation. The current step was better in the 100-dimensional case, 0.518284 versus 0.600331. The faster step choice is therefore not a universal statistical winner. Both 16-lane methods used 7,984 actual moves and roughly 1.0–1.3 online seconds; the unchanged original 256-particle method used 128,000 moves and roughly 8.3–12.5 seconds. The runtime reduction includes fewer particles, retained gradients and avoided historical evaluations, not just Python batching.", "",
            f"The campaign took {campaign['wall_seconds']:.3f} seconds. One worker and one BLAS thread ran at a time; every worker had the 2 GiB Windows hard commit cap. Five new recorder/control checks passed before measurement. No timeouts, allocation failures or worker errors occurred.", "",
            "Values below average the three sampler seeds for each fixed dataset. These are measured logistic losses, not proved regret bounds.", "",
            "| d, B | Method | Raw loss/round | c4 loss/round | Online seconds | Process-wall seconds | Peak private MiB | Moves |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        text.append(f"| {g['d']}, {g['B']:g} | {labels[g['method']]} | {g['raw_loss_per_round']:.6f} | {g['c4_loss_per_round']:.6f} | {g['online_seconds']:.6f} | {g['wall_seconds']:.6f} | {g['private_peak_MiB']:.3f} | {g['moves']:.0f} |")
    text += ["", "The smaller original step changes only the step between the two curvature controls. Their priors, pairing, fixed tilts, summaries and one-transition caps are retained. The baseline is the unchanged original ProximalLogOdds class: it retains its original independent cloud, random tilts, normal primitives and center solver. Its c4 numbers are output-only postprocessing.", "",
             "| d | Method | Full-gradient feature/lane pairs | Full residual pairs | Native original historical pairs | Summary query coordinates | Summary rebuild coordinates | Rows avoided by summary |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        original = g["method"] == "original256"
        residual_display = "not separately counted" if original else f"{g['full_test_rows']:.0f}"
        text.append(f"| {g['d']} | {labels[g['method']]} | {g['full_gradient_feature_lane_rows']:.0f}{' (derived)' if original else ''} | {residual_display} | {g['native_history_particle_pairs']:.0f} | {g['summary_query_coordinates']:.0f} | {g['summary_rebuild_coordinates']:.0f} | {g['history_rows_avoided_by_squeeze']:.0f} |")
    text += ["", "For the original class, the derived gradient count sums (k+2) m N t from its unchanged center solver. Its native historical-pair counter includes both potential and rejection work and excludes each round's current row. Curvature full-scan counters include the current row. These columns are deliberately separate. Summary query/update/rebuild arithmetic is additional to the matrix-FLOP proxy. Incremental cache pairs, all native counters and per-round snapshots are retained in analysis.json and trajectory.npz files.", ""]
    if prior_rows and not prior_source_mismatches:
        text += ["The preceding curvature 64/1 stage used the same verified sampler source hashes and the same six fixed cases. It supplies descriptive context for reducing the population, not an additional newly randomized comparison:", "",
                 "| d | Prior curvature 64/1 raw loss/round | Prior c4 loss/round | Prior online seconds | New 16/1 raw loss/round | New c4 loss/round | New online seconds | Move ratio new/prior |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for d in [10, 100]:
            old = [r for r in prior_rows if r['d'] == d]
            new = next(g for g in groups if g['d'] == d and g['method'] == 'current16')
            means = {k:statistics.mean(r[k] for r in old) for k in ['raw_loss_per_round','c4_loss_per_round','online_seconds','moves']}
            text.append(f"| {d} | {means['raw_loss_per_round']:.6f} | {means['c4_loss_per_round']:.6f} | {means['online_seconds']:.6f} | {new['raw_loss_per_round']:.6f} | {new['c4_loss_per_round']:.6f} | {new['online_seconds']:.6f} | {new['moves']/means['moves']:.3f} |")
    elif prior_source_mismatches:
        text += ["The prior 64/1 curvature stage is retained as descriptive context in analysis.json. Its different source hash has been reconciled exactly: the only intervening source line adds the preflight memory_representation description. The historical bytes were reconstructed and matched their recorded hash; no sampler arithmetic changed. The [source provenance record](CURVATURE_SOURCE_PROVENANCE.json) links the archived measured source and preserves both hashes. This report does not treat the exploratory cross-stage comparison as an isolated population experiment. All source/data hashes within the new 18-run stage are unchanged."]
    text += ["", "No fixed-cap logarithmic-regret theorem, finite-bit certificate, or general performance superiority follows. The measured legacy-step wrapper remains explicitly practical-only. The subsequently selected [ComputationFirstLogOdds](experiments/computation_first_log_odds.py) retains this original step and separately corrects the full prescribed schedule and preflight; it is not a new measurement campaign. Full protocol: [COMPUTATION_FOLLOWUP_PLAN.md](COMPUTATION_FOLLOWUP_PLAN.md). Frozen configurations, source/data hashes, supervisor logs, results and raw/c4 trajectories: experiments/results/computation_followup/."]
    (ROOT/"COMPUTATION_FOLLOWUP_RESULT.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    print(json.dumps(dict(groups=groups, wall_seconds=campaign["wall_seconds"]), indent=2))


if __name__ == "__main__":
    analyze()
