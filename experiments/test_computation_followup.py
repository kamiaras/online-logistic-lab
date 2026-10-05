"""Checks for the bounded exploratory stage, without changing any old core."""
import json
import math
from collections import Counter

import numpy as np
import pytest

from experiments.computation_followup import LegacyStepCurvatureLogOdds
from experiments.curvature_persistent_log_odds import CurvaturePersistentLogOdds
from experiments.computation_followup_campaign import manifest
from experiments.computation_followup_worker import run
from experiments.proximal import ProximalLogOdds


def test_legacy_control_changes_only_step_and_records_recomputed_warm_count():
    config = {"lanes": 16, "transitions": 1}
    current = CurvaturePersistentLogOdds(3, 2., 1000, 41, config)
    legacy = LegacyStepCurvatureLogOdds(3, 2., 1000, 41, config)
    for model in [current, legacy]:
        model._append_feature(np.array([.8, .2, -.1]))
        model._initialize_lanes()
    np.testing.assert_array_equal(current.states, legacy.states)
    a, b = current._schedule(), legacy._schedule()
    assert b["inveta"] == legacy.lam+2*legacy.d*legacy.E
    assert b["mu"] == legacy.lam+b["inveta"]
    assert b["inveta"] > a["inveta"]
    assert b["H_warm"] == a["H_warm"] and b["gamma"] == a["gamma"]
    assert b["N_prescribed"] == math.ceil(b["H_warm"]/(2*math.log1p(b["gamma"]/b["inveta"])))
    assert b["N_actual"] == 1
    with pytest.raises(ValueError, match="practical-only"):
        LegacyStepCurvatureLogOdds.preflight(1, 1., 5, {"mode": "prescribed"})


def test_manifest_is_exactly_six_fixed_cases_and_three_methods():
    specs = manifest()
    assert len(specs) == 18
    assert Counter(s["method"] for s in specs) == dict(current16=6, legacy16=6, original256=6)
    assert {s["seed"] for s in specs} == {910001, 910002, 910003}
    assert {s["d"] for s in specs} == {10, 100}
    assert all(s["rounds"] == 500 and s["seconds"] == 90. and s["T"] == 1000 for s in specs)
    assert all(s["config"]["transitions"] == 1 for s in specs)
    for spec in specs:
        if spec["method"] == "original256":
            pre = ProximalLogOdds.preflight(spec["d"], spec["B"], spec["T"], spec["config"])
            assert pre["actual_particles"] == 256 and pre["actual_max_transitions"] == 1


@pytest.mark.parametrize("method", ["current16", "legacy16", "original256"])
def test_recorder_saves_raw_c4_counts_and_original_class_without_core_changes(tmp_path, method):
    source, out = tmp_path/"data", tmp_path/"out"
    source.mkdir()
    out.mkdir()
    np.ones((5, 1), dtype="<f8").tofile(source/"X.bin")
    np.array([1, 1, -1, 1, 1], dtype="i1").tofile(source/"y.bin")
    config = dict(mode="practical", transitions=1, particle_batch=64)
    config.update(particles=256) if method == "original256" else config.update(lanes=16, history_block_size=128)
    run(dict(method=method, d=1, B=2., T=1000, seed=34, rounds=5, seconds=10.,
             data_path=str(source), output=str(out), config=config))
    result = json.loads((out/"result.json").read_text())
    assert result["status"] == "completed" and result["round"] == 5
    expected_population = (ProximalLogOdds.preflight(1, 2., 1000, config)["actual_particles"]
                           if method == "original256" else 16)
    assert result["actual_population"] == expected_population
    data = np.load(out/"trajectory.npz")
    caps = np.log(4*np.arange(1, 6)*np.arange(2, 7))
    np.testing.assert_allclose(data["guarded_logits"], np.clip(data["logits"], -caps, caps))
    assert data["proximal_steps"][-1] > 0
    if method == "original256":
        assert result["algorithm_class"] == "ProximalLogOdds"
        assert result["original_gradient_feature_lane_rows_derived"] > 0
    else:
        assert data["summary_queries"][-1] > 0
