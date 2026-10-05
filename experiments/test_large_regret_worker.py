"""Tiny recorder/direct-call equivalence checks; no production measurements."""
import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from experiments.large_regret_worker import COLUMNS, algorithm_class, run


CONFIGS = {
    "selected": dict(mode="practical", lanes=16, transitions=1,
                     particle_batch=64, history_block_size=128),
    "ours": dict(mode="practical", particles=256, transitions=1,
                 particle_batch=64, history_block_size=128),
    "sigmoid": dict(particles=16, transitions=16, bridges=1,
                    step_multiplier=.5, lazy=True, history_block_size=128),
    "aioli": dict(matrix_backend="cholesky"),
}


@pytest.mark.parametrize("method", list(CONFIGS))
def test_streamed_outputs_equal_one_direct_prediction_per_round(tmp_path, method):
    X = np.array([[.8, .2, -.1], [.1, -.3, .6], [-.7, .1, .3],
                  [.2, .2, -.3], [.6, .4, .1], [-.1, .5, -.2],
                  [0., 0., 0.], [.2, -.5, .4]], dtype="<f8")
    y = np.array([1, 1, -1, 1, -1, 1, -1, -1], dtype="i1")
    source = tmp_path/"data"
    source.mkdir()
    X.tofile(source/"X.bin")
    y.tofile(source/"y.bin")
    config = CONFIGS[method]
    expected = []
    with threadpool_limits(limits=1):
        direct = algorithm_class(method)(3, 2., 8, 99101, config)
        for x, label in zip(X, y):
            actual = float(direct.predict(x))
            raw = float(direct._last_schedule["raw_logit"]) if method == "selected" else actual
            direct.observe(x, int(label))
            expected.append((raw, actual, float(np.logaddexp(0., -label*actual)),
                             direct.state_bytes()))
    output = tmp_path/method
    spec = dict(algorithm=method, d=3, B=2., T=8, seed=99101, seconds=120.,
                data_path=str(source), output=str(output), config=config)
    result = run(spec)
    assert result["status"] == "completed", result.get("error")
    assert result["round"] == result["committed_round"] == result["recorded_rounds"] == 8
    assert result["configured_horizon"] == result["target_rounds"] == 8
    records = np.fromfile(output/"trajectory.bin", dtype="<f8").reshape(8, 5)
    np.testing.assert_array_equal(records[:, [0, 1, 2, 4]], expected)
    assert np.all(np.isfinite(records))
    assert np.all(records[:, 3] > 0) and np.all(np.diff(records[:, 3]) >= 0)
    assert result["completed_prefix_online_seconds"] == records[-1, 3]
    assert result["state_bytes"] == records[-1, 4]
    # Snapshot diagnostics may materialize previously absent zero counters.
    assert {k: v for k, v in result["metrics"]["counters"].items() if v} == {
        k: v for k, v in direct.metrics.counters.items() if v}
    assert result["algorithm_class"] == type(direct).__name__
    assert result["algorithm"] == method
    assert result["actual_population"] == getattr(direct, "m", getattr(direct, "n_particles", None))
    assert result["trajectory"]["columns"] == list(COLUMNS)
    assert result["trajectory"]["committed_bytes"] == result["trajectory"]["file_bytes"] == 8*40
    assert all(pool["num_threads"] == 1 for pool in result["threadpools"])
    assert json.loads((output/"progress.json").read_text()) == result
    assert json.loads((output/"result.json").read_text()) == result
    if method != "selected":
        np.testing.assert_array_equal(records[:, 0], records[:, 1])
        assert "no recorder clipping" in result["primary_prediction"]
    # A restart must preserve a previous completed/partial prefix.
    saved = (output/"trajectory.bin").read_bytes()
    with pytest.raises(FileExistsError):
        run(spec)
    assert (output/"trajectory.bin").read_bytes() == saved


def test_reject_unknown_method_before_creating_output(tmp_path):
    output = tmp_path/"unknown"
    with pytest.raises(ValueError, match="Unknown algorithm"):
        run(dict(algorithm="unknown", output=str(output)))
    assert not output.exists()
