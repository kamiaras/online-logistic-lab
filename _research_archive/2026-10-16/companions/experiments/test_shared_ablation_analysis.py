"""Small analysis checks; these never instantiate or benchmark a sampler."""
import numpy as np

from .shared_ablation_analyze import guards, covariance_diagnostic, summary, aggregate_cells


def test_all_declared_guards_at_extreme_logits_and_both_labels():
    z = np.array([1e8, -1e8, 0., 1., -1.] * 100)
    t = np.arange(1, len(z) + 1)
    for factor in (1, 3, 4):
        for sign in (-1, 1):
            y = np.full(len(z), sign)
            clipped, losses = guards(z, y, factor)
            raw = np.logaddexp(0., -y * z)
            assert np.all(losses - raw <= 1 / (factor * t * (t + 1)) + 1e-12)
            assert np.all(losses <= np.log1p(factor * t * (t + 1)) + 1e-12)
            assert (losses - raw).sum() <= len(z) / (factor * (len(z) + 1)) + 1e-12
            assert np.all(np.isfinite(clipped))


def test_covariance_centers_each_tilt_across_repetitions():
    # Distinct nonstationary column means must not be treated as correlation.
    orthogonal = np.array([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
    scores = orthogonal + [0., 10000.]
    d = covariance_diagnostic(scores, scores)
    assert abs(d["mean_offdiagonal_correlation"]) < 1e-12
    assert abs(d["equivalent_independent_score_count"] - 2) < 1e-12
    same = np.stack([orthogonal[:, 0], orthogonal[:, 0] + 10000.], axis=1)
    d = covariance_diagnostic(same, same)
    assert abs(d["equivalent_independent_score_count"] - 1) < 1e-12
    assert abs(d["mean_offdiagonal_correlation"] - 1) < 1e-12


def test_missing_repetitions_are_retained_not_replaced_by_available_mean():
    result = summary([1., 2., 3.], 6)
    assert result["available_mean"] == 2
    assert result["complete_mean"] is None
    assert result["conditional_sampler_t95"] is None
    assert result["n"] == 3 and result["expected_n"] == 6
    rows, contrasts = aggregate_cells({}, ["missing_dataset"])
    assert len(rows) == 12 and len(contrasts) == 21
    assert all(len(row["seeds"]) == 6 for row in rows + contrasts)
    assert all(row["metrics"]["loss"]["n"] == 0 for row in rows + contrasts)
    assert all(row["metrics"]["loss"]["complete_mean"] is None for row in rows + contrasts)
