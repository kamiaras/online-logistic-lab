"""Independent 2D integration checks, not mixing or floating-point certificates.

Both histories contain oblique features and contradictory labels.  References
evaluate the likelihood directly, without the implementation's potential helper.
The MALA check keeps one final state from each independent trajectory; it does
not count successive states of one chain as independent Monte Carlo samples.
"""

import math

import numpy as np
import pytest
from scipy.special import expit, logsumexp

from experiments.proximal import ProximalLogOdds
from experiments.sigmoid import SigmoidMALA


X = np.array([[.8, .6], [.8, .6], [-.6, .8], [.3, -.9], [-.7, -.2]])
LABELS = np.array([1., -1., 1., -1., 1.])
N_SAMPLES = 3072


def _tensor_rule(nodes, weights):
    assert np.all(weights > 0.)
    first, second = np.meshgrid(nodes, nodes, indexing="ij")
    points = np.column_stack((first.ravel(), second.ravel()))
    log_weights = (np.log(weights)[:, None] + np.log(weights)[None, :]).ravel()
    return points, log_weights


def _features(points, query):
    """Means, full second moment, mean score, and predictive probability."""
    first, second = points.T
    score = points @ query
    return np.column_stack((first, second, first * first, first * second,
                            second * second, score, expit(score)))


def _reference_statistics(points, log_mass, query):
    log_normalizer = float(logsumexp(log_mass))
    mass = np.exp(log_mass - log_normalizer)
    features = _features(points, query)
    means = mass @ features
    # Center directly to avoid cancellation in the reference MC variances.
    variances = mass @ ((features - means) ** 2)
    return means, variances, log_normalizer


def _gaussian_reference(degree, B, query):
    nodes, weights = np.polynomial.hermite.hermgauss(degree)
    points, log_weights = _tensor_rule(math.sqrt(2.) * B * nodes,
                                      weights / math.sqrt(math.pi))
    # The Gaussian prior is already in the Hermite measure: no second prior
    # density factor belongs here.  This integrates over all of R^2.
    loss = np.logaddexp(0., -(points @ X.T) * LABELS).sum(axis=1)
    return _reference_statistics(points, log_weights - loss, query)


def _box_conditional_reference(degree, B, query, p, auxiliary, inveta):
    nodes, weights = np.polynomial.legendre.leggauss(degree)
    bound = 2. * B
    points, log_weights = _tensor_rule(bound * nodes, bound * weights)
    score = points @ query
    value = .5 * np.sum((points / B) ** 2, axis=1)
    value += np.logaddexp(0., -(points @ X.T) * LABELS).sum(axis=1)
    value += p * np.logaddexp(0., -score) + (1. - p) * np.logaddexp(0., score)
    value += .5 * inveta * np.sum((points - auxiliary) ** 2, axis=1)
    return _reference_statistics(points, log_weights - value, query)


def _check_degree_convergence(coarse, fine):
    # This is a numerical convergence diagnostic, not a rigorous error bound.
    np.testing.assert_allclose(coarse[0], fine[0], rtol=5e-8, atol=5e-8)
    np.testing.assert_allclose(coarse[1], fine[1], rtol=5e-8, atol=5e-8)
    assert coarse[2] == pytest.approx(fine[2], rel=0., abs=5e-8)


def _check_sample_statistics(samples, query, reference):
    means, variances, _ = reference
    empirical = _features(samples, query).mean(axis=0)
    # Conservative seeded smoke-test thresholds.  MALA burn-in is empirical;
    # these tolerances do not purport to certify its finite-time mixing bias.
    tolerance = 7. * np.sqrt(variances / len(samples)) + 2e-6
    assert np.all(np.abs(empirical - means) <= tolerance), (
        f"empirical={empirical}, reference={means}, tolerance={tolerance}"
    )
    return empirical, tolerance


def test_sigmoid_two_dimensional_posterior_against_gauss_hermite():
    B = .9
    query = np.array([.55, -.7])
    coarse = _gaussian_reference(50, B, query)
    reference = _gaussian_reference(80, B, query)
    _check_degree_convergence(coarse, reference)

    model = SigmoidMALA(2, B, len(X) + 1, 62051,
                        {"particles": N_SAMPLES, "transitions": 1,
                         "bridges": 1, "lazy": False, "history_block_size": 8})
    for feature, label in zip(X, LABELS):
        model.predict(feature)
        model.observe(feature, label)

    # Each row evolves with its own independent noise and decisions.  One
    # retained state per trajectory avoids an unmeasured autocorrelation ESS.
    for _ in range(128):
        model._mala_step(.25)
    logit = model.predict(query)
    empirical, tolerance = _check_sample_statistics(model.theta, query, reference)

    actual_probability = float(expit(logit))
    empirical_guarded = (1. + N_SAMPLES * empirical[-1]) / (N_SAMPLES + 2.)
    assert actual_probability == pytest.approx(empirical_guarded, abs=2e-13)
    # Compare the actual public prediction with the integrated target, keeping
    # its two pseudo-observation guard exactly rather than ignoring O(1/m).
    reference_guarded = (1. + N_SAMPLES * reference[0][-1]) / (N_SAMPLES + 2.)
    assert abs(actual_probability - reference_guarded) <= (
        N_SAMPLES / (N_SAMPLES + 2.) * tolerance[-1]
    )
    counts = model.metrics.counters
    assert 0 < counts["mala_accepted"] <= counts["mala_proposals"]


def test_proximal_two_dimensional_conditional_against_box_quadrature():
    B = 1.1
    query = np.array([.45, .7])
    p = .73
    auxiliary = np.array([.8, -.55])
    # Compute the schedule independently of the algorithm's schedule helper.
    expected_inveta = B ** -2 + 2. * 2. * (np.sum(X * X) + query @ query) / 4.
    coarse = _box_conditional_reference(50, B, query, p, auxiliary, expected_inveta)
    reference = _box_conditional_reference(80, B, query, p, auxiliary, expected_inveta)
    _check_degree_convergence(coarse, reference)

    model = ProximalLogOdds(2, B, len(X) + 1, 91703)
    for feature, label in zip(X, LABELS):
        model.history.append(feature, label)
        model.sum_norm2 += float(feature @ feature)
    schedule = model._schedule(query)
    assert schedule["inveta"] == pytest.approx(expected_inveta, rel=2e-15)
    samples = model._conditional_draw(np.tile(auxiliary, (N_SAMPLES, 1)),
                                       query, p, schedule)
    assert samples.shape == (N_SAMPLES, 2)
    assert np.all(np.isfinite(samples))
    assert np.all(np.abs(samples) <= 2. * B)
    empirical, tolerance = _check_sample_statistics(samples, query, reference)

    # At this fixed tilt and fixed auxiliary Y the relevant logit contribution
    # is E[query . theta | Y,p].  It is not the full tilt-averaged online logit.
    conditional_mean_logit = float(np.mean(samples @ query))
    assert conditional_mean_logit == pytest.approx(empirical[-2], abs=2e-13)
    assert abs(conditional_mean_logit - reference[0][-2]) <= tolerance[-2]
    counts = model.metrics.counters
    assert counts["rqo_proposals"] - counts["rqo_rejections"] == N_SAMPLES
    assert counts["rqo_proposals"] / N_SAMPLES < math.e + .1
    assert counts["numerical_failures"] == 0
    assert model.diagnostics()["max_center_residual_ratio"] <= 1. + 1e-8
