"""Small direct released-class parity checks; no production measurements."""
import math
import numpy as np
import pytest

from experiments.author_baselines import AuthorAIOLI, AuthorSigmoid, author_module


X = np.array([[.8, -.2], [.1, .9], [-.5, .3], [.4, .1]])
Y = np.array([1, -1, 1, 1])


def test_author_aioli_matches_unmodified_native_class():
    adapter = AuthorAIOLI(2, 10., 4, 778)
    native = author_module().AIOLI(d=2, B=10., seed=778, **adapter.config)
    # Preserve the released initialization literally, including its scale.
    np.testing.assert_array_equal(adapter.model.L, 10.*np.eye(2))
    for x, y in zip(X, Y):
        p = native.predict_prob(x)
        logit = adapter.predict(x)
        assert adapter.last_probability == p
        assert logit == pytest.approx(float(native.theta@x), abs=1e-14)
        assert adapter.author_loss(int(y)) == author_module().logloss_from_prob(p, int(y))
        adapter.observe(x, int(y))
        native.update(x, int(y))
        for name in ("L", "b", "theta"):
            np.testing.assert_array_equal(getattr(adapter.model, name), getattr(native, name))
    assert adapter.n == 4
    assert adapter.state_bytes() == 8*(2*2+2*2)


def test_author_sigmoid_matches_native_predictions_updates_adaptation_and_rng():
    cfg = dict(n_particles=4, mala_steps_per_round=3, step_size=.01,
               grad_clip=50., adapt_step=True)
    adapter = AuthorSigmoid(2, 2., 4, 779, cfg)
    native = author_module().GaussianEW_MALA(d=2, B=2., rng=np.random.default_rng(779), **adapter.config)
    for t, (x, y) in enumerate(zip(X, Y), 1):
        p = native.predict_prob(x)
        score = adapter.predict(x)
        assert math.isfinite(score)
        assert adapter.last_probability == p
        assert math.exp(adapter.last_log_probability_positive) == pytest.approx(p, abs=2e-15)
        expected_reported = author_module().logloss_from_prob(p, int(y))
        adapter.observe(x, int(y))
        acceptance, h = native.update(X[:t], Y[:t])
        assert adapter.last_reported_loss == expected_reported
        assert adapter.last_acceptance_rate == acceptance
        assert adapter.model.h == h
        assert adapter.model.round == native.round
        assert adapter.rng.bit_generator.state == native.rng.bit_generator.state
        np.testing.assert_array_equal(adapter.model.thetas, native.thetas)
        np.testing.assert_array_equal(adapter.X_history[:t], X[:t])
        np.testing.assert_array_equal(adapter.y_history[:t], Y[:t])
    assert adapter.state_bytes() == 8*(4*2+4+4*2)
    assert adapter.metrics.counters["completed_native_particle_proposals"] == 4*3*4
    assert adapter.metrics.counters["completed_native_history_particle_evaluations"] == 2*4*3*(1+2+3+4)


def test_stable_unsmoothed_loss_and_release_reported_clipping_are_separate():
    adapter = AuthorSigmoid(1, 1., 2, 88, dict(n_particles=2, mala_steps_per_round=1))
    adapter.model.thetas[:] = 1000.
    adapter.predict(np.array([1.]))
    assert adapter.last_probability == 1.
    assert adapter.loss(-1) == pytest.approx(1000.)
    assert adapter.author_loss(-1) == author_module().logloss_from_prob(1., -1)
    assert adapter.author_loss(-1) < 28.
    assert adapter.last_stable_logit == pytest.approx(1000.)


def test_full_private_history_is_allocated_once_and_prefix_arguments_are_views(monkeypatch):
    adapter = AuthorSigmoid(2, 1., 4, 90, dict(n_particles=2, mala_steps_per_round=1))
    storage_id = id(adapter.X_history)
    observed = []
    def update(x_prefix, y_prefix):
        assert np.shares_memory(x_prefix, adapter.X_history)
        assert np.shares_memory(y_prefix, adapter.y_history)
        observed.append(len(x_prefix))
        return 1., adapter.model.h
    monkeypatch.setattr(adapter.model, "update", update)
    for x, y in zip(X, Y):
        adapter.predict(x)
        adapter.observe(x, int(y))
        assert id(adapter.X_history) == storage_id
    assert observed == [1, 2, 3, 4]


def test_released_configuration_defaults_are_explicit_and_timers_reset_on_error():
    adapter = AuthorSigmoid(2, 1000., 2, 91)
    assert adapter.model.S == 12 and adapter.model.K == 300
    assert adapter.model.h == .01 and adapter.model.grad_clip == 50.
    assert adapter.model.adapt_step and adapter.model.mala_window is None
    assert adapter.model.rm_gamma0 == .5 and adapter.model.rm_gamma_decay == .25
    assert adapter.model.h_min == 1e-4 and adapter.model.h_max == 1.
    adapter.predict(X[0])
    with pytest.raises(RuntimeError):
        adapter.predict(X[0])
    assert adapter.last_native_predict_seconds == 0.
    adapter.last_native_update_seconds = 100.
    with pytest.raises(ValueError):
        adapter.observe(X[1], 1)
    assert adapter.last_native_update_seconds == 0.
