import numpy as np
from .shared_ablation import SharedAblation, guarded_logits
from .shared_tangent import SharedTangentLogOdds


def test_poisson_fresh_sweep_exact_archived_replay():
    cfg = dict(tilts=4, burn_in=2, transitions=2, history_block_size=16)
    old = SharedTangentLogOdds(3, 2., 20, 825, cfg)
    new = SharedAblation(3, 2., 20, 825, dict(cfg, acceptance="poisson"))
    rng = np.random.default_rng(29)
    X = rng.normal(size=(20, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = rng.choice([-1, 1], size=20)
    for x, yy in zip(X, y):
        assert old.predict(x) == new.predict(x)
        old.observe(x, int(yy))
        new.observe(x, int(yy))
        np.testing.assert_array_equal(old.v, new.v)
        np.testing.assert_array_equal(old.g, new.g)


def test_equal_budget_factorial_and_persistent_states():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(10, 2))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    counts = []
    for layout in ["sweep", "lanes"]:
        for persistence in ["fresh", "warm"]:
            a = SharedAblation(2, 2., 10, 126,
                dict(layout=layout, persistence=persistence, tilts=4,
                     burn_in=2, transitions=2))
            for x in X:
                a.predict(x)
                a.observe(x, 1)
            counts.append(a.metrics.counters["proximal_steps"])
            if persistence == "warm":
                assert a.states.shape == (4 if layout == "lanes" else 1, 2)
                assert a.metrics.counters["prior_initializations"] == (4 if layout == "lanes" else 1)
    assert len(set(counts)) == 1


def test_guard_pointwise_loss_and_total_overhead():
    raw = np.array([0., 1e5, -1e5, 3., -3.]*20)
    safe = guarded_logits(raw)
    t = np.arange(1, len(raw)+1)
    for y in [-1, 1]:
        excess = np.logaddexp(0, -y*safe)-np.logaddexp(0, -y*raw)
        assert np.all(excess <= 1/(t*(t+1))+1e-12)
        assert excess.sum() < 1
        assert np.all(np.logaddexp(0, -y*safe) <= np.log1p(t*(t+1))+1e-12)
