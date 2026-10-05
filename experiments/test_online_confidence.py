"""Small standard-library checks; no sampler, BLAS or data fitting."""
import math
import random
import unittest

from .online_confidence import OnlineConfidence, PRIORS, SCALES, logsumexp, softplus


class ConfidenceChecks(unittest.TestCase):
    def test_pathwise_identity_and_each_expert_bound(self):
        rng = random.Random(173)
        model = OnlineConfidence()
        totals = [0.] * len(SCALES)
        total = 0.
        for _ in range(700):
            z, y = rng.uniform(-100, 100), rng.choice((-1, 1))
            result = model.predict(z)
            loss = model.observe(y)
            self.assertAlmostEqual(loss, softplus(-y*result), places=11)
            total += loss
            totals = [v+softplus(-y*a*z) for v, a in zip(totals, SCALES)]
            exact = -logsumexp(math.log(w)-v for w, v in zip(PRIORS, totals))
            self.assertAlmostEqual(total, exact, places=9)
            for v, w in zip(totals, PRIORS):
                self.assertLessEqual(total, v-math.log(w)+1e-8)

    def test_label_order_idempotence_and_validation(self):
        model = OnlineConfidence()
        with self.assertRaises(ValueError):
            model.observe(1)
        before = model.log_weights.copy()
        z = model.predict(10.)
        self.assertEqual(z, model.predict(10.))
        self.assertEqual(before, model.log_weights)
        with self.assertRaises(ValueError):
            model.predict(9.)
        with self.assertRaises(ValueError):
            model.observe(0)
        model.observe(-1)
        self.assertNotEqual(before, model.log_weights)
        with self.assertRaises(ValueError):
            model.predict(float("inf"))

    def test_extreme_scores(self):
        model = OnlineConfidence()
        for z, y in [(1e6, 1)]*80 + [(1e6, -1)]*80 + [(-1e6, 1)]*80:
            score = model.predict(z)
            self.assertTrue(math.isfinite(score))
            self.assertTrue(math.isfinite(model.observe(y)))
        self.assertLessEqual(model.cumulative_loss, 240*math.log(2)+math.log(4)+1e-8)

    def test_chunk_continuation_and_symmetry(self):
        import copy
        left, reflected = OnlineConfidence(), OnlineConfidence()
        for t in range(80):
            z, y = math.sin(t)*20, (1 if t % 3 else -1)
            self.assertAlmostEqual(left.predict(z), -reflected.predict(-z), places=12)
            left.observe(y)
            reflected.observe(-y)
        resumed = copy.deepcopy(left)
        for t in range(80):
            self.assertEqual(left.predict(t), resumed.predict(t))
            self.assertEqual(left.observe(-1), resumed.observe(-1))

    def test_wrapper_keeps_base_calls_and_validation(self):
        import sys
        import types
        from unittest.mock import patch
        from .online_confidence import ConfidenceAdaptiveLogOdds

        class FakeBase:
            def __init__(self, *args):
                self.pending = None
                self.events = []

            def predict(self, x):
                if self.pending is not None and x != self.pending:
                    raise ValueError("Feature mismatch")
                if self.pending is None:
                    self.pending = x
                    self.events.append(("predict", x))
                return x

            def observe(self, x, y):
                if self.pending is None or x != self.pending or y not in (-1, 1):
                    raise ValueError("Invalid observation")
                self.pending = None
                self.events.append(("observe", x, y))

        name = "experiments.computation_first_log_odds"
        fake_module = types.ModuleType(name)
        fake_module.ComputationFirstLogOdds = FakeBase
        with patch.dict(sys.modules, {name: fake_module}):
            wrapped = ConfidenceAdaptiveLogOdds(1, 1., 10, 1)
        first = wrapped.predict(2.)
        self.assertEqual(first, wrapped.predict(2.))
        before = wrapped.confidence.log_weights.copy()
        with self.assertRaises(ValueError):
            wrapped.observe(3., 1)
        self.assertEqual(before, wrapped.confidence.log_weights)
        wrapped.observe(2., 1)
        with self.assertRaises(ValueError):
            wrapped.observe(2., 1)
        self.assertEqual(wrapped.confidence.rounds, 1)
        self.assertEqual(wrapped.base.events, [("predict", 2.), ("observe", 2., 1)])


if __name__ == "__main__":
    unittest.main()
