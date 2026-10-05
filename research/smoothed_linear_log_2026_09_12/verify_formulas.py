"""Deterministic proof-identity and implementation checks, not a benchmark."""

import json
import math
import random
from pathlib import Path

from smoothed_clipped_ogd import SmoothedClippedOGD, sigmoid


def softplus(z):
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def score_loss(Y, z):
    return softplus(z if Y == 0 else -z)


def main():
    checks = 0
    max_identity_error = 0.0
    worst_local_slack = -math.inf
    rng = random.Random(912202609)
    for b in (0.05, 1.0, 10.0, 1000.0):
        for epsilon in (0.5, 0.1, 0.001):
            a = epsilon / 2.0
            q = a * (1 - a)
            k = q / (4 * (1 + 2 * b))
            M = math.log((1 - a) / a)
            for _ in range(60):
                z, r = rng.uniform(-b, b), rng.uniform(-b, b)
                s = min(M, max(-M, z))
                phat = sigmoid(s)
                actual_p = rng.uniform(epsilon, 1 - epsilon)
                d0 = z - s
                D = softplus(r) - softplus(s) - phat * (r - s)
                lower = 2 * k * (r - s) ** 2
                assert D >= lower - 2e-10 * (1 + b)
                checks += 1
                for Y in (0, 1):
                    regret = score_loss(Y, s) - score_loss(Y, r)
                    linear = (phat - Y) * (z - r)
                    identity = linear - D - (phat - Y) * d0
                    error = abs(regret - identity)
                    max_identity_error = max(max_identity_error, error)
                    assert error < 2e-10 * (1 + b)
                    noise = (Y - actual_p) * d0
                    slack = regret - linear + k * (z - r) ** 2 - noise
                    worst_local_slack = max(worst_local_slack, slack)
                    assert slack <= 2e-10 * (1 + b)
                    checks += 2
                # The same noise value is used for every comparator and has
                # exactly zero label expectation at each supplied feature.
                expected_noise = actual_p * (1 - actual_p) * d0 + (1 - actual_p) * (-actual_p) * d0
                assert abs(expected_noise) < 1e-10
                checks += 1

    max_mgf = 0.0
    for _ in range(160):
        # Exact scalar uniform-feature integral via erf, independent of the
        # proof's chord inequality. This is one-step expectation verification.
        b = rng.uniform(0.1, 10.0)
        delta = rng.uniform(0.05, 1.0)
        center = (1 - delta) * rng.uniform(-1, 1)
        w, u = rng.uniform(-b, b), rng.uniform(-b, b)
        diff = w - u
        coefficient = diff * diff / (4 * b * b)
        if coefficient < 1e-14:
            continue
        root = math.sqrt(coefficient)
        uniform_integral = (math.sqrt(math.pi) / (4 * delta * root)
                            * (math.erf(root * (center + delta))
                               - math.erf(root * (center - delta))))
        mgf = math.exp(coefficient * (delta * delta / 3.0) / 2.0) * uniform_integral
        max_mgf = max(max_mgf, mgf)
        assert mgf <= 1 + 3e-13
        checks += 1

    for Y in (0, 1):
        model = SmoothedClippedOGD(3, B=10, R=2, delta=0.2, epsilon=0.1)
        model.w[:] = [10.0, 0.0, 0.0]
        model.t = 1000000
        x = [2.0, 0.0, 0.0]
        p = model.predict(x)
        assert p == 0.95
        old = model.w[:]
        scale = (p - Y) / (model.k * model.nu * (model.t + 1))
        candidate = [old[j] - scale * x[j] / model.R for j in range(3)]
        norm = math.hypot(*candidate)
        expected = [v * min(1.0, model.b / norm) for v in candidate]
        model.observe(x, Y)
        assert max(abs(v - e) for v, e in zip(model.w, expected)) < 1e-13
        assert model.w != old  # Saturation must not erase the update.
        checks += 3

    # Rescaling physical R and B while preserving BR changes no normalized
    # learner computation or bound, and covers non-unit feature normalization.
    for _ in range(40):
        m1 = SmoothedClippedOGD(4, B=3, R=2, delta=0.4, epsilon=0.2)
        m2 = SmoothedClippedOGD(4, B=6, R=1, delta=0.4, epsilon=0.2)
        state = [rng.uniform(-1, 1) for _ in range(4)]
        h = [rng.uniform(-0.4, 0.4) for _ in range(4)]
        m1.w[:] = state
        m2.w[:] = state
        m1.t = m2.t = 1000
        x1, x2 = [2 * v for v in h], h
        assert m1.predict(x1) == m2.predict(x2)
        Y = rng.choice((0, 1))
        m1.observe(x1, Y)
        m2.observe(x2, Y)
        assert m1.w == m2.w
        assert m1.expected_regret_bound(100000) == m2.expected_regret_bound(100000)
        checks += 3

    for B, R, x in ((0, 1, [0.2, 0.3]), (1, 0, [0.0, 0.0])):
        model = SmoothedClippedOGD(2, B, R, 0.2, 0.1)
        assert model.predict(x) == 0.5
        model.observe(x, 1)
        assert model.w == [0.0, 0.0]
        assert model.expected_regret_bound(1000) == 0.0
        checks += 3

    target = SmoothedClippedOGD(1000, 1000, 1, 0.001, 0.001)
    report = {
        "status": "PASS",
        "scope": "Scalar proof identities, exact one-step uniform-feature integrals, and reference update checks; no learning benchmark.",
        "checked_assertions": checks,
        "max_local_loss_identity_error": max_identity_error,
        "max_local_regret_inequality_slack": worst_local_slack,
        "max_one_step_feature_mgf": max_mgf,
        "prior_numerical_setting_formula_only": {
            "d": 1000, "B": 1000, "R": 1, "delta": 0.001, "epsilon": 0.001,
            "T": 100000, "q": target.q, "k": target.k, "nu_normalized": target.nu,
            "bound_using_one_plus_log_T": target.expected_regret_bound(100000),
            "note": "The polynomial coefficient is large at tiny noise/floor. This is an analytic bound, not a measured loss or runtime."
        }
    }
    output = Path(__file__).resolve().parents[2] / "SMOOTHED_LINEAR_LOG_ROOT_FORMULA_CHECK.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
