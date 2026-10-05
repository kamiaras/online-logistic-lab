"""Independent local algebra checks, not a learning benchmark."""

import json
import math
import random
from decimal import Decimal, localcontext
from pathlib import Path

from diagonal_aioli import HalfLookahead, RecenteredDiagonalAIOLI, sigmoid


def reference_score(v, q, weight):
    # Independent high-precision bisection of the score equation; the prototype
    # instead bisects a bounded auxiliary rho in double precision.
    with localcontext() as ctx:
        ctx.prec = 65
        dv, dq, dw = map(lambda x: Decimal(str(x)), (v, q, weight))
        lo, hi = dv - dw * dq, dv + dw * dq
        for _ in range(220):
            mid = (lo + hi) / 2
            expz = mid.exp()
            value = mid + dw * dq * (expz - 1) / (expz + 1) - dv
            if value > 0:
                hi = mid
            else:
                lo = mid
        return float((lo + hi) / 2)


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def quadratic(a, diagonal):
    return math.fsum(x * x * d for x, d in zip(a, diagonal))


def main():
    rng = random.Random(9122026)
    count = 0
    max_score_error = max_half_identity_error = max_diag_identity_error = 0.0
    for _ in range(80):
        d = rng.randrange(1, 8)
        x = [rng.uniform(-1, 1) for _ in range(d)]
        scale = max(1, math.sqrt(dot(x, x)))
        x = [v / scale for v in x]
        initial = [rng.uniform(-2, 2) for _ in range(d)]
        diagonal = [rng.uniform(0.8, 3) for _ in range(d)]
        u = [rng.uniform(-0.2, 0.2) for _ in range(d)]
        y = rng.choice([-1, 1])

        half = HalfLookahead(d, 1, 1, 1000, lam=1.25)
        half.w[:] = initial
        z = half.predict_score(x)
        q, v = dot(x, x) / half.lam, dot(initial, x)
        exact = reference_score(v, q, 0.25)
        max_score_error = max(max_score_error, abs(z - exact))
        assert abs(z - exact) <= q * 2 ** (-half.bisections) / 4 + 1e-13
        rho = half.pending[1]
        p = sigmoid(-y * z)
        g = [-y * p * xj for xj in x]
        theta = [wj - rho * xj / (4 * half.lam) for wj, xj in zip(initial, x)]
        half.observe(x, y)
        old_square = sum(half.lam * (wj - uj) ** 2 for wj, uj in zip(initial, u))
        new_square = sum(half.lam * (wj - uj) ** 2 for wj, uj in zip(half.w, u))
        phi = z + q * math.tanh(z / 2) / 4 - v
        lhs = dot(g, [tj - uj for tj, uj in zip(theta, u)])
        rhs = old_square - new_square + q * p * (1 - p) / 4 - y * p * phi
        err = abs(lhs - rhs)
        max_half_identity_error = max(max_half_identity_error, err)
        assert err < 5e-14
        actual_regret = math.log1p(math.exp(-y * z)) - math.log1p(math.exp(-y * dot(x, u)))
        assert actual_regret <= rhs + 5e-14
        count += 3

        model = RecenteredDiagonalAIOLI(d, 1, 1, 1000, lam=0.8)
        model.w[:] = initial
        model.precision[:] = diagonal
        model.bisections = 52  # Algebraic identity, independent of loose bounds.
        z = model.predict_score(x)
        rho = model.pending[1]
        q = math.fsum(xj * xj / a for xj, a in zip(x, diagonal))
        v = dot(initial, x)
        exact = reference_score(v, q, 0.5)
        max_score_error = max(max_score_error, abs(z - exact))
        assert abs(z - exact) < 3e-14
        theta = [wj - rho * xj / (2 * a) for wj, xj, a in zip(initial, x, diagonal)]
        p = sigmoid(-y * z)
        alpha = p * (1 - p) / 4
        g = [-y * p * xj for xj in x]
        opposite = [y * (1 - p) * xj for xj in x]
        c = [a * wj for a, wj in zip(diagonal, initial)]
        H = [alpha * xj * xj for xj in x]
        expected_c = [cj + hj * tj - gj / 2 for cj, hj, tj, gj in zip(c, H, theta, g)]
        model.observe(x, y)
        updated_c = [a * wj for a, wj in zip(model.precision, model.w)]
        assert max(abs(a - b) for a, b in zip(updated_c, expected_c)) < 3e-14

        # Complete-square stability computed independently from polynomial
        # coefficients, and again from the two gradients.
        stability = (dot(g, theta) - quadratic(theta, H)
                     - math.fsum(cj * cj / a for cj, a in zip(c, diagonal))
                     + math.fsum(cj * cj / a for cj, a in zip(updated_c, model.precision)))
        gradient_identity = (math.fsum(gj * gj / a for gj, a in zip(opposite, model.precision))
                             - math.fsum((g1 + g2) ** 2 / a
                                         for g1, g2, a in zip(g, opposite, diagonal))) / 4
        err = abs(stability - gradient_identity)
        max_diag_identity_error = max(max_diag_identity_error, err)
        assert err < 8e-14
        upper = 2 * alpha * math.fsum(xj * xj / a for xj, a in zip(x, model.precision))
        assert stability <= upper + 8e-14
        count += 4

    for cls in (HalfLookahead, RecenteredDiagonalAIOLI):
        zero = cls(2, 1, 0, 1)
        assert zero.predict_score([0, 0]) == 0
        zero.observe([0, 0], 1)
        assert zero.w == [0, 0]
        try:
            cls(2, 0, 1, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("B=0 should be rejected")
        count += 2

    T = 41000
    gap = math.log1p(math.exp(-0.5)) - math.log1p(math.exp(-1.0))
    report = {
        "status": "PASS",
        "scope": "Local algebra, high-precision root comparison, and formula constants; no learning benchmark or floating-point regret proof.",
        "checked_assertions": count,
        "max_score_error_with_prescribed_half_root_tolerance": max_score_error,
        "max_half_potential_identity_error": max_half_identity_error,
        "max_diagonal_stability_identity_error": max_diag_identity_error,
        "two_phase_exact_counterexample": {
            "d": 2, "B": 1, "R": 1, "lambda": 1,
            "T": T, "first_phase": 40000, "second_phase": 1000,
            "alpha_at_half": sigmoid(0.5) * (1 - sigmoid(0.5)) / 4,
            "regret_lower_bound": 1000 * gap,
            "full_AIOLI_exact_upper_bound": 1 + 4 * math.log1p(T / 32),
            "lower_bound_per_T": gap / 41,
        },
    }
    output = Path(__file__).resolve().parents[2] / "DIAGONAL_AIOLI_ROOT_FORMULA_CHECK.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
