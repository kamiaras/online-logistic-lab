"""AIOLI quadratic-surrogate prediction with a bounded scalar solve.

The defining objective and real-arithmetic error schedule are documented in
PUBLISHED_AIOLI_COMPARISON.md. Cholesky is the default numerical backend;
Sherman--Morrison is available for independent cross-checks. These floating-
point implementations do not have a rounding-error regret certificate.
"""
import math
from time import perf_counter

import numpy as np
from scipy.linalg import norm, solve_triangular

from .common import Metrics, sigmoid


def _cholesky_rank_one(L, update):
    """In-place lower-Cholesky update using bounded orthogonal rotations."""
    w = np.array(update, dtype=np.float64, copy=True)
    d = len(w)
    for k in range(d):
        old_diagonal = L[k, k]
        r = math.hypot(float(old_diagonal), float(w[k]))
        if not math.isfinite(r) or r <= 0:
            raise FloatingPointError("Invalid Cholesky diagonal during update")
        cosine, sine = old_diagonal / r, w[k] / r
        L[k, k] = r
        if k + 1 < d:
            old_column = L[k + 1:, k].copy()
            tail = w[k + 1:]
            L[k + 1:, k] = cosine * old_column + sine * tail
            w[k + 1:] = cosine * tail - sine * old_column


def _score_root(v, q, steps, metrics):
    """Bisect rho - tanh((v-q*rho/2)/2) on [-1,1].

The score is v-q*rho/2. This equivalent coordinate avoids dividing by a
small q and has the same residual bound as the documented score bisection.
"""
    lo, hi = -1.0, 1.0
    rho = 0.0
    evaluations = iterations = 0
    stalled = False
    for _ in range(steps):
        rho = (lo + hi) / 2.0
        if rho == lo or rho == hi:
            stalled = True
            break
        z = v - (q / 2.0) * rho
        residual = rho - math.tanh(z / 2.0)
        evaluations += 1
        iterations += 1
        if residual == 0.0:
            break
        if residual > 0.0:
            hi = rho
        else:
            lo = rho
    else:
        rho = (lo + hi) / 2.0
    z = v - (q / 2.0) * rho
    residual = rho - math.tanh(z / 2.0)
    evaluations += 1
    metrics.inc("root_iterations", iterations)
    metrics.inc("root_evaluations", evaluations)
    metrics.inc("tanh_elements", evaluations)
    metrics.inc("estimated_scalar_flops", 8 * evaluations + 2 * iterations)
    if stalled:
        metrics.inc("root_rounding_stalls")
    if not math.isfinite(z) or not math.isfinite(residual):
        raise FloatingPointError("Non-finite AIOLI scalar root")
    return float(z), float(rho), float(residual)


class AIOLI:
    """Online predictor; call predict(x), then observe(the_same_x, +/-1).

    Config keys:
      R: feature norm bound (default 1).
      matrix_backend: 'cholesky' (default), or 'inverse'.
      bisections: optional explicit scalar-work override.
    """

    def __init__(self, d, B, T, seed, config=None):
        if int(d) != d or d < 1 or int(T) != T or T < 1:
            raise ValueError("d and T must be positive integers")
        self.d, self.T, self.B, self.seed = int(d), int(T), float(B), int(seed)
        config = {} if config is None else dict(config)
        self.R = float(config.get("R", 1.0))
        if not math.isfinite(self.B) or self.B <= 0:
            raise ValueError("B must be positive and finite")
        if not math.isfinite(self.R) or self.R < 0:
            raise ValueError("R must be nonnegative and finite")
        self.lam = (1.0 / self.B) ** 2
        kappa = (self.B * self.R) ** 2
        if not math.isfinite(kappa) or self.lam <= 0 or not math.isfinite(self.lam):
            raise ValueError("B and R exceed the supported float64 range")
        if kappa == 0:
            self.prescribed_bisections = 0
        else:
            log_budget = (math.log(1.5) + math.log(self.T) + math.log(kappa)
                          + math.log1p(kappa / 4.0))
            self.prescribed_bisections = max(0, math.ceil(log_budget / math.log(2.0)))
        self.bisections = config.get("bisections", self.prescribed_bisections)
        if int(self.bisections) != self.bisections or self.bisections < 0:
            raise ValueError("bisections must be a nonnegative integer")
        self.bisections = int(self.bisections)
        self.backend = config.get("matrix_backend", "cholesky")
        if self.backend not in {"cholesky", "inverse"}:
            raise ValueError("matrix_backend must be 'cholesky' or 'inverse'")
        self.L = self.M = None
        if self.backend == "cholesky":
            self.L = np.zeros((self.d, self.d), order="F")
            np.fill_diagonal(self.L, 1.0 / self.B)
        else:
            self.M = np.eye(self.d) / self.lam
        self.c = np.zeros(self.d)
        self.metrics = Metrics()
        self.metrics.inc("matrix_initializations")
        self.metrics.inc("estimated_scalar_flops", self.d)
        self.t = 0
        self._pending = None
        self._last_q = self._last_v = self._last_root_residual = 0.0
        self._max_abs_logit = 0.0

    def _feature(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.d,) or not np.isfinite(x).all():
            raise ValueError("x must be a finite vector of length d")
        xnorm = float(norm(x, check_finite=False))
        if xnorm > self.R * (1.0 + 1e-12):
            raise ValueError("Feature norm exceeds configured R")
        return x

    def predict(self, x):
        if self._pending is not None:
            raise RuntimeError("observe must follow the previous prediction")
        if self.t >= self.T:
            raise RuntimeError("Configured horizon has already been consumed")
        start = perf_counter()
        x = self._feature(x)
        if not np.any(x):
            h, q, v, z, root_residual = np.zeros(self.d), 0., 0., 0., 0.
            self.metrics.inc("zero_feature_predictions")
        else:
            if self.backend == "cholesky":
                whitened = solve_triangular(self.L, x, lower=True, check_finite=False)
                h = solve_triangular(self.L.T, whitened, lower=False, check_finite=False)
                q = float(whitened @ whitened)
                self.metrics.inc("triangular_solves", 2)
                self.metrics.inc("estimated_matmul_flops", 2 * self.d * self.d)
            else:
                h = self.M @ x
                q = float(x @ h)
                self.metrics.inc("matrix_vector_products")
                self.metrics.inc("estimated_matmul_flops", 2 * self.d * self.d)
            v = float(self.c @ h)
            self.metrics.inc("dot_products", 2)
            self.metrics.inc("estimated_scalar_flops", 4 * self.d)
            if q <= 0 or not math.isfinite(q) or not math.isfinite(v):
                raise FloatingPointError("Invalid quadratic form in AIOLI prediction")
            z, _, root_residual = _score_root(v, q, self.bisections, self.metrics)
        self._pending = {"x": x.copy(), "h": h, "q": q, "logit": z}
        self._last_q, self._last_v = q, v
        self._last_root_residual = abs(root_residual)
        self._max_abs_logit = max(self._max_abs_logit, abs(z))
        self.metrics.inc("predictions")
        self.metrics.times["predict"] += perf_counter() - start
        return float(z)

    def observe(self, x, y):
        if self._pending is None:
            raise RuntimeError("predict must precede observe")
        x = self._feature(x)
        if not np.array_equal(x, self._pending["x"]):
            raise ValueError("observe must receive the feature used by predict")
        if y not in (-1, 1):
            raise ValueError("y must be -1 or 1")
        start = perf_counter()
        z, q, h = (self._pending[k] for k in ("logit", "q", "h"))
        if q > 0:
            # Compute curvature from the smaller tail, not p*(1-p) when p≈1.
            p, tail = sigmoid(np.array([-y * z, -abs(z)]))
            alpha = float(tail * (1.0 - tail) / (2.0 * (1.0 + self.B * self.R)))
            beta = float(alpha * z + y * p / 2.0)
            if tail == 0.0:
                self.metrics.inc("curvature_underflows")
            if alpha > 0:
                if self.backend == "cholesky":
                    _cholesky_rank_one(self.L, math.sqrt(alpha) * x)
                    self.metrics.inc("cholesky_rank1_updates")
                    self.metrics.inc("cholesky_rotations", self.d)
                    self.metrics.inc("estimated_scalar_flops",
                                     3 * self.d * (self.d - 1) + 8 * self.d)
                else:
                    coefficient = alpha / (1.0 + alpha * q)
                    self.M -= coefficient * np.outer(h, h)
                    self.metrics.inc("inverse_rank1_updates")
                    self.metrics.inc("estimated_scalar_flops", 3 * self.d * self.d + 3)
            self.c += beta * x
            self.metrics.inc("sufficient_statistic_updates")
            self.metrics.inc("sigmoid_elements", 2)
            self.metrics.inc("estimated_scalar_flops", 2 * self.d + 16)
        self.t += 1
        self._pending = None
        self.metrics.inc("observations")
        self.metrics.times["observe"] += perf_counter() - start

    def state_bytes(self):
        """Owned persistent NumPy storage; Python overhead/temporaries excluded."""
        count = self.c.nbytes
        count += self.L.nbytes if self.L is not None else self.M.nbytes
        if self._pending is not None:
            count += self._pending["x"].nbytes + self._pending["h"].nbytes
        return int(count)

    def diagnostics(self):
        return {
            "matrix_backend": self.backend,
            "rounds_observed": self.t,
            "regularizer_lambda": self.lam,
            "feature_radius": self.R,
            "bisections": self.bisections,
            "prescribed_bisections": self.prescribed_bisections,
            "uses_prescribed_bisection_budget": self.bisections >= self.prescribed_bisections,
            "last_q": self._last_q,
            "last_v": self._last_v,
            "last_scalar_root_residual": self._last_root_residual,
            "max_abs_logit": self._max_abs_logit,
            "finite_precision_certified": False,
            "flop_counts_are_estimates": True,
            "state_bytes_scope": "owned persistent ndarray storage",
        }
