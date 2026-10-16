"""Small-state logistic forecasters derived in DIAGONAL_AIOLI_RESULT.md.

RecenteredDiagonalAIOLI has a correlation-defect regret bound, not AIOLI's
general logarithmic guarantee. HalfLookahead has a square-root regret bound.
Both retain one parameter vector; the former also retains d diagonal entries.
Floating-point arithmetic here has no independent rounding-error theorem.
Call predict_score(x), then observe(x, y), with the same caller-owned x.
"""

import math


def sigmoid(z):
    if z >= 0:
        tail = math.exp(-z)
        return 1.0 / (1.0 + tail)
    tail = math.exp(z)
    return tail / (1.0 + tail)


def _root(v, q, multiplier, steps):
    """Solve rho=tanh((v-multiplier*q*rho)/2) in [-1,1]."""
    lo, hi = -1.0, 1.0
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        if mid == lo or mid == hi:
            break
        value = mid - math.tanh((v - multiplier * q * mid) / 2.0)
        if value == 0.0:
            lo = hi = mid
            break
        if value > 0.0:
            hi = mid
        else:
            lo = mid
    rho = (lo + hi) / 2.0
    z = v - multiplier * q * rho
    return z, rho, rho - math.tanh(z / 2.0)


def _steps(log_budget):
    return max(0, math.ceil(log_budget / math.log(2.0)))


class _Base:
    def _initialize(self, d, B, R, T, lam):
        if not isinstance(d, int) or d < 1 or not isinstance(T, int) or T < 1:
            raise ValueError("d and T must be positive integers")
        if not all(math.isfinite(v) for v in (B, R, lam)):
            raise ValueError("Parameters must be finite")
        if B <= 0 or R < 0 or lam <= 0:
            raise ValueError("Require B>0, R>=0, lambda>0")
        self.d, self.B, self.R, self.T, self.lam = d, B, R, T, lam
        self.w = [0.0] * d
        self.rounds = 0
        self.pending = None  # Only three scalar registers, no feature copy.

    def _check_x(self, x):
        if len(x) != self.d or not all(math.isfinite(v) for v in x):
            raise ValueError("Feature vector has invalid dimension or values")
        if math.fsum(v * v for v in x) > self.R * self.R * (1 + 1e-12):
            raise ValueError("Feature norm exceeds R")

    def predict_probability(self, x):
        return sigmoid(self.predict_score(x))

    def _begin_prediction(self, x):
        if self.pending is not None:
            raise RuntimeError("Observe the pending label first")
        if self.rounds >= self.T:
            raise RuntimeError("Prescribed horizon reached")
        self._check_x(x)

    def _begin_observation(self, x, y):
        if self.pending is None:
            raise RuntimeError("Predict before observing")
        if y not in (-1, 1):
            raise ValueError("Label must be -1 or +1")
        self._check_x(x)
        # The caller must supply the same feature as in predict_score.
        # Retaining another d-vector merely to check equality is unnecessary.
        return self.pending


class RecenteredDiagonalAIOLI(_Base):
    """O(d+K) work, 2d persistent numeric entries plus scalar registers."""

    def __init__(self, d, B, R, T, lam=None):
        if not math.isfinite(B) or B <= 0:
            raise ValueError("Require finite B>0")
        lam = 1.0 / (B * B) if lam is None else lam
        self._initialize(d, B, R, T, lam)
        self.precision = [lam] * d
        if R == 0:
            self.bisections = 0
        else:
            qmax = R * R / lam
            self.bisections = _steps(math.log(1.5 * T) + math.log(qmax)
                                     + math.log1p(qmax / 4.0))

    def predict_score(self, x):
        self._begin_prediction(x)
        q = math.fsum(xj * xj / a for xj, a in zip(x, self.precision))
        v = math.fsum(xj * wj for xj, wj in zip(x, self.w))
        self.pending = _root(v, q, 0.5, self.bisections)
        return self.pending[0]

    def observe(self, x, y):
        z, rho, _ = self._begin_observation(x, y)
        p = sigmoid(-y * z)
        # Small tail avoids destroying curvature when p rounds to one.
        tail = sigmoid(-abs(z))
        alpha = tail * (1.0 - tail) / (2.0 * (1.0 + self.B * self.R))
        for j in range(self.d):
            old = self.precision[j]
            addition = alpha * x[j] * x[j]
            theta_j = self.w[j] - 0.5 * rho * x[j] / old
            new = old + addition
            self.w[j] = (old * self.w[j] + addition * theta_j
                         + 0.5 * y * p * x[j]) / new
            self.precision[j] = new
        self.rounds += 1
        self.pending = None


class HalfLookahead(_Base):
    """One vector and scalar precision; regret <= BR sqrt(T)/2 + 1.

The displayed guarantee uses default lambda and exact real arithmetic with
the prescribed finite bisections. A user-specified lambda instead gives
lambda B^2 + T R^2/(16 lambda) + 1.
"""

    def __init__(self, d, B, R, T, lam=None):
        if not math.isfinite(B) or B <= 0:
            raise ValueError("Require finite B>0")
        lam = (R * math.sqrt(T) / (4.0 * B) if R > 0 else 1.0) if lam is None else lam
        self._initialize(d, B, R, T, lam)
        if R == 0:
            self.bisections = 0
        else:
            qmax = R * R / lam
            self.bisections = _steps(math.log(T) + math.log(qmax / 4.0)
                                     + math.log1p(qmax / 8.0))

    def predict_score(self, x):
        self._begin_prediction(x)
        q = math.fsum(xj * xj for xj in x) / self.lam
        v = math.fsum(xj * wj for xj, wj in zip(x, self.w))
        self.pending = _root(v, q, 0.25, self.bisections)
        return self.pending[0]

    def observe(self, x, y):
        z, _, _ = self._begin_observation(x, y)
        scale = y * sigmoid(-y * z) / (2.0 * self.lam)
        for j in range(self.d):
            self.w[j] += scale * x[j]
        self.rounds += 1
        self.pending = None
