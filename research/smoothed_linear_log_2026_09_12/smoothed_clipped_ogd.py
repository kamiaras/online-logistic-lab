"""Clipped projected gradient with logarithmic expected hindsight regret.

Theory: SMOOTHED_LINEAR_LOG_RESULT.md. Assumes fresh bounded feature noise
with known delta>0 and a conditional label probability in [epsilon,1-epsilon].
The state w is in normalized coordinates: feature x/R, parameter radius B*R.
One d-vector and scalar registers are retained. There is no inner solver.
This float prototype does not itself establish a finite-bit regret theorem.
"""

import math


def sigmoid(z):
    if z >= 0.0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


class SmoothedClippedOGD:
    """Call predict(x), then observe(x, Y), using the same caller-owned x.

    Labels are Y in {0,1}. For +/-1 labels use Y=(y+1)/2.
    delta is the uniform-ball feature mixing weight; epsilon is the label floor.
    No horizon is required. Runtime and stored word count are O(d).
    """

    def __init__(self, d, B, R, delta, epsilon):
        if not isinstance(d, int) or d < 1:
            raise ValueError("d must be a positive integer")
        if not all(math.isfinite(v) for v in (B, R, delta, epsilon)):
            raise ValueError("All parameters must be finite")
        if B < 0 or R < 0 or not 0 < delta <= 1 or not 0 < epsilon <= 0.5:
            raise ValueError("Require B,R>=0, 0<delta<=1, 0<epsilon<=1/2")
        self.d, self.B, self.R = d, B, R
        self.delta, self.epsilon = delta, epsilon
        self.b = B * R
        if not math.isfinite(self.b):
            raise ValueError("B*R exceeds supported floating-point range")
        self.floor = epsilon / 2.0
        self.q = self.floor * (1.0 - self.floor)
        self.k = self.q / (4.0 * (1.0 + 2.0 * self.b))
        self.nu = delta * delta / (d + 2.0)
        self.neutral = self.b == 0.0
        if not self.neutral and (self.k * self.nu <= 0 or
                                 not math.isfinite(1.0 / (self.k * self.nu))):
            raise ValueError("Step coefficient exceeds supported float range")
        self.step_coefficient = 0.0 if self.neutral else 1.0 / (self.k * self.nu)
        self.w = [0.0] * d
        self.t = 0
        self.pending_probability = None

    def _check_x(self, x):
        if len(x) != self.d or not all(math.isfinite(v) for v in x):
            raise ValueError("Invalid feature vector")
        if math.hypot(*x) > self.R * (1.0 + 1e-12):
            raise ValueError("Feature norm exceeds R")

    def predict(self, x):
        if self.pending_probability is not None:
            raise RuntimeError("Observe the pending label before predicting")
        self._check_x(x)
        if self.neutral:
            probability = 0.5
        else:
            z = math.fsum(wj * (xj / self.R) for wj, xj in zip(self.w, x))
            probability = min(1.0 - self.floor, max(self.floor, sigmoid(z)))
        self.pending_probability = probability
        return probability

    def observe(self, x, Y):
        if self.pending_probability is None:
            raise RuntimeError("Predict before observing")
        if Y not in (0, 1):
            raise ValueError("Label must be 0 or 1")
        self._check_x(x)
        self.t += 1
        if not self.neutral:
            scale = self.step_coefficient * (self.pending_probability - Y) / self.t
            for j in range(self.d):
                self.w[j] -= scale * (x[j] / self.R)
            norm = math.hypot(*self.w)
            if not math.isfinite(norm):
                raise FloatingPointError("Update exceeds supported float range")
            if norm > self.b:
                shrink = self.b / norm
                for j in range(self.d):
                    self.w[j] *= shrink
        self.pending_probability = None

    def expected_regret_bound(self, T):
        """The analytic bound, conditional on the model assumptions.

        Uses H_T <= 1+log(T) to avoid any horizon-length computation.
        This is not a data-driven certificate or a measured regret.
        """
        if not isinstance(T, int) or T < 1:
            raise ValueError("T must be a positive integer")
        if self.neutral:
            return 0.0
        return ((1.0 + math.log(T)) / (2.0 * self.k * self.nu)
                + self.k * self.b * self.b
                * (4.0 * self.d * math.log1p(2.0 * T) + 6.0))
