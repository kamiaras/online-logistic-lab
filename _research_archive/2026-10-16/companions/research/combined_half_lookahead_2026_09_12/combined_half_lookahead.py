"""Explicit half-lookahead, adaptive diagonal curvature, and paired clipping.

The model is the one in SMOOTHED_LINEAR_LOG_RESULT.md: normalized bounded
features have fresh uniform-ball perturbations of weight delta, and the
committed conditional label probability belongs to [epsilon, 1-epsilon].
The comparator is the original, unclipped logistic predictor with norm <= B.

The combined theorem uses kappa=1/[6(1+B*R)], incorporating the sharper
AIOLI comparator-radius remainder verified after the first research draft.
One learned vector and scalar registers are retained; there is no history or
inner solver. A common diagonal floor plus one coordinate boost allows exact
cylinder projection in O(d). The optional two-expert safeguard retains two
learned vectors, still O(d), and costs at most log(2) against either expert.

This is a standard-library FLOAT PROTOTYPE. The mathematical theorem is in
exact real arithmetic. Rounding, underflow, large counters, and fixed-bit
memory are not covered by a finite-precision regret certificate. Parameters
that cannot represent the clipping interval or positive metric scale are
rejected. Diagnostic values are not measured or pathwise regret bounds.
"""

import importlib.util
import math
from pathlib import Path


def sigmoid(z):
    """Overflow-safe logistic function; ordinary float rounding still applies."""
    if z >= 0.0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def _clip(value, lower, upper):
    return min(upper, max(lower, value))


def _check_horizon(T):
    if not isinstance(T, int) or isinstance(T, bool) or T < 1:
        raise ValueError("T must be a positive integer")


def _log1p_product(scale, T):
    """Compute log(1 + scale*T) without overflowing the product."""
    log_product = math.log(scale) + math.log(T)
    if log_product > 0.0:
        return log_product + math.log1p(math.exp(-log_product))
    return math.log1p(math.exp(log_product))


def _logaddexp(left, right):
    largest = max(left, right)
    if largest == -math.inf:
        return largest
    return largest + math.log1p(math.exp(min(left, right) - largest))


class CombinedHalfLookahead:
    """Call predict(x), then observe(x, Y), with the same caller-owned x.

    Labels are 0 or 1; convert a +/-1 label y with Y=(y+1)/2. The feature is
    not copied or retained, so supplying the same x at observation is a caller
    contract. State w is normalized: its score is w dot (x/R). The current
    metric is baseline*I + boost*e_coordinate*e_coordinate.T.

    The cylinder contains the entire comparator ball. Consequently the
    learner may have norm up to sqrt(2)*B*R without changing the comparator.
    """

    def __init__(self, d, B, R, delta, epsilon):
        if not isinstance(d, int) or isinstance(d, bool) or d < 1:
            raise ValueError("d must be a positive integer")
        if not all(math.isfinite(v) for v in (B, R, delta, epsilon)):
            raise ValueError("All parameters must be finite")
        if B < 0.0 or R < 0.0 or not 0.0 < delta <= 1.0 or not 0.0 < epsilon <= 0.5:
            raise ValueError("Require B,R>=0, 0<delta<=1, 0<epsilon<=1/2")
        self.d, self.B, self.R = d, B, R
        self.delta, self.epsilon = delta, epsilon
        self.b = B * R
        self.D = math.sqrt(2.0) * self.b
        if not math.isfinite(self.D):
            raise ValueError("Parameter radius exceeds supported float range")
        self.neutral = self.b == 0.0
        self.floor = epsilon / 4.0
        self.M = 0.0
        self.tau = self.kappa = self.gamma = self.omega = 0.0
        self.L1 = math.exp(0.5) / 2.0
        self.L2 = self.L1
        if not self.neutral:
            # The strict margin also protects the shifted clipping interval.
            if self.floor <= 0.0 or 1.0 - self.floor >= 1.0:
                raise ValueError("Clipping interval exceeds supported float precision")
            self.M = math.log1p(-self.floor) - math.log(self.floor)
            scale = self.D * delta
            bracket = math.sqrt(2.0 * (1.0 + scale)) + scale
            denominator = (d + 2.0 + scale) * bracket * bracket
            self.tau = 2.0 * delta * delta / denominator
            self.kappa = 1.0 / (6.0 * (1.0 + self.b))
            self.gamma = self.kappa / 2.0
            self.omega = self.kappa * self.tau / 4.0
            if (self.omega <= 0.0 or not math.isfinite(self.omega)
                    or not math.isfinite(1.0 / self.omega)):
                raise ValueError("Metric coefficient exceeds supported float range")
            self.L2 += 3.0 * self.kappa / 4.0
        self.w = [0.0] * d
        self.baseline = 1.0
        self.boost = 0.0
        self.coordinate = 0
        self.t = 0
        self.sum_curvature = 0.0
        self.sum_curvature_q = 0.0
        self.sum_curvature_q_squared = 0.0
        self.pending_probability = None
        self.pending_raw_probability = None
        self.pending_lower = None
        self.pending_upper = None
        self.pending_curvature = 0.0
        self.pending_q = 0.0

    def _check_x(self, x):
        if len(x) != self.d or not all(math.isfinite(v) for v in x):
            raise ValueError("Invalid feature vector")
        if math.hypot(*x) > self.R * (1.0 + 1e-12):
            raise ValueError("Feature norm exceeds R")

    def predict(self, x):
        """Commit the feature-dependent metric and return the paired forecast."""
        if self.pending_probability is not None:
            raise RuntimeError("Observe the pending label before predicting")
        self._check_x(x)
        if self.neutral:
            self.pending_probability = 0.5
            self.pending_raw_probability = 0.5
            self.pending_lower = self.pending_upper = 0.5
            return 0.5

        raw_score = math.fsum(wj * (xj / self.R) for wj, xj in zip(self.w, x))
        if not math.isfinite(raw_score):
            raise FloatingPointError("Score exceeds supported float range")
        p0 = _clip(sigmoid(raw_score), self.floor, 1.0 - self.floor)
        curvature = p0 * (1.0 - p0)
        optimism = p0 - 0.5

        # A fixed tie rule: the earliest coordinate attaining the maximum.
        coordinate = max(range(self.d), key=lambda j: abs(x[j]))
        selected_squared = (x[coordinate] / self.R) ** 2
        other_squared = math.fsum(
            (x[j] / self.R) ** 2 for j in range(self.d) if j != coordinate
        )
        radius_floor = self.tau / 2.0
        zeta = (radius_floor
                * (radius_floor + other_squared + selected_squared)
                / (radius_floor + other_squared))
        baseline = self.baseline + self.omega * curvature
        old_boost = self.boost if coordinate == self.coordinate else 0.0
        boost = old_boost + self.gamma * curvature * zeta
        if not all(math.isfinite(v) for v in (baseline, boost, baseline + boost)):
            raise FloatingPointError("Metric exceeds supported float range")
        q = other_squared / baseline + selected_squared / (baseline + boost)
        shift = q * optimism

        # In exact arithmetic this is sigma(clip(raw_score, [-M,M])-shift).
        # The shifted interval is also the interval used to clip the residual.
        raw_probability = sigmoid(raw_score - shift)
        lower = sigmoid(-self.M - shift)
        upper = sigmoid(self.M - shift)
        probability = _clip(raw_probability, lower, upper)
        if not 0.0 < lower <= probability <= upper < 1.0:
            raise FloatingPointError("Shifted clipping interval lost float precision")

        self.baseline, self.boost, self.coordinate = baseline, boost, coordinate
        self.pending_probability = probability
        self.pending_raw_probability = raw_probability
        self.pending_lower, self.pending_upper = lower, upper
        self.pending_curvature, self.pending_q = curvature, q
        return probability

    def observe(self, x, Y):
        """Update in place and project exactly onto the current cylinder.

        The residual is interval-clipped in coordination with the forecast.
        This is not an independent gradient-norm truncation.
        """
        if self.pending_probability is None:
            raise RuntimeError("Predict before observing")
        if Y not in (0, 1):
            raise ValueError("Label must be 0 or 1")
        self._check_x(x)
        if not self.neutral:
            residual = _clip(
                self.pending_raw_probability - Y,
                self.pending_lower - Y,
                self.pending_upper - Y,
            )
            selected = self.coordinate
            for j in range(self.d):
                metric = self.baseline + (self.boost if j == selected else 0.0)
                self.w[j] -= (residual / metric) * (x[j] / self.R)
            self.w[selected] = _clip(self.w[selected], -self.b, self.b)
            other_norm = math.hypot(*(self.w[j] for j in range(self.d)
                                      if j != selected))
            if not math.isfinite(other_norm) or not math.isfinite(self.w[selected]):
                raise FloatingPointError("Update exceeds supported float range")
            if other_norm > self.b:
                shrink = self.b / other_norm
                for j in range(self.d):
                    if j != selected:
                        self.w[j] *= shrink
            curvature, q = self.pending_curvature, self.pending_q
            self.sum_curvature += curvature
            self.sum_curvature_q += curvature * q
            self.sum_curvature_q_squared += curvature * q * q
        self.t += 1
        self.pending_probability = None
        self.pending_raw_probability = None
        self.pending_lower = self.pending_upper = None
        self.pending_curvature = self.pending_q = 0.0

    def expected_regret_bound(self, T):
        """Return the analytic, epsilon-free expected empirical regret bound.

        This is conditional on the stated noise model. It does not measure
        regret on supplied data and is not a floating-point error certificate.
        """
        _check_horizon(T)
        if self.neutral:
            return 0.0
        log_scale = _log1p_product(self.omega / 4.0, T)
        variance_charge = (self.L1 / self.omega) * log_scale
        correction_charge = ((self.L2 / self.omega)
                             * (-math.expm1(-log_scale)))
        hindsight_charge = (self.kappa * self.D * self.D * self.d / 2.0
                            * _log1p_product(2.0, T)
                            + self.kappa * self.D * self.b)
        return self.b * self.b / 2.0 + variance_charge + correction_charge + hindsight_charge

    def stability_diagnostic(self):
        """Return the data-dependent potential statistics after completed rounds.

        If V_T is the returned ``diagnostic_value``, the theorem gives
        E[regret_T] <= E[V_T] for a fixed horizon T under its assumptions.
        A realized V_T is NOT a pathwise or high-probability regret bound;
        the omitted martingale terms have only been controlled in expectation.
        """
        if self.pending_probability is not None:
            raise RuntimeError("Complete the pending round before reading diagnostics")
        stability = hindsight = 0.0
        if self.t and not self.neutral:
            stability = (self.b * self.b / 2.0
                         + self.L1 * self.sum_curvature_q
                         + self.L2 * self.sum_curvature_q_squared)
            hindsight = (self.kappa * self.D * self.D * self.d / 2.0
                         * _log1p_product(2.0, self.t)
                         + self.kappa * self.D * self.b)
        return {
            "rounds": self.t,
            "sum_curvature": self.sum_curvature,
            "sum_curvature_q": self.sum_curvature_q,
            "sum_curvature_q_squared": self.sum_curvature_q_squared,
            "potential_stability_term": stability,
            "expected_uniform_hindsight_term": hindsight,
            "diagnostic_value": stability + hindsight,
            "interpretation": "E[regret_T] <= E[diagnostic_value]; not a pathwise bound",
        }


_BASELINE_CLASS = None


def _original_baseline_class():
    """Load the existing reference implementation without modifying sys.path."""
    global _BASELINE_CLASS
    if _BASELINE_CLASS is None:
        path = (Path(__file__).resolve().parents[1]
                / "smoothed_linear_log_2026_09_12" / "smoothed_clipped_ogd.py")
        spec = importlib.util.spec_from_file_location("_combined_original_ogd", path)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load the original clipped OGD reference")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _BASELINE_CLASS = module.SmoothedClippedOGD
    return _BASELINE_CLASS


class SafeguardedHalfLookahead:
    """Equal-prior two-expert log-loss mixture; O(d) work and persistent state.

    The experts are CombinedHalfLookahead and the existing SmoothedClippedOGD.
    In exact arithmetic the mixture's loss is within log(2) of either expert
    on every realized sequence. Its expected empirical regret is therefore
    bounded by min(combined bound, original bound)+log(2).
    """

    def __init__(self, d, B, R, delta, epsilon):
        self.combined = CombinedHalfLookahead(d, B, R, delta, epsilon)
        self.original = _original_baseline_class()(d, B, R, delta, epsilon)
        self.log_weight_combined = -math.log(2.0)
        self.log_weight_original = -math.log(2.0)
        self.pending_probability = None
        self.pending_combined = None
        self.pending_original = None
        self.t = 0

    def predict(self, x):
        if self.pending_probability is not None:
            raise RuntimeError("Observe the pending label before predicting")
        combined = self.combined.predict(x)
        original = self.original.predict(x)
        normalizer = _logaddexp(self.log_weight_combined, self.log_weight_original)
        left = math.exp(self.log_weight_combined - normalizer)
        right = math.exp(self.log_weight_original - normalizer)
        probability = math.fsum((left * combined, right * original))
        probability = _clip(probability, min(combined, original), max(combined, original))
        self.pending_combined, self.pending_original = combined, original
        self.pending_probability = probability
        return probability

    def observe(self, x, Y):
        if self.pending_probability is None:
            raise RuntimeError("Predict before observing")
        if Y not in (0, 1):
            raise ValueError("Label must be 0 or 1")
        self.combined._check_x(x)
        self.original._check_x(x)
        if Y == 1:
            left = self.log_weight_combined + math.log(self.pending_combined)
            right = self.log_weight_original + math.log(self.pending_original)
        else:
            left = self.log_weight_combined + math.log1p(-self.pending_combined)
            right = self.log_weight_original + math.log1p(-self.pending_original)
        normalizer = _logaddexp(left, right)
        self.combined.observe(x, Y)
        self.original.observe(x, Y)
        self.log_weight_combined = left - normalizer
        self.log_weight_original = right - normalizer
        self.t += 1
        self.pending_probability = None
        self.pending_combined = self.pending_original = None

    def expected_regret_bound(self, T):
        _check_horizon(T)
        if self.combined.neutral:
            return 0.0
        return min(self.combined.expected_regret_bound(T),
                   self.original.expected_regret_bound(T)) + math.log(2.0)
