"""Fixed Gaussian dictionary with the canonical AIOLI feature-dependent gate.

Implements GAUSSIAN_FEATURE_GATE_COMPLETE.md, including its finite composite
Newton--Cotes forecast. The learned gate is canonical AIOLI, not the altered
released comparison code. Its prediction-time quadrature never feeds back into
the gate. All components have a common positive isotropic variance.

This is a float64 reference, without a finite-precision regret certificate.
The prescribed scalar root is computed in the equivalent bounded rho coordinate
used by AIOLI. The identical quadrature nodes are cached rather than streamed:
the extra owned storage is O(log(T)**1.5), counted by state_bytes().
"""
from fractions import Fraction
import math
from time import perf_counter

import numpy as np

from .aioli import AIOLI


def _log_sigmoid(value):
    return -float(np.logaddexp(0.0, -value))


def _log_one_minus_exp_negative(value):
    """log(1-exp(-value)), retaining precision down to subnormal value."""
    if value < 0 or math.isnan(value):
        raise FloatingPointError("Negative argument in gate likelihood difference")
    if value == 0:
        return -math.inf
    if value <= math.log(2.0):
        return math.log(-math.expm1(-value))
    return math.log1p(-math.exp(-value))


def _gate_log_weights(z, m):
    """Return log w+ and log w- without subtracting saturated sigmoids.

The exact identities are w+ = sigma(z)*(1-exp(-(m+z)))/(1-exp(-m))
and w- = sigma(-z)*(1-exp(-(m-z)))/(1-exp(-m)).
"""
    if not math.isfinite(z) or not math.isfinite(m) or m <= 0:
        raise FloatingPointError("Non-finite or nonpositive anchor score")
    if abs(z) > m:
        raise FloatingPointError("Gate score exceeds the fixed-anchor bound")
    common = _log_one_minus_exp_negative(m)
    log_plus = (_log_sigmoid(z) + _log_one_minus_exp_negative(m + z)
                - common)
    log_minus = (_log_sigmoid(-z) + _log_one_minus_exp_negative(m - z)
                 - common)
    normalizer = float(np.logaddexp(log_plus, log_minus))
    if not math.isfinite(normalizer):
        raise FloatingPointError("Invalid feature-dependent gate weights")
    return log_plus - normalizer, log_minus - normalizer


def _newton_cotes_weights(degree):
    """Exact rational integrals of the Lagrange basis at i/degree.

Rational arithmetic occurs once in initialization, before conversion to double.
No library quadrature prescription or approximate Vandermonde solve is used.
"""
    result = []
    for i in range(degree + 1):
        coefficients = [1]
        denominator = 1
        for j in range(degree + 1):
            if j == i:
                continue
            product = [0] * (len(coefficients) + 1)
            for power, coefficient in enumerate(coefficients):
                product[power] -= j * coefficient
                product[power + 1] += degree * coefficient
            coefficients = product
            denominator *= i - j
        result.append(sum((Fraction(coefficient, denominator * (power + 1))
                           for power, coefficient in enumerate(coefficients)),
                          Fraction(0)))
    if sum(result, Fraction(0)) != 1:
        raise ArithmeticError("Newton--Cotes weights fail their exact mass identity")
    return result


class _PrescribedGaussianQuadrature:
    """The prescribed finite integration rule, with a cached common grid."""

    def __init__(self, T):
        self.T = int(T)
        self.L = 1.0 + math.sqrt(2.0 * (math.log(64.0) + math.log(T)))
        self.intervals = math.ceil(16.0 * self.L)
        self.degree = math.ceil((math.log(512.0) + math.log(self.L) + math.log(T))
                                / math.log(4.0))
        rational = _newton_cotes_weights(self.degree)
        panel_weights = np.array([float(weight) for weight in rational])
        # Endpoint repetitions are retained: these are precisely the panel
        # nodes of the proof, not a changed integration scheme.
        width = 2.0 * self.L / self.intervals
        indices = (np.arange(self.intervals)[:, None]
                   + np.arange(self.degree + 1)[None, :] / self.degree)
        nodes = (-self.L + width * indices).ravel()
        # Construct an exactly antisymmetric floating grid. Analytically this
        # is the same grid; it prevents asymmetric rounding of opposite tails.
        self.nodes = (nodes - nodes[::-1]) / 2.0
        weights = np.tile(panel_weights, self.intervals)
        self.signs = np.sign(weights)
        self.log_abs_coefficients = (np.log(np.abs(weights)) + math.log(width)
                                     - 0.5 * math.log(2.0 * math.pi)
                                     - 0.5 * self.nodes**2)
        if not np.isfinite(self.log_abs_coefficients).all():
            raise FloatingPointError("Non-finite prescribed quadrature coefficients")
        self.last_condition = 1.0
        self.max_condition = 1.0

    def log_likelihood(self, a, variance):
        """Approximate log E sigma(a+sqrt(variance)Z) directly for either tail."""
        if not math.isfinite(a) or not math.isfinite(variance) or not 0 <= variance <= 1:
            raise FloatingPointError("Gaussian quadrature requires finite a and variance in [0,1]")
        shifts = math.sqrt(variance) * self.nodes
        # Subtracting log-sigmoids at a huge negative anchor would erase the
        # O(sqrt(variance)) shift. These identities retain it explicitly.
        if a >= 0:
            log_ratios = np.logaddexp(0.0, -a) - np.logaddexp(0.0, -a - shifts)
        else:
            log_ratios = shifts + np.logaddexp(0.0, a) - np.logaddexp(0.0, a + shifts)
        terms = self.log_abs_coefficients + log_ratios
        scale = float(np.max(terms))
        scaled = np.exp(terms - scale) * self.signs
        total = math.fsum(scaled.tolist())
        absolute_total = float(np.sum(np.abs(scaled)))
        if total <= 0 or not math.isfinite(total):
            raise FloatingPointError("Signed Gaussian quadrature produced nonpositive likelihood")
        self.last_condition = absolute_total / total
        self.max_condition = max(self.max_condition, self.last_condition)
        # This is a failure detector, not a floating-point certificate. Refuse
        # a rule whose elementary cancellation estimate exhausts its budget.
        if 64.0 * np.finfo(float).eps * self.last_condition >= 1.0 / (7.0 * self.T):
            raise FloatingPointError("Quadrature cancellation exceeds the numerical accuracy budget")
        log_integral = scale + math.log(total)
        result = _log_sigmoid(a) + log_integral
        if not math.isfinite(result):
            raise FloatingPointError("Non-finite Gaussian component log likelihood")
        return result

    def state_bytes(self):
        return int(self.nodes.nbytes + self.signs.nbytes + self.log_abs_coefficients.nbytes)


class GaussianFeatureGate(AIOLI):
    """predict(x)->logit, then observe(x,y), for the actual Gaussian mixture.

Config accepts R and bisections from AIOLI. The inverse matrix backend is
required; an explicit different backend is rejected. The default bisection and
quadrature budgets are prescribed. No practical quadrature or covariance caps
are supplied. seed is retained for interface compatibility; the rule is
deterministic. B must be positive; R=0 is handled by neutral predictions.

Public last_base_logit is the internal canonical score; last_mixture_logit is
the emitted finite Gaussian forecast. last_covariance_variance is coordinate
variance tau at prediction, and last_projected_variance is tau*||x||**2.
"""

    def __init__(self, d, B, T, seed, config=None):
        start = perf_counter()
        config = {} if config is None else dict(config)
        if config.get("matrix_backend", "inverse") != "inverse":
            raise ValueError("GaussianFeatureGate requires matrix_backend='inverse'")
        config["matrix_backend"] = "inverse"
        super().__init__(d, B, T, seed, config)
        self.anchor_scale = self.T * self.R * (1.0 / self.lam) * math.sqrt(self.d) / 2.0
        self.covariance_precision = self.T * self.R * self.R if self.R else 1.0
        if (not math.isfinite(self.anchor_scale)
                or not math.isfinite(self.covariance_precision)
                or self.covariance_precision <= 0
                or (self.R > 0 and self.anchor_scale <= 0)):
            raise ValueError("Gaussian dictionary or covariance exceeds the float64 range")
        self.quadrature = _PrescribedGaussianQuadrature(self.T)
        self.last_base_logit = self.last_mixture_logit = 0.0
        self.last_covariance_variance = 1.0 / self.covariance_precision
        if not math.isfinite(self.last_covariance_variance) or self.last_covariance_variance <= 0:
            raise ValueError("Initial covariance is not a positive finite float64 variance")
        self.last_projected_variance = self.last_anchor_score = 0.0
        self.last_anchor_coordinate = 0
        self.last_log_weight_plus = self.last_log_weight_minus = -math.log(2.0)
        self._max_abs_mixture_logit = self._max_abs_smoothing_difference = 0.0
        self._max_smoothing_loss_excess = self._sum_projected_variance = 0.0
        self._signed_smoothing_loss_sum = 0.0
        self.metrics.inc("quadrature_grid_initializations")
        self.metrics.inc("quadrature_grid_nodes", len(self.quadrature.nodes))
        self.metrics.times["gaussian_initialize"] += perf_counter() - start

    @property
    def tau(self):
        return 1.0 / self.covariance_precision

    def predict(self, x):
        base = super().predict(x)
        start = perf_counter()
        pending_x = self._pending["x"]
        self.last_base_logit = base
        self.last_covariance_variance = self.tau
        norm_squared = float(pending_x @ pending_x)
        variance = self.last_covariance_variance * norm_squared
        if variance < 0 or not math.isfinite(variance):
            raise FloatingPointError("Non-finite projected covariance")
        self.last_projected_variance = variance
        self._pending["norm_squared"] = norm_squared
        if self._pending["q"] == 0:
            mixture = 0.0
            self.last_anchor_score = 0.0
            self.last_anchor_coordinate = 0
            self.last_log_weight_plus = self.last_log_weight_minus = -math.log(2.0)
        else:
            j = int(np.argmax(np.abs(pending_x)))
            m = self.anchor_scale * abs(float(pending_x[j]))
            log_plus, log_minus = _gate_log_weights(base, m)
            log_likelihood_plus = self.quadrature.log_likelihood(m, variance)
            log_likelihood_minus = self.quadrature.log_likelihood(-m, variance)
            # The opposite component exchanges the two directly evaluated
            # likelihoods by symmetry of the Gaussian and the common grid.
            log_positive = float(np.logaddexp(log_plus + log_likelihood_plus,
                                              log_minus + log_likelihood_minus))
            log_negative = float(np.logaddexp(log_plus + log_likelihood_minus,
                                              log_minus + log_likelihood_plus))
            # Normalizing both masses leaves this log odds unchanged.
            mixture = log_positive - log_negative
            if not math.isfinite(mixture):
                raise FloatingPointError("Non-finite Gaussian mixture log odds")
            self.last_anchor_score, self.last_anchor_coordinate = m, j
            self.last_log_weight_plus, self.last_log_weight_minus = log_plus, log_minus
            self.metrics.inc("gaussian_component_integrals", 2)
            self.metrics.inc("quadrature_node_evaluations", 2 * len(self.quadrature.nodes))
            self.metrics.inc("quadrature_logistic_elements", 2 * len(self.quadrature.nodes))
        self.last_mixture_logit = float(mixture)
        self._pending["mixture_logit"] = float(mixture)
        self._max_abs_mixture_logit = max(self._max_abs_mixture_logit, abs(mixture))
        self._max_abs_smoothing_difference = max(self._max_abs_smoothing_difference,
                                                abs(mixture - base))
        self._sum_projected_variance += variance
        if mixture == base:
            self.metrics.inc("forecast_equals_gate_in_float64")
        self.metrics.times["gaussian_forecast"] += perf_counter() - start
        self.metrics.times["predict"] += perf_counter() - start
        return float(mixture)

    def observe(self, x, y):
        if self._pending is None:
            raise RuntimeError("predict must precede observe")
        pending = self._pending
        # The parent validates matching x and y before changing its gate state.
        super().observe(x, y)
        start = perf_counter()
        precision = self.covariance_precision + pending["norm_squared"] / (4.0 * self.d)
        if not math.isfinite(precision) or precision <= 0 or 1.0 / precision <= 0:
            raise FloatingPointError("Covariance update lost positive finite variance")
        self.covariance_precision = precision
        extra_loss = float(np.logaddexp(0.0, -y * pending["mixture_logit"])
                           - np.logaddexp(0.0, -y * pending["logit"]))
        self._max_smoothing_loss_excess = max(self._max_smoothing_loss_excess, extra_loss)
        self._signed_smoothing_loss_sum += extra_loss
        self.metrics.inc("covariance_precision_updates")
        self.metrics.times["observe"] += perf_counter() - start

    def state_bytes(self):
        return super().state_bytes() + self.quadrature.state_bytes()

    def diagnostics(self):
        result = super().diagnostics()
        result.update({
            "method": "fixed_gaussian_feature_gate",
            "gate_source": "canonical AIOLI; inverse update; original half statistic increment",
            "gaussian_count": 2 * self.d,
            "active_gaussians_per_nonzero_prediction": 2,
            "fixed_anchor_scale": self.anchor_scale,
            "covariance_precision": self.covariance_precision,
            "coordinate_variance": self.tau,
            "last_base_logit": self.last_base_logit,
            "last_gate_logit": self.last_base_logit,
            "last_mixture_logit": self.last_mixture_logit,
            "last_covariance_variance": self.last_covariance_variance,
            "last_projected_variance": self.last_projected_variance,
            "last_anchor_score": self.last_anchor_score,
            "last_anchor_coordinate": self.last_anchor_coordinate,
            "max_abs_mixture_logit": self._max_abs_mixture_logit,
            "max_abs_gaussian_minus_gate_logit": self._max_abs_smoothing_difference,
            "maximum_observed_smoothing_loss_excess": self._max_smoothing_loss_excess,
            "cumulative_observed_gaussian_minus_gate_loss": self._signed_smoothing_loss_sum,
            "cumulative_projected_variance": self._sum_projected_variance,
            "quadrature_prescribed": True,
            "quadrature_degree": self.quadrature.degree,
            "quadrature_intervals": self.quadrature.intervals,
            "quadrature_nodes_per_integral": len(self.quadrature.nodes),
            "quadrature_cached_bytes": self.quadrature.state_bytes(),
            "quadrature_max_signed_sum_condition": self.quadrature.max_condition,
            "quadrature_memory": "cached O(log(T)^1.5) grid rather than proof's streamed O(log(T)) cells",
            "integration_error_feeds_gate": False,
            "history_rows_retained": 0,
            "finite_precision_certified": False,
        })
        return result
