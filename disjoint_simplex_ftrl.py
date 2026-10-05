"""Reference implementation of Gaussian-prior, disjoint-slab simplex FTRL.

Each continuous component is m + s H U, with U uniform on a centered
standard simplex. Positive cubature approximates its expected loss; the
continuous-density KL is retained exactly. Bounds returned here assume exact
arithmetic: ordinary NumPy floating point is not an interval certificate.

Only NumPy is required. The user-selectable cubature degree is distinct from
the conservative theorem's potentially enormous accuracy schedule.
"""

from dataclasses import dataclass
import math
import numpy as np


def _compositions(total, parts):
    if parts == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for rest in _compositions(total - first, parts - 1):
                yield (first,) + rest


def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -np.asarray(z)))


@dataclass
class Fit:
    mean: np.ndarray
    scale: float
    cost_upper: float
    optimization_gap_bound: float
    cubature_gap_bound: float
    iterations: int
    degree: int

    @property
    def total_gap_bound(self):
        return self.optimization_gap_bound + self.cubature_gap_bound


class DisjointSimplexFTRL:
    def __init__(self, dimension, components, B=1.0, tau=1.0,
                 degree=24, optimization_tolerance=1e-8, max_nodes=200000):
        if dimension < 1 or components < 1 or B <= 0 or tau <= 0 or degree < 1:
            raise ValueError("Positive dimension, components, B, tau, degree required")
        self.d, self.M = int(dimension), int(components)
        self.B, self.tau, self.C = float(B), float(tau), 2.0 * float(B)
        self.degree = int(degree)
        self.tolerance = float(optimization_tolerance)
        self.max_nodes = int(max_nodes)
        self.h = np.full(self.d, self.C)
        self.h[0] /= self.M
        self.v = (self.d / ((self.d + 1)**2 * (self.d + 2))) * float(self.h @ self.h)
        self.prior_constant = (math.log(self.M) + math.lgamma(self.d + 1)
                               + self.d / 2 * math.log(2 * math.pi * self.tau**2)
                               - self.d * math.log(self.C))
        self.X = np.empty((0, self.d))
        self.y = np.empty(0)
        self.fits = []
        self._nodes_cache = {}
        self.refit()

    def cell(self, index):
        lower = np.full(self.d, -self.C)
        upper = np.full(self.d, self.C)
        lower[0] = -self.C + 2 * self.C * index / self.M
        upper[0] = -self.C + 2 * self.C * (index + 1) / self.M
        return lower, upper

    def center(self, index):
        lower, upper = self.cell(index)
        return (lower + upper) / 2

    def cubature_nodes(self, degree=None):
        k = self.degree if degree is None else int(degree)
        if k < 1:
            raise ValueError("Cubature degree must be positive")
        count = math.comb(k + self.d, self.d)
        if count > self.max_nodes:
            raise ValueError(f"Requested {count} cubature nodes exceeds max_nodes={self.max_nodes}")
        if k not in self._nodes_cache:
            counts = np.asarray(list(_compositions(k, self.d + 1)), dtype=float)
            u = counts[:, 1:] / k - 1.0 / (self.d + 1)
            self._nodes_cache[k] = u * self.h
        return self._nodes_cache[k]

    def scale_floor(self):
        s1 = float(np.linalg.norm(self.X, axis=1).sum())
        return self.d / (self.d + 2 * self.C * math.sqrt(self.d) * s1
                         + self.C**2 / (2 * self.tau**2))

    def project(self, point, index, floor=None):
        """Euclidean projection via a scalar piecewise-quadratic problem.

        Breakpoints determine the exact formula in real arithmetic. The finite
        scan costs O(d^2), rather than invoking a generic constrained solver.
        """
        point = np.asarray(point, dtype=float)
        mean, scale = point[:-1], float(point[-1])
        eta = self.scale_floor() if floor is None else float(floor)
        lower, upper = self.cell(index)
        alpha = self.h / (self.d + 1)
        beta = self.d * alpha
        breaks = [eta, 1.0]
        for value in np.concatenate(((mean - lower) / alpha, (upper - mean) / beta)):
            if eta < value < 1.0:
                breaks.append(float(value))
        breaks = sorted(set(breaks))

        def derivative(s):
            return (s - scale + alpha @ np.maximum(lower + alpha * s - mean, 0.0)
                    + beta @ np.maximum(mean - upper + beta * s, 0.0))

        if derivative(eta) >= 0:
            optimum = eta
        elif derivative(1.0) <= 0:
            optimum = 1.0
        else:
            optimum = None
            for left, right in zip(breaks[:-1], breaks[1:]):
                midpoint = (left + right) / 2
                active_lower = mean < lower + alpha * midpoint
                active_upper = mean > upper - beta * midpoint
                slope = 1 + float(alpha[active_lower] @ alpha[active_lower])
                slope += float(beta[active_upper] @ beta[active_upper])
                intercept = -scale + float(alpha[active_lower] @ (lower - mean)[active_lower])
                intercept += float(beta[active_upper] @ (mean - upper)[active_upper])
                root = -intercept / slope
                if left - 1e-13 <= root <= right + 1e-13:
                    optimum = min(right, max(left, root))
                    break
            if optimum is None:
                raise ArithmeticError("Projection breakpoint root was not found")
        projected_mean = np.clip(mean, lower + alpha * optimum, upper - beta * optimum)
        return np.append(projected_mean, optimum)

    def objective_gradient(self, point, degree=None):
        mean, scale = point[:-1], float(point[-1])
        if scale <= 0:
            raise ValueError("Scale must be positive")
        nodes = self.cubature_nodes(degree)
        cost = (float(mean @ mean) + scale**2 * self.v) / (2 * self.tau**2)
        cost += -self.d * math.log(scale) + self.prior_constant
        gradient_mean = mean / self.tau**2
        gradient_scale = scale * self.v / self.tau**2 - self.d / scale
        if len(self.y):
            offsets = self.X @ nodes.T
            scores = (self.X @ mean)[:, None] + scale * offsets
            margins = -self.y[:, None] * scores
            probabilities = sigmoid(margins)
            cost += float(np.logaddexp(0.0, margins).mean(axis=1).sum())
            gradient_mean = gradient_mean - self.X.T @ (self.y * probabilities.mean(axis=1))
            gradient_scale -= float((self.y[:, None] * offsets * probabilities).mean(axis=1).sum())
        return cost, np.append(gradient_mean, gradient_scale)

    def fit_component(self, index, start=None, max_iterations=20000):
        if start is None:
            start = np.append(self.center(index), 1.0)
        point = self.project(start, index)
        mu = min(1 / self.tau**2, self.d + self.v / self.tau**2)
        lipschitz = max(mu, 1.0)
        cost, gradient = self.objective_gradient(point)
        for iteration in range(1, max_iterations + 1):
            trial_L = max(mu, lipschitz * 0.8)
            for _ in range(100):
                new_point = self.project(point - gradient / trial_L, index)
                new_cost, new_gradient = self.objective_gradient(new_point)
                step = new_point - point
                majorizer = cost + float(gradient @ step) + trial_L * float(step @ step) / 2
                if new_cost <= majorizer + 1e-13 * (1 + abs(cost)):
                    break
                trial_L *= 2
            else:
                raise ArithmeticError("Backtracking did not find an acceptable step")
            # The projection supplies a normal vector at new_point. Adding it
            # to the new gradient gives a constrained subgradient residual.
            residual = new_gradient - gradient + trial_L * (point - new_point)
            gap = float(residual @ residual) / (2 * mu)
            point, cost, gradient, lipschitz = new_point, new_cost, new_gradient, trial_L
            if gap <= self.tolerance:
                s2 = float((self.X * self.X).sum())
                return Fit(point[:-1].copy(), float(point[-1]), cost, gap,
                           self.C**2 * s2 / (8 * self.degree), iteration, self.degree)
        raise RuntimeError(f"Component {index} did not meet its requested optimization tolerance")

    def refit(self):
        previous = self.fits
        self.fits = [self.fit_component(i, None if not previous else
                     np.append(previous[i].mean, previous[i].scale)) for i in range(self.M)]
        costs = np.asarray([fit.cost_upper for fit in self.fits])
        unnormalized = np.exp(-(costs - costs.min()))
        self.weights = unnormalized / unnormalized.sum()

    def predict(self, x, degree=None):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.d,):
            raise ValueError("Feature dimension mismatch")
        nodes = self.cubature_nodes(degree)
        component_p = np.asarray([sigmoid(x @ fit.mean + fit.scale * (nodes @ x)).mean()
                                  for fit in self.fits])
        return float(self.weights @ component_p)

    def observe(self, x, y):
        if y not in (-1, 1):
            raise ValueError("Labels must be -1 or +1")
        x = np.asarray(x, dtype=float)
        if x.shape != (self.d,):
            raise ValueError("Feature dimension mismatch")
        self.X = np.vstack((self.X, x))
        self.y = np.append(self.y, float(y))
        self.refit()

    def vertices(self, fit):
        u = np.vstack((np.zeros(self.d), np.eye(self.d))) - 1 / (self.d + 1)
        return fit.mean + fit.scale * u * self.h

    def regret_bound(self, R, prediction_and_fit_allowance=0.0):
        """The proved ideal-fit bound, plus an explicitly supplied error allowance.

        Practical degree/tolerance defaults do NOT automatically give allowance
        one. Use the theorem schedules or accumulate their actual bounds.
        """
        a = float(sigmoid(self.C * R * math.sqrt(self.d))
                  * sigmoid(-self.C * R * math.sqrt(self.d)))
        gram = self.X.T @ self.X
        s2 = float(np.trace(gram))
        h1 = self.C / self.M
        K = self.C**2 / ((self.d + 1) * (self.d + 2)) + h1**2 / 4
        V = self.v + h1**2 / 4
        A = K * s2 / 8 + V / (2 * self.tau**2)
        _, logdet = np.linalg.slogdet(np.eye(self.d) + a * self.tau**2 * gram)
        return (self.prior_constant + self.B**2 / (2 * self.tau**2)
                + self.d / 2 * math.log(4 + 2 * A / self.d) + self.d / 2
                + float(logdet) / (2 * a) + prediction_and_fit_allowance)


def theorem_degrees(n, d, B, tau, R, s2):
    """The conservative finite theorem's fit budget and positive cubature degrees."""
    C = 2 * B
    D0 = max(1.0, math.sqrt(2) * (d + 1) * tau * R)
    epsilon = 1 / (16 * D0**2 * (n + 1)**4)
    fit_degree = max(1, math.ceil(C**2 * s2 / (4 * epsilon)))
    pmin = float(sigmoid(-C * R * math.sqrt(d)))
    t = n + 1
    prediction_degree = max(1, math.ceil(C**2 * R**2 * (t + 1)**2 / (2 * pmin)))
    return {"epsilon": epsilon, "optimization_tolerance": epsilon / 2,
            "fit_degree": fit_degree, "prediction_degree": prediction_degree,
            "fit_nodes": math.comb(fit_degree + d, d),
            "prediction_nodes": math.comb(prediction_degree + d, d)}
