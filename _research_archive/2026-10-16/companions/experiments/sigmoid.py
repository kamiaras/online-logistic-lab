"""Empirical expected-loss sigmoid/MALA baseline.

The prediction guard follows PUBLISHED_SIGMOID_COMPARISON.md, but the default
population and transition counts are practical budgets. The MALA theorem's
unreported universal constants are not assigned numerical values here. No
theoretical mixing or regret certification is claimed for these settings.
"""

from time import perf_counter

import numpy as np

from .common import History, Metrics, potential, sigmoid


def mala_log_acceptance(current, proposed, value_current, gradient_current,
                        value_proposed, gradient_proposed, step_size):
    """Log MH ratio for proposal N(x - h grad V(x), 2h I).

    This is V(x)-V(z)+log q(x|z)-log q(z|x). Expanding the two
    squared proposal residuals avoids dividing their difference by a tiny h.
    The last array dimension is the coordinate dimension.
    """
    delta = np.asarray(proposed) - np.asarray(current)
    gradient_sum = np.asarray(gradient_current) + np.asarray(gradient_proposed)
    gradient_difference = (np.asarray(gradient_current)
                           - np.asarray(gradient_proposed))
    return (np.asarray(value_current) - np.asarray(value_proposed)
            + 0.5 * np.sum(delta * gradient_sum, axis=-1)
            + 0.25 * step_size
            * np.sum(gradient_difference * gradient_sum, axis=-1))


def _positive_int(name, value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if integer != value or integer < 1:
        raise ValueError(f"{name} must be a positive integer")
    return integer


class SigmoidMALA:
    """Independent persistent particles with a fixed per-round MALA budget.

    ``transitions`` counts lazy Markov steps *per trajectory*, totaled across
    all bridges. It is not a budget per bridge. The quotient/remainder split
    assigns any extra steps to the first bridges. ``step_multiplier`` sets
    h = multiplier / (sqrt(d) * (B^-2 + sum_past ||x||^2 / 4)).

    ``state_bytes`` counts owned persistent NumPy buffers, including allocated
    history capacity and a pending input copy, but not Python-object overhead
    or transient evaluation buffers. The harness measures process peak memory.
    """

    _DEFAULTS = {
        "particles": 16,
        "transitions": 4,
        "bridges": 1,
        "step_multiplier": 0.5,
        "lazy": True,
        "history_block_size": 1024,
    }

    def __init__(self, d, B, T, seed, config=None):
        started = perf_counter()
        self.d = _positive_int("d", d)
        self.T = _positive_int("T", T)
        self.B = float(B)
        if not np.isfinite(self.B) or self.B <= 0:
            raise ValueError("B must be finite and positive")
        b_squared = self.B * self.B
        if not np.isfinite(b_squared) or b_squared == 0:
            raise ValueError("B squared must be representable in float64")
        self._inverse_b_squared = 1.0 / b_squared
        if not np.isfinite(self._inverse_b_squared):
            raise ValueError("B^-2 must be representable in float64")

        config = {} if config is None else dict(config)
        unknown = set(config) - set(self._DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown sigmoid configuration keys: {sorted(unknown)}")
        self.config = {**self._DEFAULTS, **config}
        for name in ("particles", "transitions", "bridges", "history_block_size"):
            self.config[name] = _positive_int(name, self.config[name])
        self.n_particles = self.config["particles"]
        self.transitions = self.config["transitions"]
        self.bridges = self.config["bridges"]
        if self.transitions < self.bridges:
            raise ValueError("transitions must be at least bridges: it is the total budget")
        multiplier = float(self.config["step_multiplier"])
        if not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("step_multiplier must be finite and positive")
        self.config["step_multiplier"] = multiplier
        if not isinstance(self.config["lazy"], (bool, np.bool_)):
            raise ValueError("lazy must be a boolean")
        self.config["lazy"] = bool(self.config["lazy"])
        quotient, remainder = divmod(self.transitions, self.bridges)
        self._bridge_steps = tuple(quotient + (j < remainder)
                                   for j in range(self.bridges))

        self.metrics = Metrics()
        self.rng = np.random.default_rng(seed)
        self.history = History(self.d, self.config["history_block_size"])
        self.theta = self.B * self.rng.standard_normal((self.n_particles, self.d))
        if not np.all(np.isfinite(self.theta)):
            raise FloatingPointError("Nonfinite Gaussian prior draw")
        self.metrics.inc("normal_draws", self.n_particles * self.d)
        self.metrics.inc("prior_particles", self.n_particles)
        self._squared_feature_sum = 0.0
        self._pending_x = None
        self._last_step_size = None
        self._min_step_size = None
        self._max_step_size = None
        self._last_logit = None
        self._squared_displacement_sum = 0.0
        self._max_squared_displacement = 0.0
        self._displacement_squares_finite = True
        self.metrics.times["initialization"] += perf_counter() - started

    def _input(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.d,) or not np.all(np.isfinite(x)):
            raise ValueError(f"x must be a finite vector with shape ({self.d},)")
        return x

    def _step_size(self):
        smoothness = self._inverse_b_squared + self._squared_feature_sum / 4.0
        h = self.config["step_multiplier"] / (np.sqrt(self.d) * smoothness)
        if not np.isfinite(h) or h <= 0 or not np.isfinite(2.0 * h):
            raise FloatingPointError("MALA step size is not representable in float64")
        return h

    def _mala_step(self, step_size, last_weight=1.0):
        """One independent lazy step for every particle, at a fixed target."""
        self.metrics.inc("mala_trials", self.n_particles)
        if self.config["lazy"]:
            active = self.rng.random(self.n_particles) >= 0.5
            self.metrics.inc("uniform_draws", self.n_particles)
            self.metrics.inc("lazy_skips", np.count_nonzero(~active))
        else:
            active = np.ones(self.n_particles, dtype=bool)
        indices = np.flatnonzero(active)
        count = len(indices)
        if count == 0:
            return

        self.metrics.inc("mala_proposals", count)
        current = self.theta[indices]
        value_current, gradient_current = potential(
            current, self.B, self.history, metrics=self.metrics,
            last_weight=last_weight)
        noise = self.rng.standard_normal(current.shape)
        self.metrics.inc("normal_draws", count * self.d)
        proposed = current - step_size * gradient_current + np.sqrt(2.0 * step_size) * noise
        if not np.all(np.isfinite(proposed)):
            raise FloatingPointError("Nonfinite MALA proposal")
        value_proposed, gradient_proposed = potential(
            proposed, self.B, self.history, metrics=self.metrics,
            last_weight=last_weight)
        self.metrics.inc("gradient_particle_evaluations", 2 * count)
        ratio = mala_log_acceptance(
            current, proposed, value_current, gradient_current,
            value_proposed, gradient_proposed, step_size)
        if not np.all(np.isfinite(ratio)):
            raise FloatingPointError("Nonfinite MALA log acceptance ratio")
        # 1-U is in (0,1], avoiding log(0) from a possible floating-point U=0.
        log_uniform = np.log1p(-self.rng.random(count))
        self.metrics.inc("uniform_draws", count)
        accepted = log_uniform <= np.minimum(0.0, ratio)
        # In float64, a proposal can round back to its starting point even if
        # it is accepted. High acceptance alone does not demonstrate movement
        # or mixing. Rejections and lazy steps contribute zero displacement.
        movement = proposed[accepted] - current[accepted]
        if len(movement):
            zero_moves = np.count_nonzero(np.all(movement == 0.0, axis=1))
            self.metrics.inc("accepted_zero_displacement", zero_moves)
            self.metrics.inc("accepted_nonzero_displacement", len(movement) - zero_moves)
            with np.errstate(over="ignore", invalid="ignore"):
                squares = np.sum(movement * movement, axis=1)
                total = self._squared_displacement_sum + float(np.sum(squares))
            if np.all(np.isfinite(squares)) and np.isfinite(total):
                self._squared_displacement_sum = total
                self._max_squared_displacement = max(
                    self._max_squared_displacement, float(np.max(squares)))
            else:
                # Preserve a JSON-safe explicit failure flag instead of
                # silently emitting an infinite diagnostic at extreme scales.
                self._displacement_squares_finite = False
        self.theta[indices[accepted]] = proposed[accepted]
        self.metrics.inc("mala_accepted", np.count_nonzero(accepted))
        self.metrics.inc("mala_rejected", np.count_nonzero(~accepted))

    def predict(self, x):
        """Return a finite guarded logit; refresh only against observed history."""
        if self._pending_x is not None:
            raise RuntimeError("observe must follow predict before another prediction")
        if self.history.n >= self.T:
            raise RuntimeError("The configured horizon has already been observed")
        x = self._input(x)
        refresh_start = perf_counter()
        if self.history.n:
            step_size = self._step_size()
            self._last_step_size = step_size
            self._min_step_size = (step_size if self._min_step_size is None
                                   else min(self._min_step_size, step_size))
            self._max_step_size = (step_size if self._max_step_size is None
                                   else max(self._max_step_size, step_size))
            for j, count in enumerate(self._bridge_steps, start=1):
                weight = j / self.bridges
                self.metrics.inc("bridge_refreshes", self.n_particles)
                for _ in range(count):
                    self._mala_step(step_size, last_weight=weight)
            self.metrics.inc("refresh_rounds")
        self.metrics.times["refresh"] += perf_counter() - refresh_start

        prediction_start = perf_counter()
        score = self.theta @ x
        if not np.all(np.isfinite(score)):
            raise FloatingPointError("Nonfinite prediction score")
        positive = float(np.sum(sigmoid(score)))
        negative = float(np.sum(sigmoid(-score)))
        logit = float(np.log1p(positive) - np.log1p(negative))
        self.metrics.inc("predictions")
        self.metrics.inc("sigmoid_elements", 2 * self.n_particles)
        self.metrics.inc("estimated_matmul_flops", 2 * self.n_particles * self.d)
        self._last_logit = logit
        self._pending_x = x.copy()
        self.metrics.times["prediction"] += perf_counter() - prediction_start
        return logit

    def observe(self, x, y):
        """Append the revealed example, with no posterior update at this point."""
        started = perf_counter()
        if self._pending_x is None:
            raise RuntimeError("predict must precede observe")
        x = self._input(x)
        if not np.array_equal(x, self._pending_x):
            raise ValueError("Observed x differs from the pending prediction input")
        if not np.isscalar(y) or y not in (-1, 1):
            raise ValueError("y must be -1 or +1")
        squared_sum = self._squared_feature_sum + float(x @ x)
        if not np.isfinite(squared_sum):
            raise FloatingPointError("Feature squared-norm sum overflowed")
        self.history.append(x, int(y))
        self._squared_feature_sum = squared_sum
        self._pending_x = None
        self.metrics.inc("observations")
        self.metrics.times["observe"] += perf_counter() - started

    def state_bytes(self):
        pending = 0 if self._pending_x is None else self._pending_x.nbytes
        return int(self.theta.nbytes + self.history.nbytes + pending)

    def diagnostics(self):
        proposals = self.metrics.counters.get("mala_proposals", 0)
        accepted = self.metrics.counters.get("mala_accepted", 0)
        trials = self.metrics.counters.get("mala_trials", 0)
        mean_square = (self._squared_displacement_sum / trials
                       if trials and self._displacement_squares_finite else None)
        return {
            "algorithm": "expected_loss_sigmoid_mala",
            "configuration": dict(self.config),
            "schedule_certified": False,
            "schedule_note": "Empirical fixed budget; universal MALA theorem constants are unspecified.",
            "step_size_rule": "step_multiplier / (sqrt(d) * (B^-2 + sum_past_norm2/4))",
            "observed_rounds": int(self.history.n),
            "pending_prediction": self._pending_x is not None,
            "bridge_steps_per_trajectory": list(self._bridge_steps),
            "total_steps_per_trajectory_per_refresh": int(sum(self._bridge_steps)),
            "last_step_size": self._last_step_size,
            "minimum_step_size": self._min_step_size,
            "maximum_step_size": self._max_step_size,
            "last_logit": self._last_logit,
            "acceptance_rate": accepted / proposals if proposals else None,
            "accepted_zero_displacement": self.metrics.counters.get("accepted_zero_displacement", 0),
            "mean_squared_displacement_per_trial": mean_square,
            "max_squared_displacement": (self._max_squared_displacement
                                         if trials and self._displacement_squares_finite else None),
            "displacement_squares_finite": self._displacement_squares_finite,
            "displacement_note": "Actual float64 moves; mean includes rejected and lazy zero moves. Acceptance is not a mixing diagnostic.",
            "state_bytes": self.state_bytes(),
            "state_bytes_scope": "Persistent owned ndarray buffers; includes history allocation capacity.",
            "potential_time_is_within_refresh": True,
            **self.metrics.as_dict(),
        }
