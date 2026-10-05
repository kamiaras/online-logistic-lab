"""Probability-averaging algorithm from SOFT_BAYES_COMPLETE.md, Sections 1--5.

The default is the prescribed schedule: m=T independent fresh Gaussian-prior
starts on every nonsymmetric round, N exact-real proximal moves per start, and
add-one probability smoothing.  ``mode='practical'`` explicitly caps samples and
transitions; it does not inherit the manuscript's regret guarantee.  Neither
normal rejection nor conditional rejection has a trial cap.  This is float64
experimental code, not a finite-bit implementation of the exact-real theorem.

The optional instrumented global operation cap in Section 10 is not implemented.
Preflight work/memory limits refuse a run before allocation; they never alter a
transition.  Event counters and dense-arithmetic estimates are not exact counts
of the manuscript's charged primitive operations.
"""
from time import perf_counter
import math
import sys

import numpy as np

from .common import Metrics
from .proximal import _positive_integer, logistic_bregman


class PrescribedBudgetExceeded(ValueError):
    """An allocation-free resource check refused the requested schedule."""

    def __init__(self, details):
        self.details = details
        super().__init__("Soft-Bayes preflight declined: "
                         + "; ".join(details["budget_reasons"]))


def _open_uniform(rng, metrics=None):
    while True:
        if metrics is not None:
            metrics.inc("uniform_draws")
        u = float(rng.random())
        if 0. < u < 1.:
            return u
        if not math.isfinite(u) or u < 0. or u > 1.:
            raise FloatingPointError("Uniform generator returned an invalid value")
        if metrics is not None:
            metrics.inc("uniform_endpoint_redraws")


def normal_standard(rng, metrics=None):
    """Section 3's scalar rejection normal; only independent uniforms are used."""
    while True:
        sign = -1. if _open_uniform(rng, metrics) < .5 else 1.
        q = -math.log(_open_uniform(rng, metrics))
        u = _open_uniform(rng, metrics)
        if metrics is not None:
            metrics.inc("normal_proposals")
        if u <= math.exp(-.5*(q-1.)**2):
            if metrics is not None:
                metrics.inc("normal_draws")
            return sign*q
        if metrics is not None:
            metrics.inc("normal_rejections")


def normal_vector(d, rng, metrics=None):
    """Independent Section 3 calls, processed coordinate by coordinate."""
    out = np.empty(d)
    for j in range(d):
        out[j] = normal_standard(rng, metrics)
    return out


def _log1pexp(z):
    if z > 0.:
        return z+math.log1p(math.exp(-z))
    return math.log1p(math.exp(z))


def _finite_exp(z):
    return math.exp(z) if z < math.log(sys.float_info.max) else None


def _mixing_count(H, log_chi_over_lambda):
    """ceil(H/log1p(lambda/chi)); retain logs if Python loop count is too big."""
    if H == 0.:
        return 0, None
    log_ratio = -log_chi_over_lambda
    # Avoid underflow in lambda/chi.  In this branch log(log1p(r))=log(r)
    # to float64 accuracy, while the log count remains meaningful.
    if log_ratio < -35.:
        log_denominator = log_ratio
    else:
        log_denominator = math.log(math.log1p(math.exp(log_ratio)))
    log_N = math.log(H)-log_denominator
    if log_N >= math.log(sys.maxsize):
        return None, log_N
    ratio = math.exp(log_ratio)
    return math.ceil(H/math.log1p(ratio)), log_N


def _parameters(d, B, T, R):
    neutral = B == 0. or R == 0.
    if neutral:
        return dict(m=T, lam=None, kappa=0., H=0., A=0., log_A=None,
                    J=0, N_max=0, log_N_max=None,
                    expected_charged_operations_bound=4*T+16,
                    log_expected_charged_operations_bound=math.log(4*T+16),
                    theoretical_cells=16, neutral=True)
    if B:
        B2 = B*B
        if not math.isfinite(B2) or B2 == 0.:
            raise ValueError("B squared must be positive and representable")
        lam = 1./B2
        if not math.isfinite(lam) or lam <= 0.:
            raise ValueError("The prior precision must be positive and representable")
    else:
        lam = None
    log_kappa = 2.*(math.log(B)+math.log(R))
    kappa = _finite_exp(log_kappa)
    if kappa is None or kappa == 0.:
        raise ValueError("(BR) squared must be positive and representable")
    log_A = (log_kappa+math.log(T)+math.log(T-1)-math.log(2.)
             if T > 1 else -math.inf)
    H = _log1pexp(math.log(T+1)+log_A)
    log_chi_ratio = (_log1pexp(log_kappa+math.log(T-1)-math.log(2.))
                     if T > 1 else 0.)
    N_max, log_N_max = _mixing_count(H, log_chi_ratio)
    J = math.ceil(math.log(max(1., (T-1)/2.))/(2.*math.log(2.)))
    log_first = math.log(1000.)+math.log(d)+2.*math.log(T)
    if T == 1:
        log_C = log_first
    else:
        log_triangle = math.log(T)+math.log(T-1)-math.log(2.)
        log_term1 = math.log1p(1.5*H)+log_triangle
        log_term2 = (math.log(H)+log_kappa+math.log(T)+math.log(T-1)
                     +math.log(2*T-1)-math.log(12.))
        log_bracket = max(log_term1, log_term2)+math.log1p(
            math.exp(-abs(log_term1-log_term2)))
        log_second = (math.log(320.)+math.log(T)+math.log(d)
                      +math.log(J+2)+log_bracket)
        log_C = max(log_first, log_second)+math.log1p(
            math.exp(-abs(log_first-log_second)))
    return dict(m=T, lam=lam, kappa=kappa, H=H,
                A=0. if T == 1 else _finite_exp(log_A),
                log_A=None if T == 1 else log_A, J=J,
                N_max=N_max, log_N_max=log_N_max,
                expected_charged_operations_bound=_finite_exp(log_C),
                log_expected_charged_operations_bound=log_C,
                theoretical_cells=T*d+16*d+128, neutral=False)


class _RawHistory:
    """Raw features only; the final block never allocates past configured T."""

    def __init__(self, d, T, block_size):
        self.d, self.T, self.block_size = d, T, block_size
        self.n, self.blocks = 0, []

    def append(self, x):
        if self.n >= self.T:
            raise ValueError("Configured horizon exceeded")
        row = self.n % self.block_size
        if row == 0:
            self.blocks.append(np.empty((min(self.block_size, self.T-self.n), self.d)))
        self.blocks[-1][row] = x
        self.n += 1

    def batches(self):
        remaining = self.n
        for block in self.blocks:
            count = min(remaining, len(block))
            yield block[:count]
            remaining -= count

    @property
    def nbytes(self):
        return sum(block.nbytes for block in self.blocks)


class SoftBayes:
    """Online add-one-smoothed mean of posterior probabilities.

    ``predict(x)`` returns log(p(+1)/p(-1)); call ``observe(x,y)`` after it.
    Repeating a prediction before its observation returns the pending result.
    Use ``predict_proba(x)`` for an array in [-1,+1] label order.

    Prescribed preflight defaults to at most 10 million worst-case transitions,
    a refusal threshold that can be raised explicitly with ``max_total_moves``
    or disabled with ``max_total_moves=None`` for externally supervised probes.
    Practical mode caps only m and N with ``samples=16, transitions=1`` defaults;
    its fresh starts, probability averaging, smoothing, and conditional rejection
    are otherwise identical.  Practical smoothing uses its actual sample count.
    """

    @staticmethod
    def preflight(d, B, T, config=None):
        cfg = dict(config or {})
        if cfg.get("global_operation_cap") is not None:
            raise ValueError("Section 10's exact charged-operation cap is not implemented")
        d, T = _positive_integer("d", d), _positive_integer("T", T)
        B, R = float(B), float(cfg.get("R", 1.))
        if not math.isfinite(B) or B < 0. or not math.isfinite(R) or R < 0.:
            raise ValueError("B and R must be finite and nonnegative")
        mode = cfg.get("mode", "prescribed")
        if mode not in ("prescribed", "practical"):
            raise ValueError("mode must be prescribed or practical")
        sample_cap = _positive_integer("samples", cfg.get("samples", 16))
        transition_cap = _positive_integer("transitions", cfg.get("transitions", 1))
        block = min(T, _positive_integer("history_block_size",
                                         cfg.get("history_block_size", min(1024, d))))
        p = _parameters(d, B, T, R)
        m = T if mode == "prescribed" else min(T, sample_cap)
        N = p["N_max"]
        if mode == "practical":
            N = min(N, transition_cap) if N is not None else transition_cap
        # Round 1 is always symmetric.  Later symmetry can lower this bound.
        moves = (T-1)*m*N if N is not None else None
        max_moves_setting = cfg.get("max_total_moves", 10_000_000)
        max_moves = (None if max_moves_setting is None else
                     _positive_integer("max_total_moves", max_moves_setting))
        # Conservative NumPy storage/scratch allowance, separate from theorem (5).
        cells = T*d+24*d+12*block+256
        reasons = []
        if moves is None:
            reasons.append("prescribed count exceeds the representable loop range")
        elif (max_moves is not None and (mode == "prescribed" or "max_total_moves" in cfg)
              and moves > max_moves):
            reasons.append(f"worst-case transitions: {moves} > {max_moves}")
        if "max_state_bytes" in cfg:
            limit = _positive_integer("max_state_bytes", cfg["max_state_bytes"])
            if 8*cells > limit:
                reasons.append(f"estimated array and scratch bytes: {8*cells} > {limit}")
        return dict(d=d, B=B, T=T, R=R, mode=mode, prescribed=p,
                    actual_samples=m, actual_max_transitions=N,
                    estimated_total_moves=moves, max_total_moves=max_moves,
                    estimated_state_bytes=8*cells, history_block_size=block,
                    memory_representation="raw feature history plus scalar/vector summaries",
                    feasible=not reasons, budget_reasons=reasons,
                    floating_point_certificate=False,
                    exact_primitive_operation_counter=False,
                    section_10_global_operation_cap_implemented=False)

    def __init__(self, d, B, T, seed, config=None):
        start = perf_counter()
        self.config = dict(config or {})
        self._preflight = self.preflight(d, B, T, self.config)
        if not self._preflight["feasible"]:
            raise PrescribedBudgetExceeded(self._preflight)
        self.d, self.B, self.T = int(d), float(B), int(T)
        self.R, self.mode = self._preflight["R"], self._preflight["mode"]
        self.m = self._preflight["actual_samples"]
        self.lam = self._preflight["prescribed"]["lam"]
        self.H = self._preflight["prescribed"]["H"]
        self.metrics, self.rng = Metrics(), np.random.default_rng(seed)
        self.history = _RawHistory(self.d, self.T, self._preflight["history_block_size"])
        self.a_obs, self.E = np.zeros(self.d), 0.
        self._pending_x = self._pending_logit = self._pending_probabilities = None
        self._last_schedule = None
        self._total_prescribed_moves = 0
        self._peak_state_bytes = self.state_bytes()
        self.metrics.times["initialization"] += perf_counter()-start

    def _fail(self, message):
        self.metrics.inc("numerical_failures")
        raise FloatingPointError(message)

    def _feature(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.d,) or not np.all(np.isfinite(x)):
            raise ValueError("Feature must be a finite d-vector")
        norm = float(np.linalg.norm(x))
        if (not math.isfinite(norm) or (self.R == 0. and np.any(x))
                or norm > self.R*(1.+1e-12)+1e-15):
            raise ValueError("Feature norm exceeds configured R")
        return x

    def _schedule(self):
        if self._preflight["prescribed"]["neutral"]:
            return dict(n=self.history.n, t=self.history.n+1, E=self.E, H=self.H,
                        chi=None, mu=None, N_prescribed=0, log_N_prescribed=None,
                        N_actual=0, samples=self.m)
        chi, mu = self.lam+2.*self.E, 2.*self.lam+2.*self.E
        if not all(math.isfinite(q) and q > 0. for q in (chi, mu)):
            self._fail("Unrepresentable proximal precision")
        N, log_N = _mixing_count(self.H, math.log(chi)-math.log(self.lam))
        if self.mode == "prescribed":
            if N is None:
                self._fail("Prescribed mixing count is not representable")
            actual = N
        else:
            cap = int(self.config.get("transitions", 1))
            actual = min(N, cap) if N is not None else cap
        return dict(n=self.history.n, t=self.history.n+1, E=self.E, H=self.H,
                    chi=chi, mu=mu, sqrt_chi=math.sqrt(chi), inv_sqrt_mu=1./math.sqrt(mu),
                    N_prescribed=N, log_N_prescribed=log_N, N_actual=actual,
                    samples=self.m)

    def _phi_grad(self, v):
        start = perf_counter()
        phi, g = 0., np.zeros(self.d)
        for xb in self.history.batches():
            score = xb@v
            phi += float(np.sum(np.logaddexp(score/2., -score/2.)))
            # h'(s)=sigma(s)-1/2, evaluated through the tail to retain stability.
            tail = np.exp(-np.abs(score))
            derivative = .5*np.sign(score)*(-np.expm1(-np.abs(score)))/(1.+tail)
            g += xb.T@derivative
            self.metrics.inc("gradient_refresh_rows", len(xb))
            self.metrics.inc("estimated_matmul_flops", 4*len(xb)*self.d)
            self.metrics.inc("softplus_elements", len(xb))
            self.metrics.inc("sigmoid_elements", len(xb))
        self.metrics.inc("gradient_refreshes")
        self.metrics.times["gradient_refresh"] += perf_counter()-start
        if not math.isfinite(phi) or not np.all(np.isfinite(g)):
            self._fail("Nonfinite tangent value or gradient")
        return phi, g

    def _full_residual(self, z, v):
        """Equation (16) summed as stable, nonnegative per-row Bregman gaps."""
        start = perf_counter()
        total = 0.
        for xb in self.history.batches():
            total += float(np.sum(logistic_bregman(xb@v, xb@z, self.metrics)))
            self.metrics.inc("residual_rows", len(xb))
            self.metrics.inc("estimated_matmul_flops", 4*len(xb)*self.d)
        self.metrics.inc("full_residual_calls")
        self.metrics.times["full_residual"] += perf_counter()-start
        if not math.isfinite(total) or total < 0.:
            self._fail("Nonfinite or negative conditional rejection residual")
        return total

    def _conditional(self, r, mu):
        """Sections 4--5: hold r fixed and advance v=c only after rejection."""
        r = np.asarray(r, dtype=np.float64)
        if r.shape != (self.d,) or not np.all(np.isfinite(r)) or not math.isfinite(mu) or mu <= 0.:
            self._fail("Invalid conditional Gaussian parameters")
        v, sd = r/mu, 1./math.sqrt(mu)
        trials = 0
        while True:
            _, g = self._phi_grad(v)
            c = (r-g)/mu
            z = c+sd*normal_vector(self.d, self.rng, self.metrics)
            if not np.all(np.isfinite(z)) or not np.all(np.isfinite(c)):
                self._fail("Nonfinite unbounded conditional Gaussian proposal")
            residual = self._full_residual(z, v)
            threshold = -math.log(_open_uniform(self.rng, self.metrics))
            trials += 1
            self.metrics.inc("rqo_proposals")
            # -log(U)>=R is the same test as U<=exp(-R), avoiding underflow.
            if threshold >= residual:
                self.metrics.inc("rqo_accepts")
                if trials == 1:
                    self.metrics.inc("rqo_first_accepts")
                self.metrics.counters["rqo_max_trials"] = max(
                    self.metrics.counters["rqo_max_trials"], trials)
                return z
            self.metrics.inc("rqo_rejections")
            v = c

    def _transition(self, theta, schedule):
        r = (schedule["chi"]*theta+self.a_obs
             +schedule["sqrt_chi"]*normal_vector(self.d, self.rng, self.metrics))
        result = self._conditional(r, schedule["mu"])
        self.metrics.inc("proximal_steps")
        return result

    def predict(self, x):
        start = perf_counter()
        x = self._feature(x)
        if self._pending_x is not None:
            if not np.array_equal(x, self._pending_x):
                raise ValueError("Observe the pending prediction before changing features")
            return self._pending_logit
        if self.history.n >= self.T:
            raise ValueError("Configured horizon exceeded")
        # The current feature is deliberately absent from past-row posterior.
        schedule = self._schedule()
        if (self._preflight["prescribed"]["neutral"] or not np.any(x)
                or not np.any(self.a_obs)):
            logit, probabilities = 0., np.array([.5, .5])
            self.metrics.inc("symmetry_skips")
            schedule = {**schedule, "skipped": True, "N_actual": 0}
        else:
            plus, minus, plus_correction, minus_correction = 0., 0., 0., 0.
            for _ in range(self.m):
                theta = self.B*normal_vector(self.d, self.rng, self.metrics)
                self.metrics.inc("prior_initializations")
                if not np.all(np.isfinite(theta)):
                    self._fail("Nonfinite fresh Gaussian prior sample")
                for _ in range(schedule["N_actual"]):
                    theta = self._transition(theta, schedule)
                score = float(x@theta)
                if not math.isfinite(score):
                    self._fail("Nonfinite sampled predictive score")
                tail = math.exp(-abs(score))
                positive, negative = ((1./(1.+tail), tail/(1.+tail)) if score >= 0.
                                      else (tail/(1.+tail), 1./(1.+tail)))
                # Compensated streaming sums; no population or score history.
                adjusted = positive-plus_correction
                updated = plus+adjusted
                plus_correction = (updated-plus)-adjusted
                plus = updated
                adjusted = negative-minus_correction
                updated = minus+adjusted
                minus_correction = (updated-minus)-adjusted
                minus = updated
                self.metrics.inc("posterior_samples")
                self.metrics.inc("sigmoid_elements", 2)
            logit = math.log1p(plus)-math.log1p(minus)
            probabilities = np.array([(1.+minus)/(self.m+2.), (1.+plus)/(self.m+2.)])
            N = schedule["N_prescribed"]
            if N is None or self._total_prescribed_moves is None:
                self._total_prescribed_moves = None
            else:
                self._total_prescribed_moves += self.T*N
            schedule = {**schedule, "skipped": False}
        if not math.isfinite(logit) or not np.all(np.isfinite(probabilities)):
            self._fail("Nonfinite smoothed predictive probability")
        self._pending_x, self._pending_logit = x.copy(), logit
        self._pending_probabilities = probabilities
        self._last_schedule = schedule
        self._peak_state_bytes = max(self._peak_state_bytes, self.state_bytes())
        self.metrics.inc("predictions")
        self.metrics.times["predict"] += perf_counter()-start
        return logit

    def predict_proba(self, x):
        self.predict(x)
        return self._pending_probabilities.copy()

    def observe(self, x, y):
        start = perf_counter()
        x = self._feature(x)
        if isinstance(y, (bool, np.bool_)) or y not in (-1, 1):
            raise ValueError("Label must be -1 or +1")
        if self._pending_x is None or not np.array_equal(x, self._pending_x):
            raise ValueError("Observation must match the pending prediction")
        if self._preflight["prescribed"]["neutral"]:
            # No feature history or sufficient-statistic work is needed here.
            self.history.n += 1
        else:
            new_E = self.E+float(x@x)/4.
            new_a = self.a_obs+.5*int(y)*x
            if not math.isfinite(new_E) or not np.all(np.isfinite(new_a)):
                self._fail("Nonfinite observed-label summary or feature energy")
            self.history.append(x)
            self.E, self.a_obs = new_E, new_a
            self.metrics.inc("history_appends")
        self._pending_x = self._pending_logit = self._pending_probabilities = None
        self._peak_state_bytes = max(self._peak_state_bytes, self.state_bytes())
        self.metrics.inc("observations")
        self.metrics.times["observe"] += perf_counter()-start

    def state_bytes(self):
        return int(self.history.nbytes+self.a_obs.nbytes
                   +(0 if self._pending_x is None else self._pending_x.nbytes)
                   +(0 if self._pending_probabilities is None else self._pending_probabilities.nbytes))

    def diagnostics(self):
        proposals = self.metrics.counters["rqo_proposals"]
        moves = self.metrics.counters["proximal_steps"]
        return dict(mode=self.mode, samples=self.m, prescribed_samples=self.T,
                    last_schedule=self._last_schedule,
                    prescribed_total_moves_on_sampled_rounds=self._total_prescribed_moves,
                    actual_total_moves=moves,
                    rqo_acceptance_rate=(self.metrics.counters["rqo_accepts"]/proposals
                                         if proposals else None),
                    rqo_first_acceptance_rate=(self.metrics.counters["rqo_first_accepts"]/moves
                                               if moves else None),
                    state_bytes=self.state_bytes(), peak_persistent_state_bytes=self._peak_state_bytes,
                    history_length=self.history.n, history_representation="raw_features_only",
                    initialization="fresh independent unbounded Gaussian prior per sample",
                    prediction="add-one smoothed average of posterior probabilities",
                    conditional_rejection_cap=None, normal_rejection_cap=None,
                    floating_point_certificate=False,
                    section_10_global_operation_cap_implemented=False,
                    numerical_failures=self.metrics.counters["numerical_failures"],
                    preflight=self._preflight, metrics=self.metrics.as_dict(),
                    certificate="Exact-real prescribed algorithm only; float64 and practical caps have no inherited regret theorem")
