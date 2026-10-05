"""Persistent antithetic midpoint pairs with guarded Bayesian log odds.

Reference for SHARED_STATISTICAL_REPAIR_RESEARCH.md. The exact-real schedule is
gap-aware and uses even M=2 max(1,ceil(2 Gamma)), with no score-range floor. The
shared tangent, unconstrained proposal center and rejection-only repair are
reused unchanged. Acceptance uses nonnegative full-history residual blocks;
there is no weighted history index. Float64 and practical caps are not covered
by the mathematical regret guarantee. No production benchmark uses this module.
"""
from time import perf_counter
import math

import numpy as np

from .common import Metrics
from .proximal import logistic_bregman, _positive_integer
from .shared_tangent import (
    SharedTangentLogOdds, PrescribedBudgetExceeded, _parameters,
    _logaddexp_scalar, _ceil_from_log, _mixing_counts,
)
from .shared_tangent_primitives import normal_interval, normal_standard


class FeatureHistory:
    """Raw feature blocks only; no labels, row norms or categorical prefixes."""

    def __init__(self, d, block_size=128, metrics=None):
        self.d = _positive_integer("d", d)
        self.block_size = _positive_integer("block_size", block_size)
        self.metrics, self.n, self.blocks = metrics, 0, []

    def append(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.d,) or not np.all(np.isfinite(x)):
            raise ValueError("History feature must be one finite d-vector")
        pos = self.n % self.block_size
        if pos == 0:
            block = np.empty((self.block_size, self.d))
            block[0] = x
            self.blocks.append(block)
        else:
            self.blocks[-1][pos] = x
        self.n += 1
        if self.metrics is not None:
            self.metrics.inc("raw_history_appends")
            self.metrics.inc("raw_history_coordinates", self.d)

    def batches(self):
        remaining = self.n
        for block in self.blocks:
            count = min(remaining, self.block_size)
            if count <= 0:
                break
            yield block[:count]
            remaining -= count

    @property
    def nbytes(self):
        return sum(block.nbytes for block in self.blocks)


def _persistent_parameters(d, B, T, R):
    old = _parameters(d, B, T, R)
    if not math.isfinite(2.*old["lam"]):
        raise ValueError("Twice the prior precision must be representable")
    log_Gamma = old["log_Gamma"]
    log_pairs = max(0., math.log(2.)+log_Gamma) if log_Gamma is not None else 0.
    pairs = _ceil_from_log(log_pairs)
    m = 2*pairs if pairs is not None else None
    log_m = math.log(m) if m is not None else math.log(2.)+log_pairs
    log_epsilon = (-math.log(4.)-math.log1p(old["score_bound"])
                   -math.log(T)-math.log(T+1))
    log_kappa = math.log(old["kappa"]) if old["kappa"] else -math.inf
    log_Kmax = log_kappa+2.*math.log(T)-math.log(2.)
    Hmax = _logaddexp_scalar(0., log_Kmax)-math.log(2.)-2.*log_epsilon
    # N_t <= 1+Hmax+Hmax*max(d,2)*kappa*T/8; use logs before allocation.
    log_Nmax = _logaddexp_scalar(
        math.log1p(Hmax),
        math.log(Hmax)+math.log(max(d, 2))+log_kappa+math.log(T)-math.log(8.))
    return dict(m=m, log_m=log_m, log_Gamma=log_Gamma,
                a=old["a"], log_a=old["log_a"], lam=old["lam"],
                kappa=old["kappa"], score_bound=old["score_bound"],
                log_epsilon=log_epsilon, epsilon=math.exp(log_epsilon),
                H_max=Hmax, N_max=_ceil_from_log(log_Nmax),
                log_N_max=log_Nmax,
                global_curvature_underflow=old["global_curvature_underflow"])


class PersistentLogOdds(SharedTangentLogOdds):
    """Opposite midpoint lanes start at theta and -theta, then move separately.

    Pairs are independent in exact arithmetic; lanes within a pair need not be.
    predict returns a guarded logit. Practical defaults cap lanes at 256 and
    each lane's prescribed warm transitions at one. Skips always use prescribed
    M. Conditional rejection and repair have no practical cap. Prescribed mode
    refuses an infeasible allocation-free preflight instead of reducing counts.
    A numerical/allocation failure is terminal: partially updated history and
    lanes are not rolled back. State-byte estimates count ndarray storage only,
    including padded history blocks, not whole-process or peak temporary memory.
    """

    @staticmethod
    def preflight(d, B, T, config=None):
        cfg = dict(config or {})
        allowed = {"R", "mode", "lanes", "transitions", "history_block_size",
                   "max_total_moves", "max_state_bytes"}
        unknown = set(cfg)-allowed
        if unknown:
            raise ValueError("Unknown persistent-lane settings: "+", ".join(sorted(unknown)))
        d, T = _positive_integer("d", d), _positive_integer("T", T)
        B, R = float(B), float(cfg.get("R", 1.))
        if not math.isfinite(B) or B <= 0. or not math.isfinite(R) or R < 0.:
            raise ValueError("B must be positive and R nonnegative, both finite")
        mode = cfg.get("mode", "practical")
        if mode not in ("practical", "prescribed"):
            raise ValueError("mode must be practical or prescribed")
        default_block = min(1024, max(16, 1 << max(0, (1048576//d).bit_length()-1)))
        lane_cap = _positive_integer("lanes", cfg.get("lanes", 256))
        if lane_cap % 2:
            raise ValueError("lanes must be even for antithetic midpoint pairing")
        move_cap = _positive_integer("transitions", cfg.get("transitions", 1))
        block = _positive_integer("history_block_size", cfg.get("history_block_size", default_block))
        max_moves = _positive_integer("max_total_moves", cfg.get("max_total_moves", 10_000_000))
        p = _persistent_parameters(d, B, T, R)
        m = p["m"] if mode == "prescribed" else min(p["m"] or lane_cap, lane_cap)
        N = p["N_max"] if mode == "prescribed" else min(p["N_max"] or move_cap, move_cap)
        moves = 0 if R == 0. else (None if m is None or N is None else T*m*N)
        rows = ((T+block-1)//block)*block
        cells = None if m is None else (rows+m)*d+32*d+256
        reasons = []
        if m is None or (mode == "prescribed" and N is None):
            reasons.append("prescribed count exceeds the representable schedule range")
        if ((mode == "prescribed" or "max_total_moves" in cfg)
                and moves is not None and moves > max_moves):
            reasons.append(f"worst-case transitions: {moves} > {max_moves}")
        if cells is not None and cells > np.iinfo(np.intp).max//8:
            reasons.append("persistent array shape exceeds the platform index range")
        if "max_state_bytes" in cfg:
            limit = _positive_integer("max_state_bytes", cfg["max_state_bytes"])
            if cells is None or 8*cells > limit:
                reasons.append("estimated persistent state exceeds max_state_bytes")
        return dict(mode=mode, d=d, B=B, T=T, R=R, prescribed=p,
                    actual_lanes=m, actual_max_transitions=N,
                    population_rule="2*max(1,ceil(2*Gamma))", guard_factor=4,
                    prior_pairing="opposite_midpoints",
                    history_block_size=block, estimated_total_moves=moves,
                    estimated_state_bytes=None if cells is None else 8*cells,
                    memory_representation="raw feature blocks plus persistent coefficient lanes",
                    max_total_moves=max_moves, feasible=not reasons, budget_reasons=reasons,
                    floating_point_certificate=False)

    def __init__(self, d, B, T, seed, config=None):
        start = perf_counter()
        self.config = dict(config or {})
        self._preflight = self.preflight(d, B, T, self.config)
        if not self._preflight["feasible"]:
            raise PrescribedBudgetExceeded(self._preflight)
        self.d, self.B, self.T = int(d), float(B), int(T)
        self.R, self.mode = self._preflight["R"], self._preflight["mode"]
        self.m = self._preflight["actual_lanes"]
        self.lam, self.bound = self._preflight["prescribed"]["lam"], 2.*self.B
        self.metrics, self.rng = Metrics(), np.random.default_rng(seed)
        self.history = FeatureHistory(self.d, self._preflight["history_block_size"], self.metrics)
        self.v, self.g, self.a_obs = (np.zeros(self.d) for _ in range(3))
        self.phi = self.E = self.E1 = 0.
        self.curvature_diagonal, self.curvature_low = np.zeros(self.d), 0.
        self.states = None
        self.reference_round, self.reference_gamma, self.gap_norm = 0, self.lam, 0.
        self._pending_x = self._pending_logit = self._last_schedule = None
        self._failed = False
        self._total_prescribed_moves = 0
        self._prescribed_moves_representable = True
        self._peak_state_bytes = self.state_bytes()
        self.metrics.times["initialization"] += perf_counter()-start

    def _append_feature(self, x):
        super()._append_feature(x)
        self.gap_norm += float(np.linalg.norm(x))
        if not math.isfinite(self.gap_norm):
            self._fail("Nonfinite warm-start feature gap")

    def _schedule(self):
        P = max(self.E1, 2.*self.E)
        inveta, mu, gamma = self.lam+P, 2.*self.lam+P, self.lam+self.curvature_low
        if not all(math.isfinite(z) and z > 0. for z in (inveta, mu, gamma)):
            self._fail("Unrepresentable proximal or curvature precision")
        log_K = (2.*math.log(self.gap_norm)-math.log(2.)-math.log(self.reference_gamma)
                 if self.gap_norm else -math.inf)
        H = (_logaddexp_scalar(0., log_K)-math.log(2.)
             -2.*self._preflight["prescribed"]["log_epsilon"])
        N, _, log_N, _ = _mixing_counts(H, gamma, inveta)
        if self.mode == "prescribed":
            if N is None:
                self._fail("Prescribed warm mixing count is not representable")
            actual = N
        else:
            cap = int(self.config.get("transitions", 1))
            actual = min(N if N is not None else cap, cap)
        return dict(t=self.history.n, E=self.E, E1=self.E1, P=P,
                    inveta=inveta, eta=1./inveta, mu=mu, gamma=gamma,
                    reference_round=self.reference_round,
                    reference_gamma=self.reference_gamma, gap_norm=self.gap_norm,
                    K_gap=math.exp(log_K) if log_K < 700. else None,
                    log_K_gap=None if log_K == -math.inf else log_K,
                    H_warm=H, N_prescribed=N, log_N_prescribed=log_N, N_actual=actual)

    def _residual_test(self, z):
        """One exponential threshold; positive partial sums may reject early.

        Accepting requires visiting all history blocks. In exact arithmetic this
        is exactly exp(-sum(row residuals)), irrespective of block size/order.
        The inherited stable float64 Bregman routine diagnoses its roundoff.
        """
        start = perf_counter()
        threshold, total = -math.log(self._uniform()), 0.
        self.metrics.inc("full_tests")
        for xb in self.history.batches():
            residuals = logistic_bregman(xb@self.v, xb@z, self.metrics)
            block_sum = float(np.sum(residuals))
            if not math.isfinite(block_sum) or block_sum < 0.:
                self._fail("Nonfinite or negative residual block")
            total += block_sum
            if not math.isfinite(total):
                self._fail("Nonfinite cumulative residual")
            self.metrics.inc("full_test_rows", len(xb))
            self.metrics.inc("estimated_matmul_flops", 4*len(xb)*self.d)
            if total > threshold:
                self.metrics.inc("full_test_rejections")
                self.metrics.times["full_test"] += perf_counter()-start
                return False
        self.metrics.inc("full_test_accepts")
        self.metrics.times["full_test"] += perf_counter()-start
        return True

    def _initialize_lanes(self):
        states = np.empty((self.m, self.d))
        for j in range(self.m//2):
            theta = self.B*normal_interval(np.full(self.d, -2.), np.full(self.d, 2.),
                                           self.rng, self.metrics)
            states[j] = theta
            states[self.m-1-j] = -theta
        self.states = states
        self.metrics.inc("prior_initializations", self.m//2)
        self.metrics.inc("prior_coordinates", self.m*self.d//2)
        self.metrics.inc("prior_pair_initializations", self.m//2)
        self.metrics.inc("initialized_lanes", self.m)
        self.metrics.inc("prior_cloud_initializations")

    def predict(self, x):
        if self._failed:
            raise RuntimeError("This sampler failed during an earlier prediction; create a new instance")
        try:
            return self._predict_unchecked(x)
        except (MemoryError, FloatingPointError):
            self._failed = True
            raise

    def _predict_unchecked(self, x):
        start = perf_counter()
        x = self._feature(x)
        if self._pending_x is not None:
            if np.array_equal(x, self._pending_x):
                return self._pending_logit
            raise ValueError("Observe the pending prediction before predicting new features")
        if self.history.n >= self.T:
            raise ValueError("Configured horizon has been exhausted")
        self._append_feature(x)
        schedule = self._schedule()
        norm2, row_l1 = float(x@x), float(np.sum(np.abs(x)))
        obs_norm = float(np.linalg.norm(self.a_obs))
        delta = min(2.*self.B*row_l1, obs_norm*math.sqrt(norm2)/schedule["gamma"])
        b_lower = norm2/(self.lam+4.*self._preflight["prescribed"]["a"]*self.E)
        if delta == 0.:
            skip = True
        elif b_lower == 0.:
            skip = False
        else:
            skip = math.log(delta)-math.log(2.) <= (
                math.log(3./8.)+math.log(b_lower)-self._preflight["prescribed"]["log_m"])
        schedule.update(delta_bound=delta, b_lower=b_lower, skipped=skip,
                        m_actual=self.m, m_prescribed=self._preflight["prescribed"]["m"])
        prescribed_m, prescribed_N = schedule["m_prescribed"], schedule["N_prescribed"]
        planned = (None if prescribed_m is None or prescribed_N is None
                   else prescribed_m*prescribed_N)
        schedule["moves_prescribed_without_skip"] = planned
        schedule["moves_actual"] = 0 if skip else self.m*schedule["N_actual"]
        if skip:
            raw_logit = 0.
            self.metrics.inc("skipped_rounds")
            if delta == 0.:
                self.metrics.inc("exact_symmetry_skips")
        else:
            self.metrics.inc("sampled_rounds")
            if planned is None:
                self._prescribed_moves_representable = False
            else:
                self._total_prescribed_moves += planned
            if self.states is None:
                self._initialize_lanes()
            for j in range(self.m):
                theta = self.states[j].copy()
                tilt = -.5+(j+.5)/self.m
                linear = self.a_obs+tilt*x
                for _ in range(schedule["N_actual"]):
                    G = normal_standard(self.d, self.rng, self.metrics)
                    r = schedule["inveta"]*theta+linear+math.sqrt(schedule["inveta"])*G
                    if not np.all(np.isfinite(r)):
                        self._fail("Nonfinite combined auxiliary vector")
                    theta = self._conditional_draw(r, schedule)
                    self.metrics.inc("proximal_steps")
                    self.metrics.inc("warm_lane_steps")
                self.states[j] = theta
                self.metrics.inc("recorded_lanes")
            # Pair first: the exact estimator is unchanged and initial opposite
            # states cancel without summing separate large signed scores.
            score_sum = sum(float(x@(self.states[j]+self.states[self.m-1-j]))
                            for j in range(self.m//2))
            raw_logit = score_sum/self.m
            self.reference_round = self.history.n
            self.reference_gamma = schedule["gamma"]
            self.gap_norm = math.sqrt(norm2)  # Include old fractional-row replacement.
        if not math.isfinite(raw_logit):
            self._fail("Nonfinite predictive logit")
        t = self.history.n
        guard = math.log(self._preflight["guard_factor"])+math.log(t)+math.log(t+1)
        logit = float(np.clip(raw_logit, -guard, guard))
        if logit != raw_logit:
            self.metrics.inc("guard_clips")
        schedule.update(raw_logit=raw_logit, guarded_logit=logit, guard=guard)
        self._pending_x, self._pending_logit = x.copy(), logit
        self._last_schedule = schedule
        self._peak_state_bytes = max(self._peak_state_bytes, self.state_bytes())
        self.metrics.inc("predictions")
        self.metrics.times["predict"] += perf_counter()-start
        return logit

    def state_bytes(self):
        return super().state_bytes()+(0 if self.states is None else self.states.nbytes)

    def diagnostics(self):
        moves = self.metrics.counters["proximal_steps"]
        proposals = self.metrics.counters["rqo_proposals"]
        return dict(mode=self.mode, lanes=self.m,
                    prescribed_lanes=self._preflight["prescribed"]["m"],
                    prescribed_log_lanes=self._preflight["prescribed"]["log_m"],
                    skip_uses_prescribed_lanes=True, last_schedule=self._last_schedule,
                    reference_round=self.reference_round, reference_gamma=self.reference_gamma,
                    gap_norm=self.gap_norm, actual_total_moves=moves,
                    prescribed_total_moves_on_sampled_rounds=(self._total_prescribed_moves
                        if self._prescribed_moves_representable else None),
                    rqo_acceptance_rate=None if proposals == 0 else self.metrics.counters["rqo_accepts"]/proposals,
                    cache_first_acceptance_rate=None if moves == 0 else self.metrics.counters["cached_first_accepts"]/moves,
                    state_bytes=self.state_bytes(), peak_persistent_state_bytes=self._peak_state_bytes,
                    history_length=self.history.n, history_representation="raw_feature_blocks",
                    history_block_size=self._preflight["history_block_size"],
                    coefficient_state_persistent=True, prior_pairing="opposite_midpoints",
                    midpoint_pairs_independent_in_exact_arithmetic=True,
                    within_pair_independence_assumed=False,
                    failed=self._failed,
                    global_curvature_underflow=self._preflight["prescribed"]["global_curvature_underflow"],
                    floating_point_certificate=False, numerical_failures=self.metrics.counters["numerical_failures"],
                    preflight=self._preflight, metrics=self.metrics.as_dict(),
                    certificate="Ideal-real prescribed schedule only; practical caps and float64 are not certified")
