"""The shared-tangent, deterministic log-odds sweep in the user's manuscript.

Only m, N_t and ell_t may be reduced in practical mode. The Poisson continuation,
cache repair, curvature ledger and deterministic skip test retain the prescribed
rules. This is a float64 experiment, not a finite-precision theorem. Raw features
and a separate weighted prefix are stored for stable lookup; this deliberately
uses more memory than the manuscript's exact-real signed-prefix representation.
"""
from time import perf_counter
import math
import numpy as np

from .common import Metrics
from .proximal import logistic_bregman, _positive_integer
from .shared_tangent_primitives import (
    WeightedHistory, normal_interval, normal_standard, box_normal,
)


class PrescribedBudgetExceeded(ValueError):
    """An allocation-free resource check refused the requested schedule."""

    def __init__(self, details):
        self.details = details
        super().__init__("Shared-tangent budget exceeded: "
                         + "; ".join(details["budget_reasons"]))


def _logaddexp_scalar(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    largest = max(a, b)
    return largest + math.log1p(math.exp(min(a, b)-largest))


def _ceil_from_log(log_count):
    """Keep impossible counts as logarithmic metadata, never float infinity."""
    if log_count > 690.:
        return None
    return max(1, math.ceil(math.exp(log_count)))


def _mixing_counts(H, gamma, inveta):
    log_ratio = math.log(gamma)-math.log(inveta)
    # log(log(1+exp(log_ratio))) without underflow of the ratio.
    if log_ratio < -35.:
        log_denominator = log_ratio
    elif log_ratio > 35.:
        log_denominator = math.log(log_ratio+math.log1p(math.exp(-log_ratio)))
    else:
        log_denominator = math.log(math.log1p(math.exp(log_ratio)))
    log_N = math.log(H/2.)-log_denominator
    log_ell = math.log(math.log(32.))-log_denominator
    return (_ceil_from_log(log_N), _ceil_from_log(log_ell),
            log_N, log_ell)


def _parameters(d, B, T, R):
    B2, BR = B*B, B*R
    if (not math.isfinite(B2) or B2 <= 0. or not math.isfinite(1./B2)
            or not math.isfinite(BR) or not math.isfinite(BR*BR)):
        raise ValueError("B squared, its inverse and (BR) squared must be representable")
    lam, kappa = 1./B2, BR*BR
    L = 2.*BR*math.sqrt(d)
    if not math.isfinite(L):
        raise ValueError("The global score bound must be representable")
    log_a = -L-2.*math.log1p(math.exp(-L))
    a = math.exp(log_a)
    log_T = math.log(T)
    if kappa == 0.:
        log_Gamma, log_m, m = -math.inf, 0., 1
        log_kappa = -math.inf
    else:
        log_kappa = math.log(kappa)
        log_z = log_a+log_kappa+log_T-math.log(d)
        # Gamma = kappa*T/8 * log(1+z)/z, z=a*kappa*T/d.
        # At underflow this uses its continuous limit rather than dividing by a.
        if log_z < -35.:
            z = math.exp(log_z)
            log_ratio = math.log1p(-z/2.+z*z/3.)
        elif log_z > 35.:
            log_ratio = math.log(log_z+math.log1p(math.exp(-log_z)))-log_z
        else:
            z = math.exp(log_z)
            log_ratio = math.log(math.log1p(z)/z)
        log_Gamma = log_kappa+log_T-math.log(8.)+log_ratio
        log_m = max(0., math.log(L) if L else -math.inf,
                    math.log(3.)+log_Gamma)
        m = _ceil_from_log(log_m)
        if m is not None:
            log_m = math.log(m)
    H = (math.log(4.)+_logaddexp_scalar(0., log_kappa+2.*log_T)
         +2.*math.log1p(L)+2.*log_T)
    # Gamma_t >= lambda and P_t <= max(d,2)*T*R^2/4.
    log_P = (math.log(max(d, 2))+log_T+2.*math.log(R)-math.log(4.)
             if R else -math.inf)
    log_inveta = _logaddexp_scalar(math.log(lam), log_P)
    if log_inveta < 700.:
        Nmax, ellmax, log_Nmax, log_ellmax = _mixing_counts(
            H, lam, math.exp(log_inveta))
    else:
        log_Nmax = math.log(H/2.)+log_inveta-math.log(lam)
        log_ellmax = math.log(math.log(32.))+log_inveta-math.log(lam)
        Nmax, ellmax = _ceil_from_log(log_Nmax), _ceil_from_log(log_ellmax)
    return dict(m=m, log_m=log_m, log_Gamma=log_Gamma if kappa else None,
                a=a, log_a=log_a, H=H, lam=lam, kappa=kappa,
                score_bound=L, N_max=Nmax, ell_max=ellmax,
                log_N_max=log_Nmax, log_ell_max=log_ellmax,
                global_curvature_underflow=(a == 0.))


class SharedTangentLogOdds:
    """predict(x) returns a logit; observe(x,y) appends only the label summary.

    Practical defaults: tilts=16, burn_in=1, transitions=1. These are caps on
    prescribed m, N_t, ell_t, respectively; there is one streamed chain. The skip
    threshold always uses prescribed m. History scans and tangent repairs are not
    capped. The scalar helper's numerical guard may fail explicitly; it never
    returns an approximate draw. A prescribed preflight refuses more than
    max_total_moves=10,000,000 worst-case moves; it never silently caps them.
    """

    @staticmethod
    def preflight(d, B, T, config=None):
        cfg = dict(config or {})
        d, T = _positive_integer("d", d), _positive_integer("T", T)
        B, R = float(B), float(cfg.get("R", 1.))
        if not math.isfinite(B) or B <= 0. or not math.isfinite(R) or R < 0.:
            raise ValueError("B must be positive and R nonnegative, both finite")
        mode = cfg.get("mode", "practical")
        if mode not in ("practical", "prescribed"):
            raise ValueError("mode must be practical or prescribed")
        default_block = min(1024, max(16, 1 << max(0, (1048576//d).bit_length()-1)))
        caps = {name: _positive_integer(name, cfg.get(name, default))
                for name, default in (("tilts", 16), ("burn_in", 1),
                                      ("transitions", 1), ("history_block_size", default_block))}
        max_moves = _positive_integer("max_total_moves", cfg.get("max_total_moves", 10_000_000))
        p = _parameters(d, B, T, R)
        def count(prescribed, key):
            return prescribed if mode == "prescribed" else min(
                prescribed if prescribed is not None else caps[key], caps[key])
        m, N, ell = count(p["m"], "tilts"), count(p["N_max"], "burn_in"), count(p["ell_max"], "transitions")
        moves = None if any(z is None for z in (m, N, ell)) else T*(N+(m-1)*ell)
        block = caps["history_block_size"]
        rows = ((T+block-1)//block)*block
        # Raw+prefix arrays and per-row L1 norms; spare vector allowance is
        # explicit. Supervisor memory measurements include Python and work arrays.
        cells = rows*(2*d+1)+32*d+256
        reasons = []
        if mode == "prescribed" and moves is None:
            reasons.append("prescribed count exceeds the representable schedule range")
        if (mode == "prescribed" or "max_total_moves" in cfg) and moves is not None and moves > max_moves:
            reasons.append(f"worst-case transitions: {moves} > {max_moves}")
        if "max_state_bytes" in cfg:
            limit = _positive_integer("max_state_bytes", cfg["max_state_bytes"])
            if 8*cells > limit:
                reasons.append(f"estimated state bytes: {8*cells} > {limit}")
        return dict(mode=mode, d=d, B=B, T=T, R=R, prescribed=p,
                    actual_tilts=m, actual_max_burn_in=N, actual_max_transitions=ell,
                    history_block_size=block,
                    estimated_total_moves=moves, estimated_state_bytes=8*cells,
                    memory_representation="raw features plus separate weighted prefix",
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
        self.m = self._preflight["actual_tilts"]
        self.lam, self.bound = self._preflight["prescribed"]["lam"], 2.*self.B
        self.metrics, self.rng = Metrics(), np.random.default_rng(seed)
        self.history = WeightedHistory(self.d, self._preflight["history_block_size"], self.metrics)
        self.v, self.g, self.a_obs = (np.zeros(self.d) for _ in range(3))
        self.phi, self.E, self.E1 = 0., 0., 0.
        self.curvature_diagonal, self.curvature_low = np.zeros(self.d), 0.
        self._pending_x = self._pending_logit = self._last_schedule = None
        self._total_prescribed_moves = 0
        self._prescribed_moves_representable = True
        self._peak_state_bytes = self.state_bytes()
        self.metrics.times["initialization"] += perf_counter()-start

    def _fail(self, message):
        self.metrics.inc("numerical_failures")
        raise FloatingPointError(message)

    def _uniform(self):
        # Open interval: endpoints are not accepted as exact continuous draws.
        while True:
            self.metrics.inc("uniform_draws")
            z = float(self.rng.random())
            if 0. < z < 1.:
                return z
            self.metrics.inc("uniform_endpoint_redraws")

    def _feature(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.d,) or not np.all(np.isfinite(x)):
            raise ValueError("Feature must be a finite d-vector")
        if np.linalg.norm(x) > self.R*(1.+1e-12)+1e-15:
            raise ValueError("Feature norm exceeds configured R")
        return x

    def _append_feature(self, x):
        start = perf_counter()
        self.history.append(x)
        norm2, row_l1 = float(x@x), float(np.sum(np.abs(x)))
        self.E += norm2/4.
        self.E1 += row_l1*row_l1/4.
        row_log_a = -2.*self.B*row_l1-2.*math.log1p(math.exp(-2.*self.B*row_l1))
        row_a = math.exp(row_log_a)
        if row_a == 0. and row_l1:
            self.metrics.inc("row_curvature_underflows")
        self.curvature_diagonal += row_a*(2.*x*x-np.abs(x)*row_l1)
        self.curvature_low = max(0., self.curvature_low, float(np.min(self.curvature_diagonal)))
        score = float(x@self.v)
        self.g += (.5*math.tanh(score/2.))*x
        self.phi += float(np.logaddexp(score/2., -score/2.))
        if not (np.all(np.isfinite(self.g)) and math.isfinite(self.phi)
                and np.all(np.isfinite(self.curvature_diagonal))
                and math.isfinite(self.E) and math.isfinite(self.E1)):
            self._fail("Nonfinite feature energy, curvature or cached tangent")
        self.metrics.inc("cache_appends")
        self.metrics.inc("cache_append_rows")
        self.metrics.inc("estimated_matmul_flops", 2*self.d)
        self.metrics.times["cache_append"] += perf_counter()-start

    def _schedule(self):
        P = max(self.E1, 2.*self.E)
        inveta, mu, gamma = self.lam+P, 2.*self.lam+P, self.lam+self.curvature_low
        N, ell, log_N, log_ell = _mixing_counts(self._preflight["prescribed"]["H"], gamma, inveta)
        def capped(value, name, default):
            if self.mode == "prescribed":
                if value is None:
                    self._fail("Prescribed mixing count is not representable")
                return value
            cap = int(self.config.get(name, default))
            return min(value if value is not None else cap, cap)
        return dict(t=self.history.n, E=self.E, E1=self.E1, P=P, inveta=inveta,
                    eta=1./inveta, mu=mu, gamma=gamma,
                    N_prescribed=N, ell_prescribed=ell, log_N_prescribed=log_N,
                    log_ell_prescribed=log_ell,
                    N_actual=capped(N, "burn_in", 1), ell_actual=capped(ell, "transitions", 1))

    def _refresh_cache(self):
        start = perf_counter()
        self.g.fill(0.)
        self.phi = 0.
        for xb in self.history.batches():
            score = xb@self.v
            self.phi += float(np.sum(np.logaddexp(score/2., -score/2.)))
            self.g += xb.T@(.5*np.tanh(score/2.))
            self.metrics.inc("gradient_refresh_rows", len(xb))
            self.metrics.inc("estimated_matmul_flops", 4*len(xb)*self.d)
            self.metrics.inc("softplus_elements", len(xb))
            self.metrics.inc("tanh_elements", len(xb))
        self.metrics.inc("gradient_refreshes")
        if not math.isfinite(self.phi) or not np.all(np.isfinite(self.g)):
            self._fail("Nonfinite full tangent refresh")
        self.metrics.times["gradient_refresh"] += perf_counter()-start

    def _full_residual(self, z):
        """R2's full residual, summed as nonnegative stable row Bregmans."""
        start = perf_counter()
        total = 0.
        for xb in self.history.batches():
            total += float(np.sum(logistic_bregman(xb@self.v, xb@z, self.metrics)))
            self.metrics.inc("fallback_rows", len(xb))
            self.metrics.inc("estimated_matmul_flops", 4*len(xb)*self.d)
        self.metrics.inc("full_residual_calls")
        self.metrics.times["full_residual"] += perf_counter()-start
        if not math.isfinite(total) or total < 0.:
            self._fail("Nonfinite or negative full residual")
        return total

    def _residual_test(self, z):
        start = perf_counter()
        difference2 = np.square(z-self.v)
        weights = self.history.W*difference2
        prefix = np.cumsum(weights)
        total = float(prefix[-1])
        C = total/8.
        if not math.isfinite(C) or C < 0.:
            self._fail("Nonfinite Poisson majorant")
        self.metrics.inc("poisson_tests")
        self.metrics.inc("poisson_coordinate_weights", self.d)
        if C == 0.:
            self.metrics.inc("poisson_zero_majorants")
            self.metrics.times["poisson_test"] += perf_counter()-start
            return True
        t = self.history.n
        # ceil(log2(t+1)) = bit_length(t) for positive integral t.
        q = t.bit_length()+4
        cap = max(1, (t*self.d+self.d+q-1)//(self.d+q))
        u = 0.
        for _ in range(cap):
            u -= math.log(self._uniform())
            if u >= C:
                self.metrics.inc("poisson_early_accepts")
                self.metrics.times["poisson_test"] += perf_counter()-start
                return True
            self.metrics.inc("poisson_arrivals")
            threshold = self._uniform()*total
            if threshold >= total:
                threshold = float(np.nextafter(total, 0.))
                self.metrics.inc("roundoff_lookup_clamps")
            j = int(np.searchsorted(prefix, threshold, side="right"))
            if j >= self.d or self.history.W[j] <= 0. or weights[j] <= 0.:
                self._fail("Poisson coordinate lookup selected a zero weight")
            row_threshold = self._uniform()*float(self.history.W[j])
            if row_threshold >= self.history.W[j]:
                row_threshold = float(np.nextafter(self.history.W[j], 0.))
                self.metrics.inc("roundoff_lookup_clamps")
            i = self.history.sample_row(j, row_threshold)
            xi = self.history.row(i)
            ci = float(self.history.row_l1(i))*float(np.abs(xi)@difference2)/8.
            rho = float(logistic_bregman(float(xi@self.v), float(xi@z), self.metrics))
            self.metrics.inc("poisson_rows")
            self.metrics.inc("estimated_matmul_flops", 6*self.d)
            if not math.isfinite(ci) or ci <= 0.:
                self._fail("Selected Poisson factor has an unrepresentable majorant")
            if rho > ci:
                tolerance = 128.*np.finfo(float).eps*(1.+abs(rho)+ci)
                if rho-ci > tolerance:
                    self._fail("Poisson residual exceeds its quadratic upper bound")
                self.metrics.inc("roundoff_acceptance_clamps")
                rho = ci
            if self._uniform() < rho/ci:
                self.metrics.inc("poisson_factor_rejections")
                self.metrics.times["poisson_test"] += perf_counter()-start
                return False
        self.metrics.inc("poisson_cap_continuations")
        residual = self._full_residual(z)
        remaining = max(0., 1.-u/C)
        accepted = math.log(self._uniform()) <= -remaining*residual
        self.metrics.inc("poisson_continuation_accepts" if accepted else "poisson_continuation_rejections")
        self.metrics.times["poisson_test"] += perf_counter()-start
        return accepted

    def _conditional_draw(self, r, schedule):
        start = perf_counter()
        first = True
        proposals_this_move = 0
        while True:
            center = (r-self.g)/schedule["mu"]  # Must remain UNCLIPPED.
            proposal_start = perf_counter()
            z = box_normal(center, 1./math.sqrt(schedule["mu"]), self.bound, self.rng, self.metrics)
            self.metrics.times["conditional_proposal"] += perf_counter()-proposal_start
            self.metrics.inc("rqo_proposals")
            proposals_this_move += 1
            if self._residual_test(z):
                self.metrics.inc("rqo_accepts")
                if first:
                    self.metrics.inc("cached_first_accepts")
                self.metrics.counters["max_proposals_one_move"] = max(
                    self.metrics.counters["max_proposals_one_move"], proposals_this_move)
                self.metrics.times["conditional_draw"] += perf_counter()-start
                return z
            self.metrics.inc("rqo_rejections")
            self.metrics.inc("rejection_repairs")
            if first:
                self.metrics.inc("cache_misses")
            self.v[:] = np.clip(center, -self.bound, self.bound)
            self._refresh_cache()
            first = False

    def predict(self, x):
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
        # A log-domain comparison keeps the prescribed-m skip conservative even
        # when m is too large for a floating-point count; practical caps never
        # enlarge this threshold.
        if delta == 0.:
            skip = True
        elif b_lower == 0.:
            skip = False
        else:
            skip = (math.log(delta)-math.log(2.) <= math.log(3./8.)
                    +math.log(b_lower)-self._preflight["prescribed"]["log_m"])
        schedule.update(delta_bound=delta, b_lower=b_lower, skipped=skip,
                        m_actual=self.m, m_prescribed=self._preflight["prescribed"]["m"])
        planned = (None if schedule["N_prescribed"] is None or schedule["ell_prescribed"] is None
                   or self._preflight["prescribed"]["m"] is None else
                   schedule["N_prescribed"]+(self._preflight["prescribed"]["m"]-1)*schedule["ell_prescribed"])
        schedule["moves_prescribed_without_skip"] = planned
        schedule["moves_actual"] = 0 if skip else schedule["N_actual"]+(self.m-1)*schedule["ell_actual"]
        if not skip:
            if planned is None:
                self._prescribed_moves_representable = False
            else:
                self._total_prescribed_moves += planned
        if skip:
            logit = 0.
            self.metrics.inc("skipped_rounds")
            if obs_norm == 0. or norm2 == 0.:
                self.metrics.inc("exact_symmetry_skips")
        else:
            self.metrics.inc("sampled_rounds")
            theta = self.B*normal_interval(np.full(self.d, -2.), np.full(self.d, 2.), self.rng, self.metrics)
            self.metrics.inc("prior_initializations")
            self.metrics.inc("prior_coordinates", self.d)
            score_sum = 0.
            for j in range(self.m):
                tilt = -.5+(j+.5)/self.m
                linear = self.a_obs+tilt*x
                moves = schedule["N_actual"] if j == 0 else schedule["ell_actual"]
                for _ in range(moves):
                    G = normal_standard(self.d, self.rng, self.metrics)
                    r = schedule["inveta"]*theta+linear+math.sqrt(schedule["inveta"])*G
                    if not np.all(np.isfinite(r)):
                        self._fail("Nonfinite combined auxiliary vector")
                    theta = self._conditional_draw(r, schedule)
                    self.metrics.inc("proximal_steps")
                    self.metrics.inc("burn_in_steps" if j == 0 else "inter_tilt_steps")
                score_sum += float(x@theta)
                self.metrics.inc("recorded_tilts")
            logit = score_sum/self.m
        if not math.isfinite(logit):
            self._fail("Nonfinite predictive logit")
        self._pending_x, self._pending_logit = x.copy(), logit
        self._last_schedule = schedule
        self._peak_state_bytes = max(self._peak_state_bytes, self.state_bytes())
        self.metrics.inc("predictions")
        self.metrics.times["predict"] += perf_counter()-start
        return logit

    def observe(self, x, y):
        start = perf_counter()
        x = self._feature(x)
        if y not in (-1, 1):
            raise ValueError("Label must be -1 or +1")
        if self._pending_x is None or not np.array_equal(x, self._pending_x):
            raise ValueError("Observation must match the pending prediction")
        self.a_obs += .5*int(y)*x
        if not np.all(np.isfinite(self.a_obs)):
            self._fail("Nonfinite observed-label statistic")
        self._pending_x = self._pending_logit = None
        self.metrics.inc("observations")
        self.metrics.times["observe"] += perf_counter()-start

    def state_bytes(self):
        return int(self.history.nbytes+self.v.nbytes+self.g.nbytes+self.a_obs.nbytes
                   +self.curvature_diagonal.nbytes
                   +(0 if self._pending_x is None else self._pending_x.nbytes))

    def diagnostics(self):
        proposals = self.metrics.counters["rqo_proposals"]
        moves = self.metrics.counters["proximal_steps"]
        return dict(mode=self.mode, tilts=self.m,
                    prescribed_tilts=self._preflight["prescribed"]["m"],
                    prescribed_log_tilts=self._preflight["prescribed"]["log_m"],
                    skip_uses_prescribed_tilts=True, last_schedule=self._last_schedule,
                    prescribed_total_moves_on_sampled_rounds=(self._total_prescribed_moves
                        if self._prescribed_moves_representable else None),
                    actual_total_moves=moves,
                    rqo_acceptance_rate=None if proposals == 0 else self.metrics.counters["rqo_accepts"]/proposals,
                    cache_first_acceptance_rate=None if moves == 0 else self.metrics.counters["cached_first_accepts"]/moves,
                    state_bytes=self.state_bytes(), peak_persistent_state_bytes=self._peak_state_bytes,
                    history_length=self.history.n, history_representation="raw_plus_weighted_prefix",
                    history_block_size=self._preflight["history_block_size"],
                    global_curvature_underflow=self._preflight["prescribed"]["global_curvature_underflow"],
                    floating_point_certificate=False, numerical_failures=self.metrics.counters["numerical_failures"],
                    preflight=self._preflight, metrics=self.metrics.as_dict(),
                    certificate="Ideal-real identities only; practical mixing/tilt caps do not inherit the regret theorem")
