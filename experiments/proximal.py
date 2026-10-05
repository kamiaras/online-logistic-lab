"""Box-conditioned proximal log-odds sampling from SMALL_CLOUD_COMPLETE.md.

The rejection identities are exact in real arithmetic; this NumPy implementation
is a floating-point experiment, not a floating-point regret certificate.
Practical mode caps only particles and proximal transitions. R defaults to one.
"""
from time import perf_counter
import math
import numpy as np

from .common import History, Metrics, potential, sigmoid


class PrescribedBudgetExceeded(ValueError):
    """A prescribed run failed its allocation-free resource preflight."""
    def __init__(self, details):
        self.details = details
        super().__init__("Prescribed proximal budget exceeded: "
                         + "; ".join(details["budget_reasons"]))


def _positive_integer(name, value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name+" must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name+" must be a positive integer") from exc
    if integer != value or integer < 1:
        raise ValueError(name+" must be a positive integer")
    return integer


def _fail(metrics, message):
    if metrics is not None:
        metrics.inc("numerical_failures")
    raise FloatingPointError(message)


def _uniform(rng, n, metrics):
    if metrics is not None:
        metrics.inc("uniform_draws", n)
    return rng.random(n)


def _normal(rng, shape, metrics):
    if metrics is not None:
        metrics.inc("normal_draws", np.prod(shape, dtype=np.int64))
    return rng.standard_normal(shape)


def _clip_roundoff(values, lower, upper, metrics):
    values, lower, upper = np.broadcast_arrays(values, lower, upper)
    if np.any(~np.isfinite(values)):
        _fail(metrics, "Nonfinite bounded proposal")
    outside = (values < lower) | (values > upper)
    if np.any(outside):
        scale = np.maximum(np.abs(values), np.maximum(np.abs(lower), np.abs(upper)))
        tolerance = 64*np.finfo(float).eps*scale+8*np.nextafter(0., 1.)
        violation = np.maximum(lower-values, values-upper)
        if np.any(violation > tolerance):
            _fail(metrics, "Bounded proposal escaped beyond roundoff tolerance")
        if metrics is not None:
            metrics.inc("roundoff_boundary_clamps", np.count_nonzero(outside))
    return np.clip(values, lower, upper)


def _positive_tail_offsets(alpha, width, rng, metrics=None):
    """Normal tail offsets on [0,width], without forming alpha+width."""
    alpha, width = np.broadcast_arrays(alpha, width)
    a, w = alpha.ravel(), width.ravel()
    out = np.empty_like(a)
    pending = np.arange(len(a))
    for _ in range(10000):
        if not len(pending):
            return out.reshape(alpha.shape)
        rate = a[pending] + 1.
        with np.errstate(over="ignore", under="ignore"):
            mass = -np.expm1(-rate * w[pending])
        z = -np.log1p(-_uniform(rng, len(pending), metrics) * mass) / rate
        if not np.all(np.isfinite(z)):
            _fail(metrics, "Nonfinite truncated-exponential proposal")
        z = _clip_roundoff(z, 0., w[pending], metrics)
        log_accept = -.5 * (z - 1.) ** 2
        good = np.log1p(-_uniform(rng, len(pending), metrics)) <= log_accept
        if metrics is not None:
            metrics.inc("tn_proposals", len(pending))
            metrics.inc("tn_rejections", np.count_nonzero(~good))
        out[pending[good]] = z[good]
        pending = pending[~good]
    _fail(metrics, "Truncated-normal rejection loop exceeded numerical guard")


def truncated_standard_normal(alpha, beta, rng, metrics=None):
    """Broadcasting finite-interval TN sampler; no normal CDF/inverse CDF.

    Scalar inputs return a float. Work arrays are bounded to 4096 coordinates.
    Unrepresentable (equal floating-point) endpoints are rejected explicitly.
    """
    aa, bb = np.broadcast_arrays(np.asarray(alpha, float), np.asarray(beta, float))
    if np.any(~np.isfinite(aa)) or np.any(~np.isfinite(bb)) or np.any(aa >= bb):
        raise ValueError("TN requires finite, strictly ordered endpoints")
    result = np.empty(aa.size)
    for lo in range(0, aa.size, 4096):
        a, b = aa.ravel()[lo:lo+4096], bb.ravel()[lo:lo+4096]
        sample = np.empty_like(a)
        positive, negative = a >= 0., b <= 0.
        for mask, left, right, sign in ((positive, a, b, 1.),
                                        (negative, -b, -a, -1.)):
            if np.any(mask):
                offset = _positive_tail_offsets(left[mask], right[mask]-left[mask],
                                                rng, metrics)
                sample[mask] = sign * (left[mask] + offset)
        cross = ~(positive | negative)
        short = cross & ((b-a) < 1.)
        pending = np.flatnonzero(cross)
        for _ in range(10000):
            if not len(pending):
                break
            is_short = short[pending]
            z = np.empty(len(pending))
            ix = pending[is_short]
            z[is_short] = a[ix] + (b[ix]-a[ix])*_uniform(rng, len(ix), metrics)
            z[~is_short] = _normal(rng, (np.count_nonzero(~is_short),), metrics)
            good = (z >= a[pending]) & (z <= b[pending])
            good[is_short] &= (np.log1p(-_uniform(rng, len(ix), metrics))
                               <= -.5*z[is_short]**2)
            if metrics is not None:
                metrics.inc("tn_proposals", len(pending))
                metrics.inc("tn_rejections", np.count_nonzero(~good))
            sample[pending[good]] = z[good]
            pending = pending[~good]
        else:
            _fail(metrics, "Truncated-normal rejection loop exceeded numerical guard")
        if np.any(~np.isfinite(sample)):
            _fail(metrics, "Nonfinite truncated-normal sample")
        result[lo:lo+len(a)] = _clip_roundoff(sample, a, b, metrics)
    shaped = result.reshape(aa.shape)
    return float(shaped) if shaped.ndim == 0 else shaped


def _box_normal(center, sd, bound, rng, metrics):
    """Sample a diagonal normal on [-bound,bound], avoiding tail cancellation."""
    center = np.asarray(center, float)
    if not np.isfinite(sd) or sd <= 0. or np.any(~np.isfinite(center)):
        _fail(metrics, "Invalid conditional Gaussian parameters")
    out = np.empty_like(center)
    cf, of = center.ravel(), out.ravel()
    width = (2.*bound)/sd
    if not np.isfinite(width) or width <= 0.:
        _fail(metrics, "Unrepresentable standardized box width")
    for lo in range(0, len(cf), 4096):
        c = cf[lo:lo+4096]
        z = np.empty_like(c)
        left, right = c <= -bound, c >= bound
        for mask, distance, sign in ((left, -bound-c, 1.),
                                     (right, c-bound, -1.)):
            if np.any(mask):
                alpha = distance[mask]/sd
                if np.any(~np.isfinite(alpha)):
                    _fail(metrics, "Unrepresentable conditional Gaussian tail")
                offset = _positive_tail_offsets(alpha, np.full(len(alpha), width),
                                                rng, metrics)
                z[mask] = sign * (-bound + sd*offset)
        inside = ~(left | right)
        if np.any(inside):
            ci = c[inside]
            draw = truncated_standard_normal((-bound-ci)/sd, (bound-ci)/sd,
                                              rng, metrics)
            z[inside] = ci + sd*draw
        of[lo:lo+len(c)] = _clip_roundoff(z, -bound, bound, metrics)
    return out


def logistic_bregman(old, new, metrics=None):
    """softplus(new)-softplus(old)-sigmoid(old)*(new-old), stably.

    Fractional and observed label linear terms cancel in this remainder.
    """
    old, new = np.broadcast_arrays(np.asarray(old, float), np.asarray(new, float))
    delta = new-old
    sign = np.where(old >= 0., -1., 1.)
    p = sigmoid(-np.abs(old))
    small = np.abs(delta) <= .5
    answer = np.empty_like(delta)
    ds = sign[small]*delta[small]
    answer[small] = np.log1p(p[small]*np.expm1(ds))-p[small]*ds
    dl = sign[~small]*delta[~small]
    answer[~small] = (np.logaddexp(0., sign[~small]*new[~small])
                      - np.logaddexp(0., -np.abs(old[~small]))-p[~small]*dl)
    tolerance = 64*np.finfo(float).eps*(1.+np.abs(delta))
    if np.any(~np.isfinite(answer)) or np.any(answer < -tolerance):
        _fail(metrics, "Invalid logistic majorant gap")
    if metrics is not None:
        metrics.inc("roundoff_gap_clamps", np.count_nonzero(answer < 0.))
    return np.maximum(answer, 0.)


def _parameters(d, B, T, R):
    b = B*R
    if not math.isfinite(b) or not math.isfinite(B*B) or B*B == 0.:
        raise ValueError("B and BR must have representable squared scales")
    if not math.isfinite(1./(B*B)):
        raise ValueError("B^-2 must be representable in float64")
    b2 = b*b
    if not math.isfinite(b2):
        raise ValueError("Squared BR exceeds the floating-point range")
    L = 2*b*math.sqrt(d)
    if not math.isfinite(L):
        raise ValueError("Score bound exceeds the floating-point range")
    loga = -L-2*math.log1p(math.exp(-L))
    if b == 0.:
        m, log_m, logz = 1, 0., None
    else:
        logz = loga+2*math.log(b)+math.log(T)-math.log(d)
        if logz < -18.:
            z = math.exp(logz)
            log_ratio = math.log1p(-z/2.+z*z/3.)
        elif logz > 50.:
            log_ratio = math.log(logz+math.log1p(math.exp(-logz)))-logz
        else:
            z = math.exp(logz)
            log_ratio = math.log(math.log1p(z)/z)
        log_f = math.log1p(b2/12.)+2*math.log(b)+math.log(T)-math.log(8.)+log_ratio
        # Very large theoretical counts remain logarithmic metadata, never arrays.
        m = max(1, math.ceil(math.exp(log_f))) if log_f < 690. else None
        log_m = math.log(m) if m is not None else log_f
    H = math.log1p(2*b2)-math.log(2.)+2*(math.log1p(L)+log_m+math.log(T)+math.log(T+1))
    contraction = 1./(1.+d*b2*T/2.)
    n_worst = math.ceil(H/(2*math.log1p(contraction))) if contraction > 0. else None
    J = max(0, math.ceil((math.log(3*math.sqrt(2.))+math.log(T)+math.log(b))/math.log(3.))) if b else 0
    return dict(m=m, log_m=log_m, log_a=loga, log_z=logz, H=H, J=J,
                N_max=n_worst, score_bound=L)


class ProximalLogOdds:
    """predict(x) returns a logit; observe(x,y) commits the current observation.

    Config: mode ('practical'/'prescribed'), particles (cap, default 16),
    transitions (cap, default 1), R (default 1), particle_batch (default 64).
    Prescribed preflight limits: max_particles=100000,
    max_particle_steps=1000000, max_operations=1e9, max_state_bytes=512 MiB.
    Resource checks refuse work; they never silently cap prescribed schedules.
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
        p = _parameters(d, B, T, R)
        cap_m, cap_n = cfg.get("particles", 16), cfg.get("transitions", 1)
        for name, value in (("particles", cap_m), ("transitions", cap_n),
                            ("particle_batch", cfg.get("particle_batch", 64))):
            _positive_integer(name, value)
        actual_m = p["m"] if mode == "prescribed" else min(p["m"] or int(cap_m), int(cap_m))
        actual_n = p["N_max"] if mode == "prescribed" else min(p["N_max"] or int(cap_n), int(cap_n))
        reasons = []
        if actual_m is None or actual_n is None:
            reasons.append("theoretical count exceeds representable preflight range")
            steps = cells = operations = None
        else:
            steps = actual_m*actual_n*T
            # History capacity is block-rounded by common.History.
            cells = ((T+1023)//1024)*1024*(d+1)+actual_m*(d+1)+16*d+128
            operations = (1000*actual_m*d*(T+1)
                          +100*actual_m*d*actual_n*(p["J"]+10)*T*(T+1)//2)
            checks = (("particles", "max_particles", actual_m, 100000),
                      ("particle transitions", "max_particle_steps", steps, 1000000),
                      ("estimated operations", "max_operations", operations, 1e9),
                      ("state bytes", "max_state_bytes", 8*cells, 512*1024**2))
            for name, key, count, default in checks:
                limit = cfg.get(key, default)
                enforce = (mode == "prescribed" or key in cfg
                           or key in ("max_particles", "max_state_bytes"))
                if enforce and count > limit:
                    reasons.append(f"{name}: {count} > {limit}")
        return dict(mode=mode, d=d, B=B, R=R, T=T, prescribed=p,
                    actual_particles=actual_m, actual_max_transitions=actual_n,
                    estimated_particle_steps=steps, estimated_operations=operations,
                    estimated_state_bytes=None if cells is None else 8*cells,
                    feasible=not reasons, budget_reasons=reasons,
                    floating_point_certificate=False)

    def __init__(self, d, B, T, seed, config=None):
        started = perf_counter()
        self.config = dict(config or {})
        self._preflight = self.preflight(d, B, T, self.config)
        if not self._preflight["feasible"]:
            raise PrescribedBudgetExceeded(self._preflight)
        self.d, self.B, self.T = int(d), float(B), int(T)
        self.R = float(self._preflight["R"])
        self.mode = self._preflight["mode"]
        self.metrics, self.rng = Metrics(), np.random.default_rng(seed)
        self.history = History(self.d)
        self.m = self._preflight["actual_particles"]
        self.batch = min(64, int(self.config.get("particle_batch", 64)))
        self.lam, self.bound = 1./(self.B*self.B), 2*self.B
        self.theta = self.B*truncated_standard_normal(
            np.full((self.m, self.d), -2.), np.full((self.m, self.d), 2.),
            self.rng, self.metrics)
        self.metrics.inc("prior_particles", self.m)
        self.tilts = np.zeros(self.m)
        self.sum_norm2 = 0.
        self._pending_x = None
        self._pending_logit = None
        self._last_schedule = None
        self._max_residual_ratio = 0.
        self._n_theoretical_sum = 0
        self._n_actual_sum = 0
        self.metrics.times["initialization"] += perf_counter()-started

    def _feature(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.d,) or not np.all(np.isfinite(x)):
            raise ValueError("Feature must be a finite d-vector")
        if np.linalg.norm(x) > self.R*(1+1e-12)+1e-15:
            raise ValueError("Feature norm exceeds configured R")
        return x

    def _schedule(self, x):
        t = self.history.n+1
        E = (self.sum_norm2+float(x@x))/4.
        inveta = self.lam+2*self.d*E
        mu = self.lam+inveta
        k = max(0, math.ceil(math.log(max(1., 6*t*self.R/math.sqrt(mu)), 3)))
        N = math.ceil(self._preflight["prescribed"]["H"]
                      /(2*math.log1p(self.lam/inveta)))
        return dict(t=t, E=E, inveta=inveta, eta=1./inveta, mu=mu,
                    smooth=mu+E, k=k, N_prescribed=N,
                    N_actual=N if self.mode == "prescribed"
                    else min(N, int(self.config.get("transitions", 1))))

    def _conditional_center(self, Y, x, p, schedule):
        Y = np.atleast_2d(np.asarray(Y, float))
        inveta, smooth = schedule["inveta"], schedule["smooth"]
        w = np.clip(Y/(1+self.lam/inveta), -self.bound, self.bound)
        def gradient(z):
            _, grad = potential(z, self.B, self.history, self.metrics,
                                extra_x=x, extra_p=p, want_value=False)
            return grad+inveta*(z-Y)
        for _ in range(schedule["k"]):
            w = np.clip(w-gradient(w)/smooth, -self.bound, self.bound)
            self.metrics.inc("projected_center_steps", len(Y))
        gw = gradient(w)
        v = np.clip(w-gw/smooth, -self.bound, self.bound)
        self.metrics.inc("projected_center_steps", len(Y))
        value, g = potential(v, self.B, self.history, self.metrics,
                             extra_x=x, extra_p=p)
        g += inveta*(v-Y)
        value += .5*inveta*np.sum((v-Y)**2, axis=1)
        normal = smooth*(w-v)-gw
        residual = g+normal
        if not all(np.all(np.isfinite(z)) for z in (v, g, value, residual)):
            _fail(self.metrics, "Nonfinite conditional envelope")
        ratio = float(np.max(np.linalg.norm(residual, axis=1)/math.sqrt(schedule["mu"])))
        self._max_residual_ratio = max(self._max_residual_ratio, ratio)
        return dict(v=v, g=g, value=value, normal=normal, residual=residual,
                    center=v-g/schedule["mu"])

    def _rejection_gap(self, v, z, x):
        gap = np.zeros(len(z))
        for xb, _ in self.history.batches():
            old, new = xb@v.T, xb@z.T
            gap += np.sum(logistic_bregman(old, new, self.metrics), axis=0)
        gap += logistic_bregman(v@x, z@x, self.metrics)
        pairs = len(z)*(self.history.n+1)
        self.metrics.inc("rejection_gap_calls")
        self.metrics.inc("history_particle_pairs", len(z)*self.history.n)
        self.metrics.inc("logistic_pairs", pairs)
        self.metrics.inc("estimated_matmul_flops", 4*pairs*self.d)
        self.metrics.inc("sigmoid_elements", pairs)
        self.metrics.inc("softplus_elements", 2*pairs)
        if not np.all(np.isfinite(gap)):
            _fail(self.metrics, "Nonfinite rejection gap")
        return gap

    def _conditional_draw(self, Y, x, p, schedule):
        Y = np.atleast_2d(np.asarray(Y, float))
        if len(Y) > self.batch:
            result = np.empty_like(Y)
            pp = np.asarray(p)
            for lo in range(0, len(Y), self.batch):
                hi = min(lo+self.batch, len(Y))
                result[lo:hi] = self._conditional_draw(
                    Y[lo:hi], x, pp if pp.ndim == 0 else pp[lo:hi], schedule)
            return result
        env = self._conditional_center(Y, x, p, schedule)
        result = np.empty_like(Y)
        pending = np.arange(len(Y))
        for _ in range(10000):
            if not len(pending):
                return result
            z = _box_normal(env["center"][pending], 1./math.sqrt(schedule["mu"]),
                            self.bound, self.rng, self.metrics)
            gap = self._rejection_gap(env["v"][pending], z, x)
            good = np.log1p(-_uniform(self.rng, len(pending), self.metrics)) <= -gap
            self.metrics.inc("rqo_proposals", len(pending))
            self.metrics.inc("rqo_rejections", np.count_nonzero(~good))
            result[pending[good]] = z[good]
            pending = pending[~good]
        _fail(self.metrics, "Conditional rejection loop exceeded numerical guard")

    def predict(self, x):
        start = perf_counter()
        x = self._feature(x)
        if self._pending_x is not None:
            if np.array_equal(x, self._pending_x):
                return self._pending_logit
            raise ValueError("Observe the pending prediction before predicting new features")
        if self.history.n >= self.T:
            raise ValueError("Configured horizon has been exhausted")
        schedule = self._schedule(x)
        self.tilts = _uniform(self.rng, self.m, self.metrics)-.5
        for lo in range(0, self.m, self.batch):
            hi = min(lo+self.batch, self.m)
            th = self.theta[lo:hi].copy()
            for _ in range(schedule["N_actual"]):
                Y = th+math.sqrt(schedule["eta"])*_normal(self.rng, th.shape, self.metrics)
                th = self._conditional_draw(Y, x, .5+self.tilts[lo:hi], schedule)
                self.metrics.inc("proximal_steps", len(th))
            self.theta[lo:hi] = th
        logit = float(np.mean(self.theta@x))
        if not math.isfinite(logit):
            _fail(self.metrics, "Nonfinite predictive logit")
        self._pending_x, self._pending_logit = x.copy(), logit
        self._last_schedule = schedule
        self._n_theoretical_sum += schedule["N_prescribed"]
        self._n_actual_sum += schedule["N_actual"]
        self.metrics.inc("predictions")
        self.metrics.times["predict"] += perf_counter()-start
        return logit

    def observe(self, x, y):
        x = self._feature(x)
        if y not in (-1, 1):
            raise ValueError("Label must be -1 or +1")
        if self._pending_x is None or not np.array_equal(x, self._pending_x):
            raise ValueError("Observation must match the pending prediction")
        self.history.append(x, y)
        self.sum_norm2 += float(x@x)
        self._pending_x = self._pending_logit = None
        self.metrics.inc("observations")

    def state_bytes(self):
        """Owned ndarray capacity, including allocated history padding."""
        return int(self.history.nbytes+self.theta.nbytes+self.tilts.nbytes
                   +(0 if self._pending_x is None else self._pending_x.nbytes))

    def diagnostics(self):
        proposals = self.metrics.counters["rqo_proposals"]
        rejections = self.metrics.counters["rqo_rejections"]
        return dict(mode=self.mode, particles=self.m,
                    prescribed_particles=self._preflight["prescribed"]["m"],
                    prescribed_log_particles=self._preflight["prescribed"]["log_m"],
                    last_schedule=self._last_schedule,
                    theoretical_transitions_per_trajectory_sum=self._n_theoretical_sum,
                    actual_transitions_per_trajectory_sum=self._n_actual_sum,
                    max_center_residual_ratio=self._max_residual_ratio,
                    rqo_acceptance_rate=None if proposals == 0 else 1-rejections/proposals,
                    numerical_failures=self.metrics.counters["numerical_failures"],
                    state_bytes=self.state_bytes(), history_length=self.history.n,
                    preflight=self._preflight, metrics=self.metrics.as_dict(),
                    certificate="Ideal-real rejection identities; no floating-point regret certificate")
