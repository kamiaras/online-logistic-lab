"""Controlled ablations of the archived shared-tangent implementation.

Historical modules are imported, never edited. These practical schedules carry
no prescribed-regret certificate. Tests vary acceptance, lane layout, and reuse.
"""
import math
from time import perf_counter
import numpy as np
from .shared_tangent import SharedTangentLogOdds
from .shared_tangent_primitives import normal_interval, normal_standard


class SharedAblation(SharedTangentLogOdds):
    def __init__(self, d, B, T, seed, config=None):
        cfg = dict(config or {})
        self.acceptance = cfg.get("acceptance", "full")
        self.layout = cfg.get("layout", "sweep")
        self.persistence = cfg.get("persistence", "fresh")
        if self.acceptance not in ("full", "poisson"):
            raise ValueError("unknown acceptance")
        if self.layout not in ("sweep", "lanes"):
            raise ValueError("unknown layout")
        if self.persistence not in ("fresh", "warm"):
            raise ValueError("unknown persistence")
        self.states = None
        self.last_scores = None
        self.last_initial_scores = None
        self.capture = []
        self._capture_checkpoints = set(cfg.get("capture_checkpoints", []))
        self._capture_per_round = int(cfg.get("capture_per_round", 2))
        self._captures_this_round = 0
        self._current_r = None
        super().__init__(d, B, T, seed, cfg)

    def _conditional_draw(self, r, schedule):
        self._current_r = r
        return super()._conditional_draw(r, schedule)

    def _residual_test(self, z):
        if (self.history.n in self._capture_checkpoints
                and self._captures_this_round < self._capture_per_round):
            # Deterministic diagnostic capture, without drawing random numbers.
            self.capture.append(dict(t=self.history.n, v=self.v.copy(),
                                     z=z.copy(), g=self.g.copy(),
                                     r=self._current_r.copy()))
            self._captures_this_round += 1
        if self.acceptance == "poisson":
            return super()._residual_test(z)
        start = perf_counter()
        residual = self._full_residual(z)
        accepted = -math.log(self._uniform()) >= residual
        self.metrics.inc("direct_tests")
        self.metrics.times["direct_test"] += perf_counter()-start
        return accepted

    def _prior(self):
        self.metrics.inc("prior_initializations")
        self.metrics.inc("prior_coordinates", self.d)
        return self.B*normal_interval(np.full(self.d, -2.), np.full(self.d, 2.),
                                     self.rng, self.metrics)

    def predict(self, x):
        start = perf_counter()
        x = self._feature(x)
        if self._pending_x is not None:
            if np.array_equal(x, self._pending_x):
                return self._pending_logit
            raise ValueError("Pending observation")
        if self.history.n >= self.T:
            raise ValueError("Exhausted horizon")
        self._append_feature(x)
        self._captures_this_round = 0
        schedule = self._schedule()
        norm2 = float(x@x)
        row_l1 = float(np.sum(np.abs(x)))
        obs_norm = float(np.linalg.norm(self.a_obs))
        delta = min(2*self.B*row_l1, obs_norm*math.sqrt(norm2)/schedule["gamma"])
        b_lower = norm2/(self.lam+4*self._preflight["prescribed"]["a"]*self.E)
        if delta == 0:
            skip = True
        elif b_lower == 0:
            skip = False
        else:
            skip = math.log(delta)-math.log(2) <= (
                math.log(3/8)+math.log(b_lower)-self._preflight["prescribed"]["log_m"])
        schedule.update(delta_bound=delta, b_lower=b_lower, skipped=skip,
                        m_actual=self.m, m_prescribed=self._preflight["prescribed"]["m"])
        if skip:
            logit = 0.
            self.last_scores = np.full(self.m, np.nan)
            self.last_initial_scores = np.full(self.m, np.nan)
            self.metrics.inc("skipped_rounds")
        else:
            self.metrics.inc("sampled_rounds")
            scores, initial = [], []
            final_states = []
            theta = None
            if self.layout == "sweep":
                theta = (self._prior() if self.persistence == "fresh" or self.states is None
                         else self.states[0].copy())
            for j in range(self.m):
                if self.layout == "lanes":
                    theta = (self._prior() if self.persistence == "fresh" or self.states is None
                             else self.states[j].copy())
                initial.append(float(x@theta))
                tilt = -.5+(j+.5)/self.m
                linear = self.a_obs+tilt*x
                moves = schedule["N_actual"] if j == 0 else schedule["ell_actual"]
                for _ in range(moves):
                    G = normal_standard(self.d, self.rng, self.metrics)
                    r = schedule["inveta"]*theta+linear+math.sqrt(schedule["inveta"])*G
                    theta = self._conditional_draw(r, schedule)
                    self.metrics.inc("proximal_steps")
                    self.metrics.inc("burn_in_steps" if j == 0 else "inter_tilt_steps")
                scores.append(float(x@theta))
                self.metrics.inc("recorded_tilts")
                if self.layout == "lanes":
                    final_states.append(theta.copy())
            if self.persistence == "warm":
                self.states = (np.stack(final_states) if self.layout == "lanes"
                               else theta[None, :].copy())
            self.last_scores = np.asarray(scores)
            self.last_initial_scores = np.asarray(initial)
            logit = float(np.mean(scores))
        if not math.isfinite(logit):
            self._fail("Nonfinite ablation score")
        self._pending_x, self._pending_logit = x.copy(), logit
        self._last_schedule = schedule
        self._peak_state_bytes = max(self._peak_state_bytes, self.state_bytes())
        self.metrics.inc("predictions")
        self.metrics.times["predict"] += perf_counter()-start
        return logit

    def state_bytes(self):
        return super().state_bytes()+(0 if self.states is None else self.states.nbytes)


def guarded_logits(logits):
    logits = np.asarray(logits)
    rounds = np.arange(1, len(logits)+1, dtype=float)
    cap = np.log(rounds*(rounds+1))
    return np.clip(logits, -cap, cap)
