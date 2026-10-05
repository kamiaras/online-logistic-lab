"""Batched exact conditional returns with a persistent tangent for every lane.

The outer paired population, warm schedule, skip and c4 guard are unchanged from
persistent_log_odds.py. Two extra M*d arrays buy matrix operations and batched
normal draws. Conditional loops are uncapped; accepted lanes leave the pending
set, and only rejected lanes refresh their own tangent. History blocks <=128
and lane blocks <=64 bound score-matrix scratch. Float64/caps remain uncertified.
The frozen sequential reference and existing experiment harness are not edited.
"""
from time import perf_counter
import math

import numpy as np

from .persistent_log_odds import PersistentLogOdds
from .proximal import _positive_integer, logistic_bregman
from .shared_tangent_primitives import box_normal, normal_interval, normal_standard, open_uniform


class BatchedPersistentLogOdds(PersistentLogOdds):
    @staticmethod
    def preflight(d, B, T, config=None):
        cfg = dict(config or {})
        alias = cfg.pop("lane_batch_size", None)
        batch = _positive_integer("particle_batch", cfg.pop("particle_batch", alias if alias is not None else 64))
        if alias is not None and alias != batch:
            raise ValueError("particle_batch and lane_batch_size disagree")
        block = _positive_integer("history_block_size", cfg.get("history_block_size", 128))
        if batch > 64 or block > 128:
            raise ValueError("Bounded scratch requires lane_batch_size<=64 and history_block_size<=128")
        cfg["history_block_size"] = block
        memory_limit = cfg.pop("max_state_bytes", None)
        pre = PersistentLogOdds.preflight(d, B, T, cfg)
        m = pre["actual_lanes"]
        extra = None if m is None else 16*m*pre["d"]
        state = None if extra is None else pre["estimated_state_bytes"]+extra
        reasons = list(pre["budget_reasons"])
        if state is not None and state > np.iinfo(np.intp).max:
            reasons.append("batched persistent state exceeds the platform index range")
        if memory_limit is not None:
            limit = _positive_integer("max_state_bytes", memory_limit)
            if state is None or state > limit:
                reasons.append("estimated batched persistent state exceeds max_state_bytes")
        pre.update(lane_batch_size=batch, particle_batch=batch, history_block_size=block,
                   estimated_state_bytes=state, extra_lane_cache_bytes=extra,
                   memory_representation="raw feature blocks, paired states, per-lane anchors and gradients",
                   tangent_cache="one exact anchor/gradient per lane", phi_stored_per_lane=False,
                   feasible=not reasons, budget_reasons=reasons)
        return pre

    def __init__(self, d, B, T, seed, config=None):
        self.anchors = self.gradients = None
        super().__init__(d, B, T, seed, config)
        self.lane_batch_size = self._preflight["lane_batch_size"]
        self.particle_batch = self.lane_batch_size
        cache_start = perf_counter()
        self.anchors = np.zeros((self.m, self.d))
        self.gradients = np.zeros((self.m, self.d))
        self._peak_state_bytes = self.state_bytes()
        self.metrics.times["initialization"] += perf_counter()-cache_start

    def _initialize_lanes(self):
        states = np.empty((self.m, self.d))
        for lo in range(0, self.m//2, self.lane_batch_size):
            hi = min(self.m//2, lo+self.lane_batch_size)
            shape = (hi-lo, self.d)
            draws = self.B*normal_interval(np.full(shape, -2.), np.full(shape, 2.),
                                           self.rng, self.metrics)
            states[lo:hi] = draws
            states[self.m-hi:self.m-lo] = -draws[::-1]
        self.states = states
        self.metrics.inc("prior_initializations", self.m//2)
        self.metrics.inc("prior_coordinates", self.m*self.d//2)
        self.metrics.inc("prior_pair_initializations", self.m//2)
        self.metrics.inc("initialized_lanes", self.m)
        self.metrics.inc("prior_cloud_initializations")

    def _append_feature(self, x):
        start = perf_counter()
        self.history.append(x)
        norm2, row_l1 = float(x@x), float(np.sum(np.abs(x)))
        self.E += norm2/4.
        self.E1 += row_l1*row_l1/4.
        score_bound = 2.*self.B*row_l1
        row_a = math.exp(-score_bound-2.*math.log1p(math.exp(-score_bound)))
        if row_a == 0. and row_l1:
            self.metrics.inc("row_curvature_underflows")
        self.curvature_diagonal += row_a*(2.*x*x-np.abs(x)*row_l1)
        self.curvature_low = max(0., self.curvature_low, float(np.min(self.curvature_diagonal)))
        for lo in range(0, self.m, self.lane_batch_size):
            hi = min(self.m, lo+self.lane_batch_size)
            scores = self.anchors[lo:hi]@x
            self.gradients[lo:hi] += (.5*np.tanh(scores/2.))[:, None]*x
            if not np.all(np.isfinite(self.gradients[lo:hi])):
                self._fail("Nonfinite incremental lane gradients")
        self.gap_norm += math.sqrt(norm2)
        if (not np.all(np.isfinite(self.curvature_diagonal))
                or not all(math.isfinite(z) for z in (self.E, self.E1, self.gap_norm))):
            self._fail("Nonfinite lane cache, feature energy or curvature")
        self.metrics.inc("cache_appends", self.m)
        self.metrics.inc("cache_append_rows", self.m)
        self.metrics.inc("estimated_matmul_flops", 2*self.m*self.d)
        self.metrics.times["cache_append"] += perf_counter()-start

    def _refresh_lanes(self, lanes):
        """Recompute gradients only for the rejected lanes, using bounded blocks."""
        lanes = np.asarray(lanes, dtype=np.intp)
        if len(lanes) == 0:
            return
        if len(lanes) > self.lane_batch_size:
            raise ValueError("Refresh exceeds bounded lane batch")
        start = perf_counter()
        anchors = self.anchors[lanes]
        gradients = np.zeros_like(anchors)
        for xb in self.history.batches():
            scores = xb@anchors.T
            gradients += (.5*np.tanh(scores/2.)).T@xb
            self.metrics.inc("gradient_refresh_rows", len(xb)*len(lanes))
            self.metrics.inc("estimated_matmul_flops", 4*len(xb)*len(lanes)*self.d)
            self.metrics.inc("tanh_elements", len(xb)*len(lanes))
        if not np.all(np.isfinite(gradients)):
            self._fail("Nonfinite batched gradient refresh")
        self.gradients[lanes] = gradients
        self.metrics.inc("gradient_refreshes", len(lanes))
        self.metrics.inc("gradient_refresh_batches")
        self.metrics.times["gradient_refresh"] += perf_counter()-start

    def _residual_test_batch(self, lanes, proposals, thresholds=None):
        """Independent exponential thresholds and exact nonnegative row sums.

        thresholds is a deterministic testing hook, never used by predict.
        Accepting lanes scan every block; rejected lanes can stop at a block.
        """
        start = perf_counter()
        lanes = np.asarray(lanes, dtype=np.intp)
        count = len(lanes)
        if count > self.lane_batch_size:
            raise ValueError("Residual test exceeds bounded lane batch")
        if thresholds is None:
            thresholds = -np.log(open_uniform(self.rng, count, self.metrics))
        else:
            thresholds = np.asarray(thresholds, dtype=float)
            if thresholds.shape != (count,) or np.any(~np.isfinite(thresholds)) or np.any(thresholds < 0.):
                raise ValueError("Thresholds must be a finite nonnegative batch vector")
        anchors = self.anchors[lanes]
        totals, accepted = np.zeros(count), np.ones(count, dtype=bool)
        live = np.arange(count)
        for xb in self.history.batches():
            if not len(live):
                break
            old, new = xb@anchors[live].T, xb@proposals[live].T
            sums = np.sum(logistic_bregman(old, new, self.metrics), axis=0)
            totals[live] += sums
            if not np.all(np.isfinite(totals[live])) or np.any(sums < 0.):
                self._fail("Nonfinite or negative batched residual")
            self.metrics.inc("full_test_rows", len(xb)*len(live))
            self.metrics.inc("estimated_matmul_flops", 4*len(xb)*len(live)*self.d)
            bad = totals[live] > thresholds[live]
            accepted[live[bad]] = False
            live = live[~bad]
        self.metrics.inc("full_tests", count)
        self.metrics.inc("full_test_accepts", np.count_nonzero(accepted))
        self.metrics.inc("full_test_rejections", np.count_nonzero(~accepted))
        self.metrics.times["full_test"] += perf_counter()-start
        return accepted

    def _conditional_batch(self, r, lanes, schedule):
        """Complete independent exact returns for a batch; never cap retries."""
        start = perf_counter()
        lanes = np.asarray(lanes, dtype=np.intp)
        if len(lanes) > self.lane_batch_size or r.shape != (len(lanes), self.d):
            raise ValueError("Invalid conditional batch shape")
        result = np.empty_like(r)
        pending = np.arange(len(lanes))
        counts = np.zeros(len(lanes), dtype=np.int64)
        while len(pending):
            active = lanes[pending]
            centers = (r[pending]-self.gradients[active])/schedule["mu"]
            if not np.all(np.isfinite(centers)):
                self._fail("Nonfinite batched untruncated center")
            proposal_start = perf_counter()
            proposals = box_normal(centers, 1./math.sqrt(schedule["mu"]),
                                   self.bound, self.rng, self.metrics)
            self.metrics.times["conditional_proposal"] += perf_counter()-proposal_start
            counts[pending] += 1
            self.metrics.inc("rqo_proposals", len(pending))
            accepted = self._residual_test_batch(active, proposals)
            result[pending[accepted]] = proposals[accepted]
            self.metrics.inc("rqo_accepts", np.count_nonzero(accepted))
            self.metrics.inc("cached_first_accepts", np.count_nonzero(accepted & (counts[pending] == 1)))
            rejected = ~accepted
            if np.any(rejected):
                rejected_lanes = active[rejected]
                self.metrics.inc("rqo_rejections", len(rejected_lanes))
                self.metrics.inc("rejection_repairs", len(rejected_lanes))
                self.metrics.inc("cache_misses", np.count_nonzero(rejected & (counts[pending] == 1)))
                self.anchors[rejected_lanes] = np.clip(centers[rejected], -self.bound, self.bound)
                self._refresh_lanes(rejected_lanes)
            pending = pending[rejected]
        self.metrics.counters["max_proposals_one_move"] = max(
            self.metrics.counters["max_proposals_one_move"], int(np.max(counts)))
        self.metrics.inc("conditional_batches")
        self.metrics.times["conditional_draw"] += perf_counter()-start
        return result

    def _advance_lanes(self, x, schedule):
        for _ in range(schedule["N_actual"]):
            for lo in range(0, self.m, self.lane_batch_size):
                hi = min(self.m, lo+self.lane_batch_size)
                lanes = np.arange(lo, hi)
                tilts = -.5+(lanes+.5)/self.m
                G = normal_standard((hi-lo, self.d), self.rng, self.metrics)
                r = (schedule["inveta"]*self.states[lo:hi]+self.a_obs
                     +tilts[:, None]*x+math.sqrt(schedule["inveta"])*G)
                if not np.all(np.isfinite(r)):
                    self._fail("Nonfinite batched combined auxiliary vectors")
                self.states[lo:hi] = self._conditional_batch(r, lanes, schedule)
                self.metrics.inc("proximal_steps", hi-lo)
                self.metrics.inc("warm_lane_steps", hi-lo)
        self.metrics.inc("recorded_lanes", self.m)

    def _predict_unchecked(self, x):
        # This keeps the frozen outer reference's arithmetic and bookkeeping;
        # the selected differences are only cache layout and _advance_lanes.
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
        pm, pn = schedule["m_prescribed"], schedule["N_prescribed"]
        planned = None if pm is None or pn is None else pm*pn
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
            self._advance_lanes(x, schedule)
            # Bounded pair-first accumulation, retaining initial cancellation.
            score_sum = 0.
            half = self.m//2
            for lo in range(0, half, self.lane_batch_size):
                hi = min(half, lo+self.lane_batch_size)
                paired = self.states[lo:hi]+self.states[self.m-hi:self.m-lo][::-1]
                score_sum += float(np.sum(paired@x))
            raw_logit = score_sum/self.m
            self.reference_round = self.history.n
            self.reference_gamma = schedule["gamma"]
            self.gap_norm = math.sqrt(norm2)
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
        return (super().state_bytes()
                +(0 if self.anchors is None else self.anchors.nbytes)
                +(0 if self.gradients is None else self.gradients.nbytes))

    def diagnostics(self):
        out = super().diagnostics()
        out.update(tangent_cache="per_lane", lane_batch_size=self.lane_batch_size,
                   extra_lane_cache_bytes=self.anchors.nbytes+self.gradients.nbytes,
                   phi_stored_per_lane=False)
        return out
