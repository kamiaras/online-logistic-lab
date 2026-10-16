"""Eight curvature summaries certify some exact acceptances without history.

Same paired population, warm schedule, conditional kernels and c4 output guard
as the frozen batched reference. Each lane stores eight extra diagonal weight
vectors, grouped by the absolute score at its own tangent. A certified upper
bound below one independent exponential threshold accepts immediately; otherwise
the full residual continues with that SAME threshold. No Poisson test is used.
Summary maintenance and its additional 8*M*d cells are explicitly charged.
"""
from time import perf_counter
import math

import numpy as np

from .batched_persistent_log_odds import BatchedPersistentLogOdds
from .proximal import _positive_integer
from .shared_tangent_primitives import open_uniform


CURVATURE_BIN_EDGES = np.array([0., 1., 2., 4., 8., 16., 32., 64.])


class CurvaturePersistentLogOdds(BatchedPersistentLogOdds):
    @staticmethod
    def preflight(d, B, T, config=None):
        cfg = dict(config or {})
        memory_limit = cfg.pop("max_state_bytes", None)
        pre = BatchedPersistentLogOdds.preflight(d, B, T, cfg)
        m = pre["actual_lanes"]
        extra = None if m is None else 8*8*m*pre["d"]
        state = None if extra is None else pre["estimated_state_bytes"]+extra
        reasons = list(pre["budget_reasons"])
        if state is not None and state > np.iinfo(np.intp).max:
            reasons.append("curvature-summary state exceeds the platform index range")
        if memory_limit is not None:
            limit = _positive_integer("max_state_bytes", memory_limit)
            if state is None or state > limit:
                reasons.append("estimated curvature-summary state exceeds max_state_bytes")
        pre.update(estimated_state_bytes=state, extra_curvature_summary_bytes=extra,
                   memory_representation="raw features, paired states, lane tangents and eight diagonal curvature summaries",
                   curvature_bin_edges=CURVATURE_BIN_EDGES.tolist(),
                   acceptance="curvature upper-bound squeeze then same-threshold full residual",
                   feasible=not reasons, budget_reasons=reasons)
        return pre

    def __init__(self, d, B, T, seed, config=None):
        self.curvature_summaries = None
        super().__init__(d, B, T, seed, config)
        start = perf_counter()
        self.curvature_summaries = np.zeros((self.m, 8, self.d))
        self._peak_state_bytes = self.state_bytes()
        self.metrics.times["initialization"] += perf_counter()-start

    def _append_feature(self, x):
        # Update gradient and its summary using the same current-row score.
        # All zero-anchor summaries are maintained from t=1, before lazy prior
        # initialization, so no historical initialization pass is required.
        start = perf_counter()
        self.history.append(x)
        absolute = np.abs(x)
        norm2, row_l1 = float(x@x), float(np.sum(absolute))
        row_weights = row_l1*absolute
        self.E += norm2/4.
        self.E1 += row_l1*row_l1/4.
        score_bound = 2.*self.B*row_l1
        row_a = math.exp(-score_bound-2.*math.log1p(math.exp(-score_bound)))
        if row_a == 0. and row_l1:
            self.metrics.inc("row_curvature_underflows")
        self.curvature_diagonal += row_a*(2.*x*x-absolute*row_l1)
        self.curvature_low = max(0., self.curvature_low, float(np.min(self.curvature_diagonal)))
        for lo in range(0, self.m, self.lane_batch_size):
            hi = min(self.m, lo+self.lane_batch_size)
            lanes = np.arange(lo, hi)
            scores = self.anchors[lo:hi]@x
            self.gradients[lo:hi] += (.5*np.tanh(scores/2.))[:, None]*x
            bins = np.searchsorted(CURVATURE_BIN_EDGES, np.abs(scores), side="right")-1
            self.curvature_summaries[lanes, bins] += row_weights
            if (not np.all(np.isfinite(self.gradients[lo:hi]))
                    or not np.all(np.isfinite(self.curvature_summaries[lanes, bins]))):
                self._fail("Nonfinite appended gradient or curvature summary")
        self.gap_norm += math.sqrt(norm2)
        if (not np.all(np.isfinite(self.curvature_diagonal))
                or not all(math.isfinite(z) for z in (self.E, self.E1, self.gap_norm))):
            self._fail("Nonfinite feature energy or curvature")
        self.metrics.inc("cache_appends", self.m)
        self.metrics.inc("cache_append_rows", self.m)
        self.metrics.inc("summary_append_lane_rows", self.m)
        self.metrics.inc("summary_append_coordinates", self.m*self.d)
        self.metrics.inc("estimated_matmul_flops", 2*self.m*self.d)
        self.metrics.times["cache_append"] += perf_counter()-start

    def _refresh_lanes(self, lanes):
        lanes = np.asarray(lanes, dtype=np.intp)
        if len(lanes) == 0:
            return
        if len(lanes) > self.lane_batch_size:
            raise ValueError("Refresh exceeds bounded lane batch")
        start = perf_counter()
        anchors = self.anchors[lanes]
        gradients = np.zeros_like(anchors)
        summaries = np.zeros((len(lanes), 8, self.d))
        for xb in self.history.batches():
            scores = xb@anchors.T
            gradients += (.5*np.tanh(scores/2.)).T@xb
            absolute = np.abs(xb)
            row_weights = absolute*np.sum(absolute, axis=1)[:, None]
            bins = np.searchsorted(CURVATURE_BIN_EDGES, np.abs(scores), side="right")-1
            # One addition of each row vector per rejected lane, not eight
            # separate dense matrix passes or eight scans of feature history.
            for j in range(len(lanes)):
                np.add.at(summaries[j], bins[:, j], row_weights)
            pairs = len(xb)*len(lanes)
            self.metrics.inc("gradient_refresh_rows", pairs)
            self.metrics.inc("summary_rebuild_lane_rows", pairs)
            self.metrics.inc("summary_rebuild_coordinates", pairs*self.d)
            self.metrics.inc("summary_rebuild_history_rows", len(xb))
            self.metrics.inc("estimated_matmul_flops", 4*pairs*self.d)
            self.metrics.inc("tanh_elements", pairs)
        if not np.all(np.isfinite(gradients)) or not np.all(np.isfinite(summaries)):
            self._fail("Nonfinite rebuilt gradient or curvature summary")
        self.gradients[lanes] = gradients
        self.curvature_summaries[lanes] = summaries
        self.metrics.inc("gradient_refreshes", len(lanes))
        self.metrics.inc("gradient_refresh_batches")
        self.metrics.inc("summary_rebuilds", len(lanes))
        self.metrics.times["gradient_refresh"] += perf_counter()-start

    def _curvature_majorant(self, lanes, proposals):
        lanes = np.asarray(lanes, dtype=np.intp)
        if len(lanes) > self.lane_batch_size or proposals.shape != (len(lanes), self.d):
            raise ValueError("Invalid summary-query batch")
        start = perf_counter()
        difference = proposals-self.anchors[lanes]
        squared = difference*difference
        radius = self.R*np.sqrt(np.sum(squared, axis=1))
        lower_scores = np.maximum(0., CURVATURE_BIN_EDGES[None, :]-radius[:, None])
        exponential = np.exp(-lower_scores)  # lower_scores <=64, no tail underflow.
        beta = exponential/np.square(1.+exponential)
        weighted = np.einsum("nkd,nd->nk", self.curvature_summaries[lanes], squared)
        C = .5*np.sum(beta*weighted, axis=1)
        if not np.all(np.isfinite(C)) or np.any(C < 0.):
            self._fail("Nonfinite or negative curvature majorant")
        self.metrics.inc("summary_queries", len(lanes))
        self.metrics.inc("summary_query_coordinates", 8*len(lanes)*self.d)
        self.metrics.times["summary_query"] += perf_counter()-start
        return C

    def _residual_test_batch(self, lanes, proposals, thresholds=None):
        lanes = np.asarray(lanes, dtype=np.intp)
        if thresholds is None:
            thresholds = -np.log(open_uniform(self.rng, len(lanes), self.metrics))
        else:
            thresholds = np.asarray(thresholds, dtype=float)
            if (thresholds.shape != (len(lanes),) or np.any(~np.isfinite(thresholds))
                    or np.any(thresholds < 0.)):
                raise ValueError("Thresholds must be a finite nonnegative batch vector")
        C = self._curvature_majorant(lanes, proposals)
        accepted = thresholds >= C
        certified = int(np.count_nonzero(accepted))
        self.metrics.inc("summary_certified_accepts", certified)
        # On these same proposals/thresholds the full backend necessarily reads
        # all t rows, because every partial residual is <=R<=C<=threshold.
        self.metrics.inc("history_rows_avoided_by_squeeze", certified*self.history.n)
        self.metrics.inc("estimated_matmul_flops_saved_by_squeeze", 4*certified*self.history.n*self.d)
        unresolved = np.flatnonzero(~accepted)
        self.metrics.inc("summary_full_continuations", len(unresolved))
        if len(unresolved):
            accepted[unresolved] = super()._residual_test_batch(
                lanes[unresolved], proposals[unresolved], thresholds[unresolved])
        return accepted

    def state_bytes(self):
        return super().state_bytes()+(0 if self.curvature_summaries is None
                                     else self.curvature_summaries.nbytes)

    def diagnostics(self):
        out = super().diagnostics()
        queries = self.metrics.counters["summary_queries"]
        certified = self.metrics.counters["summary_certified_accepts"]
        out.update(acceptance="curvature_summary_squeeze",
                   curvature_bin_edges=CURVATURE_BIN_EDGES.tolist(),
                   extra_curvature_summary_bytes=self.curvature_summaries.nbytes,
                   summary_accept_fraction=None if queries == 0 else certified/queries,
                   squeeze_uses_same_threshold=True)
        return out
