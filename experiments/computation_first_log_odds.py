"""Selected paired sampler with the original proximal step and curvature squeeze.

The step is inveta=lambda+2*d*E. Its prescribed warm counts and allocation-free
worst-case count bound are recomputed for this step. The provisional practical
default is 16 paired lanes and one transition; these caps and float64 do not
inherit the exact-real prescribed regret theorem. Frozen measured implementations
remain unchanged. All acceptance, cache, pairing and guard code is inherited.
"""
import math

from .curvature_persistent_log_odds import CurvaturePersistentLogOdds
from .shared_tangent import _ceil_from_log, _logaddexp_scalar, _mixing_counts


class ComputationFirstLogOdds(CurvaturePersistentLogOdds):
    """Original step, persistent antithetic lanes, exact curvature-summary test."""

    @staticmethod
    def preflight(d, B, T, config=None):
        cfg = {"lanes": 16, "transitions": 1, **dict(config or {})}
        pre = CurvaturePersistentLogOdds.preflight(d, B, T, cfg)
        p = dict(pre["prescribed"])
        H, kappa = p["H_max"], p["kappa"]
        # log(1+z)>=z/(1+z) gives
        # N_t <= 1+H_max+H_max*d*kappa*T/4 for inveta=lambda+2*d*E_t.
        # Thus the old uniform coefficient max(d,2) is replaced by 2*d.
        if pre["R"] == 0.:
            p.update(N_max=0, log_N_max=None)
        else:
            log_kappa = math.log(kappa) if kappa else -math.inf
            log_N = _logaddexp_scalar(
                math.log1p(H), math.log(H)+math.log(2*pre["d"])
                +log_kappa+math.log(pre["T"])-math.log(8.))
            p.update(N_max=_ceil_from_log(log_N), log_N_max=log_N)
        mode, m = pre["mode"], pre["actual_lanes"]
        cap = int(cfg["transitions"])
        if pre["R"] == 0.:
            N, moves = 0, 0
        else:
            N = p["N_max"] if mode == "prescribed" else min(p["N_max"] or cap, cap)
            moves = None if m is None or N is None else pre["T"]*m*N
        # Preserve all parent shape/storage validations, replacing only the
        # transition-budget diagnostic computed with the previous step bound.
        reasons = [reason for reason in pre["budget_reasons"]
                   if not reason.startswith("worst-case transitions:")]
        count_reason = "prescribed count exceeds the representable schedule range"
        if mode == "prescribed" and N is None and count_reason not in reasons:
            reasons.append(count_reason)
        if ((mode == "prescribed" or "max_total_moves" in cfg)
                and moves is not None and moves > pre["max_total_moves"]):
            reasons.append(f"worst-case transitions: {moves} > {pre['max_total_moves']}")
        pre.update(prescribed=p, actual_max_transitions=N, estimated_total_moves=moves,
                   step_rule="inveta=lambda+2*d*E", uniform_step_coefficient=2*pre["d"],
                   practical_default_lanes=16, practical_default_transitions=1,
                   feasible=not reasons, budget_reasons=reasons)
        return pre

    def __init__(self, d, B, T, seed, config=None):
        cfg = {"lanes": 16, "transitions": 1, **dict(config or {})}
        super().__init__(d, B, T, seed, cfg)

    def _schedule(self):
        P = 2*self.d*self.E
        inveta, mu, gamma = self.lam+P, 2*self.lam+P, self.lam+self.curvature_low
        if not all(math.isfinite(z) and z > 0. for z in (inveta, mu, gamma)):
            self._fail("Unrepresentable proximal or curvature precision")
        log_K = (2.*math.log(self.gap_norm)-math.log(2.)-math.log(self.reference_gamma)
                 if self.gap_norm else -math.inf)
        H = (_logaddexp_scalar(0., log_K)-math.log(2.)
             -2.*self._preflight["prescribed"]["log_epsilon"])
        if self.R == 0.:
            N, log_N, actual = 0, None, 0
        else:
            N, _, log_N, _ = _mixing_counts(H, gamma, inveta)
            if self.mode == "prescribed":
                if N is None:
                    self._fail("Prescribed warm mixing count is not representable")
                actual = N
            else:
                cap = int(self.config["transitions"])
                actual = min(N if N is not None else cap, cap)
        return dict(t=self.history.n, E=self.E, E1=self.E1, P=P,
                    inveta=inveta, eta=1./inveta, mu=mu, gamma=gamma,
                    reference_round=self.reference_round,
                    reference_gamma=self.reference_gamma, gap_norm=self.gap_norm,
                    K_gap=math.exp(log_K) if log_K < 700. else None,
                    log_K_gap=None if log_K == -math.inf else log_K,
                    H_warm=H, N_prescribed=N, log_N_prescribed=log_N, N_actual=actual,
                    step_rule="inveta=lambda+2*d*E")

