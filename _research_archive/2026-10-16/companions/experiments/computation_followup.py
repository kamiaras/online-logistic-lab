"""Practical-only original-step control; frozen algorithm modules are unchanged."""
from .curvature_persistent_log_odds import CurvaturePersistentLogOdds
from .shared_tangent import _mixing_counts


class LegacyStepCurvatureLogOdds(CurvaturePersistentLogOdds):
    @staticmethod
    def preflight(d, B, T, config=None):
        cfg = dict(config or {})
        if cfg.get("mode", "practical") != "practical":
            raise ValueError("Legacy-step follow-up is a practical-only control")
        out = CurvaturePersistentLogOdds.preflight(d, B, T, cfg)
        out["step_control"] = "inveta=lambda+2*d*E; practical-only"
        out["prescribed_schedule_certificate"] = False
        return out

    def _schedule(self):
        schedule = super()._schedule()
        P = 2*self.d*self.E
        inveta, mu = self.lam+P, 2*self.lam+P
        N, _, log_N, _ = _mixing_counts(schedule["H_warm"], schedule["gamma"], inveta)
        cap = int(self.config.get("transitions", 1))
        schedule.update(P_current=schedule["P"], P=P, inveta=inveta, eta=1/inveta,
                        mu=mu, N_prescribed=N, log_N_prescribed=log_N,
                        N_actual=min(N if N is not None else cap, cap),
                        step_control="original", prescribed_schedule_certificate=False)
        return schedule
