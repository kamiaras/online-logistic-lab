"""Predictable confidence adaptation for one evolving log-odds predictor.

All state is scalar. The base predictor is run once; its output is transformed
at fixed scales and combined in probability space. Updates use past labels
only. In exact arithmetic the total loss is at most the base loss + log(2)
and at most T*log(2) + log(4). No finite-precision certificate is asserted.
"""
import math


SCALES = (1., 0., .5, .25, .125, .0625, .03125, .015625, .0078125, .00390625)
PRIORS = (.5, .25) + (.03125,) * 8


def softplus(z):
    return max(z, 0.) + math.log1p(math.exp(-abs(z)))


def logsumexp(values):
    values = tuple(values)
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(v - maximum) for v in values))


class OnlineConfidence:
    """Call predict(base_logit), then observe(y), once for each round.

    Repeat predict calls for the same pending logit are idempotent. Probabilities
    and label updates are computed in log space so an expert is not discarded
    merely because its ordinary floating-point weight would underflow.
    """

    def __init__(self):
        self.log_weights = [math.log(w) for w in PRIORS]
        self.rounds = 0
        self.cumulative_loss = 0.
        self._pending = None

    def predict(self, base_logit):
        z = float(base_logit)
        if not math.isfinite(z):
            raise ValueError("The base logit must be finite")
        if self._pending is not None:
            if z != self._pending[0]:
                raise ValueError("Observe the pending prediction first")
            return self._pending[1]
        plus = [-softplus(-a*z) for a in SCALES]
        minus = [-softplus(a*z) for a in SCALES]
        lp = logsumexp(w+p for w, p in zip(self.log_weights, plus))
        lm = logsumexp(w+p for w, p in zip(self.log_weights, minus))
        # Numerical normalization; in exact arithmetic this normalizer is zero.
        normalizer = logsumexp((lp, lm))
        lp, lm = lp-normalizer, lm-normalizer
        output = lp-lm
        self._pending = (z, output, plus, minus, lp, lm)
        return output

    def observe(self, y):
        if y not in (-1, 1):
            raise ValueError("The label must be -1 or +1")
        if self._pending is None:
            raise ValueError("Predict before observing a label")
        _, _, plus, minus, lp, lm = self._pending
        likelihood = plus if y == 1 else minus
        updated = [w+p for w, p in zip(self.log_weights, likelihood)]
        normalizer = logsumexp(updated)
        self.log_weights = [w-normalizer for w in updated]
        loss = -lp if y == 1 else -lm
        self.cumulative_loss += loss
        self.rounds += 1
        self._pending = None
        return loss

    def diagnostics(self):
        return dict(scales=SCALES, initial_weights=PRIORS,
                    log_weights=list(self.log_weights),
                    weights=[math.exp(w) for w in self.log_weights],
                    rounds=self.rounds, cumulative_loss=self.cumulative_loss,
                    exact_arithmetic_base_loss_overhead=math.log(2.),
                    exact_arithmetic_neutral_loss_overhead=math.log(4.),
                    extra_sampler_moves=0, floating_point_certificate=False)


class ConfidenceAdaptiveLogOdds:
    """Opt-in confidence refinement of the frozen ComputationFirstLogOdds.

    Import the numerical sampler only when constructing the online model.
    Confidence decisions never feed back into its state, cached gradient,
    schedule, RNG or label statistic. Its diagnostics remain nested and retain
    the original meaning of raw and guarded sampler logits.
    """

    def __init__(self, d, B, T, seed, config=None):
        from .computation_first_log_odds import ComputationFirstLogOdds
        self.base = ComputationFirstLogOdds(d, B, T, seed, config)
        self.confidence = OnlineConfidence()

    def predict(self, x):
        return self.confidence.predict(self.base.predict(x))

    def observe(self, x, y):
        # The base validates the feature/label and updates from the true label.
        # A failed validation must not mutate the confidence weights.
        self.base.observe(x, y)
        self.confidence.observe(y)

    def state_bytes(self):
        """Owned ndarray bytes; the additional O(10) Python scalars are separate."""
        return self.base.state_bytes()

    def diagnostics(self):
        return dict(base=self.base.diagnostics(), confidence=self.confidence.diagnostics(),
                    state_bytes=self.state_bytes(),
                    additional_state="O(10) Python scalars and bounded pending scratch",
                    sampler_feedback=False, implementation_status="opt-in exploratory refinement")
