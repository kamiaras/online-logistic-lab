"""Change only initial lane dependence in the controlled old-schedule harness."""
from .shared_ablation import SharedAblation


class AntitheticSharedAblation(SharedAblation):
    def __init__(self, d, B, T, seed, config=None):
        super().__init__(d, B, T, seed, config)
        if self.layout != "lanes" or self.persistence != "warm" or self.m % 2:
            raise ValueError("This controlled extension requires even persistent lanes")
        self._paired_prior_bank = []
        self._paired_prior_index = 0

    def _prior(self):
        j = self._paired_prior_index
        self._paired_prior_index += 1
        if j < self.m // 2:
            theta = super()._prior()
            self._paired_prior_bank.append(theta.copy())
            self.metrics.inc("independent_prior_vectors")
            return theta
        self.metrics.inc("prior_initializations")
        self.metrics.inc("antithetic_prior_vectors")
        theta = -self._paired_prior_bank[self.m - 1 - j]
        if j == self.m - 1:
            self._paired_prior_bank = []
        return theta
