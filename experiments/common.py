"""Shared stable arithmetic and bounded-temporary full-history evaluation.

FLOP counts are kernel-size estimates, not measured CPU instructions.
"""
from collections import Counter
from time import perf_counter
import numpy as np


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


class Metrics:
    def __init__(self):
        self.counters = Counter()
        self.times = Counter()

    def inc(self, key, n=1):
        self.counters[key] += int(n)

    def as_dict(self):
        return {"counters": dict(self.counters), "times": dict(self.times)}


class History:
    def __init__(self, d, block_size=1024):
        self.d, self.block_size, self.n = d, block_size, 0
        self.blocks = []

    def append(self, x, y):
        j = self.n % self.block_size
        if j == 0:
            self.blocks.append((np.empty((self.block_size, self.d)),
                                np.empty(self.block_size)))
        self.blocks[-1][0][j] = x
        self.blocks[-1][1][j] = y
        self.n += 1

    def batches(self):
        remaining = self.n
        for x, y in self.blocks:
            count = min(remaining, self.block_size)
            yield x[:count], y[:count]
            remaining -= count

    @property
    def nbytes(self):
        return sum(x.nbytes + y.nbytes for x, y in self.blocks)


def potential(theta, B, history, metrics=None, extra_x=None, extra_p=None,
              last_weight=1., want_value=True, want_grad=True):
    """Gaussian negative log density plus history and optional fractional loss.

    theta is (m,d), or (d,); returned values/gradients restore the input rank.
    last_weight applies only to the newest historical observation.
    extra_p is the probability of positive label (scalar or length m).
    """
    start = perf_counter()
    theta = np.asarray(theta, dtype=np.float64)
    single = theta.ndim == 1
    th = np.atleast_2d(theta)
    m, d = th.shape
    values = .5 * np.sum((th / B) ** 2, axis=1) if want_value else None
    grad = th / (B * B) if want_grad else None
    for lo in range(0, m, 64):
        hi = min(m, lo + 64)
        pos = 0
        for x, y in history.batches():
            score = x @ th[lo:hi].T
            signed = -y[:, None] * score
            weight = np.ones(len(y))
            if pos + len(y) == history.n:
                weight[-1] = last_weight
            if want_value:
                values[lo:hi] += np.sum(weight[:, None] * np.logaddexp(0., signed), axis=0)
            if want_grad:
                coeff = -weight[:, None] * y[:, None] * sigmoid(signed)
                grad[lo:hi] += coeff.T @ x
            pos += len(y)
        if extra_x is not None:
            z = th[lo:hi] @ extra_x
            p = np.asarray(extra_p)
            if p.ndim:
                p = p[lo:hi]
            if want_value:
                values[lo:hi] += p * np.logaddexp(0., -z) + (1-p) * np.logaddexp(0., z)
            if want_grad:
                grad[lo:hi] += (sigmoid(z) - p)[:, None] * extra_x
    if metrics is not None:
        pairs = m * (history.n + int(extra_x is not None))
        metrics.inc("potential_calls")
        metrics.inc("history_particle_pairs", m * history.n)
        metrics.inc("logistic_pairs", pairs)
        metrics.inc("estimated_matmul_flops", 2 * pairs * d * (1 + int(want_grad)))
        metrics.inc("sigmoid_elements", pairs * int(want_grad))
        metrics.inc("softplus_elements", pairs * int(want_value))
        metrics.times["potential"] += perf_counter() - start
    if single:
        return (values[0] if want_value else None, grad[0] if want_grad else None)
    return values, grad
