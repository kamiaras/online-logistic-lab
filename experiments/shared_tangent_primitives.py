"""Uniform-only Gaussian draws and raw/prefix history for the shared-tangent loop.

The probability identities are those of sections 2, 3 and 5 of the supplied
manuscript. This is float64 experimental code, not a finite-bit certificate.
Raw features are retained separately from sampling prefixes, as explicitly
allowed in section 5, so recovering a feature never subtracts close prefixes.
"""
from __future__ import annotations

import numpy as np


_CHUNK = 4096
_EPS = np.finfo(np.float64).eps


def _inc(metrics, key, count=1):
    if metrics is not None:
        metrics.inc(key, count)


def _fail(metrics, message):
    _inc(metrics, "numerical_failures")
    raise FloatingPointError(message)


def open_uniform(rng, size=None, metrics=None):
    """Independent open-unit uniforms; redraw the representable endpoint zero."""
    out = np.asarray(rng.random(size), dtype=np.float64)
    _inc(metrics, "uniform_draws", out.size)
    if np.any(~np.isfinite(out)) or np.any((out < 0) | (out >= 1)):
        _fail(metrics, "Uniform source returned a value outside [0,1)")
    if out.ndim == 0:
        while float(out) == 0:
            out = np.asarray(rng.random(), dtype=np.float64)
            _inc(metrics, "uniform_draws")
            _inc(metrics, "uniform_zero_redraws")
            if not np.isfinite(out) or out < 0 or out >= 1:
                _fail(metrics, "Uniform source returned a value outside [0,1)")
        return float(out)
    zero = out == 0
    while np.any(zero):
        count = int(np.count_nonzero(zero))
        replacement = np.asarray(rng.random(count), dtype=np.float64)
        _inc(metrics, "uniform_draws", count)
        _inc(metrics, "uniform_zero_redraws", count)
        if np.any(~np.isfinite(replacement)) or np.any((replacement < 0) | (replacement >= 1)):
            _fail(metrics, "Uniform source returned a value outside [0,1)")
        out[zero] = replacement
        zero = out == 0
    return out


def _bounded_roundoff(z, lo, hi, metrics):
    if np.any(~np.isfinite(z)):
        _fail(metrics, "Nonfinite Gaussian proposal")
    bad = (z < lo) | (z > hi)
    if np.any(bad):
        scale = np.maximum(np.abs(z), np.maximum(np.abs(lo), np.abs(hi)))
        tolerance = 32 * _EPS * scale + 8 * np.nextafter(0., 1.)
        if np.any(np.maximum(lo - z, z - hi) > tolerance):
            _fail(metrics, "Gaussian proposal escaped its interval beyond roundoff")
        _inc(metrics, "roundoff_boundary_clamps", np.count_nonzero(bad))
        z = np.minimum(np.maximum(z, lo), hi)
    return z


def _signed_offsets(mode, left_width, right_width, rng, metrics):
    """One exact envelope rule, returning the signed displacement from its mode.

    mode is the standardized normal mode projected onto the interval. Widths
    are measured in standard deviations, and may be +inf for the whole line.
    Returning offsets avoids cancellation when a box mean is far outside it.
    """
    rate = 1. + np.abs(mode)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        left_mass = -np.expm1(-rate * left_width)
        right_mass = -np.expm1(-rate * right_width)
    total = left_mass + right_mass
    if (np.any(~np.isfinite(rate)) or np.any(~np.isfinite(total))
            or np.any(total <= 0)):
        _fail(metrics, "Gaussian envelope has unrepresentable mass or rate")
    left_probability = left_mass / total
    out = np.empty_like(mode)
    pending = np.arange(mode.size)
    for _ in range(10000):
        if pending.size == 0:
            return out
        count = pending.size
        # Consume three independent uniforms even when only one side has mass.
        left = open_uniform(rng, count, metrics) < left_probability[pending]
        mass = np.where(left, left_mass[pending], right_mass[pending])
        if np.any(mass <= 0):
            _fail(metrics, "A zero-mass Gaussian-envelope side was selected")
        z = -np.log1p(-open_uniform(rng, count, metrics) * mass) / rate[pending]
        width = np.where(left, left_width[pending], right_width[pending])
        z = _bounded_roundoff(z, np.zeros_like(z), width, metrics)
        with np.errstate(over="ignore"):
            log_accept = -.5 * (z - 1.) ** 2
        accept = np.log(open_uniform(rng, count, metrics)) <= log_accept
        _inc(metrics, "tn_proposals", count)
        _inc(metrics, "tn_rejections", np.count_nonzero(~accept))
        out[pending[accept]] = np.where(left[accept], -z[accept], z[accept])
        pending = pending[~accept]
    _fail(metrics, "Gaussian envelope exceeded the numerical rejection guard")


def normal_interval(alpha, beta, rng, metrics=None):
    """Broadcast standard normals conditioned on finite intervals or all R.

    Half-infinite intervals are deliberately unsupported: the manuscript uses
    only finite intervals and the full real line. Scalar inputs return float.
    """
    aa, bb = np.broadcast_arrays(np.asarray(alpha, dtype=np.float64),
                                 np.asarray(beta, dtype=np.float64))
    finite = np.isfinite(aa) & np.isfinite(bb) & (aa < bb)
    entire = np.isneginf(aa) & np.isposinf(bb)
    if not np.all(finite | entire):
        raise ValueError("Intervals must have finite alpha < beta or be (-inf,+inf)")
    out = np.empty(aa.size, dtype=np.float64)
    a, b = aa.ravel(), bb.ravel()
    for start in range(0, out.size, _CHUNK):
        stop = min(start + _CHUNK, out.size)
        lo, hi = a[start:stop], b[start:stop]
        mode = np.minimum(np.maximum(np.zeros_like(lo), lo), hi)
        with np.errstate(over="ignore"):
            left_width, right_width = mode - lo, hi - mode
        offset = _signed_offsets(mode, left_width, right_width, rng, metrics)
        values = mode + offset
        _inc(metrics, "normal_offset_rounding_stalls",
             np.count_nonzero((offset != 0) & (values == mode)))
        out[start:stop] = _bounded_roundoff(values, lo, hi, metrics)
    _inc(metrics, "normal_draws", out.size)
    result = out.reshape(aa.shape)
    return float(result) if result.ndim == 0 else result


def normal_standard(size, rng, metrics=None):
    """Standard normals through precisely the same two-sided envelope."""
    shape = () if size is None else size
    empty = np.empty(shape, dtype=np.float64)
    return normal_interval(np.broadcast_to(-np.inf, empty.shape),
                           np.broadcast_to(np.inf, empty.shape), rng, metrics)


def box_normal(center, sd, bound, rng, metrics=None):
    """N(center,sd**2) conditioned on [-bound,bound], coordinatewise.

    Samples are formed as clipped physical mode plus a scaled tail offset,
    rather than center + sd * (standardized mode + offset). The latter loses
    the whole interval when a distant center makes its endpoints coalesce.
    """
    cc, ss, bb = np.broadcast_arrays(np.asarray(center, dtype=np.float64),
                                     np.asarray(sd, dtype=np.float64),
                                     np.asarray(bound, dtype=np.float64))
    if (np.any(~np.isfinite(cc)) or np.any(~np.isfinite(ss)) or np.any(ss <= 0)
            or np.any(~np.isfinite(bb)) or np.any(bb <= 0)):
        raise ValueError("Box centers must be finite; standard deviations and bounds positive finite")
    out = np.empty(cc.size, dtype=np.float64)
    centers, scales, bounds = cc.ravel(), ss.ravel(), bb.ravel()
    for start in range(0, out.size, _CHUNK):
        stop = min(start + _CHUNK, out.size)
        c, scale, bound_part = centers[start:stop], scales[start:stop], bounds[start:stop]
        mode = np.minimum(np.maximum(c, -bound_part), bound_part)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            standard_mode = (mode - c) / scale
            left_width = (mode + bound_part) / scale
            right_width = (bound_part - mode) / scale
        if np.any(~np.isfinite(standard_mode)):
            _fail(metrics, "Box Gaussian standardized mode is not representable")
        offset = _signed_offsets(standard_mode, left_width, right_width, rng, metrics)
        displacement = scale * offset
        values = mode + displacement
        _inc(metrics, "normal_offset_rounding_stalls",
             np.count_nonzero((offset != 0) & (values == mode)))
        out[start:stop] = _bounded_roundoff(values, -bound_part, bound_part, metrics)
    _inc(metrics, "normal_draws", out.size)
    result = out.reshape(cc.shape)
    return float(result) if result.ndim == 0 else result


class WeightedHistory:
    """Block-allocated raw features and weighted categorical prefix sums.

    Valid state is exactly the first n rows. An append performs validation and
    allocations before committing n, so allocation/numerical failures leave
    prior valid rows and their last cumulative weight intact.
    """

    def __init__(self, d, block_size=128, metrics=None):
        for name, value in (("d", d), ("block_size", block_size)):
            if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
                raise ValueError(name + " must be a positive integer")
        self.d, self.block_size = int(d), int(block_size)
        self.metrics, self.n = metrics, 0
        self.blocks = []
        self._zero = np.zeros(self.d, dtype=np.float64)

    @property
    def W(self):
        if self.n == 0:
            return self._zero
        block, position = divmod(self.n - 1, self.block_size)
        return self.blocks[block][1][position]

    @property
    def nbytes(self):
        return self._zero.nbytes + sum(sum(array.nbytes for array in block)
                                      for block in self.blocks)

    def append(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.d,) or np.any(~np.isfinite(x)):
            raise ValueError("History feature must be one finite vector of length d")
        absolute = np.abs(x)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            row_l1 = float(np.sum(absolute))
            increment = row_l1 * absolute
            cumulative = self.W + increment
        if not np.isfinite(row_l1) or np.any(~np.isfinite(cumulative)):
            _fail(self.metrics, "Weighted history prefix overflow")
        if np.any((absolute > 0) & (increment == 0)):
            _fail(self.metrics, "Positive history sampling weight underflowed")
        if np.any((increment > 0) & (cumulative <= self.W)):
            _fail(self.metrics, "Positive history sampling weight is invisible in float64 prefix")
        position = self.n % self.block_size
        if position == 0:
            # Keep partially allocated arrays local until all three exist.
            raw = np.empty((self.block_size, self.d), dtype=np.float64)
            prefix = np.empty((self.block_size, self.d), dtype=np.float64)
            norms = np.empty(self.block_size, dtype=np.float64)
            raw[0], prefix[0], norms[0] = x, cumulative, row_l1
            self.blocks.append((raw, prefix, norms))
        else:
            raw, prefix, norms = self.blocks[-1]
            raw[position], prefix[position], norms[position] = x, cumulative, row_l1
        self.n += 1
        _inc(self.metrics, "weighted_history_appends")
        _inc(self.metrics, "weighted_history_coordinates", self.d)
        return self.W

    def _position(self, i):
        if isinstance(i, (bool, np.bool_)) or int(i) != i or not 0 <= i < self.n:
            raise IndexError("History row is outside the valid prefix")
        return divmod(int(i), self.block_size)

    def row(self, i):
        block, position = self._position(i)
        return self.blocks[block][0][position]

    def row_l1(self, i):
        block, position = self._position(i)
        return float(self.blocks[block][2][position])

    def batches(self):
        remaining = self.n
        for raw, _, _ in self.blocks:
            count = min(remaining, self.block_size)
            if count == 0:
                break
            yield raw[:count]
            remaining -= count

    def sample_row(self, j, threshold):
        """First cumulative prefix strictly above an absolute [0,W[j]) threshold."""
        if isinstance(j, (bool, np.bool_)) or int(j) != j or not 0 <= j < self.d:
            raise IndexError("History sampling coordinate is outside dimension")
        j = int(j)
        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0 <= threshold < self.W[j]:
            raise ValueError("Sampling threshold must be in [0,W[j]) with W[j] > 0")
        lo, hi, comparisons = 0, self.n, 0
        while lo < hi:
            mid = (lo + hi) // 2
            block, position = divmod(mid, self.block_size)
            comparisons += 1
            if self.blocks[block][1][position, j] <= threshold:
                lo = mid + 1
            else:
                hi = mid
        _inc(self.metrics, "weighted_history_lookups")
        _inc(self.metrics, "weighted_history_lookup_comparisons", comparisons)
        return lo
