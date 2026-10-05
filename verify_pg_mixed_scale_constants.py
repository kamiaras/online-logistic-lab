"""Exact rational certificates for two scalar integral bounds; no learning run."""
from fractions import Fraction as F
from math import factorial
import json
from pathlib import Path


def exp_interval(x, order=40):
    low = sum((x**j / factorial(j) for j in range(order + 1)), F(0))
    next_term = x ** (order + 1) / factorial(order + 1)
    high = low + next_term / (1 - x / (order + 2))
    return low, high


n = 16
f_lower = []
a_upper = []
for j in range(n + 1):
    z = F(2 * j, n)
    lo, hi = exp_interval(z)
    f_lower.append(z / (1 + hi))
    # t/(1+t)^2 is decreasing for t>=1.
    a_upper.append(z * z * lo / (1 + lo) ** 2)
m_lower = sum((f_lower[j - 1] + f_lower[j] for j in range(1, n + 1)), F(0)) / (2 * n)
A_upper = sum(a_upper[1:], F(0)) / n
assert m_lower > F(2184, 10000)
assert A_upper < F(2124, 10000)
floor = F(1, 1000)
weak = 2 * F(2124, 10000) / ((1 - 2 * floor) * (F(2184, 10000) - floor))
strong = 48 * floor**2 / ((1 - 2 * floor)**5 * (1 - floor))
assert weak < F(1958, 1000)
assert strong < F(5, 100000)
assert weak + strong < 2
result = {
    "method": "exact Fraction arithmetic with 40th-order exponential Taylor remainder intervals",
    "trapezoid_intervals": n,
    "m_lower_display": float(m_lower),
    "A_upper_display": float(A_upper),
    "weak_coefficient_upper_display": float(weak),
    "strong_coefficient_upper_display": float(strong),
    "all_strict_inequalities_verified_exactly": True,
    "scope": "scalar analytic constants only; no finite-n gap or learning benchmark",
}
output = Path(__file__).with_name("PG_MIXED_SCALE_CONSTANTS_CHECK.json")
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result))
