"""Finite scalar quadrature/certificate arithmetic; no learning benchmark.

The analytic composite-Simpson remainder is explicit below. Decimal is used
only to reproduce the finite sums to much higher accuracy than that remainder.
This does not implement a general finite-bit sampler.
"""
from decimal import Decimal as D, getcontext
from fractions import Fraction
import json

getcontext().prec = 65


def atan_bounds(q, count=40):
    total = sum((Fraction((-1) ** j, (2*j+1)*q**(2*j+1))
                 for j in range(count)), Fraction(0))
    nxt = Fraction((-1) ** count, (2*count+1)*q**(2*count+1))
    return min(total, total+nxt), max(total, total+nxt)


def dec(q):
    return D(q.numerator)/D(q.denominator)


lo5, hi5 = atan_bounds(5)
lo239, hi239 = atan_bounds(239)
pi_lo = 16*lo5-4*hi239
pi_hi = 16*hi5-4*lo239
pi = (dec(pi_lo)+dec(pi_hi))/2
H, a = D('25.001'), D('1.55')
N = 16384
step = 2*a/N
normal = (H/(2*pi)).sqrt()
fourth = D(10)*(113+4*111*87+3*111**2+6*111**3+111**4)
error = 2*a/180*step**4*fourth
sums = {(p, b): D(0) for p in (1, 9) for b in (0, 1)}
for j in range(N+1):
    t = -a+j*step
    cosh = ((t/2).exp()+(-t/2).exp())/2
    logcosh = cosh.ln()
    g = 100*(t*t/8-logcosh)
    weight = 1 if j in (0, N) else (4 if j % 2 else 2)
    for p in (1, 9):
        f0 = -H*t*t/2+p*g
        sums[p, 0] += weight*f0.exp()
        sums[p, 1] += weight*(f0+t-1/(2*H)).exp()
integrals = {key: normal*step/3*value for key, value in sums.items()}
# The Gaussian core probability has a direct union-bound certificate.
q0 = 1-999*2*(-H*a*a/2).exp()-2*(-H*(a-1/H)**2/2).exp()
# The analytic remainder dwarfs Decimal reproduction error; retain an extra
# 1e-40 guard rather than interpreting the raw finite sums as exact integrals.
err = error+D('1e-40')
lower_log_Z = 999*(integrals[1, 0]-err).ln()+(integrals[1, 1]-err).ln()
upper_log_Up = (999*(integrals[9, 0]+err).ln()
                +(integrals[9, 1]+err).ln()-D('22.5')-q0.ln())

f = D('.001')*a*a/2+100*(((a/2).exp()+(-a/2).exp())/2).ln()
fp = D('.001')*a+50*((a/2).exp()-(-a/2).exp())/((a/2).exp()+(-a/2).exp())
Zbase = (2*pi/H).sqrt()
tail0 = 2*(-f).exp()/fp/Zbase
tail1 = (((-f+a).exp()/(fp-1)+(-f-a).exp()/(fp+1))
         /Zbase/(1/(2*H)).exp())
tail = 999*tail0+tail1
p, clip = 9, D('4e-10')
cp = D(8)**8/D(9)**9
zmin = D('-.001').exp()
K = ((cp*D('3.70').exp()/(clip*zmin)).ln()/8).exp()
Nbar = K/(zmin*(1-clip))
cap = int((Nbar*(1/clip).ln()).to_integral_value(rounding='ROUND_CEILING'))

out = {
    'subintervals_per_integral': N,
    'simpson_uniform_error': str(error),
    'scalar_integrals': {f'p{p}_b{b}': str(v) for (p,b),v in integrals.items()},
    'lower_log_raw_evidence': str(lower_log_Z),
    'upper_log_centered_p9_moment': str(upper_log_Up),
    'gaussian_core_probability_lower': str(q0),
    'original_posterior_omitted_mass_upper': str(tail),
    'clipping_threshold': str(K),
    'expected_core_proposals_upper': str(Nbar),
    'expected_raw_gaussians_upper': str(Nbar/q0),
    'proposal_cap': cap,
    'total_tv_upper': str(tail+2*clip),
}
assert lower_log_Z > D('2.499')
assert upper_log_Up < D('3.70')
assert tail < D('1.498e-10')
assert q0 > 1-D('1.82e-10')
assert Nbar/q0 < 17
assert cap == 348
print(json.dumps(out, indent=2))
