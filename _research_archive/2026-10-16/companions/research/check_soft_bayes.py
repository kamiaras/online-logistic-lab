"""Focused finite mathematical checks for the soft-average Gaussian Bayes proof.

This is not a performance experiment, sampler-convergence experiment, formal
proof, or finite-bit regret certificate. Existing measured sources are untouched.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import binom


ROOT = Path(__file__).resolve().parents[1]
SEED = 9112606
RNG = np.random.default_rng(SEED)
CHECKS: dict[str, object] = {}


def reciprocal_formula(m: int, p: float) -> float:
    if p == 0:
        return 1.0
    if p == 1:
        return 1.0 / (m + 1)
    return -math.expm1((m + 1) * math.log1p(-p)) / ((m + 1) * p)


def finite_expectation(values, probabilities, function):
    terms = []
    for indices in itertools.product(*(range(len(v)) for v in values)):
        probability = math.prod(probabilities[j][k] for j, k in enumerate(indices))
        total = math.fsum(values[j][k] for j, k in enumerate(indices))
        terms.append(probability * function(total))
    return math.fsum(terms)


# 1. Exact finite sums: no Monte Carlo sampling error in these loss checks.
max_reciprocal_residual = 0.0
max_binomial_log_loss_excess = -math.inf
binomial_cases = 0
for m in [1, 2, 3, 10, 100, 1000]:
    for p in [0.0, 1e-300, 1e-100, 1e-12, 1e-4, .01, .1, .5, .9, 1-1e-12, 1.0]:
        counts = np.arange(m + 1)
        mass = binom.pmf(counts, m, p)
        actual = float(np.dot(mass, 1 / (1 + counts)))
        formula = reciprocal_formula(m, p)
        residual = abs(actual - formula) / max(1e-300, formula)
        max_reciprocal_residual = max(max_reciprocal_residual, residual)
        assert residual <= 3e-12
        if p > 0:
            assert formula <= 1 / ((m + 1) * p) * (1 + 2e-14)
            expected_loss = float(np.dot(mass, math.log(m + 2) - np.log1p(counts)))
            bound = -math.log(p) + math.log1p(1 / (m + 1))
            excess = expected_loss - bound
            max_binomial_log_loss_excess = max(max_binomial_log_loss_excess, excess)
            assert excess <= 3e-12
        binomial_cases += 1

discrete_cases = 0
max_chord_excess = -math.inf
max_general_log_loss_excess = -math.inf
for m in range(1, 6):
    for same_mean in [False, True]:
        for _ in range(16):
            common_mean = float(RNG.uniform(.02, .98))
            values, probabilities, means = [], [], []
            for _ in range(m):
                interior = float(RNG.uniform(.01, .99))
                if same_mean:
                    middle_weight = .5 * min(common_mean / interior,
                                             (1-common_mean) / (1-interior))
                    one_weight = common_mean - interior * middle_weight
                    weights = [1-middle_weight-one_weight, middle_weight, one_weight]
                else:
                    weights = RNG.dirichlet(np.ones(3)).tolist()
                vals = [0., interior, 1.]
                mean = float(np.dot(vals, weights))
                values.append(vals)
                probabilities.append(weights)
                means.append(mean)
            actual = finite_expectation(values, probabilities, lambda s: 1/(1+s))
            chord = finite_expectation([[0., 1.]] * m,
                                       [[1-p, p] for p in means], lambda s: 1/(1+s))
            max_chord_excess = max(max_chord_excess, actual-chord)
            assert actual <= chord + 2e-13
            if same_mean:
                formula = reciprocal_formula(m, common_mean)
                assert abs(chord-formula) <= 2e-13
                loss = finite_expectation(values, probabilities,
                                          lambda s: math.log(m+2)-math.log1p(s))
                bound = -math.log(common_mean) + math.log1p(1/(m+1))
                max_general_log_loss_excess = max(max_general_log_loss_excess, loss-bound)
                assert loss <= bound + 2e-13
            discrete_cases += 1
CHECKS['soft_average_finite_expectations'] = {
    'passed': True, 'binomial_cases': binomial_cases, 'independent_discrete_cases': discrete_cases,
    'smallest_positive_p': 1e-300,
    'maximum_relative_binomial_reciprocal_residual': max_reciprocal_residual,
    'maximum_binomial_expected_log_loss_minus_bound': max_binomial_log_loss_excess,
    'maximum_arbitrary_discrete_reciprocal_minus_endpoint_chord': max_chord_excess,
    'maximum_same_mean_discrete_expected_log_loss_minus_bound': max_general_log_loss_excess,
    'method': 'Exact enumeration of finite distributions in float64; no Monte Carlo confidence claim.',
}


# 2. Sensitivity to all score errors at once.
sensitivity_cases = 0
max_gradient_one_norm = 0.0
max_sensitivity_excess = -math.inf
for m in [1, 2, 5, 50, 100]:
    for scale in [1e-6, .1, 1., 10., 1000.]:
        for _ in range(20):
            z = scale * RNG.normal(size=m)
            perturbation = 10**float(RNG.uniform(-7, 2)) * RNG.uniform(-1, 1, m)
            for y in [-1., 1.]:
                probability = expit(y*z)
                gradient_one_norm = float(np.sum(probability*(1-probability)) /
                                          (1+np.sum(probability)))
                max_gradient_one_norm = max(max_gradient_one_norm, gradient_one_norm)
                assert gradient_one_norm <= 1 + 1e-14
                first = math.log(m+2) - math.log1p(float(np.sum(probability)))
                second = math.log(m+2) - math.log1p(float(np.sum(expit(y*(z+perturbation)))))
                excess = abs(first-second) - float(np.max(np.abs(perturbation)))
                max_sensitivity_excess = max(max_sensitivity_excess, excess)
                assert excess <= 2e-13
                sensitivity_cases += 1
CHECKS['joint_maximum_score_sensitivity'] = {
    'passed': True, 'cases': sensitivity_cases,
    'maximum_gradient_one_norm': max_gradient_one_norm,
    'maximum_loss_difference_minus_maximum_score_difference': max_sensitivity_excess,
}


def potential_gradient(x, z):
    scores = x @ z
    values = np.logaddexp(0., scores) - scores/2
    return float(np.sum(values)), x.T @ (expit(scores)-.5)


def k_count(n):
    return math.ceil(math.log(max(1., n/2))/(2*math.log(2)))


# 3. Logistic residuals, matrix Taylor upper bound, residual contraction, trace.
conditional_cases = 0
minimum_residual_scaled = math.inf
minimum_upper_margin_scaled = math.inf
maximum_residual_identity_relative_error = 0.0
maximum_contraction_excess_scaled = -math.inf
maximum_certified_residual_ratio = 0.0
maximum_trace_acceptance_cost = 0.0
for d in [1, 2, 5, 12]:
    for n in [1, 2, 3, 8, 20]:
        for _ in range(12):
            x = 10**float(RNG.uniform(-1.3, .7)) * RNG.normal(size=(n, d))
            matrix = x.T @ x / 4
            energy = float(np.trace(matrix))
            B = 10**float(RNG.uniform(-1, 1))
            lam = B**-2
            mu = 2*lam + 2*energy
            r = mu * RNG.normal(size=d)
            v = r/mu
            val, gradient = potential_gradient(x, v)
            assert np.linalg.norm(gradient) <= math.sqrt(n*energy) + 1e-10
            residual_gradient = gradient.copy()
            for step in range(k_count(n) + 3):
                delta = RNG.normal(size=d)
                proposal = v + delta
                proposal_value, _ = potential_gradient(x, proposal)
                residual = proposal_value-val-float(gradient @ delta)
                upper = float(delta @ matrix @ delta)/2
                scale = 1 + abs(proposal_value) + abs(val) + abs(float(gradient @ delta)) + upper
                minimum_residual_scaled = min(minimum_residual_scaled, residual/scale)
                minimum_upper_margin_scaled = min(minimum_upper_margin_scaled, (upper-residual)/scale)
                assert residual >= -2e-12*scale
                assert residual <= upper + 2e-12*scale

                if step >= k_count(n):
                    ratio = float(np.linalg.norm(residual_gradient)/math.sqrt(mu))
                    maximum_certified_residual_ratio = max(maximum_certified_residual_ratio, ratio)
                    assert ratio <= 1 + 2e-11
                    expected_upper = energy/(2*mu) + float(residual_gradient @ matrix @ residual_gradient)/(2*mu**2)
                    maximum_trace_acceptance_cost = max(maximum_trace_acceptance_cost, expected_upper)
                    assert expected_upper <= energy/mu + 2e-11
                    assert expected_upper <= .5 + 2e-11

                center = (r-gradient)/mu
                next_value, next_gradient = potential_gradient(x, center)
                next_residual = next_gradient-gradient
                direct_residual = mu*center-r+next_gradient
                identity_error = float(np.linalg.norm(next_residual-direct_residual) /
                                       (1+np.linalg.norm(r)+np.linalg.norm(next_gradient)))
                maximum_residual_identity_relative_error = max(maximum_residual_identity_relative_error,
                                                               identity_error)
                assert identity_error <= 2e-13
                contraction_excess = (float(np.linalg.norm(next_residual)) -
                                      energy/mu*float(np.linalg.norm(residual_gradient)))
                scaled_excess = contraction_excess/(1+float(np.linalg.norm(gradient)))
                maximum_contraction_excess_scaled = max(maximum_contraction_excess_scaled, scaled_excess)
                assert scaled_excess <= 2e-12
                v, val, gradient, residual_gradient = center, next_value, next_gradient, next_residual
            conditional_cases += 1
CHECKS['conditional_residual_and_trace_certificate'] = {
    'passed': True, 'random_feature_matrix_cases': conditional_cases,
    'minimum_scaled_rejection_residual': minimum_residual_scaled,
    'minimum_scaled_matrix_upper_bound_margin': minimum_upper_margin_scaled,
    'maximum_relative_fixed_point_residual_identity_error': maximum_residual_identity_relative_error,
    'maximum_scaled_residual_contraction_excess': maximum_contraction_excess_scaled,
    'maximum_residual_norm_over_sqrt_mu_after_k_rejections': maximum_certified_residual_ratio,
    'maximum_certified_expected_quadratic_rejection_cost': maximum_trace_acceptance_cost,
    'implementation_note': 'Matrices are formed here only for checking proof identities; the algorithm stores their trace only.',
}


# 4. Exact density cancellation and an independently integrated acceptance rate.
normal_density_residual = 0.0
for z in [-30., -10., -2., -1., -.1, 0., .1, 1., 2., 10., 30.]:
    log_proposal = -math.log(2)-abs(z)
    log_acceptance = -(abs(z)-1)**2/2
    difference = log_proposal+log_acceptance+z*z/2
    normal_density_residual = max(normal_density_residual, abs(difference+math.log(2)+.5))
assert normal_density_residual <= 1e-12
acceptance, quadrature_error = quad(lambda e: math.exp(-e-(e-1)**2/2), 0., math.inf,
                                  epsabs=1e-13, epsrel=1e-13)
closed_acceptance = math.sqrt(math.pi/(2*math.e))
assert abs(acceptance-closed_acceptance) <= 2e-13
assert 1/acceptance < 3
CHECKS['uniform_only_normal_density'] = {
    'passed': True, 'maximum_log_density_cancellation_residual': normal_density_residual,
    'integrated_acceptance_probability': acceptance,
    'closed_form_acceptance_probability': closed_acceptance,
    'quadrature_reported_absolute_error': quadrature_error,
    'expected_trials': 1/acceptance,
}


# 5. Full-matrix Gaussian comparator KL and variance cancellation.
gaussian_cases = 0
maximum_gaussian_cancellation_relative_error = 0.0
maximum_covariance_trace_identity_error = 0.0
for d in [1, 2, 8]:
    for n in [0, 1, 3, 20]:
        for B in [.01, .1, 1., 10.]:
            for _ in range(8):
                x = RNG.normal(size=(n, d))
                matrix = x.T @ x/4
                lam = B**-2
                precision = lam*np.eye(d)+matrix
                covariance = np.linalg.inv(precision)
                direction = RNG.normal(size=d)
                u = B*float(RNG.uniform(0, 1))*direction/np.linalg.norm(direction)
                sign, logdet_covariance = np.linalg.slogdet(covariance)
                assert sign == 1
                kl = .5*(lam*np.trace(covariance)+lam*float(u@u)-d-
                         (d*math.log(lam)+logdet_covariance))
                smoothness = .5*float(np.trace(covariance@matrix))
                sign, logdet = np.linalg.slogdet(np.eye(d)+B*B*matrix)
                assert sign == 1
                target = .5*lam*float(u@u)+.5*logdet
                error = abs(kl+smoothness-target)/(1+abs(target))
                maximum_gaussian_cancellation_relative_error = max(maximum_gaussian_cancellation_relative_error, error)
                assert error <= 2e-10
                trace_error = abs(float(np.trace(covariance@precision))-d)
                maximum_covariance_trace_identity_error = max(maximum_covariance_trace_identity_error, trace_error)
                assert trace_error <= 2e-9
                gaussian_cases += 1
CHECKS['gaussian_kl_variance_cancellation'] = {
    'passed': True, 'cases': gaussian_cases,
    'maximum_relative_KL_plus_smoothness_cancellation_error': maximum_gaussian_cancellation_relative_error,
    'maximum_covariance_precision_trace_error': maximum_covariance_trace_identity_error,
}


# 6. Explicit rounded schedules and operation-sum majorization, including edges.
maximum_log_inequality_relative_excess = -math.inf
for u in np.geomspace(1e-100, 1., 1000):
    excess = (2*u/(2+u)-math.log1p(float(u)))/float(u)
    maximum_log_inequality_relative_excess = max(maximum_log_inequality_relative_excess, excess)
    assert excess <= 3e-15

parameter_cases = 0
maximum_N_bound_relative_excess = -math.inf
maximum_work_bound_relative_excess = -math.inf
maximum_mixing_budget_excess = -math.inf
maximum_total_prediction_budget = 0.0
largest_evaluated_bound = 0.0
for T in [1, 2, 3, 4, 9, 32, 100]:
    for B in [1e-9, .1, 1., 10., 1e9]:
        for R in [0., 1e-9, .1, 1., 10., 1e9]:
            kappa = (B*R)**2
            H = math.log1p(kappa*T*(T-1)*(T+1)/2)
            J = k_count(T-1)
            mixing_bound = kappa*T*(T-1)/2*math.exp(-H)
            mixing_excess = mixing_bound-1/(T+1)
            maximum_mixing_budget_excess = max(maximum_mixing_budget_excess, mixing_excess)
            assert mixing_excess <= 3e-13
            total_budget = T*math.log1p(1/(T+1))+mixing_bound
            maximum_total_prediction_budget = max(maximum_total_prediction_budget, total_budget)
            assert total_budget <= 1 + 3e-13
            for kind in ['zero_energy', 'maximum_energy', 'random_energy']:
                fractions = (np.zeros(T-1) if kind == 'zero_energy' else
                             np.ones(T-1) if kind == 'maximum_energy' else RNG.uniform(0, 1, T-1))
                energies = R*R*np.cumsum(fractions)/4
                discrete_sum = 0.0
                for n, energy in enumerate(energies, start=1):
                    denominator = math.log1p(1/(1+2*B*B*float(energy)))
                    N = math.ceil(H/denominator)
                    N_bound = 1+1.5*H+H*kappa*n/2
                    excess = (N-N_bound)/(1+N_bound)
                    maximum_N_bound_relative_excess = max(maximum_N_bound_relative_excess, excess)
                    assert excess <= 3e-13
                    assert k_count(n) <= J
                    discrete_sum += n*N*(k_count(n)+2)
                for d in [1, 3, 100]:
                    evaluated = 320*T*d*discrete_sum+1000*d*T*T
                    bound = (320*T*d*(J+2)*((1+1.5*H)*T*(T-1)/2+
                             H*kappa*T*(T-1)*(2*T-1)/12)+1000*d*T*T)
                    excess = (evaluated-bound)/(1+bound)
                    maximum_work_bound_relative_excess = max(maximum_work_bound_relative_excess, excess)
                    largest_evaluated_bound = max(largest_evaluated_bound, bound)
                    assert excess <= 4e-13
                    parameter_cases += 1
CHECKS['explicit_schedule_and_work_bound'] = {
    'passed': True, 'parameter_energy_dimension_cases': parameter_cases,
    'T_grid': [1, 2, 3, 4, 9, 32, 100],
    'B_grid': [1e-9, .1, 1., 10., 1e9], 'R_grid': [0., 1e-9, .1, 1., 10., 1e9],
    'energy_profiles': ['zero_energy', 'maximum_energy', 'random_energy'],
    'maximum_relative_excess_for_log_lower_bound': maximum_log_inequality_relative_excess,
    'maximum_relative_rounded_N_excess': maximum_N_bound_relative_excess,
    'maximum_relative_explicit_work_sum_excess': maximum_work_bound_relative_excess,
    'maximum_mixing_bound_minus_allocated_budget': maximum_mixing_budget_excess,
    'maximum_combined_soft_average_and_mixing_budget': maximum_total_prediction_budget,
    'largest_evaluated_explicit_operation_bound': largest_evaluated_bound,
    'note': 'Huge transition counts are evaluated algebraically only; no sampler or online trajectory is run.',
}


report = {
    'status': 'passed', 'groups': len(CHECKS), 'seed': SEED,
    'script': 'research/check_soft_bayes.py',
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'scope': 'Supplementary finite numerical checks of identities and bounds; not a mathematical proof, finite-bit certificate, timing benchmark, or convergence experiment.',
    'checks': CHECKS,
}
(ROOT/'SOFT_BAYES_CHECKS.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
print(json.dumps({'status': report['status'], 'groups': report['groups'],
                  'report': str(ROOT/'SOFT_BAYES_CHECKS.json')}, indent=2))
