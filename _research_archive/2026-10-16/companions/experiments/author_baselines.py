"""Thin adapters around the unchanged pinned released author classes.

The native prediction and update methods are called as published. Stable
unsmoothed log losses are diagnostics computed from the same represented scores;
the release's separately clipped reported loss is also exposed. No sampler,
AIOLI update, clipping, adaptation or native scheduling code is replaced.
"""
import importlib.util
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.special import logsumexp

from .common import Metrics


ARCHIVE = Path(__file__).resolve().parents[1]/"research/author_code_comparison_2026-09-11"
COMMIT = "22c6ec54cd45908c0a39cbfa931a60e55646308f"


def author_module():
    name = "ols2_pinned_author_a9a_experiment"
    if name in sys.modules:
        return sys.modules[name]
    for directory in (ARCHIVE, ARCHIVE/"dependencies"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(name, ARCHIVE/"a9a_experiment.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


class _AuthorAdapter:
    def _setup(self, d, B, T, seed, config):
        self.d, self.T, self.B, self.seed = int(d), int(T), float(B), int(seed)
        if (self.d < 1 or self.T < 1 or self.d != d or self.T != T
                or not math.isfinite(self.B) or self.B <= 0.):
            raise ValueError("Require positive integral d,T and positive finite B")
        self.config = dict(config)
        self.metrics = Metrics()
        self.n = 0
        self._pending_x = None
        self.last_probability = self.last_log_probability_positive = None
        self.last_log_probability_negative = self.last_reported_loss = None
        self.last_native_predict_seconds = self.last_native_update_seconds = 0.
        self.last_native_initialization_seconds = 0.
        self.last_stable_logit = None
        self.last_acceptance_rate = None
        self.model = None
        self._failed = False

    def _feature(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.d,) or not np.all(np.isfinite(x)):
            raise ValueError("Require one finite d-dimensional feature")
        return x

    def predict(self, x):
        self.last_native_predict_seconds = 0.
        if self._failed:
            raise RuntimeError("The native algorithm previously failed")
        if self._pending_x is not None:
            raise RuntimeError("observe must follow predict")
        if self.n >= self.T:
            raise RuntimeError("Configured horizon exhausted")
        x = self._feature(x)
        started = perf_counter()
        try:
            probability = float(self.model.predict_prob(x))
        finally:
            self.last_native_predict_seconds = perf_counter()-started
            self.metrics.times["native_predict"] += self.last_native_predict_seconds
        if not math.isfinite(probability) or not 0. <= probability <= 1.:
            self._failed = True
            raise FloatingPointError("Native author prediction is not a probability")
        diagnostic_start = perf_counter()
        if self.algorithm == "author_aioli":
            score = float(self.model.theta@x)
            log_p = -float(np.logaddexp(0., -score))
            log_q = -float(np.logaddexp(0., score))
        else:
            scores = self.model.thetas@x
            if not np.all(np.isfinite(scores)):
                self._failed = True
                raise FloatingPointError("Native author particle scores are nonfinite")
            log_p = float(logsumexp(-np.logaddexp(0., -scores))-math.log(self.model.S))
            log_q = float(logsumexp(-np.logaddexp(0., scores))-math.log(self.model.S))
        logit = log_p-log_q
        if not all(math.isfinite(z) for z in (log_p, log_q, logit)):
            self._failed = True
            raise FloatingPointError("Stable native-score prediction is nonfinite")
        self.last_probability = probability
        self.last_log_probability_positive, self.last_log_probability_negative = log_p, log_q
        self.last_stable_logit = logit
        self.last_reported_loss = None
        self._pending_x = x.copy()
        self.metrics.inc("predictions")
        self.metrics.times["stable_prediction_diagnostic"] += perf_counter()-diagnostic_start
        return logit

    def loss(self, y):
        if y not in (-1, 1) or self.last_probability is None:
            raise ValueError("A prediction and a binary label are required")
        return -(self.last_log_probability_positive if y == 1 else self.last_log_probability_negative)

    def author_loss(self, y):
        if y not in (-1, 1) or self.last_probability is None:
            raise ValueError("A prediction and a binary label are required")
        return float(author_module().logloss_from_prob(self.last_probability, int(y)))

    def observe(self, x, y):
        self.last_native_update_seconds = 0.
        if self._failed:
            raise RuntimeError("The native algorithm previously failed")
        x = self._feature(x)
        if (y not in (-1, 1) or self._pending_x is None
                or not np.array_equal(x, self._pending_x)):
            raise ValueError("Observation must match a pending prediction with label -1 or +1")
        self.last_reported_loss = self.author_loss(y)
        if self.algorithm == "author_sigmoid":
            self.X_history[self.n] = x
            self.y_history[self.n] = int(y)
        started = perf_counter()
        try:
            if self.algorithm == "author_aioli":
                self.model.update(x, int(y))
            else:
                acceptance, _ = self.model.update(self.X_history[:self.n+1], self.y_history[:self.n+1])
                self.last_acceptance_rate = float(acceptance)
        except Exception:
            self._failed = True
            raise
        finally:
            self.last_native_update_seconds = perf_counter()-started
            self.metrics.times["native_update"] += self.last_native_update_seconds
        self.n += 1
        self._pending_x = None
        self.metrics.inc("observations")
        if self.algorithm == "author_sigmoid":
            effective_n = (min(self.n, int(self.model.mala_window))
                           if self.model.mala_window is not None and self.model.mala_window > 0 else self.n)
            self.metrics.inc("completed_native_mala_steps", self.model.K)
            self.metrics.inc("completed_native_particle_proposals", self.model.K*self.model.S)
            self.metrics.inc("completed_native_potential_particle_evaluations", 2*self.model.K*self.model.S)
            self.metrics.inc("completed_native_history_particle_evaluations", 2*self.model.K*self.model.S*effective_n)
            self.metrics.inc("estimated_native_history_matmul_flops", 8*self.model.K*self.model.S*effective_n*self.d)

    def state_bytes(self):
        total = 0 if self._pending_x is None else self._pending_x.nbytes
        if self.model is not None:
            arrays = (self.model.L, self.model.b, self.model.theta) if self.algorithm == "author_aioli" else (self.model.thetas, self.X_history, self.y_history)
            total += sum(array.nbytes for array in arrays)
        return int(total)

    def diagnostics(self):
        result = dict(algorithm=self.algorithm, native_class=type(self.model).__name__,
                      native_source=str(ARCHIVE/"a9a_experiment.py"), pinned_commit=COMMIT,
                      configuration=dict(self.config), observed_rounds=self.n,
                      last_probability=self.last_probability,
                      last_stable_logit=self.last_stable_logit,
                      last_log_probability_positive=self.last_log_probability_positive,
                      last_log_probability_negative=self.last_log_probability_negative,
                      last_reported_loss=self.last_reported_loss,
                      last_native_predict_seconds=self.last_native_predict_seconds,
                      last_native_update_seconds=self.last_native_update_seconds,
                      state_bytes=self.state_bytes(), failed=self._failed,
                      native_prediction_and_update_unchanged=True,
                      stable_loss_is_unsmoothed_native_score_diagnostic=True,
                      author_reported_loss_clips_probability_at=1e-12,
                      native_timing_excludes_adapter_diagnostics_and_history_append=True,
                      counters_scope="Completed native updates only; interrupted update work is not counted. FLOPs are dense-kernel estimates.",
                      finite_precision_certificate=False, metrics=self.metrics.as_dict())
        if self.algorithm == "author_sigmoid":
            result.update(population=self.model.S, steps_per_round=self.model.K,
                          current_step_size=self.model.h,
                          last_acceptance_rate=self.last_acceptance_rate,
                          adaptation_counter=self.model.round,
                          allocated_history_rows=self.T,
                          history_bytes=self.X_history.nbytes+self.y_history.nbytes,
                          full_prefix=(self.model.mala_window is None or self.model.mala_window <= 0))
        return result


class AuthorAIOLI(_AuthorAdapter):
    algorithm = "author_aioli"

    def __init__(self, d, B, T, seed, config=None):
        parameters = dict(R=1., lam=1./(float(B)*float(B)), T_inner=80, omega_solver="lbfgs")
        parameters.update(config or {})
        self._setup(d, B, T, seed, parameters)
        native = author_module()
        tick = perf_counter()
        self.model = native.AIOLI(d=self.d, B=self.B, seed=self.seed, **parameters)
        self.last_native_initialization_seconds = perf_counter()-tick
        self.metrics.times["native_initialization"] += self.last_native_initialization_seconds
        self.rng = self.model.rng


class AuthorSigmoid(_AuthorAdapter):
    algorithm = "author_sigmoid"

    def __init__(self, d, B, T, seed, config=None):
        parameters = dict(n_particles=12, step_size=1e-2, mala_steps_per_round=300,
                          grad_clip=50., adapt_step=True, mala_window=None,
                          target_accept=.57, rm_gamma0=.5, rm_gamma_decay=.25,
                          step_size_min=1e-4, step_size_max=1.)
        parameters.update(config or {})
        self._setup(d, B, T, seed, parameters)
        # One owned private history; prefix slices passed to the native class
        # are views. No second dense dataset or permutation is retained.
        self.X_history = np.empty((self.T, self.d), dtype=np.float64)
        self.y_history = np.empty(self.T, dtype=np.int64)
        self.rng = np.random.default_rng(self.seed)
        native = author_module()
        tick = perf_counter()
        self.model = native.GaussianEW_MALA(d=self.d, B=self.B, rng=self.rng, **parameters)
        self.last_native_initialization_seconds = perf_counter()-tick
        self.metrics.times["native_initialization"] += self.last_native_initialization_seconds


CLASSES = {"author_aioli": AuthorAIOLI, "author_sigmoid": AuthorSigmoid}
