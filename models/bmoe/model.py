"""
Stage 6 --- Bayesian mixture of experts.

The clamped mixing weights of stage 5 become a latent switch with a
transition of its own, and the transition gets a Dirichlet prior:

    s_t | s_{t-1} = i  ~  Cat(A_i)
    A_i                ~  Dir(c_i)

This is the same move the talk already made twice: something that was
clamped is given a prior and inferred. Here it buys persistence. A bridge
that froze on Thursday morning is still frozen on Thursday afternoon, and a
switch with a learned transition keeps that belief, while the static mixture
of stage 5 has to rediscover the regime at every sample.

Inference is the forward pass of the switch combined with the branch
predictives. The transition counts are accumulated from the expected
pairwise occupancies, the usual online variational update for a Dirichlet
over the rows of A.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from ..mixture import MixtureBase, default_bank


class BayesianMixtureOfExperts(MixtureBase):

    name = "bmoe"
    label = "Bayesian mixture"

    def __init__(self, order: int = 4, n_experts: int = 3, omega: float = 0.0,
                 vartheta: float = 1e3, alpha_lo: float = 1e2, alpha_hi: float = 1e8,
                 kappa: float = 1.0, c_off: float = 1.0, c_diag: float = 50.0,
                 count_scale: float = 1.0):
        super().__init__(default_bank(order, n_experts, omega, vartheta,
                                      alpha_lo, alpha_hi, kappa))
        K = self.K
        # Dirichlet counts, one row per originating state. The diagonal boost
        # is a mild prior belief that regimes persist; the data overwhelms it
        # within a few hundred samples.
        self.counts = np.full((K, K), float(c_off)) + np.eye(K) * float(c_diag)
        self.count_scale = float(count_scale)
        self.belief = np.full(K, 1.0 / K)     # filtered q(s_{t-1})
        self._prev = self.belief.copy()

    # -- the transition, as the Dirichlet expects it ----------------------
    def transition(self) -> np.ndarray:
        return self.counts / self.counts.sum(axis=1, keepdims=True)

    def log_prior_weights(self) -> np.ndarray:
        self._prev = self.belief.copy()
        pred = self._prev @ self.transition()
        pred = np.maximum(pred, 1e-300)
        return np.log(pred / pred.sum())

    def absorb_responsibilities(self, logw, loglik, resp) -> None:
        # expected pairwise occupancy xi_ij for this step
        log_xi = (np.log(np.maximum(self._prev, 1e-300))[:, None]
                  + np.log(self.transition())
                  + loglik[None, :])
        xi = np.exp(log_xi - logsumexp(log_xi))
        self.counts += self.count_scale * xi
        self.belief = resp

    def diagnostics(self):
        d = super().diagnostics()
        A = self.transition()
        d["stick"] = np.diag(A).copy()
        return d


def build(order: int = 4, **kw) -> BayesianMixtureOfExperts:
    return BayesianMixtureOfExperts(order=order, **kw)
