"""
Shared machinery for the two mixture stages.

`Expert` is the stage-4 block treated as a single reusable node: a
coefficient random walk plus a volatility chain, with the drift precision
clamped rather than inferred. Instantiating it K times with K different
drift precisions gives a bank that spans "the coefficients barely move" to
"the coefficients can jump", which is exactly the question a bridge poses
when it freezes or cracks.

`MixtureBase` holds what stages 5 and 6 share: predict from every branch,
mix, form responsibilities, and temper each branch's update by its own
responsibility. The only difference between the two stages is where the
mixing weights come from, which is one overridden method.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.special import logsumexp

from .base import OnlineModel, Predictive
from .blocks import CoefficientChain, VolatilityChain


LOG2PI = float(np.log(2.0 * np.pi))


class Expert:
    """One stage-4 block: TVAR coefficients with an HGF on the noise."""

    def __init__(self, order: int, alpha0: float, omega: float,
                 vartheta: float, kappa: float = 1.0, p0: float = 1.0,
                 sigma0: float = 1.0):
        self.alpha0 = float(alpha0)
        self.theta = CoefficientChain(order, alpha0=alpha0, learn_alpha=False, p0=p0)
        self.z = VolatilityChain(kappa=kappa, omega=omega, theta=vartheta,
                                 mu0=0.0, sigma0=sigma0)

    def predict(self, x: np.ndarray):
        m, _ = self.theta.predict()
        self.z.predict()
        loc = float(m @ x)
        var = self.theta.pred_var(x) + self.z.expected_variance()
        return loc, var

    def update(self, x: np.ndarray, y: float, weight: float) -> None:
        noise_var = 1.0 / self.z.expected_precision()
        eps = self.theta.update(x, y, noise_var=noise_var, weight=weight)
        self.z.update(eps, weight=weight)


def default_bank(order: int, n_experts: int, omega: float, vartheta: float,
                 alpha_lo: float, alpha_hi: float, kappa: float = 1.0) -> List[Expert]:
    """K experts whose drift precisions are spaced logarithmically.

    Low precision means a loose walk, so expert 0 is the agile one and the
    last expert is the near-static one.
    """
    alphas = np.logspace(np.log10(alpha_lo), np.log10(alpha_hi), n_experts)
    return [Expert(order, a, omega, vartheta, kappa) for a in alphas]


class MixtureBase(OnlineModel):
    """K experts behind one selector node."""

    def __init__(self, experts: Sequence[Expert], resp_floor: float = 0.02):
        # Every branch learns with at least `resp_floor` weight. Without a
        # floor a branch that loses early stops being updated, its covariance
        # inflates unchecked, and it comes back as a diffuse catch-all that
        # wins on outliers and then holds the switch. Keeping a trickle of
        # data flowing to every branch is what interacting-multiple-model
        # filters do for the same reason; the *predictive* weights stay exact.
        self.resp_floor = float(resp_floor)
        self.experts = list(experts)
        self.K = len(self.experts)
        self._loc = np.zeros(self.K)
        self._var = np.ones(self.K)
        self._logw = np.full(self.K, -np.log(self.K))
        self._resp = np.full(self.K, 1.0 / self.K)

    # -- stages differ only here -----------------------------------------
    def log_prior_weights(self) -> np.ndarray:
        """Log mixing weights for this step, before seeing y_t."""
        raise NotImplementedError

    def absorb_responsibilities(self, logw: np.ndarray, loglik: np.ndarray,
                                resp: np.ndarray) -> None:
        """Hook for a stage that learns something about the switch."""

    # -- common ------------------------------------------------------------
    def predict(self, x: np.ndarray) -> Predictive:
        for k, e in enumerate(self.experts):
            self._loc[k], self._var[k] = e.predict(x)
        self._logw = self.log_prior_weights()
        return Predictive(loc=self._loc.copy(),
                          scale=np.sqrt(self._var),
                          df=np.full(self.K, np.inf),
                          logw=self._logw.copy())

    def update(self, x: np.ndarray, y: float) -> None:
        loglik = -0.5 * (LOG2PI + np.log(self._var) + (y - self._loc) ** 2 / self._var)
        joint = self._logw + loglik
        resp = np.exp(joint - logsumexp(joint))
        self._resp = resp
        self.absorb_responsibilities(self._logw, loglik, resp)
        w = np.maximum(resp, self.resp_floor)
        for k, e in enumerate(self.experts):
            e.update(x, y, weight=float(w[k]))

    def diagnostics(self) -> Dict[str, np.ndarray]:
        return dict(resp=self._resp.copy(),
                    z=np.array([e.z.mu for e in self.experts]))
