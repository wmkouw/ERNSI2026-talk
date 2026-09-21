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
from .blocks import CoefficientChain, MaternHalf, VolatilityChain


LOG2PI = float(np.log(2.0 * np.pi))


class Expert:
    """One stage-4 block: TVAR coefficients with an HGF on the noise.

    With `learn_alpha` this is literally the stage-4 model, drift precision
    and all, used as a node. That is the point of the mixture slide: the
    branch is not a simplified stand-in for stage 4, it is stage 4.
    """

    def __init__(self, order: int, alpha0: float, omega: float,
                 vartheta: float, kappa: float = 1.0, p0: float = 1.0,
                 sigma0: float = 1.0, learn_alpha: bool = False,
                 alpha_init: float = 1e6, lr: float = 4.0,
                 half_life: float = 2000.0):
        self.alpha0 = float(alpha0)
        self.theta = CoefficientChain(order, alpha0=alpha0,
                                      learn_alpha=learn_alpha,
                                      alpha_init=alpha_init, lr=lr,
                                      half_life=half_life, p0=p0)
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


class GPExpert:
    """A Matern-1/2 GP on the signal itself, as one branch of the bank.

    The other experts all say the same thing about the bridge -- that it is
    an autoregression whose coefficients drift -- and differ only in how fast
    they let them drift. This one says something else: that y_t is a smooth
    process with a correlation time and nothing more. It is a poor model of a
    vibrating deck, being first order and non-oscillatory, and it is meant to
    be. It is the branch that catches what the autoregression cannot: a truck
    impulse, a transient, the first samples of a regime the coefficients have
    not reached.

    The GP runs in normalised units and the volatility chain supplies the
    scale. That split is not cosmetic. Letting the GP's own marginal variance
    be a multiple of the noise the chain was estimating made the pair
    unidentified -- the residual then scales with the prior, the chain sees
    E[lambda eps] = 1 at every scale, and it random-walks to its clip while
    the branch quietly reports a predictive standard deviation of ten
    thousand. In normalised units the GP's marginal variance is fixed at one,
    its predictive variance in those units is a known constant, and dividing
    the squared prediction error by that constant leaves a quantity whose
    expectation is the scale and nothing else.

    This is also, exactly, what the Transformer's instance normalisation does
    by hand. Here the scale is inferred rather than measured off the context,
    and it comes with a posterior.
    """

    def __init__(self, length: float, omega: float, vartheta: float,
                 snr: float = 4.0, kappa: float = 1.0, sigma0: float = 1.0):
        self.length = float(length)
        self.snr = float(snr)
        self.r = 1.0 / float(snr)            # observation noise, normalised
        self.gp = MaternHalf(length, p0=1.0)
        self.z = VolatilityChain(kappa=kappa, omega=omega, theta=vartheta,
                                 mu0=0.0, sigma0=sigma0)
        self._scale = 1.0
        self._mu_u = 0.0
        self._var_u = 1.0

    def predict(self, x: np.ndarray):
        self.z.predict()
        self._scale = float(np.sqrt(max(self.z.expected_variance(), 1e-300)))
        m, P = self.gp.predict(stationary_var=1.0)
        self._mu_u = m
        self._var_u = P + self.r
        return self._scale * m, self._scale ** 2 * self._var_u

    def update(self, x: np.ndarray, y: float, weight: float) -> None:
        s = self._scale
        self.gp.update(y / s, obs_var=self.r, weight=weight)
        # squared prediction error over its known variance in normalised
        # units: what is left is an estimate of the scale itself
        eps = (y - s * self._mu_u) ** 2 / max(self._var_u, 1e-300)
        self.z.update(eps, weight=weight)


def default_bank(order: int, n_experts: int, omega: float, vartheta: float,
                 alpha_lo: float, alpha_hi: float, kappa: float = 1.0) -> List:
    """K autoregressive experts whose drift precisions are spaced apart.

    Kept for reference and for the earlier runs; the talk now uses
    `paired_bank`, where the two branches are different models rather than
    the same model at different settings.
    """
    alphas = np.logspace(np.log10(alpha_lo), np.log10(alpha_hi), n_experts)
    return [Expert(order, a, omega, vartheta, kappa) for a in alphas]


def paired_bank(order: int, omega: float, vartheta: float, gp_length: float,
                gp_snr: float = 4.0, kappa: float = 1.0, lr: float = 4.0,
                half_life: float = 2000.0, alpha_init: float = 1e6) -> List:
    """The two experts the mixture slides show.

    Branch 1 is the stage-4 block itself: time-varying autoregressive
    coefficients with an inferred drift precision and a volatility chain on
    the noise. Branch 2 is a Matern-1/2 Gaussian process in state-space form.

    They are different models, not the same model at two settings, which is
    what makes the mixture worth drawing. One says the bridge is a resonant
    structure whose parameters drift; the other says it is a smooth process
    with a correlation time. The switch decides, sample by sample, which
    description the data supports.
    """
    return [
        Expert(order, alpha0=1e6, omega=omega, vartheta=vartheta, kappa=kappa,
               learn_alpha=True, alpha_init=alpha_init, lr=lr,
               half_life=half_life),
        GPExpert(gp_length, omega, vartheta, gp_snr, kappa),
    ]


class MixtureBase(OnlineModel):
    """K experts behind one selector node."""

    def __init__(self, experts: Sequence[Expert], resp_floor: float = 0.2):
        # Every branch learns with at least `resp_floor` weight, and wherever
        # else a responsibility is carried forward rather than used as a
        # density it is floored the same way. Without a floor a branch that
        # loses early stops being updated, its covariance inflates unchecked,
        # and it comes back as a diffuse catch-all that wins on outliers and
        # then holds the switch. Keeping a share of every observation flowing
        # to every branch is what interacting-multiple-model filters do for
        # the same reason; the *predictive* weights stay exact.
        #
        # With two branches rather than a bank of near-copies the floor stops
        # being a safety net and becomes the setting that matters most, so it
        # is tuned on the burn-in like anything else. See models/README.md.
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
