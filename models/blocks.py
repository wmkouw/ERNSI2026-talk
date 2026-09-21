"""
Reusable sub-graphs.

These are the pieces that get composed as the model ladder grows. Each one
corresponds to a band of the factor graph in `graphs/keynote`:

    CoefficientChain   the theta row      -- a Gaussian random walk on the AR
                                             coefficients, with the walk's
                                             precision either clamped (stage 2)
                                             or inferred (stage 3+)
    NoisePrecision     the gamma slot     -- a static Gamma posterior (stages
                                             1 to 3)
    VolatilityChain    the z row          -- the hierarchical Gaussian filter
                                             that makes the observation
                                             precision time-varying (stage 4+)

Every update is the variational message that the corresponding node emits;
nothing here needs a rule that is not already standard. All of them accept a
`weight` in [0, 1], which is the responsibility a mixture node assigns to the
branch. Weight 1 is the ordinary single-model case; the mixtures temper each
branch's likelihood by its responsibility, which is the usual mixture-node
rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .base import sym


# --------------------------------------------------------------------------- #
# theta: a Gaussian random walk on the AR coefficients
# --------------------------------------------------------------------------- #

class CoefficientChain:
    """theta_t = theta_{t-1} + w_t,  w_t ~ N(0, alpha^-1 I).

    `alpha` is either clamped to `alpha0` (stage 2) or inferred (stage 3+).

    A note on inferring alpha, because the obvious route does not work here.
    The plain variational update for the precision of a Gaussian transition
    is driven by the expected squared increment E||theta_t - theta_{t-1}||^2.
    Taken under the filtered posterior that expectation equals tr(Q) exactly,
    whatever Q is, so every alpha is a fixed point. Taking it under a
    fixed-lag smoothed posterior removes the degeneracy in principle -- it is
    then the Shumway-Stoffer EM step -- but on this signal the map has a
    derivative of about 0.998, so the estimate crawls and never leaves its
    prior. Both were tried; neither moves.

    What does identify alpha is the marginal likelihood, through the
    innovation variance S = x' (P + q I) x + noise_var. So the alpha node is
    updated by a stochastic gradient step on the free energy instead of the
    VMP fixed point:

        d(-log N(e; 0, S)) / d log q   =   -rho (e^2 / S - 1) / 2,
        rho = q x'x / S

    rho is the share of the predictive variance that the drift injection is
    responsible for, so the step is automatically scaled: the model only
    revises its drift rate when the drift is actually doing work. Larger
    innovations than predicted push alpha down (allow more drift), smaller
    ones push it up. This is the innovation-based process-noise adaptation
    that adaptive filtering has used for decades, arrived at here as
    gradient descent on the same free energy the rest of the graph minimises.

    The Gamma summary (a, b) is carried along for reporting, with `a` acting
    as a count of how much evidence has gone into the estimate.
    """

    def __init__(self, dim: int, alpha0: float = 1e6, learn_alpha: bool = False,
                 alpha_init: float = 1e6, lr: float = 1.0, half_life: float = 0.0,
                 alpha_lo: float = 1e2, alpha_hi: float = 1e12,
                 m0: Optional[np.ndarray] = None, p0: float = 1.0):
        self.d = dim
        self.m = np.zeros(dim) if m0 is None else np.asarray(m0, float).copy()
        self.P = np.eye(dim) * p0
        self.learn_alpha = learn_alpha
        self.alpha0 = float(alpha0)
        self.lr = float(lr)
        # half_life = 0 keeps the step size constant, so alpha tracks; a
        # positive value decays it as 1/(1 + t/half_life), the Robbins-Monro
        # schedule that converges to a single static alpha, which is what the
        # factor graph actually draws.
        self.half_life = float(half_life)
        self.lo, self.hi = float(alpha_lo), float(alpha_hi)
        self._logq = -float(np.log(alpha_init))
        self._n = 0
        self.a = 1e-3          # Gamma shape, a running evidence count
        self._Pm: Optional[np.ndarray] = None

    # -- alpha ------------------------------------------------------------
    @property
    def alpha(self) -> float:
        if not self.learn_alpha:
            return self.alpha0
        return float(np.clip(np.exp(-self._logq), self.lo, self.hi))

    @property
    def b(self) -> float:
        """Gamma rate implied by the current mean, for reporting only."""
        return self.a / max(self.alpha, 1e-300)

    # -- forward ----------------------------------------------------------
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Push theta through the transition; returns (m^-, P^-)."""
        q = 1.0 / max(self.alpha, 1e-300)
        self._Pm = sym(self.P + q * np.eye(self.d))
        return self.m, self._Pm

    def update(self, x: np.ndarray, y: float, noise_var: float,
               weight: float = 1.0) -> float:
        """Absorb y with observation variance `noise_var`, tempered by `weight`.

        Returns the expected squared residual E[(y - theta'x)^2] under the
        posterior, which the node upstairs needs.
        """
        Pm = self._Pm
        w = max(float(weight), 1e-12)
        s = float(x @ Pm @ x) + noise_var / w      # tempering scales precision
        k = (Pm @ x) / s
        innov = y - float(self.m @ x)

        if self.learn_alpha:
            q = 1.0 / max(self.alpha, 1e-300)
            rho = q * float(x @ x) / s             # share of S due to drift
            grad = 0.5 * rho * (innov * innov / s - 1.0)
            # The raw gradient is used, deliberately. Normalising it by its
            # running magnitude (RMSProp and friends) turns the step into
            # something close to its sign, and e^2 / S is a chi-square with
            # one degree of freedom whose median is 0.45, not 1. A sign-like
            # step therefore drifts steadily downward on pure noise. The raw
            # gradient has the right mean and does not.
            rate = self.lr if self.half_life <= 0 else self.lr / (1.0 + self._n / self.half_life)
            self._n += 1
            self._logq += rate * w * float(np.clip(grad, -0.5, 0.5))
            self._logq = float(np.clip(self._logq, -np.log(self.hi), -np.log(self.lo)))
            self.a += 0.5 * w

        self.m = self.m + k * innov
        self.P = sym(Pm - np.outer(k, x @ Pm))

        resid = y - float(self.m @ x)
        return resid ** 2 + float(x @ self.P @ x)

    # -- reporting --------------------------------------------------------
    def pred_var(self, x: np.ndarray) -> float:
        """Variance the coefficient uncertainty contributes to the forecast."""
        return float(x @ self._Pm @ x)


# --------------------------------------------------------------------------- #
# gamma: a static Gamma posterior on the observation precision
# --------------------------------------------------------------------------- #

class NoisePrecision:
    """gamma ~ Gamma(a, b), shared across time (stages 1 to 3)."""

    def __init__(self, a: float = 1e-3, b: float = 1e-3):
        self.a = float(a)
        self.b = float(b)

    @property
    def mean(self) -> float:
        return self.a / self.b

    @property
    def var_estimate(self) -> float:
        """E[gamma]^-1, used as the observation variance in the filter."""
        return self.b / self.a

    @property
    def df(self) -> float:
        return 2.0 * self.a

    def update(self, expected_sq_resid: float, weight: float = 1.0) -> None:
        w = max(float(weight), 0.0)
        self.a += 0.5 * w
        self.b += 0.5 * w * max(expected_sq_resid, 0.0)


# --------------------------------------------------------------------------- #
# f: a Matern-1/2 Gaussian process, as a state-space model
# --------------------------------------------------------------------------- #

class MaternHalf:
    """A Matern-1/2 GP, which is an AR(1) once you write it down properly.

        k(tau) = sigma^2 exp(-|tau| / ell)

    is the covariance of an Ornstein-Uhlenbeck process,

        df(t) = -(1/ell) f(t) dt + dW(t),

    and sampling that at a fixed step dt gives exactly

        f_t = a f_{t-1} + w_t,   a = exp(-dt/ell),  Var(w_t) = sigma^2 (1 - a^2).

    So the kernel and the recursion are two views of one object. Inference is
    the Kalman filter and it is exact: no inducing points, no Cholesky of an
    n-by-n matrix, no O(n^3). On a factor graph it is one more chain, drawn
    like every other chain in this talk, and the only thing that distinguishes
    it from the coefficient chain is that `a` is less than one, so it forgets
    rather than wanders.

    `length` is in samples. The random walk of the other chains is the limit
    ell to infinity, where a goes to 1 and the stationary variance diverges.
    """

    def __init__(self, length: float, m0: float = 0.0, p0: float = 1.0):
        self.length = float(length)
        self.a = float(np.exp(-1.0 / max(self.length, 1e-9)))
        self.m = float(m0)
        self.P = float(p0)
        self._mm: Optional[float] = None
        self._Pm: Optional[float] = None

    def predict(self, stationary_var: float) -> Tuple[float, float]:
        """Push f through the transition; returns (m^-, P^-).

        `stationary_var` is sigma^2, the marginal variance of the GP. It is
        passed in per step rather than fixed at construction because on this
        signal the amplitude moves by three orders of magnitude, so the scale
        comes from the volatility chain above.
        """
        q = max(stationary_var, 0.0) * (1.0 - self.a ** 2)
        self._mm = self.a * self.m
        self._Pm = self.a ** 2 * self.P + q
        return self._mm, self._Pm

    def update(self, y: float, obs_var: float, weight: float = 1.0) -> float:
        """Absorb y with observation variance `obs_var`, tempered by `weight`.

        Returns E[(y - f)^2] under the posterior, for the node upstairs.
        """
        w = max(float(weight), 1e-12)
        s = self._Pm + obs_var / w
        k = self._Pm / s
        self.m = self._mm + k * (y - self._mm)
        self.P = self._Pm - k * self._Pm
        r = y - self.m
        return r * r + self.P


# --------------------------------------------------------------------------- #
# z: the hierarchical Gaussian filter on log observation precision
# --------------------------------------------------------------------------- #

class VolatilityChain:
    """gamma_t = exp(kappa z_t + omega),  z_t = z_{t-1} + v_t, v_t ~ N(0, theta^-1).

    The update of z is the variational message out of the exponential link,
    which is the familiar HGF volatility step: a Gauss-Newton step on

        f(z) = -(z - mu^-)^2 / (2 sigma^-2)
               + w * [ (kappa z + omega) / 2 - exp(kappa z + omega) * eps / 2 ]

    with eps the expected squared residual.

    The curvature is the observed one, kappa^2 * lambda * eps / 2, floored at
    its expected value kappa^2 / 2. The floor matters. A single squared
    residual is a chi-square with one degree of freedom, so it lands near
    zero a quarter of the time; there the observed curvature vanishes and the
    raw Newton step is unbounded upward. Flooring it makes every step lie in
    [-1/kappa, +1/kappa] in either direction, which is what keeps the
    recursion stable across 70k samples. At the fixed point E[lambda eps] = 1
    and the floor is inactive, so nothing is distorted where it counts.
    """

    def __init__(self, kappa: float = 1.0, omega: float = 0.0,
                 theta: float = 1e3, mu0: float = 0.0, sigma0: float = 1.0,
                 n_newton: int = 2, z_clip: float = 20.0):
        self.kappa = float(kappa)
        self.omega = float(omega)
        self.theta = float(theta)          # precision of the z random walk
        self.mu = float(mu0)
        self.sigma = float(sigma0)         # variance of q(z)
        self.n_newton = int(n_newton)
        self.z_clip = float(z_clip)
        self._mum: Optional[float] = None
        self._sigmam: Optional[float] = None

    def predict(self) -> Tuple[float, float]:
        self._mum = self.mu
        self._sigmam = self.sigma + 1.0 / max(self.theta, 1e-300)
        return self._mum, self._sigmam

    # -- moments of the implied precision --------------------------------
    def expected_precision(self) -> float:
        """E[exp(kappa z + omega)] under the predictive q(z^-)."""
        a = self.kappa * self._mum + self.omega + 0.5 * self.kappa ** 2 * self._sigmam
        return float(np.exp(np.clip(a, -self.z_clip * 2, self.z_clip * 2)))

    def expected_variance(self) -> float:
        """E[exp(-(kappa z + omega))] under the predictive q(z^-)."""
        a = -self.kappa * self._mum - self.omega + 0.5 * self.kappa ** 2 * self._sigmam
        return float(np.exp(np.clip(a, -self.z_clip * 2, self.z_clip * 2)))

    def update(self, expected_sq_resid: float, weight: float = 1.0) -> None:
        eps = max(float(expected_sq_resid), 1e-300)
        w = max(float(weight), 0.0)
        k, om = self.kappa, self.omega
        prec_prior = 1.0 / self._sigmam
        z = self._mum
        curv = prec_prior + 0.5 * w * k ** 2
        for _ in range(self.n_newton):
            lam = float(np.exp(np.clip(k * z + om, -self.z_clip, self.z_clip)))
            curv = prec_prior + 0.5 * w * k ** 2 * max(lam * eps, 1.0)
            grad = -(z - self._mum) * prec_prior + 0.5 * w * k * (1.0 - lam * eps)
            z = float(np.clip(z + grad / curv, self._mum - 8.0, self._mum + 8.0))
        self.mu = float(np.clip(z, -self.z_clip, self.z_clip))
        self.sigma = 1.0 / curv
