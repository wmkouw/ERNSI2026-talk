"""Fit the Matern-1/2 GP on its own. `python -m models.matern.run`

This is a reference run, not a rung of the ladder. The GP is one of the two
experts the mixture slides put behind a switch, and the honest expectation
here is "not much on its own", because a first-order smooth process is a poor
description of a deck with two resonances. The number exists so the mixture
result can be read properly: if the pair beats both of its members, the
switch is doing something neither model can do alone.

The length scale and signal-to-noise ratio are chosen here, on the burn-in
day, and then carried into the mixture. Tuning them inside the mixture would
cost far more and would let the switch paper over a bad choice.
"""

from __future__ import annotations

from .. import base
from ..base import Protocol, product_grid
from ..runner import run_stage
from .model import build


def main(protocol=None, verbose=True):
    # The grid runs well past where the optimum turned out to be, on purpose:
    # an optimum sitting on a boundary is not an optimum. Large snr is the
    # interesting limit, where the observation noise vanishes, f_t collapses
    # onto y_t and the GP degenerates into an AR(1) with coefficient
    # exp(-1/ell).
    return run_stage(
        "matern", "Matern-1/2 GP", 7, build,
        grid=product_grid(length=[1.0, 2.0, 4.0, 8.0, 20.0],
                          snr=[0.25, 1.0, 4.0, 16.0, 64.0, 256.0]),
        fixed=dict(omega=0.0, vartheta=1e2, kappa=1.0),
        protocol=protocol or Protocol(), verbose=verbose)


if __name__ == "__main__":
    main()
    print(open(base.write_summary()).read())
