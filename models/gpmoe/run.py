"""
Fit the Matern-1/2 GP, alone and as an extra branch of the mixture.

    python -m models.gpmoe.run

Two entries are written.

`matern`
    The GP on its own: an Ornstein-Uhlenbeck process observed in noise, with
    the amplitude tracked by a volatility chain. This is what the branch is
    worth by itself, and the honest expectation is "not much", because a
    first-order smooth process is a poor description of a deck with two
    resonances. The number exists so the next one can be read properly.

`bmoe_gp`
    Stage 6 with that GP appended to the bank. Every other setting, including
    the switch, its Dirichlet transition and the three autoregressive experts,
    is taken from stage 6's own result file, so the only difference between
    this and `bmoe` is the extra branch.

The GP's length scale and signal-to-noise ratio are chosen on the burn-in, on
the standalone model, and then carried into the mixture. Tuning them inside
the mixture would cost four times as much and let the switch paper over a bad
choice.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .. import base
from ..base import RESULTS, Protocol, product_grid
from ..runner import run_stage
from ..moe.run import BANK
from .model import build, build_gp


def _stage6_settings():
    """Reuse stage 6's chosen volatility, so only the branch differs."""
    vartheta = 1e1
    p = os.path.join(RESULTS, "bmoe.json")
    if os.path.exists(p):
        with open(p) as fh:
            vartheta = json.load(fh)["hyper"].get("vartheta", vartheta)
    return dict(BANK, vartheta=vartheta, c_off=1.0, c_diag=50.0)


def main(protocol=None, verbose=True):
    protocol = protocol or Protocol()
    shared = _stage6_settings()

    # ------------------------------------------------- the GP on its own
    m_gp = run_stage(
        "matern", "Matern-1/2 GP", 7, build_gp,
        # The grid runs well past where the optimum turned out to be, on
        # purpose: an optimum sitting on a boundary is not an optimum. Large
        # snr is the interesting limit, where the observation noise vanishes,
        # f_t collapses onto y_t and the GP degenerates into an AR(1) with
        # coefficient exp(-1/ell).
        grid=product_grid(length=[1.0, 2.0, 4.0, 8.0, 20.0],
                          snr=[0.25, 1.0, 4.0, 16.0, 64.0, 256.0]),
        fixed=dict(omega=shared["omega"], vartheta=shared["vartheta"],
                   kappa=shared["kappa"]),
        protocol=protocol, verbose=verbose)

    with open(os.path.join(RESULTS, "matern.json")) as fh:
        chosen = json.load(fh)["hyper"]

    # --------------------------------- stage 6 with the GP as one branch
    m_mix = run_stage(
        "bmoe_gp", "Bayes. mixture + GP", 8, build, grid=None,
        fixed=dict(shared, gp_length=chosen["length"], gp_snr=chosen["snr"]),
        protocol=protocol, verbose=verbose)

    return dict(matern=m_gp, bmoe_gp=m_mix)


if __name__ == "__main__":
    main()
    print(open(base.write_summary()).read())
