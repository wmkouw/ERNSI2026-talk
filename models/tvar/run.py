"""Fit stage 2 and write its predictions. `python -m models.tvar.run`"""
import numpy as np
from ..base import Protocol, logspace_grid
from ..runner import run_stage
from .model import build

def main(protocol=None):
    # alpha_0 is clamped, so it has to be guessed; the burn-in guesses it.
    return run_stage("tvar", "TVAR", 2, build,
                     grid=logspace_grid("alpha0", 4, 11, 8),
                     fixed=dict(a0=1e-2, b0=1e-6),
                     protocol=protocol or Protocol())

if __name__ == "__main__":
    main()
