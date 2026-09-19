"""Fit stage 1 and write its predictions. `python -m models.ar.run`"""
from ..base import Protocol
from ..runner import run_stage
from .model import build

def main(protocol=None):
    # The normal-gamma prior is conjugate and weak; nothing to search over.
    return run_stage("ar", "AR", 1, build, grid=None,
                     fixed=dict(lam0=1e-2, a0=1e-2, b0=1e-6),
                     protocol=protocol or Protocol())

if __name__ == "__main__":
    main()
