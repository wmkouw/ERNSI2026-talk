"""Fit stage 6 and write its predictions. `python -m models.bmoe.run`"""
import json, os
from ..base import Protocol, RESULTS
from ..runner import run_stage
from ..moe.run import BANK
from .model import build

def main(protocol=None):
    # Stage 6 reuses stage 5's chosen volatility so the only difference
    # between the two results is the switch. If stage 5 has not been run,
    # fall back to the middle of its grid.
    vartheta = 1e2
    p = os.path.join(RESULTS, "moe.json")
    if os.path.exists(p):
        with open(p) as fh:
            vartheta = json.load(fh)["hyper"].get("vartheta", vartheta)
    return run_stage("bmoe", "Bayesian mixture", 6, build, grid=None,
                     fixed=dict(BANK, vartheta=vartheta, c_off=1.0, c_diag=50.0),
                     protocol=protocol or Protocol())

if __name__ == "__main__":
    main()
