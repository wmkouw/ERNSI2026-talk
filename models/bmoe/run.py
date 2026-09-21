"""Fit stage 6 and write its predictions. `python -m models.bmoe.run`"""
import json, os
from ..base import Protocol, RESULTS
from ..runner import run_stage
from ..moe.run import BANK
from .model import build

# What stage 5 settled on, so the only difference between the two results is
# the switch. If stage 5 has not been run, fall back to the middle of its grid.
FALLBACK = dict(vartheta=1e1, gp_length=2.0, gp_snr=64.0, resp_floor=0.2)


def inherited():
    p = os.path.join(RESULTS, "moe.json")
    if not os.path.exists(p):
        return dict(FALLBACK)
    with open(p) as fh:
        h = json.load(fh)["hyper"]
    return {k: h.get(k, v) for k, v in FALLBACK.items()}


def main(protocol=None):
    return run_stage("bmoe", "Bayesian mixture", 6, build, grid=None,
                     fixed=dict(BANK, c_off=1.0, c_diag=50.0, **inherited()),
                     protocol=protocol or Protocol())

if __name__ == "__main__":
    main()
