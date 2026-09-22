"""
The question from the second slide, answered with a number.

    python -m models.decision

Slide two asks whether it is still safe for vehicles to cross. That is not a
forecasting question, it is a decision, and a decision needs a threshold and a
margin rather than a point estimate.

We pick one hour of the scenario, state the conditions in the terms an operator
would use, and ask each model what the deck is doing. Deployed the way a monitor
would be, online and one step ahead, a model produces one predictive
distribution per sample. Over an hour that is 600 of them, summarised by two
numbers in the units of the measurement:

    mu    = r.m.s. of the predicted means      -- the motion the model expects
    sigma = r.m.s. of the predictive s.d.s     -- the motion it cannot pin down

Their Pythagorean sum is the predicted r.m.s. acceleration, which is the
quantity the threshold is set on, and it can be checked against what the deck
actually did. Their plain sum is the one-sigma margin, which is what the
decision is made on. The gap between the two is the price of uncertainty.

Everything here is read at full rate. The thinned arrays in `results/*.npz`
carry one sample in twenty, which is ample for a curve over five days and far
too coarse for the r.m.s. of a single bursty hour.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import base
from .base import Protocol, RESULTS, prepare
from .ar.model import build as build_ar
from .hgf.model import build as build_hgf
from .bmoe.model import build as build_bmoe


# Friday afternoon, three hours after the deck cracked, at the peak of the
# working day. Chosen because the decision flips inside the margin there. The
# predicted mean clears the limit and the mean plus one standard deviation does
# not. LATER_HOUR is the same evening, once the loading has eased, where the
# whole margin clears.
HOUR = 112.0
THRESHOLD_MM = 150.0        # illustrative serviceability limit, r.m.s. mm/s^2
LATER_HOUR = 114.0          # two hours on, with the loading eased

MODELS = [("ar", build_ar), ("hgf", build_hgf), ("bmoe", build_bmoe)]


def _hyper(name: str) -> dict:
    with open(os.path.join(RESULTS, f"{name}.json")) as fh:
        h = dict(json.load(fh)["hyper"])
    h.pop("_search", None)
    h.pop("order", None)
    return h


def conditions(trace_csv: str, hour: float) -> dict:
    """What an operator would read off the gauges for that hour."""
    d = pd.read_csv(trace_csv, usecols=["hour", "traffic_rate", "wind",
                                        "T_deck", "damage", "regime"])
    s = d[(d.hour >= hour) & (d.hour < hour + 1.0)]
    first_damage = d.hour[d.damage > 0]
    return dict(
        traffic_veh_per_h=round(float(s.traffic_rate.mean())),
        wind_m_per_s=round(float(s.wind.mean()), 1),
        deck_temp_c=round(float(s.T_deck.mean()), 1),
        damaged=bool(s.damage.max() > 0),
        damage_from_hour=(None if first_damage.empty
                          else round(float(first_damage.iloc[0]), 1)),
        regime=int(s.regime.max()),
    )


def band(name, build, X, y, idx, hours, windows):
    """Run the model over the whole trace, summarise the chosen windows."""
    mdl = build(order=Protocol().order, **_hyper(name))
    mu = np.empty(len(y))
    sd = np.empty(len(y))
    for k in range(len(y)):
        pr = mdl.predict(X[k])
        mu[k], sd[k] = pr.mean, pr.std
        mdl.update(X[k], y[k])
    out = {}
    for h in windows:
        s = (hours >= h) & (hours < h + 1.0)
        m = float(np.sqrt(np.mean(mu[s] ** 2)) * 1e3)
        v = float(np.sqrt(np.mean(sd[s] ** 2)) * 1e3)
        out[f"{h:g}"] = dict(mu_mm=round(m, 1), sigma_mm=round(v, 1),
                             margin_mm=round(m + v, 1),
                             rms_pred_mm=round(float(np.hypot(m, v)), 1))
    return out


def main(protocol=None, verbose=True):
    protocol = protocol or Protocol()
    trace, X, y, idx = prepare(protocol)
    hours = trace.hour[idx]
    windows = (HOUR, LATER_HOUR)

    realised = {}
    for h in windows:
        s = (hours >= h) & (hours < h + 1.0)
        realised[f"{h:g}"] = dict(
            rms_mm=round(float(np.sqrt(np.mean(y[s] ** 2)) * 1e3), 1),
            peak_mm=round(float(np.max(np.abs(y[s])) * 1e3), 1))

    out = dict(hour=HOUR, later_hour=LATER_HOUR, threshold_mm=THRESHOLD_MM,
               conditions=conditions(protocol.trace, HOUR),
               later_conditions=conditions(protocol.trace, LATER_HOUR),
               realised=realised, models={})
    for name, build in MODELS:
        out["models"][name] = band(name, build, X, y, idx, hours, windows)
        if verbose:
            r = out["models"][name][f"{HOUR:g}"]
            print(f"[{name:5}] mu {r['mu_mm']:6.1f}  sigma {r['sigma_mm']:6.1f}  "
                  f"mu+sigma {r['margin_mm']:6.1f}  predicted r.m.s. "
                  f"{r['rms_pred_mm']:6.1f} mm/s^2")

    path = os.path.join(RESULTS, "decision.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    if verbose:
        c = out["conditions"]
        print(f"\nhour {HOUR:g}: {c['traffic_veh_per_h']} veh/h, wind "
              f"{c['wind_m_per_s']} m/s, deck {c['deck_temp_c']} C, "
              f"damaged since hour {c['damage_from_hour']}")
        print(f"realised r.m.s. {realised[f'{HOUR:g}']['rms_mm']} mm/s^2, "
              f"limit {THRESHOLD_MM:g} mm/s^2  ->  {path}")
    return out


if __name__ == "__main__":
    main()
