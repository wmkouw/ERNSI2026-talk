"""
The same decision, with the conditions no longer known exactly.

    python -m models.decision_forecast

`models.decision` asks whether the deck is safe to cross at one hour, with
the wind and the deck temperature read off the gauges. That is the easy
case. A crossing decision is usually taken ahead of time, and by then the
wind and the temperature are a forecast rather than a reading.

Nothing in `models/decision.py` is touched. That file is still the one
behind res_decision0 and res_decision1. This one keeps the same models and
the same threshold and makes only the conditions uncertain.

The hour
--------
Hour 36 of the scenario, the morning a storm builds. The wind runs at
about 15 m/s and the traffic is ordinary, so the deck is wind driven and a
wind forecast is the thing that matters. At hour 112, the hour on the other
two slides, 1500 vehicles an hour dominate a 6 m/s breeze, and uncertainty
about the wind would barely move the answer. Choosing an hour where the
uncertain input is the one that drives the response is the difference
between a demonstration and a formality.

How the conditions reach the prediction
---------------------------------------
The models are autoregressive in the measured acceleration. Wind and
temperature never enter them as regressors, they act on the deck, and the
models see the result. So the route from conditions to prediction is one
the models have already traced out, and it can be read back off their own
output. For every whole hour of the record we have what the model
predicted,

    r = sqrt( rms(mu)^2 + rms(sigma)^2 )      predicted r.m.s. acceleration

and what the conditions were. Regressing log r on the conditions recovers
the exogenous part of the model, the part an ARX would carry explicitly,

    log r  =  b0 + b1 log U + b2 (log U)^2 + b3 T + b4 frost(T)
                                              + b5 log q  +  noise

with U the mean wind, T the deck temperature, frost a smooth indicator of
an icy deck and q the traffic rate. The features follow the simulator.
Buffeting load goes as U^2 and so does the response, hence the logs, and
the square lets that sensitivity switch on as the wind rises, since in a
breeze it is the traffic that drives the deck. Frost stiffens the deck
sharply below zero. The fit is conjugate normal inverse gamma, so the
slopes come with a posterior rather than a point value.

The two cases
-------------
    known      the model's own predictive for that hour, unchanged. This
               is the band `models/decision.py` reports.
    forecast   the same band, moved by the conditions:

                   r  =  r_known * exp( b . [x(U,T) - x(U_obs,T_obs)] )

               with U ~ N(U_obs, WIND_SD), T ~ N(T_obs, TEMP_SD) and b
               drawn from its posterior.

Only the differences in the features enter, so the link's intercept and
its residual scatter never touch the answer. Traffic is the same in both
and cancels. The known case is therefore exactly the model's own
uncertainty and the forecast case is that same uncertainty with the input
uncertainty propagated through on top, which is the one thing being shown.

In log space the split is exact,

    Var[log r]  =  s_model^2   +   Var[ b . dx ]
                   what the        what not knowing
                   model cannot    the conditions
                   pin down        costs

and the decision-relevant number, the chance of exceeding the
serviceability limit, is reported for both. The model and the threshold do
not change between them. Only the knowledge of the inputs does.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .base import Protocol, RESULTS, prepare
from .decision import THRESHOLD_MM, MODELS, _hyper, conditions


# ------------------------------------------------------------------ the hour
HOUR = 36.0                 # storm building, wind driven, ordinary traffic

# --------------------------------------------------------------- the forecast
# A day-ahead forecast, in the units of the gauges. Widen these and the
# predictive band widens with them, which is the whole demonstration.
WIND_SD = 3.0               # m/s, one standard deviation on the mean wind
TEMP_SD = 2.0               # degC, one standard deviation on the deck temp
WIND_FLOOR = 0.5            # m/s, still air rather than negative wind

N_DRAWS = 40_000
SEED = 20260922

# The grid the two predictive densities are written out on, r.m.s. mm/s^2.
GRID_MM = np.linspace(0.0, 480.0, 481)


# ------------------------------------------------------------------ features
def frost(T):
    """Smooth indicator of an icy deck, the simulator's own shape."""
    return 1.0 / (1.0 + np.exp((np.asarray(T, float) - 0.5) / 0.4))


def design(U, T, q):
    """One row per condition, [1, log U, (log U)^2, T, frost(T), log q].

    The square is what lets the wind sensitivity depend on the wind. In a
    breeze the deck is driven by traffic and the wind slope is near zero.
    In a storm the buffeting load goes as U^2 and so does the response, so
    the slope should approach two. A single log-linear term has to split
    the difference and reaches only about 0.7. With the square the fitted
    local slope at 15 m/s comes out at 1.95, which is the physics.""" 
    U = np.maximum(np.asarray(U, float), WIND_FLOOR)
    T = np.asarray(T, float)
    q = np.maximum(np.asarray(q, float), 1.0)
    lu = np.log(U)
    return np.stack([np.ones_like(U), lu, lu ** 2, T, frost(T), np.log(q)],
                    axis=-1)


# ------------------------------------------------------- conjugate regression
def fit_link(Xd, z, a0=1e-3, b0=1e-3, tau=1e3):
    """Normal inverse gamma posterior for z = Xd b + e, e ~ N(0, s^2)."""
    n, p = Xd.shape
    VN = np.linalg.inv(np.eye(p) / tau + Xd.T @ Xd)
    mN = VN @ (Xd.T @ z)
    aN = a0 + n / 2
    bN = b0 + 0.5 * float(z @ z - mN @ np.linalg.inv(VN) @ mN)
    return dict(m=mN, V=VN, a=aN, b=bN)


def draw_coefficients(post, n, rng):
    """Sample b from the marginal posterior, a multivariate Student t."""
    p = post["m"].size
    inv = rng.gamma(post["a"], 1.0 / post["b"], n)
    s2 = 1.0 / inv
    L = np.linalg.cholesky(post["V"] + 1e-12 * np.eye(p))
    return post["m"] + np.sqrt(s2)[:, None] * (rng.standard_normal((n, p)) @ L.T)


# --------------------------------------------------------------- the per-hour
def hourly(name, build, X, y, hours):
    """Run the model once over the record, summarise every whole hour."""
    mdl = build(order=Protocol().order, **_hyper(name))
    mu = np.empty(len(y))
    sd = np.empty(len(y))
    for k in range(len(y)):
        pr = mdl.predict(X[k])
        mu[k], sd[k] = pr.mean, pr.std
        mdl.update(X[k], y[k])

    out = {}
    for h in np.arange(np.floor(hours.min()), np.floor(hours.max()) + 1.0):
        s = (hours >= h) & (hours < h + 1.0)
        if s.sum() < 100:
            continue
        m = np.sqrt(np.mean(mu[s] ** 2))
        v = np.sqrt(np.mean(sd[s] ** 2))
        # The conjugate AR model starts with a Student t of two or fewer
        # degrees of freedom, so its variance is infinite for the first few
        # hours. Those hours carry no information and are dropped.
        if not (np.isfinite(m) and np.isfinite(v)) or m <= 0 or v <= 0:
            continue
        out[float(h)] = dict(mu=float(m), sigma=float(v), r=float(np.hypot(m, v)))
    return out


def hourly_conditions(trace_csv):
    """Mean wind, deck temperature and traffic rate for every whole hour."""
    d = pd.read_csv(trace_csv, usecols=["hour", "traffic_rate", "wind", "T_deck"])
    g = d.assign(h=np.floor(d.hour)).groupby("h").mean(numeric_only=True)
    return {float(h): dict(U=float(r.wind), T=float(r.T_deck),
                           q=float(r.traffic_rate)) for h, r in g.iterrows()}


# ------------------------------------------------------------------- reporting
def summarise(r_mm, threshold):
    return dict(
        mean_mm=round(float(np.mean(r_mm)), 1),
        sd_mm=round(float(np.std(r_mm)), 1),
        median_mm=round(float(np.median(r_mm)), 1),
        margin_mm=round(float(np.mean(r_mm) + np.std(r_mm)), 1),
        q05_mm=round(float(np.percentile(r_mm, 5)), 1),
        q95_mm=round(float(np.percentile(r_mm, 95)), 1),
        p_exceed=round(float(np.mean(r_mm > threshold)), 4),
    )


def density(r_mm):
    """A smoothed density on GRID_MM, for the figure to draw."""
    s = r_mm[::20]
    bw = 1.06 * np.std(s) * s.size ** (-0.2)
    z = (GRID_MM[:, None] - s[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (np.sqrt(2 * np.pi) * bw * s.size)


# ------------------------------------------------------------------- the case
def run_model(name, build, X, y, hours, cond, rng, verbose=True):
    per_hour = hourly(name, build, X, y, hours)
    if float(HOUR) not in per_hour:
        raise SystemExit(f"{name}: hour {HOUR:g} has no usable prediction")

    fit_hours = [h for h in sorted(per_hour) if h in cond and h != float(HOUR)]
    Xd = design([cond[h]["U"] for h in fit_hours],
                [cond[h]["T"] for h in fit_hours],
                [cond[h]["q"] for h in fit_hours])
    z = np.log([per_hour[h]["r"] for h in fit_hours])
    post = fit_link(Xd, z)
    r2 = 1.0 - np.var(z - Xd @ post["m"]) / np.var(z)

    now = cond[float(HOUR)]
    here = per_hour[float(HOUR)]

    # --- the model's own band for this hour, in log space -----------------
    # r is lognormal with the model's predicted r.m.s. as its centre and the
    # model's own predictive spread, sigma / r, as its relative width. That
    # is the band models/decision.py already reports.
    r_hat = here["r"] * 1e3
    s_model = here["sigma"] / here["r"]

    # --- known conditions -------------------------------------------------
    r_known = r_hat * np.exp(s_model * rng.standard_normal(N_DRAWS))

    # --- forecast conditions ----------------------------------------------
    U = rng.normal(now["U"], WIND_SD, N_DRAWS)
    T = rng.normal(now["T"], TEMP_SD, N_DRAWS)
    dx = design(U, T, np.full(N_DRAWS, now["q"])) \
        - design(now["U"], now["T"], now["q"])
    b = draw_coefficients(post, N_DRAWS, rng)
    shift = np.einsum("ni,ni->n", b, dx)
    r_fore = r_hat * np.exp(shift + s_model * rng.standard_normal(N_DRAWS))

    # --- the split, exact in log space ------------------------------------
    v_model = s_model ** 2
    v_cond = float(np.var(shift))

    out = dict(
        conditions_now=dict(wind_m_per_s=round(now["U"], 2),
                            deck_temp_c=round(now["T"], 2),
                            traffic_veh_per_h=round(now["q"])),
        model_point=dict(mu_mm=round(here["mu"] * 1e3, 1),
                         sigma_mm=round(here["sigma"] * 1e3, 1),
                         rms_pred_mm=round(r_hat, 1)),
        known=summarise(r_known, THRESHOLD_MM),
        forecast=summarise(r_fore, THRESHOLD_MM),
        split=dict(sd_log_model=round(float(np.sqrt(v_model)), 3),
                   sd_log_conditions=round(float(np.sqrt(v_cond)), 3),
                   sd_log_total=round(float(np.sqrt(v_model + v_cond)), 3),
                   share_conditions=round(v_cond / (v_model + v_cond), 3)),
        link=dict(r2=round(float(r2), 3), hours_fitted=len(fit_hours),
                  resid_sd_log=round(float(np.sqrt(post["b"] / post["a"])), 3),
                  coefficients=dict(zip(
                      ["const", "log_wind", "log_wind_sq", "deck_temp",
                       "frost", "log_traffic"],
                      [round(float(c), 4) for c in post["m"]])),
                  local_wind_slope=round(float(
                      post["m"][1] + 2 * post["m"][2] * np.log(now["U"])), 3)),
        density=dict(known=[round(float(v), 8) for v in density(r_known)],
                     forecast=[round(float(v), 8) for v in density(r_fore)]),
    )
    if verbose:
        k, f, s = out["known"], out["forecast"], out["split"]
        print(f"[{name:5}] link R2 {r2:5.3f} on {len(fit_hours)} hours, "
              f"local wind slope {out['link']['local_wind_slope']:+.2f}")
        print(f"         known     mean {k['mean_mm']:6.1f}  sd {k['sd_mm']:6.1f}"
              f"  90% [{k['q05_mm']:5.1f},{k['q95_mm']:6.1f}]"
              f"  P(exceed) {k['p_exceed']:.3f}")
        print(f"         forecast  mean {f['mean_mm']:6.1f}  sd {f['sd_mm']:6.1f}"
              f"  90% [{f['q05_mm']:5.1f},{f['q95_mm']:6.1f}]"
              f"  P(exceed) {f['p_exceed']:.3f}")
        print(f"         log sd {s['sd_log_model']:.2f} -> {s['sd_log_total']:.2f}"
              f", conditions account for {100 * s['share_conditions']:.0f}%")
    return out


def main(protocol=None, verbose=True):
    protocol = protocol or Protocol()
    trace, X, y, idx = prepare(protocol)
    hours = trace.hour[idx]
    cond = hourly_conditions(protocol.trace)
    rng = np.random.default_rng(SEED)

    out = dict(hour=HOUR, threshold_mm=THRESHOLD_MM,
               wind_sd=WIND_SD, temp_sd=TEMP_SD, draws=N_DRAWS,
               grid_mm=[float(v) for v in GRID_MM],
               conditions=conditions(protocol.trace, HOUR),
               models={})
    for name, build in MODELS:
        out["models"][name] = run_model(name, build, X, y, hours, cond, rng,
                                        verbose=verbose)

    path = os.path.join(RESULTS, "decision_forecast.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    if verbose:
        c = out["conditions"]
        print(f"\nhour {HOUR:g}: wind {c['wind_m_per_s']} +/- {WIND_SD} m/s, "
              f"deck {c['deck_temp_c']} +/- {TEMP_SD} C, "
              f"traffic {c['traffic_veh_per_h']} veh/h known, "
              f"limit {THRESHOLD_MM:g} mm/s^2")
        print("->", path)
    return out


if __name__ == "__main__":
    main()
