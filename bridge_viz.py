"""
Plotly figures for the bridge benchmark (see bridge.py).

    problem_animation(sim)   opening slide: traffic, weather and the bridge's own
                             condition all reach one accelerometer. Time-lapse
                             over the scenario, with play/pause and a scrubber.
    week_overview(sim)       the same timelines as a static figure.
    save_html(fig, path)     standalone, offline copy (plotly.js embedded).

Colours follow the talk: ink for the measured signal, one hue per input
(traffic orange, wind cyan, temperature violet), and red only for damage.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import bridge as br

C_INK = "#111827"
C_MUTED = "#6B7280"
C_GRID = "#E5E7EB"
C_TRAFFIC = "#EA580C"
C_TRAFFIC_FAR = "#FDBA74"
C_WIND = "#0891B2"
C_TEMP = "#7C3AED"
C_ICE = "#0EA5E9"
C_DAMAGE = "#DC2626"
C_DECK = "#9CA3AF"
C_BANK = "#D1D5DB"
C_RIVER = "#BFDBFE"
SKY_DAY = (219, 234, 254)
SKY_NIGHT = (30, 41, 59)
FONT = dict(family="Inter, Helvetica Neue, Arial, sans-serif", size=15, color=C_INK)

SCENE_X = (-7.0, 43.0)
SCENE_Y = (-8.0, 14.0)
SUNRISE, SUNSET = 8.0, 16.75       # late autumn, the Netherlands
DEFLECTION_SCALE = 60.0            # visual exaggeration of the deck motion
THERMO_X = 38.6
TH_BASE = 4.6                      # bottom of the thermometer (above the road)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _daylight(hod: float) -> float:
    up = np.clip((hod - (SUNRISE - 0.75)) / 1.5, 0.0, 1.0)
    down = np.clip(((SUNSET + 0.75) - hod) / 1.5, 0.0, 1.0)
    return float(min(up, down))


def _sky(hod: float) -> str:
    d = _daylight(hod)
    r, g, b = (int(n + (dd - n) * d) for n, dd in zip(SKY_NIGHT, SKY_DAY))
    return f"rgb({r},{g},{b})"


def _sun_or_moon(hod: float):
    if SUNRISE - 0.4 <= hod <= SUNSET + 0.4:
        s = (hod - SUNRISE) / (SUNSET - SUNRISE)
        return -4 + 44 * s, 6.5 + 6.0 * np.sin(np.pi * np.clip(s, 0, 1)), "#F59E0B", 34
    night = 24.0 - (SUNSET - SUNRISE)
    s = ((hod - SUNSET) % 24.0) / night
    return -4 + 44 * s, 8.5 + 3.5 * np.sin(np.pi * s), "#F8FAFC", 22


def _f32(a):
    return np.asarray(a, dtype=np.float32)


def _block_mean(x, n):
    m = len(x) // n * n
    return x[:m].reshape(-1, n).mean(axis=1)


def _poly(xs, ys):
    """Concatenate polygons with NaN gaps (fill='toself' draws each one)."""
    if not xs:
        return np.array([np.nan]), np.array([np.nan])
    X = np.concatenate([np.r_[x, x[0], np.nan] for x in xs])
    Y = np.concatenate([np.r_[y, y[0], np.nan] for y in ys])
    return X, Y


def _segments(x0, y0, x1, y1):
    n = len(x0)
    X = np.empty(3 * n); Y = np.empty(3 * n)
    X[0::3], X[1::3], X[2::3] = x0, x1, np.nan
    Y[0::3], Y[1::3], Y[2::3] = y0, y1, np.nan
    return (X, Y) if n else (np.array([np.nan]), np.array([np.nan]))


def _vehicle_shape(x0, yb, kind, direction):
    """Side-view outline(s) of a car (0) or truck (1) whose rear is at x0."""
    if kind == 0:
        l = 4.2
        px = np.array([0, 1, 1, .78, .62, .26, .14, 0]) * l
        py = np.array([.35, .35, .95, .95, 1.55, 1.55, .95, .95])
        parts = [(px, py)]
        wheels = [.2 * l, .8 * l]
    else:
        l = 10.0
        parts = [(np.array([0, .76, .76, 0]) * l, np.array([.45, .45, 3.1, 3.1])),
                 (np.array([.79, 1, 1, .95, .79]) * l, np.array([.45, .45, 1.7, 2.3, 2.3]))]
        wheels = [.12 * l, .26 * l, .88 * l]
    xs, ys = [], []
    for px, py in parts:
        if direction < 0:          # drive to the left: mirror around the centre
            px = l - px
        xs.append(x0 - l / 2 + px)
        ys.append(yb + py)
    wx = [x0 - l / 2 + (l - w if direction < 0 else w) for w in wheels]
    return xs, ys, wx, [yb + 0.3] * len(wheels)


def _cartoon_traffic(rates, icy, n_frames, rng, veh_per_screen=1 / 260.0):
    """Illustrative traffic whose density follows the true rate profile.

    The time-lapse is far too coarse to show the actual vehicles, so each frame
    shows vehicles moving ~9 m/frame with an on-screen density proportional
    to the expected vehicles per hour.
    """
    road = (SCENE_X[0] - 6, SCENE_X[1] + 6)
    K = (road[1] - road[0]) / 9.0
    lanes = {+1: [], -1: []}
    frames = []
    for f in range(n_frames):
        step = 9.0 * (1.0 - 0.3 * icy[f])
        for d, lane in lanes.items():
            for v in lane:
                v["x"] += d * step
            lane[:] = [v for v in lane if road[0] - 12 <= v["x"] <= road[1] + 12]
            lam = rates[f] * veh_per_screen / K / 2 * 2.0
            for _ in range(rng.poisson(lam)):
                kind = 1 if rng.uniform() < 0.15 else 0
                x = road[0] - rng.uniform(0, 9) if d > 0 else road[1] + rng.uniform(0, 9)
                ends = [v["x"] for v in lane]
                while ends and min(abs(x - e) for e in ends) < 13:
                    x -= d * 13
                lane.append(dict(x=x, kind=kind))
        frames.append({d: [dict(v) for v in lane] for d, lane in lanes.items()})
    return frames


# --------------------------------------------------------------------------- #
# timelines (shared by the animation and the static overview)
# --------------------------------------------------------------------------- #

def _timeline_data(sim: br.BridgeSim, show_hidden: bool):
    sc = sim.scenario
    n = int(round(sc.seconds_per_hour * sim.fs / 6))          # 10 scenario-minutes
    h = _block_mean(sim.hour, n)
    rows = [
        dict(key="traffic", title="Traffic  [vehicles/h]", color=C_TRAFFIC,
             y=_block_mean(sim.traffic_rate, n), range=(0, 1.1 * sim.traffic_rate.max()),
             fill=True),
        dict(key="wind", title="Wind  [m/s]", color=C_WIND,
             y=_block_mean(sim.U, n), range=(0, 1.1 * sim.U.max()), fill=True),
        dict(key="temp", title="Deck temperature  [°C]", color=C_TEMP,
             y=_block_mean(sim.T_deck, n),
             range=(np.floor(sim.T_deck.min() - 1), np.ceil(sim.T_deck.max() + 1)),
             fill=False),
        dict(key="acc", title="Sensor RMS  [mm/s²]", color=C_INK,
             y=np.sqrt(_block_mean(sim.acc[:, 0] ** 2, n)) * 1e3,
             range=(0.0, 3.0), log=True, fill=False),
    ]
    if show_hidden:
        f1 = _block_mean(sim.f_nat[:, 0], n)
        rows.append(dict(key="f1", title="Hidden: 1st natural frequency  [Hz]",
                         color=C_INK, y=f1, dash="dot",
                         range=(f1.min() - 0.04, f1.max() + 0.04), fill=False))
    return h, rows


def _add_timelines(fig, sim, rows, h, y_top, y_bottom, first_axis, show_day_names=True):
    """One small-multiple strip per row. Returns the (x-axis, y-axis) names."""
    k = len(rows)
    gap = 0.014
    height = (y_top - y_bottom - gap * (k - 1)) / k
    days = sim.scenario.days
    first_x = "x" if first_axis == 1 else f"x{first_axis}"
    for i, row in enumerate(rows):
        ax = first_axis + i
        sfx = "" if ax == 1 else str(ax)
        xa, ya = f"x{sfx}", f"y{sfx}"
        top = y_top - i * (height + gap)
        is_log = row.get("log", False)
        rng = row["range"]
        fig.update_layout(**{
            f"xaxis{sfx}": dict(domain=[0.0, 1.0], anchor=ya, range=[0, 24 * days],
                               matches=None if i == 0 else first_x,
                               showticklabels=(i == k - 1) and show_day_names,
                               tickvals=[24 * d + 12 for d in range(days)],
                               ticktext=[sim.label(24 * d)[:3] for d in range(days)],
                               ticks="", showgrid=False, zeroline=False,
                               showline=False, fixedrange=True),
            f"yaxis{sfx}": dict(domain=[top - height, top], anchor=xa,
                               range=list(rng), type="log" if is_log else "linear",
                               showgrid=True, gridcolor=C_GRID, zeroline=False,
                               tickfont=dict(size=11, color=C_MUTED), nticks=3,
                               fixedrange=True, side="right"),
        })
        y = row["y"]
        if row["key"] == "temp":
            fig.add_trace(go.Scatter(x=h, y=np.minimum(y, 0.0), xaxis=xa, yaxis=ya,
                                     fill="tozeroy", fillcolor="rgba(14,165,233,0.30)",
                                     line=dict(width=0), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=[0, 24 * days], y=[0, 0], xaxis=xa, yaxis=ya,
                                     mode="lines", line=dict(color=C_ICE, width=1, dash="dash"),
                                     hoverinfo="skip", showlegend=False))
        fill_rgba = None
        if row.get("fill"):
            c = row["color"].lstrip("#")
            fill_rgba = "rgba({},{},{},0.15)".format(*(int(c[j:j + 2], 16) for j in (0, 2, 4)))
        fig.add_trace(go.Scatter(
            x=h, y=y, xaxis=xa, yaxis=ya, mode="lines",
            line=dict(color=row["color"], width=2, dash=row.get("dash", "solid")),
            fill="tozeroy" if fill_rgba else None, fillcolor=fill_rgba,
            name=row["title"], showlegend=False,
            hovertemplate="%{y:.3g}<extra>" + row["title"] + "</extra>"))
        if row["key"] == "f1" and sim.scenario.damage_at_h is not None:
            fig.add_trace(go.Scatter(
                x=[sim.scenario.damage_at_h], y=[rng[1] - 0.25 * (rng[1] - rng[0])],
                xaxis=xa, yaxis=ya, mode="markers+text", text=["damage"],
                textposition="middle right", textfont=dict(color=C_DAMAGE, size=12),
                marker=dict(symbol="x-thin-open", size=12, color=C_DAMAGE, line=dict(width=3)),
                hoverinfo="skip", showlegend=False))
        fig.add_annotation(x=0.004, y=top - 0.004, xref="paper", yref="paper",
                           xanchor="left", yanchor="top", showarrow=False,
                           text=f"<b>{row['title']}</b>",
                           font=dict(size=12, color=row["color"]),
                           bgcolor="rgba(255,255,255,0.8)")
        for d in range(1, days):
            fig.add_shape(type="line", xref=xa, yref=f"{ya} domain", x0=24 * d, x1=24 * d,
                          y0=0, y1=1, line=dict(color=C_GRID, width=1))
        row["axes"] = (xa, ya)
        row["yspan"] = (10 ** rng[0], 10 ** rng[1]) if is_log else rng
    return rows


# --------------------------------------------------------------------------- #
# the opening animation
# --------------------------------------------------------------------------- #

def problem_animation(sim: br.BridgeSim, frames_per_hour: int = 3,
                      frame_ms: int = 110, show_hidden: bool = False,
                      start_hour: float = 2.0, height: int = 760,
                      seed: int = 0) -> go.Figure:
    """Time-lapse of the scenario: inputs -> bridge -> one accelerometer."""
    rng = np.random.default_rng(seed)
    pr, sc, fs = sim.params, sim.scenario, sim.fs
    L = pr.L
    hours = np.arange(start_hour, sc.hours + 1e-9, 1.0 / frames_per_hour)
    idx = np.array([sim.index(hh) for hh in hours])
    nF = len(hours)
    fig = go.Figure()

    # ---------- scene: static background ------------------------------------
    sx = dict(domain=[0.0, 0.64], anchor="y", range=list(SCENE_X), visible=False,
              fixedrange=True, constrain="domain")
    sy = dict(domain=[0.42, 0.935], anchor="x", range=list(SCENE_Y), visible=False,
              scaleanchor="x", scaleratio=1, constrain="domain", fixedrange=True)
    fig.update_layout(xaxis=sx, yaxis=sy)
    S = dict(xaxis="x", yaxis="y", hoverinfo="skip", showlegend=False)

    def dyn(trace):              # add a trace that the frames will update
        fig.add_trace(trace)
        return len(fig.data) - 1

    hod0 = hours[0] % 24
    t_sky = dyn(go.Scatter(x=[SCENE_X[0], SCENE_X[1], SCENE_X[1], SCENE_X[0]],
                           y=[SCENE_Y[0], SCENE_Y[0], SCENE_Y[1], SCENE_Y[1]], fill="toself",
                           fillcolor=_sky(hod0), line=dict(width=0), mode="lines", **S))
    star_x = rng.uniform(SCENE_X[0] + 1, SCENE_X[1] - 1, 22)
    star_y = rng.uniform(7.5, 13.5, 22)
    t_stars = dyn(go.Scatter(x=star_x, y=star_y, mode="markers",
                             marker=dict(symbol="star", size=5, color="#F8FAFC"),
                             opacity=1 - _daylight(hod0), **S))
    sun = _sun_or_moon(hod0)
    t_sun = dyn(go.Scatter(x=[sun[0]], y=[sun[1]], mode="markers",
                           marker=dict(size=sun[3], color=sun[2]), **S))
    cloud_c = np.array([[2, 12.3], [13, 12.8], [24, 12.2], [33, 12.9]])
    cx = np.concatenate([c[0] + np.array([-2.2, 0, 2.2, 1.0, -1.0]) for c in cloud_c])
    cy = np.concatenate([c[1] + np.array([0, 0.5, 0, -0.4, -0.4]) for c in cloud_c])
    t_clouds = dyn(go.Scatter(x=cx, y=cy, mode="markers",
                              marker=dict(size=44, color="#94A3B8"), opacity=0.0, **S))
    rain_x0 = rng.uniform(SCENE_X[0], SCENE_X[1], 36)
    rain_ph = rng.uniform(0, 1, 36)
    t_rain = dyn(go.Scatter(x=[np.nan], y=[np.nan], mode="lines",
                            line=dict(color="#60A5FA", width=1.5), opacity=0.0, **S))
    wind_y = rng.uniform(3.0, 11.5, 16)
    wind_ph = rng.uniform(0, 1, 16)
    t_wind = dyn(go.Scatter(x=[np.nan], y=[np.nan], mode="lines",
                            line=dict(color=C_WIND, width=2.5), **S))
    t_windh = dyn(go.Scatter(x=[np.nan], y=[np.nan], mode="markers",
                             marker=dict(symbol="triangle-right", size=10, color=C_WIND), **S))

    # river and banks (static)
    fig.add_trace(go.Scatter(x=[0, L, L, 0], y=[-8, -8, -5.6, -5.6], fill="toself",
                             fillcolor=C_RIVER, line=dict(width=0), mode="lines", **S))
    for bx in ([SCENE_X[0], 0.0, 3.0, SCENE_X[0]], [SCENE_X[1], L, L - 3.0, SCENE_X[1]]):
        fig.add_trace(go.Scatter(x=bx, y=[-0.25, -0.25, -8, -8], fill="toself",
                                 fillcolor=C_BANK, line=dict(width=0), mode="lines", **S))
    fig.add_trace(go.Scatter(x=[SCENE_X[0], 0, None, L, SCENE_X[1]], y=[0, 0, None, 0, 0],
                             mode="lines", line=dict(color=C_DECK, width=9), **S))
    fig.add_trace(go.Scatter(x=[0, L], y=[-0.75, -0.75], mode="markers",
                             marker=dict(symbol="triangle-up", size=16, color="#4B5563"), **S))

    # thermometer board (static)
    fig.add_trace(go.Scatter(x=[THERMO_X - 1.8, THERMO_X + 3.4, THERMO_X + 3.4, THERMO_X - 1.8],
                             y=[TH_BASE - 1.4, TH_BASE - 1.4, 13.6, 13.6], fill="toself",
                             fillcolor="rgba(255,255,255,0.88)", line=dict(width=0),
                             mode="lines", **S))
    T_lo, T_hi = -10.0, 20.0
    th_y = lambda T: TH_BASE + 0.4 + 8.0 * (np.clip(T, T_lo, T_hi) - T_lo) / (T_hi - T_lo)
    fig.add_trace(go.Scatter(x=[THERMO_X, THERMO_X], y=[TH_BASE, th_y(T_hi) + 0.3], mode="lines",
                             line=dict(color=C_GRID, width=14), **S))
    fig.add_trace(go.Scatter(x=[THERMO_X + 0.8] * 4, y=[th_y(T) for T in (-10, 0, 10, 20)],
                             mode="text", text=["−10", "0", "10", "20 °C"],
                             textposition="middle right", textfont=dict(size=11, color=C_MUTED),
                             **S))
    fig.add_trace(go.Scatter(x=[THERMO_X - 0.8, THERMO_X + 0.8], y=[th_y(0)] * 2,
                             mode="lines", line=dict(color=C_ICE, width=1.5), **S))

    # ---------- scene: dynamic foreground ----------------------------------
    xs = np.linspace(0, L, 61)
    Phi = np.array([pr.phi(xs / L, n + 1) for n in range(pr.n_modes)])      # (M,61)

    def deck_y(k):
        return np.clip(-DEFLECTION_SCALE * (sim.q[k] @ Phi), -2.2, 2.2)

    t_deck = dyn(go.Scatter(x=xs, y=deck_y(idx[0]), mode="lines",
                            line=dict(color=C_DECK, width=9, shape="spline"), **S))
    frost_x = np.linspace(1.5, L - 1.5, 15)
    t_frost = dyn(go.Scatter(x=frost_x, y=np.zeros_like(frost_x), mode="markers",
                             marker=dict(symbol="asterisk-open", size=11, color=C_ICE,
                                         line=dict(width=2)), opacity=0.0, **S))
    t_far = dyn(go.Scatter(x=[np.nan], y=[np.nan], fill="toself", mode="lines",
                           fillcolor=C_TRAFFIC_FAR, line=dict(width=0), **S))
    t_near = dyn(go.Scatter(x=[np.nan], y=[np.nan], fill="toself", mode="lines",
                            fillcolor=C_TRAFFIC, line=dict(width=0), **S))
    t_wheels = dyn(go.Scatter(x=[np.nan], y=[np.nan], mode="markers",
                              marker=dict(size=7, color="#1F2937"), **S))
    xs_s = pr.sensor_x[0] * L
    t_sensor = dyn(go.Scatter(x=[xs_s], y=[-0.7], mode="markers",
                              marker=dict(symbol="square", size=12, color=C_INK,
                                          line=dict(color="white", width=1.5)), **S))
    fig.add_annotation(x=xs_s, y=-2.3, xref="x", yref="y", showarrow=False,
                       text="accelerometer", font=dict(size=12, color=C_INK),
                       bgcolor="rgba(255,255,255,0.85)", borderpad=2)
    t_thermo = dyn(go.Scatter(x=[THERMO_X, THERMO_X], y=[TH_BASE, th_y(sim.T_deck[idx[0]])],
                              mode="lines", line=dict(color=C_TEMP, width=8), **S))
    fig.add_trace(go.Scatter(x=[THERMO_X], y=[TH_BASE - 0.1], mode="markers",
                             marker=dict(size=22, color=C_TEMP), **S))

    # ---------- readout row ------------------------------------------------
    fig.update_layout(xaxis2=dict(domain=[0.0, 0.64], anchor="y2", range=[0, 1],
                                  visible=False, fixedrange=True),
                      yaxis2=dict(domain=[0.945, 1.0], anchor="x2", range=[0, 1],
                                  visible=False, fixedrange=True))

    def readout(k, hh):
        T = sim.T_deck[k]
        state = "  <span style='color:%s'><b>frozen</b></span>" % C_ICE if T < 0 else ""
        return [f"<b>{sim.label(hh)}</b>",
                f"traffic <b>{sim.traffic_rate[k]:,.0f}</b> veh/h".replace(",", " "),
                f"wind <b>{sim.U[k]:.0f}</b> m/s",
                f"deck <b>{T:+.1f}</b> °C{state}"]

    t_read = dyn(go.Scatter(x=[0.0, 0.22, 0.50, 0.72], y=[0.5] * 4, mode="text",
                            text=readout(idx[0], hours[0]), textposition="middle right",
                            textfont=dict(size=17, color=[C_INK, C_TRAFFIC, C_WIND, C_TEMP]),
                            xaxis="x2", yaxis="y2", hoverinfo="skip", showlegend=False))

    # ---------- oscilloscope -------------------------------------------------
    W = int(round(10 * fs))
    ts = (np.arange(W) - W + 1) / fs
    y_acc_max = 0.8
    fig.update_layout(
        xaxis3=dict(domain=[0.70, 1.0], anchor="y3", range=[ts[0], 0], fixedrange=True,
                    showgrid=False, zeroline=False, tickfont=dict(size=11, color=C_MUTED),
                    ticksuffix=" s"),
        yaxis3=dict(domain=[0.74, 0.935], anchor="x3", range=[-y_acc_max, y_acc_max],
                    fixedrange=True, gridcolor=C_GRID, zeroline=True, zerolinecolor=C_GRID,
                    tickfont=dict(size=11, color=C_MUTED), nticks=5))
    acc = sim.acc[:, 0]

    def scope(k):
        seg = acc[max(0, k - W + 1):k + 1]
        return np.r_[np.full(W - len(seg), np.nan), seg].astype(np.float32)

    t_scope = dyn(go.Scatter(x=ts, y=scope(idx[0]), mode="lines", xaxis="x3", yaxis="y3",
                             line=dict(color=C_INK, width=1.4), hoverinfo="skip",
                             showlegend=False))

    def rms_text(k):
        seg = acc[max(0, k - W + 1):k + 1]
        return [f"RMS <b>{np.sqrt(np.mean(seg ** 2)) * 1e3:.0f}</b> mm/s²"]

    t_rms = dyn(go.Scatter(x=[ts[0] + 0.2], y=[0.8 * y_acc_max], mode="text",
                           text=rms_text(idx[0]), textposition="middle right",
                           textfont=dict(size=13, color=C_INK), xaxis="x3", yaxis="y3",
                           hoverinfo="skip", showlegend=False))
    fig.add_annotation(x=0.70, y=0.975, xref="paper", yref="paper", xanchor="left",
                       showarrow=False, text="<b>Sensor</b>: acceleration, last 10 s  [m/s²]",
                       font=dict(size=13, color=C_INK))

    # ---------- spectrum -----------------------------------------------------
    Wp = int(round(60 * fs))
    f_psd, _ = br.psd(acc[:Wp], fs, 256)

    def spectrum(k):
        seg = acc[max(0, k - Wp + 1):k + 1]
        _, P = br.psd(seg, fs, 256)
        return (10 * np.log10(P + 1e-14)).astype(np.float32)

    fig.update_layout(
        xaxis4=dict(domain=[0.70, 1.0], anchor="y4", range=[0, fs / 2], fixedrange=True,
                    showgrid=False, zeroline=False, tickfont=dict(size=11, color=C_MUTED),
                    ticksuffix=" Hz"),
        yaxis4=dict(domain=[0.44, 0.64], anchor="x4", range=[-85, 0], fixedrange=True,
                    gridcolor=C_GRID, zeroline=False, tickfont=dict(size=11, color=C_MUTED),
                    nticks=4))
    for n, f0 in enumerate(pr.f_ref):
        fig.add_trace(go.Scatter(x=[f0, f0], y=[-85, 0], mode="lines", xaxis="x4", yaxis="y4",
                                 line=dict(color=C_MUTED, width=1, dash="dot"),
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[pr.f_ref[-1] + 0.15], y=[-6], mode="text", xaxis="x4",
                             yaxis="y4", text=["reference modes"], textposition="middle right",
                             textfont=dict(size=11, color=C_MUTED), hoverinfo="skip",
                             showlegend=False))
    t_psd = dyn(go.Scatter(x=f_psd, y=spectrum(idx[0]), mode="lines", xaxis="x4", yaxis="y4",
                           line=dict(color=C_INK, width=1.8), hoverinfo="skip",
                           showlegend=False))
    t_true_f = None
    if show_hidden:
        t_true_f = dyn(go.Scatter(x=sim.f_nat[idx[0]], y=[-3, -3], mode="markers",
                                  xaxis="x4", yaxis="y4",
                                  marker=dict(symbol="triangle-down", size=11, color=C_DAMAGE),
                                  hoverinfo="skip", showlegend=False))
    fig.add_annotation(x=0.70, y=0.675, xref="paper", yref="paper", xanchor="left",
                       showarrow=False, text="<b>Spectrum</b>, last 60 s  [dB]",
                       font=dict(size=13, color=C_INK))

    # ---------- timelines + fog of the future ----------------------------------
    h_tl, rows = _timeline_data(sim, show_hidden)
    rows = _add_timelines(fig, sim, rows, h_tl, y_top=0.365, y_bottom=0.035, first_axis=5)
    t_fog, t_cur = [], []
    for row in rows:
        xa, ya = row["axes"]
        y0, y1 = row["yspan"]
        c = hours[0]
        t_fog.append(dyn(go.Scatter(x=[c, sc.hours, sc.hours, c], y=[y0, y0, y1, y1],
                                    fill="toself", fillcolor="rgba(255,255,255,0.82)",
                                    line=dict(width=0), mode="lines", xaxis=xa, yaxis=ya,
                                    hoverinfo="skip", showlegend=False)))
        t_cur.append(dyn(go.Scatter(x=[c, c], y=[y0, y1], mode="lines", xaxis=xa, yaxis=ya,
                                    line=dict(color=C_INK, width=1.5), hoverinfo="skip",
                                    showlegend=False)))

    # ---------- frames -------------------------------------------------------
    rates = sim.traffic_rate[idx]
    icy = (sim.T_deck[idx] < 0.5).astype(float)
    traffic = _cartoon_traffic(rates, icy, nF, rng)
    dyn_ids = [t_sky, t_stars, t_sun, t_clouds, t_rain, t_wind, t_windh, t_deck, t_frost,
               t_far, t_near, t_wheels, t_sensor, t_thermo, t_read, t_scope, t_rms, t_psd]
    if t_true_f is not None:
        dyn_ids.append(t_true_f)
    dyn_ids += t_fog + t_cur

    n_traces = len(fig.data)
    frames = []
    for f in range(nF):
        k, hh = int(idx[f]), float(hours[f])
        hod = hh % 24
        U, T = float(sim.U[k]), float(sim.T_deck[k])
        dy = deck_y(k)
        on_deck = lambda x: np.where((x >= 0) & (x <= L), np.interp(x, xs, dy), 0.0)
        data = []
        data.append(go.Scatter(fillcolor=_sky(hod)))
        data.append(go.Scatter(opacity=1 - _daylight(hod)))
        sx_, sy_, sc_, ss_ = _sun_or_moon(hod)
        data.append(go.Scatter(x=[sx_], y=[sy_], marker=dict(size=ss_, color=sc_)))
        data.append(go.Scatter(opacity=float(np.clip((U - 8) / 9, 0, 0.9))))
        ry = SCENE_Y[1] - ((rain_ph * 17 + f * 3.1) % 17)
        rx = rain_x0 - 0.02 * U * (SCENE_Y[1] - ry)
        rain_op = float(np.clip((U - 11) / 7, 0, 0.8))
        X, Y = _segments(rx, ry, rx - 0.04 * U, ry - 1.3) if rain_op > 0 else ([np.nan],) * 2
        data.append(go.Scatter(x=_f32(X), y=_f32(Y), opacity=rain_op))
        n_arrow = int(np.clip(round(U / 1.3), 1, 16))
        span = SCENE_X[1] - SCENE_X[0] + 8
        wx0 = SCENE_X[0] - 4 + (wind_ph * span + f * 0.9 * U) % span
        ln = 0.9 + 0.28 * U
        wx0, wy = wx0[:n_arrow], wind_y[:n_arrow]
        X, Y = _segments(wx0, wy, wx0 + ln, wy)
        data.append(go.Scatter(x=_f32(X), y=_f32(Y), opacity=float(0.45 + 0.55 * min(U / 18, 1))))
        data.append(go.Scatter(x=_f32(wx0 + ln), y=_f32(wy),
                               opacity=float(0.45 + 0.55 * min(U / 18, 1))))
        frozen = float(1 / (1 + np.exp(T / 0.4)))
        data.append(go.Scatter(y=_f32(dy), line=dict(color=C_ICE if T < 0 else C_DECK, width=9,
                                               shape="spline")))
        data.append(go.Scatter(y=_f32(np.interp(frost_x, xs, dy) + 0.7), opacity=frozen))
        for d in (-1, +1):
            px, py, wx, wy2 = [], [], [], []
            for v in traffic[f][d]:
                a, b, c_, e = _vehicle_shape(v["x"], float(on_deck(np.array([v["x"]]))[0]),
                                             v["kind"], d)
                px += a; py += b; wx += c_; wy2 += e
            X, Y = _poly(px, py)
            data.append(go.Scatter(x=_f32(X), y=_f32(Y)))
            if d < 0:
                far_wheels = (wx, wy2)
        wx_all = far_wheels[0] + wx
        wy_all = far_wheels[1] + wy2
        data.append(go.Scatter(x=wx_all or [np.nan], y=wy_all or [np.nan]))
        data.append(go.Scatter(y=[float(np.interp(xs_s, xs, dy)) - 0.7]))
        data.append(go.Scatter(y=[TH_BASE, th_y(T)],
                               line=dict(color=C_ICE if T < 0 else C_TEMP, width=8)))
        data.append(go.Scatter(text=readout(k, hh)))
        data.append(go.Scatter(y=scope(k)))
        data.append(go.Scatter(text=rms_text(k)))
        data.append(go.Scatter(y=spectrum(k)))
        if t_true_f is not None:
            data.append(go.Scatter(x=sim.f_nat[k].astype(np.float32)))
        for row in rows:
            y0, y1 = row["yspan"]
            data.append(go.Scatter(x=[hh, sc.hours, sc.hours, hh]))
        for row in rows:
            data.append(go.Scatter(x=[hh, hh]))
        # Give every trace an entry (static ones empty) so plotly redraws the
        # full, ordered trace stack; partial `traces` updates can reorder layers.
        full = [go.Scatter() for _ in range(n_traces)]
        for i, d in zip(dyn_ids, data):
            full[i] = d
        frames.append(go.Frame(data=full, name=str(f)))
    fig.frames = frames

    # ---------- controls -------------------------------------------------------
    play = dict(frame=dict(duration=frame_ms, redraw=True), transition=dict(duration=0),
                fromcurrent=True, mode="immediate")
    fig.update_layout(
        height=height, template="plotly_white", font=FONT,
        margin=dict(l=12, r=48, t=10, b=70), plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, hovermode=False, dragmode=False,
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.035,
                          xanchor="left", yanchor="top", pad=dict(t=0, r=6),
                          showactive=False,
                          buttons=[dict(label="▶ Play", method="animate", args=[None, play]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                           mode="immediate",
                                                           transition=dict(duration=0))])])],
        sliders=[dict(active=0, x=0.14, len=0.86, y=-0.02, xanchor="left", yanchor="top",
                      pad=dict(t=0, b=0), ticklen=0, minorticklen=0,
                      font=dict(color="rgba(0,0,0,0)", size=1),
                      currentvalue=dict(visible=True, xanchor="right", prefix="",
                                        font=dict(size=13, color=C_MUTED)),
                      steps=[dict(method="animate", label=sim.label(hh),
                                  args=[[str(f)], dict(frame=dict(duration=0, redraw=True),
                                                       mode="immediate",
                                                       transition=dict(duration=0))])
                             for f, hh in enumerate(hours)])])
    return fig


# --------------------------------------------------------------------------- #
# static overview
# --------------------------------------------------------------------------- #

def week_overview(sim: br.BridgeSim, show_hidden: bool = True, height: int = 520) -> go.Figure:
    fig = go.Figure()
    h, rows = _timeline_data(sim, show_hidden)
    _add_timelines(fig, sim, rows, h, y_top=0.99, y_bottom=0.06, first_axis=1)
    fig.update_layout(height=height, template="plotly_white", font=FONT, showlegend=False,
                      margin=dict(l=12, r=48, t=10, b=30), plot_bgcolor="white")
    return fig


def save_html(fig: go.Figure, path: str) -> str:
    """Standalone copy for presenting without the notebook (works offline)."""
    fig.write_html(path, include_plotlyjs=True, full_html=True, auto_play=False,
                   config=dict(displayModeBar=False, responsive=True))
    return path


if __name__ == "__main__":
    import time
    t0 = time.time()
    sim = br.simulate(seed=0)
    fig = problem_animation(sim)
    print(f"{len(fig.frames)} frames built in {time.time() - t0:.1f} s")
    save_html(fig, "bridge_problem.html")
    print("wrote bridge_problem.html")
