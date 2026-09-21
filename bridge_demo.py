import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import bridge as br
    import bridge_viz as bv

    return br, bv, mo


@app.cell
def _(mo):
    mo.md(r"""
    # One bridge, one sensor, many causes

    A 30 m road bridge carries a single accelerometer. Its signal never sits still:
    **rush hour** and a **storm** make it loud, a **frozen deck** makes it stiffer,
    and at some point this week something in the structure **breaks**.
    Traffic and wind are only partly measured; the bridge's own condition not at all.

    *From the sensor alone: what changed — the load, the weather, or the bridge?*
    """)
    return


@app.cell
def _(mo):
    speed = mo.ui.dropdown({"slow": 180, "normal": 110, "fast": 60}, value="normal",
                           label="playback")
    fph = mo.ui.dropdown({"2 per hour": 2, "3 per hour": 3, "4 per hour": 4},
                         value="3 per hour", label="frames")
    seed = mo.ui.number(start=0, stop=999, step=1, value=0, label="seed")
    reveal = mo.ui.switch(value=False, label="reveal hidden truth")
    mo.hstack([speed, fph, seed, reveal], justify="start", gap=2)
    return fph, reveal, seed, speed


@app.cell
def _(br, seed):
    sim = br.simulate(seed=int(seed.value))
    return (sim,)


@app.cell
def _(bv, fph, reveal, sim, speed):
    problem_fig = bv.problem_animation(sim, frames_per_hour=fph.value,
                                       frame_ms=speed.value, show_hidden=reveal.value)
    problem_fig
    return (problem_fig,)


@app.cell
def _(mo):
    save = mo.ui.run_button(label="Save standalone HTML (offline backup)")
    save
    return (save,)


@app.cell
def _(bv, mo, problem_fig, save):
    mo.stop(not save.value)
    _path = bv.save_html(problem_fig, str(mo.notebook_dir() / "bridge_problem.html"))
    mo.md(f"Saved `{_path}`: opens in any browser, no internet needed.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reading the animation.** Time-lapse: one scenario hour is 30 s of 20 Hz
    vibration, so the full week is one hour of data (72 000 samples). The scene
    shows the inputs (traffic density, wind, deck temperature); top right is what
    the accelerometer records: the last 10 s, and its spectrum over the last 60 s
    against the reference modes. The strips below trace the week; the future is
    greyed out.
    """)
    return


@app.cell
def _(bv, mo, sim):
    mo.accordion({
        "Behind the scenes (spoilers)": mo.vstack([
            mo.md(
                r"""
                The natural frequencies drift ~0.3 %/°C with deck temperature, jump
                ~+10 % while the deck is frozen (Wednesday night to Thursday morning,
                briefly again on Friday morning), and drop ~6 % after damage on
                Friday at 13:00. Switch on *reveal hidden truth* to see the true
                modes in the spectrum and the true first frequency in the strips.
                """
            ),
            bv.week_overview(sim),
        ])
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Part two --- does the model forecast?

    One-step-ahead forecasts from the six models in `models/`, on the same
    bridge trace the animation above plays back. Every model predicted each
    sample before it saw it, on the same window, with hyperparameters chosen on
    day one only.

    The plotting code below is meant to be edited. Change `STYLE`, change a
    `*_figure` function, and every panel re-renders. Running this file as a
    script (`python bridge_demo.py`) writes the slide PDFs into `../figures/`.
    """)
    return


@app.cell
def _():
    import json
    import os

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HERE = os.path.dirname(os.path.abspath(__file__))
    RESULTS = os.path.join(HERE, "results")
    FIGURES = os.path.abspath(os.path.join(HERE, "..", "figures"))

    STAGES = [
        ("ar", 1, "AR"),
        ("tvar", 2, "TVAR"),
        ("htvar", 3, "hier. TVAR"),
        ("hgf", 4, "+ HGF noise"),
        ("moe", 5, "mixture"),
        ("bmoe", 6, "Bayes. mixture"),
    ]
    # Outside comparisons, scored the same way but not rungs of the ladder,
    # so the per-stage slide figures keep their six slots.
    BASELINES = [
        ("matern", 7, "Matern-1/2 GP"),
        ("transformer", 8, "Transformer"),
        ("transformer_nightly", 9, "Transformer (nightly)"),
    ]
    return BASELINES, FIGURES, HERE, RESULTS, STAGES, json, np, os, plt


@app.cell
def _(BASELINES, RESULTS, STAGES, json, np, os):
    def load(name):
        npz = dict(np.load(os.path.join(RESULTS, f"{name}.npz")))
        with open(os.path.join(RESULTS, f"{name}.json")) as fh:
            meta = json.load(fh)
        return npz, meta

    RES = {}
    for _n, _s, _l in STAGES + BASELINES:
        if os.path.exists(os.path.join(RESULTS, f"{_n}.npz")):
            RES[_n] = load(_n)

    AVAILABLE = [(n, s, l) for n, s, l in STAGES if n in RES]
    HAVE_BASE = [(n, s, l) for n, s, l in BASELINES if n in RES]
    PREV = {n: (AVAILABLE[i - 1][0] if i else None)
            for i, (n, _, _) in enumerate(AVAILABLE)}
    return AVAILABLE, HAVE_BASE, PREV, RES, load


@app.cell
def _():
    # ---------------------------------------------------------------- style
    # One ramp for the six stages, one contrast hue for the deep baseline.
    #
    # The stages are ordered, so they get an ordered ramp rather than six
    # unrelated colours. It runs 226 deg to 35 deg in hue and 0.74 to 0.46 in
    # OKLab lightness, which means neighbouring stages are separated by
    # lightness as well as hue and survive being printed grey or seen by a
    # deuteranope. Adjacent steps are 9.0 to 11.7 apart in CIEDE2000 and the
    # ends are 38 apart; the blue ramp this replaced managed 8.1 to 9.9 with
    # two neighbours that were hard to tell apart on a projector.
    #
    # The Transformer is not a rung of this ladder, so it gets a hue that is
    # not on the ramp at all. {deep, stage[-1], stage[0]} pass the categorical
    # separation checks as a triple.
    STYLE = dict(
        stage=["#5DB8DB", "#25AFA6", "#4C9D5D",
               "#7D7D10", "#8D5C0B", "#9E250C"],
        deep="#A146A7",       # the end-to-end network
        now="#9E250C",        # single-model panels
        before="#25AFA6",     # the one thing being contrasted with it
        ink="#111827",
        muted="#6B7280",
        faint="#D1D5DB",
        grid="#E5E7EB",
        lw=1.7,
        fs=8.5,               # base font size, tuned for a 16:9 beamer frame
        figsize=(7.2, 2.5),   # inches; \gfx scales it to the frame
        dpi=200,
    )

    # How far back a stage is from the one on the slide decides how loud it
    # is. The current stage is solid, the one before it is still readable so
    # the pairwise comparison can be made, everything older is a ghost.
    FADE = [(1.00, 1.00), (0.55, 0.75), (0.20, 0.62), (0.12, 0.55)]

    def fade(age):
        """(alpha, line-width factor) for a stage `age` steps in the past."""
        return FADE[min(age, len(FADE) - 1)]

    def apply_style(style):
        import matplotlib.pyplot as _plt
        _plt.rcParams.update({
            "font.size": style["fs"],
            "axes.titlesize": style["fs"],
            "axes.labelsize": style["fs"],
            "xtick.labelsize": style["fs"] - 0.5,
            "ytick.labelsize": style["fs"] - 0.5,
            "legend.fontsize": style["fs"] - 0.5,
            "axes.edgecolor": style["muted"],
            "axes.labelcolor": style["ink"],
            "text.color": style["ink"],
            "xtick.color": style["muted"],
            "ytick.color": style["muted"],
            "axes.linewidth": 0.6,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
        })
    return FADE, STYLE, apply_style, fade


@app.cell
def _(STYLE, apply_style, plt):
    apply_style(STYLE)

    def tidy(ax, style=STYLE, grid_axis="y"):
        """Recessive grid, no box, ticks outward. Applied to every panel."""
        ax.grid(True, axis=grid_axis, color=style["grid"], lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=2.5, width=0.6)
        return ax
    return (tidy,)


@app.cell
def _(AVAILABLE, RES, STYLE, fade, np, plt, tidy):
    # ------------------------------------------------------- fixed geometry
    # Both panels on the stage slides keep the same axes from the first stage
    # to the last. If the axis moves underneath a sequence of slides, every
    # model looks equally good and the point of the sequence is lost.
    LADDER = [(n, s, l) for n, s, l in AVAILABLE]
    _runs = [RES[n][0]["run_nlpd"] for n, _, _ in LADDER]
    RUN_YLIM = (min(float(r.min()) for r in _runs) - 0.15,
                max(float(r.max()) for r in _runs) + 0.25)
    _bars = [RES[n][1]["metrics"]["nlpd"] for n, _, _ in LADDER]
    BAR_YLIM = (min(_bars) * 1.18, 0.0)

    def stage_index(name):
        return [i for i, (n, _, _) in enumerate(LADDER) if n == name][0]

    def stage_color(name, style=STYLE):
        return style["stage"][stage_index(name) % len(style["stage"])]

    def running_panel(ax, name, style=STYLE):
        """Running NLPD over the scored window, this stage on top of its past.

        Every stage up to this one is drawn, each in its own colour, each
        fainter the further back it is. Only the current stage and the one
        before it are named in the legend, so the slide asks the audience to
        make one comparison rather than six.
        """
        here = stage_index(name)
        for i in range(here + 1):
            n, _, _ = LADDER[i]
            npz, meta = RES[n]
            alpha, lwf = fade(here - i)
            ax.plot(npz["step"] / 600.0, npz["run_nlpd"],
                    color=style["stage"][i % len(style["stage"])],
                    lw=style["lw"] * lwf, alpha=alpha, zorder=2 + i,
                    label=meta["label"] if here - i <= 1 else None)
        ax.set_xlabel("scenario hour")
        ax.set_ylabel("running NLPD  [nats]")
        ax.set_ylim(*RUN_YLIM)
        if here > 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], frameon=False, loc="best",
                      handlelength=1.4, borderpad=0.2)
        ax.annotate("lower is better", (0.985, 0.04), xycoords="axes fraction",
                    ha="right", fontsize=style["fs"] - 2, color=style["muted"])
        tidy(ax)
        return ax

    def nlpd_panel(ax, name, style=STYLE):
        """Mean NLPD, one bar per stage, on an axis that never moves.

        The axis always spans all six slots and the same range, whichever
        stage is on the slide, so the bars fill in from left to right as the
        talk goes and the heights stay comparable across slides.
        """
        here = stage_index(name)
        xs = np.arange(len(LADDER))
        for i, (n, _, _) in enumerate(LADDER):
            if i > here:
                break
            alpha, _ = fade(here - i)
            v = RES[n][1]["metrics"]["nlpd"]
            ax.bar([i], [v], color=style["stage"][i % len(style["stage"])],
                   alpha=alpha, width=0.62, zorder=3)
            ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points",
                        xytext=(0, -11), ha="center",
                        fontsize=style["fs"] - 1.5,
                        color=style["ink"] if i == here else style["muted"],
                        alpha=1.0 if here - i <= 1 else 0.55)
        ax.set_xticks(xs)
        ax.set_xticklabels([l for _, _, l in LADDER], rotation=20, ha="right")
        ax.set_xlim(-0.75, len(LADDER) - 0.25)
        ax.set_ylim(*BAR_YLIM)
        ax.set_ylabel("mean NLPD  [nats]")
        tidy(ax)
        return ax

    def stage_figure(name, style=STYLE):
        """The figure that follows each factor-graph slide."""
        fig, axes = plt.subplots(1, 2, figsize=style["figsize"],
                                 gridspec_kw=dict(width_ratios=[1.35, 1.0],
                                                  wspace=0.32))
        running_panel(axes[0], name, style)
        nlpd_panel(axes[1], name, style)
        return fig
    return (BAR_YLIM, LADDER, RUN_YLIM, nlpd_panel, running_panel,
            stage_color, stage_figure, stage_index)


@app.cell
def _(AVAILABLE, RES, STYLE, np, plt, tidy):
    def ladder_figure(style=STYLE):
        """All six stages at once, for the discussion slide."""
        names = [n for n, _, _ in AVAILABLE]
        labels = [l for _, _, l in AVAILABLE]
        nlpd = np.array([RES[n][1]["metrics"]["nlpd"] for n in names])
        rmse = np.array([RES[n][1]["metrics"]["rmse"] for n in names]) * 1e3
        cov = np.array([RES[n][1]["metrics"]["coverage90"] for n in names])
        xs = np.arange(len(names))

        fig, axes = plt.subplots(1, 3, figsize=(style["figsize"][0], 2.3),
                                 gridspec_kw=dict(wspace=0.46))
        # the ramp runs light to dark with the stage number, so the eye reads
        # the order of the ladder off the colour without a legend
        cols = [style["stage"][i % len(style["stage"])]
                for i in range(len(names))]
        for ax, v, lab in zip(axes[:2], (nlpd, rmse),
                              ("mean NLPD  [nats]", r"RMSE  [mm/s$^2$]")):
            ax.bar(xs, v, color=cols, width=0.66, zorder=3)
            ax.set_ylabel(lab)
            ax.margins(y=0.18)
        axes[0].annotate("lower is better", (0.03, 0.04), xycoords="axes fraction",
                         fontsize=style["fs"] - 2, color=style["muted"])

        # coverage is a proportion against a nominal value, so a zero-baseline
        # bar would waste the whole panel; a dot against the reference reads
        ax = axes[2]
        ax.axhline(0.9, color=style["ink"], lw=0.7, ls=(0, (3, 2)), zorder=2)
        ax.vlines(xs, 0.9, cov, color=style["faint"], lw=1.2, zorder=3)
        ax.scatter(xs, cov, s=26, color=cols, zorder=4, linewidths=0)
        ax.set_ylabel("90% coverage")
        ax.set_ylim(min(0.86, cov.min() - 0.01), max(0.98, cov.max() + 0.01))
        ax.annotate("nominal", (len(xs) - 0.6, 0.9), fontsize=style["fs"] - 2,
                    color=style["ink"], va="bottom", ha="right")

        for ax in axes:
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=35, ha="right")
            tidy(ax)
        return fig
    return (ladder_figure,)


@app.cell
def _(AVAILABLE, HAVE_BASE, LADDER, RES, STYLE, np, plt, tidy):
    def _day_nlpd(name, day_hours=24.0):
        """Mean NLPD per scenario day, from the stored two-hour blocks."""
        npz = RES[name][0]
        h, v = npz["win_hour"], npz["win_nlpd"]
        day = np.floor(h / day_hours).astype(int)
        days = np.unique(day)
        return days + 1, np.array([v[day == d].mean() for d in days])

    def gp_figure(style=STYLE):
        """What the Matern-1/2 branch is worth.

        Left: the branch on its own, the stage-4 block on its own, and the two
        of them behind a switch. The GP is much the worst of the three, which
        is the interesting part: the pair beats both of its members.
        Right: how much responsibility the switch actually gives the branch,
        which is the answer to whether it earned its place.
        """
        need = ("matern", "hgf", "moe", "bmoe")
        if not all(n in RES for n in need):
            return None
        labels = [RES[n][1]["label"] for n in need]
        vals = np.array([RES[n][1]["metrics"]["nlpd"] for n in need])
        cols = [style["muted"], style["stage"][3],
                style["stage"][4], style["stage"][5]]
        xs = np.arange(len(need))

        fig, axes = plt.subplots(1, 2, figsize=(style["figsize"][0], 2.5),
                                 gridspec_kw=dict(width_ratios=[1.0, 1.4],
                                                  wspace=0.30))
        ax = axes[0]
        ax.bar(xs, vals, color=cols, width=0.6, zorder=3)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                        xytext=(0, -11), ha="center",
                        fontsize=style["fs"] - 2, color=style["muted"])
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("mean NLPD  [nats]")
        ax.set_ylim(vals.min() * 1.20, 0)
        tidy(ax)

        ax = axes[1]
        npz = RES["bmoe"][0]
        resp = npz["diag_resp"]
        h = npz["thin_index"] / 600.0
        n = min(len(h), len(resp))
        h, r = h[:n], resp[:n, -1]
        # The raw sequence switches sample to sample, so on a slide it is a
        # solid block of ink. The block is drawn faintly because the fact that
        # it reaches both rails is worth seeing; the line on top is a rolling
        # mean over roughly an hour, which is what can actually be read.
        ax.plot(h, r, color=style["stage"][5], lw=0.4, alpha=0.16, zorder=2)
        k = max(3, n // 60)
        sm = np.convolve(r, np.ones(k) / k, mode="valid")
        ax.plot(h[k - 1:], sm, color=style["stage"][5], lw=style["lw"],
                zorder=3, label="hourly mean")
        share = float(r.mean())
        ax.set_xlabel("scenario hour")
        ax.set_ylabel("GP branch responsibility")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(frameon=False, loc="upper left", handlelength=1.4,
                  fontsize=style["fs"] - 1.5)
        ax.annotate(f"overall mean {share:.2f}", (0.98, 0.90),
                    xycoords="axes fraction", ha="right",
                    fontsize=style["fs"] - 1.5, color=style["muted"])
        tidy(ax, grid_axis="both")
        return fig

    def transformer_figure(style=STYLE):
        """The whole stack against an end-to-end deep model, on one slide.

        Left: mean NLPD over the scored window. The six stages keep the ramp
        they had throughout the talk, so the audience reads them as the same
        objects; the Transformer sits off the ramp because it is a different
        kind of thing, not a seventh rung.

        Right: the same metric day by day, over the scored window. Day one is
        the burn-in both sides were fitted on and is not scored, so what is
        plotted is days two to five, where the deck freezes and then cracks
        and a network fitted on day one has to extrapolate.
        """
        deep = [e for e in HAVE_BASE if e[0].startswith("transformer")]
        if not deep:
            return None
        entries = LADDER + deep
        labels = [l for _, _, l in entries]
        vals = np.array([RES[n][1]["metrics"]["nlpd"] for n, _, _ in entries])
        cols = ([style["stage"][i % len(style["stage"])]
                 for i in range(len(LADDER))] + [style["deep"]] * len(deep))
        xs = np.arange(len(entries))

        fig, axes = plt.subplots(1, 2, figsize=(style["figsize"][0], 2.9),
                                 gridspec_kw=dict(width_ratios=[1.5, 1.0],
                                                  wspace=0.34))
        ax = axes[0]
        ax.bar(xs, vals, color=cols, width=0.66, zorder=3)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                        xytext=(0, -11), ha="center",
                        fontsize=style["fs"] - 2, color=style["muted"])
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("mean NLPD  [nats]")
        ax.set_ylim(vals.min() * 1.20, 0)
        ax.annotate("lower is better", (0.02, 0.04), xycoords="axes fraction",
                    fontsize=style["fs"] - 2, color=style["muted"])
        tidy(ax)

        ax = axes[1]
        best = LADDER[-1][0]
        ax.plot(*_day_nlpd(best), color=style["stage"][len(LADDER) - 1],
                lw=style["lw"], marker="o", ms=3.5, zorder=4,
                label=RES[best][1]["label"])
        dashes = [(None, None), (3, 2), (1, 1.6)]
        for (n, _, l), dash in zip(deep, dashes):
            d, v = _day_nlpd(n)
            ax.plot(d, v, color=style["deep"], lw=style["lw"], marker="s",
                    ms=3.0, dashes=dash, zorder=3, label=l)
        ax.set_xlabel("scenario day")
        ax.set_ylabel("mean NLPD  [nats]")
        ax.set_xticks(sorted(set(_day_nlpd(best)[0])))
        ax.legend(frameon=False, loc="best", handlelength=1.8,
                  fontsize=style["fs"] - 1.5)
        tidy(ax)
        return fig
    return gp_figure, transformer_figure



@app.cell
def _(HERE, STYLE, np, os, plt, tidy):
    def bridge_figure(style=STYLE):
        """What the simulator does, in the two channels the models care about.

        Left: the true first natural frequency. It drifts with deck temperature,
        jumps when the deck freezes, and steps down when the structure cracks.
        That is the coefficient chain's job.

        Right: the excitation r.m.s., on a log axis because it spans three
        orders of magnitude between a quiet night and the evening peak. That is
        the volatility chain's job.

        Both are simulator truth. No model in this deck is shown either one.
        """
        import pandas as pd
        d = pd.read_csv(os.path.join(HERE, "data", "sim_001.csv"),
                        usecols=["hour", "f1", "exc_rms", "regime"])
        d = d.iloc[::20]
        h, f1, exc, reg = (d.hour.values, d.f1.values,
                           d.exc_rms.values, d.regime.values)

        # Short and wide on purpose: the slide caps the height, so a flatter
        # figure is rendered larger, not smaller.
        fig, axes = plt.subplots(1, 2, figsize=(style["figsize"][0], 1.55),
                                 gridspec_kw=dict(wspace=0.28))

        def shade(ax):
            for code, col, lab in ((1, style["stage"][0], "frozen"),
                                   (2, style["stage"][5], "damaged")):
                m = reg == code
                if not m.any():
                    continue
                cut = np.flatnonzero(np.diff(m.astype(int)) != 0) + 1
                for blk in np.split(np.flatnonzero(m), np.searchsorted(
                        np.flatnonzero(m), cut)):
                    if len(blk):
                        ax.axvspan(h[blk[0]], h[blk[-1]], color=col, alpha=0.13,
                                   lw=0, zorder=1,
                                   label=lab if blk[0] == np.flatnonzero(m)[0] else None)

        ax = axes[0]
        shade(ax)
        ax.plot(h, f1, color=style["ink"], lw=1.0, zorder=3)
        ax.set_ylabel(r"true $f_1$  [Hz]")
        ax.legend(frameon=False, loc="lower left", handlelength=1.0,
                  fontsize=style["fs"] - 1.5, ncols=2, columnspacing=1.0)
        tidy(ax, grid_axis="both")

        ax = axes[1]
        shade(ax)
        # Raw is a solid block of ink at this rate, so it goes in faint and the
        # readable line is an hourly rolling median over it.
        ax.plot(h, exc * 1e3, color=style["deep"], lw=0.4, alpha=0.20, zorder=2)
        k = max(3, len(h) // 120)
        med = pd.Series(exc * 1e3).rolling(k, center=True, min_periods=1).median()
        ax.plot(h, med.values, color=style["deep"], lw=style["lw"], zorder=3)
        ax.set_yscale("log")
        ax.set_ylabel(r"excitation  [mm/s$^2$]")
        tidy(ax, grid_axis="both")

        for ax in axes:
            ax.set_xlabel("scenario hour")
            ax.set_xlim(0, 120)
            ax.set_xticks([0, 24, 48, 72, 96, 120])
        return fig
    return (bridge_figure,)


@app.cell
def _(bridge_figure):
    bridge_figure()
    return



@app.cell
def _(RES, STYLE, np, plt, tidy):
    def trace_figure(name, hour_from=95.0, hour_to=98.0, style=STYLE):
        """A zoom on the forecast and its band, for a chosen stretch."""
        npz, meta = RES[name]
        h = npz["thin_index"] / 600.0
        sel = (h >= hour_from) & (h <= hour_to)
        fig, ax = plt.subplots(figsize=(style["figsize"][0], 1.9))
        ax.fill_between(h[sel],
                        (npz["thin_mean"][sel] - 2 * npz["thin_std"][sel]) * 1e3,
                        (npz["thin_mean"][sel] + 2 * npz["thin_std"][sel]) * 1e3,
                        color=style["now"], alpha=0.18, lw=0, zorder=2,
                        label=r"$\pm 2\sigma$ forecast")
        ax.plot(h[sel], npz["thin_y"][sel] * 1e3, color=style["ink"], lw=0.7,
                zorder=3, label="measured")
        ax.set_xlabel("scenario hour")
        ax.set_ylabel(r"acceleration  [mm/s$^2$]")
        ax.legend(frameon=False, loc="upper left", ncols=2, handlelength=1.4)
        tidy(ax, grid_axis="both")
        return fig

    def volatility_figure(name="hgf", style=STYLE):
        """What the volatility chain tracked, against the true excitation."""
        npz, meta = RES[name]
        if "diag_z" not in npz:
            return None
        h = npz["thin_index"] / 600.0
        z = npz["diag_z"]
        z = z[:, 0] if z.ndim == 2 else z
        fig, ax = plt.subplots(figsize=(style["figsize"][0], 1.9))
        ax.plot(h[:len(z)], z, color=style["now"], lw=style["lw"], zorder=3,
                label=r"inferred $z_t$  (log precision)")
        ax.set_xlabel("scenario hour")
        ax.set_ylabel(r"$z_t$")
        ax.legend(frameon=False, loc="best", handlelength=1.4)
        tidy(ax, grid_axis="both")
        return fig
    return trace_figure, volatility_figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Live: run a model on a stretch of the signal

    The cells below import the models themselves rather than reading the stored
    predictions, so you can point one at any window and look at the signal
    against its forecast at full rate. Hyperparameters come from
    `results/<stage>.json`, so a live run matches the stored one.

    The filter is warmed up for `warmup` hours before the window, not from the
    start of the trace, so this is a preview rather than the scored run. A
    couple of hours is plenty for everything except the drift precision in
    stages 3 and 4, which takes longer to settle.
    """)
    return


@app.cell
def _(HERE, RES, np, os):
    import sys
    if HERE not in sys.path:
        sys.path.insert(0, HERE)

    from models.base import Protocol, prepare, run_filter
    from models.ar.model import build as build_ar
    from models.tvar.model import build as build_tvar
    from models.htvar.model import build as build_htvar
    from models.hgf.model import build as build_hgf
    from models.moe.model import build as build_moe
    from models.bmoe.model import build as build_bmoe

    BUILDERS = dict(ar=build_ar, tvar=build_tvar, htvar=build_htvar,
                    hgf=build_hgf, moe=build_moe, bmoe=build_bmoe)

    PROTOCOL = Protocol()
    TRACE, XALL, YALL, IDXALL = prepare(PROTOCOL)
    PER_HOUR = len(TRACE.y) / (TRACE.hour[-1] - TRACE.hour[0])   # samples/hour

    def hyper_for(name):
        """The settings the stored run used, minus the search log."""
        if name not in RES:
            return {}
        h = dict(RES[name][1]["hyper"])
        h.pop("_search", None)
        h.pop("order", None)
        return h

    def run_window(name, hour_from, hour_to, warmup=3.0, **overrides):
        """Filter from `warmup` hours before the window to its end.

        Returns the full-rate forecast over the window only. The model still
        predicts every sample before it sees it; it simply has not been
        running since the start of the trace.
        """
        lo = max(0, int((hour_from - warmup) * PER_HOUR))
        hi = min(len(YALL), int(hour_to * PER_HOUR))
        cfg = dict(hyper_for(name), **overrides)
        out = run_filter(BUILDERS[name](order=PROTOCOL.order, **cfg),
                         XALL[lo:hi], YALL[lo:hi], IDXALL[lo:hi], record=True)
        h = TRACE.hour[out.index]
        keep = h >= hour_from
        return dict(
            hour=h[keep], y=YALL[lo:hi][keep],
            mean=out.pred_mean[keep], std=out.pred_std[keep],
            logpdf=out.logpdf[keep],
            exc=TRACE.truth["exc_rms"][out.index][keep] if "exc_rms" in TRACE.truth else None,
            regime=TRACE.truth["regime"][out.index][keep] if "regime" in TRACE.truth else None,
            rmse=float(np.sqrt(np.mean((YALL[lo:hi][keep] - out.pred_mean[keep]) ** 2))),
            nlpd=float(-np.mean(out.logpdf[keep])),
            cfg=cfg, label=RES[name][1]["label"] if name in RES else name,
        )
    return (BUILDERS, PER_HOUR, PROTOCOL, TRACE, XALL, YALL, IDXALL,
            hyper_for, run_window, sys)


@app.cell
def _(STYLE, np, plt, tidy):
    def signal_figure(res, style=STYLE, band=2.0, show_excitation=True):
        """Top: the signal and the forecast band. Bottom: how wide the model
        thinks the band should be, against the simulator's excitation."""
        rows = 2 if (show_excitation and res["exc"] is not None) else 1
        fig, axes = plt.subplots(rows, 1, sharex=True,
                                 figsize=(style["figsize"][0], 2.0 + 1.5 * rows),
                                 gridspec_kw=dict(hspace=0.16,
                                                  height_ratios=[2, 1][:rows]))
        axes = np.atleast_1d(axes)

        ax = axes[0]
        ax.fill_between(res["hour"], (res["mean"] - band * res["std"]) * 1e3,
                        (res["mean"] + band * res["std"]) * 1e3,
                        color=style["now"], alpha=0.20, lw=0, zorder=2,
                        label=rf"$\pm{band:g}\sigma$ forecast")
        ax.plot(res["hour"], res["y"] * 1e3, color=style["ink"], lw=0.6,
                zorder=3, label="measured")
        ax.plot(res["hour"], res["mean"] * 1e3, color=style["now"], lw=0.8,
                zorder=4, label="one-step mean")
        ax.set_ylabel(r"acceleration  [mm/s$^2$]")
        ax.margins(y=0.30)              # headroom, so the legend clears the data
        ax.legend(frameon=False, ncols=3, loc="upper left", handlelength=1.4)
        ax.set_title(f"{res['label']}   ---   RMSE {res['rmse']*1e3:.1f} mm/s$^2$, "
                     f"NLPD {res['nlpd']:+.2f} nats on this window",
                     loc="left", color=style["muted"])
        tidy(ax, grid_axis="both")

        if rows == 2:
            ax = axes[1]
            ax.plot(res["hour"], res["std"] * 1e3, color=style["now"],
                    lw=style["lw"], zorder=3, label="predictive s.d.")
            ax.plot(res["hour"], res["exc"] * 1e3, color=style["deep"],
                    lw=1.0, zorder=2, label="true excitation r.m.s.")
            ax.set_yscale("log")
            _top = float(np.nanmax(np.concatenate([res["std"], res["exc"]]))) * 1e3
            ax.set_ylim(top=_top * 5.0)
            ax.set_ylabel(r"[mm/s$^2$]")
            ax.legend(frameon=False, ncols=2, loc="upper left", handlelength=1.4)
            tidy(ax, grid_axis="both")

        axes[-1].set_xlabel("scenario hour")
        return fig

    def spread_figure(results, truth_hour=None, truth=None, style=STYLE):
        """Every model's predictive s.d. on one window.

        The stages keep the ramp they have everywhere else in the deck; the
        simulator's excitation is drawn in the contrast hue.
        """
        fig, ax = plt.subplots(figsize=(style["figsize"][0], 2.6))
        shades = [style["stage"][i % len(style["stage"])]
                  for i in range(len(results))]
        stack = []
        if truth is not None:
            ax.plot(truth_hour, truth * 1e3, color=style["deep"], lw=1.4,
                    zorder=2, label="true excitation r.m.s.")
            stack.append(truth)
        for c, res in zip(shades, results):
            ax.plot(res["hour"], res["std"] * 1e3, color=c, lw=1.3, zorder=3,
                    label=res["label"])
            stack.append(res["std"])
        ax.set_yscale("log")
        ax.set_ylim(top=float(np.nanmax(np.concatenate(stack))) * 1e3 * 12.0)
        ax.set_xlabel("scenario hour")
        ax.set_ylabel(r"predictive s.d.  [mm/s$^2$]")
        ax.legend(frameon=False, ncols=4, loc="upper left", handlelength=1.4,
                  fontsize=style["fs"] - 1.5, columnspacing=1.2)
        tidy(ax, grid_axis="both")
        return fig
    return signal_figure, spread_figure


@app.cell(hide_code=True)
def _(AVAILABLE, mo):
    live_stage = mo.ui.dropdown(
        options={f"{s}. {l}": n for n, s, l in AVAILABLE},
        value=f"{AVAILABLE[-1][1]}. {AVAILABLE[-1][2]}", label="model")
    live_from = mo.ui.number(start=0.0, stop=119.0, step=0.5, value=95.0,
                             label="from hour")
    live_span = mo.ui.number(start=0.1, stop=12.0, step=0.1, value=2.0,
                             label="span [h]")
    live_warm = mo.ui.number(start=0.2, stop=24.0, step=0.2, value=3.0,
                             label="warm-up [h]")
    live_all = mo.ui.checkbox(value=False, label="compare all six")
    mo.hstack([live_stage, live_from, live_span, live_warm, live_all],
              justify="start", gap=1.0)
    return live_all, live_from, live_span, live_stage, live_warm


@app.cell
def _(live_from, live_span, live_stage, live_warm, run_window, signal_figure):
    live_res = run_window(live_stage.value, live_from.value,
                          live_from.value + live_span.value,
                          warmup=live_warm.value)
    signal_figure(live_res)
    return (live_res,)


@app.cell
def _(AVAILABLE, live_all, live_from, live_res, live_span, live_warm,
      run_window, spread_figure):
    # Guarded by the checkbox: six filters over the same window is a few
    # seconds, and you do not want that on every slider nudge.
    if live_all.value:
        _runs = [run_window(n, live_from.value,
                            live_from.value + live_span.value,
                            warmup=live_warm.value)
                 for n, _, _ in AVAILABLE]
        _fig = spread_figure(_runs, live_res["hour"], live_res["exc"])
    else:
        _fig = None
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The decision the forecast is for

    Slide two asks whether it is still safe to drive over. That is a decision,
    not a forecast, so it needs a threshold and a margin rather than a point
    estimate. `models/decision.py` runs each model over the whole trace at full
    rate and summarises one chosen hour, writing `results/decision.json`. The
    thinned arrays the panels above use are far too coarse for the r.m.s.\ of a
    single bursty hour, which is why it is a separate pass.

    The slide figure is drawn by `../plot_decision.py`, a plain script in the
    deck folder, so it is not repeated here. Run it after re-running
    `models.decision`.
    """)
    return



@app.cell(hide_code=True)
def _(AVAILABLE, mo):
    stage_pick = mo.ui.dropdown(
        options={f"{s}. {l}": n for n, s, l in AVAILABLE},
        value=f"{AVAILABLE[-1][1]}. {AVAILABLE[-1][2]}",
        label="stage",
    )
    stage_pick
    return (stage_pick,)


@app.cell
def _(stage_figure, stage_pick):
    stage_figure(stage_pick.value)
    return


@app.cell
def _(ladder_figure):
    ladder_figure()
    return


@app.cell
def _(transformer_figure):
    transformer_figure()
    return


@app.cell
def _(gp_figure):
    gp_figure()
    return


@app.cell
def _(stage_pick, trace_figure):
    # The damage step lands at scenario hour 96.1; the deck freezes earlier.
    trace_figure(stage_pick.value, 95.0, 98.0)
    return


@app.cell(hide_code=True)
def _(RES, AVAILABLE, mo):
    _rows = "\n".join(
        f"| {s} | {l} | {RES[n][1]['metrics']['rmse'] * 1e3:.2f} | "
        f"{RES[n][1]['metrics']['nlpd']:+.3f} | "
        f"{RES[n][1]['metrics']['coverage90']:.3f} |"
        for n, s, l in AVAILABLE)
    mo.md(
        "| stage | model | RMSE [mm/s²] | NLPD [nats] | 90% coverage |\n"
        "|---|---|---|---|---|\n" + _rows
    )
    return


@app.cell
def _(AVAILABLE, FIGURES, bridge_figure, gp_figure, ladder_figure, os, plt,
      stage_figure, transformer_figure, volatility_figure):
    def export(figdir=FIGURES):
        """Write the slide PDFs. Also runs when this file is executed directly."""
        os.makedirs(figdir, exist_ok=True)
        written = []
        for _n, _s, _l in AVAILABLE:
            _f = stage_figure(_n)
            _p = os.path.join(figdir, f"res_{_s}_{_n}.pdf")
            _f.savefig(_p)
            plt.close(_f)
            written.append(_p)
        _f = ladder_figure()
        _p = os.path.join(figdir, "res_ladder.pdf")
        _f.savefig(_p)
        plt.close(_f)
        written.append(_p)
        for _name, _fn in (("res_transformer", transformer_figure),
                           ("res_gp", gp_figure),
                           ("res_bridge", bridge_figure)):
            _f = _fn()
            if _f is not None:
                _p = os.path.join(figdir, f"{_name}.pdf")
                _f.savefig(_p)
                plt.close(_f)
                written.append(_p)
        _f = volatility_figure("hgf")
        if _f is not None:
            _p = os.path.join(figdir, "res_volatility.pdf")
            _f.savefig(_p)
            plt.close(_f)
            written.append(_p)
        return written

    exported = export()
    return export, exported


@app.cell(hide_code=True)
def _(exported, mo, os):
    mo.md("Wrote:\n\n" + "\n".join(f"- `{os.path.basename(p)}`" for p in exported))
    return


if __name__ == "__main__":
    app.run()
