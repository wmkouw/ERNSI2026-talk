import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        # ERNSI 2026 --- forecasting results

        One-step-ahead forecasts from the six models in `models/`, on the bridge
        trace in `data/`. Every model predicted each sample before it saw it, on
        the same window, with hyperparameters chosen on day one only.

        The plotting code below is meant to be edited. Change `STYLE`, change a
        `*_figure` function, and every panel re-renders. Running this file as a
        script (`python results_notebook.py`) writes the slide PDFs into
        `../figures/`.
        """
    )
    return (mo,)


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
    return FIGURES, HERE, RESULTS, STAGES, json, np, os, plt


@app.cell
def _(RESULTS, STAGES, json, np, os):
    def load(name):
        npz = dict(np.load(os.path.join(RESULTS, f"{name}.npz")))
        with open(os.path.join(RESULTS, f"{name}.json")) as fh:
            meta = json.load(fh)
        return npz, meta

    RES = {}
    for _n, _s, _l in STAGES:
        if os.path.exists(os.path.join(RESULTS, f"{_n}.npz")):
            RES[_n] = load(_n)

    AVAILABLE = [(n, s, l) for n, s, l in STAGES if n in RES]
    PREV = {n: (AVAILABLE[i - 1][0] if i else None)
            for i, (n, _, _) in enumerate(AVAILABLE)}
    return AVAILABLE, PREV, RES, load


@app.cell
def _():
    # ---------------------------------------------------------------- style
    # Two hues only: the stage on the slide, and the stage before it. Both
    # pass the colourblind-separation and chroma checks against a light
    # surface, so the pair is safe without relying on position alone.
    STYLE = dict(
        now="#2563EB",        # this stage
        before="#B45309",     # the previous stage
        ink="#111827",
        muted="#6B7280",
        faint="#D1D5DB",
        grid="#E5E7EB",
        lw=1.7,
        fs=8.5,               # base font size, tuned for a 16:9 beamer frame
        figsize=(7.2, 2.5),   # inches; \gfx scales it to the frame
        dpi=200,
    )

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
    return STYLE, apply_style


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
def _(PREV, RES, STAGES, STYLE, np, plt, tidy):
    def running_panel(ax, name, prev, style=STYLE):
        """Running RMSE over the scored window: this stage against the last."""
        npz, meta = RES[name]
        hour = npz["step"] / 600.0          # 600 samples per scenario hour
        if prev is not None:
            p_npz, p_meta = RES[prev]
            ax.plot(p_npz["step"] / 600.0, p_npz["run_rmse"] * 1e3,
                    color=style["before"], lw=style["lw"], zorder=2,
                    label=p_meta["label"])
        ax.plot(hour, npz["run_rmse"] * 1e3, color=style["now"],
                lw=style["lw"], zorder=3, label=meta["label"])
        ax.set_xlabel("scenario hour")
        ax.set_ylabel(r"running RMSE  [mm/s$^2$]")
        ax.legend(frameon=False, loc="best", handlelength=1.4, borderpad=0.2)
        tidy(ax)
        return ax

    def nlpd_panel(ax, name, upto=True, style=STYLE, full_axis=True):
        """Mean negative log predictive density, one bar per stage.

        The axis always spans all six slots, whichever stage is on the slide,
        so the bars fill in from left to right as the talk goes and nothing
        moves underneath the audience.
        """
        allst = [(n, s, l) for n, s, l in STAGES if n in RES]
        here = [s for n, s, l in allst if n == name][0]
        shown = [(n, s, l) for n, s, l in allst if s <= here] if upto else allst
        vals = [RES[n][1]["metrics"]["nlpd"] for n, _, _ in shown]
        cols = [style["now"] if n == name else style["faint"] for n, _, _ in shown]
        xs = np.arange(len(shown))
        ax.bar(xs, vals, color=cols, width=0.62, zorder=3)
        for x, v, (n, _, _) in zip(xs, vals, shown):
            ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                        xytext=(0, -11 if v < 0 else 3), ha="center",
                        fontsize=style["fs"] - 1.5,
                        color=style["now"] if n == name else style["muted"])
        span = allst if full_axis else shown
        ax.set_xticks(np.arange(len(span)))
        ax.set_xticklabels([l for _, _, l in span], rotation=20, ha="right")
        ax.set_xlim(-0.75, len(span) - 0.25)
        lo = min(RES[n][1]["metrics"]["nlpd"] for n, _, _ in allst)
        ax.set_ylim(lo * 1.18, 0)
        ax.set_ylabel("mean NLPD  [nats]")
        ax.annotate("lower is better", (0.02, 0.04), xycoords="axes fraction",
                    fontsize=style["fs"] - 2, color=style["muted"])
        tidy(ax)
        return ax

    def stage_figure(name, style=STYLE):
        """The figure that follows each factor-graph slide."""
        fig, axes = plt.subplots(1, 2, figsize=style["figsize"],
                                 gridspec_kw=dict(width_ratios=[1.35, 1.0],
                                                  wspace=0.32))
        running_panel(axes[0], name, PREV[name], style)
        nlpd_panel(axes[1], name, upto=True, style=style)
        return fig
    return nlpd_panel, running_panel, stage_figure


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
        # the last stage is highlighted in all three panels, so the eye
        # follows one model across the metrics rather than three winners
        last = len(names) - 1
        cols = [style["now"] if i == last else style["faint"] for i in range(len(names))]
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
def _(AVAILABLE, FIGURES, ladder_figure, os, plt, stage_figure, volatility_figure):
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
