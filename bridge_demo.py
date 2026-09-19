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


if __name__ == "__main__":
    app.run()
