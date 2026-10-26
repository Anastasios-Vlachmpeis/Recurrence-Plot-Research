"""Rotatable 3D recurrence onset surfaces for consecutive windows of an intraday price series.

Same construction as the 3D cells in recplay.ipynb: pair (i, j) is recurrent for every
ε >= D(i, j), so z = D(i, j) is the onset surface and the z-axis stops at ε_max, where RR = 100 %.
The series is cut into windows of --chunk-len samples, starting every --step samples; the slider
(or the left/right arrow keys) moves through them in time order. The panel below shows the price
with the current window shaded. The embedding is the (m, τ), m >= 2, that keeps the most vectors
per window.

Run:  python recplay_3d_viewer.py [--ticker CSG.AS] [--start 2026-09-01 --end 2026-09-02]
                                  [--chunk-len 20] [--step 1] [--eps 0.3] [--fixed-z]
Without --start/--end the most recent trading day is used (Yahoo keeps 2-min data ~60 days).
--fixed-z uses one z scale (the largest ε_max over all windows) so heights compare across time;
by default each window's z-axis ends at its own ε_max.
Drag with the left mouse button to rotate, right button to zoom; the view is kept across windows.
"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from matplotlib.widgets import Slider

HERE = Path(__file__).resolve().parent
yf.set_tz_cache_location(str(HERE / "../../data/raw/yahoo/py-yfinance-cache"))


def best_embedding(n):
    """(m, τ) with m >= 2 and at least 2 vectors that maximises N = n - (m - 1)τ; larger m breaks ties."""
    candidates = [(m, t) for m in range(2, n) for t in range(1, n) if n - (m - 1) * t >= 2]
    return max(candidates, key=lambda c: (n - (c[0] - 1) * c[1], c[0]))


def window_distances(series, chunk_len, step, m, t):
    """Distance matrices of every window: shape (windows, N, N), N = chunk_len - (m - 1)t."""
    chunks = np.lib.stride_tricks.sliding_window_view(series, chunk_len)[::step]       # (W, L)
    vecs = np.lib.stride_tricks.sliding_window_view(chunks, (m - 1) * t + 1, axis=1)[:, :, ::t]  # (W, N, m)
    return np.sqrt(((vecs[:, :, None, :] - vecs[:, None, :, :]) ** 2).sum(axis=-1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticker", default="CSG.AS")
    p.add_argument("--start", help="YYYY-MM-DD; needs --end")
    p.add_argument("--end", help="YYYY-MM-DD, exclusive")
    p.add_argument("--chunk-len", type=int, default=20)
    p.add_argument("--step", type=int, default=1, help="samples between window starts (chunk-len = back to back)")
    p.add_argument("--eps", type=float, default=0.3, help="threshold drawn as the red plane")
    p.add_argument("--fixed-z", action="store_true", help="same z scale for every window")
    args = p.parse_args()

    if args.start:
        df = yf.download(args.ticker, start=args.start, end=args.end, interval="2m", prepost=True, progress=False)
    else:
        df = yf.download(args.ticker, period="1d", interval="2m", prepost=True, progress=False)
    if df.empty:
        raise SystemExit(f"No 2-min data for {args.ticker} in that range (Yahoo keeps ~60 days).")
    series = df["Close"].squeeze("columns").to_numpy()
    times = df.index
    L, step = args.chunk_len, args.step
    if len(series) < L:
        raise SystemExit(f"Series has {len(series)} points, fewer than --chunk-len {L}.")

    m, t = best_embedding(L)

    t0 = time.perf_counter()
    D = window_distances(series, L, step, m, t)
    W, N = D.shape[0], D.shape[1]
    D_sorted = np.sort(D.reshape(W, -1), axis=1)
    eps_max = D_sorted[:, -1]  # per window: smallest ε with RR = 100 %
    z_global = eps_max.max()
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    t_data = time.perf_counter()
    print(f"embedding: m = {m}, tau = {t} -> {N} of {L} points as vectors")
    print(f"{W} windows of {L} samples, step {step}, from {len(series)} samples")
    print(f"data   (embed, D, RR, all windows): {t_data - t0:8.4f} s", flush=True)
    for keymap in ("keymap.back", "keymap.forward"):  # arrows step the slider instead of view history
        plt.rcParams[keymap] = [k for k in plt.rcParams[keymap] if k not in ("left", "right")]

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_axes([0.0, 0.27, 0.88, 0.68], projection="3d")
    cax = fig.add_axes([0.88, 0.40, 0.02, 0.45])
    ax_price = fig.add_axes([0.08, 0.10, 0.84, 0.13])
    ax_slider = fig.add_axes([0.15, 0.03, 0.70, 0.025])

    ax_price.plot(np.arange(len(series)), series, color="0.3", lw=1)
    ax_price.set_xlim(0, len(series) - 1)
    ax_price.set_xlabel("sample")
    ax_price.set_ylabel("Close")
    span = Rectangle((0, 0), L - 1, 1, transform=blended_transform_factory(ax_price.transData, ax_price.transAxes),
                     color="tab:orange", alpha=0.3)
    ax_price.add_patch(span)

    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.set_zlabel("ε | RR", labelpad=18)
    ax.view_init(elev=25, azim=-60)
    note = ax.text2D(0.02, 0.97, "", color="tab:red", transform=ax.transAxes)
    drawn = []  # artists of the current window, replaced on every slider move
    cbar = None

    def build(k):
        nonlocal cbar
        for a in drawn:
            a.remove()
        drawn.clear()

        s = k * step
        Dk, top = D[k], eps_max[k]

        def rr(e):
            return 100 * np.searchsorted(D_sorted[k], e, side="right") / N**2

        z_top = z_global if args.fixed_z else top
        if z_top == 0:  # flat window: every vector identical
            z_top = 1.0
        surf = ax.plot_surface(I, J, Dk, cmap="gist_earth", vmin=0, vmax=z_top,
                               rstride=1, cstride=1, linewidth=0, antialiased=False)
        drawn.append(surf)
        if args.eps < top:
            # Slice at ε: its footprint below the plane is the thresholded RP
            drawn.append(ax.plot_surface(I[[0, -1]][:, [0, -1]], J[[0, -1]][:, [0, -1]], np.full((2, 2), args.eps),
                                         color="tab:red", alpha=0.25))
            note.set_text(f"red plane: ε = {args.eps}  (RR = {rr(args.eps):.1f} %)")
        else:
            note.set_text(f"ε = {args.eps} ≥ ε_max: RR = 100 %, plane above the plot")

        ax.set_zlim(0, z_top)
        z_ticks = np.linspace(0, z_top, 6)
        ax.set_zticks(z_ticks)
        ax.set_zticklabels([f"{z:.2f} | {rr(z):.0f}%" for z in z_ticks])
        flat = " (flat window)" if top == 0 else ""
        ax.set_title(f"{args.ticker} window {k + 1}/{W}: samples [{s}:{s + L}], "
                     f"{times[s]:%H:%M}–{times[s + L - 1]:%H:%M}{flat}\n"
                     f"m = {m}, τ = {t}, N = {N}; RR = 100 % at ε = {top:.3f}")

        if cbar is None:
            cbar = fig.colorbar(surf, cax=cax, label="onset ε")
        else:
            cbar.update_normal(surf)
        span.set_x(s)

    slider = Slider(ax_slider, "window", 0, W - 1, valinit=0, valstep=1, valfmt="%d")

    def on_change(val):
        k = int(val)
        t1 = time.perf_counter()
        build(k)
        t2 = time.perf_counter()
        fig.canvas.draw()
        t3 = time.perf_counter()
        print(f"window {k:4d}: build {t2 - t1:7.4f} s, render {t3 - t2:7.4f} s", flush=True)

    def on_key(event):
        if event.key in ("left", "right"):
            k = int(slider.val) + (1 if event.key == "right" else -1)
            slider.set_val(min(max(k, 0), W - 1))

    t1 = time.perf_counter()
    build(0)
    t2 = time.perf_counter()
    fig.canvas.draw()  # first full render, timed here because plt.show() blocks until the window closes
    t3 = time.perf_counter()
    print(f"build  (first window)             : {t2 - t1:8.4f} s")
    print(f"render (first window)             : {t3 - t2:8.4f} s")
    print(f"total                             : {t3 - t0:8.4f} s", flush=True)

    slider.on_changed(on_change)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
