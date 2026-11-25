"""
Motivation
----------
Show what the network does at one point of the decay-time sweeps: the
rastergram, the population rates and their spectra. These are the panels
Fig. 1(c-e) (both inhibitory decay times equal: 2.0, 3.5 and 8.0 ms) and
Fig. 2(d-f) and Fig. 3(d-f) (one inhibitory decay time fixed) of
Kheiri et al. (2026).

What it measures
----------------
One 10 s run with one thread and the fixed seeds. Left: spikes of all
neurons (blue excitatory, red inhibitory) and the smoothed population rates
between 1000 and 1100 ms. Right: the rate PSD of the inhibitory (top) and
excitatory (bottom) population, 0 to 200 Hz.

Usage
-----
    python raster.py --tau-inh-exc 2.0 --tau-inh-inh 2.0     # Fig. 1(c)
    python raster.py --tau-inh-exc 3.5 --tau-inh-inh 3.5     # Fig. 1(d)
    python raster.py --tau-inh-exc 8.0 --tau-inh-inh 8.0     # Fig. 1(e)
    python raster.py --tau-inh-exc 4.0 --tau-inh-inh 1.5     # a Fig. 2 panel
"""
import argparse
import os

import numpy as np
import nest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator

from network import T_START, set_kernel, build_network, split_spikes
from analysis import population_rate


def style_ticks(ax, major_x, minor_x, major_len=14):
    ax.xaxis.set_major_locator(MultipleLocator(major_x))
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.xaxis.set_minor_locator(MultipleLocator(minor_x))
    ax.tick_params(which="major", length=major_len, width=1.4, color="k")
    ax.tick_params(which="minor", length=8, width=1.0, color="k")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def psd_axis(ax, freq, psd, color, fmax=200):
    ax.plot(freq, psd, lw=2.2, color=color)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(30)
    ax.set_xlim(0, fmax + 5)
    ax.set_ylim(0,)
    ax.set_ylabel("PSD", fontsize=50, labelpad=6)
    ax.xaxis.set_tick_params(labelsize=30, pad=6)
    ax.yaxis.set_tick_params(labelsize=30, pad=6)
    style_ticks(ax, 100, 25)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tau-inh-exc", type=float, default=2.0, help="I -> E decay time (ms)")
    p.add_argument("--tau-inh-inh", type=float, default=2.0, help="I -> I decay time (ms)")
    p.add_argument("--order", type=int, default=300)
    p.add_argument("--probability", type=float, default=0.4)
    p.add_argument("--tmax", type=float, default=10000.0)
    p.add_argument("--window", nargs=2, type=float, default=[1000.0, 1100.0],
                   help="time window shown, ms")
    p.add_argument("--outdir", default="results/raster")
    args = p.parse_args()

    plt.rcdefaults()
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Liberation Sans", "DejaVu Sans"]})
    set_kernel(threads=1)
    net = build_network(args.order, args.probability,
                        tau_inh_exc=args.tau_inh_exc, tau_inh_inh=args.tau_inh_inh)
    nest.Simulate(args.tmax)
    t_e, s_e, t_i, s_i = split_spikes(net)
    start, tmax = T_START, args.tmax

    bins = np.arange(start, tmax + 1.0, 1.0)
    rate_e = population_rate(t_e, net["n_exc"], start, tmax)
    rate_i = population_rate(t_i, net["n_inh"], start, tmax)
    # The original panels show 2 |FFT|^2 dt / n, the same scale as fft_psd.
    n = len(rate_e)
    dt0 = 1e-3
    keep = np.arange(1, np.floor(n / 2), dtype="int")
    freq = np.fft.fftfreq(n, dt0)[keep]
    psd_e = 2 * (np.abs(np.fft.fft(rate_e)) ** 2 * (dt0 / n))[keep]
    psd_i = 2 * (np.abs(np.fft.fft(rate_i)) ** 2 * (dt0 / n))[keep]

    fig = plt.figure(figsize=(13, 6.5))
    gs = gridspec.GridSpec(6, 4, figure=fig, height_ratios=[2.2, 0.2, 1, 2.2, 1.2, 2.5],
                           width_ratios=[30, 5, 0.8, 12], wspace=0.5, hspace=0.5)
    ax_raster = fig.add_subplot(gs[0:4, 0:2])
    ax_psd_i = fig.add_subplot(gs[0:2, 3])
    ax_psd_e = fig.add_subplot(gs[3:5, 3])
    ax_rate = fig.add_subplot(gs[5, 0:2])
    began, end = args.window

    ax_raster.plot(t_e, s_e, "|", color="b", markersize=4.0, markeredgewidth=1.5)
    ax_raster.plot(t_i, s_i, "|", color="r", markersize=4.0, markeredgewidth=1.5)
    ax_raster.set_ylabel("Neuron", size=50, labelpad=12)
    ax_raster.tick_params(labelsize=35, pad=6.0)
    ax_raster.yaxis.set_major_locator(MultipleLocator(100))
    ax_raster.yaxis.set_major_formatter("{x:.0f}")
    ax_raster.yaxis.set_minor_locator(MultipleLocator(50))
    style_ticks(ax_raster, 50, 10)
    ax_raster.set_xlim(began - 1, end + 1)
    ax_raster.set_xticklabels([])

    ax_rate.fill_between(bins[:-1], rate_i, color="red")
    ax_rate.fill_between(bins[:-1], rate_e, color="blue")
    ax_rate.set_xlabel("t(ms)", fontsize=50, labelpad=12)
    ax_rate.set_ylabel("Rate(Hz)", size=50, labelpad=12)
    ax_rate.tick_params(labelsize=35, width=2, length=10, pad=6.0)
    style_ticks(ax_rate, 50, 10, major_len=12)
    ax_rate.set_xlim(began - 1, end + 1)

    psd_axis(ax_psd_i, freq, psd_i, "r")
    ax_psd_i.set_xticklabels([])
    psd_axis(ax_psd_e, freq, psd_e, "b")
    ax_psd_e.set_xlabel("F(Hz)", fontsize=50, labelpad=6)

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir,
                       f"raster_ie{args.tau_inh_exc}_ii{args.tau_inh_inh}.png")
    fig.savefig(out, dpi=350, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
