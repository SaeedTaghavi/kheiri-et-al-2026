"""
Motivation
----------
Draw Fig. 4 of Kheiri et al. (2026) from the output of ``peak_tau.py``.

What it measures
----------------
Nothing new. (A) peak frequency and (B) peak power in dB of the excitatory
population, (C) and (D) the same for the inhibitory population, in the plane
of the I -> I decay time (x) and the I -> E decay time (y). Grid points
where no peak was found (frequency 0) are left white in (A), (B) and (D).

Usage
-----
    python peak_tau_plot.py --indir results/fig4
"""
import argparse
import os

import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def add_bold_label(ax, label):
    ax.text(-0.1, 1.25, rf"$\mathbf{{{label}}}$", transform=ax.transAxes,
            fontsize=28, va="top", ha="right")


def frequency_colorbar(fig, im, ax):
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks(np.arange(0, 200, 50))
    cbar.ax.yaxis.set_minor_locator(MultipleLocator(10))
    cbar.ax.tick_params(labelsize=25, width=1.5, length=10, direction="out", pad=3)
    cbar.ax.tick_params(which="minor", width=1.0, length=6, color="k")
    cbar.ax.yaxis.set_major_formatter("{x:.0f}")
    cbar.ax.set_title("F(Hz)", fontsize=25, pad=20)


def power_colorbar(fig, im, ax, minor=None):
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_title("PSD(dB)", fontsize=25, pad=20)
    cbar.ax.tick_params(labelsize=25, width=1.5, length=10, direction="out", pad=3)
    if minor:
        cbar.ax.yaxis.set_minor_locator(MultipleLocator(minor))
        cbar.ax.yaxis.set_major_formatter("{x:.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--indir", default="results/fig4")
    p.add_argument("--out", default=None, help="default: <indir>/peak3D.png")
    args = p.parse_args()

    plt.rcdefaults()
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                         "mathtext.fontset": "custom", "mathtext.bf": "DejaVu Serif:bold"})
    load = lambda name: np.loadtxt(os.path.join(args.indir, name))
    tau = load("tau.txt")
    freq_exc, freq_inh = load("freq_exc.txt"), load("freq_inh.txt")
    power_exc, power_inh = load("power_exc.txt"), load("power_inh.txt")

    freq_exc_m = ma.masked_where(freq_exc == 0, freq_exc)
    power_exc_db = 10 * np.ma.log10(ma.masked_where(freq_exc == 0, power_exc))
    power_inh_db = 10 * np.ma.log10(ma.masked_where(freq_inh == 0, power_inh))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10.0))
    for ax, label in zip(axes.flat, "ABCD"):
        add_bold_label(ax, label)
    x, y = np.meshgrid(tau, tau)   # x: I -> I, y: I -> E
    cmap = plt.cm.jet.copy()
    cmap.set_under("white")

    im = axes[0, 0].pcolormesh(x, y, freq_exc_m, cmap=cmap, vmin=0, vmax=171, shading="auto")
    frequency_colorbar(fig, im, axes[0, 0])
    im = axes[0, 1].pcolormesh(x, y, power_exc_db, cmap=cmap, shading="auto")
    power_colorbar(fig, im, axes[0, 1], minor=20)
    im = axes[1, 0].pcolormesh(x, y, freq_inh, cmap=cmap, vmin=0, vmax=171, shading="auto")
    frequency_colorbar(fig, im, axes[1, 0])
    im = axes[1, 1].pcolormesh(x, y, power_inh_db, cmap=cmap, shading="auto")
    power_colorbar(fig, im, axes[1, 1])

    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\tau_{inh \to exc}$ (ms)", size=30, labelpad=20)
    for ax in axes[1, :]:
        ax.set_xlabel(r"$\tau_{inh \to inh}$ (ms)", size=30, labelpad=10)
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    for ax in axes.flat:
        ax.tick_params(labelsize=25, axis="both", width=1.5, length=12, pad=8.0)
        ax.set_xlim(1, 10)
        ax.set_ylim(1, 10)
        ax.set_xticks(np.arange(1, 11, 2))
        ax.set_xticks(np.arange(2, 12, 2), minor=True)
        ax.set_yticks(np.arange(1, 11, 2))
        ax.set_yticks(np.arange(2, 12, 2), minor=True)
        ax.tick_params(which="major", length=10, width=1.5, color="k")
        ax.tick_params(which="minor", length=5, width=1.0, color="k")

    out = args.out or os.path.join(args.indir, "peak3D.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
