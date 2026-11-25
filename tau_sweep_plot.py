"""
Motivation
----------
Draw the PSD maps of Figs. 1, 2 and 3 of Kheiri et al. (2026) from the
output of ``tau_sweep.py``. The Fig. 1 layout is the one in the paper. The
original plotting script for Figs. 2 and 3 was not found, so for those
sweeps this draws the same quantities (PSD map and pairwise correlation) in
the Fig. 1 layout.

What it measures
----------------
Nothing new. Top: the PSD in dB (Gaussian-smoothed, sigma = 1 bin in both
directions) against the swept decay time and frequency. Bottom: the pairwise
correlation (green, left axis) and, if it was computed, the SPIKE-distance
D_Sa (black, right axis).

Usage
-----
    python tau_sweep_plot.py --indir results/fig1_n300
    python tau_sweep_plot.py --indir results/fig1_n1000 --labels F G \
        --sd-ylim-exc 0.278 0.301
    python tau_sweep_plot.py --indir results/fig2_ie4
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

XLABEL = {"both": r"$\tau_{inh}$ (ms)",
          "inh_inh": r"$\tau_{inh \to inh}$ (ms)",
          "inh_exc": r"$\tau_{inh \to exc}$ (ms)"}


def panel(tau, frequency, psd, pc, sd, label, xlabel, pc_ylim, sd_ylim, legend_loc, path):
    fig = plt.figure(figsize=(6.0, 6.0), dpi=120)
    gs = GridSpec(2, 2, height_ratios=[4, 1.6], width_ratios=[1, 0.05], hspace=0.05)
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_pc = fig.add_subplot(gs[1, 0])

    psd_db = 10 * np.log10(gaussian_filter(psd, sigma=1.0) + 1e-10)
    im = ax_spec.pcolormesh(tau, frequency, psd_db.T, shading="nearest", cmap="jet")
    if label:
        ax_spec.set_title(label, size=25, loc="left", family="serif", weight="bold", x=-0.3, pad=10)
    ax_spec.set_ylim(0, 200)
    ax_spec.set_ylabel("Freq (Hz)", size=25)
    ax_spec.tick_params(labelsize=20, which="major", length=10)
    ax_spec.set_xticks([])
    cbar = fig.colorbar(im, cax=fig.add_subplot(gs[0, 1]))
    cbar.set_label("dB", size=20)
    cbar.ax.tick_params(labelsize=15)

    lines = ax_pc.plot(tau, pc, lw=3.0, color="green", label="Pairwise Corr")
    ax_pc.set_ylabel("PC", color="green", size=25)
    ax_pc.tick_params(axis="y", labelcolor="green", labelsize=20)
    if pc_ylim:
        ax_pc.set_ylim(*pc_ylim)
    ax_pc.set_xlabel(xlabel, size=20)
    if sd is not None:
        ax_sd = ax_pc.twinx()
        lines += ax_sd.plot(tau, sd, lw=3.0, color="k", label="Spike Dist.")
        ax_sd.set_ylabel("Spike Dist.", color="k", size=20)
        if sd_ylim:
            ax_sd.set_ylim(*sd_ylim)
        ax_sd.tick_params(axis="y", labelcolor="k", labelsize=20)
        ax_pc.legend(lines, [l.get_label() for l in lines], loc=legend_loc, fontsize=12)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="results/sweep")
    p.add_argument("--outdir", default=None, help="default: <indir>/plots")
    p.add_argument("--labels", nargs=2, default=None,
                   help="panel letters, exc then inh (default: A B for --vary both, none otherwise)")
    p.add_argument("--sd-ylim-exc", nargs=2, type=float, default=[0.275, 0.31])
    p.add_argument("--sd-ylim-inh", nargs=2, type=float, default=[0.0, 0.6])
    args = p.parse_args()

    plt.rcdefaults()
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Liberation Sans", "DejaVu Sans"]})
    path = lambda name: os.path.join(args.indir, name)
    load = lambda name: np.loadtxt(path(name))
    maybe = lambda name: load(name) if os.path.exists(path(name)) else None
    with open(path("params.json")) as f:
        vary = json.load(f)["vary"]
    labels = args.labels or (["A", "B"] if vary == "both" else [None, None])
    outdir = args.outdir or os.path.join(args.indir, "plots")
    os.makedirs(outdir, exist_ok=True)
    tau, frequency = load("tau.txt"), load("frequency.txt")

    panel(tau, frequency, load("PSD_exc.txt"), load("pc_exc.txt"),
          maybe("int_spike_distance_exc.txt"), labels[0], XLABEL[vary], None,
          args.sd_ylim_exc, "upper center", os.path.join(outdir, "Exc_Analysis.png"))
    panel(tau, frequency, load("PSD_inh.txt"), load("pc_inh.txt"),
          maybe("int_spike_distance_inh.txt"), labels[1], XLABEL[vary], (0, 0.6),
          args.sd_ylim_inh, "upper right", os.path.join(outdir, "Inh_Analysis.png"))


if __name__ == "__main__":
    main()
