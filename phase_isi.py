"""
Motivation
----------
Fig. 5 of Kheiri et al. (2026): are the HFO and gamma regimes different
ways of coordinating excitation and inhibition? At one decay time (2, 4 or
8 ms for both inhibitory pathways) this measures the phase relation between
the excitatory and inhibitory population rates and the spike-interval
statistics of each population.

What it measures
----------------
From the unsmoothed population rates (1 ms bins, Hz per neuron), after
subtracting the mean:
- polar histograms (50 bins, density) of the instantaneous E-I phase
  difference in the fixed gamma band (30-80 Hz) and HFO band (100-200 Hz):
  5th-order Butterworth band-pass (sosfiltfilt), Hilbert phase, difference
  wrapped to (-pi, pi]
- the same phase difference against time, filtered +-10 Hz around each
  population's own Welch peak in that band, with its circular mean (red)
- pooled ISI distributions (5 ms bins, 0-200 ms, density) of the excitatory
  and inhibitory neurons that fired at least 10 spikes
The paper shows the HFO band for 2 ms, both bands for 4 ms and the gamma
band for 8 ms; this script draws both bands every time.

The computation is the one in the original ``study2test.py``; its plotting
code for this figure was commented out there and is reassembled here, so
the layout is close to, not identical with, the published figure. The
original was last set to N = 1000 and 2 s; the published panels span 10 s,
so 10 s is the default here. Check N with the authors before relying on it.

Usage
-----
    python phase_isi.py --tau-inh 2.0
    python phase_isi.py --tau-inh 4.0
    python phase_isi.py --tau-inh 8.0 --order 1000
"""
import argparse
import os

import numpy as np
import nest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import butter, hilbert, sosfiltfilt, welch

from network import T_START, set_kernel, build_network, split_spikes

FS = 1000.0  # Hz, 1 ms bins
BW = 10.0    # Hz, half width around the peak


def bandpass(x, low, high, fs=FS, order=5):
    nyq = 0.5 * fs
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def peak_in_band(rate, fmin, fmax, nperseg):
    freq, psd = welch(rate, fs=FS, nperseg=nperseg)
    mask = (freq >= fmin) & (freq <= fmax)
    if not mask.any():
        raise RuntimeError("Frequency range empty; check fs and signal length")
    return freq[mask][np.argmax(psd[mask])]


def phase_difference(x_e, x_i):
    return np.angle(np.exp(1j * (np.angle(hilbert(x_e)) - np.angle(hilbert(x_i)))))


def pooled_isi(times, senders, min_spikes=10):
    isis = []
    for gid in np.unique(senders):
        st = np.sort(times[senders == gid])
        d = np.diff(st)
        if st.size >= 2 and len(d) >= min_spikes - 1:
            isis.extend(d.tolist())
    return np.array(isis)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tau-inh", type=float, default=2.0,
                   help="I -> E and I -> I decay time (ms)")
    p.add_argument("--order", type=int, default=300)
    p.add_argument("--probability", type=float, default=0.4)
    p.add_argument("--tmax", type=float, default=10000.0)
    p.add_argument("--outdir", default="results/fig5")
    args = p.parse_args()

    plt.rcdefaults()
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Liberation Sans", "DejaVu Sans"]})
    set_kernel(threads=1)
    net = build_network(args.order, args.probability,
                        tau_inh_exc=args.tau_inh, tau_inh_inh=args.tau_inh)
    nest.Simulate(args.tmax)
    t_e, s_e, t_i, s_i = split_spikes(net)
    start, tmax = T_START, args.tmax

    bins = np.arange(start, tmax + 1.0, 1.0)
    rate_e = 1e3 * np.histogram(t_e, bins)[0] / net["n_exc"]
    rate_i = 1e3 * np.histogram(t_i, bins)[0] / net["n_inh"]
    nperseg = min(2048, len(rate_e))
    r_e, r_i = rate_e - rate_e.mean(), rate_i - rate_i.mean()

    bands = {"Gamma": (30.0, 80.0), "HFO": (100.0, 200.0)}
    polar, trace = {}, {}
    for name, (lo, hi) in bands.items():
        polar[name] = phase_difference(bandpass(rate_e, lo, hi), bandpass(rate_i, lo, hi))
        f_e = peak_in_band(rate_e, lo, hi, nperseg)
        f_i = peak_in_band(rate_i, lo, hi, nperseg)
        trace[name] = phase_difference(bandpass(r_e, f_e - BW, f_e + BW),
                                       bandpass(r_i, f_i - BW, f_i + BW))
        mean = np.angle(np.mean(np.exp(1j * polar[name])))
        print(f"{name}: peak E {f_e:.1f} Hz, I {f_i:.1f} Hz, mean E-I phase {mean:.3f} rad")

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"tau{args.tau_inh}"
    t = np.arange(len(rate_e)) / FS

    fig = plt.figure(figsize=(14, 9))
    for row, name in enumerate(bands):
        ax_p = fig.add_subplot(2, 2, 2 * row + 1, projection="polar")
        ax_p.hist(polar[name], bins=50, density=True, bottom=0.0,
                  color="mediumseagreen", alpha=0.8, edgecolor="k")
        ax_p.set_title(f"{name} Band", va="bottom", y=1.15, fontsize=28)
        ax_p.tick_params(axis="y", labelsize=18)
        ax_p.tick_params(axis="x", labelsize=18, pad=10.0)
        ax_p.set_rlabel_position(135)
        ax_t = fig.add_subplot(2, 2, 2 * row + 2)
        ax_t.plot(t, trace[name], color="purple", linewidth=2.0)
        ax_t.axhline(np.angle(np.mean(np.exp(1j * trace[name]))), color="red",
                     linestyle="--", dashes=(6, 4), linewidth=2.0)
        ax_t.set_ylabel(r"$\Delta \phi$ (rad)", fontsize=26)
        ax_t.set_xlabel("t (s)", fontsize=26)
        ax_t.tick_params(labelsize=20)
    fig.tight_layout()
    out = os.path.join(args.outdir, f"phase_{tag}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)

    isi_bins = np.arange(0, 200 + 5, 5)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, isis, color in [(axs[0], pooled_isi(t_e, s_e), "skyblue"),
                            (axs[1], pooled_isi(t_i, s_i), "tomato")]:
        ax.hist(isis, bins=isi_bins, density=True, alpha=0.8, color=color, edgecolor="k")
        ax.set_xlabel("ISI (ms)", fontsize=34, labelpad=8.0)
        ax.tick_params(labelsize=22, pad=8.0)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.tick_params(which="major", length=14, width=1.8)
        ax.tick_params(which="minor", length=6, width=1.0)
    axs[0].set_ylabel("PDF", fontsize=34, labelpad=8.0)
    fig.tight_layout()
    out = os.path.join(args.outdir, f"isi_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
