"""
Motivation
----------
Fig. 4 of Kheiri et al. (2026): how do the two inhibitory pathways share the
control of the rhythm? The I -> E and I -> I decay times are varied on a
grid, and the dominant oscillation of each population is located in the
plane of the two.

What it measures
----------------
For every (tau I->E, tau I->I) on the grid, for the excitatory and the
inhibitory population: the frequency and power of the most prominent peak
of the rate PSD between 0 and 200 Hz, after Gaussian smoothing of the PSD
with sigma = 2 Hz (``extract_dominant_frequency`` in ``analysis.py``).
Output: ``freq_exc.txt``, ``freq_inh.txt``, ``power_exc.txt``,
``power_inh.txt`` (rows: I->E, columns: I->I) and ``tau.txt``.

Seeding, as in the original script: the kernel is set up once with the
fixed seeds, and inside the loop only ``nest.ResetKernel()`` is called. That
resets NEST to its default seed and to one thread for every grid point,
while the initial membrane potentials keep drawing from one numpy stream
across the whole grid.

Usage
-----
    python peak_tau.py --outdir results/fig4            # 46 x 46 grid, 10 s each
    python peak_tau.py --step 1.0 --tmax 2000 --outdir results/fig4_quick
    python peak_tau_plot.py --indir results/fig4
"""
import argparse
import os
import time

import numpy as np
import nest

from network import T_START, set_kernel, build_network, split_spikes
from analysis import population_rate, fft_psd_full, extract_dominant_frequency


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--order", type=int, default=300)
    p.add_argument("--probability", type=float, default=0.4)
    p.add_argument("--tmax", type=float, default=10000.0)
    p.add_argument("--tau-min", type=float, default=1.0)
    p.add_argument("--tau-max", type=float, default=10.0)
    p.add_argument("--step", type=float, default=0.2)
    p.add_argument("--outdir", default="results/fig4")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    taus = np.arange(args.tau_min, args.tau_max + 0.1 * args.step, args.step)
    start, tmax = T_START, args.tmax
    shape = (len(taus), len(taus))
    freq_exc, freq_inh = np.zeros(shape), np.zeros(shape)
    power_exc, power_inh = np.zeros(shape), np.zeros(shape)

    t0 = time.time()
    set_kernel(threads=1)
    for i, tau_ie in enumerate(taus):
        for j, tau_ii in enumerate(taus):
            print(f"I->E {tau_ie:.1f} ms, I->I {tau_ii:.1f} ms", flush=True)
            nest.ResetKernel()
            net = build_network(args.order, args.probability, tau_inh_exc=tau_ie, tau_inh_inh=tau_ii)
            nest.Simulate(tmax)
            t_e, _, t_i, _ = split_spikes(net)
            freq, psd_e = fft_psd_full(population_rate(t_e, net["n_exc"], start, tmax))
            _, psd_i = fft_psd_full(population_rate(t_i, net["n_inh"], start, tmax))
            freq_exc[i, j], power_exc[i, j], _ = extract_dominant_frequency(
                psd_e, freq, band_low=0, band_high=200, sigma_hz=2, min_prominence=0.01)
            freq_inh[i, j], power_inh[i, j], _ = extract_dominant_frequency(
                psd_i, freq, band_low=0, band_high=200, sigma_hz=2, min_prominence=0.01)

    out = lambda name: os.path.join(args.outdir, name)
    np.savetxt(out("tau.txt"), taus)
    np.savetxt(out("freq_exc.txt"), freq_exc)
    np.savetxt(out("freq_inh.txt"), freq_inh)
    np.savetxt(out("power_exc.txt"), power_exc)
    np.savetxt(out("power_inh.txt"), power_inh)
    print(f"Wrote {args.outdir} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
