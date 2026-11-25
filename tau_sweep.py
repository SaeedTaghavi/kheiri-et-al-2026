"""
Motivation
----------
Figs. 1, 2 and 3 of Kheiri et al. (2026): does the inhibitory decay time
change the network's rhythm gradually, or does it switch between regimes?
The excitatory decay time stays at 5 ms. Three sweeps:

- ``--vary both``: I -> E and I -> I decay times varied together
  (Fig. 1(a,b) with N = 300, Fig. 1(f,g) with N = 1000)
- ``--vary inh_inh --fixed X``: I -> E held at X, I -> I varied (Fig. 2,
  X = 2, 4, 7 ms)
- ``--vary inh_exc --fixed X``: I -> I held at X, I -> E varied (Fig. 3,
  X = 2, 4, 7 ms)

What it measures
----------------
For each decay time, separately for the excitatory and inhibitory
populations:
- the PSD of the smoothed population rate (FFT, 1 ms bins): rows of
  ``PSD_exc.txt`` and ``PSD_inh.txt``, frequencies in ``frequency.txt``
- the mean pairwise correlation of 1 ms binned spike trains
  (``pc_exc.txt``, ``pc_inh.txt``)
- unless ``--no-spike-distance``: the SPIKE-distance D_Sa plotted in Fig. 1
  (``int_spike_distance_*.txt``) and the mean of the pairwise SPIKE-distance
  matrix (``avg_spike_distance_*.txt``)

Every point starts a fresh kernel with the same seeds and 16 threads, as in
the published Fig. 1 runs. NEST's random streams depend on the number of
threads, so keep ``--threads 16`` to reproduce those numbers.

Usage
-----
    python tau_sweep.py --outdir results/fig1_n300
    python tau_sweep.py --order 1000 --probability 0.15 --tmax 5000 \
        --outdir results/fig1_n1000
    python tau_sweep.py --vary inh_inh --fixed 4.0 --no-spike-distance \
        --outdir results/fig2_ie4
    python tau_sweep.py --vary inh_exc --fixed 4.0 --no-spike-distance \
        --outdir results/fig3_ii4
    # then draw it
    python tau_sweep_plot.py --indir results/fig1_n300
"""
import argparse
import json
import os
import time

import numpy as np
import nest

from network import T_START, set_kernel, build_network, split_spikes
from analysis import population_rate, fft_psd, pairwise_correlation, spike_distance


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vary", choices=["both", "inh_inh", "inh_exc"], default="both",
                   help="which inhibitory decay time is swept")
    p.add_argument("--fixed", type=float, default=None,
                   help="value in ms of the decay time that is not swept")
    p.add_argument("--order", type=int, default=300, help="number of neurons N")
    p.add_argument("--probability", type=float, default=0.4, help="connection probability P")
    p.add_argument("--tmax", type=float, default=10000.0, help="simulated time in ms")
    p.add_argument("--tau-min", type=float, default=1.0)
    p.add_argument("--tau-max", type=float, default=10.0)
    p.add_argument("--tau-step", type=float, default=0.1)
    p.add_argument("--threads", type=int, default=16,
                   help="NEST threads; keep 16 to reproduce the published numbers")
    p.add_argument("--no-spike-distance", action="store_true",
                   help="skip PySpike (Figs. 2 and 3 do not show it)")
    p.add_argument("--outdir", default="results/sweep")
    args = p.parse_args()
    if (args.vary == "both") != (args.fixed is None):
        p.error("--fixed is required with --vary inh_inh or inh_exc, and not allowed with both")

    os.makedirs(args.outdir, exist_ok=True)
    taus = np.arange(args.tau_min, args.tau_max + 0.1 * args.tau_step, args.tau_step)
    start, tmax = T_START, args.tmax
    n = len(taus)

    psd_exc, psd_inh = [], []
    pc_exc, pc_inh = np.zeros(n), np.zeros(n)
    avg_sd_exc, avg_sd_inh = np.full(n, np.nan), np.full(n, np.nan)
    int_sd_exc, int_sd_inh = np.full(n, np.nan), np.full(n, np.nan)
    frequency = None

    t0 = time.time()
    for k, tau in enumerate(taus):
        tau_ie = args.fixed if args.vary == "inh_inh" else tau
        tau_ii = args.fixed if args.vary == "inh_exc" else tau
        print(f"{k + 1}/{n}  tau I->E = {tau_ie:.1f} ms, tau I->I = {tau_ii:.1f} ms", flush=True)
        set_kernel(threads=args.threads)
        net = build_network(args.order, args.probability, tau_inh_exc=tau_ie, tau_inh_inh=tau_ii)
        nest.Simulate(tmax)
        t_e, s_e, t_i, s_i = split_spikes(net)

        pc_exc[k] = pairwise_correlation(t_e, s_e, start, tmax)
        pc_inh[k] = pairwise_correlation(t_i, s_i, start, tmax)
        if not args.no_spike_distance:
            avg_sd_exc[k], int_sd_exc[k] = spike_distance(t_e, s_e, start, tmax)
            avg_sd_inh[k], int_sd_inh[k] = spike_distance(t_i, s_i, start, tmax)

        frequency, pe = fft_psd(population_rate(t_e, net["n_exc"], start, tmax))
        _, pi = fft_psd(population_rate(t_i, net["n_inh"], start, tmax))
        psd_exc.append(pe)
        psd_inh.append(pi)
        print(f"   PC exc {pc_exc[k]:.4f} inh {pc_inh[k]:.4f}   "
              f"D_Sa exc {int_sd_exc[k]:.4f} inh {int_sd_inh[k]:.4f}", flush=True)

    out = lambda name: os.path.join(args.outdir, name)
    np.savetxt(out("tau.txt"), taus)
    np.savetxt(out("frequency.txt"), frequency)
    np.savetxt(out("PSD_exc.txt"), np.array(psd_exc))
    np.savetxt(out("PSD_inh.txt"), np.array(psd_inh))
    np.savetxt(out("pc_exc.txt"), pc_exc)
    np.savetxt(out("pc_inh.txt"), pc_inh)
    if not args.no_spike_distance:
        np.savetxt(out("avg_spike_distance_exc.txt"), avg_sd_exc)
        np.savetxt(out("avg_spike_distance_inh.txt"), avg_sd_inh)
        np.savetxt(out("int_spike_distance_exc.txt"), int_sd_exc)
        np.savetxt(out("int_spike_distance_inh.txt"), int_sd_inh)
    with open(out("params.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Wrote {args.outdir} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
