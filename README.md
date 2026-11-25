# Inhibitory synaptic timescales drive oscillatory regime transitions

Simulation and analysis code for

> Kheiri S, Taghavi S, Perc M, Valizadeh A (2026).
> Inhibitory synaptic timescales drive oscillatory regime transitions in recurrent excitatory-inhibitory networks.
> *Physical Review Research* 8: 023251. https://doi.org/10.1103/zmn7-7kf6

A recurrent network of 240 excitatory and 60 inhibitory conductance-based
leaky integrate-and-fire neurons (NEST `iaf_cond_beta`) is driven by Poisson
input. The excitatory synaptic decay time stays at 5 ms while the decay times
of the inhibitory synapses are varied. Instead of shifting the rhythm
gradually, they switch the network between a coherent high-frequency regime
(HFO, above 100 Hz), a coherent gamma regime (30 to 80 Hz) and a transition
state with low synchrony in between. The I to I decay time selects the
regime; the I to E decay time mostly tunes the frequency and the width of
the transition.

## Figures and scripts

| Figure | What it shows | Simulate | Draw |
|---|---|---|---|
| 1(a,b) | PSD, pairwise correlation and SPIKE-distance against tau_inh, N = 300 | `tau_sweep.py` | `tau_sweep_plot.py` |
| 1(c-e) | rasters, rates and spectra at tau_inh = 2.0, 3.5, 8.0 ms | `raster.py` | (same script) |
| 1(f,g) | as (a,b) for N = 1000 | `tau_sweep.py --order 1000 --probability 0.15 --tmax 5000` | `tau_sweep_plot.py --labels F G` |
| 2(a-c) | PSD maps, I to E fixed at 2, 4, 7 ms, I to I varied | `tau_sweep.py --vary inh_inh --fixed X` | `tau_sweep_plot.py` (see note) |
| 2(d-f), 3(d-f) | rasters at single points | `raster.py --tau-inh-exc X --tau-inh-inh Y` | (same script) |
| 3(a-c) | PSD maps, I to I fixed at 2, 4, 7 ms, I to E varied | `tau_sweep.py --vary inh_exc --fixed X` | `tau_sweep_plot.py` (see note) |
| 4 | peak frequency and power on the (I to I, I to E) grid | `peak_tau.py` | `peak_tau_plot.py` |
| 5 | E-I phase difference and ISI distributions at 2, 4, 8 ms | `phase_isi.py --tau-inh X` | (same script) |

`network.py` builds the network and holds the parameters of Table 1;
`analysis.py` holds the rate, PSD, pairwise correlation, SPIKE-distance and
peak-detection code. Every script starts with a docstring that says what it
measures and how to run it, and `python <script> --help` lists its options.
Output goes to `results/`, which is not committed.

## Running

The paper used NEST 3.8.0. With Docker:

```bash
docker run --rm -it -v "$PWD":/work -w /work nest/nest-simulator:3.8 bash
pip install pyspike          # inside the container
python tau_sweep.py --outdir results/fig1_n300
python tau_sweep_plot.py --indir results/fig1_n300
```

All of Fig. 1, for example:

```bash
python tau_sweep.py --outdir results/fig1_n300
python tau_sweep.py --order 1000 --probability 0.15 --tmax 5000 --outdir results/fig1_n1000
for t in 2.0 3.5 8.0; do python raster.py --tau-inh-exc $t --tau-inh-inh $t; done
python tau_sweep_plot.py --indir results/fig1_n300
python tau_sweep_plot.py --indir results/fig1_n1000 --labels F G --sd-ylim-exc 0.278 0.301
```

Cost: one point of a sweep (N = 300, 10 s, 16 threads, with the
SPIKE-distance) took 2.5 s on a 16-core workstation, so a 91-point sweep
takes a few minutes; N = 1000 takes longer, mostly for the SPIKE-distance.
`peak_tau.py` runs 46 x 46 = 2116 simulations with one thread each; for a
quick look use `--step 1.0 --tmax 2000`.

## Reproducibility

- Seeds: NEST `rng_seed = 22587425`, numpy `seed(31360215)`. The initial
  membrane potentials are drawn uniformly in [-70, -51] mV with numpy.
- `tau_sweep.py` resets both seeds at every point and uses 16 threads, as
  the published Fig. 1 runs did. NEST's random streams depend on the number
  of threads, so other thread counts give statistically equivalent but not
  identical results.
- `peak_tau.py` keeps the seeding of the original script: after the first
  setup only `nest.ResetKernel()` is called, which returns NEST to its
  default seed and one thread at every grid point.
- `raster.py` and `phase_isi.py` run once with one thread.

## Notes on the original scripts

These scripts were reorganised from the ones used for the paper. Model,
parameters and analysis are unchanged. With NEST 3.9 and shortened runs
(1 s, two decay times), `tau_sweep.py`, `peak_tau.py` and the network of
`raster.py` gave bit-identical spikes, PSDs, correlations, SPIKE-distances
and peak frequencies to the original scripts. `phase_isi.py` was only run,
not compared, because the original was left at other settings. What
differs:

- Figs. 2 and 3: the original runs (February 2026) used a version of the
  sweep with one decay time fixed, 16 threads, and NEST's default seed with
  numpy seed 123. `tau_sweep.py --vary` does the same sweep with the seeds
  above, so the maps agree in structure but not point by point. The
  original plotting script for these two figures was not found;
  `tau_sweep_plot.py` draws the same quantities in the Fig. 1 layout.
- Fig. 5: the plotting code was commented out in the original
  `study2test.py` and is reassembled in `phase_isi.py`. The published
  panels span 10 s, which is the default here.
- The multimeter that the original scripts attached to every neuron is
  still created, so node ids are unchanged, but it only records when asked.

## Citation

```bibtex
@article{Kheiri2026InhibitoryTimescales,
  author  = {Kheiri, Siamak and Taghavi, Saeed and Perc, Matja{\v{z}} and Valizadeh, Alireza},
  title   = {Inhibitory Synaptic Timescales Drive Oscillatory Regime Transitions in Recurrent Excitatory-Inhibitory Networks},
  journal = {Physical Review Research},
  year    = {2026},
  volume  = {8},
  number  = {2},
  pages   = {023251},
  doi     = {10.1103/zmn7-7kf6}
}
```

## Licence

No licence has been chosen yet. Until one is added, the code is visible but
not licensed for reuse.
