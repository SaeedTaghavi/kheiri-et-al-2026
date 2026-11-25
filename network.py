"""
Motivation
----------
Every figure in Kheiri et al. (2026) uses the same recurrent E-I network of
conductance-based LIF neurons (NEST ``iaf_cond_beta``). This module builds it
once, in exactly the order the original scripts did, so that the same seeds
give the same spikes in every figure script.

What it measures
----------------
Nothing by itself. ``build_network`` creates the populations, the Poisson
drive, the recurrent connections and a spike recorder, and returns them.
``set_kernel`` applies the kernel settings used by the paper's scripts.

Usage
-----
    import nest
    from network import set_kernel, build_network
    set_kernel(threads=1)
    net = build_network(order=300, probability=0.4,
                        tau_inh_exc=2.0, tau_inh_inh=2.0)
    nest.Simulate(10000.0)
    spikes = net["spikes"].get("events")

Parameters (Table 1 of the paper)
---------------------------------
80 % excitatory and 20 % inhibitory neurons. Excitatory: C_m = 200 pF,
inhibitory: C_m = 100 pF, both g_L = 10 nS (tau_m = 20 and 10 ms),
V_th = -52 mV, V_reset = -59 mV, t_ref = 1 ms, E_ex = 0 mV, E_in = -70 mV.
Synaptic rise time 0.5 ms, excitatory decay 5 ms, delay 2 ms. Fixed in-degree
C = P * N_pre without autapses or multapses. Weights are a fixed total
conductance divided by the in-degree, which gives g_ee = 0.05, g_ie = 0.6,
g_ei = 0.9 and g_ii = 3.5 nS for N = 300 and P = 0.4. External drive: one
Poisson generator per population at 4 kHz (independent trains per neuron),
0.2 nS onto excitatory and 0.25 nS onto inhibitory neurons.
"""
import numpy as np
import nest

NEST_SEED = 22587425
NUMPY_SEED = 31360215
DT = 0.1            # ms
T_START = 200.0     # ms, transient discarded by the recorders

V_REST = -70.0      # lower bound of the random initial V_m
V_INIT_MAX = -51.0  # upper bound of the random initial V_m
TAU_RISE = 0.5
TAU_EXC = 5.0       # E -> E and E -> I decay time, fixed in the paper
DELAY = 2.0
EXTERNAL_RATE = 4000.0

# Total conductance per target, divided by the in-degree below.
G_TOTAL = {"ee": 4.8, "ie": 57.6, "ei": -21.6, "ii": -84.0}


def set_kernel(threads=1, seed=NEST_SEED, numpy_seed=NUMPY_SEED):
    """Reset NEST and apply the settings the paper's scripts used."""
    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")
    nest.SetKernelStatus({"overwrite_files": True, "print_time": False,
                          "local_num_threads": threads})
    nest.rng_seed = seed
    np.random.seed(numpy_seed)
    nest.resolution = DT


def build_network(order, probability, tau_inh_exc, tau_inh_inh,
                  tau_exc_exc=TAU_EXC, tau_exc_inh=TAU_EXC,
                  start=T_START, record_vm=False):
    """Create and connect the network; return its parts in a dict.

    tau_inh_exc is the I -> E decay time (``tau_decay_in`` of the
    excitatory neurons), tau_inh_inh the I -> I decay time (``tau_decay_in``
    of the inhibitory neurons). Initial V_m is uniform in [-70, -51] mV,
    drawn from numpy's global generator, excitatory neurons first.
    """
    n_exc, n_inh = int(0.8 * order), int(0.2 * order)

    exc_params = {"C_m": 200.0, "g_L": 10.0, "t_ref": 1.0,
                  "E_ex": 0.0, "E_in": -70.0,
                  "tau_rise_ex": TAU_RISE, "tau_decay_ex": tau_exc_exc,
                  "tau_rise_in": TAU_RISE, "tau_decay_in": tau_inh_exc,
                  "V_reset": -59.0, "V_th": -52.0, "I_e": 0.0,
                  "V_m": [V_REST + (V_INIT_MAX - V_REST) * np.random.rand()
                          for _ in range(n_exc)]}
    inh_params = {"C_m": 100.0, "g_L": 10.0, "t_ref": 1.0,
                  "E_ex": 0.0, "E_in": -70.0,
                  "tau_rise_ex": TAU_RISE, "tau_decay_ex": tau_exc_inh,
                  "tau_rise_in": TAU_RISE, "tau_decay_in": tau_inh_inh,
                  "V_reset": -59.0, "V_th": -52.0, "I_e": 0.0,
                  "V_m": [V_REST + (V_INIT_MAX - V_REST) * np.random.rand()
                          for _ in range(n_inh)]}

    pop_exc = nest.Create("iaf_cond_beta", n_exc, params=exc_params)
    pop_inh = nest.Create("iaf_cond_beta", n_inh, params=inh_params)
    pop = pop_exc + pop_inh

    # The original scripts always created this multimeter; it is kept so that
    # node ids stay the same, but it only records when asked to.
    multimeter = nest.Create("multimeter", {"record_from": ["V_m", "g_ex", "g_in"]})
    multimeter.set({"interval": 0.1, "start": start})
    spikes = nest.Create("spike_recorder", {"start": start})
    if record_vm:
        nest.Connect(multimeter, pop)
    nest.Connect(pop, spikes)

    noise_exc = nest.Create("poisson_generator", params={"rate": EXTERNAL_RATE})
    noise_inh = nest.Create("poisson_generator", params={"rate": EXTERNAL_RATE})
    nest.Connect(noise_exc, pop_exc, syn_spec={"weight": 0.2, "delay": 0.1})
    nest.Connect(noise_inh, pop_inh, syn_spec={"weight": 0.25, "delay": 0.1})

    c_ee = int(probability * n_exc)   # E -> E in-degree
    c_ie = int(probability * n_exc)   # E -> I in-degree
    c_ei = int(probability * n_inh)   # I -> E in-degree
    c_ii = int(probability * n_inh)   # I -> I in-degree

    def conn(indegree):
        return {"rule": "fixed_indegree", "indegree": indegree,
                "allow_autapses": False, "allow_multapses": False}

    nest.Connect(pop_exc, pop_exc, conn(c_ee),
                 syn_spec={"weight": G_TOTAL["ee"] / c_ee, "delay": DELAY})
    nest.Connect(pop_exc, pop_inh, conn(c_ie),
                 syn_spec={"weight": G_TOTAL["ie"] / c_ie, "delay": DELAY})
    nest.Connect(pop_inh, pop_exc, conn(c_ei),
                 syn_spec={"weight": G_TOTAL["ei"] / c_ei, "delay": DELAY})
    nest.Connect(pop_inh, pop_inh, conn(c_ii),
                 syn_spec={"weight": G_TOTAL["ii"] / c_ii, "delay": DELAY})

    return {"pop_exc": pop_exc, "pop_inh": pop_inh, "n_exc": n_exc,
            "n_inh": n_inh, "spikes": spikes, "multimeter": multimeter,
            "start": start}


def split_spikes(net):
    """Return (times_exc, senders_exc, times_inh, senders_inh)."""
    ev = net["spikes"].get("events")
    senders, times = ev["senders"], ev["times"]
    is_exc = senders <= net["n_exc"]
    return times[is_exc], senders[is_exc], times[~is_exc], senders[~is_exc]
