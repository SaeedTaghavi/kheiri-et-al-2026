"""
Motivation
----------
The analysis steps shared by the figure scripts of Kheiri et al. (2026),
copied from the original scripts without changing the computations.

What it measures
----------------
- ``population_rate``: spike-count histogram in 1 ms bins, Gaussian
  smoothing (sigma = 1 bin), in Hz per neuron (paper Eqs. 6 to 8).
- ``fft_psd``: one-sided periodogram of a rate, ``2 |FFT|^2 dt / n``, for
  the positive frequencies without 0 Hz.
- ``pairwise_correlation``: mean zero-lag Pearson correlation of 1 ms binned
  spike trains over all pairs (paper Eq. 9).
- ``spike_distance``: the SPIKE-distance D_Sa of a population, with PySpike
  (paper Eq. 10). PySpike is imported only when it is used.
- ``extract_dominant_frequency``: the peak of a Gaussian-smoothed PSD in a
  band (Fig. 4).

Usage
-----
    from analysis import population_rate, fft_psd
    rate = population_rate(times_exc, n_exc, start=200, tmax=10000)
    freq, psd = fft_psd(rate)
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

BIN = 1.0  # ms


def population_rate(times, n_neurons, start, tmax, bin_ms=BIN, sigma=1.0):
    """Smoothed population rate in Hz per neuron, one value per bin."""
    bins = np.arange(start, tmax + bin_ms, bin_ms)
    counts, _ = np.histogram(times, bins)
    return gaussian_filter1d(counts, sigma=sigma, mode="reflect") * (1e3 / (n_neurons * bin_ms))


def fft_psd(rate, bin_ms=BIN):
    """Return (frequency in Hz, PSD) for indices 1 .. n/2 - 1."""
    n = len(rate)
    dt0 = bin_ms * 1e-3
    keep = np.arange(1, np.floor(n / 2), dtype="int")
    fhat = np.fft.fft(rate)
    psd = 2 * (np.abs(fhat) ** 2) * (dt0 / n)
    freq = np.fft.fftfreq(fhat.size, dt0)
    return freq[keep], psd[keep]


def fft_psd_full(rate, bin_ms=BIN):
    """Return (frequency, PSD) over all FFT bins, as used for Fig. 4."""
    n = len(rate)
    dt0 = bin_ms * 1e-3
    fhat = np.fft.fft(rate)
    return np.fft.fftfreq(fhat.size, dt0), 2 * (np.abs(fhat) ** 2) * (dt0 / n)


def binned_spike_trains(spike_times, neuron_ids, t_min, t_max, bin_ms=BIN):
    """Spike counts per neuron (rows) and bin (columns), for neurons that spiked."""
    neuron_ids = np.asarray(neuron_ids)
    spike_times = np.asarray(spike_times)
    bins = np.arange(t_min, t_max + bin_ms, bin_ms)
    trains = [np.histogram(spike_times[neuron_ids == nid], bins)[0]
              for nid in np.unique(neuron_ids)]
    return np.array(trains), bins[:-1]


def pairwise_correlation(spike_times, neuron_ids, t_min, t_max, bin_ms=BIN):
    """Mean zero-lag Pearson correlation over all pairs of spiking neurons.

    Pairs with a constant train give NaN in np.corrcoef; as in the original
    scripts these are set to 0 and kept in the mean.
    """
    trains, _ = binned_spike_trains(spike_times, neuron_ids, t_min, t_max, bin_ms)
    if trains.shape[0] < 2:
        return np.nan
    corr = np.nan_to_num(np.corrcoef(trains))
    return np.mean(corr[np.triu_indices_from(corr, k=1)])


def spike_distance(spike_times, neuron_ids, t_min, t_max):
    """Return (mean of the pairwise SPIKE-distance matrix, D_Sa of the population).

    The paper's figures plot the second value, ``pyspike.spike_distance``
    over all trains of spiking neurons in [t_min, t_max].
    """
    import pyspike as spk
    trains = [spk.SpikeTrain(spike_times[neuron_ids == nid], [t_min, t_max])
              for nid in np.unique(neuron_ids)]
    matrix = spk.spike_distance_matrix(trains)
    return np.mean(matrix[np.triu_indices_from(matrix, k=1)]), spk.spike_distance(trains)


# ---- dominant frequency (Fig. 4) -------------------------------------------

def preprocess_psd(psd, freq):
    """Sort the frequency axis in ascending order and the PSD with it."""
    freq = np.asarray(freq)
    psd = np.asarray(psd)
    idx = np.argsort(freq)
    return (psd[idx] if psd.ndim == 1 else psd[..., idx]), freq[idx]


def smooth_psd_gaussian(psd, freq, sigma_hz):
    """Gaussian smoothing along the frequency axis."""
    df = np.mean(np.diff(freq))
    if df <= 0:
        raise ValueError("Frequency axis must be strictly increasing.")
    sigma_bins = sigma_hz / df
    if sigma_bins < 1e-6:
        return psd
    return gaussian_filter1d(psd, sigma=sigma_bins, axis=-1)


def dominant_peak_in_band(psd, freq, low, high, min_prominence=0.0):
    """Most prominent peak in [low, high]; the maximum if there is no peak."""
    mask = (freq >= low) & (freq <= high)
    if not np.any(mask):
        return 0.0, 0.0
    psd_band = psd[..., mask]
    freq_band = freq[mask]
    if psd_band.ndim > 1:
        raise ValueError("This function expects 1D PSD for peak detection.")
    peaks, props = find_peaks(psd_band, prominence=min_prominence)
    if peaks.size > 0:
        idx = peaks[np.argmax(props["prominences"])]
    else:
        idx = np.argmax(psd_band)
    return freq_band[idx], psd_band[idx]


def extract_dominant_frequency(psd, freq, band_low, band_high, sigma_hz=2,
                               min_prominence=0.0):
    """Sort, smooth, then return (peak frequency, peak power, smoothed PSD)."""
    psd, freq = preprocess_psd(psd, freq)
    psd_smooth = smooth_psd_gaussian(psd, freq, sigma_hz)
    peak_freq, peak_power = dominant_peak_in_band(psd_smooth, freq, band_low,
                                                  band_high, min_prominence)
    return peak_freq, peak_power, psd_smooth
