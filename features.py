"""
features.py
NeuroCast — Feature extraction from averaged SSEP response sequences.

Extracts 9 features per response using a rolling 18-response window (15 min):
  [0] amp_smooth   — mean of last 3 response amplitudes / adaptive baseline
  [1] lat_shift    — regression-predicted latency shift vs adaptive baseline
  [2] slope_norm   — amplitude trend slope, normalised by adaptive baseline
  [3] amp_var      — amplitude variance over the rolling window
  [4-8] wav_e0–e4  — Daubechies-4 wavelet sub-band energies (5 levels)

Adaptive baseline is established from the first 8 responses, consistent
with real IONM practice (Natus Xltek patient-specific baseline calibration).
"""

import numpy as np
import pywt
from scipy import stats


def extract_features(amps, lats, sweeps, baseline_amp, baseline_lat,
                     window=18, adapt_n=8):
    """
    Rolling-window feature extraction over a sequence of averaged SSEP responses.

    Parameters
    ----------
    amps         : ndarray (n,)     P40 amplitude per response (µV)
    lats         : ndarray (n,)     P40 latency per response (ms)
    sweeps       : ndarray (n,500)  averaged waveforms
    baseline_amp : float            nominal amplitude (kept for API compatibility)
    baseline_lat : float            nominal latency (kept for API compatibility)
    window       : int              rolling window length (default 18 responses)
    adapt_n      : int              responses used for adaptive baseline (default 8)

    Returns
    -------
    features  : ndarray (n−window, 9)
    indices   : list[int]
    adapt_amp : float
    adapt_lat : float
    """
    # Patient-specific baseline from the opening stabilisation period
    adapt_amp = float(np.mean(amps[:adapt_n]))
    adapt_lat = float(np.mean(lats[:adapt_n]))

    feat_rows, indices = [], []

    for i in range(window, len(amps)):
        w_amp = amps[i - window:i]
        w_lat = lats[i - window:i]
        w_sw  = sweeps[i - window:i]
        t_arr = np.arange(len(w_amp))

        # Amplitude: mean of last 3 responses (noise-robust at σ ≈ 20%)
        amp_smooth = float(np.mean(w_amp[-3:])) / adapt_amp

        # Slope: full-window linear regression, normalised by adaptive baseline
        slope_a, *_ = stats.linregress(t_arr, w_amp)
        slope_norm  = float(slope_a) / adapt_amp

        # Latency: regression-predicted value at current timestep
        slope_l, intercept_l, *_ = stats.linregress(t_arr, w_lat)
        lat_pred  = slope_l * (len(w_lat) - 1) + intercept_l
        lat_shift = (float(lat_pred) - adapt_lat) / adapt_lat

        # Amplitude variance over window
        amp_var = float(np.var(w_amp))

        # Wavelet morphology features (db4, 5 levels)
        mean_sw = np.mean(w_sw, axis=0)
        coeffs  = pywt.wavedec(mean_sw, 'db4', level=5)
        wav_e   = [float(np.sum(c**2) / max(len(mean_sw), 1)) for c in coeffs[:5]]

        feat_rows.append([amp_smooth, lat_shift, slope_norm, amp_var] + wav_e)
        indices.append(i)

    return np.array(feat_rows, dtype=np.float32), indices, adapt_amp, adapt_lat


if __name__ == '__main__':
    from generate_ssep import build_ssep_sequence

    times, amps, lats, sweeps, b_amp, b_lat = build_ssep_sequence()
    feats, idx, adapt_amp, adapt_lat = extract_features(
        amps, lats, sweeps, b_amp, b_lat, window=18, adapt_n=8)

    print(f"Feature matrix shape: {feats.shape}")
    print(f"  Rows = {feats.shape[0]}  (responses after warmup window)")
    print(f"  Cols = {feats.shape[1]}  (9 features)")
    print(f"\nAdaptive baseline:  amp = {adapt_amp:.3f} µV  lat = {adapt_lat:.2f} ms")

    names = ["amp_smooth", "lat_shift", "slope_norm", "amp_var",
             "wav_e0", "wav_e1", "wav_e2", "wav_e3", "wav_e4"]

    print("\nFeature values at first window (stable baseline):")
    for n, v in zip(names, feats[0]):
        print(f"  {n:<12} = {v:.5f}")

    print("\nFeature values at last window (plateau, post-alert):")
    for n, v in zip(names, feats[-1]):
        print(f"  {n:<12} = {v:.5f}")
