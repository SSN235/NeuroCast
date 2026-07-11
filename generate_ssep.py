"""
generate_ssep.py
NeuroCast — Synthetic SSEP averaged-response sequence generator.

Simulates the data stream NeuroCast would receive from the IONM hardware
(Natus Xltek via SDK API). Each data point represents one averaged SSEP
response acquired over ~50 seconds (~200 stimulations at 4 Hz).

Waveform model: sum of three Gaussians (P40 complex — Chiappa 2004).
Degradation model: linear amplitude fall over 6.7 min (Nuwer 1995).
Noise model: σ = 20% inter-response variability (literature: 30–40%).
"""

import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_averaged_response(amplitude_uv, latency_ms, residual_noise_std=0.25):
    """
    One averaged SSEP response (P40 complex, 100 ms post-stimulus window).
    Modelled as primary P40 peak + N35 pre-peak + N50 post-peak + residual noise.
    """
    t         = np.linspace(0, 100, 500)
    primary   =  amplitude_uv * np.exp(-0.5 * ((t - latency_ms) / 2.8)**2)
    pre_peak  = -0.30 * amplitude_uv * np.exp(-0.5 * ((t - (latency_ms - 7.5)) / 2.0)**2)
    post_peak = -0.45 * amplitude_uv * np.exp(-0.5 * ((t - (latency_ms + 9.0)) / 3.0)**2)
    noise     = np.random.normal(0, residual_noise_std, len(t))
    return t, primary + pre_peak + post_peak + noise


def build_ssep_sequence(n_sweeps=75, dt_s=50.0,
                        degrade_start=30, degrade_end=38,
                        seed=42):
    """
    Build a time series of averaged SSEP responses over a simulated surgical case.

    Default case structure (75 responses × 50 s = 62 min):
      Responses  0–29 : stable baseline (t = 0–25 min)
      Responses 30–37 : gradual degradation, 100% → 42% amplitude (t = 25–32 min)
      Responses 38–74 : plateau at 42%, cause not reversed (t = 32–62 min)

    Amplitude noise: σ = 20% baseline (baseline), 15% (degradation), 12% (plateau).
    Latency noise:   σ = 1.2 ms (~3% of 40 ms; literature limit < 6%).
    """
    np.random.seed(seed)
    BASELINE_AMP = 3.2   # µV  (Chiappa 2004)
    BASELINE_LAT = 40.0  # ms  (Nuwer 1995)

    times, amps, lats, sweeps = [], [], [], []

    for i in range(n_sweeps):
        t_s = i * dt_s

        if i < degrade_start:
            a = BASELINE_AMP + np.random.normal(0, 0.20 * BASELINE_AMP)
            l = BASELINE_LAT + np.random.normal(0, 1.2)

        elif i < degrade_end:
            frac = (i - degrade_start) / (degrade_end - degrade_start)
            a = BASELINE_AMP * (1.0 - 0.58 * frac) + np.random.normal(0, 0.15 * BASELINE_AMP)
            l = BASELINE_LAT * (1.0 + 0.09 * frac) + np.random.normal(0, 1.2)

        else:
            a = BASELINE_AMP * 0.42 + np.random.normal(0, 0.12 * BASELINE_AMP)
            l = BASELINE_LAT * 1.09  + np.random.normal(0, 1.0)

        _, sw = generate_averaged_response(max(a, 0.05), max(l, 30.0))
        times.append(t_s); amps.append(a); lats.append(l); sweeps.append(sw)

    return (np.array(times), np.array(amps), np.array(lats),
            np.array(sweeps), BASELINE_AMP, BASELINE_LAT)


if __name__ == '__main__':
    times, amps, lats, sweeps, b_amp, b_lat = build_ssep_sequence()
    alert_mask = amps / b_amp < 0.5
    alert_idx  = int(alert_mask.argmax()) if alert_mask.any() else -1

    print(f"Sequence:          {len(times)} averaged responses, dt=50s, total={times[-1]/60:.1f} min")
    print(f"Nominal baseline:  {b_amp} µV amplitude | {b_lat} ms latency")
    print(f"Adaptive baseline  (first 8 responses): {amps[:8].mean():.3f} µV")
    print(f"Baseline rel. SD:  {amps[:30].std()/b_amp*100:.1f}%  (literature: 30–40%)")
    if alert_idx >= 0:
        print(f"Standard alert:    response {alert_idx}, t = {times[alert_idx]/60:.1f} min")
        print(f"Amplitude at alert:{amps[alert_idx]:.3f} µV ({amps[alert_idx]/b_amp*100:.1f}% of baseline)")
    else:
        print("Standard alert:    never fired")
