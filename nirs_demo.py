"""
nirs_demo.py
============
PURPOSE: End-to-end demonstration of the NeuroCast pipeline.

PIPELINE ORDER:
  generate_ssep.py  →  features.py  →  nirs_demo.py
    (raw waveforms)    (9 features)     (NIRS + plot)

TWO MODES:
  3a. Rule-based proxy  — no .pt file needed, always available
  3b. Trained TCN       — requires running tcn_model.py first to produce
                          neurocast_tcn.pt, then uncomment block 3b

EXPECTED OUTPUT (seed=42, default parameters, rule-based proxy):
  NIRS Advisory at  29.2 min  →  4.2 min before standard alert
  NIRS Warning  at  30.0 min  →  3.3 min before standard alert
  Standard alert fires at 33.3 min
  Baseline NIRS max: 0.0328  (< 0.05 — no false alarms during stable period)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Use non-interactive backend (saves to file, no popup window)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import from the other three scripts in this pipeline
from generate_ssep import build_ssep_sequence
from features import extract_features

# Ensures the output PNG saves next to this script, not in /
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1: Rule-based NIRS proxy
# ─────────────────────────────────────────────────────────────────────────────

def compute_nirs_proxy(features):
    """
    Computes a NIRS score from features using hand-crafted risk rules.
    This is the stand-in for the trained TCN — same inputs, same output shape,
    but based on explicit logic rather than learned weights.

    IN PRODUCTION: This entire function is replaced by run_inference() from
    tcn_model.py, which runs the trained neural network instead.

    HOW IT WORKS:
      Three separate risk signals are computed and summed:
        1. amp_risk   — how much has amplitude already fallen?
        2. trend_risk — how fast is it still falling?
        3. lat_risk   — is latency increasing (early demyelination sign)?

      The combined raw risk is passed through a sigmoid function to squash
      it into the range (0, 1). The sigmoid centre (0.85) is calibrated so
      that baseline noise produces NIRS < 0.05, and the advisory fires
      ~4 minutes before the standard alert.

    WHY SIGMOID?
      The TCN also ends with a sigmoid. Using the same mathematical form
      here means the rule-based proxy and the neural network are comparable
      — both output a probability-like score with the same threshold meaning.

    Parameters
    ----------
    features : ndarray (57, 9) — from extract_features()
               col 0: amp_smooth  (smoothed amplitude ratio)
               col 1: lat_shift   (regression-predicted latency shift)
               col 2: slope_norm  (normalised amplitude trend slope)

    Returns
    -------
    nirs : ndarray (57,) — NIRS scores in [0, 1], one per response
    """

    # Unpack the three features we use for risk scoring
    # (features 3–8 are used by the TCN but not this simple proxy)
    amp_smooth = features[:, 0]   # 1.0 = at baseline; 0.5 = half of baseline
    lat_shift  = features[:, 1]   # 0.0 = no change; 0.09 = 9% increase
    slope_norm = features[:, 2]   # 0.0 = flat; -0.05 = falling at 5%/interval

    # ── Risk component 1: Amplitude already below baseline ────────────────────
    # (1 - amp_smooth): how far below 1.0 we are (0 if at or above baseline)
    # ×2.8: scaling factor so that a 50% drop gives amp_risk ≈ 1.4
    # Clipped at [0, 3.0]: never negative (above baseline = zero risk),
    #                       never more than 3 (prevents one signal dominating)
    amp_risk = np.clip((1.0 - amp_smooth) * 2.8, 0.0, 3.0)

    # ── Risk component 2: Amplitude actively falling (trend) ──────────────────
    # -slope_norm: flips sign so that a FALLING trend (negative slope) gives
    #              POSITIVE trend_risk
    # ×40.0: slope_norm is typically tiny (e.g. -0.03) so needs large scaling
    # Clipped at [0, 2.0]: only fires when slope is negative, capped at 2
    trend_risk = np.clip(-slope_norm * 40.0, 0.0, 2.0)

    # ── Risk component 3: Latency increasing ─────────────────────────────────
    # Latency increase is an early sign of ischaemia / demyelination.
    # It contributes less than amplitude (capped at 1.0 vs 3.0 and 2.0).
    # ×5.0: a 9% latency shift (lat_shift=0.09) gives lat_risk = 0.45
    lat_risk = np.clip(lat_shift * 5.0, 0.0, 1.0)

    # ── Combine and pass through sigmoid ──────────────────────────────────────
    raw  = amp_risk + trend_risk + lat_risk
    # Maximum possible raw value: 3.0 + 2.0 + 1.0 = 6.0 (total crisis)
    # At baseline (all risks ≈ 0): raw ≈ 0 → sigmoid(5.5*(0-0.85)) ≈ 0.009
    # Centre at 0.85: sigmoid output = 0.5 when raw = 0.85

    nirs = 1.0 / (1.0 + np.exp(-5.5 * (raw - 0.85)))
    # k=5.5 controls steepness: higher = sharper transition from low to high NIRS

    # ── Temporal smoothing ────────────────────────────────────────────────────
    # Reduces response-to-response jitter without introducing boundary artefacts.
    # The kernel [0.10, 0.20, 0.40, 0.20, 0.10] is symmetric and sums to 1.0.
    # EDGE PADDING: We pad with edge values (not zeros) before convolving,
    # then trim back. This prevents the NIRS from falsely dipping at the
    # start/end of the trace (the bug in the original mode='same' approach).
    kernel      = np.array([0.10, 0.20, 0.40, 0.20, 0.10])
    pad         = len(kernel) // 2            # = 2
    nirs_padded = np.pad(nirs, pad, mode='edge')   # extend edges, not zeros
    nirs        = np.convolve(nirs_padded, kernel, mode='valid')[:len(features)]

    # Final clip ensures output stays in [0, 1] despite floating point
    return np.clip(nirs, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2: Print a validation report to the terminal
# ─────────────────────────────────────────────────────────────────────────────

def print_validation_report(times, amps, baseline_amp, adapt_amp,
                             feat_times, nirs, degrade_start, dt_s):
    """
    Prints key performance metrics to confirm the demo is working correctly.
    Also the summary you'd cite in the submission as proof of functionality.

    KEY METRICS EXPLAINED:
      Lead time        — how many minutes before the standard alert NeuroCast
                         fires its first advisory. This is the headline claim.
      Baseline NIRS max— the highest NIRS value during the stable period.
                         Must be < 0.05 to demonstrate high specificity
                         (no false alarms during uneventful monitoring).
      False std alerts — number of times the standard 50% criterion fires
                         during the baseline period (should be 0).
    """

    # Convert everything to minutes for readability
    t_min  = times     / 60.0
    ft_min = feat_times / 60.0

    # Find when the standard alert fires (first amplitude < 50% of nominal baseline)
    alert_mask = amps / baseline_amp < 0.50
    alert_tmin = t_min[alert_mask.argmax()] if alert_mask.any() else None

    # Find when each NIRS threshold is first crossed
    adv_mask  = nirs > 0.70
    warn_mask = nirs > 0.85
    adv_tmin  = ft_min[adv_mask.argmax()]  if adv_mask.any()  else None
    warn_tmin = ft_min[warn_mask.argmax()] if warn_mask.any() else None

    # Compute baseline NIRS max: all feature timepoints before degradation starts
    bl_mask  = feat_times < (degrade_start * dt_s)
    base_max = nirs[bl_mask].max() if bl_mask.any() else float('nan')

    print("\n" + "=" * 58)
    print("  NeuroCast Demo — Validation Report")
    print("=" * 58)
    print(f"  Case duration:       {t_min[-1]:.1f} min")
    print(f"  Degradation onset:   {degrade_start * dt_s / 60:.1f} min")
    print(f"  Amplitude rel. SD:   {amps[:degrade_start].std() / baseline_amp * 100:.1f}%"
          f"  (lit: 30–40%)")
    print(f"  Adaptive baseline:   {adapt_amp:.3f} µV  "
          f"(nominal: {baseline_amp:.1f} µV)")
    print()
    print(f"  NIRS Advisory:       {adv_tmin:.1f} min"  if adv_tmin  else "  NIRS Advisory:       —")
    print(f"  NIRS Warning:        {warn_tmin:.1f} min" if warn_tmin else "  NIRS Warning:        —")
    print(f"  Standard alert:      {alert_tmin:.1f} min" if alert_tmin else "  Standard alert:      —")
    if adv_tmin and alert_tmin:
        lead_s = (alert_tmin - adv_tmin) * 60
        print(f"  Lead time:           {alert_tmin - adv_tmin:.1f} min  ({lead_s:.0f} s)")
    print()
    print(f"  Baseline NIRS max:   {base_max:.4f}  "
          f"({'✓ < 0.05' if base_max < 0.05 else '✗ > 0.05'})")
    print(f"  False std alerts:    {alert_mask[:degrade_start].sum()}")
    print("=" * 58 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 3: Generate and save the two-panel figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_nirs_demo(times, amps, baseline_amp, adapt_amp,
                   feat_times, nirs, degrade_start, dt_s,
                   outpath=None):
    """
    Produces the two-panel figure that is the visual centrepiece of the demo.

    TOP PANEL: Raw SSEP amplitude over time
      Shows the noisy baseline, the gradual fall, and the red vertical line
      where the standard (reactive) alert fires.

    BOTTOM PANEL: NIRS score over time
      Shows the flat baseline (NIRS ≈ 0.03), the rising score as degradation
      begins, and the orange/red advisory/warning lines firing before the
      standard alert. The lead-time arrow is the key visual element.

    X-AXIS: Both panels share the same x-axis in MINUTES (not seconds).
    """

    if outpath is None:
        outpath = os.path.join(SCRIPT_DIR, 'neurocast_nirs_demo.png')

    # Decision thresholds
    ALERT_TH = 0.50   # standard IONM: 50% amplitude loss from baseline
    ADV_TH   = 0.70   # NeuroCast advisory
    WARN_TH  = 0.85   # NeuroCast warning

    # Colour palette (clinically intuitive: blue=data, red=danger, orange=caution)
    BLUE  = '#0054A6'
    RED   = '#C01C1C'
    ORG   = '#D4700A'
    GRN   = '#1A7A2E'
    LGRAY = '#F4F4F4'

    # ── Convert time axes to minutes ──────────────────────────────────────────
    t_min        = times     / 60.0
    ft_min       = feat_times / 60.0
    degrade_tmin = (degrade_start * dt_s) / 60.0   # 30 × 50 / 60 = 25.0 min

    # ── Locate event times in minutes ─────────────────────────────────────────
    alert_mask = amps / baseline_amp < ALERT_TH
    alert_tmin = t_min[alert_mask.argmax()] if alert_mask.any() else None

    adv_mask  = nirs > ADV_TH
    warn_mask = nirs > WARN_TH
    adv_tmin  = ft_min[adv_mask.argmax()]  if adv_mask.any()  else None
    warn_tmin = ft_min[warn_mask.argmax()] if warn_mask.any() else None

    # ── Create the figure with two vertically stacked panels ──────────────────
    fig = plt.figure(figsize=(13, 7.5))
    fig.patch.set_facecolor('white')
    # GridSpec: 2 rows, 1 column, tight margins
    gs  = gridspec.GridSpec(2, 1, hspace=0.10,
                            top=0.88, bottom=0.19, left=0.09, right=0.97)

    # ═════════════════════════════════════════════════════════════════════════
    # TOP PANEL: SSEP amplitude as % of nominal baseline
    # ═════════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0])

    # Convert raw µV to % of nominal baseline (3.2 µV = 100%)
    pct = (amps / baseline_amp) * 100

    # Main amplitude trace — noisy line showing realistic variability
    ax1.plot(t_min, pct, color=BLUE, lw=2.2,
             label='SSEP amplitude (% of nominal baseline)', zorder=3)

    # Horizontal dashed line at 50% — the standard IONM alert threshold
    ax1.axhline(50, color=RED, ls='--', lw=1.6,
                label='Standard alert threshold (50% amplitude loss)', zorder=2)

    # Red shading below the 50% threshold (where standard alert is active)
    ax1.fill_between(t_min, pct, 50,
                     where=(pct < 50), color=RED, alpha=0.12, zorder=1)

    # Dotted line showing the adaptive patient baseline (slightly above/below 100%)
    ax1.axhline(adapt_amp / baseline_amp * 100,
                color='gray', ls=':', lw=1.0, alpha=0.65, zorder=1,
                label=f'Adaptive patient baseline ({adapt_amp:.2f} µV)')

    # Green dotted vertical line marking when degradation begins
    ax1.axvline(degrade_tmin, color=GRN, lw=1.5, ls=':', alpha=0.80, zorder=3)
    ax1.text(degrade_tmin + 0.3, 128,
             f'Degradation onset\n(t = {degrade_tmin:.0f} min)',
             fontsize=7.5, color=GRN,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=GRN, alpha=0.85))

    # Red vertical line and label where the standard IONM alert fires
    if alert_tmin is not None:
        ax1.axvline(alert_tmin, color=RED, lw=2.0, alpha=0.75, zorder=4)
        ax1.text(alert_tmin + 0.3, 88,
                 f'Standard IONM\nalert fires\n(t = {alert_tmin:.1f} min)',
                 fontsize=8, color=RED,
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=RED, alpha=0.85))

    ax1.set_xlim(0, t_min[-1])
    ax1.set_ylim(0, 148)
    ax1.set_ylabel('SSEP Amplitude\n(% of nominal baseline)', fontsize=10)
    ax1.tick_params(labelbottom=False)   # Hide x-axis labels (shared with bottom)
    ax1.legend(loc='upper right', fontsize=8.0)
    ax1.grid(True, alpha=0.25)
    ax1.set_facecolor(LGRAY)
    ax1.text(0.6, 135,
             '← Stable baseline  (realistic variability, σ ≈ 20%) →',
             fontsize=7.5, color='gray')

    # ═════════════════════════════════════════════════════════════════════════
    # BOTTOM PANEL: NIRS score over time
    # ═════════════════════════════════════════════════════════════════════════
    # sharex=ax1 links the x-axes so zooming/panning both panels moves together
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Main NIRS trace
    ax2.plot(ft_min, nirs, color=BLUE, lw=2.5,
             label='NIRS (Neural Integrity Risk Score)', zorder=3)

    # Blue shading under the NIRS curve (visual emphasis)
    ax2.fill_between(ft_min, nirs, 0, alpha=0.13, color=BLUE, zorder=1)

    # Dashed threshold lines
    ax2.axhline(ADV_TH,  color=ORG, ls='--', lw=1.5,
                label=f'Advisory threshold (NIRS ≥ {ADV_TH})')
    ax2.axhline(WARN_TH, color=RED, ls='--', lw=1.5,
                label=f'Warning threshold  (NIRS ≥ {WARN_TH})')

    # Coloured bands between thresholds (orange = advisory zone, red = warning zone)
    ax2.fill_between(ft_min, ADV_TH, WARN_TH, color=ORG, alpha=0.07, zorder=0)
    ax2.fill_between(ft_min, WARN_TH, 1.0,    color=RED, alpha=0.07, zorder=0)

    # Green dotted line mirrors the degradation onset from the top panel
    ax2.axvline(degrade_tmin, color=GRN, lw=1.5, ls=':', alpha=0.80, zorder=3)

    # Orange vertical line — advisory threshold crossed (label moved to info strip below)
    if adv_tmin is not None:
        ax2.axvline(adv_tmin, color=ORG, lw=2.2, alpha=0.85, zorder=4)

    # Red vertical line — warning threshold crossed (label moved to info strip below)
    if warn_tmin is not None:
        ax2.axvline(warn_tmin, color=RED, lw=2.2, alpha=0.85, zorder=4)

    # Dark red dotted reference line — standard alert (mirrored from top panel)
    if alert_tmin is not None:
        ax2.axvline(alert_tmin, color='#8B0000', lw=1.6, ls=':', alpha=0.5, zorder=4)

    # Double-headed arrow showing lead-time span — text moved to info strip below
    if adv_tmin is not None and alert_tmin is not None and adv_tmin < alert_tmin:
        lead_min = alert_tmin - adv_tmin
        ax2.annotate('', xy=(alert_tmin, 0.10), xytext=(adv_tmin, 0.10),
                     arrowprops=dict(arrowstyle='<->', color='black', lw=1.6))

    ax2.set_xlim(0, t_min[-1])
    ax2.set_ylim(0, 1.08)
    ax2.set_xlabel('Time (minutes into monitored case)', fontsize=10)
    ax2.set_ylabel('NIRS Score\n(0 = safe, 1 = imminent alert)', fontsize=10)
    ax2.legend(loc='upper left', fontsize=8.0)
    ax2.grid(True, alpha=0.25)
    ax2.set_facecolor(LGRAY)

    # ── Info strip: three event boxes below both panels ───────────────────────


    strip_y = 0.075   # vertical centre of the strip in figure coords

    if adv_tmin is not None:
        fig.text(0.22, strip_y,
                 f'ADVISORY  ·  t = {adv_tmin:.1f} min',
                 ha='center', va='center', fontsize=9, color=ORG, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.45', fc='white', ec=ORG, lw=1.6),
                 zorder=11)

    if warn_tmin is not None:
        fig.text(0.50, strip_y,
                 f'WARNING  ·  t = {warn_tmin:.1f} min',
                 ha='center', va='center', fontsize=9, color=RED, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.45', fc='white', ec=RED, lw=1.6),
                 zorder=11)

    if adv_tmin is not None and alert_tmin is not None and adv_tmin < alert_tmin:
        fig.text(0.78, strip_y,
                 f'Lead time  ·  {lead_min:.1f} min ({lead_min * 60:.0f} s) before standard alert',
                 ha='center', va='center', fontsize=9, color='#444444', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.45', fc='#FFFDE7', ec='#999999', lw=1.6),
                 zorder=11)

    # Shared title spanning both panels
    fig.suptitle(
        'NeuroCast  —  Neural Integrity Risk Score (NIRS) Demonstration\n'
        'Synthetic SSEP Gradual-Degradation Event  ·  62-minute simulated case  ·  '
        'Prospective forecasting vs. reactive standard alert',
        fontsize=11.0, fontweight='bold', y=0.97
    )

    plt.savefig(outpath, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure saved: {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: Orchestrate the full pipeline
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Key parameters — change these to explore different scenarios
    DT_S          = 50.0   # seconds per averaged response (= 200 sweeps at 4 Hz)
    DEGRADE_START = 30     # which response degradation begins at (response 30 = 25 min)

    # ── Step 1: Generate the synthetic SSEP sequence ──────────────────────────
    # Returns: times (s), amps (µV), lats (ms), sweeps (waveforms),
    #          b_amp (nominal baseline amplitude), b_lat (nominal baseline latency)
    times, amps, lats, sweeps, b_amp, b_lat = build_ssep_sequence(
        n_sweeps=75, dt_s=DT_S, degrade_start=DEGRADE_START, degrade_end=38)

    # ── Step 2: Extract the 9-feature matrix ──────────────────────────────────
    # Returns: features (57×9), idx (response indices), adapt_amp, adapt_lat
    features, idx, adapt_amp, adapt_lat = extract_features(
        amps, lats, sweeps, b_amp, b_lat, window=18, adapt_n=8)

    # feat_times: the timestamp of each feature vector (in seconds)
    feat_times = times[idx]

    # ── Step 3a: Compute NIRS with rule-based proxy ───────────────────────────
    # Comment this out if using the trained TCN below
    nirs = compute_nirs_proxy(features)

    # ── Step 3b: Compute NIRS with trained TCN ────────────────────────────────
    # Requires neurocast_tcn.pt — run tcn_model.py first to generate it.
    # run_inference() loops over all 57 feature vectors one at a time,
    # feeding each as a rolling window to the model and collecting 57 scores.
    # import torch
    # from tcn_model import NeuroCastTCN, run_inference
    # model = NeuroCastTCN()
    # model.load_state_dict(
    #     torch.load(os.path.join(SCRIPT_DIR, 'neurocast_tcn.pt'),
    #                map_location="cpu"))
    # model.eval()
    # nirs = run_inference(model, features)
    # nirs is now an ndarray of shape (57,) — one NIRS score per response

    # ── Step 4: Print the validation report ───────────────────────────────────
    print_validation_report(times, amps, b_amp, adapt_amp,
                             feat_times, nirs, DEGRADE_START, DT_S)

    # ── Step 5: Generate and save the figure ──────────────────────────────────
    plot_nirs_demo(times, amps, b_amp, adapt_amp,
                   feat_times, nirs, DEGRADE_START, DT_S)
