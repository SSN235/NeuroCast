# NeuroCast

**Predicting neural injury before it happens: a real-time forecasting layer for intraoperative neuromonitoring.**

Every year, roughly 1.5 million spinal surgeries are performed in the U.S., and somewhere between 0.3% and 5% of them result in neurological injury. The tools surgeons rely on to catch that injury in progress, intraoperative neuromonitoring (IONM) systems, are purely reactive. They only sound an alarm once nerve signal amplitude has already dropped 50% or latency has already climbed 10%. By then, the damage may already be underway, and the surgeon's window to fix it (relax a retractor, restore perfusion, ease off distraction) may already be closing.

NeuroCast is a software layer that sits on top of existing IONM hardware and tries to see that drop coming, minutes before it crosses the standard alert threshold.

## How it works

NeuroCast doesn't replace the monitoring equipment already in the OR, it listens to it. It pulls SSEP and MEP waveform data straight off a Natus Xltek amplifier through its SDK, extracts nine features from every averaged response (smoothed amplitude, latency shift, trend slope, variance, and five wavelet sub-band energies), and feeds a rolling window of those features into a small temporal convolutional network (TCN).

The output is a single number: the **Neural Integrity Risk Score (NIRS)**, from 0 (all clear) to 1 (injury imminent). An advisory fires at NIRS >= 0.70, a warning at NIRS >= 0.85. The standard 50%-amplitude / 10%-latency alert is never touched, it keeps running in parallel as the fallback it's always been.

![NeuroCast system architecture](images/neurocast_architecture.png)

## Does it actually work?

There's no clinical data yet, that's the honest next step, but the pipeline itself is fully built and has been validated on physiologically realistic synthetic SSEP data (waveform parameters from Chiappa 2004 and Nuwer 1995).

In a simulated 62-minute surgical case with a gradual degradation event, NIRS crossed the advisory threshold **4.2 minutes before** the standard alert fired, and stayed below 0.05 for the entire 25-minute stable baseline, no false alarms while nothing was wrong.

![NIRS demonstration on a simulated degradation event](images/neurocast_nirs_demo.png)

Across 50 additional simulated events with randomized onset, ramp speed, and depth, the median lead time was 1.7 minutes (IQR 0.8 to 3.1). Specificity holds near 100% up through the noise levels seen in real averaged-response SSEP data, and only degrades once variability climbs well past what's clinically typical.

![Population-level validation across simulated cases](images/neurocast_population_validation.png)

**One important caveat:** the results above come from a rule-based proxy scorer, not the trained TCN. The TCN architecture (5 causal dilated conv layers, dilations 1, 2, 4, 8, 16) is fully implemented and trains end-to-end on synthetic data, but with only a 4.3% positive-class rate in simulation, it needs real labeled OR data, not more synthetic sequences, to actually outperform the proxy. That's the next milestone: training on de-identified retrospective IONM data through an IRB/DUA agreement with a clinical partner.

## The pipeline, in code

```
generate_ssep.py   ->   features.py   ->   nirs_demo.py / tcn_model.py
(synthetic waveforms)   (9 features)       (NIRS score + plots)
```

- **`generate_ssep.py`**: simulates the averaged SSEP response stream an amplifier like the Xltek would produce, including a realistic gradual-degradation event and inter-response noise.
- **`features.py`**: rolls an 18-response (15-minute) window over the stream and extracts the 9 features NeuroCast runs its scoring on.
- **`tcn_model.py`**: the production model architecture and training loop (PyTorch), plus the reasoning for why it isn't the one generating the demo numbers yet.
- **`nirs_demo.py`**: the rule-based proxy scorer used for the current validation results, and the plotting code behind the figures above.

## Why this doesn't exist already

A patent and literature search (USPTO, ClinicalTrials.gov, FDA device database, PitchBook, Crunchbase) turned up plenty of reactive threshold-alert IONM systems and some retrospective classification work (Cofano et al. 2024 used a Random Forest to identify MEP muscles after the fact, for example), but nothing that forecasts a threshold breach before it happens. NeuroCast is aiming to be the first.

## Regulatory path

NeuroCast is being developed toward De Novo classification as a Class II SaMD (advisory-only, no automated surgical action), with Viz.ai's ContaCT stroke-alert system (De Novo DEN170073) as the closest precedent. Estimated timeline: 24 to 36 months.

## References

Key sources behind the modeling assumptions: Bai, Kolter & Koltun (2018) on TCNs; Chiappa (2004) and Nuwer et al. (1995) on evoked-potential waveform standards; Wiedemayer (2002) and MacDonald et al. (2019) on SSEP variability in the OR; Cofano et al. (2024) and Hou et al. (2022) on prior ML-for-IONM work. Full list in the project writeup.

---

*This is an early-stage technical feasibility project. NIRS results shown here are generated from synthetic data using a rule-based proxy standing in for the trained model. They demonstrate that the pipeline and alert logic work end-to-end, not clinical predictive accuracy.*
