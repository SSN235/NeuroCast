"""
tcn_model.py
NeuroCast — Temporal Convolutional Network (TCN) for NIRS prediction.

Architecture: 5 layers of causal dilated convolutions (dilation 1,2,4,8,16),
kernel size 3, 32 hidden channels, sigmoid output ∈ (0, 1).
Causal constraint ensures no future data leakage (Bai et al. 2018).

Label: y = 1 if the standard 50% alert fires within the next 3 responses
       (3 × 50 s = 150 s ≈ 2.5 min look-ahead). y = 0 otherwise.

Training: 120 synthetic sequences, randomised degradation onset (response
20–35) and span (5–12 responses = 4.2–10 min).

Reference: Bai S, Kolter JZ, Koltun V (2018). arXiv:1803.01271.
"""

# ─────────────────────────────────────────────────────────────────────────────
# DISCLAIMER — READ BEFORE RUNNING
# ─────────────────────────────────────────────────────────────────────────────
# This file is NOT used in the primary demo (nirs_demo.py).
#
# The demo figure and all reported metrics (4.2 min lead time, baseline
# NIRS max 0.0328) are produced by the rule-based proxy in nirs_demo.py,
# which encodes the same clinical logic the TCN is designed to learn.
#
# This file exists to demonstrate three things:
#   1. The production TCN architecture is fully specified and implemented
#      (NeuroCastTCN class — causal dilated convolutions, 5 layers, sigmoid output)
#   2. A complete training pipeline is in place and runs end-to-end on
#      synthetic data (train_tcn_on_synthetic)
#   3. A rolling inference helper (run_inference) correctly produces one
#      NIRS score per averaged response — matching the output shape the
#      rest of the pipeline expects
#
# WHY NOT USE THE TCN FOR THE DEMO?
#   The TCN is trained here on synthetic data only, with a 4.3% positive
#   class rate (319 positive vs 7121 negative samples). This imbalance
#   causes the model to produce a slightly elevated baseline NIRS (~0.13)
#   rather than the near-zero baseline the proxy achieves. The fix —
#   weighted loss via BCEWithLogitsLoss(pos_weight=22.3) — requires real
#   labelled OR data to validate properly.
#
# NEXT MILESTONE:
#   Train on retrospective de-identified IONM data via IRB/DUA with a
#   clinical partner (target: Inova Health System). That dataset provides
#   both the class balance and the signal diversity needed for the TCN to
#   outperform the rule-based proxy in production.
#
# To run the demo with the TCN instead of the proxy, see the commented
# block labelled "Step 3b" in nirs_demo.py.
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DT_S               = 50.0   # must match generate_ssep.py
LOOKAHEAD_RESPONSES = 3      # 3 × 50 s = 150 s ≈ 2.5 min look-ahead


class CausalConv1d(nn.Module):
    """1D convolution with left-side padding only (causal constraint)."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=self.pad)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out


class TCNBlock(nn.Module):
    """Residual TCN block: 2× CausalConv → ReLU → Dropout + skip connection."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.15):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch,  out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(dropout)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        residual = x if self.skip is None else self.skip(x)
        out = self.drop(self.relu(self.conv1(x)))
        out = self.drop(self.relu(self.conv2(out)))
        return self.relu(out + residual)


class NeuroCastTCN(nn.Module):
    """
    NeuroCast TCN.
    Input:  (batch, n_features=9, timesteps)
    Output: (batch,)  — scalar NIRS ∈ (0, 1)
    """
    def __init__(self, n_features=9, n_hidden=32,
                 kernel_size=3, n_layers=5, dropout=0.15):
        super().__init__()
        layers, in_ch = [], n_features
        for d in [2**i for i in range(n_layers)]:
            layers.append(TCNBlock(in_ch, n_hidden, kernel_size, d, dropout))
            in_ch = n_hidden
        self.tcn     = nn.Sequential(*layers)
        self.head    = nn.Linear(n_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.tcn(x)
        out = self.head(out[:, :, -1])
        return self.sigmoid(out).squeeze(-1)


def train_tcn_on_synthetic(n_sequences=120, epochs=60,
                           lr=1e-3, seed=42, verbose=True):
    """
    Train the TCN on synthetic SSEP sequences with randomised degradation timing.
    Saves weights to neurocast_tcn.pt in the script directory.
    """
    from generate_ssep import build_ssep_sequence
    from features import extract_features

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    X_all, y_all = [], []

    for k in range(n_sequences):
        ds   = int(rng.integers(20, 36))
        de   = ds + int(rng.integers(5, 13))

        _, amps, lats, sweeps, b_amp, _ = build_ssep_sequence(
            n_sweeps=80, dt_s=DT_S, degrade_start=ds, degrade_end=de, seed=k)
        feats, idx, *_ = extract_features(amps, lats, sweeps, b_amp, 40.0,
                                          window=18, adapt_n=8)

        below      = amps / b_amp < 0.5
        alert_resp = int(below.argmax()) if below.any() else len(amps) + 999

        for i, resp_i in enumerate(idx):
            steps  = alert_resp - resp_i
            label  = 1.0 if 0 < steps <= LOOKAHEAD_RESPONSES else 0.0
            start  = max(0, i - 17)
            X_all.append(feats[start:i + 1].T)
            y_all.append(label)

    max_len = max(x.shape[1] for x in X_all)
    X_pad   = np.zeros((len(X_all), 9, max_len), dtype=np.float32)
    for i, x in enumerate(X_all):
        X_pad[i, :, -x.shape[1]:] = x
    y_arr = np.array(y_all, dtype=np.float32)

    if verbose:
        pos = int(y_arr.sum())
        print(f"Training set: {len(y_arr)} samples  "
              f"({pos} positive, {len(y_arr) - pos} negative)")

    X_t, y_t   = torch.from_numpy(X_pad), torch.from_numpy(y_arr)
    model       = NeuroCastTCN()
    optim       = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn     = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_t))
        total_loss, n_batches = 0.0, 0
        for i in range(0, len(X_t), 32):
            xb, yb = X_t[perm[i:i+32]], y_t[perm[i:i+32]]
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); optim.step()
            total_loss += loss.item(); n_batches += 1
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}  loss = {total_loss/n_batches:.4f}")

    save_path = os.path.join(SCRIPT_DIR, 'neurocast_tcn.pt')
    torch.save(model.state_dict(), save_path)
    if verbose:
        print(f"Model saved → {save_path}")
    return model


def run_inference(model, features):
    """
    Rolling inference — one NIRS score per averaged response.
    Returns ndarray (n,) matching the shape expected by nirs_demo.py.
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(len(features)):
            start  = max(0, i - 17)
            window = features[start:i + 1].T
            if window.shape[1] < 18:
                pad    = np.zeros((9, 18 - window.shape[1]), dtype=np.float32)
                window = np.concatenate([pad, window], axis=1)
            scores.append(model(torch.from_numpy(window[np.newaxis])).item())
    return np.array(scores, dtype=np.float32)


if __name__ == '__main__':
    print("Training NeuroCast TCN on synthetic SSEP data...")
    print(f"Label horizon: {LOOKAHEAD_RESPONSES} responses × {DT_S:.0f} s "
          f"= {LOOKAHEAD_RESPONSES * DT_S:.0f} s ≈ "
          f"{LOOKAHEAD_RESPONSES * DT_S / 60:.1f} min look-ahead\n")

    model = train_tcn_on_synthetic(n_sequences=120, epochs=60)
    print("Training complete.\n")

    dummy = torch.zeros(1, 9, 18)
    baseline_input = torch.zeros(1, 9, 18)
    baseline_input[0, 0, :] = 1.0
    with torch.no_grad():
        out_zero     = model(dummy).item()
        out_baseline = model(baseline_input).item()

    print(f"Sanity check — zero input NIRS:     {out_zero:.4f}  (expect low)")
    print(f"Sanity check — baseline amp NIRS:   {out_baseline:.4f}  (expect < 0.10)")
