from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Config
# ----------------------------
FILE = "ambient_temperature_system_failure.csv"
SERIES_DIR = Path("data/raw/NAB/realKnownCause")
# SERIES_DIR = Path("data/raw/NAB/realAWSCloudwatch")

SEQ_LEN = 60
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3

TRAIN_FRACTION = 0.60
PERCENTILE = float(sys.argv[1]) if len(sys.argv) > 1 else 99.9
MERGE_GAP = "2h"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_SIGNAL = FIG_DIR / "06_lstm_ae_anomalies.png"
OUT_ERROR = FIG_DIR / "07_lstm_ae_error.png"


# ----------------------------
# Helpers
# ----------------------------
def make_windows(arr: np.ndarray, seq_len: int) -> np.ndarray:
    X = []
    for i in range(len(arr) - seq_len + 1):
        X.append(arr[i : i + seq_len])
    return np.stack(X, axis=0)


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features=1, hidden_size=64, num_layers=1):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        latent = h[-1]
        repeated = latent.unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(repeated)
        out = self.output_layer(dec_out)
        return out


def reconstruction_errors(model, loader):
    model.eval()
    errs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            mse = torch.mean((recon - batch) ** 2, dim=(1, 2))
            errs.append(mse.detach().cpu().numpy())
    return np.concatenate(errs, axis=0)


def merge_anomaly_events(timestamps: pd.Series, flags: np.ndarray, gap: str = "2h"):
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    flagged = ts[flags].reset_index(drop=True)

    if len(flagged) == 0:
        return []

    gap_td = pd.Timedelta(gap)
    events = []
    start = flagged.iloc[0]
    prev = flagged.iloc[0]

    for t in flagged.iloc[1:]:
        if (t - prev) <= gap_td:
            prev = t
        else:
            events.append((start, prev))
            start = t
            prev = t

    events.append((start, prev))
    return events


# ----------------------------
# Load & preprocess
# ----------------------------
file_path = SERIES_DIR / FILE
df = pd.read_csv(file_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

values = df["value"].astype(float).values.reshape(-1, 1)

train_end = int(len(values) * TRAIN_FRACTION)
scaler = StandardScaler()
scaler.fit(values[:train_end])
values_scaled = scaler.transform(values)

X_all = make_windows(values_scaled, SEQ_LEN)
ts_win = df["timestamp"].iloc[SEQ_LEN - 1 :].reset_index(drop=True)
val_win = df["value"].iloc[SEQ_LEN - 1 :].reset_index(drop=True)

train_windows_end = int(X_all.shape[0] * TRAIN_FRACTION)
X_train = X_all[:train_windows_end]

train_loader = DataLoader(SeqDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
full_loader = DataLoader(SeqDataset(X_all), batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# Train model
# ----------------------------
model = LSTMAutoencoder(n_features=1, hidden_size=64, num_layers=1).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

model.train()
for epoch in range(1, EPOCHS + 1):
    epoch_losses = []
    for batch in train_loader:
        batch = batch.to(DEVICE)
        opt.zero_grad()
        recon = model(batch)
        loss = loss_fn(recon, batch)
        loss.backward()
        opt.step()
        epoch_losses.append(loss.item())

    print(f"Epoch {epoch}/{EPOCHS} | loss={np.mean(epoch_losses):.6f}")

# ----------------------------
# Score anomalies
# ----------------------------
train_errs = reconstruction_errors(model, train_loader)
full_errs = reconstruction_errors(model, full_loader)

thr = np.percentile(train_errs, PERCENTILE)
flags = full_errs > thr
events = merge_anomaly_events(ts_win, flags, gap=MERGE_GAP)

print(f"Loaded: {file_path} | rows={len(df):,}")
print(f"Train windows: {len(train_errs):,}")
print(f"All windows:   {len(full_errs):,}")
print(f"Threshold: {thr:.6f} (train percentile={PERCENTILE})")
print(f"Detected anomalies (windows): {int(flags.sum())}")
print(f"Merged anomaly events: {len(events)} (merge gap={MERGE_GAP})")
print(f"Device: {DEVICE}")

# ----------------------------
# Plot 1: Signal + anomaly markers
# ----------------------------
plt.figure()
plt.plot(ts_win, val_win, label="Signal")

plt.scatter(
    ts_win[flags],
    val_win[flags],
    s=12,
    label="LSTM AE anomaly windows"
)

for (start, end) in events:
    plt.axvspan(start, end, alpha=0.2)

plt.title(f"LSTM Autoencoder Anomalies: {SERIES_DIR.name}/{FILE}")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_SIGNAL, dpi=200)
plt.close()

# ----------------------------
# Plot 2: Error + threshold
# ----------------------------
plt.figure()
plt.plot(ts_win, full_errs, label="Reconstruction Error")
plt.axhline(thr, linestyle="--", label=f"Threshold (p={PERCENTILE})")
plt.title("LSTM Autoencoder Reconstruction Error")
plt.xlabel("Time")
plt.ylabel("Error (MSE)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_ERROR, dpi=200)
plt.close()

print(f"Saved → {OUT_SIGNAL}")
print(f"Saved → {OUT_ERROR}")

if events:
    print("First anomaly events:")
    for s, e in events[:5]:
        print(f" - {s} to {e}")
