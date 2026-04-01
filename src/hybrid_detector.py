#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_nab_series
from src.features import make_features, FeatureConfig
from src.evaluation import merge_anomaly_events
from src.nab_scoring import (
    load_combined_windows,
    guess_nab_key_from_csv_path,
    load_label_windows_for_series_key,
    score_events_against_windows,
)

CSV_PATH = "data/raw/NAB/realKnownCause/ambient_temperature_system_failure.csv"
EVENT_GAP = "2h"

# Isolation Forest settings
IF_CONTAMINATION = 0.01
IF_RANDOM_STATE = 42

# LSTM settings
SEQ_LEN = 60
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
TRAIN_FRACTION = 0.60
PERCENTILE = 99.9

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = Path("reports/hybrid_detector_results.csv")
OUT_MD = Path("reports/hybrid_detector_results.md")
OUT_FIG = FIG_DIR / "12_hybrid_detector_events.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SeqDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        latent = h[-1]
        rep = latent.unsqueeze(1).repeat(1, x.size(1), 1)
        dec, _ = self.decoder(rep)
        return self.out(dec)


def make_windows(arr, seq_len):
    X = []
    for i in range(len(arr) - seq_len + 1):
        X.append(arr[i:i + seq_len])
    return np.stack(X, axis=0)


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


def severity_from_score(score, low, high):
    if score >= high:
        return "high"
    if score >= low:
        return "medium"
    return "low"


def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    series = load_nab_series(CSV_PATH)
    df = series.df.copy()

    # ----------------------------
    # Isolation Forest branch
    # ----------------------------
    cfg = FeatureConfig(
        rolling_window=24,
        ewma_span=24,
        diff_lags=(1, 2, 6, 24),
        zscore_window=48,
    )
    Xdf = make_features(df, cfg=cfg)

    X_if = Xdf.values
    X_if = (X_if - X_if.mean(axis=0)) / (X_if.std(axis=0) + 1e-9)

    if_model = IsolationForest(
        n_estimators=300,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
    )
    if_model.fit(X_if)
    if_pred = if_model.predict(X_if)
    if_flags = (if_pred == -1)

    # ----------------------------
    # LSTM branch
    # ----------------------------
    values = df["value"].astype(float).values.reshape(-1, 1)
    train_end = int(len(values) * TRAIN_FRACTION)

    scaler = StandardScaler()
    scaler.fit(values[:train_end])
    values_scaled = scaler.transform(values)

    X_all = make_windows(values_scaled, SEQ_LEN)
    ts_lstm = df.index[SEQ_LEN - 1:]
    vals_lstm = df["value"].iloc[SEQ_LEN - 1:].values

    train_windows_end = int(X_all.shape[0] * TRAIN_FRACTION)
    X_train = X_all[:train_windows_end]

    train_loader = DataLoader(SeqDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
    full_loader = DataLoader(SeqDataset(X_all), batch_size=BATCH_SIZE, shuffle=False)

    lstm_model = LSTMAutoencoder(n_features=1, hidden_size=64, num_layers=1).to(DEVICE)
    opt = torch.optim.Adam(lstm_model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    lstm_model.train()
    for _ in range(EPOCHS):
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            recon = lstm_model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()

    train_errs = reconstruction_errors(lstm_model, train_loader)
    full_errs = reconstruction_errors(lstm_model, full_loader)
    thr = np.percentile(train_errs, PERCENTILE)
    lstm_flags = full_errs > thr

    # ----------------------------
    # Align both branches
    # ----------------------------
    common_index = Xdf.index.intersection(ts_lstm)
    if_common = pd.Series(if_flags, index=Xdf.index).reindex(common_index).fillna(False).astype(bool)
    lstm_common = pd.Series(lstm_flags, index=ts_lstm).reindex(common_index).fillna(False).astype(bool)

    hybrid_flags = (if_common & lstm_common).values

    # severity from normalized LSTM reconstruction error
    err_series = pd.Series(full_errs, index=ts_lstm).reindex(common_index)
    low_thr = np.percentile(train_errs, 99.0)
    high_thr = np.percentile(train_errs, 99.9)

    events = merge_anomaly_events(common_index, hybrid_flags, gap=EVENT_GAP)

    # label scoring
    windows_dict = load_combined_windows()
    series_key = guess_nab_key_from_csv_path(series.path, windows_dict=windows_dict)
    label_windows = load_label_windows_for_series_key(series_key, windows_dict=windows_dict)
    scoring = score_events_against_windows(events, label_windows)

    # build event table
    rows = []
    for start, end in events:
        event_err = err_series[(err_series.index >= start) & (err_series.index <= end)]
        peak_score = float(event_err.max()) if len(event_err) else 0.0
        severity = severity_from_score(peak_score, low_thr, high_thr)
        rows.append(
            {
                "event_start": start,
                "event_end": end,
                "peak_reconstruction_error": round(peak_score, 6),
                "severity": severity,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    md_lines = [
        "# Hybrid Detector Results",
        "",
        f"Series: `{series_key}`",
        "",
        "## Event-Level Performance",
        "",
        f"- TP events: {int(scoring['tp_events'])}",
        f"- FP events: {int(scoring['fp_events'])}",
        f"- FN windows: {int(scoring['fn_windows'])}",
        f"- Precision: {scoring['precision']:.3f}",
        f"- Recall: {scoring['recall']:.3f}",
        "",
        "## Detected Hybrid Events",
        "",
        "| Event Start | Event End | Peak Reconstruction Error | Severity |",
        "|---|---|---:|---|",
    ]

    for _, r in out_df.iterrows():
        md_lines.append(
            f"| {r['event_start']} | {r['event_end']} | {r['peak_reconstruction_error']:.6f} | {r['severity']} |"
        )

    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    # figure
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(common_index, df.reindex(common_index)["value"].values, label="Signal")
    plt.scatter(
        common_index[hybrid_flags],
        df.reindex(common_index)["value"].values[hybrid_flags],
        s=18,
        label="Hybrid anomaly"
    )
    for start, end in events:
        plt.axvspan(start, end, alpha=0.2)
    plt.title("Hybrid Detector (Isolation Forest + LSTM Autoencoder)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200)
    plt.close()

    print("Hybrid detector completed.")
    print(f"Series: {series_key}")
    print(f"Events: {len(events)}")
    print(f"Precision: {scoring['precision']:.3f}")
    print(f"Recall: {scoring['recall']:.3f}")
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_MD}")
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    main()
