#!/usr/bin/env python3

import argparse
from pathlib import Path

from src.data_loader import load_nab_series
from src.evaluation import merge_anomaly_events


def run_zscore(series):
    from src.baseline_zscore import run_zscore_detector
    return run_zscore_detector(series)


def run_isolation(series):
    from src.baseline_isolation_forest import run_isolation_forest
    return run_isolation_forest(series)


def run_lstm(series):
    from src.model_lstm_autoencoder import main as run_lstm_script
    print("Running LSTM Autoencoder...")
    run_lstm_script()
    return None


def run_hybrid(series):
    from src.hybrid_detector import main as run_hybrid_script
    print("Running Hybrid Detector...")
    run_hybrid_script()
    return None


def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection CLI Tool")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["zscore", "isolation", "lstm", "hybrid"],
        help="Model to run"
    )

    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to CSV file"
    )

    args = parser.parse_args()

    file_path = Path(args.file)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    print("\n=== Running Detection ===")
    print(f"Model: {args.model}")
    print(f"File: {file_path}\n")

    series = load_nab_series(str(file_path))

    if args.model == "zscore":
        run_zscore(series)

    elif args.model == "isolation":
        run_isolation(series)

    elif args.model == "lstm":
        run_lstm(series)

    elif args.model == "hybrid":
        run_hybrid(series)

    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
