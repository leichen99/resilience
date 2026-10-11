#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply a trained GDP prediction model (best_model_with_params1.pkl) to new data.

This script reproduces the exact inference pipeline used during training:
- load and clean the new dataset
- feature engineering (LOG_SUM and _per_AREA features)
- standardization with the saved StandardScaler
- feature selection with saved feature indices
- prediction on log-GDP and exponentiation to original GDP scale
- write results with a Predicted_GDP column

"""

import argparse
import logging
import sys
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd


# ----------------------------- Logging config ----------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ----------------------------- Core utilities ----------------------------- #
REQUIRED_BASE_COLUMNS = ["SUM", "COUNT", "AREA"]  # minimal set required for inference


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Load data (Excel or CSV) and create LOG_SUM, dropping rows with missing essentials."""
    logging.info(f"Loading data from {file_path}")

    # Try Excel first, fall back to CSV
    try:
        df = pd.read_excel(file_path)
    except Exception:
        df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)

    # Check required columns
    missing = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)[:25]}..."
        )

    # Drop rows missing essential fields (no GDP required at inference)
    before = len(df)
    df = df.dropna(subset=REQUIRED_BASE_COLUMNS)
    after = len(df)
    if after < before:
        logging.info(f"Dropped {before - after} rows with missing {REQUIRED_BASE_COLUMNS}")

    # Create LOG_SUM
    # Guard against non-positive SUM values prior to log
    nonpos = (df["SUM"] <= 0).sum()
    if nonpos > 0:
        logging.warning(f"Found {nonpos} rows with non-positive SUM; these rows will be dropped.")
        df = df[df["SUM"] > 0]

    df["LOG_SUM"] = np.log(df["SUM"].astype(float))
    logging.info(f"Data loaded and cleaned. Final rows: {len(df)}")
    return df


def detect_bin_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Detect nightlight distribution columns used during training."""
    bin_count = [c for c in df.columns if c.startswith("Bin_") and c.endswith("_COUNT")]
    bin_sum = [c for c in df.columns if c.startswith("Bin_") and c.endswith("_SUM")]
    bin_count_pct = [c for c in df.columns if c.startswith("Bin_") and c.endswith("_COUNT_PCT")]
    bin_sum_pct = [c for c in df.columns if c.startswith("Bin_") and c.endswith("_SUM_PCT")]
    return bin_count, bin_sum, bin_count_pct, bin_sum_pct


def make_feature_list(df: pd.DataFrame) -> List[str]:
    """Reconstruct the exact feature list used in training."""
    base = ["COUNT", "AREA", "MIN", "MAX", "RANGE", "MEAN", "STD", "LOG_SUM"]
    missing_base = [c for c in base if c not in df.columns]
    if missing_base:
        logging.warning(
            f"Some base features are missing and will be skipped: {missing_base}"
        )
    base = [c for c in base if c in df.columns]

    bcount, bsum, bcount_pct, bsum_pct = detect_bin_columns(df)
    features = base + bcount + bsum + bcount_pct + bsum_pct
    if not features:
        raise ValueError(
            "No usable features detected. Check that your input contains base and/or Bin_* columns."
        )
    return features


def add_area_adjusted_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Create _per_AREA features for all features except AREA itself."""
    new_cols = {}
    if "AREA" not in df.columns:
        raise ValueError("AREA column is required to compute _per_AREA features.")
    zero_area = (df["AREA"] == 0).sum()
    if zero_area:
        logging.warning(f"Found {zero_area} rows with AREA == 0; these rows will be dropped.")
        df = df[df["AREA"] != 0]

    for f in features:
        if f != "AREA" and f in df.columns:
            new_cols[f"{f}_per_AREA"] = df[f] / df["AREA"]

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def apply_model(
    df: pd.DataFrame, model_path: str, output_path: str, pred_col: str = "Predicted_GDP"
) -> None:
    """Load model+scaler+selected_features, engineer features, predict, and save results."""
    # Load serialized model pack
    saved = joblib.load(model_path)
    model = saved["model"]
    scaler = saved["scaler"]
    selected_indices = saved["selected_features"]

    # Rebuild feature set and engineer per-area features
    features = make_feature_list(df)
    df = add_area_adjusted_features(df, features)
    derived = [f"{f}_per_AREA" for f in features if f != "AREA"]
    full_feature_order = features + derived  # must match training construction order

    # Ensure all features exist (some bases may be missing in the new data)
    missing_now = [c for c in full_feature_order if c not in df.columns]
    if missing_now:
        # If missing engineered features occur (due to missing bases), drop them consistently
        logging.warning(
            f"{len(missing_now)} expected features are missing and will be dropped. "
            f"Examples: {missing_now[:8]}"
        )
        full_feature_order = [c for c in full_feature_order if c in df.columns]

    # Transform and select features
    X = df[full_feature_order].astype(float).values
    X_scaled = scaler.transform(X)
    X_sel = X_scaled[:, selected_indices]

    # Predict log-GDP then exponentiate
    y_log = model.predict(X_sel)
    y = np.exp(y_log)

    # Write outputs
    out = df.copy()
    out[pred_col] = y

    # Optionally drop engineered features to keep the file tidy
    to_drop = [c for c in out.columns if c in full_feature_order]
    out = out.drop(columns=to_drop)

    # CSV is the default archival format
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(f"Predictions saved to: {output_path} (column: {pred_col})")


# ---------------------------------- CLI ---------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply best_model_with_params.pkl to produce Predicted_GDP for new data."
    )
    parser.add_argument(
        "--input", required=True, help="Path to the new dataset (Excel .xlsx or CSV)."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained model pack (best_model_with_params.pkl).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output CSV with predictions (will be created/overwritten).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        df = load_and_clean_data(args.input)
        apply_model(df, args.model, args.output, pred_col="Predicted_GDP")
    except Exception as e:
        logging.error(f"Failed to apply model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
