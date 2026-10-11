#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute counterfactual GDP trajectories with Prophet (per eFUA).

Logic:
- Read a panel file with columns: eFUA_ID, DATE (e.g., "mm-YYYY"), GDP_incr.
- Smooth GDP_incr with a 3-month rolling mean within each eFUA.
- For each eFUA, train Prophet **only on dates before a cutoff** (default: 2020-01-01)
  to obtain a counterfactual path (no COVID shock).
- Forecast a user-defined number of months (default: 42).
- Save concatenated forecasts and per-eFUA metrics to CSV.

"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


def parse_args():
    ap = argparse.ArgumentParser(description="Compute counterfactual GDP with Prophet per eFUA.")
    ap.add_argument("--input", required=True, help="Path to input CSV with eFUA panel (eFUA_ID, DATE, GDP_incr).")
    ap.add_argument("--out_forecast", required=True, help="Output CSV path for concatenated forecasts.")
    ap.add_argument("--out_metrics", required=True, help="Output CSV path for metrics by eFUA.")
    ap.add_argument("--cutoff", default="2020-01-01", help="Training cutoff date (YYYY-MM-DD), inclusive-exclusive.")
    ap.add_argument("--horizon", type=int, default=42, help="Forecast horizon in months.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="Parallel workers.")
    return ap.parse_args()


def prepare_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "DATE" not in df.columns or "eFUA_ID" not in df.columns or "GDP_incr" not in df.columns:
        raise KeyError("Input must contain columns: eFUA_ID, DATE, GDP_incr.")
    # Robust date parsing: try generic first; fallback to "%m-%Y"
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce", infer_datetime_format=True)
    mask = df["DATE"].isna()
    if mask.any():
        df.loc[mask, "DATE"] = pd.to_datetime(df.loc[mask, "DATE"], format="%m-%Y", errors="coerce")
    if df["DATE"].isna().any():
        raise ValueError("Some DATE values could not be parsed. Expect 'mm-YYYY' or parseable dates.")
    # 3-month rolling mean within eFUA
    df = df.sort_values(["eFUA_ID", "DATE"])
    df["GDP_incr_smoothed"] = (
        df.groupby("eFUA_ID")["GDP_incr"].transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )
    return df


def fit_predict_one(city_df: pd.DataFrame, cutoff: pd.Timestamp, horizon: int):
    city_df = city_df.sort_values("DATE")
    train = city_df[city_df["DATE"] < cutoff]
    if len(train) < 6:
        # too short for reliable Prophet fit
        return None, None

    df_train = train[["DATE", "GDP_incr_smoothed"]].rename(columns={"DATE": "ds", "GDP_incr_smoothed": "y"})
    # Prophet with conservative changepoints; additive seasonality aligns with smoothed signal
    model = Prophet(seasonality_mode="additive", changepoint_prior_scale=0.01, interval_width=0.90)
    model.fit(df_train)

    future = model.make_future_dataframe(periods=horizon, freq="M")
    forecast = model.predict(future)

    # Evaluate on in-sample (training period)
    fc_train = forecast.iloc[: len(df_train)]
    actual = df_train["y"].values
    yhat = fc_train["yhat"].values
    lower = fc_train["yhat_lower"].values
    upper = fc_train["yhat_upper"].values

    mae = mean_absolute_error(actual, yhat)
    rmse = np.sqrt(mean_squared_error(actual, yhat))
    picp = float(((actual >= lower) & (actual <= upper)).mean())
    mp iw = float((upper - lower).mean())
    rrmse = float(rmse / np.maximum(1e-8, actual.mean()))
    nrmse = float(rmse / np.maximum(1e-8, (actual.max() - actual.min())))

    # Optional smoothing of PI for visualization stability
    forecast["yhat_lower_smoothed"] = forecast["yhat_lower"].rolling(window=12, min_periods=1).mean()
    forecast["yhat_upper_smoothed"] = forecast["yhat_upper"].rolling(window=12, min_periods=1).mean()
    return forecast, {"mae": mae, "rmse": rmse, "picp": picp, "mpiw": mpiw, "rrmse": rrmse, "nrmse": nrmse}


def main():
    args = parse_args()
    df = prepare_data(args.input)
    cutoff_ts = pd.Timestamp(args.cutoff)

    results = []
    metrics = []

    ids = df["eFUA_ID"].drop_duplicates().tolist()

    def task(eid):
        cdf = df[df["eFUA_ID"] == eid]
        fc, mt = fit_predict_one(cdf, cutoff_ts, args.horizon)
        if fc is None:
            return None
        fc = fc.copy()
        fc["eFUA_ID"] = eid
        return fc, {"eFUA_ID": eid, **mt}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(task, eid): eid for eid in ids}
        for fut in as_completed(futures):
            eid = futures[fut]
            try:
                out = fut.result()
                if out is None:
                    continue
                fc, mt = out
                results.append(fc)
                metrics.append(mt)
            except Exception as e:
                # Keep going even if one ID fails
                print(f"[WARN] eFUA_ID {eid} failed: {e}")

    if not results:
        raise RuntimeError("No forecasts were produced. Check data and cutoff.")

    # Concatenate and save
    fc_all = pd.concat(results, ignore_index=True)
    # Keep key fields concise
    keep_cols = [
        "eFUA_ID", "ds", "yhat", "yhat_lower", "yhat_upper",
        "yhat_lower_smoothed", "yhat_upper_smoothed"
    ]
    keep_cols = [c for c in keep_cols if c in fc_all.columns]
    fc_all = fc_all[keep_cols]
    fc_all.rename(columns={"ds": "DATE", "yhat": "GDP_incr"}, inplace=True)
    fc_all.to_csv(args.out_forecast, index=False, encoding="utf-8-sig")

    mt_all = pd.DataFrame(metrics)
    mt_all.to_csv(args.out_metrics, index=False, encoding="utf-8-sig")

    print(f"Saved forecast: {args.out_forecast}")
    print(f"Saved metrics:  {args.out_metrics}")


if __name__ == "__main__":
    main()
