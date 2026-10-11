#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute urban economic resilience metrics per eFUA.

Inputs:
  CSV with columns:
    - eFUA_ID
    - DATE (e.g., "mm-YYYY" or parseable date)
    - GDP_incr (observed monthly GDP increment)
    - yhat (Prophet counterfactual mean)
    - yhat_lower_smoothed, yhat_upper_smoothed (smoothed PI bounds)

Method (per eFUA):
  1) Define shock peak as the post-cutoff date maximizing (yhat_lower_smoothed - GDP_incr).
  2) Shock period: [cutoff, peak]; Recovery period: (peak, end_date].
  3) Resistance  = (area_above - area_below) / sum(|yhat|) over shock period.
  4) Recoverability = (area_above - area_below) / sum(|yhat|) over recovery period.
  5) Resilience = Resistance + Recoverability.
  6) Durations (months), ratios, and Instability_Count (crossings below lower bound).

"""

import argparse
import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="Compute resilience metrics per eFUA.")
    ap.add_argument("--input", required=True, help="Path to input CSV.")
    ap.add_argument("--output", required=True, help="Path to output CSV.")
    ap.add_argument("--cutoff", default="2020-01-01", help="Shock start cutoff date (YYYY-MM-DD).")
    ap.add_argument("--end", default="2023-12-01", help="Analysis end date (YYYY-MM-DD).")
    return ap.parse_args()


def month_diff(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((end.year - start.year) * 12 + (end.month - start.month))


def non_continuous_below_duration(df: pd.DataFrame) -> int:
    """Sum lengths (in months) of all disjoint runs where GDP_incr < yhat_lower_smoothed."""
    if df.empty:
        return 0
    period = df.copy()
    period["below"] = period["GDP_incr"] < period["yhat_lower_smoothed"]
    durations = []
    start_idx = None
    for i in range(1, len(period)):
        prev_below, curr_below = period["below"].iloc[i - 1], period["below"].iloc[i]
        if prev_below and not curr_below:
            end_idx = i - 1
            if start_idx is not None:
                durations.append(month_diff(period["DATE"].iloc[start_idx], period["DATE"].iloc[end_idx]))
            start_idx = None
        elif not prev_below and curr_below:
            start_idx = i
    if start_idx is not None:
        durations.append(month_diff(period["DATE"].iloc[start_idx], period["DATE"].iloc[-1]))
    return int(np.sum(durations)) if durations else 0


def area_metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    """Compute (area_above, area_below, denom) using yhat PI bounds and yhat (denominator)."""
    if df.empty:
        return 0.0, 0.0, 0.0
    above = (df.loc[df["GDP_incr"] > df["yhat_upper_smoothed"], "GDP_incr"] - df["yhat_upper_smoothed"]).sum()
    below = (df.loc[df["GDP_incr"] < df["yhat_lower_smoothed"], "yhat_lower_smoothed"] - df["GDP_incr"]).sum()
    denom = np.abs(df["yhat"]).sum()
    return float(above), float(below), float(denom)


def instability_count_after_cutoff(df: pd.DataFrame) -> int:
    """Count transitions from >= lower bound to < lower bound after cutoff."""
    if len(df) < 2:
        return 0
    was_above = True
    count = 0
    for i in range(1, len(df)):
        curr = df["GDP_incr"].iloc[i]
        lower = df["yhat_lower_smoothed"].iloc[i]
        if was_above and curr < lower:
            count += 1
            was_above = False
        elif curr >= lower:
            was_above = True
    return int(count)


def main():
    args = parse_args()
    df = pd.read_csv(args.input, low_memory=False)

    # Basic checks
    needed = {"eFUA_ID", "DATE", "GDP_incr", "yhat", "yhat_lower_smoothed", "yhat_upper_smoothed"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Parse and sort dates
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce", infer_datetime_format=True)
    if df["DATE"].isna().any():
        df["DATE"] = pd.to_datetime(df["DATE"], format="%m-%Y", errors="coerce")
    if df["DATE"].isna().any():
        raise ValueError("DATE parsing failed. Expect parseable dates or 'mm-YYYY'.")
    df = df.sort_values(["eFUA_ID", "DATE"])

    cutoff = pd.Timestamp(args.cutoff)
    end_date = pd.Timestamp(args.end)

    results = []
    for eid, g in df.groupby("eFUA_ID", sort=False):
        g = g.copy()
        # Use only within analysis window from cutoff to end_date for shock detection & durations
        post = g[(g["DATE"] >= cutoff) & (g["DATE"] <= end_date)].copy()

        # Identify shock peak date
        if post.empty or post[["yhat_lower_smoothed", "GDP_incr"]].isna().all().any():
            peak = None
        else:
            post["diff"] = post["yhat_lower_smoothed"] - post["GDP_incr"]
            peak_row = post.loc[post["diff"].idxmax()] if not post["diff"].isna().all() else None
            peak = peak_row["DATE"] if peak_row is not None else None

        # Defaults
        metrics = dict(
            eFUA_ID=eid,
            Resistance=np.nan,
            Recoverability=np.nan,
            Resilience=np.nan,
            Shock_Duration_Months=0,
            Shock_Below_Lower_Duration=0,
            Shock_Below_Lower_Ratio=1.0,
            Recovery_Duration_Months=0,
            Recovery_Below_Lower_Duration=0,
            Recovery_Below_Lower_Ratio=1.0,
            Instability_Count=0,
        )

        if peak is not None:
            shock = post[(post["DATE"] >= cutoff) & (post["DATE"] <= peak)].copy()
            recovery = post[(post["DATE"] > peak) & (post["DATE"] <= end_date)].copy()

            # Durations
            if not shock.empty:
                metrics["Shock_Duration_Months"] = month_diff(shock["DATE"].min(), shock["DATE"].max())
            if not recovery.empty:
                metrics["Recovery_Duration_Months"] = month_diff(recovery["DATE"].min(), recovery["DATE"].max())

            # Areas and normalized scores
            a_up_s, a_dn_s, denom_s = area_metrics(shock)
            a_up_r, a_dn_r, denom_r = area_metrics(recovery)
            metrics["Resistance"] = (a_up_s - a_dn_s) / denom_s if denom_s > 0 else np.nan
            metrics["Recoverability"] = (a_up_r - a_dn_r) / denom_r if denom_r > 0 else np.nan
            # Composite
            metrics["Resilience"] = np.nansum([metrics["Resistance"], metrics["Recoverability"]])

            # Durations below lower bound (non-continuous)
            s_dur = non_continuous_below_duration(shock)
            r_dur = non_continuous_below_duration(recovery)
            metrics["Shock_Below_Lower_Duration"] = s_dur
            metrics["Recovery_Below_Lower_Duration"] = r_dur
            metrics["Shock_Below_Lower_Ratio"] = (s_dur / metrics["Shock_Duration_Months"]) if metrics["Shock_Duration_Months"] > 0 else 1.0
            metrics["Recovery_Below_Lower_Ratio"] = (r_dur / metrics["Recovery_Duration_Months"]) if metrics["Recovery_Duration_Months"] > 0 else 1.0

            # Instability count on post-cutoff window
            metrics["Instability_Count"] = instability_count_after_cutoff(post)

        results.append(metrics)

    out = pd.DataFrame(results)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved resilience metrics to: {args.output}")


if __name__ == "__main__":
    main()
