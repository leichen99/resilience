#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute national adjustment factors to align city-level Predicted_GDP to country totals.

Logic:
1) Read all CSVs under city and country folders.
2) Parse DATE, derive Year and Quarter.
3) For each (GID_0, DATE): Adjustment_Factor = Country_Predicted_GDP / Sum(City_Predicted_GDP>0).
4) If the denominator is zero/missing, impute with the mean of valid factors.
5) Output two CSVs: by (GID_0, Year) and by (GID_0, Year, Quarter).

"""

import argparse
import os
import glob
import numpy as np
import pandas as pd


def read_concat_csvs(folder: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    return pd.concat(dfs, ignore_index=True)


def prepare_dates(df: pd.DataFrame) -> pd.DataFrame:
    # Flexible parsing: accept "mm-YYYY" or other common formats.
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce", infer_datetime_format=True)
    if df["DATE"].isna().any():
        # Fallback to the explicit "%m-%Y" if needed.
        mask = df["DATE"].isna()
        df.loc[mask, "DATE"] = pd.to_datetime(df.loc[mask, "DATE"], format="%m-%Y", errors="coerce")
    if df["DATE"].isna().any():
        raise ValueError("Some DATE values could not be parsed. Ensure DATE is like 'mm-YYYY'.")
    df["Year"] = df["DATE"].dt.year
    df["Quarter"] = df["DATE"].dt.quarter
    return df


def compute_adjustments(city_df: pd.DataFrame, country_df: pd.DataFrame) -> pd.DataFrame:
    # Keep only positive predictions for the ratio as in the original logic
    country = country_df.loc[country_df["Predicted_GDP"] > 0, ["GID_0", "DATE", "Year", "Quarter", "Predicted_GDP"]]
    city = city_df.loc[city_df["Predicted_GDP"] > 0, ["GID_0", "DATE", "Predicted_GDP"]]

    city_sum = (
        city.groupby(["GID_0", "DATE"], as_index=False)["Predicted_GDP"]
        .sum()
        .rename(columns={"Predicted_GDP": "City_Pred_Sum"})
    )

    df = country.merge(city_sum, on=["GID_0", "DATE"], how="left")
    df["Adjustment_Factor"] = df["Predicted_GDP"] / df["City_Pred_Sum"]

    # Clean and impute with global mean of valid factors
    df["Adjustment_Factor"].replace([np.inf, -np.inf], np.nan, inplace=True)
    avg_factor = df["Adjustment_Factor"].mean(skipna=True)
    if pd.isna(avg_factor) or avg_factor <= 0:
        avg_factor = 1.0  # safe fallback
    df["Adjustment_Factor"] = df["Adjustment_Factor"].fillna(avg_factor)

    return df[["GID_0", "DATE", "Year", "Quarter", "Adjustment_Factor"]]


def main():
    ap = argparse.ArgumentParser(description="Compute adjustment factors by year and by year-quarter.")
    ap.add_argument("--city_dir", required=True, help="Folder containing city-level CSVs with Predicted_GDP.")
    ap.add_argument("--country_dir", required=True, help="Folder containing country-level CSVs with Predicted_GDP.")
    ap.add_argument("--out_year", required=True, help="Output CSV path for (GID_0, Year) averages.")
    ap.add_argument("--out_yq", required=True, help="Output CSV path for (GID_0, Year, Quarter) averages.")
    args = ap.parse_args()

    # Read data
    city = read_concat_csvs(args.city_dir)
    country = read_concat_csvs(args.country_dir)

    # Minimal required columns
    for name, df in [("city", city), ("country", country)]:
        for col in ["GID_0", "DATE", "Predicted_GDP"]:
            if col not in df.columns:
                raise KeyError(f"Missing column '{col}' in {name} data.")

    # Prepare dates and derive Year/Quarter
    city = prepare_dates(city)
    country = prepare_dates(country)

    # Compute factors at (GID_0, DATE)
    factors = compute_adjustments(city, country)

    # Aggregate by year
    by_year = factors.groupby(["GID_0", "Year"], as_index=False)["Adjustment_Factor"].mean()
    by_year.to_csv(args.out_year, index=False, encoding="utf-8-sig")

    # Aggregate by year-quarter
    by_yq = factors.groupby(["GID_0", "Year", "Quarter"], as_index=False)["Adjustment_Factor"].mean()
    by_yq.to_csv(args.out_yq, index=False, encoding="utf-8-sig")

    print(f"Saved:\n- {args.out_year}\n- {args.out_yq}")


if __name__ == "__main__":
    main()
