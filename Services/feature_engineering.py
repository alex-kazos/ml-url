from pathlib import Path

import numpy as np
import pandas as pd

# Import services
from Services.merge_data import merge_data_service
from Services.preprocess_data import preprocess_data_service

# Import utility functions
from Utilities.Services.feature_engineering_utils import add_all_new_features

# Import configuration
from Utilities.config import PROCESSED_DATA_PATH

# # Columns to drop before saving the ML-ready dataset.
# - Text columns that are not numeric features.
# - `Label` is redundant with `label_binary`.
# - `TLDLegitimateProb` is derived from labels → data leakage.
COLUMNS_TO_DROP = [
    "URL",
    "Domain",
    "TLD",
    "Label",
    "TLDLegitimateProb",
]


def feature_engineering_service(df:pd.DataFrame=None,pkl_path:Path=None) -> pd.DataFrame:
    """Run feature engineering and produce an ML-ready dataset.

    1. Loads ``final_dataset.pkl`` (or ``.csv`` fallback).
    2. Adds new engineered features.
    3. Drops non-numeric / leaky columns.
    4. Replaces ``inf`` / ``NaN`` values.
    5. Saves the result to ``ml_ready_dataset.csv`` and ``.pkl``.

    Returns
    -------
    pd.DataFrame
        The ML-ready dataset.
    """
    if df is None:
        # Load data
        read_df = PROCESSED_DATA_PATH / "final_dataset.pkl"
        df = pd.read_pickle(read_df)
        print(f"Loaded dataset from {pkl_path}  (shape: {df.shape})")

    # Add new features
    df = add_all_new_features(df)
    print(f"After feature engineering: {df.shape[1]} columns")

    # Drop non-numeric and leaky columns
    # cols_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    # df.drop(columns=cols_present, inplace=True)
    # print(f"Dropped columns: {cols_present}")

    # Handle inf / NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    nan_before = df.isna().sum().sum()
    df.fillna(0, inplace=True)
    print(f"Replaced {nan_before} NaN/inf values with 0")

    if pkl_path:
        # Save
        pkl_path.mkdir(parents=True, exist_ok=True)
        out_pkl = pkl_path / "ml_ready_dataset.pkl"
        out_csv = pkl_path / "ml_ready_dataset.csv"

        df.to_pickle(out_pkl)
        df.to_csv(out_csv, index=False)

    return df


if __name__ == "__main__":
    # # this should run automatically in the pipeline.

    # process raw data to get the preprocessed datasets
    # dataToMerge = preprocess_data_service()
    # # pass data to merge service to merge and clean the data
    # merged_df = merge_data_service(dataToMerge)
    # # pass merged data to final feature engineering
    # feature_engineering_service(merged_df)

    # By default, this script assumes that the extract and preprocess steps have already been run
    feature_engineering_service(pkl_path=PROCESSED_DATA_PATH)
