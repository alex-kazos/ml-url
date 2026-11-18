from pathlib import Path

import numpy as np
import pandas as pd

# Import utility functions
from Utilities.Services.feature_engineering_utils import add_all_new_features

# Import configuration
from Utilities.config import PROCESSED_DATA_PATH


def feature_engineering_service(df:pd.DataFrame=None,pkl_path:Path=None) -> pd.DataFrame:
    """Run feature engineering and produce an ML-ready dataset.

    1. Loads ``final_dataset.pkl`` when no dataframe is passed.
    2. Adds new engineered features.
    3. Replaces ``inf`` / ``NaN`` values.
    4. Saves the result to ``ml_ready_dataset.csv`` and ``.pkl`` when
       ``pkl_path`` is given.

    Non-feature columns (``URL``, ``Domain``, ``TLD``, ``Label``,
    ``TLDLegitimateProb``) are kept on purpose: training fits the URL character
    model on ``URL``, checks ``label_binary`` against ``Label``, and drops them
    all itself (``COLUMNS_TO_DROP`` in ``Services/model_training.py``).

    Returns
    -------
    pd.DataFrame
        The ML-ready dataset.
    """
    if df is None:
        # Load data
        read_path = PROCESSED_DATA_PATH / "final_dataset.pkl"
        df = pd.read_pickle(read_path)
        print(f"Loaded dataset from {read_path}  (shape: {df.shape})")

    # Add new features
    df = add_all_new_features(df)
    print(f"After feature engineering: {df.shape[1]} columns")

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
