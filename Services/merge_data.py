import pandas as pd
from pathlib import Path

# Import configuration
from Utilities.config import PROCESSED_DATA_PATH
from Utilities.Services.preprocess_data_utils import (
    preprocess_phishing,
)

# Import utility functions
from Utilities.Services.merge_data_utils import (
    remove_duplicates_and_nan,
    balance_dataset, keep_common_columns,
)
from Services.preprocess_data import preprocess_data_service

# Import data class
from Classes.DataToMerge import DataToMerge

def merge_data_service(dataToMerge:DataToMerge=None,pkl_path:Path=None)->pd.DataFrame:
    """
    Service to merge the preprocessed datasets, add features, and save the final dataset.

    Parameters
    ----------
    dataToMerge : DataToMerge
        Instance containing the preprocessed datasets to merge.
    pkl_path : Path, optional
        Path to save the final merged dataset. If None, uses PROCESSED_DATA_PATH from config.

    Returns
    -------
    pd.DataFrame
        The final merged, cleaned, and balanced dataset ready for feature engineering.
    """
    # extract data if not already done
    if dataToMerge is None:
        urls_uci = pd.read_pickle(PROCESSED_DATA_PATH / "uci_phishing.pkl")
        urls_kaggle = pd.read_pickle(PROCESSED_DATA_PATH / "kaggle_phishing_preprocessed.pkl")
    else:
        # Load the preprocessed datasets
        urls_uci = dataToMerge.urls_uci
        urls_kaggle = dataToMerge.urls_kaggle

    # Prepare UCI dataset
    common_columns = urls_kaggle.columns.tolist()
    urls_uci = keep_common_columns(urls_uci, common_columns)
    # Merge the datasets
    merged_df = pd.concat([urls_uci, urls_kaggle], ignore_index=True)

    # Remove duplicates and NaN values
    cleaned_df = remove_duplicates_and_nan(merged_df)

    # Balance the dataset
    balanced_df = balance_dataset(cleaned_df)

    if pkl_path:
         # Persist the final dataset
        pkl_path.mkdir(parents=True, exist_ok=True)
        # Save the final dataset
        balanced_df.to_pickle(PROCESSED_DATA_PATH / "final_dataset.pkl")

    return balanced_df


if __name__ == "__main__":
    # this should run automatically in the pipeline.
    # dataToMerge = preprocess_data_service()
    # merge_data_service(dataToMerge)

    # By default, assume preprocess step has already been run and preprocessed datasets are available.
    merge_data_service()
