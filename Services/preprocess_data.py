from pathlib import Path

# Import configuration
from Utilities.config import PROCESSED_DATA_PATH

# Import utility functions
from Utilities.Services.preprocess_data_utils import (
    read_uci_phishing_data,
    preprocess_kaggle_phishing,
    read_kaggle_phishing_data,
)
# Import data class
from Classes.DataToMerge import DataToMerge


def preprocess_data_service(pkl_path:Path=None) -> DataToMerge:
    """Service to load and preprocess all datasets.

    This step assumes that the raw data has been downloaded by
    :mod:`Services.extract_data` into the paths configured in
    :mod:`Utilities.config`. Optionally, it can trigger the extract
    step directly.

    Returns
    -------
    DataToMerge
        Instance containing three preprocessed dataframes:

        - ``urls_uci``: UCI phishing dataset (already feature-rich).
        - ``urls_kaggle``: Kaggle phishing URLs with engineered
          URL-based features (mirrors the notebook logic).
        - ``top_urls``: Top 1M websites dataframe (as loaded).
    """
    # Load datasets
    urls_uci = read_uci_phishing_data()
    urls_kaggle = read_kaggle_phishing_data()

    # Preprocess Kaggle phishing data (add URL-based features)
    urls_kaggle = preprocess_kaggle_phishing(urls_kaggle)

    if pkl_path:
        # Persist processed versions for downstream use
        pkl_path.mkdir(parents=True, exist_ok=True)
        # save the preprocessed datasets for downstream use
        urls_uci.to_pickle(pkl_path / "uci_phishing.pkl")
        urls_kaggle.to_pickle(pkl_path / "kaggle_phishing_preprocessed.pkl")

    dataToMerge = DataToMerge(
        urls_uci=urls_uci,
        urls_kaggle=urls_kaggle,
    )
    
    return dataToMerge


if __name__ == "__main__":
    # By default, assume extract step has already been run.
    preprocess_data_service(pkl_path=PROCESSED_DATA_PATH)
