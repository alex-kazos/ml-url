from Utilities.config import PROCESSED_DATA_PATH
from Utilities.Services.preprocess_data_utils import (
    read_uci_phishing_data,
    read_kaggle_top_searches,
    preprocess_kaggle_phishing,
    read_kaggle_phishing_data,
)

from Classes.DataToMerge import DataToMerge

def preprocess_data_service(run_extract: bool = False) -> DataToMerge:
    """Service to load and preprocess all datasets.

    This step assumes that the raw data has been downloaded by
    :mod:`Services.extract_data` into the paths configured in
    :mod:`Utilities.config`. Optionally, it can trigger the extract
    step directly.

    Parameters
    ----------
    run_extract : bool, optional
        If ``True``, run :func:`Services.extract_data.extract_data_service`
        before preprocessing to ensure the raw files exist.

    Returns
    -------
    DataToMerge
        Instance containing three preprocessed dataframes:

        - ``urls_uci``: UCI phishing dataset (already feature-rich).
        - ``urls_kaggle``: Kaggle phishing URLs with engineered
          URL-based features (mirrors the notebook logic).
        - ``top_urls``: Top 1M websites dataframe (as loaded).
    """
    if run_extract:
        # Import lazily to avoid circular imports at module load time.
        from Services.extract_data import extract_data_service

        extract_data_service()

    # Load datasets
    urls_uci = read_uci_phishing_data()
    urls_kaggle = read_kaggle_phishing_data()

    urls_kaggle = preprocess_kaggle_phishing(urls_kaggle)

    # Persist processed versions for downstream use
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    urls_uci.to_pickle(PROCESSED_DATA_PATH / "uci_phishing.pkl")
    urls_kaggle.to_pickle(PROCESSED_DATA_PATH / "kaggle_phishing_preprocessed.pkl")

    dataToMerge = DataToMerge(
        urls_uci=urls_uci,
        urls_kaggle=urls_kaggle,
    )
    
    return dataToMerge


if __name__ == "__main__":
    # By default, assume extract step has already been run.
    preprocess_data_service(run_extract=False)
