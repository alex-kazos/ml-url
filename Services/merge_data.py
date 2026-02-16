
import pandas as pd
from Utilities.config import PROCESSED_DATA_PATH
from Utilities.Services.preprocess_data_utils import (
    preprocess_kaggle_phishing,
)
from Utilities.Services.merge_data_utils import (
    remove_duplicates_and_nan,
    balance_dataset, keep_common_columns,
)


def merge_data_service():
    """
    Service to merge the preprocessed datasets, add features, and save the final dataset.
    """
    # Load the preprocessed datasets
    urls_uci = pd.read_pickle(PROCESSED_DATA_PATH / "uci_phishing.pkl")
    urls_kaggle = pd.read_pickle(PROCESSED_DATA_PATH / "kaggle_phishing_preprocessed.pkl")

    # Prepare UCI dataset
    common_columns = urls_kaggle.columns.tolist()
    urls_uci = keep_common_columns(urls_uci, common_columns)
    # Merge the datasets
    merged_df = pd.concat([urls_uci, urls_kaggle], ignore_index=True)

    # Remove duplicates and NaN values
    cleaned_df = remove_duplicates_and_nan(merged_df)

    # Balance the dataset
    balanced_df = balance_dataset(cleaned_df)

    # Save the final dataset
    balanced_df.to_pickle(PROCESSED_DATA_PATH / "final_dataset.pkl")
    balanced_df.to_csv(PROCESSED_DATA_PATH / "final_dataset.csv", index=False)

    print("Data merging, feature engineering, and balancing complete.")
    print(f"Final dataset saved to {PROCESSED_DATA_PATH / 'final_dataset.pkl'}")
    print(f"Shape of the final dataset: {balanced_df.shape}")
    print("Class distribution:")
    print(balanced_df["Label"].value_counts())


if __name__ == "__main__":
    merge_data_service()
