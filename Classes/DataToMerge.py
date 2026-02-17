from pandas import DataFrame

class DataToMerge:
    """Container class for preprocessed datasets ready for merging.

    Attributes
    ----------
    urls_uci : pd.DataFrame
        UCI phishing dataset (already feature-rich).
    urls_kaggle : pd.DataFrame
        Kaggle phishing URLs with engineered URL-based features.
    top_urls : pd.DataFrame
        Top 1M websites dataframe (as loaded).
    """

    def __init__(
            self,
            urls_uci: DataFrame,
            urls_kaggle: DataFrame,
    ):
        """Initialize DataToMerge with three preprocessed dataframes.

        Parameters
        ----------
        urls_uci : pd.DataFrame
            UCI phishing dataset.
        urls_kaggle : pd.DataFrame
            Kaggle phishing URLs with engineered features.
        """
        self.urls_uci = urls_uci
        self.urls_kaggle = urls_kaggle