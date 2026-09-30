
import pandas as pd

from Utilities.Services.preprocess_data_utils import normalise_url, preprocess_phishing


def normalise_uci(df: pd.DataFrame) -> pd.DataFrame:
    """Bring the UCI dataset into the Kaggle schema.

    UCI ships a lowercase ``label`` where 1 = legitimate and 0 = phishing, and
    its own page-level features that cannot be computed from a URL at
    inference time. Keep only the URL, map the label to Kaggle's "good"/"bad",
    and derive the URL features the same way as for Kaggle rows.
    """
    if "Label" in df.columns:
        return df
    uci = pd.DataFrame(
        {
            "URL": df["URL"].astype(str),
            "Label": df["label"].map({1: "good", 0: "bad"}),
        }
    ).dropna(subset=["Label"])
    return preprocess_phishing(uci)


def keep_common_columns(df: pd.DataFrame, common_cols: list) -> pd.DataFrame:
    """
    Keeps only the common columns in the dataframe.
    """
    for col in common_cols:
        if col not in df.columns:
            df[col] = 0 # or some other default value
    return df[common_cols].copy()


def remove_duplicates_and_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate URLs and rows with NaN values.
    """
    df.drop_duplicates(subset=["URL"], keep="first", inplace=True)
    df.dropna(inplace=True)
    return df


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balances the dataset between phishing and safe URLs.
    """
    phishing_df = df[df["Label"] == "bad"]
    safe_df = df[df["Label"] == "good"]

    # Undersample the majority class (safe URLs)
    if len(safe_df) > len(phishing_df):
        safe_df = safe_df.sample(n=len(phishing_df), random_state=42)

    balanced_df = pd.concat([phishing_df, safe_df], ignore_index=True)
    return balanced_df


def summarise_sources(
    urls_uci: pd.DataFrame,
    urls_kaggle: pd.DataFrame,
    merged_df: pd.DataFrame,
) -> dict:
    """Count how many rows each source contributes before and after merging.

    Logged to MLflow with every training run, so a source that silently drops
    out of the training data shows up as a zero instead of going unnoticed.
    """
    def _urls(df: pd.DataFrame) -> pd.Series:
        return df["URL"].astype(str).map(normalise_url) if "URL" in df else pd.Series(dtype=str)

    merged_urls = _urls(merged_df)
    uci_urls = set(_urls(urls_uci))
    kaggle_urls = set(_urls(urls_kaggle))
    return {
        "rows_uci_raw": int(len(urls_uci)),
        "rows_kaggle_raw": int(len(urls_kaggle)),
        "rows_after_merge": int(len(merged_df)),
        "rows_from_uci": int(merged_urls.isin(uci_urls).sum()),
        "rows_from_kaggle": int(merged_urls.isin(kaggle_urls).sum()),
    }
