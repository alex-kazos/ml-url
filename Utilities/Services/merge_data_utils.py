
import pandas as pd


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
    merged_urls = merged_df["URL"].astype(str)
    uci_urls = set(urls_uci["URL"].astype(str)) if "URL" in urls_uci else set()
    kaggle_urls = set(urls_kaggle["URL"].astype(str)) if "URL" in urls_kaggle else set()
    return {
        "rows_uci_raw": int(len(urls_uci)),
        "rows_kaggle_raw": int(len(urls_kaggle)),
        "rows_after_merge": int(len(merged_df)),
        "rows_from_uci": int(merged_urls.isin(uci_urls).sum()),
        "rows_from_kaggle": int(merged_urls.isin(kaggle_urls).sum()),
    }
