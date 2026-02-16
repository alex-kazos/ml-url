
import pandas as pd


def keep_common_columns(df: pd.DataFrame, common_cols: list) -> pd.DataFrame:
    """
    Keeps only the common columns in the dataframe.
    """
    for col in common_cols:
        if col not in df.columns:
            df[col] = 0 # or some other default value
    return df[common_cols].copy()



def merge_datasets(
    df_uci: pd.DataFrame, df_kaggle: pd.DataFrame, df_top1m: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges the three datasets into one, aligning their columns.
    """
    # The UCI dataset has a different structure.
    # We'll select common columns and rename them for consistency.
    # For this example, we'll focus on URL and label.
    uci_renamed = df_uci.rename(columns={"label": "Label"})
    uci_simplified = uci_renamed[["url", "Label"]].copy()
    uci_simplified.rename(columns={"url": "URL"}, inplace=True)


    # The Kaggle and Top1M datasets already have engineered features.
    # We can select a common subset of columns to merge.
    common_cols = [
        "URL",
        "Label",
        "URLLength",
        "Domain",
        "IsHTTPS",
        "HasTitle",
        "HasFavicon",
        "DomainLength",
        "IsDomainIP",
        "NumSubdomains",
        "TLD",
        "TLDLength",
    ]
    
    # Ensure all dataframes have the common columns
    # For UCI, we will have to add the missing columns with default values
    for col in common_cols:
        if col not in uci_simplified.columns:
            uci_simplified[col] = 0 # or some other default value


    merged_df = pd.concat(
        [uci_simplified[common_cols], df_kaggle[common_cols], df_top1m[common_cols]],
        ignore_index=True,
    )
    return merged_df


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

