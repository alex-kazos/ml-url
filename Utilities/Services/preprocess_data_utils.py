from collections import Counter

import numpy as np
import pandas as pd

# Import configuration
from Utilities.config import (
    UCI_PHISHING_FILE_PATH,
    KAGGLE_PHISHING_FILE_PATH,
    KAGGLE_TOP_SEARCHES_FILE_PATH,
)


def read_uci_phishing_data() -> pd.DataFrame:
    """Read phishing URL data from UCI pickle file.

    Returns
    -------
    pd.DataFrame
        Phishing URL data from UCI repository.
    """

    df = pd.read_pickle(UCI_PHISHING_FILE_PATH)
    return df


def read_kaggle_phishing_data() -> pd.DataFrame:
    """Read Kaggle phishing data from CSV file.

    Returns
    -------
    pd.DataFrame
        Phishing site URLs data from Kaggle.
    """

    df = pd.read_csv(KAGGLE_PHISHING_FILE_PATH)
    return df


def read_kaggle_top_searches() -> pd.DataFrame:
    """Read Kaggle top searches (Top 1M websites) data from CSV file.

    Returns
    -------
    pd.DataFrame
        Top 1M websites data from Kaggle.
    """

    df = pd.read_csv(KAGGLE_TOP_SEARCHES_FILE_PATH)
    return df


def _char_continuation_rate(u: str) -> float:
    """Compute character continuation rate for a URL string.

    This is a direct functionalisation of the logic used in the
    `Notebook/explore_data.ipynb` notebook.
    """
    s = str(u)
    if len(s) < 2:
        return 0.0

    def cat(ch: str) -> str:
        if ch.isalpha():
            return "A"
        if ch.isdigit():
            return "D"
        return "S"  # specials

    prev = cat(s[0])
    same, total = 0, 0
    for ch in s[1:]:
        c = cat(ch)
        if c == prev:
            same += 1
        total += 1
        prev = c

    return same / total if total else 0.0


def _add_basic_url_parts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Add basic URL-derived columns (length, domain, protocol, etc.).

    Returns the mutated dataframe together with frequently reused
    `url_s` (stringified URLs) and `lower_url` (lower-cased URLs).
    """
    # Work on a string view of the URL column so we can reuse it
    url_s = df["URL"].astype(str)

    # Total character length of each URL
    df["URLLength"] = url_s.apply(len)

    # Crude domain extraction: take the 3rd slash-separated component if present
    df["Domain"] = url_s.apply(
        lambda x: x.split("/")[2] if len(x.split("/")) > 2 else ""
    )

    # HTTPS flag based on scheme prefix
    df["IsHTTPS"] = url_s.apply(
        lambda x: 1 if x.startswith("https://") else 0
    )

    # Lower-cased URLs reused for keyword-based features
    lower_url = url_s.str.lower()

    # Simple indicators for the presence of `<title>` and `favicon` tokens
    df["HasTitle"] = lower_url.str.contains("<title>").astype(int)
    df["HasFavicon"] = lower_url.str.contains("favicon").astype(int)

    # Domain-level aggregates
    domain_s = df["Domain"].astype(str)
    df["DomainLength"] = domain_s.apply(len)
    df["IsDomainIP"] = domain_s.apply(
        # Treat domain as IP if all dot-separated parts are numeric
        lambda x: 1 if x and all(part.isdigit() for part in x.split(".")) else 0
    )

    # Number of components in the domain (e.g. a.b.c => 3)
    df["NumSubdomains"] = domain_s.apply(
        lambda x: len(x.split(".")) - 1
    )

    # Top-level domain (last dot-separated token) and its length
    df["TLD"] = domain_s.apply(
        lambda x: x.split(".")[-1] if len(x.split(".")) > 1 else ""
    )
    df["TLDLength"] = df["TLD"].astype(str).apply(len)

    # Number of subdomains beyond the main domain and TLD
    df["NoOfSubDomain"] = domain_s.apply(
        lambda d: max(len(d.split(".")) - 2, 0) if d else 0
    )

    return df, url_s, lower_url


def _add_char_count_and_ratio_features(
    df: pd.DataFrame, url_s: pd.Series
) -> pd.DataFrame:
    """Add character count and ratio-based URL features."""
    # Raw counts of alphabetic and digit characters
    df["NoOfLettersInURL"] = url_s.str.count(r"[A-Za-z]")
    df["NoOfDegitsInURL"] = url_s.str.count(r"[0-9]")

    # Ratios of letters / digits to total URL length
    df["LetterRatioInURL"] = (
        df["NoOfLettersInURL"] / df["URLLength"]
    ).fillna(0.0)

    df["DegitRatioInURL"] = (
        df["NoOfDegitsInURL"] / df["URLLength"]
    ).fillna(0.0)

    # Counts of specific special characters
    df["NoOfEqualsInURL"] = url_s.str.count(r"=")
    df["NoOfQMarkInURL"] = url_s.str.count(r"\?")
    df["NoOfAmpersandInURL"] = url_s.str.count(r"&")

    # Total non-alphanumeric characters
    total_special = url_s.str.count(r"[^A-Za-z0-9]")

    # All other specials that are not =, ?, or &
    df["NoOfOtherSpecialCharsInURL"] = (
        total_special
        - df["NoOfEqualsInURL"]
        - df["NoOfQMarkInURL"]
        - df["NoOfAmpersandInURL"]
    ).clip(lower=0)

    # Overall ratio of special characters to total length
    df["SpacialCharRatioInURL"] = (
        df["NoOfEqualsInURL"]
        + df["NoOfQMarkInURL"]
        + df["NoOfAmpersandInURL"]
        + df["NoOfOtherSpecialCharsInURL"]
    ) / df["URLLength"]
    df["SpacialCharRatioInURL"] = df["SpacialCharRatioInURL"].fillna(0.0)

    return df


def _add_obfuscation_features(df: pd.DataFrame, url_s: pd.Series) -> pd.DataFrame:
    """Add URL obfuscation-related features (e.g. `%` or `@` characters)."""
    # Count characters typically used in obfuscation
    df["NoOfObfuscatedChar"] = url_s.str.count(r"[%@]")
    # Binary flag for any presence of obfuscating characters
    df["HasObfuscation"] = (df["NoOfObfuscatedChar"] > 0).astype(int)
    # Obfuscation intensity relative to URL length
    df["ObfuscationRatio"] = (
        df["NoOfObfuscatedChar"] / df["URLLength"]
    ).fillna(0.0)

    return df


def _add_keyword_flags(df: pd.DataFrame, lower_url: pd.Series) -> pd.DataFrame:
    """Add simple keyword presence flags (bank, pay, crypto)."""
    df["Bank"] = lower_url.str.contains("bank").astype(int)
    df["Pay"] = lower_url.str.contains("pay").astype(int)
    df["Crypto"] = lower_url.str.contains("crypto|bitcoin|btc|eth").astype(int)
    return df


def _add_char_continuation_feature(df: pd.DataFrame, url_s: pd.Series) -> pd.DataFrame:
    """Add the `CharContinuationRate` feature based on URL character types."""
    df["CharContinuationRate"] = url_s.apply(_char_continuation_rate)
    return df


def _add_label_and_tld_prob_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add label-based features and TLD-level legitimacy probabilities."""
    # Encode labels: assume 'good' => 1 (legitimate), everything else => 0
    df["label_binary"] = df["Label"].apply(lambda x: 1 if x == "good" else 0)

    # Average legitimacy rate per TLD
    tld_legit_prob = df.groupby("TLD")["label_binary"].mean()
    default_tld_prob = df["label_binary"].mean()

    # Map TLDs to their empirical legitimacy probabilities
    df["TLDLegitimateProb"] = (
        df["TLD"]
        .map(tld_legit_prob)
        .fillna(default_tld_prob)
    )

    return df


def _add_url_char_prob_and_similarity(df: pd.DataFrame, url_s: pd.Series) -> pd.DataFrame:
    """Add `URLCharProb` and `URLSimilarityIndex` based on character frequencies."""
    # Concatenate all URL strings to build a character frequency table
    all_chars = "".join(url_s.tolist())
    char_counts = Counter(all_chars)
    total_chars = sum(char_counts.values())

    # Simple Laplace smoothing over printable ASCII characters
    alphabet = [chr(i) for i in range(32, 127)]
    alpha_size = len(alphabet)
    denom = total_chars + alpha_size

    # Smoothed probability per character
    char_probs = {ch: (char_counts.get(ch, 0) + 1) / denom for ch in alphabet}

    def url_char_prob(u: str) -> float:
        """Geometric mean probability of a URL under the character model."""
        s = str(u)
        if not s:
            return 0.0
        log_p = 0.0
        for ch in s:
            # Fall back to a uniform low probability for unseen characters
            p = char_probs.get(ch, 1 / denom)
            log_p += float(np.log(p))
        avg_log = log_p / len(s)
        # Back to (0,1]: geometric mean probability per character
        return float(np.exp(avg_log))

    # Per-URL character probability and resulting similarity index
    df["URLCharProb"] = url_s.apply(url_char_prob)
    df["URLSimilarityIndex"] = df["URLCharProb"] * 100.0

    return df


def preprocess_phishing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to the raw Kaggle phishing URL dataset.

    The transformations mirror the logic implemented in
    `Notebook/explore_data.ipynb` (cell 17), so that the engineered
    features are consistent between the notebook and this service code.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Kaggle phishing URL dataframe as loaded by
        :func:`read_kaggle_phishing_data`.

    Returns
    -------
    pd.DataFrame
        Dataframe with engineered URL-based features.
    """
    df = df.copy()

    # Basic URL decomposition
    df, url_s, lower_url = _add_basic_url_parts(df)

    # Character counts and ratio-based metrics
    df = _add_char_count_and_ratio_features(df, url_s)

    # Obfuscation-related signals
    df = _add_obfuscation_features(df, url_s)

    # Keyword flags derived from the URL string
    df = _add_keyword_flags(df, lower_url)

    # Character continuation behavior along the URL
    df = _add_char_continuation_feature(df, url_s)

    # Label encoding and TLD-based legitimacy probabilities
    df = _add_label_and_tld_prob_features(df)

    # Step 7: URL character probability model and similarity index
    df = _add_url_char_prob_and_similarity(df, url_s)

    return df
