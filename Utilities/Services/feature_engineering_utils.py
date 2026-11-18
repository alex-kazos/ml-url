import math
import re
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from Utilities.Services.url_utils import split_registered_domain


SUSPICIOUS_TLDS = frozenset({
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "buzz", "club",
    "work", "date", "bid", "stream", "download", "racing", "win",
    "review", "accountant", "science", "cricket", "party", "faith",
    "loan", "men", "click", "link", "info", "pw", "cc",
})

PHISHING_KEYWORDS = [
    "login", "verify", "secure", "update", "account", "signin",
    "confirm", "password", "billing", "suspend", "alert", "expire",
    "unlock", "validate", "authenticate", "webscr", "cmd",
]

BRAND_NAMES = [
    "google", "facebook", "paypal", "apple", "microsoft", "amazon",
    "netflix", "ebay", "chase", "wells", "citi", "instagram",
    "linkedin", "twitter", "yahoo", "dropbox", "adobe", "whatsapp",
]

# Registered-domain labels the brands above really own. A brand name inside
# one of these is the brand's own site, not an impersonation; it differs from
# the token for "wells" (wellsfargo.com) and "citi" (citibank.com).
BRAND_OWN_DOMAIN_LABELS = frozenset(BRAND_NAMES) | {"wellsfargo", "citibank"}


def _shannon_entropy(text: str) -> float:
    """Compute Shannon entropy of a string."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _extract_path(url: str) -> str:
    """Return the path component of a URL string."""
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        return parsed.path or ""
    except Exception:
        return ""


def _extract_query(url: str) -> str:
    """Return the query string component of a URL string."""
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        return parsed.query or ""
    except Exception:
        return ""


# Pre-compiled regex for port detection  (e.g. `:8080/`)
_PORT_RE = re.compile(r":(\d{2,5})(/|$)")


def add_entropy_features(df: pd.DataFrame, url_s: pd.Series) -> pd.DataFrame:
    """Add Shannon entropy of URL and domain."""
    df["URLEntropy"] = url_s.apply(_shannon_entropy)

    domain_s = df["Domain"].astype(str)
    df["DomainEntropy"] = domain_s.apply(_shannon_entropy)
    return df


def add_path_features(df: pd.DataFrame, url_s: pd.Series) -> pd.DataFrame:
    """Add path-related features: length, token count, avg/max token length."""
    paths = url_s.apply(_extract_path)

    df["PathLength"] = paths.apply(len)

    # Split on '/' and drop empty strings
    tokens = paths.apply(lambda p: [t for t in p.split("/") if t])
    df["PathTokenCount"] = tokens.apply(len)

    df["AvgTokenLength"] = tokens.apply(
        lambda ts: np.mean([len(t) for t in ts]) if ts else 0.0
    )
    df["MaxTokenLength"] = tokens.apply(
        lambda ts: max(len(t) for t in ts) if ts else 0
    )
    return df


def add_query_features(df: pd.DataFrame, url_s: pd.Series) -> pd.DataFrame:
    """Add query string length."""
    df["QueryLength"] = url_s.apply(lambda u: len(_extract_query(u)))
    return df


def add_punctuation_counts(df: pd.DataFrame, url_s: pd.Series) -> pd.DataFrame:
    """Add counts of specific punctuation characters in the URL."""
    df["NumDots"] = url_s.str.count(r"\.")
    df["NumHyphens"] = url_s.str.count(r"-")
    df["NumUnderscores"] = url_s.str.count(r"_")
    df["NumSlashes"] = url_s.str.count(r"/")
    df["NumAtSymbol"] = url_s.str.count(r"@")
    return df


def add_port_feature(df: pd.DataFrame, url_s: pd.Series) -> pd.DataFrame:
    """Add binary flag for the presence of a port number in the URL."""
    df["HasPortNumber"] = url_s.apply(
        lambda u: 1 if _PORT_RE.search(u) else 0
    )
    return df


def add_suspicious_tld_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add flag for suspicious TLDs commonly used in phishing."""
    tld_lower = df["TLD"].astype(str).str.lower()
    df["HasSuspiciousTLD"] = tld_lower.isin(SUSPICIOUS_TLDS).astype(int)
    return df


def add_suspicious_word_count(
    df: pd.DataFrame, lower_url: pd.Series
) -> pd.DataFrame:
    """Count how many phishing-related keywords appear in each URL."""
    pattern = "|".join(PHISHING_KEYWORDS)
    df["NumSuspiciousWords"] = lower_url.str.count(pattern)
    return df


def add_brand_name_feature(
    df: pd.DataFrame, lower_url: pd.Series
) -> pd.DataFrame:
    """Flag whether a well-known brand name appears in the URL."""
    pattern = "|".join(BRAND_NAMES)
    df["HasBrandName"] = lower_url.str.contains(pattern).astype(int)
    return df


def add_brand_and_pay_features(df: pd.DataFrame, url_s: pd.Series) -> pd.DataFrame:
    """Brand names and "pay" only where they point to impersonation.

    ``HasBrandName`` and ``Pay`` also fire on a brand's own domain, so v9
    scores ``paypal.com`` 0.79. These versions ignore the registered domain
    when the brand owns it (``paypal.com``, ``login.paypal.com/signin``) and
    count the brand when it sits in user info, a subdomain, path or query
    (``paypal.com.account-verify.xyz``) or inside another registered domain
    (``paypal-secure.com``).
    """
    parts = url_s.map(split_registered_domain)
    label = parts.str[0]
    outside = parts.str[1]
    own_brand_domain = label.isin(BRAND_OWN_DOMAIN_LABELS)
    brands = "|".join(BRAND_NAMES)

    df["BrandOutsideDomain"] = outside.str.contains(brands).astype(int)
    df["BrandInDomainNotExact"] = (label.str.contains(brands) & ~own_brand_domain).astype(int)
    df["PayNotBrandDomain"] = (
        outside.str.contains("pay") | (label.str.contains("pay") & ~own_brand_domain)
    ).astype(int)
    return df


_EMAIL_RE = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
_HOST_PART_RE = r"^[^/?#]*"


def add_at_sign_features(df: pd.DataFrame, url_s: pd.Series) -> pd.DataFrame:
    """Tell the "@" tricks phishing uses apart from "@" handles.

    ``NumAtSymbol`` counts every "@", so v9 scores ``substack.com/@alexkazos``
    0.81: profile handles (Substack, Medium, YouTube) are newer than the
    training data. What phishing really uses is an email address in the URL
    (the victim's, in a query such as ``?email=jo@example.com``) and user info
    before the host (``paypal.com@evil.example/login``).
    """
    df["NumEmailsInURL"] = url_s.str.count(_EMAIL_RE)
    host_part = url_s.str.extract(f"({_HOST_PART_RE})", expand=False).fillna("")
    df["HasUserInfo"] = host_part.str.contains("@", regex=False).astype(int)
    # Percent-encoding alone: the obfuscation half of NoOfObfuscatedChar ([%@]).
    df["NumPercentChars"] = url_s.str.count("%")
    return df


def add_vowel_ratio(df: pd.DataFrame, url_s: pd.Series) -> pd.DataFrame:
    """Add ratio of vowels to total letters in the URL."""
    num_vowels = url_s.str.count(r"[aeiouAEIOU]")
    num_letters = df["NoOfLettersInURL"]
    df["VowelRatio"] = (num_vowels / num_letters).fillna(0.0)
    return df



def add_all_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every new feature engineering step to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataset loaded from ``final_dataset.pkl`` or
        ``final_dataset.csv``.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional feature columns appended.
    """
    df = df.copy()

    url_s = df["URL"].astype(str)
    lower_url = url_s.str.lower()

    df = add_entropy_features(df, url_s)
    df = add_path_features(df, url_s)
    df = add_query_features(df, url_s)
    df = add_punctuation_counts(df, url_s)
    df = add_port_feature(df, url_s)
    df = add_suspicious_tld_feature(df)
    df = add_suspicious_word_count(df, lower_url)
    df = add_brand_name_feature(df, lower_url)
    df = add_brand_and_pay_features(df, url_s)
    df = add_at_sign_features(df, url_s)
    df = add_vowel_ratio(df, url_s)

    return df
