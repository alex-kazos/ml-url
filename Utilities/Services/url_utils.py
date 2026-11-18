"""Host and registered-domain helpers shared by the features and the lookalike rule."""
from urllib.parse import urlparse

COMMON_SECOND_LEVEL_SUFFIXES = {
    "ac",
    "co",
    "com",
    "edu",
    "gov",
    "net",
    "org",
}


def hostname_from_url(url: str) -> str:
    """Lower-cased host of a URL, with or without a scheme ("" when there is none)."""
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        return (parsed.hostname or "").lower().strip(".")
    except ValueError:
        # e.g. an invalid IPv6 literal such as "http://[::1"
        return ""


def registered_domain(hostname: str) -> str:
    """``a.b.example.co.uk`` -> ``example.co.uk``; ``a.example.com`` -> ``example.com``."""
    labels = [label for label in hostname.split(".") if label]
    if len(labels) < 2:
        return labels[0] if labels else ""

    if (
        len(labels) >= 3
        and labels[-2] in COMMON_SECOND_LEVEL_SUFFIXES
        and len(labels[-1]) == 2
    ):
        return ".".join(labels[-3:])

    return ".".join(labels[-2:])


def registered_domain_label(hostname: str) -> str:
    """The name part of the registered domain: ``a.example.co.uk`` -> ``example``."""
    domain = registered_domain(hostname)
    return domain.split(".", 1)[0] if domain else ""


def split_registered_domain(url: str) -> tuple[str, str]:
    """Split a URL into (registered label, everything outside the registered domain).

    ``me@login.paypal.com.evil.xyz/pay?x=1`` -> ``("evil", "me@ login.paypal.com /pay?x=1")``:
    user info, subdomains, path, query and fragment. Text-based features use
    this to tell a brand's own domain apart from a brand name placed anywhere
    else in the URL.
    """
    lower = str(url).lower()
    host = hostname_from_url(lower)
    if not host or host not in lower:
        return "", lower
    before, _, after = lower.partition(host)
    domain = registered_domain(host)
    subdomains = host[: -len(domain)].rstrip(".") if host.endswith(domain) else host
    return registered_domain_label(host), f"{before} {subdomains} {after}"
