"""Exercise local URL feature extraction without any network I/O."""

# Import urllib.request only in the test, to monkeypatch urlopen if it were called.
import urllib.request

# Import Path so a static source check can locate url_features.py.
from pathlib import Path

# Import pytest for monkeypatch and assertion helpers.
import pytest

from secure_chat_ml.url_features import (
    URL_FEATURE_NAMES,
    UrlFeatureExtractor,
    extract_message_url_features,
    extract_urls,
    suspicious_url_flag_count,
    url_feature_matrix,
)


# Confirm a lunch DM with no link produces a zero URL-risk vector.
def test_lunch_message_has_no_url_and_low_risk() -> None:
    """Assert 'lunch tomorrow' sets has_url=0 and no phishing flags."""

    # Score a short legitimate chat line that contains no URL.
    features = extract_message_url_features("lunch tomorrow")
    # No URL should be detected.
    assert features["has_url"] == 0.0
    # URL count should be zero when has_url is zero.
    assert features["url_count"] == 0.0
    # No phishing-like binary flags should be set on a link-free ham line.
    assert suspicious_url_flag_count(features) == 0


# Confirm an IP host plus /login path lights up the high-risk lexical flags.
def test_ip_login_url_sets_high_risk_features() -> None:
    """Assert a TEST-NET IP login URL is flagged as IP host + credential path."""

    # Use documentation-range 192.0.2.1 so the example is not a real host.
    text = "verify account https://192.0.2.1/login"
    # Score the message that pairs an IP host with a login path.
    features = extract_message_url_features(text)
    # The URL must be detected.
    assert features["has_url"] == 1.0
    # The host is an IPv4 literal.
    assert features["host_is_ip"] == 1.0
    # The path contains a credential-harvesting keyword.
    assert features["path_has_login_verify_update_password_keywords"] == 1.0
    # HTTPS is used (scheme only; no live fetch).
    assert features["uses_https"] == 1.0
    # The IP host contains digits.
    assert features["digits_in_host"] > 0
    # High-risk flags must be at least IP + login path.
    assert suspicious_url_flag_count(features) >= 2


# Confirm an ordinary Google Docs https link is milder than IP + /verify.
def test_google_docs_url_is_ham_like_compared_to_ip_verify() -> None:
    """Assert docs.google.com scores fewer suspicious flags than an IP /verify URL."""

    # Ordinary collaborative-doc URL of the kind a legitimate DM shares.
    ham_like = extract_message_url_features(
        "see the doc https://docs.google.com/document/d/abc123"
    )
    # Phishing-like IP URL with a verify path.
    phish_like = extract_message_url_features("pls check https://192.0.2.1/verify")
    # Both messages contain a URL.
    assert ham_like["has_url"] == 1.0
    assert phish_like["has_url"] == 1.0
    # The docs host is not an IP literal.
    assert ham_like["host_is_ip"] == 0.0
    # The phishing host is an IP literal.
    assert phish_like["host_is_ip"] == 1.0
    # The docs path should not match login/verify/update/password keywords.
    assert ham_like["path_has_login_verify_update_password_keywords"] == 0.0
    # The phishing path should match the verify keyword.
    assert phish_like["path_has_login_verify_update_password_keywords"] == 1.0
    # The phishing message must score strictly more suspicious flags.
    assert suspicious_url_flag_count(phish_like) > suspicious_url_flag_count(ham_like)


# Confirm HTML href values are harvested when rewrite input still has markup.
def test_extract_urls_reads_href_attributes() -> None:
    """Assert an href target is extracted from leftover HTML."""

    # Mimic an email body that still contains an anchor tag.
    html = '<p>Click <a href="https://evil.example/login">here</a> please</p>'
    # Extract URLs including the href value.
    urls = extract_urls(html)
    # The href target must be present even if visible text is only "here".
    assert "https://evil.example/login" in urls


# Confirm known shorteners are flagged from a frozen list, without resolving them.
def test_known_shortener_is_flagged_without_resolving() -> None:
    """Assert bit.ly is labeled a shortener via the frozen host list."""

    # A shortener URL; the extractor must not HTTP-fetch the destination.
    features = extract_message_url_features("pls check https://bit.ly/abc123")
    # The frozen list should recognize bit.ly.
    assert features["is_known_shortener"] == 1.0
    # The host is not an IP.
    assert features["host_is_ip"] == 0.0


# Confirm malformed IPv6 URLs do not crash on-device feature extraction.
def test_malformed_ipv6_url_does_not_raise() -> None:
    """Assert a truncated IPv6 URL still yields a feature dict."""

    # Phishing corpora sometimes contain unclosed IPv6 literals.
    features = extract_message_url_features("see http://[::1/login")
    # The extractor must still record that a URL-like span was present.
    assert features["has_url"] == 1.0
    # The path keyword should still be visible in the fallback split.
    assert features["path_has_login_verify_update_password_keywords"] == 1.0
def test_punycode_host_sets_flag() -> None:
    """Assert an IDN punycode label is detected locally."""

    # xn-- is the ACE prefix for internationalized domain labels.
    features = extract_message_url_features("open https://xn--80ak6aa92e.com/login")
    # The punycode flag must be set.
    assert features["punycode_xn"] == 1.0


# Confirm the sklearn transformer emits a matrix whose width matches the name list.
def test_url_feature_matrix_shape_matches_feature_names() -> None:
    """Assert one row per text and one column per URL_FEATURE_NAMES entry."""

    # Mix a no-URL row with a URL row so zeros and non-zeros both appear.
    texts = ["lunch tomorrow", "see https://example.com/path"]
    # Build the dense matrix used by StandardScaler.
    matrix = url_feature_matrix(texts)
    # Two input texts must produce two rows.
    assert matrix.shape == (2, len(URL_FEATURE_NAMES))
    # The first row is link-free, so has_url (column 0) is 0.
    assert matrix[0, 0] == 0.0
    # The second row has a URL, so has_url is 1.
    assert matrix[1, 0] == 1.0
    # The extractor transform path must match the helper matrix.
    transformed = UrlFeatureExtractor().fit_transform(texts)
    assert transformed.shape == matrix.shape


# Confirm the extractor never performs network I/O, statically and at runtime.
def test_url_features_never_perform_network_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert url_features.py has no fetch imports and does not call urlopen."""

    # Locate the module source for a static ban on fetch libraries.
    source_path = Path(__file__).resolve().parents[1] / "src" / "secure_chat_ml" / "url_features.py"
    # Read the module source as text.
    source = source_path.read_text(encoding="utf-8")
    # Ban common HTTP/reputation client imports in this file.
    for banned in (
        "urlopen",
        "urllib.request",
        "http.client",
        "import requests",
        "import httpx",
        "socket.socket",
    ):
        # The source must not mention fetch/reputation clients.
        assert banned not in source, f"banned token {banned!r} found in url_features.py"

    # If any code path called urlopen, fail the test immediately.
    def _forbid_urlopen(*_args: object, **_kwargs: object) -> None:
        # Raise so a regression that starts fetching cannot pass silently.
        raise AssertionError("urlopen must not be called during URL feature extraction")

    # Patch the stdlib fetch entry point for the duration of this test.
    monkeypatch.setattr(urllib.request, "urlopen", _forbid_urlopen)
    # Extract features from several URL shapes, including a shortener that must not be resolved.
    extract_message_url_features(
        "verify account https://192.0.2.1/login and also https://bit.ly/abc"
    )
    # Extract from HTML hrefs as well.
    extract_urls('<a href="https://example.com/x">x</a>')
