"""Extract on-device lexical and structural URL features from message text.

Every feature is computed from the URL string itself. This module never
opens a socket, never HTTP-fetches a URL, never follows a shortener, and
never calls VirusTotal, Google Safe Browsing, PhishTank, or any other
remote reputation API. Legitimate DMs share ordinary https links; a
present URL is not treated as automatic evidence of a scam.
"""

# Import re to find URL-like spans and HTML href attributes in raw text.
import re

# Import Iterable for the public matrix helper's input type.
from collections.abc import Iterable

# Import urlparse to split a URL into scheme, host, path, and query locally.
from urllib.parse import urlparse

# Import numpy to build the dense per-message feature matrix for sklearn.
import numpy as np

# Import csr_matrix so FeatureUnion can hstack with sparse TF-IDF without densifying.
from scipy.sparse import csr_matrix

# Import sklearn base classes so the extractor plugs into FeatureUnion.
from sklearn.base import BaseEstimator, TransformerMixin

# Import Pipeline so extract → scale → sparse is one reusable transformer.
from sklearn.pipeline import Pipeline

# Import StandardScaler to z-score the small numeric URL block on TRAIN only.
from sklearn.preprocessing import StandardScaler

# Frozen shortener hostnames; compared after lowercasing and stripping a leading www.
KNOWN_SHORTENER_HOSTS: frozenset[str] = frozenset(
    {
        "bit.ly",
        "t.co",
        "tinyurl.com",
        "tiny.cc",
        "tiny.one",
        "goo.gl",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "cutt.ly",
        "rebrand.ly",
        "shorturl.at",
        "lnkd.in",
        "db.tt",
        "qr.ae",
        "adf.ly",
        "bit.do",
        "t.ly",
        "v.gd",
        "x.co",
        "rb.gy",
        "trib.ly",
        "soo.gd",
        "s.id",
        "bl.ink",
    }
)

# Frozen TLDs that are cheap to register or commonly abused in phishing kits.
SUSPICIOUS_TLDS: frozenset[str] = frozenset(
    {
        "zip",
        "mov",
        "xyz",
        "top",
        "click",
        "link",
        "tk",
        "ml",
        "ga",
        "cf",
        "gq",
        "country",
        "stream",
        "download",
        "racing",
        "review",
        "work",
        "party",
        "science",
        "bid",
        "loan",
        "win",
        "date",
        "faith",
        "cricket",
        "webcam",
        "pw",
        "rest",
        "accountants",
        "support",
        "fit",
        "gdn",
        "quest",
        "cfd",
        "sbs",
        "cyou",
        "icu",
        "cam",
        "hair",
        "makeup",
        "autos",
        "boats",
        "yachts",
        "motorcycles",
        "homes",
        "security",
    }
)

# Path tokens that often mark credential-harvesting landing pages.
_PATH_KEYWORDS: tuple[str, ...] = (
    "login",
    "log-in",
    "signin",
    "sign-in",
    "verify",
    "update",
    "password",
    "passwd",
    "wp-login",
)

# Match http(s) URLs and www. hosts; trailing sentence punctuation is trimmed later.
_SCHEME_OR_WWW_RE = re.compile(r"""(?i)\b(?:https?://|www\.)[^\s<>"'\)\]]+""")

# Match schemeless host/path pairs only when a path slash is present (avoids "U.S.").
_SCHEMLESS_RE = re.compile(
    r"""(?i)\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|co|info|biz|xyz|top|click|link|ru|cn|"""
    r"""uk|us|ly|me|cc|tv|app|dev|edu|gov)/(?:[^\s<>"'\)\]]+)"""
)

# Pull href='...' / href="..." values out of leftover HTML before tags are stripped.
_HREF_RE = re.compile(r"""(?i)href\s*=\s*['"]([^'"]+)['"]""")

# Recognize IPv4 hosts such as 192.0.2.1 without performing any DNS lookup.
_IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")

# Strip trailing punctuation that URL regexes commonly over-capture from prose.
_TRAILING_PUNCT = ".,);]!?'\""

# Schemes that are not fetchable web links and must not enter the feature vector.
_SKIP_SCHEMES = frozenset({"javascript", "mailto", "tel", "sms", "data", "file"})

# Ordered names of the aggregated per-message feature vector consumed by sklearn.
URL_FEATURE_NAMES: tuple[str, ...] = (
    "has_url",
    "url_count",
    "uses_https",
    "host_is_ip",
    "has_at_sign",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "url_length",
    "path_length",
    "num_subdomains",
    "is_known_shortener",
    "suspicious_tld",
    "path_has_login_verify_update_password_keywords",
    "punycode_xn",
    "digits_in_host",
    "num_dots_mean",
    "num_hyphens_mean",
    "url_length_mean",
    "path_length_mean",
)


# Trim punctuation and whitespace that is not part of the URL itself.
def _trim_url(raw: str) -> str:
    """Return a cleaned URL candidate with wrapping punctuation removed."""

    # Drop surrounding whitespace copied from the message span.
    cleaned = raw.strip()
    # Repeatedly peel trailing sentence punctuation the regex may have included.
    cleaned = cleaned.rstrip(_TRAILING_PUNCT)
    # Drop a wrapping closing parenthesis if the opener was not inside the URL.
    if cleaned.endswith(")") and cleaned.count("(") < cleaned.count(")"):
        # Remove the extra closing paren that belonged to surrounding prose.
        cleaned = cleaned[:-1]
    # Return the cleaned candidate for scheme filtering.
    return cleaned


# Decide whether a candidate string is a web URL worth scoring locally.
def _is_skippable(raw: str) -> bool:
    """Return True when the candidate is empty or a non-http(s) URI scheme."""

    # Empty strings carry no URL signal.
    if not raw:
        # Treat missing candidates as skippable.
        return True
    # Read the scheme up to the first colon, if any.
    scheme = raw.split(":", 1)[0].lower()
    # Skip javascript/mailto/tel/data/file URIs that are not web links.
    return scheme in _SKIP_SCHEMES


# Find http(s), www, schemeless host/path, and HTML href URLs in one message.
def extract_urls(text: str) -> list[str]:
    """Return unique URL strings in first-seen order, without network I/O."""

    # Treat missing text as having no URLs.
    if not text:
        # Return an empty list so callers can iterate uniformly.
        return []
    # Accumulate unique URLs while preserving first-seen order.
    found: list[str] = []
    # Track already-emitted strings so duplicates from href + visible text collapse.
    seen: set[str] = set()

    # Add one cleaned candidate if it is a real web URL and not a duplicate.
    def _add(candidate: str) -> None:
        # Normalize wrapping punctuation before the uniqueness check.
        cleaned = _trim_url(candidate)
        # Ignore empty, non-web, or already-recorded candidates.
        if _is_skippable(cleaned) or cleaned in seen:
            # Skip this candidate without mutating the output list.
            return
        # Remember the candidate so later duplicate spans are ignored.
        seen.add(cleaned)
        # Append in first-seen order for stable urls_json output.
        found.append(cleaned)

    # Prefer explicit href targets so leftover HTML still yields the real link.
    for match in _HREF_RE.finditer(text):
        # Record the href value as a URL candidate.
        _add(match.group(1))
    # Collect canonical http(s) and www. spans from visible text.
    for match in _SCHEME_OR_WWW_RE.finditer(text):
        # Record the matched http(s)/www span.
        _add(match.group(0))
    # Collect regex-safe schemeless host/path pairs such as bit.ly/abc.
    for match in _SCHEMLESS_RE.finditer(text):
        # Record the schemeless candidate; parsing will add a default scheme.
        _add(match.group(0))
    # Return the ordered unique URL list for feature extraction and rewrites.
    return found


# Give urlparse a scheme so www. and schemeless hosts populate hostname/path.
def _for_parse(raw: str) -> str:
    """Return a URL string urlparse can split into host and path locally."""

    # Leave already-schemed URLs unchanged.
    if re.match(r"^[a-z][a-z0-9+.-]*://", raw, flags=re.IGNORECASE):
        # Return the original schemed URL for parsing.
        return raw
    # Pretend www./host/path strings are http so hostname is populated, not path.
    return "http://" + raw


# Lowercase a hostname and strip a leading www. for list lookups.
def _normalize_host(host: str) -> str:
    """Return a lowercase hostname without a leading www. label."""

    # Lowercase so list membership is case-insensitive.
    lowered = host.lower()
    # Strip a leading www. which is not part of shortener or TLD identity.
    if lowered.startswith("www."):
        # Drop the www. prefix for frozen-list comparisons.
        return lowered[4:]
    # Return the already-normalized host.
    return lowered


# Detect IPv4 or IPv6 hosts without DNS or socket calls.
def _host_is_ip(host: str) -> bool:
    """Return True when the hostname is a literal IPv4 or IPv6 address."""

    # Empty hosts cannot be IP literals.
    if not host:
        # Treat missing hosts as not-IP.
        return False
    # Match dotted-decimal IPv4 used in many phishing kits.
    if _IPV4_RE.match(host):
        # An IPv4 literal is always scored as host_is_ip.
        return True
    # IPv6 literals contain colons; urlparse may wrap them in brackets.
    stripped = host.strip("[]")
    # A colon in the stripped host is the cheap IPv6 heuristic (no network).
    return ":" in stripped


# Count labels to the left of the registrable domain; IP hosts have zero.
def _subdomain_count(host: str) -> int:
    """Return how many subdomain labels sit left of domain.tld."""

    # IP literals have no subdomain structure.
    if not host or _host_is_ip(host):
        # Report zero subdomains for IPs and empty hosts.
        return 0
    # Split the normalized host on dots to count labels.
    labels = [part for part in _normalize_host(host).split(".") if part]
    # Need at least domain + tld before anything counts as a subdomain.
    if len(labels) <= 2:
        # example.com has zero subdomains.
        return 0
    # Treat every extra label as a subdomain (www.a.b.example.com → 3).
    return len(labels) - 2


# Read the rightmost label as a TLD heuristic (no Public Suffix List fetch).
def _tld(host: str) -> str:
    """Return the last dotted label of a hostname, or empty for IPs."""

    # IP literals have no TLD.
    if not host or _host_is_ip(host):
        # Report an empty TLD so suspicious_tld stays 0.
        return ""
    # Split on dots and take the last non-empty label.
    labels = [part for part in host.lower().split(".") if part]
    # Return the last label, or empty if the host was only dots.
    return labels[-1] if labels else ""


# Split a URL locally, falling back to string heuristics when urlparse rejects it.
def _split_url(url: str) -> tuple[str, str, str, str]:
    """Return (scheme, host, path, query) without network I/O.

    Phishing corpora contain malformed IPv6 and truncated URLs that make
    urllib.parse.urlparse raise ValueError. Those strings still get lexical
    features from this fallback path.
    """

    # Prefer stdlib parsing when the string is a well-formed URL.
    try:
        # Normalize schemeless hosts so hostname is populated.
        parsed = urlparse(_for_parse(url))
        # hostname can itself raise ValueError on broken IPv6 literals.
        host = parsed.hostname or ""
        # Return the four fields used by the lexical feature function.
        return parsed.scheme or "", host, parsed.path or "", parsed.query or ""
    except ValueError:
        # Fall through to linear string heuristics for malformed URLs.
        pass
    # Work on a scheme-normalized copy even in the fallback path.
    raw = _for_parse(url)
    # Detect https vs http from the prefix only.
    lowered = raw.lower()
    # Record the scheme token used by uses_https.
    scheme = "https" if lowered.startswith("https://") else "http"
    # Drop the scheme prefix when present.
    rest = raw.split("://", 1)[-1] if "://" in raw else raw
    # Keep the last @-segment as the host side of userinfo@host.
    if "@" in rest:
        # user:pass@host is a classic phishing obfuscation; keep the host part.
        rest = rest.rsplit("@", 1)[-1]
    # Split host from path at the first slash.
    host_part, _, path_and_query = rest.partition("/")
    # Drop a trailing port when it is a single :digits suffix, not IPv6.
    if host_part.count(":") == 1 and not host_part.startswith("["):
        # Keep only the hostname to the left of the port.
        host_part = host_part.split(":", 1)[0]
    # Split path and query at the first '?'.
    path, _, query = path_and_query.partition("?")
    # Drop a fragment if one leaked into the query string.
    query = query.split("#", 1)[0]
    # Return heuristic fields; host may still be an opaque malformed literal.
    return scheme, host_part.strip("[]"), ("/" + path if path else ""), query


# Score one URL into the lexical/structural features listed in the spec.
def extract_single_url_features(url: str) -> dict[str, float]:
    """Return local features for one URL string; never performs I/O."""

    # Split scheme/host/path/query locally, tolerating malformed phishing URLs.
    scheme, host, path, query = _split_url(url)
    # Combine path and query so keyword checks see /login?next=bank as well.
    path_and_query = path + (("?" + query) if query else "")
    # Normalize the host for shortener and TLD lookups.
    host_norm = _normalize_host(host)
    # Count dots in the hostname (classic phishing visual-confusion feature).
    num_dots = float(host.count("."))
    # Count hyphens in the hostname (brand-impersonation feature).
    num_hyphens = float(host.count("-"))
    # Count digit characters in the full original URL string.
    num_digits = float(sum(char.isdigit() for char in url))
    # Count digit characters in the hostname (paypa1.com, 192.0.2.1).
    digits_in_host = float(sum(char.isdigit() for char in host))
    # Measure the raw URL length including scheme and query.
    url_length = float(len(url))
    # Measure the path length excluding the host.
    path_length = float(len(path))
    # Flag https; http-only and schemeless (treated as http) score 0.
    uses_https = 1.0 if scheme.lower() == "https" else 0.0
    # Flag IP-literal hosts used to skip brand-looking domains.
    host_is_ip = 1.0 if _host_is_ip(host) else 0.0
    # Flag '@' which can hide the real host in http://user@evil/ paths.
    has_at_sign = 1.0 if "@" in url else 0.0
    # Flag frozen shortener hosts; this is a list lookup, not a resolve.
    is_known_shortener = 1.0 if (
        host_norm in KNOWN_SHORTENER_HOSTS
        or any(host_norm.endswith("." + name) for name in KNOWN_SHORTENER_HOSTS)
    ) else 0.0
    # Flag frozen suspicious TLDs using the last hostname label only.
    suspicious_tld = 1.0 if _tld(host) in SUSPICIOUS_TLDS else 0.0
    # Flag credential-harvesting keywords anywhere in path or query.
    lowered_path = path_and_query.lower()
    # Score 1 when any configured path keyword is a substring of path/query.
    path_keywords = 1.0 if any(keyword in lowered_path for keyword in _PATH_KEYWORDS) else 0.0
    # Flag punycode labels used in IDN homograph lookalikes.
    punycode = 1.0 if "xn--" in host.lower() else 0.0
    # Count subdomain labels left of domain.tld.
    num_subdomains = float(_subdomain_count(host))
    # Return a dict so aggregators can max/mean/any by name.
    return {
        "uses_https": uses_https,
        "host_is_ip": host_is_ip,
        "has_at_sign": has_at_sign,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_digits": num_digits,
        "url_length": url_length,
        "path_length": path_length,
        "num_subdomains": num_subdomains,
        "is_known_shortener": is_known_shortener,
        "suspicious_tld": suspicious_tld,
        "path_has_login_verify_update_password_keywords": path_keywords,
        "punycode_xn": punycode,
        "digits_in_host": digits_in_host,
    }


# Collapse every URL in a message into one fixed-length vector (max/mean/any).
def extract_message_url_features(text: str) -> dict[str, float]:
    """Return aggregated URL features for one message, or zeros when none exist."""

    # Collect URL strings from the message without touching the network.
    urls = extract_urls(text)
    # Build the zero vector used when the message has no URLs at all.
    zeros = {name: 0.0 for name in URL_FEATURE_NAMES}
    # Short-circuit so legitimate link-free DMs do not look "missing".
    if not urls:
        # has_url stays 0 and every other feature stays 0.
        return zeros
    # Score each URL independently before aggregating.
    per_url = [extract_single_url_features(url) for url in urls]
    # Binary/any features: a single suspicious URL should surface.
    any_names = (
        "uses_https",
        "host_is_ip",
        "has_at_sign",
        "is_known_shortener",
        "suspicious_tld",
        "path_has_login_verify_update_password_keywords",
        "punycode_xn",
    )
    # Continuous features: keep the max (worst/longest) plus a mean copy.
    max_mean_names = ("num_dots", "num_hyphens", "num_digits", "url_length", "path_length")
    # Start from zeros and overwrite with aggregates.
    aggregated = dict(zeros)
    # Record that at least one URL was present.
    aggregated["has_url"] = 1.0
    # Record how many unique URLs the message contained.
    aggregated["url_count"] = float(len(urls))
    # Reduce binary flags with max, which is equivalent to logical any.
    for name in any_names:
        # Take the strongest (1) observation across URLs.
        aggregated[name] = max(row[name] for row in per_url)
    # Reduce continuous features with both max and mean as the spec allows.
    for name in max_mean_names:
        # Collect the per-URL values for this continuous feature.
        values = [row[name] for row in per_url]
        # Store the max under the canonical spec name (num_dots, url_length, ...).
        aggregated[name] = max(values)
        # Store the mean under an explicit _mean suffix for the extra columns.
        aggregated[f"{name}_mean"] = float(sum(values) / len(values))
    # num_subdomains and digits_in_host are max-aggregated (worst host wins).
    aggregated["num_subdomains"] = max(row["num_subdomains"] for row in per_url)
    # digits_in_host uses max so an IP host is not averaged away by a second URL.
    aggregated["digits_in_host"] = max(row["digits_in_host"] for row in per_url)
    # Return the fixed-key dict matching URL_FEATURE_NAMES.
    return aggregated


# Turn aggregated dicts into the column order sklearn will train on.
def features_to_vector(features: dict[str, float]) -> list[float]:
    """Return feature values in URL_FEATURE_NAMES order."""

    # Look up each name so the vector layout cannot drift from the name tuple.
    return [float(features[name]) for name in URL_FEATURE_NAMES]


# Count high-risk binary flags for unit tests (not used as the model target).
def suspicious_url_flag_count(features: dict[str, float]) -> int:
    """Return how many phishing-like binary flags are set on a message."""

    # These flags distinguish IP+/login phishing pages from ordinary https docs.
    flag_names = (
        "host_is_ip",
        "has_at_sign",
        "is_known_shortener",
        "suspicious_tld",
        "path_has_login_verify_update_password_keywords",
        "punycode_xn",
    )
    # Sum the 0/1 flags as an integer for easy test comparisons.
    return int(sum(features[name] for name in flag_names))


# Convert an iterable of texts into a dense (n_samples, n_features) matrix.
def url_feature_matrix(texts: Iterable[str]) -> np.ndarray:
    """Return a float64 matrix of aggregated URL features, one row per text."""

    # Materialize rows so the shape is known before stacking.
    rows = [features_to_vector(extract_message_url_features(str(text))) for text in texts]
    # Handle the empty-input case so sklearn gets a 2-D array with the right width.
    if not rows:
        # Return shape (0, n_features) rather than a 1-D empty array.
        return np.zeros((0, len(URL_FEATURE_NAMES)), dtype=np.float64)
    # Stack rows into a contiguous float64 matrix for StandardScaler.
    return np.asarray(rows, dtype=np.float64)


# Accept pandas Series or numpy arrays of strings from FeatureUnion.
def _as_text_list(X: object) -> list[str]:
    """Return a list of strings from the array-like FeatureUnion passes in."""

    # pandas Series supports fillna; use it when present so NaN does not become "nan".
    if hasattr(X, "fillna") and hasattr(X, "astype") and hasattr(X, "tolist"):
        # Normalize missing cells to empty strings before vectorizing URLs.
        return X.fillna("").astype(str).tolist()
    # Fall back to a numpy object array for lists and ndarrays.
    flattened = np.asarray(X, dtype=object).ravel()
    # Map None to empty string; stringify everything else.
    return ["" if item is None else str(item) for item in flattened]


# Sklearn transformer: message text → dense URL feature matrix.
class UrlFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract aggregated local URL features; fit() is a no-op (no learned state)."""

    # sklearn calls fit on TRAIN only; URL features are stateless heuristics.
    def fit(self, X: object, y: object = None) -> "UrlFeatureExtractor":
        """Return self; URL features do not learn parameters from training data."""

        # Mark unused y explicitly so linters accept the sklearn fit signature.
        del y
        # Record that fit ran so sklearn 1.7 Pipeline.check_is_fitted succeeds.
        self.n_features_out_ = len(URL_FEATURE_NAMES)
        # Stateless fit still needs to return self for Pipeline compatibility.
        return self

    # Transform any array-like of texts into the URL feature matrix.
    def transform(self, X: object) -> np.ndarray:
        """Return the dense URL feature matrix for the given texts."""

        # Convert the FeatureUnion input into a list of strings.
        texts = _as_text_list(X)
        # Build the numeric matrix used by StandardScaler.
        return url_feature_matrix(texts)

    # Expose stable names so FeatureUnion can prefix them in get_feature_names_out.
    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return the aggregated URL feature names in vector order."""

        # input_features is unused because this transformer does not select columns.
        del input_features
        # Return a numpy array of names as sklearn's feature-name API requires.
        return np.asarray(URL_FEATURE_NAMES, dtype=object)


# Convert the scaled dense URL block to sparse so FeatureUnion stays sparse.
class DenseToSparse(BaseEstimator, TransformerMixin):
    """Wrap a dense ndarray as csr_matrix so TF-IDF is not densified."""

    # No learned state; fit only exists for Pipeline API completeness.
    def fit(self, X: object, y: object = None) -> "DenseToSparse":
        """Return self after recording input width so the pipeline is 'fitted'."""

        # sklearn Pipelines always pass y; this transformer ignores it.
        del y
        # Capture the feature width sklearn 1.7 uses to decide the pipeline is fitted.
        array = np.asarray(X)
        # Store n_features_in_ so check_is_fitted finds a trailing-underscore attribute.
        self.n_features_in_ = int(array.shape[1]) if array.ndim == 2 else 1
        # Return self so later steps can call transform.
        return self

    # Convert the scaled dense matrix into CSR sparse form.
    def transform(self, X: object) -> csr_matrix:
        """Return a CSR matrix with the same values as the dense input."""

        # csr_matrix accepts array-like; dtype float64 matches StandardScaler output.
        return csr_matrix(np.asarray(X, dtype=np.float64))


# Build extract → StandardScaler → sparse as one FeatureUnion branch.
def build_url_feature_pipeline() -> Pipeline:
    """Return the URL-feature branch: extract, z-score on TRAIN, then sparsify.

    StandardScaler is fit on training rows only when this pipeline is inside
    the outer TF-IDF + classifier Pipeline. Messages with no URL keep a zero
    vector (has_url=0) rather than being dropped.
    """

    # Compose the three steps so baseline.py can drop this in as one transformer.
    return Pipeline(
        steps=[
            # Compute the lexical/structural URL matrix from raw message text.
            ("extract", UrlFeatureExtractor()),
            # Z-score the small dense block; with_mean is fine because this block is tiny.
            ("scale", StandardScaler()),
            # Convert to CSR so FeatureUnion hstacks with sparse TF-IDF.
            ("to_sparse", DenseToSparse()),
        ]
    )
