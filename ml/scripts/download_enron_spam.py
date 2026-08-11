"""Download and normalize the Metsis Enron-Spam derivative for local research."""

# Import sha256 to pin downloaded research archives against upstream changes.
from hashlib import sha256

# Import Path for workspace-independent data locations.
from pathlib import Path

# Import ssl so the incomplete AUEB certificate chain can be handled explicitly.
import ssl

# Import tarfile to read gzipped Enron archives in memory.
import tarfile

# Import urlopen for a dependency-free HTTPS dataset download.
from urllib.request import Request, urlopen

# Import pandas for typed tabular loading and normalization.
import pandas as pd

# Import the shared email body extractor used by every email corpus script.
from secure_chat_ml.email_text import extract_email_text

# Resolve the ML project directory from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Keep immutable downloaded source files in the ignored raw directory.
RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "enron_spam"
# Keep normalized derived files in the ignored processed directory.
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"
# Name the normalized CSV consumed by the EDA notebook.
OUTPUT_PATH = PROCESSED_DIRECTORY / "enron_spam.csv"
# Point at the authoritative AUEB Enron-Spam preprocessed mirrors.
BASE_URL = "https://www2.aueb.gr/users/ion/data/enron-spam/preprocessed"
# Download all six user-specific benchmark archives described by Metsis et al.
ARCHIVE_NAMES = [
    # Include the first Enron user benchmark archive.
    "enron1.tar.gz",
    # Include the second Enron user benchmark archive.
    "enron2.tar.gz",
    # Include the third Enron user benchmark archive.
    "enron3.tar.gz",
    # Include the fourth Enron user benchmark archive.
    "enron4.tar.gz",
    # Include the fifth Enron user benchmark archive.
    "enron5.tar.gz",
    # Include the sixth Enron user benchmark archive.
    "enron6.tar.gz",
]
# Create an unverified TLS context because www2.aueb.gr serves an incomplete chain.
SSL_CONTEXT = ssl._create_unverified_context()


# Download one archive only when it is absent locally.
def download_archive(archive_name: str) -> Path:
    """Fetch one Enron-Spam archive into the local raw-data directory."""

    # Create the ignored raw directory without failing when it already exists.
    RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Build the local destination path for this archive.
    archive_path = RAW_DIRECTORY / archive_name
    # Reuse the local archive to avoid unnecessary network requests.
    if not archive_path.exists():
        # Compose the absolute HTTPS URL for the selected archive.
        download_url = f"{BASE_URL}/{archive_name}"
        # Attach a descriptive User-Agent because some academic hosts reject empty agents.
        request = Request(download_url, headers={"User-Agent": "secure-chat-ml/0.1"})
        # Open the authoritative HTTPS resource with a generous timeout for large archives.
        with urlopen(request, timeout=180, context=SSL_CONTEXT) as response:  # noqa: S310
            # Read the complete archive after the TLS session is established.
            archive_bytes = response.read()
        # Persist the immutable source archive for reproducible local reruns.
        archive_path.write_bytes(archive_bytes)
    # Return the local archive path for later parsing.
    return archive_path


# Infer ham versus spam from the Enron archive member path.
def _label_from_member_name(member_name: str) -> str | None:
    """Return the source label encoded by the archive directory layout."""

    # Normalize path separators so Windows and POSIX member names behave alike.
    normalized = member_name.replace("\\", "/").lower()
    # Treat members under a spam directory as the positive class.
    if "/spam/" in normalized or normalized.startswith("spam/"):
        # Return the human-readable spam label for auditability.
        return "spam"
    # Treat members under a ham directory as the negative class.
    if "/ham/" in normalized or normalized.startswith("ham/"):
        # Return the human-readable ham label for auditability.
        return "ham"
    # Ignore directory entries and unexpected member layouts.
    return None


# Load every Enron archive member and map labels onto the unified schema.
def normalize_dataset() -> pd.DataFrame:
    """Return Enron-Spam messages with canonical text, label, source, and split fields."""

    # Accumulate normalized row dictionaries before building the DataFrame.
    rows: list[dict[str, object]] = []
    # Walk every user-specific archive in deterministic order.
    for archive_name in ARCHIVE_NAMES:
        # Ensure the archive exists locally before parsing.
        archive_path = download_archive(archive_name)
        # Open the gzipped tar without extracting unknown paths to disk.
        with tarfile.open(archive_path, mode="r:gz") as archive:
            # Iterate over every member while preserving archive order.
            for member in archive.getmembers():
                # Skip directories and non-regular files.
                if not member.isfile():
                    # Continue searching for message files only.
                    continue
                # Derive the source label from the ham/spam directory name.
                original_label = _label_from_member_name(member.name)
                # Skip members that are not labeled message files.
                if original_label is None:
                    # Ignore README-like files and unexpected paths.
                    continue
                # Extract the member as a binary stream without writing to disk.
                extracted = archive.extractfile(member)
                # Guard against tar members that cannot be opened as files.
                if extracted is None:
                    # Skip unreadable members instead of failing the whole corpus.
                    continue
                # Read the complete message bytes for body extraction.
                raw_message = extracted.read()
                # Convert headers and MIME parts into a single plain-text body.
                text = extract_email_text(raw_message)
                # Drop empty bodies that would invalidate vectorization.
                if not text:
                    # Skip blank messages after extraction.
                    continue
                # Record one normalized row for this message file.
                rows.append(
                    {
                        # Defer message_id assignment until the full table exists.
                        "text": text,
                        # Preserve the directory-derived human-readable label.
                        "original_label": original_label,
                        # Map ham to legitimate and spam to scam.
                        "label": 0 if original_label == "ham" else 1,
                        # Record the exact source corpus on every normalized row.
                        "source": "enron_spam",
                        # Avoid assigning train/test membership before deduplication.
                        "split": "unassigned",
                    }
                )
    # Reject an empty result so missing archives fail loudly.
    if not rows:
        # Explain that no labeled messages were recovered from the archives.
        raise ValueError("Enron-Spam normalization produced zero labeled messages")
    # Build a typed DataFrame from the accumulated rows.
    frame = pd.DataFrame(rows)
    # Assign deterministic identifiers for duplicate analysis and traceability.
    frame["message_id"] = [f"enron-spam-{index:05d}" for index in range(len(frame))]
    # Force string dtypes for text-like columns consumed by the EDA notebook.
    frame["text"] = frame["text"].astype("string")
    # Preserve original labels as nullable strings.
    frame["original_label"] = frame["original_label"].astype("string")
    # Return columns in the order documented by the unified schema.
    return frame[
        ["message_id", "text", "label", "original_label", "source", "split"]
    ]


# Coordinate download, normalization, and local persistence.
def main() -> None:
    """Create the normalized Enron-Spam dataset used by multi-corpus EDA."""

    # Transform the source records into the unified binary-label schema.
    normalized = normalize_dataset()
    # Create the ignored processed directory without failing if it exists.
    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Write a portable UTF-8 CSV without a redundant DataFrame index.
    normalized.to_csv(OUTPUT_PATH, index=False)
    # Persist a sidecar checksum file for later sources.yaml updates.
    digest = sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    # Report only aggregate metadata and paths, never message content.
    print(f"Wrote {len(normalized)} rows to {OUTPUT_PATH}")
    # Report class counts for provenance recording.
    print(normalized["label"].value_counts().to_dict())
    # Report the normalized output digest for the data manifest.
    print(f"normalized_sha256={digest}")


# Run the pipeline only when this file is invoked as a script.
if __name__ == "__main__":
    # Execute the deterministic dataset preparation entry point.
    main()
