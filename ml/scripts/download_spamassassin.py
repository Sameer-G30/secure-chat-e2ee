"""Download and normalize the Apache SpamAssassin public corpus for local research."""

# Import tarfile to read bzip2 SpamAssassin archives in memory.
import tarfile

# Import sha256 to pin downloaded research archives against upstream changes.
from hashlib import sha256

# Import Path for workspace-independent data locations.
from pathlib import Path

# Import urlopen for a dependency-free HTTPS dataset download.
from urllib.request import Request, urlopen

# Import pandas for typed tabular loading and normalization.
import pandas as pd

# Import the shared email body extractor used by every email corpus script.
from secure_chat_ml.email_text import extract_email_text

# Resolve the ML project directory from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Keep immutable downloaded source files in the ignored raw directory.
RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "spamassassin"
# Keep normalized derived files in the ignored processed directory.
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"
# Name the normalized CSV consumed by the EDA notebook.
OUTPUT_PATH = PROCESSED_DIRECTORY / "spamassassin.csv"
# Point at the authoritative Apache SpamAssassin public corpus mirror.
BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
# Download the reviewed public-corpus archives matching the literature counts.
ARCHIVE_SPECS = [
    # Include easy legitimate mail from the February 2003 assembly.
    {"name": "20030228_easy_ham.tar.bz2", "original_label": "easy_ham", "label": 0},
    # Include a second easy-ham batch from the same assembly family.
    {"name": "20030228_easy_ham_2.tar.bz2", "original_label": "easy_ham", "label": 0},
    # Include hard legitimate mail that resembles promotional spam.
    {"name": "20030228_hard_ham.tar.bz2", "original_label": "hard_ham", "label": 0},
    # Include the original spam batch from the February 2003 assembly.
    {"name": "20030228_spam.tar.bz2", "original_label": "spam", "label": 1},
    # Prefer the March 2005 spam_2 revision that removed a mislabeled ham.
    {"name": "20050311_spam_2.tar.bz2", "original_label": "spam", "label": 1},
]


# Download one archive only when it is absent locally.
def download_archive(archive_name: str) -> Path:
    """Fetch one SpamAssassin archive into the local raw-data directory."""

    # Create the ignored raw directory without failing when it already exists.
    RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Build the local destination path for this archive.
    archive_path = RAW_DIRECTORY / archive_name
    # Reuse the local archive to avoid unnecessary network requests.
    if not archive_path.exists():
        # Compose the absolute HTTPS URL for the selected archive.
        download_url = f"{BASE_URL}/{archive_name}"
        # Attach a descriptive User-Agent for polite corpus retrieval.
        request = Request(download_url, headers={"User-Agent": "secure-chat-ml/0.1"})
        # Open the authoritative HTTPS resource with a generous timeout.
        with urlopen(request, timeout=180) as response:  # noqa: S310
            # Read the complete archive after TLS verification succeeds.
            archive_bytes = response.read()
        # Persist the immutable source archive for reproducible local reruns.
        archive_path.write_bytes(archive_bytes)
    # Return the local archive path for later parsing.
    return archive_path


# Load every SpamAssassin archive member and map labels onto the unified schema.
def normalize_dataset() -> pd.DataFrame:
    """Return SpamAssassin messages with canonical text, label, source, and split fields."""

    # Accumulate normalized row dictionaries before building the DataFrame.
    rows: list[dict[str, object]] = []
    # Walk every reviewed archive in deterministic order.
    for archive_spec in ARCHIVE_SPECS:
        # Ensure the archive exists locally before parsing.
        archive_path = download_archive(str(archive_spec["name"]))
        # Open the bzip2 tar without extracting unknown paths to disk.
        with tarfile.open(archive_path, mode="r:bz2") as archive:
            # Iterate over every member while preserving archive order.
            for member in archive.getmembers():
                # Skip directories and non-regular files.
                if not member.isfile():
                    # Continue searching for message files only.
                    continue
                # Skip cmds index files that accompany each corpus directory.
                if Path(member.name).name == "cmds":
                    # Ignore SpamAssassin helper index files.
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
                        # Store the extracted plain-text body.
                        "text": text,
                        # Preserve the corpus-partition label for EDA.
                        "original_label": archive_spec["original_label"],
                        # Map ham partitions to 0 and spam partitions to 1.
                        "label": archive_spec["label"],
                        # Record the exact source corpus on every normalized row.
                        "source": "spamassassin",
                        # Avoid assigning train/test membership before deduplication.
                        "split": "unassigned",
                    }
                )
    # Reject an empty result so missing archives fail loudly.
    if not rows:
        # Explain that no labeled messages were recovered from the archives.
        raise ValueError("SpamAssassin normalization produced zero labeled messages")
    # Build a typed DataFrame from the accumulated rows.
    frame = pd.DataFrame(rows)
    # Assign deterministic identifiers for duplicate analysis and traceability.
    frame["message_id"] = [f"spamassassin-{index:05d}" for index in range(len(frame))]
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
    """Create the normalized SpamAssassin dataset used by multi-corpus EDA."""

    # Transform the source records into the unified binary-label schema.
    normalized = normalize_dataset()
    # Create the ignored processed directory without failing if it exists.
    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Write a portable UTF-8 CSV without a redundant DataFrame index.
    normalized.to_csv(OUTPUT_PATH, index=False)
    # Persist a digest for later sources.yaml updates.
    digest = sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    # Report only aggregate metadata and paths, never message content.
    print(f"Wrote {len(normalized)} rows to {OUTPUT_PATH}")
    # Report class counts for provenance recording.
    print(normalized["label"].value_counts().to_dict())
    # Report original-label counts for hard-ham analysis.
    print(normalized["original_label"].value_counts().to_dict())
    # Report the normalized output digest for the data manifest.
    print(f"normalized_sha256={digest}")


# Run the pipeline only when this file is invoked as a script.
if __name__ == "__main__":
    # Execute the deterministic dataset preparation entry point.
    main()
