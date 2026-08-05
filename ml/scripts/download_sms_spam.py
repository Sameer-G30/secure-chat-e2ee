"""Download and normalize the UCI SMS Spam Collection for local research."""

# Import csv to parse every tab-delimited record without NA heuristics.
import csv

# Import sha256 to pin downloaded research data against upstream changes.
from hashlib import sha256

# Import TextIOWrapper to decode the ZIP member as UTF-8 text.
from io import TextIOWrapper

# Import Path for workspace-independent data locations.
from pathlib import Path

# Import urlopen for a dependency-free HTTPS dataset download.
from urllib.request import urlopen

# Import ZipFile to read the archive without extracting unknown paths.
from zipfile import ZipFile

# Import pandas for typed tabular loading and normalization.
import pandas as pd

# Reference UCI's official static archive for dataset identifier 228.
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# Pin the authoritative archive bytes retrieved for this reviewed slice.
EXPECTED_ARCHIVE_SHA256 = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"
# Resolve the ML project directory from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Keep immutable downloaded source files in the ignored raw directory.
RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw"
# Keep normalized derived files in the ignored processed directory.
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"
# Name the local copy of UCI's ZIP archive.
ARCHIVE_PATH = RAW_DIRECTORY / "sms_spam_collection.zip"
# Name the normalized CSV consumed by the EDA notebook.
OUTPUT_PATH = PROCESSED_DIRECTORY / "sms_spam.csv"


# Download the source archive only when it is absent.
def download_archive() -> None:
    """Fetch the official UCI archive into the local raw-data directory."""

    # Create the ignored raw directory without failing when it already exists.
    RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Reuse the local archive to avoid unnecessary network requests.
    if not ARCHIVE_PATH.exists():
        # Open the authoritative HTTPS resource with a finite timeout.
        with urlopen(DATASET_URL, timeout=60) as response:  # noqa: S310
            # Read the complete small archive after TLS verification succeeds.
            archive_bytes = response.read()
        # Persist the immutable source archive for reproducible local reruns.
        ARCHIVE_PATH.write_bytes(archive_bytes)
    # Read the local source bytes whether they were cached or just downloaded.
    archive_bytes = ARCHIVE_PATH.read_bytes()
    # Compute the archive's lowercase hexadecimal SHA-256 digest.
    archive_digest = sha256(archive_bytes).hexdigest()
    # Reject upstream replacement or local corruption before parsing.
    if archive_digest != EXPECTED_ARCHIVE_SHA256:
        # Explain the integrity failure without processing unreviewed data.
        raise ValueError("UCI SMS archive checksum does not match the reviewed source")


# Load the archive member and map source labels onto the unified schema.
def normalize_dataset() -> pd.DataFrame:
    """Return UCI messages with canonical text, label, source, and split fields."""

    # Open only the expected local ZIP file.
    with ZipFile(ARCHIVE_PATH) as archive:
        # Open the explicitly named member instead of extracting archive paths.
        with archive.open("SMSSpamCollection") as source_file:
            # Decode the archived bytes without writing untrusted member paths.
            with TextIOWrapper(source_file, encoding="utf-8") as text_stream:
                # Parse tabs literally and retain text that resembles pandas NA markers.
                rows = list(
                    csv.reader(
                        # Read directly from the decoded archive-member stream.
                        text_stream,
                        # Split source labels from message text at tab delimiters.
                        delimiter="\t",
                        # Treat quote characters as message content in this unquoted corpus.
                        quoting=csv.QUOTE_NONE,
                    )
                )
            # Reject malformed source rows before constructing the normalized table.
            if any(len(row) != 2 for row in rows):
                # Fail clearly if the upstream corpus format changes.
                raise ValueError("UCI SMS dataset contains a malformed row")
            # Build a typed DataFrame without pandas' automatic NA interpretation.
            frame = pd.DataFrame(
                # Preserve each parsed source label and message exactly.
                rows,
                # Assign canonical names because the source file has no header.
                columns=["original_label", "text"],
                # Preserve both source fields as nullable strings.
                dtype="string",
            )
    # Map legitimate and spam source labels to the unified binary target.
    frame["label"] = frame["original_label"].map({"ham": 0, "spam": 1})
    # Reject unexpected labels instead of silently producing incomplete targets.
    if frame["label"].isna().any():
        # Raise a clear error when the upstream label vocabulary changes.
        raise ValueError("UCI SMS dataset contains an unmapped label")
    # Assign deterministic identifiers for duplicate analysis and traceability.
    frame["message_id"] = [f"uci-sms-{index:05d}" for index in range(len(frame))]
    # Record the exact source corpus on every normalized row.
    frame["source"] = "uci_sms_spam"
    # Avoid assigning train/test membership before deduplication and stratification.
    frame["split"] = "unassigned"
    # Return columns in the order documented by the unified schema.
    return frame[
        ["message_id", "text", "label", "original_label", "source", "split"]
    ]


# Coordinate download, normalization, and local persistence.
def main() -> None:
    """Create the normalized UCI SMS dataset used by Slice 1 EDA."""

    # Ensure the authoritative source archive is available.
    download_archive()
    # Transform the source records into the unified binary-label schema.
    normalized = normalize_dataset()
    # Create the ignored processed directory without failing if it exists.
    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Write a portable UTF-8 CSV without a redundant DataFrame index.
    normalized.to_csv(OUTPUT_PATH, index=False)
    # Report only aggregate metadata and paths, never message content.
    print(f"Wrote {len(normalized)} rows to {OUTPUT_PATH}")


# Run the pipeline only when this file is invoked as a script.
if __name__ == "__main__":
    # Execute the deterministic dataset preparation entry point.
    main()
