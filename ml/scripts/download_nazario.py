"""Download and normalize Jose Nazario's phishing corpus for local research."""

# Import mailbox to iterate mbox-formatted phishing snapshots.
import mailbox

# Import sha256 to pin downloaded research files against upstream changes.
from hashlib import sha256

# Import Path for workspace-independent data locations.
from pathlib import Path

# Import urlopen for a dependency-free HTTPS dataset download.
from urllib.request import Request, urlopen

# Import pandas for typed tabular loading and normalization.
import pandas as pd

# Import the shared Message-based extractor used for mbox snapshots.
from secure_chat_ml.email_text import extract_message_text

# Resolve the ML project directory from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Keep immutable downloaded source files in the ignored raw directory.
RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "nazario"
# Keep normalized derived files in the ignored processed directory.
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"
# Name the normalized CSV consumed by the EDA notebook.
OUTPUT_PATH = PROCESSED_DIRECTORY / "nazario.csv"
# Point at Jose Nazario's original phishing corpus directory.
BASE_URL = "https://monkey.org/~jose/phishing"
# Freeze the public snapshots available at retrieval time, excluding private mailboxes.
FILE_NAMES = [
    # Include the earliest historical phishing mailbox.
    "phishing0.mbox",
    # Include the second early historical phishing mailbox.
    "phishing1.mbox",
    # Include the November 2005 snapshot mailbox.
    "20051114.mbox",
    # Include the third early historical phishing mailbox.
    "phishing2.mbox",
    # Include the fourth early historical phishing mailbox.
    "phishing3.mbox",
    # Include the 2015 annual phishing batch.
    "phishing-2015",
    # Include the 2016 annual phishing batch.
    "phishing-2016",
    # Include the 2017 annual phishing batch.
    "phishing-2017",
    # Include the 2018 annual phishing batch.
    "phishing-2018",
    # Include the 2019 annual phishing batch.
    "phishing-2019",
    # Include the 2020 annual phishing batch.
    "phishing-2020",
    # Include the 2021 annual phishing batch.
    "phishing-2021",
    # Include the 2022 annual phishing batch.
    "phishing-2022",
    # Include the 2023 annual phishing batch.
    "phishing-2023",
    # Include the 2024 annual phishing batch.
    "phishing-2024",
    # Include the 2025 annual phishing batch.
    "phishing-2025",
]


# Download one corpus file only when it is absent locally.
def download_file(file_name: str) -> Path:
    """Fetch one Nazario corpus file into the local raw-data directory."""

    # Create the ignored raw directory without failing when it already exists.
    RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Build the local destination path for this corpus file.
    file_path = RAW_DIRECTORY / file_name
    # Reuse the local file to avoid unnecessary network requests.
    if not file_path.exists():
        # Compose the absolute HTTPS URL for the selected file.
        download_url = f"{BASE_URL}/{file_name}"
        # Attach a descriptive User-Agent for polite corpus retrieval.
        request = Request(download_url, headers={"User-Agent": "secure-chat-ml/0.1"})
        # Open the authoritative HTTPS resource with a generous timeout.
        with urlopen(request, timeout=300) as response:  # noqa: S310
            # Read the complete file after TLS verification succeeds.
            file_bytes = response.read()
        # Persist the immutable source file for reproducible local reruns.
        file_path.write_bytes(file_bytes)
    # Return the local file path for later parsing.
    return file_path


# Load every public Nazario message and map it onto the unified schema.
def normalize_dataset() -> pd.DataFrame:
    """Return Nazario phishing messages with canonical text, label, source, and split fields."""

    # Accumulate normalized row dictionaries before building the DataFrame.
    rows: list[dict[str, object]] = []
    # Walk every frozen public snapshot in deterministic order.
    for file_name in FILE_NAMES:
        # Ensure the snapshot exists locally before parsing.
        file_path = download_file(file_name)
        # Open the mailbox using the standard mbox reader.
        mbox = mailbox.mbox(file_path)
        # Iterate every message contained in the snapshot.
        for message in mbox:
            # Convert headers and MIME parts into a single plain-text body.
            text = extract_message_text(message)
            # Drop empty bodies that would invalidate vectorization.
            if not text:
                # Skip blank messages after extraction.
                continue
            # Record one normalized phishing-only row.
            rows.append(
                {
                    # Store the extracted plain-text body.
                    "text": text,
                    # Preserve the phishing-only source label for auditability.
                    "original_label": "phishing",
                    # Map every Nazario message to the scam/warning class.
                    "label": 1,
                    # Record the exact source corpus on every normalized row.
                    "source": "nazario",
                    # Avoid assigning train/test membership before deduplication.
                    "split": "unassigned",
                }
            )
        # Close the mailbox handle after exhausting its messages.
        mbox.close()
    # Reject an empty result so missing files fail loudly.
    if not rows:
        # Explain that no phishing messages were recovered from the snapshots.
        raise ValueError("Nazario normalization produced zero phishing messages")
    # Build a typed DataFrame from the accumulated rows.
    frame = pd.DataFrame(rows)
    # Assign deterministic identifiers for duplicate analysis and traceability.
    frame["message_id"] = [f"nazario-{index:05d}" for index in range(len(frame))]
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
    """Create the normalized Nazario dataset used by multi-corpus EDA."""

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
    # Report the normalized output digest for the data manifest.
    print(f"normalized_sha256={digest}")


# Run the pipeline only when this file is invoked as a script.
if __name__ == "__main__":
    # Execute the deterministic dataset preparation entry point.
    main()
