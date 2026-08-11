"""Normalize the locally downloaded Kaggle phishing-email compilation."""

# Import csv and raise the field-size limit for long email bodies.
import csv

# Import sys so the CSV parser can accept the largest practical fields.
import sys

# Import sha256 to pin the local source file and normalized output.
from hashlib import sha256

# Import Path for workspace-independent data locations.
from pathlib import Path

# Import pandas for typed tabular loading and normalization.
import pandas as pd

# Allow extremely large CSV fields present in the Kaggle phishing compilation.
csv.field_size_limit(sys.maxsize)

# Resolve the ML project directory from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Keep the manually downloaded source CSV in the ignored raw directory.
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Phishing_Email.csv"
# Keep normalized derived files in the ignored processed directory.
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"
# Name the normalized CSV consumed by the EDA notebook.
OUTPUT_PATH = PROCESSED_DIRECTORY / "kaggle_phishing.csv"
# Map the Kaggle uploader's class names onto the unified binary target.
LABEL_MAP = {
    # Map safe email rows to the legitimate class.
    "Safe Email": 0,
    # Map phishing email rows to the scam/warning class.
    "Phishing Email": 1,
}


# Load the local Kaggle CSV and map labels onto the unified schema.
def normalize_dataset() -> pd.DataFrame:
    """Return Kaggle phishing rows with canonical text, label, source, and split fields."""

    # Fail clearly when the manually downloaded source file is missing.
    if not RAW_PATH.exists():
        # Explain exactly which local path the operator must populate.
        raise FileNotFoundError(
            f"Place the Kaggle CSV at {RAW_PATH} before running this script"
        )
    # Accumulate normalized row dictionaries before building the DataFrame.
    rows: list[dict[str, object]] = []
    # Open the source CSV with UTF-8 replacement for any malformed bytes.
    with RAW_PATH.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        # Parse the headered CSV while preserving long quoted email bodies.
        reader = csv.DictReader(handle)
        # Iterate every source row in file order.
        for source_row in reader:
            # Read the email body column published by the Kaggle dataset.
            text = (source_row.get("Email Text") or "").strip()
            # Drop null, empty, or literal pandas-style NA body strings.
            if not text or text.lower() == "nan":
                # Skip unusable bodies instead of inventing placeholders.
                continue
            # Read the human-readable source label column.
            original_label = (source_row.get("Email Type") or "").strip()
            # Map the source label onto the unified binary target.
            label = LABEL_MAP.get(original_label)
            # Reject unexpected label vocabularies instead of silent coercion.
            if label is None:
                # Explain which unexpected label blocked normalization.
                raise ValueError(f"Unmapped Kaggle label: {original_label!r}")
            # Record one normalized row for this usable email.
            rows.append(
                {
                    # Store the cleaned email body text.
                    "text": text,
                    # Preserve the Kaggle class name for auditability.
                    "original_label": original_label,
                    # Store the unified binary decision target.
                    "label": label,
                    # Record the exact source corpus on every normalized row.
                    "source": "kaggle_phishing",
                    # Avoid assigning train/test membership before deduplication.
                    "split": "unassigned",
                }
            )
    # Reject an empty result so a broken source file fails loudly.
    if not rows:
        # Explain that no usable labeled rows were recovered.
        raise ValueError("Kaggle phishing normalization produced zero usable rows")
    # Build a typed DataFrame from the accumulated rows.
    frame = pd.DataFrame(rows)
    # Assign deterministic identifiers for duplicate analysis and traceability.
    frame["message_id"] = [f"kaggle-phishing-{index:05d}" for index in range(len(frame))]
    # Force string dtypes for text-like columns consumed by the EDA notebook.
    frame["text"] = frame["text"].astype("string")
    # Preserve original labels as nullable strings.
    frame["original_label"] = frame["original_label"].astype("string")
    # Return columns in the order documented by the unified schema.
    return frame[
        ["message_id", "text", "label", "original_label", "source", "split"]
    ]


# Coordinate normalization and local persistence for the manual Kaggle download.
def main() -> None:
    """Create the normalized Kaggle phishing dataset used by multi-corpus EDA."""

    # Compute the raw source digest before any transformation.
    raw_digest = sha256(RAW_PATH.read_bytes()).hexdigest()
    # Transform the source records into the unified binary-label schema.
    normalized = normalize_dataset()
    # Create the ignored processed directory without failing if it exists.
    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Write a portable UTF-8 CSV without a redundant DataFrame index.
    normalized.to_csv(OUTPUT_PATH, index=False)
    # Compute the normalized output digest for the data manifest.
    normalized_digest = sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    # Report only aggregate metadata and paths, never message content.
    print(f"Wrote {len(normalized)} rows to {OUTPUT_PATH}")
    # Report class counts for provenance recording.
    print(normalized["label"].value_counts().to_dict())
    # Report original-label counts for schema verification.
    print(normalized["original_label"].value_counts().to_dict())
    # Report the raw source digest for the data manifest.
    print(f"raw_sha256={raw_digest}")
    # Report the normalized output digest for the data manifest.
    print(f"normalized_sha256={normalized_digest}")


# Run the pipeline only when this file is invoked as a script.
if __name__ == "__main__":
    # Execute the deterministic dataset preparation entry point.
    main()
